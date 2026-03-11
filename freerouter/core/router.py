"""
freerouter/core/router.py
Routing decision engine: task type detection + platform selection.
"""
from __future__ import annotations
import logging
import re
from typing import Optional

from freerouter.core.config import FreeRouterConfig
from freerouter.core.models import (
    ProbeResult, ProbeStatus, RoutingDecision, TaskType
)
from freerouter.core.platforms import (
    ALL_PLATFORMS, PLATFORM_MAP, Platform, PlatformModel
)
from freerouter.core.prober import Prober

log = logging.getLogger("freerouter.router")

# ─── Task type detection heuristics ─────────────────────────────────────────

_CODE_COMPLETION_PATTERNS = [
    r"complete\s+the\s+(code|function|method)",
    r"fill\s+in\s+the\s+(blank|rest)",
    r"<\|fim",           # fill-in-middle tokens
    r"// ?TODO",
    r"# ?TODO",
]

_CODE_GENERATION_PATTERNS = [
    r"```[\w+\-]*\n",    # code block in prompt
    r"\bdef\s+\w+\(",
    r"\bfunction\s+\w+\(",
    r"\bclass\s+\w+",
    r"implement\b",
    r"write\s+(\w+\s+)*(function|class|method|script|program|api|service|server|cli|tool)",
    r"\bcode\s+(for|to|that)\b",
    r"\bgenerate\s+(code|a\s+function|a\s+class)\b",
    r"\.(py|ts|js|go|rs|java|cpp|c)\b",
]

_REASONING_PATTERNS = [
    r"prove\b",
    r"proof\b",
    r"derive\b",
    r"mathematical(ly)?\b",
    r"step[- ]by[- ]step\b",
    r"reason(ing)?\s+through",
    r"∑|∫|∂|∈|∀|∃|→|⟹",
]

_AGENTIC_PATTERNS = [
    r"multi[- ]?step",
    r"workflow\b",
    r"plan\s+(and\s+)?execute",
    r"use\s+(the\s+)?tool",
    r"call\s+(the\s+)?function",
]


def detect_task_type(messages: list[dict], estimated_tokens: int) -> TaskType:
    """Heuristic task type detection from chat messages."""
    # Long context always wins — only NVIDIA Build can handle it
    if estimated_tokens > 128_000:
        return TaskType.LONG_CONTEXT

    # Collect all text content
    text = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str)
        else " ".join(
            c.get("text", "") for c in m.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        )
        for m in messages
    ).lower()

    if any(re.search(p, text, re.IGNORECASE) for p in _AGENTIC_PATTERNS):
        return TaskType.AGENTIC

    if any(re.search(p, text, re.IGNORECASE) for p in _REASONING_PATTERNS):
        return TaskType.REASONING

    if any(re.search(p, text, re.IGNORECASE) for p in _CODE_COMPLETION_PATTERNS):
        return TaskType.CODE_COMPLETION

    if any(re.search(p, text, re.IGNORECASE) for p in _CODE_GENERATION_PATTERNS):
        return TaskType.CODE_GENERATION

    return TaskType.GENERAL


# ─── Routing priority tables ─────────────────────────────────────────────────
# For each task type: ordered list of (platform_name, preferred_model_filter)
# Router picks first available (not congested, not rate-limited) platform.

ROUTING_TABLE: dict[TaskType, list[str]] = {
    TaskType.CODE_COMPLETION: [
        "groq", # 300 t/s — fastest, 14K req/day
        "cerebras", # 2500 t/s, 1M tokens/day
        "nvidia_build", # Reliable fallback
        "sambanova",
        "mistral", # Codestral for code
    ],
    TaskType.CODE_GENERATION: [
        "nvidia_build", # Qwen3 Coder 480B — world #1
        "mistral", # Codestral, Devstral
        "groq",
        "cerebras",
        "sambanova",
    ],
    TaskType.AGENTIC: [
        "cerebras", # 1M tokens/day, fast
        "nvidia_build", # Reliable, good context
        "groq", # fast per-step
        "sambanova",
        "mistral", # Devstral for agents
    ],
    TaskType.REASONING: [
        "nvidia_build", # DeepSeek R1, QwQ-32b
        "mistral", # Magistral models
        "groq", # QwQ, DeepSeek R1 Distill
        "sambanova",
        "cerebras",
    ],
    TaskType.LONG_CONTEXT: [
        "nvidia_build", # Only platform with 1M+ context
    ],
    TaskType.GENERAL: [
        "nvidia_build", # Most models
        "groq", # Fastest
        "cerebras", # Good for agentic
        "sambanova", # High free quota
        "mistral", # Quality models
    ],
}


def _best_model_for_task(platform: Platform, task: TaskType, avoid_rate_limited: bool = True) -> Optional[str]:
    """Return the highest-priority model ID for this task on this platform."""
    from freerouter.core.model_selector import select_best_model
    
    choice = select_best_model(platform, task, avoid_rate_limited=avoid_rate_limited)
    return choice.model_id if choice else None


class Router:
    def __init__(self, config: FreeRouterConfig, prober: Prober):
        self.config = config
        self.prober = prober

    def decide(
        self,
        messages: list[dict],
        estimated_tokens: int,
        user_override_platform: Optional[str] = None,
        user_override_model: Optional[str] = None,
    ) -> RoutingDecision:
        """Return a routing decision for this request."""

        # 1. Hard user override
        if user_override_platform and user_override_model:
            return RoutingDecision(
                task_type=TaskType.GENERAL,
                selected_platform=user_override_platform,
                selected_model=user_override_model,
                reason="user_override",
                fallback_chain=[],
                probe_ttft_ms=0,
            )

        # 2. Detect task
        task = detect_task_type(messages, estimated_tokens)
        log.debug("Detected task type: %s", task.value)

        # 3. Walk the routing table
        probe_results = self.prober.results
        ordered_platforms = ROUTING_TABLE.get(task, ROUTING_TABLE[TaskType.GENERAL])
        available_platforms = self.config.api_keys.available_platforms()

        fallback_chain: list[str] = []
        threshold = self.config.routing.ttft_threshold_ms

        for platform_name in ordered_platforms:
            if platform_name not in available_platforms:
                continue  # no API key

            platform = PLATFORM_MAP[platform_name]
            probe = probe_results.get(platform_name)

            fallback_chain.append(platform_name)

            # No probe data yet — assume available (first run)
            if probe is None:
                model_id = _best_model_for_task(platform, task) or platform.probe_model
                return RoutingDecision(
                    task_type=task,
                    selected_platform=platform_name,
                    selected_model=model_id,
                    reason="no_probe_data_yet",
                    fallback_chain=fallback_chain,
                    probe_ttft_ms=0,
                )

            if not probe.is_available:
                log.debug("Skipping %s: status=%s", platform_name, probe.status)
                continue

            if probe.ttft_ms > threshold:
                log.debug("Skipping %s: congested (%.0fms > %.0fms)",
                          platform_name, probe.ttft_ms, threshold)
                continue

            # ✓ This platform is available and not congested
            model_id = _best_model_for_task(platform, task) or platform.probe_model
            return RoutingDecision(
                task_type=task,
                selected_platform=platform_name,
                selected_model=model_id,
                reason=f"ttft={probe.ttft_ms:.0f}ms",
                fallback_chain=fallback_chain,
                probe_ttft_ms=probe.ttft_ms,
            )

        # 4. Everything congested — pick least-bad available
        log.warning("All platforms congested; selecting least-bad option")
        for platform_name in ordered_platforms:
            if platform_name not in available_platforms:
                continue
            platform = PLATFORM_MAP[platform_name]
            probe = probe_results.get(platform_name)
            if probe and probe.status == ProbeStatus.OK:
                model_id = _best_model_for_task(platform, task) or platform.probe_model
                return RoutingDecision(
                    task_type=task,
                    selected_platform=platform_name,
                    selected_model=model_id,
                    reason="all_congested_least_bad",
                    fallback_chain=fallback_chain,
                    probe_ttft_ms=probe.ttft_ms,
                )

        # 5. Total failure — raise so proxy can return 503
        raise RuntimeError(
            "No available platforms. Check API keys and network connectivity."
        )
