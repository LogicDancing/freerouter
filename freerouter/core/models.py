"""
freerouter/core/models.py
Data models for platforms, probes, routing decisions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class TaskType(str, Enum):
    CODE_COMPLETION = "code_completion"    # fast, high-freq
    CODE_GENERATION = "code_generation"    # quality matters
    AGENTIC = "agentic"                    # multi-step, token-heavy
    REASONING = "reasoning"               # complex logic/math
    LONG_CONTEXT = "long_context"         # >128K tokens
    GENERAL = "general"                   # default


class ProbeStatus(str, Enum):
    OK = "ok"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class PlatformModel:
    model_id: str
    context_window: int           # in tokens
    priority: int                 # 1 = highest
    task_types: list[TaskType]
    description: str = ""


@dataclass
class Platform:
    name: str                     # "groq", "cerebras", etc.
    display_name: str
    api_base: str
    probe_model: str              # cheapest/fastest model for probing
    models: list[PlatformModel]
    daily_request_limit: Optional[int] = None
    daily_token_limit: Optional[int] = None
    rpm_limit: int = 30
    # strengths as a short tag list for display
    tags: list[str] = field(default_factory=list)


@dataclass
class ProbeResult:
    platform_name: str
    model_id: str
    ttft_ms: float               # time-to-first-token in milliseconds
    status: ProbeStatus
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    cooldown_until: Optional[float] = None  # epoch seconds

    @property
    def is_available(self) -> bool:
        if self.status != ProbeStatus.OK:
            return False
        if self.cooldown_until and time.time() < self.cooldown_until:
            return False
        return True

    @property
    def is_congested(self) -> bool:
        return self.ttft_ms > 3000  # 3 seconds threshold


@dataclass
class RoutingDecision:
    task_type: TaskType
    selected_platform: str
    selected_model: str
    reason: str
    fallback_chain: list[str]    # platforms tried / would try
    probe_ttft_ms: float
    timestamp: float = field(default_factory=time.time)
