"""
freerouter/core/model_selector.py
Intelligent model selection with rate limit awareness.
"""
from __future__ import annotations
import logging
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from freerouter.core.models import ProbeResult, ProbeStatus, TaskType
from freerouter.core.platforms import Platform, PlatformModel, PLATFORM_MAP

log = logging.getLogger("freerouter.model_selector")


@dataclass
class ModelChoice:
    """A model choice with context."""
    platform_name: str
    model_id: str
    priority: int
    is_rate_limited: bool = False
    quota_tier: str = "unknown"  # "high", "medium", "low"


# Rate limit tiers for SambaNova models (requests per day)
SAMBANOVA_QUOTA_TIERS = {
    # High quota (288K req/day) - most resilient
    "Meta-Llama-3.1-8B-Instruct": "high",
    # Medium quota (48K req/day)
    "Meta-Llama-3.3-70B-Instruct": "medium",
    "DeepSeek-R1-Distill-Llama-70B": "medium",
    # Low quota (12K req/day) - most restrictive
    "DeepSeek-V3.1": "low",
    "DeepSeek-V3.0324": "low",
    "DeepSeek-R1-0528": "low",
    "Qwen3-32B": "low",
    "Llama-4-Maverick-17B-128E-Instruct": "low",
    "Qwen2.5-Coder-32B-Instruct": "low",
    "Qwen3-235B": "low",
    "gpt-oss-120b": "low",
    "MiniMax-M2.5": "low",
}

# Track rate limit cooldowns per model
_rate_limit_cache: dict[str, datetime] = {}
RATE_LIMIT_COOLDOWN = timedelta(minutes=5)


def mark_rate_limited(platform_name: str, model_id: str) -> None:
    """Mark a model as rate limited."""
    key = f"{platform_name}:{model_id}"
    _rate_limit_cache[key] = datetime.utcnow()
    log.debug(f"Marked rate limited: {key}")


def is_rate_limited(platform_name: str, model_id: str) -> bool:
    """Check if a model is currently rate limited."""
    key = f"{platform_name}:{model_id}"
    if key not in _rate_limit_cache:
        return False
    
    # Check if cooldown has passed
    limited_at = _rate_limit_cache[key]
    if datetime.utcnow() - limited_at > RATE_LIMIT_COOLDOWN:
        del _rate_limit_cache[key]
        return False
    
    return True


def get_sambanova_quota_tier(model_id: str) -> str:
    """Get the quota tier for a SambaNova model."""
    return SAMBANOVA_QUOTA_TIERS.get(model_id, "unknown")


def select_best_model(
    platform: Platform,
    task: TaskType,
    probe_result: Optional[ProbeResult] = None,
    avoid_rate_limited: bool = True,
) -> Optional[ModelChoice]:
    """
    Select the best model for a task on a platform.
    
    Priority order:
    1. Task type match
    2. Priority number (lower = better)
    3. Quota tier (high > medium > low)
    4. Not rate limited
    
    Args:
        platform: The platform to select from
        task: The task type
        probe_result: Current probe status
        avoid_rate_limited: Whether to skip rate-limited models
    
    Returns:
        Best model choice or None
    """
    # Get candidates matching task type
    candidates = []
    for model in platform.models:
        if task in model.task_types:
            candidates.append(model)
    
    # Fall back to all models if no match
    if not candidates:
        candidates = platform.models
    
    if not candidates:
        return None
    
    # Sort by priority (lower is better)
    candidates.sort(key=lambda m: m.priority)
    
    # For SambaNova, prefer high-quota models and avoid rate-limited ones
    if platform.name == "sambanova":
        return _select_sambanova_model(candidates, task, avoid_rate_limited)
    
    # For other platforms, just pick the best priority
    best = candidates[0]
    
    # Check if rate limited
    if avoid_rate_limited and is_rate_limited(platform.name, best.model_id):
        # Try next model
        for model in candidates[1:]:
            if not is_rate_limited(platform.name, model.model_id):
                return ModelChoice(
                    platform_name=platform.name,
                    model_id=model.model_id,
                    priority=model.priority,
                    is_rate_limited=False,
                )
        # All rate limited, return first anyway
        return ModelChoice(
            platform_name=platform.name,
            model_id=best.model_id,
            priority=best.priority,
            is_rate_limited=True,
        )
    
    return ModelChoice(
        platform_name=platform.name,
        model_id=best.model_id,
        priority=best.priority,
        is_rate_limited=is_rate_limited(platform.name, best.model_id),
    )


def _select_sambanova_model(
    candidates: list[PlatformModel],
    task: TaskType,
    avoid_rate_limited: bool,
) -> Optional[ModelChoice]:
    """Select best SambaNova model considering quota tiers."""
    # Score each candidate
    scored = []
    for model in candidates:
        tier = get_sambanova_quota_tier(model.model_id)
        tier_score = {"high": 0, "medium": 1, "low": 2, "unknown": 3}.get(tier, 3)
        rate_limited = is_rate_limited("sambanova", model.model_id)
        
        # Skip rate limited if avoiding
        if avoid_rate_limited and rate_limited:
            continue
        
        # Score: lower is better
        # Priority * 10 + tier_score gives us a good ordering
        score = model.priority * 10 + tier_score
        if rate_limited:
            score += 100  # Penalty for rate limited
        
        scored.append((model, score, rate_limited, tier))
    
    if not scored:
        return None
    
    # Sort by score
    scored.sort(key=lambda x: x[1])
    
    best, score, rate_limited, tier = scored[0]
    
    return ModelChoice(
        platform_name="sambanova",
        model_id=best.model_id,
        priority=best.priority,
        is_rate_limited=rate_limited,
        quota_tier=tier,
    )


def get_fallback_models(
    platform: Platform,
    task: TaskType,
    excluded_model_ids: list[str],
) -> list[ModelChoice]:
    """Get fallback model options, excluding certain models."""
    choices = []
    
    for model in platform.models:
        if model.model_id in excluded_model_ids:
            continue
        if task in model.task_types or task == TaskType.GENERAL:
            tier = get_sambanova_quota_tier(model.model_id) if platform.name == "sambanova" else "unknown"
            choices.append(ModelChoice(
                platform_name=platform.name,
                model_id=model.model_id,
                priority=model.priority,
                is_rate_limited=is_rate_limited(platform.name, model.model_id),
                quota_tier=tier,
            ))
    
    # Sort by priority, then by quota tier
    choices.sort(key=lambda c: (c.priority, {"high": 0, "medium": 1, "low": 2, "unknown": 3}.get(c.quota_tier, 3)))
    
    return choices
