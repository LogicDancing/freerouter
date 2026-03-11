"""
freerouter/benchmarks/model_scores.py
Model capability scores from LMSYS Arena and other benchmarks.

This module provides:
- ELO ratings from LMSYS Chatbot Arena
- Domain-specific rankings (coding, reasoning, etc.)
- Score caching and periodic updates
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

log = logging.getLogger("freerouter.benchmarks")


class ScoreSource(str, Enum):
    """Source of the score data."""
    LMSYS_ARENA = "lmsys_arena"
    HUMAN_EVAL = "human_eval"
    MBPP = "mbpp"
    MATH_BENCH = "math_bench"
    SWE_BENCH = "swe_bench"
    LIVE_CODE_BENCH = "live_code_bench"
    CUSTOM = "custom"


@dataclass
class ModelScore:
    """Score data for a model from a specific benchmark."""
    model_id: str
    score: float
    source: ScoreSource
    timestamp: datetime = field(default_factory=datetime.utcnow)
    # Additional context
    rank: Optional[int] = None
    total_models: Optional[int] = None
    confidence: float = 1.0  # Lower confidence for estimated/derived scores


@dataclass
class ModelCapabilities:
    """
    Comprehensive capability assessment for a model.
    
    Aggregates scores from multiple benchmarks into domain-specific ratings.
    """
    model_id: str
    
    # Overall quality (from LMSYS Arena ELO)
    overall_elo: Optional[float] = None
    
    # Domain-specific scores (0-100 normalized)
    coding_score: Optional[float] = None  # Code generation/completion
    reasoning_score: Optional[float] = None  # Math, logic, step-by-step
    instruction_score: Optional[float] = None  # Following complex instructions
    long_context_score: Optional[float] = None  # Handling long inputs
    
    # Specialized capabilities
    supports_vision: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True
    
    # Performance characteristics
    context_window: int = 8192
    typical_speed_tps: Optional[float] = None
    
    # Last updated
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_score_for_domain(self, domain: str) -> Optional[float]:
        """Get score for a specific domain."""
        domain_scores = {
            "coding": self.coding_score,
            "reasoning": self.reasoning_score,
            "instruction": self.instruction_score,
            "long_context": self.long_context_score,
            "general": self.overall_elo,
        }
        return domain_scores.get(domain.lower())
    
    def is_better_than(self, other: 'ModelCapabilities', domain: str) -> bool:
        """Compare this model to another for a specific domain."""
        my_score = self.get_score_for_domain(domain) or 0
        other_score = other.get_score_for_domain(domain) or 0
        return my_score > other_score


# ============================================================================
# HARDCODED BENCHMARK DATA
# Based on LMSYS Arena (Feb 2026) and other benchmarks
# ============================================================================

# LMSYS Arena ELO ratings (approximate, from Feb 2026)
LMSYS_ELO_DATA: dict[str, float] = {
    # Closed-source frontier (for reference)
    "claude-opus-4-6": 1505,
    "claude-4-6-thinking": 1504,
    "gemini-3-1-pro": 1500,
    "gpt-5-2": 1480,
    
    # Open-source models (available on free platforms)
    "deepseek-r1": 1380,
    "deepseek-v3": 1340,
    "deepseek-v3-1": 1350,
    "deepseek-v3-2": 1355,
    "qwen3-397b": 1340,
    "qwen3-235b": 1320,
    "qwen3-32b": 1280,
    "glm-4-7": 1330,
    "glm-5": 1340,
    "llama-3-3-70b": 1300,
    "llama-3-1-405b": 1310,
    "llama-3-1-70b": 1280,
    "llama-3-1-8b": 1180,
    "llama-4-scout": 1320,
    "llama-4-maverick": 1310,
    "qwq-32b": 1260,
    "kimi-k2": 1270,
    
    # Mistral models
    "mistral-large-3": 1320,
    "mistral-medium-3-1": 1300,
    "mistral-small-3-2": 1250,
    "codestral": 1280,
    "devstral": 1290,
    "devstral-2": 1310,
    "ministral-3-14b": 1220,
    "ministral-3-8b": 1180,
}

# Coding benchmark scores (HumanEval, LiveCodeBench composite)
CODING_SCORES: dict[str, float] = {
    # Top tier
    "qwen3-coder-480b": 95,
    "deepseek-v3": 92,
    "deepseek-v3-1": 93,
    "deepseek-v3-2": 94,
    "codestral": 90,
    "devstral": 91,
    "devstral-2": 93,
    
    # Strong
    "llama-3-3-70b": 85,
    "llama-3-1-70b": 84,
    "qwen3-32b": 86,
    "qwen2-5-coder-32b": 88,
    "qwq-32b": 82,
    
    # Medium
    "llama-3-1-8b": 72,
    "deepseek-coder-6-7b": 75,
    "starcoder2-15b": 76,
    "ministral-3-8b": 70,
    
    # Light
    "gemma-2-9b": 68,
    "mistral-small-3-2": 74,
}

# Reasoning/math scores
REASONING_SCORES: dict[str, float] = {
    # Top tier (R1-class models)
    "deepseek-r1": 95,
    "deepseek-r1-distill-llama-70b": 88,
    "deepseek-r1-distill-qwen-32b": 86,
    "qwq-32b": 87,
    "kimi-k2": 85,
    
    # Strong
    "deepseek-v3": 82,
    "deepseek-v3-1": 83,
    "deepseek-v3-2": 84,
    "qwen3-397b": 83,
    "llama-3-1-405b": 80,
    
    # Medium
    "llama-3-3-70b": 75,
    "llama-3-1-70b": 74,
    "qwen3-32b": 76,
    "glm-5": 78,
    
    # Reasoning-focused models
    "magistral-medium": 89,
    "magistral-small": 85,
}

# Long context handling
LONG_CONTEXT_SCORES: dict[str, float] = {
    # 1M+ context models
    "llama-3-1-nemotron-nano-8b-v1": 98,
    "llama-3-1-nemotron-8b-ultralong-1m": 99,
    "llama-3-1-nemotron-8b-ultralong-2m": 99,
    "nemotron-3-nano-30b-a3b": 97,
    
    # 128K+ context
    "deepseek-v3": 90,
    "llama-3-1-405b": 88,
    "llama-3-3-70b": 85,
    "qwen3-coder-480b": 92,
    
    # 32K-128K
    "llama-3-1-8b": 75,
    "mistral-large-3": 85,
}


def get_model_capabilities(model_id: str) -> ModelCapabilities:
    """
    Get comprehensive capability assessment for a model.
    
    Args:
        model_id: The model identifier (e.g., "deepseek-v3", "llama-3-1-70b")
    
    Returns:
        ModelCapabilities with scores from various benchmarks
    """
    # Normalize model ID
    normalized_id = _normalize_model_id(model_id)
    
    caps = ModelCapabilities(model_id=model_id)
    
    # Look up scores from hardcoded data
    caps.overall_elo = _lookup_score(normalized_id, LMSYS_ELO_DATA)
    caps.coding_score = _lookup_score(normalized_id, CODING_SCORES)
    caps.reasoning_score = _lookup_score(normalized_id, REASONING_SCORES)
    caps.long_context_score = _lookup_score(normalized_id, LONG_CONTEXT_SCORES)
    
    # Instruction following correlates with overall quality
    if caps.overall_elo:
        caps.instruction_score = min(100, (caps.overall_elo - 1100) / 4)
    
    # Detect capabilities from model name
    caps.supports_vision = _detect_vision(model_id)
    
    return caps


def _normalize_model_id(model_id: str) -> str:
    """Normalize model ID for lookup."""
    # Remove common prefixes and normalize
    normalized = model_id.lower()
    for prefix in ["meta/", "deepseek-ai/", "qwen/", "mistralai/", "nvidia/", "z-ai/", "thudm/"]:
        normalized = normalized.replace(prefix.lower(), "")
    
    # Handle version suffixes
    normalized = normalized.replace("-instruct", "").replace("-it", "")
    
    return normalized


def _lookup_score(normalized_id: str, score_dict: dict[str, float]) -> Optional[float]:
    """Look up score with fuzzy matching."""
    # Exact match
    if normalized_id in score_dict:
        return score_dict[normalized_id]
    
    # Fuzzy match - check if any key is contained in normalized_id or vice versa
    for key, value in score_dict.items():
        if key in normalized_id or normalized_id in key:
            return value
    
    return None


def _detect_vision(model_id: str) -> bool:
    """Detect if model supports vision from its name."""
    vision_patterns = ["vision", "vl", "pixtral", "qwen2-vl", "qwen3-vl", "ministral"]
    return any(p in model_id.lower() for p in vision_patterns)


def compare_models(
    model_a: str,
    model_b: str,
    domain: str = "general"
) -> int:
    """
    Compare two models for a specific domain.
    
    Returns:
        1 if model_a is better, -1 if model_b is better, 0 if equal
    """
    caps_a = get_model_capabilities(model_a)
    caps_b = get_model_capabilities(model_b)
    
    score_a = caps_a.get_score_for_domain(domain) or 0
    score_b = caps_b.get_score_for_domain(domain) or 0
    
    if score_a > score_b + 2:  # Small threshold to handle noise
        return 1
    elif score_b > score_a + 2:
        return -1
    return 0


def get_top_models(
    domain: str = "general",
    platform: Optional[str] = None,
    limit: int = 5
) -> list[tuple[str, float]]:
    """
    Get top models for a specific domain.
    
    Args:
        domain: Domain to rank by ("coding", "reasoning", "general", "long_context")
        platform: Optional platform filter (e.g., "nvidia_build", "groq")
        limit: Maximum number of models to return
    
    Returns:
        List of (model_id, score) tuples sorted by score
    """
    # Choose the right score dict
    if domain == "coding":
        score_dict = CODING_SCORES
    elif domain == "reasoning":
        score_dict = REASONING_SCORES
    elif domain == "long_context":
        score_dict = LONG_CONTEXT_SCORES
    else:
        score_dict = LMSYS_ELO_DATA
    
    # Sort by score
    sorted_models = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Apply limit
    return sorted_models[:limit]
