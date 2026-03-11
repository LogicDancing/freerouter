"""
freerouter/benchmarks/__init__.py
Benchmark and scoring module for model capability assessment.
"""
from .model_scores import (
    ModelScore,
    ModelCapabilities,
    ScoreSource,
    get_model_capabilities,
    compare_models,
    get_top_models,
    LMSYS_ELO_DATA,
    CODING_SCORES,
    REASONING_SCORES,
    LONG_CONTEXT_SCORES,
)

__all__ = [
    "ModelScore",
    "ModelCapabilities",
    "ScoreSource",
    "get_model_capabilities",
    "compare_models",
    "get_top_models",
    "LMSYS_ELO_DATA",
    "CODING_SCORES",
    "REASONING_SCORES",
    "LONG_CONTEXT_SCORES",
]
