"""
freerouter/discovery/__init__.py
Model discovery module for dynamically fetching available models from platforms.
"""
from .base import (
    DiscoveredModel,
    DiscoveryResult,
    PlatformDiscoverer,
    detect_capabilities,
)
from .nvidia import NvidiaDiscoverer
from .groq import GroqDiscoverer
from .cerebras import CerebrasDiscoverer
from .sambanova import SambaNovaDiscoverer
from .mistral import MistralDiscoverer
from .discovery_manager import (
    DiscoveryManager,
    DiscoveryConfig,
    discovery_loop,
)

__all__ = [
    # Base classes
    "DiscoveredModel",
    "DiscoveryResult",
    "PlatformDiscoverer",
    "detect_capabilities",
    # Platform discoverers
    "NvidiaDiscoverer",
    "GroqDiscoverer",
    "CerebrasDiscoverer",
    "SambaNovaDiscoverer",
    "MistralDiscoverer",
    # Manager
    "DiscoveryManager",
    "DiscoveryConfig",
    "discovery_loop",
]
