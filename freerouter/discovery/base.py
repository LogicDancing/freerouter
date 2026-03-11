"""
freerouter/discovery/base.py
Base class and common utilities for model discovery.
"""
from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

log = logging.getLogger("freerouter.discovery")


@dataclass
class DiscoveredModel:
    """A model discovered from a platform API."""
    model_id: str
    platform: str
    context_window: int = 8192
    owned_by: str = ""
    # Capabilities
    supports_vision: bool = False
    supports_tools: bool = False
    supports_streaming: bool = True
    # Rate limits (if available from API)
    rpm_limit: Optional[int] = None
    rpd_limit: Optional[int] = None
    tpd_limit: Optional[int] = None
    # Raw metadata from API
    raw: dict = field(default_factory=dict)
    # Discovery metadata
    discovered_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DiscoveryResult:
    """Result of a discovery operation."""
    platform: str
    models: list[DiscoveredModel]
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PlatformDiscoverer(ABC):
    """Abstract base class for platform model discovery."""
    
    platform_name: str
    api_base: str
    models_endpoint: str = "/v1/models"
    
    def __init__(self, api_key: Optional[str] = None, timeout: float = 15.0):
        self.api_key = api_key
        self.timeout = timeout
    
    @abstractmethod
    async def fetch_models(self) -> list[DiscoveredModel]:
        """Fetch available models from the platform."""
        pass
    
    async def discover(self) -> DiscoveryResult:
        """Run discovery and return result."""
        try:
            models = await self.fetch_models()
            log.info(f"Discovered {len(models)} models from {self.platform_name}")
            return DiscoveryResult(
                platform=self.platform_name,
                models=models,
                success=True
            )
        except Exception as e:
            log.warning(f"Discovery failed for {self.platform_name}: {e}")
            return DiscoveryResult(
                platform=self.platform_name,
                models=[],
                success=False,
                error=str(e)
            )
    
    def normalize_model_id(self, model_id: str) -> str:
        """Normalize model ID for consistent comparison."""
        return model_id.lower().strip()


# Common model ID patterns for capability detection
VISION_PATTERNS = ["vision", "vlm", "llava", "qwen-vl", "gemini", "claude-3"]
CODER_PATTERNS = ["coder", "code", "starcoder", "codellama", "deepseek-coder"]
REASONING_PATTERNS = ["reasoning", "qwq", "deepseek-r1", "o1", "thinking"]


def detect_capabilities(model_id: str, metadata: dict = None) -> dict:
    """Detect model capabilities from model ID and metadata."""
    model_id_lower = model_id.lower()
    metadata = metadata or {}
    
    return {
        "supports_vision": any(p in model_id_lower for p in VISION_PATTERNS) or metadata.get("supports_vision", False),
        "is_coder": any(p in model_id_lower for p in CODER_PATTERNS),
        "is_reasoning": any(p in model_id_lower for p in REASONING_PATTERNS),
    }
