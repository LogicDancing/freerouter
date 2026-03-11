"""
freerouter/discovery/discovery_manager.py
Unified discovery manager that coordinates model discovery across all platforms.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable
from dataclasses import dataclass, field

from .base import DiscoveredModel, DiscoveryResult
from .nvidia import NvidiaDiscoverer
from .groq import GroqDiscoverer
from .cerebras import CerebrasDiscoverer
from .sambanova import SambaNovaDiscoverer
from .mistral import MistralDiscoverer

log = logging.getLogger("freerouter.discovery")


@dataclass
class DiscoveryConfig:
    """Configuration for discovery behavior."""
    poll_interval_hours: float = 6.0  # How often to refresh model lists
    timeout_seconds: float = 15.0
    enable_nvidia: bool = True
    enable_groq: bool = True
    enable_cerebras: bool = True
    enable_sambanova: bool = True
    enable_mistral: bool = True


@dataclass
class CachedDiscovery:
    """Cached discovery results."""
    results: dict[str, DiscoveryResult] = field(default_factory=dict)
    last_updated: Optional[datetime] = None
    all_models: list[DiscoveredModel] = field(default_factory=list)


class DiscoveryManager:
    """
    Manages model discovery across all platforms.
    
    Features:
    - Parallel discovery from multiple platforms
    - Caching with configurable TTL
    - Callback for new model notifications
    - Graceful handling of failures
    """
    
    def __init__(
        self,
        api_keys: dict[str, str],
        config: Optional[DiscoveryConfig] = None,
    ):
        self.config = config or DiscoveryConfig()
        self.api_keys = api_keys
        self._cache = CachedDiscovery()
        self._discoverers = self._create_discoverers()
    
    def _create_discoverers(self) -> dict[str, type]:
        """Create discoverer instances based on config and available keys."""
        discoverers = {}
        
        if self.config.enable_nvidia and "nvidia_build" in self.api_keys:
            discoverers["nvidia_build"] = NvidiaDiscoverer
        if self.config.enable_groq and "groq" in self.api_keys:
            discoverers["groq"] = GroqDiscoverer
        if self.config.enable_cerebras and "cerebras" in self.api_keys:
            discoverers["cerebras"] = CerebrasDiscoverer
        if self.config.enable_sambanova and "sambanova" in self.api_keys:
            discoverers["sambanova"] = SambaNovaDiscoverer
        if self.config.enable_mistral and "mistral" in self.api_keys:
            discoverers["mistral"] = MistralDiscoverer
        
        return discoverers
    
    async def discover_all(self) -> dict[str, DiscoveryResult]:
        """
        Run discovery on all configured platforms in parallel.
        
        Returns:
            Dict mapping platform names to discovery results.
        """
        tasks = {}
        
        for platform_name, discoverer_class in self._discoverers.items():
            api_key = self.api_keys.get(platform_name)
            discoverer = discoverer_class(api_key=api_key, timeout=self.config.timeout_seconds)
            tasks[platform_name] = discoverer.discover()
        
        # Run all discoveries in parallel
        results = {}
        for platform_name, task in tasks.items():
            try:
                result = await task
                results[platform_name] = result
            except Exception as e:
                log.error(f"Discovery failed for {platform_name}: {e}")
                results[platform_name] = DiscoveryResult(
                    platform=platform_name,
                    models=[],
                    success=False,
                    error=str(e)
                )
        
        # Update cache
        self._cache.results = results
        self._cache.last_updated = datetime.utcnow()
        self._cache.all_models = self._flatten_models(results)
        
        log.info(f"Discovery complete: {len(self._cache.all_models)} models across {len(results)} platforms")
        
        return results
    
    def _flatten_models(self, results: dict[str, DiscoveryResult]) -> list[DiscoveredModel]:
        """Flatten all discovered models into a single list."""
        models = []
        for result in results.values():
            if result.success:
                models.extend(result.models)
        return models
    
    def get_models_for_platform(self, platform: str) -> list[DiscoveredModel]:
        """Get cached models for a specific platform."""
        result = self._cache.results.get(platform)
        return result.models if result and result.success else []
    
    def get_all_models(self) -> list[DiscoveredModel]:
        """Get all cached models across all platforms."""
        return self._cache.all_models
    
    def find_model(self, model_id: str) -> Optional[DiscoveredModel]:
        """Find a specific model by ID across all platforms."""
        for model in self._cache.all_models:
            if model.model_id == model_id:
                return model
        return None
    
    def get_platforms_with_model(self, model_id: str) -> list[str]:
        """Find all platforms that have a specific model."""
        platforms = []
        for result in self._cache.results.values():
            if result.success:
                for model in result.models:
                    if model.model_id == model_id:
                        platforms.append(result.platform)
        return platforms
    
    def needs_refresh(self) -> bool:
        """Check if cache needs refreshing based on TTL."""
        if not self._cache.last_updated:
            return True
        
        ttl = timedelta(hours=self.config.poll_interval_hours)
        return datetime.utcnow() - self._cache.last_updated > ttl
    
    async def refresh_if_needed(self) -> dict[str, DiscoveryResult]:
        """Refresh discovery if cache is stale."""
        if self.needs_refresh():
            return await self.discover_all()
        return self._cache.results


async def discovery_loop(
    manager: DiscoveryManager,
    callback: Optional[Callable[[list[DiscoveredModel]], None]] = None,
) -> None:
    """
    Run continuous discovery loop.
    
    Args:
        manager: DiscoveryManager instance
        callback: Optional callback for new model notifications
    """
    known_ids: set[str] = set()
    
    while True:
        await manager.discover_all()
        
        current_ids = {m.model_id for m in manager.get_all_models()}
        new_ids = current_ids - known_ids
        
        if new_ids and known_ids:  # Skip notification on first run
            new_models = [m for m in manager.get_all_models() if m.model_id in new_ids]
            log.info(f"🆕 New models discovered: {[m.model_id for m in new_models]}")
            if callback:
                callback(new_models)
        
        known_ids = current_ids
        
        # Wait for next poll
        await asyncio.sleep(manager.config.poll_interval_hours * 3600)
