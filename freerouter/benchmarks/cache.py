"""
freerouter/benchmarks/cache.py
Caching layer for benchmark data with persistence.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import threading

from .model_scores import (
    ModelCapabilities,
    get_model_capabilities,
    get_top_models,
)

log = logging.getLogger("freerouter.benchmarks.cache")


class BenchmarkCache:
    """
    Thread-safe cache for benchmark data with disk persistence.
    
    Features:
    - In-memory caching for fast access
    - Disk persistence for restarts
    - Automatic expiration
    - Thread-safe operations
    """
    
    DEFAULT_CACHE_TTL = timedelta(hours=24)
    CACHE_FILENAME = "benchmark_cache.json"
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl: timedelta = DEFAULT_CACHE_TTL,
    ):
        self.cache_dir = cache_dir or Path.home() / ".config" / "freerouter" / "cache"
        self.ttl = ttl
        self._cache: dict[str, ModelCapabilities] = {}
        self._lock = threading.RLock()
        self._loaded = False
    
    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_path(self) -> Path:
        """Get path to cache file."""
        return self.cache_dir / self.CACHE_FILENAME
    
    def load(self) -> None:
        """Load cache from disk."""
        with self._lock:
            if self._loaded:
                return
            
            cache_path = self._cache_path()
            if not cache_path.exists():
                log.debug("No cache file found, starting fresh")
                self._loaded = True
                return
            
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                
                # Check if cache is expired
                cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
                if datetime.utcnow() - cached_at > self.ttl:
                    log.info("Cache expired, will refresh")
                    self._loaded = True
                    return
                
                # Load capabilities
                for model_id, cap_data in data.get("capabilities", {}).items():
                    self._cache[model_id] = ModelCapabilities(
                        model_id=model_id,
                        overall_elo=cap_data.get("overall_elo"),
                        coding_score=cap_data.get("coding_score"),
                        reasoning_score=cap_data.get("reasoning_score"),
                        instruction_score=cap_data.get("instruction_score"),
                        long_context_score=cap_data.get("long_context_score"),
                        supports_vision=cap_data.get("supports_vision", False),
                        supports_tools=cap_data.get("supports_tools", True),
                        context_window=cap_data.get("context_window", 8192),
                        updated_at=datetime.fromisoformat(cap_data.get("updated_at", datetime.utcnow().isoformat())),
                    )
                
                log.info(f"Loaded {len(self._cache)} cached model capabilities")
                
            except Exception as e:
                log.warning(f"Failed to load cache: {e}")
            
            self._loaded = True
    
    def save(self) -> None:
        """Save cache to disk."""
        with self._lock:
            self._ensure_cache_dir()
            
            data = {
                "cached_at": datetime.utcnow().isoformat(),
                "capabilities": {}
            }
            
            for model_id, caps in self._cache.items():
                data["capabilities"][model_id] = {
                    "overall_elo": caps.overall_elo,
                    "coding_score": caps.coding_score,
                    "reasoning_score": caps.reasoning_score,
                    "instruction_score": caps.instruction_score,
                    "long_context_score": caps.long_context_score,
                    "supports_vision": caps.supports_vision,
                    "supports_tools": caps.supports_tools,
                    "context_window": caps.context_window,
                    "updated_at": caps.updated_at.isoformat(),
                }
            
            cache_path = self._cache_path()
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            
            log.debug(f"Saved cache to {cache_path}")
    
    def get_capabilities(self, model_id: str) -> ModelCapabilities:
        """
        Get capabilities for a model, using cache or computing fresh.
        
        Args:
            model_id: The model identifier
        
        Returns:
            ModelCapabilities for the model
        """
        with self._lock:
            self.load()
            
            # Check cache
            if model_id in self._cache:
                caps = self._cache[model_id]
                # Check if stale
                if datetime.utcnow() - caps.updated_at < self.ttl:
                    return caps
            
            # Compute fresh
            caps = get_model_capabilities(model_id)
            self._cache[model_id] = caps
            
            # Save to disk
            self.save()
            
            return caps
    
    def get_top_models_for_domain(
        self,
        domain: str,
        available_models: list[str],
        limit: int = 5
    ) -> list[tuple[str, float]]:
        """
        Get top models for a domain from available models.
        
        Args:
            domain: Domain to rank by
            available_models: List of available model IDs
            limit: Maximum models to return
        
        Returns:
            List of (model_id, score) tuples
        """
        # Get capabilities for all available models
        scored = []
        for model_id in available_models:
            caps = self.get_capabilities(model_id)
            score = caps.get_score_for_domain(domain)
            if score:
                scored.append((model_id, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:limit]
    
    def warm_up(self, model_ids: list[str]) -> None:
        """
        Pre-populate cache for a list of models.
        
        Args:
            model_ids: List of model IDs to warm up
        """
        with self._lock:
            for model_id in model_ids:
                if model_id not in self._cache:
                    self._cache[model_id] = get_model_capabilities(model_id)
            
            self.save()
            log.info(f"Warmed up cache for {len(model_ids)} models")
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            cache_path = self._cache_path()
            if cache_path.exists():
                cache_path.unlink()
            log.info("Cleared benchmark cache")


# Global cache instance
_global_cache: Optional[BenchmarkCache] = None


def get_benchmark_cache() -> BenchmarkCache:
    """Get the global benchmark cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = BenchmarkCache()
    return _global_cache
