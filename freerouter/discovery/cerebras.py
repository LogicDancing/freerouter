"""
freerouter/discovery/cerebras.py
Discover models from Cerebras API.
"""
from __future__ import annotations
import logging
from typing import Optional
import httpx

from .base import PlatformDiscoverer, DiscoveredModel, detect_capabilities

log = logging.getLogger("freerouter.discovery.cerebras")


class CerebrasDiscoverer(PlatformDiscoverer):
    """Discover models from Cerebras API."""
    
    platform_name = "cerebras"
    api_base = "https://api.cerebras.ai/v1"
    
    # Known models with their specs
    KNOWN_MODELS = {
        "llama3.1-8b": {
            "context_window": 131072,
            "speed_tps": 2200,
        },
        "gpt-oss-120b": {
            "context_window": 131072,
            "speed_tps": 3000,
        },
        "qwen-3-235b-a22b-instruct-2507": {
            "context_window": 131072,
            "speed_tps": 1400,
        },
        "zai-glm-4.7": {
            "context_window": 131072,
            "speed_tps": 1000,
        },
        "llama-4-scout-17b-16e": {
            "context_window": 131072,
            "speed_tps": 1800,
        },
        "qwen-3-32b": {
            "context_window": 131072,
            "speed_tps": 1600,
        },
    }
    
    async def fetch_models(self) -> list[DiscoveredModel]:
        """Fetch models from Cerebras API."""
        if not self.api_key:
            log.warning("No Cerebras API key configured, skipping discovery")
            return []
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.api_base}/models",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
        
        models = []
        for model_data in data.get("data", []):
            model_id = model_data.get("id", "")
            if not model_id:
                continue
            
            # Get known model info
            known = self.KNOWN_MODELS.get(model_id, {})
            
            # Detect capabilities
            caps = detect_capabilities(model_id, model_data)
            
            model = DiscoveredModel(
                model_id=model_id,
                platform=self.platform_name,
                context_window=known.get("context_window", 131072),
                owned_by=model_data.get("owned_by", ""),
                supports_vision=caps["supports_vision"],
                supports_tools=True,
                supports_streaming=True,
                raw=model_data
            )
            models.append(model)
        
        return models
