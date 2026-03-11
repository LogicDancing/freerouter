"""
freerouter/discovery/sambanova.py
Discover models from SambaNova API.
"""
from __future__ import annotations
import logging
from typing import Optional
import httpx

from .base import PlatformDiscoverer, DiscoveredModel, detect_capabilities

log = logging.getLogger("freerouter.discovery.sambanova")


class SambaNovaDiscoverer(PlatformDiscoverer):
    """Discover models from SambaNova API."""
    
    platform_name = "sambanova"
    api_base = "https://api.sambanova.ai/v1"
    
    # Known models with their specs and rate limits
    KNOWN_MODELS = {
        # High quota models (1440 RPM / 288000 RPD)
        "Meta-Llama-3.1-8B-Instruct": {
            "context_window": 131072,
            "rpm_limit": 1440,
            "rpd_limit": 288000,
            "speed_tps": 132,
        },
        # Medium quota models (240 RPM / 48000 RPD)
        "Meta-Llama-3.3-70B-Instruct": {
            "context_window": 131072,
            "rpm_limit": 240,
            "rpd_limit": 48000,
            "speed_tps": 100,
        },
        "DeepSeek-R1-Distill-Llama-70B": {
            "context_window": 131072,
            "rpm_limit": 240,
            "rpd_limit": 48000,
            "speed_tps": 80,
        },
        # Low quota models (60 RPM / 12000 RPD)
        "DeepSeek-V3-0324": {
            "context_window": 8192,
            "rpm_limit": 60,
            "rpd_limit": 12000,
            "speed_tps": 60,
        },
        "DeepSeek-V3.1": {
            "context_window": 131072,
            "rpm_limit": 60,
            "rpd_limit": 12000,
            "speed_tps": 60,
        },
        "DeepSeek-R1-0528": {
            "context_window": 131072,
            "rpm_limit": 60,
            "rpd_limit": 12000,
            "speed_tps": 50,
        },
        "Qwen3-32B": {
            "context_window": 131072,
            "rpm_limit": 60,
            "rpd_limit": 12000,
            "speed_tps": 70,
        },
        "Llama-4-Maverick-17B-128E-Instruct": {
            "context_window": 131072,
            "rpm_limit": 60,
            "rpd_limit": 12000,
            "speed_tps": 65,
        },
        "Qwen2.5-Coder-32B-Instruct": {
            "context_window": 16384,
            "rpm_limit": 60,
            "rpd_limit": 12000,
            "speed_tps": 70,
        },
        "QwQ-32B-Preview": {
            "context_window": 16384,
            "rpm_limit": 60,
            "rpd_limit": 12000,
            "speed_tps": 65,
        },
    }
    
    async def fetch_models(self) -> list[DiscoveredModel]:
        """Fetch models from SambaNova API."""
        if not self.api_key:
            log.warning("No SambaNova API key configured, skipping discovery")
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
            
            # Try to get context length from API response
            context_window = model_data.get("context_length", known.get("context_window", 16384))
            
            # Detect capabilities
            caps = detect_capabilities(model_id, model_data)
            
            model = DiscoveredModel(
                model_id=model_id,
                platform=self.platform_name,
                context_window=context_window,
                owned_by=model_data.get("owned_by", ""),
                supports_vision=caps["supports_vision"],
                supports_tools=True,
                supports_streaming=True,
                rpm_limit=known.get("rpm_limit"),
                rpd_limit=known.get("rpd_limit"),
                raw=model_data
            )
            models.append(model)
        
        return models
