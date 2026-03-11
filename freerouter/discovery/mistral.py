"""
freerouter/discovery/mistral.py
Discover models from Mistral AI API.
"""
from __future__ import annotations
import logging
from typing import Optional
import httpx

from .base import PlatformDiscoverer, DiscoveredModel, detect_capabilities

log = logging.getLogger("freerouter.discovery.mistral")


class MistralDiscoverer(PlatformDiscoverer):
    """Discover models from Mistral AI API."""
    
    platform_name = "mistral"
    api_base = "https://api.mistral.ai/v1"
    
    # Known models with their specs
    # Free tier: 2 RPM, 1B tokens/month
    KNOWN_MODELS = {
        # Frontier Models - Generalist
        "mistral-large-3-2512": {
            "context_window": 131072,
            "description": "Mistral Large 3 - Open-weight multimodal flagship",
            "is_open": True,
        },
        "mistral-medium-3-1-2508": {
            "context_window": 131072,
            "description": "Mistral Medium 3.1 - Frontier multimodal",
        },
        "mistral-small-3-2-2506": {
            "context_window": 131072,
            "description": "Mistral Small 3.2 - Open-weight update",
            "is_open": True,
        },
        # Ministral family
        "ministral-3-14b-2512": {
            "context_window": 131072,
            "description": "Ministral 3 14B - Open-weight",
            "is_open": True,
        },
        "ministral-3-8b-2512": {
            "context_window": 131072,
            "description": "Ministral 3 8B - Open-weight",
            "is_open": True,
        },
        "ministral-3-3b-2512": {
            "context_window": 131072,
            "description": "Ministral 3 3B - Tiny efficient",
            "is_open": True,
        },
        # Reasoning Models
        "magistral-medium-1-2-2509": {
            "context_window": 131072,
            "description": "Magistral Medium 1.2 - Reasoning model",
        },
        "magistral-small-1-2-2509": {
            "context_window": 131072,
            "description": "Magistral Small 1.2 - Open reasoning",
            "is_open": True,
        },
        # Code Models
        "codestral-2508": {
            "context_window": 262144,
            "description": "Codestral - Code generation specialist",
            "is_coder": True,
        },
        "devstral-2-2512": {
            "context_window": 131072,
            "description": "Devstral 2 - Code agents model",
            "is_open": True,
            "is_coder": True,
        },
        "devstral-small-2-2512": {
            "context_window": 131072,
            "description": "Devstral Small 2 - SWE agent",
            "is_open": True,
            "is_coder": True,
        },
        "devstral-medium-1-0-2507": {
            "context_window": 131072,
            "description": "Devstral Medium 1.0 - Enterprise SWE",
            "is_coder": True,
        },
        # Legacy but still useful
        "mistral-nemo-12b-2407": {
            "context_window": 131072,
            "description": "Mistral Nemo 12B - Best multilingual open",
            "is_open": True,
        },
        "pixtral-large-2411": {
            "context_window": 131072,
            "description": "Pixtral Large - Multimodal frontier",
            "supports_vision": True,
        },
    }
    
    async def fetch_models(self) -> list[DiscoveredModel]:
        """Fetch models from Mistral API."""
        if not self.api_key:
            log.warning("No Mistral API key configured, skipping discovery")
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
                context_window=known.get("context_window", 32768),
                owned_by=model_data.get("owned_by", "mistralai"),
                supports_vision=known.get("supports_vision", caps["supports_vision"]),
                supports_tools=True,  # Mistral supports function calling
                supports_streaming=True,
                rpm_limit=2,  # Free tier: 2 RPM
                tpd_limit=1_000_000_000,  # Free tier: 1B tokens/month
                raw=model_data
            )
            models.append(model)
        
        return models
