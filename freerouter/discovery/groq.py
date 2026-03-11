"""
freerouter/discovery/groq.py
Discover models from Groq API.
"""
from __future__ import annotations
import logging
from typing import Optional
import httpx

from .base import PlatformDiscoverer, DiscoveredModel, detect_capabilities

log = logging.getLogger("freerouter.discovery.groq")


class GroqDiscoverer(PlatformDiscoverer):
    """Discover models from Groq API."""
    
    platform_name = "groq"
    api_base = "https://api.groq.com/openai/v1"
    
    # Known model context windows and rate limits
    KNOWN_MODELS = {
        "llama-3.1-8b-instant": {
            "context_window": 131072,
            "rpd_limit": 14400,
            "rpm_limit": 30,
            "tpm_limit": 6000,
        },
        "llama-3.3-70b-versatile": {
            "context_window": 131072,
            "rpd_limit": 1000,
            "rpm_limit": 30,
            "tpm_limit": 12000,
        },
        "llama-4-scout-17b-16e-instruct": {
            "context_window": 131072,
            "rpd_limit": 1000,
            "rpm_limit": 30,
            "tpm_limit": 30000,
        },
        "llama-4-maverick-17b-128e-instruct": {
            "context_window": 131072,
            "rpd_limit": 1000,
            "rpm_limit": 30,
            "tpm_limit": 6000,
        },
        "qwen-qwq-32b": {
            "context_window": 131072,
            "rpd_limit": 1000,
            "rpm_limit": 30,
            "tpm_limit": 6000,
        },
        "deepseek-r1-distill-llama-70b": {
            "context_window": 131072,
            "rpd_limit": 1000,
            "rpm_limit": 30,
            "tpm_limit": 6000,
        },
        "gemma-2-9b-it": {
            "context_window": 8192,
            "rpd_limit": 14400,
            "rpm_limit": 30,
            "tpm_limit": 6000,
        },
        "moonshotai/kimi-k2-instruct": {
            "context_window": 131072,
            "rpd_limit": 1000,
            "rpm_limit": 30,
            "tpm_limit": 10000,
        },
        "openai/gpt-oss-120b": {
            "context_window": 131072,
            "rpd_limit": 1000,
            "rpm_limit": 30,
            "tpm_limit": 8000,
        },
        "openai/gpt-oss-20b": {
            "context_window": 131072,
            "rpd_limit": 1000,
            "rpm_limit": 30,
            "tpm_limit": 8000,
        },
        "meta-llama/llama-guard-4-12b": {
            "context_window": 131072,
            "rpd_limit": 14400,
            "rpm_limit": 30,
            "tpm_limit": 15000,
        },
        "groq/compound": {
            "context_window": 131072,
            "rpd_limit": 250,
            "rpm_limit": 30,
            "tpm_limit": 70000,
        },
        "groq/compound-mini": {
            "context_window": 131072,
            "rpd_limit": 250,
            "rpm_limit": 30,
            "tpm_limit": 70000,
        },
    }
    
    async def fetch_models(self) -> list[DiscoveredModel]:
        """Fetch models from Groq API."""
        if not self.api_key:
            log.warning("No Groq API key configured, skipping discovery")
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
            
            # Get known model info or use defaults
            known = self.KNOWN_MODELS.get(model_id, {})
            
            # Detect capabilities
            caps = detect_capabilities(model_id, model_data)
            
            model = DiscoveredModel(
                model_id=model_id,
                platform=self.platform_name,
                context_window=known.get("context_window", 8192),
                owned_by=model_data.get("owned_by", "groq"),
                supports_vision=caps["supports_vision"],
                supports_tools=True,  # Groq models support tool calling
                supports_streaming=True,
                rpm_limit=known.get("rpm_limit", 30),
                rpd_limit=known.get("rpd_limit"),
                raw=model_data
            )
            models.append(model)
        
        return models
