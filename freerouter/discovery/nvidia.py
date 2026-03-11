"""
freerouter/discovery/nvidia.py
Discover models from NVIDIA Build API.
"""
from __future__ import annotations
import logging
from typing import Optional
import httpx

from .base import PlatformDiscoverer, DiscoveredModel, detect_capabilities

log = logging.getLogger("freerouter.discovery.nvidia")


class NvidiaDiscoverer(PlatformDiscoverer):
    """Discover models from NVIDIA Build (NIM) API."""
    
    platform_name = "nvidia_build"
    api_base = "https://integrate.api.nvidia.com/v1"
    
    # Known model context windows (NVIDIA doesn't always return this)
    KNOWN_CONTEXT_WINDOWS = {
        "meta/llama-3.1-8b-instruct": 131072,
        "meta/llama-3.1-70b-instruct": 131072,
        "meta/llama-3.1-405b-instruct": 131072,
        "meta/llama-3.2-1b-instruct": 131072,
        "meta/llama-3.2-3b-instruct": 131072,
        "meta/llama-3.2-11b-vision-instruct": 131072,
        "meta/llama-3.2-90b-vision-instruct": 131072,
        "meta/llama-3.3-70b-instruct": 131072,
        "deepseek-ai/deepseek-v3": 131072,
        "deepseek-ai/deepseek-v3.1": 131072,
        "deepseek-ai/deepseek-r1": 131072,
        "deepseek-ai/deepseek-coder-6.7b-instruct": 16384,
        "qwen/qwen3-coder-480b-a35b-instruct": 262144,
        "qwen/qwen2.5-coder-32b-instruct": 131072,
        "qwen/qwen3.5-397b-a17b": 200000,
        "qwen/qwq-32b": 131072,
        "z-ai/glm5": 200000,
        "z-ai/glm4.7": 131072,
        "thudm/chatglm3-6b": 8192,
        "mistralai/mistral-7b-instruct-v0.3": 32768,
        "mistralai/mixtral-8x7b-instruct-v0.1": 32768,
        "mistralai/mistral-nemo-12b-instruct": 131072,
        "google/gemma-2-9b-it": 8192,
        "google/gemma-2-27b-it": 8192,
        "google/recurrentgemma-2b-it": 8192,
        "nvidia/llama-3.1-nemotron-nano-8b-v1": 1048576,  # 1M context!
        "nvidia/llama-3.1-nemotron-70b-instruct": 131072,
        "bigcode/starcoder2-15b": 16384,
        "bigcode/starcoder2-7b": 16384,
        "microsoft/phi-3-mini-4k-instruct": 4096,
        "microsoft/phi-3-mini-128k-instruct": 131072,
        "microsoft/phi-3.5-mini-instruct": 131072,
        "snowflake/arctic": 4096,
    }
    
    async def fetch_models(self) -> list[DiscoveredModel]:
        """Fetch models from NVIDIA Build API."""
        if not self.api_key:
            log.warning("No NVIDIA API key configured, skipping discovery")
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
            
            # Get context window from known mapping or default
            context_window = self.KNOWN_CONTEXT_WINDOWS.get(model_id, 8192)
            
            # Detect capabilities
            caps = detect_capabilities(model_id, model_data)
            
            model = DiscoveredModel(
                model_id=model_id,
                platform=self.platform_name,
                context_window=context_window,
                owned_by=model_data.get("owned_by", "nvidia"),
                supports_vision=caps["supports_vision"],
                supports_tools=model_data.get("supports_tools", True),
                supports_streaming=True,
                rpm_limit=40,  # NVIDIA free tier default
                raw=model_data
            )
            models.append(model)
        
        return models
    
    async def get_model_details(self, model_id: str) -> Optional[dict]:
        """Get detailed info for a specific model."""
        if not self.api_key:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.api_base}/models/{model_id}",
                headers=headers
            )
            if response.status_code == 200:
                return response.json()
        return None
