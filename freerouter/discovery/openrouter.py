"""
freerouter/discovery/openrouter.py
Discover free models from OpenRouter's public API.
No auth required — polls every 6 hours.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Optional

import httpx

log = logging.getLogger("freerouter.discovery")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_POLL_INTERVAL = 6 * 3600  # 6 hours


async def fetch_free_models() -> list[dict]:
    """Fetch all :free models from OpenRouter. No API key required."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(OPENROUTER_MODELS_URL)
            r.raise_for_status()
            data = r.json()
            models = data.get("data", [])
            free = [m for m in models if m.get("id", "").endswith(":free")]
            log.info("Discovered %d free models on OpenRouter", len(free))
            return free
    except Exception as e:
        log.warning("OpenRouter discovery failed: %s", e)
        return []


async def discovery_loop(callback=None) -> None:
    """
    Run the discovery loop forever.
    Optional callback(new_models: list[dict]) is called when new models are found.
    """
    known_ids: set[str] = set()
    while True:
        models = await fetch_free_models()
        current_ids = {m["id"] for m in models}
        new_ids = current_ids - known_ids
        if new_ids and known_ids:  # skip on first run
            new_models = [m for m in models if m["id"] in new_ids]
            log.info("🆕 New free models found: %s", [m["id"] for m in new_models])
            if callback:
                callback(new_models)
        known_ids = current_ids
        await asyncio.sleep(_POLL_INTERVAL)
