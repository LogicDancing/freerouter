"""
freerouter/core/prober.py
Async probe engine: measures TTFT for all platforms every N seconds.
Stores results in SQLite. Used by the router to make decisions.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Optional

import aiosqlite
import httpx

from freerouter.core.config import FreeRouterConfig
from freerouter.core.models import ProbeResult, ProbeStatus
from freerouter.core.platforms import ALL_PLATFORMS, Platform, PLATFORM_MAP

log = logging.getLogger("freerouter.prober")

# Platform-specific rate limit handling
PLATFORM_RATE_LIMITS = {
    "sambanova": {"cooldown": 180, "read_retry_after": True},  # 3min cooldown
    "groq": {"cooldown": 60, "read_retry_after": True},
    "cerebras": {"cooldown": 60, "read_retry_after": False},
    "nvidia_build": {"cooldown": 60, "read_retry_after": False},
    "mistral": {"cooldown": 60, "read_retry_after": False},
}


class ProbeStore:
    """SQLite-backed store for probe results."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS probes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    ttft_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    ts REAL NOT NULL,
                    cooldown_until REAL
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_platform_ts ON probes(platform, ts)"
            )
            await db.commit()

    async def save(self, result: ProbeResult, cooldown_until: Optional[float] = None) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO probes(platform, model_id, ttft_ms, status, error, ts, cooldown_until) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (result.platform_name, result.model_id, result.ttft_ms,
                 result.status.value, result.error, result.timestamp, cooldown_until)
            )
            await db.commit()

    async def latest(self, platform_name: str) -> Optional[ProbeResult]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT platform, model_id, ttft_ms, status, error, ts, cooldown_until "
                "FROM probes WHERE platform=? ORDER BY ts DESC LIMIT 1",
                (platform_name,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                
                cooldown_until = row[6]
                if cooldown_until and time.time() < cooldown_until:
                    return ProbeResult(
                        platform_name=row[0],
                        model_id=row[1],
                        ttft_ms=9999,
                        status=ProbeStatus.RATE_LIMITED,
                        error="in cooldown",
                        timestamp=row[5],
                    )
                
                return ProbeResult(
                    platform_name=row[0],
                    model_id=row[1],
                    ttft_ms=row[2],
                    status=ProbeStatus(row[3]),
                    error=row[4],
                    timestamp=row[5],
                )

    async def all_latest(self) -> dict[str, ProbeResult]:
        results = {}
        for platform in ALL_PLATFORMS:
            r = await self.latest(platform.name)
            if r:
                results[platform.name] = r
        return results


async def _probe_one(
    platform: Platform,
    api_key: str,
    client: httpx.AsyncClient,
    timeout: float = 10.0,
) -> tuple[ProbeResult, Optional[float]]:
    """Send a minimal 1-token streaming request and measure TTFT."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": platform.probe_model,
        "messages": [{"role": "user", "content": "1"}],
        "max_tokens": 1,
        "stream": True,
    }
    start = time.monotonic()
    
    platform_config = PLATFORM_RATE_LIMITS.get(platform.name, {"cooldown": 60, "read_retry_after": False})
    default_cooldown = platform_config.get("cooldown", 60)
    
    try:
        async with client.stream(
            "POST",
            f"{platform.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        ) as response:
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                if retry_after and platform_config.get("read_retry_after"):
                    try:
                        cooldown_seconds = int(retry_after)
                    except ValueError:
                        cooldown_seconds = default_cooldown
                else:
                    cooldown_seconds = default_cooldown
                
                cooldown_until = time.time() + cooldown_seconds
                
                return ProbeResult(
                    platform_name=platform.name,
                    model_id=platform.probe_model,
                    ttft_ms=9999,
                    status=ProbeStatus.RATE_LIMITED,
                    cooldown_until=cooldown_until,
                ), cooldown_until
            
            if response.status_code >= 500:
                return ProbeResult(
                    platform_name=platform.name,
                    model_id=platform.probe_model,
                    ttft_ms=9999,
                    status=ProbeStatus.ERROR,
                    error=f"HTTP {response.status_code}",
                ), None
            
            async for _ in response.aiter_bytes(chunk_size=1):
                ttft_ms = (time.monotonic() - start) * 1000
                return ProbeResult(
                    platform_name=platform.name,
                    model_id=platform.probe_model,
                    ttft_ms=ttft_ms,
                    status=ProbeStatus.OK,
                ), None
            
            return ProbeResult(
                platform_name=platform.name,
                model_id=platform.probe_model,
                ttft_ms=9999,
                status=ProbeStatus.ERROR,
                error="Empty response",
            ), None
            
    except httpx.TimeoutException:
        return ProbeResult(
            platform_name=platform.name,
            model_id=platform.probe_model,
            ttft_ms=timeout * 1000,
            status=ProbeStatus.ERROR,
            error="Timeout",
        ), None
    except Exception as e:
        return ProbeResult(
            platform_name=platform.name,
            model_id=platform.probe_model,
            ttft_ms=9999,
            status=ProbeStatus.ERROR,
            error=str(e),
        ), None


class Prober:
    """Runs probe loops for all configured platforms."""

    def __init__(self, config: FreeRouterConfig, store: ProbeStore):
        self.config = config
        self.store = store
        self._results: dict[str, ProbeResult] = {}
        self._task: Optional[asyncio.Task] = None

    @property
    def results(self) -> dict[str, ProbeResult]:
        return self._results

    async def probe_now(self) -> dict[str, ProbeResult]:
        """Probe all available platforms. Skip platforms in cooldown."""
        api_keys = self.config.api_keys.as_dict()
        current_time = time.time()
        
        # Load latest results from DB to check cooldowns
        for platform_name in api_keys.keys():
            if platform_name in PLATFORM_MAP:
                latest = await self.store.latest(platform_name)
                if latest:
                    self._results[platform_name] = latest
        
        # Filter platforms: must have key and not in cooldown
        platforms_to_probe = []
        for name, key in api_keys.items():
            if not key or name not in PLATFORM_MAP:
                continue
            
            if name in self._results:
                result = self._results[name]
                if result.status == ProbeStatus.RATE_LIMITED:
                    if hasattr(result, 'cooldown_until') and result.cooldown_until and current_time < result.cooldown_until:
                        log.debug(f"Skipping {name} (in cooldown)")
                        continue
            
            platforms_to_probe.append(PLATFORM_MAP[name])

        if not platforms_to_probe:
            return self._results

        # Probe sequentially
        async with httpx.AsyncClient() as client:
            for platform in platforms_to_probe:
                result, cooldown_until = await _probe_one(
                    platform, api_keys[platform.name], client
                )
                
                self._results[result.platform_name] = result
                await self.store.save(result, cooldown_until)
                
                log.debug(
                    "%s %s %.0fms [%s]",
                    "OK" if result.status == ProbeStatus.OK else "FAIL",
                    result.platform_name,
                    result.ttft_ms, result.status.value
                )
                
                await asyncio.sleep(1)

        return self._results

    async def start_loop(self) -> None:
        """Run probe loop forever at configured interval."""
        log.info("Prober started (interval=%ds)", self.config.routing.probe_interval)
        
        # Wait before first probe
        await asyncio.sleep(10)
        
        while True:
            try:
                # Load latest from DB first
                latest = await self.store.all_latest()
                for name, r in latest.items():
                    self._results[name] = r
                
                await self.probe_now()
            except Exception as e:
                log.error("Probe loop error: %s", e)
            
            await asyncio.sleep(self.config.routing.probe_interval)

    def start_background(self) -> asyncio.Task:
        self._task = asyncio.create_task(self.start_loop())
        return self._task

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
