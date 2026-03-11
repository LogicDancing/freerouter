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
                    ts REAL NOT NULL
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_platform_ts ON probes(platform, ts)"
            )
            await db.commit()

    async def save(self, result: ProbeResult) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO probes(platform, model_id, ttft_ms, status, error, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (result.platform_name, result.model_id, result.ttft_ms,
                 result.status.value, result.error, result.timestamp)
            )
            await db.commit()

    async def latest(self, platform_name: str) -> Optional[ProbeResult]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT platform, model_id, ttft_ms, status, error, ts "
                "FROM probes WHERE platform=? ORDER BY ts DESC LIMIT 1",
                (platform_name,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
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
) -> ProbeResult:
    """Send a minimal 1-token streaming request and measure TTFT."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": platform.probe_model,
        "messages": [{"role": "user", "content": "1"}],
        "max_tokens": 1,
        "stream": True,
    }
    start = time.monotonic()
    try:
        async with client.stream(
            "POST",
            f"{platform.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        ) as response:
            if response.status_code == 429:
                # Mark the probe model as rate limited
                try:
                    from freerouter.core.model_selector import mark_rate_limited
                    mark_rate_limited(platform.name, platform.probe_model)
                except ImportError:
                    pass
                return ProbeResult(
                    platform_name=platform.name,
                    model_id=platform.probe_model,
                    ttft_ms=9999,
                    status=ProbeStatus.RATE_LIMITED,
                    cooldown_until=time.time() + 60,
                )
            if response.status_code >= 500:
                return ProbeResult(
                    platform_name=platform.name,
                    model_id=platform.probe_model,
                    ttft_ms=9999,
                    status=ProbeStatus.ERROR,
                    error=f"HTTP {response.status_code}",
                )
            async for _ in response.aiter_bytes(chunk_size=1):
                ttft_ms = (time.monotonic() - start) * 1000
                return ProbeResult(
                    platform_name=platform.name,
                    model_id=platform.probe_model,
                    ttft_ms=ttft_ms,
                    status=ProbeStatus.OK,
                )
            # Empty response
            return ProbeResult(
                platform_name=platform.name,
                model_id=platform.probe_model,
                ttft_ms=9999,
                status=ProbeStatus.ERROR,
                error="Empty response",
            )
    except httpx.TimeoutException:
        return ProbeResult(
            platform_name=platform.name,
            model_id=platform.probe_model,
            ttft_ms=timeout * 1000,
            status=ProbeStatus.ERROR,
            error="Timeout",
        )
    except Exception as e:
        return ProbeResult(
            platform_name=platform.name,
            model_id=platform.probe_model,
            ttft_ms=9999,
            status=ProbeStatus.ERROR,
            error=str(e),
        )


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
        """Probe all available platforms concurrently. Returns results dict."""
        api_keys = self.config.api_keys.as_dict()
        platforms_to_probe = [
            PLATFORM_MAP[name]
            for name, key in api_keys.items()
            if key and name in PLATFORM_MAP
        ]

        if not platforms_to_probe:
            log.warning("No API keys configured. Run 'freerouter init' to set up.")
            return {}

        async with httpx.AsyncClient() as client:
            tasks = [
                _probe_one(platform, api_keys[platform.name], client)
                for platform in platforms_to_probe
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    log.error("Probe exception: %s", result)
                    continue
                self._results[result.platform_name] = result
                await self.store.save(result)
                status_icon = "OK" if result.status == ProbeStatus.OK else "FAIL"
                log.debug(
                    "%s %s %.0fms [%s]",
                    status_icon, result.platform_name,
                    result.ttft_ms, result.status.value
                )

            return self._results

    async def start_loop(self) -> None:
        """Run probe loop forever at configured interval."""
        log.info("Prober started (interval=%ds)", self.config.routing.probe_interval)
        while True:
            try:
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
