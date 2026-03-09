"""
freerouter/proxy/server.py
FastAPI local proxy: intercepts OpenAI-compatible requests,
routes them to the best available free platform, and streams the response.
"""
from __future__ import annotations
import json
import logging
import time
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from freerouter.core.config import FreeRouterConfig
from freerouter.core.models import RoutingDecision
from freerouter.core.platforms import PLATFORM_MAP
from freerouter.core.prober import Prober, ProbeStore
from freerouter.core.router import Router

log = logging.getLogger("freerouter.proxy")


def create_app(config: FreeRouterConfig) -> FastAPI:
    app = FastAPI(
        title="FreeRouter",
        description="Free LLM Forever — auto-routing proxy",
        version="0.1.0",
    )

    store = ProbeStore(config.db_path)
    prober = Prober(config, store)
    router = Router(config, prober)

    @app.on_event("startup")
    async def startup():
        await store.init()
        # Initial probe so first request has data
        await prober.probe_now()
        # Start background probe loop
        prober.start_background()
        log.info("FreeRouter started on %s:%s", config.host, config.port)

    @app.on_event("shutdown")
    async def shutdown():
        prober.stop()

    # ── Health / status endpoints ────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/v1/status")
    async def status():
        """Return current probe results for all platforms."""
        results = prober.results
        return {
            name: {
                "ttft_ms": r.ttft_ms,
                "status": r.status.value,
                "is_available": r.is_available,
                "is_congested": r.is_congested,
                "timestamp": r.timestamp,
            }
            for name, r in results.items()
        }

    # ── OpenAI-compatible chat completions ───────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Rough token estimate (4 chars ≈ 1 token)
        raw_text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        estimated_tokens = len(raw_text) // 4

        # Parse optional override from model field: "platform/model-id"
        model_field: str = body.get("model", "auto")
        user_platform: Optional[str] = None
        user_model: Optional[str] = None
        if "/" in model_field and model_field != "auto":
            parts = model_field.split("/", 1)
            if parts[0] in PLATFORM_MAP:
                user_platform, user_model = parts[0], parts[1]

        try:
            decision = router.decide(
                messages=messages,
                estimated_tokens=estimated_tokens,
                user_override_platform=user_platform,
                user_override_model=user_model,
            )
        except RuntimeError as e:
            return Response(
                content=json.dumps({"error": {"message": str(e), "type": "router_error"}}),
                status_code=503,
                media_type="application/json",
            )

        log.info(
            "→ %s/%s  [%s]  %.0fms  task=%s",
            decision.selected_platform,
            decision.selected_model,
            decision.reason,
            decision.probe_ttft_ms,
            decision.task_type.value,
        )

        # Build forwarded request body
        platform = PLATFORM_MAP[decision.selected_platform]
        api_keys = config.api_keys.as_dict()
        api_key = api_keys.get(decision.selected_platform, "")

        forward_body = {**body, "model": decision.selected_model}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Attach routing metadata for transparency
            "X-FreeRouter-Platform": decision.selected_platform,
            "X-FreeRouter-Model": decision.selected_model,
            "X-FreeRouter-Task": decision.task_type.value,
            "X-FreeRouter-TTFT": str(int(decision.probe_ttft_ms)),
        }

        target_url = f"{platform.api_base}/chat/completions"

        if stream:
            return StreamingResponse(
                _stream_response(target_url, headers, forward_body, decision),
                media_type="text/event-stream",
                headers={
                    "X-FreeRouter-Platform": decision.selected_platform,
                    "X-FreeRouter-Model": decision.selected_model,
                },
            )
        else:
            return await _non_stream_response(target_url, headers, forward_body, decision)

    return app


async def _stream_response(
    url: str,
    headers: dict,
    body: dict,
    decision: RoutingDecision,
) -> AsyncIterator[bytes]:
    """Stream SSE response from upstream, injecting routing metadata."""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, headers=headers, json=body) as r:
            if r.status_code != 200:
                error_body = await r.aread()
                yield _sse_error(r.status_code, error_body.decode())
                return

            # Inject a comment with routing info as first SSE event
            yield f": routed-via={decision.selected_platform}/{decision.selected_model}\n\n".encode()

            async for chunk in r.aiter_bytes(chunk_size=512):
                yield chunk


async def _non_stream_response(
    url: str,
    headers: dict,
    body: dict,
    decision: RoutingDecision,
) -> Response:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
        response_body = r.json()
        # Inject routing metadata into response
        if "choices" in response_body:
            response_body["_freerouter"] = {
                "platform": decision.selected_platform,
                "model": decision.selected_model,
                "task_type": decision.task_type.value,
                "probe_ttft_ms": decision.probe_ttft_ms,
            }
        return Response(
            content=json.dumps(response_body),
            status_code=r.status_code,
            media_type="application/json",
        )


def _sse_error(status_code: int, body: str) -> bytes:
    payload = json.dumps({
        "error": {
            "message": f"Upstream returned {status_code}: {body[:200]}",
            "type": "upstream_error",
            "code": status_code,
        }
    })
    return f"data: {payload}\n\ndata: [DONE]\n\n".encode()
