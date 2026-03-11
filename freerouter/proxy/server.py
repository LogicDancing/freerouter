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

    @app.get("/")
    async def dashboard():
        """Simple HTML dashboard showing available models and status."""
        from freerouter.core.platforms import ALL_PLATFORMS
        
        results = prober.results
        available_keys = config.api_keys.available_platforms()
        
        # Build platform status HTML
        platform_rows = ""
        for platform in ALL_PLATFORMS:
            result = results.get(platform.name)
            has_key = platform.name in available_keys
            
            if not has_key:
                status_badge = '<span style="color:#888">No API Key</span>'
                ttft = "-"
            elif result and result.status.value == "ok":
                color = "#22c55e" if result.ttft_ms < 1000 else "#eab308"
                status_badge = f'<span style="color:{color}">Available</span>'
                ttft = f"{result.ttft_ms:.0f}ms"
            elif result and result.status.value == "rate_limited":
                status_badge = '<span style="color:#f97316">Rate Limited</span>'
                ttft = "-"
            else:
                status_badge = f'<span style="color:#ef4444">{result.error[:30] if result and result.error else "Error"}</span>'
                ttft = "-"
            
            platform_rows += f"""
            <tr>
                <td><strong>{platform.display_name}</strong></td>
                <td>{status_badge}</td>
                <td>{ttft}</td>
                <td style="color:#666;font-size:12px">{', '.join(platform.tags)}</td>
            </tr>"""
        
        # Build model list HTML
        model_rows = ""
        for platform in ALL_PLATFORMS:
            if platform.name not in available_keys:
                continue
            model_rows += f"<tr><td colspan='4' style='background:#f5f5f5;padding:8px'><strong>{platform.display_name}</strong></td></tr>"
            for model in platform.models[:5]:
                task_types = ', '.join(t.value for t in model.task_types)
                model_rows += f"""
                <tr>
                    <td style="padding-left:20px;font-family:monospace;font-size:12px">{model.model_id}</td>
                    <td>{model.context_window:,}</td>
                    <td>{task_types}</td>
                    <td style="color:#666">{model.description}</td>
                </tr>"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FreeRouter — Free LLM Forever</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1000px; margin: 40px auto; padding: 20px; }}
                h1 {{ color: #22c55e; margin-bottom: 10px; }}
                h2 {{ color: #333; border-bottom: 2px solid #22c55e; padding-bottom: 8px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #eee; }}
                th {{ background: #f9fafb; font-weight: 600; }}
                .endpoint {{ background: #f0fdf4; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                code {{ background: #1f2937; color: #22c55e; padding: 2px 8px; border-radius: 4px; font-family: monospace; }}
                .refresh {{ float: right; margin-top: -40px; }}
            </style>
        </head>
        <body>
            <h1>🚀 FreeRouter v0.1.0</h1>
            <p style="color:#666">Free LLM Forever — auto-route across NVIDIA Build, Groq, Cerebras & SambaNova</p>
            
            <a href="/" class="refresh" style="text-decoration:none;padding:8px 16px;background:#22c55e;color:white;border-radius:6px">Refresh</a>
            
            <h2>📊 Platform Status</h2>
            <table>
                <tr><th>Platform</th><th>Status</th><th>TTFT</th><th>Tags</th></tr>
                {platform_rows}
            </table>
            
            <h2>🤖 Available Models</h2>
            <table>
                <tr><th>Model</th><th>Context</th><th>Task Types</th><th>Notes</th></tr>
                {model_rows}
            </table>
            
            <div class="endpoint">
                <h3 style="margin-top:0">📡 API Endpoint</h3>
                <p><code>POST http://{config.host}:{config.port}/v1/chat/completions</code></p>
                <p style="color:#666;font-size:14px">Use <code>model: "auto"</code> to let FreeRouter choose the best platform, or specify <code>model: "platform/model-id"</code> to override.</p>
            </div>
            
            <p style="color:#999;text-align:center;margin-top:40px">Powered by FreeRouter • <a href="https://github.com/LogicDancing/freerouter">GitHub</a></p>
        </body>
        </html>
        """
        return Response(content=html, media_type="text/html")
    
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
        
        # Handle non-JSON responses (e.g., Cloudflare errors)
        try:
            response_body = r.json()
        except json.JSONDecodeError:
            return Response(
                content=json.dumps({"error": {"message": f"Upstream error {r.status_code}: {r.text[:200]}", "type": "upstream_error"}}),
                status_code=502,
                media_type="application/json",
            )
        
        # Handle upstream errors
        if r.status_code >= 400:
            return Response(
                content=json.dumps(response_body),
                status_code=r.status_code,
                media_type="application/json",
            )
        
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
