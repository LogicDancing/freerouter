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
from freerouter.core.smart_router import SmartRouter
from freerouter.discovery import DiscoveryManager, DiscoveryConfig

log = logging.getLogger("freerouter.proxy")


def create_app(config: FreeRouterConfig) -> FastAPI:
    app = FastAPI(
        title="FreeRouter",
        description="Free LLM Forever — auto-routing proxy",
        version="0.1.0",
    )

    store = ProbeStore(config.db_path)
    prober = Prober(config, store)

    # Initialize discovery manager for dynamic model discovery
    discovery_manager = DiscoveryManager(
        api_keys=config.api_keys.as_dict(),
        config=DiscoveryConfig()
    )
    router = SmartRouter(config, prober, discovery_manager=discovery_manager)

    @app.on_event("startup")
    async def startup():
        await store.init()
        # Run dynamic model discovery
        log.info("Discovering models from all platforms...")
        await discovery_manager.discover_all()
        model_count = len(discovery_manager.get_all_models())
        log.info("Discovered %d models across all platforms", model_count)
        # Start probing platforms
        await prober.probe_now()
        prober.start_background()
        log.info("FreeRouter started on %s:%s", config.host, config.port)

    @app.on_event("shutdown")
    async def shutdown():
        prober.stop()
        log.info("FreeRouter stopped")

    @app.get("/")
    async def dashboard():
        """Simple HTML dashboard showing available models and status."""
        from freerouter.core.platforms import ALL_PLATFORMS
        
        results = prober.results
        available_keys = config.api_keys.available_platforms()
        
        platform_rows = ""
        for platform in ALL_PLATFORMS:
            result = results.get(platform.name)
            has_key = platform.name in available_keys
            
            if not has_key:
                status_badge = '<span style="color:#888">未配置密钥</span>'
                ttft = "-"
            elif result and result.status.value == "ok":
                color = "#22c55e" if result.ttft_ms < 1000 else "#eab308"
                status_badge = f'<span style="color:{color}">✓ 可用</span>'
                ttft = f"{result.ttft_ms:.0f}ms"
            elif result and result.status.value == "rate_limited":
                status_badge = '<span style="color:#f97316">⚠ 限流中</span>'
                ttft = "-"
            else:
                error_msg = result.error[:20] if result and result.error else "错误"
                status_badge = f'<span style="color:#ef4444">✗ {error_msg}</span>'
                ttft = "-"
            
            platform_rows += f"""
            <tr>
                <td><strong>{platform.display_name}</strong></td>
                <td>{status_badge}</td>
                <td>{ttft}</td>
                <td style="color:#666;font-size:12px">{', '.join(platform.tags)}</td>
            </tr>"""
        
        model_rows = ""
        model_count = len(discovery_manager.get_all_models())
        for platform in ALL_PLATFORMS:
            if platform.name not in available_keys:
                continue
            model_rows += f"<tr><td colspan='4' style='background:#f5f5f5;padding:8px'><strong>{platform.display_name}</strong></td></tr>"
            for model in platform.models[:5]:
                task_map = {
                    "code_completion": "代码补全",
                    "code_generation": "代码生成",
                    "agentic": "智能体",
                    "reasoning": "推理",
                    "long_context": "长文本",
                    "general": "通用"
                }
                task_types = ', '.join(task_map.get(t.value, t.value) for t in model.task_types)
                model_rows += f"""
                <tr>
                    <td style="padding-left:20px;font-family:monospace;font-size:12px">{model.model_id}</td>
                    <td>{model.context_window:,} tokens</td>
                    <td>{task_types}</td>
                    <td style="color:#666">{model.description}</td>
                </tr>"""
        
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <title>FreeRouter — 永久免费的 LLM 路由</title>
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
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: #f9fafb; padding: 15px 20px; border-radius: 8px; flex: 1; }}
                .stat-card h3 {{ margin: 0 0 5px 0; color: #666; font-size: 14px; }}
                .stat-card p {{ margin: 0; font-size: 24px; font-weight: bold; color: #22c55e; }}
            </style>
        </head>
        <body>
            <a href="/" class="refresh" style="text-decoration:none;padding:8px 16px;background:#22c55e;color:white;border-radius:6px">刷新</a>
            
            <h1>🚀 FreeRouter v0.1.0</h1>
            <p style="color:#666">永久免费的 LLM 路由 — 自动选择 NVIDIA Build、Groq、Cerebras、SambaNova 中最快的平台</p>
            
            <div class="stats">
                <div class="stat-card">
                    <h3>已配置平台</h3>
                    <p>{len(available_keys)}</p>
                </div>
                <div class="stat-card">
                    <h3>可用模型</h3>
                    <p>{model_count}</p>
                </div>
                <div class="stat-card">
                    <h3>路由策略</h3>
                    <p>自动</p>
                </div>
            </div>
            
            <h2>📊 平台状态</h2>
            <table>
                <tr><th>平台</th><th>状态</th><th>延迟</th><th>标签</th></tr>
                {platform_rows}
            </table>
            
            <h2>🤖 可用模型</h2>
            <table>
                <tr><th>模型</th><th>上下文</th><th>任务类型</th><th>说明</th></tr>
                {model_rows}
            </table>
            
            <div class="endpoint">
                <h3 style="margin-top:0">📡 API 端点</h3>
                <p><code>POST http://{config.host}:{config.port}/v1/chat/completions</code></p>
                <p style="color:#666;font-size:14px">
                    使用 <code>model: "auto"</code> 让 FreeRouter 自动选择最优平台，<br>
                    或指定 <code>model: "平台/模型ID"</code> 强制使用特定模型。
                </p>
            </div>
            
            <p style="color:#999;text-align:center;margin-top:40px">Powered by FreeRouter • <a href="https://github.com/LogicDancing/freerouter">GitHub</a></p>
        </body>
        </html>
        """
        return Response(content=html, media_type="text/html")

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

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        raw_text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        estimated_tokens = len(raw_text) // 4

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
            "→ %s/%s [%s] %.0fms task=%s",
            decision.selected_platform,
            decision.selected_model,
            decision.reason,
            decision.probe_ttft_ms,
            decision.task_type.value,
        )

        platform = PLATFORM_MAP[decision.selected_platform]
        api_keys = config.api_keys.as_dict()
        api_key = api_keys.get(decision.selected_platform, "")

        forward_body = {**body, "model": decision.selected_model}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
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

        try:
            response_body = r.json()
        except json.JSONDecodeError:
            return Response(
                content=json.dumps({"error": {"message": f"Upstream error {r.status_code}: {r.text[:200]}", "type": "upstream_error"}}),
                status_code=502,
                media_type="application/json",
            )

        if r.status_code >= 400:
            return Response(
                content=json.dumps(response_body),
                status_code=r.status_code,
                media_type="application/json",
            )

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
