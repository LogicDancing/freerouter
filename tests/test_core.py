"""
tests/test_core.py
Core logic tests — run with: pytest tests/
No network calls, all external deps mocked.
"""
import sys
import unittest.mock as mock

# Mock all external deps before any freerouter import
for mod in [
    "aiosqlite", "httpx", "fastapi", "fastapi.responses",
    "uvicorn", "typer", "yaml", "watchdog", "watchdog.observers",
    "tiktoken",
    "rich", "rich.console", "rich.live", "rich.panel",
    "rich.table", "rich.text", "rich.layout", "rich.box",
]:
    sys.modules[mod] = mock.MagicMock()

import pytest
from freerouter.core.models import ProbeStatus, ProbeResult, TaskType
from freerouter.core.config import FreeRouterConfig, ApiKeys
from freerouter.core.platforms import ALL_PLATFORMS, PLATFORM_MAP
from freerouter.core.router import Router, detect_task_type, ROUTING_TABLE
from freerouter.tools.updater import (
    update_opencode, FREEROUTER_BASE_URL, FREEROUTER_MODEL
)


# ── Platform registry ────────────────────────────────────────────────────────

def test_platform_count():
    assert len(ALL_PLATFORMS) == 5  # nvidia_build, groq, cerebras, sambanova, mistral

def test_total_models():
    total = sum(len(p.models) for p in ALL_PLATFORMS)
    assert total >= 28

def test_nvidia_has_1m_context():
    nvidia = PLATFORM_MAP["nvidia_build"]
    assert any(m.context_window >= 1_000_000 for m in nvidia.models)

def test_all_routing_table_covered():
    for task in TaskType:
        assert task in ROUTING_TABLE, f"Missing: {task}"


# ── Task type detection ──────────────────────────────────────────────────────

@pytest.mark.parametrize("content,tokens,expected", [
    ("implement binary search in Python",        50,      TaskType.CODE_GENERATION),
    ("def fib(n): # TODO",                        30,      TaskType.CODE_COMPLETION),
    ("prove sqrt(2) is irrational, step by step", 50,      TaskType.REASONING),
    ("write a REST API in Python using FastAPI",  50,      TaskType.CODE_GENERATION),
    ("plan and execute a multi-step pipeline",    50,      TaskType.AGENTIC),
    ("how are you",                               10,      TaskType.GENERAL),
    ("anything",                                  130_001, TaskType.LONG_CONTEXT),
    ("write code for a binary tree",              50,      TaskType.CODE_GENERATION),
])
def test_task_detection(content, tokens, expected):
    result = detect_task_type([{"role": "user", "content": content}], tokens)
    assert result == expected, f"Got {result} for: {content!r}"


# ── Routing decisions ────────────────────────────────────────────────────────

def _router(keys_kw, probe_data, threshold=3000):
    cfg = FreeRouterConfig()
    cfg.api_keys = ApiKeys(**keys_kw)
    cfg.routing.ttft_threshold_ms = threshold
    class FakeProber:
        results = probe_data
    return Router(cfg, FakeProber())


def test_no_probe_data_returns_first_available():
    r = _router({"groq": "k", "cerebras": "k"}, {})
    d = r.decide([{"role": "user", "content": "hello"}], 10)
    assert d.selected_platform in ("groq", "cerebras")


def test_code_completion_routes_to_cerebras():
    r = _router(
        {"cerebras": "k", "groq": "k", "nvidia_build": "k"},
        {
            "cerebras": ProbeResult("cerebras", "m", 280.0, ProbeStatus.OK),
            "groq": ProbeResult("groq", "m", 200.0, ProbeStatus.OK),  # Groq is faster
            "nvidia_build": ProbeResult("nvidia_build", "m", 4200.0, ProbeStatus.OK),
        },
    )
    d = r.decide([{"role": "user", "content": "def fib(n): # TODO"}], 30)
    # Groq is now first in routing table for code_completion
    assert d.selected_platform in ("groq", "cerebras")


def test_congested_nvidia_falls_back_to_sambanova():
    r = _router(
        {"groq": "k", "nvidia_build": "k", "sambanova": "k"},
        {
            "nvidia_build": ProbeResult("nvidia_build", "m", 9000.0, ProbeStatus.OK),
            "groq": ProbeResult("groq", "m", 400.0, ProbeStatus.OK),
            "sambanova": ProbeResult("sambanova", "m", 600.0, ProbeStatus.OK),
        },
    )
    d = r.decide([{"role": "user", "content": "write a REST API in Python"}], 50)
    # Groq is now prioritized over nvidia for code_generation
    assert d.selected_platform in ("groq", "sambanova")


def test_all_congested_returns_least_bad():
    r = _router(
        {"groq": "k", "cerebras": "k"},
        {
            "groq":     ProbeResult("groq",     "m", 1200.0, ProbeStatus.OK),
            "cerebras": ProbeResult("cerebras", "m", 1500.0, ProbeStatus.OK),
        },
        threshold=1000,
    )
    d = r.decide([{"role": "user", "content": "hello"}], 10)
    assert d.reason == "all_congested_least_bad"


def test_rate_limited_platform_skipped():
    r = _router(
        {"groq": "k", "cerebras": "k"},
        {
            "groq":     ProbeResult("groq",     "m", 9999.0, ProbeStatus.RATE_LIMITED),
            "cerebras": ProbeResult("cerebras", "m",  310.0, ProbeStatus.OK),
        },
    )
    d = r.decide([{"role": "user", "content": "hello"}], 10)
    assert d.selected_platform == "cerebras"


# ── opencode updater ─────────────────────────────────────────────────────────

def test_update_opencode_sets_correct_url(tmp_path):
    import json
    cfg_file = tmp_path / "opencode.json"
    cfg_file.write_text(json.dumps({"existingKey": "preserved"}))

    ok, out_path = update_opencode(cfg_file)
    data = json.loads(out_path.read_text())

    assert ok
    assert data["openai"]["baseURL"] == FREEROUTER_BASE_URL
    assert data["openai"]["model"]   == FREEROUTER_MODEL
    assert data["existingKey"]       == "preserved"   # no data loss


def test_update_opencode_creates_if_missing(tmp_path):
    import json
    cfg_file = tmp_path / "new_dir" / "opencode.json"
    assert not cfg_file.exists()

    ok, out_path = update_opencode(cfg_file)
    assert ok
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["openai"]["baseURL"] == FREEROUTER_BASE_URL
