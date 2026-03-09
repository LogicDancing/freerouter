"""
freerouter/tools/updater.py
Configure AI coding tools to point at FreeRouter proxy.

MVP  (v0.1): opencode
V1.0 (v0.2): Cursor, Continue, Cline  <- not yet
"""
from __future__ import annotations
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger("freerouter.tools")

FREEROUTER_BASE_URL = "http://localhost:8080/v1"
FREEROUTER_API_KEY  = "freerouter-local"
FREEROUTER_MODEL    = "auto"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, suffix=".tmp"
    ) as tf:
        tf.write(content)
        tmp = tf.name
    shutil.move(tmp, path)


def find_opencode_config() -> Optional[Path]:
    candidates = [
        Path.home() / ".config" / "opencode" / "opencode.json",
        Path.home() / ".opencode" / "opencode.json",
        Path.cwd() / "opencode.json",
    ]
    return next((p for p in candidates if p.exists()), None)


def update_opencode(config_path: Optional[Path] = None) -> tuple[bool, Path]:
    """Point opencode at FreeRouter. Creates config if missing."""
    path = config_path or find_opencode_config()
    if path is None:
        path = Path.home() / ".config" / "opencode" / "opencode.json"

    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning("Could not parse %s — will overwrite", path)

    existing.setdefault("openai", {})
    existing["openai"]["baseURL"] = FREEROUTER_BASE_URL
    existing["openai"]["apiKey"]  = FREEROUTER_API_KEY
    existing["openai"]["model"]   = FREEROUTER_MODEL

    _atomic_write(path, json.dumps(existing, indent=2))
    log.info("opencode config updated -> %s", path)
    return True, path


def revert_opencode(
    api_key: str,
    model: str = "nvidia/qwen3-coder-480b-a35b-instruct",
    config_path: Optional[Path] = None,
) -> tuple[bool, Path]:
    """Restore opencode to direct NVIDIA Build (bypass FreeRouter)."""
    path = config_path or find_opencode_config()
    if path is None:
        return False, Path()

    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}

    existing.setdefault("openai", {})
    existing["openai"]["baseURL"] = "https://integrate.api.nvidia.com/v1"
    existing["openai"]["apiKey"]  = api_key
    existing["openai"]["model"]   = model

    _atomic_write(path, json.dumps(existing, indent=2))
    log.info("opencode config reverted -> %s", path)
    return True, path


# V1.0 stubs
def update_cursor(*_, **__):   return False, Path()
def update_continue(*_, **__): return False, Path()
def update_cline(*_, **__):    return False, Path()
