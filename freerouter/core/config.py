"""
freerouter/core/config.py
Configuration loading, validation, and defaults.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "freerouter" / "freerouter.yaml"


@dataclass
class RoutingConfig:
    strategy: str = "auto"          # auto | speed | quality | quota
    probe_interval: int = 30        # seconds
    ttft_threshold_ms: float = 3000 # ms; above = congested
    fallback_to_openrouter: bool = True
    max_retries: int = 3


@dataclass
class ToolsConfig:
    opencode: bool = True
    cursor: bool = True
    continue_dev: bool = True
    cline: bool = True
    # Auto-detected paths; override if needed
    opencode_config: Optional[str] = None
    cursor_config: Optional[str] = None
    continue_config: Optional[str] = None


@dataclass
class ApiKeys:
    nvidia_build: Optional[str] = None
    groq: Optional[str] = None
    cerebras: Optional[str] = None
    sambanova: Optional[str] = None
    openrouter: Optional[str] = None

    def as_dict(self) -> dict[str, Optional[str]]:
        return {
            "nvidia_build": self.nvidia_build,
            "groq": self.groq,
            "cerebras": self.cerebras,
            "sambanova": self.sambanova,
            "openrouter": self.openrouter,
        }

    def available_platforms(self) -> list[str]:
        """Return platform names that have an API key configured."""
        mapping = {
            "nvidia_build": self.nvidia_build,
            "groq": self.groq,
            "cerebras": self.cerebras,
            "sambanova": self.sambanova,
        }
        return [name for name, key in mapping.items() if key]


@dataclass
class FreeRouterConfig:
    api_keys: ApiKeys = field(default_factory=ApiKeys)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    port: int = 8080
    host: str = "127.0.0.1"
    db_path: str = str(Path.home() / ".config" / "freerouter" / "probes.db")
    log_level: str = "INFO"


def _load_api_keys_from_env(keys: ApiKeys) -> ApiKeys:
    """Override config keys with environment variables if set."""
    if v := os.getenv("NVIDIA_API_KEY"):
        keys.nvidia_build = v
    if v := os.getenv("GROQ_API_KEY"):
        keys.groq = v
    if v := os.getenv("CEREBRAS_API_KEY"):
        keys.cerebras = v
    if v := os.getenv("SAMBANOVA_API_KEY"):
        keys.sambanova = v
    if v := os.getenv("OPENROUTER_API_KEY"):
        keys.openrouter = v
    return keys


def load_config(path: Optional[Path] = None) -> FreeRouterConfig:
    """Load config from YAML file, falling back to env vars and defaults."""
    config_path = path or DEFAULT_CONFIG_PATH
    cfg = FreeRouterConfig()

    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        # API keys
        if keys_data := data.get("api_keys", {}):
            cfg.api_keys = ApiKeys(
                nvidia_build=keys_data.get("nvidia_build"),
                groq=keys_data.get("groq"),
                cerebras=keys_data.get("cerebras"),
                sambanova=keys_data.get("sambanova"),
                openrouter=keys_data.get("openrouter"),
            )

        # Routing
        if r := data.get("routing", {}):
            cfg.routing = RoutingConfig(
                strategy=r.get("strategy", "auto"),
                probe_interval=r.get("probe_interval", 30),
                ttft_threshold_ms=r.get("ttft_threshold_ms", 3000),
                fallback_to_openrouter=r.get("fallback_to_openrouter", True),
                max_retries=r.get("max_retries", 3),
            )

        # Tools
        if t := data.get("tools", {}):
            cfg.tools = ToolsConfig(
                opencode=t.get("opencode", True),
                cursor=t.get("cursor", True),
                continue_dev=t.get("continue", True),
                cline=t.get("cline", True),
                opencode_config=t.get("opencode_config"),
                cursor_config=t.get("cursor_config"),
                continue_config=t.get("continue_config"),
            )

        cfg.port = data.get("port", 8080)
        cfg.host = data.get("host", "127.0.0.1")
        cfg.log_level = data.get("log_level", "INFO")

    # Env vars always win
    cfg.api_keys = _load_api_keys_from_env(cfg.api_keys)
    return cfg


def save_default_config(path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Write a commented example config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    template = """\
# FreeRouter Configuration
# Free LLM Forever — https://github.com/yourname/freerouter
#
# API keys can also be set via environment variables:
#   NVIDIA_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY,
#   SAMBANOVA_API_KEY, OPENROUTER_API_KEY

api_keys:
  nvidia_build: ""   # https://build.nvidia.com  (no credit card)
  groq: ""           # https://console.groq.com  (no credit card)
  cerebras: ""       # https://cloud.cerebras.ai (no credit card)
  sambanova: ""      # https://cloud.sambanova.ai(no credit card)
  openrouter: ""     # https://openrouter.ai     (fallback, free models)

routing:
  strategy: auto          # auto | speed | quality | quota
  probe_interval: 30      # probe all platforms every N seconds
  ttft_threshold_ms: 3000 # mark platform congested if TTFT > this
  fallback_to_openrouter: true
  max_retries: 3

tools:
  opencode: true
  cursor: true
  continue: true
  cline: true

port: 8080
host: 127.0.0.1
log_level: INFO
"""
    path.write_text(template)
