"""
freerouter/cli.py
MVP CLI: init · start · probe · status · setup
"""
from __future__ import annotations
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(
    name="freerouter",
    help="[bold green]Free LLM Forever[/bold green] — auto-route across NVIDIA Build, Groq, Cerebras & SambaNova",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

BANNER = """
[bold green]FreeRouter[/bold green] [dim]v0.1.0[/dim]
[dim]Free LLM Forever — route smart, build free[/dim]
[dim]NVIDIA Build · Groq · Cerebras · SambaNova[/dim]
"""


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ─── init ────────────────────────────────────────────────────────────────────

@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
):
    """
    Interactive setup: save API keys and configure opencode.

    All platforms are FREE — no credit card required.
    """
    from freerouter.core.config import DEFAULT_CONFIG_PATH, save_default_config

    console.print(BANNER)

    if DEFAULT_CONFIG_PATH.exists() and not force:
        console.print(f"[yellow]Config already exists:[/yellow] {DEFAULT_CONFIG_PATH}")
        console.print("Run with [bold]--force[/bold] to reconfigure.\n")
        raise typer.Exit()

    console.print("[bold]Step 1 of 2 — API Keys[/bold]")
    console.print("[dim]Get free keys (no credit card) at:[/dim]")
    console.print("  NVIDIA Build  →  [cyan]https://build.nvidia.com[/cyan]")
    console.print("  Groq          →  [cyan]https://console.groq.com/keys[/cyan]")
    console.print("  Cerebras      →  [cyan]https://cloud.cerebras.ai[/cyan]")
    console.print("  SambaNova     →  [cyan]https://cloud.sambanova.ai[/cyan]\n")

    keys: dict[str, str] = {}
    platform_defs = [
        ("NVIDIA Build",  "nvidia_build", "NVIDIA_API_KEY"),
        ("Groq",          "groq",         "GROQ_API_KEY"),
        ("Cerebras",      "cerebras",     "CEREBRAS_API_KEY"),
        ("SambaNova",     "sambanova",    "SAMBANOVA_API_KEY"),
    ]

    for display, key_name, env_var in platform_defs:
        # Check env var first
        env_val = os.getenv(env_var, "")
        if env_val:
            console.print(f"  [green]✓[/green]  {display:<14} (from env {env_var})")
            keys[key_name] = env_val
            continue

        val = typer.prompt(
            f"  {display:<14} API key (Enter to skip)",
            default="",
            hide_input=True,
            prompt_suffix=" > ",
        ).strip()
        if val:
            keys[key_name] = val
            console.print(f"  [green]✓[/green]  {display} saved")
        else:
            console.print(f"  [dim]–  {display} skipped[/dim]")

    if not keys:
        console.print("\n[red]No keys entered.[/red] Set environment variables and run again:")
        console.print("  export NVIDIA_API_KEY=nvapi-...")
        console.print("  export GROQ_API_KEY=gsk_...")
        raise typer.Exit(1)

    console.print(f"\n[dim]Configured platforms: {', '.join(keys.keys())}[/dim]")

    # Write config
    save_default_config(DEFAULT_CONFIG_PATH)
    # Inject keys into config file
    cfg_text = DEFAULT_CONFIG_PATH.read_text()
    for k, v in keys.items():
        cfg_text = cfg_text.replace(f'{k}: ""', f'{k}: "{v}"')
    DEFAULT_CONFIG_PATH.write_text(cfg_text)

    console.print(f"\n[green]✓[/green] Config saved: [cyan]{DEFAULT_CONFIG_PATH}[/cyan]")

    # Step 2: configure opencode
    console.print("\n[bold]Step 2 of 2 — Configure opencode[/bold]")
    do_setup = typer.confirm("  Configure opencode to use FreeRouter?", default=True)
    if do_setup:
        from freerouter.tools.updater import update_opencode
        ok, path = update_opencode()
        if ok:
            console.print(f"  [green]✓[/green] opencode configured → [cyan]{path}[/cyan]")
    else:
        console.print("  [dim]Skipped. Run [bold]freerouter setup[/bold] later.[/dim]")

    console.print(Panel(
        "[bold]You're ready![/bold]\n\n"
        "  Start the proxy:  [bold cyan]freerouter start[/bold cyan]\n"
        "  Check status:     [bold cyan]freerouter status[/bold cyan]\n"
        "  Probe platforms:  [bold cyan]freerouter probe[/bold cyan]",
        border_style="green",
    ))


# ─── setup ───────────────────────────────────────────────────────────────────

@app.command()
def setup():
    """Configure opencode to use FreeRouter as its LLM backend."""
    from freerouter.tools.updater import update_opencode, find_opencode_config

    existing = find_opencode_config()
    if existing:
        console.print(f"Found opencode config: [cyan]{existing}[/cyan]")
    else:
        console.print("[dim]opencode config not found — will create default location[/dim]")

    ok, path = update_opencode()
    if ok:
        console.print(f"[green]✓[/green] opencode now points to FreeRouter")
        console.print(f"  Config: [cyan]{path}[/cyan]")
        console.print(f"  base_url: [cyan]http://localhost:8080/v1[/cyan]")
        console.print(f"  model: [cyan]auto[/cyan] (FreeRouter decides)\n")
        console.print("[dim]Start FreeRouter with: [bold]freerouter start[/bold][/dim]")
    else:
        console.print("[red]Failed to update opencode config.[/red]")
        raise typer.Exit(1)


# ─── start ───────────────────────────────────────────────────────────────────

@app.command()
def start(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    port: int   = typer.Option(8080,        "--port",  "-p"),
    host: str   = typer.Option("127.0.0.1", "--host"),
    log_level: str = typer.Option("WARNING", "--log-level", "-l",
                                  help="WARNING keeps output clean; use INFO to debug"),
    no_dashboard: bool = typer.Option(False, "--no-dashboard"),
):
    """
    Start the FreeRouter proxy server.

    Your tools send requests to localhost:8080/v1 —
    FreeRouter routes them to the fastest free platform automatically.
    """
    _setup_logging(log_level)

    from freerouter.core.config import load_config
    cfg = load_config(config)
    cfg.port = port
    cfg.host = host

    available = cfg.api_keys.available_platforms()
    if not available:
        console.print("[bold red]No API keys found.[/bold red]")
        console.print("Run [bold cyan]freerouter init[/bold cyan] first.\n")
        raise typer.Exit(1)

    console.print(BANNER)
    console.print(f"  Proxy     → [bold cyan]http://{host}:{port}/v1[/bold cyan]")
    console.print(f"  Platforms → [cyan]{', '.join(available)}[/cyan]")
    console.print(f"  Strategy  → [cyan]{cfg.routing.strategy}[/cyan]  "
                  f"(probe every {cfg.routing.probe_interval}s)")
    console.print(f"  Threshold → [cyan]{cfg.routing.ttft_threshold_ms:.0f}ms[/cyan] "
                  f"(above = congested)\n")

    import uvicorn
    from freerouter.proxy.server import create_app

    async def _run():
        from freerouter.core.prober import Prober, ProbeStore

        store = ProbeStore(cfg.db_path)
        await store.init()
        prober = Prober(cfg, store)

        console.print("[dim]Probing platforms...[/dim]")
        results = await prober.probe_now()
        _print_probe_table(results)
        console.print()

        probe_task = prober.start_background()

        if not no_dashboard:
            from freerouter.tui.dashboard import live_dashboard
            server_config = uvicorn.Config(
                create_app(cfg), host=host, port=port,
                log_level=log_level.lower(), access_log=False,
            )
            server = uvicorn.Server(server_config)
            await asyncio.gather(
                server.serve(),
                live_dashboard(prober),
            )
        else:
            uvicorn.run(
                create_app(cfg), host=host, port=port,
                log_level=log_level.lower(), access_log=False,
            )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n[dim]FreeRouter stopped.[/dim]")


# ─── probe ───────────────────────────────────────────────────────────────────

@app.command()
def probe(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    watch:  bool = typer.Option(False, "--watch", "-w", help="Keep probing every 10s"),
):
    """
    Probe all platforms and show current latency ranking.

    Use --watch for a live-updating view.
    """
    from freerouter.core.config import load_config
    from freerouter.core.prober import Prober, ProbeStore

    cfg = load_config(config)

    async def _run():
        store = ProbeStore(cfg.db_path)
        await store.init()
        prober = Prober(cfg, store)

        if watch:
            console.print("[dim]Probing every 10s — Ctrl+C to stop[/dim]\n")
            while True:
                results = await prober.probe_now()
                console.clear()
                _print_probe_table(results)
                await asyncio.sleep(10)
        else:
            console.print("[dim]Probing...[/dim]")
            results = await prober.probe_now()
            _print_probe_table(results)

            available = [
                (name, r) for name, r in results.items()
                if r.status.value == "ok"
            ]
            if available:
                best = min(available, key=lambda x: x[1].ttft_ms)
                console.print(
                    f"\n[bold green]Fastest right now:[/bold green] "
                    f"[cyan]{best[0]}[/cyan]  ({best[1].ttft_ms:.0f}ms TTFT)"
                )

    asyncio.run(_run())


# ─── status ──────────────────────────────────────────────────────────────────

@app.command()
def status():
    """Check if FreeRouter is running and show current routing state."""
    try:
        import httpx
        r = httpx.get("http://localhost:8080/v1/status", timeout=3)
        r.raise_for_status()
        data = r.json()
        console.print("\n[bold green]FreeRouter is running[/bold green]  "
                      "[dim]http://localhost:8080/v1[/dim]\n")
        for name, info in data.items():
            icon  = "[green]✓[/green]" if info["is_available"] else "[red]✗[/red]"
            ttft  = f"[green]{info['ttft_ms']:.0f}ms[/green]" \
                    if info["is_available"] and info["ttft_ms"] < 3000 \
                    else f"[yellow]{info['ttft_ms']:.0f}ms[/yellow]" \
                    if info["is_available"] else f"[dim]{info['status']}[/dim]"
            console.print(f"  {icon}  {name:<20} {ttft}")
        console.print()
    except Exception:
        console.print("[red]FreeRouter is not running.[/red]")
        console.print("Start with: [bold cyan]freerouter start[/bold cyan]")
        raise typer.Exit(1)


# ─── Shared display helper ────────────────────────────────────────────────────

def _print_probe_table(results: dict) -> None:
    from rich.table import Table
    from rich import box as rbox
    from freerouter.core.models import ProbeStatus
    from freerouter.core.platforms import ALL_PLATFORMS

    table = Table(box=rbox.SIMPLE_HEAD, show_header=True, header_style="bold dim")
    table.add_column("Platform",   min_width=16)
    table.add_column("TTFT",       justify="right", min_width=10)
    table.add_column("Status",     min_width=14)
    table.add_column("Tags",       style="dim")

    for platform in ALL_PLATFORMS:
        r = results.get(platform.name)
        if r is None:
            table.add_row(platform.display_name, "–", "[dim]no key[/dim]", "")
            continue

        if r.status == ProbeStatus.RATE_LIMITED:
            ttft_s, status_s = "[yellow]–[/yellow]", "[yellow]rate limited[/yellow]"
        elif r.status == ProbeStatus.ERROR:
            ttft_s, status_s = "[red]–[/red]", f"[red]{r.error or 'error'}[/red]"
        elif r.ttft_ms > 3000:
            ttft_s  = f"[yellow]{r.ttft_ms:.0f}ms[/yellow]"
            status_s = "[yellow]congested[/yellow]"
        else:
            color   = "bright_green" if r.ttft_ms < 500 else "green"
            ttft_s  = f"[{color}]{r.ttft_ms:.0f}ms[/{color}]"
            status_s = f"[{color}]available[/{color}]"

        table.add_row(
            platform.display_name,
            ttft_s,
            status_s,
            "  ".join(platform.tags[:3]),
        )

    console.print(table)


if __name__ == "__main__":
    app()
