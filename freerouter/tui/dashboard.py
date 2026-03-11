"""
freerouter/tui/dashboard.py
Rich-based terminal dashboard showing real-time platform status.
"""
from __future__ import annotations
import asyncio
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from freerouter.core.models import ProbeResult, ProbeStatus
from freerouter.core.platforms import ALL_PLATFORMS
from freerouter.core.prober import Prober

console = Console()


def _ttft_color(ttft_ms: float, status: ProbeStatus) -> str:
    if status == ProbeStatus.RATE_LIMITED:
        return "yellow"
    if status == ProbeStatus.ERROR:
        return "red"
    if ttft_ms < 500:
        return "bright_green"
    if ttft_ms < 1500:
        return "green"
    if ttft_ms < 3000:
        return "yellow"
    return "red"


def _status_icon(result: Optional[ProbeResult]) -> str:
    if result is None:
        return "[dim]·[/dim]"
    if result.status == ProbeStatus.RATE_LIMITED:
        return "[yellow]⏸[/yellow]"
    if result.status == ProbeStatus.ERROR:
        return "[red]✗[/red]"
    if result.is_congested:
        return "[yellow]⚠[/yellow]"
    return "[bright_green]✓[/bright_green]"


def build_status_table(probe_results: dict[str, ProbeResult]) -> Table:
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title="[bold]FreeRouter — Platform Status[/bold]",
        title_style="bold white",
        expand=True,
    )
    table.add_column("", width=3)
    table.add_column("Platform", style="bold", min_width=16)
    table.add_column("TTFT", justify="right", min_width=10)
    table.add_column("Status", min_width=14)
    table.add_column("Strengths", style="dim")

    for platform in ALL_PLATFORMS:
        result = probe_results.get(platform.name)
        icon = _status_icon(result)

        if result is None:
            ttft_str = "[dim]–[/dim]"
            status_str = "[dim]no data[/dim]"
        elif result.status == ProbeStatus.RATE_LIMITED:
            ttft_str = "[yellow]–[/yellow]"
            status_str = "[yellow]rate limited[/yellow]"
        elif result.status == ProbeStatus.ERROR:
            ttft_str = "[red]–[/red]"
            status_str = f"[red]error[/red]"
        else:
            color = _ttft_color(result.ttft_ms, result.status)
            ttft_str = f"[{color}]{result.ttft_ms:.0f}ms[/{color}]"
            if result.is_congested:
                status_str = "[yellow]congested[/yellow]"
            else:
                status_str = f"[{color}]available[/{color}]"

        tags_str = "  ".join(platform.tags[:3])
        table.add_row(icon, platform.display_name, ttft_str, status_str, tags_str)

    return table


def print_probe_summary(probe_results: dict[str, ProbeResult]) -> None:
    """One-shot print of current platform status."""
    table = build_status_table(probe_results)
    console.print(table)


async def live_dashboard(prober: Prober, refresh_interval: float = 2.0) -> None:
    """Live-updating terminal dashboard."""
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            table = build_status_table(prober.results)
            footer = Text(
                f"  Probing every {prober.config.routing.probe_interval}s  "
                "·  Ctrl+C to exit  "
                "·  freerouter listening on "
                f"http://{prober.config.host}:{prober.config.port}/v1",
                style="dim"
            )
            panel = Panel(table, subtitle=footer)
            live.update(panel)
            await asyncio.sleep(refresh_interval)
