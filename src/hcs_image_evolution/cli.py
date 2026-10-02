"""CLI entrypoint for the Autonomous Image Model Evolution Lab."""

import click
from rich.console import Console

from hcs_image_evolution.agent.planner import ResearchPlanner
from hcs_image_evolution.agent.state_machine import StateManager
from hcs_image_evolution.utils.logging import setup_logger
from hcs_image_evolution.utils.system import get_system_specs

console = Console()


@click.group()
def main():
    """HCS Image Model Evolution Lab CLI."""
    setup_logger()


@main.group()
def agent():
    """Autonomous agent controller commands."""


@agent.command("run")
@click.option("--config", default="configs/base.yaml", help="Path to base configuration YAML.")
def run_agent(config: str):
    """Executes the autonomous closed-loop research and evolution cycle."""
    console.print(f"[bold cyan]Starting HCS Autonomous Agent[/bold cyan] with config: {config}")
    state_mgr = StateManager()
    planner = ResearchPlanner()

    console.print(f"Active run: [bold green]{state_mgr.state.run_id}[/bold green]")
    console.print(f"Current phase: [bold yellow]{state_mgr.state.stage}[/bold yellow]")

    # Run cycle step
    next_phase, adjustments = planner.plan_next_action(state_mgr.state, {})
    console.print(f"Next planned phase: [bold magenta]{next_phase}[/bold magenta]")
    state_mgr.transition_to(next_phase)


@main.command("info")
def system_info():
    """Displays local hardware, compute, and Vulkan capabilities."""
    specs = get_system_specs()
    console.print("[bold green]=== System Diagnostics ===[/bold green]")
    for k, v in specs.items():
        console.print(f"[cyan]{k}[/cyan]: {v}")


if __name__ == "__main__":
    main()
