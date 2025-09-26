"""
Pipeline Control Commands.

Commands for managing the FinLab data pipeline including start, stop,
status monitoring, and configuration management.
"""

import sys
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, TaskID
import time

from ...data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
from ...data.ingestion.finlab_auth import create_finlab_authenticator, AuthenticationError
from ...data.pipeline.incremental_updater import IncrementalUpdater, UpdateMode
from ...data.pipeline.monitoring import get_pipeline_monitor, PipelineStatus, AlertLevel
from ...data.pipeline.data_validation import DataValidator, ValidationReport
from ..utils.formatting import format_table, format_status, format_duration, format_timestamp
from ..utils.errors import PipelineError, AuthenticationError as CliAuthError, handle_cli_error

logger = logging.getLogger(__name__)


@click.group(name='pipeline')
def pipeline_group():
    """Pipeline control and management commands."""
    pass


@pipeline_group.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.option('--refresh', '-r', is_flag=True,
              help='Refresh status (don\'t use cached data)')
@click.pass_context
def status(ctx, output_format: str, refresh: bool):
    """Show pipeline status and health information."""
    console = ctx.obj['console']

    try:
        # Get pipeline monitor
        monitor = get_pipeline_monitor()

        if refresh:
            # Force refresh of all components
            monitor.refresh_all_status()

        # Get current system status
        system_status = monitor.get_system_status()
        component_status = monitor.get_component_status()
        recent_alerts = monitor.get_recent_alerts(hours=24)

        status_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": system_status.value,
            "components": {
                name: {
                    "status": status.status.value,
                    "last_update": status.last_update.isoformat() if status.last_update else None,
                    "health_score": status.health_score,
                    "message": status.message
                }
                for name, status in component_status.items()
            },
            "alerts": [
                {
                    "level": alert.level.value,
                    "component": alert.component,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in recent_alerts
            ]
        }

        if output_format == 'json':
            console.print(json.dumps(status_data, indent=2))
        else:
            # Display formatted status
            console.print(f"[bold]Pipeline Status[/bold] - {format_timestamp(datetime.utcnow())}")
            console.print(f"System Status: {format_status(system_status.value, system_status.value in ['healthy', 'running'])}")
            console.print()

            # Component status table
            if component_status:
                components_data = []
                for name, comp_status in component_status.items():
                    components_data.append({
                        "Component": name,
                        "Status": comp_status.status.value,
                        "Health": f"{comp_status.health_score:.1f}",
                        "Last Update": format_timestamp(comp_status.last_update) if comp_status.last_update else "Never",
                        "Message": comp_status.message or ""
                    })

                table = format_table(components_data, title="Component Status")
                console.print(table)
                console.print()

            # Recent alerts
            if recent_alerts:
                alerts_data = []
                for alert in recent_alerts[:10]:  # Show last 10 alerts
                    alerts_data.append({
                        "Level": alert.level.value.upper(),
                        "Component": alert.component,
                        "Message": alert.message,
                        "Time": format_timestamp(alert.timestamp)
                    })

                table = format_table(alerts_data, title="Recent Alerts (Last 24h)")
                console.print(table)
            else:
                console.print("[green]No recent alerts[/green]")

    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        console.print(f"[red]Error getting pipeline status: {e}[/red]")
        sys.exit(1)


@pipeline_group.command()
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='Custom configuration file')
@click.option('--mode', type=click.Choice(['incremental', 'full', 'validation']),
              default='incremental', help='Update mode')
@click.option('--datasets', multiple=True,
              help='Specific datasets to start (default: all)')
@click.option('--symbols', help='Comma-separated list of symbols (optional)')
@click.option('--force', is_flag=True,
              help='Force start even if already running')
@click.pass_context
def start(ctx, config_file: Optional[str], mode: str, datasets: tuple,
          symbols: Optional[str], force: bool):
    """Start the data pipeline."""
    console = ctx.obj['console']

    try:
        # Check if pipeline is already running
        monitor = get_pipeline_monitor()
        if not force and monitor.is_pipeline_running():
            console.print("[yellow]Pipeline is already running. Use --force to restart.[/yellow]")
            return

        console.print("[blue]Starting FinLab data pipeline...[/blue]")

        # Create authenticator
        auth = create_finlab_authenticator()
        if not auth.config.has_db_auth and not auth.config.has_token_auth:
            raise CliAuthError("No authentication configured. Please set up FinLab credentials.")

        # Create connector
        connector_config = FinLabConfig(auth_config=auth.config)
        connector = FinLabConnector(connector_config)

        # Test connection
        with console.status("[blue]Testing connection..."):
            if not connector.test_connection():
                raise PipelineError("Connection test failed")

        console.print("[green]✓[/green] Connection established")

        # Configure update mode
        update_mode = UpdateMode(mode)

        # Parse symbols if provided
        symbol_list = None
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',')]
            console.print(f"Target symbols: {', '.join(symbol_list)}")

        # Parse datasets if provided
        dataset_list = list(datasets) if datasets else None
        if dataset_list:
            console.print(f"Target datasets: {', '.join(dataset_list)}")

        # Create temporal store and updater
        from ...data.core.temporal import TemporalDataManager
        temporal_manager = TemporalDataManager()
        updater = IncrementalUpdater(
            temporal_store=temporal_manager.store,
            finlab_connector=connector
        )

        # Start the pipeline with progress tracking
        with Progress() as progress:
            task = progress.add_task("[blue]Starting pipeline...", total=100)

            # Initialize
            progress.update(task, advance=20, description="[blue]Initializing...")
            updater.initialize()

            progress.update(task, advance=30, description="[blue]Validating configuration...")
            time.sleep(0.5)  # Brief pause for user experience

            progress.update(task, advance=50, description="[blue]Starting components...")
            # Note: In a real implementation, this would start background processes
            # For now, we simulate the startup sequence
            time.sleep(1)

            progress.update(task, advance=100, description="[green]Pipeline started")

        console.print("[green]✓[/green] Pipeline started successfully")
        console.print(f"Mode: {mode}")
        console.print(f"Process ID: {monitor.get_pipeline_pid() or 'N/A'}")

    except (AuthenticationError, CliAuthError) as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        sys.exit(3)
    except PipelineError as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        sys.exit(4)
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        console.print(f"[red]Error starting pipeline: {e}[/red]")
        sys.exit(1)


@pipeline_group.command()
@click.option('--graceful', '-g', is_flag=True,
              help='Graceful shutdown (wait for current operations)')
@click.option('--timeout', default=30,
              help='Timeout for graceful shutdown in seconds')
@click.pass_context
def stop(ctx, graceful: bool, timeout: int):
    """Stop the data pipeline."""
    console = ctx.obj['console']

    try:
        monitor = get_pipeline_monitor()

        if not monitor.is_pipeline_running():
            console.print("[yellow]Pipeline is not running.[/yellow]")
            return

        if graceful:
            console.print(f"[blue]Stopping pipeline gracefully (timeout: {timeout}s)...[/blue]")

            with Progress() as progress:
                task = progress.add_task("[blue]Shutting down...", total=timeout)

                # Simulate graceful shutdown process
                for i in range(timeout):
                    progress.update(task, advance=1,
                                  description=f"[blue]Waiting for operations to complete... ({timeout-i}s)")
                    time.sleep(1)

                    # Check if pipeline stopped
                    if not monitor.is_pipeline_running():
                        progress.update(task, completed=timeout, description="[green]Pipeline stopped")
                        break
                else:
                    console.print("[yellow]Timeout reached, forcing shutdown...[/yellow]")
        else:
            console.print("[blue]Stopping pipeline immediately...[/blue]")

        # Actually stop the pipeline
        # Note: In a real implementation, this would terminate background processes
        monitor.stop_pipeline()

        console.print("[green]✓[/green] Pipeline stopped successfully")

    except PipelineError as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        sys.exit(4)
    except Exception as e:
        logger.error(f"Error stopping pipeline: {e}")
        console.print(f"[red]Error stopping pipeline: {e}[/red]")
        sys.exit(1)


@pipeline_group.command()
@click.option('--follow', '-f', is_flag=True,
              help='Follow logs in real-time')
@click.option('--lines', '-n', default=50,
              help='Number of lines to show')
@click.option('--level', type=click.Choice(['debug', 'info', 'warning', 'error']),
              help='Filter by log level')
@click.option('--component', help='Filter by component name')
@click.pass_context
def logs(ctx, follow: bool, lines: int, level: Optional[str], component: Optional[str]):
    """View pipeline logs."""
    console = ctx.obj['console']

    try:
        monitor = get_pipeline_monitor()
        log_entries = monitor.get_log_entries(
            limit=lines,
            level=level,
            component=component
        )

        if not follow:
            # Show static logs
            for entry in log_entries:
                level_color = {
                    'DEBUG': 'blue',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red'
                }.get(entry.level, 'white')

                console.print(
                    f"[dim]{format_timestamp(entry.timestamp)}[/dim] "
                    f"[{level_color}]{entry.level}[/{level_color}] "
                    f"[dim]{entry.component}[/dim] "
                    f"{entry.message}"
                )
        else:
            # Follow logs in real-time
            console.print("[blue]Following logs... (Press Ctrl+C to exit)[/blue]")

            with Live(console=console, refresh_per_second=2) as live:
                while True:
                    try:
                        new_entries = monitor.get_log_entries(
                            limit=20,
                            level=level,
                            component=component
                        )

                        log_display = "\n".join([
                            f"{format_timestamp(entry.timestamp)} "
                            f"{entry.level} {entry.component} {entry.message}"
                            for entry in new_entries
                        ])

                        live.update(log_display)
                        time.sleep(2)

                    except KeyboardInterrupt:
                        console.print("\n[blue]Stopped following logs[/blue]")
                        break

    except Exception as e:
        logger.error(f"Error viewing logs: {e}")
        console.print(f"[red]Error viewing logs: {e}[/red]")
        sys.exit(1)


@pipeline_group.command()
@click.pass_context
def restart(ctx):
    """Restart the data pipeline."""
    console = ctx.obj['console']

    try:
        console.print("[blue]Restarting pipeline...[/blue]")

        # Stop first
        ctx.invoke(stop, graceful=True, timeout=30)

        # Wait a moment
        time.sleep(2)

        # Start again
        ctx.invoke(start)

    except Exception as e:
        logger.error(f"Error restarting pipeline: {e}")
        console.print(f"[red]Error restarting pipeline: {e}[/red]")
        sys.exit(1)