"""
Main CLI Entry Point for FinLab Data Integration Pipeline.

This module provides the main command-line interface for managing and monitoring
the FinLab data integration system with comprehensive tools for administration,
troubleshooting, and operational control.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# Install rich traceback handler for better error display
install(show_locals=True)

# Import command groups
from .commands.pipeline import pipeline_group
from .commands.validation import validation_group
from .commands.monitoring import monitoring_group
from .commands.data import data_group
from .commands.config import config_group
from .commands.batch import batch_group
from .commands.troubleshoot import troubleshoot_group
from .utils.logging import setup_logging, get_log_level
from .utils.config import load_cli_config, CliConfig

console = Console()
logger = logging.getLogger(__name__)


@click.group(name='finlab-cli')
@click.option('--verbose', '-v', count=True,
              help='Increase verbosity (-v, -vv, -vvv for debug)')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress all output except errors')
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--log-file', type=click.Path(),
              help='Path to log file (default: logs/finlab-cli.log)')
@click.option('--no-color', is_flag=True,
              help='Disable colored output')
@click.pass_context
def cli(ctx, verbose: int, quiet: bool, config_file: Optional[str],
        log_file: Optional[str], no_color: bool):
    """
    FinLab Data Integration CLI - Command-line tools for managing the FinLab
    data pipeline, including data validation, monitoring, and system control.

    Examples:
        # Check pipeline status
        finlab-cli pipeline status

        # Run data validation
        finlab-cli validation run --dataset fundamental

        # Monitor system health
        finlab-cli monitoring health

        # Sync data manually
        finlab-cli data sync --symbols 2330,2317 --days 30

        # View configuration
        finlab-cli config show
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Disable colors if requested or if not in a tty
    if no_color or not sys.stdout.isatty():
        console._color_system = None

    # Setup logging
    log_level = get_log_level(verbose, quiet)
    setup_logging(log_level, log_file, no_color)

    # Load configuration
    try:
        config = load_cli_config(config_file)
        ctx.obj['config'] = config
        ctx.obj['console'] = console
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)

    # Store CLI options in context
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    ctx.obj['no_color'] = no_color

    logger.debug(f"CLI initialized with log level: {logging.getLevelName(log_level)}")


@cli.command()
@click.pass_context
def version(ctx):
    """Show version information."""
    from .. import __version__
    console = ctx.obj['console']
    console.print(f"FinLab CLI version {__version__}")


@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['text', 'json']),
              default='text', help='Output format')
@click.pass_context
def info(ctx, output_format: str):
    """Show system information and configuration."""
    import json
    from datetime import datetime
    from ..data.ingestion.finlab_auth import create_finlab_authenticator
    from ..data.pipeline.monitoring import get_pipeline_monitor

    console = ctx.obj['console']
    config = ctx.obj['config']

    try:
        # Get system information
        auth = create_finlab_authenticator()
        monitor = get_pipeline_monitor()

        info_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "cli_version": "1.0.0",
            "config_file": config.config_file,
            "authentication": {
                "has_token_auth": auth.config.has_token_auth if auth else False,
                "has_db_auth": auth.config.has_db_auth if auth else False
            },
            "pipeline": {
                "status": monitor.get_system_status().value if monitor else "unknown"
            },
            "environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform
            }
        }

        if output_format == 'json':
            console.print(json.dumps(info_data, indent=2))
        else:
            console.print(f"[bold]FinLab CLI System Information[/bold]")
            console.print(f"CLI Version: {info_data['cli_version']}")
            console.print(f"Timestamp: {info_data['timestamp']}")
            console.print(f"Config File: {info_data['config_file']}")
            console.print(f"Authentication: Token={info_data['authentication']['has_token_auth']}, DB={info_data['authentication']['has_db_auth']}")
            console.print(f"Pipeline Status: {info_data['pipeline']['status']}")
            console.print(f"Python: {info_data['environment']['python_version']} ({info_data['environment']['platform']})")

    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        if output_format == 'json':
            console.print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


# Register command groups
cli.add_command(pipeline_group, name='pipeline')
cli.add_command(validation_group, name='validation')
cli.add_command(monitoring_group, name='monitoring')
cli.add_command(data_group, name='data')
cli.add_command(config_group, name='config')
cli.add_command(batch_group, name='batch')
cli.add_command(troubleshoot_group, name='troubleshoot')


if __name__ == '__main__':
    cli()