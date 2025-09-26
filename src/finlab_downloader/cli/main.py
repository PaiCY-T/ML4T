"""
Main CLI interface for FinLab downloader.

Provides command-line interface with subcommands for download, validation, configuration, and listing operations.
"""

import sys
import click
from pathlib import Path
from typing import Optional

from ..config.manager import ConfigManager
from ..core.exceptions import FinLabDownloaderError, ConfigurationError
from ..utils.logger import configure_logging_from_config, get_logger
from .commands import (
    download_command,
    validate_command,
    list_command,
    config_command
)


# Global configuration manager instance
config_manager = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager


@click.group()
@click.option(
    '--config',
    '-c',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file path'
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--quiet',
    '-q',
    is_flag=True,
    help='Suppress non-error output'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    help='Set logging level'
)
@click.pass_context
def cli(ctx, config: Optional[Path], verbose: bool, quiet: bool, log_level: Optional[str]):
    """
    FinLab Data Downloader - Download and manage financial data from FinLab sources.

    This tool provides a command-line interface for downloading financial data,
    validating downloads, and managing configuration settings.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    try:
        # Initialize configuration manager
        global config_manager
        if config:
            config_manager = ConfigManager(config)
        else:
            config_manager = ConfigManager()

        # Store configuration in context
        ctx.obj['config_manager'] = config_manager

        # Configure logging
        if log_level:
            # Override log level from command line
            config_manager.set('logging.level', log_level)

        if verbose and not quiet:
            config_manager.set('logging.level', 'DEBUG')
        elif quiet and not verbose:
            config_manager.set('logging.level', 'ERROR')

        # Setup logging
        configure_logging_from_config(config_manager)

        # Store CLI flags in context
        ctx.obj['verbose'] = verbose
        ctx.obj['quiet'] = quiet

    except FinLabDownloaderError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('symbols', nargs=-1, required=True)
@click.option(
    '--start-date',
    '-s',
    required=True,
    help='Start date (YYYY-MM-DD)'
)
@click.option(
    '--end-date',
    '-e',
    required=True,
    help='End date (YYYY-MM-DD)'
)
@click.option(
    '--source',
    type=str,
    help='Data source name'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(path_type=Path),
    help='Output directory or file'
)
@click.option(
    '--format',
    type=click.Choice(['csv', 'json', 'parquet']),
    default='csv',
    help='Output format'
)
@click.option(
    '--overwrite',
    is_flag=True,
    help='Overwrite existing files'
)
@click.pass_context
def download(
    ctx,
    symbols: tuple,
    start_date: str,
    end_date: str,
    source: Optional[str],
    output: Optional[Path],
    format: str,
    overwrite: bool
):
    """
    Download financial data for specified symbols.

    SYMBOLS: One or more symbol identifiers to download (e.g., 2330 2317 2454)
    """
    try:
        config_manager = ctx.obj['config_manager']
        logger = get_logger('cli.download')

        logger.info(f"Starting download for symbols: {list(symbols)}")

        # Call download command implementation
        download_command(
            config_manager=config_manager,
            symbols=list(symbols),
            start_date=start_date,
            end_date=end_date,
            source=source,
            output=output,
            format=format,
            overwrite=overwrite,
            verbose=ctx.obj.get('verbose', False)
        )

    except FinLabDownloaderError as e:
        click.echo(f"Download failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger = get_logger('cli.download')
        logger.exception("Unexpected error during download")
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('path', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--strict',
    is_flag=True,
    help='Use strict validation mode'
)
@click.option(
    '--report',
    '-r',
    type=click.Path(path_type=Path),
    help='Save validation report to file'
)
@click.pass_context
def validate(ctx, path: Path, strict: bool, report: Optional[Path]):
    """
    Validate downloaded data files.

    PATH: Path to data file or directory to validate
    """
    try:
        config_manager = ctx.obj['config_manager']
        logger = get_logger('cli.validate')

        logger.info(f"Starting validation of: {path}")

        # Call validate command implementation
        validate_command(
            config_manager=config_manager,
            path=path,
            strict=strict,
            report=report,
            verbose=ctx.obj.get('verbose', False)
        )

    except FinLabDownloaderError as e:
        click.echo(f"Validation failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger = get_logger('cli.validate')
        logger.exception("Unexpected error during validation")
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    '--source',
    type=str,
    help='List symbols for specific data source'
)
@click.option(
    '--format',
    type=click.Choice(['table', 'json', 'csv']),
    default='table',
    help='Output format'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(path_type=Path),
    help='Save output to file'
)
@click.pass_context
def list(ctx, source: Optional[str], format: str, output: Optional[Path]):
    """
    List available symbols and data sources.
    """
    try:
        config_manager = ctx.obj['config_manager']
        logger = get_logger('cli.list')

        logger.info("Listing available symbols and sources")

        # Call list command implementation
        list_command(
            config_manager=config_manager,
            source=source,
            format=format,
            output=output,
            verbose=ctx.obj.get('verbose', False)
        )

    except FinLabDownloaderError as e:
        click.echo(f"List operation failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger = get_logger('cli.list')
        logger.exception("Unexpected error during list operation")
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.group()
@click.pass_context
def config(ctx):
    """
    Configuration management commands.
    """
    pass


@config.command('show')
@click.option(
    '--key',
    type=str,
    help='Show specific configuration key'
)
@click.pass_context
def config_show(ctx, key: Optional[str]):
    """Show current configuration."""
    try:
        config_manager = ctx.obj['config_manager']

        # Call config show implementation
        config_command(
            action='show',
            config_manager=config_manager,
            key=key,
            verbose=ctx.obj.get('verbose', False)
        )

    except FinLabDownloaderError as e:
        click.echo(f"Config operation failed: {e}", err=True)
        sys.exit(1)


@config.command('set')
@click.argument('key', type=str)
@click.argument('value', type=str)
@click.pass_context
def config_set(ctx, key: str, value: str):
    """Set configuration value."""
    try:
        config_manager = ctx.obj['config_manager']

        # Call config set implementation
        config_command(
            action='set',
            config_manager=config_manager,
            key=key,
            value=value,
            verbose=ctx.obj.get('verbose', False)
        )

    except FinLabDownloaderError as e:
        click.echo(f"Config operation failed: {e}", err=True)
        sys.exit(1)


@config.command('init')
@click.argument('path', type=click.Path(path_type=Path))
@click.option(
    '--overwrite',
    is_flag=True,
    help='Overwrite existing configuration file'
)
@click.pass_context
def config_init(ctx, path: Path, overwrite: bool):
    """Initialize new configuration file."""
    try:
        config_manager = ctx.obj['config_manager']

        # Call config init implementation
        config_command(
            action='init',
            config_manager=config_manager,
            path=path,
            overwrite=overwrite,
            verbose=ctx.obj.get('verbose', False)
        )

    except FinLabDownloaderError as e:
        click.echo(f"Config operation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def version(ctx):
    """Show version information."""
    from .. import __version__
    click.echo(f"FinLab Downloader v{__version__}")


if __name__ == '__main__':
    cli()