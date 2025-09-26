"""Download command implementation."""

import click
from pathlib import Path
from typing import List, Optional

from ...config.manager import ConfigManager
from ...core.exceptions import FinLabDownloaderError
from ...utils.logger import get_logger
from ...utils.validation import validate_symbols, validate_date_range


def download_command(
    config_manager: ConfigManager,
    symbols: List[str],
    start_date: str,
    end_date: str,
    source: Optional[str] = None,
    output: Optional[Path] = None,
    format: str = 'csv',
    overwrite: bool = False,
    verbose: bool = False
) -> None:
    """
    Execute download command.

    Args:
        config_manager: Configuration manager instance
        symbols: List of symbols to download
        start_date: Start date string
        end_date: End date string
        source: Data source name
        output: Output path
        format: Output format
        overwrite: Whether to overwrite existing files
        verbose: Verbose output flag
    """
    logger = get_logger('download_command')

    try:
        # Validate symbols
        validated_symbols = validate_symbols(symbols)
        logger.info(f"Validated symbols: {validated_symbols}")

        # Validate date range
        start_dt, end_dt = validate_date_range(start_date, end_date)
        logger.info(f"Date range: {start_dt} to {end_dt}")

        # Determine data source
        if source is None:
            data_sources = config_manager.list_data_sources()
            if not data_sources:
                raise FinLabDownloaderError(
                    "No data sources configured and none specified"
                )
            source = data_sources[0]  # Use first available source
            logger.info(f"Using default data source: {source}")

        # Get data source configuration
        source_config = config_manager.get_data_source_config(source)
        logger.info(f"Data source config: {source_config}")

        # Determine output path
        if output is None:
            output = Path(config_manager.get('general.data_directory', './data'))

        # Create output directory
        output.mkdir(parents=True, exist_ok=True)

        # For now, just show what would be downloaded (placeholder implementation)
        click.echo(f"📥 Download Configuration:")
        click.echo(f"   Symbols: {', '.join(validated_symbols)}")
        click.echo(f"   Date Range: {start_dt} to {end_dt}")
        click.echo(f"   Data Source: {source}")
        click.echo(f"   Output: {output}")
        click.echo(f"   Format: {format}")
        click.echo(f"   Overwrite: {overwrite}")

        # Placeholder for actual download implementation
        click.echo("✅ Download framework ready (implementation pending)")
        click.echo("   This will be implemented in issue #66 (Data Source Connectors)")

        logger.info("Download command completed successfully")

    except Exception as e:
        logger.error(f"Download command failed: {e}")
        raise