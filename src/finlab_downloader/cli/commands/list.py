"""List command implementation."""

import click
import json
from pathlib import Path
from typing import Optional

from ...config.manager import ConfigManager
from ...core.exceptions import FinLabDownloaderError
from ...utils.logger import get_logger
from ...utils.formatting import format_table


def list_command(
    config_manager: ConfigManager,
    source: Optional[str] = None,
    format: str = 'table',
    output: Optional[Path] = None,
    verbose: bool = False
) -> None:
    """
    Execute list command.

    Args:
        config_manager: Configuration manager instance
        source: Specific data source to list
        format: Output format
        output: Output file path
        verbose: Verbose output flag
    """
    logger = get_logger('list_command')

    try:
        logger.info("Starting list operation")

        # Get available data sources
        data_sources = config_manager.list_data_sources()

        if not data_sources:
            click.echo("⚠️  No data sources configured")
            click.echo("   Use 'finlab-downloader config init' to create a configuration file")
            return

        # Filter by specific source if requested
        if source:
            if source not in data_sources:
                raise FinLabDownloaderError(f"Data source '{source}' not found")
            sources_to_list = [source]
        else:
            sources_to_list = data_sources

        # Prepare data for output
        results = {
            'data_sources': [],
            'symbols': {}
        }

        for src in sources_to_list:
            src_config = config_manager.get_data_source_config(src)
            results['data_sources'].append({
                'name': src,
                'type': src_config.get('type', 'unknown'),
                'status': 'configured'
            })

            # Placeholder for symbol listing
            # This will be implemented with actual data source connectors
            results['symbols'][src] = [
                '2330', '2317', '2454', '2881', '2882'  # Example symbols
            ]

        # Output results
        if format == 'json':
            output_data = json.dumps(results, indent=2, ensure_ascii=False)
        elif format == 'csv':
            # For CSV, just output data sources
            lines = ['name,type,status']
            for ds in results['data_sources']:
                lines.append(f"{ds['name']},{ds['type']},{ds['status']}")
            output_data = '\n'.join(lines)
        else:  # table format
            # Format as table
            click.echo("📊 Available Data Sources:")
            if results['data_sources']:
                headers = ['Name', 'Type', 'Status']
                data = [[ds['name'], ds['type'], ds['status']] for ds in results['data_sources']]
                table = format_table(data, headers)
                click.echo(table)
            else:
                click.echo("   No data sources configured")

            click.echo("\n🏷️  Available Symbols (Sample):")
            for src, symbols in results['symbols'].items():
                click.echo(f"   {src}: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

            output_data = None

        # Save to file if requested
        if output and output_data:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_data)
            click.echo(f"📄 Output saved to: {output}")

        # Show placeholder message
        if format == 'table':
            click.echo("\n✅ List framework ready (implementation pending)")
            click.echo("   Actual symbol lists will be available with issue #66 (Data Source Connectors)")

        logger.info("List command completed successfully")

    except Exception as e:
        logger.error(f"List command failed: {e}")
        raise