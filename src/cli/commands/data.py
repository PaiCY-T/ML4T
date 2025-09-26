"""
Data Management Commands.

Commands for manual data operations, synchronization,
and data quality management.
"""

import sys
import logging
import json
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, date, timedelta

import click
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from ...data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
from ...data.ingestion.finlab_auth import create_finlab_authenticator
from ...data.pipeline.incremental_updater import IncrementalUpdater, UpdateConfig, UpdateMode
from ...data.pipeline.finlab_dataset_config import FinLabDatasetConfig
from ...data.core.temporal import DataType, is_taiwan_trading_day, get_previous_trading_day
from ...data.pipeline.data_validation import DataValidator
from ..utils.formatting import format_table, format_status, format_bytes, format_timestamp, format_duration
from ..utils.errors import DataError, AuthenticationError, handle_cli_error

logger = logging.getLogger(__name__)


@click.group(name='data')
def data_group():
    """Data management and synchronization commands."""
    pass


@data_group.command()
@click.option('--symbols', help='Comma-separated list of symbols to sync')
@click.option('--datasets', help='Comma-separated list of datasets to sync')
@click.option('--start-date', type=click.DateTime(formats=['%Y-%m-%d']),
              help='Start date for sync (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(formats=['%Y-%m-%d']),
              help='End date for sync (YYYY-MM-DD)')
@click.option('--days', type=int, help='Number of days back to sync (alternative to date range)')
@click.option('--mode', type=click.Choice(['incremental', 'full', 'validation']),
              default='incremental', help='Sync mode')
@click.option('--batch-size', default=1000, help='Batch size for processing')
@click.option('--dry-run', is_flag=True, help='Show what would be synced without doing it')
@click.option('--force', is_flag=True, help='Force sync even if data exists')
@click.pass_context
def sync(ctx, symbols: Optional[str], datasets: Optional[str], start_date: Optional[datetime],
         end_date: Optional[datetime], days: Optional[int], mode: str, batch_size: int,
         dry_run: bool, force: bool):
    """Manually sync data from FinLab."""
    console = ctx.obj['console']

    try:
        # Create connector and updater
        auth = create_finlab_authenticator()
        if not auth.config.has_db_auth and not auth.config.has_token_auth:
            raise AuthenticationError("No authentication configured")

        config = FinLabConfig(auth_config=auth.config)
        connector = FinLabConnector(config)

        # Test connection
        with console.status("[blue]Testing connection..."):
            if not connector.test_connection():
                raise DataError("Connection test failed")

        console.print("[green]✓[/green] Connection established")

        # Parse parameters
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
            console.print(f"Target symbols: {', '.join(symbol_list)}")

        dataset_list = None
        if datasets:
            dataset_list = [d.strip() for d in datasets.split(',')]
            console.print(f"Target datasets: {', '.join(dataset_list)}")

        # Calculate date range
        if days:
            end_dt = date.today()
            start_dt = end_dt - timedelta(days=days)
        else:
            start_dt = start_date.date() if start_date else date.today() - timedelta(days=30)
            end_dt = end_date.date() if end_date else date.today()

        console.print(f"Date range: {start_dt} to {end_dt}")
        console.print(f"Sync mode: {mode}")

        if dry_run:
            console.print("[yellow]DRY RUN MODE - No data will be modified[/yellow]")

        # Create updater
        update_config = UpdateConfig(
            update_mode=UpdateMode(mode.upper()),
            batch_size=batch_size,
            force_update=force,
            start_date=start_dt,
            end_date=end_dt
        )

        updater = IncrementalUpdater(connector, update_config)

        # Get sync plan
        sync_plan = updater.create_sync_plan(
            symbols=symbol_list,
            datasets=dataset_list,
            start_date=start_dt,
            end_date=end_dt
        )

        # Display sync plan
        console.print()
        console.print("[bold]Sync Plan:[/bold]")

        plan_data = []
        total_operations = 0
        for dataset, operations in sync_plan.items():
            plan_data.append({
                "Dataset": dataset,
                "Operations": len(operations),
                "Symbols": len(set(op.symbol for op in operations if op.symbol)),
                "Date Range": f"{min(op.date for op in operations if op.date)} to {max(op.date for op in operations if op.date)}" if operations else "N/A"
            })
            total_operations += len(operations)

        if plan_data:
            table = format_table(plan_data, title="Sync Operations")
            console.print(table)
            console.print(f"Total operations: {total_operations}")
        else:
            console.print("[yellow]No sync operations needed[/yellow]")
            return

        if dry_run:
            console.print("\n[blue]Dry run completed. Use --dry-run=false to execute.[/blue]")
            return

        # Confirm execution
        if not force and not click.confirm(f"\nProceed with {total_operations} sync operations?"):
            console.print("Sync cancelled")
            return

        # Execute sync
        console.print()
        console.print("[blue]Starting data synchronization...[/blue]")

        with Progress() as progress:
            overall_task = progress.add_task("[blue]Syncing data...", total=total_operations)
            completed_ops = 0

            for dataset, operations in sync_plan.items():
                if not operations:
                    continue

                dataset_task = progress.add_task(f"[blue]Syncing {dataset}...", total=len(operations))

                try:
                    # Process operations in batches
                    batch_operations = []
                    for i, operation in enumerate(operations):
                        batch_operations.append(operation)

                        if len(batch_operations) >= batch_size or i == len(operations) - 1:
                            # Execute batch
                            results = updater.execute_batch_operations(batch_operations)

                            # Update progress
                            progress.update(dataset_task, advance=len(batch_operations))
                            progress.update(overall_task, advance=len(batch_operations))
                            completed_ops += len(batch_operations)

                            # Log results
                            successful = sum(1 for r in results if r.success)
                            if successful < len(batch_operations):
                                failed = len(batch_operations) - successful
                                logger.warning(f"Batch completed with {failed} failures")

                            batch_operations = []

                    progress.update(dataset_task, completed=len(operations))

                except Exception as e:
                    logger.error(f"Error syncing dataset {dataset}: {e}")
                    console.print(f"[red]Error syncing {dataset}: {e}[/red]")

        console.print(f"[green]✓[/green] Sync completed: {completed_ops} operations processed")

    except AuthenticationError as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        sys.exit(3)
    except DataError as e:
        console.print(f"[red]Data error: {e}[/red]")
        sys.exit(6)
    except Exception as e:
        logger.error(f"Error syncing data: {e}")
        console.print(f"[red]Error syncing data: {e}[/red]")
        sys.exit(1)


@data_group.command()
@click.argument('symbol')
@click.option('--dataset', help='Specific dataset to query')
@click.option('--start-date', type=click.DateTime(formats=['%Y-%m-%d']),
              help='Start date (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(formats=['%Y-%m-%d']),
              help='End date (YYYY-MM-DD)')
@click.option('--limit', default=100, help='Maximum number of records to show')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save results to file')
@click.pass_context
def query(ctx, symbol: str, dataset: Optional[str], start_date: Optional[datetime],
          end_date: Optional[datetime], limit: int, output_format: str, output: Optional[str]):
    """Query data for a specific symbol."""
    console = ctx.obj['console']

    try:
        # Create connector
        auth = create_finlab_authenticator()
        config = FinLabConfig(auth_config=auth.config)
        connector = FinLabConnector(config)

        symbol = symbol.upper()
        console.print(f"[blue]Querying data for symbol: {symbol}[/blue]")

        # Set default date range
        end_dt = end_date.date() if end_date else date.today()
        start_dt = start_date.date() if start_date else end_dt - timedelta(days=30)

        console.print(f"Date range: {start_dt} to {end_dt}")
        if dataset:
            console.print(f"Dataset: {dataset}")

        # Query data
        with console.status(f"[blue]Fetching data for {symbol}..."):
            if dataset:
                data_records = connector.get_symbol_data(
                    symbol=symbol,
                    dataset=dataset,
                    start_date=start_dt,
                    end_date=end_dt,
                    limit=limit
                )
            else:
                # Get all available data types
                data_records = connector.get_comprehensive_symbol_data(
                    symbol=symbol,
                    start_date=start_dt,
                    end_date=end_dt,
                    limit=limit
                )

        if not data_records:
            console.print(f"[yellow]No data found for {symbol}[/yellow]")
            return

        # Process results
        result_data = {
            "symbol": symbol,
            "dataset": dataset,
            "date_range": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "record_count": len(data_records),
            "records": [
                {
                    "date": record.date.isoformat() if hasattr(record, 'date') and record.date else None,
                    "data_type": record.data_type.value if hasattr(record, 'data_type') else None,
                    "value": record.value if hasattr(record, 'value') else None,
                    "metadata": record.metadata if hasattr(record, 'metadata') else {}
                }
                for record in data_records
            ]
        }

        # Output results
        if output:
            if output_format == 'json':
                with open(output, 'w') as f:
                    json.dump(result_data, f, indent=2)
            elif output_format == 'csv':
                import csv
                with open(output, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Date', 'Data Type', 'Value', 'Metadata'])
                    for record in result_data['records']:
                        writer.writerow([
                            record.get('date', ''),
                            record.get('data_type', ''),
                            record.get('value', ''),
                            json.dumps(record.get('metadata', {}))
                        ])
            else:
                with open(output, 'w') as f:
                    f.write(f"Data Query Results for {symbol}\n")
                    f.write("=" * 50 + "\n\n")
                    for record in result_data['records']:
                        f.write(f"Date: {record.get('date', 'N/A')}\n")
                        f.write(f"Type: {record.get('data_type', 'N/A')}\n")
                        f.write(f"Value: {record.get('value', 'N/A')}\n")
                        f.write(f"Metadata: {record.get('metadata', {})}\n\n")

            console.print(f"[green]Results saved to {output}[/green]")

        # Display results
        if output_format == 'json':
            console.print(json.dumps(result_data, indent=2))
        else:
            console.print(f"[bold]Query Results for {symbol}[/bold]")
            console.print(f"Records found: {len(data_records)}")
            console.print()

            if data_records:
                # Display as table
                display_data = []
                for record in data_records[:limit]:
                    display_data.append({
                        "Date": record.date.isoformat() if hasattr(record, 'date') and record.date else 'N/A',
                        "Type": record.data_type.value if hasattr(record, 'data_type') else 'N/A',
                        "Value": str(record.value) if hasattr(record, 'value') else 'N/A'
                    })

                table = format_table(display_data, title=f"Data for {symbol}", max_rows=50)
                console.print(table)

                if len(data_records) > limit:
                    console.print(f"[dim]... showing {limit} of {len(data_records)} records[/dim]")

    except Exception as e:
        logger.error(f"Error querying data for {symbol}: {e}")
        console.print(f"[red]Error querying data for {symbol}: {e}[/red]")
        sys.exit(1)


@data_group.command()
@click.option('--dataset', help='Specific dataset to list symbols for')
@click.option('--active-only', is_flag=True, help='Only show actively traded symbols')
@click.option('--market', type=click.Choice(['TSE', 'OTC', 'ALL']),
              default='ALL', help='Filter by market')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def symbols(ctx, dataset: Optional[str], active_only: bool, market: str, output_format: str):
    """List available symbols in the database."""
    console = ctx.obj['console']

    try:
        # Create connector
        auth = create_finlab_authenticator()
        config = FinLabConfig(auth_config=auth.config)
        connector = FinLabConnector(config)

        console.print("[blue]Fetching symbol list...[/blue]")

        # Get symbols
        with console.status("[blue]Loading symbols..."):
            symbol_info = connector.get_available_symbols(
                dataset=dataset,
                active_only=active_only,
                market=market if market != 'ALL' else None
            )

        if not symbol_info:
            console.print("[yellow]No symbols found[/yellow]")
            return

        symbols_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "filters": {
                "dataset": dataset,
                "active_only": active_only,
                "market": market
            },
            "count": len(symbol_info),
            "symbols": [
                {
                    "symbol": info.symbol,
                    "name": info.name,
                    "market": info.market,
                    "sector": info.sector,
                    "active": info.active,
                    "last_update": info.last_update.isoformat() if info.last_update else None
                }
                for info in symbol_info
            ]
        }

        if output_format == 'json':
            console.print(json.dumps(symbols_data, indent=2))
        else:
            # Display symbol list
            console.print(f"[bold]Available Symbols[/bold]")
            console.print(f"Total symbols: {len(symbol_info)}")

            if dataset:
                console.print(f"Dataset: {dataset}")
            if active_only:
                console.print("Filter: Active only")
            if market != 'ALL':
                console.print(f"Market: {market}")
            console.print()

            # Group by market
            from collections import defaultdict
            symbols_by_market = defaultdict(list)
            for info in symbol_info:
                symbols_by_market[info.market or 'Unknown'].append(info)

            for mkt, symbols_list in symbols_by_market.items():
                console.print(f"[bold]{mkt} Market ({len(symbols_list)} symbols)[/bold]")

                display_data = []
                for info in sorted(symbols_list, key=lambda x: x.symbol)[:100]:  # Show first 100 per market
                    display_data.append({
                        "Symbol": info.symbol,
                        "Name": info.name[:30] + '...' if len(info.name) > 30 else info.name,
                        "Sector": info.sector or 'N/A',
                        "Active": "Yes" if info.active else "No",
                        "Last Update": format_timestamp(info.last_update) if info.last_update else "Never"
                    })

                if display_data:
                    table = format_table(display_data, max_rows=100)
                    console.print(table)

                    if len(symbols_list) > 100:
                        console.print(f"[dim]... and {len(symbols_list) - 100} more symbols[/dim]")

                console.print()

    except Exception as e:
        logger.error(f"Error listing symbols: {e}")
        console.print(f"[red]Error listing symbols: {e}[/red]")
        sys.exit(1)


@data_group.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def datasets(ctx, output_format: str):
    """List available datasets and their configurations."""
    console = ctx.obj['console']

    try:
        console.print("[blue]Loading dataset configurations...[/blue]")

        # Get dataset configurations
        dataset_configs = FinLabDatasetConfig.get_all_dataset_configs()

        datasets_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(dataset_configs),
            "datasets": [
                {
                    "name": config.dataset_name,
                    "description": config.description,
                    "table_name": config.table_name,
                    "update_strategy": config.update_strategy.value,
                    "field_count": len(config.fields),
                    "primary_keys": config.primary_keys,
                    "enabled": config.enabled
                }
                for config in dataset_configs
            ]
        }

        if output_format == 'json':
            console.print(json.dumps(datasets_data, indent=2))
        else:
            console.print(f"[bold]Available Datasets[/bold]")
            console.print(f"Total datasets: {len(dataset_configs)}")
            console.print()

            display_data = []
            for config in dataset_configs:
                display_data.append({
                    "Name": config.dataset_name,
                    "Description": config.description[:50] + '...' if len(config.description) > 50 else config.description,
                    "Table": config.table_name,
                    "Strategy": config.update_strategy.value,
                    "Fields": len(config.fields),
                    "Enabled": "Yes" if config.enabled else "No"
                })

            table = format_table(display_data, title="Dataset Configurations")
            console.print(table)

    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        console.print(f"[red]Error listing datasets: {e}[/red]")
        sys.exit(1)


@data_group.command()
@click.option('--symbols', help='Comma-separated list of symbols to clean')
@click.option('--datasets', help='Comma-separated list of datasets to clean')
@click.option('--days', type=int, help='Clean data older than N days')
@click.option('--dry-run', is_flag=True, help='Show what would be cleaned without doing it')
@click.option('--force', is_flag=True, help='Force cleanup without confirmation')
@click.pass_context
def cleanup(ctx, symbols: Optional[str], datasets: Optional[str], days: Optional[int],
            dry_run: bool, force: bool):
    """Clean up old or invalid data."""
    console = ctx.obj['console']

    try:
        # Create connector
        auth = create_finlab_authenticator()
        config = FinLabConfig(auth_config=auth.config)
        connector = FinLabConnector(config)

        # Parse parameters
        symbol_list = [s.strip().upper() for s in symbols.split(',')] if symbols else None
        dataset_list = [d.strip() for d in datasets.split(',')] if datasets else None

        console.print("[blue]Analyzing data for cleanup...[/blue]")

        # Analyze what needs to be cleaned
        cleanup_plan = connector.analyze_cleanup_requirements(
            symbols=symbol_list,
            datasets=dataset_list,
            older_than_days=days
        )

        if not cleanup_plan:
            console.print("[green]No cleanup required[/green]")
            return

        # Display cleanup plan
        console.print()
        console.print("[bold]Cleanup Plan:[/bold]")

        plan_data = []
        total_records = 0
        for item in cleanup_plan:
            plan_data.append({
                "Dataset": item.dataset,
                "Symbol": item.symbol or "ALL",
                "Records": item.record_count,
                "Size": format_bytes(item.size_bytes) if item.size_bytes else "N/A",
                "Reason": item.reason
            })
            total_records += item.record_count

        table = format_table(plan_data, title="Cleanup Operations")
        console.print(table)
        console.print(f"Total records to clean: {total_records:,}")

        if dry_run:
            console.print("\n[blue]Dry run completed. Use --dry-run=false to execute.[/blue]")
            return

        # Confirm cleanup
        if not force and not click.confirm(f"\nProceed with cleanup of {total_records:,} records?"):
            console.print("Cleanup cancelled")
            return

        # Execute cleanup
        console.print()
        console.print("[blue]Starting cleanup...[/blue]")

        with Progress() as progress:
            task = progress.add_task("[blue]Cleaning up data...", total=len(cleanup_plan))

            cleaned_records = 0
            for i, item in enumerate(cleanup_plan):
                try:
                    progress.update(task, description=f"[blue]Cleaning {item.dataset}...")
                    result = connector.cleanup_data(item)
                    cleaned_records += result.cleaned_count
                    progress.update(task, advance=1)

                except Exception as e:
                    logger.error(f"Error cleaning {item.dataset}: {e}")
                    progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Cleanup completed: {cleaned_records:,} records removed")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        console.print(f"[red]Error during cleanup: {e}[/red]")
        sys.exit(1)