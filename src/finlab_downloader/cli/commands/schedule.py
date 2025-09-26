"""
CLI commands for financial statement scheduler.

Provides command-line interface for scheduling, monitoring, and managing
financial statement downloads.
"""

import asyncio
import logging
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...config.manager import ConfigurationManager
from ...scheduler.integration import SchedulerIntegration
from ...scheduler.financial_calendar import IndustryType, ReportingPeriod
from ...scheduler.scheduler_engine import SchedulePriority, ScheduleStatus

logger = logging.getLogger(__name__)
console = Console()


@click.group()
@click.pass_context
def schedule(ctx):
    """Financial statement scheduler commands."""
    pass


@schedule.command()
@click.argument('company_ticker')
@click.option('--fiscal-year', '-y', type=int, default=None,
              help='Fiscal year (defaults to current year)')
@click.option('--period', '-p', multiple=True,
              type=click.Choice(['Q1', 'Q2', 'Q3', 'Q4']),
              help='Reporting periods to schedule (can specify multiple)')
@click.option('--industry', '-i',
              type=click.Choice([t.value for t in IndustryType]),
              help='Industry type (auto-detected if not specified)')
@click.option('--priority',
              type=click.Choice([p.name.lower() for p in SchedulePriority]),
              default='normal', help='Schedule priority')
@click.option('--datasets', '-d', multiple=True,
              help='Specific datasets to download (defaults to all financial statements)')
@click.pass_context
def add(ctx, company_ticker: str, fiscal_year: Optional[int], period: tuple,
        industry: Optional[str], priority: str, datasets: tuple):
    """Schedule financial statement downloads for a company."""
    try:
        # Initialize scheduler integration
        config_manager = ctx.obj['config_manager']
        integration = SchedulerIntegration(config_manager)

        # Set defaults
        if fiscal_year is None:
            fiscal_year = datetime.now().year

        if not period:
            period = ('Q1', 'Q2', 'Q3', 'Q4')

        # Convert parameters
        reporting_periods = [ReportingPeriod(p) for p in period]
        industry_type = IndustryType(industry) if industry else None
        schedule_priority = SchedulePriority[priority.upper()]
        dataset_filter = list(datasets) if datasets else None

        # Schedule downloads
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Scheduling downloads for {company_ticker}...", total=None)

            schedule_ids = integration.schedule_company_financials(
                company_ticker=company_ticker,
                reporting_periods=reporting_periods,
                fiscal_year=fiscal_year,
                industry_type=industry_type,
                priority=schedule_priority,
                dataset_filter=dataset_filter
            )

        # Display results
        if schedule_ids:
            console.print(f"✅ Successfully scheduled {len(schedule_ids)} downloads for {company_ticker}",
                         style="green")

            table = Table(title=f"Scheduled Downloads - {company_ticker} FY{fiscal_year}")
            table.add_column("Schedule ID", style="cyan")
            table.add_column("Period", style="magenta")
            table.add_column("Priority", style="yellow")

            for i, schedule_id in enumerate(schedule_ids):
                period_name = reporting_periods[i % len(reporting_periods)].value
                table.add_row(schedule_id, period_name, priority.upper())

            console.print(table)
        else:
            console.print("❌ No downloads were scheduled", style="red")

    except Exception as e:
        console.print(f"❌ Error scheduling downloads: {e}", style="red")
        raise click.ClickException(str(e))


@schedule.command()
@click.argument('companies_file', type=click.Path(exists=True))
@click.option('--fiscal-year', '-y', type=int, default=None,
              help='Fiscal year (defaults to current year)')
@click.option('--period', '-p', multiple=True,
              type=click.Choice(['Q1', 'Q2', 'Q3', 'Q4']),
              help='Reporting periods to schedule (defaults to all)')
@click.option('--priority',
              type=click.Choice([p.name.lower() for p in SchedulePriority]),
              default='normal', help='Schedule priority')
@click.pass_context
def bulk(ctx, companies_file: str, fiscal_year: Optional[int], period: tuple, priority: str):
    """Schedule downloads for multiple companies from a file."""
    try:
        # Initialize scheduler integration
        config_manager = ctx.obj['config_manager']
        integration = SchedulerIntegration(config_manager)

        # Read companies file
        companies_path = Path(companies_file)
        with open(companies_path, 'r') as f:
            company_tickers = [line.strip() for line in f if line.strip()]

        if not company_tickers:
            raise click.ClickException("No company tickers found in file")

        # Set defaults
        if fiscal_year is None:
            fiscal_year = datetime.now().year

        if not period:
            period = ('Q1', 'Q2', 'Q3', 'Q4')

        # Convert parameters
        reporting_periods = [ReportingPeriod(p) for p in period]
        schedule_priority = SchedulePriority[priority.upper()]

        # Schedule bulk downloads
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Scheduling {len(company_tickers)} companies...", total=None)

            results = integration.schedule_bulk_companies(
                company_tickers=company_tickers,
                reporting_periods=reporting_periods,
                fiscal_year=fiscal_year,
                priority=schedule_priority
            )

        # Display results
        successful = len([ids for ids in results.values() if ids])
        total_schedules = sum(len(ids) for ids in results.values())

        console.print(f"✅ Bulk scheduling completed:", style="green")
        console.print(f"   Companies processed: {len(company_tickers)}")
        console.print(f"   Successfully scheduled: {successful}")
        console.print(f"   Total schedules created: {total_schedules}")

        # Show summary table
        table = Table(title=f"Bulk Scheduling Results - FY{fiscal_year}")
        table.add_column("Company", style="cyan")
        table.add_column("Schedules", style="magenta")
        table.add_column("Status", style="green")

        for ticker, schedule_ids in results.items():
            status = "✅ Success" if schedule_ids else "❌ Failed"
            table.add_row(ticker, str(len(schedule_ids)), status)

        console.print(table)

    except Exception as e:
        console.print(f"❌ Error in bulk scheduling: {e}", style="red")
        raise click.ClickException(str(e))


@schedule.command()
@click.option('--status', '-s',
              type=click.Choice([s.value for s in ScheduleStatus]),
              help='Filter by schedule status')
@click.option('--company', '-c', help='Filter by company ticker')
@click.option('--period', '-p',
              type=click.Choice(['Q1', 'Q2', 'Q3', 'Q4']),
              help='Filter by reporting period')
@click.pass_context
def list(ctx, status: Optional[str], company: Optional[str], period: Optional[str]):
    """List scheduled downloads with optional filtering."""
    try:
        # Initialize scheduler integration
        config_manager = ctx.obj['config_manager']
        integration = SchedulerIntegration(config_manager)

        # Get schedules
        status_filter = ScheduleStatus(status) if status else None
        schedules = integration.scheduler.list_schedules(status_filter)

        # Apply additional filters
        if company:
            schedules = [s for s in schedules if s.request.company_ticker.upper() == company.upper()]

        if period:
            period_enum = ReportingPeriod(period)
            schedules = [s for s in schedules if s.request.reporting_period == period_enum]

        if not schedules:
            console.print("No schedules found matching the criteria", style="yellow")
            return

        # Display schedules table
        table = Table(title=f"Scheduled Downloads ({len(schedules)} found)")
        table.add_column("Schedule ID", style="cyan")
        table.add_column("Company", style="magenta")
        table.add_column("Period", style="yellow")
        table.add_column("Fiscal Year", style="blue")
        table.add_column("Scheduled Time", style="green")
        table.add_column("Status", style="red")
        table.add_column("Priority", style="white")

        for schedule in sorted(schedules, key=lambda s: s.scheduled_time):
            table.add_row(
                schedule.schedule_id,
                schedule.request.company_ticker,
                schedule.request.reporting_period.value,
                str(schedule.request.fiscal_year),
                schedule.scheduled_time.strftime("%Y-%m-%d %H:%M"),
                schedule.status.value,
                schedule.request.priority.name
            )

        console.print(table)

    except Exception as e:
        console.print(f"❌ Error listing schedules: {e}", style="red")
        raise click.ClickException(str(e))


@schedule.command()
@click.pass_context
def status(ctx):
    """Show scheduler status and summary statistics."""
    try:
        # Initialize scheduler integration
        config_manager = ctx.obj['config_manager']
        integration = SchedulerIntegration(config_manager)

        # Get summary
        summary = integration.get_schedule_summary()

        # Display summary
        console.print(Panel.fit(
            f"📊 Scheduler Status Summary\n\n"
            f"Total Schedules: {summary['total_schedules']}\n"
            f"Pending: {summary['by_status'].get('pending', 0)}\n"
            f"Scheduled: {summary['by_status'].get('scheduled', 0)}\n"
            f"Running: {summary['by_status'].get('running', 0)}\n"
            f"Completed: {summary['by_status'].get('completed', 0)}\n"
            f"Failed: {summary['by_status'].get('failed', 0)}",
            title="Scheduler Status",
            border_style="green"
        ))

        # Show upcoming deadlines
        if summary['upcoming_deadlines']:
            table = Table(title="🔔 Upcoming Deadlines (Next 7 Days)")
            table.add_column("Company", style="cyan")
            table.add_column("Period", style="magenta")
            table.add_column("Scheduled", style="yellow")
            table.add_column("Days Until", style="green")

            for deadline in summary['upcoming_deadlines']:
                table.add_row(
                    deadline['company'],
                    deadline['period'],
                    deadline['scheduled_time'][:10],  # Date only
                    str(deadline['days_until'])
                )

            console.print(table)

        # Show overdue schedules
        if summary['overdue_schedules']:
            table = Table(title="⚠️ Overdue Schedules")
            table.add_column("Company", style="cyan")
            table.add_column("Period", style="magenta")
            table.add_column("Was Scheduled", style="yellow")
            table.add_column("Days Overdue", style="red")

            for overdue in summary['overdue_schedules']:
                table.add_row(
                    overdue['company'],
                    overdue['period'],
                    overdue['scheduled_time'][:10],  # Date only
                    str(overdue['days_overdue'])
                )

            console.print(table)

    except Exception as e:
        console.print(f"❌ Error getting scheduler status: {e}", style="red")
        raise click.ClickException(str(e))


@schedule.command()
@click.pass_context
def run(ctx):
    """Execute all pending scheduled downloads."""
    try:
        # Initialize scheduler integration
        config_manager = ctx.obj['config_manager']
        integration = SchedulerIntegration(config_manager)

        # Run async execution
        async def execute():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Executing scheduled downloads...", total=None)
                results = await integration.run_scheduled_downloads()
                return results

        results = asyncio.run(execute())

        # Display results
        if not results:
            console.print("No pending schedules to execute", style="yellow")
            return

        successful = len([r for r in results if r.status.value == "completed"])
        failed = len([r for r in results if r.status.value == "failed"])

        console.print(f"✅ Execution completed:", style="green")
        console.print(f"   Total executed: {len(results)}")
        console.print(f"   Successful: {successful}")
        console.print(f"   Failed: {failed}")

        # Show results table
        if results:
            table = Table(title="Execution Results")
            table.add_column("Company", style="cyan")
            table.add_column("Period", style="magenta")
            table.add_column("Status", style="green")
            table.add_column("Downloaded", style="blue")
            table.add_column("Failed", style="red")
            table.add_column("Duration", style="yellow")

            for result in results:
                status_color = "green" if result.status.value == "completed" else "red"
                table.add_row(
                    result.request.company_ticker,
                    result.request.reporting_period.value,
                    f"[{status_color}]{result.status.value}[/{status_color}]",
                    str(len(result.downloaded_datasets)),
                    str(len(result.failed_datasets)),
                    f"{result.duration_minutes:.1f}m"
                )

            console.print(table)

    except Exception as e:
        console.print(f"❌ Error executing schedules: {e}", style="red")
        raise click.ClickException(str(e))


@schedule.command()
@click.argument('schedule_id')
@click.pass_context
def cancel(ctx, schedule_id: str):
    """Cancel a scheduled download."""
    try:
        # Initialize scheduler integration
        config_manager = ctx.obj['config_manager']
        integration = SchedulerIntegration(config_manager)

        # Cancel schedule
        success = integration.scheduler.cancel_schedule(schedule_id)

        if success:
            console.print(f"✅ Successfully cancelled schedule {schedule_id}", style="green")
        else:
            console.print(f"❌ Failed to cancel schedule {schedule_id} (not found or cannot be cancelled)",
                         style="red")

    except Exception as e:
        console.print(f"❌ Error cancelling schedule: {e}", style="red")
        raise click.ClickException(str(e))


@schedule.command()
@click.option('--output', '-o', type=click.Path(),
              default='schedule_config.yaml',
              help='Output file path for configuration export')
@click.pass_context
def export(ctx, output: str):
    """Export current schedule configuration."""
    try:
        # Initialize scheduler integration
        config_manager = ctx.obj['config_manager']
        integration = SchedulerIntegration(config_manager)

        # Export configuration
        integration.export_schedule_config(output)

        console.print(f"✅ Configuration exported to {output}", style="green")

    except Exception as e:
        console.print(f"❌ Error exporting configuration: {e}", style="red")
        raise click.ClickException(str(e))