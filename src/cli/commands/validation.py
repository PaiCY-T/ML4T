"""
Data Validation Commands.

Commands for running data validation checks, quality assessments,
and generating validation reports.
"""

import sys
import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ...data.pipeline.data_validation import DataValidator, ValidationReport, ValidationSeverity
from ...data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
from ...data.ingestion.finlab_auth import create_finlab_authenticator
from ...data.pipeline.finlab_dataset_config import FinLabDatasetConfig
from ...data.core.temporal import DataType
from ..utils.formatting import format_table, format_status, format_percentage, format_timestamp
from ..utils.errors import ValidationError, handle_cli_error

logger = logging.getLogger(__name__)


@click.group(name='validation')
def validation_group():
    """Data validation and quality check commands."""
    pass


@validation_group.command()
@click.option('--dataset', multiple=True,
              help='Dataset to validate (can be used multiple times)')
@click.option('--symbols', help='Comma-separated list of symbols to validate')
@click.option('--start-date', type=click.DateTime(formats=['%Y-%m-%d']),
              help='Start date for validation (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(formats=['%Y-%m-%d']),
              help='End date for validation (YYYY-MM-DD)')
@click.option('--severity', type=click.Choice(['info', 'warning', 'error', 'critical']),
              default='warning', help='Minimum severity level to report')
@click.option('--output', '-o', type=click.Path(),
              help='Save report to file')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
@click.pass_context
def run(ctx, dataset: tuple, symbols: Optional[str], start_date: Optional[datetime],
        end_date: Optional[datetime], severity: str, output: Optional[str],
        output_format: str):
    """Run data validation checks."""
    console = ctx.obj['console']

    try:
        # Create validator
        auth = create_finlab_authenticator()
        config = FinLabConfig(auth_config=auth.config)
        connector = FinLabConnector(config)
        validator = DataValidator(connector)

        # Parse parameters
        symbol_list = None
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',')]

        dataset_list = list(dataset) if dataset else None
        min_severity = ValidationSeverity(severity.upper())

        # Convert datetime to date
        start_dt = start_date.date() if start_date else None
        end_dt = end_date.date() if end_date else None

        console.print("[blue]Running data validation...[/blue]")

        if dataset_list:
            console.print(f"Datasets: {', '.join(dataset_list)}")
        if symbol_list:
            console.print(f"Symbols: {', '.join(symbol_list[:5])}{' ...' if len(symbol_list) > 5 else ''}")
        if start_dt or end_dt:
            console.print(f"Date range: {start_dt or 'earliest'} to {end_dt or 'latest'}")

        # Run validation with progress tracking
        validation_results = []

        with Progress() as progress:
            if dataset_list:
                total_datasets = len(dataset_list)
                task = progress.add_task("[blue]Validating datasets...", total=total_datasets)

                for i, dataset_name in enumerate(dataset_list):
                    progress.update(task, description=f"[blue]Validating {dataset_name}...")

                    try:
                        # Get dataset configuration
                        dataset_config = FinLabDatasetConfig.get_dataset_config(dataset_name)
                        if not dataset_config:
                            console.print(f"[yellow]Warning: Unknown dataset '{dataset_name}'[/yellow]")
                            continue

                        # Run validation for this dataset
                        report = validator.validate_dataset(
                            dataset_name=dataset_name,
                            symbols=symbol_list,
                            start_date=start_dt,
                            end_date=end_dt
                        )

                        validation_results.append(report)
                        progress.update(task, advance=1)

                    except Exception as e:
                        logger.error(f"Error validating dataset {dataset_name}: {e}")
                        console.print(f"[red]Error validating {dataset_name}: {e}[/red]")
                        progress.update(task, advance=1)
            else:
                # Run comprehensive validation
                task = progress.add_task("[blue]Running comprehensive validation...", total=100)

                progress.update(task, advance=20, description="[blue]Validating market data...")
                market_report = validator.validate_market_data(
                    symbols=symbol_list,
                    start_date=start_dt,
                    end_date=end_dt
                )
                validation_results.append(market_report)

                progress.update(task, advance=40, description="[blue]Validating fundamental data...")
                fundamental_report = validator.validate_fundamental_data(
                    symbols=symbol_list,
                    start_date=start_dt,
                    end_date=end_dt
                )
                validation_results.append(fundamental_report)

                progress.update(task, advance=80, description="[blue]Validating data consistency...")
                consistency_report = validator.validate_data_consistency(
                    symbols=symbol_list,
                    start_date=start_dt,
                    end_date=end_dt
                )
                validation_results.append(consistency_report)

                progress.update(task, advance=100, description="[green]Validation complete")

        # Process results
        all_issues = []
        summary_stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'issues_by_severity': {severity.value: 0 for severity in ValidationSeverity}
        }

        for report in validation_results:
            summary_stats['total_checks'] += len(report.checks)
            summary_stats['passed_checks'] += len([c for c in report.checks if c.passed])

            for issue in report.issues:
                if issue.severity.value >= min_severity.value:
                    all_issues.append(issue)
                summary_stats['issues_by_severity'][issue.severity.name] += 1

        # Display results
        if output_format == 'json':
            result_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'summary': summary_stats,
                'issues': [
                    {
                        'severity': issue.severity.name,
                        'category': issue.category,
                        'message': issue.message,
                        'symbol': issue.symbol,
                        'data_type': issue.data_type.value if issue.data_type else None,
                        'timestamp': issue.timestamp.isoformat() if issue.timestamp else None,
                        'metadata': issue.metadata
                    }
                    for issue in all_issues
                ]
            }

            if output:
                with open(output, 'w') as f:
                    json.dump(result_data, f, indent=2)
                console.print(f"[green]Report saved to {output}[/green]")
            else:
                console.print(json.dumps(result_data, indent=2))

        else:
            # Display summary
            console.print()
            console.print("[bold]Validation Summary[/bold]")
            console.print(f"Total checks: {summary_stats['total_checks']}")
            console.print(f"Passed checks: {summary_stats['passed_checks']}")

            pass_rate = summary_stats['passed_checks'] / max(summary_stats['total_checks'], 1)
            console.print(f"Pass rate: {format_percentage(pass_rate)}")
            console.print()

            # Issues by severity
            severity_data = []
            for sev_name, count in summary_stats['issues_by_severity'].items():
                if count > 0:
                    severity_data.append({
                        'Severity': sev_name,
                        'Count': count
                    })

            if severity_data:
                table = format_table(severity_data, title="Issues by Severity")
                console.print(table)
                console.print()

            # Detailed issues
            if all_issues:
                issues_data = []
                for issue in all_issues[:50]:  # Limit to 50 most recent
                    issues_data.append({
                        'Severity': issue.severity.name,
                        'Category': issue.category,
                        'Symbol': issue.symbol or 'N/A',
                        'Message': issue.message[:80] + '...' if len(issue.message) > 80 else issue.message
                    })

                table = format_table(issues_data, title="Validation Issues", max_rows=50)
                console.print(table)

                if len(all_issues) > 50:
                    console.print(f"[dim]... and {len(all_issues) - 50} more issues[/dim]")

                if output:
                    # Save detailed report
                    with open(output, 'w') as f:
                        if output_format == 'csv':
                            import csv
                            writer = csv.writer(f)
                            writer.writerow(['Severity', 'Category', 'Symbol', 'Data Type', 'Message', 'Timestamp'])
                            for issue in all_issues:
                                writer.writerow([
                                    issue.severity.name,
                                    issue.category,
                                    issue.symbol,
                                    issue.data_type.value if issue.data_type else '',
                                    issue.message,
                                    issue.timestamp.isoformat() if issue.timestamp else ''
                                ])
                        else:
                            f.write("Validation Report\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")

                            for issue in all_issues:
                                f.write(f"[{issue.severity.name}] {issue.category}\n")
                                f.write(f"Symbol: {issue.symbol or 'N/A'}\n")
                                f.write(f"Message: {issue.message}\n")
                                if issue.timestamp:
                                    f.write(f"Time: {issue.timestamp.isoformat()}\n")
                                f.write("\n")

                    console.print(f"[green]Detailed report saved to {output}[/green]")
            else:
                console.print("[green]No validation issues found![/green]")

    except Exception as e:
        logger.error(f"Error running validation: {e}")
        console.print(f"[red]Error running validation: {e}[/red]")
        sys.exit(1)


@validation_group.command()
@click.option('--days', default=7, help='Number of days to check')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def history(ctx, days: int, output_format: str):
    """Show validation history and trends."""
    console = ctx.obj['console']

    try:
        # Create validator
        auth = create_finlab_authenticator()
        config = FinLabConfig(auth_config=auth.config)
        connector = FinLabConnector(config)
        validator = DataValidator(connector)

        console.print(f"[blue]Loading validation history for last {days} days...[/blue]")

        # Get validation history
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        history_reports = validator.get_validation_history(start_date, end_date)

        if output_format == 'json':
            history_data = [
                {
                    'date': report.timestamp.date().isoformat(),
                    'total_checks': len(report.checks),
                    'passed_checks': len([c for c in report.checks if c.passed]),
                    'issues_by_severity': {
                        severity.name: len([i for i in report.issues if i.severity == severity])
                        for severity in ValidationSeverity
                    }
                }
                for report in history_reports
            ]
            console.print(json.dumps(history_data, indent=2))
        else:
            # Display history table
            if history_reports:
                history_data = []
                for report in history_reports:
                    total_checks = len(report.checks)
                    passed_checks = len([c for c in report.checks if c.passed])
                    pass_rate = passed_checks / max(total_checks, 1)

                    critical_issues = len([i for i in report.issues if i.severity == ValidationSeverity.CRITICAL])
                    error_issues = len([i for i in report.issues if i.severity == ValidationSeverity.ERROR])
                    warning_issues = len([i for i in report.issues if i.severity == ValidationSeverity.WARNING])

                    history_data.append({
                        'Date': report.timestamp.date().isoformat(),
                        'Checks': total_checks,
                        'Pass Rate': format_percentage(pass_rate),
                        'Critical': critical_issues,
                        'Errors': error_issues,
                        'Warnings': warning_issues
                    })

                table = format_table(history_data, title=f"Validation History ({days} days)")
                console.print(table)
            else:
                console.print("[yellow]No validation history found[/yellow]")

    except Exception as e:
        logger.error(f"Error getting validation history: {e}")
        console.print(f"[red]Error getting validation history: {e}[/red]")
        sys.exit(1)


@validation_group.command()
@click.argument('symbol')
@click.option('--dataset', help='Specific dataset to check')
@click.option('--days', default=30, help='Number of days to analyze')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def symbol(ctx, symbol: str, dataset: Optional[str], days: int, output_format: str):
    """Validate data for a specific symbol."""
    console = ctx.obj['console']

    try:
        # Create validator
        auth = create_finlab_authenticator()
        config = FinLabConfig(auth_config=auth.config)
        connector = FinLabConnector(config)
        validator = DataValidator(connector)

        console.print(f"[blue]Validating data for symbol {symbol}...[/blue]")

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Run symbol-specific validation
        report = validator.validate_symbol(
            symbol=symbol,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date
        )

        if output_format == 'json':
            result_data = {
                'symbol': symbol,
                'dataset': dataset,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'summary': {
                    'total_checks': len(report.checks),
                    'passed_checks': len([c for c in report.checks if c.passed]),
                    'issues_count': len(report.issues)
                },
                'issues': [
                    {
                        'severity': issue.severity.name,
                        'category': issue.category,
                        'message': issue.message,
                        'timestamp': issue.timestamp.isoformat() if issue.timestamp else None
                    }
                    for issue in report.issues
                ]
            }
            console.print(json.dumps(result_data, indent=2))
        else:
            # Display symbol validation results
            console.print(f"[bold]Validation Results for {symbol}[/bold]")

            if dataset:
                console.print(f"Dataset: {dataset}")
            console.print(f"Date Range: {start_date} to {end_date}")
            console.print()

            total_checks = len(report.checks)
            passed_checks = len([c for c in report.checks if c.passed])
            pass_rate = passed_checks / max(total_checks, 1)

            console.print(f"Total Checks: {total_checks}")
            console.print(f"Passed: {passed_checks}")
            console.print(f"Pass Rate: {format_percentage(pass_rate)}")
            console.print()

            if report.issues:
                issues_data = []
                for issue in report.issues:
                    issues_data.append({
                        'Severity': issue.severity.name,
                        'Category': issue.category,
                        'Message': issue.message,
                        'Time': format_timestamp(issue.timestamp) if issue.timestamp else 'N/A'
                    })

                table = format_table(issues_data, title="Issues Found")
                console.print(table)
            else:
                console.print("[green]No issues found for this symbol![/green]")

    except Exception as e:
        logger.error(f"Error validating symbol {symbol}: {e}")
        console.print(f"[red]Error validating symbol {symbol}: {e}[/red]")
        sys.exit(1)