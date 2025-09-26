"""
System Monitoring Commands.

Commands for monitoring system health, performance metrics,
alerts, and pipeline status.
"""

import sys
import logging
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress
from rich.spinner import Spinner

from ...data.pipeline.monitoring import (
    get_pipeline_monitor, PipelineMonitor, AlertLevel, PerformanceMetric, MetricType
)
from ...data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
from ...data.ingestion.finlab_auth import create_finlab_authenticator
from ..utils.formatting import (
    format_table, format_status, format_duration, format_bytes,
    format_percentage, format_timestamp
)
from ..utils.errors import PipelineError, handle_cli_error

logger = logging.getLogger(__name__)


@click.group(name='monitoring')
def monitoring_group():
    """System monitoring and health check commands."""
    pass


@monitoring_group.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.option('--refresh', '-r', is_flag=True,
              help='Refresh status (don\'t use cached data)')
@click.pass_context
def health(ctx, output_format: str, refresh: bool):
    """Show comprehensive system health status."""
    console = ctx.obj['console']

    try:
        monitor = get_pipeline_monitor()

        if refresh:
            monitor.refresh_all_status()

        # Get health information
        system_health = monitor.get_system_health()
        component_health = monitor.get_component_health()
        performance_metrics = monitor.get_recent_metrics(hours=1)

        health_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": {
                "overall_score": system_health.overall_score,
                "status": system_health.status.value,
                "uptime_seconds": system_health.uptime_seconds,
                "last_error": system_health.last_error.isoformat() if system_health.last_error else None
            },
            "components": {
                name: {
                    "health_score": health.health_score,
                    "status": health.status.value,
                    "last_check": health.last_check.isoformat() if health.last_check else None,
                    "error_count": health.error_count,
                    "response_time_ms": health.response_time_ms
                }
                for name, health in component_health.items()
            },
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "type": metric.metric_type.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat()
                }
                for metric in performance_metrics
            ]
        }

        if output_format == 'json':
            console.print(json.dumps(health_data, indent=2))
        else:
            # Display formatted health status
            console.print(f"[bold]System Health Status[/bold] - {format_timestamp(datetime.utcnow())}")
            console.print()

            # Overall health
            overall_score = system_health.overall_score
            status_color = "green" if overall_score >= 0.8 else "yellow" if overall_score >= 0.6 else "red"
            console.print(f"Overall Health: [{status_color}]{format_percentage(overall_score)}[/{status_color}]")
            console.print(f"System Status: {format_status(system_health.status.value, overall_score >= 0.8)}")
            console.print(f"Uptime: {format_duration(system_health.uptime_seconds)}")

            if system_health.last_error:
                console.print(f"Last Error: {format_timestamp(system_health.last_error)}")
            console.print()

            # Component health
            if component_health:
                health_data_list = []
                for name, health in component_health.items():
                    health_data_list.append({
                        "Component": name,
                        "Health": f"{health.health_score:.1f}",
                        "Status": health.status.value,
                        "Response Time": f"{health.response_time_ms}ms" if health.response_time_ms else "N/A",
                        "Errors": str(health.error_count),
                        "Last Check": format_timestamp(health.last_check) if health.last_check else "Never"
                    })

                table = format_table(health_data_list, title="Component Health")
                console.print(table)
                console.print()

            # Key performance metrics
            if performance_metrics:
                metrics_by_type = {}
                for metric in performance_metrics:
                    if metric.metric_type not in metrics_by_type:
                        metrics_by_type[metric.metric_type] = []
                    metrics_by_type[metric.metric_type].append(metric)

                for metric_type, metrics in metrics_by_type.items():
                    if len(metrics) > 0:
                        console.print(f"[bold]{metric_type.value.title()} Metrics:[/bold]")

                        metrics_data = []
                        for metric in metrics[-10:]:  # Show last 10 metrics of each type
                            value_str = str(metric.value)
                            if metric.unit:
                                value_str += f" {metric.unit}"

                            metrics_data.append({
                                "Name": metric.name,
                                "Value": value_str,
                                "Time": format_timestamp(metric.timestamp)
                            })

                        table = format_table(metrics_data)
                        console.print(table)
                        console.print()

    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        console.print(f"[red]Error getting health status: {e}[/red]")
        sys.exit(1)


@monitoring_group.command()
@click.option('--hours', default=24, help='Hours of alerts to show')
@click.option('--level', type=click.Choice(['info', 'warning', 'error', 'critical']),
              help='Filter by alert level')
@click.option('--component', help='Filter by component name')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def alerts(ctx, hours: int, level: Optional[str], component: Optional[str], output_format: str):
    """Show system alerts and notifications."""
    console = ctx.obj['console']

    try:
        monitor = get_pipeline_monitor()

        # Get alerts
        alert_level = AlertLevel(level.upper()) if level else None
        alerts = monitor.get_recent_alerts(
            hours=hours,
            level=alert_level,
            component=component
        )

        alerts_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "filters": {
                "hours": hours,
                "level": level,
                "component": component
            },
            "alerts": [
                {
                    "level": alert.level.value,
                    "title": alert.title,
                    "message": alert.message,
                    "component": alert.component,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
                }
                for alert in alerts
            ]
        }

        if output_format == 'json':
            console.print(json.dumps(alerts_data, indent=2))
        else:
            # Display alerts
            console.print(f"[bold]System Alerts[/bold] - Last {hours} hours")

            if level:
                console.print(f"Filter: {level.upper()} level and above")
            if component:
                console.print(f"Component: {component}")
            console.print()

            if alerts:
                alert_display = []
                for alert in alerts:
                    level_colors = {
                        'info': 'blue',
                        'warning': 'yellow',
                        'error': 'red',
                        'critical': 'bold red'
                    }
                    level_color = level_colors.get(alert.level.value, 'white')

                    status_text = "RESOLVED" if alert.resolved else "ACTIVE"
                    status_color = "green" if alert.resolved else "red"

                    alert_display.append({
                        "Level": f"[{level_color}]{alert.level.value.upper()}[/{level_color}]",
                        "Component": alert.component,
                        "Title": alert.title,
                        "Status": f"[{status_color}]{status_text}[/{status_color}]",
                        "Time": format_timestamp(alert.timestamp)
                    })

                table = format_table(alert_display, title="Recent Alerts")
                console.print(table)

                # Show detailed messages for critical alerts
                critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL and not a.resolved]
                if critical_alerts:
                    console.print()
                    console.print("[bold red]Critical Alert Details:[/bold red]")
                    for alert in critical_alerts:
                        console.print(f"[red]• {alert.component}: {alert.message}[/red]")

            else:
                console.print("[green]No alerts found![/green]")

    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        console.print(f"[red]Error getting alerts: {e}[/red]")
        sys.exit(1)


@monitoring_group.command()
@click.option('--hours', default=6, help='Hours of metrics to show')
@click.option('--metric-type', type=click.Choice(['counter', 'gauge', 'histogram', 'timing']),
              help='Filter by metric type')
@click.option('--component', help='Filter by component name')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def metrics(ctx, hours: int, metric_type: Optional[str], component: Optional[str], output_format: str):
    """Show performance metrics and statistics."""
    console = ctx.obj['console']

    try:
        monitor = get_pipeline_monitor()

        # Get performance metrics
        metrics = monitor.get_recent_metrics(
            hours=hours,
            metric_type=MetricType(metric_type.upper()) if metric_type else None,
            component=component
        )

        metrics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "filters": {
                "hours": hours,
                "metric_type": metric_type,
                "component": component
            },
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "type": metric.metric_type.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "tags": metric.tags
                }
                for metric in metrics
            ]
        }

        if output_format == 'json':
            console.print(json.dumps(metrics_data, indent=2))
        else:
            # Display metrics
            console.print(f"[bold]Performance Metrics[/bold] - Last {hours} hours")

            if metric_type:
                console.print(f"Type: {metric_type.upper()}")
            if component:
                console.print(f"Component: {component}")
            console.print()

            if metrics:
                # Group metrics by name for summary statistics
                from collections import defaultdict
                import statistics

                metrics_by_name = defaultdict(list)
                for metric in metrics:
                    metrics_by_name[metric.name].append(metric)

                summary_data = []
                for name, metric_list in metrics_by_name.items():
                    values = [m.value for m in metric_list]
                    latest_metric = max(metric_list, key=lambda x: x.timestamp)

                    summary_data.append({
                        "Metric": name,
                        "Type": latest_metric.metric_type.value,
                        "Latest": f"{values[-1]:.2f}" + (f" {latest_metric.unit}" if latest_metric.unit else ""),
                        "Average": f"{statistics.mean(values):.2f}",
                        "Min": f"{min(values):.2f}",
                        "Max": f"{max(values):.2f}",
                        "Count": len(values)
                    })

                table = format_table(summary_data, title="Metrics Summary")
                console.print(table)

                # Show recent individual metrics
                if len(metrics) <= 20:
                    console.print()
                    recent_data = []
                    for metric in sorted(metrics, key=lambda x: x.timestamp, reverse=True)[:20]:
                        value_str = f"{metric.value:.2f}"
                        if metric.unit:
                            value_str += f" {metric.unit}"

                        recent_data.append({
                            "Metric": metric.name,
                            "Value": value_str,
                            "Type": metric.metric_type.value,
                            "Time": format_timestamp(metric.timestamp)
                        })

                    table = format_table(recent_data, title="Recent Metrics")
                    console.print(table)

            else:
                console.print("[yellow]No metrics found for the specified criteria[/yellow]")

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        console.print(f"[red]Error getting metrics: {e}[/red]")
        sys.exit(1)


@monitoring_group.command()
@click.option('--interval', default=5, help='Refresh interval in seconds')
@click.option('--duration', default=300, help='Total monitoring duration in seconds')
@click.pass_context
def watch(ctx, interval: int, duration: int):
    """Watch system status in real-time."""
    console = ctx.obj['console']

    try:
        monitor = get_pipeline_monitor()
        start_time = time.time()

        console.print(f"[blue]Monitoring system status (refresh: {interval}s, duration: {duration}s)[/blue]")
        console.print("[dim]Press Ctrl+C to exit[/dim]")

        with Live(console=console, refresh_per_second=1/interval) as live:
            while time.time() - start_time < duration:
                try:
                    # Create layout
                    layout = Layout()
                    layout.split_column(
                        Layout(name="header", size=3),
                        Layout(name="body"),
                        Layout(name="footer", size=3)
                    )

                    layout["body"].split_row(
                        Layout(name="left"),
                        Layout(name="right")
                    )

                    # Header
                    current_time = datetime.utcnow()
                    layout["header"].update(
                        Panel(f"FinLab Pipeline Monitor - {format_timestamp(current_time)}",
                              style="bold blue")
                    )

                    # System status (left panel)
                    system_health = monitor.get_system_health()
                    component_health = monitor.get_component_health()

                    status_text = f"Overall Health: {format_percentage(system_health.overall_score)}\n"
                    status_text += f"Status: {system_health.status.value}\n"
                    status_text += f"Uptime: {format_duration(system_health.uptime_seconds)}\n\n"

                    status_text += "Component Status:\n"
                    for name, health in list(component_health.items())[:10]:
                        status_text += f"  {name}: {health.status.value} ({health.health_score:.1f})\n"

                    layout["left"].update(Panel(status_text, title="System Status", border_style="green"))

                    # Recent alerts (right panel)
                    recent_alerts = monitor.get_recent_alerts(hours=1)
                    alerts_text = ""

                    if recent_alerts:
                        for alert in recent_alerts[-10:]:  # Last 10 alerts
                            status = "RESOLVED" if alert.resolved else "ACTIVE"
                            alerts_text += f"{alert.level.value.upper()}: {alert.title} ({status})\n"
                    else:
                        alerts_text = "No recent alerts"

                    layout["right"].update(Panel(alerts_text, title="Recent Alerts", border_style="yellow"))

                    # Footer
                    elapsed = int(time.time() - start_time)
                    remaining = max(0, duration - elapsed)
                    layout["footer"].update(
                        Panel(f"Elapsed: {elapsed}s | Remaining: {remaining}s | Press Ctrl+C to exit",
                              style="dim")
                    )

                    live.update(layout)
                    time.sleep(interval)

                except KeyboardInterrupt:
                    console.print("\n[blue]Monitoring stopped by user[/blue]")
                    break

            else:
                console.print(f"\n[blue]Monitoring completed ({duration}s)[/blue]")

    except Exception as e:
        logger.error(f"Error monitoring system: {e}")
        console.print(f"[red]Error monitoring system: {e}[/red]")
        sys.exit(1)


@monitoring_group.command()
@click.argument('component')
@click.option('--timeout', default=30, help='Health check timeout in seconds')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.pass_context
def check(ctx, component: str, timeout: int, output_format: str):
    """Run health check for a specific component."""
    console = ctx.obj['console']

    try:
        monitor = get_pipeline_monitor()

        console.print(f"[blue]Running health check for {component}...[/blue]")

        with console.status(f"[blue]Checking {component} health..."):
            # Run component-specific health check
            start_time = time.time()
            health_result = monitor.check_component_health(component, timeout=timeout)
            check_duration = time.time() - start_time

        check_data = {
            "component": component,
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": check_duration,
            "result": {
                "healthy": health_result.healthy,
                "status": health_result.status.value,
                "health_score": health_result.health_score,
                "response_time_ms": health_result.response_time_ms,
                "error_count": health_result.error_count,
                "message": health_result.message
            }
        }

        if output_format == 'json':
            console.print(json.dumps(check_data, indent=2))
        else:
            # Display health check results
            console.print(f"[bold]Health Check Results: {component}[/bold]")
            console.print(f"Check Duration: {format_duration(check_duration)}")
            console.print()

            status_color = "green" if health_result.healthy else "red"
            console.print(f"Status: [{status_color}]{health_result.status.value}[/{status_color}]")
            console.print(f"Health Score: {format_percentage(health_result.health_score)}")

            if health_result.response_time_ms:
                console.print(f"Response Time: {health_result.response_time_ms}ms")

            if health_result.error_count > 0:
                console.print(f"Recent Errors: {health_result.error_count}")

            if health_result.message:
                console.print(f"Message: {health_result.message}")

            # Overall result
            console.print()
            if health_result.healthy:
                console.print(f"[green]✓ {component} is healthy[/green]")
            else:
                console.print(f"[red]✗ {component} has issues[/red]")

    except Exception as e:
        logger.error(f"Error checking component {component}: {e}")
        console.print(f"[red]Error checking component {component}: {e}[/red]")
        sys.exit(1)