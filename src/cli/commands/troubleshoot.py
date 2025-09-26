"""
Troubleshooting and Diagnostic Commands.

Commands for diagnosing system issues, debugging problems,
and providing system diagnostics.
"""

import sys
import logging
import json
import traceback
import subprocess
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel

from ...data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
from ...data.ingestion.finlab_auth import create_finlab_authenticator
from ...data.pipeline.monitoring import get_pipeline_monitor
from ...data.pipeline.data_validation import DataValidator
from ..utils.formatting import format_table, format_status, format_bytes, format_timestamp
from ..utils.errors import PipelineError, AuthenticationError, DataError

logger = logging.getLogger(__name__)


@click.group(name='troubleshoot')
def troubleshoot_group():
    """Troubleshooting and diagnostic commands."""
    pass


@troubleshoot_group.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
@click.option('--include-sensitive', is_flag=True,
              help='Include potentially sensitive information')
@click.pass_context
def diagnose(ctx, output_format: str, include_sensitive: bool):
    """Run comprehensive system diagnostics."""
    console = ctx.obj['console']

    try:
        console.print("[blue]Running system diagnostics...[/blue]")

        diagnostics = {
            "timestamp": datetime.utcnow().isoformat(),
            "python_info": {},
            "system_info": {},
            "authentication": {},
            "database": {},
            "dependencies": {},
            "configuration": {},
            "issues": []
        }

        # Python information
        console.print("[dim]Checking Python environment...[/dim]")
        import platform
        import sys

        diagnostics["python_info"] = {
            "version": sys.version,
            "version_info": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "executable": sys.executable,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0]
        }

        # System information
        diagnostics["system_info"] = {
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }

        # Check for WSL
        try:
            with open('/proc/version', 'r') as f:
                proc_version = f.read().lower()
                if 'microsoft' in proc_version:
                    diagnostics["system_info"]["wsl"] = True
                    diagnostics["system_info"]["wsl_version"] = "WSL2" if "wsl2" in proc_version else "WSL1"
        except:
            diagnostics["system_info"]["wsl"] = False

        # Authentication check
        console.print("[dim]Checking authentication...[/dim]")
        try:
            auth = create_finlab_authenticator()
            diagnostics["authentication"] = {
                "has_token_auth": auth.config.has_token_auth,
                "has_db_auth": auth.config.has_db_auth,
                "config_source": "configured" if auth.config else "not_configured"
            }

            if include_sensitive:
                diagnostics["authentication"].update({
                    "db_host": auth.config.db_host if auth.config else None,
                    "db_port": auth.config.db_port if auth.config else None,
                    "db_database": auth.config.db_database if auth.config else None,
                    "db_username": auth.config.db_username if auth.config else None
                })

        except Exception as e:
            diagnostics["authentication"] = {
                "error": str(e),
                "has_token_auth": False,
                "has_db_auth": False
            }
            diagnostics["issues"].append({
                "category": "authentication",
                "severity": "error",
                "message": f"Authentication setup failed: {e}"
            })

        # Database connection check
        console.print("[dim]Checking database connection...[/dim]")
        try:
            if diagnostics["authentication"].get("has_db_auth") or diagnostics["authentication"].get("has_token_auth"):
                auth = create_finlab_authenticator()
                config = FinLabConfig(auth_config=auth.config)
                connector = FinLabConnector(config)

                connection_test = connector.test_connection()
                diagnostics["database"] = {
                    "connection_available": connection_test,
                    "connection_info": connector.get_connection_info() if connection_test else None
                }

                if not connection_test:
                    diagnostics["issues"].append({
                        "category": "database",
                        "severity": "error",
                        "message": "Database connection test failed"
                    })

            else:
                diagnostics["database"] = {
                    "connection_available": False,
                    "reason": "No authentication configured"
                }

        except Exception as e:
            diagnostics["database"] = {
                "connection_available": False,
                "error": str(e)
            }
            diagnostics["issues"].append({
                "category": "database",
                "severity": "error",
                "message": f"Database connection error: {e}"
            })

        # Dependency check
        console.print("[dim]Checking dependencies...[/dim]")
        required_packages = [
            'click', 'rich', 'sqlalchemy', 'pandas', 'numpy', 'aiohttp',
            'psutil', 'cerberus', 'typer'
        ]

        dependencies_status = {}
        missing_deps = []

        for package in required_packages:
            try:
                __import__(package)
                dependencies_status[package] = "available"
            except ImportError:
                dependencies_status[package] = "missing"
                missing_deps.append(package)

        diagnostics["dependencies"] = {
            "status": dependencies_status,
            "missing_count": len(missing_deps),
            "missing_packages": missing_deps
        }

        if missing_deps:
            diagnostics["issues"].append({
                "category": "dependencies",
                "severity": "error",
                "message": f"Missing required packages: {', '.join(missing_deps)}"
            })

        # Configuration check
        console.print("[dim]Checking configuration...[/dim]")
        config = ctx.obj['config']
        diagnostics["configuration"] = {
            "config_file": config.config_file,
            "has_finlab_auth": config.has_finlab_auth,
            "log_dir_exists": Path(config.log_dir).exists(),
            "log_dir_writable": Path(config.log_dir).is_dir() and os.access(config.log_dir, os.W_OK)
        }

        # Check log directory
        if not diagnostics["configuration"]["log_dir_exists"]:
            diagnostics["issues"].append({
                "category": "configuration",
                "severity": "warning",
                "message": f"Log directory does not exist: {config.log_dir}"
            })
        elif not diagnostics["configuration"]["log_dir_writable"]:
            diagnostics["issues"].append({
                "category": "configuration",
                "severity": "warning",
                "message": f"Log directory is not writable: {config.log_dir}"
            })

        # Pipeline status check
        console.print("[dim]Checking pipeline status...[/dim]")
        try:
            monitor = get_pipeline_monitor()
            system_health = monitor.get_system_health()
            diagnostics["pipeline"] = {
                "monitor_available": True,
                "system_health_score": system_health.overall_score,
                "system_status": system_health.status.value,
                "uptime_seconds": system_health.uptime_seconds
            }

            if system_health.overall_score < 0.5:
                diagnostics["issues"].append({
                    "category": "pipeline",
                    "severity": "warning",
                    "message": f"System health score is low: {system_health.overall_score:.1f}"
                })

        except Exception as e:
            diagnostics["pipeline"] = {
                "monitor_available": False,
                "error": str(e)
            }

        # Overall health assessment
        critical_issues = [i for i in diagnostics["issues"] if i["severity"] == "error"]
        warning_issues = [i for i in diagnostics["issues"] if i["severity"] == "warning"]

        diagnostics["summary"] = {
            "overall_health": "healthy" if not critical_issues else "unhealthy",
            "critical_issues": len(critical_issues),
            "warnings": len(warning_issues),
            "recommendations": []
        }

        # Generate recommendations
        if missing_deps:
            diagnostics["summary"]["recommendations"].append(
                f"Install missing dependencies: pip install {' '.join(missing_deps)}"
            )

        if not diagnostics["authentication"].get("has_db_auth") and not diagnostics["authentication"].get("has_token_auth"):
            diagnostics["summary"]["recommendations"].append(
                "Configure FinLab authentication using: finlab-cli config init --interactive"
            )

        if not diagnostics["configuration"]["log_dir_writable"]:
            diagnostics["summary"]["recommendations"].append(
                f"Create writable log directory: mkdir -p {config.log_dir}"
            )

        # Output results
        if output_format == 'json':
            console.print(json.dumps(diagnostics, indent=2))
        else:
            # Display formatted diagnostics
            console.print()
            console.print("[bold]System Diagnostics Report[/bold]")
            console.print(f"Generated: {format_timestamp(datetime.utcnow())}")
            console.print()

            # Overall status
            overall_health = diagnostics["summary"]["overall_health"]
            health_color = "green" if overall_health == "healthy" else "red"
            console.print(f"Overall Status: [{health_color}]{overall_health.upper()}[/{health_color}]")
            console.print(f"Critical Issues: {diagnostics['summary']['critical_issues']}")
            console.print(f"Warnings: {diagnostics['summary']['warnings']}")
            console.print()

            # System information
            sys_info = diagnostics["system_info"]
            python_info = diagnostics["python_info"]

            console.print("[bold]System Information[/bold]")
            info_data = [
                {"Component": "Operating System", "Value": f"{sys_info['platform']} {sys_info['release']}"},
                {"Component": "Architecture", "Value": sys_info['machine']},
                {"Component": "Python Version", "Value": python_info['version_info']},
                {"Component": "Python Executable", "Value": python_info['executable']},
            ]

            if sys_info.get("wsl"):
                info_data.append({"Component": "WSL Version", "Value": sys_info.get("wsl_version", "Unknown")})

            table = format_table(info_data)
            console.print(table)
            console.print()

            # Authentication status
            auth_info = diagnostics["authentication"]
            console.print("[bold]Authentication Status[/bold]")

            if auth_info.get("error"):
                console.print(f"[red]Error: {auth_info['error']}[/red]")
            else:
                console.print(f"Token Auth: {'✓' if auth_info['has_token_auth'] else '✗'}")
                console.print(f"Database Auth: {'✓' if auth_info['has_db_auth'] else '✗'}")

            console.print()

            # Database status
            db_info = diagnostics["database"]
            console.print("[bold]Database Status[/bold]")

            if db_info.get("error"):
                console.print(f"[red]Error: {db_info['error']}[/red]")
            elif db_info["connection_available"]:
                console.print("[green]✓ Database connection available[/green]")
                if db_info.get("connection_info"):
                    for key, value in db_info["connection_info"].items():
                        console.print(f"  {key}: {value}")
            else:
                reason = db_info.get("reason", "Unknown")
                console.print(f"[red]✗ Database connection unavailable: {reason}[/red]")

            console.print()

            # Dependencies
            deps_info = diagnostics["dependencies"]
            console.print("[bold]Dependencies[/bold]")

            if deps_info["missing_count"] > 0:
                console.print(f"[red]Missing {deps_info['missing_count']} required packages[/red]")
                for pkg in deps_info["missing_packages"]:
                    console.print(f"  [red]✗ {pkg}[/red]")
            else:
                console.print("[green]✓ All required dependencies available[/green]")

            console.print()

            # Issues and recommendations
            if diagnostics["issues"]:
                console.print("[bold]Issues Found[/bold]")

                issues_data = []
                for issue in diagnostics["issues"]:
                    severity_color = {"error": "red", "warning": "yellow", "info": "blue"}.get(issue["severity"], "white")
                    issues_data.append({
                        "Category": issue["category"].title(),
                        "Severity": f"[{severity_color}]{issue['severity'].upper()}[/{severity_color}]",
                        "Message": issue["message"]
                    })

                table = format_table(issues_data)
                console.print(table)
                console.print()

            if diagnostics["summary"]["recommendations"]:
                console.print("[bold]Recommendations[/bold]")
                for i, rec in enumerate(diagnostics["summary"]["recommendations"], 1):
                    console.print(f"{i}. {rec}")

    except Exception as e:
        logger.error(f"Error running diagnostics: {e}")
        console.print(f"[red]Error running diagnostics: {e}[/red]")
        if output_format == 'json':
            error_data = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            console.print(json.dumps(error_data, indent=2))
        sys.exit(1)


@troubleshoot_group.command()
@click.argument('component', required=False)
@click.option('--lines', '-n', default=100, help='Number of log lines to show')
@click.option('--level', type=click.Choice(['debug', 'info', 'warning', 'error']),
              help='Filter by log level')
@click.pass_context
def logs(ctx, component: Optional[str], lines: int, level: Optional[str]):
    """Show recent error logs and debug information."""
    console = ctx.obj['console']

    try:
        console.print("[blue]Fetching recent logs...[/blue]")

        # Get pipeline monitor
        monitor = get_pipeline_monitor()

        # Fetch logs
        log_entries = monitor.get_log_entries(
            limit=lines,
            level=level,
            component=component
        )

        if not log_entries:
            console.print("[yellow]No log entries found[/yellow]")
            return

        console.print(f"[bold]Recent Logs[/bold] ({len(log_entries)} entries)")
        if component:
            console.print(f"Component: {component}")
        if level:
            console.print(f"Level: {level.upper()}")
        console.print()

        # Group logs by severity
        from collections import defaultdict
        logs_by_level = defaultdict(list)

        for entry in log_entries:
            logs_by_level[entry.level].append(entry)

        # Display logs by severity (errors first)
        for log_level in ['ERROR', 'WARNING', 'INFO', 'DEBUG']:
            entries = logs_by_level.get(log_level, [])
            if not entries:
                continue

            level_color = {
                'ERROR': 'red',
                'WARNING': 'yellow',
                'INFO': 'blue',
                'DEBUG': 'dim'
            }.get(log_level, 'white')

            console.print(f"[bold {level_color}]{log_level} ({len(entries)} entries)[/bold {level_color}]")

            for entry in entries[-20:]:  # Show last 20 of each level
                timestamp = format_timestamp(entry.timestamp)
                console.print(
                    f"[dim]{timestamp}[/dim] "
                    f"[{level_color}]{entry.component}[/{level_color}] "
                    f"{entry.message}"
                )

            console.print()

    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        console.print(f"[red]Error fetching logs: {e}[/red]")
        sys.exit(1)


@troubleshoot_group.command()
@click.option('--fix-permissions', is_flag=True, help='Fix file permissions')
@click.option('--create-dirs', is_flag=True, help='Create missing directories')
@click.option('--reset-config', is_flag=True, help='Reset configuration to defaults')
@click.option('--dry-run', is_flag=True, help='Show what would be fixed without doing it')
@click.pass_context
def repair(ctx, fix_permissions: bool, create_dirs: bool, reset_config: bool, dry_run: bool):
    """Attempt to repair common system issues."""
    console = ctx.obj['console']
    config = ctx.obj['config']

    try:
        console.print("[blue]Running system repair...[/blue]")

        if dry_run:
            console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")

        repairs_made = []
        issues_found = []

        # Check and fix log directory
        log_dir = Path(config.log_dir)
        if not log_dir.exists():
            issues_found.append(f"Log directory missing: {log_dir}")
            if create_dirs:
                if not dry_run:
                    log_dir.mkdir(parents=True, exist_ok=True)
                repairs_made.append(f"Created log directory: {log_dir}")
            else:
                console.print(f"[yellow]Issue: Log directory missing: {log_dir}[/yellow]")
                console.print("Use --create-dirs to fix")

        elif not log_dir.is_dir():
            issues_found.append(f"Log directory is not a directory: {log_dir}")
        elif fix_permissions and not dry_run:
            try:
                # Try to make directory writable
                import stat
                current_mode = log_dir.stat().st_mode
                new_mode = current_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR
                if current_mode != new_mode:
                    log_dir.chmod(new_mode)
                    repairs_made.append(f"Fixed permissions for: {log_dir}")
            except Exception as e:
                issues_found.append(f"Cannot fix permissions for {log_dir}: {e}")

        # Check Python dependencies
        console.print("[dim]Checking Python dependencies...[/dim]")
        missing_deps = []
        required_packages = ['click', 'rich', 'sqlalchemy', 'pandas', 'numpy']

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_deps.append(package)

        if missing_deps:
            issues_found.append(f"Missing dependencies: {', '.join(missing_deps)}")
            console.print(f"[yellow]Missing dependencies: {', '.join(missing_deps)}[/yellow]")
            console.print("Run: pip install " + " ".join(missing_deps))

        # Check configuration
        if reset_config:
            config_file = config.config_file
            if config_file:
                if not dry_run:
                    # Create backup
                    backup_file = f"{config_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    Path(config_file).rename(backup_file)

                    # Create new config
                    from ..utils.config import CliConfig, save_cli_config
                    new_config = CliConfig()
                    save_cli_config(new_config, config_file)

                    repairs_made.append(f"Reset configuration (backup: {backup_file})")
                else:
                    repairs_made.append(f"Would reset configuration: {config_file}")

        # Check authentication
        try:
            auth = create_finlab_authenticator()
            if not auth.config.has_db_auth and not auth.config.has_token_auth:
                issues_found.append("No authentication configured")
                console.print("[yellow]No authentication configured[/yellow]")
                console.print("Run: finlab-cli config init --interactive")
        except Exception as e:
            issues_found.append(f"Authentication error: {e}")

        # Display results
        console.print()
        console.print("[bold]Repair Results[/bold]")

        if issues_found:
            console.print(f"Issues found: {len(issues_found)}")
            for issue in issues_found:
                console.print(f"  [red]✗ {issue}[/red]")
            console.print()

        if repairs_made:
            console.print(f"Repairs made: {len(repairs_made)}")
            for repair in repairs_made:
                console.print(f"  [green]✓ {repair}[/green]")
        elif not dry_run:
            console.print("[blue]No repairs needed[/blue]")

        if dry_run and (create_dirs or fix_permissions or reset_config):
            console.print("\n[blue]Run without --dry-run to apply fixes[/blue]")

    except Exception as e:
        logger.error(f"Error during repair: {e}")
        console.print(f"[red]Error during repair: {e}[/red]")
        sys.exit(1)


@troubleshoot_group.command()
@click.argument('error_message', required=False)
@click.option('--category', type=click.Choice(['authentication', 'database', 'pipeline', 'configuration']),
              help='Error category')
@click.pass_context
def suggest(ctx, error_message: Optional[str], category: Optional[str]):
    """Get suggestions for fixing common issues."""
    console = ctx.obj['console']

    try:
        if not error_message and not category:
            console.print("[yellow]Please provide an error message or category[/yellow]")
            console.print("Example: finlab-cli troubleshoot suggest 'Connection refused'")
            console.print("Categories: authentication, database, pipeline, configuration")
            return

        console.print("[blue]Analyzing issue and generating suggestions...[/blue]")

        suggestions = []

        # Error message analysis
        if error_message:
            error_lower = error_message.lower()

            # Authentication errors
            if any(term in error_lower for term in ['authentication', 'auth', 'credentials', 'login', 'token']):
                suggestions.extend([
                    {
                        "title": "Authentication Configuration",
                        "description": "Set up FinLab credentials",
                        "commands": [
                            "finlab-cli config init --interactive",
                            "finlab-cli config test"
                        ]
                    },
                    {
                        "title": "Check Environment Variables",
                        "description": "Verify FINLAB_TOKEN or database credentials are set",
                        "commands": ["echo $FINLAB_TOKEN", "finlab-cli config show"]
                    }
                ])

            # Connection errors
            if any(term in error_lower for term in ['connection', 'connect', 'refused', 'timeout', 'network']):
                suggestions.extend([
                    {
                        "title": "Database Connection",
                        "description": "Check database connectivity and credentials",
                        "commands": [
                            "finlab-cli config test",
                            "finlab-cli troubleshoot diagnose"
                        ]
                    },
                    {
                        "title": "Network Connectivity",
                        "description": "Verify network access to FinLab services",
                        "commands": ["ping finlab.io", "curl -I https://finlab.io"]
                    }
                ])

            # Permission errors
            if any(term in error_lower for term in ['permission', 'access', 'denied', 'forbidden']):
                suggestions.extend([
                    {
                        "title": "File Permissions",
                        "description": "Fix file and directory permissions",
                        "commands": [
                            "finlab-cli troubleshoot repair --fix-permissions",
                            "chmod 755 ~/.config/finlab-cli"
                        ]
                    }
                ])

            # Import/module errors
            if any(term in error_lower for term in ['import', 'module', 'package']):
                suggestions.extend([
                    {
                        "title": "Missing Dependencies",
                        "description": "Install required Python packages",
                        "commands": [
                            "pip install -r requirements.txt",
                            "finlab-cli troubleshoot diagnose"
                        ]
                    }
                ])

        # Category-based suggestions
        if category == 'authentication':
            suggestions.extend([
                {
                    "title": "Authentication Setup",
                    "description": "Configure FinLab API credentials",
                    "commands": [
                        "finlab-cli config init --interactive",
                        "finlab-cli config set finlab.token YOUR_TOKEN",
                        "finlab-cli config test"
                    ]
                },
                {
                    "title": "Database Authentication",
                    "description": "Set up database connection credentials",
                    "commands": [
                        "finlab-cli config set finlab.db.host YOUR_HOST",
                        "finlab-cli config set finlab.db.username YOUR_USERNAME",
                        "finlab-cli config set finlab.db.password YOUR_PASSWORD"
                    ]
                }
            ])

        elif category == 'database':
            suggestions.extend([
                {
                    "title": "Database Connection Test",
                    "description": "Verify database connectivity",
                    "commands": [
                        "finlab-cli config test",
                        "finlab-cli pipeline status"
                    ]
                },
                {
                    "title": "Database Configuration",
                    "description": "Check database connection settings",
                    "commands": [
                        "finlab-cli config show",
                        "finlab-cli troubleshoot diagnose --include-sensitive"
                    ]
                }
            ])

        elif category == 'pipeline':
            suggestions.extend([
                {
                    "title": "Pipeline Status Check",
                    "description": "Check pipeline health and status",
                    "commands": [
                        "finlab-cli pipeline status",
                        "finlab-cli monitoring health",
                        "finlab-cli pipeline logs"
                    ]
                },
                {
                    "title": "Pipeline Restart",
                    "description": "Restart the data pipeline",
                    "commands": [
                        "finlab-cli pipeline stop --graceful",
                        "finlab-cli pipeline start"
                    ]
                }
            ])

        elif category == 'configuration':
            suggestions.extend([
                {
                    "title": "Configuration Reset",
                    "description": "Reset configuration to defaults",
                    "commands": [
                        "finlab-cli config template --file config-new.yaml",
                        "finlab-cli troubleshoot repair --reset-config"
                    ]
                },
                {
                    "title": "Configuration Validation",
                    "description": "Validate current configuration",
                    "commands": [
                        "finlab-cli config test",
                        "finlab-cli troubleshoot diagnose"
                    ]
                }
            ])

        # Default suggestions if none found
        if not suggestions:
            suggestions = [
                {
                    "title": "General Diagnostics",
                    "description": "Run comprehensive system diagnostics",
                    "commands": ["finlab-cli troubleshoot diagnose"]
                },
                {
                    "title": "View Logs",
                    "description": "Check recent error logs",
                    "commands": ["finlab-cli troubleshoot logs --level error"]
                },
                {
                    "title": "System Repair",
                    "description": "Attempt to repair common issues",
                    "commands": ["finlab-cli troubleshoot repair --create-dirs --fix-permissions"]
                }
            ]

        # Display suggestions
        console.print()
        console.print("[bold]Suggested Solutions[/bold]")

        for i, suggestion in enumerate(suggestions, 1):
            console.print()
            console.print(f"[bold blue]{i}. {suggestion['title']}[/bold blue]")
            console.print(f"   {suggestion['description']}")

            if suggestion.get('commands'):
                console.print("   Commands:")
                for cmd in suggestion['commands']:
                    console.print(f"     [dim]$[/dim] {cmd}")

    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        console.print(f"[red]Error generating suggestions: {e}[/red]")
        sys.exit(1)