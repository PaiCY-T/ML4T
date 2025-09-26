"""
Configuration Management Commands.

Commands for managing CLI configuration, authentication settings,
and system configuration.
"""

import sys
import logging
import json
import os
from typing import Optional, Dict, Any
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from ...data.ingestion.finlab_auth import create_finlab_authenticator, AuthConfig
from ..utils.config import CliConfig, load_cli_config, save_cli_config
from ..utils.formatting import format_table
from ..utils.errors import ConfigError, handle_cli_error

logger = logging.getLogger(__name__)


@click.group(name='config')
def config_group():
    """Configuration management commands."""
    pass


@config_group.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'yaml']),
              default='table', help='Output format')
@click.option('--show-sensitive', is_flag=True,
              help='Show sensitive values (passwords, tokens)')
@click.pass_context
def show(ctx, output_format: str, show_sensitive: bool):
    """Show current configuration."""
    console = ctx.obj['console']
    config = ctx.obj['config']

    try:
        # Get configuration data
        config_data = config.to_dict()

        # Mask sensitive data unless explicitly requested
        if not show_sensitive:
            sensitive_keys = ['finlab_token', 'finlab_db_password']
            for key in sensitive_keys:
                if key in config_data and config_data[key]:
                    config_data[key] = '*' * 8

        if output_format == 'json':
            console.print(json.dumps(config_data, indent=2))
        elif output_format == 'yaml':
            import yaml
            console.print(yaml.dump(config_data, default_flow_style=False))
        else:
            # Display as formatted table
            console.print("[bold]Current Configuration[/bold]")

            if config.config_file:
                console.print(f"Configuration file: {config.config_file}")
            else:
                console.print("Configuration source: defaults + environment")
            console.print()

            # Authentication settings
            auth_data = []
            auth_data.append({"Setting": "FinLab Token", "Value": config_data.get('finlab_token', 'Not set')})
            auth_data.append({"Setting": "FinLab DB Host", "Value": config_data.get('finlab_db_host', 'Not set')})
            auth_data.append({"Setting": "FinLab DB Port", "Value": str(config.finlab_db_port)})
            auth_data.append({"Setting": "FinLab DB Name", "Value": config.finlab_db_name})
            auth_data.append({"Setting": "FinLab DB Username", "Value": config_data.get('finlab_db_username', 'Not set')})
            auth_data.append({"Setting": "FinLab DB Password", "Value": config_data.get('finlab_db_password', 'Not set')})

            table = format_table(auth_data, title="Authentication Settings")
            console.print(table)
            console.print()

            # CLI settings
            cli_data = []
            cli_data.append({"Setting": "Default Batch Size", "Value": str(config.default_batch_size)})
            cli_data.append({"Setting": "Default Timeout", "Value": f"{config.default_timeout}s"})
            cli_data.append({"Setting": "Default Output Format", "Value": config.default_output_format})
            cli_data.append({"Setting": "Max Display Rows", "Value": str(config.max_display_rows)})
            cli_data.append({"Setting": "Log Directory", "Value": config.log_dir})
            cli_data.append({"Setting": "Log Retention", "Value": f"{config.log_retention_days} days"})

            table = format_table(cli_data, title="CLI Settings")
            console.print(table)

            # Status
            console.print()
            console.print("[bold]Authentication Status:[/bold]")
            if config.has_finlab_auth:
                console.print("[green]✓ FinLab authentication configured[/green]")
            else:
                console.print("[red]✗ FinLab authentication not configured[/red]")

    except Exception as e:
        logger.error(f"Error showing configuration: {e}")
        console.print(f"[red]Error showing configuration: {e}[/red]")
        sys.exit(1)


@config_group.command()
@click.option('--file', '-f', 'config_file', type=click.Path(),
              help='Configuration file to create/update')
@click.option('--interactive', '-i', is_flag=True,
              help='Interactive configuration setup')
@click.pass_context
def init(ctx, config_file: Optional[str], interactive: bool):
    """Initialize or update configuration."""
    console = ctx.obj['console']

    try:
        if not config_file:
            # Choose default location
            default_locations = [
                Path.cwd() / "finlab-cli.yaml",
                Path.home() / ".config" / "finlab-cli" / "config.yaml",
                Path.home() / ".finlab-cli.yaml"
            ]

            console.print("[blue]Choose configuration file location:[/blue]")
            for i, path in enumerate(default_locations, 1):
                console.print(f"  {i}. {path}")

            choice = Prompt.ask("Select location", choices=[str(i) for i in range(1, len(default_locations) + 1)], default="1")
            config_file = str(default_locations[int(choice) - 1])

        config_path = Path(config_file)

        # Load existing config if it exists
        if config_path.exists():
            console.print(f"[yellow]Configuration file {config_file} already exists[/yellow]")
            if not Confirm.ask("Update existing configuration?"):
                console.print("Configuration initialization cancelled")
                return

            try:
                existing_config = load_cli_config(config_file)
            except Exception as e:
                console.print(f"[red]Error loading existing config: {e}[/red]")
                existing_config = CliConfig()
        else:
            existing_config = CliConfig()

        new_config = existing_config

        if interactive:
            console.print("[blue]Interactive Configuration Setup[/blue]")
            console.print("Press Enter to keep current values or provide new ones.")
            console.print()

            # Authentication settings
            console.print("[bold]FinLab Authentication Settings[/bold]")

            current_token = new_config.finlab_token or ""
            token = Prompt.ask(
                "FinLab Token",
                default=current_token,
                password=True,
                show_default=bool(current_token)
            )
            if token != current_token:
                new_config.finlab_token = token or None

            current_host = new_config.finlab_db_host or ""
            host = Prompt.ask("FinLab DB Host", default=current_host)
            if host != current_host:
                new_config.finlab_db_host = host or None

            if new_config.finlab_db_host:
                port = Prompt.ask("FinLab DB Port", default=str(new_config.finlab_db_port))
                new_config.finlab_db_port = int(port)

                db_name = Prompt.ask("FinLab DB Name", default=new_config.finlab_db_name)
                new_config.finlab_db_name = db_name

                current_username = new_config.finlab_db_username or ""
                username = Prompt.ask("FinLab DB Username", default=current_username)
                if username != current_username:
                    new_config.finlab_db_username = username or None

                current_password = new_config.finlab_db_password or ""
                password = Prompt.ask(
                    "FinLab DB Password",
                    default=current_password,
                    password=True,
                    show_default=bool(current_password)
                )
                if password != current_password:
                    new_config.finlab_db_password = password or None

            # CLI settings
            console.print()
            console.print("[bold]CLI Settings[/bold]")

            batch_size = Prompt.ask("Default Batch Size", default=str(new_config.default_batch_size))
            new_config.default_batch_size = int(batch_size)

            timeout = Prompt.ask("Default Timeout (seconds)", default=str(new_config.default_timeout))
            new_config.default_timeout = int(timeout)

            output_format = Prompt.ask(
                "Default Output Format",
                choices=['table', 'json', 'csv'],
                default=new_config.default_output_format
            )
            new_config.default_output_format = output_format

            max_rows = Prompt.ask("Max Display Rows", default=str(new_config.max_display_rows))
            new_config.max_display_rows = int(max_rows)

            log_dir = Prompt.ask("Log Directory", default=new_config.log_dir)
            new_config.log_dir = log_dir

        else:
            # Use environment variables or defaults
            new_config.finlab_token = os.getenv('FINLAB_TOKEN', new_config.finlab_token)
            new_config.finlab_db_host = os.getenv('FINLAB_DB_HOST', new_config.finlab_db_host)
            new_config.finlab_db_username = os.getenv('FINLAB_DB_USERNAME', new_config.finlab_db_username)
            new_config.finlab_db_password = os.getenv('FINLAB_DB_PASSWORD', new_config.finlab_db_password)

        # Save configuration
        save_cli_config(new_config, config_file)
        console.print(f"[green]✓[/green] Configuration saved to {config_file}")

        # Test configuration
        if new_config.has_finlab_auth:
            console.print()
            if Confirm.ask("Test FinLab connection?"):
                ctx.invoke(test)

    except Exception as e:
        logger.error(f"Error initializing configuration: {e}")
        console.print(f"[red]Error initializing configuration: {e}[/red]")
        sys.exit(1)


@config_group.command()
@click.argument('key')
@click.argument('value', required=False)
@click.option('--file', '-f', 'config_file', type=click.Path(exists=True),
              help='Configuration file to update')
@click.pass_context
def set(ctx, key: str, value: Optional[str], config_file: Optional[str]):
    """Set a configuration value."""
    console = ctx.obj['console']

    try:
        # Load current config
        if config_file:
            config = load_cli_config(config_file)
            config.config_file = config_file
        else:
            config = ctx.obj['config']
            config_file = config.config_file

        if not config_file:
            raise ConfigError("No configuration file specified. Use --file or run 'config init' first.")

        # Map key to configuration attribute
        key_mapping = {
            'finlab.token': 'finlab_token',
            'finlab.db.host': 'finlab_db_host',
            'finlab.db.port': 'finlab_db_port',
            'finlab.db.name': 'finlab_db_name',
            'finlab.db.username': 'finlab_db_username',
            'finlab.db.password': 'finlab_db_password',
            'cli.batch_size': 'default_batch_size',
            'cli.timeout': 'default_timeout',
            'cli.output_format': 'default_output_format',
            'cli.max_rows': 'max_display_rows',
            'cli.log_dir': 'log_dir'
        }

        if key not in key_mapping:
            console.print(f"[red]Unknown configuration key: {key}[/red]")
            console.print("Available keys:")
            for k in key_mapping.keys():
                console.print(f"  {k}")
            sys.exit(1)

        attr_name = key_mapping[key]

        # Get current value if no new value provided
        if value is None:
            current_value = getattr(config, attr_name, None)
            console.print(f"Current value of {key}: {current_value}")
            return

        # Set new value with type conversion
        try:
            if attr_name in ['finlab_db_port', 'default_batch_size', 'default_timeout', 'max_display_rows']:
                value = int(value)
            elif value.lower() in ['none', 'null', '']:
                value = None

            setattr(config, attr_name, value)

            # Save configuration
            save_cli_config(config, config_file)
            console.print(f"[green]✓[/green] Set {key} = {value}")

        except ValueError as e:
            console.print(f"[red]Invalid value for {key}: {e}[/red]")
            sys.exit(1)

    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Error setting configuration: {e}")
        console.print(f"[red]Error setting configuration: {e}[/red]")
        sys.exit(1)


@config_group.command()
@click.argument('key')
@click.option('--file', '-f', 'config_file', type=click.Path(exists=True),
              help='Configuration file to read from')
@click.pass_context
def get(ctx, key: str, config_file: Optional[str]):
    """Get a configuration value."""
    console = ctx.obj['console']

    try:
        # Load config
        if config_file:
            config = load_cli_config(config_file)
        else:
            config = ctx.obj['config']

        # Map key to configuration attribute
        key_mapping = {
            'finlab.token': 'finlab_token',
            'finlab.db.host': 'finlab_db_host',
            'finlab.db.port': 'finlab_db_port',
            'finlab.db.name': 'finlab_db_name',
            'finlab.db.username': 'finlab_db_username',
            'finlab.db.password': 'finlab_db_password',
            'cli.batch_size': 'default_batch_size',
            'cli.timeout': 'default_timeout',
            'cli.output_format': 'default_output_format',
            'cli.max_rows': 'max_display_rows',
            'cli.log_dir': 'log_dir'
        }

        if key not in key_mapping:
            console.print(f"[red]Unknown configuration key: {key}[/red]")
            sys.exit(1)

        attr_name = key_mapping[key]
        value = getattr(config, attr_name, None)

        # Mask sensitive values
        if key in ['finlab.token', 'finlab.db.password'] and value:
            console.print('*' * 8)
        else:
            console.print(str(value) if value is not None else "")

    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        console.print(f"[red]Error getting configuration: {e}[/red]")
        sys.exit(1)


@config_group.command()
@click.pass_context
def test(ctx):
    """Test the current configuration and connections."""
    console = ctx.obj['console']
    config = ctx.obj['config']

    try:
        console.print("[blue]Testing configuration...[/blue]")

        # Test authentication
        console.print()
        console.print("[bold]Authentication Test[/bold]")

        try:
            auth = create_finlab_authenticator()

            if auth.config.has_token_auth:
                console.print("[green]✓[/green] Token authentication configured")
            else:
                console.print("[yellow]○[/yellow] Token authentication not configured")

            if auth.config.has_db_auth:
                console.print("[green]✓[/green] Database authentication configured")
            else:
                console.print("[yellow]○[/yellow] Database authentication not configured")

            if not auth.config.has_token_auth and not auth.config.has_db_auth:
                console.print("[red]✗ No authentication method configured[/red]")
                return

        except Exception as e:
            console.print(f"[red]✗ Authentication configuration error: {e}[/red]")
            return

        # Test database connection
        console.print()
        console.print("[bold]Database Connection Test[/bold]")

        try:
            from ...data.ingestion.finlab_connector import FinLabConnector, FinLabConfig

            connector_config = FinLabConfig(auth_config=auth.config)
            connector = FinLabConnector(connector_config)

            with console.status("[blue]Testing database connection..."):
                if connector.test_connection():
                    console.print("[green]✓[/green] Database connection successful")

                    # Get basic info
                    info = connector.get_connection_info()
                    console.print(f"  Host: {info.get('host', 'N/A')}")
                    console.print(f"  Database: {info.get('database', 'N/A')}")
                    console.print(f"  Version: {info.get('version', 'N/A')}")

                else:
                    console.print("[red]✗[/red] Database connection failed")

        except Exception as e:
            console.print(f"[red]✗ Database connection error: {e}[/red]")

        # Test CLI settings
        console.print()
        console.print("[bold]CLI Settings Test[/bold]")

        # Check log directory
        log_path = Path(config.log_dir)
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]✓[/green] Log directory accessible: {log_path}")
        except Exception as e:
            console.print(f"[red]✗[/red] Log directory error: {e}")

        # Validate settings
        if config.default_batch_size > 0:
            console.print(f"[green]✓[/green] Batch size: {config.default_batch_size}")
        else:
            console.print("[yellow]○[/yellow] Invalid batch size")

        if config.default_timeout > 0:
            console.print(f"[green]✓[/green] Timeout: {config.default_timeout}s")
        else:
            console.print("[yellow]○[/yellow] Invalid timeout")

        console.print()
        console.print("[bold]Configuration test completed[/bold]")

    except Exception as e:
        logger.error(f"Error testing configuration: {e}")
        console.print(f"[red]Error testing configuration: {e}[/red]")
        sys.exit(1)


@config_group.command()
@click.option('--file', '-f', 'config_file', type=click.Path(),
              help='Configuration file to create')
@click.pass_context
def template(ctx, config_file: Optional[str]):
    """Generate a configuration template file."""
    console = ctx.obj['console']

    try:
        if not config_file:
            config_file = "finlab-cli-template.yaml"

        # Create template configuration
        template_config = {
            "# FinLab Authentication": None,
            "finlab_token": "your_finlab_token_here",
            "finlab_db_host": "your_db_host_here",
            "finlab_db_port": 5432,
            "finlab_db_name": "finlab",
            "finlab_db_username": "your_username_here",
            "finlab_db_password": "your_password_here",
            "": None,
            "# CLI Settings": None,
            "default_batch_size": 1000,
            "default_timeout": 300,
            "default_output_format": "table",
            "max_display_rows": 50,
            "log_dir": "logs",
            "log_retention_days": 30
        }

        # Write template
        import yaml

        config_path = Path(config_file)
        with open(config_path, 'w') as f:
            # Write header comment
            f.write("# FinLab CLI Configuration Template\n")
            f.write("# Copy this file and customize the values for your environment\n")
            f.write("# Remove or comment out sections you don't need\n\n")

            # Write configuration with comments
            for key, value in template_config.items():
                if value is None:
                    if key:
                        f.write(f"\n{key}\n")
                    continue
                elif key.startswith("#"):
                    continue
                else:
                    if isinstance(value, str) and "your_" in value:
                        f.write(f"# {key}: \"{value}\"  # TODO: Replace with your actual value\n")
                    else:
                        f.write(f"{key}: {yaml.dump(value).strip()}\n")

        console.print(f"[green]✓[/green] Configuration template created: {config_file}")
        console.print()
        console.print("Next steps:")
        console.print("1. Edit the template file with your actual values")
        console.print("2. Remove or comment out unused sections")
        console.print(f"3. Use it with: finlab-cli --config-file {config_file}")
        console.print("4. Or copy to a standard location (see 'config init' command)")

    except Exception as e:
        logger.error(f"Error creating template: {e}")
        console.print(f"[red]Error creating template: {e}[/red]")
        sys.exit(1)