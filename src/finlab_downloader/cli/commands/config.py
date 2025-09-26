"""Config command implementation."""

import click
import yaml
from pathlib import Path
from typing import Optional, Any

from ...config.manager import ConfigManager
from ...core.exceptions import FinLabDownloaderError
from ...utils.logger import get_logger


def config_command(
    action: str,
    config_manager: ConfigManager,
    key: Optional[str] = None,
    value: Optional[Any] = None,
    path: Optional[Path] = None,
    overwrite: bool = False,
    verbose: bool = False
) -> None:
    """
    Execute config command.

    Args:
        action: Action to perform (show, set, init)
        config_manager: Configuration manager instance
        key: Configuration key
        value: Configuration value
        path: Configuration file path
        overwrite: Whether to overwrite existing files
        verbose: Verbose output flag
    """
    logger = get_logger('config_command')

    try:
        if action == 'show':
            _show_config(config_manager, key, verbose)
        elif action == 'set':
            _set_config(config_manager, key, value, verbose)
        elif action == 'init':
            _init_config(config_manager, path, overwrite, verbose)
        else:
            raise FinLabDownloaderError(f"Unknown config action: {action}")

        logger.info(f"Config {action} command completed successfully")

    except Exception as e:
        logger.error(f"Config {action} command failed: {e}")
        raise


def _show_config(
    config_manager: ConfigManager,
    key: Optional[str],
    verbose: bool
) -> None:
    """Show configuration."""
    if key:
        # Show specific key
        value = config_manager.get(key)
        if value is None:
            click.echo(f"Configuration key '{key}' not found")
        else:
            click.echo(f"{key}: {value}")
    else:
        # Show entire configuration
        click.echo("📝 Current Configuration:")

        if config_manager.config_path:
            click.echo(f"   File: {config_manager.config_path}")
        else:
            click.echo("   File: Using defaults (no config file loaded)")

        if verbose:
            # Show full configuration
            config_yaml = yaml.dump(
                config_manager.config,
                default_flow_style=False,
                indent=2,
                sort_keys=True
            )
            click.echo(f"\n{config_yaml}")
        else:
            # Show summary
            general = config_manager.get('general', {})
            data_sources = config_manager.get('data_sources', {})

            click.echo(f"\n📂 General Settings:")
            click.echo(f"   Data Directory: {general.get('data_directory', 'Not set')}")
            click.echo(f"   Cache Directory: {general.get('cache_directory', 'Not set')}")
            click.echo(f"   Log Level: {general.get('log_level', 'INFO')}")

            click.echo(f"\n🔗 Data Sources: {len(data_sources)} configured")
            for name, config in data_sources.items():
                click.echo(f"   {name}: {config.get('type', 'unknown')}")

            click.echo(f"\n💡 Use --verbose or --key option for more details")


def _set_config(
    config_manager: ConfigManager,
    key: str,
    value: Any,
    verbose: bool
) -> None:
    """Set configuration value."""
    if not key:
        raise FinLabDownloaderError("Configuration key is required")

    if value is None:
        raise FinLabDownloaderError("Configuration value is required")

    # Try to parse value as appropriate type
    parsed_value = _parse_config_value(value)

    # Set the value
    old_value = config_manager.get(key)
    config_manager.set(key, parsed_value)

    click.echo(f"✅ Configuration updated:")
    click.echo(f"   Key: {key}")
    click.echo(f"   Old Value: {old_value}")
    click.echo(f"   New Value: {parsed_value}")

    # Save configuration if file path exists
    if config_manager.config_path:
        config_manager.save_config()
        click.echo(f"   Saved to: {config_manager.config_path}")
    else:
        click.echo("   ⚠️  No config file loaded - changes are temporary")
        click.echo("   Use 'config init' to create a configuration file")


def _init_config(
    config_manager: ConfigManager,
    path: Path,
    overwrite: bool,
    verbose: bool
) -> None:
    """Initialize configuration file."""
    if not path:
        raise FinLabDownloaderError("Configuration file path is required")

    # Check if file exists
    if path.exists() and not overwrite:
        raise FinLabDownloaderError(
            f"Configuration file already exists: {path}\n"
            "Use --overwrite flag to replace it"
        )

    # Create default configuration
    config_manager.create_default_config(path)

    click.echo(f"✅ Configuration file created: {path}")

    if verbose:
        click.echo("\n📝 Default configuration created with:")
        click.echo("   - Basic logging settings")
        click.echo("   - Default directories")
        click.echo("   - Empty data sources (add your own)")
        click.echo("   - Validation settings")

    click.echo(f"\n💡 Edit the file to add your data sources and customize settings")


def _parse_config_value(value: str) -> Any:
    """
    Parse configuration value string to appropriate type.

    Args:
        value: String value to parse

    Returns:
        Parsed value with appropriate type
    """
    # Try boolean
    if value.lower() in ('true', 'yes', '1', 'on'):
        return True
    elif value.lower() in ('false', 'no', '0', 'off'):
        return False

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try JSON (for complex types)
    try:
        import json
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass

    # Return as string
    return value