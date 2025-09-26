"""
Configuration management for CLI interface.

Handles loading and managing CLI configuration from files and environment
variables with proper validation and default values.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import json
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CliConfig:
    """CLI configuration with defaults and validation."""

    # Configuration file path
    config_file: Optional[str] = None

    # FinLab connection settings
    finlab_token: Optional[str] = None
    finlab_db_host: Optional[str] = None
    finlab_db_port: int = 5432
    finlab_db_name: str = "finlab"
    finlab_db_username: Optional[str] = None
    finlab_db_password: Optional[str] = None

    # Default settings
    default_batch_size: int = 1000
    default_timeout: int = 300
    default_retry_attempts: int = 3

    # Output settings
    default_output_format: str = "table"  # table, json, csv
    max_display_rows: int = 50

    # Logging settings
    log_dir: str = "logs"
    log_retention_days: int = 30

    # Monitoring settings
    health_check_timeout: int = 30
    status_refresh_interval: int = 5

    # Data pipeline settings
    pipeline_status_file: str = ".pipeline_status"
    pipeline_lock_file: str = ".pipeline_lock"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CliConfig':
        """Create config from dictionary."""
        # Filter only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @property
    def has_finlab_auth(self) -> bool:
        """Check if FinLab authentication is configured."""
        return (self.finlab_token is not None or
                (self.finlab_db_host and self.finlab_db_username))


def load_cli_config(config_file: Optional[str] = None) -> CliConfig:
    """
    Load CLI configuration from file and environment.

    Args:
        config_file: Optional path to config file

    Returns:
        CliConfig instance
    """
    config_data = {}

    # Try to load from config file
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.suffix.lower() in ['.yaml', '.yml']:
                        config_data = yaml.safe_load(f) or {}
                    else:
                        config_data = json.load(f)
                logger.debug(f"Loaded config from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
    else:
        # Try default config locations
        default_locations = [
            Path.cwd() / "finlab-cli.yaml",
            Path.cwd() / "finlab-cli.json",
            Path.home() / ".config" / "finlab-cli" / "config.yaml",
            Path.home() / ".finlab-cli.yaml"
        ]

        for config_path in default_locations:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        if config_path.suffix.lower() in ['.yaml', '.yml']:
                            config_data = yaml.safe_load(f) or {}
                        else:
                            config_data = json.load(f)
                    config_file = str(config_path)
                    logger.debug(f"Loaded config from {config_path}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load config file {config_path}: {e}")

    # Override with environment variables
    env_mappings = {
        'FINLAB_TOKEN': 'finlab_token',
        'FINLAB_DB_HOST': 'finlab_db_host',
        'FINLAB_DB_PORT': 'finlab_db_port',
        'FINLAB_DB_NAME': 'finlab_db_name',
        'FINLAB_DB_USERNAME': 'finlab_db_username',
        'FINLAB_DB_PASSWORD': 'finlab_db_password',
        'CLI_DEFAULT_BATCH_SIZE': 'default_batch_size',
        'CLI_DEFAULT_TIMEOUT': 'default_timeout',
        'CLI_LOG_DIR': 'log_dir',
        'CLI_OUTPUT_FORMAT': 'default_output_format'
    }

    for env_var, config_key in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            # Convert numeric values
            if config_key in ['finlab_db_port', 'default_batch_size', 'default_timeout']:
                try:
                    env_value = int(env_value)
                except ValueError:
                    logger.warning(f"Invalid numeric value for {env_var}: {env_value}")
                    continue
            config_data[config_key] = env_value

    # Create config object
    config = CliConfig.from_dict(config_data)
    config.config_file = config_file

    return config


def save_cli_config(config: CliConfig, config_file: str) -> None:
    """
    Save CLI configuration to file.

    Args:
        config: CliConfig to save
        config_file: Path to save config
    """
    config_path = Path(config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = config.to_dict()
    # Remove the config_file field from saved data
    config_data.pop('config_file', None)

    try:
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_data, f, default_flow_style=False)
            else:
                json.dump(config_data, f, indent=2)
        logger.info(f"Configuration saved to {config_file}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_file}: {e}")
        raise