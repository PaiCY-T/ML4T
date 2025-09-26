"""
Configuration manager for FinLab downloader.

Handles loading, validation, and management of configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ..core.exceptions import ConfigurationError, FileOperationError
from .schema import ConfigSchema


class ConfigManager:
    """Configuration manager with YAML support and validation."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default locations.
        """
        self.config_path = None
        self.config = {}
        self._loaded = False

        if config_path:
            self.load_config(config_path)
        else:
            self._find_and_load_config()

    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Raises:
            ConfigurationError: If configuration is invalid
            FileOperationError: If file cannot be read
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileOperationError(
                f"Configuration file not found: {config_path}",
                file_path=str(config_path),
                operation="read"
            )

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)

            if raw_config is None:
                raw_config = {}

            # Validate configuration
            ConfigSchema.validate(raw_config)

            # Normalize configuration
            self.config = ConfigSchema.normalize(raw_config)

            self.config_path = config_path
            self._loaded = True

        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                config_path=str(config_path),
                cause=e
            )
        except IOError as e:
            raise FileOperationError(
                f"Cannot read configuration file: {e}",
                file_path=str(config_path),
                operation="read",
                cause=e
            )

    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.

        Args:
            config_path: Path to save configuration. If None, uses current config_path.

        Raises:
            ConfigurationError: If no configuration is loaded
            FileOperationError: If file cannot be written
        """
        if not self._loaded:
            raise ConfigurationError("No configuration loaded to save")

        save_path = Path(config_path) if config_path else self.config_path

        if save_path is None:
            raise ConfigurationError("No configuration path specified")

        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    indent=2,
                    sort_keys=True
                )

            self.config_path = save_path

        except IOError as e:
            raise FileOperationError(
                f"Cannot write configuration file: {e}",
                file_path=str(save_path),
                operation="write",
                cause=e
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'general.data_directory')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if not self._loaded:
            return default

        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'general.data_directory')
            value: Value to set
        """
        if not self._loaded:
            self.config = ConfigSchema.normalize({})
            self._loaded = True

        keys = key.split('.')
        target = self.config

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # Set the value
        target[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply
        """
        if not self._loaded:
            self.config = ConfigSchema.normalize({})
            self._loaded = True

        self._deep_merge(self.config, updates)

        # Re-validate after updates
        ConfigSchema.validate(self.config)

    def get_data_source_config(self, source_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific data source.

        Args:
            source_name: Name of the data source

        Returns:
            Data source configuration

        Raises:
            ConfigurationError: If data source not configured
        """
        data_sources = self.get("data_sources", {})

        if source_name not in data_sources:
            raise ConfigurationError(
                f"Data source '{source_name}' not configured",
                field=f"data_sources.{source_name}"
            )

        return data_sources[source_name]

    def list_data_sources(self) -> list:
        """
        Get list of configured data sources.

        Returns:
            List of data source names
        """
        return list(self.get("data_sources", {}).keys())

    def is_loaded(self) -> bool:
        """
        Check if configuration is loaded.

        Returns:
            True if configuration is loaded, False otherwise
        """
        return self._loaded

    def reload(self) -> None:
        """
        Reload configuration from file.

        Raises:
            ConfigurationError: If no configuration file path is set
        """
        if self.config_path is None:
            raise ConfigurationError("No configuration file path to reload from")

        self.load_config(self.config_path)

    def create_default_config(self, config_path: Union[str, Path]) -> None:
        """
        Create a default configuration file.

        Args:
            config_path: Path where to create the configuration file

        Raises:
            FileOperationError: If file cannot be created
        """
        config_path = Path(config_path)

        # Use default configuration
        default_config = ConfigSchema.DEFAULT_CONFIG.copy()

        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    default_config,
                    f,
                    default_flow_style=False,
                    indent=2,
                    sort_keys=True
                )

        except IOError as e:
            raise FileOperationError(
                f"Cannot create configuration file: {e}",
                file_path=str(config_path),
                operation="create",
                cause=e
            )

    def _find_and_load_config(self) -> None:
        """Find and load configuration from default locations."""
        default_locations = [
            "config.yaml",
            "finlab_downloader.yaml",
            "~/.finlab_downloader/config.yaml",
            "~/.config/finlab_downloader/config.yaml",
        ]

        for location in default_locations:
            config_path = Path(location).expanduser()
            if config_path.exists():
                self.load_config(config_path)
                return

        # No configuration file found, use defaults
        self.config = ConfigSchema.normalize({})
        self._loaded = True

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def __str__(self) -> str:
        """String representation of configuration."""
        if self._loaded:
            return f"ConfigManager(loaded=True, path={self.config_path})"
        else:
            return "ConfigManager(loaded=False)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()