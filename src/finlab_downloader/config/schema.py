"""
Configuration schema validation for FinLab downloader.

Defines the expected structure and validation rules for configuration files.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import os

from ..core.exceptions import ConfigurationError


class ConfigSchema:
    """Configuration schema validator."""

    # Default configuration structure
    DEFAULT_CONFIG = {
        "version": "1.0",
        "general": {
            "data_directory": "./data",
            "cache_directory": "./cache",
            "log_level": "INFO",
            "log_file": "finlab_downloader.log",
            "max_workers": 4,
            "request_timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 1.0
        },
        "data_sources": {},
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_rotation": {
                "max_bytes": 10485760,  # 10MB
                "backup_count": 5
            },
            "console_output": True
        },
        "cache": {
            "enabled": True,
            "type": "memory",
            "ttl": 3600,
            "max_size": 1000
        },
        "validation": {
            "strict_mode": False,
            "check_duplicates": True,
            "validate_dates": True,
            "max_date_gap": 365
        }
    }

    # Required fields that must be present
    REQUIRED_FIELDS = [
        "version",
        "general",
        "data_sources",
        "logging"
    ]

    # Valid log levels
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # Valid cache types
    VALID_CACHE_TYPES = ["memory", "file", "redis", "disabled"]

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Validated and normalized configuration

        Raises:
            ConfigurationError: If validation fails
        """
        # Check required fields
        cls._validate_required_fields(config)

        # Validate version
        cls._validate_version(config.get("version"))

        # Validate general settings
        cls._validate_general_settings(config.get("general", {}))

        # Validate logging settings
        cls._validate_logging_settings(config.get("logging", {}))

        # Validate cache settings
        cls._validate_cache_settings(config.get("cache", {}))

        # Validate data sources
        cls._validate_data_sources(config.get("data_sources", {}))

        # Validate validation settings
        cls._validate_validation_settings(config.get("validation", {}))

        return config

    @classmethod
    def normalize(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize configuration by filling in defaults.

        Args:
            config: Configuration dictionary

        Returns:
            Normalized configuration with defaults applied
        """
        normalized = cls.DEFAULT_CONFIG.copy()

        # Deep merge the provided config
        cls._deep_merge(normalized, config)

        # Expand paths
        cls._expand_paths(normalized)

        return normalized

    @classmethod
    def _validate_required_fields(cls, config: Dict[str, Any]) -> None:
        """Validate that all required fields are present."""
        missing_fields = []
        for field in cls.REQUIRED_FIELDS:
            if field not in config:
                missing_fields.append(field)

        if missing_fields:
            raise ConfigurationError(
                f"Missing required configuration fields: {missing_fields}",
                field="required_fields"
            )

    @classmethod
    def _validate_version(cls, version: Any) -> None:
        """Validate configuration version."""
        if not isinstance(version, str):
            raise ConfigurationError(
                "Configuration version must be a string",
                field="version"
            )

        # For now, just check it's not empty
        if not version.strip():
            raise ConfigurationError(
                "Configuration version cannot be empty",
                field="version"
            )

    @classmethod
    def _validate_general_settings(cls, general: Dict[str, Any]) -> None:
        """Validate general settings."""
        if not isinstance(general, dict):
            raise ConfigurationError(
                "General settings must be a dictionary",
                field="general"
            )

        # Validate numeric fields
        numeric_fields = ["max_workers", "request_timeout", "retry_attempts"]
        for field in numeric_fields:
            if field in general:
                value = general[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ConfigurationError(
                        f"Field '{field}' must be a positive number",
                        field=f"general.{field}",
                        value=value
                    )

        # Validate retry_delay
        if "retry_delay" in general:
            value = general["retry_delay"]
            if not isinstance(value, (int, float)) or value < 0:
                raise ConfigurationError(
                    "retry_delay must be a non-negative number",
                    field="general.retry_delay",
                    value=value
                )

    @classmethod
    def _validate_logging_settings(cls, logging_config: Dict[str, Any]) -> None:
        """Validate logging settings."""
        if not isinstance(logging_config, dict):
            raise ConfigurationError(
                "Logging settings must be a dictionary",
                field="logging"
            )

        # Validate log level
        if "level" in logging_config:
            level = logging_config["level"]
            if level not in cls.VALID_LOG_LEVELS:
                raise ConfigurationError(
                    f"Invalid log level '{level}'. Must be one of: {cls.VALID_LOG_LEVELS}",
                    field="logging.level",
                    value=level
                )

        # Validate file rotation settings
        if "file_rotation" in logging_config:
            rotation = logging_config["file_rotation"]
            if not isinstance(rotation, dict):
                raise ConfigurationError(
                    "File rotation settings must be a dictionary",
                    field="logging.file_rotation"
                )

            for field in ["max_bytes", "backup_count"]:
                if field in rotation:
                    value = rotation[field]
                    if not isinstance(value, int) or value <= 0:
                        raise ConfigurationError(
                            f"Field '{field}' must be a positive integer",
                            field=f"logging.file_rotation.{field}",
                            value=value
                        )

    @classmethod
    def _validate_cache_settings(cls, cache_config: Dict[str, Any]) -> None:
        """Validate cache settings."""
        if not isinstance(cache_config, dict):
            return  # Cache config is optional

        # Validate cache type
        if "type" in cache_config:
            cache_type = cache_config["type"]
            if cache_type not in cls.VALID_CACHE_TYPES:
                raise ConfigurationError(
                    f"Invalid cache type '{cache_type}'. Must be one of: {cls.VALID_CACHE_TYPES}",
                    field="cache.type",
                    value=cache_type
                )

        # Validate numeric fields
        numeric_fields = ["ttl", "max_size"]
        for field in numeric_fields:
            if field in cache_config:
                value = cache_config[field]
                if not isinstance(value, int) or value <= 0:
                    raise ConfigurationError(
                        f"Field '{field}' must be a positive integer",
                        field=f"cache.{field}",
                        value=value
                    )

    @classmethod
    def _validate_data_sources(cls, data_sources: Dict[str, Any]) -> None:
        """Validate data sources configuration."""
        if not isinstance(data_sources, dict):
            raise ConfigurationError(
                "Data sources must be a dictionary",
                field="data_sources"
            )

        for name, config in data_sources.items():
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Data source '{name}' configuration must be a dictionary",
                    field=f"data_sources.{name}"
                )

            # Each data source should have a type
            if "type" not in config:
                raise ConfigurationError(
                    f"Data source '{name}' must specify a type",
                    field=f"data_sources.{name}.type"
                )

    @classmethod
    def _validate_validation_settings(cls, validation_config: Dict[str, Any]) -> None:
        """Validate validation settings."""
        if not isinstance(validation_config, dict):
            return  # Validation config is optional

        # Validate boolean fields
        bool_fields = ["strict_mode", "check_duplicates", "validate_dates"]
        for field in bool_fields:
            if field in validation_config:
                value = validation_config[field]
                if not isinstance(value, bool):
                    raise ConfigurationError(
                        f"Field '{field}' must be a boolean",
                        field=f"validation.{field}",
                        value=value
                    )

        # Validate max_date_gap
        if "max_date_gap" in validation_config:
            value = validation_config["max_date_gap"]
            if not isinstance(value, int) or value <= 0:
                raise ConfigurationError(
                    "max_date_gap must be a positive integer",
                    field="validation.max_date_gap",
                    value=value
                )

    @classmethod
    def _deep_merge(cls, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                cls._deep_merge(target[key], value)
            else:
                target[key] = value

    @classmethod
    def _expand_paths(cls, config: Dict[str, Any]) -> None:
        """Expand relative paths to absolute paths."""
        # Expand directory paths
        general = config.get("general", {})
        path_fields = ["data_directory", "cache_directory"]

        for field in path_fields:
            if field in general:
                path = general[field]
                if not os.path.isabs(path):
                    general[field] = os.path.abspath(path)

        # Expand log file path
        logging_config = config.get("logging", {})
        if "file" in logging_config:
            log_file = logging_config["file"]
            if not os.path.isabs(log_file):
                logging_config["file"] = os.path.abspath(log_file)