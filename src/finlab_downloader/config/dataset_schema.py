"""
Dataset-specific configuration schemas for FinLab downloader.

Extends the base configuration schema with dataset catalog and download specifications.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from ..core.exceptions import ConfigurationError
from ..core.dataset import DatasetCatalog, DownloadMethod, DataType
from .schema import ConfigSchema


class DatasetConfigSchema(ConfigSchema):
    """Extended configuration schema with dataset support."""

    # Default dataset configuration
    DEFAULT_DATASET_CONFIG = {
        "datasets": {
            "catalog_file": "finlab_database_cleaned.csv",
            "auto_discover": True,
            "validation": {
                "enabled": True,
                "strict_mode": False,
                "business_logic": True,
                "taiwan_market_rules": True
            },
            "download": {
                "batch_size": 100,
                "parallel_downloads": 4,
                "retry_failed": True,
                "cache_responses": True
            },
            "categories": {
                "market_data": {
                    "enabled": True,
                    "priority": 1,
                    "cache_ttl": 3600
                },
                "financial_statements": {
                    "enabled": True,
                    "priority": 2,
                    "cache_ttl": 86400
                },
                "fundamental_analysis": {
                    "enabled": True,
                    "priority": 3,
                    "cache_ttl": 86400
                }
            }
        }
    }

    # Dataset-specific validation rules
    DATASET_VALIDATION_RULES = {
        "catalog_file": {
            "type": "string",
            "required": True,
            "description": "Path to dataset catalog CSV file"
        },
        "auto_discover": {
            "type": "boolean",
            "default": True,
            "description": "Automatically discover datasets from catalog"
        },
        "validation.enabled": {
            "type": "boolean",
            "default": True,
            "description": "Enable dataset validation"
        },
        "validation.strict_mode": {
            "type": "boolean",
            "default": False,
            "description": "Use strict validation rules"
        },
        "validation.business_logic": {
            "type": "boolean",
            "default": True,
            "description": "Apply business logic validation"
        },
        "validation.taiwan_market_rules": {
            "type": "boolean",
            "default": True,
            "description": "Apply Taiwan market specific rules"
        },
        "download.batch_size": {
            "type": "integer",
            "min": 1,
            "max": 1000,
            "default": 100,
            "description": "Number of datasets to download in batch"
        },
        "download.parallel_downloads": {
            "type": "integer",
            "min": 1,
            "max": 16,
            "default": 4,
            "description": "Number of parallel download threads"
        },
        "download.retry_failed": {
            "type": "boolean",
            "default": True,
            "description": "Retry failed downloads"
        },
        "download.cache_responses": {
            "type": "boolean",
            "default": True,
            "description": "Cache download responses"
        }
    }

    @classmethod
    def validate_with_datasets(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration with dataset-specific rules.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Validated and normalized configuration

        Raises:
            ConfigurationError: If validation fails
        """
        # First validate base configuration
        config = cls.validate(config)

        # Validate dataset-specific configuration
        cls._validate_dataset_config(config.get("datasets", {}))

        return config

    @classmethod
    def normalize_with_datasets(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize configuration with dataset defaults.

        Args:
            config: Configuration dictionary

        Returns:
            Normalized configuration with dataset defaults applied
        """
        # Start with base normalization
        normalized = cls.normalize(config)

        # Deep merge dataset defaults
        if "datasets" not in normalized:
            normalized["datasets"] = {}

        cls._deep_merge(normalized["datasets"], cls.DEFAULT_DATASET_CONFIG["datasets"])

        return normalized

    @classmethod
    def _validate_dataset_config(cls, dataset_config: Dict[str, Any]) -> None:
        """Validate dataset-specific configuration."""
        if not isinstance(dataset_config, dict):
            return  # Dataset config is optional

        # Validate catalog file
        cls._validate_catalog_file(dataset_config.get("catalog_file"))

        # Validate validation settings
        cls._validate_dataset_validation_settings(dataset_config.get("validation", {}))

        # Validate download settings
        cls._validate_download_settings(dataset_config.get("download", {}))

        # Validate category settings
        cls._validate_category_settings(dataset_config.get("categories", {}))

    @classmethod
    def _validate_catalog_file(cls, catalog_file: Any) -> None:
        """Validate catalog file configuration."""
        if catalog_file is None:
            return

        if not isinstance(catalog_file, str):
            raise ConfigurationError(
                "Catalog file must be a string path",
                field="datasets.catalog_file",
                value=catalog_file
            )

        if not catalog_file.strip():
            raise ConfigurationError(
                "Catalog file path cannot be empty",
                field="datasets.catalog_file",
                value=catalog_file
            )

        # Check if file exists (if it's not a relative path)
        catalog_path = Path(catalog_file)
        if catalog_path.is_absolute() and not catalog_path.exists():
            raise ConfigurationError(
                f"Catalog file not found: {catalog_file}",
                field="datasets.catalog_file",
                value=catalog_file,
                file_path=catalog_file
            )

    @classmethod
    def _validate_dataset_validation_settings(cls, validation_config: Dict[str, Any]) -> None:
        """Validate dataset validation settings."""
        if not isinstance(validation_config, dict):
            return

        # Validate boolean fields
        bool_fields = ["enabled", "strict_mode", "business_logic", "taiwan_market_rules"]
        for field in bool_fields:
            if field in validation_config:
                value = validation_config[field]
                if not isinstance(value, bool):
                    raise ConfigurationError(
                        f"Field '{field}' must be a boolean",
                        field=f"datasets.validation.{field}",
                        value=value
                    )

    @classmethod
    def _validate_download_settings(cls, download_config: Dict[str, Any]) -> None:
        """Validate download settings."""
        if not isinstance(download_config, dict):
            return

        # Validate numeric fields
        numeric_fields = {
            "batch_size": (1, 1000),
            "parallel_downloads": (1, 16)
        }

        for field, (min_val, max_val) in numeric_fields.items():
            if field in download_config:
                value = download_config[field]
                if not isinstance(value, int) or not (min_val <= value <= max_val):
                    raise ConfigurationError(
                        f"Field '{field}' must be an integer between {min_val} and {max_val}",
                        field=f"datasets.download.{field}",
                        value=value,
                        min_value=min_val,
                        max_value=max_val
                    )

        # Validate boolean fields
        bool_fields = ["retry_failed", "cache_responses"]
        for field in bool_fields:
            if field in download_config:
                value = download_config[field]
                if not isinstance(value, bool):
                    raise ConfigurationError(
                        f"Field '{field}' must be a boolean",
                        field=f"datasets.download.{field}",
                        value=value
                    )

    @classmethod
    def _validate_category_settings(cls, categories_config: Dict[str, Any]) -> None:
        """Validate category settings."""
        if not isinstance(categories_config, dict):
            return

        for category_name, category_config in categories_config.items():
            if not isinstance(category_config, dict):
                raise ConfigurationError(
                    f"Category '{category_name}' configuration must be a dictionary",
                    field=f"datasets.categories.{category_name}"
                )

            # Validate enabled field
            if "enabled" in category_config:
                enabled = category_config["enabled"]
                if not isinstance(enabled, bool):
                    raise ConfigurationError(
                        f"Category '{category_name}' enabled field must be a boolean",
                        field=f"datasets.categories.{category_name}.enabled",
                        value=enabled
                    )

            # Validate priority field
            if "priority" in category_config:
                priority = category_config["priority"]
                if not isinstance(priority, int) or priority < 1:
                    raise ConfigurationError(
                        f"Category '{category_name}' priority must be a positive integer",
                        field=f"datasets.categories.{category_name}.priority",
                        value=priority
                    )

            # Validate cache_ttl field
            if "cache_ttl" in category_config:
                cache_ttl = category_config["cache_ttl"]
                if not isinstance(cache_ttl, int) or cache_ttl < 0:
                    raise ConfigurationError(
                        f"Category '{category_name}' cache_ttl must be a non-negative integer",
                        field=f"datasets.categories.{category_name}.cache_ttl",
                        value=cache_ttl
                    )


class DatasetDownloadConfig:
    """Configuration for dataset download operations."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize download configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_config = config.get("datasets", {})
        self.download_config = self.dataset_config.get("download", {})

    @property
    def batch_size(self) -> int:
        """Get batch size for downloads."""
        return self.download_config.get("batch_size", 100)

    @property
    def parallel_downloads(self) -> int:
        """Get number of parallel downloads."""
        return self.download_config.get("parallel_downloads", 4)

    @property
    def retry_failed(self) -> bool:
        """Get whether to retry failed downloads."""
        return self.download_config.get("retry_failed", True)

    @property
    def cache_responses(self) -> bool:
        """Get whether to cache responses."""
        return self.download_config.get("cache_responses", True)

    @property
    def catalog_file(self) -> str:
        """Get catalog file path."""
        return self.dataset_config.get("catalog_file", "finlab_database_cleaned.csv")

    @property
    def auto_discover(self) -> bool:
        """Get whether to auto-discover datasets."""
        return self.dataset_config.get("auto_discover", True)

    def get_category_config(self, category: str) -> Dict[str, Any]:
        """
        Get configuration for a specific category.

        Args:
            category: Category name

        Returns:
            Category configuration
        """
        categories = self.dataset_config.get("categories", {})
        return categories.get(category, {
            "enabled": True,
            "priority": 5,
            "cache_ttl": 3600
        })

    def is_category_enabled(self, category: str) -> bool:
        """
        Check if a category is enabled.

        Args:
            category: Category name

        Returns:
            True if category is enabled
        """
        category_config = self.get_category_config(category)
        return category_config.get("enabled", True)

    def get_category_priority(self, category: str) -> int:
        """
        Get priority for a category.

        Args:
            category: Category name

        Returns:
            Category priority (lower number = higher priority)
        """
        category_config = self.get_category_config(category)
        return category_config.get("priority", 5)

    def get_category_cache_ttl(self, category: str) -> int:
        """
        Get cache TTL for a category.

        Args:
            category: Category name

        Returns:
            Cache TTL in seconds
        """
        category_config = self.get_category_config(category)
        return category_config.get("cache_ttl", 3600)


class DatasetValidationConfig:
    """Configuration for dataset validation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validation configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_config = config.get("datasets", {})
        self.validation_config = self.dataset_config.get("validation", {})

    @property
    def enabled(self) -> bool:
        """Get whether validation is enabled."""
        return self.validation_config.get("enabled", True)

    @property
    def strict_mode(self) -> bool:
        """Get whether strict mode is enabled."""
        return self.validation_config.get("strict_mode", False)

    @property
    def business_logic(self) -> bool:
        """Get whether business logic validation is enabled."""
        return self.validation_config.get("business_logic", True)

    @property
    def taiwan_market_rules(self) -> bool:
        """Get whether Taiwan market rules are enabled."""
        return self.validation_config.get("taiwan_market_rules", True)

    def should_validate(self, dataset_name: str) -> bool:
        """
        Check if a dataset should be validated.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if dataset should be validated
        """
        return self.enabled

    def get_validation_level(self) -> str:
        """
        Get validation level based on configuration.

        Returns:
            Validation level: 'strict', 'standard', or 'minimal'
        """
        if not self.enabled:
            return "minimal"
        elif self.strict_mode:
            return "strict"
        else:
            return "standard"


def create_sample_dataset_config() -> Dict[str, Any]:
    """
    Create a sample dataset configuration.

    Returns:
        Sample configuration dictionary
    """
    return {
        "version": "1.0",
        "general": {
            "data_directory": "./data",
            "cache_directory": "./cache",
            "log_level": "INFO"
        },
        "datasets": {
            "catalog_file": "example/finlab_database_cleaned.csv",
            "auto_discover": True,
            "validation": {
                "enabled": True,
                "strict_mode": False,
                "business_logic": True,
                "taiwan_market_rules": True
            },
            "download": {
                "batch_size": 50,
                "parallel_downloads": 2,
                "retry_failed": True,
                "cache_responses": True
            },
            "categories": {
                "market_data": {
                    "enabled": True,
                    "priority": 1,
                    "cache_ttl": 1800
                },
                "financial_statements": {
                    "enabled": True,
                    "priority": 2,
                    "cache_ttl": 86400
                },
                "fundamental_analysis": {
                    "enabled": True,
                    "priority": 3,
                    "cache_ttl": 86400
                }
            }
        },
        "data_sources": {
            "finlab": {
                "type": "finlab_api",
                "base_url": "https://api.finlab.tw",
                "api_version": "v1"
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console_output": True
        }
    }