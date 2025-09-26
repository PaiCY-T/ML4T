"""
Factory for creating FinLab clients with configuration integration.

Provides convenient factory methods for creating configured FinLab API clients.
"""

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

from ..utils.logger import get_logger
from .client import FinLabClient, RateLimitConfig, ProgressConfig
from .dataset import DatasetCatalog
from .exceptions import ConfigurationError

if TYPE_CHECKING:
    from ..config.manager import ConfigManager


class FinLabClientFactory:
    """Factory for creating configured FinLab API clients."""

    @staticmethod
    def create_client(
        config_manager: Optional["ConfigManager"] = None,
        config_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ) -> FinLabClient:
        """
        Create a FinLab client from configuration.

        Args:
            config_manager: Existing configuration manager
            config_path: Path to configuration file (if config_manager not provided)
            logger: Optional logger instance

        Returns:
            Configured FinLabClient instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Initialize configuration manager if not provided
        if config_manager is None:
            from ..config.manager import ConfigManager
            config_manager = ConfigManager(config_path)

        # Get logger
        if logger is None:
            logger = get_logger("FinLabClientFactory")

        # Extract configuration sections
        try:
            config = config_manager.get_config()

            # Extract FinLab-specific configuration
            finlab_config = config.get('finlab', {})
            if not finlab_config:
                raise ConfigurationError("Missing 'finlab' configuration section")

            # Extract API configuration
            api_config = finlab_config.get('api', {})
            if not api_config:
                raise ConfigurationError("Missing 'finlab.api' configuration section")

            # Create rate limiting configuration
            rate_limit_config = FinLabClientFactory._create_rate_limit_config(
                finlab_config.get('rate_limiting', {})
            )

            # Create progress configuration
            progress_config = FinLabClientFactory._create_progress_config(
                finlab_config.get('progress', {})
            )

            # Create and return client
            client = FinLabClient(
                config=api_config,
                rate_limit_config=rate_limit_config,
                progress_config=progress_config,
                logger=logger
            )

            logger.info("Successfully created FinLab client from configuration")
            return client

        except Exception as e:
            logger.error(f"Failed to create FinLab client: {e}")
            raise ConfigurationError(f"Client creation failed: {e}")

    @staticmethod
    def create_client_with_defaults(
        api_token: str,
        logger: Optional[logging.Logger] = None
    ) -> FinLabClient:
        """
        Create a FinLab client with default configuration and provided API token.

        Args:
            api_token: FinLab API token
            logger: Optional logger instance

        Returns:
            FinLabClient with default configuration
        """
        if logger is None:
            logger = get_logger("FinLabClientFactory")

        config = {
            'api_token': api_token
        }

        rate_limit_config = RateLimitConfig()
        progress_config = ProgressConfig()

        return FinLabClient(
            config=config,
            rate_limit_config=rate_limit_config,
            progress_config=progress_config,
            logger=logger
        )

    @staticmethod
    def _create_rate_limit_config(rate_limit_settings: Dict[str, Any]) -> RateLimitConfig:
        """Create rate limiting configuration from settings."""
        return RateLimitConfig(
            max_requests_per_minute=rate_limit_settings.get('max_requests_per_minute', 60),
            max_requests_per_hour=rate_limit_settings.get('max_requests_per_hour', 1000),
            max_requests_per_day=rate_limit_settings.get('max_requests_per_day', 10000),
            backoff_factor=rate_limit_settings.get('backoff_factor', 1.5),
            max_retries=rate_limit_settings.get('max_retries', 3)
        )

    @staticmethod
    def _create_progress_config(progress_settings: Dict[str, Any]) -> ProgressConfig:
        """Create progress configuration from settings."""
        return ProgressConfig(
            show_progress=progress_settings.get('show_progress', True),
            update_interval=progress_settings.get('update_interval', 0.1),
            chunk_size=progress_settings.get('chunk_size', 1000),
            verbose=progress_settings.get('verbose', False)
        )


class DatasetCatalogFactory:
    """Factory for creating dataset catalogs with configuration integration."""

    @staticmethod
    def create_catalog(
        config_manager: Optional["ConfigManager"] = None,
        config_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ) -> DatasetCatalog:
        """
        Create a dataset catalog from configuration.

        Args:
            config_manager: Existing configuration manager
            config_path: Path to configuration file (if config_manager not provided)
            csv_path: Override path to dataset CSV file
            logger: Optional logger instance

        Returns:
            Configured DatasetCatalog instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Initialize configuration manager if not provided
        if config_manager is None:
            from ..config.manager import ConfigManager
            config_manager = ConfigManager(config_path)

        # Get logger
        if logger is None:
            logger = get_logger("DatasetCatalogFactory")

        try:
            config = config_manager.get_config()

            # Extract dataset configuration
            dataset_config = config.get('datasets', {})

            # Determine CSV file path
            if csv_path is None:
                csv_path = dataset_config.get('catalog_file', 'finlab_database_cleaned.csv')

            # Create catalog
            if csv_path:
                catalog = DatasetCatalog.from_csv(csv_path)
                logger.info(f"Loaded dataset catalog from {csv_path}")
            else:
                catalog = DatasetCatalog()
                logger.info("Created empty dataset catalog")

            return catalog

        except Exception as e:
            logger.error(f"Failed to create dataset catalog: {e}")
            raise ConfigurationError(f"Dataset catalog creation failed: {e}")


class IntegratedDownloaderFactory:
    """Factory for creating fully integrated downloaders with client and catalog."""

    @staticmethod
    def create_integrated_downloader(
        config_manager: Optional["ConfigManager"] = None,
        config_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        """
        Create a complete downloader setup with client and catalog.

        Args:
            config_manager: Existing configuration manager
            config_path: Path to configuration file
            logger: Optional logger instance

        Returns:
            Dictionary containing 'client' and 'catalog' instances

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if logger is None:
            logger = get_logger("IntegratedDownloaderFactory")

        try:
            # Create client
            client = FinLabClientFactory.create_client(
                config_manager=config_manager,
                config_path=config_path,
                logger=logger
            )

            # Create catalog
            catalog = DatasetCatalogFactory.create_catalog(
                config_manager=config_manager,
                config_path=config_path,
                logger=logger
            )

            logger.info("Successfully created integrated downloader")

            return {
                'client': client,
                'catalog': catalog,
                'config_manager': config_manager
            }

        except Exception as e:
            logger.error(f"Failed to create integrated downloader: {e}")
            raise ConfigurationError(f"Integrated downloader creation failed: {e}")