"""
Base classes and interfaces for FinLab downloader components.

Provides abstract base classes that define the interface for downloaders and validators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
import logging

from .exceptions import FinLabDownloaderError


class BaseDownloader(ABC):
    """Abstract base class for all data downloaders."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the downloader.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration for this downloader."""
        pass

    @abstractmethod
    def download(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Download data for the specified symbols and date range.

        Args:
            symbols: List of symbol identifiers
            start_date: Start date for data download
            end_date: End date for data download
            **kwargs: Additional parameters specific to the downloader

        Returns:
            Dictionary containing downloaded data and metadata

        Raises:
            DataSourceError: When download fails
            ValidationError: When parameters are invalid
        """
        pass

    @abstractmethod
    def list_available_symbols(self) -> List[str]:
        """
        Get a list of all available symbols for this data source.

        Returns:
            List of available symbol identifiers

        Raises:
            DataSourceError: When unable to retrieve symbol list
        """
        pass

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the data source.

        Returns:
            Dictionary with health status information
        """
        try:
            # Basic connectivity test
            symbols = self.list_available_symbols()
            return {
                "status": "healthy",
                "available_symbols": len(symbols),
                "timestamp": datetime.now().isoformat(),
                "source": self.__class__.__name__
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "source": self.__class__.__name__
            }


class BaseValidator(ABC):
    """Abstract base class for data validators."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the validator.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Validate the provided data.

        Args:
            data: Data to validate
            **kwargs: Additional validation parameters

        Returns:
            Dictionary with validation results including:
            - is_valid: boolean indicating if data is valid
            - errors: list of validation errors
            - warnings: list of validation warnings
            - metrics: validation metrics

        Raises:
            ValidationError: When validation cannot be performed
        """
        pass

    def quick_validate(self, data: Any) -> bool:
        """
        Perform a quick validation check.

        Args:
            data: Data to validate

        Returns:
            True if data passes basic validation, False otherwise
        """
        try:
            result = self.validate(data)
            return result.get("is_valid", False)
        except Exception:
            return False


class BaseDataSource(ABC):
    """Abstract base class for data source implementations."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the data source.

        Args:
            name: Name of the data source
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.name = name
        self.config = config
        self.logger = logger or logging.getLogger(f"{self.__class__.__name__}.{name}")

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the data source."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class BaseCache(ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        pass