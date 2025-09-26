"""Core module containing base classes and interfaces."""

from .exceptions import (
    FinLabDownloaderError,
    ConfigurationError,
    DataSourceError,
    ValidationError,
    AuthenticationError,
    RateLimitError,
)
from .base import BaseDownloader, BaseValidator
from .client import FinLabClient, RateLimitConfig, ProgressConfig
from .factory import FinLabClientFactory, DatasetCatalogFactory, IntegratedDownloaderFactory
from .dataset import DatasetCatalog, DatasetSpecification, DownloadMethod, DataType

__all__ = [
    "FinLabDownloaderError",
    "ConfigurationError",
    "DataSourceError",
    "ValidationError",
    "AuthenticationError",
    "RateLimitError",
    "BaseDownloader",
    "BaseValidator",
    "FinLabClient",
    "RateLimitConfig",
    "ProgressConfig",
    "FinLabClientFactory",
    "DatasetCatalogFactory",
    "IntegratedDownloaderFactory",
    "DatasetCatalog",
    "DatasetSpecification",
    "DownloadMethod",
    "DataType",
]