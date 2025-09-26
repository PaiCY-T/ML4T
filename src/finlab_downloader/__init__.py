"""
FinLab Data Downloader

A comprehensive framework for downloading and managing financial data from FinLab sources.
Provides CLI interface, configuration management, and structured data handling.
"""

__version__ = "0.1.0"
__author__ = "ML4T Team"

from .config.manager import ConfigManager
from .core.exceptions import FinLabDownloaderError
from .utils.logger import get_logger

__all__ = [
    "ConfigManager",
    "FinLabDownloaderError",
    "get_logger",
    "__version__",
]