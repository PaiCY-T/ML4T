"""
Logging utilities for FinLab downloader.

Provides structured logging with file rotation, console output, and configurable levels.
"""

import logging
import logging.handlers
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import os

from ..core.exceptions import ConfigurationError


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            # Add color to level name
            colored_levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            )
            record.levelname = colored_levelname

        return super().format(record)


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup logging configuration.

    Args:
        config: Logging configuration dictionary
    """
    if config is None:
        config = {}

    # Default configuration
    log_config = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "finlab_downloader.log",
        "console_output": True,
        "file_rotation": {
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5
        }
    }

    # Update with provided config
    log_config.update(config)

    # Get log level
    level = getattr(logging, log_config["level"].upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Setup console handler
    if log_config.get("console_output", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Use colored formatter for console
        console_formatter = ColoredFormatter(log_config["format"])
        console_handler.setFormatter(console_formatter)

        root_logger.addHandler(console_handler)

    # Setup file handler
    if "file" in log_config:
        log_file = Path(log_config["file"])

        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup rotating file handler
        file_rotation = log_config.get("file_rotation", {})
        max_bytes = file_rotation.get("max_bytes", 10485760)
        backup_count = file_rotation.get("backup_count", 5)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)

        # Use plain formatter for file
        file_formatter = logging.Formatter(log_config["format"])
        file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)

    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""

    def __init__(self, context: Dict[str, Any]):
        """
        Initialize context filter.

        Args:
            context: Context dictionary to add to log records
        """
        super().__init__()
        self.context = context

    def filter(self, record):
        """Add context to log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class PerformanceLogger:
    """Logger for performance metrics and timing."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.

        Args:
            logger: Base logger to use
        """
        self.logger = logger

    def log_duration(self, operation: str, duration: float, **kwargs):
        """
        Log operation duration.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            **kwargs: Additional context
        """
        context = {"operation": operation, "duration_ms": round(duration * 1000, 2)}
        context.update(kwargs)

        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        self.logger.info(f"Performance: {context_str}")

    def log_throughput(self, operation: str, items_processed: int, duration: float, **kwargs):
        """
        Log throughput metrics.

        Args:
            operation: Name of the operation
            items_processed: Number of items processed
            duration: Duration in seconds
            **kwargs: Additional context
        """
        throughput = items_processed / duration if duration > 0 else 0

        context = {
            "operation": operation,
            "items": items_processed,
            "duration_s": round(duration, 2),
            "throughput_per_s": round(throughput, 2)
        }
        context.update(kwargs)

        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        self.logger.info(f"Throughput: {context_str}")


class StructuredLogger:
    """Logger with structured output support."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize structured logger.

        Args:
            logger: Base logger to use
        """
        self.logger = logger

    def log_structured(self, level: str, message: str, **kwargs):
        """
        Log structured message with context.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            **kwargs: Structured context
        """
        # Create structured message
        context_items = []
        for key, value in kwargs.items():
            if isinstance(value, str):
                context_items.append(f'{key}="{value}"')
            else:
                context_items.append(f'{key}={value}')

        if context_items:
            structured_message = f"{message} [{', '.join(context_items)}]"
        else:
            structured_message = message

        # Log at appropriate level
        log_method = getattr(self.logger, level.lower())
        log_method(structured_message)

    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.log_structured("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.log_structured("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.log_structured("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self.log_structured("error", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self.log_structured("critical", message, **kwargs)


def configure_logging_from_config(config_manager) -> None:
    """
    Configure logging from configuration manager.

    Args:
        config_manager: ConfigManager instance
    """
    try:
        logging_config = config_manager.get("logging", {})

        # Get log file path
        log_file = config_manager.get("general.log_file")
        if log_file:
            logging_config["file"] = log_file

        setup_logging(logging_config)

    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.error(f"Failed to configure logging from config: {e}")


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name

    Returns:
        StructuredLogger instance
    """
    base_logger = get_logger(name)
    return StructuredLogger(base_logger)


def get_performance_logger(name: str) -> PerformanceLogger:
    """
    Get a performance logger instance.

    Args:
        name: Logger name

    Returns:
        PerformanceLogger instance
    """
    base_logger = get_logger(name)
    return PerformanceLogger(base_logger)