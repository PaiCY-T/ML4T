"""
Logging utilities for CLI interface.

Provides consistent logging configuration across all CLI commands with
rich formatting and proper log level management.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def get_log_level(verbose: int, quiet: bool) -> int:
    """
    Determine log level based on verbosity flags.

    Args:
        verbose: Verbosity count (0-3)
        quiet: Quiet flag

    Returns:
        Logging level constant
    """
    if quiet:
        return logging.ERROR
    elif verbose >= 3:
        return logging.DEBUG
    elif verbose == 2:
        return logging.INFO
    elif verbose == 1:
        return logging.WARNING
    else:
        return logging.ERROR


def setup_logging(level: int = logging.WARNING,
                 log_file: Optional[str] = None,
                 no_color: bool = False) -> None:
    """
    Setup logging configuration for CLI.

    Args:
        level: Logging level
        log_file: Optional log file path
        no_color: Disable colored output
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Setup console handler with rich formatting
    console = Console(force_terminal=not no_color)
    console_handler = RichHandler(
        console=console,
        show_path=level <= logging.DEBUG,
        show_time=level <= logging.INFO,
        rich_tracebacks=True,
        tracebacks_show_locals=level <= logging.DEBUG
    )
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # Setup file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always debug level for files
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set root logger level
    root_logger.setLevel(level)

    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)

    # Ensure our modules log at the requested level
    for module in ['src.data', 'src.cli']:
        logging.getLogger(module).setLevel(level)