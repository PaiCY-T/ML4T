"""
Custom error classes for CLI operations.

Provides structured error handling with appropriate exit codes and
user-friendly error messages.
"""

import sys
from typing import Optional, Any


class CliError(Exception):
    """Base exception for CLI operations."""

    def __init__(self, message: str, exit_code: int = 1, context: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        self.context = context

    def __str__(self) -> str:
        return self.message


class ConfigError(CliError):
    """Configuration-related errors."""

    def __init__(self, message: str, config_file: Optional[str] = None):
        super().__init__(message, exit_code=2)
        self.config_file = config_file

    def __str__(self) -> str:
        if self.config_file:
            return f"Configuration error in {self.config_file}: {self.message}"
        return f"Configuration error: {self.message}"


class AuthenticationError(CliError):
    """Authentication-related errors."""

    def __init__(self, message: str, auth_type: Optional[str] = None):
        super().__init__(message, exit_code=3)
        self.auth_type = auth_type

    def __str__(self) -> str:
        if self.auth_type:
            return f"Authentication error ({self.auth_type}): {self.message}"
        return f"Authentication error: {self.message}"


class PipelineError(CliError):
    """Pipeline operation errors."""

    def __init__(self, message: str, component: Optional[str] = None):
        super().__init__(message, exit_code=4)
        self.component = component

    def __str__(self) -> str:
        if self.component:
            return f"Pipeline error in {self.component}: {self.message}"
        return f"Pipeline error: {self.message}"


class ValidationError(CliError):
    """Data validation errors."""

    def __init__(self, message: str, dataset: Optional[str] = None, severity: str = "error"):
        super().__init__(message, exit_code=5)
        self.dataset = dataset
        self.severity = severity

    def __str__(self) -> str:
        if self.dataset:
            return f"Validation {self.severity} in {self.dataset}: {self.message}"
        return f"Validation {self.severity}: {self.message}"


class DataError(CliError):
    """Data operation errors."""

    def __init__(self, message: str, symbol: Optional[str] = None, data_type: Optional[str] = None):
        super().__init__(message, exit_code=6)
        self.symbol = symbol
        self.data_type = data_type

    def __str__(self) -> str:
        parts = []
        if self.symbol:
            parts.append(f"symbol {self.symbol}")
        if self.data_type:
            parts.append(f"type {self.data_type}")

        if parts:
            return f"Data error ({', '.join(parts)}): {self.message}"
        return f"Data error: {self.message}"


def handle_cli_error(error: Exception, quiet: bool = False) -> int:
    """
    Handle CLI errors with appropriate messaging and exit codes.

    Args:
        error: Exception to handle
        quiet: Suppress error messages

    Returns:
        Exit code
    """
    if isinstance(error, CliError):
        if not quiet:
            print(f"Error: {error}", file=sys.stderr)
        return error.exit_code
    else:
        if not quiet:
            print(f"Unexpected error: {error}", file=sys.stderr)
        return 1


def format_error_details(error: Exception) -> dict:
    """
    Format error details for structured output.

    Args:
        error: Exception to format

    Returns:
        Dictionary with error details
    """
    base_details = {
        "error_type": type(error).__name__,
        "message": str(error),
        "exit_code": getattr(error, 'exit_code', 1)
    }

    if isinstance(error, ConfigError):
        base_details["config_file"] = error.config_file
    elif isinstance(error, AuthenticationError):
        base_details["auth_type"] = error.auth_type
    elif isinstance(error, PipelineError):
        base_details["component"] = error.component
    elif isinstance(error, ValidationError):
        base_details.update({
            "dataset": error.dataset,
            "severity": error.severity
        })
    elif isinstance(error, DataError):
        base_details.update({
            "symbol": error.symbol,
            "data_type": error.data_type
        })

    return base_details