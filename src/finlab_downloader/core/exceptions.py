"""
Custom exception hierarchy for FinLab downloader.

Provides domain-specific exceptions with structured error handling capabilities.
"""

from typing import Optional, Dict, Any


class FinLabDownloaderError(Exception):
    """Base exception for all FinLab downloader errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause

    def __str__(self) -> str:
        """String representation with error code and context."""
        parts = [self.message]

        if self.error_code:
            parts.append(f"(Error Code: {self.error_code})")

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"[Context: {context_str}]")

        return " ".join(parts)


class ConfigurationError(FinLabDownloaderError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        field: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if config_path:
            context['config_path'] = config_path
        if field:
            context['field'] = field

        # Filter out known kwargs before passing to parent
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k in ['cause', 'context']}

        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            context=context,
            **filtered_kwargs
        )


class DataSourceError(FinLabDownloaderError):
    """Raised when data source operations fail."""

    def __init__(
        self,
        message: str,
        source: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if source:
            context['source'] = source
        if endpoint:
            context['endpoint'] = endpoint
        if status_code:
            context['status_code'] = status_code

        super().__init__(
            message,
            error_code="DATA_SOURCE_ERROR",
            context=context,
            **kwargs
        )


class ValidationError(FinLabDownloaderError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        expected: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if field:
            context['field'] = field
        if value is not None:
            context['value'] = value
        if expected:
            context['expected'] = expected

        # Filter out known kwargs before passing to parent
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k in ['cause', 'context']}

        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            context=context,
            **filtered_kwargs
        )


class AuthenticationError(FinLabDownloaderError):
    """Raised when authentication fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="AUTH_ERROR",
            **kwargs
        )


class RateLimitError(FinLabDownloaderError):
    """Raised when rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if retry_after:
            context['retry_after'] = retry_after

        super().__init__(
            message,
            error_code="RATE_LIMIT_ERROR",
            context=context,
            **kwargs
        )


class FileOperationError(FinLabDownloaderError):
    """Raised when file operations fail."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if file_path:
            context['file_path'] = file_path
        if operation:
            context['operation'] = operation

        super().__init__(
            message,
            error_code="FILE_ERROR",
            context=context,
            **kwargs
        )


class SchedulingError(FinLabDownloaderError):
    """Raised when scheduling operations fail."""

    def __init__(
        self,
        message: str,
        schedule_id: Optional[str] = None,
        company_ticker: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if schedule_id:
            context['schedule_id'] = schedule_id
        if company_ticker:
            context['company_ticker'] = company_ticker

        super().__init__(
            message,
            error_code="SCHEDULING_ERROR",
            context=context,
            **kwargs
        )