"""
FinLab API client for robust data retrieval.

Provides a comprehensive wrapper around finlab.data.get() with authentication,
error handling, rate limiting, and progress tracking.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, date
from dataclasses import dataclass
from threading import Lock
import traceback

import finlab
import finlab.data
from tqdm import tqdm

from .base import BaseDownloader, BaseDataSource
from .exceptions import (
    FinLabDownloaderError,
    DataSourceError,
    ValidationError,
    AuthenticationError,
    RateLimitError
)
from .dataset import DatasetSpecification, DownloadMethod


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_requests_per_day: int = 10000
    backoff_factor: float = 1.5
    max_retries: int = 3


@dataclass
class ProgressConfig:
    """Configuration for progress tracking."""
    show_progress: bool = True
    update_interval: float = 0.1
    chunk_size: int = 1000
    verbose: bool = False


class RateLimiter:
    """Thread-safe rate limiter for API requests."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._minute_requests = []
        self._hour_requests = []
        self._day_requests = []
        self._lock = Lock()

    def can_proceed(self) -> bool:
        """Check if a request can proceed without hitting rate limits."""
        with self._lock:
            now = time.time()
            self._cleanup_old_requests(now)

            return (
                len(self._minute_requests) < self.config.max_requests_per_minute and
                len(self._hour_requests) < self.config.max_requests_per_hour and
                len(self._day_requests) < self.config.max_requests_per_day
            )

    def wait_if_needed(self) -> float:
        """Wait if necessary to respect rate limits. Returns wait time."""
        wait_time = 0.0

        while not self.can_proceed():
            wait_seconds = min(60, 1.0 * self.config.backoff_factor)
            time.sleep(wait_seconds)
            wait_time += wait_seconds

        # Record the request
        with self._lock:
            now = time.time()
            self._minute_requests.append(now)
            self._hour_requests.append(now)
            self._day_requests.append(now)

        return wait_time

    def _cleanup_old_requests(self, now: float) -> None:
        """Remove old request timestamps."""
        # Keep only requests from the last minute, hour, and day
        self._minute_requests = [t for t in self._minute_requests if now - t < 60]
        self._hour_requests = [t for t in self._hour_requests if now - t < 3600]
        self._day_requests = [t for t in self._day_requests if now - t < 86400]


class FinLabClient(BaseDownloader, BaseDataSource):
    """
    Comprehensive FinLab API client with robust error handling and rate limiting.

    This client wraps finlab.data.get() to provide:
    - Authentication management
    - Rate limiting and quota management
    - Progress tracking for long downloads
    - Retry logic with exponential backoff
    - Data validation and conversion
    """

    def __init__(
        self,
        config: Dict[str, Any],
        rate_limit_config: Optional[RateLimitConfig] = None,
        progress_config: Optional[ProgressConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize FinLab API client.

        Args:
            config: Configuration dictionary containing API credentials
            rate_limit_config: Rate limiting configuration
            progress_config: Progress tracking configuration
            logger: Optional logger instance
        """
        # Initialize base classes
        BaseDownloader.__init__(self, config, logger)
        BaseDataSource.__init__(self, "finlab", config, logger)

        # Configuration
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.progress_config = progress_config or ProgressConfig()

        # Rate limiting
        self.rate_limiter = RateLimiter(self.rate_limit_config)

        # Connection state
        self._connected = False
        self._authenticated = False
        self._token = None

        # Statistics
        self.request_count = 0
        self.error_count = 0
        self.total_wait_time = 0.0

    def _validate_config(self) -> None:
        """Validate the configuration for FinLab client."""
        required_keys = ['api_token']

        for key in required_keys:
            if key not in self.config:
                raise ValidationError(f"Missing required configuration key: {key}")

        # Validate token format if provided
        if self.config.get('api_token'):
            token = self.config['api_token']
            if not isinstance(token, str) or len(token.strip()) == 0:
                raise ValidationError("API token must be a non-empty string")

    def connect(self) -> None:
        """Establish connection and authenticate with FinLab API."""
        if self._connected:
            return

        try:
            self.logger.info("Connecting to FinLab API...")

            # Set up authentication
            api_token = self.config.get('api_token')
            if api_token:
                self._token = api_token
                # Set FinLab token if available
                if hasattr(finlab, 'login'):
                    finlab.login(api_token)
                elif hasattr(finlab, 'set_token'):
                    finlab.set_token(api_token)
                else:
                    self.logger.warning("Unable to set FinLab token - login method not found")

                self._authenticated = True
                self.logger.info("Successfully authenticated with FinLab API")
            else:
                self.logger.warning("No API token provided - operating in limited mode")
                self._authenticated = False

            self._connected = True
            self.logger.info("FinLab API connection established")

        except Exception as e:
            self.logger.error(f"Failed to connect to FinLab API: {e}")
            raise AuthenticationError(f"FinLab API connection failed: {e}")

    def disconnect(self) -> None:
        """Close connection to FinLab API."""
        if not self._connected:
            return

        try:
            # FinLab doesn't require explicit disconnection
            self._connected = False
            self._authenticated = False
            self.logger.info("Disconnected from FinLab API")
        except Exception as e:
            self.logger.warning(f"Error during disconnect: {e}")

    def is_connected(self) -> bool:
        """Check if connected to FinLab API."""
        return self._connected

    def is_authenticated(self) -> bool:
        """Check if authenticated with FinLab API."""
        return self._authenticated

    def download(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        dataset_spec: Optional[DatasetSpecification] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Download data using FinLab API with robust error handling.

        Args:
            symbols: List of symbol identifiers
            start_date: Start date for data download
            end_date: End date for data download
            dataset_spec: Dataset specification for download method
            **kwargs: Additional parameters for download

        Returns:
            Dictionary containing downloaded data and metadata

        Raises:
            DataSourceError: When download fails
            ValidationError: When parameters are invalid
            RateLimitError: When rate limits are exceeded
        """
        if not self._connected:
            self.connect()

        # Validate parameters
        self._validate_download_params(symbols, start_date, end_date)

        # Prepare download parameters
        download_params = self._prepare_download_params(
            symbols, start_date, end_date, dataset_spec, **kwargs
        )

        # Execute download with retry logic
        return self._execute_download_with_retry(download_params)

    def download_dataset(
        self,
        dataset_spec: DatasetSpecification,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Download data for a specific dataset specification.

        Args:
            dataset_spec: Dataset specification
            start_date: Optional start date override
            end_date: Optional end date override
            **kwargs: Additional parameters

        Returns:
            Dictionary containing downloaded data and metadata
        """
        self.logger.info(f"Downloading dataset: {dataset_spec.name}")

        # Use dataset-specific download method
        if dataset_spec.download_method == DownloadMethod.ETL:
            return self._download_etl_data(dataset_spec, start_date, end_date, **kwargs)
        elif dataset_spec.download_method == DownloadMethod.FINANCIAL_STATEMENT:
            return self._download_financial_statement(dataset_spec, start_date, end_date, **kwargs)
        else:
            return self._download_generic_data(dataset_spec, start_date, end_date, **kwargs)

    def _download_etl_data(
        self,
        dataset_spec: DatasetSpecification,
        start_date: Optional[Union[str, date, datetime]],
        end_date: Optional[Union[str, date, datetime]],
        **kwargs
    ) -> Dict[str, Any]:
        """Download ETL data using finlab.data.get()."""
        try:
            # Wait for rate limit if needed
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                self.total_wait_time += wait_time
                self.logger.info(f"Waited {wait_time:.2f}s for rate limit")

            # Prepare progress tracking
            progress_bar = None
            if self.progress_config.show_progress:
                progress_bar = tqdm(
                    desc=f"Downloading {dataset_spec.name}",
                    unit="records",
                    disable=not self.progress_config.verbose
                )

            try:
                # Make the API call
                self.logger.debug(f"Calling finlab.data.get('{dataset_spec.download_key}')")

                start_time = time.time()
                data = finlab.data.get(dataset_spec.download_key)
                end_time = time.time()

                self.request_count += 1

                # Update progress
                if progress_bar:
                    if hasattr(data, '__len__'):
                        progress_bar.total = len(data)
                        progress_bar.update(len(data))
                    progress_bar.close()

                # Validate and convert data
                validated_data = self._validate_and_convert_data(data, dataset_spec)

                self.logger.info(
                    f"Successfully downloaded {dataset_spec.name} in {end_time - start_time:.2f}s"
                )

                return {
                    'data': validated_data,
                    'metadata': {
                        'dataset_name': dataset_spec.name,
                        'download_method': dataset_spec.download_method.value,
                        'download_key': dataset_spec.download_key,
                        'download_time': datetime.now().isoformat(),
                        'duration_seconds': end_time - start_time,
                        'record_count': len(validated_data) if hasattr(validated_data, '__len__') else None,
                        'data_type': dataset_spec.data_type.value
                    }
                }

            finally:
                if progress_bar:
                    progress_bar.close()

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Failed to download ETL data for {dataset_spec.name}: {e}")
            raise DataSourceError(f"ETL download failed: {e}")

    def _download_financial_statement(
        self,
        dataset_spec: DatasetSpecification,
        start_date: Optional[Union[str, date, datetime]],
        end_date: Optional[Union[str, date, datetime]],
        **kwargs
    ) -> Dict[str, Any]:
        """Download financial statement data."""
        try:
            # Wait for rate limit
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                self.total_wait_time += wait_time

            self.logger.debug(f"Downloading financial statement: {dataset_spec.download_key}")

            start_time = time.time()

            # Use appropriate finlab method for financial statements
            if hasattr(finlab.data, 'financial_statement'):
                data = finlab.data.financial_statement(dataset_spec.download_key)
            else:
                # Fallback to generic get method
                data = finlab.data.get(dataset_spec.download_key)

            end_time = time.time()
            self.request_count += 1

            # Validate and convert data
            validated_data = self._validate_and_convert_data(data, dataset_spec)

            return {
                'data': validated_data,
                'metadata': {
                    'dataset_name': dataset_spec.name,
                    'download_method': dataset_spec.download_method.value,
                    'download_key': dataset_spec.download_key,
                    'download_time': datetime.now().isoformat(),
                    'duration_seconds': end_time - start_time,
                    'record_count': len(validated_data) if hasattr(validated_data, '__len__') else None,
                    'data_type': dataset_spec.data_type.value
                }
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Failed to download financial statement {dataset_spec.name}: {e}")
            raise DataSourceError(f"Financial statement download failed: {e}")

    def _download_generic_data(
        self,
        dataset_spec: DatasetSpecification,
        start_date: Optional[Union[str, date, datetime]],
        end_date: Optional[Union[str, date, datetime]],
        **kwargs
    ) -> Dict[str, Any]:
        """Download data using generic method."""
        try:
            # Wait for rate limit
            wait_time = self.rate_limiter.wait_if_needed()
            if wait_time > 0:
                self.total_wait_time += wait_time

            self.logger.debug(f"Downloading generic data: {dataset_spec.download_key}")

            start_time = time.time()
            data = finlab.data.get(dataset_spec.download_key)
            end_time = time.time()

            self.request_count += 1

            # Validate and convert data
            validated_data = self._validate_and_convert_data(data, dataset_spec)

            return {
                'data': validated_data,
                'metadata': {
                    'dataset_name': dataset_spec.name,
                    'download_method': dataset_spec.download_method.value,
                    'download_key': dataset_spec.download_key,
                    'download_time': datetime.now().isoformat(),
                    'duration_seconds': end_time - start_time,
                    'record_count': len(validated_data) if hasattr(validated_data, '__len__') else None,
                    'data_type': dataset_spec.data_type.value
                }
            }

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Failed to download generic data {dataset_spec.name}: {e}")
            raise DataSourceError(f"Generic download failed: {e}")

    def _validate_download_params(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime]
    ) -> None:
        """Validate download parameters."""
        if not symbols:
            raise ValidationError("Symbols list cannot be empty")

        if not isinstance(symbols, list):
            raise ValidationError("Symbols must be provided as a list")

        # Date validation would go here if needed

    def _prepare_download_params(
        self,
        symbols: List[str],
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        dataset_spec: Optional[DatasetSpecification],
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare parameters for download."""
        return {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'dataset_spec': dataset_spec,
            'kwargs': kwargs
        }

    def _execute_download_with_retry(self, download_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute download with retry logic and exponential backoff."""
        max_retries = self.rate_limit_config.max_retries
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                if download_params.get('dataset_spec'):
                    return self.download_dataset(
                        download_params['dataset_spec'],
                        download_params.get('start_date'),
                        download_params.get('end_date'),
                        **download_params.get('kwargs', {})
                    )
                else:
                    # Generic download fallback
                    raise NotImplementedError("Generic download without dataset_spec not yet implemented")

            except (DataSourceError, RateLimitError) as e:
                if attempt == max_retries:
                    raise

                delay = base_delay * (self.rate_limit_config.backoff_factor ** attempt)
                self.logger.warning(
                    f"Download attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

        raise DataSourceError("Maximum retry attempts exceeded")

    def _validate_and_convert_data(self, data: Any, dataset_spec: DatasetSpecification) -> Any:
        """Validate and convert downloaded data according to specification."""
        if data is None:
            if not dataset_spec.nullable:
                raise ValidationError(f"Dataset {dataset_spec.name} returned null data but is not nullable")
            return data

        # Type-specific validation would go here
        # For now, return data as-is since finlab.data handles most conversion

        return data

    def list_available_symbols(self) -> List[str]:
        """Get list of available symbols from FinLab."""
        if not self._connected:
            self.connect()

        try:
            # Try to get available symbols
            if hasattr(finlab.data, 'search'):
                # Use search function to get available data
                available_data = finlab.data.search()
                if isinstance(available_data, list):
                    return available_data
                elif hasattr(available_data, 'keys'):
                    return list(available_data.keys())

            # Fallback: return empty list
            self.logger.warning("Unable to retrieve symbol list from FinLab API")
            return []

        except Exception as e:
            self.logger.error(f"Failed to retrieve available symbols: {e}")
            raise DataSourceError(f"Symbol list retrieval failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get client usage statistics."""
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.request_count),
            'total_wait_time': self.total_wait_time,
            'connected': self._connected,
            'authenticated': self._authenticated,
            'rate_limit_config': {
                'max_requests_per_minute': self.rate_limit_config.max_requests_per_minute,
                'max_requests_per_hour': self.rate_limit_config.max_requests_per_hour,
                'max_requests_per_day': self.rate_limit_config.max_requests_per_day
            }
        }