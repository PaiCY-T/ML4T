"""
Unit tests for FinLab API client.

Tests the core functionality of the FinLabClient including authentication,
rate limiting, error handling, and data downloading.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime

from src.finlab_downloader.core.client import (
    FinLabClient,
    RateLimitConfig,
    ProgressConfig,
    RateLimiter
)
from src.finlab_downloader.core.dataset import (
    DatasetSpecification,
    DownloadMethod,
    DataType
)
from src.finlab_downloader.core.exceptions import (
    ValidationError,
    AuthenticationError,
    DataSourceError,
    RateLimitError
)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RateLimitConfig(
            max_requests_per_minute=5,
            max_requests_per_hour=100,
            max_requests_per_day=1000,
            backoff_factor=1.5,
            max_retries=3
        )
        self.rate_limiter = RateLimiter(self.config)

    def test_can_proceed_initially(self):
        """Test that requests can proceed initially."""
        self.assertTrue(self.rate_limiter.can_proceed())

    def test_rate_limit_enforcement(self):
        """Test that rate limits are enforced."""
        # Fill up the minute quota
        for _ in range(self.config.max_requests_per_minute):
            self.rate_limiter.wait_if_needed()

        # Next request should be blocked
        self.assertFalse(self.rate_limiter.can_proceed())

    def test_cleanup_old_requests(self):
        """Test cleanup of old request timestamps."""
        # Simulate old requests by directly manipulating timestamps
        old_time = time.time() - 3700  # More than 1 hour old
        self.rate_limiter._hour_requests = [old_time]

        # Should clean up old requests
        self.assertTrue(self.rate_limiter.can_proceed())
        self.assertEqual(len(self.rate_limiter._hour_requests), 0)


class TestFinLabClient(unittest.TestCase):
    """Test FinLab API client functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'api_token': 'test_token_123'
        }
        self.rate_limit_config = RateLimitConfig(max_requests_per_minute=10)
        self.progress_config = ProgressConfig(show_progress=False)

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )
        self.assertIsInstance(client, FinLabClient)

    def test_config_validation_missing_token(self):
        """Test configuration validation with missing token."""
        invalid_config = {}
        with self.assertRaises(ValidationError) as cm:
            FinLabClient(
                config=invalid_config,
                rate_limit_config=self.rate_limit_config,
                progress_config=self.progress_config
            )
        self.assertIn("api_token", str(cm.exception))

    def test_config_validation_empty_token(self):
        """Test configuration validation with empty token."""
        invalid_config = {'api_token': ''}
        with self.assertRaises(ValidationError) as cm:
            FinLabClient(
                config=invalid_config,
                rate_limit_config=self.rate_limit_config,
                progress_config=self.progress_config
            )
        self.assertIn("non-empty string", str(cm.exception))

    @patch('src.finlab_downloader.core.client.finlab')
    def test_connect_success(self, mock_finlab):
        """Test successful connection."""
        mock_finlab.login = Mock()

        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )

        client.connect()

        self.assertTrue(client.is_connected())
        self.assertTrue(client.is_authenticated())
        mock_finlab.login.assert_called_once_with('test_token_123')

    @patch('src.finlab_downloader.core.client.finlab')
    def test_connect_without_login_method(self, mock_finlab):
        """Test connection when finlab doesn't have login method."""
        # Remove login method
        if hasattr(mock_finlab, 'login'):
            delattr(mock_finlab, 'login')
        if hasattr(mock_finlab, 'set_token'):
            delattr(mock_finlab, 'set_token')

        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )

        # Should still connect but with warning
        client.connect()

        self.assertTrue(client.is_connected())
        self.assertTrue(client.is_authenticated())

    @patch('src.finlab_downloader.core.client.finlab')
    def test_connect_failure(self, mock_finlab):
        """Test connection failure."""
        mock_finlab.login = Mock(side_effect=Exception("Connection failed"))

        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )

        with self.assertRaises(AuthenticationError):
            client.connect()

    def test_disconnect(self):
        """Test disconnection."""
        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )

        # Manually set connection state
        client._connected = True
        client._authenticated = True

        client.disconnect()

        self.assertFalse(client.is_connected())
        self.assertFalse(client.is_authenticated())

    @patch('src.finlab_downloader.core.client.finlab.data')
    def test_download_etl_data_success(self, mock_finlab_data):
        """Test successful ETL data download."""
        mock_data = {'symbol1': [1, 2, 3], 'symbol2': [4, 5, 6]}
        mock_finlab_data.get = Mock(return_value=mock_data)

        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )
        client._connected = True
        client._authenticated = True

        dataset_spec = DatasetSpecification(
            name="Test ETL Dataset",
            download_method=DownloadMethod.ETL,
            download_key="test_etl_key",
            data_type=DataType.FLOAT
        )

        result = client.download_dataset(dataset_spec)

        self.assertIn('data', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['data'], mock_data)
        self.assertEqual(result['metadata']['dataset_name'], "Test ETL Dataset")
        self.assertEqual(result['metadata']['download_method'], "etl")

        mock_finlab_data.get.assert_called_once_with("test_etl_key")

    @patch('src.finlab_downloader.core.client.finlab.data')
    def test_download_financial_statement_success(self, mock_finlab_data):
        """Test successful financial statement download."""
        mock_data = {'revenue': [100, 200, 300]}
        mock_finlab_data.get = Mock(return_value=mock_data)

        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )
        client._connected = True
        client._authenticated = True

        dataset_spec = DatasetSpecification(
            name="Test Financial Statement",
            download_method=DownloadMethod.FINANCIAL_STATEMENT,
            download_key="revenue",
            data_type=DataType.FLOAT
        )

        result = client.download_dataset(dataset_spec)

        self.assertIn('data', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['data'], mock_data)

        mock_finlab_data.get.assert_called_once_with("revenue")

    @patch('src.finlab_downloader.core.client.finlab.data')
    def test_download_with_retry(self, mock_finlab_data):
        """Test download with retry logic."""
        # First call fails, second succeeds
        mock_finlab_data.get = Mock(side_effect=[
            Exception("Network error"),
            {'data': [1, 2, 3]}
        ])

        client = FinLabClient(
            config=self.config,
            rate_limit_config=RateLimitConfig(max_retries=2, backoff_factor=1.1),
            progress_config=self.progress_config
        )
        client._connected = True
        client._authenticated = True

        dataset_spec = DatasetSpecification(
            name="Test Dataset",
            download_method=DownloadMethod.ETL,
            download_key="test_key",
            data_type=DataType.FLOAT
        )

        # Should succeed on retry
        result = client.download_dataset(dataset_spec)

        self.assertIn('data', result)
        self.assertEqual(mock_finlab_data.get.call_count, 2)

    @patch('src.finlab_downloader.core.client.finlab.data')
    def test_download_max_retries_exceeded(self, mock_finlab_data):
        """Test download when max retries are exceeded."""
        mock_finlab_data.get = Mock(side_effect=Exception("Persistent error"))

        client = FinLabClient(
            config=self.config,
            rate_limit_config=RateLimitConfig(max_retries=1, backoff_factor=1.1),
            progress_config=self.progress_config
        )
        client._connected = True
        client._authenticated = True

        dataset_spec = DatasetSpecification(
            name="Test Dataset",
            download_method=DownloadMethod.ETL,
            download_key="test_key",
            data_type=DataType.FLOAT
        )

        with self.assertRaises(DataSourceError):
            client.download_dataset(dataset_spec)

    def test_download_validation_empty_symbols(self):
        """Test download parameter validation with empty symbols."""
        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )

        with self.assertRaises(ValidationError) as cm:
            client.download([], '2023-01-01', '2023-12-31')
        self.assertIn("empty", str(cm.exception))

    def test_download_validation_non_list_symbols(self):
        """Test download parameter validation with non-list symbols."""
        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )

        with self.assertRaises(ValidationError) as cm:
            client.download("not_a_list", '2023-01-01', '2023-12-31')
        self.assertIn("list", str(cm.exception))

    @patch('src.finlab_downloader.core.client.finlab.data')
    def test_list_available_symbols(self, mock_finlab_data):
        """Test listing available symbols."""
        mock_finlab_data.search = Mock(return_value=['AAPL', 'GOOGL', 'TSLA'])

        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )
        client._connected = True

        symbols = client.list_available_symbols()

        self.assertEqual(symbols, ['AAPL', 'GOOGL', 'TSLA'])
        mock_finlab_data.search.assert_called_once()

    @patch('src.finlab_downloader.core.client.finlab.data')
    def test_list_available_symbols_no_search_method(self, mock_finlab_data):
        """Test listing symbols when search method doesn't exist."""
        if hasattr(mock_finlab_data, 'search'):
            delattr(mock_finlab_data, 'search')

        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )
        client._connected = True

        symbols = client.list_available_symbols()

        self.assertEqual(symbols, [])

    def test_get_statistics(self):
        """Test getting client statistics."""
        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )

        # Simulate some usage
        client.request_count = 10
        client.error_count = 2
        client.total_wait_time = 5.5

        stats = client.get_statistics()

        self.assertEqual(stats['request_count'], 10)
        self.assertEqual(stats['error_count'], 2)
        self.assertEqual(stats['error_rate'], 0.2)
        self.assertEqual(stats['total_wait_time'], 5.5)
        self.assertIn('rate_limit_config', stats)

    def test_context_manager(self):
        """Test client as context manager."""
        client = FinLabClient(
            config=self.config,
            rate_limit_config=self.rate_limit_config,
            progress_config=self.progress_config
        )

        with patch.object(client, 'connect') as mock_connect, \
             patch.object(client, 'disconnect') as mock_disconnect:

            with client:
                pass

            mock_connect.assert_called_once()
            mock_disconnect.assert_called_once()


if __name__ == '__main__':
    unittest.main()