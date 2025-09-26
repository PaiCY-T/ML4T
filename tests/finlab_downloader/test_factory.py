"""
Unit tests for FinLab factory classes.

Tests the factory methods for creating configured clients and catalogs.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.finlab_downloader.core.factory import (
    FinLabClientFactory,
    DatasetCatalogFactory,
    IntegratedDownloaderFactory
)
from src.finlab_downloader.core.client import FinLabClient, RateLimitConfig, ProgressConfig
from src.finlab_downloader.core.dataset import DatasetCatalog
from src.finlab_downloader.config.manager import ConfigManager
from src.finlab_downloader.core.exceptions import ConfigurationError


class TestFinLabClientFactory(unittest.TestCase):
    """Test FinLab client factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'finlab': {
                'api': {
                    'api_token': 'test_token_123'
                },
                'rate_limiting': {
                    'max_requests_per_minute': 30,
                    'max_requests_per_hour': 500,
                    'max_retries': 5
                },
                'progress': {
                    'show_progress': False,
                    'verbose': True
                }
            }
        }

    @patch('src.finlab_downloader.core.factory.ConfigManager')
    def test_create_client_success(self, mock_config_manager_class):
        """Test successful client creation from configuration."""
        # Mock config manager
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = self.mock_config
        mock_config_manager_class.return_value = mock_config_manager

        client = FinLabClientFactory.create_client(config_path='/fake/path')

        self.assertIsInstance(client, FinLabClient)
        self.assertEqual(client.config['api_token'], 'test_token_123')
        self.assertEqual(client.rate_limit_config.max_requests_per_minute, 30)
        self.assertEqual(client.rate_limit_config.max_requests_per_hour, 500)
        self.assertEqual(client.rate_limit_config.max_retries, 5)
        self.assertFalse(client.progress_config.show_progress)
        self.assertTrue(client.progress_config.verbose)

    def test_create_client_with_existing_config_manager(self):
        """Test client creation with existing config manager."""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = self.mock_config

        client = FinLabClientFactory.create_client(config_manager=mock_config_manager)

        self.assertIsInstance(client, FinLabClient)
        mock_config_manager.get_config.assert_called_once()

    @patch('src.finlab_downloader.core.factory.ConfigManager')
    def test_create_client_missing_finlab_section(self, mock_config_manager_class):
        """Test client creation with missing finlab configuration section."""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = {}
        mock_config_manager_class.return_value = mock_config_manager

        with self.assertRaises(ConfigurationError) as cm:
            FinLabClientFactory.create_client()

        self.assertIn("Missing 'finlab' configuration section", str(cm.exception))

    @patch('src.finlab_downloader.core.factory.ConfigManager')
    def test_create_client_missing_api_section(self, mock_config_manager_class):
        """Test client creation with missing API configuration section."""
        config_without_api = {'finlab': {}}
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = config_without_api
        mock_config_manager_class.return_value = mock_config_manager

        with self.assertRaises(ConfigurationError) as cm:
            FinLabClientFactory.create_client()

        self.assertIn("Missing 'finlab.api' configuration section", str(cm.exception))

    def test_create_client_with_defaults(self):
        """Test client creation with defaults."""
        client = FinLabClientFactory.create_client_with_defaults('test_token_456')

        self.assertIsInstance(client, FinLabClient)
        self.assertEqual(client.config['api_token'], 'test_token_456')
        self.assertIsInstance(client.rate_limit_config, RateLimitConfig)
        self.assertIsInstance(client.progress_config, ProgressConfig)

    def test_create_rate_limit_config(self):
        """Test rate limit configuration creation."""
        settings = {
            'max_requests_per_minute': 120,
            'max_requests_per_hour': 2000,
            'max_requests_per_day': 20000,
            'backoff_factor': 2.0,
            'max_retries': 5
        }

        config = FinLabClientFactory._create_rate_limit_config(settings)

        self.assertEqual(config.max_requests_per_minute, 120)
        self.assertEqual(config.max_requests_per_hour, 2000)
        self.assertEqual(config.max_requests_per_day, 20000)
        self.assertEqual(config.backoff_factor, 2.0)
        self.assertEqual(config.max_retries, 5)

    def test_create_rate_limit_config_defaults(self):
        """Test rate limit configuration creation with defaults."""
        config = FinLabClientFactory._create_rate_limit_config({})

        self.assertEqual(config.max_requests_per_minute, 60)
        self.assertEqual(config.max_requests_per_hour, 1000)
        self.assertEqual(config.max_requests_per_day, 10000)
        self.assertEqual(config.backoff_factor, 1.5)
        self.assertEqual(config.max_retries, 3)

    def test_create_progress_config(self):
        """Test progress configuration creation."""
        settings = {
            'show_progress': False,
            'update_interval': 0.5,
            'chunk_size': 2000,
            'verbose': True
        }

        config = FinLabClientFactory._create_progress_config(settings)

        self.assertFalse(config.show_progress)
        self.assertEqual(config.update_interval, 0.5)
        self.assertEqual(config.chunk_size, 2000)
        self.assertTrue(config.verbose)

    def test_create_progress_config_defaults(self):
        """Test progress configuration creation with defaults."""
        config = FinLabClientFactory._create_progress_config({})

        self.assertTrue(config.show_progress)
        self.assertEqual(config.update_interval, 0.1)
        self.assertEqual(config.chunk_size, 1000)
        self.assertFalse(config.verbose)


class TestDatasetCatalogFactory(unittest.TestCase):
    """Test dataset catalog factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'datasets': {
                'catalog_file': '/path/to/finlab_database.csv'
            }
        }

    @patch('src.finlab_downloader.core.factory.DatasetCatalog')
    @patch('src.finlab_downloader.core.factory.ConfigManager')
    def test_create_catalog_success(self, mock_config_manager_class, mock_catalog_class):
        """Test successful catalog creation from configuration."""
        # Mock config manager
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = self.mock_config
        mock_config_manager_class.return_value = mock_config_manager

        # Mock catalog
        mock_catalog = Mock()
        mock_catalog_class.from_csv.return_value = mock_catalog

        catalog = DatasetCatalogFactory.create_catalog(config_path='/fake/path')

        self.assertEqual(catalog, mock_catalog)
        mock_catalog_class.from_csv.assert_called_once_with('/path/to/finlab_database.csv')

    @patch('src.finlab_downloader.core.factory.DatasetCatalog')
    def test_create_catalog_with_existing_config_manager(self, mock_catalog_class):
        """Test catalog creation with existing config manager."""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = self.mock_config

        mock_catalog = Mock()
        mock_catalog_class.from_csv.return_value = mock_catalog

        catalog = DatasetCatalogFactory.create_catalog(config_manager=mock_config_manager)

        self.assertEqual(catalog, mock_catalog)
        mock_config_manager.get_config.assert_called_once()

    @patch('src.finlab_downloader.core.factory.DatasetCatalog')
    def test_create_catalog_with_csv_path_override(self, mock_catalog_class):
        """Test catalog creation with CSV path override."""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = {}

        mock_catalog = Mock()
        mock_catalog_class.from_csv.return_value = mock_catalog

        catalog = DatasetCatalogFactory.create_catalog(
            config_manager=mock_config_manager,
            csv_path='/override/path.csv'
        )

        self.assertEqual(catalog, mock_catalog)
        mock_catalog_class.from_csv.assert_called_once_with('/override/path.csv')

    @patch('src.finlab_downloader.core.factory.DatasetCatalog')
    @patch('src.finlab_downloader.core.factory.ConfigManager')
    def test_create_catalog_no_csv_path(self, mock_config_manager_class, mock_catalog_class):
        """Test catalog creation without CSV path."""
        # Config without datasets section
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = {}
        mock_config_manager_class.return_value = mock_config_manager

        # Mock empty catalog
        mock_catalog = Mock()
        mock_catalog_class.return_value = mock_catalog

        catalog = DatasetCatalogFactory.create_catalog()

        self.assertEqual(catalog, mock_catalog)
        mock_catalog_class.assert_called_once()
        mock_catalog_class.from_csv.assert_not_called()

    @patch('src.finlab_downloader.core.factory.DatasetCatalog')
    @patch('src.finlab_downloader.core.factory.ConfigManager')
    def test_create_catalog_default_csv_filename(self, mock_config_manager_class, mock_catalog_class):
        """Test catalog creation with default CSV filename."""
        # Config with empty datasets section
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = {'datasets': {}}
        mock_config_manager_class.return_value = mock_config_manager

        mock_catalog = Mock()
        mock_catalog_class.from_csv.return_value = mock_catalog

        catalog = DatasetCatalogFactory.create_catalog()

        self.assertEqual(catalog, mock_catalog)
        mock_catalog_class.from_csv.assert_called_once_with('finlab_database_cleaned.csv')


class TestIntegratedDownloaderFactory(unittest.TestCase):
    """Test integrated downloader factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'finlab': {
                'api': {
                    'api_token': 'test_token_123'
                }
            },
            'datasets': {
                'catalog_file': '/path/to/finlab_database.csv'
            }
        }

    @patch('src.finlab_downloader.core.factory.DatasetCatalogFactory.create_catalog')
    @patch('src.finlab_downloader.core.factory.FinLabClientFactory.create_client')
    def test_create_integrated_downloader_success(self, mock_create_client, mock_create_catalog):
        """Test successful integrated downloader creation."""
        mock_client = Mock()
        mock_catalog = Mock()

        mock_create_client.return_value = mock_client
        mock_create_catalog.return_value = mock_catalog

        result = IntegratedDownloaderFactory.create_integrated_downloader()

        self.assertEqual(result['client'], mock_client)
        self.assertEqual(result['catalog'], mock_catalog)
        self.assertIn('config_manager', result)

        mock_create_client.assert_called_once()
        mock_create_catalog.assert_called_once()

    @patch('src.finlab_downloader.core.factory.DatasetCatalogFactory.create_catalog')
    @patch('src.finlab_downloader.core.factory.FinLabClientFactory.create_client')
    def test_create_integrated_downloader_with_config_manager(self, mock_create_client, mock_create_catalog):
        """Test integrated downloader creation with existing config manager."""
        mock_client = Mock()
        mock_catalog = Mock()
        mock_config_manager = Mock()

        mock_create_client.return_value = mock_client
        mock_create_catalog.return_value = mock_catalog

        result = IntegratedDownloaderFactory.create_integrated_downloader(
            config_manager=mock_config_manager
        )

        self.assertEqual(result['client'], mock_client)
        self.assertEqual(result['catalog'], mock_catalog)
        self.assertEqual(result['config_manager'], mock_config_manager)

        mock_create_client.assert_called_once_with(
            config_manager=mock_config_manager,
            config_path=None,
            logger=unittest.mock.ANY
        )
        mock_create_catalog.assert_called_once_with(
            config_manager=mock_config_manager,
            config_path=None,
            logger=unittest.mock.ANY
        )

    @patch('src.finlab_downloader.core.factory.DatasetCatalogFactory.create_catalog')
    @patch('src.finlab_downloader.core.factory.FinLabClientFactory.create_client')
    def test_create_integrated_downloader_client_failure(self, mock_create_client, mock_create_catalog):
        """Test integrated downloader creation when client creation fails."""
        mock_create_client.side_effect = ConfigurationError("Client creation failed")

        with self.assertRaises(ConfigurationError) as cm:
            IntegratedDownloaderFactory.create_integrated_downloader()

        self.assertIn("Integrated downloader creation failed", str(cm.exception))

    @patch('src.finlab_downloader.core.factory.DatasetCatalogFactory.create_catalog')
    @patch('src.finlab_downloader.core.factory.FinLabClientFactory.create_client')
    def test_create_integrated_downloader_catalog_failure(self, mock_create_client, mock_create_catalog):
        """Test integrated downloader creation when catalog creation fails."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        mock_create_catalog.side_effect = ConfigurationError("Catalog creation failed")

        with self.assertRaises(ConfigurationError) as cm:
            IntegratedDownloaderFactory.create_integrated_downloader()

        self.assertIn("Integrated downloader creation failed", str(cm.exception))


if __name__ == '__main__':
    unittest.main()