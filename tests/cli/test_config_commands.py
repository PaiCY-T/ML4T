"""
Configuration command tests.

Tests for configuration management CLI commands.
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cli.main import cli
from cli.utils.config import CliConfig


class TestConfigCommands:
    """Test configuration management commands."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()

    def test_config_show_default(self):
        """Test config show with default configuration."""
        with patch('cli.utils.config.load_cli_config') as mock_load:
            mock_config = CliConfig()
            mock_config.config_file = None
            mock_load.return_value = mock_config

            result = self.runner.invoke(cli, ['config', 'show'])
            assert result.exit_code == 0
            assert 'Current Configuration' in result.output
            assert 'Authentication Settings' in result.output

    def test_config_show_json(self):
        """Test config show with JSON output."""
        with patch('cli.utils.config.load_cli_config') as mock_load:
            mock_config = CliConfig()
            mock_config.finlab_token = "test_token"
            mock_load.return_value = mock_config

            result = self.runner.invoke(cli, ['config', 'show', '--format', 'json'])
            assert result.exit_code == 0
            # Should mask sensitive data
            assert '"finlab_token": "********"' in result.output

    def test_config_show_sensitive(self):
        """Test config show with sensitive data."""
        with patch('cli.utils.config.load_cli_config') as mock_load:
            mock_config = CliConfig()
            mock_config.finlab_token = "test_token"
            mock_load.return_value = mock_config

            result = self.runner.invoke(cli, ['config', 'show', '--format', 'json', '--show-sensitive'])
            assert result.exit_code == 0
            # Should show actual token
            assert '"finlab_token": "test_token"' in result.output

    def test_config_template(self):
        """Test config template generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_file = Path(temp_dir) / "test-template.yaml"

            result = self.runner.invoke(cli, ['config', 'template', '--file', str(template_file)])
            assert result.exit_code == 0
            assert template_file.exists()

            # Check template content
            content = template_file.read_text()
            assert 'finlab_token:' in content
            assert 'finlab_db_host:' in content

    def test_config_get_value(self):
        """Test getting configuration value."""
        with patch('cli.utils.config.load_cli_config') as mock_load:
            mock_config = CliConfig()
            mock_config.default_batch_size = 1000
            mock_load.return_value = mock_config

            result = self.runner.invoke(cli, ['config', 'get', 'cli.batch_size'])
            assert result.exit_code == 0
            assert '1000' in result.output

    def test_config_get_invalid_key(self):
        """Test getting invalid configuration key."""
        result = self.runner.invoke(cli, ['config', 'get', 'invalid.key'])
        assert result.exit_code != 0
        assert 'Unknown configuration key' in result.output

    def test_config_set_value(self):
        """Test setting configuration value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test-config.yaml"

            # Create initial config
            mock_config = CliConfig()
            mock_config.config_file = str(config_file)

            with patch('cli.utils.config.load_cli_config', return_value=mock_config), \
                 patch('cli.utils.config.save_cli_config') as mock_save:

                result = self.runner.invoke(cli, [
                    'config', 'set', 'cli.batch_size', '2000',
                    '--file', str(config_file)
                ])

                # Should succeed (even though file doesn't exist, mocked)
                mock_save.assert_called_once()

    def test_config_test_no_auth(self):
        """Test config test with no authentication."""
        with patch('cli.commands.config.create_finlab_authenticator') as mock_auth:
            mock_auth_instance = Mock()
            mock_auth_instance.config.has_token_auth = False
            mock_auth_instance.config.has_db_auth = False
            mock_auth.return_value = mock_auth_instance

            result = self.runner.invoke(cli, ['config', 'test'])
            assert result.exit_code == 0
            assert 'Testing configuration' in result.output
            assert 'No authentication method configured' in result.output

    def test_config_test_with_db_auth(self):
        """Test config test with database authentication."""
        with patch('cli.commands.config.create_finlab_authenticator') as mock_auth, \
             patch('cli.commands.config.FinLabConnector') as mock_connector:

            # Mock authenticator
            mock_auth_instance = Mock()
            mock_auth_instance.config.has_token_auth = False
            mock_auth_instance.config.has_db_auth = True
            mock_auth.return_value = mock_auth_instance

            # Mock connector
            mock_connector_instance = Mock()
            mock_connector_instance.test_connection.return_value = True
            mock_connector_instance.get_connection_info.return_value = {
                'host': 'localhost',
                'database': 'finlab',
                'version': '13.0'
            }
            mock_connector.return_value = mock_connector_instance

            result = self.runner.invoke(cli, ['config', 'test'])
            assert result.exit_code == 0
            assert 'Database connection successful' in result.output