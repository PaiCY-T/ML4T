"""
Basic CLI functionality tests.

Tests for core CLI features, command structure, and basic functionality.
"""

import pytest
import sys
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cli.main import cli


class TestCLIBasic:
    """Test basic CLI functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'FinLab Data Integration CLI' in result.output
        assert 'pipeline' in result.output
        assert 'validation' in result.output
        assert 'monitoring' in result.output

    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(cli, ['version'])
        assert result.exit_code == 0
        assert 'FinLab CLI version' in result.output

    def test_info_command(self):
        """Test info command."""
        with patch('cli.main.create_finlab_authenticator') as mock_auth, \
             patch('cli.main.get_pipeline_monitor') as mock_monitor:

            # Mock authenticator
            mock_auth_instance = Mock()
            mock_auth_instance.config.has_token_auth = False
            mock_auth_instance.config.has_db_auth = False
            mock_auth.return_value = mock_auth_instance

            # Mock monitor
            mock_monitor_instance = Mock()
            mock_monitor_instance.get_system_status.return_value.value = "healthy"
            mock_monitor.return_value = mock_monitor_instance

            result = self.runner.invoke(cli, ['info'])
            assert result.exit_code == 0
            assert 'FinLab CLI System Information' in result.output

    def test_pipeline_help(self):
        """Test pipeline command group help."""
        result = self.runner.invoke(cli, ['pipeline', '--help'])
        assert result.exit_code == 0
        assert 'Pipeline control and management' in result.output
        assert 'status' in result.output
        assert 'start' in result.output
        assert 'stop' in result.output

    def test_validation_help(self):
        """Test validation command group help."""
        result = self.runner.invoke(cli, ['validation', '--help'])
        assert result.exit_code == 0
        assert 'Data validation and quality check' in result.output
        assert 'run' in result.output

    def test_monitoring_help(self):
        """Test monitoring command group help."""
        result = self.runner.invoke(cli, ['monitoring', '--help'])
        assert result.exit_code == 0
        assert 'System monitoring and health check' in result.output
        assert 'health' in result.output

    def test_data_help(self):
        """Test data command group help."""
        result = self.runner.invoke(cli, ['data', '--help'])
        assert result.exit_code == 0
        assert 'Data management and synchronization' in result.output
        assert 'sync' in result.output

    def test_config_help(self):
        """Test config command group help."""
        result = self.runner.invoke(cli, ['config', '--help'])
        assert result.exit_code == 0
        assert 'Configuration management' in result.output
        assert 'show' in result.output

    def test_batch_help(self):
        """Test batch command group help."""
        result = self.runner.invoke(cli, ['batch', '--help'])
        assert result.exit_code == 0
        assert 'Batch operations and scheduling' in result.output
        assert 'run' in result.output

    def test_troubleshoot_help(self):
        """Test troubleshoot command group help."""
        result = self.runner.invoke(cli, ['troubleshoot', '--help'])
        assert result.exit_code == 0
        assert 'Troubleshooting and diagnostic' in result.output
        assert 'diagnose' in result.output

    def test_invalid_command(self):
        """Test invalid command handling."""
        result = self.runner.invoke(cli, ['invalid-command'])
        assert result.exit_code != 0
        assert 'No such command' in result.output

    def test_verbose_flag(self):
        """Test verbose flag."""
        result = self.runner.invoke(cli, ['-v', '--help'])
        assert result.exit_code == 0

    def test_quiet_flag(self):
        """Test quiet flag."""
        result = self.runner.invoke(cli, ['-q', 'version'])
        # Should still work but with less output
        assert result.exit_code == 0

    def test_no_color_flag(self):
        """Test no color flag."""
        result = self.runner.invoke(cli, ['--no-color', 'version'])
        assert result.exit_code == 0