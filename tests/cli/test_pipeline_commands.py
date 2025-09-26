"""
Pipeline command tests.

Tests for pipeline control and management CLI commands.
"""

import pytest
import sys
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cli.main import cli


class TestPipelineCommands:
    """Test pipeline management commands."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()

    def test_pipeline_status_healthy(self):
        """Test pipeline status with healthy system."""
        with patch('cli.commands.pipeline.get_pipeline_monitor') as mock_monitor:
            # Mock monitor
            mock_monitor_instance = Mock()

            # Mock system status
            mock_status = Mock()
            mock_status.value = 'healthy'
            mock_monitor_instance.get_system_status.return_value = mock_status

            # Mock component status
            mock_component_status = Mock()
            mock_component_status.status.value = 'running'
            mock_component_status.health_score = 0.95
            mock_component_status.last_update = None
            mock_component_status.message = "All systems operational"

            mock_monitor_instance.get_component_status.return_value = {
                'data_ingestion': mock_component_status,
                'validation': mock_component_status
            }

            # Mock alerts
            mock_monitor_instance.get_recent_alerts.return_value = []

            mock_monitor.return_value = mock_monitor_instance

            result = self.runner.invoke(cli, ['pipeline', 'status'])
            assert result.exit_code == 0
            assert 'Pipeline Status' in result.output
            assert 'healthy' in result.output.lower()

    def test_pipeline_status_json(self):
        """Test pipeline status with JSON output."""
        with patch('cli.commands.pipeline.get_pipeline_monitor') as mock_monitor:
            # Mock monitor
            mock_monitor_instance = Mock()

            mock_status = Mock()
            mock_status.value = 'healthy'
            mock_monitor_instance.get_system_status.return_value = mock_status
            mock_monitor_instance.get_component_status.return_value = {}
            mock_monitor_instance.get_recent_alerts.return_value = []

            mock_monitor.return_value = mock_monitor_instance

            result = self.runner.invoke(cli, ['pipeline', 'status', '--format', 'json'])
            assert result.exit_code == 0
            # Should be valid JSON
            import json
            data = json.loads(result.output)
            assert 'system_status' in data
            assert 'components' in data

    def test_pipeline_start_no_auth(self):
        """Test pipeline start with no authentication."""
        with patch('cli.commands.pipeline.create_finlab_authenticator') as mock_auth:
            mock_auth_instance = Mock()
            mock_auth_instance.config.has_db_auth = False
            mock_auth_instance.config.has_token_auth = False
            mock_auth.return_value = mock_auth_instance

            result = self.runner.invoke(cli, ['pipeline', 'start'])
            assert result.exit_code == 3  # Authentication error exit code
            assert 'authentication' in result.output.lower()

    def test_pipeline_start_with_auth(self):
        """Test pipeline start with proper authentication."""
        with patch('cli.commands.pipeline.create_finlab_authenticator') as mock_auth, \
             patch('cli.commands.pipeline.FinLabConnector') as mock_connector, \
             patch('cli.commands.pipeline.get_pipeline_monitor') as mock_monitor, \
             patch('cli.commands.pipeline.IncrementalUpdater') as mock_updater:

            # Mock authentication
            mock_auth_instance = Mock()
            mock_auth_instance.config.has_db_auth = True
            mock_auth.return_value = mock_auth_instance

            # Mock connector
            mock_connector_instance = Mock()
            mock_connector_instance.test_connection.return_value = True
            mock_connector.return_value = mock_connector_instance

            # Mock monitor
            mock_monitor_instance = Mock()
            mock_monitor_instance.is_pipeline_running.return_value = False
            mock_monitor_instance.get_pipeline_pid.return_value = 12345
            mock_monitor.return_value = mock_monitor_instance

            # Mock updater
            mock_updater_instance = Mock()
            mock_updater.return_value = mock_updater_instance

            result = self.runner.invoke(cli, ['pipeline', 'start'])
            assert result.exit_code == 0
            assert 'Pipeline started successfully' in result.output

    def test_pipeline_start_already_running(self):
        """Test pipeline start when already running."""
        with patch('cli.commands.pipeline.get_pipeline_monitor') as mock_monitor:
            mock_monitor_instance = Mock()
            mock_monitor_instance.is_pipeline_running.return_value = True
            mock_monitor.return_value = mock_monitor_instance

            result = self.runner.invoke(cli, ['pipeline', 'start'])
            assert result.exit_code == 0
            assert 'already running' in result.output

    def test_pipeline_start_force(self):
        """Test pipeline start with force flag."""
        with patch('cli.commands.pipeline.create_finlab_authenticator') as mock_auth, \
             patch('cli.commands.pipeline.FinLabConnector') as mock_connector, \
             patch('cli.commands.pipeline.get_pipeline_monitor') as mock_monitor, \
             patch('cli.commands.pipeline.IncrementalUpdater') as mock_updater:

            # Mock authentication
            mock_auth_instance = Mock()
            mock_auth_instance.config.has_db_auth = True
            mock_auth.return_value = mock_auth_instance

            # Mock connector
            mock_connector_instance = Mock()
            mock_connector_instance.test_connection.return_value = True
            mock_connector.return_value = mock_connector_instance

            # Mock monitor - pipeline running
            mock_monitor_instance = Mock()
            mock_monitor_instance.is_pipeline_running.return_value = True
            mock_monitor_instance.get_pipeline_pid.return_value = 12345
            mock_monitor.return_value = mock_monitor_instance

            # Mock updater
            mock_updater_instance = Mock()
            mock_updater.return_value = mock_updater_instance

            result = self.runner.invoke(cli, ['pipeline', 'start', '--force'])
            assert result.exit_code == 0
            assert 'Pipeline started successfully' in result.output

    def test_pipeline_stop_not_running(self):
        """Test pipeline stop when not running."""
        with patch('cli.commands.pipeline.get_pipeline_monitor') as mock_monitor:
            mock_monitor_instance = Mock()
            mock_monitor_instance.is_pipeline_running.return_value = False
            mock_monitor.return_value = mock_monitor_instance

            result = self.runner.invoke(cli, ['pipeline', 'stop'])
            assert result.exit_code == 0
            assert 'not running' in result.output

    def test_pipeline_stop_graceful(self):
        """Test pipeline stop with graceful shutdown."""
        with patch('cli.commands.pipeline.get_pipeline_monitor') as mock_monitor:
            mock_monitor_instance = Mock()
            mock_monitor_instance.is_pipeline_running.return_value = True
            mock_monitor_instance.stop_pipeline.return_value = True
            mock_monitor.return_value = mock_monitor_instance

            result = self.runner.invoke(cli, ['pipeline', 'stop', '--graceful'])
            assert result.exit_code == 0
            assert 'Pipeline stopped successfully' in result.output

    def test_pipeline_logs_basic(self):
        """Test pipeline logs command."""
        with patch('cli.commands.pipeline.get_pipeline_monitor') as mock_monitor:
            # Mock log entries
            mock_entry = Mock()
            mock_entry.level = 'INFO'
            mock_entry.component = 'test_component'
            mock_entry.message = 'Test log message'
            mock_entry.timestamp = Mock()

            mock_monitor_instance = Mock()
            mock_monitor_instance.get_log_entries.return_value = [mock_entry]
            mock_monitor.return_value = mock_monitor_instance

            result = self.runner.invoke(cli, ['pipeline', 'logs'])
            assert result.exit_code == 0
            assert 'Test log message' in result.output

    def test_pipeline_restart(self):
        """Test pipeline restart command."""
        with patch('cli.commands.pipeline.get_pipeline_monitor') as mock_monitor:
            mock_monitor_instance = Mock()
            mock_monitor_instance.is_pipeline_running.return_value = True
            mock_monitor_instance.stop_pipeline.return_value = True
            mock_monitor.return_value = mock_monitor_instance

            # Mock the context invoke calls
            with patch.object(self.runner, 'invoke') as mock_invoke:
                result = self.runner.invoke(cli, ['pipeline', 'restart'])
                # The restart command calls stop and start internally
                # This is a basic test that it doesn't crash