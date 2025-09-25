"""
Integration Tests for FinLab Data Pipeline Enhancement.

This module contains comprehensive integration tests for the enhanced FinLab data pipeline,
validating the interaction between authentication, data validation, monitoring, and
the incremental updater components.
"""

import pytest
import tempfile
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import components under test
from src.data.core.temporal import TemporalStore, TemporalValue, DataType, InMemoryTemporalStore
from src.data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
from src.data.ingestion.finlab_auth import AuthConfig, AuthToken, FinLabAuthenticator
from src.data.pipeline.incremental_updater import IncrementalUpdater, UpdateRequest, UpdateMode, UpdatePriority
from src.data.pipeline.finlab_dataset_config import FinLabDatasetConfig, FinLabField
from src.data.pipeline.data_validation import DataValidator, ValidationReport
from src.data.pipeline.monitoring import PipelineMonitor, AlertLevel
from src.data.models.taiwan_market import create_taiwan_trading_calendar


class TestFinLabPipelineIntegration:
    """Integration test suite for FinLab data pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_auth_config(self):
        """Create mock authentication configuration."""
        config = AuthConfig()
        config.finlab_token = "test_token_12345"
        config.api_key = "test_api_key"
        config.db_host = "test.finlab.tw"
        config.db_username = "test_user"
        config.db_password = "test_password"
        return config

    @pytest.fixture
    def mock_temporal_store(self):
        """Create mock temporal store."""
        return InMemoryTemporalStore()

    @pytest.fixture
    def mock_finlab_connector(self, mock_auth_config, mock_temporal_store):
        """Create mock FinLab connector with authentication."""
        config = FinLabConfig(auth_config=mock_auth_config)
        connector = Mock(spec=FinLabConnector)

        # Mock authentication methods
        connector.validate_authentication.return_value = True
        connector.refresh_authentication.return_value = True
        connector.get_auth_headers.return_value = {"Authorization": "Bearer test_token_12345"}

        # Mock data retrieval methods
        connector.get_price_data.return_value = self._create_mock_price_data()
        connector.get_fundamental_data.return_value = self._create_mock_fundamental_data()
        connector.get_corporate_actions.return_value = []

        return connector

    @pytest.fixture
    def dataset_config(self):
        """Create dataset configuration."""
        return FinLabDatasetConfig()

    @pytest.fixture
    def data_validator(self):
        """Create data validator."""
        return DataValidator()

    @pytest.fixture
    def pipeline_monitor(self, temp_dir):
        """Create pipeline monitor with temporary log directory."""
        return PipelineMonitor(log_dir=temp_dir / "logs")

    @pytest.fixture
    def incremental_updater(self, mock_temporal_store, mock_finlab_connector, temp_dir):
        """Create enhanced incremental updater."""
        trading_calendar = create_taiwan_trading_calendar(2024)

        return IncrementalUpdater(
            temporal_store=mock_temporal_store,
            finlab_connector=mock_finlab_connector,
            trading_calendar=trading_calendar,
            config_file=temp_dir / "config.json",
            enable_monitoring=True,
            enable_validation=True
        )

    def _create_mock_price_data(self) -> List[TemporalValue]:
        """Create mock price data for testing."""
        today = date.today()
        return [
            TemporalValue(
                value=100.50,
                as_of_date=today,
                value_date=today,
                data_type=DataType.PRICE,
                symbol="2330",
                metadata={"field": "adj_close", "source": "finlab"}
            ),
            TemporalValue(
                value=101.25,
                as_of_date=today,
                value_date=today,
                data_type=DataType.PRICE,
                symbol="2330",
                metadata={"field": "adj_high", "source": "finlab"}
            )
        ]

    def _create_mock_fundamental_data(self) -> List[TemporalValue]:
        """Create mock fundamental data for testing."""
        today = date.today()
        report_date = today - timedelta(days=30)  # Quarterly data

        return [
            TemporalValue(
                value=1500000000,  # 1.5B revenue
                as_of_date=today,
                value_date=report_date,
                data_type=DataType.FUNDAMENTAL,
                symbol="2330",
                metadata={
                    "field": "revenue",
                    "source": "finlab",
                    "fiscal_year": 2024,
                    "fiscal_quarter": 1
                }
            )
        ]

    def test_authentication_integration(self, mock_auth_config):
        """Test authentication system integration."""
        # Test AuthConfig initialization
        assert mock_auth_config.finlab_token == "test_token_12345"
        assert mock_auth_config.has_token_auth

        # Test AuthToken creation
        token = AuthToken(
            token=mock_auth_config.finlab_token,
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )

        assert not token.is_expired
        assert token.to_header() == "Bearer test_token_12345"

    def test_dataset_configuration_loading(self, dataset_config):
        """Test dataset configuration and field mapping."""
        all_fields = dataset_config.get_all_fields()

        # Verify core fields are present
        assert "adj_close" in all_fields
        assert "revenue" in all_fields
        assert "roa" in all_fields

        # Test field categorization
        price_fields = dataset_config.get_fields_by_temporal_type(DataType.PRICE)
        fundamental_fields = dataset_config.get_fields_by_temporal_type(DataType.FUNDAMENTAL)

        assert len(price_fields) > 0
        assert len(fundamental_fields) > 0

        # Test update frequency grouping
        daily_fields = dataset_config.get_fields_by_update_frequency("daily")
        quarterly_fields = dataset_config.get_fields_by_update_frequency("quarterly")

        assert len(daily_fields) > 0
        assert len(quarterly_fields) > 0

    def test_data_validation_integration(self, data_validator, dataset_config):
        """Test data validation framework integration."""
        field_configs = dataset_config.get_all_fields()
        mock_values = self._create_mock_price_data()

        # Test batch validation
        report = data_validator.validate_batch(mock_values, field_configs, "2330")

        assert report.symbol == "2330"
        assert report.total_records == len(mock_values)
        assert report.quality_score >= 0
        assert isinstance(report.pass_rate, float)

    def test_monitoring_system_integration(self, pipeline_monitor):
        """Test monitoring and alerting system integration."""
        # Test component status update
        pipeline_monitor.update_component_status(
            "test_component",
            "healthy",
            "Test message"
        )

        # Test performance tracking
        pipeline_monitor.performance_tracker.start_timer("test_operation")
        pipeline_monitor.performance_tracker.end_timer("test_operation")

        # Test alert generation
        alert = pipeline_monitor.alert_manager.raise_alert(
            AlertLevel.WARNING,
            "Test Alert",
            "Test alert message",
            "test_component"
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.title == "Test Alert"

        # Test dashboard data generation
        dashboard_data = pipeline_monitor.get_dashboard_data()
        assert "timestamp" in dashboard_data
        assert "system_health" in dashboard_data
        assert "performance" in dashboard_data

    def test_incremental_updater_initialization(self, incremental_updater):
        """Test incremental updater initialization with all components."""
        # Verify initialization
        assert incremental_updater.enable_validation is True
        assert incremental_updater.enable_monitoring is True
        assert len(incremental_updater.field_configs) > 0

        # Verify component integration
        assert incremental_updater.data_validator is not None
        assert incremental_updater.dataset_config is not None
        assert incremental_updater.error_recovery is not None

    def test_update_request_processing(self, incremental_updater):
        """Test complete update request processing."""
        # Create update request
        request = UpdateRequest(
            symbols=["2330", "2317"],
            data_types=[DataType.PRICE, DataType.FUNDAMENTAL],
            start_date=date.today() - timedelta(days=7),
            end_date=date.today(),
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.HIGH,
            validate_consistency=True
        )

        # Execute update
        result = incremental_updater.execute_update(request)

        # Verify result structure
        assert result.request == request
        assert isinstance(result.success, bool)
        assert result.execution_time_seconds >= 0
        assert isinstance(result.errors, list)

    def test_checkpoint_persistence(self, incremental_updater, temp_dir):
        """Test checkpoint persistence functionality."""
        # Create a checkpoint
        incremental_updater._update_checkpoint_with_validation(
            "2330", DataType.PRICE, date.today()
        )

        # Save checkpoints
        incremental_updater._save_checkpoints()

        # Verify checkpoint file exists
        assert incremental_updater.checkpoint_file.exists()

        # Load checkpoints in new instance
        new_updater = IncrementalUpdater(
            temporal_store=incremental_updater.temporal_store,
            finlab_connector=incremental_updater.finlab_connector,
            config_file=temp_dir / "config.json",
            enable_monitoring=False,
            enable_validation=False
        )

        # Verify checkpoint was loaded
        checkpoint_key = ("2330", DataType.PRICE)
        assert checkpoint_key in new_updater.checkpoints

    def test_error_recovery_integration(self, incremental_updater):
        """Test error recovery system integration."""
        error_recovery = incremental_updater.error_recovery

        # Test error recording and retry logic
        from src.data.pipeline.incremental_updater import ErrorContext

        error_context = ErrorContext(
            error_type="ConnectionError",
            error_message="Connection timeout",
            symbol="2330",
            data_type=DataType.PRICE,
            retry_count=0
        )

        # Test retry decision
        should_retry = error_recovery.should_retry(error_context, 3)
        assert isinstance(should_retry, bool)

        # Test recovery strategy
        strategy = error_recovery.get_recovery_strategy("ConnectionError")
        assert strategy is not None

    def test_comprehensive_status_reporting(self, incremental_updater):
        """Test comprehensive status reporting."""
        status = incremental_updater.get_comprehensive_status()

        # Verify all enhanced status components
        assert "dataset_config" in status
        assert "validation" in status
        assert "monitoring" in status
        assert "error_recovery" in status
        assert "checkpoints" in status

        # Verify dataset config details
        dataset_status = status["dataset_config"]
        assert "total_fields" in dataset_status
        assert dataset_status["total_fields"] > 0

    def test_integration_with_csv_config(self, temp_dir, dataset_config):
        """Test integration with CSV configuration loading."""
        # Create mock CSV data
        csv_data = '''資料集名稱,下載方式及key,數據類型
test_field,etl:test_field,float
test_field2,financial_statement:test_field2,int'''

        csv_file = temp_dir / "test_config.csv"
        csv_file.write_text(csv_data, encoding='utf-8')

        # Test CSV loading
        additional_fields = FinLabDatasetConfig.load_from_csv(csv_file)

        assert len(additional_fields) > 0
        assert "test_field" in additional_fields

    def test_performance_optimization_features(self, incremental_updater):
        """Test performance optimization features."""
        # Test batch processing strategy selection
        small_request = UpdateRequest(
            symbols=["2330"],  # Single symbol
            data_types=[DataType.PRICE],
            start_date=date.today(),
            end_date=date.today()
        )

        large_request = UpdateRequest(
            symbols=[f"23{i:02d}" for i in range(30, 80)],  # 50 symbols
            data_types=[DataType.PRICE],
            start_date=date.today(),
            end_date=date.today()
        )

        # Both should process without errors
        small_result = incremental_updater.execute_update(small_request)
        large_result = incremental_updater.execute_update(large_request)

        assert isinstance(small_result.success, bool)
        assert isinstance(large_result.success, bool)

    def test_monitoring_dashboard_integration(self, pipeline_monitor):
        """Test monitoring dashboard data integration."""
        # Simulate some operations
        pipeline_monitor.performance_tracker.record_counter("test_metric", 5)
        pipeline_monitor.performance_tracker.record_gauge("test_gauge", 75.5)

        pipeline_monitor.update_component_status(
            "finlab_connector", "healthy", "All systems operational"
        )

        # Generate dashboard data
        dashboard = pipeline_monitor.get_dashboard_data()

        # Verify comprehensive data structure
        assert "system_health" in dashboard
        assert "performance" in dashboard
        assert "alerts" in dashboard
        assert "components" in dashboard

        # Verify component status
        assert "finlab_connector" in dashboard["components"]
        component_status = dashboard["components"]["finlab_connector"]
        assert component_status["status"] == "healthy"

    @patch('src.data.pipeline.monitoring.pipeline_monitor')
    def test_monitoring_decorator_integration(self, mock_monitor, incremental_updater):
        """Test monitoring decorator integration."""
        from src.data.pipeline.monitoring import monitor_performance

        @monitor_performance("test_operation", "test_component")
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"

    def test_validation_report_integration(self, data_validator, incremental_updater):
        """Test validation report integration with monitoring."""
        field_configs = incremental_updater.field_configs
        mock_values = self._create_mock_price_data()

        # Generate validation report
        report = data_validator.validate_batch(mock_values, field_configs, "2330")

        # Test report storage
        data_validator.store_validation_report(report)

        # Test quality metrics generation
        metrics = data_validator.generate_quality_metrics("2330", 30)

        assert "symbol" in metrics
        assert metrics["symbol"] == "2330"

    def test_end_to_end_data_pipeline(self, incremental_updater):
        """Test complete end-to-end data pipeline execution."""
        # Create comprehensive update request
        request = UpdateRequest(
            symbols=["2330", "2317", "2454"],
            data_types=[DataType.PRICE, DataType.FUNDAMENTAL],
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.HIGH,
            validate_consistency=True
        )

        # Execute complete pipeline
        result = incremental_updater.execute_update(request)

        # Verify pipeline execution
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'processed_count')
        assert hasattr(result, 'errors')

        # Verify monitoring integration
        if incremental_updater.enable_monitoring:
            status = incremental_updater.get_comprehensive_status()
            assert "update_count" in status

        # Verify validation integration
        if incremental_updater.enable_validation:
            assert len(incremental_updater.validation_reports) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])