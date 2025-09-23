"""
Tests for walk-forward validation engine.

Tests the core walk-forward validation functionality including window generation,
validation, and Taiwan market-specific features.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from src.backtesting.validation.walk_forward import (
    WalkForwardSplitter,
    WalkForwardConfig,
    ValidationWindow,
    ValidationResult,
    WalkForwardValidator,
    WindowType,
    ValidationStatus,
    create_default_config
)
from src.backtesting.integration.pit_validator import (
    PITValidator,
    PITBiasDetector,
    BiasCheckResult,
    PITValidationConfig,
    BiasType,
    ValidationLevel,
    create_strict_pit_validator
)
from src.data.core.temporal import TemporalStore, DataType, TemporalValue
from src.data.models.taiwan_market import TaiwanTradingCalendar
from src.data.pipeline.pit_engine import PointInTimeEngine, BiasCheckLevel
from src.data.quality.validation_engine import ValidationEngine, QualityMonitor


class TestWalkForwardConfig:
    """Test WalkForwardConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WalkForwardConfig()
        
        assert config.train_weeks == 156
        assert config.test_weeks == 26
        assert config.purge_weeks == 2
        assert config.rebalance_weeks == 4
        assert config.window_type == WindowType.SLIDING
        assert config.min_history_weeks == 260
        assert config.use_taiwan_calendar is True
        assert config.settlement_lag_days == 2
        assert config.bias_check_level == BiasCheckLevel.STRICT
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WalkForwardConfig(
            train_weeks=52,
            test_weeks=13,
            purge_weeks=1,
            window_type=WindowType.EXPANDING
        )
        
        assert config.train_weeks == 52
        assert config.test_weeks == 13
        assert config.purge_weeks == 1
        assert config.window_type == WindowType.EXPANDING
    
    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            WalkForwardConfig(train_weeks=0)
        
        with pytest.raises(ValueError):
            WalkForwardConfig(test_weeks=-1)
        
        with pytest.raises(ValueError):
            WalkForwardConfig(purge_weeks=-1)
        
        with pytest.raises(ValueError):
            WalkForwardConfig(rebalance_weeks=0)


class TestValidationWindow:
    """Test ValidationWindow class."""
    
    def test_window_creation(self):
        """Test validation window creation."""
        window = ValidationWindow(
            window_id="test_001",
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 15),
            test_end=date(2021, 6, 30),
            purge_start=date(2021, 1, 1),
            purge_end=date(2021, 1, 14),
            window_number=1,
            total_train_days=365,
            total_test_days=166,
            purge_days=14,
            trading_days_train=252,
            trading_days_test=120
        )
        
        assert window.window_id == "test_001"
        assert window.window_number == 1
        assert window.status == ValidationStatus.PENDING
        assert window.total_train_days == 365
        assert window.total_test_days == 166
    
    def test_window_serialization(self):
        """Test window to_dict method."""
        window = ValidationWindow(
            window_id="test_001",
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 15),
            test_end=date(2021, 6, 30),
            purge_start=date(2021, 1, 1),
            purge_end=date(2021, 1, 14),
            window_number=1,
            total_train_days=365,
            total_test_days=166,
            purge_days=14,
            trading_days_train=252,
            trading_days_test=120
        )
        
        window_dict = window.to_dict()
        
        assert window_dict['window_id'] == "test_001"
        assert window_dict['train_start'] == "2020-01-01"
        assert window_dict['test_end'] == "2021-06-30"
        assert window_dict['status'] == "pending"


class TestWalkForwardSplitter:
    """Test WalkForwardSplitter class."""
    
    @pytest.fixture
    def mock_temporal_store(self):
        """Create mock temporal store."""
        return Mock(spec=TemporalStore)
    
    @pytest.fixture
    def mock_pit_engine(self):
        """Create mock PIT engine."""
        mock_engine = Mock(spec=PointInTimeEngine)
        mock_engine.check_data_availability.return_value = True
        mock_engine.query.return_value = {}
        return mock_engine
    
    @pytest.fixture
    def mock_taiwan_calendar(self):
        """Create mock Taiwan calendar."""
        calendar = Mock(spec=TaiwanTradingCalendar)
        calendar.is_trading_day.return_value = True
        return calendar
    
    @pytest.fixture
    def default_config(self):
        """Create default configuration for testing."""
        return WalkForwardConfig(
            train_weeks=52,
            test_weeks=13,
            purge_weeks=1,
            rebalance_weeks=4,
            min_history_weeks=104
        )
    
    @pytest.fixture
    def splitter(self, default_config, mock_temporal_store, mock_pit_engine, mock_taiwan_calendar):
        """Create WalkForwardSplitter instance."""
        return WalkForwardSplitter(
            config=default_config,
            temporal_store=mock_temporal_store,
            pit_engine=mock_pit_engine,
            taiwan_calendar=mock_taiwan_calendar
        )
    
    def test_splitter_initialization(self, splitter, default_config):
        """Test splitter initialization."""
        assert splitter.config == default_config
        assert splitter.temporal_store is not None
        assert splitter.pit_engine is not None
        assert splitter.taiwan_calendar is not None
    
    def test_date_range_validation(self, splitter):
        """Test date range validation."""
        # Valid date range
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)
        
        # Should not raise exception
        splitter._validate_date_range(start_date, end_date)
        
        # Invalid date range
        with pytest.raises(ValueError, match="End date must be after start date"):
            splitter._validate_date_range(end_date, start_date)
        
        # Insufficient data period
        short_end_date = date(2020, 6, 1)
        with pytest.raises(ValueError, match="Insufficient data period"):
            splitter._validate_date_range(start_date, short_end_date)
    
    def test_window_generation(self, splitter):
        """Test validation window generation."""
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)
        symbols = ["2330.TW", "2317.TW"]
        
        windows = splitter.generate_windows(start_date, end_date, symbols)
        
        assert len(windows) > 0
        assert all(isinstance(w, ValidationWindow) for w in windows)
        
        # Check window ordering
        for i in range(1, len(windows)):
            assert windows[i].window_number == windows[i-1].window_number + 1
            assert windows[i].train_start >= windows[i-1].train_start
    
    def test_sliding_window_generation(self, default_config, mock_temporal_store, mock_pit_engine, mock_taiwan_calendar):
        """Test sliding window generation."""
        config = default_config
        config.window_type = WindowType.SLIDING
        
        splitter = WalkForwardSplitter(
            config=config,
            temporal_store=mock_temporal_store,
            pit_engine=mock_pit_engine,
            taiwan_calendar=mock_taiwan_calendar
        )
        
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)
        
        windows = splitter.generate_windows(start_date, end_date)
        
        # In sliding window, training period should be constant
        if len(windows) > 1:
            train_period_1 = (windows[0].train_end - windows[0].train_start).days
            train_period_2 = (windows[1].train_end - windows[1].train_start).days
            
            # Allow some variation due to calendar adjustments
            assert abs(train_period_1 - train_period_2) <= 7
    
    def test_expanding_window_generation(self, default_config, mock_temporal_store, mock_pit_engine, mock_taiwan_calendar):
        """Test expanding window generation."""
        config = default_config
        config.window_type = WindowType.EXPANDING
        
        splitter = WalkForwardSplitter(
            config=config,
            temporal_store=mock_temporal_store,
            pit_engine=mock_pit_engine,
            taiwan_calendar=mock_taiwan_calendar
        )
        
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)
        
        windows = splitter.generate_windows(start_date, end_date)
        
        # In expanding window, training period should grow
        if len(windows) > 1:
            train_period_1 = (windows[0].train_end - windows[0].train_start).days
            train_period_2 = (windows[1].train_end - windows[1].train_start).days
            
            assert train_period_2 > train_period_1
    
    def test_window_validation(self, splitter, mock_pit_engine):
        """Test window validation."""
        window = ValidationWindow(
            window_id="test_001",
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 15),
            test_end=date(2021, 6, 30),
            purge_start=date(2021, 1, 1),
            purge_end=date(2021, 1, 14),
            window_number=1,
            total_train_days=365,
            total_test_days=166,
            purge_days=14,
            trading_days_train=252,
            trading_days_test=120
        )
        
        symbols = ["2330.TW"]
        
        # Mock successful validation
        mock_pit_engine.check_data_availability.return_value = True
        
        result = splitter.validate_window(window, symbols)
        
        assert result is True
        assert mock_pit_engine.check_data_availability.call_count == 2  # train and test
    
    def test_window_validation_failure(self, splitter, mock_pit_engine):
        """Test window validation failure."""
        window = ValidationWindow(
            window_id="test_001",
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 15),
            test_end=date(2021, 6, 30),
            purge_start=date(2021, 1, 1),
            purge_end=date(2021, 1, 14),
            window_number=1,
            total_train_days=365,
            total_test_days=166,
            purge_days=14,
            trading_days_train=252,
            trading_days_test=120
        )
        
        symbols = ["2330.TW"]
        
        # Mock failed validation
        mock_pit_engine.check_data_availability.return_value = False
        
        result = splitter.validate_window(window, symbols)
        
        assert result is False
        assert window.error_message == "Insufficient training data"
    
    def test_trading_day_counting(self, splitter, mock_taiwan_calendar):
        """Test trading day counting."""
        start_date = date(2023, 1, 1)  # Sunday
        end_date = date(2023, 1, 31)   # Tuesday
        
        # Mock trading day function
        def is_trading_day(d):
            # Monday to Friday only
            return d.weekday() < 5
        
        mock_taiwan_calendar.is_trading_day.side_effect = is_trading_day
        
        count = splitter._count_trading_days(start_date, end_date)
        
        # January 2023: 22 weekdays
        assert count == 22
    
    def test_taiwan_calendar_adjustment(self, splitter, mock_taiwan_calendar):
        """Test Taiwan calendar date adjustment."""
        # Mock calendar that excludes weekends
        def is_trading_day(d):
            return d.weekday() < 5
        
        mock_taiwan_calendar.is_trading_day.side_effect = is_trading_day
        
        # Start on Saturday (non-trading day)
        saturday = date(2023, 1, 7)  # Saturday
        
        adjusted = splitter._adjust_for_trading_calendar(saturday)
        
        # Should adjust to Monday
        assert adjusted.weekday() == 0  # Monday
        assert adjusted == date(2023, 1, 9)


class TestWalkForwardValidator:
    """Test WalkForwardValidator class."""
    
    @pytest.fixture
    def mock_splitter(self):
        """Create mock splitter."""
        splitter = Mock(spec=WalkForwardSplitter)
        
        # Mock window generation
        mock_windows = [
            ValidationWindow(
                window_id=f"test_{i:03d}",
                train_start=date(2020, 1, 1) + timedelta(weeks=i*4),
                train_end=date(2020, 12, 31) + timedelta(weeks=i*4),
                test_start=date(2021, 1, 15) + timedelta(weeks=i*4),
                test_end=date(2021, 6, 30) + timedelta(weeks=i*4),
                purge_start=date(2021, 1, 1) + timedelta(weeks=i*4),
                purge_end=date(2021, 1, 14) + timedelta(weeks=i*4),
                window_number=i+1,
                total_train_days=365,
                total_test_days=166,
                purge_days=14,
                trading_days_train=252,
                trading_days_test=120
            )
            for i in range(3)
        ]
        
        splitter.generate_windows.return_value = mock_windows
        splitter.validate_window.return_value = True
        
        return splitter
    
    @pytest.fixture
    def validator(self, mock_splitter):
        """Create WalkForwardValidator instance."""
        symbols = ["2330.TW", "2317.TW"]
        return WalkForwardValidator(mock_splitter, symbols)
    
    def test_validator_initialization(self, validator, mock_splitter):
        """Test validator initialization."""
        assert validator.splitter == mock_splitter
        assert len(validator.symbols) == 2
        assert DataType.PRICE in validator.data_types
        assert DataType.VOLUME in validator.data_types
    
    def test_validation_run_success(self, validator, mock_splitter):
        """Test successful validation run."""
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)
        
        result = validator.run_validation(start_date, end_date, validate_windows=True)
        
        assert isinstance(result, ValidationResult)
        assert result.total_windows == 3
        assert result.successful_windows == 3
        assert result.failed_windows == 0
        assert result.success_rate() == 1.0
        assert result.total_runtime_seconds > 0
    
    def test_validation_run_with_failures(self, validator, mock_splitter):
        """Test validation run with some failures."""
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)
        
        # Mock some validation failures
        mock_splitter.validate_window.side_effect = [True, False, True]
        
        result = validator.run_validation(start_date, end_date, validate_windows=True)
        
        assert result.total_windows == 3
        assert result.successful_windows == 2
        assert result.failed_windows == 1
        assert result.success_rate() == 2/3
    
    def test_validation_without_window_validation(self, validator, mock_splitter):
        """Test validation run without window validation."""
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)
        
        result = validator.run_validation(start_date, end_date, validate_windows=False)
        
        assert result.total_windows == 3
        assert result.successful_windows == 3  # All marked as successful
        assert result.failed_windows == 0
        assert mock_splitter.validate_window.call_count == 0  # No validation calls


class TestValidationResult:
    """Test ValidationResult class."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample validation result."""
        config = WalkForwardConfig()
        windows = [
            ValidationWindow(
                window_id=f"test_{i:03d}",
                train_start=date(2020, 1, 1),
                train_end=date(2020, 12, 31),
                test_start=date(2021, 1, 15),
                test_end=date(2021, 6, 30),
                purge_start=date(2021, 1, 1),
                purge_end=date(2021, 1, 14),
                window_number=i+1,
                total_train_days=365,
                total_test_days=166,
                purge_days=14,
                trading_days_train=252,
                trading_days_test=120,
                status=ValidationStatus.COMPLETED if i < 2 else ValidationStatus.FAILED
            )
            for i in range(3)
        ]
        
        return ValidationResult(
            config=config,
            windows=windows,
            total_windows=3,
            successful_windows=2,
            failed_windows=1,
            total_runtime_seconds=120.5
        )
    
    def test_success_rate(self, sample_result):
        """Test success rate calculation."""
        assert sample_result.success_rate() == 2/3
        
        # Test zero windows
        empty_result = ValidationResult(
            config=WalkForwardConfig(),
            windows=[],
            total_windows=0,
            successful_windows=0,
            failed_windows=0,
            total_runtime_seconds=0
        )
        assert empty_result.success_rate() == 0.0
    
    def test_get_window_by_id(self, sample_result):
        """Test getting window by ID."""
        window = sample_result.get_window_by_id("test_001")
        assert window is not None
        assert window.window_id == "test_001"
        assert window.window_number == 2  # 0-indexed
        
        # Test non-existent window
        assert sample_result.get_window_by_id("non_existent") is None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_default_config(self):
        """Test create_default_config function."""
        config = create_default_config()
        assert isinstance(config, WalkForwardConfig)
        assert config.train_weeks == 156  # Default value
        
        # Test with overrides
        config = create_default_config(train_weeks=52, test_weeks=13)
        assert config.train_weeks == 52
        assert config.test_weeks == 13
        
        # Test invalid parameter
        with pytest.raises(ValueError):
            create_default_config(invalid_param=123)


class TestIntegration:
    """Integration tests for walk-forward validation."""
    
    @pytest.fixture
    def integration_setup(self):
        """Set up integration test environment."""
        # Create mock dependencies
        temporal_store = Mock(spec=TemporalStore)
        pit_engine = Mock(spec=PointInTimeEngine)
        taiwan_calendar = Mock(spec=TaiwanTradingCalendar)
        
        # Configure mocks
        pit_engine.check_data_availability.return_value = True
        pit_engine.query.return_value = {}
        taiwan_calendar.is_trading_day.return_value = True
        
        return temporal_store, pit_engine, taiwan_calendar
    
    def test_end_to_end_validation(self, integration_setup):
        """Test end-to-end validation process."""
        temporal_store, pit_engine, taiwan_calendar = integration_setup
        
        # Create configuration
        config = WalkForwardConfig(
            train_weeks=52,
            test_weeks=13,
            purge_weeks=1,
            rebalance_weeks=4,
            min_history_weeks=104
        )
        
        # Create splitter
        splitter = WalkForwardSplitter(
            config=config,
            temporal_store=temporal_store,
            pit_engine=pit_engine,
            taiwan_calendar=taiwan_calendar
        )
        
        # Create validator
        symbols = ["2330.TW", "2317.TW"]
        validator = WalkForwardValidator(splitter, symbols)
        
        # Run validation
        start_date = date(2020, 1, 1)
        end_date = date(2022, 12, 31)
        
        result = validator.run_validation(start_date, end_date)
        
        # Verify results
        assert isinstance(result, ValidationResult)
        assert result.total_windows > 0
        assert result.total_runtime_seconds > 0
        assert all(w.status == ValidationStatus.COMPLETED for w in result.windows)
    
    def test_taiwan_market_features(self, integration_setup):
        """Test Taiwan market-specific features."""
        temporal_store, pit_engine, taiwan_calendar = integration_setup
        
        config = WalkForwardConfig(
            use_taiwan_calendar=True,
            settlement_lag_days=2,
            handle_lunar_new_year=True
        )
        
        splitter = WalkForwardSplitter(
            config=config,
            temporal_store=temporal_store,
            pit_engine=pit_engine,
            taiwan_calendar=taiwan_calendar
        )
        
        # Test date adjustment
        saturday = date(2023, 1, 7)  # Saturday
        
        # Mock calendar to exclude weekends
        taiwan_calendar.is_trading_day.side_effect = lambda d: d.weekday() < 5
        
        adjusted = splitter._adjust_for_trading_calendar(saturday)
        
        assert adjusted.weekday() == 0  # Should be Monday
        assert taiwan_calendar.is_trading_day.called


class TestPITValidationConfig:
    """Test PITValidationConfig class."""
    
    def test_default_config(self):
        """Test default PIT validation configuration."""
        config = PITValidationConfig()
        
        assert config.bias_check_level == BiasCheckLevel.STRICT
        assert config.validation_level == ValidationLevel.STRICT
        assert config.enable_look_ahead_detection is True
        assert config.enable_survivorship_detection is True
        assert config.max_concurrent_validations == 4
        assert config.min_data_completeness == 0.95
        assert config.require_quality_validation is True
    
    def test_custom_config(self):
        """Test custom PIT validation configuration."""
        config = PITValidationConfig(
            validation_level=ValidationLevel.PARANOID,
            max_concurrent_validations=8,
            min_data_completeness=0.99
        )
        
        assert config.validation_level == ValidationLevel.PARANOID
        assert config.max_concurrent_validations == 8
        assert config.min_data_completeness == 0.99
    
    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            PITValidationConfig(max_concurrent_validations=0)
        
        with pytest.raises(ValueError):
            PITValidationConfig(validation_timeout_seconds=-1)
        
        with pytest.raises(ValueError):
            PITValidationConfig(min_data_completeness=1.5)


class TestPITBiasDetector:
    """Test PITBiasDetector class."""
    
    @pytest.fixture
    def mock_pit_engine(self):
        """Create mock PIT engine for bias detection."""
        mock_engine = Mock(spec=PointInTimeEngine)
        mock_engine.query.return_value = {}
        return mock_engine
    
    @pytest.fixture
    def mock_taiwan_calendar(self):
        """Create mock Taiwan calendar."""
        calendar = Mock(spec=TaiwanTradingCalendar)
        calendar.is_trading_day.return_value = True
        return calendar
    
    @pytest.fixture
    def bias_detector(self, mock_pit_engine, mock_taiwan_calendar):
        """Create PITBiasDetector instance."""
        return PITBiasDetector(mock_pit_engine, mock_taiwan_calendar)
    
    @pytest.fixture
    def sample_window(self):
        """Create sample validation window."""
        return ValidationWindow(
            window_id="test_bias_001",
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 15),
            test_end=date(2021, 6, 30),
            purge_start=date(2021, 1, 1),
            purge_end=date(2021, 1, 14),
            window_number=1,
            total_train_days=365,
            total_test_days=166,
            purge_days=14,
            trading_days_train=252,
            trading_days_test=120
        )
    
    def test_no_bias_detected(self, bias_detector, sample_window, mock_pit_engine):
        """Test when no bias is detected."""
        # Mock no future data available (correct behavior)
        mock_pit_engine.query.return_value = {}
        
        symbols = ["2330.TW"]
        data_types = [DataType.PRICE]
        
        results = bias_detector.detect_all_biases(
            sample_window, symbols, data_types, ValidationLevel.STRICT
        )
        
        # Should return empty list or results with detected=False
        detected_biases = [r for r in results if r.detected]
        assert len(detected_biases) == 0
    
    def test_look_ahead_bias_detected(self, bias_detector, sample_window, mock_pit_engine):
        """Test look-ahead bias detection."""
        # Mock future data being available (incorrect behavior)
        mock_temporal_value = TemporalValue(
            value=100.0,
            as_of_date=date(2020, 12, 31),
            value_date=date(2021, 2, 1),  # Future date
            data_type=DataType.PRICE,
            symbol="2330.TW"
        )
        mock_pit_engine.query.return_value = {"2330.TW": [mock_temporal_value]}
        
        symbols = ["2330.TW"]
        data_types = [DataType.PRICE]
        
        results = bias_detector.detect_all_biases(
            sample_window, symbols, data_types, ValidationLevel.STRICT
        )
        
        # Should detect look-ahead bias
        look_ahead_results = [r for r in results if r.bias_type == BiasType.LOOK_AHEAD and r.detected]
        assert len(look_ahead_results) > 0
        assert look_ahead_results[0].severity == "critical"
    
    def test_survivorship_bias_detection(self, bias_detector, sample_window, mock_pit_engine):
        """Test survivorship bias detection."""
        # Mock corporate action data indicating delisting
        mock_ca_value = TemporalValue(
            value="DELISTING",
            as_of_date=date(2021, 3, 1),
            value_date=date(2021, 3, 1),
            data_type=DataType.CORPORATE_ACTION,
            symbol="2330.TW",
            metadata={'action_type': 'DELISTING'}
        )
        mock_pit_engine.query.return_value = {"2330.TW": [mock_ca_value]}
        
        symbols = ["2330.TW"]
        
        results = bias_detector.detect_all_biases(
            sample_window, symbols, [DataType.PRICE], ValidationLevel.STRICT
        )
        
        # Should detect survivorship bias
        survivorship_results = [r for r in results if r.bias_type == BiasType.SURVIVORSHIP and r.detected]
        assert len(survivorship_results) > 0
    
    def test_temporal_leakage_detection(self, bias_detector, sample_window, mock_pit_engine):
        """Test temporal leakage detection."""
        # Mock data with impossible temporal ordering
        mock_temporal_value = TemporalValue(
            value=100.0,
            as_of_date=date(2020, 6, 1),  # Available before it should be
            value_date=date(2020, 6, 30),
            data_type=DataType.FUNDAMENTAL,  # Should have 60-day lag
            symbol="2330.TW"
        )
        mock_pit_engine.query.return_value = {"2330.TW": [mock_temporal_value]}
        
        symbols = ["2330.TW"]
        data_types = [DataType.FUNDAMENTAL]
        
        results = bias_detector.detect_all_biases(
            sample_window, symbols, data_types, ValidationLevel.STRICT
        )
        
        # Should detect temporal leakage
        temporal_results = [r for r in results if r.bias_type == BiasType.TEMPORAL_LEAKAGE and r.detected]
        assert len(temporal_results) > 0


class TestPITValidator:
    """Test PITValidator integration class."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for PITValidator."""
        temporal_store = Mock(spec=TemporalStore)
        pit_engine = Mock(spec=PointInTimeEngine)
        validation_engine = Mock(spec=ValidationEngine)
        quality_monitor = Mock(spec=QualityMonitor)
        taiwan_calendar = Mock(spec=TaiwanTradingCalendar)
        
        # Configure mocks
        pit_engine.check_data_availability.return_value = True
        pit_engine.query.return_value = {}
        quality_monitor.check_data_completeness.return_value = 0.98
        quality_monitor.calculate_quality_score.return_value = 0.85
        taiwan_calendar.is_trading_day.return_value = True
        
        return {
            'temporal_store': temporal_store,
            'pit_engine': pit_engine,
            'validation_engine': validation_engine,
            'quality_monitor': quality_monitor,
            'taiwan_calendar': taiwan_calendar
        }
    
    @pytest.fixture
    def pit_validator(self, mock_dependencies):
        """Create PITValidator instance."""
        config = PITValidationConfig(
            validation_level=ValidationLevel.STRICT,
            enable_parallel_processing=False  # Disable for testing
        )
        
        return PITValidator(
            config=config,
            temporal_store=mock_dependencies['temporal_store'],
            pit_engine=mock_dependencies['pit_engine'],
            validation_engine=mock_dependencies['validation_engine'],
            quality_monitor=mock_dependencies['quality_monitor'],
            taiwan_calendar=mock_dependencies['taiwan_calendar']
        )
    
    def test_validator_initialization(self, pit_validator):
        """Test PITValidator initialization."""
        assert pit_validator.config.validation_level == ValidationLevel.STRICT
        assert pit_validator.temporal_store is not None
        assert pit_validator.pit_engine is not None
        assert pit_validator.validation_engine is not None
        assert pit_validator.quality_monitor is not None
        assert pit_validator.bias_detector is not None
    
    def test_walk_forward_scenario_validation(self, pit_validator):
        """Test comprehensive walk-forward scenario validation."""
        wf_config = WalkForwardConfig(
            train_weeks=52,
            test_weeks=13,
            purge_weeks=1,
            rebalance_weeks=4,
            min_history_weeks=104
        )
        
        symbols = ["2330.TW", "2317.TW"]
        start_date = date(2020, 1, 1)
        end_date = date(2022, 12, 31)
        data_types = [DataType.PRICE, DataType.VOLUME]
        
        # Run validation
        result = pit_validator.validate_walk_forward_scenario(
            wf_config=wf_config,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            enable_performance_validation=True
        )
        
        # Verify result structure
        assert 'validation_summary' in result
        assert 'windows' in result
        assert 'validation_results' in result
        assert 'bias_detection_results' in result
        assert 'quality_validation_results' in result
        assert 'performance_validation_results' in result
        assert 'summary_statistics' in result
        
        # Verify summary content
        summary = result['validation_summary']
        assert summary['total_windows'] > 0
        assert summary['total_runtime_seconds'] > 0
        assert 'successful_validations' in summary
    
    def test_single_window_validation(self, pit_validator):
        """Test single window validation."""
        window = ValidationWindow(
            window_id="test_pit_001",
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 15),
            test_end=date(2021, 6, 30),
            purge_start=date(2021, 1, 1),
            purge_end=date(2021, 1, 14),
            window_number=1,
            total_train_days=365,
            total_test_days=166,
            purge_days=14,
            trading_days_train=252,
            trading_days_test=120
        )
        
        symbols = ["2330.TW"]
        data_types = [DataType.PRICE]
        
        result = pit_validator._validate_single_window(
            window, symbols, data_types, enable_performance_validation=True
        )
        
        # Verify result structure
        assert 'validation' in result
        assert 'bias' in result
        assert 'quality' in result
        assert 'performance' in result
        
        # Verify validation success
        assert result['validation']['success'] is True
    
    def test_bias_detection_integration(self, pit_validator, mock_dependencies):
        """Test bias detection integration."""
        # Configure mock to return future data (bias scenario)
        future_data = {
            "2330.TW": [TemporalValue(
                value=100.0,
                as_of_date=date(2020, 12, 31),
                value_date=date(2021, 2, 1),
                data_type=DataType.PRICE,
                symbol="2330.TW"
            )]
        }
        mock_dependencies['pit_engine'].query.return_value = future_data
        
        window = ValidationWindow(
            window_id="test_bias_001",
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 15),
            test_end=date(2021, 6, 30),
            purge_start=date(2021, 1, 1),
            purge_end=date(2021, 1, 14),
            window_number=1,
            total_train_days=365,
            total_test_days=166,
            purge_days=14,
            trading_days_train=252,
            trading_days_test=120
        )
        
        symbols = ["2330.TW"]
        data_types = [DataType.PRICE]
        
        result = pit_validator._validate_single_window(
            window, symbols, data_types, enable_performance_validation=False
        )
        
        # Should detect bias and mark validation as failed
        assert len(result['bias']) > 0
        bias_detected = any(bias['detected'] for bias in result['bias'])
        assert bias_detected
        
        # Critical bias should cause validation failure
        critical_bias = any(bias['severity'] == 'critical' for bias in result['bias'])
        if critical_bias:
            assert result['validation']['success'] is False
    
    def test_quality_validation_integration(self, pit_validator, mock_dependencies):
        """Test quality validation integration."""
        # Configure mock to return low data completeness
        mock_dependencies['quality_monitor'].check_data_completeness.return_value = 0.50  # Below threshold
        
        window = ValidationWindow(
            window_id="test_quality_001",
            train_start=date(2020, 1, 1),
            train_end=date(2020, 12, 31),
            test_start=date(2021, 1, 15),
            test_end=date(2021, 6, 30),
            purge_start=date(2021, 1, 1),
            purge_end=date(2021, 1, 14),
            window_number=1,
            total_train_days=365,
            total_test_days=166,
            purge_days=14,
            trading_days_train=252,
            trading_days_test=120
        )
        
        symbols = ["2330.TW"]
        data_types = [DataType.PRICE]
        
        result = pit_validator._validate_single_window(
            window, symbols, data_types, enable_performance_validation=False
        )
        
        # Should detect quality issues
        assert len(result['quality']) > 0
        quality_issues = [q for q in result['quality'] if q.get('type') == 'data_completeness']
        assert len(quality_issues) > 0
    
    def test_health_score_calculation(self, pit_validator):
        """Test overall health score calculation."""
        # Create mock comprehensive result
        comprehensive_result = {
            'validation_summary': {
                'total_windows': 10,
                'successful_validations': 8,
                'total_runtime_seconds': 100.0
            },
            'bias_detection_results': [
                [{'severity': 'medium'}, {'severity': 'low'}],  # Window 1
                [{'severity': 'critical'}],                      # Window 2
                []                                               # Window 3
            ],
            'quality_validation_results': [
                [{'severity': 'high'}],   # Window 1
                [],                       # Window 2
                [{'severity': 'low'}]     # Window 3
            ]
        }
        
        health_score = pit_validator._calculate_health_score(comprehensive_result)
        
        # Should be less than 100 due to issues
        assert 0 <= health_score <= 100
        assert health_score < 80  # Should be penalized for critical bias


class TestPITIntegrationUtilities:
    """Test utility functions for PIT integration."""
    
    @pytest.fixture
    def mock_temporal_store(self):
        """Create mock temporal store."""
        return Mock(spec=TemporalStore)
    
    def test_create_standard_pit_validator(self, mock_temporal_store):
        """Test creating standard PIT validator."""
        validator = create_strict_pit_validator(mock_temporal_store)
        
        assert isinstance(validator, PITValidator)
        assert validator.config.validation_level == ValidationLevel.STRICT
        assert validator.config.enable_look_ahead_detection is True
        assert validator.config.require_quality_validation is True
    
    def test_create_strict_pit_validator_with_overrides(self, mock_temporal_store):
        """Test creating strict PIT validator with config overrides."""
        validator = create_strict_pit_validator(
            mock_temporal_store,
            max_concurrent_validations=8,
            min_data_completeness=0.99
        )
        
        assert validator.config.max_concurrent_validations == 8
        assert validator.config.min_data_completeness == 0.99
        assert validator.config.validation_level == ValidationLevel.STRICT  # Default preserved


class TestBiasCheckResult:
    """Test BiasCheckResult class."""
    
    def test_bias_check_result_creation(self):
        """Test creating bias check result."""
        result = BiasCheckResult(
            bias_type=BiasType.LOOK_AHEAD,
            detected=True,
            severity="critical",
            description="Look-ahead bias detected in price data",
            affected_windows=["window_001"],
            affected_symbols=["2330.TW"],
            remediation="Review data access patterns",
            confidence=0.95
        )
        
        assert result.bias_type == BiasType.LOOK_AHEAD
        assert result.detected is True
        assert result.severity == "critical"
        assert result.confidence == 0.95
    
    def test_bias_check_result_serialization(self):
        """Test bias check result serialization."""
        result = BiasCheckResult(
            bias_type=BiasType.SURVIVORSHIP,
            detected=False,
            severity="low",
            description="No survivorship bias detected",
            confidence=0.80
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['bias_type'] == 'survivorship'
        assert result_dict['detected'] is False
        assert result_dict['severity'] == "low"
        assert result_dict['confidence'] == 0.80


if __name__ == "__main__":
    pytest.main([__file__])