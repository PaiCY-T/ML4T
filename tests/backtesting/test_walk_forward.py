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
from src.data.core.temporal import TemporalStore, DataType, TemporalValue
from src.data.models.taiwan_market import TaiwanTradingCalendar
from src.data.pipeline.pit_engine import PointInTimeEngine, BiasCheckLevel


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


if __name__ == "__main__":
    pytest.main([__file__])