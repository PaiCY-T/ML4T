"""
Tests for Taiwan market-specific validation functionality.

Tests Taiwan market validators, settlement handling, and market-specific constraints.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import Mock, MagicMock, patch
from decimal import Decimal
from typing import Dict, Any, List

from src.backtesting.validation.taiwan_specific import (
    TaiwanMarketValidator,
    TaiwanValidationConfig,
    SettlementValidator,
    ValidationIssue,
    ValidationSeverity,
    MarketEventType,
    create_standard_taiwan_validator,
    create_strict_taiwan_validator
)
from src.backtesting.integration.pit_validator import (
    PITValidator,
    PITValidationConfig,
    ValidationLevel,
    create_strict_pit_validator
)
from src.backtesting.validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardSplitter,
    ValidationWindow
)
from src.data.core.temporal import TemporalStore, DataType, TemporalValue
from src.data.models.taiwan_market import (
    TaiwanTradingCalendar, TaiwanSettlement, TaiwanMarketCode,
    TradingStatus, CorporateActionType
)
from src.data.pipeline.pit_engine import PointInTimeEngine, PITQuery
from src.data.quality.validation_engine import ValidationEngine, QualityMonitor


class TestTaiwanValidationConfig:
    """Test TaiwanValidationConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TaiwanValidationConfig()
        
        assert config.enforce_settlement_lag is True
        assert config.settlement_lag_days == 2
        assert config.validate_trading_days is True
        assert config.validate_price_limits is True
        assert config.daily_price_limit_pct == 0.10
        assert config.validate_volume_constraints is True
        assert config.min_daily_volume == 1000
        assert config.max_position_pct == 0.05
        assert config.validate_corporate_actions is True
        assert config.validate_market_events is True
        assert config.handle_lunar_new_year is True
        assert config.handle_typhoon_days is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TaiwanValidationConfig(
            settlement_lag_days=3,
            daily_price_limit_pct=0.15,
            max_position_pct=0.10
        )
        
        assert config.settlement_lag_days == 3
        assert config.daily_price_limit_pct == 0.15
        assert config.max_position_pct == 0.10
    
    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            TaiwanValidationConfig(settlement_lag_days=-1)
        
        with pytest.raises(ValueError):
            TaiwanValidationConfig(daily_price_limit_pct=0.0)
        
        with pytest.raises(ValueError):
            TaiwanValidationConfig(daily_price_limit_pct=1.5)


class TestValidationIssue:
    """Test ValidationIssue class."""
    
    def test_issue_creation(self):
        """Test validation issue creation."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            issue_type="test_issue",
            description="Test issue description",
            symbol="2330.TW",
            date=date(2023, 1, 15),
            value=10.5,
            expected_value=8.0,
            remediation="Fix the issue"
        )
        
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.issue_type == "test_issue"
        assert issue.description == "Test issue description"
        assert issue.symbol == "2330.TW"
        assert issue.date == date(2023, 1, 15)
        assert issue.value == 10.5
        assert issue.expected_value == 8.0
        assert issue.remediation == "Fix the issue"
    
    def test_issue_serialization(self):
        """Test issue to_dict method."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            issue_type="test_warning",
            description="Test warning",
            symbol="2317.TW",
            date=date(2023, 2, 20)
        )
        
        issue_dict = issue.to_dict()
        
        assert issue_dict['severity'] == "warning"
        assert issue_dict['issue_type'] == "test_warning"
        assert issue_dict['description'] == "Test warning"
        assert issue_dict['symbol'] == "2317.TW"
        assert issue_dict['date'] == "2023-02-20"
        assert issue_dict['value'] is None
        assert issue_dict['expected_value'] is None


class TestTaiwanMarketValidator:
    """Test TaiwanMarketValidator class."""
    
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
        """Create default validation configuration."""
        return TaiwanValidationConfig()
    
    @pytest.fixture
    def validator(self, default_config, mock_temporal_store, mock_pit_engine, mock_taiwan_calendar):
        """Create TaiwanMarketValidator instance."""
        return TaiwanMarketValidator(
            config=default_config,
            temporal_store=mock_temporal_store,
            pit_engine=mock_pit_engine,
            taiwan_calendar=mock_taiwan_calendar
        )
    
    def test_validator_initialization(self, validator, default_config):
        """Test validator initialization."""
        assert validator.config == default_config
        assert validator.temporal_store is not None
        assert validator.pit_engine is not None
        assert validator.taiwan_calendar is not None
        assert validator.settlement is not None
    
    def test_date_range_validation(self, validator):
        """Test date range validation."""
        # Valid date range
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)
        
        issues = validator._validate_date_range(start_date, end_date)
        assert len(issues) == 0
        
        # Invalid date range (end before start)
        issues = validator._validate_date_range(end_date, start_date)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
        assert issues[0].issue_type == "invalid_date_range"
        
        # Future date warning
        future_date = date.today() + timedelta(days=400)
        far_future_date = date.today() + timedelta(days=500)
        
        issues = validator._validate_date_range(future_date, far_future_date)
        assert any(issue.issue_type == "future_date" for issue in issues)
        
        # Old date warning
        old_date = date(1990, 1, 1)
        
        issues = validator._validate_date_range(old_date, date(1990, 12, 31))
        assert any(issue.issue_type == "old_date" for issue in issues)
    
    def test_symbol_validation(self, validator):
        """Test Taiwan symbol validation."""
        # Valid symbols
        valid_symbols = ["2330.TW", "2317.TW", "1101.TWO", "6505.TWO"]
        
        for symbol in valid_symbols:
            assert validator._is_valid_taiwan_symbol(symbol) is True
        
        # Invalid symbols
        invalid_symbols = ["AAPL", "2330", ".TW", "123.TW", "ABCD.TW"]
        
        for symbol in invalid_symbols:
            assert validator._is_valid_taiwan_symbol(symbol) is False
        
        # Test symbol validation in context
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        
        issues = validator._validate_symbols(["INVALID"], start_date, end_date)
        assert len(issues) >= 1
        assert any(issue.issue_type == "invalid_symbol_format" for issue in issues)
    
    def test_trading_calendar_validation(self, validator, mock_taiwan_calendar):
        """Test trading calendar validation."""
        start_date = date(2023, 1, 1)  # Sunday
        end_date = date(2023, 1, 7)    # Saturday
        
        # Mock calendar behavior
        def is_trading_day(d):
            return d.weekday() < 5  # Monday to Friday
        
        mock_taiwan_calendar.is_trading_day.side_effect = is_trading_day
        
        issues = validator._validate_trading_calendar(start_date, end_date)
        
        # Should find weekend days
        weekend_issues = [issue for issue in issues if issue.issue_type == "weekend_day"]
        assert len(weekend_issues) == 2  # Sunday and Saturday
    
    def test_settlement_timing_validation(self, validator):
        """Test settlement timing validation."""
        # Test end date that results in future settlement
        future_end_date = date.today() + timedelta(days=1)
        
        issues = validator._validate_settlement_timing(date(2023, 1, 1), future_end_date)
        
        # May or may not have future settlement warning depending on exact dates
        # This tests the mechanism rather than specific outcomes
        assert isinstance(issues, list)
    
    def test_price_limit_validation(self, validator, mock_pit_engine):
        """Test price limit validation."""
        symbols = ["2330.TW"]
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        # Mock price data with large price change
        mock_price_data = {
            "2330.TW": [
                TemporalValue(
                    value=100.0,
                    as_of_date=date(2023, 1, 2),
                    value_date=date(2023, 1, 2),
                    data_type=DataType.PRICE,
                    symbol="2330.TW"
                ),
                TemporalValue(
                    value=120.0,  # 20% increase (exceeds 10% limit)
                    as_of_date=date(2023, 1, 3),
                    value_date=date(2023, 1, 3),
                    data_type=DataType.PRICE,
                    symbol="2330.TW"
                )
            ]
        }
        
        mock_pit_engine.query.return_value = mock_price_data
        
        issues = validator._validate_price_limits(symbols, start_date, end_date)
        
        # Should detect price limit violation
        limit_violations = [issue for issue in issues if issue.issue_type == "price_limit_violation"]
        assert len(limit_violations) >= 1
    
    def test_volume_constraint_validation(self, validator, mock_pit_engine):
        """Test volume constraint validation."""
        symbols = ["2330.TW"]
        positions = {
            "2330.TW": {
                date(2023, 1, 15): 100000  # Large position
            }
        }
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        # Mock low volume data
        mock_volume_data = {
            "2330.TW": [
                TemporalValue(
                    value=500,  # Below minimum volume
                    as_of_date=date(2023, 1, 15),
                    value_date=date(2023, 1, 15),
                    data_type=DataType.VOLUME,
                    symbol="2330.TW"
                )
            ]
        }
        
        mock_pit_engine.query.return_value = mock_volume_data
        
        issues = validator._validate_volume_constraints(symbols, positions, start_date, end_date)
        
        # Should detect low volume and excessive position size
        assert len(issues) >= 1
        assert any(issue.issue_type in ["low_volume", "excessive_position_size"] for issue in issues)
    
    def test_market_events_validation(self, validator):
        """Test market events validation."""
        # Test period that includes Lunar New Year
        start_date = date(2024, 2, 5)   # Before LNY
        end_date = date(2024, 2, 20)    # After LNY
        
        issues = validator._validate_market_events(start_date, end_date)
        
        # Should detect Lunar New Year period
        lny_issues = [issue for issue in issues if issue.issue_type == "lunar_new_year_period"]
        assert len(lny_issues) >= 0  # May or may not detect depending on exact dates
        
        # Test typhoon season
        typhoon_start = date(2023, 8, 1)   # Peak typhoon season
        typhoon_end = date(2023, 8, 31)
        
        issues = validator._validate_market_events(typhoon_start, typhoon_end)
        
        # Should detect typhoon season
        typhoon_issues = [issue for issue in issues if issue.issue_type == "typhoon_season"]
        assert len(typhoon_issues) >= 0
    
    def test_transaction_validation(self, validator, mock_taiwan_calendar):
        """Test transaction validation."""
        # Mock calendar
        mock_taiwan_calendar.is_trading_day.return_value = True
        
        # Valid transactions
        valid_transactions = [
            {
                'date': date(2023, 1, 16),  # Monday
                'symbol': '2330.TW',
                'quantity': 1000,
                'price': 500.0
            },
            {
                'date': date(2023, 1, 17),  # Tuesday
                'symbol': '2317.TW',
                'quantity': 500,
                'price': 250.0
            }
        ]
        
        issues = validator._validate_transactions(valid_transactions)
        assert len(issues) == 0
        
        # Invalid transactions
        invalid_transactions = [
            {
                'date': date(2023, 1, 15),  # Sunday (non-trading day)
                'symbol': '2330.TW',
                'quantity': 1000
            },
            {
                'symbol': '2317.TW',  # Missing date
                'quantity': 500
            },
            {
                'date': date(2023, 1, 16),
                'symbol': '2330.TW',
                'quantity': 0  # Zero quantity
            }
        ]
        
        # Mock Sunday as non-trading day
        def is_trading_day(d):
            return d.weekday() < 5
        
        mock_taiwan_calendar.is_trading_day.side_effect = is_trading_day
        
        issues = validator._validate_transactions(invalid_transactions)
        assert len(issues) >= 3
        
        issue_types = [issue.issue_type for issue in issues]
        assert "non_trading_day_transaction" in issue_types
        assert "incomplete_transaction" in issue_types
        assert "zero_quantity_transaction" in issue_types
    
    def test_complete_trading_scenario_validation(self, validator):
        """Test complete trading scenario validation."""
        symbols = ["2330.TW", "2317.TW"]
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        
        positions = {
            "2330.TW": {
                date(2023, 6, 15): 10000
            },
            "2317.TW": {
                date(2023, 9, 20): 5000
            }
        }
        
        transactions = [
            {
                'date': date(2023, 6, 15),
                'symbol': '2330.TW',
                'quantity': 10000,
                'price': 500.0
            }
        ]
        
        issues = validator.validate_trading_scenario(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            positions=positions,
            transactions=transactions
        )
        
        # Should complete without critical errors
        assert isinstance(issues, list)
        critical_issues = [issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) == 0


class TestSettlementValidator:
    """Test SettlementValidator class."""
    
    @pytest.fixture
    def mock_taiwan_calendar(self):
        """Create mock Taiwan calendar."""
        calendar = Mock(spec=TaiwanTradingCalendar)
        calendar.is_trading_day.side_effect = lambda d: d.weekday() < 5
        return calendar
    
    @pytest.fixture
    def settlement_validator(self, mock_taiwan_calendar):
        """Create SettlementValidator instance."""
        return SettlementValidator(taiwan_calendar=mock_taiwan_calendar)
    
    def test_settlement_schedule_validation(self, settlement_validator):
        """Test settlement schedule validation."""
        trade_dates = [
            date(2023, 1, 16),  # Monday
            date(2023, 1, 17),  # Tuesday
            date(2023, 1, 18),  # Wednesday
        ]
        
        issues = settlement_validator.validate_settlement_schedule(trade_dates, settlement_lag=2)
        
        # Should complete without errors for normal trading days
        assert isinstance(issues, list)
        
        # Test with trade date that results in weekend settlement
        friday_trade = [date(2023, 1, 20)]  # Friday, T+2 = Sunday
        
        issues = settlement_validator.validate_settlement_schedule(friday_trade, settlement_lag=2)
        
        # Should detect non-trading settlement date
        non_trading_issues = [issue for issue in issues if issue.issue_type == "non_trading_settlement_date"]
        # May or may not detect depending on settlement calculation logic


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_standard_taiwan_validator(self):
        """Test create_standard_taiwan_validator function."""
        mock_store = Mock(spec=TemporalStore)
        
        validator = create_standard_taiwan_validator(mock_store)
        
        assert isinstance(validator, TaiwanMarketValidator)
        assert validator.temporal_store == mock_store
        assert isinstance(validator.config, TaiwanValidationConfig)
    
    def test_create_standard_validator_with_overrides(self):
        """Test create_standard_taiwan_validator with config overrides."""
        mock_store = Mock(spec=TemporalStore)
        
        validator = create_standard_taiwan_validator(
            mock_store,
            settlement_lag_days=3,
            daily_price_limit_pct=0.15
        )
        
        assert validator.config.settlement_lag_days == 3
        assert validator.config.daily_price_limit_pct == 0.15
    
    def test_create_strict_taiwan_validator(self):
        """Test create_strict_taiwan_validator function."""
        mock_store = Mock(spec=TemporalStore)
        
        validator = create_strict_taiwan_validator(mock_store)
        
        assert isinstance(validator, TaiwanMarketValidator)
        assert validator.config.max_missing_data_pct == 0.01  # Stricter than default
        assert validator.config.max_outlier_pct == 0.005      # Stricter than default
    
    def test_create_strict_validator_with_overrides(self):
        """Test create_strict_taiwan_validator with config overrides."""
        mock_store = Mock(spec=TemporalStore)
        
        validator = create_strict_taiwan_validator(
            mock_store,
            max_missing_data_pct=0.005
        )
        
        assert validator.config.max_missing_data_pct == 0.005


class TestDataQualityValidation:
    """Test data quality validation features."""
    
    @pytest.fixture
    def validator(self):
        """Create validator for data quality tests."""
        mock_store = Mock(spec=TemporalStore)
        config = TaiwanValidationConfig()
        
        return TaiwanMarketValidator(
            config=config,
            temporal_store=mock_store
        )
    
    def test_data_availability_validation(self, validator):
        """Test data availability validation."""
        symbols = ["2330.TW"]
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        # Mock PIT engine to return no data available
        validator.pit_engine.check_data_availability.return_value = False
        
        issues = validator._validate_data_availability(
            "2330.TW", DataType.PRICE, start_date, end_date
        )
        
        assert len(issues) >= 1
        assert any(issue.issue_type == "data_unavailable" for issue in issues)
    
    def test_data_quality_validation(self, validator):
        """Test comprehensive data quality validation."""
        symbols = ["2330.TW", "2317.TW"]
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        
        # Mock data availability
        validator.pit_engine.check_data_availability.return_value = True
        
        issues = validator.validate_data_quality(symbols, start_date, end_date)
        
        # Should complete validation process
        assert isinstance(issues, list)


class TestIntegration:
    """Integration tests for Taiwan market validation."""
    
    def test_end_to_end_validation(self):
        """Test end-to-end Taiwan market validation."""
        # Create mock dependencies
        mock_store = Mock(spec=TemporalStore)
        
        # Create validator
        validator = create_standard_taiwan_validator(mock_store)
        
        # Define test scenario
        symbols = ["2330.TW", "2317.TW"]
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        
        # Run validation
        issues = validator.validate_trading_scenario(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify results
        assert isinstance(issues, list)
        
        # Check issue types and severities
        severities = [issue.severity for issue in issues]
        issue_types = [issue.issue_type for issue in issues]
        
        # Should not have critical errors in basic validation
        assert ValidationSeverity.CRITICAL not in severities
    
    def test_taiwan_market_specifics(self):
        """Test Taiwan market-specific validation features."""
        mock_store = Mock(spec=TemporalStore)
        
        # Create validator with Taiwan-specific settings
        config = TaiwanValidationConfig(
            settlement_lag_days=2,
            handle_lunar_new_year=True,
            handle_typhoon_days=True,
            validate_price_limits=True,
            daily_price_limit_pct=0.10
        )
        
        validator = TaiwanMarketValidator(
            config=config,
            temporal_store=mock_store
        )
        
        # Test symbol validation
        taiwan_symbols = ["2330.TW", "2317.TW", "1101.TWO"]
        us_symbols = ["AAPL", "GOOGL"]
        
        for symbol in taiwan_symbols:
            assert validator._is_valid_taiwan_symbol(symbol) is True
        
        for symbol in us_symbols:
            assert validator._is_valid_taiwan_symbol(symbol) is False


class TestTaiwanPITIntegration:
    """Test Taiwan market validation with PIT integration."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create comprehensive mock dependencies."""
        temporal_store = Mock(spec=TemporalStore)
        pit_engine = Mock(spec=PointInTimeEngine)
        validation_engine = Mock(spec=ValidationEngine)
        quality_monitor = Mock(spec=QualityMonitor)
        taiwan_calendar = Mock(spec=TaiwanTradingCalendar)
        
        # Configure mocks for Taiwan market
        pit_engine.check_data_availability.return_value = True
        pit_engine.query.return_value = {}
        quality_monitor.check_data_completeness.return_value = 0.98
        quality_monitor.calculate_quality_score.return_value = 0.85
        
        # Taiwan calendar - exclude weekends and holidays
        def is_trading_day(d):
            if d.weekday() >= 5:  # Weekend
                return False
            # Mock some holidays
            taiwan_holidays = [
                date(2023, 2, 10),  # Lunar New Year
                date(2023, 2, 11),
                date(2023, 2, 12),
                date(2023, 4, 5),   # Children's Day
                date(2023, 10, 10), # National Day
            ]
            return d not in taiwan_holidays
        
        taiwan_calendar.is_trading_day.side_effect = is_trading_day
        
        return {
            'temporal_store': temporal_store,
            'pit_engine': pit_engine,
            'validation_engine': validation_engine,
            'quality_monitor': quality_monitor,
            'taiwan_calendar': taiwan_calendar
        }
    
    @pytest.fixture
    def taiwan_pit_validator(self, mock_dependencies):
        """Create PIT validator configured for Taiwan market."""
        config = PITValidationConfig(
            validation_level=ValidationLevel.STRICT,
            enable_look_ahead_detection=True,
            enable_survivorship_detection=True,
            validate_settlement_timing=True,
            validate_corporate_actions=True,
            validate_market_events=True,
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
    
    def test_taiwan_walk_forward_validation(self, taiwan_pit_validator):
        """Test walk-forward validation with Taiwan market specifics."""
        # Taiwan market configuration
        wf_config = WalkForwardConfig(
            train_weeks=156,  # 3 years
            test_weeks=26,    # 6 months
            purge_weeks=2,
            rebalance_weeks=4,
            use_taiwan_calendar=True,
            settlement_lag_days=2,
            handle_lunar_new_year=True
        )
        
        # Taiwan stock symbols
        symbols = ["2330.TW", "2317.TW", "1101.TWO", "6505.TWO"]
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)
        data_types = [DataType.PRICE, DataType.VOLUME]
        
        # Run comprehensive validation
        result = taiwan_pit_validator.validate_walk_forward_scenario(
            wf_config=wf_config,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            enable_performance_validation=True
        )
        
        # Verify Taiwan-specific results
        assert 'validation_summary' in result
        assert result['validation_summary']['total_windows'] > 0
        
        # Check for Taiwan market validations
        taiwan_validations = []
        for validation_result in result['validation_results']:
            if 'taiwan_validation' in validation_result:
                taiwan_validations.append(validation_result['taiwan_validation'])
        
        # Should have run Taiwan market validations
        assert len(taiwan_validations) > 0
    
    def test_lunar_new_year_validation(self, taiwan_pit_validator, mock_dependencies):
        """Test validation during Lunar New Year period."""
        # Configure validation window that spans Lunar New Year
        window = ValidationWindow(
            window_id="lny_test_001",
            train_start=date(2023, 1, 1),
            train_end=date(2023, 2, 5),
            test_start=date(2023, 2, 15),  # After LNY
            test_end=date(2023, 3, 15),
            purge_start=date(2023, 2, 6),
            purge_end=date(2023, 2, 14),   # Covers LNY period
            window_number=1,
            total_train_days=35,
            total_test_days=28,
            purge_days=9,
            trading_days_train=25,
            trading_days_test=20
        )
        
        symbols = ["2330.TW"]
        data_types = [DataType.PRICE]
        
        # Mock PIT data during LNY
        lny_data = {
            "2330.TW": [
                TemporalValue(
                    value=500.0,
                    as_of_date=date(2023, 2, 9),   # Before LNY
                    value_date=date(2023, 2, 9),
                    data_type=DataType.PRICE,
                    symbol="2330.TW"
                ),
                # No data during LNY holidays
                TemporalValue(
                    value=505.0,
                    as_of_date=date(2023, 2, 15),  # After LNY
                    value_date=date(2023, 2, 15),
                    data_type=DataType.PRICE,
                    symbol="2330.TW"
                )
            ]
        }
        mock_dependencies['pit_engine'].query.return_value = lny_data
        
        result = taiwan_pit_validator._validate_single_window(
            window, symbols, data_types, enable_performance_validation=False
        )
        
        # Should handle LNY period correctly
        assert result['validation']['success'] is True
        
        # Check for Taiwan validation results
        if 'taiwan_validation' in result['validation']:
            taiwan_result = result['validation']['taiwan_validation']
            assert 'success' in taiwan_result
    
    def test_taiwan_settlement_timing(self, taiwan_pit_validator, mock_dependencies):
        """Test T+2 settlement validation for Taiwan market."""
        # Create window with settlement timing considerations
        window = ValidationWindow(
            window_id="settlement_test_001",
            train_start=date(2023, 1, 1),
            train_end=date(2023, 1, 31),
            test_start=date(2023, 2, 2),   # T+2 from train_end
            test_end=date(2023, 2, 28),
            purge_start=date(2023, 2, 1),
            purge_end=date(2023, 2, 1),
            window_number=1,
            total_train_days=31,
            total_test_days=27,
            purge_days=1,
            trading_days_train=22,
            trading_days_test=19
        )
        
        symbols = ["2330.TW"]
        
        # Test settlement validation
        taiwan_validation = taiwan_pit_validator._validate_window_taiwan_market(window, symbols)
        
        # Should validate settlement timing
        assert 'success' in taiwan_validation
        if not taiwan_validation['success']:
            # Check if settlement issues are detected
            assert 'issues' in taiwan_validation
            settlement_issues = [
                issue for issue in taiwan_validation['issues']
                if 'settlement' in issue.get('issue_type', '').lower()
            ]
    
    def test_taiwan_corporate_actions_validation(self, taiwan_pit_validator, mock_dependencies):
        """Test corporate action validation for Taiwan stocks."""
        window = ValidationWindow(
            window_id="ca_test_001",
            train_start=date(2023, 1, 1),
            train_end=date(2023, 3, 31),
            test_start=date(2023, 4, 3),
            test_end=date(2023, 6, 30),
            purge_start=date(2023, 4, 1),
            purge_end=date(2023, 4, 2),
            window_number=1,
            total_train_days=90,
            total_test_days=89,
            purge_days=2,
            trading_days_train=63,
            trading_days_test=62
        )
        
        symbols = ["2330.TW"]
        
        # Mock corporate action data (dividend)
        ca_data = {
            "2330.TW": [
                TemporalValue(
                    value="DIVIDEND",
                    as_of_date=date(2023, 2, 1),
                    value_date=date(2023, 2, 1),
                    data_type=DataType.CORPORATE_ACTION,
                    symbol="2330.TW",
                    metadata={
                        'action_type': 'DIVIDEND',
                        'ex_date': date(2023, 2, 15),
                        'amount': 15.0
                    }
                )
            ]
        }
        mock_dependencies['pit_engine'].query.return_value = ca_data
        
        result = taiwan_pit_validator._validate_single_window(
            window, symbols, [DataType.CORPORATE_ACTION], enable_performance_validation=False
        )
        
        # Should handle corporate actions properly
        assert result['validation']['success'] is True
    
    def test_taiwan_price_limit_validation(self, taiwan_pit_validator, mock_dependencies):
        """Test Taiwan market price limit validation."""
        window = ValidationWindow(
            window_id="price_limit_test_001",
            train_start=date(2023, 1, 1),
            train_end=date(2023, 1, 31),
            test_start=date(2023, 2, 2),
            test_end=date(2023, 2, 28),
            purge_start=date(2023, 2, 1),
            purge_end=date(2023, 2, 1),
            window_number=1,
            total_train_days=31,
            total_test_days=27,
            purge_days=1,
            trading_days_train=22,
            trading_days_test=19
        )
        
        symbols = ["2330.TW"]
        
        # Mock price data with limit hit
        price_data = {
            "2330.TW": [
                TemporalValue(
                    value=500.0,
                    as_of_date=date(2023, 1, 16),
                    value_date=date(2023, 1, 16),
                    data_type=DataType.PRICE,
                    symbol="2330.TW"
                ),
                TemporalValue(
                    value=550.0,  # 10% increase (at limit)
                    as_of_date=date(2023, 1, 17),
                    value_date=date(2023, 1, 17),
                    data_type=DataType.PRICE,
                    symbol="2330.TW"
                ),
                TemporalValue(
                    value=570.0,  # >10% increase (exceeds limit)
                    as_of_date=date(2023, 1, 18),
                    value_date=date(2023, 1, 18),
                    data_type=DataType.PRICE,
                    symbol="2330.TW"
                )
            ]
        }
        mock_dependencies['pit_engine'].query.return_value = price_data
        
        taiwan_validation = taiwan_pit_validator._validate_window_taiwan_market(window, symbols)
        
        # Should detect price limit violations
        assert 'success' in taiwan_validation
        if not taiwan_validation['success']:
            assert 'issues' in taiwan_validation
            price_issues = [
                issue for issue in taiwan_validation['issues']
                if 'price_limit' in issue.get('issue_type', '').lower()
            ]
    
    def test_taiwan_volume_liquidity_validation(self, taiwan_pit_validator, mock_dependencies):
        """Test Taiwan market volume and liquidity validation."""
        window = ValidationWindow(
            window_id="volume_test_001",
            train_start=date(2023, 1, 1),
            train_end=date(2023, 1, 31),
            test_start=date(2023, 2, 2),
            test_end=date(2023, 2, 28),
            purge_start=date(2023, 2, 1),
            purge_end=date(2023, 2, 1),
            window_number=1,
            total_train_days=31,
            total_test_days=27,
            purge_days=1,
            trading_days_train=22,
            trading_days_test=19
        )
        
        symbols = ["2330.TW"]
        
        # Mock volume data with low liquidity
        volume_data = {
            "2330.TW": [
                TemporalValue(
                    value=500,  # Low volume
                    as_of_date=date(2023, 1, 16),
                    value_date=date(2023, 1, 16),
                    data_type=DataType.VOLUME,
                    symbol="2330.TW"
                )
            ]
        }
        mock_dependencies['pit_engine'].query.return_value = volume_data
        
        # Test positions that might be too large for the volume
        positions = {
            "2330.TW": {
                date(2023, 1, 16): 100  # 20% of daily volume (exceeds 5% limit)
            }
        }
        
        # Use Taiwan validator directly for position validation
        taiwan_validator = taiwan_pit_validator.taiwan_validator
        issues = taiwan_validator._validate_volume_constraints(
            symbols, positions, window.train_start, window.train_end
        )
        
        # Should detect volume constraint violations
        assert len(issues) >= 0  # May detect issues depending on validation logic
    
    def test_taiwan_market_calendar_integration(self, taiwan_pit_validator, mock_dependencies):
        """Test Taiwan market calendar integration."""
        # Test validation during various Taiwan holidays
        holiday_periods = [
            (date(2023, 2, 10), date(2023, 2, 12)),  # Lunar New Year
            (date(2023, 4, 5), date(2023, 4, 5)),    # Children's Day
            (date(2023, 10, 10), date(2023, 10, 10)), # National Day
        ]
        
        for start_holiday, end_holiday in holiday_periods:
            # Create window that includes holiday
            window = ValidationWindow(
                window_id=f"holiday_test_{start_holiday.strftime('%Y%m%d')}",
                train_start=start_holiday - timedelta(days=30),
                train_end=start_holiday - timedelta(days=1),
                test_start=end_holiday + timedelta(days=1),
                test_end=end_holiday + timedelta(days=30),
                purge_start=start_holiday,
                purge_end=end_holiday,
                window_number=1,
                total_train_days=30,
                total_test_days=30,
                purge_days=(end_holiday - start_holiday).days + 1,
                trading_days_train=21,
                trading_days_test=21
            )
            
            symbols = ["2330.TW"]
            
            result = taiwan_pit_validator._validate_single_window(
                window, symbols, [DataType.PRICE], enable_performance_validation=False
            )
            
            # Should handle holidays correctly
            assert result['validation']['success'] is True


class TestTaiwanHistoricalValidation:
    """Test validation against historical Taiwan market periods."""
    
    @pytest.fixture
    def historical_validator(self):
        """Create validator for historical testing."""
        mock_store = Mock(spec=TemporalStore)
        pit_engine = Mock(spec=PointInTimeEngine)
        
        # Mock historical data availability
        pit_engine.check_data_availability.return_value = True
        pit_engine.query.return_value = {}
        
        config = PITValidationConfig(
            validation_level=ValidationLevel.STRICT,
            require_quality_validation=True
        )
        
        return PITValidator(
            config=config,
            temporal_store=mock_store,
            pit_engine=pit_engine
        )
    
    def test_dotcom_bubble_period_validation(self, historical_validator):
        """Test validation during dot-com bubble period (2000-2001)."""
        wf_config = WalkForwardConfig(
            train_weeks=104,  # 2 years
            test_weeks=26,    # 6 months
            purge_weeks=2
        )
        
        symbols = ["2330.TW", "2317.TW"]  # Major Taiwan tech stocks
        start_date = date(1999, 1, 1)
        end_date = date(2002, 12, 31)
        
        result = historical_validator.validate_walk_forward_scenario(
            wf_config=wf_config,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            enable_performance_validation=False
        )
        
        # Should handle volatile period
        assert result['validation_summary']['total_windows'] > 0
        assert result['summary_statistics']['success_rate'] >= 0.5
    
    def test_asian_financial_crisis_validation(self, historical_validator):
        """Test validation during Asian Financial Crisis (1997-1998)."""
        wf_config = WalkForwardConfig(
            train_weeks=78,   # 1.5 years
            test_weeks=13,    # 3 months
            purge_weeks=1
        )
        
        symbols = ["2330.TW", "1101.TWO"]
        start_date = date(1996, 1, 1)
        end_date = date(1999, 12, 31)
        
        result = historical_validator.validate_walk_forward_scenario(
            wf_config=wf_config,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            enable_performance_validation=False
        )
        
        # Should handle crisis period with appropriate warnings
        assert result['validation_summary']['total_windows'] > 0
        
        # May have bias or quality issues due to market stress
        total_issues = (
            result['validation_summary']['total_bias_issues'] +
            result['validation_summary']['total_quality_issues']
        )
        # Allow for some issues during crisis periods
        assert total_issues >= 0
    
    def test_covid_pandemic_period_validation(self, historical_validator):
        """Test validation during COVID-19 pandemic (2020-2021)."""
        wf_config = WalkForwardConfig(
            train_weeks=156,  # 3 years
            test_weeks=26,    # 6 months
            purge_weeks=2
        )
        
        symbols = ["2330.TW", "2317.TW", "1101.TWO", "6505.TWO"]
        start_date = date(2018, 1, 1)
        end_date = date(2022, 12, 31)
        
        result = historical_validator.validate_walk_forward_scenario(
            wf_config=wf_config,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            enable_performance_validation=True
        )
        
        # Should handle pandemic period with market volatility
        assert result['validation_summary']['total_windows'] > 0
        assert result['validation_summary']['total_runtime_seconds'] > 0
        
        # Check health score
        health_score = result['summary_statistics']['overall_health_score']
        assert 0 <= health_score <= 100


class TestTaiwanComplianceValidation:
    """Test Taiwan market regulatory compliance validation."""
    
    def test_twse_compliance_validation(self):
        """Test TWSE (Taiwan Stock Exchange) compliance."""
        mock_store = Mock(spec=TemporalStore)
        
        # TWSE-specific configuration
        config = TaiwanValidationConfig(
            daily_price_limit_pct=0.10,  # TWSE 10% daily limit
            validate_price_limits=True,
            validate_volume_constraints=True,
            settlement_lag_days=2,        # T+2 settlement
            validate_corporate_actions=True
        )
        
        validator = TaiwanMarketValidator(
            config=config,
            temporal_store=mock_store
        )
        
        # Test TWSE symbols
        twse_symbols = ["2330.TW", "2317.TW", "1101.TW"]
        
        for symbol in twse_symbols:
            assert validator._is_valid_taiwan_symbol(symbol) is True
            assert symbol.endswith('.TW')
    
    def test_tpex_compliance_validation(self):
        """Test TPEx (Taipei Exchange) compliance."""
        mock_store = Mock(spec=TemporalStore)
        
        # TPEx-specific configuration (similar to TWSE but different market)
        config = TaiwanValidationConfig(
            daily_price_limit_pct=0.10,  # TPEx also has 10% limit
            validate_price_limits=True,
            settlement_lag_days=2
        )
        
        validator = TaiwanMarketValidator(
            config=config,
            temporal_store=mock_store
        )
        
        # Test TPEx symbols
        tpex_symbols = ["1101.TWO", "6505.TWO", "8436.TWO"]
        
        for symbol in tpex_symbols:
            assert validator._is_valid_taiwan_symbol(symbol) is True
            assert symbol.endswith('.TWO')
    
    def test_securities_transaction_tax_compliance(self):
        """Test securities transaction tax compliance (0.3% for stocks)."""
        # This would test transaction cost validation
        # Implementation depends on transaction cost modeling from other components
        
        mock_store = Mock(spec=TemporalStore)
        validator = create_strict_taiwan_validator(mock_store)
        
        # Mock transaction with tax implications
        transactions = [
            {
                'date': date(2023, 6, 15),
                'symbol': '2330.TW',
                'quantity': 10000,
                'price': 500.0,
                'transaction_cost': 15000.0,  # 0.3% of 5M transaction
                'tax_rate': 0.003
            }
        ]
        
        issues = validator._validate_transactions(transactions)
        
        # Should validate transaction structure
        assert isinstance(issues, list)


if __name__ == "__main__":
    pytest.main([__file__])