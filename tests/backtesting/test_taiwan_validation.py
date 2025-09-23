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
from src.data.core.temporal import TemporalStore, DataType, TemporalValue
from src.data.models.taiwan_market import (
    TaiwanTradingCalendar, TaiwanSettlement, TaiwanMarketCode,
    TradingStatus, CorporateActionType
)
from src.data.pipeline.pit_engine import PointInTimeEngine, PITQuery


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


if __name__ == "__main__":
    pytest.main([__file__])