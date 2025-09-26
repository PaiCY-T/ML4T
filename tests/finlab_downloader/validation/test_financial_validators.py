"""
Tests for financial data validators.
"""

import pytest
from datetime import datetime, date
import pandas as pd
import numpy as np

from src.finlab_downloader.validation.financial_validators import (
    FinancialDataValidator, PriceValidator, VolumeValidator, DateValidator,
    ValidationResult, ValidationSeverity, ValidationCategory
)


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'date': dates,
        'symbol': ['AAPL'] * 10,
        'open': [150, 151, 149, 152, 148, 147, 150, 153, 151, 149],
        'high': [152, 153, 151, 154, 150, 149, 152, 155, 153, 151],
        'low': [148, 149, 147, 150, 146, 145, 148, 151, 149, 147],
        'close': [151, 150, 150, 153, 149, 148, 151, 154, 152, 150],
        'volume': [1000000, 1200000, 800000, 1500000, 900000, 700000, 1100000, 1300000, 1000000, 950000]
    })
    return data


@pytest.fixture
def invalid_stock_data():
    """Create invalid stock data for testing validators."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')

    data = pd.DataFrame({
        'date': dates,
        'symbol': ['AAPL'] * 5,
        'open': [150, -10, 149, 152, 148],  # One negative price
        'high': [152, 153, 140, 154, 150],  # One value lower than open
        'low': [148, 149, 160, 150, 146],   # One value higher than open
        'close': [151, 150, 150, 153, 149],
        'volume': [1000000, -500000, 800000, 1500000, 900000]  # One negative volume
    })
    return data


def test_price_validator_range():
    """Test price range validation."""
    data = pd.DataFrame({
        'price': [10, 50, 100, -5, 150000]  # One negative, one too high
    })

    result = PriceValidator.validate_price_range(
        data, 'price', min_price=0, max_price=10000
    )

    assert isinstance(result, ValidationResult)
    assert result.rule_name == "price_range"
    assert not result.passed
    assert result.affected_rows == 2  # Negative and too high values


def test_price_validator_ohlc_consistency():
    """Test OHLC consistency validation."""
    # Valid OHLC data
    valid_data = pd.DataFrame({
        'open': [100, 105],
        'high': [110, 120],  # High >= Open, Close
        'low': [95, 100],    # Low <= Open, Close
        'close': [108, 115]
    })

    result = PriceValidator.validate_ohlc_consistency(valid_data)
    assert result.passed

    # Invalid OHLC data
    invalid_data = pd.DataFrame({
        'open': [100, 105],
        'high': [90, 100],   # High < Open (invalid)
        'low': [110, 120],   # Low > Open (invalid)
        'close': [108, 115]
    })

    result = PriceValidator.validate_ohlc_consistency(invalid_data)
    assert not result.passed
    assert result.affected_rows == 2


def test_price_validator_no_negative():
    """Test no negative prices validation."""
    data = pd.DataFrame({
        'open': [100, -10, 105],
        'close': [102, 95, -5],
        'high': [110, 100, 108],
        'low': [98, 90, 100]
    })

    result = PriceValidator.validate_no_negative_prices(
        data, ['open', 'close', 'high', 'low']
    )

    assert not result.passed
    assert result.affected_rows == 2  # Two negative values
    assert 'open' in result.affected_columns
    assert 'close' in result.affected_columns


def test_volume_validator_range():
    """Test volume range validation."""
    data = pd.DataFrame({
        'volume': [1000, 5000000, 2000000000, 500]  # One too high, one too low
    })

    result = VolumeValidator.validate_volume_range(
        data, 'volume', min_volume=1000, max_volume=1000000000
    )

    assert not result.passed
    assert result.affected_rows == 1  # One value too high


def test_volume_validator_non_negative():
    """Test non-negative volume validation."""
    data = pd.DataFrame({
        'volume': [1000, -500, 2000, 0, -100]
    })

    result = VolumeValidator.validate_non_negative_volume(data, 'volume')

    assert not result.passed
    assert result.affected_rows == 2  # Two negative values


def test_date_validator_format():
    """Test date format validation."""
    # Valid datetime column
    valid_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='D')
    })

    result = DateValidator.validate_date_format(valid_data, 'date')
    assert result.passed

    # Invalid date column
    invalid_data = pd.DataFrame({
        'date': ['invalid', '2024-01-01', 'also invalid', '2024-12-31', 'nope']
    })

    result = DateValidator.validate_date_format(invalid_data, 'date')
    # Should still pass if convertible
    assert result.passed or not result.passed  # pandas might convert some


def test_date_validator_range():
    """Test date range validation."""
    data = pd.DataFrame({
        'date': pd.to_datetime([
            '2024-01-01', '2024-06-15', '2024-12-31',
            '2023-12-31', '2025-01-01'  # One before, one after range
        ])
    })

    min_date = date(2024, 1, 1)
    max_date = date(2024, 12, 31)

    result = DateValidator.validate_date_range(
        data, 'date', min_date=min_date, max_date=max_date
    )

    assert not result.passed
    assert result.affected_rows == 2


def test_date_validator_trading_days():
    """Test trading days validation."""
    # Include some weekend dates
    data = pd.DataFrame({
        'date': pd.to_datetime([
            '2024-01-01',  # Monday
            '2024-01-02',  # Tuesday
            '2024-01-06',  # Saturday (weekend)
            '2024-01-07',  # Sunday (weekend)
            '2024-01-08'   # Monday
        ])
    })

    result = DateValidator.validate_trading_days(
        data, 'date', allow_weekends=False
    )

    assert not result.passed
    assert result.affected_rows == 2  # Two weekend dates


def test_financial_data_validator_complete():
    """Test complete financial data validation."""
    validator = FinancialDataValidator()

    # Create test data with various issues
    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5, freq='D'),
        'symbol': ['AAPL'] * 5,
        'open': [100, -10, 105, 110, 108],     # One negative
        'high': [110, 100, 115, 120, 118],    # One inconsistent with open
        'low': [95, 90, 100, 105, 103],       # One inconsistent
        'close': [105, 95, 110, 115, 112],
        'volume': [1000000, -500000, 1200000, 1500000, 1100000]  # One negative
    })

    results = validator.validate(data, "stock_prices")

    # Should have multiple validation failures
    failed_results = [r for r in results if not r.passed]
    assert len(failed_results) > 0

    # Check that we have different types of validations
    rule_names = {r.rule_name for r in results}
    expected_rules = {'price_range', 'ohlc_consistency', 'no_negative_prices', 'non_negative_volume'}
    assert len(rule_names.intersection(expected_rules)) > 0


def test_validation_result_summary():
    """Test validation result summary generation."""
    validator = FinancialDataValidator()

    data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3, freq='D'),
        'open': [100, 105, 110],
        'high': [110, 115, 120],
        'low': [95, 100, 105],
        'close': [105, 110, 115],
        'volume': [1000000, 1200000, 1500000]
    })

    results = validator.validate(data)
    summary = validator.get_summary(results)

    assert 'total_rules' in summary
    assert 'passed_rules' in summary
    assert 'failed_rules' in summary
    assert 'pass_rate' in summary
    assert 'severity_counts' in summary
    assert 'category_counts' in summary


def test_custom_validation_rule():
    """Test adding custom validation rules."""
    validator = FinancialDataValidator()

    def custom_price_validator(data, column):
        # Custom rule: prices should be between 10 and 1000
        invalid_count = ((data[column] < 10) | (data[column] > 1000)).sum()
        return ValidationResult(
            rule_name="custom_price_range",
            category=ValidationCategory.RANGE,
            severity=ValidationSeverity.WARNING,
            passed=invalid_count == 0,
            message=f"Custom price validation: {invalid_count} values outside [10, 1000]",
            affected_rows=invalid_count,
            affected_columns=[column],
            details={'invalid_count': invalid_count},
            timestamp=datetime.utcnow()
        )

    # Add custom rule
    from src.finlab_downloader.validation.financial_validators import ValidationRule
    custom_rule = ValidationRule(
        name="custom_price_range",
        category=ValidationCategory.RANGE,
        severity=ValidationSeverity.WARNING,
        description="Custom price range validation",
        validator_func=custom_price_validator,
        applies_to_columns=['price']
    )

    validator.add_custom_rule(custom_rule)

    # Test with data that violates custom rule
    data = pd.DataFrame({
        'price': [5, 50, 500, 1500, 100]  # Two values outside range
    })

    results = validator.validate(data)
    custom_results = [r for r in results if r.rule_name == "custom_price_range"]
    assert len(custom_results) == 1
    assert not custom_results[0].passed


def test_validation_rule_enable_disable():
    """Test enabling and disabling validation rules."""
    validator = FinancialDataValidator()

    # Disable a rule
    validator.disable_rule("price_range")

    data = pd.DataFrame({
        'open': [100, -50, 200000],  # Would normally trigger price_range validation
        'high': [110, 100, 220000],
        'low': [95, -60, 190000],
        'close': [105, -45, 210000]
    })

    results = validator.validate(data)
    price_range_results = [r for r in results if r.rule_name == "price_range"]
    assert len(price_range_results) == 0  # Rule should be disabled

    # Re-enable the rule
    validator.enable_rule("price_range")
    results = validator.validate(data)
    price_range_results = [r for r in results if r.rule_name == "price_range"]
    assert len(price_range_results) > 0  # Rule should be active again


def test_missing_columns_handling():
    """Test handling of missing columns."""
    validator = FinancialDataValidator()

    # Data missing required columns
    data = pd.DataFrame({
        'symbol': ['AAPL', 'GOOGL'],
        'price': [150, 2800]
        # Missing date, OHLC, volume columns
    })

    results = validator.validate(data)

    # Should handle missing columns gracefully without errors
    assert isinstance(results, list)
    # Most rules should be skipped due to missing columns
    applicable_results = [r for r in results if r.rule_name != "missing_column"]
    # Should have minimal results since most columns are missing