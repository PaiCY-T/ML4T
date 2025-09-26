"""
Financial data validators with comprehensive rules for market data validation.

Provides specialized validation rules for financial datasets including
price validation, volume checks, and data integrity verification.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import re

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation rules."""
    DATA_TYPE = "data_type"
    RANGE = "range"
    FORMAT = "format"
    BUSINESS_LOGIC = "business_logic"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_name: str
    category: ValidationCategory
    severity: ValidationSeverity
    passed: bool
    message: str
    affected_rows: int
    affected_columns: List[str]
    details: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_name': self.rule_name,
            'category': self.category.value,
            'severity': self.severity.value,
            'passed': self.passed,
            'message': self.message,
            'affected_rows': self.affected_rows,
            'affected_columns': self.affected_columns,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ValidationRule:
    """Definition of a validation rule."""
    name: str
    category: ValidationCategory
    severity: ValidationSeverity
    description: str
    validator_func: Callable
    applies_to_columns: List[str]
    enabled: bool = True
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class PriceValidator:
    """Validator for price-related data."""

    @staticmethod
    def validate_price_range(data: pd.DataFrame,
                           column: str,
                           min_price: float = 0.0,
                           max_price: float = 100000.0) -> ValidationResult:
        """
        Validate price values are within reasonable range.

        Args:
            data: DataFrame to validate
            column: Price column name
            min_price: Minimum valid price
            max_price: Maximum valid price

        Returns:
            ValidationResult
        """
        if column not in data.columns:
            return ValidationResult(
                rule_name="price_range",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Column '{column}' not found in data",
                affected_rows=0,
                affected_columns=[column],
                details={'min_price': min_price, 'max_price': max_price},
                timestamp=datetime.utcnow()
            )

        # Check for invalid prices
        invalid_mask = (data[column] < min_price) | (data[column] > max_price)
        invalid_count = invalid_mask.sum()

        passed = invalid_count == 0
        severity = ValidationSeverity.WARNING if invalid_count > 0 else ValidationSeverity.INFO

        return ValidationResult(
            rule_name="price_range",
            category=ValidationCategory.RANGE,
            severity=severity,
            passed=passed,
            message=f"Found {invalid_count} price values outside range [{min_price}, {max_price}]",
            affected_rows=invalid_count,
            affected_columns=[column],
            details={
                'min_price': min_price,
                'max_price': max_price,
                'invalid_values': data[invalid_mask][column].tolist()[:10] if invalid_count > 0 else []
            },
            timestamp=datetime.utcnow()
        )

    @staticmethod
    def validate_ohlc_consistency(data: pd.DataFrame,
                                 open_col: str = 'open',
                                 high_col: str = 'high',
                                 low_col: str = 'low',
                                 close_col: str = 'close') -> ValidationResult:
        """
        Validate OHLC data consistency (high >= open, close; low <= open, close).

        Args:
            data: DataFrame with OHLC data
            open_col: Open price column
            high_col: High price column
            low_col: Low price column
            close_col: Close price column

        Returns:
            ValidationResult
        """
        required_cols = [open_col, high_col, low_col, close_col]
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            return ValidationResult(
                rule_name="ohlc_consistency",
                category=ValidationCategory.BUSINESS_LOGIC,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Missing OHLC columns: {missing_cols}",
                affected_rows=0,
                affected_columns=missing_cols,
                details={'missing_columns': missing_cols},
                timestamp=datetime.utcnow()
            )

        # Check OHLC consistency
        inconsistent_mask = (
            (data[high_col] < data[open_col]) |
            (data[high_col] < data[close_col]) |
            (data[low_col] > data[open_col]) |
            (data[low_col] > data[close_col])
        )

        inconsistent_count = inconsistent_mask.sum()
        passed = inconsistent_count == 0

        return ValidationResult(
            rule_name="ohlc_consistency",
            category=ValidationCategory.BUSINESS_LOGIC,
            severity=ValidationSeverity.ERROR if inconsistent_count > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=f"Found {inconsistent_count} OHLC inconsistencies",
            affected_rows=inconsistent_count,
            affected_columns=required_cols,
            details={
                'inconsistent_indices': data[inconsistent_mask].index.tolist()[:10] if inconsistent_count > 0 else []
            },
            timestamp=datetime.utcnow()
        )

    @staticmethod
    def validate_no_negative_prices(data: pd.DataFrame, columns: List[str]) -> ValidationResult:
        """
        Validate that price columns don't contain negative values.

        Args:
            data: DataFrame to validate
            columns: Price columns to check

        Returns:
            ValidationResult
        """
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            return ValidationResult(
                rule_name="no_negative_prices",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Missing columns: {missing_cols}",
                affected_rows=0,
                affected_columns=missing_cols,
                details={'missing_columns': missing_cols},
                timestamp=datetime.utcnow()
            )

        negative_count = 0
        negative_columns = []

        for col in columns:
            negative_mask = data[col] < 0
            col_negative_count = negative_mask.sum()
            if col_negative_count > 0:
                negative_count += col_negative_count
                negative_columns.append(col)

        passed = negative_count == 0

        return ValidationResult(
            rule_name="no_negative_prices",
            category=ValidationCategory.RANGE,
            severity=ValidationSeverity.ERROR if negative_count > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=f"Found {negative_count} negative price values",
            affected_rows=negative_count,
            affected_columns=negative_columns,
            details={'negative_columns': negative_columns},
            timestamp=datetime.utcnow()
        )


class VolumeValidator:
    """Validator for volume-related data."""

    @staticmethod
    def validate_volume_range(data: pd.DataFrame,
                            column: str,
                            min_volume: int = 0,
                            max_volume: int = 1000000000) -> ValidationResult:
        """
        Validate volume values are within reasonable range.

        Args:
            data: DataFrame to validate
            column: Volume column name
            min_volume: Minimum valid volume
            max_volume: Maximum valid volume

        Returns:
            ValidationResult
        """
        if column not in data.columns:
            return ValidationResult(
                rule_name="volume_range",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Column '{column}' not found",
                affected_rows=0,
                affected_columns=[column],
                details={'min_volume': min_volume, 'max_volume': max_volume},
                timestamp=datetime.utcnow()
            )

        invalid_mask = (data[column] < min_volume) | (data[column] > max_volume)
        invalid_count = invalid_mask.sum()

        passed = invalid_count == 0

        return ValidationResult(
            rule_name="volume_range",
            category=ValidationCategory.RANGE,
            severity=ValidationSeverity.WARNING if invalid_count > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=f"Found {invalid_count} volume values outside range [{min_volume}, {max_volume}]",
            affected_rows=invalid_count,
            affected_columns=[column],
            details={
                'min_volume': min_volume,
                'max_volume': max_volume,
                'outlier_count': invalid_count
            },
            timestamp=datetime.utcnow()
        )

    @staticmethod
    def validate_non_negative_volume(data: pd.DataFrame, column: str) -> ValidationResult:
        """
        Validate volume is non-negative.

        Args:
            data: DataFrame to validate
            column: Volume column name

        Returns:
            ValidationResult
        """
        if column not in data.columns:
            return ValidationResult(
                rule_name="non_negative_volume",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Column '{column}' not found",
                affected_rows=0,
                affected_columns=[column],
                details={},
                timestamp=datetime.utcnow()
            )

        negative_mask = data[column] < 0
        negative_count = negative_mask.sum()

        passed = negative_count == 0

        return ValidationResult(
            rule_name="non_negative_volume",
            category=ValidationCategory.RANGE,
            severity=ValidationSeverity.ERROR if negative_count > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=f"Found {negative_count} negative volume values",
            affected_rows=negative_count,
            affected_columns=[column],
            details={'negative_count': negative_count},
            timestamp=datetime.utcnow()
        )


class DateValidator:
    """Validator for date-related data."""

    @staticmethod
    def validate_date_format(data: pd.DataFrame, column: str) -> ValidationResult:
        """
        Validate date column format.

        Args:
            data: DataFrame to validate
            column: Date column name

        Returns:
            ValidationResult
        """
        if column not in data.columns:
            return ValidationResult(
                rule_name="date_format",
                category=ValidationCategory.FORMAT,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Column '{column}' not found",
                affected_rows=0,
                affected_columns=[column],
                details={},
                timestamp=datetime.utcnow()
            )

        # Check if column is datetime
        is_datetime = pd.api.types.is_datetime64_any_dtype(data[column])

        if not is_datetime:
            # Try to convert to datetime
            try:
                pd.to_datetime(data[column])
                convertible = True
            except (ValueError, TypeError):
                convertible = False
        else:
            convertible = True

        passed = is_datetime or convertible

        return ValidationResult(
            rule_name="date_format",
            category=ValidationCategory.FORMAT,
            severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
            passed=passed,
            message=f"Date column format validation: {'passed' if passed else 'failed'}",
            affected_rows=len(data) if not passed else 0,
            affected_columns=[column],
            details={
                'is_datetime': is_datetime,
                'convertible': convertible,
                'current_dtype': str(data[column].dtype)
            },
            timestamp=datetime.utcnow()
        )

    @staticmethod
    def validate_date_range(data: pd.DataFrame,
                          column: str,
                          min_date: Optional[date] = None,
                          max_date: Optional[date] = None) -> ValidationResult:
        """
        Validate dates are within specified range.

        Args:
            data: DataFrame to validate
            column: Date column name
            min_date: Minimum valid date
            max_date: Maximum valid date

        Returns:
            ValidationResult
        """
        if column not in data.columns:
            return ValidationResult(
                rule_name="date_range",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Column '{column}' not found",
                affected_rows=0,
                affected_columns=[column],
                details={},
                timestamp=datetime.utcnow()
            )

        # Convert to datetime if needed
        try:
            dates = pd.to_datetime(data[column])
        except (ValueError, TypeError):
            return ValidationResult(
                rule_name="date_range",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Cannot convert column '{column}' to datetime",
                affected_rows=len(data),
                affected_columns=[column],
                details={},
                timestamp=datetime.utcnow()
            )

        # Apply range validation
        invalid_mask = pd.Series(False, index=data.index)

        if min_date:
            invalid_mask |= dates < pd.Timestamp(min_date)

        if max_date:
            invalid_mask |= dates > pd.Timestamp(max_date)

        invalid_count = invalid_mask.sum()
        passed = invalid_count == 0

        return ValidationResult(
            rule_name="date_range",
            category=ValidationCategory.RANGE,
            severity=ValidationSeverity.WARNING if invalid_count > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=f"Found {invalid_count} dates outside valid range",
            affected_rows=invalid_count,
            affected_columns=[column],
            details={
                'min_date': min_date.isoformat() if min_date else None,
                'max_date': max_date.isoformat() if max_date else None,
                'invalid_count': invalid_count
            },
            timestamp=datetime.utcnow()
        )

    @staticmethod
    def validate_trading_days(data: pd.DataFrame,
                            column: str,
                            allow_weekends: bool = False,
                            allow_holidays: bool = False) -> ValidationResult:
        """
        Validate dates are valid trading days.

        Args:
            data: DataFrame to validate
            column: Date column name
            allow_weekends: Whether to allow weekend dates
            allow_holidays: Whether to allow holiday dates

        Returns:
            ValidationResult
        """
        if column not in data.columns:
            return ValidationResult(
                rule_name="trading_days",
                category=ValidationCategory.BUSINESS_LOGIC,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Column '{column}' not found",
                affected_rows=0,
                affected_columns=[column],
                details={},
                timestamp=datetime.utcnow()
            )

        try:
            dates = pd.to_datetime(data[column])
        except (ValueError, TypeError):
            return ValidationResult(
                rule_name="trading_days",
                category=ValidationCategory.BUSINESS_LOGIC,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Cannot convert column '{column}' to datetime",
                affected_rows=len(data),
                affected_columns=[column],
                details={},
                timestamp=datetime.utcnow()
            )

        invalid_dates = 0

        if not allow_weekends:
            # Check for weekends (Saturday=5, Sunday=6)
            weekend_mask = dates.dt.dayofweek.isin([5, 6])
            invalid_dates += weekend_mask.sum()

        passed = invalid_dates == 0

        return ValidationResult(
            rule_name="trading_days",
            category=ValidationCategory.BUSINESS_LOGIC,
            severity=ValidationSeverity.WARNING if invalid_dates > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=f"Found {invalid_dates} non-trading days",
            affected_rows=invalid_dates,
            affected_columns=[column],
            details={
                'allow_weekends': allow_weekends,
                'allow_holidays': allow_holidays,
                'weekend_count': invalid_dates
            },
            timestamp=datetime.utcnow()
        )


class FinancialDataValidator:
    """
    Comprehensive financial data validator.

    Combines multiple validation rules for complete financial data validation.
    """

    def __init__(self):
        """Initialize validator with default rules."""
        self.rules = self._create_default_rules()
        self.results_history = []

    def _create_default_rules(self) -> List[ValidationRule]:
        """Create default validation rules."""
        return [
            ValidationRule(
                name="price_range",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.WARNING,
                description="Validate price values are within reasonable range",
                validator_func=PriceValidator.validate_price_range,
                applies_to_columns=['open', 'high', 'low', 'close', 'price']
            ),
            ValidationRule(
                name="ohlc_consistency",
                category=ValidationCategory.BUSINESS_LOGIC,
                severity=ValidationSeverity.ERROR,
                description="Validate OHLC data consistency",
                validator_func=PriceValidator.validate_ohlc_consistency,
                applies_to_columns=['open', 'high', 'low', 'close']
            ),
            ValidationRule(
                name="no_negative_prices",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.ERROR,
                description="Validate no negative price values",
                validator_func=PriceValidator.validate_no_negative_prices,
                applies_to_columns=['open', 'high', 'low', 'close', 'price']
            ),
            ValidationRule(
                name="volume_range",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.WARNING,
                description="Validate volume values are within reasonable range",
                validator_func=VolumeValidator.validate_volume_range,
                applies_to_columns=['volume']
            ),
            ValidationRule(
                name="non_negative_volume",
                category=ValidationCategory.RANGE,
                severity=ValidationSeverity.ERROR,
                description="Validate volume is non-negative",
                validator_func=VolumeValidator.validate_non_negative_volume,
                applies_to_columns=['volume']
            ),
            ValidationRule(
                name="date_format",
                category=ValidationCategory.FORMAT,
                severity=ValidationSeverity.ERROR,
                description="Validate date column format",
                validator_func=DateValidator.validate_date_format,
                applies_to_columns=['date', 'datetime', 'timestamp']
            ),
            ValidationRule(
                name="trading_days",
                category=ValidationCategory.BUSINESS_LOGIC,
                severity=ValidationSeverity.WARNING,
                description="Validate dates are valid trading days",
                validator_func=DateValidator.validate_trading_days,
                applies_to_columns=['date', 'datetime', 'timestamp']
            )
        ]

    def validate(self, data: pd.DataFrame, dataset_type: str = "generic") -> List[ValidationResult]:
        """
        Validate DataFrame against all applicable rules.

        Args:
            data: DataFrame to validate
            dataset_type: Type of dataset for context-specific validation

        Returns:
            List of validation results
        """
        results = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Check if rule applies to this data
            applicable_columns = [col for col in rule.applies_to_columns if col in data.columns]
            if not applicable_columns:
                continue

            try:
                # Execute validation rule
                if rule.name == "price_range":
                    for col in applicable_columns:
                        result = rule.validator_func(data, col, **rule.parameters)
                        results.append(result)

                elif rule.name == "ohlc_consistency":
                    if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                        result = rule.validator_func(data)
                        results.append(result)

                elif rule.name == "no_negative_prices":
                    result = rule.validator_func(data, applicable_columns)
                    results.append(result)

                elif rule.name == "volume_range":
                    for col in applicable_columns:
                        result = rule.validator_func(data, col, **rule.parameters)
                        results.append(result)

                elif rule.name == "non_negative_volume":
                    for col in applicable_columns:
                        result = rule.validator_func(data, col)
                        results.append(result)

                elif rule.name == "date_format":
                    for col in applicable_columns:
                        result = rule.validator_func(data, col)
                        results.append(result)

                elif rule.name == "trading_days":
                    for col in applicable_columns:
                        result = rule.validator_func(data, col, **rule.parameters)
                        results.append(result)

            except Exception as e:
                logger.error(f"Error executing validation rule '{rule.name}': {e}")
                error_result = ValidationResult(
                    rule_name=rule.name,
                    category=rule.category,
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    message=f"Validation rule failed with error: {e}",
                    affected_rows=0,
                    affected_columns=[],
                    details={'error': str(e)},
                    timestamp=datetime.utcnow()
                )
                results.append(error_result)

        self.results_history.extend(results)
        return results

    def add_custom_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.rules.append(rule)

    def disable_rule(self, rule_name: str) -> None:
        """Disable a validation rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                break

    def enable_rule(self, rule_name: str) -> None:
        """Enable a validation rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                break

    def get_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Get validation summary.

        Args:
            results: Validation results

        Returns:
            Summary dictionary
        """
        total_rules = len(results)
        passed_rules = sum(1 for r in results if r.passed)
        failed_rules = total_rules - passed_rules

        severity_counts = {
            severity.value: sum(1 for r in results if r.severity == severity)
            for severity in ValidationSeverity
        }

        category_counts = {
            category.value: sum(1 for r in results if r.category == category)
            for category in ValidationCategory
        }

        return {
            'total_rules': total_rules,
            'passed_rules': passed_rules,
            'failed_rules': failed_rules,
            'pass_rate': passed_rules / total_rules if total_rules > 0 else 0,
            'severity_counts': severity_counts,
            'category_counts': category_counts,
            'critical_failures': [r.to_dict() for r in results if r.severity == ValidationSeverity.CRITICAL and not r.passed],
            'timestamp': datetime.utcnow().isoformat()
        }