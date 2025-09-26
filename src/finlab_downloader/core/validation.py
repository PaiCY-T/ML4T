"""
Enhanced validation utilities for FinLab datasets.

Provides comprehensive validation and data type conversion with business logic.
"""

import re
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass

from .exceptions import ValidationError
from .dataset import DataType, DatasetSpecification


@dataclass
class ValidationRule:
    """A validation rule with custom logic."""

    name: str
    validator: Callable[[Any], bool]
    error_message: str
    error_code: Optional[str] = None


class DataTypeValidator:
    """Enhanced data type validator with business logic."""

    # Common validation patterns
    PATTERNS = {
        'percentage': re.compile(r'^-?\d+(\.\d+)?%?$'),
        'currency': re.compile(r'^-?\d+(\.\d{1,2})?$'),
        'stock_code': re.compile(r'^\d{4,6}$'),
        'date_string': re.compile(r'^\d{4}-\d{2}-\d{2}$'),
        'positive_number': re.compile(r'^\d+(\.\d+)?$'),
    }

    # Taiwan market specific validators
    TAIWAN_STOCK_CODE_RANGE = (1000, 9999)
    MAX_MARKET_CAP = 50_000_000_000_000  # 50 trillion TWD
    MAX_DAILY_VOLUME = 10_000_000_000     # 10 billion shares

    @staticmethod
    def validate_and_convert(value: Any, data_type: DataType,
                           spec: Optional[DatasetSpecification] = None,
                           rules: Optional[List[ValidationRule]] = None) -> Any:
        """
        Validate and convert value with enhanced business logic.

        Args:
            value: Value to validate and convert
            data_type: Target data type
            spec: Dataset specification for additional validation
            rules: Custom validation rules

        Returns:
            Validated and converted value

        Raises:
            ValidationError: If validation fails
        """
        # Handle null values
        if value is None or (isinstance(value, str) and value.strip() == ""):
            nullable = spec.nullable if spec else True
            if not nullable:
                raise ValidationError(
                    f"Value cannot be null",
                    field=spec.name if spec else "unknown",
                    value=value
                )
            return None

        # Pre-processing for string values
        if isinstance(value, str):
            value = value.strip()

        # Type-specific validation and conversion
        try:
            if data_type == DataType.FLOAT:
                converted = DataTypeValidator._convert_to_float(value)
            elif data_type == DataType.INT:
                converted = DataTypeValidator._convert_to_int(value)
            elif data_type == DataType.STRING:
                converted = DataTypeValidator._convert_to_string(value)
            elif data_type == DataType.BOOLEAN:
                converted = DataTypeValidator._convert_to_boolean(value)
            elif data_type == DataType.DATETIME:
                converted = DataTypeValidator._convert_to_datetime(value)
            else:
                converted = value

        except Exception as e:
            raise ValidationError(
                f"Cannot convert '{value}' to {data_type.value}",
                field=spec.name if spec else "unknown",
                value=value,
                expected_type=data_type.value,
                cause=e
            )

        # Apply dataset-specific validation
        if spec:
            converted = spec.validate_value(converted)

        # Apply custom validation rules
        if rules:
            for rule in rules:
                if not rule.validator(converted):
                    raise ValidationError(
                        rule.error_message,
                        field=spec.name if spec else "unknown",
                        value=converted,
                        error_code=rule.error_code
                    )

        return converted

    @staticmethod
    def _convert_to_float(value: Any) -> float:
        """Convert value to float with enhanced handling."""
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Handle percentage strings
            if value.endswith('%'):
                return float(value[:-1]) / 100.0

            # Handle currency strings (remove commas)
            value = value.replace(',', '')

            # Handle special cases
            if value.lower() in ('inf', 'infinity'):
                return float('inf')
            if value.lower() in ('-inf', '-infinity'):
                return float('-inf')
            if value.lower() == 'nan':
                return float('nan')

        return float(value)

    @staticmethod
    def _convert_to_int(value: Any) -> int:
        """Convert value to int with enhanced handling."""
        if isinstance(value, int):
            return value

        if isinstance(value, float):
            # Handle float to int conversion
            if value.is_integer():
                return int(value)
            else:
                raise ValueError(f"Cannot convert non-integer float {value} to int")

        if isinstance(value, str):
            # Handle currency strings (remove commas)
            value = value.replace(',', '')

            # Handle percentage strings
            if value.endswith('%'):
                float_val = float(value[:-1]) / 100.0
                if float_val.is_integer():
                    return int(float_val)
                else:
                    raise ValueError(f"Percentage {value} is not an integer")

            # Try direct conversion
            return int(float(value))  # Use float as intermediate to handle "1.0"

        return int(value)

    @staticmethod
    def _convert_to_string(value: Any) -> str:
        """Convert value to string with enhanced handling."""
        if isinstance(value, str):
            return value

        if isinstance(value, (int, float)):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, date):
            return value.isoformat()

        return str(value)

    @staticmethod
    def _convert_to_boolean(value: Any) -> bool:
        """Convert value to boolean with enhanced handling."""
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ('true', '1', 'yes', 'on', 'y', 't'):
                return True
            if value_lower in ('false', '0', 'no', 'off', 'n', 'f'):
                return False
            raise ValueError(f"Cannot convert string '{value}' to boolean")

        if isinstance(value, (int, float)):
            return bool(value)

        return bool(value)

    @staticmethod
    def _convert_to_datetime(value: Any) -> datetime:
        """Convert value to datetime with enhanced handling."""
        if isinstance(value, datetime):
            return value

        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())

        if isinstance(value, str):
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d',
                '%Y/%m/%d %H:%M:%S',
                '%Y/%m/%d %H:%M',
                '%Y/%m/%d',
                '%d/%m/%Y %H:%M:%S',
                '%d/%m/%Y %H:%M',
                '%d/%m/%Y',
                '%d-%m-%Y %H:%M:%S',
                '%d-%m-%Y %H:%M',
                '%d-%m-%Y',
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue

            raise ValueError(f"Cannot parse datetime from '{value}'")

        if isinstance(value, (int, float)):
            # Assume Unix timestamp
            return datetime.fromtimestamp(value)

        raise ValueError(f"Cannot convert {type(value)} to datetime")


class BusinessLogicValidator:
    """Validator with Taiwan market specific business logic."""

    @staticmethod
    def create_taiwan_stock_validators() -> List[ValidationRule]:
        """Create validation rules for Taiwan stock data."""
        return [
            ValidationRule(
                name="valid_stock_code",
                validator=lambda x: (isinstance(x, str) and
                                   BusinessLogicValidator.PATTERNS['stock_code'].match(x) and
                                   BusinessLogicValidator.TAIWAN_STOCK_CODE_RANGE[0] <= int(x) <=
                                   BusinessLogicValidator.TAIWAN_STOCK_CODE_RANGE[1]),
                error_message="Invalid Taiwan stock code",
                error_code="INVALID_STOCK_CODE"
            ),
            ValidationRule(
                name="positive_price",
                validator=lambda x: isinstance(x, (int, float)) and x > 0,
                error_message="Stock price must be positive",
                error_code="NEGATIVE_PRICE"
            ),
            ValidationRule(
                name="reasonable_market_cap",
                validator=lambda x: (isinstance(x, (int, float)) and
                                   0 < x <= BusinessLogicValidator.MAX_MARKET_CAP),
                error_message=f"Market cap exceeds reasonable limit ({BusinessLogicValidator.MAX_MARKET_CAP:,})",
                error_code="UNREASONABLE_MARKET_CAP"
            ),
            ValidationRule(
                name="reasonable_volume",
                validator=lambda x: (isinstance(x, (int, float)) and
                                   0 <= x <= BusinessLogicValidator.MAX_DAILY_VOLUME),
                error_message=f"Daily volume exceeds reasonable limit ({BusinessLogicValidator.MAX_DAILY_VOLUME:,})",
                error_code="UNREASONABLE_VOLUME"
            )
        ]

    @staticmethod
    def create_financial_validators() -> List[ValidationRule]:
        """Create validation rules for financial statement data."""
        return [
            ValidationRule(
                name="reasonable_revenue",
                validator=lambda x: isinstance(x, (int, float)) and x >= 0,
                error_message="Revenue cannot be negative",
                error_code="NEGATIVE_REVENUE"
            ),
            ValidationRule(
                name="percentage_range",
                validator=lambda x: isinstance(x, (int, float)) and -100 <= x <= 1000,
                error_message="Percentage value outside reasonable range (-100% to 1000%)",
                error_code="UNREASONABLE_PERCENTAGE"
            ),
            ValidationRule(
                name="ratio_range",
                validator=lambda x: isinstance(x, (int, float)) and -1000 <= x <= 1000,
                error_message="Ratio value outside reasonable range (-1000 to 1000)",
                error_code="UNREASONABLE_RATIO"
            )
        ]

    # Copy patterns from DataTypeValidator for backward compatibility
    PATTERNS = DataTypeValidator.PATTERNS
    TAIWAN_STOCK_CODE_RANGE = DataTypeValidator.TAIWAN_STOCK_CODE_RANGE
    MAX_MARKET_CAP = DataTypeValidator.MAX_MARKET_CAP
    MAX_DAILY_VOLUME = DataTypeValidator.MAX_DAILY_VOLUME


class ValidationContext:
    """Context for validation operations with custom rules."""

    def __init__(self):
        """Initialize validation context."""
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.type_mappings: Dict[str, DataType] = {}

    def add_rule(self, dataset_name: str, rule: ValidationRule) -> None:
        """Add a validation rule for a specific dataset."""
        if dataset_name not in self.rules:
            self.rules[dataset_name] = []
        self.rules[dataset_name].append(rule)

    def add_rules(self, dataset_name: str, rules: List[ValidationRule]) -> None:
        """Add multiple validation rules for a specific dataset."""
        for rule in rules:
            self.add_rule(dataset_name, rule)

    def get_rules(self, dataset_name: str) -> List[ValidationRule]:
        """Get validation rules for a specific dataset."""
        return self.rules.get(dataset_name, [])

    def set_type_mapping(self, pattern: str, data_type: DataType) -> None:
        """Set data type mapping for datasets matching a pattern."""
        self.type_mappings[pattern] = data_type

    def get_mapped_type(self, dataset_name: str) -> Optional[DataType]:
        """Get mapped data type for a dataset."""
        for pattern, data_type in self.type_mappings.items():
            if re.search(pattern, dataset_name, re.IGNORECASE):
                return data_type
        return None

    def validate(self, dataset_name: str, value: Any,
                spec: DatasetSpecification) -> Any:
        """
        Validate a value in this context.

        Args:
            dataset_name: Name of the dataset
            value: Value to validate
            spec: Dataset specification

        Returns:
            Validated and converted value
        """
        # Get context-specific rules
        rules = self.get_rules(dataset_name)

        # Use mapped type if available
        data_type = self.get_mapped_type(dataset_name) or spec.data_type

        return DataTypeValidator.validate_and_convert(
            value, data_type, spec, rules
        )


# Pre-defined validation contexts
def create_taiwan_market_context() -> ValidationContext:
    """Create validation context for Taiwan market data."""
    context = ValidationContext()

    # Stock code validation
    context.add_rules("stock_code", BusinessLogicValidator.create_taiwan_stock_validators()[:1])

    # Price data validation
    price_patterns = ["price", "high", "low", "open", "close", "adj_"]
    for pattern in price_patterns:
        context.add_rules(pattern, [BusinessLogicValidator.create_taiwan_stock_validators()[1]])

    # Volume validation
    context.add_rules("volume", [BusinessLogicValidator.create_taiwan_stock_validators()[3]])

    # Financial data validation
    financial_rules = BusinessLogicValidator.create_financial_validators()
    revenue_patterns = ["revenue", "營收", "營業收入"]
    for pattern in revenue_patterns:
        context.add_rules(pattern, [financial_rules[0]])

    # Percentage data
    percentage_patterns = ["rate", "ratio", "growth", "增減", "比率"]
    for pattern in percentage_patterns:
        context.add_rules(pattern, [financial_rules[1]])

    return context


def create_strict_validation_context() -> ValidationContext:
    """Create strict validation context with comprehensive rules."""
    context = create_taiwan_market_context()

    # Add all business logic rules
    all_stock_rules = BusinessLogicValidator.create_taiwan_stock_validators()
    all_financial_rules = BusinessLogicValidator.create_financial_validators()

    # Apply to all datasets
    context.add_rules(".*", all_stock_rules + all_financial_rules)

    return context