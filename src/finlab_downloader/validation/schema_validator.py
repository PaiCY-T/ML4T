"""
Schema validation for financial data with pandas DataFrame schema enforcement.

Provides comprehensive schema validation including data types, constraints,
and financial data specific rules.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from .financial_validators import ValidationResult, ValidationSeverity, ValidationCategory

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for validation."""
    INTEGER = "int"
    FLOAT = "float"
    STRING = "str"
    DATETIME = "datetime"
    DATE = "date"
    BOOLEAN = "bool"
    CATEGORY = "category"


@dataclass
class FieldConstraint:
    """Constraints for a field."""
    nullable: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None
    unique: bool = False


@dataclass
class FieldSchema:
    """Schema definition for a field."""
    name: str
    data_type: DataType
    constraints: FieldConstraint
    description: Optional[str] = None


@dataclass
class FinancialDataSchema:
    """Complete schema for financial datasets."""
    name: str
    version: str
    fields: List[FieldSchema]
    required_fields: List[str]
    index_field: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class SchemaValidator:
    """
    Comprehensive schema validator for financial data.

    Validates DataFrame structure, data types, constraints, and
    financial data specific requirements.
    """

    def __init__(self):
        """Initialize schema validator."""
        self.predefined_schemas = self._create_predefined_schemas()

    def _create_predefined_schemas(self) -> Dict[str, FinancialDataSchema]:
        """Create predefined schemas for common financial datasets."""
        schemas = {}

        # Stock price schema
        schemas['stock_prices'] = FinancialDataSchema(
            name='stock_prices',
            version='1.0',
            fields=[
                FieldSchema('date', DataType.DATETIME, FieldConstraint(nullable=False)),
                FieldSchema('symbol', DataType.STRING, FieldConstraint(nullable=False, max_length=10)),
                FieldSchema('open', DataType.FLOAT, FieldConstraint(nullable=False, min_value=0)),
                FieldSchema('high', DataType.FLOAT, FieldConstraint(nullable=False, min_value=0)),
                FieldSchema('low', DataType.FLOAT, FieldConstraint(nullable=False, min_value=0)),
                FieldSchema('close', DataType.FLOAT, FieldConstraint(nullable=False, min_value=0)),
                FieldSchema('volume', DataType.INTEGER, FieldConstraint(nullable=False, min_value=0)),
                FieldSchema('adj_close', DataType.FLOAT, FieldConstraint(nullable=True, min_value=0))
            ],
            required_fields=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
            index_field='date',
            description='Daily stock price data with OHLCV format'
        )

        # Financial statements schema
        schemas['financial_statements'] = FinancialDataSchema(
            name='financial_statements',
            version='1.0',
            fields=[
                FieldSchema('date', DataType.DATETIME, FieldConstraint(nullable=False)),
                FieldSchema('symbol', DataType.STRING, FieldConstraint(nullable=False, max_length=10)),
                FieldSchema('period_type', DataType.STRING, FieldConstraint(
                    nullable=False, allowed_values=['Q', 'A'])),
                FieldSchema('revenue', DataType.FLOAT, FieldConstraint(nullable=True)),
                FieldSchema('net_income', DataType.FLOAT, FieldConstraint(nullable=True)),
                FieldSchema('total_assets', DataType.FLOAT, FieldConstraint(nullable=True, min_value=0)),
                FieldSchema('total_liabilities', DataType.FLOAT, FieldConstraint(nullable=True, min_value=0)),
                FieldSchema('shareholders_equity', DataType.FLOAT, FieldConstraint(nullable=True))
            ],
            required_fields=['date', 'symbol', 'period_type'],
            index_field='date',
            description='Financial statement data (income statement, balance sheet)'
        )

        # Market data schema
        schemas['market_data'] = FinancialDataSchema(
            name='market_data',
            version='1.0',
            fields=[
                FieldSchema('date', DataType.DATETIME, FieldConstraint(nullable=False)),
                FieldSchema('symbol', DataType.STRING, FieldConstraint(nullable=False, max_length=10)),
                FieldSchema('price', DataType.FLOAT, FieldConstraint(nullable=False, min_value=0)),
                FieldSchema('market_cap', DataType.FLOAT, FieldConstraint(nullable=True, min_value=0)),
                FieldSchema('pe_ratio', DataType.FLOAT, FieldConstraint(nullable=True, min_value=0)),
                FieldSchema('dividend_yield', DataType.FLOAT, FieldConstraint(nullable=True, min_value=0, max_value=1))
            ],
            required_fields=['date', 'symbol', 'price'],
            index_field='date',
            description='Market data with valuation metrics'
        )

        return schemas

    def validate_schema(self, data: pd.DataFrame, schema: FinancialDataSchema) -> List[ValidationResult]:
        """
        Validate DataFrame against schema.

        Args:
            data: DataFrame to validate
            schema: Schema to validate against

        Returns:
            List of validation results
        """
        results = []

        # Check required fields
        results.extend(self._validate_required_fields(data, schema))

        # Check field types and constraints
        for field in schema.fields:
            if field.name in data.columns:
                results.extend(self._validate_field(data, field))

        # Check index field if specified
        if schema.index_field:
            results.extend(self._validate_index_field(data, schema))

        # Financial data specific validations
        results.extend(self._validate_financial_constraints(data, schema))

        return results

    def _validate_required_fields(self, data: pd.DataFrame, schema: FinancialDataSchema) -> List[ValidationResult]:
        """Validate that all required fields are present."""
        results = []
        missing_fields = [field for field in schema.required_fields if field not in data.columns]

        if missing_fields:
            results.append(ValidationResult(
                rule_name="required_fields",
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Missing required fields: {missing_fields}",
                affected_rows=0,
                affected_columns=missing_fields,
                details={'missing_fields': missing_fields},
                timestamp=datetime.utcnow()
            ))
        else:
            results.append(ValidationResult(
                rule_name="required_fields",
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.INFO,
                passed=True,
                message="All required fields present",
                affected_rows=0,
                affected_columns=[],
                details={'required_fields': schema.required_fields},
                timestamp=datetime.utcnow()
            ))

        return results

    def _validate_field(self, data: pd.DataFrame, field: FieldSchema) -> List[ValidationResult]:
        """Validate a specific field against its schema."""
        results = []
        column = field.name
        constraints = field.constraints

        if column not in data.columns:
            return results

        series = data[column]

        # Data type validation
        results.append(self._validate_data_type(series, field))

        # Nullability validation
        if not constraints.nullable:
            null_count = series.isnull().sum()
            results.append(ValidationResult(
                rule_name=f"{column}_nullability",
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.ERROR if null_count > 0 else ValidationSeverity.INFO,
                passed=null_count == 0,
                message=f"Found {null_count} null values in non-nullable field '{column}'",
                affected_rows=null_count,
                affected_columns=[column],
                details={'null_count': null_count},
                timestamp=datetime.utcnow()
            ))

        # Value range validation
        if constraints.min_value is not None or constraints.max_value is not None:
            results.append(self._validate_value_range(series, field))

        # Length validation for strings
        if field.data_type == DataType.STRING and (constraints.min_length or constraints.max_length):
            results.append(self._validate_string_length(series, field))

        # Allowed values validation
        if constraints.allowed_values:
            results.append(self._validate_allowed_values(series, field))

        # Uniqueness validation
        if constraints.unique:
            results.append(self._validate_uniqueness(series, field))

        return results

    def _validate_data_type(self, series: pd.Series, field: FieldSchema) -> ValidationResult:
        """Validate data type of a series."""
        expected_type = field.data_type
        column = field.name

        # Check current type
        is_valid = False
        current_dtype = str(series.dtype)

        if expected_type == DataType.INTEGER:
            is_valid = pd.api.types.is_integer_dtype(series)
        elif expected_type == DataType.FLOAT:
            is_valid = pd.api.types.is_numeric_dtype(series)
        elif expected_type == DataType.STRING:
            is_valid = pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
        elif expected_type == DataType.DATETIME:
            is_valid = pd.api.types.is_datetime64_any_dtype(series)
        elif expected_type == DataType.DATE:
            is_valid = pd.api.types.is_datetime64_any_dtype(series) or isinstance(series.iloc[0], date)
        elif expected_type == DataType.BOOLEAN:
            is_valid = pd.api.types.is_bool_dtype(series)
        elif expected_type == DataType.CATEGORY:
            is_valid = pd.api.types.is_categorical_dtype(series)

        return ValidationResult(
            rule_name=f"{column}_data_type",
            category=ValidationCategory.DATA_TYPE,
            severity=ValidationSeverity.ERROR if not is_valid else ValidationSeverity.INFO,
            passed=is_valid,
            message=f"Data type validation for '{column}': expected {expected_type.value}, got {current_dtype}",
            affected_rows=len(series) if not is_valid else 0,
            affected_columns=[column],
            details={'expected_type': expected_type.value, 'actual_type': current_dtype},
            timestamp=datetime.utcnow()
        )

    def _validate_value_range(self, series: pd.Series, field: FieldSchema) -> ValidationResult:
        """Validate value range constraints."""
        column = field.name
        constraints = field.constraints

        invalid_count = 0
        issues = []

        if constraints.min_value is not None:
            below_min = series < constraints.min_value
            below_count = below_min.sum()
            if below_count > 0:
                invalid_count += below_count
                issues.append(f"{below_count} values below minimum {constraints.min_value}")

        if constraints.max_value is not None:
            above_max = series > constraints.max_value
            above_count = above_max.sum()
            if above_count > 0:
                invalid_count += above_count
                issues.append(f"{above_count} values above maximum {constraints.max_value}")

        passed = invalid_count == 0
        message = f"Value range validation for '{column}': {'; '.join(issues) if issues else 'all values within range'}"

        return ValidationResult(
            rule_name=f"{column}_value_range",
            category=ValidationCategory.RANGE,
            severity=ValidationSeverity.WARNING if invalid_count > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=message,
            affected_rows=invalid_count,
            affected_columns=[column],
            details={
                'min_value': constraints.min_value,
                'max_value': constraints.max_value,
                'invalid_count': invalid_count
            },
            timestamp=datetime.utcnow()
        )

    def _validate_string_length(self, series: pd.Series, field: FieldSchema) -> ValidationResult:
        """Validate string length constraints."""
        column = field.name
        constraints = field.constraints

        invalid_count = 0
        issues = []

        # Convert to string and get lengths
        str_lengths = series.astype(str).str.len()

        if constraints.min_length is not None:
            too_short = str_lengths < constraints.min_length
            short_count = too_short.sum()
            if short_count > 0:
                invalid_count += short_count
                issues.append(f"{short_count} values shorter than {constraints.min_length}")

        if constraints.max_length is not None:
            too_long = str_lengths > constraints.max_length
            long_count = too_long.sum()
            if long_count > 0:
                invalid_count += long_count
                issues.append(f"{long_count} values longer than {constraints.max_length}")

        passed = invalid_count == 0
        message = f"String length validation for '{column}': {'; '.join(issues) if issues else 'all lengths valid'}"

        return ValidationResult(
            rule_name=f"{column}_string_length",
            category=ValidationCategory.FORMAT,
            severity=ValidationSeverity.WARNING if invalid_count > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=message,
            affected_rows=invalid_count,
            affected_columns=[column],
            details={
                'min_length': constraints.min_length,
                'max_length': constraints.max_length,
                'invalid_count': invalid_count
            },
            timestamp=datetime.utcnow()
        )

    def _validate_allowed_values(self, series: pd.Series, field: FieldSchema) -> ValidationResult:
        """Validate allowed values constraint."""
        column = field.name
        allowed_values = field.constraints.allowed_values

        invalid_mask = ~series.isin(allowed_values)
        invalid_count = invalid_mask.sum()

        passed = invalid_count == 0
        message = f"Allowed values validation for '{column}': {invalid_count} invalid values"

        return ValidationResult(
            rule_name=f"{column}_allowed_values",
            category=ValidationCategory.RANGE,
            severity=ValidationSeverity.ERROR if invalid_count > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=message,
            affected_rows=invalid_count,
            affected_columns=[column],
            details={
                'allowed_values': allowed_values,
                'invalid_count': invalid_count,
                'invalid_values': series[invalid_mask].unique().tolist()[:10] if invalid_count > 0 else []
            },
            timestamp=datetime.utcnow()
        )

    def _validate_uniqueness(self, series: pd.Series, field: FieldSchema) -> ValidationResult:
        """Validate uniqueness constraint."""
        column = field.name

        duplicate_count = series.duplicated().sum()
        passed = duplicate_count == 0

        return ValidationResult(
            rule_name=f"{column}_uniqueness",
            category=ValidationCategory.CONSISTENCY,
            severity=ValidationSeverity.ERROR if duplicate_count > 0 else ValidationSeverity.INFO,
            passed=passed,
            message=f"Uniqueness validation for '{column}': {duplicate_count} duplicate values",
            affected_rows=duplicate_count,
            affected_columns=[column],
            details={'duplicate_count': duplicate_count},
            timestamp=datetime.utcnow()
        )

    def _validate_index_field(self, data: pd.DataFrame, schema: FinancialDataSchema) -> List[ValidationResult]:
        """Validate index field requirements."""
        results = []
        index_field = schema.index_field

        if index_field not in data.columns:
            results.append(ValidationResult(
                rule_name="index_field_exists",
                category=ValidationCategory.COMPLETENESS,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Index field '{index_field}' not found in data",
                affected_rows=0,
                affected_columns=[index_field],
                details={'index_field': index_field},
                timestamp=datetime.utcnow()
            ))
            return results

        # Check if index field is suitable for indexing
        index_series = data[index_field]

        # Should be sortable
        try:
            sorted_index = index_series.sort_values()
            sortable = True
        except Exception:
            sortable = False

        results.append(ValidationResult(
            rule_name="index_field_sortable",
            category=ValidationCategory.CONSISTENCY,
            severity=ValidationSeverity.WARNING if not sortable else ValidationSeverity.INFO,
            passed=sortable,
            message=f"Index field '{index_field}' sortability: {'passed' if sortable else 'failed'}",
            affected_rows=0 if sortable else len(data),
            affected_columns=[index_field],
            details={'index_field': index_field, 'sortable': sortable},
            timestamp=datetime.utcnow()
        ))

        return results

    def _validate_financial_constraints(self, data: pd.DataFrame, schema: FinancialDataSchema) -> List[ValidationResult]:
        """Validate financial data specific constraints."""
        results = []

        # Check for OHLC consistency if all OHLC fields are present
        ohlc_fields = ['open', 'high', 'low', 'close']
        if all(field in data.columns for field in ohlc_fields):
            inconsistent_mask = (
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            )
            inconsistent_count = inconsistent_mask.sum()

            results.append(ValidationResult(
                rule_name="ohlc_consistency",
                category=ValidationCategory.BUSINESS_LOGIC,
                severity=ValidationSeverity.ERROR if inconsistent_count > 0 else ValidationSeverity.INFO,
                passed=inconsistent_count == 0,
                message=f"OHLC consistency check: {inconsistent_count} inconsistent rows",
                affected_rows=inconsistent_count,
                affected_columns=ohlc_fields,
                details={'inconsistent_count': inconsistent_count},
                timestamp=datetime.utcnow()
            ))

        # Check for reasonable financial ratios
        if 'pe_ratio' in data.columns:
            extreme_pe = ((data['pe_ratio'] < 0) | (data['pe_ratio'] > 1000)).sum()
            results.append(ValidationResult(
                rule_name="pe_ratio_reasonable",
                category=ValidationCategory.BUSINESS_LOGIC,
                severity=ValidationSeverity.WARNING if extreme_pe > 0 else ValidationSeverity.INFO,
                passed=extreme_pe == 0,
                message=f"PE ratio reasonableness check: {extreme_pe} extreme values",
                affected_rows=extreme_pe,
                affected_columns=['pe_ratio'],
                details={'extreme_count': extreme_pe},
                timestamp=datetime.utcnow()
            ))

        return results

    def get_schema(self, schema_name: str) -> Optional[FinancialDataSchema]:
        """Get predefined schema by name."""
        return self.predefined_schemas.get(schema_name)

    def register_schema(self, schema: FinancialDataSchema) -> None:
        """Register a new schema."""
        self.predefined_schemas[schema.name] = schema

    def list_schemas(self) -> List[str]:
        """List available schema names."""
        return list(self.predefined_schemas.keys())

    def infer_schema(self, data: pd.DataFrame, schema_name: str) -> FinancialDataSchema:
        """
        Infer schema from DataFrame.

        Args:
            data: DataFrame to analyze
            schema_name: Name for the inferred schema

        Returns:
            Inferred schema
        """
        fields = []

        for column in data.columns:
            # Infer data type
            if pd.api.types.is_integer_dtype(data[column]):
                data_type = DataType.INTEGER
            elif pd.api.types.is_float_dtype(data[column]):
                data_type = DataType.FLOAT
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                data_type = DataType.DATETIME
            elif pd.api.types.is_bool_dtype(data[column]):
                data_type = DataType.BOOLEAN
            elif pd.api.types.is_categorical_dtype(data[column]):
                data_type = DataType.CATEGORY
            else:
                data_type = DataType.STRING

            # Infer constraints
            constraints = FieldConstraint(
                nullable=data[column].isnull().any(),
                min_value=data[column].min() if pd.api.types.is_numeric_dtype(data[column]) else None,
                max_value=data[column].max() if pd.api.types.is_numeric_dtype(data[column]) else None,
                unique=data[column].nunique() == len(data)
            )

            fields.append(FieldSchema(column, data_type, constraints))

        # Detect likely required fields
        required_fields = [col for col in data.columns if not data[col].isnull().any()]

        # Detect likely index field
        index_field = None
        for col in ['date', 'datetime', 'timestamp']:
            if col in data.columns:
                index_field = col
                break

        return FinancialDataSchema(
            name=schema_name,
            version='1.0',
            fields=fields,
            required_fields=required_fields,
            index_field=index_field,
            description=f'Inferred schema for {schema_name}'
        )