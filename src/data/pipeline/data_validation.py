"""
Data Validation and Quality Framework for FinLab Pipeline.

This module provides comprehensive data validation, quality assessment,
and consistency checking for the FinLab data integration pipeline.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal
import statistics
import json

from ..core.temporal import TemporalValue, DataType
from .finlab_dataset_config import FinLabField, FinLabDatasetType

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    CRITICAL = "critical"  # Data unusable, blocks pipeline
    HIGH = "high"         # Significant quality issues
    MEDIUM = "medium"     # Minor quality issues
    LOW = "low"          # Informational warnings


class ValidationCategory(Enum):
    """Validation category types."""
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    DATA_QUALITY = "data_quality"
    BUSINESS_RULES = "business_rules"
    COMPLETENESS = "completeness"
    INTEGRITY = "integrity"


@dataclass
class ValidationIssue:
    """Data validation issue representation."""
    category: ValidationCategory
    severity: ValidationSeverity
    field_name: str
    message: str
    value: Any = None
    expected_range: Optional[Tuple[float, float]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "field_name": self.field_name,
            "message": self.message,
            "value": str(self.value) if self.value is not None else None,
            "expected_range": self.expected_range,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    symbol: str
    validation_date: date
    total_records: int
    passed_validations: int
    failed_validations: int
    issues: List[ValidationIssue] = field(default_factory=list)
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Calculate validation pass rate."""
        total = self.passed_validations + self.failed_validations
        return self.passed_validations / max(total, 1)

    @property
    def critical_issues_count(self) -> int:
        """Count critical issues."""
        return len([i for i in self.issues if i.severity == ValidationSeverity.CRITICAL])

    def to_summary(self) -> Dict[str, Any]:
        """Generate summary dictionary."""
        issue_summary = {}
        for issue in self.issues:
            key = f"{issue.category.value}_{issue.severity.value}"
            issue_summary[key] = issue_summary.get(key, 0) + 1

        return {
            "symbol": self.symbol,
            "validation_date": self.validation_date.isoformat(),
            "total_records": self.total_records,
            "pass_rate": round(self.pass_rate, 4),
            "quality_score": round(self.quality_score, 2),
            "critical_issues": self.critical_issues_count,
            "issue_summary": issue_summary
        }


class DataValidator:
    """Comprehensive data validator for FinLab datasets."""

    def __init__(self):
        self.business_rules = self._load_business_rules()
        self.validation_history: Dict[str, List[ValidationReport]] = {}

    def _load_business_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load business validation rules."""
        return {
            # Price data rules
            "adj_close": {
                "min_value": 0.01,
                "max_value": 10000.0,
                "max_daily_change": 0.10,  # 10% max daily change
                "null_tolerance": 0.0
            },
            "adj_high": {
                "min_value": 0.01,
                "max_value": 10000.0,
                "null_tolerance": 0.0
            },
            "adj_low": {
                "min_value": 0.01,
                "max_value": 10000.0,
                "null_tolerance": 0.0
            },
            "adj_open": {
                "min_value": 0.01,
                "max_value": 10000.0,
                "null_tolerance": 0.0
            },

            # Financial statement rules
            "total_assets": {
                "min_value": 0.0,
                "max_growth_rate": 5.0,  # 500% max growth
                "null_tolerance": 0.0
            },
            "revenue": {
                "min_value": 0.0,
                "max_quarterly_growth": 3.0,  # 300% max quarterly growth
                "null_tolerance": 0.0
            },
            "net_income": {
                # Net income can be negative (losses)
                "max_quarterly_change": 10.0,  # 1000% max change
                "null_tolerance": 0.0
            },

            # Ratio rules
            "roa": {
                "min_value": -1.0,
                "max_value": 1.0,
                "null_tolerance": 0.05
            },
            "roe": {
                "min_value": -2.0,
                "max_value": 2.0,
                "null_tolerance": 0.05
            },
            "current_ratio": {
                "min_value": 0.0,
                "max_value": 50.0,
                "null_tolerance": 0.05
            }
        }

    def validate_temporal_value(self,
                               value: TemporalValue,
                               field_config: FinLabField,
                               historical_values: Optional[List[TemporalValue]] = None) -> List[ValidationIssue]:
        """Validate a single temporal value comprehensively."""
        issues = []

        # Temporal consistency validation
        issues.extend(self._validate_temporal_consistency(value, field_config))

        # Data quality validation
        issues.extend(self._validate_data_quality(value, field_config))

        # Business rules validation
        issues.extend(self._validate_business_rules(value, field_config))

        # Historical consistency validation
        if historical_values:
            issues.extend(self._validate_historical_consistency(value, historical_values, field_config))

        return issues

    def _validate_temporal_consistency(self,
                                     value: TemporalValue,
                                     field_config: FinLabField) -> List[ValidationIssue]:
        """Validate temporal consistency."""
        issues = []

        # Check as_of_date vs value_date consistency
        expected_lag = timedelta(days=field_config.lag_days)
        actual_lag = value.as_of_date - value.value_date

        if actual_lag < timedelta(days=0):
            issues.append(ValidationIssue(
                category=ValidationCategory.TEMPORAL_CONSISTENCY,
                severity=ValidationSeverity.CRITICAL,
                field_name=field_config.name,
                message=f"as_of_date {value.as_of_date} is before value_date {value.value_date}",
                metadata={"expected_lag_days": field_config.lag_days, "actual_lag_days": actual_lag.days}
            ))

        # Check expected data lag
        max_expected_lag = expected_lag + timedelta(days=5)  # Allow 5-day buffer
        if actual_lag > max_expected_lag:
            issues.append(ValidationIssue(
                category=ValidationCategory.TEMPORAL_CONSISTENCY,
                severity=ValidationSeverity.HIGH,
                field_name=field_config.name,
                message=f"Data lag {actual_lag.days} days exceeds expected {field_config.lag_days} days",
                metadata={"expected_lag_days": field_config.lag_days, "actual_lag_days": actual_lag.days}
            ))

        # Check future data
        if value.value_date > date.today():
            issues.append(ValidationIssue(
                category=ValidationCategory.TEMPORAL_CONSISTENCY,
                severity=ValidationSeverity.CRITICAL,
                field_name=field_config.name,
                message=f"Future data detected: value_date {value.value_date} is after today",
                value=value.value_date
            ))

        return issues

    def _validate_data_quality(self,
                             value: TemporalValue,
                             field_config: FinLabField) -> List[ValidationIssue]:
        """Validate data quality."""
        issues = []

        # Null value check
        if value.value is None:
            issues.append(ValidationIssue(
                category=ValidationCategory.DATA_QUALITY,
                severity=ValidationSeverity.HIGH,
                field_name=field_config.name,
                message="Null value detected",
                value=None
            ))
            return issues

        # Data type validation
        expected_type = field_config.data_type
        if expected_type == "float":
            if not isinstance(value.value, (int, float, Decimal)):
                issues.append(ValidationIssue(
                    category=ValidationCategory.DATA_QUALITY,
                    severity=ValidationSeverity.HIGH,
                    field_name=field_config.name,
                    message=f"Expected float, got {type(value.value).__name__}",
                    value=value.value
                ))
        elif expected_type == "int":
            if not isinstance(value.value, int):
                issues.append(ValidationIssue(
                    category=ValidationCategory.DATA_QUALITY,
                    severity=ValidationSeverity.HIGH,
                    field_name=field_config.name,
                    message=f"Expected int, got {type(value.value).__name__}",
                    value=value.value
                ))

        # Value range validation for numeric data
        if isinstance(value.value, (int, float, Decimal)):
            numeric_value = float(value.value)

            # Check for infinite or NaN values
            if not np.isfinite(numeric_value):
                issues.append(ValidationIssue(
                    category=ValidationCategory.DATA_QUALITY,
                    severity=ValidationSeverity.CRITICAL,
                    field_name=field_config.name,
                    message=f"Invalid numeric value: {numeric_value}",
                    value=numeric_value
                ))

            # Price-specific validations
            if field_config.temporal_type == DataType.PRICE:
                if numeric_value <= 0:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.DATA_QUALITY,
                        severity=ValidationSeverity.CRITICAL,
                        field_name=field_config.name,
                        message=f"Non-positive price value: {numeric_value}",
                        value=numeric_value
                    ))

        return issues

    def _validate_business_rules(self,
                               value: TemporalValue,
                               field_config: FinLabField) -> List[ValidationIssue]:
        """Validate against business rules."""
        issues = []

        rules = self.business_rules.get(field_config.name, {})
        if not rules or value.value is None:
            return issues

        if isinstance(value.value, (int, float, Decimal)):
            numeric_value = float(value.value)

            # Min/max value checks
            if "min_value" in rules and numeric_value < rules["min_value"]:
                issues.append(ValidationIssue(
                    category=ValidationCategory.BUSINESS_RULES,
                    severity=ValidationSeverity.HIGH,
                    field_name=field_config.name,
                    message=f"Value {numeric_value} below minimum {rules['min_value']}",
                    value=numeric_value,
                    expected_range=(rules["min_value"], rules.get("max_value"))
                ))

            if "max_value" in rules and numeric_value > rules["max_value"]:
                issues.append(ValidationIssue(
                    category=ValidationCategory.BUSINESS_RULES,
                    severity=ValidationSeverity.HIGH,
                    field_name=field_config.name,
                    message=f"Value {numeric_value} above maximum {rules['max_value']}",
                    value=numeric_value,
                    expected_range=(rules.get("min_value"), rules["max_value"])
                ))

        return issues

    def _validate_historical_consistency(self,
                                       value: TemporalValue,
                                       historical_values: List[TemporalValue],
                                       field_config: FinLabField) -> List[ValidationIssue]:
        """Validate consistency with historical data."""
        issues = []

        if not historical_values or value.value is None:
            return issues

        # Find most recent historical value
        sorted_values = sorted(historical_values, key=lambda v: v.value_date, reverse=True)
        most_recent = sorted_values[0] if sorted_values else None

        if most_recent and most_recent.value is not None:
            current_val = float(value.value) if isinstance(value.value, (int, float, Decimal)) else None
            previous_val = float(most_recent.value) if isinstance(most_recent.value, (int, float, Decimal)) else None

            if current_val is not None and previous_val is not None and previous_val != 0:
                # Calculate change rate
                change_rate = abs(current_val - previous_val) / abs(previous_val)

                rules = self.business_rules.get(field_config.name, {})

                # Daily change validation for prices
                if field_config.temporal_type == DataType.PRICE:
                    max_daily_change = rules.get("max_daily_change", 0.20)  # 20% default
                    days_diff = (value.value_date - most_recent.value_date).days

                    if days_diff == 1 and change_rate > max_daily_change:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.BUSINESS_RULES,
                            severity=ValidationSeverity.MEDIUM,
                            field_name=field_config.name,
                            message=f"Large daily change: {change_rate:.2%} exceeds {max_daily_change:.2%}",
                            value=current_val,
                            metadata={
                                "previous_value": previous_val,
                                "change_rate": change_rate,
                                "days_diff": days_diff
                            }
                        ))

                # Growth rate validation for fundamental data
                elif field_config.temporal_type == DataType.FUNDAMENTAL:
                    if field_config.dataset_type == FinLabDatasetType.FINANCIAL_STATEMENT:
                        max_growth = rules.get("max_quarterly_growth", 5.0)  # 500% default
                        if change_rate > max_growth:
                            issues.append(ValidationIssue(
                                category=ValidationCategory.BUSINESS_RULES,
                                severity=ValidationSeverity.MEDIUM,
                                field_name=field_config.name,
                                message=f"Large quarterly change: {change_rate:.2%} exceeds {max_growth:.2%}",
                                value=current_val,
                                metadata={
                                    "previous_value": previous_val,
                                    "change_rate": change_rate
                                }
                            ))

        return issues

    def validate_batch(self,
                      values: List[TemporalValue],
                      field_configs: Dict[str, FinLabField],
                      symbol: str) -> ValidationReport:
        """Validate a batch of temporal values."""
        report = ValidationReport(
            symbol=symbol,
            validation_date=date.today(),
            total_records=len(values)
        )

        passed = 0
        failed = 0

        # Group values by field for historical consistency checking
        values_by_field: Dict[str, List[TemporalValue]] = {}
        for value in values:
            field_name = value.metadata.get("field", "unknown")
            if field_name not in values_by_field:
                values_by_field[field_name] = []
            values_by_field[field_name].append(value)

        # Validate each value
        for value in values:
            field_name = value.metadata.get("field", "unknown")
            field_config = field_configs.get(field_name)

            if not field_config:
                report.issues.append(ValidationIssue(
                    category=ValidationCategory.INTEGRITY,
                    severity=ValidationSeverity.MEDIUM,
                    field_name=field_name,
                    message=f"Unknown field configuration for {field_name}",
                    value=field_name
                ))
                failed += 1
                continue

            # Get historical values for this field (excluding current value)
            historical_values = [v for v in values_by_field[field_name]
                                if v.value_date < value.value_date]

            # Validate the value
            value_issues = self.validate_temporal_value(value, field_config, historical_values)

            if value_issues:
                report.issues.extend(value_issues)
                failed += 1
            else:
                passed += 1

        report.passed_validations = passed
        report.failed_validations = failed
        report.quality_score = self._calculate_quality_score(report)

        return report

    def _calculate_quality_score(self, report: ValidationReport) -> float:
        """Calculate overall quality score (0-100)."""
        if report.total_records == 0:
            return 100.0

        base_score = report.pass_rate * 100

        # Penalty for critical issues
        critical_penalty = min(report.critical_issues_count * 10, 50)

        # Penalty for high-severity issues
        high_issues = len([i for i in report.issues if i.severity == ValidationSeverity.HIGH])
        high_penalty = min(high_issues * 5, 25)

        # Penalty for medium-severity issues
        medium_issues = len([i for i in report.issues if i.severity == ValidationSeverity.MEDIUM])
        medium_penalty = min(medium_issues * 2, 15)

        final_score = max(0, base_score - critical_penalty - high_penalty - medium_penalty)
        return round(final_score, 2)

    def generate_quality_metrics(self,
                                symbol: str,
                                lookback_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive quality metrics for a symbol."""
        recent_reports = []
        cutoff_date = date.today() - timedelta(days=lookback_days)

        # Get recent validation reports
        if symbol in self.validation_history:
            recent_reports = [
                r for r in self.validation_history[symbol]
                if r.validation_date >= cutoff_date
            ]

        if not recent_reports:
            return {
                "symbol": symbol,
                "period": f"last_{lookback_days}_days",
                "no_data": True
            }

        # Calculate aggregate metrics
        total_records = sum(r.total_records for r in recent_reports)
        total_passed = sum(r.passed_validations for r in recent_reports)
        total_failed = sum(r.failed_validations for r in recent_reports)

        avg_quality_score = statistics.mean([r.quality_score for r in recent_reports])
        total_critical_issues = sum(r.critical_issues_count for r in recent_reports)

        # Issue category analysis
        issue_categories = {}
        for report in recent_reports:
            for issue in report.issues:
                cat_sev = f"{issue.category.value}_{issue.severity.value}"
                issue_categories[cat_sev] = issue_categories.get(cat_sev, 0) + 1

        return {
            "symbol": symbol,
            "period": f"last_{lookback_days}_days",
            "total_records": total_records,
            "overall_pass_rate": total_passed / max(total_passed + total_failed, 1),
            "average_quality_score": round(avg_quality_score, 2),
            "total_critical_issues": total_critical_issues,
            "validation_runs": len(recent_reports),
            "issue_breakdown": issue_categories,
            "trend": self._calculate_quality_trend(recent_reports)
        }

    def _calculate_quality_trend(self, reports: List[ValidationReport]) -> str:
        """Calculate quality trend from recent reports."""
        if len(reports) < 2:
            return "insufficient_data"

        # Sort by date
        sorted_reports = sorted(reports, key=lambda r: r.validation_date)

        recent_half = sorted_reports[len(sorted_reports)//2:]
        earlier_half = sorted_reports[:len(sorted_reports)//2]

        if not earlier_half or not recent_half:
            return "insufficient_data"

        recent_avg = statistics.mean([r.quality_score for r in recent_half])
        earlier_avg = statistics.mean([r.quality_score for r in earlier_half])

        diff = recent_avg - earlier_avg

        if diff > 5:
            return "improving"
        elif diff < -5:
            return "deteriorating"
        else:
            return "stable"

    def store_validation_report(self, report: ValidationReport) -> None:
        """Store validation report in history."""
        if report.symbol not in self.validation_history:
            self.validation_history[report.symbol] = []

        self.validation_history[report.symbol].append(report)

        # Keep only last 100 reports per symbol
        if len(self.validation_history[report.symbol]) > 100:
            self.validation_history[report.symbol] = \
                self.validation_history[report.symbol][-100:]

    def export_validation_summary(self,
                                 symbol: Optional[str] = None,
                                 days: int = 7) -> List[Dict[str, Any]]:
        """Export validation summary for reporting."""
        summaries = []
        cutoff_date = date.today() - timedelta(days=days)

        symbols_to_process = [symbol] if symbol else list(self.validation_history.keys())

        for sym in symbols_to_process:
            if sym not in self.validation_history:
                continue

            recent_reports = [
                r for r in self.validation_history[sym]
                if r.validation_date >= cutoff_date
            ]

            for report in recent_reports:
                summaries.append(report.to_summary())

        return summaries