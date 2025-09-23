"""
Data Quality Validators and Monitoring Hooks.

This module provides comprehensive data quality validation, anomaly detection,
and monitoring capabilities for the point-in-time data system.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics
import asyncio
import threading
from abc import ABC, abstractmethod

from ..core.temporal import TemporalValue, DataType
from ..models.taiwan_market import TaiwanMarketData, TaiwanMarketDataValidator

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Data quality issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityCheckType(Enum):
    """Types of data quality checks."""
    COMPLETENESS = "completeness"        # Missing data detection
    ACCURACY = "accuracy"                # Value correctness
    CONSISTENCY = "consistency"          # Internal consistency
    TIMELINESS = "timeliness"           # Data freshness
    VALIDITY = "validity"               # Format and range validation
    ANOMALY = "anomaly"                 # Statistical anomaly detection
    TEMPORAL = "temporal"               # Temporal consistency
    BUSINESS_RULES = "business_rules"   # Business logic validation


@dataclass
class QualityIssue:
    """Data quality issue report."""
    check_type: QualityCheckType
    severity: SeverityLevel
    symbol: str
    data_type: DataType
    data_date: date
    issue_date: datetime
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_action: Optional[str] = None
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class QualityMetrics:
    """Data quality metrics."""
    symbol: str
    data_type: DataType
    period_start: date
    period_end: date
    completeness_score: float  # 0.0 to 1.0
    accuracy_score: float      # 0.0 to 1.0
    timeliness_score: float    # 0.0 to 1.0
    overall_score: float       # 0.0 to 1.0
    total_records: int
    error_count: int
    warning_count: int
    last_updated: datetime = field(default_factory=datetime.utcnow)


class DataQualityValidator(ABC):
    """Abstract base class for data quality validators."""
    
    @abstractmethod
    def validate(self, value: TemporalValue, context: Dict[str, Any] = None) -> List[QualityIssue]:
        """Validate a temporal value and return quality issues."""
        pass
    
    @abstractmethod
    def get_validator_name(self) -> str:
        """Get the name of this validator."""
        pass


class CompletenessValidator(DataQualityValidator):
    """Validates data completeness and detects missing values."""
    
    def __init__(self, required_fields: Dict[DataType, Set[str]]):
        self.required_fields = required_fields
    
    def validate(self, value: TemporalValue, context: Dict[str, Any] = None) -> List[QualityIssue]:
        """Check for missing required data fields."""
        issues = []
        
        required = self.required_fields.get(value.data_type, set())
        
        if isinstance(value.value, dict):
            missing_fields = required - set(value.value.keys())
            
            if missing_fields:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.COMPLETENESS,
                    severity=SeverityLevel.WARNING,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Missing required fields: {missing_fields}",
                    details={"missing_fields": list(missing_fields)},
                    suggested_action="Verify data source completeness"
                ))
        
        elif value.value is None:
            issues.append(QualityIssue(
                check_type=QualityCheckType.COMPLETENESS,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol,
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description="Null value for required data",
                suggested_action="Check data source"
            ))
        
        return issues
    
    def get_validator_name(self) -> str:
        return "completeness_validator"


class AccuracyValidator(DataQualityValidator):
    """Validates data accuracy and range constraints."""
    
    def __init__(self):
        # Taiwan stock market constraints
        self.price_constraints = {
            "min_price": Decimal("0.01"),
            "max_price": Decimal("10000.00"),
            "min_volume": 0,
            "max_daily_change": 0.10  # 10% daily limit
        }
    
    def validate(self, value: TemporalValue, context: Dict[str, Any] = None) -> List[QualityIssue]:
        """Check data accuracy and range constraints."""
        issues = []
        
        if value.data_type == DataType.PRICE and isinstance(value.value, (int, float, Decimal)):
            price = Decimal(str(value.value))
            
            # Check price range
            if price < self.price_constraints["min_price"]:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Price {price} below minimum {self.price_constraints['min_price']}",
                    details={"price": float(price), "min_allowed": float(self.price_constraints["min_price"])},
                    suggested_action="Verify price data source"
                ))
            
            elif price > self.price_constraints["max_price"]:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.WARNING,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Price {price} unusually high",
                    details={"price": float(price)},
                    suggested_action="Verify for stock splits or unusual events"
                ))
        
        elif value.data_type == DataType.VOLUME and isinstance(value.value, (int, float)):
            volume = int(value.value)
            
            if volume < 0:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Negative volume: {volume}",
                    details={"volume": volume},
                    suggested_action="Check data source for volume calculation errors"
                ))
        
        return issues
    
    def get_validator_name(self) -> str:
        return "accuracy_validator"


class TimelinessValidator(DataQualityValidator):
    """Validates data timeliness and freshness."""
    
    def __init__(self, max_lag_hours: Dict[DataType, int]):
        self.max_lag_hours = max_lag_hours
    
    def validate(self, value: TemporalValue, context: Dict[str, Any] = None) -> List[QualityIssue]:
        """Check data timeliness constraints."""
        issues = []
        
        max_lag = self.max_lag_hours.get(value.data_type, 24)
        lag_hours = (datetime.utcnow() - value.created_at).total_seconds() / 3600
        
        if lag_hours > max_lag:
            severity = SeverityLevel.WARNING if lag_hours < max_lag * 2 else SeverityLevel.ERROR
            
            issues.append(QualityIssue(
                check_type=QualityCheckType.TIMELINESS,
                severity=severity,
                symbol=value.symbol,
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Data lag {lag_hours:.1f} hours exceeds maximum {max_lag} hours",
                details={"lag_hours": lag_hours, "max_lag_hours": max_lag},
                suggested_action="Check data ingestion pipeline"
            ))
        
        return issues
    
    def get_validator_name(self) -> str:
        return "timeliness_validator"


class AnomalyDetector(DataQualityValidator):
    """Statistical anomaly detection for financial data."""
    
    def __init__(self, lookback_days: int = 30, z_score_threshold: float = 3.0):
        self.lookback_days = lookback_days
        self.z_score_threshold = z_score_threshold
        self.historical_data: Dict[Tuple[str, DataType], deque] = defaultdict(
            lambda: deque(maxlen=lookback_days)
        )
    
    def update_historical_data(self, value: TemporalValue):
        """Update historical data for anomaly detection."""
        key = (value.symbol, value.data_type)
        
        if isinstance(value.value, (int, float, Decimal)):
            self.historical_data[key].append(float(value.value))
    
    def validate(self, value: TemporalValue, context: Dict[str, Any] = None) -> List[QualityIssue]:
        """Detect statistical anomalies in data."""
        issues = []
        
        if not isinstance(value.value, (int, float, Decimal)):
            return issues
        
        key = (value.symbol, value.data_type)
        historical = list(self.historical_data[key])
        
        if len(historical) < 10:  # Need sufficient history
            return issues
        
        current_value = float(value.value)
        
        try:
            mean = statistics.mean(historical)
            stdev = statistics.stdev(historical)
            
            if stdev > 0:
                z_score = abs((current_value - mean) / stdev)
                
                if z_score > self.z_score_threshold:
                    severity = SeverityLevel.WARNING if z_score < self.z_score_threshold * 1.5 else SeverityLevel.ERROR
                    
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.ANOMALY,
                        severity=severity,
                        symbol=value.symbol,
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Statistical anomaly detected: z-score {z_score:.2f}",
                        details={
                            "current_value": current_value,
                            "historical_mean": mean,
                            "historical_stdev": stdev,
                            "z_score": z_score,
                            "threshold": self.z_score_threshold
                        },
                        suggested_action="Verify data accuracy and check for corporate actions"
                    ))
        
        except statistics.StatisticsError:
            # Not enough variation in historical data
            pass
        
        # Update historical data after validation
        self.update_historical_data(value)
        
        return issues
    
    def get_validator_name(self) -> str:
        return "anomaly_detector"


class TemporalConsistencyValidator(DataQualityValidator):
    """Validates temporal consistency and ordering."""
    
    def validate(self, value: TemporalValue, context: Dict[str, Any] = None) -> List[QualityIssue]:
        """Check temporal consistency."""
        issues = []
        
        # Check if as_of_date is after value_date (basic temporal constraint)
        if value.as_of_date < value.value_date:
            issues.append(QualityIssue(
                check_type=QualityCheckType.TEMPORAL,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol,
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"as_of_date {value.as_of_date} before value_date {value.value_date}",
                details={
                    "as_of_date": value.as_of_date.isoformat(),
                    "value_date": value.value_date.isoformat()
                },
                suggested_action="Check data timestamp logic"
            ))
        
        # Check for future dates
        today = date.today()
        if value.value_date > today:
            issues.append(QualityIssue(
                check_type=QualityCheckType.TEMPORAL,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol,
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Future value_date {value.value_date}",
                details={"value_date": value.value_date.isoformat(), "today": today.isoformat()},
                suggested_action="Check system clock and data source"
            ))
        
        if value.as_of_date > today:
            issues.append(QualityIssue(
                check_type=QualityCheckType.TEMPORAL,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol,
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Future as_of_date {value.as_of_date}",
                details={"as_of_date": value.as_of_date.isoformat(), "today": today.isoformat()},
                suggested_action="Check system clock and data source"
            ))
        
        return issues
    
    def get_validator_name(self) -> str:
        return "temporal_consistency_validator"


class QualityMonitor:
    """Data quality monitoring and alerting system."""
    
    def __init__(self, 
                 validators: List[DataQualityValidator],
                 alert_thresholds: Dict[SeverityLevel, int] = None,
                 enable_alerts: bool = True):
        
        self.validators = validators
        self.alert_thresholds = alert_thresholds or {
            SeverityLevel.CRITICAL: 1,  # Alert immediately
            SeverityLevel.ERROR: 5,     # Alert after 5 errors
            SeverityLevel.WARNING: 20,  # Alert after 20 warnings
            SeverityLevel.INFO: 100     # Alert after 100 info items
        }
        self.enable_alerts = enable_alerts
        
        # Issue tracking
        self.issues: List[QualityIssue] = []
        self.issue_counts: Dict[SeverityLevel, int] = defaultdict(int)
        self.metrics_cache: Dict[Tuple[str, DataType], QualityMetrics] = {}
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[QualityIssue], None]] = []
        
        # Performance tracking
        self.validation_count = 0
        self.total_validation_time = 0.0
        
        logger.info(f"Quality monitor initialized with {len(validators)} validators")
    
    def add_alert_callback(self, callback: Callable[[QualityIssue], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def validate_value(self, value: TemporalValue, context: Dict[str, Any] = None) -> List[QualityIssue]:
        """Validate a temporal value using all validators."""
        start_time = datetime.utcnow()
        all_issues = []
        
        for validator in self.validators:
            try:
                issues = validator.validate(value, context)
                all_issues.extend(issues)
                
                # Store issues and update counts
                for issue in issues:
                    self.issues.append(issue)
                    self.issue_counts[issue.severity] += 1
                    
                    # Check alert thresholds
                    if (self.enable_alerts and 
                        self.issue_counts[issue.severity] >= self.alert_thresholds[issue.severity]):
                        self._trigger_alert(issue)
                
            except Exception as e:
                logger.error(f"Validator {validator.get_validator_name()} failed: {e}")
                
                # Create a critical issue for validator failure
                critical_issue = QualityIssue(
                    check_type=QualityCheckType.VALIDITY,
                    severity=SeverityLevel.CRITICAL,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Validator {validator.get_validator_name()} failed: {e}",
                    suggested_action="Check validator configuration"
                )
                all_issues.append(critical_issue)
                self.issues.append(critical_issue)
        
        # Update performance metrics
        self.validation_count += 1
        validation_time = (datetime.utcnow() - start_time).total_seconds()
        self.total_validation_time += validation_time
        
        return all_issues
    
    def _trigger_alert(self, issue: QualityIssue):
        """Trigger alert for quality issue."""
        for callback in self.alert_callbacks:
            try:
                callback(issue)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def get_quality_metrics(self, 
                           symbol: str,
                           data_type: DataType,
                           period_days: int = 30) -> QualityMetrics:
        """Calculate quality metrics for symbol/data_type."""
        end_date = date.today()
        start_date = end_date - timedelta(days=period_days)
        
        # Filter issues for this symbol/data_type in the period
        relevant_issues = [
            issue for issue in self.issues
            if (issue.symbol == symbol and 
                issue.data_type == data_type and
                start_date <= issue.data_date <= end_date)
        ]
        
        # Count issues by severity
        error_count = sum(1 for issue in relevant_issues 
                         if issue.severity in [SeverityLevel.ERROR, SeverityLevel.CRITICAL])
        warning_count = sum(1 for issue in relevant_issues 
                          if issue.severity == SeverityLevel.WARNING)
        
        # Estimate total records (simplified)
        total_records = period_days  # Assume daily data
        
        # Calculate scores (simplified scoring)
        completeness_score = max(0.0, 1.0 - (error_count * 0.1))
        accuracy_score = max(0.0, 1.0 - (error_count * 0.05) - (warning_count * 0.01))
        timeliness_score = 0.9  # Simplified
        overall_score = (completeness_score + accuracy_score + timeliness_score) / 3
        
        return QualityMetrics(
            symbol=symbol,
            data_type=data_type,
            period_start=start_date,
            period_end=end_date,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score,
            overall_score=overall_score,
            total_records=total_records,
            error_count=error_count,
            warning_count=warning_count
        )
    
    def get_summary_report(self, days: int = 7) -> Dict[str, Any]:
        """Get quality summary report."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Filter recent issues
        recent_issues = [
            issue for issue in self.issues
            if start_date <= issue.issue_date <= end_date
        ]
        
        # Group by severity
        severity_counts = defaultdict(int)
        for issue in recent_issues:
            severity_counts[issue.severity] += 1
        
        # Group by symbol
        symbol_counts = defaultdict(int)
        for issue in recent_issues:
            symbol_counts[issue.symbol] += 1
        
        # Performance metrics
        avg_validation_time = (
            self.total_validation_time / max(self.validation_count, 1)
        )
        
        return {
            "period_days": days,
            "total_issues": len(recent_issues),
            "severity_breakdown": dict(severity_counts),
            "top_problematic_symbols": dict(
                sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "validation_performance": {
                "total_validations": self.validation_count,
                "avg_validation_time_seconds": avg_validation_time,
                "total_validation_time_seconds": self.total_validation_time
            },
            "active_validators": [v.get_validator_name() for v in self.validators]
        }
    
    def clear_resolved_issues(self):
        """Remove resolved issues from tracking."""
        self.issues = [issue for issue in self.issues if not issue.resolved]
        
        # Recalculate issue counts
        self.issue_counts.clear()
        for issue in self.issues:
            self.issue_counts[issue.severity] += 1


# Factory function for creating standard quality monitor

def create_standard_quality_monitor(enable_anomaly_detection: bool = True) -> QualityMonitor:
    """Create a standard quality monitor with common validators."""
    
    # Required fields by data type
    required_fields = {
        DataType.PRICE: {"close_price"},
        DataType.VOLUME: {"volume"},
        DataType.FUNDAMENTAL: {"revenue", "net_income"},
        DataType.MARKET_DATA: set()
    }
    
    # Maximum lag hours by data type
    max_lag_hours = {
        DataType.PRICE: 2,        # Price data should be available within 2 hours
        DataType.VOLUME: 2,       # Volume data should be available within 2 hours
        DataType.FUNDAMENTAL: 72, # Fundamental data can have 3-day lag
        DataType.MARKET_DATA: 4   # Market data within 4 hours
    }
    
    # Create validators
    validators = [
        CompletenessValidator(required_fields),
        AccuracyValidator(),
        TimelinessValidator(max_lag_hours),
        TemporalConsistencyValidator()
    ]
    
    if enable_anomaly_detection:
        validators.append(AnomalyDetector())
    
    return QualityMonitor(validators)


# Example alert callback functions

def log_alert_callback(issue: QualityIssue):
    """Log quality issues as alerts."""
    logger.warning(f"QUALITY ALERT: {issue.severity.value.upper()} - "
                  f"{issue.symbol} {issue.data_type.value} - {issue.description}")


def email_alert_callback(issue: QualityIssue):
    """Send email alert for quality issues (placeholder)."""
    if issue.severity in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]:
        # Placeholder for email notification
        logger.info(f"Would send email alert for: {issue.description}")


def slack_alert_callback(issue: QualityIssue):
    """Send Slack alert for quality issues (placeholder)."""
    if issue.severity == SeverityLevel.CRITICAL:
        # Placeholder for Slack notification
        logger.info(f"Would send Slack alert for: {issue.description}")


# Enhanced validators using the new plugin architecture

from .validation_engine import ValidationPlugin, ValidationContext, ValidationOutput, ValidationResult, ValidationPriority


class EnhancedCompletenessValidator(ValidationPlugin):
    """Enhanced completeness validator using the plugin architecture."""
    
    def __init__(self, required_fields: Dict[DataType, Set[str]]):
        self.required_fields = required_fields
    
    @property
    def name(self) -> str:
        return "enhanced_completeness_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.HIGH
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return set(self.required_fields.keys())
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return value.data_type in self.required_fields
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Perform enhanced completeness validation."""
        start_time = datetime.utcnow()
        issues = []
        
        required = self.required_fields.get(value.data_type, set())
        
        if isinstance(value.value, dict):
            missing_fields = required - set(value.value.keys())
            
            if missing_fields:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.COMPLETENESS,
                    severity=SeverityLevel.WARNING,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Missing required fields: {missing_fields}",
                    details={"missing_fields": list(missing_fields)},
                    suggested_action="Verify data source completeness"
                ))
            
            # Check for null values in required fields
            null_fields = []
            for field in required:
                if field in value.value and value.value[field] is None:
                    null_fields.append(field)
            
            if null_fields:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.COMPLETENESS,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Null values in required fields: {null_fields}",
                    details={"null_fields": null_fields},
                    suggested_action="Check data source for null handling"
                ))
        
        elif value.value is None and required:
            issues.append(QualityIssue(
                check_type=QualityCheckType.COMPLETENESS,
                severity=SeverityLevel.CRITICAL,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description="Null value for required data",
                suggested_action="Check data source"
            ))
        
        result = ValidationResult.FAIL if any(i.severity == SeverityLevel.CRITICAL for i in issues) else ValidationResult.PASS
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )


class EnhancedAccuracyValidator(ValidationPlugin):
    """Enhanced accuracy validator with Taiwan market constraints."""
    
    def __init__(self):
        # Taiwan market constraints
        self.price_constraints = {
            "min_price": Decimal("0.01"),
            "max_price": Decimal("10000.00"),
            "min_volume": 0,
            "max_daily_change": 0.10  # 10% daily limit
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_validation_time_ms": 5.0  # Target < 5ms for accuracy validation
        }
    
    @property
    def name(self) -> str:
        return "enhanced_accuracy_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.HIGH
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.PRICE, DataType.VOLUME, DataType.MARKET_DATA}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return value.data_type in self.supported_data_types
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Perform enhanced accuracy validation with Taiwan market rules."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            if value.data_type == DataType.PRICE:
                issues.extend(self._validate_price_accuracy(value, context))
            elif value.data_type == DataType.VOLUME:
                issues.extend(self._validate_volume_accuracy(value, context))
            elif value.data_type == DataType.MARKET_DATA:
                issues.extend(self._validate_market_data_accuracy(value, context))
        
        except Exception as e:
            logger.error(f"Error in enhanced accuracy validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
        
        result = ValidationResult.FAIL if any(i.severity == SeverityLevel.CRITICAL for i in issues) else ValidationResult.PASS
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Performance check
        if execution_time > self.performance_thresholds["max_validation_time_ms"]:
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.WARNING,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation performance warning: {execution_time:.2f}ms > {self.performance_thresholds['max_validation_time_ms']}ms",
                details={"execution_time_ms": execution_time, "threshold_ms": self.performance_thresholds["max_validation_time_ms"]}
            ))
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )
    
    def _validate_price_accuracy(self, value: TemporalValue, context: ValidationContext) -> List[QualityIssue]:
        """Validate price accuracy."""
        issues = []
        
        if isinstance(value.value, (int, float, Decimal)):
            price = Decimal(str(value.value))
            
            # Check price range
            if price < self.price_constraints["min_price"]:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Price {price} below minimum {self.price_constraints['min_price']}",
                    details={"price": float(price), "min_allowed": float(self.price_constraints["min_price"])},
                    suggested_action="Verify price data source"
                ))
            
            elif price > self.price_constraints["max_price"]:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.WARNING,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Price {price} unusually high",
                    details={"price": float(price)},
                    suggested_action="Verify for stock splits or unusual events"
                ))
        
        return issues
    
    def _validate_volume_accuracy(self, value: TemporalValue, context: ValidationContext) -> List[QualityIssue]:
        """Validate volume accuracy."""
        issues = []
        
        if isinstance(value.value, (int, float)):
            volume = int(value.value)
            
            if volume < 0:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Negative volume: {volume}",
                    details={"volume": volume},
                    suggested_action="Check data source for volume calculation errors"
                ))
        
        return issues
    
    def _validate_market_data_accuracy(self, value: TemporalValue, context: ValidationContext) -> List[QualityIssue]:
        """Validate market data accuracy."""
        issues = []
        
        if isinstance(value.value, dict):
            # Check OHLC consistency
            ohlc_fields = ["open_price", "high_price", "low_price", "close_price"]
            ohlc_values = {}
            
            for field in ohlc_fields:
                if field in value.value and value.value[field] is not None:
                    try:
                        ohlc_values[field] = Decimal(str(value.value[field]))
                    except (ValueError, TypeError):
                        issues.append(QualityIssue(
                            check_type=QualityCheckType.ACCURACY,
                            severity=SeverityLevel.ERROR,
                            symbol=value.symbol or "",
                            data_type=value.data_type,
                            data_date=value.value_date,
                            issue_date=datetime.utcnow(),
                            description=f"Invalid {field} value: {value.value[field]}",
                            suggested_action="Check data type conversion"
                        ))
            
            # OHLC consistency checks
            if len(ohlc_values) >= 2:
                high = ohlc_values.get("high_price")
                low = ohlc_values.get("low_price")
                open_price = ohlc_values.get("open_price")
                close = ohlc_values.get("close_price")
                
                if high and low and high < low:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=SeverityLevel.ERROR,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"High price {high} < Low price {low}",
                        details={"high": float(high), "low": float(low)},
                        suggested_action="Check OHLC data consistency"
                    ))
                
                if high and open_price and open_price > high:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=SeverityLevel.ERROR,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Open price {open_price} > High price {high}",
                        details={"open": float(open_price), "high": float(high)},
                        suggested_action="Check OHLC data consistency"
                    ))
                
                if low and close and close < low:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=SeverityLevel.ERROR,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Close price {close} < Low price {low}",
                        details={"close": float(close), "low": float(low)},
                        suggested_action="Check OHLC data consistency"
                    ))
        
        return issues


class ConsistencyValidator(ValidationPlugin):
    """Validates data consistency across related fields and time periods."""
    
    def __init__(self):
        self.consistency_rules = {
            # Market cap should be price * shares outstanding
            "market_cap_consistency": {
                "fields": ["close_price", "outstanding_shares", "market_cap"],
                "tolerance": 0.01  # 1% tolerance
            }
        }
    
    @property
    def name(self) -> str:
        return "consistency_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.MEDIUM
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.MARKET_DATA, DataType.FUNDAMENTAL}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return value.data_type in self.supported_data_types
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Perform consistency validation."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            if isinstance(value.value, dict):
                # Check market cap consistency
                issues.extend(self._check_market_cap_consistency(value))
                
                # Check financial ratios consistency
                if value.data_type == DataType.FUNDAMENTAL:
                    issues.extend(self._check_financial_ratios_consistency(value))
        
        except Exception as e:
            logger.error(f"Error in consistency validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
        
        result = ValidationResult.PASS if not issues else ValidationResult.WARNING
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )
    
    def _check_market_cap_consistency(self, value: TemporalValue) -> List[QualityIssue]:
        """Check market capitalization consistency."""
        issues = []
        data = value.value
        
        required_fields = ["close_price", "outstanding_shares", "market_cap"]
        if all(field in data and data[field] is not None for field in required_fields):
            try:
                price = Decimal(str(data["close_price"]))
                shares = Decimal(str(data["outstanding_shares"]))
                reported_market_cap = Decimal(str(data["market_cap"]))
                
                calculated_market_cap = price * shares
                tolerance = self.consistency_rules["market_cap_consistency"]["tolerance"]
                
                if calculated_market_cap > 0:
                    relative_diff = abs(calculated_market_cap - reported_market_cap) / calculated_market_cap
                    
                    if relative_diff > tolerance:
                        issues.append(QualityIssue(
                            check_type=QualityCheckType.CONSISTENCY,
                            severity=SeverityLevel.WARNING,
                            symbol=value.symbol or "",
                            data_type=value.data_type,
                            data_date=value.value_date,
                            issue_date=datetime.utcnow(),
                            description=f"Market cap inconsistency: calculated {calculated_market_cap}, reported {reported_market_cap}",
                            details={
                                "calculated_market_cap": float(calculated_market_cap),
                                "reported_market_cap": float(reported_market_cap),
                                "relative_difference": float(relative_diff),
                                "tolerance": tolerance
                            },
                            suggested_action="Verify market cap calculation or outstanding shares data"
                        ))
            
            except (ValueError, TypeError, ZeroDivisionError) as e:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.CONSISTENCY,
                    severity=SeverityLevel.WARNING,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Market cap consistency check failed: {e}",
                    suggested_action="Check data types and values"
                ))
        
        return issues
    
    def _check_financial_ratios_consistency(self, value: TemporalValue) -> List[QualityIssue]:
        """Check financial ratios consistency."""
        issues = []
        data = value.value
        
        # Check if ROE = Net Income / Total Equity
        if all(field in data and data[field] is not None for field in ["roe", "net_income", "total_equity"]):
            try:
                reported_roe = Decimal(str(data["roe"]))
                net_income = Decimal(str(data["net_income"]))
                total_equity = Decimal(str(data["total_equity"]))
                
                if total_equity != 0:
                    calculated_roe = net_income / total_equity
                    tolerance = 0.01  # 1% tolerance
                    
                    if abs(calculated_roe - reported_roe) / max(abs(calculated_roe), abs(reported_roe)) > tolerance:
                        issues.append(QualityIssue(
                            check_type=QualityCheckType.CONSISTENCY,
                            severity=SeverityLevel.WARNING,
                            symbol=value.symbol or "",
                            data_type=value.data_type,
                            data_date=value.value_date,
                            issue_date=datetime.utcnow(),
                            description=f"ROE inconsistency: calculated {calculated_roe:.4f}, reported {reported_roe:.4f}",
                            details={
                                "calculated_roe": float(calculated_roe),
                                "reported_roe": float(reported_roe),
                                "net_income": float(net_income),
                                "total_equity": float(total_equity)
                            },
                            suggested_action="Verify ROE calculation"
                        ))
            
            except (ValueError, TypeError, ZeroDivisionError):
                pass  # Skip if calculation fails
        
        return issues


# Factory function for creating enhanced validators

def create_enhanced_validators() -> List[ValidationPlugin]:
    """Create enhanced validation plugins."""
    
    # Required fields by data type
    required_fields = {
        DataType.PRICE: {"close_price"},
        DataType.VOLUME: {"volume"},
        DataType.FUNDAMENTAL: {"revenue", "net_income"},
        DataType.MARKET_DATA: {"close_price", "volume"}
    }
    
    return [
        EnhancedCompletenessValidator(required_fields),
        EnhancedAccuracyValidator(),
        ConsistencyValidator()
    ]