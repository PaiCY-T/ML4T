"""
Business Logic Validator for Feature Selection.

Validates features against business rules, domain constraints,
and logical consistency for Taiwan stock market applications.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Business logic validation severity levels."""
    CRITICAL = "critical"  # Must fix - breaks business logic
    HIGH = "high"         # Should fix - questionable business logic  
    MEDIUM = "medium"     # May fix - suboptimal business logic
    LOW = "low"          # Nice to fix - minor optimization
    INFO = "info"        # Informational - no action needed


class LogicCategory(Enum):
    """Business logic validation categories."""
    MATHEMATICAL = "mathematical"      # Mathematical consistency
    DOMAIN_BOUNDS = "domain_bounds"    # Value range validity  
    TEMPORAL = "temporal"             # Time-series logic
    CROSS_SECTIONAL = "cross_sectional"  # Cross-asset logic
    SECTOR_LOGIC = "sector_logic"     # Sector-specific rules
    RISK_LOGIC = "risk_logic"         # Risk management rules
    REGULATORY = "regulatory"         # Regulatory constraints
    DATA_QUALITY = "data_quality"     # Data integrity rules


@dataclass
class BusinessRule:
    """Represents a business logic validation rule."""
    rule_id: str
    name: str
    category: LogicCategory
    severity: ValidationSeverity
    description: str
    validation_function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    taiwan_specific: bool = False


@dataclass
class ValidationResult:
    """Result of business logic validation."""
    feature_name: str
    rule_id: str
    category: LogicCategory
    severity: ValidationSeverity
    status: str  # "pass", "fail", "warning", "error"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    remediation: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BusinessLogicConfig:
    """Configuration for business logic validation."""
    
    # Mathematical bounds
    max_absolute_correlation: float = 0.95  # Maximum feature correlation
    min_variance: float = 1e-8              # Minimum feature variance
    max_skewness: float = 10.0              # Maximum distribution skewness  
    max_kurtosis: float = 50.0              # Maximum distribution kurtosis
    max_outlier_ratio: float = 0.05         # Maximum outlier percentage
    
    # Financial constraints
    max_return_threshold: float = 0.5       # 50% maximum single-period return
    min_price_threshold: float = 0.01       # Minimum valid price (TWD)
    max_price_threshold: float = 10000.0    # Maximum reasonable price (TWD)
    max_pe_ratio: float = 200.0             # Maximum P/E ratio
    max_pb_ratio: float = 50.0              # Maximum P/B ratio
    min_market_cap: float = 1e8             # Minimum market cap (TWD)
    max_leverage_ratio: float = 10.0        # Maximum debt/equity ratio
    
    # Taiwan market specifics
    max_daily_return: float = 0.10          # 10% daily price limit
    trading_days_per_year: int = 250        # Trading days in Taiwan
    market_hours_per_day: float = 4.5       # TSE trading hours
    min_trading_volume: float = 1000.0      # Minimum daily volume
    
    # Temporal constraints
    max_lookback_days: int = 252 * 5        # Maximum 5-year lookback
    min_data_points: int = 60               # Minimum observations
    max_missing_ratio: float = 0.10         # Maximum missing data ratio
    
    # Cross-sectional constraints
    min_universe_size: int = 50             # Minimum universe size
    max_concentration: float = 0.20         # Maximum single asset weight
    
    # Risk management
    max_portfolio_beta: float = 1.5         # Maximum portfolio beta
    max_tracking_error: float = 0.08        # Maximum tracking error
    min_sharpe_threshold: float = 0.5       # Minimum Sharpe ratio
    max_drawdown_threshold: float = 0.20    # Maximum drawdown
    
    # Regulatory constraints (Taiwan)
    foreign_ownership_limit: float = 0.30   # Foreign ownership limit
    margin_requirement: float = 0.50        # Margin requirement
    position_size_limit: float = 0.10       # Position size limit


class BaseBusinessValidator(ABC):
    """Base class for business logic validators."""
    
    def __init__(self, config: BusinessLogicConfig):
        self.config = config
    
    @abstractmethod
    def validate_feature(
        self,
        feature_name: str,
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate feature against business rules."""
        pass
    
    @abstractmethod
    def get_validation_category(self) -> LogicCategory:
        """Get the validation category this validator handles."""
        pass


class MathematicalValidator(BaseBusinessValidator):
    """Validates mathematical consistency and bounds."""
    
    def validate_feature(
        self,
        feature_name: str,
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate mathematical properties."""
        
        results = []
        
        if feature_data is None or feature_name not in feature_data.columns:
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="MATH-001",
                category=LogicCategory.MATHEMATICAL,
                severity=ValidationSeverity.CRITICAL,
                status="error",
                message="Feature data not available for validation",
                remediation=["Provide feature data for mathematical validation"]
            ))
            return results
        
        feature_series = feature_data[feature_name]
        
        # Remove NaN values for analysis
        valid_data = feature_series.dropna()
        
        if len(valid_data) == 0:
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="MATH-002",
                category=LogicCategory.MATHEMATICAL,
                severity=ValidationSeverity.CRITICAL,
                status="fail",
                message="Feature contains no valid data",
                remediation=["Check data generation process", "Verify data sources"]
            ))
            return results
        
        # 1. Variance check
        variance = valid_data.var()
        if variance < self.config.min_variance:
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="MATH-003",
                category=LogicCategory.MATHEMATICAL,
                severity=ValidationSeverity.HIGH,
                status="fail",
                message=f"Feature variance too low: {variance:.2e}",
                details={"variance": variance, "threshold": self.config.min_variance},
                remediation=["Check for constant or near-constant values", "Consider feature transformation"]
            ))
        
        # 2. Infinite/NaN check
        inf_count = np.isinf(feature_series).sum()
        nan_count = feature_series.isna().sum()
        total_count = len(feature_series)
        
        if inf_count > 0:
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="MATH-004",
                category=LogicCategory.MATHEMATICAL,
                severity=ValidationSeverity.CRITICAL,
                status="fail",
                message=f"Feature contains {inf_count} infinite values",
                details={"infinite_count": inf_count, "total_count": total_count},
                remediation=["Handle division by zero", "Apply proper data cleaning"]
            ))
        
        nan_ratio = nan_count / total_count
        if nan_ratio > self.config.max_missing_ratio:
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="MATH-005",
                category=LogicCategory.MATHEMATICAL,
                severity=ValidationSeverity.MEDIUM,
                status="warning",
                message=f"High missing data ratio: {nan_ratio:.1%}",
                details={"missing_ratio": nan_ratio, "threshold": self.config.max_missing_ratio},
                remediation=["Improve data coverage", "Consider imputation methods"]
            ))
        
        # 3. Distribution checks
        if len(valid_data) >= 30:  # Need sufficient data for distribution analysis
            skewness = stats.skew(valid_data)
            kurtosis = stats.kurtosis(valid_data)
            
            if abs(skewness) > self.config.max_skewness:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="MATH-006",
                    category=LogicCategory.MATHEMATICAL,
                    severity=ValidationSeverity.MEDIUM,
                    status="warning",
                    message=f"Extreme skewness detected: {skewness:.2f}",
                    details={"skewness": skewness, "threshold": self.config.max_skewness},
                    remediation=["Consider log transformation", "Apply winsorization", "Check for outliers"]
                ))
            
            if kurtosis > self.config.max_kurtosis:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="MATH-007",
                    category=LogicCategory.MATHEMATICAL,
                    severity=ValidationSeverity.MEDIUM,
                    status="warning",
                    message=f"Extreme kurtosis detected: {kurtosis:.2f}",
                    details={"kurtosis": kurtosis, "threshold": self.config.max_kurtosis},
                    remediation=["Investigate extreme values", "Consider robust transformations"]
                ))
        
        # 4. Outlier detection
        if len(valid_data) >= 10:
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                outlier_bounds = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
                outliers = (valid_data < outlier_bounds[0]) | (valid_data > outlier_bounds[1])
                outlier_ratio = outliers.sum() / len(valid_data)
                
                if outlier_ratio > self.config.max_outlier_ratio:
                    results.append(ValidationResult(
                        feature_name=feature_name,
                        rule_id="MATH-008",
                        category=LogicCategory.MATHEMATICAL,
                        severity=ValidationSeverity.MEDIUM,
                        status="warning",
                        message=f"High outlier ratio: {outlier_ratio:.1%}",
                        details={
                            "outlier_ratio": outlier_ratio,
                            "threshold": self.config.max_outlier_ratio,
                            "outlier_bounds": outlier_bounds
                        },
                        remediation=["Apply outlier treatment", "Consider winsorization"]
                    ))
        
        # If no issues found, add success result
        if not any(r.status == "fail" for r in results):
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="MATH-000",
                category=LogicCategory.MATHEMATICAL,
                severity=ValidationSeverity.INFO,
                status="pass",
                message="Mathematical properties are valid",
                details={
                    "variance": variance,
                    "skewness": stats.skew(valid_data) if len(valid_data) >= 30 else None,
                    "kurtosis": stats.kurtosis(valid_data) if len(valid_data) >= 30 else None
                }
            ))
        
        return results
    
    def get_validation_category(self) -> LogicCategory:
        return LogicCategory.MATHEMATICAL


class DomainBoundsValidator(BaseBusinessValidator):
    """Validates domain-specific value bounds and constraints."""
    
    def validate_feature(
        self,
        feature_name: str,
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate domain bounds."""
        
        results = []
        
        if feature_data is None or feature_name not in feature_data.columns:
            return [ValidationResult(
                feature_name=feature_name,
                rule_id="DOMAIN-001",
                category=LogicCategory.DOMAIN_BOUNDS,
                severity=ValidationSeverity.CRITICAL,
                status="error",
                message="Feature data not available for domain validation"
            )]
        
        feature_series = feature_data[feature_name]
        valid_data = feature_series.dropna()
        
        if len(valid_data) == 0:
            return [ValidationResult(
                feature_name=feature_name,
                rule_id="DOMAIN-002",
                category=LogicCategory.DOMAIN_BOUNDS,
                severity=ValidationSeverity.CRITICAL,
                status="fail",
                message="No valid data for domain validation"
            )]
        
        feature_lower = feature_name.lower()
        
        # 1. Return-based features
        if any(keyword in feature_lower for keyword in ['return', 'ret_', 'pct_change']):
            max_return = valid_data.max()
            min_return = valid_data.min()
            
            if max_return > self.config.max_return_threshold:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="DOMAIN-003",
                    category=LogicCategory.DOMAIN_BOUNDS,
                    severity=ValidationSeverity.HIGH,
                    status="fail",
                    message=f"Extreme positive return: {max_return:.1%}",
                    details={"max_return": max_return, "threshold": self.config.max_return_threshold},
                    remediation=["Check for data errors", "Apply return capping", "Verify price data"]
                ))
            
            if min_return < -self.config.max_return_threshold:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="DOMAIN-004",
                    category=LogicCategory.DOMAIN_BOUNDS,
                    severity=ValidationSeverity.HIGH,
                    status="fail",
                    message=f"Extreme negative return: {min_return:.1%}",
                    details={"min_return": min_return, "threshold": -self.config.max_return_threshold},
                    remediation=["Check for data errors", "Apply return capping", "Verify price data"]
                ))
        
        # 2. Price-based features
        if any(keyword in feature_lower for keyword in ['price', 'close', 'open', 'high', 'low']):
            min_price = valid_data.min()
            max_price = valid_data.max()
            
            if min_price < self.config.min_price_threshold:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="DOMAIN-005",
                    category=LogicCategory.DOMAIN_BOUNDS,
                    severity=ValidationSeverity.MEDIUM,
                    status="warning",
                    message=f"Very low price detected: {min_price:.4f} TWD",
                    details={"min_price": min_price, "threshold": self.config.min_price_threshold},
                    remediation=["Check for stock splits", "Verify data quality", "Consider penny stock filtering"]
                ))
            
            if max_price > self.config.max_price_threshold:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="DOMAIN-006",
                    category=LogicCategory.DOMAIN_BOUNDS,
                    severity=ValidationSeverity.MEDIUM,
                    status="warning",
                    message=f"Very high price detected: {max_price:.2f} TWD",
                    details={"max_price": max_price, "threshold": self.config.max_price_threshold},
                    remediation=["Check for data errors", "Verify unusual price movements"]
                ))
        
        # 3. Ratio-based features
        if 'pe' in feature_lower and 'ratio' in feature_lower:
            max_pe = valid_data.max()
            if max_pe > self.config.max_pe_ratio:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="DOMAIN-007",
                    category=LogicCategory.DOMAIN_BOUNDS,
                    severity=ValidationSeverity.MEDIUM,
                    status="warning",
                    message=f"Extreme P/E ratio: {max_pe:.1f}",
                    details={"max_pe": max_pe, "threshold": self.config.max_pe_ratio},
                    remediation=["Apply P/E capping", "Filter loss-making companies", "Check earnings data"]
                ))
        
        if 'pb' in feature_lower and 'ratio' in feature_lower:
            max_pb = valid_data.max()
            if max_pb > self.config.max_pb_ratio:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="DOMAIN-008",
                    category=LogicCategory.DOMAIN_BOUNDS,
                    severity=ValidationSeverity.MEDIUM,
                    status="warning",
                    message=f"Extreme P/B ratio: {max_pb:.1f}",
                    details={"max_pb": max_pb, "threshold": self.config.max_pb_ratio},
                    remediation=["Apply P/B capping", "Check book value data", "Consider asset quality"]
                ))
        
        # 4. Volume-based features
        if any(keyword in feature_lower for keyword in ['volume', 'turnover', 'shares']):
            min_volume = valid_data.min()
            if min_volume < self.config.min_trading_volume:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="DOMAIN-009",
                    category=LogicCategory.DOMAIN_BOUNDS,
                    severity=ValidationSeverity.LOW,
                    status="warning",
                    message=f"Low trading volume detected: {min_volume:.0f}",
                    details={"min_volume": min_volume, "threshold": self.config.min_trading_volume},
                    remediation=["Consider liquidity filtering", "Check volume data quality"]
                ))
        
        # 5. Beta-based features
        if 'beta' in feature_lower:
            max_beta = abs(valid_data).max()
            if max_beta > 3.0:  # Extreme beta
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="DOMAIN-010",
                    category=LogicCategory.DOMAIN_BOUNDS,
                    severity=ValidationSeverity.MEDIUM,
                    status="warning",
                    message=f"Extreme beta detected: {max_beta:.2f}",
                    details={"max_beta": max_beta},
                    remediation=["Apply beta winsorization", "Check calculation method", "Verify benchmark data"]
                ))
        
        # Success case
        if not results:
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="DOMAIN-000",
                category=LogicCategory.DOMAIN_BOUNDS,
                severity=ValidationSeverity.INFO,
                status="pass",
                message="Domain bounds are valid"
            ))
        
        return results
    
    def get_validation_category(self) -> LogicCategory:
        return LogicCategory.DOMAIN_BOUNDS


class TemporalValidator(BaseBusinessValidator):
    """Validates temporal logic and time-series consistency."""
    
    def validate_feature(
        self,
        feature_name: str,
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate temporal logic."""
        
        results = []
        
        if feature_data is None or feature_name not in feature_data.columns:
            return [ValidationResult(
                feature_name=feature_name,
                rule_id="TEMP-001",
                category=LogicCategory.TEMPORAL,
                severity=ValidationSeverity.CRITICAL,
                status="error",
                message="Feature data not available for temporal validation"
            )]
        
        feature_series = feature_data[feature_name]
        
        # Check if index is datetime
        if not isinstance(feature_series.index, pd.DatetimeIndex):
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="TEMP-002",
                category=LogicCategory.TEMPORAL,
                severity=ValidationSeverity.HIGH,
                status="fail",
                message="Feature data requires datetime index for temporal validation",
                remediation=["Set datetime index", "Ensure proper time alignment"]
            ))
            return results
        
        # 1. Data frequency consistency
        if len(feature_series) > 1:
            time_diffs = feature_series.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                mode_diff = time_diffs.mode()
                if len(mode_diff) > 0:
                    expected_freq = mode_diff.iloc[0]
                    inconsistent_freq = (time_diffs != expected_freq).sum()
                    
                    if inconsistent_freq / len(time_diffs) > 0.05:  # More than 5% inconsistent
                        results.append(ValidationResult(
                            feature_name=feature_name,
                            rule_id="TEMP-003",
                            category=LogicCategory.TEMPORAL,
                            severity=ValidationSeverity.MEDIUM,
                            status="warning",
                            message=f"Inconsistent time frequency detected: {inconsistent_freq} irregular intervals",
                            details={
                                "expected_frequency": str(expected_freq),
                                "inconsistent_count": inconsistent_freq,
                                "total_intervals": len(time_diffs)
                            },
                            remediation=["Check data collection process", "Apply time resampling", "Fill missing dates"]
                        ))
        
        # 2. Weekend/holiday data check
        if len(feature_series) > 0:
            # Check for weekend data (should be minimal for stock data)
            weekend_data = feature_series.index.weekday >= 5  # Saturday=5, Sunday=6
            weekend_count = weekend_data.sum()
            
            if weekend_count > 0:
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="TEMP-004",
                    category=LogicCategory.TEMPORAL,
                    severity=ValidationSeverity.LOW,
                    status="warning",
                    message=f"Weekend data detected: {weekend_count} observations",
                    details={"weekend_count": weekend_count, "total_count": len(feature_series)},
                    remediation=["Filter weekend data", "Check data source", "Apply business day calendar"]
                ))
        
        # 3. Data staleness check
        if len(feature_series) > 0:
            latest_date = feature_series.index.max()
            days_since_latest = (datetime.now() - latest_date).days
            
            if days_since_latest > 30:  # More than 30 days old
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="TEMP-005",
                    category=LogicCategory.TEMPORAL,
                    severity=ValidationSeverity.MEDIUM,
                    status="warning",
                    message=f"Stale data detected: {days_since_latest} days since latest observation",
                    details={"latest_date": latest_date, "days_since_latest": days_since_latest},
                    remediation=["Update data source", "Check data pipeline", "Verify data availability"]
                ))
        
        # 4. Sufficient history check
        valid_data = feature_series.dropna()
        if len(valid_data) < self.config.min_data_points:
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="TEMP-006",
                category=LogicCategory.TEMPORAL,
                severity=ValidationSeverity.HIGH,
                status="fail",
                message=f"Insufficient data points: {len(valid_data)}",
                details={"available_points": len(valid_data), "required_points": self.config.min_data_points},
                remediation=["Extend data history", "Check data availability", "Consider feature redesign"]
            ))
        
        # 5. Look-ahead bias check
        feature_lower = feature_name.lower()
        lookahead_keywords = ['future', 'forward', 'next', 'lead', 'ahead']
        if any(keyword in feature_lower for keyword in lookahead_keywords):
            # Check if proper lag is indicated
            lag_keywords = ['lag', 'prev', 'delayed', 't-']
            if not any(keyword in feature_lower for keyword in lag_keywords):
                results.append(ValidationResult(
                    feature_name=feature_name,
                    rule_id="TEMP-007",
                    category=LogicCategory.TEMPORAL,
                    severity=ValidationSeverity.CRITICAL,
                    status="fail",
                    message="Potential look-ahead bias detected",
                    details={"suspicious_keywords": [k for k in lookahead_keywords if k in feature_lower]},
                    remediation=["Apply appropriate lag", "Check calculation timing", "Verify data availability"]
                ))
        
        # Success case
        if not any(r.status == "fail" for r in results):
            results.append(ValidationResult(
                feature_name=feature_name,
                rule_id="TEMP-000",
                category=LogicCategory.TEMPORAL,
                severity=ValidationSeverity.INFO,
                status="pass",
                message="Temporal logic is valid",
                details={"data_points": len(valid_data), "date_range": f"{feature_series.index.min()} to {feature_series.index.max()}"}
            ))
        
        return results
    
    def get_validation_category(self) -> LogicCategory:
        return LogicCategory.TEMPORAL


class BusinessLogicValidator:
    """
    Business Logic Validator for Feature Selection.
    
    Validates features against comprehensive business rules including:
    1. Mathematical consistency
    2. Domain bounds validation  
    3. Temporal logic validation
    4. Cross-sectional consistency
    5. Taiwan market-specific rules
    """
    
    def __init__(self, config: Optional[BusinessLogicConfig] = None):
        """Initialize business logic validator.
        
        Args:
            config: Configuration for business logic validation
        """
        self.config = config or BusinessLogicConfig()
        
        # Initialize validators
        self.validators = [
            MathematicalValidator(self.config),
            DomainBoundsValidator(self.config),
            TemporalValidator(self.config)
        ]
        
        self.validation_results: List[ValidationResult] = []
        
        logger.info("Business Logic Validator initialized")
        logger.info(f"Loaded {len(self.validators)} validator categories")
    
    def validate_features(
        self,
        features: List[str],
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[ValidationResult]]:
        """
        Validate features against business logic rules.
        
        Args:
            features: List of feature names to validate
            feature_data: Feature data for validation
            market_data: Market data for context
            feature_metadata: Feature metadata
            
        Returns:
            Dictionary mapping feature names to validation results
        """
        logger.info(f"Validating {len(features)} features against business logic")
        
        all_results = {}
        self.validation_results = []
        
        for feature in features:
            feature_results = []
            
            # Run all validators on this feature
            for validator in self.validators:
                try:
                    validator_results = validator.validate_feature(
                        feature, feature_data, market_data, feature_metadata
                    )
                    feature_results.extend(validator_results)
                    self.validation_results.extend(validator_results)
                    
                except Exception as e:
                    logger.error(f"Error in {validator.__class__.__name__} for feature {feature}: {e}")
                    
                    error_result = ValidationResult(
                        feature_name=feature,
                        rule_id=f"{validator.get_validation_category().value.upper()}-ERR",
                        category=validator.get_validation_category(),
                        severity=ValidationSeverity.CRITICAL,
                        status="error",
                        message=f"Validation error: {str(e)}",
                        details={"error": str(e)},
                        remediation=["Check validator implementation", "Verify input data"]
                    )
                    
                    feature_results.append(error_result)
                    self.validation_results.append(error_result)
            
            all_results[feature] = feature_results
        
        # Log validation summary
        self._log_validation_summary(all_results)
        
        return all_results
    
    def _log_validation_summary(self, results: Dict[str, List[ValidationResult]]) -> None:
        """Log validation summary statistics."""
        
        if not results:
            return
        
        total_features = len(results)
        total_validations = sum(len(feature_results) for feature_results in results.values())
        
        # Count by severity
        severity_counts = {severity: 0 for severity in ValidationSeverity}
        status_counts = {"pass": 0, "fail": 0, "warning": 0, "error": 0}
        
        for feature_results in results.values():
            for result in feature_results:
                severity_counts[result.severity] += 1
                status_counts[result.status] += 1
        
        # Calculate success rate
        success_rate = (status_counts["pass"] + status_counts["warning"]) / total_validations if total_validations > 0 else 0
        
        logger.info(f"Business Logic Validation Summary:")
        logger.info(f"  Total Features: {total_features}")
        logger.info(f"  Total Validations: {total_validations}")
        logger.info(f"  Success Rate: {success_rate:.1%}")
        logger.info(f"  Status Distribution:")
        for status, count in status_counts.items():
            if count > 0:
                logger.info(f"    {status.title()}: {count} ({count/total_validations:.1%})")
        
        # Log critical failures
        critical_failures = [
            result for feature_results in results.values()
            for result in feature_results
            if result.severity == ValidationSeverity.CRITICAL and result.status in ["fail", "error"]
        ]
        
        if critical_failures:
            logger.warning(f"Found {len(critical_failures)} critical business logic violations:")
            for failure in critical_failures[:5]:  # Log first 5
                logger.warning(f"  {failure.feature_name}: {failure.message}")
    
    def get_valid_features(
        self,
        results: Dict[str, List[ValidationResult]],
        max_severity: ValidationSeverity = ValidationSeverity.MEDIUM
    ) -> List[str]:
        """Get features that pass business logic validation."""
        
        valid_features = []
        
        for feature, feature_results in results.items():
            # Check if feature has any critical failures
            has_critical_failure = any(
                result.severity in [ValidationSeverity.CRITICAL] and result.status in ["fail", "error"]
                for result in feature_results
            )
            
            # Check if feature has failures above threshold
            has_severe_failure = any(
                result.severity.value <= max_severity.value and result.status in ["fail", "error"]
                for result in feature_results
            )
            
            if not (has_critical_failure or has_severe_failure):
                valid_features.append(feature)
        
        logger.info(f"Found {len(valid_features)} features passing business logic validation")
        
        return valid_features
    
    def get_validation_scores(
        self,
        results: Dict[str, List[ValidationResult]]
    ) -> Dict[str, float]:
        """Calculate validation scores for features (0-1 scale)."""
        
        feature_scores = {}
        
        # Scoring weights by severity and status
        severity_weights = {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.HIGH: 0.8,
            ValidationSeverity.MEDIUM: 0.6,
            ValidationSeverity.LOW: 0.4,
            ValidationSeverity.INFO: 0.0
        }
        
        status_penalties = {
            "fail": 1.0,
            "error": 1.0,
            "warning": 0.3,
            "pass": 0.0
        }
        
        for feature, feature_results in results.items():
            if not feature_results:
                feature_scores[feature] = 0.0
                continue
            
            # Calculate penalty score
            total_penalty = 0.0
            max_possible_penalty = 0.0
            
            for result in feature_results:
                severity_weight = severity_weights[result.severity]
                status_penalty = status_penalties[result.status]
                
                penalty = severity_weight * status_penalty
                total_penalty += penalty
                max_possible_penalty += severity_weight  # Maximum possible penalty for this validation
            
            # Convert penalty to score (1 - normalized_penalty)
            if max_possible_penalty > 0:
                normalized_penalty = total_penalty / max_possible_penalty
                score = max(0.0, 1.0 - normalized_penalty)
            else:
                score = 1.0
            
            feature_scores[feature] = score
        
        return feature_scores
    
    def generate_validation_report(
        self,
        results: Dict[str, List[ValidationResult]],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate comprehensive business logic validation report."""
        
        report_data = []
        
        for feature, feature_results in results.items():
            for result in feature_results:
                report_data.append({
                    'Feature': result.feature_name,
                    'Rule_ID': result.rule_id,
                    'Category': result.category.value,
                    'Severity': result.severity.value,
                    'Status': result.status,
                    'Message': result.message,
                    'Details': str(result.details),
                    'Remediation': '; '.join(result.remediation),
                    'Timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        report_df = pd.DataFrame(report_data)
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            logger.info(f"Business logic validation report saved to {output_path}")
        
        return report_df
    
    def get_category_summary(
        self,
        results: Dict[str, List[ValidationResult]]
    ) -> Dict[str, Dict[str, Any]]:
        """Get validation summary by category."""
        
        category_summary = {}
        
        for category in LogicCategory:
            category_results = [
                result for feature_results in results.values()
                for result in feature_results
                if result.category == category
            ]
            
            if category_results:
                status_counts = {}
                for status in ["pass", "fail", "warning", "error"]:
                    status_counts[status] = sum(1 for r in category_results if r.status == status)
                
                severity_counts = {}
                for severity in ValidationSeverity:
                    severity_counts[severity.value] = sum(1 for r in category_results if r.severity == severity)
                
                category_summary[category.value] = {
                    'total_validations': len(category_results),
                    'status_distribution': status_counts,
                    'severity_distribution': severity_counts,
                    'success_rate': (status_counts['pass'] + status_counts['warning']) / len(category_results) if category_results else 0
                }
        
        return category_summary
    
    def get_remediation_suggestions(
        self,
        results: Dict[str, List[ValidationResult]],
        feature: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get remediation suggestions for features."""
        
        suggestions = {}
        
        target_results = results if feature is None else {feature: results.get(feature, [])}
        
        for feat, feature_results in target_results.items():
            feat_suggestions = []
            
            for result in feature_results:
                if result.status in ["fail", "error"] and result.remediation:
                    feat_suggestions.extend(result.remediation)
            
            # Remove duplicates while preserving order
            unique_suggestions = []
            for suggestion in feat_suggestions:
                if suggestion not in unique_suggestions:
                    unique_suggestions.append(suggestion)
            
            if unique_suggestions:
                suggestions[feat] = unique_suggestions
        
        return suggestions