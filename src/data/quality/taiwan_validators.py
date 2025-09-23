"""
Taiwan Market-Specific Data Quality Validators.

This module implements Taiwan Stock Exchange (TWSE) specific validation rules
including price limits, volume checks, trading hours, and settlement validation.
"""

import asyncio
import logging
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.temporal import TemporalValue, DataType
from ..models.taiwan_market import (
    TaiwanMarketData, TaiwanSettlement, TaiwanTradingCalendar,
    TradingStatus, MarketSession, CorporateActionType
)
from .validation_engine import (
    ValidationPlugin, ValidationContext, ValidationOutput, ValidationResult,
    ValidationPriority
)
from .validators import QualityIssue, SeverityLevel, QualityCheckType

logger = logging.getLogger(__name__)


class TaiwanFundamentalLagValidator(ValidationPlugin):
    """Validates 60-day fundamental data lag for Taiwan market regulations."""
    
    def __init__(self, quarterly_lag_days: int = 60, annual_lag_days: int = 90):
        self.quarterly_lag_days = quarterly_lag_days
        self.annual_lag_days = annual_lag_days
    
    @property
    def name(self) -> str:
        return "taiwan_fundamental_lag_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.CRITICAL
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.FUNDAMENTAL}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return (
            value.data_type == DataType.FUNDAMENTAL and
            isinstance(value.value, dict) and
            value.symbol is not None
        )
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Validate fundamental data reporting lag compliance."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            # Extract fundamental data metadata
            fundamental_data = value.value
            fiscal_quarter = fundamental_data.get("fiscal_quarter")
            report_date = value.value_date
            announcement_date = value.as_of_date
            
            # Calculate actual lag
            actual_lag_days = (announcement_date - report_date).days
            
            # Determine required lag based on reporting period
            if fiscal_quarter == 4:  # Annual report
                required_lag_days = self.annual_lag_days
                period_type = "annual"
            else:  # Quarterly report
                required_lag_days = self.quarterly_lag_days
                period_type = "quarterly"
            
            # Validate lag compliance
            if actual_lag_days > required_lag_days:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.TEMPORAL_CONSISTENCY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Fundamental data lag {actual_lag_days} days exceeds Taiwan regulatory requirement of {required_lag_days} days for {period_type} report",
                    details={
                        "actual_lag_days": actual_lag_days,
                        "required_lag_days": required_lag_days,
                        "period_type": period_type,
                        "fiscal_quarter": fiscal_quarter,
                        "report_date": report_date.isoformat(),
                        "announcement_date": announcement_date.isoformat()
                    },
                    suggested_action="Verify announcement date and regulatory filing timeline"
                ))
            elif actual_lag_days < 0:
                # Future data - critical violation
                issues.append(QualityIssue(
                    check_type=QualityCheckType.TEMPORAL_CONSISTENCY,
                    severity=SeverityLevel.CRITICAL,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Fundamental data has negative lag: announcement date before report date",
                    details={
                        "actual_lag_days": actual_lag_days,
                        "report_date": report_date.isoformat(),
                        "announcement_date": announcement_date.isoformat()
                    },
                    suggested_action="Correct announcement date - cannot be before report date"
                ))
            
            result = ValidationResult.PASS if not issues else ValidationResult.FAIL
            
        except Exception as e:
            logger.error(f"Error in Taiwan fundamental lag validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
            result = ValidationResult.FAIL
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )


class TaiwanVolumeValidator(ValidationPlugin):
    """Validates Taiwan market volume anomalies and spikes."""
    
    def __init__(self, volume_spike_threshold: float = 5.0, 
                 zero_volume_severity: str = "warning"):
        self.volume_spike_threshold = volume_spike_threshold
        self.zero_volume_severity = zero_volume_severity
        self._volume_history_cache: Dict[str, List[Tuple[date, int]]] = {}
    
    @property
    def name(self) -> str:
        return "taiwan_volume_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.HIGH
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.VOLUME, DataType.MARKET_DATA}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return value.data_type in self.supported_data_types and value.symbol is not None
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Validate volume data for anomalies."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            # Extract volume data
            current_volume = None
            
            if value.data_type == DataType.VOLUME:
                current_volume = int(value.value) if value.value is not None else None
            elif value.data_type == DataType.MARKET_DATA and isinstance(value.value, dict):
                current_volume = value.value.get("volume")
                if current_volume is not None:
                    current_volume = int(current_volume)
            
            if current_volume is None:
                return ValidationOutput(
                    validator_name=self.name,
                    validation_id=f"{self.name}_{int(start_time.timestamp())}",
                    result=ValidationResult.SKIP,
                    metadata={"reason": "No volume data found"}
                )
            
            # Check for negative volume
            if current_volume < 0:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.VALIDITY,
                    severity=SeverityLevel.CRITICAL,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Negative volume detected: {current_volume}",
                    details={"volume": current_volume},
                    suggested_action="Correct volume data - cannot be negative"
                ))
            
            # Check for zero volume
            if current_volume == 0:
                severity = SeverityLevel.WARNING if self.zero_volume_severity == "warning" else SeverityLevel.ERROR
                issues.append(QualityIssue(
                    check_type=QualityCheckType.BUSINESS_RULES,
                    severity=severity,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description="Zero volume detected - potential trading suspension",
                    details={"volume": current_volume},
                    suggested_action="Verify trading status and market conditions"
                ))
            
            # Calculate volume spike if historical data available
            avg_volume = await self._get_average_volume(value.symbol, value.value_date, context)
            
            if avg_volume and avg_volume > 0 and current_volume > 0:
                volume_ratio = current_volume / avg_volume
                
                if volume_ratio > self.volume_spike_threshold:
                    severity = SeverityLevel.ERROR if volume_ratio > self.volume_spike_threshold * 2 else SeverityLevel.WARNING
                    
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.ANOMALY_DETECTION,
                        severity=severity,
                        symbol=value.symbol,
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Volume spike detected: {volume_ratio:.1f}x average volume",
                        details={
                            "current_volume": current_volume,
                            "average_volume": avg_volume,
                            "volume_ratio": volume_ratio,
                            "threshold": self.volume_spike_threshold
                        },
                        suggested_action="Investigate market news and trading conditions"
                    ))
            
            result = ValidationResult.PASS if not issues else ValidationResult.WARNING
            
        except Exception as e:
            logger.error(f"Error in Taiwan volume validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
            result = ValidationResult.FAIL
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )
    
    async def _get_average_volume(self, symbol: str, current_date: date, 
                                context: ValidationContext) -> Optional[float]:
        """Calculate 20-day average volume."""
        volumes = []
        
        # Look for historical data in context
        if context.historical_data:
            for hist_value in context.historical_data:
                if (hist_value.symbol == symbol and 
                    hist_value.data_type in [DataType.VOLUME, DataType.MARKET_DATA] and
                    hist_value.value_date < current_date):
                    
                    volume = None
                    if hist_value.data_type == DataType.VOLUME:
                        volume = int(hist_value.value) if hist_value.value else None
                    elif isinstance(hist_value.value, dict):
                        volume = hist_value.value.get("volume")
                    
                    if volume and volume > 0:
                        volumes.append((hist_value.value_date, volume))
        
        # Sort by date and take last 20 trading days
        volumes.sort(key=lambda x: x[0], reverse=True)
        recent_volumes = [v[1] for v in volumes[:20]]
        
        if len(recent_volumes) >= 5:  # Need at least 5 days of data
            return sum(recent_volumes) / len(recent_volumes)
        
        return None


class TaiwanDataCompletenessValidator(ValidationPlugin):
    """Validates data completeness for Taiwan market data."""
    
    def __init__(self, required_fields_by_type: Optional[Dict[DataType, Set[str]]] = None):
        self.required_fields = required_fields_by_type or {
            DataType.PRICE: {"close_price"},
            DataType.VOLUME: {"volume"},
            DataType.MARKET_DATA: {"open_price", "high_price", "low_price", "close_price", "volume"},
            DataType.FUNDAMENTAL: {"revenue", "net_income", "fiscal_quarter", "fiscal_year"}
        }
    
    @property
    def name(self) -> str:
        return "taiwan_data_completeness_validator"
    
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
        return value.data_type in self.supported_data_types
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Validate data completeness."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            required_fields = self.required_fields.get(value.data_type, set())
            missing_fields = []
            
            if value.data_type in [DataType.MARKET_DATA, DataType.FUNDAMENTAL]:
                # For complex data types, check if required fields are present and not None
                if not isinstance(value.value, dict):
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.COMPLETENESS,
                        severity=SeverityLevel.ERROR,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Data value is not a dictionary for {value.data_type.value}",
                        suggested_action="Verify data structure format"
                    ))
                else:
                    for field in required_fields:
                        field_value = value.value.get(field)
                        if field_value is None or (isinstance(field_value, str) and field_value.strip() == ""):
                            missing_fields.append(field)
            else:
                # For simple data types, check if value is present
                if value.value is None:
                    missing_fields.extend(required_fields)
            
            # Report missing fields
            if missing_fields:
                severity = SeverityLevel.ERROR if len(missing_fields) > len(required_fields) // 2 else SeverityLevel.WARNING
                
                issues.append(QualityIssue(
                    check_type=QualityCheckType.COMPLETENESS,
                    severity=severity,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Missing required fields: {', '.join(missing_fields)}",
                    details={
                        "missing_fields": missing_fields,
                        "required_fields": list(required_fields),
                        "completeness_score": 1.0 - (len(missing_fields) / len(required_fields))
                    },
                    suggested_action="Verify data source and collection process"
                ))
            
            result = ValidationResult.PASS if not issues else ValidationResult.WARNING
            
        except Exception as e:
            logger.error(f"Error in Taiwan data completeness validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
            result = ValidationResult.FAIL
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )


class TaiwanDataConsistencyValidator(ValidationPlugin):
    """Validates internal data consistency for Taiwan market data."""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "taiwan_data_consistency_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.HIGH
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.MARKET_DATA, DataType.FUNDAMENTAL}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return (
            value.data_type in self.supported_data_types and
            isinstance(value.value, dict)
        )
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Validate internal data consistency."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            data = value.value
            
            if value.data_type == DataType.MARKET_DATA:
                issues.extend(await self._validate_market_data_consistency(data, value, context))
            elif value.data_type == DataType.FUNDAMENTAL:
                issues.extend(await self._validate_fundamental_consistency(data, value, context))
            
            result = ValidationResult.PASS if not issues else ValidationResult.WARNING
            
        except Exception as e:
            logger.error(f"Error in Taiwan data consistency validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
            result = ValidationResult.FAIL
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )
    
    async def _validate_market_data_consistency(self, data: Dict[str, Any], 
                                              value: TemporalValue, 
                                              context: ValidationContext) -> List[QualityIssue]:
        """Validate market data internal consistency."""
        issues = []
        
        # Extract price fields
        open_price = data.get("open_price")
        high_price = data.get("high_price")
        low_price = data.get("low_price")
        close_price = data.get("close_price")
        volume = data.get("volume")
        turnover = data.get("turnover")
        
        # Convert to Decimal for precise comparison
        def to_decimal(val):
            return Decimal(str(val)) if val is not None else None
        
        open_dec = to_decimal(open_price)
        high_dec = to_decimal(high_price)
        low_dec = to_decimal(low_price)
        close_dec = to_decimal(close_price)
        
        # Check OHLC relationships
        if high_dec and low_dec and high_dec < low_dec:
            issues.append(QualityIssue(
                check_type=QualityCheckType.ACCURACY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"High price {high_price} is less than low price {low_price}",
                details={"high_price": float(high_dec), "low_price": float(low_dec)},
                suggested_action="Verify OHLC data integrity"
            ))
        
        # Check if open/close are within high/low range
        for price_name, price_val in [("open", open_dec), ("close", close_dec)]:
            if price_val and high_dec and low_dec:
                if price_val > high_dec or price_val < low_dec:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.ACCURACY,
                        severity=SeverityLevel.ERROR,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"{price_name.capitalize()} price {price_val} is outside high-low range [{low_price}, {high_price}]",
                        details={
                            f"{price_name}_price": float(price_val),
                            "high_price": float(high_dec),
                            "low_price": float(low_dec)
                        },
                        suggested_action="Verify price data accuracy"
                    ))
        
        # Check volume and turnover consistency
        if volume and turnover and close_dec:
            # Simple check: turnover should be roughly volume * average_price
            avg_price = (high_dec + low_dec + open_dec + close_dec) / 4 if all([high_dec, low_dec, open_dec, close_dec]) else close_dec
            expected_turnover = volume * avg_price
            turnover_decimal = to_decimal(turnover)
            
            # Allow 10% variance for turnover calculation
            if abs(turnover_decimal - expected_turnover) / expected_turnover > 0.1:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.WARNING,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Turnover {turnover} inconsistent with volume {volume} and average price {avg_price}",
                    details={
                        "volume": volume,
                        "turnover": float(turnover_decimal),
                        "expected_turnover": float(expected_turnover),
                        "variance_pct": float(abs(turnover_decimal - expected_turnover) / expected_turnover)
                    },
                    suggested_action="Verify volume and turnover calculation"
                ))
        
        return issues
    
    async def _validate_fundamental_consistency(self, data: Dict[str, Any], 
                                              value: TemporalValue, 
                                              context: ValidationContext) -> List[QualityIssue]:
        """Validate fundamental data internal consistency."""
        issues = []
        
        # Extract fundamental fields
        revenue = data.get("revenue")
        operating_income = data.get("operating_income")
        net_income = data.get("net_income")
        total_assets = data.get("total_assets")
        total_equity = data.get("total_equity")
        eps = data.get("eps")
        roe = data.get("roe")
        
        # Check logical relationships
        if operating_income and revenue:
            if operating_income > revenue:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Operating income {operating_income} exceeds revenue {revenue}",
                    details={"operating_income": operating_income, "revenue": revenue},
                    suggested_action="Verify financial statement data"
                ))
        
        if total_assets and total_equity:
            if total_equity > total_assets:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Total equity {total_equity} exceeds total assets {total_assets}",
                    details={"total_equity": total_equity, "total_assets": total_assets},
                    suggested_action="Verify balance sheet data"
                ))
        
        # Check ROE calculation if all components are available
        if roe and net_income and total_equity and total_equity != 0:
            calculated_roe = net_income / total_equity
            roe_variance = abs(roe - calculated_roe) / abs(calculated_roe) if calculated_roe != 0 else 1
            
            if roe_variance > 0.05:  # 5% tolerance
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.WARNING,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"ROE {roe:.4f} inconsistent with calculated ROE {calculated_roe:.4f}",
                    details={
                        "reported_roe": roe,
                        "calculated_roe": calculated_roe,
                        "net_income": net_income,
                        "total_equity": total_equity,
                        "variance_pct": roe_variance
                    },
                    suggested_action="Verify ROE calculation method"
                ))
        
        return issues


logger = logging.getLogger(__name__)


class TaiwanPriceLimitValidator(ValidationPlugin):
    """Validates Taiwan market price limits (10% daily limit)."""
    
    def __init__(self, daily_limit_pct: float = 0.10, enable_special_stocks: bool = True):
        self.daily_limit_pct = daily_limit_pct
        self.enable_special_stocks = enable_special_stocks
        
        # Special stocks with different limits (ETFs, etc.)
        self.special_limits = {
            "ETF": 0.15,      # ETFs have 15% limit
            "WARRANT": 0.25,  # Warrants have 25% limit
            "FUTURES": None,  # No daily limits
        }
        
        # Cache for previous closing prices
        self._previous_close_cache: Dict[Tuple[str, date], Decimal] = {}
    
    @property
    def name(self) -> str:
        return "taiwan_price_limit_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.CRITICAL
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.PRICE, DataType.MARKET_DATA}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return (
            value.data_type in self.supported_data_types and
            value.symbol is not None
        )
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Validate price against Taiwan market daily limits."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            # Extract price data
            current_price = None
            previous_close = None
            
            if value.data_type == DataType.PRICE:
                current_price = Decimal(str(value.value))
            elif value.data_type == DataType.MARKET_DATA and isinstance(value.value, dict):
                if "close_price" in value.value:
                    current_price = Decimal(str(value.value["close_price"]))
            
            if current_price is None:
                return ValidationOutput(
                    validator_name=self.name,
                    validation_id=f"{self.name}_{int(start_time.timestamp())}",
                    result=ValidationResult.SKIP,
                    metadata={"reason": "No price data found"}
                )
            
            # Get previous closing price
            previous_close = await self._get_previous_close(value.symbol, value.value_date, context)
            
            if previous_close is None:
                return ValidationOutput(
                    validator_name=self.name,
                    validation_id=f"{self.name}_{int(start_time.timestamp())}",
                    result=ValidationResult.SKIP,
                    metadata={"reason": "No previous close price available"}
                )
            
            # Determine applicable limit
            applicable_limit = self._get_applicable_limit(value.symbol, context)
            
            if applicable_limit is None:
                # No limit applies (e.g., futures)
                return ValidationOutput(
                    validator_name=self.name,
                    validation_id=f"{self.name}_{int(start_time.timestamp())}",
                    result=ValidationResult.PASS,
                    metadata={"reason": "No price limit applies"}
                )
            
            # Calculate price change
            price_change = abs(current_price - previous_close) / previous_close
            
            # Check if within limits
            if price_change > applicable_limit:
                # Check if this is a known corporate action day
                if self._is_corporate_action_day(value.symbol, value.value_date, context):
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.BUSINESS_RULES,
                        severity=SeverityLevel.INFO,
                        symbol=value.symbol,
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Price change {price_change:.2%} exceeds limit {applicable_limit:.2%} on corporate action day",
                        details={
                            "current_price": float(current_price),
                            "previous_close": float(previous_close),
                            "price_change_pct": float(price_change),
                            "limit_pct": float(applicable_limit),
                            "corporate_action": True
                        },
                        suggested_action="Verify corporate action adjustment"
                    ))
                else:
                    # Violation without known corporate action
                    severity = SeverityLevel.ERROR if price_change > applicable_limit * 1.2 else SeverityLevel.WARNING
                    
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.BUSINESS_RULES,
                        severity=severity,
                        symbol=value.symbol,
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Price change {price_change:.2%} exceeds daily limit {applicable_limit:.2%}",
                        details={
                            "current_price": float(current_price),
                            "previous_close": float(previous_close),
                            "price_change_pct": float(price_change),
                            "limit_pct": float(applicable_limit)
                        },
                        suggested_action="Verify price data and check for trading halts"
                    ))
            
            result = ValidationResult.PASS if not issues or all(i.severity == SeverityLevel.INFO for i in issues) else ValidationResult.WARNING
            
        except Exception as e:
            logger.error(f"Error in Taiwan price limit validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
            result = ValidationResult.FAIL
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )
    
    async def _get_previous_close(self, symbol: str, current_date: date, 
                                context: ValidationContext) -> Optional[Decimal]:
        """Get previous trading day's closing price."""
        # Check cache first
        cache_key = (symbol, current_date)
        if cache_key in self._previous_close_cache:
            return self._previous_close_cache[cache_key]
        
        # Look for historical data in context
        if context.historical_data:
            for hist_value in reversed(context.historical_data):
                if (hist_value.symbol == symbol and 
                    hist_value.data_type == DataType.PRICE and
                    hist_value.value_date < current_date):
                    
                    prev_close = Decimal(str(hist_value.value))
                    self._previous_close_cache[cache_key] = prev_close
                    return prev_close
        
        # TODO: In production, query temporal store for previous trading day
        # For now, return None to skip validation
        return None
    
    def _get_applicable_limit(self, symbol: str, context: ValidationContext) -> Optional[float]:
        """Get applicable price limit for symbol."""
        # Check metadata for stock type
        stock_type = context.get_metadata("stock_type", "STOCK")
        
        if self.enable_special_stocks and stock_type in self.special_limits:
            return self.special_limits[stock_type]
        
        # Default to standard limit
        return self.daily_limit_pct
    
    def _is_corporate_action_day(self, symbol: str, check_date: date, 
                               context: ValidationContext) -> bool:
        """Check if there's a corporate action on this date."""
        # Check metadata for corporate actions
        corporate_actions = context.get_metadata("corporate_actions", [])
        
        for action in corporate_actions:
            if (action.get("symbol") == symbol and 
                action.get("ex_date") == check_date):
                return True
        
        return False


class TaiwanVolumeValidator(ValidationPlugin):
    """Validates volume data for Taiwan market."""
    
    def __init__(self, spike_threshold: float = 5.0, lookback_days: int = 20):
        self.spike_threshold = spike_threshold  # X times average volume
        self.lookback_days = lookback_days
        self._volume_history: Dict[str, List[Tuple[date, int]]] = {}
    
    @property
    def name(self) -> str:
        return "taiwan_volume_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.HIGH
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.VOLUME, DataType.MARKET_DATA}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return (
            value.data_type in self.supported_data_types and
            value.symbol is not None
        )
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Validate volume data."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            # Extract volume
            current_volume = None
            
            if value.data_type == DataType.VOLUME:
                current_volume = int(value.value)
            elif value.data_type == DataType.MARKET_DATA and isinstance(value.value, dict):
                if "volume" in value.value:
                    current_volume = int(value.value["volume"])
            
            if current_volume is None:
                return ValidationOutput(
                    validator_name=self.name,
                    validation_id=f"{self.name}_{int(start_time.timestamp())}",
                    result=ValidationResult.SKIP,
                    metadata={"reason": "No volume data found"}
                )
            
            # Basic validation
            if current_volume < 0:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.ACCURACY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol,
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Negative volume: {current_volume}",
                    details={"volume": current_volume},
                    suggested_action="Check data source"
                ))
            
            # Calculate average volume for spike detection
            avg_volume = self._get_average_volume(value.symbol, value.value_date, context)
            
            if avg_volume and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                
                if volume_ratio > self.spike_threshold:
                    severity = SeverityLevel.WARNING if volume_ratio < self.spike_threshold * 2 else SeverityLevel.ERROR
                    
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.ANOMALY,
                        severity=severity,
                        symbol=value.symbol,
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Volume spike: {volume_ratio:.1f}x average volume",
                        details={
                            "current_volume": current_volume,
                            "average_volume": avg_volume,
                            "ratio": volume_ratio,
                            "threshold": self.spike_threshold
                        },
                        suggested_action="Check for news or corporate actions"
                    ))
                
                # Check for unusually low volume (possible data issue)
                elif volume_ratio < 0.01:  # Less than 1% of average
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.COMPLETENESS,
                        severity=SeverityLevel.WARNING,
                        symbol=value.symbol,
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Unusually low volume: {volume_ratio:.3f}x average",
                        details={
                            "current_volume": current_volume,
                            "average_volume": avg_volume,
                            "ratio": volume_ratio
                        },
                        suggested_action="Verify trading was active"
                    ))
            
            # Update volume history
            self._update_volume_history(value.symbol, value.value_date, current_volume)
            
            result = ValidationResult.PASS if not issues else ValidationResult.WARNING
            
        except Exception as e:
            logger.error(f"Error in Taiwan volume validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
            result = ValidationResult.FAIL
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )
    
    def _get_average_volume(self, symbol: str, current_date: date, 
                          context: ValidationContext) -> Optional[float]:
        """Calculate average volume over lookback period."""
        volumes = []
        
        # Get from historical data in context
        if context.historical_data:
            for hist_value in context.historical_data:
                if (hist_value.symbol == symbol and 
                    hist_value.data_type == DataType.VOLUME and
                    hist_value.value_date < current_date and
                    (current_date - hist_value.value_date).days <= self.lookback_days):
                    
                    volumes.append(int(hist_value.value))
        
        # Get from internal history
        if symbol in self._volume_history:
            for hist_date, volume in self._volume_history[symbol]:
                if (hist_date < current_date and
                    (current_date - hist_date).days <= self.lookback_days):
                    volumes.append(volume)
        
        return sum(volumes) / len(volumes) if volumes else None
    
    def _update_volume_history(self, symbol: str, trade_date: date, volume: int):
        """Update volume history for symbol."""
        if symbol not in self._volume_history:
            self._volume_history[symbol] = []
        
        history = self._volume_history[symbol]
        history.append((trade_date, volume))
        
        # Keep only recent history
        cutoff_date = trade_date - timedelta(days=self.lookback_days * 2)
        self._volume_history[symbol] = [
            (d, v) for d, v in history if d >= cutoff_date
        ]


class TaiwanTradingHoursValidator(ValidationPlugin):
    """Validates data timestamps against Taiwan trading hours (09:00-13:30 TST)."""
    
    def __init__(self):
        self.trading_start = time(9, 0)    # 09:00 TST
        self.trading_end = time(13, 30)    # 13:30 TST
        self.taiwan_timezone_offset = 8    # UTC+8
    
    @property
    def name(self) -> str:
        return "taiwan_trading_hours_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.MEDIUM
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.PRICE, DataType.VOLUME, DataType.MARKET_DATA}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return value.data_type in self.supported_data_types
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Validate timestamp against trading hours."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            # Check if value has timestamp info
            timestamp = None
            
            if hasattr(value, 'created_at') and value.created_at:
                timestamp = value.created_at
            elif isinstance(value.value, dict) and 'timestamp' in value.value:
                timestamp = value.value['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            if timestamp is None:
                # No timestamp to validate - skip
                return ValidationOutput(
                    validator_name=self.name,
                    validation_id=f"{self.name}_{int(start_time.timestamp())}",
                    result=ValidationResult.SKIP,
                    metadata={"reason": "No timestamp found"}
                )
            
            # Convert to Taiwan time (simplified - assumes UTC input)
            taiwan_time = timestamp + timedelta(hours=self.taiwan_timezone_offset)
            trade_time = taiwan_time.time()
            
            # Check if within trading hours
            if not (self.trading_start <= trade_time <= self.trading_end):
                # Check if it's a trading day
                is_trading_day = self._is_trading_day(value.value_date, context)
                
                if is_trading_day:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.TEMPORAL,
                        severity=SeverityLevel.WARNING,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Data timestamp {trade_time} outside trading hours {self.trading_start}-{self.trading_end}",
                        details={
                            "timestamp": timestamp.isoformat(),
                            "taiwan_time": taiwan_time.isoformat(),
                            "trade_time": trade_time.isoformat(),
                            "trading_start": self.trading_start.isoformat(),
                            "trading_end": self.trading_end.isoformat()
                        },
                        suggested_action="Verify data source timing"
                    ))
                else:
                    # Not a trading day - data should not exist
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.TEMPORAL,
                        severity=SeverityLevel.ERROR,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Trading data on non-trading day {value.value_date}",
                        details={
                            "timestamp": timestamp.isoformat(),
                            "taiwan_time": taiwan_time.isoformat(),
                            "is_trading_day": is_trading_day
                        },
                        suggested_action="Check trading calendar"
                    ))
            
            result = ValidationResult.PASS if not issues else ValidationResult.WARNING
            
        except Exception as e:
            logger.error(f"Error in Taiwan trading hours validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
            result = ValidationResult.FAIL
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )
    
    def _is_trading_day(self, check_date: date, context: ValidationContext) -> bool:
        """Check if date is a trading day."""
        # Check trading calendar from context
        if context.trading_calendar and check_date in context.trading_calendar:
            calendar_entry = context.trading_calendar[check_date]
            return getattr(calendar_entry, 'is_trading_day', True)
        
        # Default: weekdays are trading days (simplified)
        return check_date.weekday() < 5


class TaiwanSettlementValidator(ValidationPlugin):
    """Validates T+2 settlement rules for Taiwan market."""
    
    def __init__(self):
        self.settlement_days = 2  # T+2 for Taiwan
    
    @property
    def name(self) -> str:
        return "taiwan_settlement_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.MEDIUM
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.PRICE, DataType.VOLUME, DataType.MARKET_DATA}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return value.data_type in self.supported_data_types
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Validate settlement timing."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            # Check settlement metadata
            settlement_info = context.get_metadata("settlement_info")
            
            if settlement_info:
                if isinstance(settlement_info, dict):
                    settlement_date = settlement_info.get("settlement_date")
                    if isinstance(settlement_date, str):
                        settlement_date = date.fromisoformat(settlement_date)
                    
                    if settlement_date:
                        # Calculate expected settlement date
                        expected_settlement = self._calculate_settlement_date(
                            value.value_date, context
                        )
                        
                        if settlement_date != expected_settlement:
                            issues.append(QualityIssue(
                                check_type=QualityCheckType.BUSINESS_RULES,
                                severity=SeverityLevel.WARNING,
                                symbol=value.symbol or "",
                                data_type=value.data_type,
                                data_date=value.value_date,
                                issue_date=datetime.utcnow(),
                                description=f"Settlement date mismatch: expected {expected_settlement}, got {settlement_date}",
                                details={
                                    "trade_date": value.value_date.isoformat(),
                                    "expected_settlement": expected_settlement.isoformat(),
                                    "actual_settlement": settlement_date.isoformat()
                                },
                                suggested_action="Check settlement calculation"
                            ))
            
            # Validate as_of_date is not before settlement
            if value.data_type in [DataType.PRICE, DataType.VOLUME]:
                # These should be available immediately, not subject to settlement delay
                pass
            else:
                # Other data types might have settlement considerations
                expected_settlement = self._calculate_settlement_date(value.value_date, context)
                
                if value.as_of_date < expected_settlement:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.TEMPORAL,
                        severity=SeverityLevel.INFO,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Data available before settlement date",
                        details={
                            "as_of_date": value.as_of_date.isoformat(),
                            "settlement_date": expected_settlement.isoformat()
                        },
                        suggested_action="Verify data timing"
                    ))
            
            result = ValidationResult.PASS if not issues else ValidationResult.WARNING
            
        except Exception as e:
            logger.error(f"Error in Taiwan settlement validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
            result = ValidationResult.FAIL
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )
    
    def _calculate_settlement_date(self, trade_date: date, context: ValidationContext) -> date:
        """Calculate T+2 settlement date considering trading calendar."""
        current_date = trade_date
        business_days_added = 0
        
        while business_days_added < self.settlement_days:
            current_date += timedelta(days=1)
            
            # Check if it's a trading day
            if self._is_trading_day(current_date, context):
                business_days_added += 1
        
        return current_date
    
    def _is_trading_day(self, check_date: date, context: ValidationContext) -> bool:
        """Check if date is a trading day."""
        # Check trading calendar from context
        if context.trading_calendar and check_date in context.trading_calendar:
            calendar_entry = context.trading_calendar[check_date]
            return getattr(calendar_entry, 'is_trading_day', True)
        
        # Default: weekdays are trading days (simplified)
        return check_date.weekday() < 5


class TaiwanFundamentalLagValidator(ValidationPlugin):
    """Validates 60-day financial data lag for Taiwan market."""
    
    def __init__(self, quarterly_lag_days: int = 60, annual_lag_days: int = 90):
        self.quarterly_lag_days = quarterly_lag_days
        self.annual_lag_days = annual_lag_days
    
    @property
    def name(self) -> str:
        return "taiwan_fundamental_lag_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.CRITICAL  # Important for look-ahead bias prevention
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        return {DataType.FUNDAMENTAL}
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this validator can process the given value."""
        return value.data_type == DataType.FUNDAMENTAL
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Validate fundamental data timing constraints."""
        start_time = datetime.utcnow()
        issues = []
        
        try:
            # Extract fundamental data info
            if not isinstance(value.value, dict):
                return ValidationOutput(
                    validator_name=self.name,
                    validation_id=f"{self.name}_{int(start_time.timestamp())}",
                    result=ValidationResult.SKIP,
                    metadata={"reason": "Invalid fundamental data format"}
                )
            
            fiscal_year = value.value.get("fiscal_year")
            fiscal_quarter = value.value.get("fiscal_quarter")
            
            if not fiscal_year or not fiscal_quarter:
                issues.append(QualityIssue(
                    check_type=QualityCheckType.COMPLETENESS,
                    severity=SeverityLevel.WARNING,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description="Missing fiscal year or quarter information",
                    suggested_action="Verify fundamental data format"
                ))
            else:
                # Calculate expected reporting deadline
                quarter_end_date = self._get_quarter_end_date(fiscal_year, fiscal_quarter)
                max_lag_days = self.annual_lag_days if fiscal_quarter == 4 else self.quarterly_lag_days
                reporting_deadline = quarter_end_date + timedelta(days=max_lag_days)
                
                # Check if data is available before reporting deadline
                if value.as_of_date < reporting_deadline:
                    # Calculate actual lag
                    actual_lag = (value.as_of_date - quarter_end_date).days
                    
                    if actual_lag < 0:
                        # Data available before quarter end - critical error
                        issues.append(QualityIssue(
                            check_type=QualityCheckType.TEMPORAL,
                            severity=SeverityLevel.CRITICAL,
                            symbol=value.symbol or "",
                            data_type=value.data_type,
                            data_date=value.value_date,
                            issue_date=datetime.utcnow(),
                            description=f"Fundamental data available {abs(actual_lag)} days before quarter end",
                            details={
                                "quarter_end_date": quarter_end_date.isoformat(),
                                "as_of_date": value.as_of_date.isoformat(),
                                "lag_days": actual_lag,
                                "fiscal_year": fiscal_year,
                                "fiscal_quarter": fiscal_quarter
                            },
                            suggested_action="Critical look-ahead bias - check data timing"
                        ))
                    else:
                        # Early reporting
                        issues.append(QualityIssue(
                            check_type=QualityCheckType.TIMELINESS,
                            severity=SeverityLevel.INFO,
                            symbol=value.symbol or "",
                            data_type=value.data_type,
                            data_date=value.value_date,
                            issue_date=datetime.utcnow(),
                            description=f"Early fundamental data reporting: {actual_lag} days after quarter end",
                            details={
                                "quarter_end_date": quarter_end_date.isoformat(),
                                "as_of_date": value.as_of_date.isoformat(),
                                "lag_days": actual_lag,
                                "max_lag_days": max_lag_days,
                                "fiscal_year": fiscal_year,
                                "fiscal_quarter": fiscal_quarter
                            },
                            suggested_action="Verify early reporting is legitimate"
                        ))
                
                # Check if lag exceeds regulatory maximum
                actual_lag = (value.as_of_date - quarter_end_date).days
                if actual_lag > max_lag_days:
                    issues.append(QualityIssue(
                        check_type=QualityCheckType.TIMELINESS,
                        severity=SeverityLevel.WARNING,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Late fundamental data reporting: {actual_lag} days exceeds {max_lag_days} day limit",
                        details={
                            "quarter_end_date": quarter_end_date.isoformat(),
                            "as_of_date": value.as_of_date.isoformat(),
                            "lag_days": actual_lag,
                            "max_lag_days": max_lag_days,
                            "fiscal_year": fiscal_year,
                            "fiscal_quarter": fiscal_quarter
                        },
                        suggested_action="Check for reporting extensions or restatements"
                    ))
            
            result = ValidationResult.FAIL if any(i.severity == SeverityLevel.CRITICAL for i in issues) else ValidationResult.PASS
            
        except Exception as e:
            logger.error(f"Error in Taiwan fundamental lag validation: {e}")
            issues.append(QualityIssue(
                check_type=QualityCheckType.VALIDITY,
                severity=SeverityLevel.ERROR,
                symbol=value.symbol or "",
                data_type=value.data_type,
                data_date=value.value_date,
                issue_date=datetime.utcnow(),
                description=f"Validation error: {str(e)}"
            ))
            result = ValidationResult.FAIL
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time
        )
    
    def _get_quarter_end_date(self, fiscal_year: int, fiscal_quarter: int) -> date:
        """Get quarter end date for fiscal year and quarter."""
        # Taiwan fiscal quarters typically align with calendar quarters
        quarter_end_months = {1: 3, 2: 6, 3: 9, 4: 12}
        
        if fiscal_quarter not in quarter_end_months:
            raise ValueError(f"Invalid fiscal quarter: {fiscal_quarter}")
        
        month = quarter_end_months[fiscal_quarter]
        
        # Get last day of quarter
        if month == 12:
            return date(fiscal_year, 12, 31)
        elif month == 6:
            return date(fiscal_year, 6, 30)
        elif month == 9:
            return date(fiscal_year, 9, 30)
        else:  # March
            return date(fiscal_year, 3, 31)


# Factory function to create Taiwan validators

def create_taiwan_validators() -> List[ValidationPlugin]:
    """Create a comprehensive set of Taiwan market validators for Stream A requirements."""
    return [
        # Core Taiwan market validators
        TaiwanPriceLimitValidator(daily_limit_pct=0.10),
        TaiwanVolumeValidator(spike_threshold=5.0),
        TaiwanTradingHoursValidator(),
        TaiwanSettlementValidator(),
        
        # New Stream A validators for comprehensive data quality
        TaiwanFundamentalLagValidator(quarterly_lag_days=60, annual_lag_days=90),
        TaiwanDataCompletenessValidator(),
        TaiwanDataConsistencyValidator(),
    ]


def create_taiwan_validation_registry() -> 'ValidationRegistry':
    """Create a validation registry configured for Taiwan market requirements."""
    from .validation_engine import ValidationRegistry
    
    registry = ValidationRegistry()
    
    # Register all Taiwan validators
    validators = create_taiwan_validators()
    
    for validator in validators:
        # Define dependencies for some validators
        dependencies = set()
        
        if validator.name == "taiwan_data_consistency_validator":
            # Consistency checks should run after completeness
            dependencies.add("taiwan_data_completeness_validator")
        
        registry.register_plugin(validator, dependencies=dependencies)
    
    return registry


def create_optimized_taiwan_engine(temporal_store: 'TemporalStore') -> 'ValidationEngine':
    """Create a high-performance validation engine optimized for Taiwan market real-time validation."""
    from .validation_engine import ValidationEngine
    
    registry = create_taiwan_validation_registry()
    
    # Optimized for <10ms latency requirements
    engine = ValidationEngine(
        registry=registry,
        temporal_store=temporal_store,
        max_workers=8,           # Higher parallelism
        timeout_ms=5000,         # 5s timeout for fail-fast
        enable_async=True,       # Async processing
        enable_fast_path=True,   # Fast-path optimizations
        max_cache_size=50000     # Large cache for better hit rates
    )
    
    return engine