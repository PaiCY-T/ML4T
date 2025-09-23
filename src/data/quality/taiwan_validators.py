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
    """Create a standard set of Taiwan market validators."""
    return [
        TaiwanPriceLimitValidator(),
        TaiwanVolumeValidator(),
        TaiwanTradingHoursValidator(),
        TaiwanSettlementValidator(),
        TaiwanFundamentalLagValidator()
    ]