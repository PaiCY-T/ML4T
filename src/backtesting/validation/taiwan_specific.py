"""
Taiwan Market-Specific Validation Rules for ML4T.

This module implements validation rules and constraints specific to the Taiwan
Stock Exchange (TWSE) and Taipei Exchange (TPEx), including settlement timing,
trading calendar, and market-specific constraints.

Key Features:
- T+2 settlement lag validation
- Taiwan trading calendar integration
- Lunar New Year and typhoon handling
- Price limit and circuit breaker validation
- Corporate action timing validation
- Market microstructure constraints
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from decimal import Decimal

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.models.taiwan_market import (
    TaiwanTradingCalendar, TaiwanSettlement, TaiwanMarketCode,
    TradingStatus, CorporateActionType, create_taiwan_trading_calendar
)
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MarketEventType(Enum):
    """Types of Taiwan market events that affect validation."""
    LUNAR_NEW_YEAR = "lunar_new_year"
    TYPHOON_DAY = "typhoon_day"
    NATIONAL_HOLIDAY = "national_holiday"
    MARKET_CLOSURE = "market_closure"
    EARLY_CLOSE = "early_close"
    CIRCUIT_BREAKER = "circuit_breaker"
    TRADING_HALT = "trading_halt"


@dataclass
class ValidationIssue:
    """A validation issue found during Taiwan market validation."""
    severity: ValidationSeverity
    issue_type: str
    description: str
    symbol: Optional[str] = None
    date: Optional[date] = None
    value: Optional[Any] = None
    expected_value: Optional[Any] = None
    remediation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'severity': self.severity.value,
            'issue_type': self.issue_type,
            'description': self.description,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'value': str(self.value) if self.value is not None else None,
            'expected_value': str(self.expected_value) if self.expected_value is not None else None,
            'remediation': self.remediation
        }


@dataclass
class TaiwanValidationConfig:
    """Configuration for Taiwan market validation."""
    # Settlement validation
    enforce_settlement_lag: bool = True
    settlement_lag_days: int = 2
    validate_settlement_calendar: bool = True
    
    # Trading calendar validation
    validate_trading_days: bool = True
    allow_partial_trading_days: bool = False
    validate_market_hours: bool = True
    
    # Price limit validation
    validate_price_limits: bool = True
    daily_price_limit_pct: float = 0.10  # 10% daily limit
    
    # Volume and liquidity validation
    validate_volume_constraints: bool = True
    min_daily_volume: int = 1000  # Minimum daily volume
    max_position_pct: float = 0.05  # Max 5% of daily volume
    
    # Corporate action validation
    validate_corporate_actions: bool = True
    ex_date_adjustment_days: int = 1  # Ex-date adjustment
    
    # Market event handling
    validate_market_events: bool = True
    handle_lunar_new_year: bool = True
    handle_typhoon_days: bool = True
    
    # Data quality thresholds
    max_missing_data_pct: float = 0.05  # Max 5% missing data
    max_outlier_pct: float = 0.01  # Max 1% outliers
    
    def __post_init__(self):
        """Validate configuration."""
        if self.settlement_lag_days < 0:
            raise ValueError("Settlement lag days cannot be negative")
        if self.daily_price_limit_pct <= 0 or self.daily_price_limit_pct > 1:
            raise ValueError("Price limit percentage must be between 0 and 1")


class TaiwanMarketValidator:
    """
    Comprehensive validator for Taiwan market-specific constraints.
    
    Validates data and trading scenarios against Taiwan Stock Exchange
    and Taipei Exchange rules and market microstructure.
    """
    
    def __init__(
        self,
        config: TaiwanValidationConfig,
        temporal_store: TemporalStore,
        pit_engine: Optional[PointInTimeEngine] = None,
        taiwan_calendar: Optional[TaiwanTradingCalendar] = None
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine or PointInTimeEngine(temporal_store)
        self.taiwan_calendar = taiwan_calendar or create_taiwan_trading_calendar()
        self.settlement = TaiwanSettlement()
        
        # Load Taiwan market-specific data
        self._load_market_constraints()
        
        logger.info("TaiwanMarketValidator initialized")
    
    def validate_trading_scenario(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        positions: Optional[Dict[str, Dict[date, float]]] = None,
        transactions: Optional[List[Dict[str, Any]]] = None
    ) -> List[ValidationIssue]:
        """
        Validate a complete trading scenario against Taiwan market rules.
        
        Args:
            symbols: List of Taiwan stock symbols
            start_date: Scenario start date
            end_date: Scenario end date
            positions: Position sizes by symbol and date
            transactions: List of transaction records
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        logger.info(f"Validating trading scenario for {len(symbols)} symbols")
        
        # 1. Validate date range and calendar
        issues.extend(self._validate_date_range(start_date, end_date))
        
        # 2. Validate symbol availability and listing status
        issues.extend(self._validate_symbols(symbols, start_date, end_date))
        
        # 3. Validate trading calendar compliance
        issues.extend(self._validate_trading_calendar(start_date, end_date))
        
        # 4. Validate settlement timing
        if self.config.enforce_settlement_lag:
            issues.extend(self._validate_settlement_timing(start_date, end_date))
        
        # 5. Validate price limits and market constraints
        if self.config.validate_price_limits:
            issues.extend(self._validate_price_limits(symbols, start_date, end_date))
        
        # 6. Validate volume and liquidity constraints
        if self.config.validate_volume_constraints and positions:
            issues.extend(self._validate_volume_constraints(symbols, positions, start_date, end_date))
        
        # 7. Validate corporate actions timing
        if self.config.validate_corporate_actions:
            issues.extend(self._validate_corporate_actions(symbols, start_date, end_date))
        
        # 8. Validate market events
        if self.config.validate_market_events:
            issues.extend(self._validate_market_events(start_date, end_date))
        
        # 9. Validate transaction timing if provided
        if transactions:
            issues.extend(self._validate_transactions(transactions))
        
        logger.info(f"Validation completed: {len(issues)} issues found")
        return issues
    
    def validate_data_quality(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        data_types: List[DataType] = None
    ) -> List[ValidationIssue]:
        """
        Validate data quality for Taiwan market data.
        
        Args:
            symbols: List of symbols to validate
            start_date: Start date for validation
            end_date: End date for validation
            data_types: Data types to validate
            
        Returns:
            List of data quality issues
        """
        if data_types is None:
            data_types = [DataType.PRICE, DataType.VOLUME]
        
        issues = []
        
        for symbol in symbols:
            for data_type in data_types:
                # Check data availability
                data_issues = self._validate_data_availability(
                    symbol, data_type, start_date, end_date
                )
                issues.extend(data_issues)
                
                # Check data consistency
                consistency_issues = self._validate_data_consistency(
                    symbol, data_type, start_date, end_date
                )
                issues.extend(consistency_issues)
        
        return issues
    
    def _load_market_constraints(self):
        """Load Taiwan market-specific constraints and parameters."""
        # Price limits by market
        self.price_limits = {
            TaiwanMarketCode.TWSE: 0.10,  # 10% daily limit
            TaiwanMarketCode.TPEx: 0.10   # 10% daily limit
        }
        
        # Trading hours
        self.trading_hours = {
            'market_open': '09:00',
            'market_close': '13:30',
            'lunch_break_start': '12:00',
            'lunch_break_end': '13:00'
        }
        
        # Settlement constraints
        self.settlement_constraints = {
            'settlement_cycle': 'T+2',
            'cut_off_time': '15:30',
            'settlement_currency': 'TWD'
        }
    
    def _validate_date_range(self, start_date: date, end_date: date) -> List[ValidationIssue]:
        """Validate date range parameters."""
        issues = []
        
        if end_date <= start_date:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                issue_type="invalid_date_range",
                description="End date must be after start date",
                date=start_date,
                value=end_date,
                remediation="Ensure end_date > start_date"
            ))
        
        # Check if dates are too far in the future
        max_future_date = date.today() + timedelta(days=365)
        if start_date > max_future_date:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                issue_type="future_date",
                description="Start date is far in the future",
                date=start_date,
                remediation="Verify date is correct"
            ))
        
        # Check for very old dates (data availability)
        min_data_date = date(2000, 1, 1)  # Approximate start of electronic trading
        if start_date < min_data_date:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                issue_type="old_date",
                description="Start date may have limited data availability",
                date=start_date,
                remediation="Check data availability for historical dates"
            ))
        
        return issues
    
    def _validate_symbols(
        self, 
        symbols: List[str], 
        start_date: date, 
        end_date: date
    ) -> List[ValidationIssue]:
        """Validate symbol format and listing status."""
        issues = []
        
        for symbol in symbols:
            # Validate Taiwan stock symbol format
            if not self._is_valid_taiwan_symbol(symbol):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    issue_type="invalid_symbol_format",
                    description=f"Invalid Taiwan stock symbol format: {symbol}",
                    symbol=symbol,
                    remediation="Use format: NNNN.TW or NNNN.TWO"
                ))
            
            # Check if symbol was listed during the period
            listing_issues = self._validate_listing_period(symbol, start_date, end_date)
            issues.extend(listing_issues)
        
        return issues
    
    def _validate_trading_calendar(self, start_date: date, end_date: date) -> List[ValidationIssue]:
        """Validate trading calendar compliance."""
        issues = []
        
        current = start_date
        while current <= end_date:
            if not self.taiwan_calendar.is_trading_day(current):
                # Check if this is expected (weekend, holiday)
                if current.weekday() >= 5:  # Weekend
                    if not self.config.allow_partial_trading_days:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            issue_type="weekend_day",
                            description=f"Weekend day in trading period: {current}",
                            date=current,
                            remediation="Exclude weekends from trading analysis"
                        ))
                else:
                    # Weekday but not trading day - likely holiday
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        issue_type="holiday_trading_day",
                        description=f"Holiday or non-trading day: {current}",
                        date=current,
                        remediation="Verify Taiwan market calendar"
                    ))
            
            current += timedelta(days=1)
        
        return issues
    
    def _validate_settlement_timing(self, start_date: date, end_date: date) -> List[ValidationIssue]:
        """Validate T+2 settlement timing constraints."""
        issues = []
        
        # Check if end date allows for T+2 settlement
        settlement_date = self.settlement.calculate_settlement_date(
            end_date, self.config.settlement_lag_days
        )
        
        if settlement_date > date.today():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                issue_type="future_settlement",
                description=f"Settlement date {settlement_date} is in the future",
                date=end_date,
                expected_value=f"T+{self.config.settlement_lag_days}",
                remediation="Ensure sufficient time for settlement"
            ))
        
        return issues
    
    def _validate_price_limits(
        self, 
        symbols: List[str], 
        start_date: date, 
        end_date: date
    ) -> List[ValidationIssue]:
        """Validate price limit compliance."""
        issues = []
        
        for symbol in symbols:
            try:
                # Query price data
                price_query = PITQuery(
                    symbols=[symbol],
                    as_of_date=end_date,
                    data_types=[DataType.PRICE],
                    start_date=start_date,
                    end_date=end_date
                )
                
                price_data = self.pit_engine.query(price_query)
                
                if symbol in price_data and len(price_data[symbol]) > 1:
                    # Check for price limit violations
                    price_violations = self._check_price_limit_violations(
                        symbol, price_data[symbol]
                    )
                    issues.extend(price_violations)
                    
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    issue_type="price_data_unavailable",
                    description=f"Could not validate price limits for {symbol}: {e}",
                    symbol=symbol,
                    remediation="Check data availability"
                ))
        
        return issues
    
    def _validate_volume_constraints(
        self,
        symbols: List[str],
        positions: Dict[str, Dict[date, float]],
        start_date: date,
        end_date: date
    ) -> List[ValidationIssue]:
        """Validate volume and liquidity constraints."""
        issues = []
        
        for symbol in symbols:
            if symbol not in positions:
                continue
                
            symbol_positions = positions[symbol]
            
            for trade_date, position_size in symbol_positions.items():
                if start_date <= trade_date <= end_date:
                    volume_issues = self._validate_position_vs_volume(
                        symbol, trade_date, position_size
                    )
                    issues.extend(volume_issues)
        
        return issues
    
    def _validate_corporate_actions(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> List[ValidationIssue]:
        """Validate corporate action timing and adjustments."""
        issues = []
        
        for symbol in symbols:
            try:
                # Query corporate action data
                ca_query = PITQuery(
                    symbols=[symbol],
                    as_of_date=end_date,
                    data_types=[DataType.CORPORATE_ACTION],
                    start_date=start_date,
                    end_date=end_date
                )
                
                ca_data = self.pit_engine.query(ca_query)
                
                if symbol in ca_data:
                    ca_issues = self._validate_corporate_action_timing(
                        symbol, ca_data[symbol]
                    )
                    issues.extend(ca_issues)
                    
            except Exception as e:
                logger.debug(f"No corporate action data for {symbol}: {e}")
        
        return issues
    
    def _validate_market_events(self, start_date: date, end_date: date) -> List[ValidationIssue]:
        """Validate market events and special conditions."""
        issues = []
        
        # Check for Lunar New Year period
        if self.config.handle_lunar_new_year:
            lny_issues = self._check_lunar_new_year_period(start_date, end_date)
            issues.extend(lny_issues)
        
        # Check for typhoon days
        if self.config.handle_typhoon_days:
            typhoon_issues = self._check_typhoon_days(start_date, end_date)
            issues.extend(typhoon_issues)
        
        return issues
    
    def _validate_transactions(self, transactions: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate transaction timing and constraints."""
        issues = []
        
        for i, transaction in enumerate(transactions):
            try:
                trade_date = transaction.get('date')
                symbol = transaction.get('symbol')
                quantity = transaction.get('quantity', 0)
                
                if not trade_date or not symbol:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        issue_type="incomplete_transaction",
                        description=f"Transaction {i} missing required fields",
                        remediation="Ensure all transactions have date and symbol"
                    ))
                    continue
                
                # Validate trading day
                if not self.taiwan_calendar.is_trading_day(trade_date):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        issue_type="non_trading_day_transaction",
                        description=f"Transaction on non-trading day: {trade_date}",
                        symbol=symbol,
                        date=trade_date,
                        remediation="Move transaction to next trading day"
                    ))
                
                # Validate quantity
                if quantity <= 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        issue_type="zero_quantity_transaction",
                        description=f"Zero or negative quantity: {quantity}",
                        symbol=symbol,
                        date=trade_date,
                        value=quantity,
                        remediation="Verify transaction quantity"
                    ))
                    
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    issue_type="transaction_validation_error",
                    description=f"Error validating transaction {i}: {e}",
                    remediation="Check transaction format"
                ))
        
        return issues
    
    def _is_valid_taiwan_symbol(self, symbol: str) -> bool:
        """Check if symbol follows Taiwan stock symbol format."""
        # Taiwan symbols: 4-digit code + .TW (TWSE) or .TWO (TPEx)
        if len(symbol) >= 6:
            code_part = symbol[:-3] if symbol.endswith('.TW') else symbol[:-4]
            suffix = symbol[-3:] if symbol.endswith('.TW') else symbol[-4:]
            
            return (
                code_part.isdigit() and 
                len(code_part) == 4 and
                suffix in ['.TW', '.TWO']
            )
        return False
    
    def _validate_listing_period(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date
    ) -> List[ValidationIssue]:
        """Validate if symbol was listed during the period."""
        issues = []
        
        # This would typically query a listing database
        # For now, implement basic validation
        
        # Check for delisted stocks (basic heuristic)
        if symbol.startswith('000') or symbol.startswith('999'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                issue_type="potentially_delisted",
                description=f"Symbol {symbol} may be delisted or suspended",
                symbol=symbol,
                remediation="Verify current listing status"
            ))
        
        return issues
    
    def _check_price_limit_violations(
        self, 
        symbol: str, 
        price_data: List[TemporalValue]
    ) -> List[ValidationIssue]:
        """Check for daily price limit violations."""
        issues = []
        
        if len(price_data) < 2:
            return issues
        
        # Sort by value_date
        sorted_data = sorted(price_data, key=lambda x: x.value_date)
        
        for i in range(1, len(sorted_data)):
            prev_price = float(sorted_data[i-1].value)
            curr_price = float(sorted_data[i].value)
            
            price_change = abs(curr_price - prev_price) / prev_price
            
            if price_change > self.config.daily_price_limit_pct:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    issue_type="price_limit_violation",
                    description=f"Price change {price_change:.2%} exceeds limit {self.config.daily_price_limit_pct:.2%}",
                    symbol=symbol,
                    date=sorted_data[i].value_date,
                    value=price_change,
                    expected_value=self.config.daily_price_limit_pct,
                    remediation="Verify price data or adjust for stock splits"
                ))
        
        return issues
    
    def _validate_position_vs_volume(
        self, 
        symbol: str, 
        trade_date: date, 
        position_size: float
    ) -> List[ValidationIssue]:
        """Validate position size against daily volume."""
        issues = []
        
        try:
            # Query volume data for the trade date
            volume_query = PITQuery(
                symbols=[symbol],
                as_of_date=trade_date,
                data_types=[DataType.VOLUME],
                start_date=trade_date,
                end_date=trade_date
            )
            
            volume_data = self.pit_engine.query(volume_query)
            
            if symbol in volume_data and len(volume_data[symbol]) > 0:
                daily_volume = float(volume_data[symbol][0].value)
                
                if daily_volume < self.config.min_daily_volume:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        issue_type="low_volume",
                        description=f"Daily volume {daily_volume} below minimum {self.config.min_daily_volume}",
                        symbol=symbol,
                        date=trade_date,
                        value=daily_volume,
                        expected_value=self.config.min_daily_volume,
                        remediation="Consider liquidity constraints"
                    ))
                
                position_pct = abs(position_size) / daily_volume
                if position_pct > self.config.max_position_pct:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        issue_type="excessive_position_size",
                        description=f"Position {position_pct:.2%} of daily volume exceeds limit {self.config.max_position_pct:.2%}",
                        symbol=symbol,
                        date=trade_date,
                        value=position_pct,
                        expected_value=self.config.max_position_pct,
                        remediation="Reduce position size or spread across multiple days"
                    ))
                    
        except Exception as e:
            logger.debug(f"Could not validate volume for {symbol} on {trade_date}: {e}")
        
        return issues
    
    def _validate_corporate_action_timing(
        self, 
        symbol: str, 
        ca_data: List[TemporalValue]
    ) -> List[ValidationIssue]:
        """Validate corporate action timing and ex-date adjustments."""
        issues = []
        
        for ca_record in ca_data:
            ca_type = ca_record.metadata.get('action_type') if ca_record.metadata else None
            ex_date = ca_record.metadata.get('ex_date') if ca_record.metadata else None
            
            if ca_type and ex_date:
                # Validate ex-date timing
                if ex_date <= ca_record.value_date:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        issue_type="invalid_ex_date",
                        description=f"Ex-date {ex_date} should be after announcement {ca_record.value_date}",
                        symbol=symbol,
                        date=ca_record.value_date,
                        remediation="Verify corporate action timeline"
                    ))
        
        return issues
    
    def _check_lunar_new_year_period(self, start_date: date, end_date: date) -> List[ValidationIssue]:
        """Check for Lunar New Year market closure period."""
        issues = []
        
        # Approximate Lunar New Year dates (would use actual calendar)
        lny_periods = [
            (date(2023, 1, 21), date(2023, 1, 27)),
            (date(2024, 2, 10), date(2024, 2, 14)),
            (date(2025, 1, 29), date(2025, 2, 2)),
        ]
        
        for lny_start, lny_end in lny_periods:
            if (start_date <= lny_end and end_date >= lny_start):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    issue_type="lunar_new_year_period",
                    description=f"Trading period includes Lunar New Year closure: {lny_start} to {lny_end}",
                    date=lny_start,
                    remediation="Account for extended market closure"
                ))
        
        return issues
    
    def _check_typhoon_days(self, start_date: date, end_date: date) -> List[ValidationIssue]:
        """Check for potential typhoon day closures."""
        issues = []
        
        # Typhoon season in Taiwan is typically June-November
        current = start_date
        while current <= end_date:
            if 6 <= current.month <= 11:  # Typhoon season
                # This would typically check historical typhoon data
                # For now, just note the season
                if current.month in [8, 9]:  # Peak season
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        issue_type="typhoon_season",
                        description=f"Trading during peak typhoon season: {current.strftime('%Y-%m')}",
                        date=current,
                        remediation="Monitor for potential typhoon-related closures"
                    ))
                    break  # Only warn once per period
            current += timedelta(days=30)  # Check monthly
        
        return issues
    
    def _validate_data_availability(
        self,
        symbol: str,
        data_type: DataType,
        start_date: date,
        end_date: date
    ) -> List[ValidationIssue]:
        """Validate data availability for symbol and period."""
        issues = []
        
        try:
            query = PITQuery(
                symbols=[symbol],
                as_of_date=end_date,
                data_types=[data_type],
                start_date=start_date,
                end_date=end_date
            )
            
            available = self.pit_engine.check_data_availability(query)
            
            if not available:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    issue_type="data_unavailable",
                    description=f"No {data_type.value} data available for {symbol}",
                    symbol=symbol,
                    date=start_date,
                    remediation="Check data source and symbol validity"
                ))
                
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                issue_type="data_check_failed",
                description=f"Could not check data availability: {e}",
                symbol=symbol,
                remediation="Verify data access configuration"
            ))
        
        return issues
    
    def _validate_data_consistency(
        self,
        symbol: str,
        data_type: DataType,
        start_date: date,
        end_date: date
    ) -> List[ValidationIssue]:
        """Validate data consistency and quality."""
        issues = []
        
        # This would implement detailed data quality checks
        # For now, return empty list as placeholder
        
        return issues


class SettlementValidator:
    """
    Specialized validator for Taiwan T+2 settlement constraints.
    
    Handles settlement date calculations, cut-off times, and
    settlement-related validation rules.
    """
    
    def __init__(self, taiwan_calendar: Optional[TaiwanTradingCalendar] = None):
        self.taiwan_calendar = taiwan_calendar or create_taiwan_trading_calendar()
        self.settlement = TaiwanSettlement()
    
    def validate_settlement_schedule(
        self,
        trade_dates: List[date],
        settlement_lag: int = 2
    ) -> List[ValidationIssue]:
        """Validate settlement schedule for given trade dates."""
        issues = []
        
        for trade_date in trade_dates:
            settlement_date = self.settlement.calculate_settlement_date(
                trade_date, settlement_lag
            )
            
            # Validate settlement date is a trading day
            if not self.taiwan_calendar.is_trading_day(settlement_date):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    issue_type="non_trading_settlement_date",
                    description=f"Settlement date {settlement_date} is not a trading day",
                    date=trade_date,
                    expected_value=f"T+{settlement_lag}",
                    remediation="Adjust settlement calculation for holidays"
                ))
            
            # Check for reasonable settlement timeline
            if (settlement_date - trade_date).days > settlement_lag + 5:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    issue_type="extended_settlement_period",
                    description=f"Settlement period {(settlement_date - trade_date).days} days exceeds normal T+{settlement_lag}",
                    date=trade_date,
                    value=(settlement_date - trade_date).days,
                    expected_value=settlement_lag,
                    remediation="Verify trading calendar for holidays"
                ))
        
        return issues


# Utility functions for creating validators
def create_standard_taiwan_validator(
    temporal_store: TemporalStore,
    **config_overrides
) -> TaiwanMarketValidator:
    """Create Taiwan market validator with standard configuration."""
    config = TaiwanValidationConfig(**config_overrides)
    return TaiwanMarketValidator(config, temporal_store)


def create_strict_taiwan_validator(
    temporal_store: TemporalStore,
    **config_overrides
) -> TaiwanMarketValidator:
    """Create Taiwan market validator with strict validation rules."""
    strict_config = TaiwanValidationConfig(
        enforce_settlement_lag=True,
        validate_trading_days=True,
        validate_price_limits=True,
        validate_volume_constraints=True,
        validate_corporate_actions=True,
        validate_market_events=True,
        max_missing_data_pct=0.01,  # Stricter
        max_outlier_pct=0.005,      # Stricter
        **config_overrides
    )
    return TaiwanMarketValidator(strict_config, temporal_store)


# Example usage
if __name__ == "__main__":
    print("Taiwan Market Validator demo")
    
    # This would be called with actual temporal store
    print("Demo of Taiwan market validation - requires actual data stores")
    
    config = TaiwanValidationConfig()
    print(f"Default config: {config}")
    print("In actual usage, initialize with TemporalStore and run validation")