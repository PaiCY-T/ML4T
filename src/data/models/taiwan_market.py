"""
Taiwan market specific data models and settlement handling.

This module implements Taiwan Stock Exchange (TWSE) specific data models,
including T+2 settlement lag, trading calendar, and market timing constraints.
"""

from datetime import datetime, date, time, timedelta
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from decimal import Decimal

from ..core.temporal import DataType, TemporalValue, MarketSession

logger = logging.getLogger(__name__)


class TaiwanMarketCode(Enum):
    """Taiwan market codes."""
    TWSE = "TWSE"  # Taiwan Stock Exchange
    TPEx = "TPEx"  # Taipei Exchange (OTC)


class TradingStatus(Enum):
    """Trading status for Taiwan stocks."""
    NORMAL = "normal"
    SUSPENDED = "suspended"
    HALTED = "halted"
    DELISTED = "delisted"
    IPO = "ipo"
    ATTENTION = "attention"  # Under regulatory attention


class CorporateActionType(Enum):
    """Types of corporate actions in Taiwan market."""
    DIVIDEND_CASH = "dividend_cash"
    DIVIDEND_STOCK = "dividend_stock"
    STOCK_SPLIT = "stock_split"
    STOCK_MERGER = "stock_merger"
    RIGHTS_ISSUE = "rights_issue"
    SPIN_OFF = "spin_off"
    CAPITAL_REDUCTION = "capital_reduction"


@dataclass
class TaiwanTradingCalendar:
    """Taiwan trading calendar with market sessions."""
    date: date
    is_trading_day: bool
    market_session: MarketSession
    morning_open: Optional[time] = time(9, 0)    # 09:00 TST
    morning_close: Optional[time] = time(13, 30)  # 13:30 TST
    notes: Optional[str] = None
    
    @property
    def trading_hours(self) -> Tuple[time, time]:
        """Get trading hours for the day."""
        if self.is_trading_day and self.market_session == MarketSession.MORNING:
            return (self.morning_open, self.morning_close)
        return (None, None)
    
    @property
    def trading_duration_minutes(self) -> int:
        """Get trading duration in minutes."""
        if not self.is_trading_day:
            return 0
        start, end = self.trading_hours
        if start and end:
            start_dt = datetime.combine(self.date, start)
            end_dt = datetime.combine(self.date, end)
            return int((end_dt - start_dt).total_seconds() / 60)
        return 0


@dataclass
class TaiwanSettlement:
    """Taiwan market settlement information with T+2 handling."""
    trade_date: Optional[date] = None
    settlement_date: Optional[date] = None
    is_regular_settlement: bool = True
    special_settlement_reason: Optional[str] = None
    
    def __init__(self, trade_date: Optional[date] = None, settlement_date: Optional[date] = None, 
                 is_regular_settlement: bool = True, special_settlement_reason: Optional[str] = None):
        self.trade_date = trade_date
        self.settlement_date = settlement_date
        self.is_regular_settlement = is_regular_settlement
        self.special_settlement_reason = special_settlement_reason
        
        if self.trade_date and self.settlement_date and self.settlement_date <= self.trade_date:
            raise ValueError(f"Settlement date {self.settlement_date} must be after trade date {self.trade_date}")
    
    @property
    def settlement_lag_days(self) -> int:
        """Calculate settlement lag in calendar days."""
        if not self.trade_date or not self.settlement_date:
            return 2  # Default T+2
        return (self.settlement_date - self.trade_date).days
    
    def calculate_settlement_date(self, trade_date: date, settlement_lag: int = 2) -> date:
        """Calculate settlement date for a given trade date."""
        # Simple T+2 calculation adjusting for weekends
        settlement_date = trade_date + timedelta(days=settlement_lag)
        
        # Skip weekends (simplified)
        while settlement_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            settlement_date += timedelta(days=1)
            
        return settlement_date
    
    def adjust_for_settlement_lag(self, trade_date: date, lag_days: int = 2) -> date:
        """Adjust a trade date for settlement lag."""
        return self.calculate_settlement_date(trade_date, lag_days)
    
    @classmethod
    def calculate_t2_settlement(cls, trade_date: date, 
                               trading_calendar: Optional[Dict[date, TaiwanTradingCalendar]] = None) -> 'TaiwanSettlement':
        """Calculate T+2 settlement date considering Taiwan holidays."""
        if trading_calendar is None:
            # Simplified calculation without calendar
            settlement_date = trade_date + timedelta(days=2)
            while settlement_date.weekday() >= 5:
                settlement_date += timedelta(days=1)
        else:
            current_date = trade_date
            business_days_added = 0
            
            while business_days_added < 2:
                current_date += timedelta(days=1)
                
                # Check if it's a trading day
                calendar_entry = trading_calendar.get(current_date)
                if calendar_entry and calendar_entry.is_trading_day:
                    business_days_added += 1
        
        return cls(
            trade_date=trade_date,
            settlement_date=current_date if trading_calendar else settlement_date,
            is_regular_settlement=True
        )


@dataclass
class TaiwanStockInfo:
    """Taiwan stock information and metadata."""
    symbol: str
    name_zh: str  # Chinese name
    name_en: Optional[str] = None  # English name
    market: TaiwanMarketCode = TaiwanMarketCode.TWSE
    industry_code: Optional[str] = None
    sector: Optional[str] = None
    listing_date: Optional[date] = None
    par_value: Optional[Decimal] = None
    outstanding_shares: Optional[int] = None
    trading_status: TradingStatus = TradingStatus.NORMAL
    
    def is_active(self, as_of_date: date) -> bool:
        """Check if stock was actively traded on a given date."""
        if self.listing_date and as_of_date < self.listing_date:
            return False
        return self.trading_status in [TradingStatus.NORMAL, TradingStatus.ATTENTION]


@dataclass
class TaiwanMarketData:
    """Taiwan market specific data point."""
    symbol: str
    data_date: date
    as_of_date: date
    open_price: Optional[Decimal] = None
    high_price: Optional[Decimal] = None
    low_price: Optional[Decimal] = None
    close_price: Optional[Decimal] = None
    volume: Optional[int] = None
    turnover: Optional[Decimal] = None  # Trading value in TWD
    market_cap: Optional[Decimal] = None
    trading_status: TradingStatus = TradingStatus.NORMAL
    price_limit_up: Optional[Decimal] = None  # Daily limit up price
    price_limit_down: Optional[Decimal] = None  # Daily limit down price
    foreign_ownership_pct: Optional[Decimal] = None
    
    def to_temporal_values(self) -> List[TemporalValue]:
        """Convert to temporal values for storage."""
        values = []
        
        # Price data
        if self.close_price is not None:
            values.append(TemporalValue(
                value=self.close_price,
                as_of_date=self.as_of_date,
                value_date=self.data_date,
                data_type=DataType.PRICE,
                symbol=self.symbol,
                metadata={"field": "close_price", "market": "TWSE"}
            ))
        
        # Volume data
        if self.volume is not None:
            values.append(TemporalValue(
                value=self.volume,
                as_of_date=self.as_of_date,
                value_date=self.data_date,
                data_type=DataType.VOLUME,
                symbol=self.symbol,
                metadata={"field": "volume", "market": "TWSE"}
            ))
            
        # Market data (turnover, market cap, etc.)
        for field_name in ["open_price", "high_price", "low_price", "turnover", "market_cap"]:
            value = getattr(self, field_name, None)
            if value is not None:
                values.append(TemporalValue(
                    value=value,
                    as_of_date=self.as_of_date,
                    value_date=self.data_date,
                    data_type=DataType.MARKET_DATA,
                    symbol=self.symbol,
                    metadata={"field": field_name, "market": "TWSE"}
                ))
        
        return values


@dataclass
class TaiwanCorporateAction:
    """Taiwan corporate action data."""
    symbol: str
    action_type: CorporateActionType
    announcement_date: date
    ex_date: date
    record_date: date
    payable_date: Optional[date] = None
    ratio: Optional[Decimal] = None  # For splits, stock dividends
    amount: Optional[Decimal] = None  # For cash dividends in TWD
    description: Optional[str] = None
    
    def to_temporal_value(self) -> TemporalValue:
        """Convert to temporal value for storage."""
        return TemporalValue(
            value={
                "action_type": self.action_type.value,
                "ex_date": self.ex_date.isoformat(),
                "record_date": self.record_date.isoformat(),
                "payable_date": self.payable_date.isoformat() if self.payable_date else None,
                "ratio": float(self.ratio) if self.ratio else None,
                "amount": float(self.amount) if self.amount else None,
                "description": self.description
            },
            as_of_date=self.announcement_date,
            value_date=self.ex_date,
            data_type=DataType.CORPORATE_ACTION,
            symbol=self.symbol,
            metadata={"action_type": self.action_type.value}
        )
    
    def affects_price_on_date(self, check_date: date) -> bool:
        """Check if this corporate action affects price on a given date."""
        return check_date >= self.ex_date


@dataclass 
class TaiwanFundamental:
    """Taiwan fundamental data with reporting lag."""
    symbol: str
    report_date: date  # Financial statement date
    announcement_date: date  # When data was made public
    fiscal_year: int
    fiscal_quarter: int
    revenue: Optional[Decimal] = None
    operating_income: Optional[Decimal] = None
    net_income: Optional[Decimal] = None
    total_assets: Optional[Decimal] = None
    total_equity: Optional[Decimal] = None
    eps: Optional[Decimal] = None  # Earnings per share
    roe: Optional[Decimal] = None  # Return on equity
    book_value_per_share: Optional[Decimal] = None
    
    def to_temporal_value(self) -> TemporalValue:
        """Convert to temporal value with proper lag handling."""
        # Fundamental data is only available after announcement
        return TemporalValue(
            value={
                "revenue": float(self.revenue) if self.revenue else None,
                "operating_income": float(self.operating_income) if self.operating_income else None,
                "net_income": float(self.net_income) if self.net_income else None,
                "total_assets": float(self.total_assets) if self.total_assets else None,
                "total_equity": float(self.total_equity) if self.total_equity else None,
                "eps": float(self.eps) if self.eps else None,
                "roe": float(self.roe) if self.roe else None,
                "book_value_per_share": float(self.book_value_per_share) if self.book_value_per_share else None,
                "fiscal_year": self.fiscal_year,
                "fiscal_quarter": self.fiscal_quarter
            },
            as_of_date=self.announcement_date,
            value_date=self.report_date,
            data_type=DataType.FUNDAMENTAL,
            symbol=self.symbol,
            metadata={
                "fiscal_year": self.fiscal_year,
                "fiscal_quarter": self.fiscal_quarter,
                "lag_days": (self.announcement_date - self.report_date).days
            }
        )
    
    @property
    def reporting_lag_days(self) -> int:
        """Calculate reporting lag in days."""
        return (self.announcement_date - self.report_date).days


class TaiwanMarketDataValidator:
    """Validator for Taiwan market data quality."""
    
    def __init__(self):
        self.price_change_limit = 0.10  # 10% daily limit (simplified)
        self.volume_spike_threshold = 5.0  # 5x average volume
        
    def validate_price_data(self, data: TaiwanMarketData, 
                           previous_close: Optional[Decimal] = None) -> List[str]:
        """Validate price data for anomalies."""
        issues = []
        
        # Check price consistency
        if (data.open_price and data.high_price and 
            data.open_price > data.high_price):
            issues.append("Open price higher than high price")
            
        if (data.low_price and data.close_price and 
            data.low_price > data.close_price):
            issues.append("Low price higher than close price")
            
        # Check daily limits if previous close available
        if previous_close and data.close_price:
            daily_change = abs(data.close_price - previous_close) / previous_close
            if daily_change > self.price_change_limit:
                issues.append(f"Price change {daily_change:.2%} exceeds daily limit")
                
        # Check volume reasonableness
        if data.volume and data.volume < 0:
            issues.append("Negative volume")
            
        return issues
    
    def validate_settlement_timing(self, trade_date: date, 
                                  settlement: TaiwanSettlement) -> List[str]:
        """Validate settlement timing rules."""
        issues = []
        
        if settlement.settlement_lag_days < 2:
            issues.append(f"Settlement lag {settlement.settlement_lag_days} days < T+2 minimum")
            
        if settlement.settlement_lag_days > 5:
            issues.append(f"Settlement lag {settlement.settlement_lag_days} days > reasonable maximum")
            
        return issues
    
    def validate_fundamental_timing(self, fundamental: TaiwanFundamental) -> List[str]:
        """Validate fundamental data timing constraints."""
        issues = []
        
        # Taiwan regulations require financial statements within 60 days for annual reports
        max_lag_days = 60 if fundamental.fiscal_quarter == 4 else 45
        
        if fundamental.reporting_lag_days > max_lag_days:
            issues.append(f"Reporting lag {fundamental.reporting_lag_days} days exceeds "
                         f"regulatory maximum {max_lag_days} days")
            
        if fundamental.announcement_date < fundamental.report_date:
            issues.append("Announcement date before report date")
            
        return issues


def create_taiwan_trading_calendar(start_year: int = None, end_year: int = None) -> 'TaiwanMarketCalendar':
    """Create Taiwan trading calendar for a date range."""
    if start_year is None:
        start_year = date.today().year - 1
    if end_year is None:
        end_year = date.today().year + 1
    
    return TaiwanMarketCalendar(start_year, end_year)


class TaiwanMarketCalendar:
    """Comprehensive Taiwan market trading calendar."""
    
    def __init__(self, start_year: int, end_year: int):
        self.start_year = start_year
        self.end_year = end_year
        self._calendar_cache = {}
        self._initialize_calendar()
    
    def _initialize_calendar(self):
        """Initialize the trading calendar."""
        for year in range(self.start_year, self.end_year + 1):
            self._calendar_cache.update(self._create_year_calendar(year))
    
    def _create_year_calendar(self, year: int) -> Dict[date, TaiwanTradingCalendar]:
        """Create Taiwan trading calendar for a given year."""
        calendar = {}
        
        # This is a simplified implementation
        # Real implementation would fetch from Taiwan Stock Exchange
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        # Get Taiwan holidays for the year
        taiwan_holidays = self._get_taiwan_holidays(year)
        
        current_date = start_date
        while current_date <= end_date:
            # Simplified: weekdays are trading days
            is_trading = current_date.weekday() < 5
            
            # Check for holidays
            if current_date in taiwan_holidays:
                is_trading = False
                
            calendar[current_date] = TaiwanTradingCalendar(
                date=current_date,
                is_trading_day=is_trading,
                market_session=MarketSession.MORNING if is_trading else MarketSession.CLOSED,
                notes=taiwan_holidays.get(current_date, '')
            )
            
            current_date += timedelta(days=1)
        
        return calendar
    
    def _get_taiwan_holidays(self, year: int) -> Dict[date, str]:
        """Get Taiwan holidays for a given year."""
        # Taiwan national holidays (simplified list)
        holidays = {
            date(year, 1, 1): "New Year's Day",
            date(year, 2, 28): "Peace Memorial Day",
            date(year, 4, 4): "Children's Day",
            date(year, 4, 5): "Tomb Sweeping Day",
            date(year, 5, 1): "Labor Day",
            date(year, 10, 10): "National Day",
        }
        
        # Add Lunar New Year (approximate dates - real implementation would calculate)
        # This is a simplified approximation
        lunar_new_year_dates = {
            2023: [(1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27)],
            2024: [(2, 10), (2, 11), (2, 12), (2, 13), (2, 14)],
            2025: [(1, 29), (1, 30), (1, 31), (2, 1), (2, 2)],
            2026: [(2, 17), (2, 18), (2, 19), (2, 20), (2, 21)],
        }
        
        if year in lunar_new_year_dates:
            for month, day in lunar_new_year_dates[year]:
                holidays[date(year, month, day)] = "Lunar New Year Holiday"
        
        return holidays
    
    def is_trading_day(self, check_date: date) -> bool:
        """Check if a date is a trading day."""
        if check_date in self._calendar_cache:
            return self._calendar_cache[check_date].is_trading_day
        
        # If not in cache, make a simple determination
        return check_date.weekday() < 5  # Monday = 0, Friday = 4
    
    def get_trading_session(self, check_date: date) -> MarketSession:
        """Get trading session for a date."""
        if check_date in self._calendar_cache:
            return self._calendar_cache[check_date].market_session
        
        return MarketSession.MORNING if self.is_trading_day(check_date) else MarketSession.CLOSED
    
    def get_next_trading_day(self, current_date: date) -> date:
        """Get the next trading day after the given date."""
        next_date = current_date + timedelta(days=1)
        while not self.is_trading_day(next_date):
            next_date += timedelta(days=1)
        return next_date
    
    def get_previous_trading_day(self, current_date: date) -> date:
        """Get the previous trading day before the given date."""
        prev_date = current_date - timedelta(days=1)
        while not self.is_trading_day(prev_date):
            prev_date -= timedelta(days=1)
        return prev_date
    
    def count_trading_days(self, start_date: date, end_date: date) -> int:
        """Count trading days between two dates (inclusive)."""
        count = 0
        current = start_date
        while current <= end_date:
            if self.is_trading_day(current):
                count += 1
            current += timedelta(days=1)
        return count
    
    def get_trading_days_in_range(self, start_date: date, end_date: date) -> List[date]:
        """Get all trading days in a date range."""
        trading_days = []
        current = start_date
        while current <= end_date:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)
        return trading_days


def get_taiwan_market_timezone() -> str:
    """Get Taiwan market timezone string."""
    return "Asia/Taipei"


def taiwan_market_time_to_utc(market_time: datetime) -> datetime:
    """Convert Taiwan market time to UTC."""
    # Simplified - real implementation would use pytz
    # Taiwan is UTC+8
    return market_time - timedelta(hours=8)