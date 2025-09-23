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
    trade_date: date
    settlement_date: date
    is_regular_settlement: bool = True
    special_settlement_reason: Optional[str] = None
    
    def __post_init__(self):
        """Validate settlement dates."""
        if self.settlement_date <= self.trade_date:
            raise ValueError(f"Settlement date {self.settlement_date} must be after trade date {self.trade_date}")
    
    @property
    def settlement_lag_days(self) -> int:
        """Calculate settlement lag in calendar days."""
        return (self.settlement_date - self.trade_date).days
    
    @classmethod
    def calculate_t2_settlement(cls, trade_date: date, 
                               trading_calendar: Dict[date, TaiwanTradingCalendar]) -> 'TaiwanSettlement':
        """Calculate T+2 settlement date considering Taiwan holidays."""
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
            settlement_date=current_date,
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


def create_taiwan_trading_calendar(year: int) -> Dict[date, TaiwanTradingCalendar]:
    """Create Taiwan trading calendar for a given year."""
    calendar = {}
    
    # This is a simplified implementation
    # Real implementation would fetch from Taiwan Stock Exchange
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    
    current_date = start_date
    while current_date <= end_date:
        # Simplified: weekdays are trading days
        is_trading = current_date.weekday() < 5
        
        # Add known holidays (simplified list)
        taiwan_holidays = {
            date(year, 1, 1),   # New Year's Day
            date(year, 10, 10), # National Day
            # Add more holidays as needed
        }
        
        if current_date in taiwan_holidays:
            is_trading = False
            
        calendar[current_date] = TaiwanTradingCalendar(
            date=current_date,
            is_trading_day=is_trading,
            market_session=MarketSession.MORNING if is_trading else MarketSession.CLOSED
        )
        
        current_date += timedelta(days=1)
    
    return calendar


def get_taiwan_market_timezone() -> str:
    """Get Taiwan market timezone string."""
    return "Asia/Taipei"


def taiwan_market_time_to_utc(market_time: datetime) -> datetime:
    """Convert Taiwan market time to UTC."""
    # Simplified - real implementation would use pytz
    # Taiwan is UTC+8
    return market_time - timedelta(hours=8)