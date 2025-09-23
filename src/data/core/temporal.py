"""
Core temporal data management for point-in-time access.

This module provides the foundation for temporal data operations in the ML4T system,
ensuring no look-ahead bias and proper handling of Taiwan market timing constraints.
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Data types with different temporal characteristics."""
    PRICE = "price"  # End-of-day prices
    VOLUME = "volume"  # Trading volumes
    FUNDAMENTAL = "fundamental"  # Financial statements (60-day lag)
    CORPORATE_ACTION = "corporate_action"  # Dividends, splits, etc.
    MARKET_DATA = "market_data"  # General market data
    NEWS = "news"  # News and events
    TECHNICAL = "technical"  # Technical indicators


class MarketSession(Enum):
    """Taiwan market trading sessions."""
    MORNING = "morning"  # 09:00-13:30 TST
    CLOSED = "closed"
    HOLIDAY = "holiday"
    SUSPENDED = "suspended"


@dataclass
class TemporalValue:
    """A value with temporal metadata."""
    value: Any
    as_of_date: date  # When this data was known
    value_date: date  # What date this data refers to
    data_type: DataType
    symbol: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    version: int = 1
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class SettlementInfo:
    """Settlement timing information for Taiwan market."""
    trade_date: date
    settlement_date: date
    is_trading_day: bool
    market_session: MarketSession
    
    @property
    def settlement_lag_days(self) -> int:
        """Calculate settlement lag in business days."""
        return (self.settlement_date - self.trade_date).days


class TemporalIndex:
    """Efficient temporal indexing for point-in-time queries."""
    
    def __init__(self):
        self._symbol_index: Dict[str, List[TemporalValue]] = {}
        self._date_index: Dict[date, List[TemporalValue]] = {}
        self._type_index: Dict[DataType, List[TemporalValue]] = {}
        
    def add(self, value: TemporalValue) -> None:
        """Add a temporal value to all indices."""
        # Symbol index
        if value.symbol:
            if value.symbol not in self._symbol_index:
                self._symbol_index[value.symbol] = []
            self._symbol_index[value.symbol].append(value)
            
        # Date index
        if value.as_of_date not in self._date_index:
            self._date_index[value.as_of_date] = []
        self._date_index[value.as_of_date].append(value)
        
        # Type index
        if value.data_type not in self._type_index:
            self._type_index[value.data_type] = []
        self._type_index[value.data_type].append(value)
        
    def get_by_symbol_date(self, symbol: str, as_of_date: date, 
                          data_type: Optional[DataType] = None) -> List[TemporalValue]:
        """Get all values for a symbol as of a specific date."""
        symbol_values = self._symbol_index.get(symbol, [])
        
        # Filter by as_of_date (only data known as of that date)
        filtered = [v for v in symbol_values if v.as_of_date <= as_of_date]
        
        # Filter by data type if specified
        if data_type:
            filtered = [v for v in filtered if v.data_type == data_type]
            
        # Sort by as_of_date descending to get most recent
        return sorted(filtered, key=lambda x: x.as_of_date, reverse=True)


class TemporalStore(ABC):
    """Abstract base class for temporal data storage."""
    
    @abstractmethod
    def store(self, value: TemporalValue) -> None:
        """Store a temporal value."""
        pass
        
    @abstractmethod
    def get_point_in_time(self, symbol: str, as_of_date: date, 
                         data_type: DataType) -> Optional[TemporalValue]:
        """Get the most recent value as of a specific date."""
        pass
        
    @abstractmethod
    def get_range(self, symbol: str, start_date: date, end_date: date,
                 data_type: DataType) -> List[TemporalValue]:
        """Get all values in a date range."""
        pass
        
    @abstractmethod
    def validate_no_lookahead(self, symbol: str, query_date: date,
                             data_date: date) -> bool:
        """Validate that no look-ahead bias exists."""
        pass


class InMemoryTemporalStore(TemporalStore):
    """In-memory implementation for development and testing."""
    
    def __init__(self):
        self._index = TemporalIndex()
        self._values: List[TemporalValue] = []
        
    def store(self, value: TemporalValue) -> None:
        """Store a temporal value."""
        self._values.append(value)
        self._index.add(value)
        logger.debug(f"Stored temporal value: {value.symbol} {value.data_type} "
                    f"as_of={value.as_of_date} value_date={value.value_date}")
        
    def get_point_in_time(self, symbol: str, as_of_date: date, 
                         data_type: DataType) -> Optional[TemporalValue]:
        """Get the most recent value as of a specific date."""
        values = self._index.get_by_symbol_date(symbol, as_of_date, data_type)
        return values[0] if values else None
        
    def get_range(self, symbol: str, start_date: date, end_date: date,
                 data_type: DataType) -> List[TemporalValue]:
        """Get all values in a date range."""
        values = self._index.get_by_symbol_date(symbol, end_date, data_type)
        return [v for v in values if start_date <= v.value_date <= end_date]
        
    def validate_no_lookahead(self, symbol: str, query_date: date,
                             data_date: date) -> bool:
        """Validate that no look-ahead bias exists."""
        # Data should only be accessible after it was known
        value = self.get_point_in_time(symbol, query_date, DataType.PRICE)
        if value and value.value_date > query_date:
            logger.warning(f"Potential look-ahead bias: accessing data from "
                          f"{value.value_date} on {query_date}")
            return False
        return True


class TemporalDataManager:
    """Main interface for temporal data operations."""
    
    def __init__(self, store: TemporalStore):
        self.store = store
        self._settlement_cache: Dict[date, SettlementInfo] = {}
        
    def get_taiwan_settlement_date(self, trade_date: date) -> date:
        """Calculate T+2 settlement date for Taiwan market."""
        # Taiwan uses T+2 settlement
        # This is a simplified version - real implementation would check
        # for holidays and weekends
        settlement_date = trade_date + timedelta(days=2)
        
        # Skip weekends (simplified)
        while settlement_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            settlement_date += timedelta(days=1)
            
        return settlement_date
        
    def get_with_settlement_lag(self, symbol: str, trade_date: date,
                               data_type: DataType) -> Optional[TemporalValue]:
        """Get data accounting for settlement lag."""
        if data_type in [DataType.PRICE, DataType.VOLUME]:
            # Price and volume data available immediately
            as_of_date = trade_date
        elif data_type == DataType.FUNDAMENTAL:
            # Financial data has 60-day lag
            as_of_date = trade_date - timedelta(days=60)
        else:
            # Default case
            as_of_date = trade_date
            
        return self.store.get_point_in_time(symbol, as_of_date, data_type)
        
    def validate_temporal_consistency(self, symbol: str, 
                                    request_date: date) -> List[str]:
        """Validate temporal consistency for a symbol."""
        issues = []
        
        # Check for future data access
        if not self.store.validate_no_lookahead(symbol, request_date, request_date):
            issues.append("Potential look-ahead bias detected")
            
        # Check settlement consistency
        settlement_date = self.get_taiwan_settlement_date(request_date)
        if settlement_date <= request_date:
            issues.append(f"Settlement date {settlement_date} not after trade date {request_date}")
            
        return issues
        
    def create_temporal_snapshot(self, symbols: List[str], as_of_date: date,
                               data_types: List[DataType]) -> Dict[str, Dict[DataType, TemporalValue]]:
        """Create a consistent temporal snapshot for multiple symbols."""
        snapshot = {}
        
        for symbol in symbols:
            snapshot[symbol] = {}
            for data_type in data_types:
                value = self.store.get_point_in_time(symbol, as_of_date, data_type)
                if value:
                    snapshot[symbol][data_type] = value
                    
        return snapshot


# Utility functions for temporal operations

def calculate_data_lag(data_type: DataType) -> timedelta:
    """Calculate expected data lag for different data types."""
    lags = {
        DataType.PRICE: timedelta(days=0),
        DataType.VOLUME: timedelta(days=0), 
        DataType.FUNDAMENTAL: timedelta(days=60),
        DataType.CORPORATE_ACTION: timedelta(days=7),  # Average
        DataType.MARKET_DATA: timedelta(days=0),
        DataType.NEWS: timedelta(days=0),
        DataType.TECHNICAL: timedelta(days=0),
    }
    return lags.get(data_type, timedelta(days=0))


def is_taiwan_trading_day(date_to_check: date) -> bool:
    """Check if a given date is a Taiwan trading day."""
    # Simplified implementation - real version would check holidays
    return date_to_check.weekday() < 5  # Monday = 0, Friday = 4


def get_previous_trading_day(current_date: date) -> date:
    """Get the previous Taiwan trading day."""
    prev_date = current_date - timedelta(days=1)
    while not is_taiwan_trading_day(prev_date):
        prev_date -= timedelta(days=1)
    return prev_date


def validate_temporal_order(values: List[TemporalValue]) -> bool:
    """Validate that temporal values are in correct chronological order."""
    for i in range(1, len(values)):
        if values[i].as_of_date < values[i-1].as_of_date:
            return False
    return True