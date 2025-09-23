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
import json
from decimal import Decimal

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

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


class PostgreSQLTemporalStore(TemporalStore):
    """PostgreSQL-backed temporal store for production use."""
    
    def __init__(self, connection_params: Dict[str, Any], pool_size: int = 10):
        if not HAS_POSTGRES:
            raise ImportError("psycopg2 is required for PostgreSQL temporal store")
        
        self.connection_params = connection_params
        self.pool = SimpleConnectionPool(1, pool_size, **connection_params)
        self._initialize_schema()
        
    def _initialize_schema(self):
        """Initialize database schema for temporal data."""
        schema_sql = """
        -- Point-in-time data table
        CREATE TABLE IF NOT EXISTS pit_data (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            data_type VARCHAR(50) NOT NULL,
            as_of_date DATE NOT NULL,
            value_date DATE NOT NULL,
            value_numeric DECIMAL(20,6),
            value_text TEXT,
            value_json JSONB,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            version INTEGER DEFAULT 1,
            -- Indexes for efficient querying
            CONSTRAINT pit_data_unique UNIQUE (symbol, data_type, as_of_date, value_date, version)
        );
        
        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_pit_symbol_asof_type ON pit_data (symbol, as_of_date, data_type);
        CREATE INDEX IF NOT EXISTS idx_pit_value_date ON pit_data (value_date);
        CREATE INDEX IF NOT EXISTS idx_pit_data_type ON pit_data (data_type);
        CREATE INDEX IF NOT EXISTS idx_pit_created_at ON pit_data (created_at);
        CREATE INDEX IF NOT EXISTS idx_pit_metadata_gin ON pit_data USING GIN (metadata);
        
        -- Settlement calendar table
        CREATE TABLE IF NOT EXISTS settlement_calendar (
            trade_date DATE PRIMARY KEY,
            settlement_date DATE NOT NULL,
            is_trading_day BOOLEAN NOT NULL DEFAULT TRUE,
            market_session VARCHAR(20) DEFAULT 'morning',
            notes TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Index for settlement calendar
        CREATE INDEX IF NOT EXISTS idx_settlement_date ON settlement_calendar (settlement_date);
        """
        
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(schema_sql)
                conn.commit()
                logger.info("PostgreSQL temporal store schema initialized")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize schema: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def _serialize_value(self, value: Any) -> Tuple[Optional[Decimal], Optional[str], Optional[Dict]]:
        """Serialize value into appropriate database columns."""
        if isinstance(value, (int, float, Decimal)):
            return Decimal(str(value)), None, None
        elif isinstance(value, str):
            return None, value, None
        elif isinstance(value, (dict, list)):
            return None, None, value
        else:
            # Convert to JSON for complex types
            return None, None, {"type": type(value).__name__, "value": str(value)}
    
    def _deserialize_value(self, numeric_val: Optional[Decimal], 
                          text_val: Optional[str], 
                          json_val: Optional[Dict]) -> Any:
        """Deserialize value from database columns."""
        if numeric_val is not None:
            return numeric_val
        elif text_val is not None:
            return text_val
        elif json_val is not None:
            if isinstance(json_val, dict) and "type" in json_val and "value" in json_val:
                # Handle serialized complex types
                return json_val["value"]
            return json_val
        else:
            return None
    
    def store(self, value: TemporalValue) -> None:
        """Store a temporal value in PostgreSQL."""
        conn = self.pool.getconn()
        try:
            numeric_val, text_val, json_val = self._serialize_value(value.value)
            
            with conn.cursor() as cursor:
                # Use ON CONFLICT to handle duplicates (update version)
                sql = """
                INSERT INTO pit_data (
                    symbol, data_type, as_of_date, value_date,
                    value_numeric, value_text, value_json, metadata, created_at, version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, data_type, as_of_date, value_date, version)
                DO UPDATE SET
                    value_numeric = EXCLUDED.value_numeric,
                    value_text = EXCLUDED.value_text,
                    value_json = EXCLUDED.value_json,
                    metadata = EXCLUDED.metadata,
                    created_at = EXCLUDED.created_at
                """
                
                cursor.execute(sql, (
                    value.symbol,
                    value.data_type.value,
                    value.as_of_date,
                    value.value_date,
                    numeric_val,
                    text_val,
                    json.dumps(json_val) if json_val else None,
                    json.dumps(value.metadata) if value.metadata else None,
                    value.created_at or datetime.utcnow(),
                    value.version
                ))
                
                conn.commit()
                logger.debug(f"Stored temporal value: {value.symbol} {value.data_type} "
                           f"as_of={value.as_of_date} value_date={value.value_date}")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store temporal value: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def get_point_in_time(self, symbol: str, as_of_date: date, 
                         data_type: DataType) -> Optional[TemporalValue]:
        """Get the most recent value as of a specific date."""
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                sql = """
                SELECT * FROM pit_data
                WHERE symbol = %s 
                AND data_type = %s
                AND as_of_date <= %s
                ORDER BY as_of_date DESC, version DESC
                LIMIT 1
                """
                
                cursor.execute(sql, (symbol, data_type.value, as_of_date))
                row = cursor.fetchone()
                
                if row:
                    value = self._deserialize_value(
                        row['value_numeric'], 
                        row['value_text'], 
                        json.loads(row['value_json']) if row['value_json'] else None
                    )
                    
                    metadata = json.loads(row['metadata']) if row['metadata'] else None
                    
                    return TemporalValue(
                        value=value,
                        as_of_date=row['as_of_date'],
                        value_date=row['value_date'],
                        data_type=DataType(row['data_type']),
                        symbol=row['symbol'],
                        metadata=metadata,
                        created_at=row['created_at'],
                        version=row['version']
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get point-in-time value: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def get_range(self, symbol: str, start_date: date, end_date: date,
                 data_type: DataType) -> List[TemporalValue]:
        """Get all values in a date range."""
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                sql = """
                SELECT * FROM pit_data
                WHERE symbol = %s 
                AND data_type = %s
                AND value_date >= %s
                AND value_date <= %s
                ORDER BY value_date ASC, as_of_date DESC, version DESC
                """
                
                cursor.execute(sql, (symbol, data_type.value, start_date, end_date))
                rows = cursor.fetchall()
                
                values = []
                for row in rows:
                    value = self._deserialize_value(
                        row['value_numeric'], 
                        row['value_text'], 
                        json.loads(row['value_json']) if row['value_json'] else None
                    )
                    
                    metadata = json.loads(row['metadata']) if row['metadata'] else None
                    
                    values.append(TemporalValue(
                        value=value,
                        as_of_date=row['as_of_date'],
                        value_date=row['value_date'],
                        data_type=DataType(row['data_type']),
                        symbol=row['symbol'],
                        metadata=metadata,
                        created_at=row['created_at'],
                        version=row['version']
                    ))
                
                return values
                
        except Exception as e:
            logger.error(f"Failed to get range values: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def validate_no_lookahead(self, symbol: str, query_date: date,
                             data_date: date) -> bool:
        """Validate that no look-ahead bias exists."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Check if any data for the symbol on data_date was available before query_date
                sql = """
                SELECT COUNT(*) FROM pit_data
                WHERE symbol = %s 
                AND value_date = %s
                AND as_of_date > %s
                """
                
                cursor.execute(sql, (symbol, data_date, query_date))
                count = cursor.fetchone()[0]
                
                if count > 0:
                    logger.warning(f"Potential look-ahead bias: {count} records for {symbol} "
                                 f"on {data_date} available after {query_date}")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to validate look-ahead bias: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def bulk_store(self, values: List[TemporalValue]) -> None:
        """Efficiently store multiple temporal values."""
        if not values:
            return
            
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Prepare data for bulk insert
                data = []
                for value in values:
                    numeric_val, text_val, json_val = self._serialize_value(value.value)
                    data.append((
                        value.symbol,
                        value.data_type.value,
                        value.as_of_date,
                        value.value_date,
                        numeric_val,
                        text_val,
                        json.dumps(json_val) if json_val else None,
                        json.dumps(value.metadata) if value.metadata else None,
                        value.created_at or datetime.utcnow(),
                        value.version
                    ))
                
                # Use COPY for maximum performance on large datasets
                sql = """
                INSERT INTO pit_data (
                    symbol, data_type, as_of_date, value_date,
                    value_numeric, value_text, value_json, metadata, created_at, version
                ) VALUES %s
                ON CONFLICT (symbol, data_type, as_of_date, value_date, version)
                DO UPDATE SET
                    value_numeric = EXCLUDED.value_numeric,
                    value_text = EXCLUDED.value_text,
                    value_json = EXCLUDED.value_json,
                    metadata = EXCLUDED.metadata,
                    created_at = EXCLUDED.created_at
                """
                
                from psycopg2.extras import execute_values
                execute_values(cursor, sql, data)
                
                conn.commit()
                logger.info(f"Bulk stored {len(values)} temporal values")
                
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to bulk store temporal values: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics for performance monitoring."""
        conn = self.pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get table statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        COUNT(DISTINCT data_type) as unique_data_types,
                        MIN(as_of_date) as earliest_as_of_date,
                        MAX(as_of_date) as latest_as_of_date,
                        MIN(value_date) as earliest_value_date,
                        MAX(value_date) as latest_value_date
                    FROM pit_data
                """)
                
                stats = dict(cursor.fetchone())
                
                # Get data type distribution
                cursor.execute("""
                    SELECT data_type, COUNT(*) as count
                    FROM pit_data
                    GROUP BY data_type
                    ORDER BY count DESC
                """)
                
                stats['data_type_distribution'] = {
                    row['data_type']: row['count'] 
                    for row in cursor.fetchall()
                }
                
                # Get table size
                cursor.execute("""
                    SELECT pg_size_pretty(pg_total_relation_size('pit_data')) as table_size
                """)
                
                size_result = cursor.fetchone()
                if size_result:
                    stats['table_size'] = size_result['table_size']
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
        finally:
            self.pool.putconn(conn)
    
    def close(self):
        """Close database connection pool."""
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL temporal store connections closed")