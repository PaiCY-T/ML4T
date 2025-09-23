"""
Point-in-Time (PIT) Query Engine with Look-Ahead Bias Prevention.

This module provides the core query engine for accessing temporal data
without look-ahead bias, optimized for high-performance backtesting scenarios.
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from abc import ABC, abstractmethod
import bisect
from collections import defaultdict

from ..core.temporal import (
    TemporalValue, TemporalStore, DataType, TemporalDataManager,
    PostgreSQLTemporalStore, calculate_data_lag, validate_temporal_order
)
from ..models.taiwan_market import (
    TaiwanSettlement, TaiwanTradingCalendar, TaiwanMarketDataValidator,
    create_taiwan_trading_calendar, TradingStatus
)

logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Query execution modes."""
    STRICT = "strict"      # Strict temporal consistency
    FAST = "fast"          # Optimized for speed
    BULK = "bulk"          # Bulk query optimization
    REALTIME = "realtime"  # Real-time streaming


class BiasCheckLevel(Enum):
    """Look-ahead bias checking levels."""
    NONE = "none"          # No bias checking
    BASIC = "basic"        # Basic date validation
    STRICT = "strict"      # Comprehensive bias checking
    PARANOID = "paranoid"  # Maximum validation


@dataclass
class PITQuery:
    """Point-in-time query specification."""
    symbols: List[str]
    as_of_date: date
    data_types: List[DataType]
    mode: QueryMode = QueryMode.STRICT
    bias_check: BiasCheckLevel = BiasCheckLevel.STRICT
    max_lag_days: Optional[int] = None
    include_metadata: bool = True
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    
    def __post_init__(self):
        """Validate query parameters."""
        if not self.symbols:
            raise ValueError("At least one symbol required")
        if not self.data_types:
            raise ValueError("At least one data type required")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("Start date must be before or equal to end date")


@dataclass
class PITResult:
    """Point-in-time query result."""
    query: PITQuery
    data: Dict[str, Dict[DataType, TemporalValue]]
    execution_time_ms: float
    cache_hits: int = 0
    total_queries: int = 0
    bias_violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return self.cache_hits / max(self.total_queries, 1)
    
    def get_value(self, symbol: str, data_type: DataType) -> Optional[TemporalValue]:
        """Get a specific value from the result."""
        return self.data.get(symbol, {}).get(data_type)
    
    def has_complete_data(self, symbol: str) -> bool:
        """Check if all requested data types are available for symbol."""
        symbol_data = self.data.get(symbol, {})
        return all(dt in symbol_data for dt in self.query.data_types)


class PITCache:
    """High-performance cache for point-in-time queries."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[TemporalValue, datetime]] = {}
        self._access_order: List[str] = []
        
    def _make_key(self, symbol: str, data_type: DataType, as_of_date: date) -> str:
        """Create cache key."""
        return f"{symbol}:{data_type.value}:{as_of_date.isoformat()}"
    
    def get(self, symbol: str, data_type: DataType, as_of_date: date) -> Optional[TemporalValue]:
        """Get cached value."""
        key = self._make_key(symbol, data_type, as_of_date)
        
        if key in self._cache:
            value, timestamp = self._cache[key]
            
            # Check TTL
            if (datetime.utcnow() - timestamp).total_seconds() < self.ttl_seconds:
                # Move to end (LRU)
                self._access_order.remove(key)
                self._access_order.append(key)
                return value
            else:
                # Expired
                del self._cache[key]
                self._access_order.remove(key)
        
        return None
    
    def put(self, symbol: str, data_type: DataType, as_of_date: date, value: TemporalValue):
        """Cache a value."""
        key = self._make_key(symbol, data_type, as_of_date)
        
        # Evict if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Remove least recently used
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        # Add/update cache
        self._cache[key] = (value, datetime.utcnow())
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size
        }


class BiasDetector:
    """Detects and prevents look-ahead bias in temporal queries."""
    
    def __init__(self, trading_calendar: Dict[date, TaiwanTradingCalendar]):
        self.trading_calendar = trading_calendar
        self.validator = TaiwanMarketDataValidator()
        
    def check_query_bias(self, query: PITQuery) -> List[str]:
        """Check query for potential bias issues."""
        violations = []
        
        if query.bias_check == BiasCheckLevel.NONE:
            return violations
            
        # Check if query date is in the future
        if query.as_of_date > date.today():
            violations.append(f"Query date {query.as_of_date} is in the future")
            
        # Check data type compatibility with query date
        for data_type in query.data_types:
            expected_lag = calculate_data_lag(data_type)
            earliest_available = query.as_of_date - expected_lag
            
            if earliest_available > query.as_of_date:
                violations.append(
                    f"Data type {data_type.value} not available until "
                    f"{expected_lag.days} days after query date"
                )
        
        return violations
    
    def check_value_bias(self, value: TemporalValue, query_date: date) -> List[str]:
        """Check if a temporal value violates look-ahead constraints."""
        violations = []
        
        # Basic temporal consistency
        if value.as_of_date > query_date:
            violations.append(
                f"Value as_of_date {value.as_of_date} is after query date {query_date}"
            )
            
        if value.value_date > query_date:
            violations.append(
                f"Value date {value.value_date} is after query date {query_date}"
            )
            
        # Data type specific checks
        expected_lag = calculate_data_lag(value.data_type)
        if value.as_of_date < value.value_date + expected_lag:
            violations.append(
                f"Value available before expected lag of {expected_lag.days} days"
            )
            
        return violations
    
    def validate_settlement_timing(self, symbol: str, trade_date: date, 
                                  settlement_date: date) -> List[str]:
        """Validate settlement timing for Taiwan market."""
        violations = []
        
        # T+2 minimum for Taiwan
        min_settlement = trade_date + timedelta(days=2)
        if settlement_date < min_settlement:
            violations.append(
                f"Settlement date {settlement_date} violates T+2 rule for trade date {trade_date}"
            )
            
        return violations


class PITQueryOptimizer:
    """Optimizes point-in-time queries for performance."""
    
    def __init__(self):
        self.symbol_affinity: Dict[str, float] = {}  # Cache affinity scores
        self.query_patterns: Dict[str, int] = {}     # Query pattern frequencies
        
    def optimize_query(self, query: PITQuery) -> PITQuery:
        """Optimize query for better performance."""
        # Sort symbols by cache affinity
        optimized_symbols = sorted(
            query.symbols, 
            key=lambda s: self.symbol_affinity.get(s, 0), 
            reverse=True
        )
        
        # Create optimized query
        return PITQuery(
            symbols=optimized_symbols,
            as_of_date=query.as_of_date,
            data_types=query.data_types,
            mode=query.mode,
            bias_check=query.bias_check,
            max_lag_days=query.max_lag_days,
            include_metadata=query.include_metadata
        )
    
    def update_affinity(self, symbol: str, cache_hit: bool):
        """Update symbol cache affinity based on query results."""
        current = self.symbol_affinity.get(symbol, 0.5)
        
        # Exponential moving average
        alpha = 0.1
        new_score = current + alpha * (1.0 if cache_hit else -0.5)
        self.symbol_affinity[symbol] = max(0.0, min(1.0, new_score))


class PointInTimeEngine:
    """High-performance point-in-time query engine."""
    
    def __init__(self, 
                 store: TemporalStore,
                 trading_calendar: Optional[Dict[date, TaiwanTradingCalendar]] = None,
                 enable_cache: bool = True,
                 max_workers: int = 4):
        self.store = store
        self.trading_calendar = trading_calendar or create_taiwan_trading_calendar(2024)
        self.cache = PITCache() if enable_cache else None
        self.bias_detector = BiasDetector(self.trading_calendar)
        self.optimizer = PITQueryOptimizer()
        self.max_workers = max_workers
        
        # Performance metrics
        self.query_count = 0
        self.total_execution_time = 0.0
        self.bias_violation_count = 0
        
    def execute_query(self, query: PITQuery) -> PITResult:
        """Execute a point-in-time query."""
        start_time = datetime.utcnow()
        
        # Optimize query
        optimized_query = self.optimizer.optimize_query(query)
        
        # Check for bias violations
        bias_violations = self.bias_detector.check_query_bias(optimized_query)
        
        if bias_violations and optimized_query.bias_check == BiasCheckLevel.STRICT:
            logger.error(f"Bias violations detected: {bias_violations}")
            raise ValueError(f"Look-ahead bias detected: {'; '.join(bias_violations)}")
        
        # Execute based on query mode
        if optimized_query.mode == QueryMode.BULK:
            result_data = self._execute_bulk_query(optimized_query)
        elif optimized_query.mode == QueryMode.FAST:
            result_data = self._execute_fast_query(optimized_query)
        else:
            result_data = self._execute_strict_query(optimized_query)
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Update metrics
        self.query_count += 1
        self.total_execution_time += execution_time
        if bias_violations:
            self.bias_violation_count += 1
        
        return PITResult(
            query=optimized_query,
            data=result_data,
            execution_time_ms=execution_time,
            bias_violations=bias_violations,
            total_queries=len(optimized_query.symbols) * len(optimized_query.data_types)
        )
    
    def _execute_strict_query(self, query: PITQuery) -> Dict[str, Dict[DataType, TemporalValue]]:
        """Execute query with strict temporal consistency."""
        result = {}
        
        for symbol in query.symbols:
            result[symbol] = {}
            
            for data_type in query.data_types:
                # Check cache first
                cached_value = None
                if self.cache:
                    cached_value = self.cache.get(symbol, data_type, query.as_of_date)
                
                if cached_value:
                    # Validate cached value for bias
                    if query.bias_check != BiasCheckLevel.NONE:
                        violations = self.bias_detector.check_value_bias(cached_value, query.as_of_date)
                        if violations and query.bias_check == BiasCheckLevel.STRICT:
                            logger.warning(f"Cached value has bias violations: {violations}")
                            continue
                    
                    result[symbol][data_type] = cached_value
                    self.optimizer.update_affinity(symbol, True)
                else:
                    # Query store
                    value = self.store.get_point_in_time(symbol, query.as_of_date, data_type)
                    
                    if value:
                        # Validate for bias
                        if query.bias_check != BiasCheckLevel.NONE:
                            violations = self.bias_detector.check_value_bias(value, query.as_of_date)
                            if violations and query.bias_check == BiasCheckLevel.STRICT:
                                logger.warning(f"Store value has bias violations: {violations}")
                                continue
                        
                        result[symbol][data_type] = value
                        
                        # Cache the result
                        if self.cache:
                            self.cache.put(symbol, data_type, query.as_of_date, value)
                    
                    self.optimizer.update_affinity(symbol, False)
        
        return result
    
    def _execute_fast_query(self, query: PITQuery) -> Dict[str, Dict[DataType, TemporalValue]]:
        """Execute query optimized for speed (minimal bias checking)."""
        result = {}
        
        # Use parallel execution for multiple symbols
        if len(query.symbols) > 1 and self.max_workers > 1:
            return self._execute_parallel_query(query)
        
        # Sequential execution for single symbol or small queries
        for symbol in query.symbols:
            result[symbol] = {}
            
            for data_type in query.data_types:
                # Check cache first
                if self.cache:
                    cached_value = self.cache.get(symbol, data_type, query.as_of_date)
                    if cached_value:
                        result[symbol][data_type] = cached_value
                        continue
                
                # Query store
                value = self.store.get_point_in_time(symbol, query.as_of_date, data_type)
                if value:
                    result[symbol][data_type] = value
                    
                    if self.cache:
                        self.cache.put(symbol, data_type, query.as_of_date, value)
        
        return result
    
    def _execute_bulk_query(self, query: PITQuery) -> Dict[str, Dict[DataType, TemporalValue]]:
        """Execute bulk query optimized for large datasets."""
        # Use range queries when possible
        result = {}
        
        for symbol in query.symbols:
            result[symbol] = {}
            
            for data_type in query.data_types:
                # For bulk queries, get a range around the query date
                start_date = query.as_of_date - timedelta(days=7)
                end_date = query.as_of_date
                
                values = self.store.get_range(symbol, start_date, end_date, data_type)
                
                # Find the most recent value as of query date
                applicable_values = [v for v in values if v.as_of_date <= query.as_of_date]
                
                if applicable_values:
                    # Sort by as_of_date and take the most recent
                    latest_value = max(applicable_values, key=lambda x: x.as_of_date)
                    result[symbol][data_type] = latest_value
        
        return result
    
    def _execute_parallel_query(self, query: PITQuery) -> Dict[str, Dict[DataType, TemporalValue]]:
        """Execute query using parallel workers."""
        result = {}
        
        def query_symbol(symbol: str) -> Tuple[str, Dict[DataType, TemporalValue]]:
            symbol_result = {}
            
            for data_type in query.data_types:
                if self.cache:
                    cached_value = self.cache.get(symbol, data_type, query.as_of_date)
                    if cached_value:
                        symbol_result[data_type] = cached_value
                        continue
                
                value = self.store.get_point_in_time(symbol, query.as_of_date, data_type)
                if value:
                    symbol_result[data_type] = value
                    
                    if self.cache:
                        self.cache.put(symbol, data_type, query.as_of_date, value)
            
            return symbol, symbol_result
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(query_symbol, symbol) for symbol in query.symbols]
            
            for future in as_completed(futures):
                symbol, symbol_data = future.result()
                result[symbol] = symbol_data
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        avg_execution_time = (
            self.total_execution_time / max(self.query_count, 1)
        )
        
        stats = {
            "query_count": self.query_count,
            "avg_execution_time_ms": avg_execution_time,
            "total_execution_time_ms": self.total_execution_time,
            "bias_violation_count": self.bias_violation_count,
            "bias_violation_rate": self.bias_violation_count / max(self.query_count, 1)
        }
        
        if self.cache:
            stats["cache_stats"] = self.cache.get_stats()
        
        return stats
    
    def warm_cache(self, symbols: List[str], date_range: Tuple[date, date], 
                   data_types: List[DataType]):
        """Pre-warm cache with frequently accessed data."""
        logger.info(f"Warming cache for {len(symbols)} symbols over {date_range}")
        
        start_date, end_date = date_range
        current_date = start_date
        
        while current_date <= end_date:
            query = PITQuery(
                symbols=symbols,
                as_of_date=current_date,
                data_types=data_types,
                mode=QueryMode.FAST,
                bias_check=BiasCheckLevel.BASIC
            )
            
            self.execute_query(query)
            current_date += timedelta(days=1)
        
        logger.info("Cache warming completed")
    
    def clear_cache(self):
        """Clear query cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Query cache cleared")
    
    def query(self, query: PITQuery, bias_check_level: Optional[BiasCheckLevel] = None) -> Dict[str, List[TemporalValue]]:
        """
        Execute query and return data in format expected by validation framework.
        
        Args:
            query: PITQuery specification
            bias_check_level: Override bias check level
            
        Returns:
            Dictionary mapping symbols to lists of temporal values
        """
        if bias_check_level:
            query.bias_check = bias_check_level
        
        result = self.execute_query(query)
        
        # Convert result format for validation framework
        converted_result = {}
        for symbol, symbol_data in result.data.items():
            converted_result[symbol] = []
            for data_type, temporal_value in symbol_data.items():
                if temporal_value:
                    converted_result[symbol].append(temporal_value)
        
        return converted_result
    
    def check_data_availability(self, query: PITQuery) -> bool:
        """
        Check if sufficient data is available for the query.
        
        Args:
            query: PITQuery specification
            
        Returns:
            True if data is available for the query
        """
        try:
            # Execute query to check availability
            result = self.execute_query(query)
            
            # Check if we have data for all requested symbols and data types
            for symbol in query.symbols:
                if symbol not in result.data:
                    return False
                
                symbol_data = result.data[symbol]
                for data_type in query.data_types:
                    if data_type not in symbol_data or symbol_data[data_type] is None:
                        return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Data availability check failed: {e}")
            return False


def create_temporal_store(connection_params: Optional[Dict[str, Any]] = None,
                         store_type: str = "auto") -> TemporalStore:
    """Factory function to create the appropriate temporal store.
    
    Args:
        connection_params: Database connection parameters for PostgreSQL store
        store_type: "auto", "postgresql", or "memory"
    
    Returns:
        TemporalStore: Configured temporal store instance
    """
    if store_type == "auto":
        # Auto-detect based on availability
        if connection_params:
            store_type = "postgresql"
        else:
            store_type = "memory"
    
    if store_type == "postgresql":
        if not connection_params:
            raise ValueError("PostgreSQL store requires connection_params")
        return PostgreSQLTemporalStore(connection_params)
    elif store_type == "memory":
        from ..core.temporal import InMemoryTemporalStore
        return InMemoryTemporalStore()
    else:
        raise ValueError(f"Unknown store type: {store_type}")


class OptimizedPointInTimeEngine(PointInTimeEngine):
    """Enhanced PIT engine with SQL-specific optimizations."""
    
    def __init__(self,
                 store: TemporalStore,
                 trading_calendar: Optional[Dict[date, TaiwanTradingCalendar]] = None,
                 enable_cache: bool = True,
                 max_workers: int = 4,
                 bulk_threshold: int = 100):
        super().__init__(store, trading_calendar, enable_cache, max_workers)
        self.bulk_threshold = bulk_threshold
        self._is_sql_store = isinstance(store, PostgreSQLTemporalStore)
        
    def execute_bulk_optimized_query(self, queries: List[PITQuery]) -> List[PITResult]:
        """Execute multiple queries with SQL-level optimization."""
        if not self._is_sql_store or len(queries) < self.bulk_threshold:
            # Fall back to individual queries
            return [self.execute_query(query) for query in queries]
        
        # SQL-specific bulk optimization
        return self._execute_sql_bulk_queries(queries)
    
    def _execute_sql_bulk_queries(self, queries: List[PITQuery]) -> List[PITResult]:
        """Execute bulk queries using SQL-specific optimizations."""
        # Group queries by data types and date ranges for optimal SQL execution
        grouped_queries = self._group_queries_for_sql(queries)
        results = []
        
        for group in grouped_queries:
            # Execute each group as a single SQL query
            group_results = self._execute_sql_query_group(group)
            results.extend(group_results)
        
        return results
    
    def _group_queries_for_sql(self, queries: List[PITQuery]) -> List[List[PITQuery]]:
        """Group queries for optimal SQL execution."""
        # Simple grouping by data types for now
        # Could be enhanced with more sophisticated grouping logic
        groups = {}
        
        for query in queries:
            key = (tuple(sorted([dt.value for dt in query.data_types])), query.mode)
            if key not in groups:
                groups[key] = []
            groups[key].append(query)
        
        return list(groups.values())
    
    def _execute_sql_query_group(self, queries: List[PITQuery]) -> List[PITResult]:
        """Execute a group of similar queries using SQL bulk operations."""
        # This is a simplified implementation
        # Real implementation would generate optimized SQL for bulk retrieval
        return [self.execute_query(query) for query in queries]
    
    def warm_cache_sql_optimized(self, symbols: List[str], 
                                date_range: Tuple[date, date],
                                data_types: List[DataType]):
        """SQL-optimized cache warming using bulk queries."""
        if not self._is_sql_store:
            return super().warm_cache(symbols, date_range, data_types)
        
        logger.info(f"SQL-optimized cache warming for {len(symbols)} symbols")
        
        start_date, end_date = date_range
        
        # Generate single bulk query for all symbols/dates/types
        bulk_query = PITQuery(
            symbols=symbols,
            as_of_date=end_date,  # Get all data up to end date
            data_types=data_types,
            mode=QueryMode.BULK,
            bias_check=BiasCheckLevel.BASIC
        )
        
        # Execute and let the SQL store handle bulk operations efficiently
        result = self.execute_query(bulk_query)
        
        logger.info(f"Cache warming completed with {result.total_queries} queries")


def create_optimized_pit_engine(connection_params: Optional[Dict[str, Any]] = None,
                               store_type: str = "auto",
                               enable_cache: bool = True,
                               max_workers: int = 4) -> PointInTimeEngine:
    """Factory function to create an optimized PIT engine.
    
    Args:
        connection_params: Database connection parameters
        store_type: Type of temporal store to use
        enable_cache: Whether to enable query caching
        max_workers: Maximum worker threads for parallel queries
    
    Returns:
        PointInTimeEngine: Configured PIT engine instance
    """
    store = create_temporal_store(connection_params, store_type)
    trading_calendar = create_taiwan_trading_calendar(2024)
    
    if isinstance(store, PostgreSQLTemporalStore):
        return OptimizedPointInTimeEngine(
            store=store,
            trading_calendar=trading_calendar,
            enable_cache=enable_cache,
            max_workers=max_workers
        )
    else:
        return PointInTimeEngine(
            store=store,
            trading_calendar=trading_calendar,
            enable_cache=enable_cache,
            max_workers=max_workers
        )