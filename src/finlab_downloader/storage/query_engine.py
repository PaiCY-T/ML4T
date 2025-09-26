"""
Query engine for efficient time-series data retrieval from Parquet storage.

Optimized for financial data queries with predicate pushdown,
column pruning, and intelligent caching.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from .base import PartitionKey, StorageError
from .metadata_manager import MetadataManager, TableMetadata

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Supported aggregation types."""
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    LAST = "last"
    COUNT = "count"
    STD = "std"
    VAR = "var"


@dataclass
class TimeSeriesQuery:
    """Time-series query specification."""
    dataset: str
    symbols: Optional[List[str]] = None
    columns: Optional[List[str]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    aggregation: Optional[AggregationType] = None
    aggregation_period: Optional[str] = None  # 'D', 'W', 'M', 'Q', 'Y'
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    order_by: Optional[str] = None
    order_desc: bool = False


@dataclass
class QueryResult:
    """Query execution result with metadata."""
    data: pd.DataFrame
    execution_time_ms: float
    rows_scanned: int
    rows_returned: int
    partitions_accessed: int
    cache_hit: bool
    query_plan: Dict[str, Any]


class QueryEngine:
    """
    High-performance query engine for financial time-series data.

    Features:
    - Predicate pushdown for efficient filtering
    - Column pruning to reduce I/O
    - Intelligent partition pruning
    - Query result caching
    - Parallel execution for multi-partition queries
    """

    def __init__(self,
                 storage_backend,
                 metadata_manager: MetadataManager,
                 enable_cache: bool = True,
                 cache_ttl_seconds: int = 3600):
        """
        Initialize query engine.

        Args:
            storage_backend: Storage backend instance
            metadata_manager: Metadata manager instance
            enable_cache: Whether to enable result caching
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.storage_backend = storage_backend
        self.metadata_manager = metadata_manager
        self.enable_cache = enable_cache
        self.cache_ttl_seconds = cache_ttl_seconds

        # Query cache: query_hash -> (result, timestamp)
        self.query_cache = {}

        # Query statistics
        self.query_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'total_execution_time_ms': 0,
            'total_rows_scanned': 0,
            'total_rows_returned': 0
        }

    def execute_query(self, query: TimeSeriesQuery) -> QueryResult:
        """
        Execute a time-series query.

        Args:
            query: Query specification

        Returns:
            Query result with metadata
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            cached_result = self._get_cached_result(cache_key)

            if cached_result:
                logger.debug(f"Cache hit for query: {query.dataset}")
                self.query_stats['cache_hits'] += 1
                return cached_result

            # Generate query plan
            query_plan = self._generate_query_plan(query)

            # Execute query
            result = self._execute_query_plan(query, query_plan)

            # Cache result if enabled
            if self.enable_cache and not result.cache_hit:
                self._cache_result(cache_key, result)

            # Update statistics
            self._update_query_stats(result)

            # Record query performance in metadata
            self.metadata_manager.record_query_stats(
                query_pattern=self._get_query_pattern(query),
                dataset=query.dataset,
                execution_time_ms=result.execution_time_ms,
                rows_scanned=result.rows_scanned,
                rows_returned=result.rows_returned,
                cache_hit=result.cache_hit
            )

            return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise StorageError(f"Query execution failed: {e}") from e

    def _generate_query_plan(self, query: TimeSeriesQuery) -> Dict[str, Any]:
        """Generate optimized query execution plan."""
        plan = {
            'partitions': [],
            'pushdown_filters': {},
            'column_pruning': [],
            'aggregation_strategy': None,
            'estimated_cost': 0
        }

        # Find relevant partitions
        relevant_tables = self.metadata_manager.list_tables(
            dataset=query.dataset,
            start_date=query.start_date,
            end_date=query.end_date,
            symbols=query.symbols
        )

        plan['partitions'] = [
            {
                'file_path': table.file_path,
                'partition_key': table.partition_key,
                'row_count': table.row_count,
                'date_range': (table.min_date, table.max_date)
            }
            for table in relevant_tables
        ]

        # Column pruning
        if query.columns:
            plan['column_pruning'] = query.columns
        else:
            # Use all available columns from first partition
            if relevant_tables:
                plan['column_pruning'] = relevant_tables[0].columns

        # Pushdown filters
        if query.start_date or query.end_date:
            plan['pushdown_filters']['date'] = {
                'start': query.start_date,
                'end': query.end_date
            }

        if query.symbols:
            plan['pushdown_filters']['symbols'] = query.symbols

        if query.filters:
            plan['pushdown_filters'].update(query.filters)

        # Aggregation strategy
        if query.aggregation:
            plan['aggregation_strategy'] = {
                'type': query.aggregation.value,
                'period': query.aggregation_period,
                'push_to_storage': self._can_pushdown_aggregation(query.aggregation)
            }

        # Estimate query cost
        plan['estimated_cost'] = sum(p['row_count'] for p in plan['partitions'])

        return plan

    def _execute_query_plan(self, query: TimeSeriesQuery, plan: Dict[str, Any]) -> QueryResult:
        """Execute the query plan."""
        start_time = time.time()

        partitions_data = []
        total_rows_scanned = 0
        partitions_accessed = 0

        # Process each partition
        for partition_info in plan['partitions']:
            try:
                # Load partition data with filters
                partition_data = self._load_partition_with_filters(
                    partition_info['file_path'],
                    plan['column_pruning'],
                    plan['pushdown_filters']
                )

                if not partition_data.empty:
                    partitions_data.append(partition_data)
                    total_rows_scanned += partition_info['row_count']
                    partitions_accessed += 1

            except Exception as e:
                logger.warning(f"Error loading partition {partition_info['file_path']}: {e}")
                continue

        # Combine partition data
        if partitions_data:
            combined_data = pd.concat(partitions_data, ignore_index=True)
        else:
            combined_data = pd.DataFrame()

        # Apply additional filters that couldn't be pushed down
        if not combined_data.empty:
            combined_data = self._apply_additional_filters(combined_data, query)

        # Apply aggregation
        if query.aggregation and not combined_data.empty:
            combined_data = self._apply_aggregation(combined_data, query)

        # Apply sorting and limiting
        if not combined_data.empty:
            if query.order_by and query.order_by in combined_data.columns:
                combined_data = combined_data.sort_values(
                    query.order_by,
                    ascending=not query.order_desc
                )

            if query.limit:
                combined_data = combined_data.head(query.limit)

        execution_time_ms = (time.time() - start_time) * 1000
        rows_returned = len(combined_data)

        return QueryResult(
            data=combined_data,
            execution_time_ms=execution_time_ms,
            rows_scanned=total_rows_scanned,
            rows_returned=rows_returned,
            partitions_accessed=partitions_accessed,
            cache_hit=False,
            query_plan=plan
        )

    def _load_partition_with_filters(self,
                                   file_path: str,
                                   columns: List[str],
                                   filters: Dict[str, Any]) -> pd.DataFrame:
        """Load partition data with applied filters."""
        try:
            # Use pyarrow for efficient loading with filters
            table = pq.read_table(
                file_path,
                columns=columns if columns else None,
                use_pandas_metadata=True
            )

            # Apply date filters using pyarrow compute
            if 'date' in filters and filters['date']:
                date_filter = filters['date']
                date_column = None

                # Find date column
                for col_name in table.column_names:
                    if 'date' in col_name.lower():
                        date_column = col_name
                        break

                if date_column:
                    date_col = table.column(date_column)

                    # Build filter expression
                    filter_expr = None
                    if date_filter.get('start'):
                        start_expr = pc.greater_equal(date_col, pa.scalar(date_filter['start']))
                        filter_expr = start_expr

                    if date_filter.get('end'):
                        end_expr = pc.less_equal(date_col, pa.scalar(date_filter['end']))
                        filter_expr = pc.and_(filter_expr, end_expr) if filter_expr else end_expr

                    if filter_expr:
                        table = table.filter(filter_expr)

            # Convert to pandas
            df = table.to_pandas()

            # Apply symbol filters (pandas level)
            if 'symbols' in filters and filters['symbols'] and not df.empty:
                symbol_columns = [col for col in df.columns if 'symbol' in col.lower()]
                if symbol_columns:
                    symbol_col = symbol_columns[0]
                    df = df[df[symbol_col].isin(filters['symbols'])]

            return df

        except Exception as e:
            logger.error(f"Error loading partition {file_path} with filters: {e}")
            return pd.DataFrame()

    def _apply_additional_filters(self, data: pd.DataFrame, query: TimeSeriesQuery) -> pd.DataFrame:
        """Apply filters that couldn't be pushed down to storage."""
        if not query.filters:
            return data

        filtered_data = data.copy()

        for column, filter_spec in query.filters.items():
            if column not in filtered_data.columns:
                continue

            if isinstance(filter_spec, dict):
                # Range filter
                if 'min' in filter_spec:
                    filtered_data = filtered_data[filtered_data[column] >= filter_spec['min']]
                if 'max' in filter_spec:
                    filtered_data = filtered_data[filtered_data[column] <= filter_spec['max']]
                if 'in' in filter_spec:
                    filtered_data = filtered_data[filtered_data[column].isin(filter_spec['in'])]
            else:
                # Equality filter
                filtered_data = filtered_data[filtered_data[column] == filter_spec]

        return filtered_data

    def _apply_aggregation(self, data: pd.DataFrame, query: TimeSeriesQuery) -> pd.DataFrame:
        """Apply aggregation to the data."""
        if not query.aggregation:
            return data

        # Find date column for grouping
        date_column = None
        for col in data.columns:
            if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(data[col]):
                date_column = col
                break

        if not date_column:
            logger.warning("No date column found for aggregation")
            return data

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])

        # Set up grouping
        grouper = None
        if query.aggregation_period:
            grouper = pd.Grouper(key=date_column, freq=query.aggregation_period)
        else:
            grouper = data[date_column].dt.date

        # Apply aggregation
        numeric_columns = data.select_dtypes(include=[float, int]).columns
        agg_func = self._get_aggregation_function(query.aggregation)

        if query.aggregation == AggregationType.FIRST:
            aggregated = data.groupby(grouper).first()
        elif query.aggregation == AggregationType.LAST:
            aggregated = data.groupby(grouper).last()
        else:
            aggregated = data.groupby(grouper)[numeric_columns].agg(agg_func)

        return aggregated.reset_index()

    def _get_aggregation_function(self, aggregation: AggregationType) -> str:
        """Get pandas aggregation function name."""
        mapping = {
            AggregationType.SUM: 'sum',
            AggregationType.MEAN: 'mean',
            AggregationType.MIN: 'min',
            AggregationType.MAX: 'max',
            AggregationType.COUNT: 'count',
            AggregationType.STD: 'std',
            AggregationType.VAR: 'var'
        }
        return mapping.get(aggregation, 'mean')

    def _can_pushdown_aggregation(self, aggregation: AggregationType) -> bool:
        """Check if aggregation can be pushed down to storage level."""
        # Simple aggregations can be pushed down
        pushdown_supported = {
            AggregationType.SUM,
            AggregationType.MIN,
            AggregationType.MAX,
            AggregationType.COUNT
        }
        return aggregation in pushdown_supported

    def _generate_cache_key(self, query: TimeSeriesQuery) -> str:
        """Generate cache key for query."""
        key_parts = [
            query.dataset,
            str(query.symbols) if query.symbols else "all",
            str(query.columns) if query.columns else "all",
            str(query.start_date) if query.start_date else "none",
            str(query.end_date) if query.end_date else "none",
            str(query.aggregation.value) if query.aggregation else "none",
            str(query.aggregation_period) if query.aggregation_period else "none",
            str(query.filters) if query.filters else "none"
        ]
        return "_".join(key_parts)

    def _get_cached_result(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached result if available and not expired."""
        if not self.enable_cache or cache_key not in self.query_cache:
            return None

        cached_data, timestamp = self.query_cache[cache_key]
        age_seconds = (datetime.utcnow() - timestamp).total_seconds()

        if age_seconds > self.cache_ttl_seconds:
            # Cache expired
            del self.query_cache[cache_key]
            return None

        # Create result with cache hit flag
        cached_result = QueryResult(
            data=cached_data.copy(),
            execution_time_ms=0.1,  # Minimal time for cache access
            rows_scanned=0,
            rows_returned=len(cached_data),
            partitions_accessed=0,
            cache_hit=True,
            query_plan={'cached': True}
        )

        return cached_result

    def _cache_result(self, cache_key: str, result: QueryResult) -> None:
        """Cache query result."""
        if not self.enable_cache:
            return

        # Only cache successful results that aren't too large
        if result.rows_returned > 0 and result.rows_returned < 100000:
            self.query_cache[cache_key] = (result.data.copy(), datetime.utcnow())

        # Clean up old cache entries periodically
        if len(self.query_cache) > 100:
            self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = datetime.utcnow()
        expired_keys = []

        for key, (data, timestamp) in self.query_cache.items():
            age_seconds = (current_time - timestamp).total_seconds()
            if age_seconds > self.cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self.query_cache[key]

        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _get_query_pattern(self, query: TimeSeriesQuery) -> str:
        """Get query pattern for statistics."""
        pattern_parts = [query.dataset]

        if query.aggregation:
            pattern_parts.append(f"agg_{query.aggregation.value}")

        if query.start_date or query.end_date:
            pattern_parts.append("date_filter")

        if query.symbols:
            pattern_parts.append("symbol_filter")

        if query.columns:
            pattern_parts.append("column_select")

        return "_".join(pattern_parts)

    def _update_query_stats(self, result: QueryResult) -> None:
        """Update query statistics."""
        self.query_stats['total_queries'] += 1
        self.query_stats['total_execution_time_ms'] += result.execution_time_ms
        self.query_stats['total_rows_scanned'] += result.rows_scanned
        self.query_stats['total_rows_returned'] += result.rows_returned

    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        stats = self.query_stats.copy()

        if stats['total_queries'] > 0:
            stats['avg_execution_time_ms'] = stats['total_execution_time_ms'] / stats['total_queries']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries']
            stats['avg_rows_scanned'] = stats['total_rows_scanned'] / stats['total_queries']
            stats['avg_rows_returned'] = stats['total_rows_returned'] / stats['total_queries']
        else:
            stats['avg_execution_time_ms'] = 0
            stats['cache_hit_rate'] = 0
            stats['avg_rows_scanned'] = 0
            stats['avg_rows_returned'] = 0

        stats['cache_entries'] = len(self.query_cache)
        stats['last_updated'] = datetime.utcnow().isoformat()

        return stats

    def clear_cache(self) -> None:
        """Clear query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")

    def warm_cache(self, common_queries: List[TimeSeriesQuery]) -> None:
        """Warm up cache with common queries."""
        logger.info(f"Warming cache with {len(common_queries)} common queries")

        for query in common_queries:
            try:
                self.execute_query(query)
            except Exception as e:
                logger.warning(f"Failed to warm cache for query: {e}")

        logger.info("Cache warming completed")