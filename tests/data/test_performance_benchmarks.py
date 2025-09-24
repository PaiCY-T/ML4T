"""
Performance Benchmarks for Point-in-Time Data Management System.

This test suite validates:
- Query performance requirements (>10K queries/sec)
- Single query latency (<100ms)
- Memory efficiency with large datasets
- Cache performance optimization
- Concurrent access patterns
- Large-scale data ingestion performance
"""

import pytest
import time
import threading
import asyncio
import statistics
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import psutil
import gc
from dataclasses import dataclass
from unittest.mock import Mock, patch

# Import components to benchmark
from src.data.core.temporal import (
    TemporalValue, InMemoryTemporalStore, TemporalDataManager,
    DataType, MarketSession
)

from src.data.pipeline.pit_engine import (
    PointInTimeEngine, PITQuery, PITResult, PITCache,
    QueryMode, BiasCheckLevel
)

from src.data.models.taiwan_market import (
    TaiwanMarketData, create_taiwan_trading_calendar
)


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    operation_name: str
    total_operations: int
    total_time_seconds: float
    operations_per_second: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_usage_mb: float
    success_rate: float = 1.0
    
    def __str__(self) -> str:
        return (
            f"{self.operation_name}:\n"
            f"  Operations: {self.total_operations:,}\n"
            f"  Total time: {self.total_time_seconds:.2f}s\n"
            f"  Throughput: {self.operations_per_second:,.0f} ops/sec\n"
            f"  Avg latency: {self.avg_latency_ms:.2f}ms\n"
            f"  P95 latency: {self.p95_latency_ms:.2f}ms\n"
            f"  P99 latency: {self.p99_latency_ms:.2f}ms\n"
            f"  Memory: {self.memory_usage_mb:.1f}MB\n"
            f"  Success rate: {self.success_rate:.1%}"
        )


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def measure_performance(self, 
                          operation_name: str,
                          operation_func,
                          iterations: int = 1000,
                          warmup_iterations: int = 100) -> PerformanceMetrics:
        """Measure performance of an operation."""
        # Warmup
        for _ in range(warmup_iterations):
            try:
                operation_func()
            except:
                pass
        
        # Force garbage collection before measurement
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure performance
        latencies = []
        successes = 0
        
        start_time = time.time()
        
        for _ in range(iterations):
            operation_start = time.time()
            try:
                operation_func()
                successes += 1
            except Exception as e:
                # Log but continue
                pass
            finally:
                operation_end = time.time()
                latencies.append((operation_end - operation_start) * 1000)  # ms
        
        end_time = time.time()
        
        # Memory measurement
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        # Calculate metrics
        total_time = end_time - start_time
        ops_per_second = iterations / total_time
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        success_rate = successes / iterations
        
        return PerformanceMetrics(
            operation_name=operation_name,
            total_operations=iterations,
            total_time_seconds=total_time,
            operations_per_second=ops_per_second,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            memory_usage_mb=memory_usage,
            success_rate=success_rate
        )


class TestDataGenerator:
    """Generate test data for performance benchmarks."""
    
    def __init__(self, num_symbols: int = 1000, num_days: int = 365):
        self.num_symbols = num_symbols
        self.num_days = num_days
        self.symbols = [f"SYM{i:04d}" for i in range(num_symbols)]
        self.start_date = date(2023, 1, 1)
        
    def generate_temporal_values(self) -> List[TemporalValue]:
        """Generate temporal values for testing."""
        values = []
        
        for i, symbol in enumerate(self.symbols):
            for day in range(self.num_days):
                test_date = self.start_date + timedelta(days=day)
                
                # Price data
                base_price = 100 + (i % 100)
                price = base_price + random.uniform(-5, 5)
                
                values.append(TemporalValue(
                    value=Decimal(f"{price:.2f}"),
                    as_of_date=test_date,
                    value_date=test_date,
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    metadata={"source": "test"}
                ))
                
                # Volume data (every 5th day to reduce dataset size)
                if day % 5 == 0:
                    volume = random.randint(100000, 10000000)
                    values.append(TemporalValue(
                        value=volume,
                        as_of_date=test_date,
                        value_date=test_date,
                        data_type=DataType.VOLUME,
                        symbol=symbol,
                        metadata={"source": "test"}
                    ))
        
        return values
    
    def generate_market_data(self) -> List[TaiwanMarketData]:
        """Generate Taiwan market data for testing."""
        market_data = []
        
        for symbol in self.symbols[:100]:  # Smaller subset for market data
            for day in range(min(self.num_days, 100)):
                test_date = self.start_date + timedelta(days=day)
                
                base_price = 100 + random.uniform(50, 500)
                price_range = base_price * 0.05
                
                open_price = base_price + random.uniform(-price_range, price_range)
                close_price = base_price + random.uniform(-price_range, price_range)
                high_price = max(open_price, close_price) + random.uniform(0, price_range)
                low_price = min(open_price, close_price) - random.uniform(0, price_range)
                
                market_data.append(TaiwanMarketData(
                    symbol=symbol,
                    data_date=test_date,
                    as_of_date=test_date,
                    open_price=Decimal(f"{open_price:.2f}"),
                    high_price=Decimal(f"{high_price:.2f}"),
                    low_price=Decimal(f"{low_price:.2f}"),
                    close_price=Decimal(f"{close_price:.2f}"),
                    volume=random.randint(100000, 10000000),
                    turnover=Decimal(f"{random.randint(1000000, 100000000)}")
                ))
        
        return market_data


class TestTemporalStorePerformance:
    """Performance tests for temporal store operations."""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    @pytest.fixture
    def data_generator(self):
        return TestDataGenerator(num_symbols=500, num_days=100)
    
    @pytest.fixture
    def populated_store(self, data_generator):
        """Create a store populated with test data."""
        store = InMemoryTemporalStore()
        values = data_generator.generate_temporal_values()
        
        for value in values:
            store.store(value)
            
        return store, data_generator
    
    def test_store_operation_performance(self, benchmark, data_generator):
        """Test temporal store operation performance."""
        store = InMemoryTemporalStore()
        values = data_generator.generate_temporal_values()
        value_iter = iter(values)
        
        def store_operation():
            try:
                value = next(value_iter)
                store.store(value)
            except StopIteration:
                # Create a new value when iterator exhausted
                store.store(TemporalValue(
                    value=random.uniform(100, 200),
                    as_of_date=date(2024, 1, 1),
                    value_date=date(2024, 1, 1),
                    data_type=DataType.PRICE,
                    symbol="TEST"
                ))
        
        metrics = benchmark.measure_performance(
            "Temporal Store Operations",
            store_operation,
            iterations=10000
        )
        
        print(f"\n{metrics}")
        
        # Performance requirements
        assert metrics.operations_per_second > 50000, f"Store ops too slow: {metrics.operations_per_second}"
        assert metrics.avg_latency_ms < 1.0, f"Store latency too high: {metrics.avg_latency_ms}"
    
    def test_point_in_time_query_performance(self, benchmark, populated_store):
        """Test point-in-time query performance."""
        store, data_generator = populated_store
        symbols = data_generator.symbols
        
        def query_operation():
            symbol = random.choice(symbols)
            query_date = date(2023, 6, 15)  # Mid-range date
            store.get_point_in_time(symbol, query_date, DataType.PRICE)
        
        metrics = benchmark.measure_performance(
            "Point-in-Time Queries",
            query_operation,
            iterations=10000
        )
        
        print(f"\n{metrics}")
        
        # Performance requirements
        assert metrics.operations_per_second > 100000, f"Query ops too slow: {metrics.operations_per_second}"
        assert metrics.avg_latency_ms < 0.1, f"Query latency too high: {metrics.avg_latency_ms}"
    
    def test_range_query_performance(self, benchmark, populated_store):
        """Test range query performance."""
        store, data_generator = populated_store
        symbols = data_generator.symbols
        
        def range_query_operation():
            symbol = random.choice(symbols)
            start_date = date(2023, 3, 1)
            end_date = date(2023, 3, 31)  # 1 month range
            store.get_range(symbol, start_date, end_date, DataType.PRICE)
        
        metrics = benchmark.measure_performance(
            "Range Queries",
            range_query_operation,
            iterations=1000
        )
        
        print(f"\n{metrics}")
        
        # Range queries should be reasonable
        assert metrics.operations_per_second > 1000, f"Range query ops too slow: {metrics.operations_per_second}"
        assert metrics.avg_latency_ms < 10.0, f"Range query latency too high: {metrics.avg_latency_ms}"


class TestPITEnginePerformance:
    """Performance tests for PIT engine."""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    @pytest.fixture
    def engine_with_data(self):
        """Create PIT engine with test data."""
        store = InMemoryTemporalStore()
        data_gen = TestDataGenerator(num_symbols=1000, num_days=250)
        values = data_gen.generate_temporal_values()
        
        for value in values:
            store.store(value)
        
        engine = PointInTimeEngine(
            store=store,
            enable_cache=True,
            max_workers=4
        )
        
        return engine, data_gen
    
    def test_single_symbol_query_performance(self, benchmark, engine_with_data):
        """Test single symbol query performance requirement (<100ms)."""
        engine, data_gen = engine_with_data
        symbols = data_gen.symbols
        
        def single_query_operation():
            symbol = random.choice(symbols)
            query = PITQuery(
                symbols=[symbol],
                as_of_date=date(2023, 6, 15),
                data_types=[DataType.PRICE],
                mode=QueryMode.FAST
            )
            engine.execute_query(query)
        
        metrics = benchmark.measure_performance(
            "Single Symbol Queries",
            single_query_operation,
            iterations=1000
        )
        
        print(f"\n{metrics}")
        
        # Requirements: <100ms average latency
        assert metrics.avg_latency_ms < 100, f"Single query latency too high: {metrics.avg_latency_ms}"
        assert metrics.p95_latency_ms < 200, f"P95 latency too high: {metrics.p95_latency_ms}"
    
    def test_bulk_query_performance_requirement(self, benchmark, engine_with_data):
        """Test bulk query performance requirement (>10K queries/sec)."""
        engine, data_gen = engine_with_data
        symbols = data_gen.symbols
        
        def bulk_query_operation():
            # Small bulk query (5 symbols)
            query_symbols = random.sample(symbols, 5)
            query = PITQuery(
                symbols=query_symbols,
                as_of_date=date(2023, 6, 15),
                data_types=[DataType.PRICE],
                mode=QueryMode.FAST,
                bias_check=BiasCheckLevel.BASIC
            )
            engine.execute_query(query)
        
        metrics = benchmark.measure_performance(
            "Bulk Queries (5 symbols)",
            bulk_query_operation,
            iterations=2000
        )
        
        print(f"\n{metrics}")
        
        # Requirements: >10K queries/sec
        assert metrics.operations_per_second > 10000, f"Bulk query throughput too low: {metrics.operations_per_second}"
    
    def test_cache_performance_impact(self, benchmark, engine_with_data):
        """Test cache performance impact."""
        engine, data_gen = engine_with_data
        symbols = data_gen.symbols[:100]  # Use subset for consistent testing
        
        # Test with cache enabled
        engine.cache.clear()
        
        def cached_query_operation():
            symbol = random.choice(symbols)
            query = PITQuery(
                symbols=[symbol],
                as_of_date=date(2023, 6, 15),
                data_types=[DataType.PRICE],
                mode=QueryMode.FAST
            )
            engine.execute_query(query)
        
        metrics_with_cache = benchmark.measure_performance(
            "Queries with Cache",
            cached_query_operation,
            iterations=2000
        )
        
        # Test without cache
        engine_no_cache = PointInTimeEngine(
            store=engine.store,
            enable_cache=False,
            max_workers=4
        )
        
        def uncached_query_operation():
            symbol = random.choice(symbols)
            query = PITQuery(
                symbols=[symbol],
                as_of_date=date(2023, 6, 15),
                data_types=[DataType.PRICE],
                mode=QueryMode.FAST
            )
            engine_no_cache.execute_query(query)
        
        metrics_no_cache = benchmark.measure_performance(
            "Queries without Cache",
            uncached_query_operation,
            iterations=2000
        )
        
        print(f"\n{metrics_with_cache}")
        print(f"\n{metrics_no_cache}")
        
        # Cache should improve performance
        cache_speedup = metrics_with_cache.operations_per_second / metrics_no_cache.operations_per_second
        print(f"\nCache speedup: {cache_speedup:.2f}x")
        
        assert cache_speedup > 1.5, f"Cache speedup too low: {cache_speedup:.2f}x"
    
    def test_parallel_query_performance(self, benchmark, engine_with_data):
        """Test parallel query performance."""
        engine, data_gen = engine_with_data
        symbols = data_gen.symbols
        
        def parallel_query_operation():
            # Large multi-symbol query
            query_symbols = random.sample(symbols, 20)
            query = PITQuery(
                symbols=query_symbols,
                as_of_date=date(2023, 6, 15),
                data_types=[DataType.PRICE, DataType.VOLUME],
                mode=QueryMode.FAST
            )
            result = engine.execute_query(query)
            # Ensure query processed multiple symbols
            assert len(result.data) > 0
        
        metrics = benchmark.measure_performance(
            "Parallel Multi-Symbol Queries",
            parallel_query_operation,
            iterations=500
        )
        
        print(f"\n{metrics}")
        
        # Should handle parallel queries efficiently
        assert metrics.operations_per_second > 100, f"Parallel query throughput too low: {metrics.operations_per_second}"
        assert metrics.avg_latency_ms < 50, f"Parallel query latency too high: {metrics.avg_latency_ms}"


class TestCachePerformance:
    """Performance tests for PIT cache."""
    
    @pytest.fixture
    def benchmark(self):
        return PerformanceBenchmark()
    
    def test_cache_hit_performance(self, benchmark):
        """Test cache hit performance."""
        cache = PITCache(max_size=10000, ttl_seconds=3600)
        
        # Populate cache
        test_symbols = [f"SYM{i:03d}" for i in range(100)]
        test_date = date(2023, 6, 15)
        
        for symbol in test_symbols:
            value = TemporalValue(
                value=random.uniform(100, 200),
                as_of_date=test_date,
                value_date=test_date,
                data_type=DataType.PRICE,
                symbol=symbol
            )
            cache.put(symbol, DataType.PRICE, test_date, value)
        
        def cache_hit_operation():
            symbol = random.choice(test_symbols)
            cache.get(symbol, DataType.PRICE, test_date)
        
        metrics = benchmark.measure_performance(
            "Cache Hit Operations",
            cache_hit_operation,
            iterations=100000
        )
        
        print(f"\n{metrics}")
        
        # Cache hits should be extremely fast
        assert metrics.operations_per_second > 1000000, f"Cache hit rate too slow: {metrics.operations_per_second}"
        assert metrics.avg_latency_ms < 0.01, f"Cache hit latency too high: {metrics.avg_latency_ms}"
    
    def test_cache_miss_performance(self, benchmark):
        """Test cache miss performance."""
        cache = PITCache(max_size=10000, ttl_seconds=3600)
        
        def cache_miss_operation():
            symbol = f"MISS{random.randint(0, 10000)}"
            cache.get(symbol, DataType.PRICE, date(2023, 6, 15))
        
        metrics = benchmark.measure_performance(
            "Cache Miss Operations",
            cache_miss_operation,
            iterations=100000
        )
        
        print(f"\n{metrics}")
        
        # Cache misses should still be fast
        assert metrics.operations_per_second > 500000, f"Cache miss rate too slow: {metrics.operations_per_second}"
        assert metrics.avg_latency_ms < 0.01, f"Cache miss latency too high: {metrics.avg_latency_ms}"
    
    def test_cache_eviction_performance(self, benchmark):
        """Test cache eviction performance with LRU."""
        cache = PITCache(max_size=1000, ttl_seconds=3600)  # Small cache for eviction testing
        
        def cache_eviction_operation():
            # Generate unique keys to force evictions
            symbol = f"EVT{random.randint(0, 10000)}"
            value = TemporalValue(
                value=random.uniform(100, 200),
                as_of_date=date(2023, 6, 15),
                value_date=date(2023, 6, 15),
                data_type=DataType.PRICE,
                symbol=symbol
            )
            cache.put(symbol, DataType.PRICE, date(2023, 6, 15), value)
        
        metrics = benchmark.measure_performance(
            "Cache Eviction Operations",
            cache_eviction_operation,
            iterations=10000
        )
        
        print(f"\n{metrics}")
        
        # Eviction should not significantly impact performance
        assert metrics.operations_per_second > 100000, f"Cache eviction too slow: {metrics.operations_per_second}"
        assert metrics.avg_latency_ms < 0.1, f"Cache eviction latency too high: {metrics.avg_latency_ms}"


class TestConcurrentAccess:
    """Test concurrent access patterns and thread safety."""
    
    def test_concurrent_store_operations(self):
        """Test concurrent store operations."""
        store = InMemoryTemporalStore()
        num_threads = 10
        operations_per_thread = 1000
        
        def worker_thread(thread_id: int):
            for i in range(operations_per_thread):
                value = TemporalValue(
                    value=random.uniform(100, 200),
                    as_of_date=date(2023, 6, 15),
                    value_date=date(2023, 6, 15),
                    data_type=DataType.PRICE,
                    symbol=f"T{thread_id}S{i}"
                )
                store.store(value)
        
        start_time = time.time()
        
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        total_operations = num_threads * operations_per_thread
        total_time = end_time - start_time
        ops_per_second = total_operations / total_time
        
        print(f"\nConcurrent Store Operations:")
        print(f"  Threads: {num_threads}")
        print(f"  Operations per thread: {operations_per_thread}")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {ops_per_second:,.0f} ops/sec")
        
        # Verify all data was stored
        assert len(store._values) == total_operations
        
        # Should maintain reasonable performance under concurrency
        assert ops_per_second > 10000, f"Concurrent store performance too low: {ops_per_second}"
    
    def test_concurrent_query_operations(self):
        """Test concurrent query operations."""
        # Prepare data
        store = InMemoryTemporalStore()
        symbols = [f"SYM{i:03d}" for i in range(100)]
        
        for symbol in symbols:
            for day in range(30):
                test_date = date(2023, 6, 1) + timedelta(days=day)
                value = TemporalValue(
                    value=random.uniform(100, 200),
                    as_of_date=test_date,
                    value_date=test_date,
                    data_type=DataType.PRICE,
                    symbol=symbol
                )
                store.store(value)
        
        num_threads = 10
        queries_per_thread = 500
        
        def query_worker(thread_id: int):
            for i in range(queries_per_thread):
                symbol = random.choice(symbols)
                query_date = date(2023, 6, 15)
                result = store.get_point_in_time(symbol, query_date, DataType.PRICE)
                # Verify result is reasonable
                assert result is not None
        
        start_time = time.time()
        
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=query_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        total_queries = num_threads * queries_per_thread
        total_time = end_time - start_time
        queries_per_second = total_queries / total_time
        
        print(f"\nConcurrent Query Operations:")
        print(f"  Threads: {num_threads}")
        print(f"  Queries per thread: {queries_per_thread}")
        print(f"  Total queries: {total_queries}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {queries_per_second:,.0f} queries/sec")
        
        # Should maintain high query performance under concurrency
        assert queries_per_second > 50000, f"Concurrent query performance too low: {queries_per_second}"


class TestMemoryEfficiency:
    """Test memory efficiency with large datasets."""
    
    def test_large_dataset_memory_usage(self):
        """Test memory usage with large datasets."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        store = InMemoryTemporalStore()
        
        # Create large dataset
        num_symbols = 2000
        num_days = 365
        symbols = [f"SYM{i:04d}" for i in range(num_symbols)]
        
        print(f"\nLoading large dataset: {num_symbols} symbols x {num_days} days")
        
        start_time = time.time()
        
        for i, symbol in enumerate(symbols):
            for day in range(num_days):
                test_date = date(2023, 1, 1) + timedelta(days=day)
                
                value = TemporalValue(
                    value=Decimal(f"{100 + random.uniform(-10, 10):.2f}"),
                    as_of_date=test_date,
                    value_date=test_date,
                    data_type=DataType.PRICE,
                    symbol=symbol
                )
                store.store(value)
            
            # Progress reporting
            if (i + 1) % 500 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"  Loaded {i + 1} symbols, memory: {current_memory:.1f}MB")
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_values = num_symbols * num_days
        load_time = end_time - start_time
        memory_used = final_memory - initial_memory
        values_per_mb = total_values / memory_used if memory_used > 0 else 0
        
        print(f"\nLarge Dataset Results:")
        print(f"  Total values: {total_values:,}")
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Load rate: {total_values / load_time:,.0f} values/sec")
        print(f"  Memory used: {memory_used:.1f}MB")
        print(f"  Values per MB: {values_per_mb:,.0f}")
        
        # Test query performance with large dataset
        query_start = time.time()
        num_queries = 1000
        
        for _ in range(num_queries):
            symbol = random.choice(symbols)
            query_date = date(2023, 6, 15)
            result = store.get_point_in_time(symbol, query_date, DataType.PRICE)
            assert result is not None
        
        query_end = time.time()
        query_time = query_end - query_start
        queries_per_second = num_queries / query_time
        
        print(f"  Query performance: {queries_per_second:,.0f} queries/sec")
        
        # Memory efficiency requirements
        assert memory_used < 1000, f"Memory usage too high: {memory_used:.1f}MB"
        assert values_per_mb > 10000, f"Memory efficiency too low: {values_per_mb:.0f} values/MB"
        
        # Query performance should not degrade significantly
        assert queries_per_second > 10000, f"Query performance degraded: {queries_per_second:.0f} queries/sec"


@pytest.mark.stress
class TestStressScenarios:
    """Stress tests for extreme conditions."""
    
    def test_high_frequency_updates(self):
        """Test high-frequency update scenario."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store, enable_cache=True)
        
        # Simulate high-frequency trading updates
        symbol = "TSMC"
        base_date = date(2024, 1, 15)
        num_updates = 10000
        
        start_time = time.time()
        
        for i in range(num_updates):
            # Micro-timestamp simulation (multiple updates per day)
            update_time = datetime.combine(base_date, datetime.min.time()) + timedelta(seconds=i)
            
            value = TemporalValue(
                value=Decimal(f"{580 + random.uniform(-5, 5):.2f}"),
                as_of_date=update_time.date(),
                value_date=update_time.date(),
                data_type=DataType.PRICE,
                symbol=symbol,
                created_at=update_time
            )
            store.store(value)
        
        update_time = time.time() - start_time
        updates_per_second = num_updates / update_time
        
        print(f"\nHigh-Frequency Updates:")
        print(f"  Updates: {num_updates:,}")
        print(f"  Time: {update_time:.2f}s")
        print(f"  Rate: {updates_per_second:,.0f} updates/sec")
        
        # Test query performance after updates
        query_start = time.time()
        
        for _ in range(1000):
            result = store.get_point_in_time(symbol, base_date, DataType.PRICE)
            assert result is not None
        
        query_time = time.time() - query_start
        queries_per_second = 1000 / query_time
        
        print(f"  Query rate after updates: {queries_per_second:,.0f} queries/sec")
        
        # Should handle high-frequency updates efficiently
        assert updates_per_second > 50000, f"Update rate too low: {updates_per_second}"
        assert queries_per_second > 10000, f"Query performance degraded: {queries_per_second}"


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "not stress"])