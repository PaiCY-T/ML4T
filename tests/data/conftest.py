"""
Pytest configuration and fixtures for Point-in-Time Data Management tests.

This module provides shared fixtures and configuration for all test modules.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
import random
import logging

# Import core components
from src.data.core.temporal import (
    TemporalValue, InMemoryTemporalStore, TemporalDataManager,
    DataType, MarketSession
)

from src.data.pipeline.pit_engine import (
    PointInTimeEngine, PITQuery, PITCache
)

from src.data.models.taiwan_market import (
    TaiwanMarketData, create_taiwan_trading_calendar,
    TaiwanMarketDataValidator
)

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "stress: marks tests as stress tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


@pytest.fixture(scope="session")
def test_symbols():
    """Standard test symbols for Taiwan market."""
    return ["2330", "2317", "1301", "2412", "2454", "0050", "0056"]


@pytest.fixture(scope="session")
def test_date_range():
    """Standard test date range."""
    return date(2023, 1, 1), date(2024, 1, 31)


@pytest.fixture(scope="session")
def taiwan_trading_calendar_2023_2024():
    """Taiwan trading calendar for test period."""
    calendar_2023 = create_taiwan_trading_calendar(2023)
    calendar_2024 = create_taiwan_trading_calendar(2024)
    calendar_2023.update(calendar_2024)
    return calendar_2023


@pytest.fixture
def empty_temporal_store():
    """Empty in-memory temporal store."""
    return InMemoryTemporalStore()


@pytest.fixture
def temporal_data_manager(empty_temporal_store):
    """Temporal data manager with empty store."""
    return TemporalDataManager(empty_temporal_store)


@pytest.fixture
def populated_temporal_store(test_symbols, test_date_range):
    """Temporal store populated with test data."""
    store = InMemoryTemporalStore()
    start_date, end_date = test_date_range
    
    # Generate test data
    current_date = start_date
    prices = {symbol: 100.0 + random.uniform(50, 500) for symbol in test_symbols}
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Weekdays only
            for symbol in test_symbols:
                # Simulate price movement
                daily_return = random.gauss(0.0, 0.02)  # 2% daily volatility
                prices[symbol] *= (1 + daily_return)
                
                # Store price data
                price_value = TemporalValue(
                    value=Decimal(f"{prices[symbol]:.2f}"),
                    as_of_date=current_date,
                    value_date=current_date,
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    metadata={"source": "test"}
                )
                store.store(price_value)
                
                # Store volume data
                volume = random.randint(1000000, 50000000)
                volume_value = TemporalValue(
                    value=volume,
                    as_of_date=current_date,
                    value_date=current_date,
                    data_type=DataType.VOLUME,
                    symbol=symbol,
                    metadata={"source": "test"}
                )
                store.store(volume_value)
                
                # Store fundamental data (quarterly)
                if current_date.month in [2, 5, 8, 11] and current_date.day == 15:
                    fundamental_value = TemporalValue(
                        value={
                            "revenue": random.uniform(1000000, 10000000),
                            "eps": random.uniform(1.0, 10.0),
                            "roe": random.uniform(10.0, 30.0)
                        },
                        as_of_date=current_date,
                        value_date=current_date - timedelta(days=60),  # 60-day lag
                        data_type=DataType.FUNDAMENTAL,
                        symbol=symbol,
                        metadata={"source": "test", "lag_days": 60}
                    )
                    store.store(fundamental_value)
        
        current_date += timedelta(days=1)
    
    return store


@pytest.fixture
def pit_engine(populated_temporal_store, taiwan_trading_calendar_2023_2024):
    """Point-in-time engine with populated data."""
    return PointInTimeEngine(
        store=populated_temporal_store,
        trading_calendar=taiwan_trading_calendar_2023_2024,
        enable_cache=True,
        max_workers=4
    )


@pytest.fixture
def taiwan_market_validator():
    """Taiwan market data validator."""
    return TaiwanMarketDataValidator()


@pytest.fixture
def sample_taiwan_market_data():
    """Sample Taiwan market data for testing."""
    return TaiwanMarketData(
        symbol="2330",
        data_date=date(2024, 1, 15),
        as_of_date=date(2024, 1, 15),
        open_price=Decimal("580.00"),
        high_price=Decimal("585.00"),
        low_price=Decimal("578.00"),
        close_price=Decimal("582.00"),
        volume=25000000,
        turnover=Decimal("14550000000"),
        market_cap=Decimal("15000000000000")
    )


@pytest.fixture
def sample_pit_query(test_symbols):
    """Sample PIT query for testing."""
    return PITQuery(
        symbols=test_symbols[:3],  # Use first 3 symbols
        as_of_date=date(2023, 6, 15),
        data_types=[DataType.PRICE, DataType.VOLUME],
        mode=QueryMode.STRICT,
        bias_check=BiasCheckLevel.STRICT
    )


@pytest.fixture
def performance_test_data():
    """Large dataset for performance testing."""
    store = InMemoryTemporalStore()
    
    # Create larger dataset for performance tests
    num_symbols = 100
    num_days = 252  # Trading year
    symbols = [f"SYM{i:03d}" for i in range(num_symbols)]
    start_date = date(2023, 1, 1)
    
    current_date = start_date
    for day in range(num_days):
        test_date = start_date + timedelta(days=day)
        
        if test_date.weekday() < 5:  # Weekdays only
            for symbol in symbols:
                # Price data
                price = 100 + random.uniform(-20, 20)
                price_value = TemporalValue(
                    value=Decimal(f"{price:.2f}"),
                    as_of_date=test_date,
                    value_date=test_date,
                    data_type=DataType.PRICE,
                    symbol=symbol
                )
                store.store(price_value)
                
                # Volume data (every 5th symbol to reduce size)
                if hash(symbol) % 5 == 0:
                    volume = random.randint(100000, 10000000)
                    volume_value = TemporalValue(
                        value=volume,
                        as_of_date=test_date,
                        value_date=test_date,
                        data_type=DataType.VOLUME,
                        symbol=symbol
                    )
                    store.store(volume_value)
    
    return store, symbols, start_date


class PerformanceProfiler:
    """Simple performance profiler for tests."""
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, operation_name: str):
        """Start timing an operation."""
        self.start_times[operation_name] = datetime.now()
    
    def end(self, operation_name: str):
        """End timing an operation."""
        if operation_name in self.start_times:
            duration = datetime.now() - self.start_times[operation_name]
            self.timings[operation_name] = duration.total_seconds() * 1000  # ms
            del self.start_times[operation_name]
    
    def get_timing(self, operation_name: str) -> float:
        """Get timing for an operation in milliseconds."""
        return self.timings.get(operation_name, 0.0)
    
    def report(self):
        """Print performance report."""
        print("\nPerformance Report:")
        for operation, timing in self.timings.items():
            print(f"  {operation}: {timing:.2f}ms")


@pytest.fixture
def performance_profiler():
    """Performance profiler for tests."""
    return PerformanceProfiler()


# Utility functions for tests

def create_sample_temporal_value(
    symbol: str = "2330",
    value: Any = 100.0,
    data_date: date = None,
    data_type: DataType = DataType.PRICE,
    **kwargs
) -> TemporalValue:
    """Create a sample temporal value for testing."""
    if data_date is None:
        data_date = date(2024, 1, 15)
    
    return TemporalValue(
        value=Decimal(str(value)) if isinstance(value, (int, float)) else value,
        as_of_date=data_date,
        value_date=data_date,
        data_type=data_type,
        symbol=symbol,
        **kwargs
    )


def assert_no_look_ahead_bias(result, query_date: date):
    """Assert that a PIT query result has no look-ahead bias."""
    assert len(result.bias_violations) == 0, f"Bias violations found: {result.bias_violations}"
    
    for symbol, data_dict in result.data.items():
        for data_type, temporal_value in data_dict.items():
            assert temporal_value.as_of_date <= query_date, \
                f"Look-ahead bias: {symbol} {data_type} as_of_date {temporal_value.as_of_date} > {query_date}"
            assert temporal_value.value_date <= query_date, \
                f"Look-ahead bias: {symbol} {data_type} value_date {temporal_value.value_date} > {query_date}"


def assert_performance_requirements(execution_time_ms: float, max_time_ms: float = 100):
    """Assert that execution time meets performance requirements."""
    assert execution_time_ms < max_time_ms, \
        f"Performance requirement failed: {execution_time_ms:.2f}ms > {max_time_ms}ms"


# Custom pytest markers and hooks

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark performance tests
        if "performance" in item.name.lower() or "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if "stress" in item.name.lower() or "large" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)


# Test data generators

class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_price_series(
        symbol: str,
        start_date: date,
        num_days: int,
        initial_price: float = 100.0,
        volatility: float = 0.02
    ) -> List[TemporalValue]:
        """Generate a price time series."""
        values = []
        current_price = initial_price
        
        for i in range(num_days):
            test_date = start_date + timedelta(days=i)
            
            # Skip weekends
            if test_date.weekday() >= 5:
                continue
            
            # Apply random walk
            daily_return = random.gauss(0.0, volatility)
            current_price *= (1 + daily_return)
            
            value = TemporalValue(
                value=Decimal(f"{current_price:.2f}"),
                as_of_date=test_date,
                value_date=test_date,
                data_type=DataType.PRICE,
                symbol=symbol
            )
            values.append(value)
        
        return values
    
    @staticmethod
    def generate_fundamental_data(
        symbol: str,
        report_dates: List[date],
        lag_days: int = 60
    ) -> List[TemporalValue]:
        """Generate fundamental data with proper lag."""
        values = []
        
        for report_date in report_dates:
            announcement_date = report_date + timedelta(days=lag_days)
            
            value = TemporalValue(
                value={
                    "revenue": random.uniform(1000000, 10000000),
                    "eps": random.uniform(1.0, 10.0),
                    "roe": random.uniform(10.0, 30.0),
                    "quarter": ((report_date.month - 1) // 3) + 1,
                    "year": report_date.year
                },
                as_of_date=announcement_date,
                value_date=report_date,
                data_type=DataType.FUNDAMENTAL,
                symbol=symbol,
                metadata={"lag_days": lag_days}
            )
            values.append(value)
        
        return values


# Export fixtures and utilities
__all__ = [
    'test_symbols',
    'test_date_range',
    'taiwan_trading_calendar_2023_2024',
    'empty_temporal_store',
    'temporal_data_manager',
    'populated_temporal_store',
    'pit_engine',
    'taiwan_market_validator',
    'sample_taiwan_market_data',
    'sample_pit_query',
    'performance_test_data',
    'performance_profiler',
    'create_sample_temporal_value',
    'assert_no_look_ahead_bias',
    'assert_performance_requirements',
    'TestDataGenerator'
]