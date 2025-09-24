"""
Advanced Point-in-Time Engine Tests.

This test suite validates:
- Complex PIT query scenarios
- Look-ahead bias prevention under stress
- Multi-threaded query execution
- Query optimization strategies
- Error handling and recovery
- Integration with FinLab connector
- Real-world backtesting scenarios
"""

import pytest
import asyncio
import threading
import time
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Import components to test
from src.data.core.temporal import (
    TemporalValue, InMemoryTemporalStore, TemporalDataManager,
    DataType, MarketSession
)

from src.data.pipeline.pit_engine import (
    PointInTimeEngine, PITQuery, PITResult, PITCache, BiasDetector,
    PITQueryOptimizer, QueryMode, BiasCheckLevel
)

from src.data.models.taiwan_market import (
    TaiwanMarketData, TaiwanSettlement, TaiwanTradingCalendar,
    TaiwanMarketDataValidator, create_taiwan_trading_calendar,
    TaiwanFundamental, TaiwanCorporateAction, CorporateActionType
)

from src.data.ingestion.finlab_connector import FinLabConnector, FinLabConfig


class TestAdvancedPITEngine:
    """Advanced PIT Engine testing scenarios."""
    
    @pytest.fixture
    def populated_engine(self):
        """Create PIT engine with comprehensive test data."""
        store = InMemoryTemporalStore()
        
        # Create test data for multiple symbols and time periods
        symbols = ["2330", "2317", "1301", "2412", "2454"]  # Major Taiwan stocks
        start_date = date(2023, 1, 1)
        end_date = date(2024, 1, 31)
        
        current_date = start_date
        while current_date <= end_date:
            for symbol in symbols:
                # Price data (available daily)
                base_price = {"2330": 580, "2317": 120, "1301": 65, "2412": 85, "2454": 750}[symbol]
                price = base_price + random.uniform(-20, 20)
                
                price_value = TemporalValue(
                    value=Decimal(f"{price:.2f}"),
                    as_of_date=current_date,
                    value_date=current_date,
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    metadata={"source": "test", "market": "TWSE"}
                )
                store.store(price_value)
                
                # Volume data
                volume = random.randint(10000000, 50000000)
                volume_value = TemporalValue(
                    value=volume,
                    as_of_date=current_date,
                    value_date=current_date,
                    data_type=DataType.VOLUME,
                    symbol=symbol,
                    metadata={"source": "test"}
                )
                store.store(volume_value)
                
                # Fundamental data (quarterly)
                if current_date.day == 15 and current_date.month in [2, 5, 8, 11]:
                    # Simulate quarterly earnings with 30-day lag
                    report_date = date(current_date.year, ((current_date.month - 2) // 3) * 3 + 1, 1)
                    if current_date.month == 2:
                        report_date = date(current_date.year - 1, 12, 31)
                    
                    fundamental_value = TemporalValue(
                        value={
                            "revenue": random.uniform(100000, 500000),
                            "eps": random.uniform(1.0, 10.0),
                            "roe": random.uniform(10.0, 30.0)
                        },
                        as_of_date=current_date,  # Announcement date
                        value_date=report_date,   # Report period end
                        data_type=DataType.FUNDAMENTAL,
                        symbol=symbol,
                        metadata={"source": "test", "lag_days": 45}
                    )
                    store.store(fundamental_value)
            
            current_date += timedelta(days=1)
        
        # Create engine with full configuration
        trading_calendar = create_taiwan_trading_calendar(2023)
        trading_calendar.update(create_taiwan_trading_calendar(2024))
        
        engine = PointInTimeEngine(
            store=store,
            trading_calendar=trading_calendar,
            enable_cache=True,
            max_workers=4
        )
        
        return engine, symbols, start_date, end_date
    
    def test_complex_multi_symbol_queries(self, populated_engine):
        """Test complex queries with multiple symbols and data types."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Complex query with all symbols and multiple data types
        query = PITQuery(
            symbols=symbols,
            as_of_date=date(2023, 6, 15),
            data_types=[DataType.PRICE, DataType.VOLUME, DataType.FUNDAMENTAL],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        result = engine.execute_query(query)
        
        assert result.success if hasattr(result, 'success') else True
        assert len(result.data) == len(symbols)
        
        # Check that each symbol has price and volume data
        for symbol in symbols:
            assert symbol in result.data
            assert DataType.PRICE in result.data[symbol]
            assert DataType.VOLUME in result.data[symbol]
            # Fundamental data may not be available for all symbols on this date
        
        # Verify no look-ahead bias
        assert len(result.bias_violations) == 0
        
        # Performance check
        assert result.execution_time_ms < 1000  # Should complete within 1 second
    
    def test_time_series_consistency(self, populated_engine):
        """Test temporal consistency across a time series."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Query same symbol across multiple dates
        symbol = "2330"
        test_dates = [start_date + timedelta(days=i * 30) for i in range(12)]  # Monthly
        
        previous_value = None
        for test_date in test_dates:
            query = PITQuery(
                symbols=[symbol],
                as_of_date=test_date,
                data_types=[DataType.PRICE],
                mode=QueryMode.STRICT
            )
            
            result = engine.execute_query(query)
            
            if symbol in result.data and DataType.PRICE in result.data[symbol]:
                current_value = result.data[symbol][DataType.PRICE]
                
                # Verify temporal consistency
                assert current_value.as_of_date <= test_date, f"Look-ahead bias: {current_value.as_of_date} > {test_date}"
                
                # If we have a previous value, ensure it's not from the future
                if previous_value:
                    assert current_value.as_of_date >= previous_value.as_of_date, "Temporal order violation"
                
                previous_value = current_value
    
    def test_settlement_lag_integration(self, populated_engine):
        """Test integration with Taiwan T+2 settlement lag."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Test trade execution scenario
        trade_date = date(2023, 6, 15)  # Thursday
        symbol = "2330"
        
        # Query for settlement calculation
        settlement_date = engine.trading_calendar.get(trade_date)
        if settlement_date and settlement_date.is_trading_day:
            # Simulate T+2 settlement
            expected_settlement = trade_date + timedelta(days=2)
            
            # Skip weekends
            while expected_settlement.weekday() >= 5:
                expected_settlement += timedelta(days=1)
            
            # Query should respect settlement timing
            query = PITQuery(
                symbols=[symbol],
                as_of_date=trade_date,
                data_types=[DataType.PRICE],
                mode=QueryMode.STRICT
            )
            
            result = engine.execute_query(query)
            
            # Should get data up to trade date
            if symbol in result.data and DataType.PRICE in result.data[symbol]:
                price_data = result.data[symbol][DataType.PRICE]
                assert price_data.value_date <= trade_date
    
    def test_fundamental_data_lag_scenarios(self, populated_engine):
        """Test fundamental data lag scenarios."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Test quarterly earnings lag
        earnings_announcement = date(2023, 5, 15)  # May 15
        query_symbol = "2330"
        
        # Query before earnings announcement
        pre_earnings_query = PITQuery(
            symbols=[query_symbol],
            as_of_date=earnings_announcement - timedelta(days=1),
            data_types=[DataType.FUNDAMENTAL],
            mode=QueryMode.STRICT
        )
        
        pre_result = engine.execute_query(pre_earnings_query)
        
        # Query after earnings announcement
        post_earnings_query = PITQuery(
            symbols=[query_symbol],
            as_of_date=earnings_announcement + timedelta(days=1),
            data_types=[DataType.FUNDAMENTAL],
            mode=QueryMode.STRICT
        )
        
        post_result = engine.execute_query(post_earnings_query)
        
        # Should have more recent fundamental data after announcement
        pre_fundamental = pre_result.data.get(query_symbol, {}).get(DataType.FUNDAMENTAL)
        post_fundamental = post_result.data.get(query_symbol, {}).get(DataType.FUNDAMENTAL)
        
        if pre_fundamental and post_fundamental:
            assert post_fundamental.as_of_date >= pre_fundamental.as_of_date
    
    def test_corporate_action_handling(self, populated_engine):
        """Test corporate action data handling."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Add corporate action data
        symbol = "2330"
        announcement_date = date(2023, 7, 20)
        ex_date = date(2023, 8, 15)
        
        # Create dividend action
        dividend_data = {
            "action_type": "dividend_cash",
            "amount": 2.75,
            "ex_date": ex_date.isoformat(),
            "record_date": (ex_date + timedelta(days=2)).isoformat(),
            "payment_date": (ex_date + timedelta(days=30)).isoformat()
        }
        
        corporate_action = TemporalValue(
            value=dividend_data,
            as_of_date=announcement_date,
            value_date=ex_date,
            data_type=DataType.CORPORATE_ACTION,
            symbol=symbol,
            metadata={"action_type": "dividend_cash"}
        )
        
        engine.store.store(corporate_action)
        
        # Query before announcement
        pre_announcement_query = PITQuery(
            symbols=[symbol],
            as_of_date=announcement_date - timedelta(days=1),
            data_types=[DataType.CORPORATE_ACTION],
            mode=QueryMode.STRICT
        )
        
        pre_result = engine.execute_query(pre_announcement_query)
        
        # Query after announcement
        post_announcement_query = PITQuery(
            symbols=[symbol],
            as_of_date=announcement_date + timedelta(days=1),
            data_types=[DataType.CORPORATE_ACTION],
            mode=QueryMode.STRICT
        )
        
        post_result = engine.execute_query(post_announcement_query)
        
        # Should not have corporate action data before announcement
        assert DataType.CORPORATE_ACTION not in pre_result.data.get(symbol, {})
        
        # Should have corporate action data after announcement
        if symbol in post_result.data:
            corporate_action_data = post_result.data[symbol].get(DataType.CORPORATE_ACTION)
            if corporate_action_data:
                assert corporate_action_data.value["amount"] == 2.75
    
    def test_bias_detection_under_stress(self, populated_engine):
        """Test look-ahead bias detection under stress conditions."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Add future data that should trigger bias detection
        future_date = date(2024, 6, 1)  # Future date
        current_query_date = date(2023, 12, 15)
        
        # Store future data
        future_value = TemporalValue(
            value=Decimal("999.99"),  # Distinctive value
            as_of_date=future_date,   # Future as_of_date
            value_date=future_date,
            data_type=DataType.PRICE,
            symbol="2330",
            metadata={"source": "future_test"}
        )
        
        engine.store.store(future_value)
        
        # Query with strict bias checking
        strict_query = PITQuery(
            symbols=["2330"],
            as_of_date=current_query_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        # Should not get future data or should raise error
        result = engine.execute_query(strict_query)
        
        # Check that we didn't get the future value
        if "2330" in result.data and DataType.PRICE in result.data["2330"]:
            price_value = result.data["2330"][DataType.PRICE]
            assert price_value.value != Decimal("999.99"), "Got future data!"
            assert price_value.as_of_date <= current_query_date, "Look-ahead bias detected!"
    
    def test_query_optimization_effectiveness(self, populated_engine):
        """Test query optimization and caching effectiveness."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Clear cache and reset optimizer
        engine.clear_cache()
        engine.optimizer = PITQueryOptimizer()
        
        # Run initial queries to build optimization data
        test_symbols = symbols[:3]
        test_date = date(2023, 6, 15)
        
        # First pass - should be slower
        first_pass_times = []
        for _ in range(10):
            query = PITQuery(
                symbols=test_symbols,
                as_of_date=test_date,
                data_types=[DataType.PRICE],
                mode=QueryMode.FAST
            )
            
            start_time = time.time()
            result = engine.execute_query(query)
            execution_time = (time.time() - start_time) * 1000
            first_pass_times.append(execution_time)
        
        avg_first_pass = sum(first_pass_times) / len(first_pass_times)
        
        # Second pass - should be faster due to cache and optimization
        second_pass_times = []
        for _ in range(10):
            query = PITQuery(
                symbols=test_symbols,
                as_of_date=test_date,
                data_types=[DataType.PRICE],
                mode=QueryMode.FAST
            )
            
            start_time = time.time()
            result = engine.execute_query(query)
            execution_time = (time.time() - start_time) * 1000
            second_pass_times.append(execution_time)
        
        avg_second_pass = sum(second_pass_times) / len(second_pass_times)
        
        print(f"\nQuery Optimization Test:")
        print(f"  First pass avg: {avg_first_pass:.2f}ms")
        print(f"  Second pass avg: {avg_second_pass:.2f}ms")
        print(f"  Speedup: {avg_first_pass / avg_second_pass:.2f}x")
        
        # Should see improvement from caching
        assert avg_second_pass < avg_first_pass, "No optimization improvement detected"
    
    def test_parallel_query_execution(self, populated_engine):
        """Test parallel query execution with threading."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Test concurrent queries
        num_threads = 8
        queries_per_thread = 50
        
        def worker_queries(thread_id: int) -> List[float]:
            execution_times = []
            
            for i in range(queries_per_thread):
                symbol = random.choice(symbols)
                test_date = start_date + timedelta(days=random.randint(30, 300))
                
                query = PITQuery(
                    symbols=[symbol],
                    as_of_date=test_date,
                    data_types=[DataType.PRICE],
                    mode=QueryMode.FAST,
                    bias_check=BiasCheckLevel.BASIC
                )
                
                start_time = time.time()
                result = engine.execute_query(query)
                execution_time = (time.time() - start_time) * 1000
                execution_times.append(execution_time)
                
                # Verify result
                assert isinstance(result, PITResult)
            
            return execution_times
        
        # Execute parallel queries
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_queries, i) for i in range(num_threads)]
            
            all_times = []
            for future in as_completed(futures):
                thread_times = future.result()
                all_times.extend(thread_times)
        
        total_time = time.time() - start_time
        total_queries = num_threads * queries_per_thread
        throughput = total_queries / total_time
        avg_latency = sum(all_times) / len(all_times)
        
        print(f"\nParallel Query Execution:")
        print(f"  Threads: {num_threads}")
        print(f"  Total queries: {total_queries}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:,.0f} queries/sec")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        
        # Performance requirements
        assert throughput > 1000, f"Parallel throughput too low: {throughput}"
        assert avg_latency < 100, f"Parallel latency too high: {avg_latency}"
    
    def test_memory_pressure_handling(self, populated_engine):
        """Test engine behavior under memory pressure."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Add large amount of additional data to create memory pressure
        large_symbol_set = [f"MEM{i:04d}" for i in range(500)]
        
        for symbol in large_symbol_set:
            for day in range(100):
                test_date = start_date + timedelta(days=day)
                
                # Add multiple data types per symbol
                for data_type in [DataType.PRICE, DataType.VOLUME]:
                    value = TemporalValue(
                        value=random.uniform(50, 500),
                        as_of_date=test_date,
                        value_date=test_date,
                        data_type=data_type,
                        symbol=symbol
                    )
                    engine.store.store(value)
        
        # Test that queries still work under memory pressure
        query = PITQuery(
            symbols=large_symbol_set[:10],
            as_of_date=start_date + timedelta(days=50),
            data_types=[DataType.PRICE, DataType.VOLUME],
            mode=QueryMode.FAST
        )
        
        result = engine.execute_query(query)
        
        # Should still function correctly
        assert len(result.data) > 0
        assert result.execution_time_ms < 5000  # Should complete within 5 seconds
    
    def test_error_recovery_scenarios(self, populated_engine):
        """Test error handling and recovery scenarios."""
        engine, symbols, start_date, end_date = populated_engine
        
        # Test query with invalid symbol
        invalid_query = PITQuery(
            symbols=["INVALID_SYMBOL"],
            as_of_date=date(2023, 6, 15),
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT
        )
        
        result = engine.execute_query(invalid_query)
        
        # Should handle gracefully
        assert isinstance(result, PITResult)
        assert "INVALID_SYMBOL" not in result.data or len(result.data["INVALID_SYMBOL"]) == 0
        
        # Test query with future date and strict bias checking
        future_query = PITQuery(
            symbols=["2330"],
            as_of_date=date(2025, 1, 1),  # Future date
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        # Should either raise exception or return result with bias violations
        try:
            result = engine.execute_query(future_query)
            assert len(result.bias_violations) > 0, "Should detect future date bias"
        except ValueError as e:
            assert "bias" in str(e).lower(), "Should raise bias-related error"
    
    def test_cache_coherency_under_updates(self, populated_engine):
        """Test cache coherency when underlying data is updated."""
        engine, symbols, start_date, end_date = populated_engine
        
        symbol = "2330"
        test_date = date(2023, 6, 15)
        
        # Initial query to populate cache
        initial_query = PITQuery(
            symbols=[symbol],
            as_of_date=test_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.FAST
        )
        
        initial_result = engine.execute_query(initial_query)
        initial_price = initial_result.data[symbol][DataType.PRICE].value
        
        # Update underlying data
        new_value = TemporalValue(
            value=Decimal("999.99"),  # Distinctive new value
            as_of_date=test_date,
            value_date=test_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            version=2  # Higher version
        )
        
        engine.store.store(new_value)
        
        # Clear cache to ensure fresh data
        engine.clear_cache()
        
        # Query again
        updated_query = PITQuery(
            symbols=[symbol],
            as_of_date=test_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.FAST
        )
        
        updated_result = engine.execute_query(updated_query)
        updated_price = updated_result.data[symbol][DataType.PRICE].value
        
        # Should get updated value
        assert updated_price == Decimal("999.99"), "Cache coherency issue: didn't get updated value"
        assert updated_price != initial_price, "Should have different value after update"


class TestBacktestingScenarios:
    """Test realistic backtesting scenarios."""
    
    @pytest.fixture
    def backtesting_engine(self):
        """Create engine optimized for backtesting scenarios."""
        store = InMemoryTemporalStore()
        
        # Create realistic backtesting dataset
        symbols = ["2330", "2317", "1301", "2412", "2454", "0050", "0056"]  # Mix of stocks and ETFs
        start_date = date(2020, 1, 1)
        end_date = date(2023, 12, 31)  # 4 years of data
        
        # Simulate daily data
        current_date = start_date
        prices = {symbol: 100.0 for symbol in symbols}  # Starting prices
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Weekdays only
                for symbol in symbols:
                    # Simulate price movement
                    daily_return = random.gauss(0.0005, 0.02)  # 0.05% daily return, 2% volatility
                    prices[symbol] *= (1 + daily_return)
                    
                    # Store price data
                    price_value = TemporalValue(
                        value=Decimal(f"{prices[symbol]:.2f}"),
                        as_of_date=current_date,
                        value_date=current_date,
                        data_type=DataType.PRICE,
                        symbol=symbol
                    )
                    store.store(price_value)
                    
                    # Store volume data
                    base_volume = 10000000
                    volume = int(base_volume * random.uniform(0.5, 2.0))
                    volume_value = TemporalValue(
                        value=volume,
                        as_of_date=current_date,
                        value_date=current_date,
                        data_type=DataType.VOLUME,
                        symbol=symbol
                    )
                    store.store(volume_value)
            
            current_date += timedelta(days=1)
        
        engine = PointInTimeEngine(
            store=store,
            enable_cache=True,
            max_workers=4
        )
        
        return engine, symbols, start_date, end_date
    
    def test_daily_rebalancing_simulation(self, backtesting_engine):
        """Test daily portfolio rebalancing simulation."""
        engine, symbols, start_date, end_date = backtesting_engine
        
        # Simulate daily rebalancing over 1 year
        backtest_start = date(2022, 1, 1)
        backtest_end = date(2022, 12, 31)
        
        portfolio = {symbol: 1.0 / len(symbols) for symbol in symbols}  # Equal weight
        rebalancing_dates = []
        
        current_date = backtest_start
        while current_date <= backtest_end:
            if current_date.weekday() < 5:  # Trading days only
                # Query all symbols for current date
                query = PITQuery(
                    symbols=symbols,
                    as_of_date=current_date,
                    data_types=[DataType.PRICE, DataType.VOLUME],
                    mode=QueryMode.FAST,
                    bias_check=BiasCheckLevel.BASIC
                )
                
                start_time = time.time()
                result = engine.execute_query(query)
                query_time = (time.time() - start_time) * 1000
                
                # Verify we got complete data
                available_symbols = [s for s in symbols if s in result.data and DataType.PRICE in result.data[s]]
                
                if len(available_symbols) >= len(symbols) * 0.8:  # At least 80% of symbols
                    rebalancing_dates.append(current_date)
                    
                    # Verify no look-ahead bias
                    for symbol in available_symbols:
                        price_data = result.data[symbol][DataType.PRICE]
                        assert price_data.as_of_date <= current_date, f"Look-ahead bias: {symbol} on {current_date}"
                
                # Performance requirement for daily rebalancing
                assert query_time < 100, f"Daily rebalancing query too slow: {query_time:.2f}ms"
            
            current_date += timedelta(days=1)
        
        print(f"\nDaily Rebalancing Simulation:")
        print(f"  Backtest period: {backtest_start} to {backtest_end}")
        print(f"  Successful rebalancing days: {len(rebalancing_dates)}")
        print(f"  Expected trading days: ~252")
        
        # Should have most trading days
        assert len(rebalancing_dates) > 200, f"Too few rebalancing days: {len(rebalancing_dates)}"
    
    def test_momentum_strategy_simulation(self, backtesting_engine):
        """Test momentum strategy backtesting with look-back periods."""
        engine, symbols, start_date, end_date = backtesting_engine
        
        # Momentum strategy: select top performers over last 30 days
        lookback_days = 30
        rebalance_frequency = 20  # Every 20 trading days
        
        backtest_start = date(2022, 6, 1)
        backtest_end = date(2022, 12, 31)
        
        current_date = backtest_start
        rebalance_count = 0
        
        while current_date <= backtest_end:
            if current_date.weekday() < 5 and rebalance_count % rebalance_frequency == 0:
                # Calculate momentum for each symbol
                momentum_scores = {}
                
                for symbol in symbols:
                    # Get current price
                    current_query = PITQuery(
                        symbols=[symbol],
                        as_of_date=current_date,
                        data_types=[DataType.PRICE],
                        mode=QueryMode.FAST
                    )
                    current_result = engine.execute_query(current_query)
                    
                    # Get price 30 days ago
                    lookback_date = current_date - timedelta(days=lookback_days)
                    lookback_query = PITQuery(
                        symbols=[symbol],
                        as_of_date=lookback_date,
                        data_types=[DataType.PRICE],
                        mode=QueryMode.FAST
                    )
                    lookback_result = engine.execute_query(lookback_query)
                    
                    # Calculate momentum if both prices available
                    if (symbol in current_result.data and DataType.PRICE in current_result.data[symbol] and
                        symbol in lookback_result.data and DataType.PRICE in lookback_result.data[symbol]):
                        
                        current_price = current_result.data[symbol][DataType.PRICE].value
                        lookback_price = lookback_result.data[symbol][DataType.PRICE].value
                        
                        momentum = float((current_price - lookback_price) / lookback_price)
                        momentum_scores[symbol] = momentum
                        
                        # Verify temporal consistency
                        assert current_result.data[symbol][DataType.PRICE].as_of_date <= current_date
                        assert lookback_result.data[symbol][DataType.PRICE].as_of_date <= lookback_date
                
                # Select top 3 momentum stocks
                if len(momentum_scores) >= 3:
                    sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
                    selected_stocks = [stock[0] for stock in sorted_stocks[:3]]
                    
                    print(f"  {current_date}: Selected {selected_stocks} (momentum: {[f'{momentum_scores[s]:.2%}' for s in selected_stocks]})")
            
            if current_date.weekday() < 5:
                rebalance_count += 1
            
            current_date += timedelta(days=1)
        
        print(f"\nMomentum Strategy Simulation:")
        print(f"  Lookback period: {lookback_days} days")
        print(f"  Rebalance frequency: {rebalance_frequency} trading days")
        print(f"  Total rebalances: {rebalance_count // rebalance_frequency}")
    
    def test_earnings_announcement_strategy(self, backtesting_engine):
        """Test strategy around earnings announcements."""
        engine, symbols, start_date, end_date = backtesting_engine
        
        # Add simulated earnings data
        earnings_calendar = []
        for symbol in symbols[:3]:  # Focus on subset
            for quarter in range(1, 5):  # 4 quarters
                for year in [2022, 2023]:
                    # Earnings typically announced 45 days after quarter end
                    quarter_end = date(year, quarter * 3, [31, 30, 30, 31][quarter-1])
                    if quarter == 2:  # Adjust for February
                        quarter_end = date(year, 6, 30)
                    
                    announcement_date = quarter_end + timedelta(days=45)
                    
                    if start_date <= announcement_date <= end_date:
                        earnings_data = TemporalValue(
                            value={
                                "eps": random.uniform(1.0, 5.0),
                                "revenue": random.uniform(1000000, 5000000),
                                "quarter": quarter,
                                "year": year
                            },
                            as_of_date=announcement_date,
                            value_date=quarter_end,
                            data_type=DataType.FUNDAMENTAL,
                            symbol=symbol
                        )
                        
                        engine.store.store(earnings_data)
                        earnings_calendar.append((symbol, announcement_date))
        
        # Test earnings momentum strategy
        successful_trades = 0
        
        for symbol, earnings_date in earnings_calendar:
            # Check price 5 days before and after earnings
            pre_earnings_date = earnings_date - timedelta(days=5)
            post_earnings_date = earnings_date + timedelta(days=5)
            
            # Get prices
            pre_query = PITQuery(
                symbols=[symbol],
                as_of_date=pre_earnings_date,
                data_types=[DataType.PRICE],
                mode=QueryMode.STRICT
            )
            
            post_query = PITQuery(
                symbols=[symbol],
                as_of_date=post_earnings_date,
                data_types=[DataType.PRICE],
                mode=QueryMode.STRICT
            )
            
            pre_result = engine.execute_query(pre_query)
            post_result = engine.execute_query(post_query)
            
            # Verify no look-ahead bias
            assert len(pre_result.bias_violations) == 0, f"Pre-earnings bias for {symbol}"
            assert len(post_result.bias_violations) == 0, f"Post-earnings bias for {symbol}"
            
            if (symbol in pre_result.data and DataType.PRICE in pre_result.data[symbol] and
                symbol in post_result.data and DataType.PRICE in post_result.data[symbol]):
                
                successful_trades += 1
        
        print(f"\nEarnings Strategy Simulation:")
        print(f"  Total earnings events: {len(earnings_calendar)}")
        print(f"  Successful trades: {successful_trades}")
        print(f"  Success rate: {successful_trades/len(earnings_calendar):.1%}")
        
        # Should be able to trade around most earnings events
        assert successful_trades / len(earnings_calendar) > 0.8, "Too many failed earnings trades"


if __name__ == "__main__":
    # Run advanced PIT engine tests
    pytest.main([__file__, "-v", "--tb=short"])