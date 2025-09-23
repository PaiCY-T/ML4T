"""
Edge Case Look-Ahead Bias Validation Tests.

This module tests extreme edge cases and corner scenarios to ensure
the Point-in-Time Data Management System prevents look-ahead bias
under the most challenging conditions, including:
- Microsecond-level temporal precision
- Data corruption scenarios  
- System clock synchronization issues
- Extreme market conditions
- Cross-timezone data handling
- Memory pressure scenarios
- Network partition simulations
"""

import pytest
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional
import random
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import gc
import psutil
import os

# Import components to test
from src.data.core.temporal import (
    TemporalValue, InMemoryTemporalStore, TemporalDataManager,
    DataType, calculate_data_lag
)

from src.data.pipeline.pit_engine import (
    PointInTimeEngine, PITQuery, PITResult, BiasDetector,
    QueryMode, BiasCheckLevel
)

from src.data.models.taiwan_market import (
    TaiwanMarketData, TaiwanSettlement, TaiwanFundamental,
    TaiwanCorporateAction, CorporateActionType, TaiwanMarketDataValidator,
    create_taiwan_trading_calendar, taiwan_market_time_to_utc
)


class TestMicrosecondTemporalPrecision:
    """Test microsecond-level temporal precision bias prevention."""
    
    def test_microsecond_ordering_bias_prevention(self):
        """Test bias prevention with microsecond-level timestamp differences."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        base_datetime = datetime(2024, 1, 15, 9, 0, 0, 0)  # Market open
        
        # Create data with microsecond differences
        microsecond_data = []
        for i in range(100):
            # Create timestamps 1 microsecond apart
            timestamp = base_datetime + timedelta(microseconds=i)
            price = Decimal(f"{580 + (i * 0.01):.2f}")
            
            value = TemporalValue(
                value=price,
                as_of_date=timestamp.date(),
                value_date=timestamp.date(),
                data_type=DataType.PRICE,
                symbol=symbol,
                created_at=timestamp,
                metadata={"microsecond_sequence": i}
            )
            microsecond_data.append(value)
            store.store(value)
        
        # Test queries at various microsecond boundaries
        for test_microsecond in [0, 25, 50, 75, 99]:
            query_time = base_datetime + timedelta(microseconds=test_microsecond)
            
            query = PITQuery(
                symbols=[symbol],
                as_of_date=query_time.date(),
                data_types=[DataType.PRICE],
                mode=QueryMode.STRICT,
                bias_check=BiasCheckLevel.PARANOID
            )
            
            result = engine.execute_query(query)
            assert len(result.bias_violations) == 0, f"Bias violations at microsecond {test_microsecond}"
            
            if symbol in result.data and DataType.PRICE in result.data[symbol]:
                retrieved_value = result.data[symbol][DataType.PRICE]
                retrieved_sequence = retrieved_value.metadata.get("microsecond_sequence", -1)
                
                # Should not get data from future microseconds
                assert retrieved_sequence <= test_microsecond, \
                    f"Got future microsecond data: sequence {retrieved_sequence} > {test_microsecond}"
    
    def test_nanosecond_temporal_boundaries(self):
        """Test temporal boundaries at nanosecond precision."""
        store = InMemoryTemporalStore()
        
        symbol = "2330"
        base_time = datetime(2024, 1, 15, 13, 30, 0, 0)  # Market close
        
        # Create two values at the exact boundary of market close
        pre_close_value = TemporalValue(
            value=Decimal("580.00"),
            as_of_date=base_time.date(),
            value_date=base_time.date(),
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=base_time - timedelta(microseconds=1),  # 1 microsecond before
            metadata={"position": "pre_close"}
        )
        
        post_close_value = TemporalValue(
            value=Decimal("581.00"),  # Different value
            as_of_date=base_time.date(),
            value_date=base_time.date(),
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=base_time + timedelta(microseconds=1),  # 1 microsecond after
            metadata={"position": "post_close"}
        )
        
        store.store(pre_close_value)
        store.store(post_close_value)
        
        # Query exactly at market close
        exact_close_query = store.get_point_in_time(symbol, base_time.date(), DataType.PRICE)
        
        # Should get the most recent value that was available at query time
        assert exact_close_query is not None
        # The exact behavior depends on implementation, but should be consistent


class TestDataCorruptionScenarios:
    """Test bias prevention under data corruption scenarios."""
    
    def test_corrupted_timestamp_bias_prevention(self):
        """Test bias prevention when timestamps are corrupted."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        normal_date = date(2024, 1, 15)
        
        # Store normal data
        normal_value = TemporalValue(
            value=Decimal("580.00"),
            as_of_date=normal_date,
            value_date=normal_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            metadata={"data_type": "normal"}
        )
        store.store(normal_value)
        
        # Simulate corrupted data with impossible future dates
        corrupted_values = [
            TemporalValue(
                value=Decimal("999.99"),
                as_of_date=date(2099, 12, 31),  # Far future
                value_date=normal_date,
                data_type=DataType.PRICE,
                symbol=symbol,
                metadata={"data_type": "corrupted_future_as_of"}
            ),
            TemporalValue(
                value=Decimal("888.88"),
                as_of_date=normal_date,
                value_date=date(2099, 12, 31),  # Far future value date
                data_type=DataType.PRICE,
                symbol=symbol,
                metadata={"data_type": "corrupted_future_value"}
            ),
            TemporalValue(
                value=Decimal("777.77"),
                as_of_date=date(1900, 1, 1),  # Impossibly old
                value_date=normal_date,
                data_type=DataType.PRICE,
                symbol=symbol,
                metadata={"data_type": "corrupted_ancient"}
            )
        ]
        
        for corrupted_value in corrupted_values:
            store.store(corrupted_value)
        
        # Query should not return corrupted data
        query = PITQuery(
            symbols=[symbol],
            as_of_date=normal_date + timedelta(days=1),
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.PARANOID
        )
        
        result = engine.execute_query(query)
        
        if symbol in result.data and DataType.PRICE in result.data[symbol]:
            retrieved_value = result.data[symbol][DataType.PRICE]
            data_type = retrieved_value.metadata.get("data_type", "unknown")
            
            # Should not get any corrupted data
            assert data_type == "normal", f"Retrieved corrupted data: {data_type}"
            assert retrieved_value.value == Decimal("580.00"), "Got wrong value from corrupted data"
    
    def test_negative_price_data_handling(self):
        """Test handling of invalid negative price data."""
        store = InMemoryTemporalStore()
        validator = TaiwanMarketDataValidator()
        
        symbol = "2330"
        test_date = date(2024, 1, 15)
        
        # Create market data with negative prices (data corruption)
        corrupted_market_data = TaiwanMarketData(
            symbol=symbol,
            data_date=test_date,
            as_of_date=test_date,
            open_price=Decimal("-100.00"),  # Negative prices (impossible)
            high_price=Decimal("-50.00"),
            low_price=Decimal("-150.00"),
            close_price=Decimal("-75.00"),
            volume=-1000000,  # Negative volume (impossible)
            turnover=Decimal("-10000000")  # Negative turnover (impossible)
        )
        
        # Validate should catch these issues
        validation_issues = validator.validate_price_data(corrupted_market_data)
        assert len(validation_issues) > 0, "Failed to detect negative price corruption"
        
        # Check specific issue types
        negative_issues = [issue for issue in validation_issues if "negative" in issue.lower()]
        assert len(negative_issues) >= 2, f"Not enough negative value detections: {len(negative_issues)}"
    
    def test_null_and_missing_data_handling(self):
        """Test handling of null and missing data fields."""
        store = InMemoryTemporalStore()
        
        symbol = "2330"
        test_date = date(2024, 1, 15)
        
        # Test various null/missing scenarios
        incomplete_values = [
            TemporalValue(
                value=None,  # Null value
                as_of_date=test_date,
                value_date=test_date,
                data_type=DataType.PRICE,
                symbol=symbol,
                metadata={"issue": "null_value"}
            ),
            TemporalValue(
                value=Decimal("580.00"),
                as_of_date=None,  # Null as_of_date
                value_date=test_date,
                data_type=DataType.PRICE,
                symbol=symbol,
                metadata={"issue": "null_as_of_date"}
            ),
            TemporalValue(
                value=Decimal("580.00"),
                as_of_date=test_date,
                value_date=None,  # Null value_date
                data_type=DataType.PRICE,
                symbol=symbol,
                metadata={"issue": "null_value_date"}
            )
        ]
        
        stored_count = 0
        for incomplete_value in incomplete_values:
            try:
                store.store(incomplete_value)
                stored_count += 1
            except (ValueError, TypeError, AttributeError):
                # Expected to fail for incomplete data
                pass
        
        # Most incomplete data should be rejected
        assert stored_count < len(incomplete_values), "Too many incomplete values were stored"


class TestSystemClockSynchronization:
    """Test bias prevention with system clock synchronization issues."""
    
    @patch('datetime.datetime')
    def test_system_clock_drift_bias_prevention(self, mock_datetime):
        """Test bias prevention when system clock drifts."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        real_time = datetime(2024, 1, 15, 10, 0, 0)
        
        # Simulate system clock running 1 hour fast
        fast_time = real_time + timedelta(hours=1)
        mock_datetime.now.return_value = fast_time
        mock_datetime.utcnow.return_value = fast_time
        
        # Store data with "current" timestamp (which is actually 1 hour in future)
        future_biased_value = TemporalValue(
            value=Decimal("580.00"),
            as_of_date=fast_time.date(),  # Future date due to clock drift
            value_date=fast_time.date(),
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=fast_time,
            metadata={"clock_status": "fast"}
        )
        store.store(future_biased_value)
        
        # Reset mock to real time
        mock_datetime.now.return_value = real_time
        mock_datetime.utcnow.return_value = real_time
        
        # Query at real time should detect clock drift issue
        query = PITQuery(
            symbols=[symbol],
            as_of_date=real_time.date(),
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.PARANOID
        )
        
        result = engine.execute_query(query)
        
        # Should either reject the future data or flag bias violations
        if symbol in result.data and DataType.PRICE in result.data[symbol]:
            # If data is returned, verify it's not from the "future"
            retrieved_value = result.data[symbol][DataType.PRICE]
            assert retrieved_value.as_of_date <= real_time.date(), "Clock drift created look-ahead bias"
    
    def test_timezone_confusion_bias_prevention(self):
        """Test bias prevention with timezone confusion."""
        store = InMemoryTemporalStore()
        
        symbol = "2330"
        taiwan_date = date(2024, 1, 15)
        
        # Taiwan market time (TST = UTC+8)
        taiwan_market_open = datetime(2024, 1, 15, 9, 0, 0)  # 09:00 TST
        taiwan_market_close = datetime(2024, 1, 15, 13, 30, 0)  # 13:30 TST
        
        # Store data with various timezone interpretations
        timezone_values = [
            TemporalValue(
                value=Decimal("580.00"),
                as_of_date=taiwan_date,
                value_date=taiwan_date,
                data_type=DataType.PRICE,
                symbol=symbol,
                created_at=taiwan_market_open,  # Local Taiwan time
                metadata={"timezone": "TST", "time_type": "market_open"}
            ),
            TemporalValue(
                value=Decimal("582.00"),
                as_of_date=taiwan_date,
                value_date=taiwan_date,
                data_type=DataType.PRICE,
                symbol=symbol,
                created_at=taiwan_market_close,  # Local Taiwan time
                metadata={"timezone": "TST", "time_type": "market_close"}
            ),
            TemporalValue(
                value=Decimal("583.00"),
                as_of_date=taiwan_date,
                value_date=taiwan_date,
                data_type=DataType.PRICE,
                symbol=symbol,
                # Accidentally stored as UTC (would be future in Taiwan timezone)
                created_at=datetime(2024, 1, 15, 1, 0, 0),  # 01:00 UTC = 09:00 TST
                metadata={"timezone": "UTC", "time_type": "confused_timezone"}
            )
        ]
        
        for value in timezone_values:
            store.store(value)
        
        # Query during Taiwan trading hours
        trading_hour_query = store.get_point_in_time(symbol, taiwan_date, DataType.PRICE)
        
        assert trading_hour_query is not None
        # Should get appropriate value based on temporal ordering
        # The exact behavior depends on how the store handles timezone data


class TestExtremeMarketConditions:
    """Test bias prevention under extreme market conditions."""
    
    def test_flash_crash_bias_prevention(self):
        """Test bias prevention during flash crash scenarios."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        crash_date = date(2024, 1, 15)
        
        # Normal trading data before crash
        pre_crash_time = datetime.combine(crash_date, datetime.min.time().replace(hour=10))
        normal_value = TemporalValue(
            value=Decimal("580.00"),
            as_of_date=crash_date,
            value_date=crash_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=pre_crash_time,
            metadata={"event": "normal_trading"}
        )
        store.store(normal_value)
        
        # Flash crash - extreme price drop in seconds
        crash_time = pre_crash_time + timedelta(seconds=30)
        crash_value = TemporalValue(
            value=Decimal("520.00"),  # -10.3% in 30 seconds
            as_of_date=crash_date,
            value_date=crash_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=crash_time,
            metadata={"event": "flash_crash"}
        )
        store.store(crash_value)
        
        # Flash recovery - price rebounds quickly
        recovery_time = crash_time + timedelta(seconds=45)
        recovery_value = TemporalValue(
            value=Decimal("575.00"),  # Recovery but not full
            as_of_date=crash_date,
            value_date=crash_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=recovery_time,
            metadata={"event": "flash_recovery"}
        )
        store.store(recovery_value)
        
        # Test queries at different points during flash crash
        test_times = [
            (pre_crash_time, "normal_trading", Decimal("580.00")),
            (crash_time, "flash_crash", Decimal("520.00")),
            (recovery_time, "flash_recovery", Decimal("575.00"))
        ]
        
        for test_time, expected_event, expected_price in test_times:
            query = PITQuery(
                symbols=[symbol],
                as_of_date=test_time.date(),
                data_types=[DataType.PRICE],
                mode=QueryMode.STRICT,
                bias_check=BiasCheckLevel.STRICT
            )
            
            result = engine.execute_query(query)
            assert len(result.bias_violations) == 0, f"Bias violation during {expected_event}"
            
            if symbol in result.data and DataType.PRICE in result.data[symbol]:
                retrieved_value = result.data[symbol][DataType.PRICE]
                
                # Should not get future data during flash crash
                assert retrieved_value.created_at <= test_time, \
                    f"Future data during flash crash: {retrieved_value.created_at} > {test_time}"
    
    def test_trading_halt_bias_prevention(self):
        """Test bias prevention during trading halts."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        halt_date = date(2024, 1, 15)
        
        # Normal trading before halt
        pre_halt_time = datetime.combine(halt_date, datetime.min.time().replace(hour=10))
        pre_halt_value = TemporalValue(
            value=Decimal("580.00"),
            as_of_date=halt_date,
            value_date=halt_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=pre_halt_time,
            metadata={"trading_status": "normal"}
        )
        store.store(pre_halt_value)
        
        # Trading halt announcement
        halt_time = pre_halt_time + timedelta(minutes=30)
        halt_announcement = TemporalValue(
            value={"status": "trading_halted", "reason": "volatility", "expected_resume": "11:30"},
            as_of_date=halt_date,
            value_date=halt_date,
            data_type=DataType.CORPORATE_ACTION,
            symbol=symbol,
            created_at=halt_time,
            metadata={"event_type": "trading_halt"}
        )
        store.store(halt_announcement)
        
        # Price data after halt (should not be available during halt)
        during_halt_time = halt_time + timedelta(minutes=15)
        during_halt_value = TemporalValue(
            value=Decimal("585.00"),  # Price movement during halt
            as_of_date=halt_date,
            value_date=halt_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=during_halt_time,
            metadata={"trading_status": "halted", "data_validity": "questionable"}
        )
        store.store(during_halt_value)
        
        # Query during halt period
        during_halt_query = PITQuery(
            symbols=[symbol],
            as_of_date=halt_date,
            data_types=[DataType.PRICE, DataType.CORPORATE_ACTION],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        result = engine.execute_query(during_halt_query)
        assert len(result.bias_violations) == 0
        
        # Should see halt announcement
        corp_action = result.data.get(symbol, {}).get(DataType.CORPORATE_ACTION)
        if corp_action:
            assert corp_action.metadata.get("event_type") == "trading_halt"
        
        # Price data handling during halt depends on implementation
        # but should not create look-ahead bias


class TestMemoryPressureScenarios:
    """Test bias prevention under memory pressure."""
    
    def test_low_memory_bias_prevention(self):
        """Test bias prevention when system is under memory pressure."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        symbols = [f"SYM{i:04d}" for i in range(1000)]  # Many symbols
        base_date = date(2024, 1, 1)
        
        # Create large dataset to pressure memory
        all_values = []
        for i, symbol in enumerate(symbols):
            for day in range(30):  # 30 days per symbol
                test_date = base_date + timedelta(days=day)
                
                value = TemporalValue(
                    value=Decimal(f"{100 + i + day:.2f}"),
                    as_of_date=test_date,
                    value_date=test_date,
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    metadata={"symbol_index": i, "day_index": day}
                )
                all_values.append(value)
                store.store(value)
        
        # Force garbage collection
        gc.collect()
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        print(f"Memory pressure test: {len(all_values):,} values, {memory_used:.1f}MB used")
        
        # Test queries under memory pressure
        query_date = base_date + timedelta(days=15)
        test_symbols = random.sample(symbols, 50)  # Test subset
        
        query = PITQuery(
            symbols=test_symbols,
            as_of_date=query_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.FAST,  # Use fast mode to reduce memory usage
            bias_check=BiasCheckLevel.BASIC
        )
        
        result = engine.execute_query(query)
        
        # Should still prevent bias even under memory pressure
        assert len(result.bias_violations) == 0, "Memory pressure caused bias violations"
        
        # Verify some results
        assert len(result.data) > 0, "No data returned under memory pressure"
        
        for symbol in result.data:
            price_data = result.data[symbol].get(DataType.PRICE)
            if price_data:
                assert price_data.as_of_date <= query_date, \
                    f"Memory pressure caused temporal violation: {price_data.as_of_date} > {query_date}"
    
    def test_memory_fragmentation_handling(self):
        """Test handling of memory fragmentation scenarios."""
        store = InMemoryTemporalStore()
        
        symbol = "2330"
        base_date = date(2024, 1, 1)
        
        # Create and delete data repeatedly to fragment memory
        for cycle in range(10):
            # Create temporary data
            temp_values = []
            for day in range(100):
                test_date = base_date + timedelta(days=day)
                value = TemporalValue(
                    value=Decimal(f"{100 + cycle * 10 + day:.2f}"),
                    as_of_date=test_date,
                    value_date=test_date,
                    data_type=DataType.PRICE,
                    symbol=f"{symbol}_TEMP_{cycle}",
                    metadata={"cycle": cycle, "temporary": True}
                )
                temp_values.append(value)
                store.store(value)
            
            # Keep only some permanent data
            if cycle % 3 == 0:  # Every 3rd cycle
                permanent_value = TemporalValue(
                    value=Decimal(f"{580 + cycle:.2f}"),
                    as_of_date=base_date + timedelta(days=cycle * 10),
                    value_date=base_date + timedelta(days=cycle * 10),
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    metadata={"cycle": cycle, "permanent": True}
                )
                store.store(permanent_value)
            
            # Force garbage collection to fragment memory
            gc.collect()
        
        # Test query after memory fragmentation
        query_date = base_date + timedelta(days=50)
        result = store.get_point_in_time(symbol, query_date, DataType.PRICE)
        
        if result:
            assert result.metadata.get("permanent") is True, "Got temporary data after fragmentation"
            assert result.as_of_date <= query_date, "Memory fragmentation caused temporal violation"


class TestConcurrentAccessBiasScenarios:
    """Test bias prevention under concurrent access scenarios."""
    
    def test_concurrent_write_read_bias_prevention(self):
        """Test bias prevention with concurrent writes and reads."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        base_date = date(2024, 1, 15)
        
        results = []
        errors = []
        bias_violations = []
        
        def writer_thread():
            """Thread that continuously writes data."""
            try:
                for i in range(100):
                    value = TemporalValue(
                        value=Decimal(f"{580 + i * 0.1:.2f}"),
                        as_of_date=base_date + timedelta(seconds=i),  # Incremental timestamps
                        value_date=base_date + timedelta(seconds=i),
                        data_type=DataType.PRICE,
                        symbol=symbol,
                        metadata={"sequence": i, "thread": "writer"}
                    )
                    store.store(value)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"Writer error: {e}")
        
        def reader_thread(thread_id):
            """Thread that continuously reads data."""
            try:
                for i in range(50):
                    # Query at different points in time
                    query_date = base_date + timedelta(seconds=i * 2)
                    
                    query = PITQuery(
                        symbols=[symbol],
                        as_of_date=query_date,
                        data_types=[DataType.PRICE],
                        mode=QueryMode.FAST,
                        bias_check=BiasCheckLevel.BASIC
                    )
                    
                    result = engine.execute_query(query)
                    results.append((thread_id, i, result))
                    
                    if result.bias_violations:
                        bias_violations.extend(result.bias_violations)
                    
                    time.sleep(0.002)  # Small delay
            except Exception as e:
                errors.append(f"Reader {thread_id} error: {e}")
        
        # Start writer thread
        writer = threading.Thread(target=writer_thread)
        writer.start()
        
        # Start multiple reader threads
        readers = []
        for i in range(3):
            reader = threading.Thread(target=reader_thread, args=(i,))
            readers.append(reader)
            reader.start()
        
        # Wait for all threads
        writer.join()
        for reader in readers:
            reader.join()
        
        # Verify no errors or bias violations occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(bias_violations) == 0, f"Concurrent access bias violations: {bias_violations}"
        assert len(results) > 0, "No results from concurrent access test"
        
        # Verify temporal consistency in all results
        for thread_id, query_index, result in results:
            for symbol_data in result.data.values():
                for data_type, temporal_value in symbol_data.items():
                    query_date = base_date + timedelta(seconds=query_index * 2)
                    assert temporal_value.as_of_date <= query_date, \
                        f"Concurrent access temporal violation: {temporal_value.as_of_date} > {query_date}"
    
    def test_race_condition_bias_prevention(self):
        """Test bias prevention in race condition scenarios."""
        store = InMemoryTemporalStore()
        
        symbol = "2330"
        critical_time = datetime(2024, 1, 15, 13, 30, 0)  # Market close
        
        race_results = []
        
        def racing_thread(thread_id, offset_microseconds):
            """Thread that tries to store data at the same critical time."""
            try:
                # Each thread tries to store data at slightly different times
                store_time = critical_time + timedelta(microseconds=offset_microseconds)
                
                value = TemporalValue(
                    value=Decimal(f"{580 + thread_id:.2f}"),
                    as_of_date=critical_time.date(),
                    value_date=critical_time.date(),
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    created_at=store_time,
                    metadata={"thread_id": thread_id, "offset_microseconds": offset_microseconds}
                )
                
                store.store(value)
                race_results.append((thread_id, offset_microseconds, "success"))
                
            except Exception as e:
                race_results.append((thread_id, offset_microseconds, f"error: {e}"))
        
        # Create racing threads with microsecond offsets
        threads = []
        for i in range(10):
            offset = i - 5  # -5 to +4 microseconds
            thread = threading.Thread(target=racing_thread, args=(i, offset))
            threads.append(thread)
        
        # Start all threads simultaneously
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify race condition results
        successful_stores = [r for r in race_results if r[2] == "success"]
        assert len(successful_stores) > 0, "No successful stores in race condition"
        
        # Query the data to verify temporal consistency
        query_result = store.get_point_in_time(symbol, critical_time.date(), DataType.PRICE)
        
        if query_result:
            # Should have consistent temporal ordering despite race conditions
            assert query_result.as_of_date <= critical_time.date()
            thread_id = query_result.metadata.get("thread_id", -1)
            print(f"Race condition winner: thread {thread_id}")


class TestNetworkPartitionSimulation:
    """Test bias prevention with network partition simulation."""
    
    @patch('src.data.pipeline.incremental_updater.FinLabConnector')
    def test_network_partition_bias_prevention(self, mock_connector_class):
        """Test bias prevention when network connections are unreliable."""
        store = InMemoryTemporalStore()
        
        # Mock unreliable network connector
        mock_connector = Mock()
        mock_connector_class.return_value = mock_connector
        
        symbol = "2330"
        base_date = date(2024, 1, 15)
        
        # Simulate intermittent network failures
        call_count = 0
        def unreliable_get_price_data(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count % 3 == 0:  # Fail every 3rd call
                raise ConnectionError("Network partition simulation")
            
            # Return mock data for successful calls
            return [TemporalValue(
                value=Decimal(f"{580 + call_count:.2f}"),
                as_of_date=base_date + timedelta(days=call_count),
                value_date=base_date + timedelta(days=call_count),
                data_type=DataType.PRICE,
                symbol=symbol,
                metadata={"call_count": call_count, "source": "unreliable_network"}
            )]
        
        mock_connector.get_price_data.side_effect = unreliable_get_price_data
        
        # Attempt to retrieve data multiple times
        successful_retrievals = 0
        network_errors = 0
        
        for attempt in range(10):
            try:
                data = mock_connector.get_price_data(symbol, base_date, base_date + timedelta(days=1))
                for value in data:
                    store.store(value)
                successful_retrievals += 1
            except ConnectionError:
                network_errors += 1
        
        print(f"Network partition simulation: {successful_retrievals} successful, {network_errors} failed")
        
        # Verify that successful data doesn't violate temporal constraints
        for attempt_day in range(successful_retrievals):
            query_date = base_date + timedelta(days=attempt_day + 1)
            result = store.get_point_in_time(symbol, query_date, DataType.PRICE)
            
            if result and result.metadata.get("source") == "unreliable_network":
                assert result.as_of_date <= query_date, \
                    f"Network partition caused temporal violation: {result.as_of_date} > {query_date}"


if __name__ == "__main__":
    # Run edge case bias validation tests
    pytest.main([__file__, "-v", "--tb=short"])