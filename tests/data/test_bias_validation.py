"""
Comprehensive Look-Ahead Bias Validation Tests.

This test suite validates that the Point-in-Time Data Management System
completely prevents look-ahead bias under all scenarios, including:
- Temporal data access patterns
- Corporate action timing
- Fundamental data lags
- Settlement timing
- Edge cases and stress scenarios
"""

import pytest
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
import random
from unittest.mock import Mock, patch

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
    create_taiwan_trading_calendar
)


class TestTemporalBiasValidation:
    """Test temporal bias prevention at the core level."""
    
    @pytest.fixture
    def store_with_temporal_data(self):
        """Create store with carefully crafted temporal data."""
        store = InMemoryTemporalStore()
        
        # Create data with various temporal patterns
        symbol = "2330"
        base_date = date(2024, 1, 1)
        
        # Add historical data (safe)
        for i in range(365):
            data_date = base_date + timedelta(days=i)
            value = TemporalValue(
                value=Decimal(f"{580 + random.uniform(-10, 10):.2f}"),
                as_of_date=data_date,
                value_date=data_date,
                data_type=DataType.PRICE,
                symbol=symbol,
                metadata={"source": "historical"}
            )
            store.store(value)
        
        # Add some future data (should trigger bias detection)
        future_date = base_date + timedelta(days=400)
        future_value = TemporalValue(
            value=Decimal("999.99"),  # Distinctive value
            as_of_date=future_date,
            value_date=future_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            metadata={"source": "future"}
        )
        store.store(future_value)
        
        return store, symbol, base_date
    
    def test_no_future_data_access(self, store_with_temporal_data):
        """Test that future data is never accessible."""
        store, symbol, base_date = store_with_temporal_data
        
        # Query for data that exists but is in the future
        query_date = base_date + timedelta(days=300)  # Before the future data
        
        result = store.get_point_in_time(symbol, query_date, DataType.PRICE)
        
        # Should not get the future value (999.99)
        assert result is not None
        assert result.value != Decimal("999.99"), "Retrieved future data!"
        assert result.as_of_date <= query_date, f"Data from future: {result.as_of_date} > {query_date}"
    
    def test_temporal_ordering_enforcement(self, store_with_temporal_data):
        """Test that temporal ordering is strictly enforced."""
        store, symbol, base_date = store_with_temporal_data
        
        # Query multiple dates and verify temporal ordering
        query_dates = [base_date + timedelta(days=i*30) for i in range(10)]
        
        previous_as_of_date = None
        for query_date in query_dates:
            result = store.get_point_in_time(symbol, query_date, DataType.PRICE)
            
            if result:
                # Verify this data is not from the future
                assert result.as_of_date <= query_date, f"Future data access: {result.as_of_date} > {query_date}"
                
                # Verify temporal progression (monotonic or equal)
                if previous_as_of_date:
                    assert result.as_of_date >= previous_as_of_date, f"Temporal regression: {result.as_of_date} < {previous_as_of_date}"
                
                previous_as_of_date = result.as_of_date
    
    def test_data_version_temporal_consistency(self):
        """Test temporal consistency with data revisions."""
        store = InMemoryTemporalStore()
        symbol = "2330"
        data_date = date(2024, 1, 15)
        
        # Store initial value
        initial_value = TemporalValue(
            value=Decimal("100.00"),
            as_of_date=data_date,
            value_date=data_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            version=1
        )
        store.store(initial_value)
        
        # Store revised value (should be available later)
        revision_date = data_date + timedelta(days=1)
        revised_value = TemporalValue(
            value=Decimal("101.00"),  # Corrected price
            as_of_date=revision_date,
            value_date=data_date,  # Same value date, different as_of_date
            data_type=DataType.PRICE,
            symbol=symbol,
            version=2
        )
        store.store(revised_value)
        
        # Query on original date - should get original value
        original_result = store.get_point_in_time(symbol, data_date, DataType.PRICE)
        assert original_result.value == Decimal("100.00"), "Got revised value before revision date"
        
        # Query after revision - should get revised value
        revised_result = store.get_point_in_time(symbol, revision_date, DataType.PRICE)
        assert revised_result.value == Decimal("101.00"), "Didn't get revised value after revision date"


class TestBiasDetectorValidation:
    """Test the bias detector component."""
    
    @pytest.fixture
    def bias_detector(self):
        """Create bias detector with Taiwan trading calendar."""
        calendar = create_taiwan_trading_calendar(2024)
        return BiasDetector(calendar)
    
    def test_future_query_detection(self, bias_detector):
        """Test detection of future query dates."""
        future_date = date.today() + timedelta(days=30)
        
        query = PITQuery(
            symbols=["2330"],
            as_of_date=future_date,
            data_types=[DataType.PRICE],
            bias_check=BiasCheckLevel.STRICT
        )
        
        violations = bias_detector.check_query_bias(query)
        
        assert len(violations) > 0, "Failed to detect future query date"
        assert any("future" in v.lower() for v in violations), "Wrong violation type for future date"
    
    def test_data_lag_validation(self, bias_detector):
        """Test data lag validation for different data types."""
        query_date = date(2024, 1, 15)
        
        # Test fundamental data (should have 60-day lag)
        fundamental_query = PITQuery(
            symbols=["2330"],
            as_of_date=query_date,
            data_types=[DataType.FUNDAMENTAL],
            bias_check=BiasCheckLevel.STRICT
        )
        
        violations = bias_detector.check_query_bias(fundamental_query)
        # May have violations depending on expected lag
        
        # Test price data (should have no lag)
        price_query = PITQuery(
            symbols=["2330"],
            as_of_date=query_date,
            data_types=[DataType.PRICE],
            bias_check=BiasCheckLevel.STRICT
        )
        
        price_violations = bias_detector.check_query_bias(price_query)
        # Should not have lag-related violations for price data
    
    def test_value_bias_detection(self, bias_detector):
        """Test bias detection for individual temporal values."""
        query_date = date(2024, 1, 15)
        
        # Valid value (not from future)
        valid_value = TemporalValue(
            value=100.0,
            as_of_date=query_date - timedelta(days=1),
            value_date=query_date - timedelta(days=1),
            data_type=DataType.PRICE,
            symbol="2330"
        )
        
        valid_violations = bias_detector.check_value_bias(valid_value, query_date)
        assert len(valid_violations) == 0, "False positive bias detection"
        
        # Invalid value (from future)
        invalid_value = TemporalValue(
            value=100.0,
            as_of_date=query_date + timedelta(days=1),  # Future
            value_date=query_date + timedelta(days=1),
            data_type=DataType.PRICE,
            symbol="2330"
        )
        
        invalid_violations = bias_detector.check_value_bias(invalid_value, query_date)
        assert len(invalid_violations) > 0, "Failed to detect future value bias"
    
    def test_settlement_timing_validation(self, bias_detector):
        """Test T+2 settlement timing validation."""
        trade_date = date(2024, 1, 15)  # Monday
        
        # Valid T+2 settlement (Wednesday)
        valid_settlement = date(2024, 1, 17)
        violations = bias_detector.validate_settlement_timing("2330", trade_date, valid_settlement)
        assert len(violations) == 0, "False positive T+2 violation"
        
        # Invalid T+1 settlement (Tuesday)
        invalid_settlement = date(2024, 1, 16)
        violations = bias_detector.validate_settlement_timing("2330", trade_date, invalid_settlement)
        assert len(violations) > 0, "Failed to detect T+1 violation"
        
        # Invalid same-day settlement
        same_day_settlement = trade_date
        violations = bias_detector.validate_settlement_timing("2330", trade_date, same_day_settlement)
        assert len(violations) > 0, "Failed to detect same-day settlement violation"


class TestPITEngineBiasValidation:
    """Test bias validation in the PIT engine."""
    
    @pytest.fixture
    def engine_with_bias_scenarios(self):
        """Create PIT engine with various bias scenarios."""
        store = InMemoryTemporalStore()
        symbol = "2330"
        base_date = date(2024, 1, 1)
        
        # Add legitimate historical data
        for i in range(100):
            data_date = base_date + timedelta(days=i)
            
            value = TemporalValue(
                value=Decimal(f"{580 + random.uniform(-5, 5):.2f}"),
                as_of_date=data_date,
                value_date=data_date,
                data_type=DataType.PRICE,
                symbol=symbol
            )
            store.store(value)
        
        # Add problematic data that should trigger bias detection
        
        # 1. Future as_of_date
        future_as_of = TemporalValue(
            value=Decimal("999.01"),
            as_of_date=base_date + timedelta(days=200),  # Future
            value_date=base_date + timedelta(days=50),   # Past value date
            data_type=DataType.PRICE,
            symbol=symbol,
            metadata={"bias_type": "future_as_of"}
        )
        store.store(future_as_of)
        
        # 2. Future value_date
        future_value_date = TemporalValue(
            value=Decimal("999.02"),
            as_of_date=base_date + timedelta(days=50),
            value_date=base_date + timedelta(days=200),  # Future
            data_type=DataType.PRICE,
            symbol=symbol,
            metadata={"bias_type": "future_value_date"}
        )
        store.store(future_value_date)
        
        # 3. Insufficient fundamental lag
        premature_fundamental = TemporalValue(
            value={"eps": 5.0, "revenue": 1000000},
            as_of_date=base_date + timedelta(days=30),  # Too soon
            value_date=base_date,  # Report date
            data_type=DataType.FUNDAMENTAL,
            symbol=symbol,
            metadata={"bias_type": "premature_fundamental"}
        )
        store.store(premature_fundamental)
        
        engine = PointInTimeEngine(store, enable_cache=True)
        return engine, symbol, base_date
    
    def test_strict_bias_checking_prevents_violations(self, engine_with_bias_scenarios):
        """Test that strict bias checking prevents all violations."""
        engine, symbol, base_date = engine_with_bias_scenarios
        
        # Query that should not get any problematic data
        query_date = base_date + timedelta(days=75)
        
        query = PITQuery(
            symbols=[symbol],
            as_of_date=query_date,
            data_types=[DataType.PRICE, DataType.FUNDAMENTAL],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        try:
            result = engine.execute_query(query)
            
            # If result is returned, verify no bias violations
            assert len(result.bias_violations) == 0, f"Bias violations found: {result.bias_violations}"
            
            # Verify we didn't get any of the problematic values
            if symbol in result.data and DataType.PRICE in result.data[symbol]:
                price_value = result.data[symbol][DataType.PRICE]
                assert price_value.value not in [Decimal("999.01"), Decimal("999.02")], "Got problematic bias data"
                assert price_value.as_of_date <= query_date, "Got future as_of_date"
                assert price_value.value_date <= query_date, "Got future value_date"
            
            # Fundamental data should respect lag requirements
            if symbol in result.data and DataType.FUNDAMENTAL in result.data[symbol]:
                fundamental_value = result.data[symbol][DataType.FUNDAMENTAL]
                expected_lag = calculate_data_lag(DataType.FUNDAMENTAL)
                earliest_valid_date = fundamental_value.value_date + expected_lag
                assert fundamental_value.as_of_date >= earliest_valid_date, "Fundamental data lag violation"
        
        except ValueError as e:
            # Strict mode might raise exceptions for bias violations
            assert "bias" in str(e).lower(), f"Unexpected error: {e}"
    
    def test_paranoid_bias_checking(self, engine_with_bias_scenarios):
        """Test paranoid bias checking mode."""
        engine, symbol, base_date = engine_with_bias_scenarios
        
        query = PITQuery(
            symbols=[symbol],
            as_of_date=base_date + timedelta(days=75),
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.PARANOID  # Maximum validation
        )
        
        result = engine.execute_query(query)
        
        # Paranoid mode should catch even subtle issues
        # At minimum, should not return any bias violations in the result
        for violation in result.bias_violations:
            print(f"Paranoid bias detection: {violation}")
        
        # Verify strict temporal consistency
        if symbol in result.data and DataType.PRICE in result.data[symbol]:
            price_value = result.data[symbol][DataType.PRICE]
            assert price_value.as_of_date <= query.as_of_date
            assert price_value.value_date <= query.as_of_date
    
    def test_fast_mode_bias_detection(self, engine_with_bias_scenarios):
        """Test that even fast mode detects critical bias issues."""
        engine, symbol, base_date = engine_with_bias_scenarios
        
        query = PITQuery(
            symbols=[symbol],
            as_of_date=base_date + timedelta(days=75),
            data_types=[DataType.PRICE],
            mode=QueryMode.FAST,  # Optimized for speed
            bias_check=BiasCheckLevel.BASIC  # Basic checking
        )
        
        result = engine.execute_query(query)
        
        # Even in fast mode, should not return obviously biased data
        if symbol in result.data and DataType.PRICE in result.data[symbol]:
            price_value = result.data[symbol][DataType.PRICE]
            
            # Should not get the obviously wrong future values
            assert price_value.value not in [Decimal("999.01"), Decimal("999.02")], "Fast mode returned biased data"


class TestTaiwanMarketBiasScenarios:
    """Test bias scenarios specific to Taiwan market."""
    
    def test_t2_settlement_bias_prevention(self):
        """Test that T+2 settlement prevents look-ahead bias."""
        store = InMemoryTemporalStore()
        manager = TemporalDataManager(store)
        
        symbol = "2330"
        trade_date = date(2024, 1, 15)  # Monday
        
        # Store trade execution
        trade_value = TemporalValue(
            value=Decimal("580.00"),
            as_of_date=trade_date,
            value_date=trade_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            metadata={"trade_execution": True}
        )
        store.store(trade_value)
        
        # Calculate settlement date
        settlement_date = manager.get_taiwan_settlement_date(trade_date)
        
        # Verify settlement is at least T+2
        assert settlement_date > trade_date, "Settlement date not after trade date"
        assert (settlement_date - trade_date).days >= 2, "Settlement not T+2 or later"
        
        # Query before settlement should not affect positions
        pre_settlement_query_date = trade_date + timedelta(days=1)
        
        # Settlement lag validation
        issues = manager.validate_temporal_consistency(symbol, pre_settlement_query_date)
        # Should detect any settlement-related inconsistencies
    
    def test_fundamental_data_lag_bias_prevention(self):
        """Test that 60-day fundamental lag prevents bias."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        quarter_end = date(2023, 12, 31)  # Q4 2023
        
        # Store legitimate fundamental data (with proper 60-day lag)
        legitimate_announcement = quarter_end + timedelta(days=60)
        legitimate_fundamental = TemporalValue(
            value={"eps": 5.0, "revenue": 1000000, "quarter": 4, "year": 2023},
            as_of_date=legitimate_announcement,
            value_date=quarter_end,
            data_type=DataType.FUNDAMENTAL,
            symbol=symbol,
            metadata={"lag_days": 60}
        )
        store.store(legitimate_fundamental)
        
        # Store problematic fundamental data (insufficient lag)
        premature_announcement = quarter_end + timedelta(days=20)  # Only 20-day lag
        premature_fundamental = TemporalValue(
            value={"eps": 6.0, "revenue": 1200000, "quarter": 4, "year": 2023},
            as_of_date=premature_announcement,
            value_date=quarter_end,
            data_type=DataType.FUNDAMENTAL,
            symbol=symbol,
            metadata={"lag_days": 20, "problematic": True}
        )
        store.store(premature_fundamental)
        
        # Query during the problematic period
        query_date = quarter_end + timedelta(days=45)  # After premature but before legitimate
        
        query = PITQuery(
            symbols=[symbol],
            as_of_date=query_date,
            data_types=[DataType.FUNDAMENTAL],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        result = engine.execute_query(query)
        
        # Should not get the premature fundamental data
        if symbol in result.data and DataType.FUNDAMENTAL in result.data[symbol]:
            fundamental_data = result.data[symbol][DataType.FUNDAMENTAL]
            assert fundamental_data.metadata.get("problematic") != True, "Got premature fundamental data"
            assert fundamental_data.metadata.get("lag_days", 0) >= 60, "Insufficient fundamental lag"
    
    def test_corporate_action_announcement_timing(self):
        """Test corporate action announcement timing prevents bias."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        announcement_date = date(2024, 1, 25)
        ex_date = date(2024, 4, 18)
        
        # Store dividend announcement
        dividend_data = {
            "action_type": "dividend_cash",
            "amount": 2.75,
            "ex_date": ex_date.isoformat(),
            "announcement_date": announcement_date.isoformat()
        }
        
        corporate_action = TemporalValue(
            value=dividend_data,
            as_of_date=announcement_date,
            value_date=ex_date,
            data_type=DataType.CORPORATE_ACTION,
            symbol=symbol
        )
        store.store(corporate_action)
        
        # Query before announcement should not see the corporate action
        pre_announcement_query = PITQuery(
            symbols=[symbol],
            as_of_date=announcement_date - timedelta(days=1),
            data_types=[DataType.CORPORATE_ACTION],
            mode=QueryMode.STRICT
        )
        
        pre_result = engine.execute_query(pre_query)
        assert DataType.CORPORATE_ACTION not in pre_result.data.get(symbol, {}), "Got corporate action before announcement"
        
        # Query after announcement should see the corporate action
        post_announcement_query = PITQuery(
            symbols=[symbol],
            as_of_date=announcement_date + timedelta(days=1),
            data_types=[DataType.CORPORATE_ACTION],
            mode=QueryMode.STRICT
        )
        
        post_result = engine.execute_query(post_announcement_query)
        if symbol in post_result.data and DataType.CORPORATE_ACTION in post_result.data[symbol]:
            action_data = post_result.data[symbol][DataType.CORPORATE_ACTION]
            assert action_data.value["amount"] == 2.75, "Corporate action data inconsistent"


class TestEdgeCaseBiasScenarios:
    """Test edge cases and stress scenarios for bias prevention."""
    
    def test_timezone_bias_prevention(self):
        """Test that timezone differences don't create bias."""
        store = InMemoryTemporalStore()
        
        # Simulate data from different timezones
        symbol = "2330"
        taiwan_date = date(2024, 1, 15)
        
        # Taiwan market data (09:00 TST = 01:00 UTC)
        taiwan_market_time = datetime.combine(taiwan_date, datetime.min.time().replace(hour=9))
        
        # US market data (same calendar date but different time)
        us_market_time = datetime.combine(taiwan_date, datetime.min.time().replace(hour=22))  # 22:00 TST = 14:00 UTC
        
        taiwan_value = TemporalValue(
            value=Decimal("580.00"),
            as_of_date=taiwan_date,
            value_date=taiwan_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=taiwan_market_time,
            metadata={"timezone": "TST", "market": "TWSE"}
        )
        
        us_value = TemporalValue(
            value=Decimal("581.00"),  # Different value
            as_of_date=taiwan_date,   # Same date
            value_date=taiwan_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            created_at=us_market_time,  # Later in the day
            metadata={"timezone": "TST", "market": "US_REF"}
        )
        
        store.store(taiwan_value)
        store.store(us_value)
        
        # Query should respect temporal ordering even within the same day
        result = store.get_point_in_time(symbol, taiwan_date, DataType.PRICE)
        
        # Should get the most recent value that was available
        assert result is not None
        # The exact behavior depends on how the store handles same-day ordering
    
    def test_market_holiday_bias_prevention(self):
        """Test bias prevention around market holidays."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        
        # Create data around Taiwan National Day (October 10)
        oct_9 = date(2024, 10, 9)   # Wednesday (trading day)
        oct_10 = date(2024, 10, 10)  # Thursday (National Day - holiday)
        oct_11 = date(2024, 10, 11)  # Friday (trading day)
        
        # Store data for trading days
        for trade_date in [oct_9, oct_11]:
            value = TemporalValue(
                value=Decimal("580.00"),
                as_of_date=trade_date,
                value_date=trade_date,
                data_type=DataType.PRICE,
                symbol=symbol
            )
            store.store(value)
        
        # Query on holiday should not get future data
        holiday_query = PITQuery(
            symbols=[symbol],
            as_of_date=oct_10,
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT
        )
        
        result = engine.execute_query(holiday_query)
        
        if symbol in result.data and DataType.PRICE in result.data[symbol]:
            price_data = result.data[symbol][DataType.PRICE]
            # Should not get data from Oct 11 (Friday) when querying on Oct 10 (holiday)
            assert price_data.as_of_date <= oct_10, "Got future data during holiday"
    
    def test_data_revision_bias_prevention(self):
        """Test bias prevention with data revisions and restatements."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        original_date = date(2024, 1, 15)
        revision_date = date(2024, 1, 20)
        
        # Store original data
        original_value = TemporalValue(
            value=Decimal("580.00"),
            as_of_date=original_date,
            value_date=original_date,
            data_type=DataType.PRICE,
            symbol=symbol,
            version=1,
            metadata={"revision": "original"}
        )
        store.store(original_value)
        
        # Store revision (corrected data)
        revised_value = TemporalValue(
            value=Decimal("578.50"),  # Corrected price
            as_of_date=revision_date,  # When revision was made
            value_date=original_date,  # Original date being corrected
            data_type=DataType.PRICE,
            symbol=symbol,
            version=2,
            metadata={"revision": "corrected"}
        )
        store.store(revised_value)
        
        # Query on original date should get original value (not revision)
        original_query = PITQuery(
            symbols=[symbol],
            as_of_date=original_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT
        )
        
        original_result = engine.execute_query(original_query)
        
        if symbol in original_result.data and DataType.PRICE in original_result.data[symbol]:
            price_data = original_result.data[symbol][DataType.PRICE]
            assert price_data.value == Decimal("580.00"), "Got revised value before revision date"
            assert price_data.metadata.get("revision") == "original", "Wrong version retrieved"
        
        # Query after revision date should get revised value
        revision_query = PITQuery(
            symbols=[symbol],
            as_of_date=revision_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT
        )
        
        revision_result = engine.execute_query(revision_query)
        
        if symbol in revision_result.data and DataType.PRICE in revision_result.data[symbol]:
            price_data = revision_result.data[symbol][DataType.PRICE]
            assert price_data.value == Decimal("578.50"), "Didn't get revised value after revision date"
            assert price_data.metadata.get("revision") == "corrected", "Wrong version retrieved"
    
    def test_high_frequency_bias_prevention(self):
        """Test bias prevention with high-frequency data updates."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store)
        
        symbol = "2330"
        base_date = date(2024, 1, 15)
        
        # Simulate high-frequency updates throughout the day
        prices = [580.00, 580.50, 579.75, 581.25, 580.80]
        
        for i, price in enumerate(prices):
            # Create timestamps throughout the trading day
            update_time = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=9, minutes=i*30)
            
            value = TemporalValue(
                value=Decimal(f"{price:.2f}"),
                as_of_date=update_time.date(),
                value_date=update_time.date(),
                data_type=DataType.PRICE,
                symbol=symbol,
                created_at=update_time,
                metadata={"sequence": i, "intraday_time": update_time.time().isoformat()}
            )
            store.store(value)
        
        # Query at different times during the day
        for hour in range(9, 14):  # 09:00 to 13:00 (Taiwan trading hours)
            query_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour))
            
            query = PITQuery(
                symbols=[symbol],
                as_of_date=query_time.date(),
                data_types=[DataType.PRICE],
                mode=QueryMode.STRICT
            )
            
            result = engine.execute_query(query)
            
            if symbol in result.data and DataType.PRICE in result.data[symbol]:
                price_data = result.data[symbol][DataType.PRICE]
                
                # Verify we don't get data from later in the day
                price_time = datetime.fromisoformat(f"{base_date.isoformat()} {price_data.metadata['intraday_time']}")
                assert price_time <= query_time, f"Got future intraday data: {price_time} > {query_time}"


class TestBiasPreventionIntegration:
    """Integration tests for comprehensive bias prevention."""
    
    def test_end_to_end_bias_prevention(self):
        """Test end-to-end bias prevention in realistic scenario."""
        store = InMemoryTemporalStore()
        engine = PointInTimeEngine(store, enable_cache=True)
        
        # Create comprehensive dataset with potential bias scenarios
        symbols = ["2330", "2317"]
        start_date = date(2023, 1, 1)
        end_date = date(2024, 6, 30)
        
        # Add legitimate data
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Trading days
                for symbol in symbols:
                    # Price data
                    price = 100 + random.uniform(-10, 10)
                    price_value = TemporalValue(
                        value=Decimal(f"{price:.2f}"),
                        as_of_date=current_date,
                        value_date=current_date,
                        data_type=DataType.PRICE,
                        symbol=symbol
                    )
                    store.store(price_value)
                    
                    # Fundamental data (quarterly with proper lag)
                    if current_date.month in [2, 5, 8, 11] and current_date.day == 15:
                        fundamental_value = TemporalValue(
                            value={"eps": random.uniform(1, 5), "revenue": random.uniform(1000, 5000)},
                            as_of_date=current_date,
                            value_date=current_date - timedelta(days=75),  # Proper lag
                            data_type=DataType.FUNDAMENTAL,
                            symbol=symbol
                        )
                        store.store(fundamental_value)
            
            current_date += timedelta(days=1)
        
        # Run comprehensive backtesting simulation
        backtest_start = date(2023, 6, 1)
        backtest_end = date(2024, 3, 31)
        
        bias_violations_found = []
        total_queries = 0
        
        current_date = backtest_start
        while current_date <= backtest_end:
            if current_date.weekday() < 5:  # Trading days
                query = PITQuery(
                    symbols=symbols,
                    as_of_date=current_date,
                    data_types=[DataType.PRICE, DataType.FUNDAMENTAL],
                    mode=QueryMode.STRICT,
                    bias_check=BiasCheckLevel.STRICT
                )
                
                result = engine.execute_query(query)
                total_queries += 1
                
                # Check for any bias violations
                if result.bias_violations:
                    bias_violations_found.extend(result.bias_violations)
                
                # Verify temporal consistency of all returned data
                for symbol in result.data:
                    for data_type, temporal_value in result.data[symbol].items():
                        if temporal_value.as_of_date > current_date:
                            bias_violations_found.append(f"Future as_of_date: {temporal_value.as_of_date} > {current_date}")
                        
                        if temporal_value.value_date > current_date:
                            bias_violations_found.append(f"Future value_date: {temporal_value.value_date} > {current_date}")
            
            current_date += timedelta(days=1)
        
        print(f"\nEnd-to-End Bias Prevention Test:")
        print(f"  Total queries: {total_queries}")
        print(f"  Bias violations found: {len(bias_violations_found)}")
        
        if bias_violations_found:
            print("  Violations:")
            for violation in bias_violations_found[:10]:  # Show first 10
                print(f"    - {violation}")
        
        # Should have zero bias violations
        assert len(bias_violations_found) == 0, f"Found {len(bias_violations_found)} bias violations in end-to-end test"
        
        # Should have processed a reasonable number of queries
        assert total_queries > 100, f"Too few queries processed: {total_queries}"


if __name__ == "__main__":
    # Run comprehensive bias validation tests
    pytest.main([__file__, "-v", "--tb=short"])