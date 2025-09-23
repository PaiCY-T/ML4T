"""
Full Data Pipeline Integration Tests.

This module tests the complete data pipeline from ingestion to API access,
ensuring all components work together correctly with temporal consistency.
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from data.core.temporal import (
    TemporalValue, InMemoryTemporalStore, DataType, TemporalDataManager
)
from data.models.taiwan_market import (
    TaiwanMarketData, TaiwanSettlement, TradingStatus,
    create_taiwan_trading_calendar
)
from data.pipeline.pit_engine import (
    PointInTimeEngine, PITQuery, QueryMode, BiasCheckLevel
)
from data.pipeline.incremental_updater import (
    IncrementalUpdater, UpdateRequest, UpdateMode, UpdatePriority
)
from data.ingestion.finlab_connector import (
    FinLabConnector, FinLabConfig, FinLabFieldMapping
)
from data.api.pit_data_service import PITDataService, PITQueryRequest, DataTypeEnum
from data.quality.validators import (
    create_standard_quality_monitor, QualityMonitor, SeverityLevel
)


class MockFinLabConnector:
    """Mock FinLab connector for testing."""
    
    def __init__(self, temporal_store):
        self.temporal_store = temporal_store
        self.field_mapping = FinLabFieldMapping.get_all_fields()
        self.query_count = 0
        self.connected = False
        
        # Generate mock data
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate mock Taiwan stock data."""
        symbols = ["2330", "2317", "2454"]  # TSMC, Hon Hai, MediaTek
        base_date = date(2024, 9, 1)
        
        self.mock_data = {}
        
        for i, symbol in enumerate(symbols):
            self.mock_data[symbol] = []
            base_price = Decimal(str(100 + i * 50))  # Different base prices
            
            for days in range(30):  # 30 days of data
                current_date = base_date + timedelta(days=days)
                
                # Generate realistic price movement
                price_change = Decimal(str((days % 7 - 3) * 2))  # Simple pattern
                current_price = base_price + price_change
                
                market_data = TaiwanMarketData(
                    symbol=symbol,
                    data_date=current_date,
                    as_of_date=current_date,
                    open_price=current_price - Decimal("1"),
                    high_price=current_price + Decimal("2"),
                    low_price=current_price - Decimal("2"),
                    close_price=current_price,
                    volume=1000000 + (days * 50000),
                    turnover=current_price * Decimal(str(1000000)),
                    trading_status=TradingStatus.NORMAL
                )
                
                self.mock_data[symbol].append(market_data)
    
    def connect(self):
        self.connected = True
    
    def disconnect(self):
        self.connected = False
    
    def get_available_symbols(self, as_of_date=None):
        return list(self.mock_data.keys())
    
    def get_price_data(self, symbol, start_date, end_date, fields=None):
        """Get mock price data."""
        self.query_count += 1
        
        if symbol not in self.mock_data:
            return []
        
        temporal_values = []
        for market_data in self.mock_data[symbol]:
            if start_date <= market_data.data_date <= end_date:
                temporal_values.extend(market_data.to_temporal_values())
        
        return temporal_values
    
    def get_fundamental_data(self, symbol, start_date, end_date, fields=None):
        """Get mock fundamental data."""
        self.query_count += 1
        return []  # Simplified for this test
    
    def get_corporate_actions(self, symbol, start_date, end_date):
        """Get mock corporate actions."""
        self.query_count += 1
        return []  # Simplified for this test
    
    def validate_data_quality(self, symbol, data_date, data):
        """Mock data quality validation."""
        return []  # No issues for mock data
    
    def get_performance_stats(self):
        return {
            "query_count": self.query_count,
            "cache_hits": 0,
            "cache_hit_rate": 0.0,
            "avg_query_time_seconds": 0.001,
            "total_query_time_seconds": self.query_count * 0.001,
            "mapped_fields_count": len(self.field_mapping),
            "cache_enabled": False
        }


@pytest.fixture
def temporal_store():
    """Create temporal store for testing."""
    return InMemoryTemporalStore()


@pytest.fixture
def mock_finlab_connector(temporal_store):
    """Create mock FinLab connector."""
    return MockFinLabConnector(temporal_store)


@pytest.fixture
def pit_engine(temporal_store):
    """Create PIT engine for testing."""
    trading_calendar = create_taiwan_trading_calendar(2024)
    return PointInTimeEngine(temporal_store, trading_calendar)


@pytest.fixture
def incremental_updater(temporal_store, mock_finlab_connector):
    """Create incremental updater for testing."""
    return IncrementalUpdater(
        temporal_store=temporal_store,
        finlab_connector=mock_finlab_connector,
        enable_queue=False  # Synchronous for testing
    )


@pytest.fixture
def quality_monitor():
    """Create quality monitor for testing."""
    return create_standard_quality_monitor(enable_anomaly_detection=False)


@pytest.fixture
def pit_data_service(temporal_store, pit_engine, incremental_updater, mock_finlab_connector):
    """Create PIT data service for testing."""
    return PITDataService(
        temporal_store=temporal_store,
        pit_engine=pit_engine,
        incremental_updater=incremental_updater,
        finlab_connector=mock_finlab_connector,
        enable_background_updates=False
    )


class TestFullPipelineIntegration:
    """Test complete data pipeline integration."""
    
    def test_end_to_end_data_flow(self, temporal_store, mock_finlab_connector, 
                                 pit_engine, incremental_updater, quality_monitor):
        """Test complete data flow from ingestion to query."""
        
        # Step 1: Connect to data source
        mock_finlab_connector.connect()
        assert mock_finlab_connector.connected
        
        # Step 2: Get available symbols
        symbols = mock_finlab_connector.get_available_symbols()
        assert len(symbols) > 0
        assert "2330" in symbols  # TSMC should be available
        
        # Step 3: Create update request
        update_request = UpdateRequest(
            symbols=["2330"],
            data_types=[DataType.PRICE, DataType.VOLUME],
            start_date=date(2024, 9, 1),
            end_date=date(2024, 9, 10),
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.HIGH,
            validate_consistency=True
        )
        
        # Step 4: Execute incremental update
        result = incremental_updater.execute_update(update_request)
        assert result.success
        assert result.new_count > 0
        
        # Step 5: Validate data was stored
        stored_value = temporal_store.get_point_in_time(
            "2330", date(2024, 9, 5), DataType.PRICE
        )
        assert stored_value is not None
        assert stored_value.symbol == "2330"
        
        # Step 6: Execute point-in-time query
        pit_query = PITQuery(
            symbols=["2330"],
            as_of_date=date(2024, 9, 5),
            data_types=[DataType.PRICE, DataType.VOLUME],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        pit_result = pit_engine.execute_query(pit_query)
        assert pit_result.data
        assert "2330" in pit_result.data
        assert DataType.PRICE in pit_result.data["2330"]
        
        # Step 7: Validate temporal consistency
        price_value = pit_result.data["2330"][DataType.PRICE]
        assert price_value.as_of_date <= date(2024, 9, 5)
        assert price_value.value_date <= date(2024, 9, 5)
        
        # Step 8: Quality validation
        quality_issues = quality_monitor.validate_value(price_value)
        # Should have no critical issues for well-formed mock data
        critical_issues = [i for i in quality_issues if i.severity == SeverityLevel.CRITICAL]
        assert len(critical_issues) == 0
        
        # Step 9: Cleanup
        mock_finlab_connector.disconnect()
        assert not mock_finlab_connector.connected
    
    def test_bias_prevention(self, temporal_store, mock_finlab_connector, pit_engine):
        """Test look-ahead bias prevention."""
        
        mock_finlab_connector.connect()
        
        # Store some future data (should be prevented from access)
        future_value = TemporalValue(
            value=Decimal("100.00"),
            as_of_date=date(2024, 9, 10),  # Data known on 9/10
            value_date=date(2024, 9, 10),  # Data for 9/10
            data_type=DataType.PRICE,
            symbol="2330"
        )
        temporal_store.store(future_value)
        
        # Try to query this data from an earlier date (should not be accessible)
        pit_query = PITQuery(
            symbols=["2330"],
            as_of_date=date(2024, 9, 5),  # Query as of 9/5
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        result = pit_engine.execute_query(pit_query)
        
        # Should not get the future data
        if "2330" in result.data and DataType.PRICE in result.data["2330"]:
            retrieved_value = result.data["2330"][DataType.PRICE]
            assert retrieved_value.as_of_date <= date(2024, 9, 5)
        
        mock_finlab_connector.disconnect()
    
    def test_settlement_lag_handling(self, temporal_store, pit_engine):
        """Test Taiwan T+2 settlement lag handling."""
        
        # Create trade data with proper settlement dates
        trade_date = date(2024, 9, 20)  # Friday
        settlement_date = date(2024, 9, 24)  # Following Tuesday (T+2)
        
        # Store price data available immediately
        price_value = TemporalValue(
            value=Decimal("100.00"),
            as_of_date=trade_date,
            value_date=trade_date,
            data_type=DataType.PRICE,
            symbol="2330"
        )
        temporal_store.store(price_value)
        
        # Query should work for same day
        pit_query = PITQuery(
            symbols=["2330"],
            as_of_date=trade_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT
        )
        
        result = pit_engine.execute_query(pit_query)
        assert "2330" in result.data
        assert DataType.PRICE in result.data["2330"]
    
    def test_data_quality_monitoring(self, quality_monitor, temporal_store):
        """Test data quality monitoring integration."""
        
        # Create test values with different quality characteristics
        good_value = TemporalValue(
            value=Decimal("100.00"),
            as_of_date=date(2024, 9, 5),
            value_date=date(2024, 9, 5),
            data_type=DataType.PRICE,
            symbol="2330"
        )
        
        bad_value = TemporalValue(
            value=Decimal("-10.00"),  # Negative price (invalid)
            as_of_date=date(2024, 9, 6),
            value_date=date(2024, 9, 6),
            data_type=DataType.PRICE,
            symbol="2330"
        )
        
        # Validate good value
        good_issues = quality_monitor.validate_value(good_value)
        assert len(good_issues) == 0
        
        # Validate bad value
        bad_issues = quality_monitor.validate_value(bad_value)
        assert len(bad_issues) > 0
        assert any(issue.severity == SeverityLevel.ERROR for issue in bad_issues)
    
    @pytest.mark.asyncio
    async def test_api_integration(self, pit_data_service):
        """Test API service integration."""
        
        # First populate some data
        symbols = ["2330"]
        update_request = UpdateRequest(
            symbols=symbols,
            data_types=[DataType.PRICE, DataType.VOLUME],
            start_date=date(2024, 9, 1),
            end_date=date(2024, 9, 5),
            mode=UpdateMode.INCREMENTAL
        )
        
        # Execute update
        result = pit_data_service.incremental_updater.execute_update(update_request)
        assert result.success
        
        # Test API query
        api_request = PITQueryRequest(
            symbols=["2330"],
            as_of_date=date(2024, 9, 5),
            data_types=[DataTypeEnum.PRICE, DataTypeEnum.VOLUME],
            mode="strict",
            bias_check="strict"
        )
        
        api_response = await pit_data_service.execute_pit_query(api_request)
        assert api_response.success
        assert "2330" in api_response.data
        
        # Test health check
        health = await pit_data_service.get_health()
        assert health.status == "healthy"
        assert health.database_status in ["connected", "not_configured"]
    
    def test_performance_requirements(self, temporal_store, pit_engine):
        """Test performance requirements are met."""
        
        # Store test data
        symbols = ["2330", "2317", "2454"]
        test_date = date(2024, 9, 5)
        
        for symbol in symbols:
            for i in range(100):  # 100 data points per symbol
                value = TemporalValue(
                    value=Decimal(str(100 + i)),
                    as_of_date=test_date,
                    value_date=test_date,
                    data_type=DataType.PRICE,
                    symbol=symbol
                )
                temporal_store.store(value)
        
        # Test query performance (should be < 100ms for single query)
        start_time = datetime.utcnow()
        
        pit_query = PITQuery(
            symbols=symbols,
            as_of_date=test_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.FAST  # Use fast mode for performance
        )
        
        result = pit_engine.execute_query(pit_query)
        
        end_time = datetime.utcnow()
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        assert result.data
        assert len(result.data) == len(symbols)
        assert execution_time_ms < 100  # Should be under 100ms
    
    def test_concurrent_access(self, temporal_store, pit_engine):
        """Test concurrent access to the system."""
        import threading
        import time
        
        # Store initial data
        test_value = TemporalValue(
            value=Decimal("100.00"),
            as_of_date=date(2024, 9, 5),
            value_date=date(2024, 9, 5),
            data_type=DataType.PRICE,
            symbol="2330"
        )
        temporal_store.store(test_value)
        
        results = []
        errors = []
        
        def query_worker():
            """Worker function for concurrent queries."""
            try:
                query = PITQuery(
                    symbols=["2330"],
                    as_of_date=date(2024, 9, 5),
                    data_types=[DataType.PRICE],
                    mode=QueryMode.FAST
                )
                result = pit_engine.execute_query(query)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Launch multiple concurrent queries
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=query_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all queries succeeded
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10
        
        # Verify all results are consistent
        for result in results:
            assert result.data
            assert "2330" in result.data
    
    def test_memory_usage(self, temporal_store):
        """Test memory usage stays reasonable with large datasets."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Store a large amount of data
        symbols = [f"{2330 + i}" for i in range(100)]  # 100 symbols
        base_date = date(2024, 9, 1)
        
        for symbol in symbols:
            for days in range(30):  # 30 days per symbol
                test_date = base_date + timedelta(days=days)
                value = TemporalValue(
                    value=Decimal(str(100 + days)),
                    as_of_date=test_date,
                    value_date=test_date,
                    data_type=DataType.PRICE,
                    symbol=symbol
                )
                temporal_store.store(value)
        
        # Check memory usage after storing data
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB for 3000 data points)
        assert memory_increase < 50, f"Memory usage increased by {memory_increase:.1f} MB"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])