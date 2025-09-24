"""
Integration tests for Stream B - Data Integration & APIs.

Tests the complete pipeline from FinLab connector through incremental updater 
to the point-in-time data API service.
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import json

# Test framework imports
from fastapi.testclient import TestClient
import uvicorn

# Stream B components
from src.data.core.temporal import (
    InMemoryTemporalStore, TemporalValue, DataType, PostgreSQLTemporalStore
)
from src.data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
from src.data.pipeline.incremental_updater import (
    IncrementalUpdater, UpdateRequest, UpdateMode, UpdatePriority
)
from src.data.api.pit_data_service import PITDataService, create_api_app
from src.data.pipeline.pit_engine import PointInTimeEngine
from src.data.models.taiwan_market import create_taiwan_trading_calendar


@pytest.fixture
def mock_finlab_config():
    """Mock FinLab configuration for testing."""
    return FinLabConfig(
        host="localhost",
        port=5432,
        database="test_finlab",
        username="test_user",
        password="test_pass"
    )


@pytest.fixture
def temporal_store():
    """In-memory temporal store for testing."""
    return InMemoryTemporalStore()


@pytest.fixture
def sample_temporal_values():
    """Sample temporal values for testing."""
    base_date = date(2024, 1, 1)
    
    values = []
    for i in range(10):
        value_date = base_date + timedelta(days=i)
        
        # Price data
        values.append(TemporalValue(
            value=100.0 + i,
            as_of_date=value_date,
            value_date=value_date,
            data_type=DataType.PRICE,
            symbol="2330",
            metadata={"field": "close_price", "source": "finlab"}
        ))
        
        # Volume data
        values.append(TemporalValue(
            value=1000000 + i * 10000,
            as_of_date=value_date,
            value_date=value_date,
            data_type=DataType.VOLUME,
            symbol="2330",
            metadata={"field": "volume", "source": "finlab"}
        ))
    
    return values


@pytest.fixture
def mock_finlab_connector(temporal_store, sample_temporal_values):
    """Mock FinLab connector with sample data."""
    connector = Mock(spec=FinLabConnector)
    
    # Mock methods
    connector.get_available_symbols.return_value = ["2330", "2317", "2454"]
    connector.get_symbol_info.return_value = {
        "symbol": "2330",
        "company_name": "Taiwan Semiconductor Manufacturing Co",
        "industry": "Semiconductors",
        "sector": "Technology",
        "listing_date": date(1994, 9, 5),
        "market_type": "TWSE",
        "outstanding_shares": 25930380458
    }
    
    def mock_get_price_data(symbol, start_date, end_date):
        return [v for v in sample_temporal_values 
                if v.symbol == symbol and v.data_type == DataType.PRICE
                and start_date <= v.value_date <= end_date]
    
    def mock_get_fundamental_data(symbol, start_date, end_date):
        # Return empty for this test
        return []
    
    def mock_get_corporate_actions(symbol, start_date, end_date):
        # Return empty for this test
        return []
    
    connector.get_price_data.side_effect = mock_get_price_data
    connector.get_fundamental_data.side_effect = mock_get_fundamental_data
    connector.get_corporate_actions.side_effect = mock_get_corporate_actions
    
    # Mock validation methods
    connector.validate_data_quality.return_value = []
    connector.validate_data_completeness.return_value = {
        "symbol": "2330",
        "date_range": (date(2024, 1, 1), date(2024, 1, 10)),
        "total_trading_days": 8,
        "missing_data_days": [],
        "data_type_coverage": {
            "price": {"total_values": 10, "coverage_rate": 1.0, "missing_dates": []}
        },
        "quality_issues": []
    }
    
    connector.get_data_quality_metrics.return_value = {
        "symbol": "2330",
        "analysis_period": (date(2024, 1, 1), date(2024, 1, 31)),
        "price_data_metrics": {
            "total_records": 20,
            "non_null_records": 20,
            "null_rate": 0.0,
            "outliers_detected": 0,
            "volatility_spike_days": 0
        },
        "fundamental_data_metrics": {},
        "overall_quality_score": 95.0,
        "issues_found": [],
        "recommendations": []
    }
    
    connector.get_performance_stats.return_value = {
        "query_count": 10,
        "cache_hits": 5,
        "cache_hit_rate": 0.5,
        "avg_query_time_seconds": 0.1
    }
    
    return connector


class TestStreamBPipeline:
    """Integration tests for the complete Stream B pipeline."""
    
    def test_finlab_connector_integration(self, temporal_store, mock_finlab_connector, sample_temporal_values):
        """Test FinLab connector integration with temporal store."""
        # Store sample data in temporal store
        for value in sample_temporal_values:
            temporal_store.store(value)
        
        # Test data retrieval
        symbols = mock_finlab_connector.get_available_symbols()
        assert "2330" in symbols
        
        # Test price data retrieval
        price_data = mock_finlab_connector.get_price_data(
            "2330", date(2024, 1, 1), date(2024, 1, 5)
        )
        assert len(price_data) == 5
        assert all(v.data_type == DataType.PRICE for v in price_data)
        
        # Test symbol info
        symbol_info = mock_finlab_connector.get_symbol_info("2330")
        assert symbol_info["symbol"] == "2330"
        assert "Taiwan Semiconductor" in symbol_info["company_name"]
    
    def test_incremental_updater_integration(self, temporal_store, mock_finlab_connector):
        """Test incremental updater with FinLab connector."""
        # Create incremental updater
        trading_calendar = create_taiwan_trading_calendar(2024)
        updater = IncrementalUpdater(
            temporal_store=temporal_store,
            finlab_connector=mock_finlab_connector,
            trading_calendar=trading_calendar,
            max_workers=2
        )
        
        # Create update request
        request = UpdateRequest(
            symbols=["2330"],
            data_types=[DataType.PRICE, DataType.VOLUME],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.HIGH
        )
        
        # Execute update
        result = updater.execute_update(request)
        
        # Verify results
        assert result.success
        assert result.processed_count == 1
        assert result.new_count > 0
        assert len(result.errors) == 0
        
        # Verify data was stored
        stored_value = temporal_store.get_point_in_time("2330", date(2024, 1, 1), DataType.PRICE)
        assert stored_value is not None
        assert stored_value.symbol == "2330"
    
    def test_pit_engine_integration(self, temporal_store, sample_temporal_values):
        """Test point-in-time engine with stored data."""
        # Store sample data
        for value in sample_temporal_values:
            temporal_store.store(value)
        
        # Create PIT engine
        trading_calendar = create_taiwan_trading_calendar(2024)
        pit_engine = PointInTimeEngine(
            store=temporal_store,
            trading_calendar=trading_calendar
        )
        
        # Create and execute query
        from src.data.pipeline.pit_engine import PITQuery, QueryMode, BiasCheckLevel
        
        query = PITQuery(
            symbols=["2330"],
            as_of_date=date(2024, 1, 5),
            data_types=[DataType.PRICE, DataType.VOLUME],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        result = pit_engine.execute_query(query)
        
        # Verify results
        assert result.data["2330"][DataType.PRICE] is not None
        assert result.data["2330"][DataType.VOLUME] is not None
        assert result.execution_time_ms > 0
        assert len(result.bias_violations) == 0
    
    @pytest.mark.asyncio
    async def test_api_service_integration(self, temporal_store, mock_finlab_connector, sample_temporal_values):
        """Test complete API service integration."""
        # Store sample data
        for value in sample_temporal_values:
            temporal_store.store(value)
        
        # Create PIT engine
        trading_calendar = create_taiwan_trading_calendar(2024)
        pit_engine = PointInTimeEngine(
            store=temporal_store,
            trading_calendar=trading_calendar
        )
        
        # Create incremental updater
        incremental_updater = IncrementalUpdater(
            temporal_store=temporal_store,
            finlab_connector=mock_finlab_connector,
            trading_calendar=trading_calendar
        )
        
        # Create API service
        service = PITDataService(
            temporal_store=temporal_store,
            pit_engine=pit_engine,
            incremental_updater=incremental_updater,
            finlab_connector=mock_finlab_connector
        )
        
        # Test PIT query
        from src.data.api.pit_data_service import PITQueryRequest, DataTypeEnum, QueryModeEnum
        
        query_request = PITQueryRequest(
            symbols=["2330"],
            as_of_date=date(2024, 1, 5),
            data_types=[DataTypeEnum.PRICE, DataTypeEnum.VOLUME],
            mode=QueryModeEnum.STRICT
        )
        
        response = await service.execute_pit_query(query_request)
        
        # Verify response
        assert response.success
        assert "2330" in response.data
        assert DataTypeEnum.PRICE in response.data["2330"]
        assert response.execution_time_ms > 0
        
        # Test data quality metrics
        metrics_response = await service.get_data_quality_metrics("2330", 30)
        assert metrics_response.symbol == "2330"
        assert metrics_response.overall_quality_score >= 90.0
        
        # Test symbol search
        from src.data.api.pit_data_service import SymbolSearchRequest
        
        search_request = SymbolSearchRequest(query="2330")
        search_results = await service.search_symbols(search_request)
        assert len(search_results) > 0
        assert search_results[0].symbol == "2330"
    
    def test_api_endpoints_integration(self, temporal_store, mock_finlab_connector, sample_temporal_values):
        """Test FastAPI endpoints integration."""
        # Store sample data
        for value in sample_temporal_values:
            temporal_store.store(value)
        
        # Create complete service stack
        trading_calendar = create_taiwan_trading_calendar(2024)
        pit_engine = PointInTimeEngine(store=temporal_store, trading_calendar=trading_calendar)
        
        incremental_updater = IncrementalUpdater(
            temporal_store=temporal_store,
            finlab_connector=mock_finlab_connector,
            trading_calendar=trading_calendar
        )
        
        service = PITDataService(
            temporal_store=temporal_store,
            pit_engine=pit_engine,
            incremental_updater=incremental_updater,
            finlab_connector=mock_finlab_connector
        )
        
        # Create FastAPI app
        app = create_api_app(service)
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # Test PIT data endpoint
        pit_request = {
            "symbols": ["2330"],
            "as_of_date": "2024-01-05",
            "data_types": ["price", "volume"],
            "mode": "strict"
        }
        
        response = client.post("/data/pit", json=pit_request)
        assert response.status_code == 200
        pit_data = response.json()
        assert pit_data["success"]
        assert "2330" in pit_data["data"]
        
        # Test trading calendar endpoint
        response = client.get("/data/calendar/trading/2024-01-01/2024-01-05")
        assert response.status_code == 200
        calendar_data = response.json()
        assert len(calendar_data) == 5
        
        # Test available symbols endpoint
        response = client.get("/symbols?limit=10")
        assert response.status_code == 200
        symbols = response.json()
        assert "2330" in symbols
        
        # Test statistics endpoint
        response = client.get("/admin/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "service" in stats
        assert "pit_engine" in stats
    
    def test_error_handling_integration(self, temporal_store):
        """Test error handling across the pipeline."""
        # Create service with no connector (should handle gracefully)
        trading_calendar = create_taiwan_trading_calendar(2024)
        pit_engine = PointInTimeEngine(store=temporal_store, trading_calendar=trading_calendar)
        
        service = PITDataService(
            temporal_store=temporal_store,
            pit_engine=pit_engine,
            incremental_updater=None,
            finlab_connector=None
        )
        
        app = create_api_app(service)
        client = TestClient(app)
        
        # Test endpoints that require connector (should return 503)
        response = client.get("/symbols")
        assert response.status_code == 503
        
        response = client.get("/symbols/2330")
        assert response.status_code == 503
        
        # Test endpoints that should still work
        response = client.get("/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["database_status"] == "not_configured"
    
    def test_performance_optimization(self, temporal_store, mock_finlab_connector):
        """Test performance optimizations in the pipeline."""
        # Create large dataset for performance testing
        large_values = []
        for symbol_idx in range(10):  # 10 symbols
            symbol = f"23{symbol_idx:02d}"
            for day_idx in range(100):  # 100 days each
                value_date = date(2024, 1, 1) + timedelta(days=day_idx)
                
                large_values.append(TemporalValue(
                    value=100.0 + day_idx,
                    as_of_date=value_date,
                    value_date=value_date,
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    metadata={"field": "close_price", "source": "finlab"}
                ))
        
        # Store large dataset
        for value in large_values:
            temporal_store.store(value)
        
        # Test bulk operations
        if hasattr(temporal_store, 'bulk_store'):
            # Test bulk storage performance
            start_time = datetime.utcnow()
            temporal_store.bulk_store(large_values[:100])
            bulk_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Should be faster than individual operations
            assert bulk_time < 1.0  # Should complete within 1 second
        
        # Test batch processing in incremental updater
        trading_calendar = create_taiwan_trading_calendar(2024)
        updater = IncrementalUpdater(
            temporal_store=temporal_store,
            finlab_connector=mock_finlab_connector,
            trading_calendar=trading_calendar,
            max_workers=4
        )
        
        # Create large update request
        symbols = [f"23{i:02d}" for i in range(25)]  # Should trigger batch processing
        request = UpdateRequest(
            symbols=symbols,
            data_types=[DataType.PRICE],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.HIGH
        )
        
        start_time = datetime.utcnow()
        result = updater.execute_update(request)
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Should complete efficiently
        assert result.success
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert result.processed_count == 25


class TestStreamBDataQuality:
    """Test data quality features in Stream B."""
    
    def test_data_validation_pipeline(self, temporal_store, mock_finlab_connector):
        """Test data validation throughout the pipeline."""
        # Create updater with validation enabled
        trading_calendar = create_taiwan_trading_calendar(2024)
        updater = IncrementalUpdater(
            temporal_store=temporal_store,
            finlab_connector=mock_finlab_connector,
            trading_calendar=trading_calendar
        )
        
        # Test with invalid data (this would be caught by validation)
        invalid_value = TemporalValue(
            value=-100.0,  # Invalid negative price
            as_of_date=date(2024, 1, 1),
            value_date=date(2024, 1, 1),
            data_type=DataType.PRICE,
            symbol="TEST",
            metadata={"field": "close_price"}
        )
        
        # Test validation catches issues
        issues = updater._validate_data_quality(invalid_value)
        assert len(issues) > 0
        assert any("Negative price" in issue for issue in issues)
    
    def test_temporal_consistency_validation(self, temporal_store):
        """Test temporal consistency validation."""
        # Create values with temporal issues
        future_value = TemporalValue(
            value=100.0,
            as_of_date=date.today() + timedelta(days=1),  # Future date
            value_date=date.today(),
            data_type=DataType.PRICE,
            symbol="TEST",
            metadata={"field": "close_price"}
        )
        
        trading_calendar = create_taiwan_trading_calendar(2024)
        updater = IncrementalUpdater(
            temporal_store=temporal_store,
            finlab_connector=Mock(),
            trading_calendar=trading_calendar
        )
        
        # Test temporal consistency check
        issues = updater._enhanced_consistency_check(future_value)
        assert len(issues) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])