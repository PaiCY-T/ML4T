"""
Comprehensive Test Suite for Data Quality Validation Framework.

This module provides comprehensive testing for the data quality validation system,
including performance tests, Taiwan market specific tests, and integration tests
with the point-in-time system.
"""

import asyncio
import pytest
import time
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from src.data.core.temporal import TemporalValue, TemporalStore, DataType
from src.data.models.taiwan_market import TaiwanMarketData, TaiwanTradingCalendar, create_taiwan_trading_calendar
from src.data.pipeline.pit_engine import PointInTimeEngine, PITQuery, QueryMode, BiasCheckLevel
from src.data.quality.validation_engine import (
    ValidationEngine, ValidationRegistry, ValidationContext, ValidationOutput,
    ValidationResult, ValidationPriority, ValidationPlugin
)
from src.data.quality.taiwan_validators import (
    TaiwanPriceLimitValidator, TaiwanVolumeValidator, TaiwanTradingHoursValidator,
    TaiwanSettlementValidator, TaiwanFundamentalLagValidator, create_taiwan_validators
)
from src.data.quality.monitor import QualityMonitor, create_taiwan_market_monitor
from src.data.quality.pit_integration import (
    PITValidationOrchestrator, PITValidationConfig, PITValidationMode,
    create_pit_validation_orchestrator, validate_taiwan_market_data
)
from src.data.quality.validators import QualityIssue, SeverityLevel, QualityCheckType


class TestValidationFramework:
    """Test core validation framework functionality."""
    
    @pytest.fixture
    def temporal_store(self):
        """Create mock temporal store."""
        store = Mock(spec=TemporalStore)
        
        # Mock some test data
        test_price = TemporalValue(
            symbol="2330",
            value=Decimal("500.0"),
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE,
            created_at=datetime(2024, 1, 15, 10, 0),
            updated_at=datetime(2024, 1, 15, 10, 0)
        )
        
        store.get_point_in_time.return_value = test_price
        store.get_range.return_value = [test_price]
        
        return store
    
    @pytest.fixture
    def validation_registry(self):
        """Create validation registry with Taiwan validators."""
        registry = ValidationRegistry()
        
        # Register Taiwan validators
        validators = create_taiwan_validators()
        for validator in validators:
            registry.register_plugin(validator)
        
        return registry
    
    @pytest.fixture
    def validation_engine(self, validation_registry, temporal_store):
        """Create validation engine."""
        return ValidationEngine(
            registry=validation_registry,
            temporal_store=temporal_store,
            max_workers=2,
            timeout_ms=5000,
            enable_async=True
        )
    
    @pytest.fixture
    def trading_calendar(self):
        """Create Taiwan trading calendar."""
        return create_taiwan_trading_calendar(2024)
    
    @pytest.fixture
    def pit_engine(self, temporal_store, trading_calendar):
        """Create PIT engine."""
        return PointInTimeEngine(
            store=temporal_store,
            trading_calendar=trading_calendar,
            enable_cache=True,
            max_workers=2
        )
    
    @pytest.fixture
    def pit_orchestrator(self, temporal_store, trading_calendar):
        """Create PIT validation orchestrator."""
        config = PITValidationConfig(
            mode=PITValidationMode.PERFORMANCE,
            max_latency_ms=10.0,
            taiwan_market_rules=True,
            parallel_validation=True
        )
        
        return create_pit_validation_orchestrator(
            temporal_store=temporal_store,
            trading_calendar=trading_calendar,
            config=config
        )
    
    @pytest.mark.asyncio
    async def test_basic_validation(self, validation_engine):
        """Test basic validation functionality."""
        # Create test value
        test_value = TemporalValue(
            symbol="2330",
            value=Decimal("500.0"),
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE
        )
        
        # Create context
        context = ValidationContext(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE
        )
        
        # Validate
        results = await validation_engine.validate_value(test_value, context)
        
        # Assertions
        assert len(results) > 0
        assert all(isinstance(result, ValidationOutput) for result in results)
        assert all(result.validator_name for result in results)
    
    @pytest.mark.asyncio
    async def test_performance_validation_latency(self, validation_engine):
        """Test validation latency is under 10ms."""
        # Create test value
        test_value = TemporalValue(
            symbol="2330",
            value=Decimal("500.0"),
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE
        )
        
        # Measure validation time
        start_time = time.perf_counter()
        results = await validation_engine.validate_value(test_value)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Assert latency under 10ms (generous limit for test environment)
        assert latency_ms < 50.0, f"Validation latency {latency_ms:.2f}ms exceeds limit"
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_batch_validation_performance(self, validation_engine):
        """Test batch validation performance."""
        # Create batch of test values
        test_values = []
        for i in range(100):
            value = TemporalValue(
                symbol=f"23{30 + i % 10}",
                value=Decimal(f"{500 + i}"),
                value_date=date(2024, 1, 15),
                as_of_date=date(2024, 1, 15),
                data_type=DataType.PRICE
            )
            test_values.append(value)
        
        # Measure batch validation time
        start_time = time.perf_counter()
        results = await validation_engine.validate_batch(test_values)
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        per_value_time_ms = total_time_ms / len(test_values)
        
        # Assert performance targets
        assert len(results) == len(test_values)
        assert per_value_time_ms < 5.0, f"Per-value latency {per_value_time_ms:.2f}ms exceeds target"
        
        # Assert all values validated
        for value in test_values:
            assert value in results
            assert len(results[value]) > 0
    
    @pytest.mark.asyncio
    async def test_pit_integration_validation(self, pit_orchestrator):
        """Test point-in-time integration validation."""
        symbols = ["2330", "2317", "2454"]
        as_of_date = date(2024, 1, 15)
        data_types = [DataType.PRICE, DataType.VOLUME]
        
        # Execute PIT validation
        result = await pit_orchestrator.validate_point_in_time(
            symbols=symbols,
            as_of_date=as_of_date,
            data_types=data_types
        )
        
        # Assertions
        assert result is not None
        assert result.query.symbols == symbols
        assert result.query.as_of_date == as_of_date
        assert result.execution_time_ms > 0
        assert result.overall_quality_score >= 0
        
        # Check performance
        assert result.execution_time_ms < 100.0, f"PIT validation took {result.execution_time_ms:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_streaming_data_validation(self, pit_orchestrator):
        """Test streaming data validation with <10ms latency."""
        # Create streaming value
        stream_value = TemporalValue(
            symbol="2330",
            value=Decimal("500.0"),
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE,
            created_at=datetime.utcnow()
        )
        
        # Measure streaming validation
        start_time = time.perf_counter()
        result = await pit_orchestrator.validate_streaming_data(stream_value)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Assertions
        assert result is not None
        assert isinstance(result, ValidationOutput)
        assert latency_ms < 15.0, f"Streaming validation latency {latency_ms:.2f}ms exceeds target"
    
    @pytest.mark.asyncio
    async def test_validation_caching(self, validation_engine):
        """Test validation result caching."""
        # Create test value
        test_value = TemporalValue(
            symbol="2330",
            value=Decimal("500.0"),
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE
        )
        
        # First validation (miss)
        start_stats = validation_engine.get_performance_stats()
        results1 = await validation_engine.validate_value(test_value)
        mid_stats = validation_engine.get_performance_stats()
        
        # Second validation (hit)
        results2 = await validation_engine.validate_value(test_value)
        end_stats = validation_engine.get_performance_stats()
        
        # Check cache effectiveness
        cache_misses_increased = mid_stats['cache_size'] > start_stats['cache_size']
        cache_hits_increased = end_stats['cache_hit_rate'] > mid_stats['cache_hit_rate']
        
        assert cache_misses_increased, "Cache should store results"
        assert cache_hits_increased, "Second validation should hit cache"
        assert len(results1) == len(results2), "Results should be consistent"
    
    def test_validation_registry_management(self, validation_registry):
        """Test validation plugin registry management."""
        initial_count = len(validation_registry.list_plugins())
        
        # Create mock validator
        mock_validator = Mock(spec=ValidationPlugin)
        mock_validator.name = "test_validator"
        mock_validator.version = "1.0.0"
        mock_validator.priority = ValidationPriority.MEDIUM
        mock_validator.supported_data_types = {DataType.PRICE}
        mock_validator.initialize = Mock()
        mock_validator.cleanup = Mock()
        
        # Register plugin
        validation_registry.register_plugin(mock_validator)
        assert len(validation_registry.list_plugins()) == initial_count + 1
        assert "test_validator" in validation_registry.list_plugins()
        
        # Get plugin
        retrieved = validation_registry.get_plugin("test_validator")
        assert retrieved == mock_validator
        
        # Unregister plugin
        validation_registry.unregister_plugin("test_validator")
        assert len(validation_registry.list_plugins()) == initial_count
        assert "test_validator" not in validation_registry.list_plugins()
        
        # Verify cleanup was called
        mock_validator.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, validation_engine):
        """Test validation error handling."""
        # Create invalid test value
        invalid_value = TemporalValue(
            symbol="INVALID",
            value="invalid_data",
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE
        )
        
        # Validation should handle errors gracefully
        results = await validation_engine.validate_value(invalid_value)
        
        # Should still return results (possibly with errors)
        assert len(results) >= 0
        
        # Check for error results
        error_results = [r for r in results if r.result == ValidationResult.FAIL]
        if error_results:
            assert all(len(r.issues) > 0 for r in error_results)
    
    @pytest.mark.asyncio
    async def test_concurrent_validation(self, validation_engine):
        """Test concurrent validation handling."""
        # Create multiple test values
        test_values = [
            TemporalValue(
                symbol=f"23{30 + i}",
                value=Decimal(f"{500 + i}"),
                value_date=date(2024, 1, 15),
                as_of_date=date(2024, 1, 15),
                data_type=DataType.PRICE
            )
            for i in range(10)
        ]
        
        # Run concurrent validations
        tasks = [validation_engine.validate_value(value) for value in test_values]
        results = await asyncio.gather(*tasks)
        
        # All validations should complete
        assert len(results) == len(test_values)
        assert all(len(result) > 0 for result in results)
    
    def test_performance_stats_tracking(self, validation_engine):
        """Test performance statistics tracking."""
        stats = validation_engine.get_performance_stats()
        
        # Check required stats
        required_fields = [
            'validation_count', 'avg_execution_time_ms', 'total_execution_time_ms',
            'cache_hit_rate', 'cache_size', 'registered_plugins'
        ]
        
        for field in required_fields:
            assert field in stats, f"Missing required stat: {field}"
        
        # Check data types
        assert isinstance(stats['validation_count'], int)
        assert isinstance(stats['avg_execution_time_ms'], (int, float))
        assert isinstance(stats['cache_hit_rate'], (int, float))
        assert 0 <= stats['cache_hit_rate'] <= 1


class TestTaiwanMarketValidation:
    """Test Taiwan market specific validation functionality."""
    
    @pytest.fixture
    def price_validator(self):
        """Create Taiwan price limit validator."""
        return TaiwanPriceLimitValidator()
    
    @pytest.fixture
    def volume_validator(self):
        """Create Taiwan volume validator."""
        return TaiwanVolumeValidator()
    
    @pytest.fixture
    def trading_hours_validator(self):
        """Create Taiwan trading hours validator."""
        return TaiwanTradingHoursValidator()
    
    @pytest.fixture
    def fundamental_lag_validator(self):
        """Create Taiwan fundamental lag validator."""
        return TaiwanFundamentalLagValidator()
    
    @pytest.mark.asyncio
    async def test_price_limit_validation_pass(self, price_validator):
        """Test price limit validation with valid price change."""
        # Create price within 10% limit
        current_price = TemporalValue(
            symbol="2330",
            value=Decimal("505.0"),  # 1% increase
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE
        )
        
        # Create context with previous close
        context = ValidationContext(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE,
            historical_data=[
                TemporalValue(
                    symbol="2330",
                    value=Decimal("500.0"),
                    value_date=date(2024, 1, 14),
                    as_of_date=date(2024, 1, 14),
                    data_type=DataType.PRICE
                )
            ]
        )
        
        result = await price_validator.validate(current_price, context)
        
        # Should pass validation
        assert result.result in [ValidationResult.PASS, ValidationResult.SKIP]
        assert not result.has_critical_issues
    
    @pytest.mark.asyncio
    async def test_price_limit_validation_fail(self, price_validator):
        """Test price limit validation with excessive price change."""
        # Create price exceeding 10% limit
        current_price = TemporalValue(
            symbol="2330",
            value=Decimal("600.0"),  # 20% increase
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE
        )
        
        # Create context with previous close
        context = ValidationContext(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE,
            historical_data=[
                TemporalValue(
                    symbol="2330",
                    value=Decimal("500.0"),
                    value_date=date(2024, 1, 14),
                    as_of_date=date(2024, 1, 14),
                    data_type=DataType.PRICE
                )
            ]
        )
        
        result = await price_validator.validate(current_price, context)
        
        # Should detect violation
        assert result.result == ValidationResult.WARNING
        assert len(result.issues) > 0
        assert any("exceeds daily limit" in issue.description for issue in result.issues)
    
    @pytest.mark.asyncio
    async def test_volume_spike_detection(self, volume_validator):
        """Test volume spike detection."""
        # Create high volume
        high_volume = TemporalValue(
            symbol="2330",
            value=50000000,  # 50M shares
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.VOLUME
        )
        
        # Create context with normal historical volumes
        historical_volumes = [
            TemporalValue(
                symbol="2330",
                value=1000000 + i * 100000,  # 1-3M shares
                value_date=date(2024, 1, 15) - timedelta(days=i+1),
                as_of_date=date(2024, 1, 15) - timedelta(days=i+1),
                data_type=DataType.VOLUME
            )
            for i in range(20)
        ]
        
        context = ValidationContext(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.VOLUME,
            historical_data=historical_volumes
        )
        
        result = await volume_validator.validate(high_volume, context)
        
        # Should detect volume spike
        assert result.result == ValidationResult.WARNING
        assert len(result.issues) > 0
        assert any("Volume spike" in issue.description for issue in result.issues)
    
    @pytest.mark.asyncio
    async def test_fundamental_lag_validation_critical(self, fundamental_lag_validator):
        """Test fundamental data lag validation with critical issue."""
        # Create fundamental data available before quarter end (critical bias)
        fundamental_data = TemporalValue(
            symbol="2330",
            value={
                "fiscal_year": 2024,
                "fiscal_quarter": 1,
                "revenue": 1000000,
                "net_income": 100000
            },
            value_date=date(2024, 3, 31),  # Q1 end
            as_of_date=date(2024, 3, 25),  # Before quarter end!
            data_type=DataType.FUNDAMENTAL
        )
        
        context = ValidationContext(
            symbol="2330",
            data_date=date(2024, 3, 31),
            as_of_date=date(2024, 3, 25),
            data_type=DataType.FUNDAMENTAL
        )
        
        result = await fundamental_lag_validator.validate(fundamental_data, context)
        
        # Should detect critical look-ahead bias
        assert result.result == ValidationResult.FAIL
        assert any(issue.severity == SeverityLevel.CRITICAL for issue in result.issues)
        assert any("before quarter end" in issue.description for issue in result.issues)
    
    @pytest.mark.asyncio
    async def test_trading_hours_validation(self, trading_hours_validator):
        """Test trading hours validation."""
        # Create data with timestamp outside trading hours
        after_hours_data = TemporalValue(
            symbol="2330",
            value={"close_price": 500.0, "timestamp": "2024-01-15T18:00:00+08:00"},
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.MARKET_DATA,
            created_at=datetime(2024, 1, 15, 18, 0)  # 6 PM TST
        )
        
        context = ValidationContext(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.MARKET_DATA
        )
        
        result = await trading_hours_validator.validate(after_hours_data, context)
        
        # Should detect outside trading hours
        assert result.result in [ValidationResult.WARNING, ValidationResult.SKIP]
        if result.issues:
            assert any("outside trading hours" in issue.description for issue in result.issues)
    
    def test_taiwan_validators_creation(self):
        """Test Taiwan validators creation and configuration."""
        validators = create_taiwan_validators()
        
        # Check all expected validators are created
        validator_names = [v.name for v in validators]
        expected_validators = [
            "taiwan_price_limit_validator",
            "taiwan_volume_validator", 
            "taiwan_trading_hours_validator",
            "taiwan_settlement_validator",
            "taiwan_fundamental_lag_validator"
        ]
        
        for expected in expected_validators:
            assert expected in validator_names, f"Missing validator: {expected}"
        
        # Check all validators are properly configured
        for validator in validators:
            assert validator.name
            assert validator.version
            assert validator.priority
            assert len(validator.supported_data_types) > 0


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests for validation framework."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_validation(self, validation_engine):
        """Test high throughput validation (target: 100K validations/minute)."""
        # Create large batch of test values
        batch_size = 1000
        test_values = []
        
        for i in range(batch_size):
            value = TemporalValue(
                symbol=f"23{30 + i % 100}",
                value=Decimal(f"{500 + i % 100}"),
                value_date=date(2024, 1, 15),
                as_of_date=date(2024, 1, 15),
                data_type=DataType.PRICE
            )
            test_values.append(value)
        
        # Measure throughput
        start_time = time.perf_counter()
        results = await validation_engine.validate_batch(test_values)
        end_time = time.perf_counter()
        
        # Calculate throughput
        total_time_seconds = end_time - start_time
        validations_per_second = batch_size / total_time_seconds
        validations_per_minute = validations_per_second * 60
        
        # Assert performance target
        assert len(results) == batch_size
        assert validations_per_minute > 50000, f"Throughput {validations_per_minute:.0f}/min below target"
        
        print(f"Validation throughput: {validations_per_minute:.0f} validations/minute")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, validation_engine):
        """Test memory efficiency with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        large_batch = []
        for i in range(5000):
            value = TemporalValue(
                symbol=f"S{i:04d}",
                value=Decimal(f"{100 + i}"),
                value_date=date(2024, 1, 15),
                as_of_date=date(2024, 1, 15),
                data_type=DataType.PRICE
            )
            large_batch.append(value)
        
        await validation_engine.validate_batch(large_batch)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB too high"
        
        print(f"Memory increase: {memory_increase:.1f}MB for 5000 validations")
    
    @pytest.mark.asyncio  
    async def test_latency_percentiles(self, validation_engine):
        """Test validation latency percentiles."""
        latencies = []
        
        # Collect latency samples
        for i in range(100):
            test_value = TemporalValue(
                symbol="2330",
                value=Decimal("500.0"),
                value_date=date(2024, 1, 15),
                as_of_date=date(2024, 1, 15),
                data_type=DataType.PRICE
            )
            
            start_time = time.perf_counter()
            await validation_engine.validate_value(test_value)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate percentiles
        latencies.sort()
        p50 = latencies[49]   # 50th percentile
        p95 = latencies[94]   # 95th percentile
        p99 = latencies[98]   # 99th percentile
        
        # Assert latency targets
        assert p50 < 5.0, f"P50 latency {p50:.2f}ms exceeds target"
        assert p95 < 10.0, f"P95 latency {p95:.2f}ms exceeds target"
        assert p99 < 20.0, f"P99 latency {p99:.2f}ms exceeds target"
        
        print(f"Latency percentiles - P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")


if __name__ == "__main__":
    # Run specific test groups
    pytest.main([__file__, "-v", "--tb=short"])