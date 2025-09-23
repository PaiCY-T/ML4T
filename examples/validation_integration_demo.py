"""
Data Quality Validation Framework Integration Demonstration.

This script demonstrates the complete integration of the data quality validation
framework with the point-in-time system, showcasing Taiwan market validation
capabilities and performance optimization.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the data quality framework components
from src.data.core.temporal import TemporalValue, TemporalStore, DataType
from src.data.models.taiwan_market import create_taiwan_trading_calendar
from src.data.pipeline.pit_engine import PointInTimeEngine
from src.data.quality.pit_integration import (
    PITValidationOrchestrator, PITValidationConfig, PITValidationMode,
    create_pit_validation_orchestrator, validate_taiwan_market_data
)


class ValidationDemonstration:
    """Demonstration of integrated validation system."""
    
    def __init__(self):
        self.setup_complete = False
        
    async def setup_demo_environment(self):
        """Set up demonstration environment with mock data."""
        logger.info("Setting up demonstration environment...")
        
        # Create mock temporal store with Taiwan market data
        from unittest.mock import Mock
        self.mock_store = Mock(spec=TemporalStore)
        
        # Taiwan stock symbols for demo
        self.symbols = ["2330", "2317", "2454", "2412", "3008"]
        
        # Mock data for demonstration
        self._setup_mock_data()
        
        # Create Taiwan trading calendar
        self.trading_calendar = create_taiwan_trading_calendar(2024)
        
        # Create PIT validation orchestrator
        config = PITValidationConfig(
            mode=PITValidationMode.STRICT,
            max_latency_ms=10.0,
            bias_check_level="strict",
            taiwan_market_rules=True,
            parallel_validation=True,
            enable_caching=True
        )
        
        self.orchestrator = create_pit_validation_orchestrator(
            temporal_store=self.mock_store,
            trading_calendar=self.trading_calendar,
            config=config
        )
        
        self.setup_complete = True
        logger.info("Demo environment setup complete")
    
    def _setup_mock_data(self):
        """Setup mock market data for demonstration."""
        # Normal price data
        normal_price = TemporalValue(
            symbol="2330",
            value=Decimal("580.0"),
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE,
            created_at=datetime(2024, 1, 15, 10, 30)
        )
        
        # Volume data
        normal_volume = TemporalValue(
            symbol="2330",
            value=25000000,  # 25M shares
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.VOLUME,
            created_at=datetime(2024, 1, 15, 10, 30)
        )
        
        # Market data
        market_data = TemporalValue(
            symbol="2330",
            value={
                "open_price": 575.0,
                "high_price": 585.0,
                "low_price": 570.0,
                "close_price": 580.0,
                "volume": 25000000,
                "timestamp": "2024-01-15T10:30:00+08:00"
            },
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.MARKET_DATA,
            created_at=datetime(2024, 1, 15, 10, 30)
        )
        
        # Configure mock returns
        def mock_get_point_in_time(symbol, as_of_date, data_type):
            if data_type == DataType.PRICE:
                return normal_price._replace(symbol=symbol)
            elif data_type == DataType.VOLUME:
                return normal_volume._replace(symbol=symbol)
            elif data_type == DataType.MARKET_DATA:
                return market_data._replace(symbol=symbol)
            return None
        
        def mock_get_range(symbol, start_date, end_date, data_type):
            # Return some historical data for context
            historical = []
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Weekdays only
                    if data_type == DataType.PRICE:
                        price_variation = Decimal("20.0") * (hash(str(current_date)) % 100) / 100
                        value = Decimal("560.0") + price_variation
                        historical.append(TemporalValue(
                            symbol=symbol,
                            value=value,
                            value_date=current_date,
                            as_of_date=current_date,
                            data_type=data_type
                        ))
                    elif data_type == DataType.VOLUME:
                        volume_variation = (hash(str(current_date)) % 20000000)
                        value = 15000000 + volume_variation
                        historical.append(TemporalValue(
                            symbol=symbol,
                            value=value,
                            value_date=current_date,
                            as_of_date=current_date,
                            data_type=data_type
                        ))
                current_date += timedelta(days=1)
            return historical[-10:]  # Return last 10 days
        
        self.mock_store.get_point_in_time.side_effect = mock_get_point_in_time
        self.mock_store.get_range.side_effect = mock_get_range
    
    async def demonstrate_basic_validation(self):
        """Demonstrate basic validation functionality."""
        if not self.setup_complete:
            await self.setup_demo_environment()
        
        logger.info("=== Basic Validation Demonstration ===")
        
        # Validate current market data
        result = await self.orchestrator.validate_point_in_time(
            symbols=["2330"],
            as_of_date=date(2024, 1, 15),
            data_types=[DataType.PRICE, DataType.VOLUME, DataType.MARKET_DATA]
        )
        
        logger.info(f"Validation completed in {result.execution_time_ms:.2f}ms")
        logger.info(f"Overall quality score: {result.overall_quality_score:.1f}")
        logger.info(f"Bias violations: {len(result.bias_violations)}")
        
        # Show detailed results
        for symbol, quality in result.quality_metrics.items():
            logger.info(f"Symbol {symbol}: Quality={quality.quality_score:.1f}, "
                       f"Latency={quality.validation_latency_ms:.2f}ms")
        
        return result
    
    async def demonstrate_taiwan_market_rules(self):
        """Demonstrate Taiwan market specific validation rules."""
        logger.info("=== Taiwan Market Rules Demonstration ===")
        
        # Test price limit validation with edge case
        price_limit_test = TemporalValue(
            symbol="2330",
            value=Decimal("638.0"),  # ~10% increase (at limit)
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE
        )
        
        result = await self.orchestrator.validate_streaming_data(price_limit_test)
        
        logger.info(f"Price limit validation result: {result.result.value}")
        if result.issues:
            for issue in result.issues:
                logger.info(f"  Issue: {issue.description} (Severity: {issue.severity.value})")
        
        # Test volume spike detection
        volume_spike_test = TemporalValue(
            symbol="2330",
            value=150000000,  # 150M shares (6x normal)
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.VOLUME
        )
        
        result = await self.orchestrator.validate_streaming_data(volume_spike_test)
        
        logger.info(f"Volume spike validation result: {result.result.value}")
        if result.issues:
            for issue in result.issues:
                logger.info(f"  Issue: {issue.description} (Severity: {issue.severity.value})")
        
        # Test fundamental data lag validation
        fundamental_test = TemporalValue(
            symbol="2330",
            value={
                "fiscal_year": 2023,
                "fiscal_quarter": 4,
                "revenue": 75900000000,
                "net_income": 25960000000
            },
            value_date=date(2023, 12, 31),  # Q4 end
            as_of_date=date(2024, 2, 29),   # 60 days later (at limit)
            data_type=DataType.FUNDAMENTAL
        )
        
        result = await self.orchestrator.validate_streaming_data(fundamental_test)
        
        logger.info(f"Fundamental lag validation result: {result.result.value}")
        if result.issues:
            for issue in result.issues:
                logger.info(f"  Issue: {issue.description} (Severity: {issue.severity.value})")
    
    async def demonstrate_performance_optimization(self):
        """Demonstrate performance optimization features."""
        logger.info("=== Performance Optimization Demonstration ===")
        
        # Test caching effectiveness
        symbols = self.symbols[:3]
        data_types = [DataType.PRICE, DataType.VOLUME]
        
        # First validation (cache miss)
        start_time = asyncio.get_event_loop().time()
        result1 = await self.orchestrator.validate_point_in_time(
            symbols=symbols,
            as_of_date=date(2024, 1, 15),
            data_types=data_types
        )
        first_time = asyncio.get_event_loop().time() - start_time
        
        # Second validation (cache hit)
        start_time = asyncio.get_event_loop().time()
        result2 = await self.orchestrator.validate_point_in_time(
            symbols=symbols,
            as_of_date=date(2024, 1, 15),
            data_types=data_types
        )
        second_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"First validation: {first_time*1000:.2f}ms")
        logger.info(f"Second validation: {second_time*1000:.2f}ms")
        logger.info(f"Cache speedup: {first_time/second_time:.1f}x")
        
        # Test batch processing
        batch_symbols = self.symbols
        start_time = asyncio.get_event_loop().time()
        
        batch_result = await self.orchestrator.validate_point_in_time(
            symbols=batch_symbols,
            as_of_date=date(2024, 1, 15),
            data_types=[DataType.PRICE, DataType.VOLUME]
        )
        
        batch_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"Batch validation ({len(batch_symbols)} symbols): {batch_time*1000:.2f}ms")
        logger.info(f"Per-symbol latency: {batch_time*1000/len(batch_symbols):.2f}ms")
        
        # Show performance stats
        stats = self.orchestrator.get_performance_stats()
        logger.info(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
        logger.info(f"Average execution time: {stats.get('avg_execution_time_ms', 0):.2f}ms")
    
    async def demonstrate_error_handling(self):
        """Demonstrate error handling and edge cases."""
        logger.info("=== Error Handling Demonstration ===")
        
        # Test with invalid data
        invalid_data = TemporalValue(
            symbol="INVALID",
            value="not_a_number",
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.PRICE
        )
        
        result = await self.orchestrator.validate_streaming_data(invalid_data)
        logger.info(f"Invalid data validation: {result.result.value}")
        
        # Test with future date (bias violation)
        future_data = TemporalValue(
            symbol="2330",
            value=Decimal("600.0"),
            value_date=date(2024, 12, 31),  # Future date
            as_of_date=date(2024, 12, 31),
            data_type=DataType.PRICE
        )
        
        try:
            result = await self.orchestrator.validate_point_in_time(
                symbols=["2330"],
                as_of_date=date(2024, 12, 31),  # Future date
                data_types=[DataType.PRICE]
            )
            if result.bias_violations:
                logger.info(f"Bias violations detected: {result.bias_violations}")
        except Exception as e:
            logger.info(f"Bias protection triggered: {e}")
    
    async def demonstrate_monitoring_integration(self):
        """Demonstrate monitoring and alerting integration."""
        logger.info("=== Monitoring Integration Demonstration ===")
        
        # Get current monitoring metrics
        if hasattr(self.orchestrator, 'monitor') and self.orchestrator.monitor:
            metrics = self.orchestrator.monitor.get_current_metrics()
            
            logger.info(f"Monitor status: {metrics.get('status', 'unknown')}")
            logger.info(f"Total validations: {metrics.get('total_validations', 0)}")
            logger.info(f"Average latency: {metrics.get('average_latency_ms', 0):.2f}ms")
            
            recent = metrics.get('recent_metrics', {})
            if recent:
                logger.info(f"Recent quality score: {recent.get('avg_quality_score', 0):.1f}")
                logger.info(f"Recent latency: {recent.get('avg_latency_ms', 0):.2f}ms")
        
        # Simulate quality threshold breach
        logger.info("Simulating quality monitoring alerts...")
        
        # This would normally trigger real alerts in production
        logger.info("Quality monitoring demonstration complete")
    
    async def run_complete_demonstration(self):
        """Run complete demonstration of all features."""
        logger.info("Starting Complete Data Quality Validation Demonstration")
        logger.info("=" * 80)
        
        try:
            await self.setup_demo_environment()
            
            # Run all demonstrations
            await self.demonstrate_basic_validation()
            await self.demonstrate_taiwan_market_rules()
            await self.demonstrate_performance_optimization()
            await self.demonstrate_error_handling()
            await self.demonstrate_monitoring_integration()
            
            logger.info("=" * 80)
            logger.info("Demonstration completed successfully!")
            
            # Final performance summary
            if hasattr(self.orchestrator, 'get_performance_stats'):
                final_stats = self.orchestrator.get_performance_stats()
                logger.info(f"Final Statistics:")
                logger.info(f"  Total validations: {final_stats.get('validation_count', 0)}")
                logger.info(f"  Average latency: {final_stats.get('avg_execution_time_ms', 0):.2f}ms")
                logger.info(f"  Cache hit rate: {final_stats.get('cache_hit_rate', 0):.1%}")
            
        except Exception as e:
            logger.error(f"Demonstration failed: {e}")
            raise
        finally:
            # Cleanup
            if hasattr(self.orchestrator, 'shutdown'):
                self.orchestrator.shutdown()


async def main():
    """Main demonstration entry point."""
    demo = ValidationDemonstration()
    await demo.run_complete_demonstration()


if __name__ == "__main__":
    asyncio.run(main())