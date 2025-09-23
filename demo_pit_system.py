#!/usr/bin/env python3
"""
Demo script for Point-in-Time Data Management System.

This script demonstrates the core functionality of Stream A implementation.
"""

from datetime import date, datetime
from decimal import Decimal
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.core.temporal import (
    TemporalValue, InMemoryTemporalStore, TemporalDataManager, DataType
)
from data.models.taiwan_market import (
    TaiwanMarketData, TaiwanSettlement, TradingStatus,
    create_taiwan_trading_calendar
)
from data.pipeline.pit_engine import (
    PointInTimeEngine, PITQuery, QueryMode, BiasCheckLevel
)


def demo_temporal_data_system():
    """Demonstrate the point-in-time data system."""
    print("🚀 ML4T Point-in-Time Data Management System Demo")
    print("=" * 60)
    
    # Initialize components
    store = InMemoryTemporalStore()
    trading_calendar = create_taiwan_trading_calendar(2024)
    pit_engine = PointInTimeEngine(store, trading_calendar)
    
    print("\n📊 1. Creating Taiwan Market Data")
    
    # Create sample Taiwan stock data for TSMC (2330)
    tsmc_data = TaiwanMarketData(
        symbol="2330",
        data_date=date(2024, 9, 20),
        as_of_date=date(2024, 9, 20),
        open_price=Decimal("925.00"),
        high_price=Decimal("932.00"),
        low_price=Decimal("920.00"),
        close_price=Decimal("928.00"),
        volume=25_467_000,
        turnover=Decimal("23_632_456_000"),
        trading_status=TradingStatus.NORMAL
    )
    
    # Convert to temporal values and store
    temporal_values = tsmc_data.to_temporal_values()
    for value in temporal_values:
        store.store(value)
    
    print(f"✅ Stored {len(temporal_values)} temporal values for TSMC (2330)")
    
    print("\n🔍 2. Point-in-Time Query Demonstration")
    
    # Create a PIT query
    query = PITQuery(
        symbols=["2330"],
        as_of_date=date(2024, 9, 20),
        data_types=[DataType.PRICE, DataType.VOLUME, DataType.MARKET_DATA],
        mode=QueryMode.STRICT,
        bias_check=BiasCheckLevel.STRICT
    )
    
    # Execute query
    result = pit_engine.execute_query(query)
    
    print(f"✅ Query executed in {result.execution_time_ms:.2f}ms")
    print(f"📈 Retrieved data for {len(result.data)} symbols")
    
    # Display results
    if "2330" in result.data:
        symbol_data = result.data["2330"]
        
        print(f"\n💰 TSMC (2330) Data as of {query.as_of_date}:")
        
        if DataType.PRICE in symbol_data:
            price_value = symbol_data[DataType.PRICE]
            print(f"  📊 Close Price: ${price_value.value}")
            print(f"  📅 Value Date: {price_value.value_date}")
            print(f"  🕐 As Of Date: {price_value.as_of_date}")
        
        if DataType.VOLUME in symbol_data:
            volume_value = symbol_data[DataType.VOLUME]
            print(f"  📦 Volume: {volume_value.value:,} shares")
        
        if DataType.MARKET_DATA in symbol_data:
            market_values = [v for v in temporal_values 
                           if v.data_type == DataType.MARKET_DATA]
            print(f"  📈 Market Data: {len(market_values)} additional fields")
    
    print("\n🛡️ 3. Bias Detection Demonstration")
    
    # Try to create a query that would cause look-ahead bias
    try:
        future_query = PITQuery(
            symbols=["2330"],
            as_of_date=date(2024, 9, 19),  # Query before data date
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        # This should work, but with warnings about temporal consistency
        future_result = pit_engine.execute_query(future_query)
        
        if future_result.bias_violations:
            print(f"⚠️  Bias violations detected: {len(future_result.bias_violations)}")
            for violation in future_result.bias_violations:
                print(f"   - {violation}")
        else:
            print("✅ No bias violations detected (data not yet available)")
    
    except ValueError as e:
        print(f"🚫 Query blocked due to bias prevention: {e}")
    
    print("\n🏦 4. Taiwan Market Settlement Demo")
    
    # Demonstrate T+2 settlement calculation
    trade_date = date(2024, 9, 20)  # Friday
    settlement = TaiwanSettlement.calculate_t2_settlement(trade_date, trading_calendar)
    
    print(f"📅 Trade Date: {settlement.trade_date}")
    print(f"💰 Settlement Date: {settlement.settlement_date}")
    print(f"⏰ Settlement Lag: {settlement.settlement_lag_days} days")
    print(f"✅ Regular Settlement: {settlement.is_regular_settlement}")
    
    print("\n📊 5. Performance Statistics")
    
    # Get performance stats
    stats = pit_engine.get_performance_stats()
    
    print(f"🔢 Total Queries: {stats['query_count']}")
    print(f"⚡ Avg Execution Time: {stats['avg_execution_time_ms']:.2f}ms")
    print(f"🎯 Bias Violation Rate: {stats['bias_violation_rate']:.1%}")
    
    if 'cache_stats' in stats:
        cache_stats = stats['cache_stats']
        print(f"💾 Cache Utilization: {cache_stats['utilization']:.1%}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext: Stream B will implement database storage and API integration")
    print("Next: Stream C will add comprehensive testing and performance benchmarks")


if __name__ == "__main__":
    demo_temporal_data_system()