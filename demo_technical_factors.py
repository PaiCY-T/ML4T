"""
Demonstration script for technical factors implementation.

This script demonstrates the technical factors functionality
with mock data, showcasing the 18 technical factors for Taiwan market.
"""

import sys
import os
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockPITEngine:
    """Mock Point-in-Time Engine for demonstration."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def query(self, pit_query):
        """Generate mock Taiwan market data."""
        
        # Generate trading dates
        dates = pd.bdate_range(
            start=pit_query.start_date,
            end=pit_query.as_of_date,
            freq='B'
        )
        
        data_list = []
        
        for symbol in pit_query.symbols:
            # Generate realistic Taiwan stock prices
            np.random.seed(hash(symbol) % 2**32)  # Deterministic for consistency
            
            # Taiwan stock characteristics
            if symbol == '2330.TW':  # TSMC
                base_price = 500
                volatility = 0.025
            elif symbol == '2317.TW':  # Hon Hai
                base_price = 100
                volatility = 0.03
            else:
                base_price = 50 + abs(hash(symbol)) % 100
                volatility = 0.02 + (abs(hash(symbol)) % 20) / 1000
            
            # Generate price series
            returns = np.random.normal(0.0005, volatility, len(dates))
            prices = [base_price]
            
            for ret in returns[1:]:
                # Apply Taiwan 10% daily price limits
                ret = np.clip(ret, -0.1, 0.1)
                new_price = prices[-1] * (1 + ret)
                prices.append(max(1.0, new_price))  # Min 1 TWD
            
            # Create OHLCV data
            for i, trade_date in enumerate(dates):
                close_price = prices[i]
                open_price = close_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = np.random.randint(500000, 5000000)
                
                data_list.append({
                    'symbol': symbol,
                    'date': trade_date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
        
        df = pd.DataFrame(data_list)
        df = df.set_index(['date', 'symbol'])
        return df


def demonstrate_technical_factors():
    """Demonstrate technical factors calculation."""
    
    logger.info("=== Taiwan Market Technical Factors Demonstration ===")
    
    # Import with fallback
    try:
        from factors.taiwan_adjustments import TaiwanMarketAdjustments
        from factors.technical import TechnicalFactors, get_technical_factor_metadata
        logger.info("✓ Successfully imported factor modules")
    except ImportError as e:
        logger.error(f"✗ Failed to import factor modules: {e}")
        return
    
    # Setup
    mock_engine = MockPITEngine()
    taiwan_adj = TaiwanMarketAdjustments()
    
    # Taiwan market symbols for demo
    symbols = [
        '2330.TW',  # TSMC
        '2317.TW',  # Hon Hai  
        '1301.TW',  # Formosa Plastics
        '2412.TW',  # Chunghwa Telecom
        '2454.TW'   # MediaTek
    ]
    
    as_of_date = date(2023, 9, 15)
    
    logger.info(f"Demo Universe: {len(symbols)} Taiwan stocks")
    logger.info(f"Calculation Date: {as_of_date}")
    
    # Initialize technical factors
    try:
        tech_factors = TechnicalFactors(mock_engine, taiwan_adj)
        logger.info("✓ TechnicalFactors initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize TechnicalFactors: {e}")
        return
    
    # Show factor metadata
    logger.info("\n=== Factor Metadata ===")
    try:
        metadata = get_technical_factor_metadata()
        for factor_name, info in metadata.items():
            logger.info(f"{factor_name:20} | {info['category']:15} | {info['description']}")
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}")
    
    # Calculate all technical factors
    logger.info("\n=== Calculating Technical Factors ===")
    start_time = datetime.now()
    
    try:
        results = tech_factors.calculate_all_factors(
            symbols, as_of_date, parallel=False
        )
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ Calculation completed in {calculation_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"✗ Factor calculation failed: {e}")
        return
    
    # Display results
    logger.info(f"\n=== Results Summary ===")
    logger.info(f"Total factors calculated: {len(results)}")
    
    if results:
        # Calculate average coverage
        coverages = [r.coverage for r in results.values() if r.coverage is not None]
        avg_coverage = np.mean(coverages) if coverages else 0
        logger.info(f"Average universe coverage: {avg_coverage:.1%}")
        
        # Show sample results for each category
        categories = tech_factors.get_factor_categories()
        
        for category_name, factor_names in categories.items():
            logger.info(f"\n--- {category_name.upper()} FACTORS ---")
            
            for factor_name in factor_names:
                if factor_name in results:
                    result = results[factor_name]
                    
                    if result.values:
                        sample_values = list(result.values.items())[:3]  # Show first 3
                        logger.info(f"{factor_name:20}: {sample_values}")
                        
                        # Show statistics
                        values = list(result.values.values())
                        if values:
                            logger.info(f"{'':20}  Mean: {np.mean(values):.4f}, "
                                      f"Std: {np.std(values):.4f}, "
                                      f"Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
    
    # Taiwan Market Adjustments Demo
    logger.info(f"\n=== Taiwan Market Adjustments ===")
    taiwan_metadata = taiwan_adj.get_taiwan_market_metadata()
    
    for key, value in taiwan_metadata.items():
        logger.info(f"{key:20}: {value}")
    
    # Performance statistics
    stats = tech_factors.get_calculation_stats()
    if as_of_date in stats:
        perf_stats = stats[as_of_date]
        logger.info(f"\n=== Performance Statistics ===")
        logger.info(f"Universe size: {perf_stats['universe_size']}")
        logger.info(f"Factors calculated: {perf_stats['factors_calculated']}")
        logger.info(f"Elapsed time: {perf_stats['elapsed_time_seconds']:.2f}s")
        logger.info(f"Average coverage: {perf_stats['average_coverage']:.1%}")
        logger.info(f"Full coverage factors: {perf_stats['factors_with_full_coverage']}")
    
    logger.info("\n=== Demonstration Complete ===")


if __name__ == '__main__':
    try:
        demonstrate_technical_factors()
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()