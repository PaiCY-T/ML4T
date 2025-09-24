"""
Integration tests for technical factors with real data pipeline.

This test demonstrates the integration between technical factors
and the existing data pipeline infrastructure.
"""

import unittest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from factors.technical import TechnicalFactors
from factors.taiwan_adjustments import TaiwanMarketAdjustments


class MockDataPipeline:
    """Mock data pipeline that simulates realistic Taiwan market data."""
    
    def __init__(self):
        self.taiwan_symbols = [
            '2330.TW',  # TSMC
            '2317.TW',  # Hon Hai
            '1301.TW',  # Formosa Plastics
            '1303.TW',  # Nan Ya Plastics
            '2412.TW',  # Chunghwa Telecom
            '2454.TW',  # MediaTek
            '1216.TW',  # Uni-President
            '2308.TW',  # Delta Electronics
            '2881.TW',  # Fubon Financial
            '2891.TW'   # CTBC Financial
        ]
        
        # Generate realistic Taiwan market data
        self.market_data = self._generate_market_data()
    
    def _generate_market_data(self):
        """Generate realistic Taiwan market data."""
        
        # Taiwan market characteristics
        trading_days = pd.bdate_range(
            start='2022-01-01',
            end='2023-12-31',
            freq='B'
        )
        
        # Remove Taiwan holidays (simplified)
        taiwan_holidays = [
            date(2023, 1, 2),   # New Year
            date(2023, 1, 23),  # Lunar New Year
            date(2023, 1, 24),  # Lunar New Year
            date(2023, 1, 25),  # Lunar New Year
            date(2023, 1, 26),  # Lunar New Year
            date(2023, 1, 27),  # Lunar New Year
            date(2023, 4, 5),   # Tomb Sweeping Day
            date(2023, 5, 1),   # Labor Day
            date(2023, 10, 10), # National Day
        ]
        
        trading_days = [d for d in trading_days if d.date() not in taiwan_holidays]
        
        market_data = {}
        
        for symbol in self.taiwan_symbols:
            # Symbol-specific characteristics
            if symbol == '2330.TW':  # TSMC - tech giant
                base_price = 500
                volatility = 0.025
                drift = 0.0003
            elif symbol == '2317.TW':  # Hon Hai - manufacturing
                base_price = 100
                volatility = 0.03
                drift = 0.0001
            elif symbol in ['2881.TW', '2891.TW']:  # Financials
                base_price = 25
                volatility = 0.02
                drift = 0.0002
            else:  # Others
                base_price = 50 + (hash(symbol) % 100)
                volatility = 0.02 + (hash(symbol) % 20) / 1000
                drift = (hash(symbol) % 10 - 5) / 100000
            
            prices = self._generate_price_series(
                trading_days, base_price, volatility, drift, symbol
            )
            
            market_data[symbol] = prices
        
        return market_data
    
    def _generate_price_series(self, dates, base_price, volatility, drift, symbol):
        """Generate realistic price series with Taiwan market constraints."""
        
        np.random.seed(hash(symbol) % 2**32)
        
        prices_data = []
        current_price = base_price
        
        for i, trade_date in enumerate(dates):
            # Generate daily return
            random_shock = np.random.normal(0, volatility)
            daily_return = drift + random_shock
            
            # Apply Taiwan daily price limits (±10%)
            daily_return = np.clip(daily_return, -0.1, 0.1)
            
            # Update price
            new_price = current_price * (1 + daily_return)
            new_price = max(1.0, new_price)  # Minimum price 1 TWD
            
            # Generate OHLCV
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, new_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, new_price) * (1 - abs(np.random.normal(0, 0.01)))
            close_price = new_price
            
            # Taiwan typical volume patterns
            base_volume = 1000000 * (1 + hash(symbol) % 10)
            volume_multiplier = 1 + np.random.exponential(0.5)  # Right-skewed volume
            volume = int(base_volume * volume_multiplier)
            
            prices_data.append({
                'date': trade_date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            current_price = close_price
        
        return pd.DataFrame(prices_data).set_index('date')
    
    def query(self, pit_query):
        """Mock PIT query that returns Taiwan market data."""
        
        result_data = []
        
        for symbol in pit_query.symbols:
            if symbol not in self.market_data:
                continue
            
            symbol_data = self.market_data[symbol]
            
            # Filter by date range
            mask = (
                (symbol_data.index >= pit_query.start_date) &
                (symbol_data.index <= pit_query.as_of_date)
            )
            filtered_data = symbol_data[mask].copy()
            
            # Add symbol column and reset index
            filtered_data['symbol'] = symbol
            filtered_data = filtered_data.reset_index()
            
            result_data.append(filtered_data)
        
        if result_data:
            combined_data = pd.concat(result_data, ignore_index=True)
            # Set multi-index as expected by factor calculations
            combined_data = combined_data.set_index(['date', 'symbol'])
            return combined_data
        else:
            return pd.DataFrame()


class TestTechnicalFactorsIntegration(unittest.TestCase):
    """Integration tests for technical factors."""
    
    def setUp(self):
        """Set up test environment."""
        self.data_pipeline = MockDataPipeline()
        self.taiwan_adj = TaiwanMarketAdjustments()
        self.tech_factors = TechnicalFactors(self.data_pipeline, self.taiwan_adj)
        
        self.test_symbols = self.data_pipeline.taiwan_symbols[:5]  # Use subset for faster testing
        self.test_date = date(2023, 6, 15)
    
    def test_full_factor_calculation_pipeline(self):
        """Test complete factor calculation pipeline."""
        
        # Calculate all technical factors
        start_time = datetime.now()
        
        results = self.tech_factors.calculate_all_factors(
            self.test_symbols, self.test_date, parallel=False
        )
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        # Performance requirements
        self.assertLess(calculation_time, 30.0)  # Should complete within 30 seconds
        
        # Validate results
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 8)  # Should have most factors
        
        # Check individual factor results
        for factor_name, result in results.items():
            with self.subTest(factor=factor_name):
                self.assertIsInstance(result.values, dict)
                self.assertGreater(len(result.values), 0)  # Should have some results
                
                # Check coverage
                if result.coverage:
                    self.assertGreaterEqual(result.coverage, 0.6)  # At least 60% coverage
                
                # Check that values are reasonable
                for symbol, value in result.values.items():
                    self.assertFalse(np.isnan(value))
                    self.assertIsInstance(value, (int, float))
                    # Sanity check: factors shouldn't be extremely large
                    self.assertLess(abs(value), 100.0)
    
    def test_factor_correlation_analysis(self):
        """Test factor correlation analysis."""
        
        # Calculate correlation matrix
        correlation_matrix = self.tech_factors.calculate_factor_correlations(
            self.test_symbols, self.test_date, lookback_days=30
        )
        
        if not correlation_matrix.empty:
            # Check correlation matrix properties
            self.assertTrue(correlation_matrix.shape[0] > 0)
            self.assertTrue(correlation_matrix.shape[1] > 0)
            
            # Diagonal should be 1 (or close to 1)
            diagonal_values = np.diag(correlation_matrix.values)
            self.assertTrue(np.all(diagonal_values >= 0.95))
            
            # Check for reasonable correlations (not all factors should be perfectly correlated)
            off_diagonal = correlation_matrix.values - np.diag(diagonal_values)
            max_correlation = np.max(np.abs(off_diagonal))
            self.assertLess(max_correlation, 0.95)  # Factors should be somewhat independent
    
    def test_taiwan_market_specific_features(self):
        """Test Taiwan market specific features are properly handled."""
        
        # Test with a date that has price limit scenarios
        results = self.tech_factors.calculate_all_factors(
            self.test_symbols, self.test_date
        )
        
        # Check that Taiwan adjustments are applied
        taiwan_metadata = self.taiwan_adj.get_taiwan_market_metadata()
        
        self.assertEqual(taiwan_metadata['price_limit'], 0.10)
        self.assertEqual(taiwan_metadata['settlement_days'], 2)
        self.assertEqual(taiwan_metadata['currency'], 'TWD')
        
        # Validate that results respect Taiwan market constraints
        for factor_name, result in results.items():
            # All factor results should be finite
            for symbol, value in result.values.items():
                self.assertTrue(np.isfinite(value))
    
    def test_factor_category_distribution(self):
        """Test that factors are properly distributed across categories."""
        
        categories = self.tech_factors.get_factor_categories()
        
        # Should have all three categories
        self.assertIn('momentum', categories)
        self.assertIn('mean_reversion', categories)
        self.assertIn('volatility', categories)
        
        # Each category should have factors
        self.assertGreater(len(categories['momentum']), 0)
        self.assertGreater(len(categories['mean_reversion']), 0)
        self.assertGreater(len(categories['volatility']), 0)
        
        # Total should match expected number of technical factors
        total_factors = sum(len(factors) for factors in categories.values())
        self.assertGreaterEqual(total_factors, 10)  # Should have at least 10 factors
    
    def test_factor_universe_validation(self):
        """Test factor universe validation."""
        
        validation_results = self.tech_factors.validate_factor_universe_coverage(
            self.test_symbols, self.test_date, min_coverage=0.7
        )
        
        self.assertIsInstance(validation_results, dict)
        
        # Most factors should pass validation
        passing_factors = sum(1 for passed in validation_results.values() if passed)
        total_factors = len(validation_results)
        
        pass_rate = passing_factors / total_factors if total_factors > 0 else 0
        self.assertGreaterEqual(pass_rate, 0.6)  # At least 60% should pass
    
    def test_performance_statistics(self):
        """Test performance statistics tracking."""
        
        # Calculate factors to generate statistics
        self.tech_factors.calculate_all_factors(self.test_symbols, self.test_date)
        
        stats = self.tech_factors.get_calculation_stats()
        
        self.assertIn(self.test_date, stats)
        
        date_stats = stats[self.test_date]
        
        # Validate statistics structure
        required_fields = [
            'universe_size', 'factors_calculated', 'elapsed_time_seconds',
            'average_coverage', 'factors_with_full_coverage'
        ]
        
        for field in required_fields:
            self.assertIn(field, date_stats)
        
        # Validate values
        self.assertEqual(date_stats['universe_size'], len(self.test_symbols))
        self.assertGreater(date_stats['factors_calculated'], 0)
        self.assertGreater(date_stats['elapsed_time_seconds'], 0)
        self.assertLessEqual(date_stats['elapsed_time_seconds'], 60)  # Should be reasonable
    
    def test_factor_definitions_export(self):
        """Test factor definitions export."""
        
        definitions = self.tech_factors.export_factor_definitions()
        
        self.assertIsInstance(definitions, dict)
        self.assertGreater(len(definitions), 8)  # Should have multiple factors
        
        # Check structure of definitions
        for factor_name, definition in definitions.items():
            with self.subTest(factor=factor_name):
                required_fields = [
                    'name', 'category', 'description', 'lookback_days',
                    'taiwan_specific', 'expected_ic'
                ]
                
                for field in required_fields:
                    self.assertIn(field, definition)
                
                # Taiwan-specific validation
                self.assertIsInstance(definition['taiwan_specific'], bool)
                if definition['expected_ic']:
                    self.assertGreater(definition['expected_ic'], 0)
                    self.assertLess(definition['expected_ic'], 1.0)
    
    def test_parallel_vs_sequential_calculation(self):
        """Test that parallel and sequential calculations give same results."""
        
        # Calculate factors sequentially
        sequential_results = self.tech_factors.calculate_all_factors(
            self.test_symbols, self.test_date, parallel=False
        )
        
        # Calculate factors in parallel
        parallel_results = self.tech_factors.calculate_all_factors(
            self.test_symbols, self.test_date, parallel=True
        )
        
        # Results should be the same
        self.assertEqual(len(sequential_results), len(parallel_results))
        
        for factor_name in sequential_results:
            if factor_name in parallel_results:
                seq_values = sequential_results[factor_name].values
                par_values = parallel_results[factor_name].values
                
                # Same symbols should be present
                self.assertEqual(set(seq_values.keys()), set(par_values.keys()))
                
                # Values should be very close (allowing for floating point differences)
                for symbol in seq_values:
                    if symbol in par_values:
                        self.assertAlmostEqual(
                            seq_values[symbol], par_values[symbol], places=6
                        )


class TestRealWorldScenarios(unittest.TestCase):
    """Test scenarios that simulate real-world usage patterns."""
    
    def setUp(self):
        self.data_pipeline = MockDataPipeline()
        self.tech_factors = TechnicalFactors(self.data_pipeline)
    
    def test_large_universe_calculation(self):
        """Test calculation with large universe of stocks."""
        
        # Use all available symbols
        large_universe = self.data_pipeline.taiwan_symbols
        test_date = date(2023, 9, 15)
        
        start_time = datetime.now()
        
        results = self.tech_factors.calculate_all_factors(
            large_universe, test_date, parallel=True
        )
        
        calculation_time = (datetime.now() - start_time).total_seconds()
        
        # Performance target: should handle 10 symbols within reasonable time
        self.assertLess(calculation_time, 60.0)  # 1 minute for 10 symbols
        
        # Coverage should be reasonable for large universe
        avg_coverage = np.mean([
            r.coverage for r in results.values() 
            if r.coverage is not None
        ])
        
        self.assertGreater(avg_coverage, 0.5)  # At least 50% average coverage
    
    def test_historical_factor_calculation(self):
        """Test calculating factors for historical dates."""
        
        # Calculate factors for multiple historical dates
        test_dates = [
            date(2023, 3, 15),
            date(2023, 6, 15),
            date(2023, 9, 15)
        ]
        
        test_symbols = self.data_pipeline.taiwan_symbols[:3]
        historical_results = {}
        
        for test_date in test_dates:
            results = self.tech_factors.calculate_all_factors(
                test_symbols, test_date, parallel=False
            )
            historical_results[test_date] = results
        
        # Should have results for all dates
        self.assertEqual(len(historical_results), len(test_dates))
        
        # Each date should have factor results
        for test_date, results in historical_results.items():
            self.assertGreater(len(results), 0)
            
            for factor_name, result in results.items():
                self.assertEqual(result.date, test_date)
                self.assertGreater(len(result.values), 0)
    
    def test_factor_subset_calculation(self):
        """Test calculating specific factor subsets for different use cases."""
        
        test_symbols = self.data_pipeline.taiwan_symbols[:5]
        test_date = date(2023, 6, 15)
        
        # Test momentum factors only
        momentum_factors = ['price_momentum', 'rsi_momentum', 'macd_signal']
        momentum_results = self.tech_factors.calculate_factor_subset(
            momentum_factors, test_symbols, test_date
        )
        
        self.assertEqual(len(momentum_results), len(momentum_factors))
        
        for factor_name in momentum_factors:
            self.assertIn(factor_name, momentum_results)
        
        # Test volatility factors only
        volatility_factors = ['realized_volatility', 'garch_volatility', 'taiwan_vix']
        volatility_results = self.tech_factors.calculate_factor_subset(
            volatility_factors, test_symbols, test_date
        )
        
        self.assertEqual(len(volatility_results), len(volatility_factors))


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Run tests with higher verbosity for integration tests
    unittest.main(verbosity=2)