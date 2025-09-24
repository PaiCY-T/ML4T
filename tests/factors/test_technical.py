"""
Comprehensive test suite for technical factors.

This test suite validates the technical factor calculations including
momentum, mean reversion, and volatility factors for Taiwan market.
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

from factors.technical import TechnicalFactors, calculate_technical_factors
from factors.taiwan_adjustments import TaiwanMarketAdjustments
from factors.base import FactorResult, FactorMetadata, FactorCategory, FactorFrequency
from factors.momentum import PriceMomentumCalculator, RSIMomentumCalculator, MACDSignalCalculator
from factors.mean_reversion import (
    MovingAverageReversionCalculator, BollingerBandPositionCalculator,
    ZScoreReversionCalculator, ShortTermReversalCalculator
)
from factors.volatility import (
    RealizedVolatilityCalculator, GARCHVolatilityCalculator,
    TaiwanVIXCalculator, VolatilityRiskPremiumCalculator
)


class MockPITEngine:
    """Mock PIT Engine for testing."""
    
    def __init__(self):
        self.query_count = 0
    
    def query(self, pit_query):
        """Mock query method that returns synthetic data."""
        self.query_count += 1
        
        # Generate synthetic OHLCV data
        dates = pd.bdate_range(
            start=pit_query.start_date,
            end=pit_query.as_of_date,
            freq='B'
        )
        
        data_list = []
        
        for symbol in pit_query.symbols:
            # Generate realistic price series with random walk
            np.random.seed(hash(symbol) % 2**32)  # Deterministic but symbol-specific
            
            base_price = 50 + (hash(symbol) % 100)  # Base price between 50-150
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
            
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                # Apply Taiwan price limits
                if abs(ret) > 0.1:  # 10% limit
                    ret = np.sign(ret) * 0.1
                    new_price = prices[-1] * (1 + ret)
                prices.append(max(1.0, new_price))  # Minimum price of 1 TWD
            
            for i, date in enumerate(dates):
                close_price = prices[i]
                open_price = close_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = np.random.randint(100000, 10000000)
                
                data_list.append({
                    'symbol': symbol,
                    'date': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
        
        df = pd.DataFrame(data_list)
        df = df.set_index(['date', 'symbol'])
        
        return df


class TestTaiwanMarketAdjustments(unittest.TestCase):
    """Test Taiwan market adjustments."""
    
    def setUp(self):
        self.taiwan_adj = TaiwanMarketAdjustments()
    
    def test_price_limit_adjustment(self):
        """Test price limit adjustments."""
        # Create test data with price limit violations
        dates = pd.date_range('2023-01-01', periods=5, freq='B')
        symbols = ['2330.TW', '2317.TW']
        
        prices_data = {
            '2330.TW': [100, 110, 121, 133, 146],  # Normal progression
            '2317.TW': [50, 60, 66, 85, 93.5]     # Contains limit violation
        }
        
        prices_df = pd.DataFrame(prices_data, index=dates)
        returns_df = prices_df.pct_change()
        
        adjusted_prices, adjusted_returns = self.taiwan_adj.adjust_for_price_limits(
            prices_df, returns_df
        )
        
        # Check that extreme returns are capped
        max_return = adjusted_returns.max().max()
        min_return = adjusted_returns.min().min()
        
        self.assertLessEqual(max_return, 0.11)  # Slightly above 10% for rounding
        self.assertGreaterEqual(min_return, -0.11)
    
    def test_settlement_lag_adjustment(self):
        """Test T+2 settlement lag adjustment."""
        factor_date = date(2023, 6, 15)  # Thursday
        adjusted_date = self.taiwan_adj.adjust_for_settlement_lag(factor_date)
        
        # Should be 2 days earlier
        expected_date = date(2023, 6, 13)  # Tuesday
        self.assertEqual(adjusted_date, expected_date)
    
    def test_market_hours_adjustment(self):
        """Test market hours adjustment."""
        # Before market open
        early_time = datetime(2023, 6, 15, 8, 30)
        adjusted = self.taiwan_adj.adjust_for_market_hours(early_time)
        self.assertEqual(adjusted.time(), self.taiwan_adj.MARKET_CLOSE_TIME)
        self.assertEqual(adjusted.date(), date(2023, 6, 14))
        
        # After market close
        late_time = datetime(2023, 6, 15, 15, 0)
        adjusted = self.taiwan_adj.adjust_for_market_hours(late_time)
        self.assertEqual(adjusted.time(), self.taiwan_adj.MARKET_CLOSE_TIME)
        self.assertEqual(adjusted.date(), date(2023, 6, 15))
    
    def test_data_validation(self):
        """Test Taiwan market data validation."""
        # Create test data with various issues
        dates = pd.date_range('2023-01-01', periods=10, freq='B')
        data = pd.DataFrame({
            'valid_symbol': np.random.uniform(50, 150, 10),
            'negative_prices': [-10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'extreme_returns': [100, 150, 200, 80, 120, 110, 105, 115, 125, 130]
        }, index=dates)
        
        validation = self.taiwan_adj.validate_taiwan_market_data(
            data, ['valid_symbol', 'negative_prices', 'extreme_returns', 'missing_symbol']
        )
        
        self.assertIn('valid_symbol', validation['valid_symbols'])
        self.assertIn('missing_symbol', validation['invalid_symbols'])
        self.assertTrue(any('negative_prices' in w for w in validation['warnings']))


class TestMomentumFactors(unittest.TestCase):
    """Test momentum factor calculations."""
    
    def setUp(self):
        self.mock_engine = MockPITEngine()
        self.taiwan_adj = TaiwanMarketAdjustments()
        self.symbols = ['2330.TW', '2317.TW', '1301.TW']
        self.as_of_date = date(2023, 6, 15)
    
    def test_price_momentum_calculator(self):
        """Test price momentum factor calculation."""
        calculator = PriceMomentumCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'price_momentum')
        self.assertEqual(result.date, self.as_of_date)
        self.assertTrue(len(result.values) > 0)
        
        # Check that values are reasonable (momentum should be bounded)
        for symbol, value in result.values.items():
            self.assertFalse(np.isnan(value))
            self.assertGreater(value, -2.0)  # Not extremely negative
            self.assertLess(value, 2.0)      # Not extremely positive
    
    def test_rsi_momentum_calculator(self):
        """Test RSI momentum factor calculation."""
        calculator = RSIMomentumCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'rsi_momentum')
        
        # RSI momentum should be bounded
        for symbol, value in result.values.items():
            self.assertFalse(np.isnan(value))
            self.assertGreater(value, -2.0)
            self.assertLess(value, 2.0)
    
    def test_macd_signal_calculator(self):
        """Test MACD signal strength calculation."""
        calculator = MACDSignalCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'macd_signal')
        
        # MACD signals should be reasonable
        for symbol, value in result.values.items():
            self.assertFalse(np.isnan(value))
            self.assertIsInstance(value, (int, float))


class TestMeanReversionFactors(unittest.TestCase):
    """Test mean reversion factor calculations."""
    
    def setUp(self):
        self.mock_engine = MockPITEngine()
        self.taiwan_adj = TaiwanMarketAdjustments()
        self.symbols = ['2330.TW', '2317.TW']
        self.as_of_date = date(2023, 6, 15)
    
    def test_moving_average_reversion(self):
        """Test moving average reversion calculation."""
        calculator = MovingAverageReversionCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'ma_reversion')
        
        # Reversion signals should be reasonable
        for symbol, value in result.values.items():
            self.assertFalse(np.isnan(value))
            self.assertIsInstance(value, (int, float))
    
    def test_bollinger_band_position(self):
        """Test Bollinger Band position calculation."""
        calculator = BollingerBandPositionCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'bb_position')
        
        # BB position should be bounded
        for symbol, value in result.values.items():
            self.assertFalse(np.isnan(value))
            self.assertGreater(value, -3.0)  # Reasonable bounds
            self.assertLess(value, 3.0)
    
    def test_zscore_reversion(self):
        """Test Z-score reversion calculation."""
        calculator = ZScoreReversionCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'zscore_reversion')
    
    def test_short_term_reversal(self):
        """Test short-term reversal calculation."""
        calculator = ShortTermReversalCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'short_term_reversal')


class TestVolatilityFactors(unittest.TestCase):
    """Test volatility factor calculations."""
    
    def setUp(self):
        self.mock_engine = MockPITEngine()
        self.taiwan_adj = TaiwanMarketAdjustments()
        self.symbols = ['2330.TW', '2317.TW']
        self.as_of_date = date(2023, 6, 15)
    
    def test_realized_volatility(self):
        """Test realized volatility calculation."""
        calculator = RealizedVolatilityCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'realized_volatility')
        
        # Volatility factors should be reasonable
        for symbol, value in result.values.items():
            self.assertFalse(np.isnan(value))
            self.assertIsInstance(value, (int, float))
    
    def test_garch_volatility(self):
        """Test GARCH volatility calculation."""
        calculator = GARCHVolatilityCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'garch_volatility')
    
    def test_taiwan_vix(self):
        """Test Taiwan VIX calculation."""
        calculator = TaiwanVIXCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'taiwan_vix')
    
    def test_volatility_risk_premium(self):
        """Test volatility risk premium calculation."""
        calculator = VolatilityRiskPremiumCalculator(self.mock_engine, self.taiwan_adj)
        
        result = calculator.calculate(self.symbols, self.as_of_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, 'vol_risk_premium')


class TestTechnicalFactorsIntegration(unittest.TestCase):
    """Test technical factors integration and orchestration."""
    
    def setUp(self):
        self.mock_engine = MockPITEngine()
        self.tech_factors = TechnicalFactors(self.mock_engine)
        self.symbols = ['2330.TW', '2317.TW', '1301.TW']
        self.as_of_date = date(2023, 6, 15)
    
    def test_factor_registration(self):
        """Test that all factors are properly registered."""
        metadata = self.tech_factors.get_factor_metadata()
        
        # Should have all technical factors
        expected_factors = {
            'price_momentum', 'rsi_momentum', 'macd_signal',  # Momentum (3)
            'ma_reversion', 'bb_position', 'zscore_reversion', 'short_term_reversal',  # Mean reversion (4)
            'realized_volatility', 'garch_volatility', 'taiwan_vix', 'vol_risk_premium'  # Volatility (4)
        }
        
        self.assertEqual(len(metadata), len(expected_factors))
        for factor_name in expected_factors:
            self.assertIn(factor_name, metadata)
    
    def test_calculate_all_factors(self):
        """Test calculating all technical factors."""
        results = self.tech_factors.calculate_all_factors(
            self.symbols, self.as_of_date, parallel=False
        )
        
        self.assertIsInstance(results, dict)
        self.assertTrue(len(results) > 0)
        
        for factor_name, result in results.items():
            self.assertIsInstance(result, FactorResult)
            self.assertEqual(result.date, self.as_of_date)
            self.assertIsInstance(result.values, dict)
    
    def test_calculate_factor_subset(self):
        """Test calculating specific factor subset."""
        factor_names = ['price_momentum', 'ma_reversion', 'realized_volatility']
        
        results = self.tech_factors.calculate_factor_subset(
            factor_names, self.symbols, self.as_of_date
        )
        
        self.assertEqual(len(results), len(factor_names))
        
        for factor_name in factor_names:
            self.assertIn(factor_name, results)
            self.assertEqual(results[factor_name].factor_name, factor_name)
    
    def test_factor_categories(self):
        """Test factor categorization."""
        categories = self.tech_factors.get_factor_categories()
        
        self.assertIn('momentum', categories)
        self.assertIn('mean_reversion', categories)
        self.assertIn('volatility', categories)
        
        # Check that factors are properly categorized
        self.assertIn('price_momentum', categories['momentum'])
        self.assertIn('ma_reversion', categories['mean_reversion'])
        self.assertIn('realized_volatility', categories['volatility'])
    
    def test_validation(self):
        """Test factor validation."""
        validation_results = self.tech_factors.validate_factor_universe_coverage(
            self.symbols, self.as_of_date, min_coverage=0.5
        )
        
        self.assertIsInstance(validation_results, dict)
        
        for factor_name, is_valid in validation_results.items():
            self.assertIsInstance(is_valid, bool)
    
    def test_performance_tracking(self):
        """Test calculation performance tracking."""
        # Calculate factors to generate stats
        self.tech_factors.calculate_all_factors(
            self.symbols, self.as_of_date, parallel=False
        )
        
        stats = self.tech_factors.get_calculation_stats()
        self.assertIn(self.as_of_date, stats)
        
        date_stats = stats[self.as_of_date]
        self.assertIn('universe_size', date_stats)
        self.assertIn('factors_calculated', date_stats)
        self.assertIn('elapsed_time_seconds', date_stats)
        self.assertIn('average_coverage', date_stats)
    
    def test_export_factor_definitions(self):
        """Test exporting factor definitions."""
        definitions = self.tech_factors.export_factor_definitions()
        
        self.assertIsInstance(definitions, dict)
        
        for factor_name, definition in definitions.items():
            self.assertIn('name', definition)
            self.assertIn('category', definition)
            self.assertIn('description', definition)
            self.assertIn('taiwan_specific', definition)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        self.mock_engine = MockPITEngine()
        self.symbols = ['2330.TW', '2317.TW']
        self.as_of_date = date(2023, 6, 15)
    
    def test_calculate_technical_factors_function(self):
        """Test convenience function for calculating technical factors."""
        results = calculate_technical_factors(
            self.mock_engine, self.symbols, self.as_of_date, parallel=False
        )
        
        self.assertIsInstance(results, dict)
        self.assertTrue(len(results) > 0)
        
        for factor_name, result in results.items():
            self.assertIsInstance(result, FactorResult)


class TestFactorResultProcessing(unittest.TestCase):
    """Test factor result processing and metrics."""
    
    def test_factor_result_initialization(self):
        """Test FactorResult initialization and derived metrics."""
        values = {'2330.TW': 0.15, '2317.TW': -0.05, '1301.TW': 0.08}
        
        metadata = FactorMetadata(
            name='test_factor',
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description='Test factor',
            lookback_days=30,
            data_requirements=[]
        )
        
        result = FactorResult(
            factor_name='test_factor',
            date=date(2023, 6, 15),
            values=values,
            metadata=metadata
        )
        
        # Check that derived metrics are calculated
        self.assertIsNotNone(result.percentile_ranks)
        self.assertIsNotNone(result.z_scores)
        self.assertIsNotNone(result.coverage)
        
        # Check coverage
        self.assertEqual(result.coverage, 1.0)  # All values are valid
        
        # Check percentile ranks
        self.assertTrue(0 <= result.percentile_ranks['2330.TW'] <= 1)
        self.assertTrue(0 <= result.percentile_ranks['2317.TW'] <= 1)
        self.assertTrue(0 <= result.percentile_ranks['1301.TW'] <= 1)
        
        # Highest value should have highest percentile
        max_value_symbol = max(values, key=values.get)
        self.assertEqual(result.percentile_ranks[max_value_symbol], 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.mock_engine = MockPITEngine()
        self.taiwan_adj = TaiwanMarketAdjustments()
        self.as_of_date = date(2023, 6, 15)
    
    def test_empty_symbol_list(self):
        """Test handling of empty symbol list."""
        tech_factors = TechnicalFactors(self.mock_engine)
        
        results = tech_factors.calculate_all_factors([], self.as_of_date)
        
        # Should return empty results without error
        for factor_name, result in results.items():
            self.assertEqual(len(result.values), 0)
    
    def test_insufficient_data(self):
        """Test handling of insufficient historical data."""
        # Create a date very close to start of data
        recent_date = date(2023, 1, 5)  # Very recent date
        symbols = ['2330.TW']
        
        tech_factors = TechnicalFactors(self.mock_engine)
        results = tech_factors.calculate_all_factors(symbols, recent_date)
        
        # Should handle gracefully without crashing
        self.assertIsInstance(results, dict)
    
    def test_invalid_factor_names(self):
        """Test handling of invalid factor names."""
        tech_factors = TechnicalFactors(self.mock_engine)
        
        results = tech_factors.calculate_factor_subset(
            ['invalid_factor'], ['2330.TW'], self.as_of_date
        )
        
        # Should return empty results for invalid factors
        self.assertEqual(len(results), 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Run tests
    unittest.main(verbosity=2)