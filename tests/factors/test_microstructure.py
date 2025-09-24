"""
Comprehensive test suite for microstructure factors.

Tests for Taiwan market microstructure factor calculations including:
- Liquidity factors (4 factors)
- Volume pattern factors (4 factors) 
- Taiwan-specific factors (4 factors)
- Tick data handling and foreign flow analysis
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from factors.microstructure import (
    MicrostructureFactorCalculator, MicrostructureFactors,
    TaiwanMarketSession, TickSizeStructure
)
from factors.liquidity import (
    LiquidityFactors, AverageDailyTurnoverCalculator, BidAskSpreadCalculator,
    PriceImpactCalculator, AmihudIlliquidityCalculator
)
from factors.volume_patterns import (
    VolumePatternFactors, VolumeWeightedMomentumCalculator,
    VolumeBreakoutCalculator, RelativeVolumeCalculator, VolumePriceCorrelationCalculator
)
from factors.taiwan_specific import (
    TaiwanSpecificFactors, ForeignFlowImpactCalculator, MarginTradingRatioCalculator,
    IndexInclusionEffectCalculator, CrossStraitSentimentCalculator
)
from factors.tick_data_handler import (
    TickDataHandler, TickData, TickDataCleaner, IntradayMetricsCalculator,
    TaiwanSessionFilter
)
from factors.foreign_flows import (
    ForeignFlowAnalyzer, ForeignFlowData, FlowDirection, FlowIntensity
)
from factors.base import FactorResult, FactorMetadata, FactorCategory, FactorFrequency


class TestTaiwanMarketSession(unittest.TestCase):
    """Test Taiwan market session utilities."""
    
    def test_taiwan_session_parameters(self):
        """Test Taiwan market session parameters."""
        session = TaiwanMarketSession()
        
        self.assertEqual(session.open_time, "09:00")
        self.assertEqual(session.close_time, "13:30")
        self.assertEqual(session.session_minutes, 270)  # 4.5 hours
        self.assertEqual(session.trading_days_per_year, 245)
        self.assertEqual(session.price_limit, 0.10)
        self.assertEqual(session.settlement_cycle, 2)


class TestTickSizeStructure(unittest.TestCase):
    """Test Taiwan tick size structure."""
    
    def test_tick_size_calculation(self):
        """Test tick size calculation for different price levels."""
        # Test various price levels
        self.assertEqual(TickSizeStructure.get_tick_size(5.0), 0.01)    # < 10
        self.assertEqual(TickSizeStructure.get_tick_size(25.0), 0.05)   # 10-50
        self.assertEqual(TickSizeStructure.get_tick_size(75.0), 0.10)   # 50-100
        self.assertEqual(TickSizeStructure.get_tick_size(250.0), 0.50)  # 100-500
        self.assertEqual(TickSizeStructure.get_tick_size(750.0), 1.00)  # 500-1000
        self.assertEqual(TickSizeStructure.get_tick_size(1500.0), 5.00) # > 1000


class TestTickDataHandler(unittest.TestCase):
    """Test tick data handling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = TickDataHandler(memory_limit_mb=100)
        self.sample_ticks = self._create_sample_tick_data()
    
    def _create_sample_tick_data(self) -> List[Dict]:
        """Create sample tick data for testing."""
        base_time = datetime(2024, 9, 24, 10, 0, 0)  # 10:00 AM
        sample_data = []
        
        for i in range(100):
            tick = {
                'timestamp': base_time + timedelta(seconds=i*10),
                'symbol': '2330.TW',  # Taiwan Semiconductor
                'price': 500.0 + np.random.normal(0, 5),
                'volume': max(1000, int(np.random.normal(5000, 1000))),
                'bid': 499.5 + np.random.normal(0, 5),
                'ask': 500.5 + np.random.normal(0, 5)
            }
            
            # Ensure ask > bid
            if tick['ask'] <= tick['bid']:
                tick['ask'] = tick['bid'] + 0.5
            
            sample_data.append(tick)
        
        return sample_data
    
    def test_tick_data_cleaning(self):
        """Test tick data cleaning process."""
        cleaned_ticks = self.handler.cleaner.clean_tick_data(self.sample_ticks)
        
        # Should have cleaned some but not all ticks
        self.assertGreater(len(cleaned_ticks), 0)
        self.assertLessEqual(len(cleaned_ticks), len(self.sample_ticks))
        
        # All cleaned ticks should be valid
        for tick in cleaned_ticks:
            self.assertIsInstance(tick, TickData)
            self.assertGreater(tick.price, 0)
            self.assertGreaterEqual(tick.volume, 0)
            if tick.bid and tick.ask:
                self.assertLess(tick.bid, tick.ask)
    
    def test_session_filtering(self):
        """Test Taiwan session time filtering."""
        session_filter = TaiwanSessionFilter()
        
        # Test trading hours
        trading_time = datetime(2024, 9, 24, 10, 30)  # 10:30 AM
        non_trading_time = datetime(2024, 9, 24, 15, 0)  # 3:00 PM
        
        self.assertTrue(session_filter.is_trading_time(trading_time))
        self.assertFalse(session_filter.is_trading_time(non_trading_time))
        
        # Test period classification
        opening_time = datetime(2024, 9, 24, 9, 15)
        self.assertTrue(session_filter.is_opening_period(opening_time))
        
        closing_time = datetime(2024, 9, 24, 13, 15)
        self.assertTrue(session_filter.is_closing_period(closing_time))
    
    def test_intraday_metrics_calculation(self):
        """Test intraday metrics calculation."""
        # Create tick data objects
        cleaned_ticks = self.handler.cleaner.clean_tick_data(self.sample_ticks)
        
        if cleaned_ticks:
            # Calculate metrics
            target_date = date(2024, 9, 24)
            metrics = self.handler.metrics_calculator.calculate_metrics(cleaned_ticks, target_date)
            
            self.assertIn('2330.TW', metrics)
            
            symbol_metrics = metrics['2330.TW']
            self.assertIsNotNone(symbol_metrics.vwap)
            self.assertGreater(symbol_metrics.total_volume, 0)
            self.assertGreater(symbol_metrics.tick_count, 0)


class TestLiquidityFactors(unittest.TestCase):
    """Test liquidity factor calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_pit_engine = Mock()
        self.liquidity_factors = LiquidityFactors(self.mock_pit_engine)
        self.test_symbols = ['2330.TW', '2454.TW', '2317.TW']
        self.test_date = date(2024, 9, 24)
    
    def _create_mock_ohlcv_data(self) -> pd.DataFrame:
        """Create mock OHLCV data."""
        dates = pd.date_range(start='2024-01-01', end='2024-09-24', freq='D')
        data = []
        
        for symbol in self.test_symbols:
            for d in dates:
                # Skip weekends (simplified)
                if d.weekday() < 5:
                    data.append({
                        'date': d.date(),
                        'symbol': symbol,
                        'open': 500.0 + np.random.normal(0, 10),
                        'high': 510.0 + np.random.normal(0, 10), 
                        'low': 490.0 + np.random.normal(0, 10),
                        'close': 500.0 + np.random.normal(0, 10),
                        'volume': max(1000, int(np.random.normal(100000, 20000)))
                    })
        
        return pd.DataFrame(data)
    
    def test_average_daily_turnover_calculator(self):
        """Test average daily turnover calculation."""
        # Mock data
        mock_data = self._create_mock_ohlcv_data()
        mock_mcap_data = pd.DataFrame([
            {'date': self.test_date, 'symbol': symbol, 'market_cap': 1e9 + np.random.normal(0, 1e8)}
            for symbol in self.test_symbols
        ])
        
        # Mock PIT engine responses
        def mock_get_data(query):
            if 'OHLCV' in str(query):
                return mock_data
            elif 'MARKET_CAP' in str(query):
                return mock_mcap_data
            return pd.DataFrame()
        
        self.mock_pit_engine.query = mock_get_data
        
        calculator = AverageDailyTurnoverCalculator(self.mock_pit_engine)
        result = calculator.calculate(self.test_symbols, self.test_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, "avg_daily_turnover")
        self.assertGreater(len(result.values), 0)
    
    def test_bid_ask_spread_calculator(self):
        """Test bid-ask spread calculation."""
        # Mock order book data
        mock_order_book = pd.DataFrame([
            {
                'date': self.test_date - timedelta(days=i),
                'symbol': symbol,
                'timestamp': datetime.combine(self.test_date - timedelta(days=i), datetime.min.time()) + timedelta(hours=10),
                'bid': 499.5 + np.random.normal(0, 2),
                'ask': 500.5 + np.random.normal(0, 2)
            }
            for symbol in self.test_symbols
            for i in range(30)
        ])
        
        def mock_get_data(query):
            return mock_order_book
        
        self.mock_pit_engine.query = mock_get_data
        
        calculator = BidAskSpreadCalculator(self.mock_pit_engine)
        result = calculator.calculate(self.test_symbols, self.test_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, "bid_ask_spread")
    
    def test_amihud_illiquidity_calculator(self):
        """Test Amihud illiquidity ratio calculation."""
        mock_data = self._create_mock_ohlcv_data()
        
        # Ensure we have sufficient data
        mock_data = mock_data[mock_data['date'] >= (self.test_date - timedelta(days=252))]
        
        def mock_get_data(query):
            return mock_data
        
        self.mock_pit_engine.query = mock_get_data
        
        calculator = AmihudIlliquidityCalculator(self.mock_pit_engine)
        result = calculator.calculate(self.test_symbols, self.test_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, "amihud_illiquidity")
        
        # Check that values are reasonable (positive for illiquidity measure)
        for value in result.values.values():
            self.assertIsInstance(value, (int, float))


class TestVolumePatternFactors(unittest.TestCase):
    """Test volume pattern factor calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_pit_engine = Mock()
        self.volume_factors = VolumePatternFactors(self.mock_pit_engine)
        self.test_symbols = ['2330.TW', '2454.TW']
        self.test_date = date(2024, 9, 24)
    
    def test_volume_weighted_momentum_calculator(self):
        """Test volume-weighted momentum calculation."""
        # Create mock data with momentum pattern
        mock_data = self._create_momentum_pattern_data()
        
        def mock_get_data(query):
            return mock_data
        
        self.mock_pit_engine.query = mock_get_data
        
        calculator = VolumeWeightedMomentumCalculator(self.mock_pit_engine)
        result = calculator.calculate(self.test_symbols, self.test_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, "volume_weighted_momentum")
    
    def test_volume_breakout_calculator(self):
        """Test volume breakout calculation."""
        mock_data = self._create_breakout_pattern_data()
        
        def mock_get_data(query):
            return mock_data
        
        self.mock_pit_engine.query = mock_get_data
        
        calculator = VolumeBreakoutCalculator(self.mock_pit_engine)
        result = calculator.calculate(self.test_symbols, self.test_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, "volume_breakout")
    
    def _create_momentum_pattern_data(self) -> pd.DataFrame:
        """Create mock data with momentum patterns."""
        dates = pd.date_range(start='2024-01-01', end='2024-09-24', freq='D')
        data = []
        
        for symbol in self.test_symbols:
            base_price = 500.0
            trend = 1.0  # Positive momentum
            
            for i, d in enumerate(dates):
                if d.weekday() < 5:  # Weekdays only
                    # Create trending pattern
                    price_change = trend * (1 + np.random.normal(0, 0.01))
                    base_price *= price_change
                    
                    # Higher volume during trend
                    base_volume = 100000
                    if price_change > 1:  # Up days get more volume
                        volume = base_volume * (1.5 + np.random.normal(0, 0.3))
                    else:
                        volume = base_volume * (0.8 + np.random.normal(0, 0.2))
                    
                    data.append({
                        'date': d.date(),
                        'symbol': symbol,
                        'open': base_price,
                        'high': base_price * 1.02,
                        'low': base_price * 0.98,
                        'close': base_price,
                        'volume': max(1000, int(volume))
                    })
        
        return pd.DataFrame(data)
    
    def _create_breakout_pattern_data(self) -> pd.DataFrame:
        """Create mock data with volume breakout patterns."""
        dates = pd.date_range(start='2024-08-01', end='2024-09-24', freq='D')
        data = []
        
        for symbol in self.test_symbols:
            base_price = 500.0
            base_volume = 100000
            
            for i, d in enumerate(dates):
                if d.weekday() < 5:  # Weekdays only
                    # Create volume spike near end of period
                    days_from_end = (dates[-1] - d).days
                    
                    if days_from_end < 5:  # Last 5 days have volume breakout
                        volume_multiplier = 3.0 + np.random.normal(0, 0.5)
                        price_move = 1.05  # Price breakout with volume
                    else:
                        volume_multiplier = 1.0 + np.random.normal(0, 0.2)
                        price_move = 1.0 + np.random.normal(0, 0.01)
                    
                    base_price *= price_move
                    volume = max(1000, int(base_volume * volume_multiplier))
                    
                    data.append({
                        'date': d.date(),
                        'symbol': symbol,
                        'open': base_price,
                        'high': base_price * 1.02,
                        'low': base_price * 0.98,
                        'close': base_price,
                        'volume': volume
                    })
        
        return pd.DataFrame(data)


class TestTaiwanSpecificFactors(unittest.TestCase):
    """Test Taiwan-specific factor calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_pit_engine = Mock()
        self.taiwan_factors = TaiwanSpecificFactors(self.mock_pit_engine)
        self.test_symbols = ['2330.TW', '2454.TW']
        self.test_date = date(2024, 9, 24)
    
    def test_foreign_flow_impact_calculator(self):
        """Test foreign flow impact calculation."""
        # Mock foreign flow data
        mock_flow_data = pd.DataFrame([
            {
                'date': self.test_date - timedelta(days=i),
                'symbol': symbol,
                'foreign_buy_value': 1e6 + np.random.normal(0, 5e5),
                'foreign_sell_value': 8e5 + np.random.normal(0, 4e5),
                'foreign_net_value': 2e5 + np.random.normal(0, 3e5),
                'foreign_ownership_pct': 0.25 + np.random.normal(0, 0.05)
            }
            for symbol in self.test_symbols
            for i in range(60)
        ])
        
        def mock_get_data(query):
            return mock_flow_data
        
        self.mock_pit_engine.query = mock_get_data
        
        calculator = ForeignFlowImpactCalculator(self.mock_pit_engine)
        result = calculator.calculate(self.test_symbols, self.test_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, "foreign_flow_impact")
    
    def test_cross_strait_sentiment_calculator(self):
        """Test cross-strait sentiment calculation."""
        # Mock sentiment data
        mock_sentiment_data = pd.DataFrame([
            {
                'date': self.test_date - timedelta(days=i),
                'sentiment_score': np.random.normal(0, 0.3),  # Centered around neutral
                'news_count': max(1, int(np.random.normal(10, 3))),
                'keyword_mentions': {'cooperation': 5, 'tension': 2}
            }
            for i in range(30)
        ])
        
        def mock_get_data(query):
            if 'NEWS_SENTIMENT' in str(query):
                return mock_sentiment_data
            return pd.DataFrame()
        
        self.mock_pit_engine.query = mock_get_data
        
        calculator = CrossStraitSentimentCalculator(self.mock_pit_engine)
        result = calculator.calculate(self.test_symbols, self.test_date)
        
        self.assertIsInstance(result, FactorResult)
        self.assertEqual(result.factor_name, "cross_strait_sentiment")
        
        # Should have values for all symbols (market-wide factor)
        self.assertEqual(len(result.values), len(self.test_symbols))


class TestForeignFlowAnalyzer(unittest.TestCase):
    """Test foreign flow analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ForeignFlowAnalyzer(lookback_days=126)
        self.test_date = date(2024, 9, 24)
        self.sample_flows = self._create_sample_flow_data()
    
    def _create_sample_flow_data(self) -> List[ForeignFlowData]:
        """Create sample foreign flow data."""
        flows = []
        
        for i in range(60):
            flow_date = self.test_date - timedelta(days=i)
            
            flows.append(ForeignFlowData(
                date=flow_date,
                symbol='2330.TW',
                foreign_buy_value=1e6 + np.random.normal(0, 3e5),
                foreign_sell_value=8e5 + np.random.normal(0, 2e5),
                foreign_net_value=2e5 + np.random.normal(0, 1e5),
                foreign_ownership_pct=0.30 + np.random.normal(0, 0.02)
            ))
        
        return flows
    
    def test_flow_analysis(self):
        """Test comprehensive flow analysis."""
        results = self.analyzer.analyze_flows(self.sample_flows, self.test_date)
        
        self.assertIn('2330.TW', results)
        
        result = results['2330.TW']
        self.assertEqual(result.symbol, '2330.TW')
        self.assertEqual(result.analysis_date, self.test_date)
        self.assertIsInstance(result.flow_direction, FlowDirection)
        self.assertIsInstance(result.flow_intensity, FlowIntensity)
        
        # Check that metrics are within reasonable ranges
        self.assertGreaterEqual(result.flow_percentile, 0.0)
        self.assertLessEqual(result.flow_percentile, 1.0)
        self.assertGreaterEqual(result.momentum_score, 0.0)
        self.assertLessEqual(result.momentum_score, 1.0)
    
    def test_flow_direction_classification(self):
        """Test flow direction classification."""
        # Test strong buy flow
        direction = self.analyzer._classify_flow_direction(5e6)
        self.assertEqual(direction, FlowDirection.NET_BUY)
        
        # Test strong sell flow
        direction = self.analyzer._classify_flow_direction(-5e6)
        self.assertEqual(direction, FlowDirection.NET_SELL)
        
        # Test balanced flow
        direction = self.analyzer._classify_flow_direction(5e5)
        self.assertEqual(direction, FlowDirection.BALANCED)
    
    def test_flow_intensity_classification(self):
        """Test flow intensity classification."""
        # Create sample flow series
        flows = pd.Series([1e6, 1.2e6, 0.8e6, 1.5e6, 0.9e6] * 10)
        
        # Test different intensity levels
        normal_flow = 1e6
        intensity = self.analyzer._classify_flow_intensity(normal_flow, flows)
        self.assertIn(intensity, [FlowIntensity.LOW, FlowIntensity.MODERATE])
        
        # Test very high flow
        extreme_flow = 5e6
        intensity = self.analyzer._classify_flow_intensity(extreme_flow, flows)
        self.assertIn(intensity, [FlowIntensity.HIGH, FlowIntensity.VERY_HIGH])


class TestIntegration(unittest.TestCase):
    """Integration tests for microstructure factors."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_pit_engine = Mock()
        
    def test_complete_microstructure_calculation(self):
        """Test complete microstructure factor calculation pipeline."""
        # Create comprehensive mock data
        symbols = ['2330.TW', '2454.TW', '2317.TW']
        test_date = date(2024, 9, 24)
        
        # Mock all data types
        mock_ohlcv = self._create_comprehensive_ohlcv_data(symbols, test_date)
        mock_flows = self._create_comprehensive_flow_data(symbols, test_date)
        mock_sentiment = self._create_comprehensive_sentiment_data(test_date)
        
        def mock_query(query):
            query_str = str(query)
            if 'OHLCV' in query_str:
                return mock_ohlcv
            elif 'FOREIGN_FLOWS' in query_str:
                return mock_flows
            elif 'NEWS_SENTIMENT' in query_str:
                return mock_sentiment
            elif 'ORDER_BOOK' in query_str:
                return self._create_order_book_data(symbols, test_date)
            elif 'MARKET_CAP' in query_str:
                return self._create_market_cap_data(symbols, test_date)
            return pd.DataFrame()
        
        self.mock_pit_engine.query = mock_query
        
        # Initialize all factor calculators
        liquidity_factors = LiquidityFactors(self.mock_pit_engine)
        volume_factors = VolumePatternFactors(self.mock_pit_engine)
        taiwan_factors = TaiwanSpecificFactors(self.mock_pit_engine)
        
        # Calculate all factors
        liquidity_results = liquidity_factors.calculate_all_factors(symbols, test_date)
        volume_results = volume_factors.calculate_all_factors(symbols, test_date)
        taiwan_results = taiwan_factors.calculate_all_factors(symbols, test_date)
        
        # Verify we got results from all categories
        self.assertGreater(len(liquidity_results), 0)
        self.assertGreater(len(volume_results), 0)
        self.assertGreater(len(taiwan_results), 0)
        
        # Verify all results are FactorResult objects
        for category_results in [liquidity_results, volume_results, taiwan_results]:
            for factor_name, result in category_results.items():
                self.assertIsInstance(result, FactorResult)
                self.assertEqual(result.date, test_date)
                self.assertIsNotNone(result.values)
    
    def _create_comprehensive_ohlcv_data(self, symbols: List[str], end_date: date) -> pd.DataFrame:
        """Create comprehensive OHLCV data for integration testing."""
        start_date = end_date - timedelta(days=252)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for symbol in symbols:
            base_price = 500.0
            for d in dates:
                if d.weekday() < 5:  # Weekdays only
                    # Add some realistic price movement
                    price_change = 1 + np.random.normal(0, 0.02)
                    base_price *= price_change
                    
                    data.append({
                        'date': d.date(),
                        'symbol': symbol,
                        'open': base_price * (1 + np.random.normal(0, 0.005)),
                        'high': base_price * (1 + abs(np.random.normal(0, 0.01))),
                        'low': base_price * (1 - abs(np.random.normal(0, 0.01))),
                        'close': base_price,
                        'volume': max(10000, int(np.random.normal(200000, 50000)))
                    })
        
        return pd.DataFrame(data)
    
    def _create_comprehensive_flow_data(self, symbols: List[str], end_date: date) -> pd.DataFrame:
        """Create comprehensive foreign flow data."""
        start_date = end_date - timedelta(days=126)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for symbol in symbols:
            for d in dates:
                if d.weekday() < 5:  # Weekdays only
                    buy_value = max(0, np.random.normal(1e6, 3e5))
                    sell_value = max(0, np.random.normal(8e5, 2e5))
                    
                    data.append({
                        'date': d.date(),
                        'symbol': symbol,
                        'foreign_buy_value': buy_value,
                        'foreign_sell_value': sell_value,
                        'foreign_net_value': buy_value - sell_value,
                        'foreign_ownership_pct': max(0.1, min(0.4, 0.25 + np.random.normal(0, 0.02)))
                    })
        
        return pd.DataFrame(data)
    
    def _create_comprehensive_sentiment_data(self, end_date: date) -> pd.DataFrame:
        """Create comprehensive sentiment data."""
        start_date = end_date - timedelta(days=63)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for d in dates:
            data.append({
                'date': d.date(),
                'sentiment_score': np.random.normal(0, 0.3),
                'news_count': max(1, int(np.random.normal(8, 2))),
                'keyword_mentions': {
                    'cooperation': max(0, int(np.random.normal(3, 1))),
                    'tension': max(0, int(np.random.normal(2, 1)))
                }
            })
        
        return pd.DataFrame(data)
    
    def _create_order_book_data(self, symbols: List[str], end_date: date) -> pd.DataFrame:
        """Create order book data."""
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for symbol in symbols:
            for d in dates:
                if d.weekday() < 5:
                    mid_price = 500.0 + np.random.normal(0, 10)
                    spread = abs(np.random.normal(1.0, 0.3))
                    
                    data.append({
                        'date': d.date(),
                        'symbol': symbol,
                        'timestamp': datetime.combine(d.date(), datetime.min.time()) + timedelta(hours=10),
                        'bid': mid_price - spread/2,
                        'ask': mid_price + spread/2
                    })
        
        return pd.DataFrame(data)
    
    def _create_market_cap_data(self, symbols: List[str], end_date: date) -> pd.DataFrame:
        """Create market cap data."""
        data = []
        base_caps = {'2330.TW': 15e12, '2454.TW': 2e12, '2317.TW': 1e12}  # TWD
        
        for symbol in symbols:
            data.append({
                'date': end_date,
                'symbol': symbol,
                'market_cap': base_caps.get(symbol, 1e12) * (1 + np.random.normal(0, 0.1))
            })
        
        return pd.DataFrame(data)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)