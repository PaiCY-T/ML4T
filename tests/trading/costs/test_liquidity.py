"""
Tests for liquidity analysis and capacity modeling components.

This module tests the liquidity analysis engine including ADV calculations,
capacity constraints, liquidity monitoring, and execution schedule optimization.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

from src.trading.costs.liquidity import (
    LiquidityAnalyzer,
    LiquidityMetrics,
    CapacityConstraints,
    LiquidityAlert,
    LiquidityTier,
    LiquidityRegime,
    create_liquidity_analyzer
)


class TestLiquidityMetrics:
    """Test liquidity metrics data structure."""
    
    def test_basic_metrics_creation(self):
        """Test basic liquidity metrics creation."""
        metrics = LiquidityMetrics(
            symbol="2330.TW",
            date=date.today(),
            adv_20d=500_000,
            adv_60d=480_000,
            lav_20d=520_000,
            current_volume=400_000,
            volume_ratio=0.8,
            shares_outstanding=25_930_000_000,
            liquidity_tier=LiquidityTier.LIQUID,
            liquidity_score=0.75
        )
        
        assert metrics.symbol == "2330.TW"
        assert metrics.adv_20d == 500_000
        assert metrics.liquidity_tier == LiquidityTier.LIQUID
        assert metrics.liquidity_score == 0.75
        
        # Test derived calculations
        expected_turnover = 400_000 / 25_930_000_000
        assert abs(metrics.daily_turnover_rate - expected_turnover) < 1e-10
        
        expected_avg_turnover = 500_000 / 25_930_000_000
        assert abs(metrics.avg_turnover_rate - expected_avg_turnover) < 1e-10


class TestLiquidityAnalyzer:
    """Test liquidity analyzer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = LiquidityAnalyzer(
            adv_window=20,
            lav_threshold=0.3,
            capacity_threshold=0.10,
            min_liquidity_score=0.3
        )
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic volume data
        base_volume = 100_000
        volumes = []
        for i in range(60):
            # Add some zero volume days (5%)
            if i % 20 == 0:
                volumes.append(0)
            else:
                vol = base_volume * (1 + np.random.normal(0, 0.3))
                volumes.append(max(0, vol))
        
        self.volume_data = pd.Series(volumes, index=dates)
        
        # Generate price data
        base_price = 100.0
        prices = [base_price + np.random.normal(0, 2) for _ in range(60)]
        self.price_data = pd.Series(prices, index=dates)
        
        self.market_data = {
            'shares_outstanding': 1_000_000_000,
            'avg_trade_size': 1000
        }
    
    def test_liquidity_metrics_calculation(self):
        """Test comprehensive liquidity metrics calculation."""
        metrics = self.analyzer.calculate_liquidity_metrics(
            symbol="TEST.TW",
            volume_data=self.volume_data,
            price_data=self.price_data,
            market_data=self.market_data
        )
        
        assert isinstance(metrics, LiquidityMetrics)
        assert metrics.symbol == "TEST.TW"
        
        # Check ADV calculations
        assert metrics.adv_20d > 0
        assert metrics.adv_60d > 0
        assert metrics.lav_20d > 0
        
        # LAV should generally be >= ADV (excludes low volume days)
        assert metrics.lav_20d >= metrics.adv_20d * 0.8  # Allow some variance
        
        # Check current metrics
        assert metrics.current_volume >= 0
        assert metrics.volume_ratio >= 0
        
        # Check zero volume day percentage
        assert 0 <= metrics.zero_volume_days_pct <= 1
        
        # Check capacity calculations
        assert metrics.daily_capacity_shares > 0
        assert metrics.daily_capacity_twd > 0
        
        # Check liquidity classification
        assert isinstance(metrics.liquidity_tier, LiquidityTier)
        assert 0 <= metrics.liquidity_score <= 1
    
    def test_adv_calculation_accuracy(self):
        """Test ADV calculation accuracy."""
        # Create simple test data
        volumes = [100] * 20 + [200] * 20 + [150] * 20
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        volume_series = pd.Series(volumes, index=dates)
        price_series = pd.Series([100] * 60, index=dates)
        
        metrics = self.analyzer.calculate_liquidity_metrics(
            symbol="TEST.TW",
            volume_data=volume_series,
            price_data=price_series
        )
        
        # Last 20 days average should be 150
        assert abs(metrics.adv_20d - 150) < 0.01
        
        # 60-day average should be (100*20 + 200*20 + 150*20) / 60 = 150
        assert abs(metrics.adv_60d - 150) < 0.01
    
    def test_lav_calculation(self):
        """Test LAV (Liquidity Adjusted Volume) calculation."""
        # Create data with some low-volume days
        volumes = [1000] * 10 + [100] * 5 + [1000] * 25  # Mix of high and low volume
        dates = pd.date_range('2024-01-01', periods=40, freq='D')
        volume_series = pd.Series(volumes, index=dates)
        price_series = pd.Series([100] * 40, index=dates)
        
        metrics = self.analyzer.calculate_liquidity_metrics(
            symbol="TEST.TW",
            volume_data=volume_series,
            price_data=price_series
        )
        
        # LAV should exclude low volume days (< 30% of ADV)
        # ADV = (1000*10 + 100*5 + 1000*20) / 30 (last 20 days) = (5000 + 20000) / 20 = 1250
        # LAV should exclude days with < 375 volume, so should be close to 1000
        expected_adv = (100 * 5 + 1000 * 15) / 20  # Last 20 days
        assert abs(metrics.adv_20d - expected_adv) < 1
        
        # LAV should be higher since it excludes low volume days
        assert metrics.lav_20d >= metrics.adv_20d
    
    def test_liquidity_score_calculation(self):
        """Test liquidity score calculation components."""
        # High liquidity scenario
        high_vol = [1_000_000] * 30
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        high_volume_series = pd.Series(high_vol, index=dates)
        price_series = pd.Series([100] * 30, index=dates)
        
        high_metrics = self.analyzer.calculate_liquidity_metrics(
            symbol="HIGH.TW",
            volume_data=high_volume_series,
            price_data=price_series
        )
        
        # Low liquidity scenario
        low_vol = [1_000] * 30
        low_volume_series = pd.Series(low_vol, index=dates)
        
        low_metrics = self.analyzer.calculate_liquidity_metrics(
            symbol="LOW.TW",
            volume_data=low_volume_series,
            price_data=price_series
        )
        
        # High volume should have higher liquidity score
        assert high_metrics.liquidity_score > low_metrics.liquidity_score
        assert high_metrics.liquidity_tier.value in ['very_liquid', 'liquid']
        assert low_metrics.liquidity_tier.value in ['illiquid', 'very_illiquid']
    
    def test_capacity_constraints_calculation(self):
        """Test capacity constraints calculation."""
        metrics = self.analyzer.calculate_liquidity_metrics(
            symbol="TEST.TW",
            volume_data=self.volume_data,
            price_data=self.price_data,
            market_data=self.market_data
        )
        
        constraints = self.analyzer.calculate_capacity_constraints(
            symbol="TEST.TW",
            liquidity_metrics=metrics,
            strategy_horizon_days=3,
            max_impact_bps=50.0
        )
        
        assert isinstance(constraints, CapacityConstraints)
        assert constraints.symbol == "TEST.TW"
        
        # Check volume constraints
        assert constraints.max_daily_volume_shares > 0
        assert constraints.max_daily_volume_twd > 0
        
        # Strategy capacity should be daily capacity * horizon
        expected_strategy_shares = constraints.max_daily_volume_shares * 3
        assert abs(metrics.strategy_capacity_shares - expected_strategy_shares) < 1000
        
        # Check participation rate is within reasonable bounds
        assert 0 < constraints.max_participation_rate <= 0.5  # Max 50%
        
        # Check execution time constraints
        assert constraints.min_execution_days >= 1.0
        assert constraints.max_execution_days >= constraints.min_execution_days
        
        # Check impact constraints
        assert constraints.max_impact_bps > 0
        assert constraints.impact_budget_bps <= constraints.max_impact_bps
    
    def test_liquidity_tier_adjustments(self):
        """Test liquidity tier adjustments in capacity calculation."""
        # Create metrics for different liquidity tiers
        base_metrics = LiquidityMetrics(
            symbol="TEST.TW",
            date=date.today(),
            adv_20d=100_000,
            adv_60d=100_000,
            lav_20d=100_000,
            current_volume=100_000,
            volume_ratio=1.0,
            daily_capacity_shares=10_000,
            daily_capacity_twd=1_000_000,
            liquidity_tier=LiquidityTier.LIQUID,
            liquidity_score=0.6
        )
        
        # Test liquid stock constraints
        liquid_constraints = self.analyzer.calculate_capacity_constraints(
            symbol="LIQUID.TW",
            liquidity_metrics=base_metrics
        )
        
        # Test illiquid stock constraints
        illiquid_metrics = base_metrics
        illiquid_metrics.liquidity_tier = LiquidityTier.ILLIQUID
        illiquid_metrics.liquidity_score = 0.2
        
        illiquid_constraints = self.analyzer.calculate_capacity_constraints(
            symbol="ILLIQUID.TW",
            liquidity_metrics=illiquid_metrics
        )
        
        # Illiquid stocks should have more conservative constraints
        assert illiquid_constraints.max_participation_rate < liquid_constraints.max_participation_rate
    
    def test_liquidity_alerts_monitoring(self):
        """Test liquidity alert monitoring."""
        symbols = ["LIQUID.TW", "ILLIQUID.TW"]
        
        # Current positions
        current_positions = {
            "LIQUID.TW": 50_000,
            "ILLIQUID.TW": 20_000
        }
        
        # Liquidity metrics
        liquidity_metrics = {
            "LIQUID.TW": LiquidityMetrics(
                symbol="LIQUID.TW",
                date=date.today(),
                adv_20d=500_000,
                adv_60d=500_000,
                lav_20d=500_000,
                current_volume=400_000,
                volume_ratio=0.8,
                liquidity_tier=LiquidityTier.LIQUID,
                liquidity_score=0.75
            ),
            "ILLIQUID.TW": LiquidityMetrics(
                symbol="ILLIQUID.TW",
                date=date.today(),
                adv_20d=10_000,
                adv_60d=10_000,
                lav_20d=10_000,
                current_volume=2_000,
                volume_ratio=0.2,
                liquidity_tier=LiquidityTier.ILLIQUID,
                liquidity_score=0.2  # Below threshold
            )
        }
        
        # Capacity constraints
        capacity_constraints = {}
        for symbol, metrics in liquidity_metrics.items():
            capacity_constraints[symbol] = self.analyzer.calculate_capacity_constraints(
                symbol=symbol,
                liquidity_metrics=metrics
            )
        
        alerts = self.analyzer.monitor_liquidity_alerts(
            symbols=symbols,
            current_positions=current_positions,
            liquidity_metrics=liquidity_metrics,
            capacity_constraints=capacity_constraints
        )
        
        assert isinstance(alerts, list)
        
        # Should have alerts for illiquid stock
        illiquid_alerts = [a for a in alerts if a.symbol == "ILLIQUID.TW"]
        assert len(illiquid_alerts) > 0
        
        # Check alert properties
        for alert in alerts:
            assert isinstance(alert, LiquidityAlert)
            assert alert.symbol in symbols
            assert alert.severity in ['low', 'medium', 'high', 'critical']
            assert alert.alert_type is not None
            assert alert.message is not None
            assert alert.recommended_action is not None
    
    def test_execution_schedule_optimization(self):
        """Test execution schedule optimization."""
        metrics = LiquidityMetrics(
            symbol="TEST.TW",
            date=date.today(),
            adv_20d=100_000,
            adv_60d=100_000,
            lav_20d=100_000,
            current_volume=100_000,
            volume_ratio=1.0,
            liquidity_tier=LiquidityTier.MODERATE,
            liquidity_score=0.5
        )
        
        constraints = self.analyzer.calculate_capacity_constraints(
            symbol="TEST.TW",
            liquidity_metrics=metrics
        )
        
        # Test small order (should complete in 1 day)
        small_schedule = self.analyzer.optimize_execution_schedule(
            symbol="TEST.TW",
            target_shares=5_000,
            liquidity_metrics=metrics,
            constraints=constraints
        )
        
        assert small_schedule['feasible'] is True
        assert small_schedule['execution_days'] >= 1
        assert len(small_schedule['daily_schedule']) == small_schedule['execution_days']
        
        # Test large order (should take multiple days)
        large_schedule = self.analyzer.optimize_execution_schedule(
            symbol="TEST.TW",
            target_shares=50_000,  # Large relative to capacity
            liquidity_metrics=metrics,
            constraints=constraints
        )
        
        assert large_schedule['execution_days'] > 1
        assert large_schedule['avg_daily_participation_rate'] <= constraints.max_participation_rate * 1.1
        
        # Check daily schedule structure
        for day_info in large_schedule['daily_schedule']:
            assert 'day' in day_info
            assert 'target_shares' in day_info
            assert 'participation_rate' in day_info
            assert 'estimated_impact_bps' in day_info
            assert day_info['target_shares'] > 0
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with insufficient data
        short_data = pd.Series([1000, 2000, 3000], index=pd.date_range('2024-01-01', periods=3, freq='D'))
        price_data = pd.Series([100, 101, 102], index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        with pytest.raises(ValueError, match="Insufficient data"):
            self.analyzer.calculate_liquidity_metrics(
                symbol="SHORT.TW",
                volume_data=short_data,
                price_data=price_data
            )
        
        # Test with all zero volumes
        zero_volumes = pd.Series([0] * 30, index=pd.date_range('2024-01-01', periods=30, freq='D'))
        zero_prices = pd.Series([100] * 30, index=pd.date_range('2024-01-01', periods=30, freq='D'))
        
        metrics = self.analyzer.calculate_liquidity_metrics(
            symbol="ZERO.TW",
            volume_data=zero_volumes,
            price_data=zero_prices
        )
        
        assert metrics.adv_20d == 0
        assert metrics.lav_20d == 0
        assert metrics.liquidity_tier == LiquidityTier.VERY_ILLIQUID
        assert metrics.zero_volume_days_pct == 1.0


class TestFactoryFunctions:
    """Test factory functions for analyzer creation."""
    
    def test_create_liquidity_analyzer_default(self):
        """Test default analyzer creation."""
        analyzer = create_liquidity_analyzer()
        
        assert isinstance(analyzer, LiquidityAnalyzer)
        assert analyzer.adv_window == 20
        assert analyzer.capacity_threshold == 0.10
        assert analyzer.min_liquidity_score == 0.3
    
    def test_create_liquidity_analyzer_conservative(self):
        """Test conservative analyzer creation."""
        analyzer = create_liquidity_analyzer(conservative=True)
        
        # Conservative settings should be more restrictive
        assert analyzer.adv_window >= 20
        assert analyzer.capacity_threshold <= 0.10
        assert analyzer.min_liquidity_score >= 0.3
    
    def test_create_liquidity_analyzer_custom(self):
        """Test custom parameter analyzer creation."""
        custom_params = {
            'adv_window': 30,
            'capacity_threshold': 0.05,
            'min_liquidity_score': 0.5
        }
        
        analyzer = create_liquidity_analyzer(custom_params=custom_params)
        
        assert analyzer.adv_window == 30
        assert analyzer.capacity_threshold == 0.05
        assert analyzer.min_liquidity_score == 0.5


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = LiquidityAnalyzer()
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large dataset (2 years of data)
        dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
        np.random.seed(42)
        
        volumes = [max(0, 100000 * (1 + np.random.normal(0, 0.3))) for _ in range(len(dates))]
        prices = [100 + np.random.normal(0, 2) for _ in range(len(dates))]
        
        volume_data = pd.Series(volumes, index=dates)
        price_data = pd.Series(prices, index=dates)
        
        start_time = datetime.now()
        
        metrics = self.analyzer.calculate_liquidity_metrics(
            symbol="LARGE.TW",
            volume_data=volume_data,
            price_data=price_data
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time (< 1 second for 2 years of data)
        assert execution_time < 1.0
        assert isinstance(metrics, LiquidityMetrics)
    
    def test_multiple_symbol_analysis(self):
        """Test analysis of multiple symbols."""
        symbols = [f"TEST{i}.TW" for i in range(10)]
        
        # Generate data for each symbol
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        
        start_time = datetime.now()
        
        all_metrics = {}
        for symbol in symbols:
            np.random.seed(hash(symbol) % 1000)  # Different seed per symbol
            volumes = [max(0, 50000 * (1 + np.random.normal(0, 0.3))) for _ in range(60)]
            prices = [100 + np.random.normal(0, 2) for _ in range(60)]
            
            volume_data = pd.Series(volumes, index=dates)
            price_data = pd.Series(prices, index=dates)
            
            metrics = self.analyzer.calculate_liquidity_metrics(
                symbol=symbol,
                volume_data=volume_data,
                price_data=price_data
            )
            all_metrics[symbol] = metrics
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Should complete all symbols in reasonable time
        assert execution_time < 5.0  # 5 seconds for 10 symbols
        assert len(all_metrics) == 10
        
        # Each symbol should have different metrics
        scores = [m.liquidity_score for m in all_metrics.values()]
        assert len(set(scores)) > 1  # Should have variety


if __name__ == "__main__":
    pytest.main([__file__])