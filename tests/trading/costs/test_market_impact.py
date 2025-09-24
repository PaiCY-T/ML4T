"""
Tests for market impact modeling components.

This module tests the Taiwan market impact models including temporary and
permanent impact calculations, regime analysis, and integration with
real market data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch

from src.trading.costs.market_impact import (
    TaiwanMarketImpactModel,
    MarketImpactParameters,
    ImpactCalculationResult,
    ImpactComponent,
    ImpactRegime,
    PortfolioImpactAnalyzer,
    create_taiwan_impact_model
)


class TestMarketImpactParameters:
    """Test market impact parameter configurations."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = MarketImpactParameters()
        
        assert params.temp_impact_coeff == 0.35
        assert params.perm_impact_coeff == 0.20
        assert params.temp_size_exponent == 0.65
        assert params.perm_size_exponent == 0.5
        assert params.liquidity_penalty_threshold == 0.10
        
        # Check session multipliers
        assert 'morning_open' in params.session_multipliers
        assert params.session_multipliers['morning_open'] == 1.3
        
        # Check market cap adjustments
        assert 'large_cap' in params.market_cap_adjustments
        assert params.market_cap_adjustments['large_cap'] == 0.8
    
    def test_parameter_customization(self):
        """Test custom parameter configuration."""
        custom_session_multipliers = {'morning_open': 1.5}
        
        params = MarketImpactParameters(
            temp_impact_coeff=0.4,
            session_multipliers=custom_session_multipliers
        )
        
        assert params.temp_impact_coeff == 0.4
        assert params.session_multipliers['morning_open'] == 1.5


class TestTaiwanMarketImpactModel:
    """Test Taiwan market impact model calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = TaiwanMarketImpactModel()
        self.sample_timestamp = datetime(2024, 6, 15, 10, 30)  # Mid-morning
    
    def test_basic_impact_calculation(self):
        """Test basic market impact calculation."""
        result = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=self.sample_timestamp
        )
        
        assert isinstance(result, ImpactCalculationResult)
        assert result.symbol == "2330.TW"
        assert result.order_size == 10_000
        assert result.participation_rate == 0.02  # 10K / 500K
        
        # Impact should be positive
        assert result.temporary_impact_bps > 0
        assert result.permanent_impact_bps > 0
        assert result.total_impact_bps > 0
        
        # Total impact should equal sum of components
        assert abs(result.total_impact_bps - (result.temporary_impact_bps + result.permanent_impact_bps)) < 0.01
        
        # TWD values should be reasonable
        assert result.total_impact_twd > 0
        trade_value = 10_000 * 500.0
        expected_twd = (result.total_impact_bps / 10_000) * trade_value
        assert abs(result.total_impact_twd - expected_twd) < 1.0
    
    def test_regime_determination(self):
        """Test market regime determination."""
        # Normal regime
        result_normal = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=5_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.20,
            timestamp=self.sample_timestamp
        )
        assert result_normal.regime == ImpactRegime.NORMAL
        
        # Volatile regime
        result_volatile = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=5_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.45,  # High volatility
            timestamp=self.sample_timestamp
        )
        assert result_volatile.regime == ImpactRegime.VOLATILE
        
        # Illiquid regime (high participation)
        result_illiquid = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=100_000,  # Large order
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.20,
            timestamp=self.sample_timestamp
        )
        assert result_illiquid.regime == ImpactRegime.ILLIQUID
    
    def test_session_effects(self):
        """Test trading session effects on impact."""
        # Morning open (high impact)
        morning_open = datetime(2024, 6, 15, 9, 15)
        result_open = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=morning_open
        )
        
        # Mid-morning (normal impact)
        mid_morning = datetime(2024, 6, 15, 10, 30)
        result_mid = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=mid_morning
        )
        
        # Opening should have higher impact than mid-morning
        assert result_open.temporary_impact_bps > result_mid.temporary_impact_bps
    
    def test_size_scaling(self):
        """Test impact scaling with order size."""
        base_size = 10_000
        large_size = 50_000
        
        result_base = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=base_size,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=self.sample_timestamp
        )
        
        result_large = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=large_size,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=self.sample_timestamp
        )
        
        # Larger orders should have higher impact
        assert result_large.total_impact_bps > result_base.total_impact_bps
        
        # Impact should scale non-linearly (less than proportional)
        size_ratio = large_size / base_size
        impact_ratio = result_large.total_impact_bps / result_base.total_impact_bps
        assert impact_ratio < size_ratio  # Non-linear scaling
        assert impact_ratio > 1.0  # But still increasing
    
    def test_volatility_effects(self):
        """Test volatility effects on impact."""
        low_vol_result = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.15,  # Low volatility
            timestamp=self.sample_timestamp
        )
        
        high_vol_result = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.35,  # High volatility
            timestamp=self.sample_timestamp
        )
        
        # Higher volatility should increase impact
        assert high_vol_result.temporary_impact_bps > low_vol_result.temporary_impact_bps
        assert high_vol_result.total_impact_bps > low_vol_result.total_impact_bps
    
    def test_market_cap_tier_effects(self):
        """Test market cap tier adjustments."""
        # Large cap stock (high price)
        large_cap_result = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,  # High price suggests large cap
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=self.sample_timestamp
        )
        
        # Small cap stock (low price)
        small_cap_result = self.model.calculate_impact(
            symbol="1234.TW",
            order_size=10_000,
            price=30.0,  # Low price suggests small cap
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=self.sample_timestamp
        )
        
        # Small cap should have higher impact
        assert small_cap_result.total_impact_bps > large_cap_result.total_impact_bps
        assert large_cap_result.market_cap_tier == 'large_cap'
        assert small_cap_result.market_cap_tier == 'small_cap'
    
    def test_decay_function(self):
        """Test temporary impact decay function."""
        result = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=self.sample_timestamp
        )
        
        decay_fn = self.model.get_decay_function(result)
        
        # Test decay behavior
        assert decay_fn(0) == 1.0  # No decay at t=0
        assert 0 < decay_fn(30) < 1.0  # Partial decay at 30 minutes
        assert decay_fn(120) < decay_fn(30)  # More decay at 120 minutes
        assert decay_fn(result.decay_half_life_minutes) == pytest.approx(0.5, rel=0.01)  # Half-life
    
    def test_model_parameters(self):
        """Test model parameter retrieval."""
        params = self.model.get_model_parameters()
        
        assert 'model_name' in params
        assert params['model_name'] == 'TaiwanMarketImpactModel'
        assert 'temp_impact_coeff' in params
        assert 'perm_impact_coeff' in params
        assert isinstance(params['temp_impact_coeff'], float)
    
    def test_result_serialization(self):
        """Test impact result serialization."""
        result = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=self.sample_timestamp
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['symbol'] == "2330.TW"
        assert result_dict['order_size'] == 10_000
        assert 'timestamp' in result_dict
        assert 'temporary_impact_bps' in result_dict
        assert 'permanent_impact_bps' in result_dict


class TestPortfolioImpactAnalyzer:
    """Test portfolio-level impact analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.impact_model = TaiwanMarketImpactModel()
        self.analyzer = PortfolioImpactAnalyzer(self.impact_model)
    
    def test_portfolio_impact_calculation(self):
        """Test portfolio impact calculation."""
        trades = [
            {
                'symbol': '2330.TW',
                'order_size': 10_000,
                'price': 500.0,
                'avg_daily_volume': 500_000,
                'volatility': 0.25,
                'timestamp': datetime(2024, 6, 15, 10, 30)
            },
            {
                'symbol': '2317.TW',
                'order_size': 15_000,
                'price': 100.0,
                'avg_daily_volume': 300_000,
                'volatility': 0.30,
                'timestamp': datetime(2024, 6, 15, 10, 45)
            }
        ]
        
        result = self.analyzer.calculate_portfolio_impact(trades)
        
        assert 'individual_impacts' in result
        assert 'portfolio_summary' in result
        assert 'timing_optimization' in result
        
        # Check portfolio summary
        summary = result['portfolio_summary']
        assert summary['trade_count'] == 2
        assert summary['total_impact_twd'] > 0
        assert summary['total_trade_value'] > 0
        assert summary['portfolio_impact_bps'] > 0
        
        # Check individual impacts
        individual_impacts = result['individual_impacts']
        assert len(individual_impacts) == 2
        for impact in individual_impacts:
            assert 'symbol' in impact
            assert 'total_impact_bps' in impact
    
    def test_timing_optimization_analysis(self):
        """Test timing optimization analysis."""
        trades = [
            {
                'symbol': '2330.TW',
                'order_size': 50_000,  # Large order for high impact
                'price': 500.0,
                'avg_daily_volume': 500_000,
                'volatility': 0.25,
                'timestamp': datetime(2024, 6, 15, 10, 30)
            },
            {
                'symbol': '2317.TW',
                'order_size': 5_000,   # Small order for low impact
                'price': 100.0,
                'avg_daily_volume': 300_000,
                'volatility': 0.30,
                'timestamp': datetime(2024, 6, 15, 10, 45)
            }
        ]
        
        result = self.analyzer.calculate_portfolio_impact(trades, timing_spread_minutes=120)
        
        timing_opt = result['timing_optimization']
        assert 'high_impact_trade_count' in timing_opt
        assert 'recommended_timing_spread_minutes' in timing_opt
        assert 'estimated_impact_reduction_pct' in timing_opt
        assert 'potential_savings_twd' in timing_opt
        
        # High impact trade should be identified
        assert timing_opt['high_impact_trade_count'] >= 1


class TestFactoryFunctions:
    """Test factory functions for model creation."""
    
    def test_create_taiwan_impact_model_default(self):
        """Test default model creation."""
        model = create_taiwan_impact_model()
        
        assert isinstance(model, TaiwanMarketImpactModel)
        assert model.name == "TaiwanMarketImpactModel"
        
        # Test that it can calculate impact
        result = model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25
        )
        assert isinstance(result, ImpactCalculationResult)
    
    def test_create_taiwan_impact_model_conservative(self):
        """Test conservative model creation."""
        model = create_taiwan_impact_model(conservative=True)
        
        # Conservative model should have higher impact coefficients
        params = model.get_model_parameters()
        assert params['temp_impact_coeff'] > 0.35  # Higher than default
        assert params['perm_impact_coeff'] > 0.20  # Higher than default
    
    def test_create_taiwan_impact_model_custom(self):
        """Test custom parameter model creation."""
        custom_params = {
            'temp_impact_coeff': 0.6,
            'perm_impact_coeff': 0.4
        }
        
        model = create_taiwan_impact_model(custom_params=custom_params)
        
        params = model.get_model_parameters()
        assert params['temp_impact_coeff'] == 0.6
        assert params['perm_impact_coeff'] == 0.4


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = TaiwanMarketImpactModel()
    
    def test_zero_volume(self):
        """Test handling of zero volume."""
        result = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=0,  # Zero volume
            volatility=0.25,
            timestamp=datetime.now()
        )
        
        # Should still calculate impact (fallback to default)
        assert result.total_impact_bps > 0
        assert result.participation_rate == float('inf')  # Order size / 0
    
    def test_very_small_order(self):
        """Test very small order sizes."""
        result = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=1,  # 1 share
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=datetime.now()
        )
        
        assert result.total_impact_bps >= 0
        assert result.participation_rate == 1 / 500_000
    
    def test_very_large_order(self):
        """Test very large order sizes."""
        result = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=1_000_000,  # 1M shares (2x daily volume)
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.25,
            timestamp=datetime.now()
        )
        
        assert result.total_impact_bps > 0
        assert result.participation_rate == 2.0  # 200% of ADV
        assert result.regime in [ImpactRegime.ILLIQUID, ImpactRegime.SEVERELY_CONSTRAINED]
    
    def test_extreme_volatility(self):
        """Test extreme volatility values."""
        # Very low volatility
        result_low = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=0.01,  # 1% annual volatility
            timestamp=datetime.now()
        )
        
        # Very high volatility
        result_high = self.model.calculate_impact(
            symbol="2330.TW",
            order_size=10_000,
            price=500.0,
            avg_daily_volume=500_000,
            volatility=2.0,  # 200% annual volatility
            timestamp=datetime.now()
        )
        
        assert result_low.total_impact_bps > 0
        assert result_high.total_impact_bps > result_low.total_impact_bps
        assert result_high.regime in [ImpactRegime.VOLATILE, ImpactRegime.STRESSED]


if __name__ == "__main__":
    pytest.main([__file__])