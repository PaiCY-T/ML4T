"""
Tests for strategy capacity modeling components.

This module tests the capacity analysis engine including strategy-level
capacity modeling, portfolio allocation optimization, and stress testing.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch

from src.trading.costs.capacity import (
    StrategyCapacityAnalyzer,
    StrategyCapacityParameters,
    CapacityAnalysisResult,
    PortfolioCapacityAllocation,
    CapacityType,
    CapacityRegime,
    create_capacity_analyzer
)
from src.trading.costs.market_impact import TaiwanMarketImpactModel
from src.trading.costs.liquidity import LiquidityAnalyzer, LiquidityMetrics, LiquidityTier


class TestStrategyCapacityParameters:
    """Test strategy capacity parameter configurations."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = StrategyCapacityParameters()
        
        assert params.max_impact_bps == 50.0
        assert params.target_impact_bps == 30.0
        assert params.impact_budget_daily_bps == 20.0
        assert params.stress_test_multiplier == 1.5
        assert params.confidence_level == 0.95
        assert params.diversification_benefit == 0.8
        assert params.min_holding_period_days == 1
        assert params.max_holding_period_days == 30
        assert params.rebalancing_frequency_days == 7
        assert params.max_concentration_pct == 0.20
        assert params.max_turnover_annual == 2.0
        assert params.market_hours_per_day == 4.5
        assert params.trading_days_per_year == 252
    
    def test_parameter_customization(self):
        """Test custom parameter configuration."""
        params = StrategyCapacityParameters(
            max_impact_bps=30.0,
            target_impact_bps=20.0,
            max_concentration_pct=0.15,
            stress_test_multiplier=2.0
        )
        
        assert params.max_impact_bps == 30.0
        assert params.target_impact_bps == 20.0
        assert params.max_concentration_pct == 0.15
        assert params.stress_test_multiplier == 2.0


class TestStrategyCapacityAnalyzer:
    """Test strategy capacity analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StrategyCapacityAnalyzer()
        
        # Sample universe and market data
        self.universe = ['2330.TW', '2317.TW', '2454.TW']
        self.market_data = {
            '2330.TW': {  # TSMC - Large cap, high liquidity
                'price': 500.0,
                'avg_daily_volume': 500_000,
                'volatility': 0.25,
                'shares_outstanding': 25_930_000_000,
                'data_date': date.today()
            },
            '2317.TW': {  # Hon Hai - Large cap, medium liquidity
                'price': 100.0,
                'avg_daily_volume': 300_000,
                'volatility': 0.30,
                'shares_outstanding': 13_800_000_000,
                'data_date': date.today()
            },
            '2454.TW': {  # MediaTek - Mid cap, medium liquidity
                'price': 800.0,
                'avg_daily_volume': 200_000,
                'volatility': 0.35,
                'shares_outstanding': 1_593_000_000,
                'data_date': date.today()
            }
        }
        
        # Mock volume/price data for each symbol
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        for symbol, data in self.market_data.items():
            np.random.seed(hash(symbol) % 1000)
            base_volume = data['avg_daily_volume']
            volumes = [max(0, base_volume * (1 + np.random.normal(0, 0.3))) for _ in range(60)]
            prices = [data['price'] * (1 + np.random.normal(0, 0.02)) for _ in range(60)]
            
            data['volume_data'] = pd.Series(volumes, index=dates)
            data['price_data'] = pd.Series(prices, index=dates)
    
    def test_strategy_capacity_analysis(self):
        """Test basic strategy capacity analysis."""
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Taiwan Large Cap Strategy",
            universe=self.universe,
            market_data=self.market_data,
            capacity_type=CapacityType.STRATEGY
        )
        
        assert isinstance(result, CapacityAnalysisResult)
        assert result.strategy_name == "Taiwan Large Cap Strategy"
        assert result.capacity_type == CapacityType.STRATEGY
        
        # Check capacity metrics
        assert result.max_portfolio_size_twd > 0
        assert result.max_daily_turnover_twd > 0
        assert result.estimated_impact_bps > 0
        assert result.impact_utilization_pct >= 0
        assert result.capacity_utilization_pct >= 0
        
        # Check position limits
        assert len(result.max_position_shares) == len(self.universe)
        for symbol in self.universe:
            assert symbol in result.max_position_shares
            assert result.max_position_shares[symbol] > 0
        
        # Check regime classification
        assert isinstance(result.capacity_regime, CapacityRegime)
        
        # Check stress test capacity
        assert result.stress_test_capacity_twd < result.max_portfolio_size_twd
        
        # Check confidence interval
        assert len(result.confidence_interval) == 2
        lower, upper = result.confidence_interval
        assert lower <= result.max_portfolio_size_twd <= upper
    
    def test_capacity_type_variations(self):
        """Test different capacity analysis types."""
        # Daily capacity
        daily_result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Daily Strategy",
            universe=self.universe,
            market_data=self.market_data,
            capacity_type=CapacityType.DAILY
        )
        
        # Monthly capacity
        monthly_result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Monthly Strategy",
            universe=self.universe,
            market_data=self.market_data,
            capacity_type=CapacityType.MONTHLY
        )
        
        # Daily capacity should have higher turnover capacity
        assert daily_result.max_daily_turnover_twd > monthly_result.max_daily_turnover_twd
    
    def test_binding_constraints_identification(self):
        """Test identification of binding constraints."""
        # Create market data with some illiquid stocks
        illiquid_market_data = self.market_data.copy()
        illiquid_market_data['ILLIQUID.TW'] = {
            'price': 50.0,
            'avg_daily_volume': 1_000,  # Very low volume
            'volatility': 0.50,  # High volatility
            'shares_outstanding': 100_000_000,
            'data_date': date.today()
        }
        
        # Add mock data
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        volumes = [max(0, 1000 * (1 + np.random.normal(0, 0.5))) for _ in range(60)]
        prices = [50 * (1 + np.random.normal(0, 0.03)) for _ in range(60)]
        illiquid_market_data['ILLIQUID.TW']['volume_data'] = pd.Series(volumes, index=dates)
        illiquid_market_data['ILLIQUID.TW']['price_data'] = pd.Series(prices, index=dates)
        
        universe_with_illiquid = self.universe + ['ILLIQUID.TW']
        
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Mixed Liquidity Strategy",
            universe=universe_with_illiquid,
            market_data=illiquid_market_data,
            capacity_type=CapacityType.STRATEGY
        )
        
        # Should have binding constraints for illiquid stock
        assert len(result.binding_constraints) > 0
        
        # Should identify illiquid stock constraints
        illiquid_constraints = [c for c in result.binding_constraints if 'ILLIQUID.TW' in c]
        assert len(illiquid_constraints) > 0
    
    def test_portfolio_capacity_allocation(self):
        """Test portfolio capacity allocation optimization."""
        target_weights = {
            '2330.TW': 0.5,   # 50% in TSMC
            '2317.TW': 0.3,   # 30% in Hon Hai
            '2454.TW': 0.2    # 20% in MediaTek
        }
        
        total_portfolio_value = 100_000_000  # 100M TWD
        
        allocation = self.analyzer.optimize_portfolio_capacity_allocation(
            target_weights=target_weights,
            market_data=self.market_data,
            total_portfolio_value=total_portfolio_value
        )
        
        assert isinstance(allocation, PortfolioCapacityAllocation)
        assert allocation.total_capacity_twd > 0
        assert allocation.utilization_pct >= 0
        assert allocation.efficiency_score >= 0
        
        # Check allocations for each symbol
        assert len(allocation.allocations) == len(target_weights)
        for symbol, target_weight in target_weights.items():
            assert symbol in allocation.allocations
            alloc = allocation.allocations[symbol]
            assert 'capacity_twd' in alloc
            assert 'weight' in alloc
            assert 'impact_bps' in alloc
            assert alloc['capacity_twd'] > 0
            assert alloc['weight'] >= 0
            assert alloc['impact_bps'] >= 0
        
        # Check risk scores
        assert 0 <= allocation.concentration_risk_score <= 1
        assert 0 <= allocation.liquidity_risk_score <= 1
        assert 0 <= allocation.impact_risk_score <= 1
        
        # Check diversification benefit
        assert allocation.diversification_benefit_twd != 0  # Should have some benefit
        
        # Risk-adjusted capacity should be <= total capacity
        assert allocation.risk_adjusted_capacity <= allocation.total_capacity_twd
    
    def test_stress_testing(self):
        """Test capacity stress testing functionality."""
        base_result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Base Strategy",
            universe=self.universe,
            market_data=self.market_data,
            capacity_type=CapacityType.STRATEGY
        )
        
        # Define stress scenarios
        stress_scenarios = [
            {
                'name': 'High Volatility',
                'volatility_multiplier': 2.0,
                'liquidity_reduction': 0.3  # 30% reduction in liquidity
            },
            {
                'name': 'Liquidity Crisis',
                'volatility_multiplier': 1.5,
                'liquidity_reduction': 0.5,  # 50% reduction
                'spread_multiplier': 3.0
            },
            {
                'name': 'Market Stress',
                'volatility_multiplier': 3.0,
                'liquidity_reduction': 0.7  # 70% reduction
            }
        ]
        
        stress_results = self.analyzer.stress_test_capacity(
            base_capacity=base_result,
            stress_scenarios=stress_scenarios,
            market_data=self.market_data
        )
        
        assert 'base_capacity' in stress_results
        assert 'stress_scenarios' in stress_results
        assert 'stress_summary' in stress_results
        
        # Check scenario results
        scenario_results = stress_results['stress_scenarios']
        assert len(scenario_results) == len(stress_scenarios)
        
        for scenario_name in ['High Volatility', 'Liquidity Crisis', 'Market Stress']:
            assert scenario_name in scenario_results
            scenario_result = scenario_results[scenario_name]
            
            if 'error' not in scenario_result:
                assert 'capacity_reduction_pct' in scenario_result
                assert 'impact_increase_bps' in scenario_result
                assert scenario_result['capacity_reduction_pct'] >= 0  # Should reduce capacity
                assert scenario_result['impact_increase_bps'] >= 0  # Should increase impact
        
        # Check stress summary
        if 'error' not in stress_results['stress_summary']:
            summary = stress_results['stress_summary']
            assert 'avg_capacity_reduction_pct' in summary
            assert 'max_capacity_reduction_pct' in summary
            assert 'stress_adjusted_capacity_twd' in summary
            assert summary['max_capacity_reduction_pct'] >= summary['avg_capacity_reduction_pct']
    
    def test_regime_determination(self):
        """Test capacity regime determination."""
        # Create scenarios for different regimes
        
        # Normal regime: moderate impact, few constraints
        normal_data = {
            'NORMAL.TW': {
                'price': 100.0,
                'avg_daily_volume': 1_000_000,  # High liquidity
                'volatility': 0.20,  # Low volatility
                'shares_outstanding': 1_000_000_000,
                'data_date': date.today()
            }
        }
        
        # Add mock data
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        volumes = [1_000_000 * (1 + np.random.normal(0, 0.1)) for _ in range(60)]
        prices = [100 * (1 + np.random.normal(0, 0.01)) for _ in range(60)]
        normal_data['NORMAL.TW']['volume_data'] = pd.Series(volumes, index=dates)
        normal_data['NORMAL.TW']['price_data'] = pd.Series(prices, index=dates)
        
        normal_result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Normal Strategy",
            universe=['NORMAL.TW'],
            market_data=normal_data
        )
        
        # Constrained regime: high impact or many binding constraints
        constrained_data = {
            'CONSTRAINED.TW': {
                'price': 50.0,
                'avg_daily_volume': 50_000,  # Lower liquidity
                'volatility': 0.40,  # Higher volatility
                'shares_outstanding': 100_000_000,
                'data_date': date.today()
            }
        }
        
        # Add mock data
        volumes = [50_000 * (1 + np.random.normal(0, 0.3)) for _ in range(60)]
        prices = [50 * (1 + np.random.normal(0, 0.02)) for _ in range(60)]
        constrained_data['CONSTRAINED.TW']['volume_data'] = pd.Series(volumes, index=dates)
        constrained_data['CONSTRAINED.TW']['price_data'] = pd.Series(prices, index=dates)
        
        constrained_result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Constrained Strategy",
            universe=['CONSTRAINED.TW'],
            market_data=constrained_data
        )
        
        # Normal regime should have better characteristics
        if normal_result.capacity_regime == CapacityRegime.NORMAL:
            assert normal_result.estimated_impact_bps <= constrained_result.estimated_impact_bps
            assert len(normal_result.binding_constraints) <= len(constrained_result.binding_constraints)
    
    def test_optimization_suggestions(self):
        """Test generation of optimization suggestions."""
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Test Strategy",
            universe=self.universe,
            market_data=self.market_data
        )
        
        assert isinstance(result.optimization_suggestions, list)
        assert len(result.optimization_suggestions) > 0
        
        # Suggestions should be strings
        for suggestion in result.optimization_suggestions:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0
    
    def test_recommended_parameters(self):
        """Test recommended parameter generation."""
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Test Strategy",
            universe=self.universe,
            market_data=self.market_data
        )
        
        # Check recommended position size
        assert 0 < result.recommended_max_position_pct <= 0.5
        
        # Check recommended rebalancing frequency
        assert result.recommended_rebalancing_frequency >= 1
        
        # Recommendations should be reasonable based on regime
        if result.capacity_regime == CapacityRegime.CONSTRAINED:
            assert result.recommended_max_position_pct <= 0.15  # Conservative for constrained regime
        elif result.capacity_regime == CapacityRegime.UNCONSTRAINED:
            assert result.recommended_max_position_pct >= 0.15  # More aggressive for unconstrained
    
    def test_result_serialization(self):
        """Test capacity analysis result serialization."""
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Serialization Test",
            universe=self.universe,
            market_data=self.market_data
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['strategy_name'] == "Serialization Test"
        assert 'analysis_date' in result_dict
        assert 'capacity_type' in result_dict
        assert 'max_portfolio_size_twd' in result_dict
        assert 'estimated_impact_bps' in result_dict
        assert 'binding_constraints' in result_dict
        assert 'optimization_suggestions' in result_dict


class TestFactoryFunctions:
    """Test factory functions for analyzer creation."""
    
    def test_create_capacity_analyzer_default(self):
        """Test default analyzer creation."""
        analyzer = create_capacity_analyzer()
        
        assert isinstance(analyzer, StrategyCapacityAnalyzer)
        assert isinstance(analyzer.impact_model, TaiwanMarketImpactModel)
        assert isinstance(analyzer.liquidity_analyzer, LiquidityAnalyzer)
        assert isinstance(analyzer.parameters, StrategyCapacityParameters)
    
    def test_create_capacity_analyzer_conservative(self):
        """Test conservative analyzer creation."""
        analyzer = create_capacity_analyzer(conservative=True)
        
        # Conservative settings should be more restrictive
        params = analyzer.parameters
        assert params.max_impact_bps <= 50.0
        assert params.target_impact_bps <= 30.0
        assert params.max_concentration_pct <= 0.20
        assert params.stress_test_multiplier >= 1.5
    
    def test_create_capacity_analyzer_custom(self):
        """Test custom parameter analyzer creation."""
        custom_params = {
            'max_impact_bps': 40.0,
            'target_impact_bps': 25.0,
            'max_concentration_pct': 0.10
        }
        
        analyzer = create_capacity_analyzer(custom_params=custom_params)
        
        params = analyzer.parameters
        assert params.max_impact_bps == 40.0
        assert params.target_impact_bps == 25.0
        assert params.max_concentration_pct == 0.10


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StrategyCapacityAnalyzer()
    
    def test_empty_universe(self):
        """Test handling of empty universe."""
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Empty Strategy",
            universe=[],
            market_data={}
        )
        
        # Should handle gracefully
        assert result.max_portfolio_size_twd >= 0
        assert len(result.max_position_shares) == 0
    
    def test_missing_market_data(self):
        """Test handling of missing market data."""
        universe = ['MISSING.TW']
        market_data = {}  # No data for symbol
        
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Missing Data Strategy",
            universe=universe,
            market_data=market_data
        )
        
        # Should handle gracefully and skip missing symbols
        assert isinstance(result, CapacityAnalysisResult)
        # May have warnings but should not crash
    
    def test_extreme_volatility(self):
        """Test handling of extreme volatility values."""
        extreme_data = {
            'EXTREME.TW': {
                'price': 100.0,
                'avg_daily_volume': 100_000,
                'volatility': 5.0,  # 500% annual volatility
                'shares_outstanding': 1_000_000_000,
                'data_date': date.today()
            }
        }
        
        # Add mock data
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        volumes = [100_000 * (1 + np.random.normal(0, 0.2)) for _ in range(60)]
        prices = [100 * (1 + np.random.normal(0, 0.02)) for _ in range(60)]
        extreme_data['EXTREME.TW']['volume_data'] = pd.Series(volumes, index=dates)
        extreme_data['EXTREME.TW']['price_data'] = pd.Series(prices, index=dates)
        
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Extreme Volatility Strategy",
            universe=['EXTREME.TW'],
            market_data=extreme_data
        )
        
        # Should handle extreme volatility
        assert result.estimated_impact_bps > 0
        assert result.capacity_regime in [CapacityRegime.CONSTRAINED, CapacityRegime.SEVERELY_CONSTRAINED]
    
    def test_zero_volume_stocks(self):
        """Test handling of stocks with zero volume."""
        zero_volume_data = {
            'ZERO.TW': {
                'price': 100.0,
                'avg_daily_volume': 0,  # No volume
                'volatility': 0.30,
                'shares_outstanding': 1_000_000_000,
                'data_date': date.today()
            }
        }
        
        # Add mock data with zero volumes
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        volumes = [0] * 60  # All zero volumes
        prices = [100] * 60
        zero_volume_data['ZERO.TW']['volume_data'] = pd.Series(volumes, index=dates)
        zero_volume_data['ZERO.TW']['price_data'] = pd.Series(prices, index=dates)
        
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Zero Volume Strategy",
            universe=['ZERO.TW'],
            market_data=zero_volume_data
        )
        
        # Should handle zero volume gracefully
        assert isinstance(result, CapacityAnalysisResult)
        # Should have very low capacity or binding constraints
        assert len(result.binding_constraints) > 0 or result.max_portfolio_size_twd == 0


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StrategyCapacityAnalyzer()
    
    def test_large_universe_performance(self):
        """Test performance with large universes."""
        # Create large universe (50 stocks)
        large_universe = [f"STOCK{i:02d}.TW" for i in range(50)]
        
        # Generate market data
        large_market_data = {}
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        
        for i, symbol in enumerate(large_universe):
            np.random.seed(i)  # Different seed per stock
            base_volume = 50_000 + i * 10_000  # Varying liquidity
            base_price = 50 + i * 10
            
            volumes = [max(0, base_volume * (1 + np.random.normal(0, 0.3))) for _ in range(60)]
            prices = [base_price * (1 + np.random.normal(0, 0.02)) for _ in range(60)]
            
            large_market_data[symbol] = {
                'price': base_price,
                'avg_daily_volume': base_volume,
                'volatility': 0.20 + i * 0.01,  # Varying volatility
                'shares_outstanding': 1_000_000_000,
                'data_date': date.today(),
                'volume_data': pd.Series(volumes, index=dates),
                'price_data': pd.Series(prices, index=dates)
            }
        
        start_time = datetime.now()
        
        result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Large Universe Strategy",
            universe=large_universe,
            market_data=large_market_data
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time (< 10 seconds for 50 stocks)
        assert execution_time < 10.0
        assert isinstance(result, CapacityAnalysisResult)
        assert len(result.max_position_shares) <= len(large_universe)  # May skip some due to data issues
    
    def test_stress_test_performance(self):
        """Test stress testing performance."""
        # Create moderate universe
        universe = [f"TEST{i}.TW" for i in range(10)]
        
        market_data = {}
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        
        for i, symbol in enumerate(universe):
            np.random.seed(i)
            volumes = [100_000 * (1 + np.random.normal(0, 0.3)) for _ in range(60)]
            prices = [100 * (1 + np.random.normal(0, 0.02)) for _ in range(60)]
            
            market_data[symbol] = {
                'price': 100.0,
                'avg_daily_volume': 100_000,
                'volatility': 0.25,
                'shares_outstanding': 1_000_000_000,
                'data_date': date.today(),
                'volume_data': pd.Series(volumes, index=dates),
                'price_data': pd.Series(prices, index=dates)
            }
        
        # Create base capacity
        base_result = self.analyzer.analyze_strategy_capacity(
            strategy_name="Performance Test",
            universe=universe,
            market_data=market_data
        )
        
        # Define multiple stress scenarios
        stress_scenarios = [
            {'name': f'Scenario_{i}', 'volatility_multiplier': 1.5 + i * 0.1, 'liquidity_reduction': 0.2 + i * 0.1}
            for i in range(10)
        ]
        
        start_time = datetime.now()
        
        stress_results = self.analyzer.stress_test_capacity(
            base_capacity=base_result,
            stress_scenarios=stress_scenarios,
            market_data=market_data
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Should complete stress testing in reasonable time
        assert execution_time < 30.0  # 30 seconds for 10 scenarios x 10 stocks
        assert 'stress_scenarios' in stress_results
        assert len(stress_results['stress_scenarios']) <= len(stress_scenarios)


if __name__ == "__main__":
    pytest.main([__file__])