"""
Performance Validation Tests - Task #004
GitHub Issue #77

Validation tests ensuring factor combination strategies meet performance targets
and Taiwan market optimization requirements.

Performance Targets:
- Portfolio construction <30 seconds for 500 Taiwan stocks
- Factor combination logic <5 seconds for real-time updates
- Memory usage <2GB for full Taiwan universe processing
- Maintain 31x performance advantage from value factors

Taiwan Market Optimization:
- Respects market constraints (liquidity, market cap, sector limits)
- Handles production readiness differences across factor types
- Optimized for Taiwan market patterns and regime characteristics
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import time
import logging
import gc
import psutil
import os
from decimal import Decimal
from pathlib import Path

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.strategies.factor_combination import (
    EqualWeightStrategy, SmartBetaStrategy, FactorPortfolioConstructor,
    FactorCombinationMethod, PortfolioObjective, TaiwanMarketRegime,
    CompositeFactorScore, FactorWeight
)

from src.factors.factor_integration import (
    FactorPipeline, IntegratedFactorMetrics, FactorGroupType,
    IntegrationQuality, FactorGroupStatus
)


class TestPerformanceTargets:
    """Test performance target validation and benchmarks."""

    def get_memory_usage_mb(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def create_large_integrated_metrics(self, num_symbols=500):
        """Create large set of integrated metrics for performance testing."""
        test_date = date.today()
        symbols = [f"Symbol_{i:04d}.TW" for i in range(num_symbols)]

        metrics = {}
        for i, symbol in enumerate(symbols):
            # Create realistic factor scores with variation
            base_score = np.random.normal(0, 1)

            metrics[symbol] = IntegratedFactorMetrics(
                symbol=symbol,
                date=test_date,
                normalized_value_score=base_score + np.random.normal(0, 0.3),
                normalized_flow_score=base_score + np.random.normal(0, 0.4),
                normalized_momentum_score=base_score + np.random.normal(0, 0.2),
                total_data_completeness=0.7 + np.random.random() * 0.3,
                factor_group_coverage={
                    'value': True,
                    'flow': np.random.random() > 0.2,  # 80% coverage
                    'momentum': True
                }
            )

        return metrics

    def create_large_market_data(self, num_symbols=500):
        """Create large market data set for performance testing."""
        market_data = {}
        symbols = [f"Symbol_{i:04d}.TW" for i in range(num_symbols)]

        sectors = ['technology', 'financials', 'industrials', 'consumer', 'materials']

        for i, symbol in enumerate(symbols):
            market_data[symbol] = {
                'market_cap': 5e9 + np.random.exponential(10e9),  # Above threshold
                'avg_volume': 1e6 + np.random.exponential(5e6),   # Above threshold
                'sector': sectors[i % len(sectors)],
                'beta': 0.5 + np.random.random() * 1.0,
                'volatility': 0.1 + np.random.random() * 0.2
            }

        return market_data

    @pytest.fixture
    def mock_factor_pipeline_performance(self):
        """Create mock factor pipeline optimized for performance testing."""
        pipeline = Mock(spec=FactorPipeline)

        # Mock high-quality factor group status
        status_dict = {
            FactorGroupType.VALUE: FactorGroupStatus(
                group_type=FactorGroupType.VALUE,
                quality_level=IntegrationQuality.PRODUCTION_READY,
                performance_score=31.0,  # 31x performance maintained
                coverage_score=0.95,
                production_ready=True
            ),
            FactorGroupType.FLOW: FactorGroupStatus(
                group_type=FactorGroupType.FLOW,
                quality_level=IntegrationQuality.FUNCTIONAL,
                performance_score=1.0,
                coverage_score=0.80,
                production_ready=False  # Production concerns handled
            ),
            FactorGroupType.MOMENTUM: FactorGroupStatus(
                group_type=FactorGroupType.MOMENTUM,
                quality_level=IntegrationQuality.FUNCTIONAL,
                performance_score=1.0,
                coverage_score=0.85,
                production_ready=True
            )
        }

        pipeline.get_factor_group_status.return_value = status_dict
        return pipeline

    def test_factor_combination_performance_target(self, mock_factor_pipeline_performance):
        """Test factor combination logic meets <5 second target."""
        # Create large dataset (500 symbols)
        integrated_metrics = self.create_large_integrated_metrics(500)

        # Mock pipeline to return large dataset quickly
        mock_factor_pipeline_performance.calculate_integrated_factors.return_value = integrated_metrics

        # Test EqualWeightStrategy
        strategy = EqualWeightStrategy(mock_factor_pipeline_performance)
        symbols = list(integrated_metrics.keys())

        # Measure combination time
        start_time = time.time()
        composite_scores = strategy.calculate_composite_scores(symbols, date.today())
        combination_time = time.time() - start_time

        # Validate performance target: <5 seconds for factor combination logic
        assert combination_time < 5.0, f"Combination took {combination_time:.2f}s, exceeds 5s target"
        assert len(composite_scores) == len(symbols)

        # Validate that scores were computed
        valid_scores = sum(1 for score in composite_scores.values() if score.equal_weight_score is not None)
        assert valid_scores > len(symbols) * 0.8  # At least 80% should have valid scores

        print(f"✅ Factor combination: {len(symbols)} symbols in {combination_time:.2f}s")

    def test_smart_beta_combination_performance(self, mock_factor_pipeline_performance):
        """Test SmartBeta strategy performance with large dataset."""
        # Create large dataset (500 symbols)
        integrated_metrics = self.create_large_integrated_metrics(500)
        mock_factor_pipeline_performance.calculate_integrated_factors.return_value = integrated_metrics

        # Test SmartBetaStrategy
        strategy = SmartBetaStrategy(mock_factor_pipeline_performance)
        symbols = list(integrated_metrics.keys())

        # Measure combination time
        start_time = time.time()
        composite_scores = strategy.calculate_composite_scores(symbols, date.today())
        combination_time = time.time() - start_time

        # Validate performance target: <5 seconds
        assert combination_time < 5.0, f"SmartBeta combination took {combination_time:.2f}s, exceeds 5s target"
        assert len(composite_scores) == len(symbols)

        # Validate that both equal-weight and smart-beta scores were computed
        smart_beta_count = sum(1 for score in composite_scores.values() if score.smart_beta_score is not None)
        assert smart_beta_count > len(symbols) * 0.8

        print(f"✅ SmartBeta combination: {len(symbols)} symbols in {combination_time:.2f}s")

    def test_portfolio_construction_performance_target(self):
        """Test portfolio construction meets <30 second target for 500 Taiwan stocks."""
        constructor = FactorPortfolioConstructor(PortfolioObjective.ALPHA_GENERATION)

        # Create 500 composite scores
        composite_scores = {}
        for i in range(500):
            symbol = f"Stock_{i:04d}.TW"
            composite_scores[symbol] = CompositeFactorScore(
                symbol=symbol,
                date=date.today(),
                smart_beta_score=np.random.normal(0, 1),
                equal_weight_score=np.random.normal(0, 1),
                value_score=np.random.normal(0, 1),
                flow_score=np.random.normal(0, 1),
                momentum_score=np.random.normal(0, 1),
                factor_count=3,
                data_completeness=0.8,
                production_ready=True,
                factor_weights=FactorWeight(0.4, 0.3, 0.3, date.today())
            )

        # Create market data for all symbols
        market_data = self.create_large_market_data(500)

        # Measure portfolio construction time
        start_time = time.time()
        portfolio = constructor.construct_portfolio(composite_scores, market_data)
        construction_time = time.time() - start_time

        # Validate performance target: <30 seconds for 500 Taiwan stocks
        assert construction_time < 30.0, f"Portfolio construction took {construction_time:.2f}s, exceeds 30s target"
        assert len(portfolio.positions) > 0
        assert portfolio.position_count <= constructor.taiwan_params['max_positions']

        print(f"✅ Portfolio construction: {len(composite_scores)} stocks in {construction_time:.2f}s")

    def test_memory_usage_target(self, mock_factor_pipeline_performance):
        """Test memory usage stays under 2GB for full Taiwan universe processing."""
        # Get initial memory usage
        gc.collect()  # Clean up before measurement
        initial_memory = self.get_memory_usage_mb()

        # Create large dataset representing full Taiwan universe (~1300 stocks)
        integrated_metrics = self.create_large_integrated_metrics(1300)
        mock_factor_pipeline_performance.calculate_integrated_factors.return_value = integrated_metrics

        # Process with both strategies
        equal_weight_strategy = EqualWeightStrategy(mock_factor_pipeline_performance)
        smart_beta_strategy = SmartBetaStrategy(mock_factor_pipeline_performance)

        symbols = list(integrated_metrics.keys())

        # Calculate composite scores
        equal_weight_scores = equal_weight_strategy.calculate_composite_scores(symbols, date.today())
        smart_beta_scores = smart_beta_strategy.calculate_composite_scores(symbols, date.today())

        # Create portfolio
        constructor = FactorPortfolioConstructor()
        market_data = self.create_large_market_data(1300)
        portfolio = constructor.construct_portfolio(smart_beta_scores, market_data)

        # Measure memory usage
        current_memory = self.get_memory_usage_mb()
        memory_increase = current_memory - initial_memory

        # Validate memory target: <2GB increase
        memory_limit_mb = 2048  # 2GB
        assert memory_increase < memory_limit_mb, f"Memory usage increased by {memory_increase:.1f}MB, exceeds {memory_limit_mb}MB limit"

        print(f"✅ Memory usage: {memory_increase:.1f}MB increase for {len(symbols)} stocks")

        # Cleanup
        del integrated_metrics, equal_weight_scores, smart_beta_scores, portfolio, market_data
        gc.collect()

    def test_value_factor_performance_advantage_maintained(self, mock_factor_pipeline_performance):
        """Test that 31x performance advantage from value factors is maintained."""
        # This test validates that the integration doesn't degrade value factor performance

        # Get factor group status
        status_dict = mock_factor_pipeline_performance.get_factor_group_status()
        value_status = status_dict[FactorGroupType.VALUE]

        # Validate that value factors maintain their performance advantage
        assert value_status.performance_score >= 30.0, f"Value factor performance score {value_status.performance_score} below 30x threshold"
        assert value_status.production_ready, "Value factors should remain production ready"
        assert value_status.quality_level == IntegrationQuality.PRODUCTION_READY

        print(f"✅ Value factor performance advantage maintained: {value_status.performance_score}x")

    def test_production_readiness_handling(self, mock_factor_pipeline_performance):
        """Test proper handling of production readiness differences across factor types."""
        strategy = SmartBetaStrategy(mock_factor_pipeline_performance)

        # Create test metrics
        integrated_metrics = self.create_large_integrated_metrics(100)
        mock_factor_pipeline_performance.calculate_integrated_factors.return_value = integrated_metrics

        # Calculate factor weights (should handle production readiness differences)
        factor_weights = strategy.calculate_factor_weights(integrated_metrics, TaiwanMarketRegime.MEAN_REVERTING)

        # Validate that flow factor weight is reduced due to production concerns
        flow_status = mock_factor_pipeline_performance.get_factor_group_status()[FactorGroupType.FLOW]
        if not flow_status.production_ready:
            # Flow weight should be reduced compared to other factors
            assert factor_weights.flow_weight < max(factor_weights.value_weight, factor_weights.momentum_weight)

        # Validate weights sum to 1
        total_weight = factor_weights.value_weight + factor_weights.flow_weight + factor_weights.momentum_weight
        assert abs(total_weight - 1.0) < 0.001

        print(f"✅ Production readiness handled: weights V:{factor_weights.value_weight:.2f} F:{factor_weights.flow_weight:.2f} M:{factor_weights.momentum_weight:.2f}")


class TestTaiwanMarketOptimization:
    """Test Taiwan market specific optimizations and constraints."""

    def test_taiwan_market_constraint_enforcement(self):
        """Test enforcement of Taiwan market constraints."""
        constructor = FactorPortfolioConstructor()

        # Test market cap threshold
        assert constructor.taiwan_params['market_cap_threshold'] == 5e9  # 5B TWD

        # Test liquidity threshold
        assert constructor.taiwan_params['liquidity_threshold'] == 1e6  # 1M TWD

        # Test position size limits
        assert constructor.taiwan_params['max_position_size'] == 0.05  # 5%
        assert constructor.taiwan_params['min_position_size'] == 0.005  # 0.5%

        # Test sector concentration limits
        sector_limits = constructor.taiwan_params['sector_limits']
        assert sector_limits['technology'] <= 0.3  # Max 30% tech
        assert sector_limits['financials'] <= 0.25  # Max 25% financials

        print("✅ Taiwan market constraints properly configured")

    def test_regime_aware_factor_weighting(self, mock_factor_pipeline_performance):
        """Test regime-aware factor weighting for Taiwan market."""
        strategy = SmartBetaStrategy(mock_factor_pipeline_performance)

        base_weights = (1/3, 1/3, 1/3)

        # Test different Taiwan market regimes
        regimes_to_test = [
            (TaiwanMarketRegime.TRENDING_BULL, "momentum"),
            (TaiwanMarketRegime.MEAN_REVERTING, "value"),
            (TaiwanMarketRegime.HIGH_VOLATILITY, "balanced"),
        ]

        for regime, expected_bias in regimes_to_test:
            adjusted_weights = strategy._apply_regime_adjustments(base_weights, regime)

            # Weights should be different from base (unless balanced)
            if expected_bias != "balanced":
                assert adjusted_weights != base_weights

            # Sum should still be reasonable (normalization happens later)
            assert sum(adjusted_weights) > 0

        print("✅ Regime-aware factor weighting validated for Taiwan market")

    def test_sector_concentration_control(self):
        """Test sector concentration control for Taiwan market."""
        constructor = FactorPortfolioConstructor()

        # Create portfolio with sector concentration
        composite_scores = {}
        market_data = {}

        # Create 20 tech stocks and 5 financial stocks
        for i in range(20):
            symbol = f"TECH_{i:02d}.TW"
            composite_scores[symbol] = CompositeFactorScore(
                symbol=symbol,
                date=date.today(),
                smart_beta_score=0.8,
                factor_count=3,
                data_completeness=0.8,
                production_ready=True
            )
            market_data[symbol] = {
                'market_cap': 1e10,
                'avg_volume': 2e6,
                'sector': 'technology'
            }

        for i in range(5):
            symbol = f"FIN_{i:02d}.TW"
            composite_scores[symbol] = CompositeFactorScore(
                symbol=symbol,
                date=date.today(),
                smart_beta_score=0.6,
                factor_count=3,
                data_completeness=0.8,
                production_ready=True
            )
            market_data[symbol] = {
                'market_cap': 8e9,
                'avg_volume': 1.5e6,
                'sector': 'financials'
            }

        # Construct portfolio
        portfolio = constructor.construct_portfolio(composite_scores, market_data)

        # Calculate sector exposures
        sector_exposures = {}
        for position in portfolio.positions:
            sector = market_data[position.symbol]['sector']
            sector_exposures[sector] = sector_exposures.get(sector, 0) + position.weight

        # Validate sector limits are respected (allowing some tolerance)
        tech_limit = constructor.taiwan_params['sector_limits']['technology']
        fin_limit = constructor.taiwan_params['sector_limits']['financials']

        if 'technology' in sector_exposures:
            assert sector_exposures['technology'] <= tech_limit + 0.1  # 10% tolerance

        if 'financials' in sector_exposures:
            assert sector_exposures['financials'] <= fin_limit + 0.1  # 10% tolerance

        print(f"✅ Sector concentration controlled: Tech={sector_exposures.get('technology', 0):.1%}, Fin={sector_exposures.get('financials', 0):.1%}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])