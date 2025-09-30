"""
Factor Combination Strategy Implementation - Task #004
GitHub Issue #77

Comprehensive factor combination algorithms implementing equal-weight and smart-beta
strategies for Taiwan market optimization. This implementation leverages the integrated
factor pipeline from Task 003 while addressing production readiness differences across
factor types.

Key Strategy Features:
1. EqualWeightStrategy: Simple equal combination of normalized factor scores
2. SmartBetaStrategy: Risk-adjusted factor weighting based on Taiwan market characteristics
3. FactorPortfolioConstructor: Taiwan-optimized portfolio construction
4. Composite scoring with factor interaction modeling
5. Production quality coordination across varying factor group readiness

Integration Context:
- Builds upon FactorPipeline from Task 003 with 18 passing integration tests
- Leverages high-quality value factors (31x performance, production-ready)
- Safely integrates flow factors (functional but documented concerns)
- Maintains unified normalization across factor groups
- Optimized for Taiwan market patterns and constraints

Performance Requirements:
- Portfolio construction <30 seconds for 500 Taiwan stocks
- Factor combination logic <5 seconds for real-time updates
- Memory usage <2GB for full Taiwan universe processing
- Maintain 31x performance advantage from value factors

Quality Framework:
- All strategy claims backed by evidence (backtesting, performance metrics)
- Address factor quality differences with robust error handling
- Comprehensive testing validating combination effectiveness
- Taiwan market optimization with regime awareness
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
from decimal import Decimal
import warnings
import time
from pathlib import Path
import json
from collections import defaultdict

# Import factor integration components
from ..factors.factor_integration import (
    FactorPipeline, IntegratedFactorMetrics, FactorCorrelationMatrix,
    FactorGroupType, IntegrationQuality, FactorGroupStatus,
    create_factor_pipeline
)

try:
    from ..data.core.temporal import DataType, TemporalValue
    from ..data.pipeline.pit_engine import PITQueryEngine, PITQuery
    from ..data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
    from ..data.models.taiwan_market import TaiwanMarketCode, TradingStatus
except ImportError:
    # For testing or standalone usage
    DataType = object
    TemporalValue = object
    PITQueryEngine = object
    PITQuery = object
    FinLabConnector = object
    FinLabConfig = object
    TaiwanMarketCode = object
    TradingStatus = object

logger = logging.getLogger(__name__)


class FactorCombinationMethod(Enum):
    """Factor combination methodologies supported by the system."""
    EQUAL_WEIGHT = "equal_weight"           # Simple equal weighting
    SMART_BETA = "smart_beta"               # Risk-adjusted weighting
    ADAPTIVE_WEIGHT = "adaptive_weight"     # Dynamic factor weighting
    REGIME_AWARE = "regime_aware"           # Regime-dependent weighting


class PortfolioObjective(Enum):
    """Portfolio construction objectives for Taiwan market."""
    ALPHA_GENERATION = "alpha_generation"   # Maximize alpha vs TAIEX
    RISK_ADJUSTED = "risk_adjusted"         # Optimize Sharpe ratio
    DRAWDOWN_CONTROL = "drawdown_control"   # Minimize maximum drawdown
    FACTOR_BALANCED = "factor_balanced"     # Balanced factor exposures


class TaiwanMarketRegime(Enum):
    """Taiwan market regime classifications for factor weighting."""
    TRENDING_BULL = "trending_bull"         # Strong upward trend
    TRENDING_BEAR = "trending_bear"         # Strong downward trend
    MEAN_REVERTING = "mean_reverting"       # Sideways/choppy market
    HIGH_VOLATILITY = "high_volatility"     # Elevated volatility regime
    RECOVERY = "recovery"                   # Post-crisis recovery


@dataclass
class FactorWeight:
    """Container for factor weights with validation and metadata."""
    value_weight: float
    flow_weight: float
    momentum_weight: float

    # Metadata
    weight_date: date
    regime: Optional[TaiwanMarketRegime] = None
    confidence_score: float = 1.0
    rebalance_trigger: Optional[str] = None

    def __post_init__(self):
        """Validate weights sum to 1.0 and are non-negative."""
        weights = [self.value_weight, self.flow_weight, self.momentum_weight]

        # Check non-negative
        if any(w < 0 for w in weights):
            raise ValueError("All factor weights must be non-negative")

        # Normalize to sum to 1.0
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")

        self.value_weight /= total_weight
        self.flow_weight /= total_weight
        self.momentum_weight /= total_weight

    @property
    def is_balanced(self) -> bool:
        """Check if weights are reasonably balanced (no single factor >60%)."""
        return max(self.value_weight, self.flow_weight, self.momentum_weight) <= 0.6


@dataclass
class CompositeFactorScore:
    """Container for composite factor scores with quality indicators."""
    symbol: str
    date: date

    # Individual factor scores (normalized)
    value_score: Optional[float] = None
    flow_score: Optional[float] = None
    momentum_score: Optional[float] = None

    # Composite scores
    equal_weight_score: Optional[float] = None
    smart_beta_score: Optional[float] = None

    # Quality indicators
    factor_count: int = 0  # Number of factors with valid scores
    data_completeness: float = 0.0  # Overall data quality
    production_ready: bool = False  # Safe for production use

    # Factor weights used (for transparency)
    factor_weights: Optional[FactorWeight] = None

    # Ranking information
    cross_sectional_rank: Optional[int] = None
    percentile_rank: Optional[float] = None
    universe_size: Optional[int] = None


@dataclass
class PortfolioPosition:
    """Container for portfolio position with factor attribution."""
    symbol: str
    weight: float
    composite_score: CompositeFactorScore

    # Factor attribution
    value_contribution: float = 0.0
    flow_contribution: float = 0.0
    momentum_contribution: float = 0.0

    # Risk metrics
    expected_volatility: Optional[float] = None
    beta_to_taiex: Optional[float] = None

    # Constraints validation
    meets_liquidity_req: bool = True
    meets_size_req: bool = True


@dataclass
class FactorPortfolio:
    """Container for factor-based portfolio with performance attribution."""
    portfolio_date: date
    positions: List[PortfolioPosition]

    # Portfolio-level metrics
    total_weight: float = 1.0
    position_count: int = 0

    # Factor exposures
    value_exposure: float = 0.0
    flow_exposure: float = 0.0
    momentum_exposure: float = 0.0

    # Risk metrics
    portfolio_beta: Optional[float] = None
    expected_volatility: Optional[float] = None
    diversification_ratio: Optional[float] = None

    # Construction metadata
    combination_method: Optional[FactorCombinationMethod] = None
    objective: Optional[PortfolioObjective] = None
    regime: Optional[TaiwanMarketRegime] = None

    def __post_init__(self):
        """Calculate derived metrics."""
        self.position_count = len(self.positions)
        self.total_weight = sum(pos.weight for pos in self.positions)

        # Calculate factor exposures
        self.value_exposure = sum(pos.weight * pos.value_contribution for pos in self.positions)
        self.flow_exposure = sum(pos.weight * pos.flow_contribution for pos in self.positions)
        self.momentum_exposure = sum(pos.weight * pos.momentum_contribution for pos in self.positions)


class FactorCombinationStrategy(ABC):
    """
    Abstract base class for factor combination strategies.

    Provides common infrastructure for factor combination while allowing
    strategy-specific implementations of combination logic and weighting schemes.
    """

    def __init__(self,
                 factor_pipeline: FactorPipeline,
                 method: FactorCombinationMethod,
                 taiwan_market_params: Optional[Dict[str, Any]] = None):
        """
        Initialize factor combination strategy.

        Args:
            factor_pipeline: Integrated factor pipeline from Task 003
            method: Factor combination method to use
            taiwan_market_params: Taiwan-specific market parameters
        """
        self.factor_pipeline = factor_pipeline
        self.method = method

        # Taiwan market parameters with defaults
        self.taiwan_params = taiwan_market_params or self._get_default_taiwan_params()

        # Performance monitoring
        self.performance_metrics = {}
        self.combination_history = []

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _get_default_taiwan_params(self) -> Dict[str, Any]:
        """Get default Taiwan market parameters."""
        return {
            'trading_days_per_year': 250,
            'market_cap_threshold': 5e9,  # 5B TWD minimum market cap
            'liquidity_threshold': 1e6,    # 1M TWD average daily volume
            'max_single_position': 0.05,   # 5% maximum position size
            'sector_concentration_limit': 0.25,  # 25% maximum sector exposure
            'rebalance_frequency': 'monthly',
            'transaction_cost': 0.003,     # 0.3% transaction costs
            'benchmark': 'TAIEX',
            'risk_free_rate': 0.01,       # 1% risk-free rate
        }

    @abstractmethod
    def calculate_factor_weights(self,
                               integrated_metrics: Dict[str, IntegratedFactorMetrics],
                               market_regime: Optional[TaiwanMarketRegime] = None) -> FactorWeight:
        """
        Calculate factor weights based on strategy methodology.

        Args:
            integrated_metrics: Integrated factor metrics from pipeline
            market_regime: Current market regime for adaptive strategies

        Returns:
            FactorWeight with strategy-specific weights
        """
        pass

    @abstractmethod
    def combine_factors(self,
                       integrated_metrics: Dict[str, IntegratedFactorMetrics],
                       factor_weights: FactorWeight) -> Dict[str, CompositeFactorScore]:
        """
        Combine individual factor scores into composite scores.

        Args:
            integrated_metrics: Integrated factor metrics from pipeline
            factor_weights: Factor weights for combination

        Returns:
            Dictionary mapping symbols to composite factor scores
        """
        pass

    def calculate_composite_scores(self,
                                 symbols: List[str],
                                 as_of_date: date,
                                 market_regime: Optional[TaiwanMarketRegime] = None) -> Dict[str, CompositeFactorScore]:
        """
        Calculate composite factor scores for given symbols and date.

        Args:
            symbols: List of Taiwan stock symbols
            as_of_date: Calculation date
            market_regime: Current market regime (optional)

        Returns:
            Dictionary mapping symbols to composite factor scores
        """
        start_time = time.time()

        try:
            # Step 1: Get integrated factor metrics from pipeline
            self.logger.info(f"Calculating composite scores for {len(symbols)} symbols on {as_of_date}")

            integrated_metrics = self.factor_pipeline.calculate_integrated_factors(
                symbols=symbols,
                as_of_date=as_of_date,
                include_correlations=True,
                validate_quality=True
            )

            # Step 2: Calculate factor weights based on strategy
            factor_weights = self.calculate_factor_weights(integrated_metrics, market_regime)

            # Step 3: Combine factors using strategy-specific logic
            composite_scores = self.combine_factors(integrated_metrics, factor_weights)

            # Step 4: Add cross-sectional ranking
            self._add_cross_sectional_ranking(composite_scores)

            # Step 5: Performance monitoring
            elapsed_time = time.time() - start_time
            self._update_performance_metrics(len(symbols), elapsed_time, composite_scores)

            self.logger.info(
                f"Completed composite score calculation: "
                f"{len(composite_scores)}/{len(symbols)} scores in {elapsed_time:.2f}s"
            )

            return composite_scores

        except Exception as e:
            self.logger.error(f"Error calculating composite scores: {e}")
            raise RuntimeError(f"Composite score calculation failed: {e}")

    def _add_cross_sectional_ranking(self, composite_scores: Dict[str, CompositeFactorScore]) -> None:
        """Add cross-sectional ranking to composite scores."""

        # Get valid scores for ranking
        valid_scores = [
            (symbol, score) for symbol, score in composite_scores.items()
            if score.equal_weight_score is not None or score.smart_beta_score is not None
        ]

        if not valid_scores:
            return

        # Sort by primary score (smart_beta if available, else equal_weight)
        def get_primary_score(item):
            symbol, score = item
            return score.smart_beta_score if score.smart_beta_score is not None else score.equal_weight_score

        sorted_scores = sorted(valid_scores, key=get_primary_score, reverse=True)
        universe_size = len(sorted_scores)

        # Assign ranks and percentiles
        for rank, (symbol, score) in enumerate(sorted_scores, 1):
            score.cross_sectional_rank = rank
            score.percentile_rank = (universe_size - rank + 1) / universe_size
            score.universe_size = universe_size

    def _update_performance_metrics(self,
                                  symbol_count: int,
                                  elapsed_time: float,
                                  composite_scores: Dict[str, CompositeFactorScore]) -> None:
        """Update performance metrics for monitoring."""

        success_count = len(composite_scores)

        self.performance_metrics.update({
            'last_calculation_time': elapsed_time,
            'last_symbol_count': symbol_count,
            'last_success_count': success_count,
            'last_success_rate': success_count / symbol_count if symbol_count > 0 else 0,
            'last_processing_rate': symbol_count / elapsed_time if elapsed_time > 0 else 0,
            'timestamp': datetime.now(),
            'method': self.method.value
        })

        # Check performance targets
        target_time = 5.0  # 5 seconds target for combination logic
        if elapsed_time > target_time:
            self.logger.warning(
                f"Performance alert: Combination time {elapsed_time:.1f}s exceeds target {target_time}s"
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()


class EqualWeightStrategy(FactorCombinationStrategy):
    """
    Equal-weight factor combination strategy.

    Simple strategy that combines normalized factor scores with equal weights,
    providing a balanced baseline approach for factor combination. Optimized
    for Taiwan market with production quality coordination.
    """

    def __init__(self,
                 factor_pipeline: FactorPipeline,
                 custom_weights: Optional[Tuple[float, float, float]] = None,
                 taiwan_market_params: Optional[Dict[str, Any]] = None):
        """
        Initialize equal-weight strategy.

        Args:
            factor_pipeline: Integrated factor pipeline from Task 003
            custom_weights: Optional custom weights (value, flow, momentum)
            taiwan_market_params: Taiwan-specific parameters
        """
        super().__init__(factor_pipeline, FactorCombinationMethod.EQUAL_WEIGHT, taiwan_market_params)

        # Set weights (default to equal, or use custom)
        if custom_weights:
            self.base_weights = custom_weights
        else:
            self.base_weights = (1/3, 1/3, 1/3)  # Equal weights

        self.logger.info(f"Initialized EqualWeightStrategy with weights: {self.base_weights}")

    def calculate_factor_weights(self,
                               integrated_metrics: Dict[str, IntegratedFactorMetrics],
                               market_regime: Optional[TaiwanMarketRegime] = None) -> FactorWeight:
        """
        Calculate equal factor weights with production quality adjustments.

        For equal-weight strategy, we use fixed weights but adjust for
        production readiness differences between factor groups.
        """

        # Get factor group quality status
        factor_status = self.factor_pipeline.get_factor_group_status()

        # Start with base equal weights
        value_weight, flow_weight, momentum_weight = self.base_weights

        # Adjust weights based on production readiness
        # Reduce flow factor weight if not production ready
        if not factor_status[FactorGroupType.FLOW].production_ready:
            self.logger.warning("Flow factors not production ready - reducing weight by 50%")
            flow_weight *= 0.5

            # Redistribute to other factors
            redistribution = flow_weight * 0.5  # Half of reduction
            value_weight += redistribution * 0.7  # Favor value (higher quality)
            momentum_weight += redistribution * 0.3

        # Create factor weight with validation
        return FactorWeight(
            value_weight=value_weight,
            flow_weight=flow_weight,
            momentum_weight=momentum_weight,
            weight_date=date.today(),
            regime=market_regime,
            confidence_score=0.9,  # High confidence for equal-weight
            rebalance_trigger="equal_weight_baseline"
        )

    def combine_factors(self,
                       integrated_metrics: Dict[str, IntegratedFactorMetrics],
                       factor_weights: FactorWeight) -> Dict[str, CompositeFactorScore]:
        """
        Combine factors using equal-weight methodology with quality coordination.
        """
        composite_scores = {}

        for symbol, metrics in integrated_metrics.items():
            # Initialize composite score
            score = CompositeFactorScore(
                symbol=symbol,
                date=metrics.date,
                factor_weights=factor_weights
            )

            # Extract individual normalized scores
            factor_scores = []
            factor_count = 0

            if metrics.normalized_value_score is not None:
                score.value_score = metrics.normalized_value_score
                factor_scores.append(metrics.normalized_value_score * factor_weights.value_weight)
                factor_count += 1

            if metrics.normalized_flow_score is not None:
                score.flow_score = metrics.normalized_flow_score
                factor_scores.append(metrics.normalized_flow_score * factor_weights.flow_weight)
                factor_count += 1

            if metrics.normalized_momentum_score is not None:
                score.momentum_score = metrics.normalized_momentum_score
                factor_scores.append(metrics.normalized_momentum_score * factor_weights.momentum_weight)
                factor_count += 1

            # Calculate composite score if we have any factors
            if factor_scores:
                score.equal_weight_score = sum(factor_scores)
                score.factor_count = factor_count
                score.data_completeness = metrics.total_data_completeness

                # Determine production readiness
                score.production_ready = self._assess_production_readiness(metrics, factor_count)

            composite_scores[symbol] = score

        return composite_scores

    def _assess_production_readiness(self,
                                   metrics: IntegratedFactorMetrics,
                                   factor_count: int) -> bool:
        """Assess if composite score is production ready."""

        # Require at least 2 factors for production use
        if factor_count < 2:
            return False

        # Require minimum data completeness
        if metrics.total_data_completeness < 0.7:
            return False

        # Must have value factors (highest quality) for production
        if metrics.normalized_value_score is None:
            return False

        return True


class SmartBetaStrategy(FactorCombinationStrategy):
    """
    Smart-beta factor combination strategy with risk-adjusted weighting.

    Advanced strategy that dynamically weights factors based on Taiwan market
    characteristics, historical performance, and risk-adjusted metrics.
    Optimized for Taiwan market patterns with regime awareness.
    """

    def __init__(self,
                 factor_pipeline: FactorPipeline,
                 lookback_periods: int = 12,
                 min_weight: float = 0.1,
                 max_weight: float = 0.6,
                 taiwan_market_params: Optional[Dict[str, Any]] = None):
        """
        Initialize smart-beta strategy.

        Args:
            factor_pipeline: Integrated factor pipeline from Task 003
            lookback_periods: Months of history for weight calculation
            min_weight: Minimum factor weight
            max_weight: Maximum factor weight
            taiwan_market_params: Taiwan-specific parameters
        """
        super().__init__(factor_pipeline, FactorCombinationMethod.SMART_BETA, taiwan_market_params)

        self.lookback_periods = lookback_periods
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Historical performance tracking for weight calculation
        self.factor_performance_history = defaultdict(list)
        self.correlation_history = []

        self.logger.info(
            f"Initialized SmartBetaStrategy with lookback={lookback_periods} months, "
            f"weight_range=({min_weight}, {max_weight})"
        )

    def calculate_factor_weights(self,
                               integrated_metrics: Dict[str, IntegratedFactorMetrics],
                               market_regime: Optional[TaiwanMarketRegime] = None) -> FactorWeight:
        """
        Calculate risk-adjusted factor weights for Taiwan market.

        Weights are determined by:
        1. Historical factor performance (Information Ratio)
        2. Factor correlations (diversification benefit)
        3. Taiwan market regime characteristics
        4. Production readiness adjustments
        """

        # Get factor group quality status
        factor_status = self.factor_pipeline.get_factor_group_status()

        # Calculate base weights using historical performance
        base_weights = self._calculate_base_weights()

        # Apply regime-specific adjustments
        regime_weights = self._apply_regime_adjustments(base_weights, market_regime)

        # Apply production readiness adjustments
        final_weights = self._apply_production_adjustments(regime_weights, factor_status)

        # Ensure weights are within bounds and sum to 1
        final_weights = self._normalize_weights(final_weights)

        return FactorWeight(
            value_weight=final_weights[0],
            flow_weight=final_weights[1],
            momentum_weight=final_weights[2],
            weight_date=date.today(),
            regime=market_regime,
            confidence_score=self._calculate_weight_confidence(final_weights),
            rebalance_trigger="smart_beta_optimization"
        )

    def _calculate_base_weights(self) -> Tuple[float, float, float]:
        """Calculate base weights using historical performance metrics."""

        # If no history, use equal weights
        if not self.factor_performance_history:
            return (1/3, 1/3, 1/3)

        # Calculate information ratios for each factor
        factor_scores = {}

        for factor_type in ['value', 'flow', 'momentum']:
            if factor_type in self.factor_performance_history:
                performance_data = self.factor_performance_history[factor_type]

                if len(performance_data) >= 3:  # Minimum data for calculation
                    returns = np.array([p['return'] for p in performance_data])
                    volatility = np.std(returns) if len(returns) > 1 else 1.0
                    mean_return = np.mean(returns)

                    # Information Ratio (risk-adjusted return)
                    factor_scores[factor_type] = mean_return / max(volatility, 0.01)
                else:
                    factor_scores[factor_type] = 0.0
            else:
                factor_scores[factor_type] = 0.0

        # Convert scores to weights
        total_score = sum(max(score, 0) for score in factor_scores.values())

        if total_score > 0:
            weights = [
                max(factor_scores['value'], 0) / total_score,
                max(factor_scores['flow'], 0) / total_score,
                max(factor_scores['momentum'], 0) / total_score
            ]
        else:
            weights = [1/3, 1/3, 1/3]

        return tuple(weights)

    def _apply_regime_adjustments(self,
                                base_weights: Tuple[float, float, float],
                                regime: Optional[TaiwanMarketRegime]) -> Tuple[float, float, float]:
        """Apply Taiwan market regime-specific weight adjustments."""

        if regime is None:
            return base_weights

        value_weight, flow_weight, momentum_weight = base_weights

        # Taiwan market regime-specific adjustments
        if regime == TaiwanMarketRegime.TRENDING_BULL:
            # Increase momentum in trending markets
            momentum_weight *= 1.3
            value_weight *= 0.9

        elif regime == TaiwanMarketRegime.TRENDING_BEAR:
            # Increase value in bear markets
            value_weight *= 1.3
            momentum_weight *= 0.8

        elif regime == TaiwanMarketRegime.MEAN_REVERTING:
            # Increase value and flow in mean-reverting markets
            value_weight *= 1.2
            flow_weight *= 1.1
            momentum_weight *= 0.7

        elif regime == TaiwanMarketRegime.HIGH_VOLATILITY:
            # Reduce all factor exposures in high volatility
            value_weight *= 0.9
            flow_weight *= 0.8
            momentum_weight *= 0.8

        elif regime == TaiwanMarketRegime.RECOVERY:
            # Balanced approach in recovery
            value_weight *= 1.1
            flow_weight *= 1.2
            momentum_weight *= 1.0

        return (value_weight, flow_weight, momentum_weight)

    def _apply_production_adjustments(self,
                                    regime_weights: Tuple[float, float, float],
                                    factor_status: Dict[FactorGroupType, FactorGroupStatus]) -> Tuple[float, float, float]:
        """Apply production readiness adjustments to weights."""

        value_weight, flow_weight, momentum_weight = regime_weights

        # Reduce flow weight significantly if not production ready
        if not factor_status[FactorGroupType.FLOW].production_ready:
            self.logger.warning("Flow factors not production ready - applying weight reduction")

            # Reduce flow weight by 60%
            original_flow = flow_weight
            flow_weight *= 0.4

            # Redistribute to value and momentum (favor value for quality)
            redistribution = original_flow - flow_weight
            value_weight += redistribution * 0.7  # 70% to value
            momentum_weight += redistribution * 0.3  # 30% to momentum

        return (value_weight, flow_weight, momentum_weight)

    def _normalize_weights(self, weights: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Normalize weights to sum to 1 and enforce min/max constraints."""

        value_weight, flow_weight, momentum_weight = weights

        # First normalize to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            value_weight /= total_weight
            flow_weight /= total_weight
            momentum_weight /= total_weight
        else:
            # Fallback to equal weights
            value_weight = flow_weight = momentum_weight = 1/3

        # Apply min/max constraints iteratively
        weights_array = np.array([value_weight, flow_weight, momentum_weight])

        # Clip to max constraint first
        clipped_high = np.clip(weights_array, 0, self.max_weight)
        excess = np.sum(weights_array - clipped_high)

        if excess > 0:
            # Redistribute excess to weights that aren't at max
            available_capacity = np.maximum(0, self.max_weight - clipped_high)
            total_capacity = np.sum(available_capacity)

            if total_capacity > 0:
                redistribution = excess * available_capacity / total_capacity
                clipped_high += redistribution

        # Apply minimum constraint
        clipped_weights = np.maximum(clipped_high, self.min_weight)

        # Final normalization to ensure sum to 1
        final_total = np.sum(clipped_weights)
        if final_total > 0:
            clipped_weights /= final_total
        else:
            clipped_weights = np.array([1/3, 1/3, 1/3])

        return tuple(clipped_weights)

    def _calculate_weight_confidence(self, weights: Tuple[float, float, float]) -> float:
        """Calculate confidence score for weight allocation."""

        # Higher confidence for more balanced weights
        weight_array = np.array(weights)
        balance_score = 1.0 - np.std(weight_array) / np.mean(weight_array)

        # Higher confidence with more historical data
        data_score = min(len(self.factor_performance_history) / 12.0, 1.0)

        # Combined confidence
        return (balance_score * 0.6 + data_score * 0.4)

    def combine_factors(self,
                       integrated_metrics: Dict[str, IntegratedFactorMetrics],
                       factor_weights: FactorWeight) -> Dict[str, CompositeFactorScore]:
        """
        Combine factors using smart-beta methodology with risk adjustments.
        """
        composite_scores = {}

        # Calculate correlation adjustments
        correlation_adjustments = self._calculate_correlation_adjustments()

        for symbol, metrics in integrated_metrics.items():
            # Initialize composite score
            score = CompositeFactorScore(
                symbol=symbol,
                date=metrics.date,
                factor_weights=factor_weights
            )

            # Extract individual normalized scores
            factor_scores = []
            factor_count = 0

            # Value factor contribution
            if metrics.normalized_value_score is not None:
                score.value_score = metrics.normalized_value_score
                value_contribution = (
                    metrics.normalized_value_score *
                    factor_weights.value_weight *
                    correlation_adjustments.get('value', 1.0)
                )
                factor_scores.append(value_contribution)
                factor_count += 1

            # Flow factor contribution
            if metrics.normalized_flow_score is not None:
                score.flow_score = metrics.normalized_flow_score
                flow_contribution = (
                    metrics.normalized_flow_score *
                    factor_weights.flow_weight *
                    correlation_adjustments.get('flow', 1.0)
                )
                factor_scores.append(flow_contribution)
                factor_count += 1

            # Momentum factor contribution
            if metrics.normalized_momentum_score is not None:
                score.momentum_score = metrics.normalized_momentum_score
                momentum_contribution = (
                    metrics.normalized_momentum_score *
                    factor_weights.momentum_weight *
                    correlation_adjustments.get('momentum', 1.0)
                )
                factor_scores.append(momentum_contribution)
                factor_count += 1

            # Calculate composite scores
            if factor_scores:
                # Smart-beta score with correlation adjustments
                score.smart_beta_score = sum(factor_scores)

                # Also calculate equal-weight for comparison
                equal_weight_scores = []
                if score.value_score is not None:
                    equal_weight_scores.append(score.value_score / 3)
                if score.flow_score is not None:
                    equal_weight_scores.append(score.flow_score / 3)
                if score.momentum_score is not None:
                    equal_weight_scores.append(score.momentum_score / 3)

                score.equal_weight_score = sum(equal_weight_scores)
                score.factor_count = factor_count
                score.data_completeness = metrics.total_data_completeness

                # Determine production readiness
                score.production_ready = self._assess_production_readiness(metrics, factor_count)

            composite_scores[symbol] = score

        return composite_scores

    def _calculate_correlation_adjustments(self) -> Dict[str, float]:
        """Calculate correlation-based adjustments to factor weights."""

        # If no correlation history, no adjustments
        if not self.correlation_history:
            return {'value': 1.0, 'flow': 1.0, 'momentum': 1.0}

        # Get latest correlation matrix
        latest_corr = self.correlation_history[-1]

        # Reduce weight for highly correlated factors
        adjustments = {}

        # Value factor adjustment
        value_corr = max(
            abs(latest_corr.value_flow_correlation),
            abs(latest_corr.value_momentum_correlation)
        )
        adjustments['value'] = 1.0 - (value_corr * 0.2)  # Reduce by up to 20%

        # Flow factor adjustment
        flow_corr = max(
            abs(latest_corr.value_flow_correlation),
            abs(latest_corr.flow_momentum_correlation)
        )
        adjustments['flow'] = 1.0 - (flow_corr * 0.2)

        # Momentum factor adjustment
        momentum_corr = max(
            abs(latest_corr.value_momentum_correlation),
            abs(latest_corr.flow_momentum_correlation)
        )
        adjustments['momentum'] = 1.0 - (momentum_corr * 0.2)

        return adjustments

    def _assess_production_readiness(self,
                                   metrics: IntegratedFactorMetrics,
                                   factor_count: int) -> bool:
        """Assess if composite score is production ready for smart-beta."""

        # Require at least 2 factors for production use
        if factor_count < 2:
            return False

        # Require higher data completeness for smart-beta
        if metrics.total_data_completeness < 0.8:
            return False

        # Must have value factors (highest quality) for production
        if metrics.normalized_value_score is None:
            return False

        # For smart-beta, prefer having momentum as well
        if metrics.normalized_momentum_score is None and factor_count < 3:
            return False

        return True

    def update_performance_history(self,
                                 factor_returns: Dict[str, float],
                                 date: date) -> None:
        """Update factor performance history for weight calculation."""

        for factor_type, return_val in factor_returns.items():
            self.factor_performance_history[factor_type].append({
                'date': date,
                'return': return_val,
                'timestamp': datetime.now()
            })

            # Keep only recent history
            if len(self.factor_performance_history[factor_type]) > self.lookback_periods:
                self.factor_performance_history[factor_type].pop(0)

    def update_correlation_history(self, correlation_matrix: FactorCorrelationMatrix) -> None:
        """Update correlation history for adjustment calculations."""

        self.correlation_history.append(correlation_matrix)

        # Keep only recent history
        if len(self.correlation_history) > self.lookback_periods:
            self.correlation_history.pop(0)


class FactorPortfolioConstructor:
    """
    Taiwan-optimized portfolio constructor using factor combination strategies.

    Builds portfolios from composite factor scores while respecting Taiwan market
    constraints, liquidity requirements, and risk management principles.
    """

    def __init__(self,
                 objective: PortfolioObjective = PortfolioObjective.ALPHA_GENERATION,
                 taiwan_market_params: Optional[Dict[str, Any]] = None):
        """
        Initialize portfolio constructor.

        Args:
            objective: Portfolio construction objective
            taiwan_market_params: Taiwan-specific market parameters
        """
        self.objective = objective
        self.taiwan_params = taiwan_market_params or self._get_default_taiwan_params()

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _get_default_taiwan_params(self) -> Dict[str, Any]:
        """Get default Taiwan market parameters for portfolio construction."""
        return {
            'max_positions': 50,           # Maximum number of positions
            'min_position_size': 0.005,    # 0.5% minimum position
            'max_position_size': 0.05,     # 5% maximum position
            'market_cap_threshold': 5e9,   # 5B TWD minimum market cap
            'liquidity_threshold': 1e6,    # 1M TWD daily volume
            'sector_limits': {             # Maximum sector exposures
                'technology': 0.3,
                'financials': 0.25,
                'industrials': 0.2,
                'consumer': 0.15,
                'materials': 0.1,
                'others': 0.1
            },
            'turnover_limit': 0.3,         # 30% maximum monthly turnover
            'cash_buffer': 0.02,           # 2% cash buffer
        }

    def construct_portfolio(self,
                          composite_scores: Dict[str, CompositeFactorScore],
                          market_data: Optional[Dict[str, Any]] = None,
                          current_portfolio: Optional[FactorPortfolio] = None) -> FactorPortfolio:
        """
        Construct optimized portfolio from composite factor scores.

        Args:
            composite_scores: Composite factor scores for universe
            market_data: Additional market data (market cap, volume, etc.)
            current_portfolio: Current portfolio for turnover optimization

        Returns:
            Optimized factor portfolio
        """
        start_time = time.time()

        try:
            self.logger.info(f"Constructing portfolio from {len(composite_scores)} factor scores")

            # Step 1: Filter universe for Taiwan market constraints
            eligible_scores = self._filter_universe(composite_scores, market_data)

            # Step 2: Rank stocks by composite scores
            ranked_stocks = self._rank_stocks(eligible_scores)

            # Step 3: Optimize portfolio weights
            positions = self._optimize_weights(ranked_stocks, current_portfolio)

            # Step 4: Create portfolio object
            portfolio = self._create_portfolio_object(positions, composite_scores)

            # Step 5: Validate portfolio constraints
            self._validate_portfolio_constraints(portfolio)

            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Portfolio construction completed: "
                f"{len(positions)} positions in {elapsed_time:.2f}s"
            )

            return portfolio

        except Exception as e:
            self.logger.error(f"Error constructing portfolio: {e}")
            raise RuntimeError(f"Portfolio construction failed: {e}")

    def _filter_universe(self,
                        composite_scores: Dict[str, CompositeFactorScore],
                        market_data: Optional[Dict[str, Any]]) -> Dict[str, CompositeFactorScore]:
        """Filter universe for Taiwan market constraints."""

        eligible_scores = {}

        for symbol, score in composite_scores.items():
            # Skip if no valid composite score
            primary_score = score.smart_beta_score or score.equal_weight_score
            if primary_score is None:
                continue

            # Skip if not production ready (for live trading)
            if not score.production_ready:
                self.logger.debug(f"Skipping {symbol}: not production ready")
                continue

            # Skip if insufficient data completeness
            if score.data_completeness < 0.7:
                self.logger.debug(f"Skipping {symbol}: low data completeness {score.data_completeness:.1%}")
                continue

            # Check market data constraints if available
            if market_data and symbol in market_data:
                market_info = market_data[symbol]

                # Market cap filter
                if market_info.get('market_cap', 0) < self.taiwan_params['market_cap_threshold']:
                    continue

                # Liquidity filter
                if market_info.get('avg_volume', 0) < self.taiwan_params['liquidity_threshold']:
                    continue

                # Update position object with market data validation
                score.meets_liquidity_req = True
                score.meets_size_req = True

            eligible_scores[symbol] = score

        self.logger.info(f"Universe filtered: {len(eligible_scores)}/{len(composite_scores)} stocks eligible")
        return eligible_scores

    def _rank_stocks(self, eligible_scores: Dict[str, CompositeFactorScore]) -> List[Tuple[str, CompositeFactorScore]]:
        """Rank stocks by composite factor scores."""

        def get_ranking_score(item):
            symbol, score = item
            # Prefer smart-beta score if available, else equal-weight
            return score.smart_beta_score if score.smart_beta_score is not None else score.equal_weight_score

        # Sort by composite score (descending)
        ranked_stocks = sorted(
            eligible_scores.items(),
            key=get_ranking_score,
            reverse=True
        )

        self.logger.debug(f"Stocks ranked: top score = {get_ranking_score(ranked_stocks[0]):.3f}")
        return ranked_stocks

    def _optimize_weights(self,
                         ranked_stocks: List[Tuple[str, CompositeFactorScore]],
                         current_portfolio: Optional[FactorPortfolio]) -> List[PortfolioPosition]:
        """Optimize portfolio weights based on objective and constraints."""

        positions = []
        max_positions = self.taiwan_params['max_positions']

        # Select top N stocks within position limit
        selected_stocks = ranked_stocks[:max_positions]

        if self.objective == PortfolioObjective.ALPHA_GENERATION:
            # Score-weighted positions
            positions = self._create_score_weighted_positions(selected_stocks)

        elif self.objective == PortfolioObjective.RISK_ADJUSTED:
            # Risk-adjusted equal weight
            positions = self._create_risk_adjusted_positions(selected_stocks)

        elif self.objective == PortfolioObjective.FACTOR_BALANCED:
            # Factor exposure balanced
            positions = self._create_factor_balanced_positions(selected_stocks)

        else:
            # Default: equal weight
            positions = self._create_equal_weight_positions(selected_stocks)

        # Apply turnover constraints if current portfolio provided
        if current_portfolio:
            positions = self._apply_turnover_constraints(positions, current_portfolio)

        return positions

    def _create_score_weighted_positions(self,
                                       selected_stocks: List[Tuple[str, CompositeFactorScore]]) -> List[PortfolioPosition]:
        """Create score-weighted portfolio positions."""

        positions = []

        # Calculate score-based weights
        scores = []
        for symbol, score in selected_stocks:
            primary_score = score.smart_beta_score if score.smart_beta_score is not None else score.equal_weight_score
            # Use rank-based scoring to reduce outlier impact
            scores.append(max(primary_score, 0))  # Ensure non-negative

        # Convert to weights
        total_score = sum(scores) if sum(scores) > 0 else len(scores)

        for i, (symbol, score) in enumerate(selected_stocks):
            weight = scores[i] / total_score

            # Apply position size constraints before creating position
            weight = np.clip(weight,
                           self.taiwan_params['min_position_size'],
                           self.taiwan_params['max_position_size'])

            # Create position with factor attribution
            position = self._create_position_with_attribution(symbol, weight, score)
            positions.append(position)

        # Normalize weights to sum to (1 - cash_buffer) while respecting position limits
        target_weight = 1.0 - self.taiwan_params['cash_buffer']
        total_weight = sum(pos.weight for pos in positions)

        if total_weight > 0:
            # Calculate scaling factor
            scale_factor = target_weight / total_weight

            # Apply scaling but respect max position size
            for position in positions:
                scaled_weight = position.weight * scale_factor

                # If scaled weight exceeds max, cap it
                if scaled_weight > self.taiwan_params['max_position_size']:
                    position.weight = self.taiwan_params['max_position_size']
                else:
                    position.weight = scaled_weight

            # Renormalize if total exceeds target due to capping
            final_total = sum(pos.weight for pos in positions)
            if final_total > target_weight:
                excess = final_total - target_weight
                uncapped_positions = [pos for pos in positions if pos.weight < self.taiwan_params['max_position_size']]

                if uncapped_positions:
                    reduction_per_position = excess / len(uncapped_positions)
                    for position in uncapped_positions:
                        position.weight = max(
                            position.weight - reduction_per_position,
                            self.taiwan_params['min_position_size']
                        )

        return positions

    def _create_equal_weight_positions(self,
                                     selected_stocks: List[Tuple[str, CompositeFactorScore]]) -> List[PortfolioPosition]:
        """Create equal-weight portfolio positions."""

        positions = []
        target_weight = 1.0 - self.taiwan_params['cash_buffer']
        equal_weight = target_weight / len(selected_stocks)

        for symbol, score in selected_stocks:
            position = self._create_position_with_attribution(symbol, equal_weight, score)
            positions.append(position)

        return positions

    def _create_risk_adjusted_positions(self,
                                      selected_stocks: List[Tuple[str, CompositeFactorScore]]) -> List[PortfolioPosition]:
        """Create risk-adjusted portfolio positions."""

        # For now, use equal weights (risk adjustment requires volatility data)
        # TODO: Implement risk adjustment when volatility data available
        return self._create_equal_weight_positions(selected_stocks)

    def _create_factor_balanced_positions(self,
                                        selected_stocks: List[Tuple[str, CompositeFactorScore]]) -> List[PortfolioPosition]:
        """Create factor exposure balanced portfolio positions."""

        # For now, use equal weights (factor balancing requires factor loadings)
        # TODO: Implement factor balancing when factor loadings available
        return self._create_equal_weight_positions(selected_stocks)

    def _create_position_with_attribution(self,
                                        symbol: str,
                                        weight: float,
                                        score: CompositeFactorScore) -> PortfolioPosition:
        """Create portfolio position with factor attribution."""

        # Calculate factor contributions based on factor weights
        factor_weights = score.factor_weights

        if factor_weights:
            value_contribution = (score.value_score or 0) * factor_weights.value_weight
            flow_contribution = (score.flow_score or 0) * factor_weights.flow_weight
            momentum_contribution = (score.momentum_score or 0) * factor_weights.momentum_weight
        else:
            # Default equal attribution
            value_contribution = (score.value_score or 0) / 3
            flow_contribution = (score.flow_score or 0) / 3
            momentum_contribution = (score.momentum_score or 0) / 3

        return PortfolioPosition(
            symbol=symbol,
            weight=weight,
            composite_score=score,
            value_contribution=value_contribution,
            flow_contribution=flow_contribution,
            momentum_contribution=momentum_contribution,
            meets_liquidity_req=getattr(score, 'meets_liquidity_req', True),
            meets_size_req=getattr(score, 'meets_size_req', True)
        )

    def _apply_turnover_constraints(self,
                                  new_positions: List[PortfolioPosition],
                                  current_portfolio: FactorPortfolio) -> List[PortfolioPosition]:
        """Apply turnover constraints to new portfolio."""

        # Calculate turnover
        current_weights = {pos.symbol: pos.weight for pos in current_portfolio.positions}
        new_weights = {pos.symbol: pos.weight for pos in new_positions}

        turnover = 0.0
        for symbol in set(current_weights.keys()) | set(new_weights.keys()):
            current_weight = current_weights.get(symbol, 0)
            new_weight = new_weights.get(symbol, 0)
            turnover += abs(new_weight - current_weight)

        turnover /= 2  # One-way turnover

        # If turnover exceeds limit, blend with current portfolio
        turnover_limit = self.taiwan_params['turnover_limit']
        if turnover > turnover_limit:
            self.logger.warning(f"Turnover {turnover:.1%} exceeds limit {turnover_limit:.1%} - applying constraints")

            # Simple approach: reduce weight changes proportionally
            reduction_factor = turnover_limit / turnover

            for position in new_positions:
                symbol = position.symbol
                current_weight = current_weights.get(symbol, 0)
                weight_change = position.weight - current_weight

                # Reduce weight change
                position.weight = current_weight + (weight_change * reduction_factor)

        return new_positions

    def _create_portfolio_object(self,
                               positions: List[PortfolioPosition],
                               original_scores: Dict[str, CompositeFactorScore]) -> FactorPortfolio:
        """Create portfolio object with metadata."""

        # Determine combination method from first score with factor weights
        combination_method = None
        for position in positions:
            if position.composite_score.factor_weights:
                # Check if weights suggest equal-weight or smart-beta
                weights = position.composite_score.factor_weights
                if abs(weights.value_weight - 1/3) < 0.05 and abs(weights.flow_weight - 1/3) < 0.05:
                    combination_method = FactorCombinationMethod.EQUAL_WEIGHT
                else:
                    combination_method = FactorCombinationMethod.SMART_BETA
                break

        portfolio = FactorPortfolio(
            portfolio_date=date.today(),
            positions=positions,
            combination_method=combination_method,
            objective=self.objective
        )

        return portfolio

    def _validate_portfolio_constraints(self, portfolio: FactorPortfolio) -> None:
        """Validate portfolio meets Taiwan market constraints."""

        # Check total weight
        if abs(portfolio.total_weight - (1.0 - self.taiwan_params['cash_buffer'])) > 0.01:
            self.logger.warning(f"Portfolio weight {portfolio.total_weight:.1%} deviates from target")

        # Check position count
        if portfolio.position_count > self.taiwan_params['max_positions']:
            raise ValueError(f"Portfolio has {portfolio.position_count} positions, exceeds limit {self.taiwan_params['max_positions']}")

        # Check individual position sizes
        for position in portfolio.positions:
            if position.weight > self.taiwan_params['max_position_size']:
                raise ValueError(f"Position {position.symbol} weight {position.weight:.1%} exceeds limit {self.taiwan_params['max_position_size']:.1%}")

            if position.weight < self.taiwan_params['min_position_size']:
                self.logger.warning(f"Position {position.symbol} weight {position.weight:.1%} below minimum {self.taiwan_params['min_position_size']:.1%}")

        self.logger.info("Portfolio constraint validation passed")


# Factory functions for easy instantiation

def create_equal_weight_strategy(factor_pipeline: FactorPipeline,
                               custom_weights: Optional[Tuple[float, float, float]] = None,
                               **kwargs) -> EqualWeightStrategy:
    """Factory function to create EqualWeightStrategy."""
    return EqualWeightStrategy(factor_pipeline, custom_weights, **kwargs)


def create_smart_beta_strategy(factor_pipeline: FactorPipeline,
                             lookback_periods: int = 12,
                             **kwargs) -> SmartBetaStrategy:
    """Factory function to create SmartBetaStrategy."""
    return SmartBetaStrategy(factor_pipeline, lookback_periods, **kwargs)


def create_portfolio_constructor(objective: PortfolioObjective = PortfolioObjective.ALPHA_GENERATION,
                               **kwargs) -> FactorPortfolioConstructor:
    """Factory function to create FactorPortfolioConstructor."""
    return FactorPortfolioConstructor(objective, **kwargs)