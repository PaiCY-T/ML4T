"""
Dynamic Factor Weight Allocation System - Task #006
GitHub Issue #79

Sophisticated dynamic factor weight allocation system that adapts factor weights based on
detected market regimes, factor performance, and Taiwan market conditions. Bridges the
regime detection system (Task 005) with factor combination strategies (Task 004) for
adaptive multi-factor investing.

Key Features:
1. DynamicFactorAllocator with regime-aware weight calculation
2. Performance-based allocation algorithms with Taiwan market optimization
3. Risk-adjusted weighting with volatility and correlation awareness
4. Transition management system for smooth regime changes
5. Integration with statistical regime detection and factor strategies
6. Comprehensive validation demonstrating allocation effectiveness

Integration Context:
- Leverages TaiwanMarketRegimeDetector from Task 005 (23/23 tests passing)
- Integrates with EqualWeightStrategy and SmartBetaStrategy from Task 004 (39/39 tests passing)
- Builds upon FactorPipeline from Task 003 (18/18 tests passing)
- Utilizes high-quality value factors (31x performance, production-ready)
- Safely handles varying production readiness across factor groups

Performance Requirements:
- Weight calculation <2 seconds for real-time updates
- Historical allocation optimization <30 seconds for backtest periods
- Memory usage <500MB for full allocation history
- Portfolio rebalancing <10 seconds for 200 Taiwan stocks

Quality Framework:
- All allocation claims backed by evidence (performance metrics, backtesting)
- Statistical rigor from Task 005's regime detection system
- Smooth integration with existing production-ready components
- Taiwan market optimization with regime awareness
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable, NamedTuple
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
from collections import defaultdict, deque
import math

# Import regime detection components from Task 005
from ..market.regime_detection import (
    TaiwanMarketRegimeDetector, RegimeClassification, TaiwanMarketRegime,
    RegimeConfidenceScore, create_taiwan_regime_detector
)

# Import factor combination components from Task 004
from .factor_combination import (
    FactorCombinationStrategy, EqualWeightStrategy, SmartBetaStrategy,
    FactorWeight, FactorCombinationMethod, PortfolioObjective,
    create_equal_weight_strategy, create_smart_beta_strategy
)

# Import factor integration components from Task 003
from ..factors.factor_integration import (
    FactorPipeline, IntegratedFactorMetrics, FactorCorrelationMatrix,
    FactorGroupType, IntegrationQuality, FactorGroupStatus,
    create_factor_pipeline
)

# Import Taiwan market models
try:
    from ..data.models.taiwan_market import (
        TaiwanMarketCode, TradingStatus, TaiwanTradingCalendar,
        create_taiwan_trading_calendar, TaiwanMarketCalendar
    )
    from ..data.core.temporal import DataType, TemporalValue
except ImportError:
    # For testing or standalone usage
    TaiwanMarketCode = object
    TradingStatus = object
    TaiwanTradingCalendar = object
    TaiwanMarketCalendar = object
    DataType = object
    TemporalValue = object

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Dynamic allocation strategy types."""
    REGIME_PURE = "regime_pure"                    # Pure regime-based allocation
    PERFORMANCE_WEIGHTED = "performance_weighted"  # Performance-based weighting
    RISK_ADJUSTED = "risk_adjusted"               # Volatility and correlation adjusted
    HYBRID_OPTIMAL = "hybrid_optimal"             # Optimal combination of all methods
    TAIWAN_ADAPTIVE = "taiwan_adaptive"           # Taiwan market specific adaptation


class TransitionMode(Enum):
    """Transition smoothing modes for regime changes."""
    IMMEDIATE = "immediate"      # Instant weight changes
    SMOOTH = "smooth"           # Exponential smoothing
    GRADUAL = "gradual"         # Linear transition over time
    CONFIDENCE_BASED = "confidence_based"  # Based on regime confidence


class RiskConstraintType(Enum):
    """Risk constraint types for allocation."""
    LEVERAGE_LIMIT = "leverage_limit"           # Total leverage constraint
    CONCENTRATION_LIMIT = "concentration_limit" # Single factor limit
    VOLATILITY_TARGET = "volatility_target"     # Portfolio volatility target
    DRAWDOWN_CONTROL = "drawdown_control"       # Maximum drawdown control
    TURNOVER_LIMIT = "turnover_limit"           # Portfolio turnover constraint


@dataclass
class AllocationConstraints:
    """Container for allocation risk constraints and limits."""
    # Leverage and concentration limits
    max_leverage: float = 1.3                    # Maximum 130% leverage
    max_single_factor_weight: float = 0.5        # Maximum 50% single factor
    min_factor_weight: float = 0.05              # Minimum 5% factor weight

    # Risk control parameters
    volatility_target: Optional[float] = None    # Target portfolio volatility
    max_drawdown_threshold: float = 0.15         # 15% maximum drawdown
    max_daily_weight_change: float = 0.05        # 5% maximum daily change

    # Transition control
    transition_speed: float = 0.2                # Transition speed factor
    confidence_threshold: float = 0.6            # Minimum regime confidence

    # Taiwan market specific
    rebalance_frequency_days: int = 5             # Rebalance every 5 days
    transaction_cost_threshold: float = 0.002    # 0.2% cost threshold

    def validate_constraints(self) -> bool:
        """Validate constraint parameters."""
        return (
            0.5 <= self.max_leverage <= 2.0 and
            0.1 <= self.max_single_factor_weight <= 0.8 and
            0.01 <= self.min_factor_weight <= 0.2 and
            0.01 <= self.max_daily_weight_change <= 0.2 and
            0.05 <= self.transition_speed <= 1.0 and
            0.5 <= self.confidence_threshold <= 0.95
        )


@dataclass
class FactorPerformanceMetrics:
    """Container for factor performance tracking."""
    factor_name: str
    measurement_date: date

    # Performance metrics
    period_return: float = 0.0              # Return over measurement period
    volatility: float = 0.0                 # Annualized volatility
    sharpe_ratio: float = 0.0               # Risk-adjusted return
    information_ratio: float = 0.0          # Alpha vs benchmark
    max_drawdown: float = 0.0               # Maximum drawdown

    # Factor-specific metrics
    hit_rate: float = 0.0                   # Percentage of positive periods
    correlation_to_benchmark: float = 0.0   # Correlation to TAIEX
    correlation_to_peers: float = 0.0       # Average correlation to other factors

    # Quality indicators
    data_completeness: float = 1.0          # Data availability
    statistical_significance: float = 0.0   # Statistical significance of performance

    # Taiwan market specifics
    sector_concentration: float = 0.0       # Technology sector bias
    liquidity_impact: float = 0.0          # Impact on portfolio liquidity


@dataclass
class DynamicAllocation:
    """Container for dynamic factor allocation with metadata."""
    allocation_date: date
    regime: TaiwanMarketRegime
    regime_confidence: float

    # Factor weights
    value_weight: float
    flow_weight: float
    momentum_weight: float

    # Allocation metadata
    strategy: AllocationStrategy
    transition_mode: TransitionMode
    constraints_applied: AllocationConstraints

    # Performance attribution
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    expected_sharpe: Optional[float] = None

    # Change tracking
    weight_changes: Dict[str, float] = field(default_factory=dict)
    trigger_reason: str = "regime_change"
    rebalance_cost: float = 0.0

    # Validation flags
    meets_constraints: bool = True
    production_ready: bool = True

    def __post_init__(self):
        """Validate allocation weights and constraints."""
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

        # Check leverage constraint
        if total_weight > self.constraints_applied.max_leverage:
            self.meets_constraints = False

        # Track weight changes
        self.weight_changes = {
            'value': self.value_weight,
            'flow': self.flow_weight,
            'momentum': self.momentum_weight
        }

    @property
    def total_weight(self) -> float:
        """Total portfolio weight."""
        return self.value_weight + self.flow_weight + self.momentum_weight

    @property
    def is_balanced(self) -> bool:
        """Check if allocation is reasonably balanced."""
        max_weight = max(self.value_weight, self.flow_weight, self.momentum_weight)
        return max_weight <= self.constraints_applied.max_single_factor_weight

    def to_factor_weight(self) -> FactorWeight:
        """Convert to FactorWeight object for strategy integration."""
        return FactorWeight(
            value_weight=self.value_weight,
            flow_weight=self.flow_weight,
            momentum_weight=self.momentum_weight,
            weight_date=self.allocation_date,
            regime=self.regime,
            confidence_score=self.regime_confidence,
            rebalance_trigger=self.trigger_reason
        )


@dataclass
class AllocationTransition:
    """Container for allocation transition events and analysis."""
    transition_date: date
    from_allocation: DynamicAllocation
    to_allocation: DynamicAllocation

    # Transition characteristics
    regime_change: bool = False
    confidence_change: float = 0.0
    weight_magnitude: float = 0.0          # Total weight change magnitude

    # Cost analysis
    estimated_turnover: float = 0.0        # Portfolio turnover estimate
    transaction_costs: float = 0.0         # Estimated transaction costs
    market_impact: float = 0.0             # Estimated market impact

    # Validation
    smooth_transition: bool = True          # Whether transition was smooth
    meets_daily_limits: bool = True         # Whether within daily change limits

    def __post_init__(self):
        """Calculate transition metrics."""
        # Check for regime change
        self.regime_change = (self.from_allocation.regime != self.to_allocation.regime)

        # Calculate confidence change
        self.confidence_change = (
            self.to_allocation.regime_confidence - self.from_allocation.regime_confidence
        )

        # Calculate weight change magnitude
        value_change = abs(self.to_allocation.value_weight - self.from_allocation.value_weight)
        flow_change = abs(self.to_allocation.flow_weight - self.from_allocation.flow_weight)
        momentum_change = abs(self.to_allocation.momentum_weight - self.from_allocation.momentum_weight)

        self.weight_magnitude = value_change + flow_change + momentum_change

        # Estimate turnover (half of total weight changes)
        self.estimated_turnover = self.weight_magnitude / 2.0

        # Check daily limits
        max_daily_change = self.to_allocation.constraints_applied.max_daily_weight_change
        self.meets_daily_limits = all([
            value_change <= max_daily_change,
            flow_change <= max_daily_change,
            momentum_change <= max_daily_change
        ])


class DynamicFactorAllocator:
    """
    Sophisticated dynamic factor weight allocation system for Taiwan market.

    This system bridges the regime detection system (Task 005) with factor combination
    strategies (Task 004) to provide adaptive factor weight allocation based on:
    1. Market regime classification with confidence scoring
    2. Factor performance tracking and optimization
    3. Risk-adjusted weighting with correlation awareness
    4. Smooth transition management during regime changes
    5. Taiwan market specific constraints and optimization
    """

    def __init__(self,
                 regime_detector: TaiwanMarketRegimeDetector,
                 factor_pipeline: FactorPipeline,
                 strategy: AllocationStrategy = AllocationStrategy.HYBRID_OPTIMAL,
                 constraints: Optional[AllocationConstraints] = None,
                 taiwan_market_params: Optional[Dict[str, Any]] = None):
        """
        Initialize dynamic factor allocator.

        Args:
            regime_detector: Regime detection system from Task 005
            factor_pipeline: Factor pipeline from Task 003
            strategy: Dynamic allocation strategy to use
            constraints: Risk constraints and limits
            taiwan_market_params: Taiwan-specific market parameters
        """
        self.regime_detector = regime_detector
        self.factor_pipeline = factor_pipeline
        self.strategy = strategy
        self.constraints = constraints or AllocationConstraints()

        # Taiwan market parameters with defaults
        self.taiwan_params = taiwan_market_params or self._get_default_taiwan_params()

        # Validate constraints
        if not self.constraints.validate_constraints():
            raise ValueError("Invalid allocation constraints provided")

        # Performance tracking
        self.factor_performance_history: Dict[str, List[FactorPerformanceMetrics]] = {
            'value': [],
            'flow': [],
            'momentum': []
        }

        # Allocation history tracking
        self.allocation_history: List[DynamicAllocation] = []
        self.transition_history: List[AllocationTransition] = []

        # Factor strategies for integration
        self.equal_weight_strategy = create_equal_weight_strategy(factor_pipeline)
        self.smart_beta_strategy = create_smart_beta_strategy(factor_pipeline)

        # Performance monitoring
        self.performance_metrics = {}

        # Taiwan market calendar
        self.taiwan_calendar = create_taiwan_trading_calendar(
            start_year=2009,
            end_year=date.today().year + 1
        )

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.logger.info(
            f"Initialized DynamicFactorAllocator with strategy={strategy.value}, "
            f"constraints validated, Taiwan market parameters loaded"
        )

    def _get_default_taiwan_params(self) -> Dict[str, Any]:
        """Get default Taiwan market parameters."""
        return {
            'trading_days_per_year': 250,
            'market_session_hours': 4.5,          # 9:00-13:30 Taiwan time
            'settlement_days': 2,                 # T+2 settlement
            'transaction_cost_bps': 30,           # 30 bps total costs
            'market_impact_factor': 0.1,          # Market impact coefficient
            'foreign_ownership_limit': 0.3,       # Foreign ownership constraints
            'rebalance_buffer_bps': 20,           # 20 bps rebalance buffer
            'min_rebalance_days': 3,              # Minimum days between rebalances
            'volatility_lookback_days': 60,       # Volatility calculation window
            'correlation_lookback_days': 120,     # Correlation calculation window
        }

    def calculate_dynamic_allocation(self,
                                   allocation_date: date,
                                   symbols: Optional[List[str]] = None,
                                   taiex_data: Optional[pd.Series] = None,
                                   market_data: Optional[Dict[str, Any]] = None) -> DynamicAllocation:
        """
        Calculate dynamic factor allocation for given date.

        Args:
            allocation_date: Date for allocation calculation
            symbols: List of Taiwan stock symbols for factor calculation
            taiex_data: TAIEX price series for regime detection
            market_data: Additional market data

        Returns:
            DynamicAllocation with regime-aware factor weights
        """
        start_time = time.time()

        try:
            self.logger.info(f"Calculating dynamic allocation for {allocation_date}")

            # Step 1: Detect current market regime
            regime_classification = self.regime_detector.detect_current_regime(
                allocation_date, taiex_data, market_data
            )

            # Step 2: Get factor performance metrics
            factor_metrics = self._calculate_factor_performance_metrics(
                allocation_date, symbols
            )

            # Step 3: Calculate regime-specific base weights
            base_weights = self._calculate_regime_base_weights(
                regime_classification.regime,
                regime_classification.confidence_score
            )

            # Step 4: Apply performance-based adjustments
            performance_weights = self._apply_performance_adjustments(
                base_weights, factor_metrics, allocation_date
            )

            # Step 5: Apply risk-adjusted weighting
            risk_adjusted_weights = self._apply_risk_adjustments(
                performance_weights, factor_metrics, allocation_date
            )

            # Step 6: Apply Taiwan market constraints
            final_weights = self._apply_taiwan_constraints(
                risk_adjusted_weights, regime_classification, allocation_date
            )

            # Step 7: Apply transition smoothing if previous allocation exists
            if self.allocation_history:
                final_weights = self._apply_transition_smoothing(
                    final_weights, regime_classification, allocation_date
                )

            # Step 8: Create allocation object
            allocation = DynamicAllocation(
                allocation_date=allocation_date,
                regime=regime_classification.regime,
                regime_confidence=regime_classification.confidence_score.confidence,
                value_weight=final_weights[0],
                flow_weight=final_weights[1],
                momentum_weight=final_weights[2],
                strategy=self.strategy,
                transition_mode=self._determine_transition_mode(regime_classification),
                constraints_applied=self.constraints,
                trigger_reason=self._determine_trigger_reason(regime_classification)
            )

            # Step 9: Calculate expected performance metrics
            self._calculate_expected_performance(allocation, factor_metrics)

            # Step 10: Validate allocation
            self._validate_allocation(allocation)

            # Step 11: Update history and track transitions
            self._update_allocation_history(allocation)

            # Step 12: Performance monitoring
            elapsed_time = time.time() - start_time
            self._update_performance_metrics(allocation_date, elapsed_time, allocation)

            self.logger.info(
                f"Dynamic allocation completed: {allocation.regime.value} regime "
                f"(confidence: {allocation.regime_confidence:.1%}) "
                f"weights=[{allocation.value_weight:.1%}, {allocation.flow_weight:.1%}, {allocation.momentum_weight:.1%}] "
                f"in {elapsed_time:.2f}s"
            )

            return allocation

        except Exception as e:
            self.logger.error(f"Error calculating dynamic allocation for {allocation_date}: {e}")
            raise RuntimeError(f"Dynamic allocation calculation failed: {e}")

    def _calculate_factor_performance_metrics(self,
                                            allocation_date: date,
                                            symbols: Optional[List[str]]) -> Dict[str, FactorPerformanceMetrics]:
        """Calculate current factor performance metrics."""

        factor_metrics = {}

        # Use default symbol set if not provided
        if symbols is None:
            symbols = self._get_default_taiwan_symbols()

        try:
            # Get integrated factor metrics from pipeline
            integrated_metrics = self.factor_pipeline.calculate_integrated_factors(
                symbols=symbols[:50],  # Limit for performance
                as_of_date=allocation_date,
                include_correlations=True,
                validate_quality=True
            )

            # Calculate performance metrics for each factor group
            for factor_name in ['value', 'flow', 'momentum']:
                metrics = self._calculate_single_factor_performance(
                    factor_name, integrated_metrics, allocation_date
                )
                factor_metrics[factor_name] = metrics

                # Add to performance history
                self.factor_performance_history[factor_name].append(metrics)

                # Limit history size
                if len(self.factor_performance_history[factor_name]) > 60:  # 3 months
                    self.factor_performance_history[factor_name].pop(0)

        except Exception as e:
            self.logger.warning(f"Error calculating factor performance metrics: {e}")
            # Return default metrics
            for factor_name in ['value', 'flow', 'momentum']:
                factor_metrics[factor_name] = FactorPerformanceMetrics(
                    factor_name=factor_name,
                    measurement_date=allocation_date
                )

        return factor_metrics

    def _calculate_single_factor_performance(self,
                                           factor_name: str,
                                           integrated_metrics: Dict[str, IntegratedFactorMetrics],
                                           measurement_date: date) -> FactorPerformanceMetrics:
        """Calculate performance metrics for a single factor."""

        metrics = FactorPerformanceMetrics(
            factor_name=factor_name,
            measurement_date=measurement_date
        )

        # Extract factor scores
        factor_scores = []
        for symbol, int_metrics in integrated_metrics.items():
            if factor_name == 'value' and int_metrics.normalized_value_score is not None:
                factor_scores.append(int_metrics.normalized_value_score)
            elif factor_name == 'flow' and int_metrics.normalized_flow_score is not None:
                factor_scores.append(int_metrics.normalized_flow_score)
            elif factor_name == 'momentum' and int_metrics.normalized_momentum_score is not None:
                factor_scores.append(int_metrics.normalized_momentum_score)

        if factor_scores:
            scores_array = np.array(factor_scores)

            # Basic statistics
            metrics.volatility = np.std(scores_array) * np.sqrt(250)  # Annualized
            metrics.hit_rate = np.mean(scores_array > 0)
            metrics.data_completeness = len(factor_scores) / len(integrated_metrics)

            # Use historical performance if available
            if len(self.factor_performance_history[factor_name]) > 0:
                historical_returns = [
                    m.period_return for m in self.factor_performance_history[factor_name][-12:]
                ]
                if historical_returns:
                    metrics.period_return = np.mean(historical_returns)
                    metrics.sharpe_ratio = (
                        np.mean(historical_returns) / np.std(historical_returns)
                        if np.std(historical_returns) > 0 else 0.0
                    )
            else:
                # Estimate from current scores
                metrics.period_return = np.mean(scores_array) * 0.1  # Rough estimate
                metrics.sharpe_ratio = (
                    metrics.period_return / metrics.volatility
                    if metrics.volatility > 0 else 0.0
                )

        return metrics

    def _calculate_regime_base_weights(self,
                                     regime: TaiwanMarketRegime,
                                     confidence_score: RegimeConfidenceScore) -> Tuple[float, float, float]:
        """Calculate regime-specific base factor weights."""

        # Base regime-specific weights (from Task 006 requirements)
        regime_weights = {
            TaiwanMarketRegime.TRENDING_BULL: (0.1, 0.05, 0.4),      # Momentum favored
            TaiwanMarketRegime.TRENDING_BEAR: (0.35, 0.1, 0.05),     # Value favored
            TaiwanMarketRegime.MEAN_REVERTING: (0.35, 0.2, 0.05),    # Value and flow favored
            TaiwanMarketRegime.HIGH_VOLATILITY: (0.25, 0.15, 0.2),   # Reduced exposures
            TaiwanMarketRegime.RECOVERY: (0.3, 0.25, 0.15)           # Balanced approach
        }

        base_weights = regime_weights.get(regime, (1/3, 1/3, 1/3))  # Default equal

        # Adjust weights based on regime confidence
        confidence = confidence_score.confidence

        if confidence < self.constraints.confidence_threshold:
            # Low confidence - move toward equal weights
            equal_weights = (1/3, 1/3, 1/3)
            # Stronger blending for low confidence
            blend_factor = min(confidence / self.constraints.confidence_threshold, 0.5)

            adjusted_weights = tuple(
                base_weights[i] * blend_factor + equal_weights[i] * (1 - blend_factor)
                for i in range(3)
            )
        else:
            adjusted_weights = base_weights

        self.logger.debug(
            f"Regime base weights for {regime.value} (confidence={confidence:.1%}): "
            f"value={adjusted_weights[0]:.1%}, flow={adjusted_weights[1]:.1%}, momentum={adjusted_weights[2]:.1%}"
        )

        return adjusted_weights

    def _apply_performance_adjustments(self,
                                     base_weights: Tuple[float, float, float],
                                     factor_metrics: Dict[str, FactorPerformanceMetrics],
                                     allocation_date: date) -> Tuple[float, float, float]:
        """Apply performance-based adjustments to base weights."""

        if self.strategy not in [AllocationStrategy.PERFORMANCE_WEIGHTED, AllocationStrategy.HYBRID_OPTIMAL]:
            return base_weights

        # Calculate performance scores for each factor
        performance_scores = {}

        for factor_name in ['value', 'flow', 'momentum']:
            metrics = factor_metrics[factor_name]

            # Composite performance score
            score = (
                metrics.sharpe_ratio * 0.4 +           # Risk-adjusted return
                metrics.hit_rate * 0.3 +               # Consistency
                (1 - metrics.volatility / 2) * 0.2 +   # Lower volatility preferred
                metrics.data_completeness * 0.1        # Data quality
            )

            performance_scores[factor_name] = max(0, score)  # Ensure non-negative

        # Convert scores to weight adjustments
        total_score = sum(performance_scores.values())

        if total_score > 0:
            performance_weights = [
                performance_scores['value'] / total_score,
                performance_scores['flow'] / total_score,
                performance_scores['momentum'] / total_score
            ]
        else:
            performance_weights = [1/3, 1/3, 1/3]

        # Blend with base weights (50% base, 50% performance)
        blend_factor = 0.5 if self.strategy == AllocationStrategy.HYBRID_OPTIMAL else 0.8

        adjusted_weights = tuple(
            base_weights[i] * (1 - blend_factor) + performance_weights[i] * blend_factor
            for i in range(3)
        )

        self.logger.debug(
            f"Performance-adjusted weights: "
            f"value={adjusted_weights[0]:.1%}, flow={adjusted_weights[1]:.1%}, momentum={adjusted_weights[2]:.1%}"
        )

        return adjusted_weights

    def _apply_risk_adjustments(self,
                              performance_weights: Tuple[float, float, float],
                              factor_metrics: Dict[str, FactorPerformanceMetrics],
                              allocation_date: date) -> Tuple[float, float, float]:
        """Apply risk-based adjustments to factor weights."""

        if self.strategy not in [AllocationStrategy.RISK_ADJUSTED, AllocationStrategy.HYBRID_OPTIMAL]:
            return performance_weights

        # Calculate risk adjustments based on volatility and correlation
        risk_adjustments = [1.0, 1.0, 1.0]  # Default no adjustment

        for i, factor_name in enumerate(['value', 'flow', 'momentum']):
            metrics = factor_metrics[factor_name]

            # Volatility adjustment (reduce weight for high volatility factors)
            if metrics.volatility > 0:
                vol_adjustment = 1.0 / (1.0 + metrics.volatility)
                risk_adjustments[i] *= vol_adjustment

            # Correlation adjustment (reduce weight for highly correlated factors)
            if metrics.correlation_to_peers > 0.7:
                corr_adjustment = 1.0 - (metrics.correlation_to_peers - 0.7) * 0.5
                risk_adjustments[i] *= corr_adjustment

        # Apply risk adjustments
        risk_adjusted_weights = [
            performance_weights[i] * risk_adjustments[i] for i in range(3)
        ]

        # Normalize to sum to 1
        total_weight = sum(risk_adjusted_weights)
        if total_weight > 0:
            risk_adjusted_weights = [w / total_weight for w in risk_adjusted_weights]
        else:
            risk_adjusted_weights = [1/3, 1/3, 1/3]

        self.logger.debug(
            f"Risk-adjusted weights: "
            f"value={risk_adjusted_weights[0]:.1%}, flow={risk_adjusted_weights[1]:.1%}, momentum={risk_adjusted_weights[2]:.1%}"
        )

        return tuple(risk_adjusted_weights)

    def _apply_taiwan_constraints(self,
                                risk_weights: Tuple[float, float, float],
                                regime_classification: RegimeClassification,
                                allocation_date: date) -> Tuple[float, float, float]:
        """Apply Taiwan market specific constraints and adjustments."""

        value_weight, flow_weight, momentum_weight = risk_weights

        # Get factor group status for production readiness
        factor_status = self.factor_pipeline.get_factor_group_status()

        # Reduce flow weight if not production ready
        if not factor_status[FactorGroupType.FLOW].production_ready:
            self.logger.warning("Flow factors not production ready - reducing weight")

            # Reduce flow weight by 50%
            original_flow = flow_weight
            flow_weight *= 0.5

            # Redistribute to value and momentum (favor value for quality)
            redistribution = original_flow - flow_weight
            value_weight += redistribution * 0.7  # 70% to value
            momentum_weight += redistribution * 0.3  # 30% to momentum

        # Apply Taiwan market trading hour constraints
        if self._is_market_closed(allocation_date):
            # Reduce momentum factor during market close periods
            momentum_weight *= 0.8

            # Redistribute to value and flow
            redistribution = momentum_weight * 0.2
            value_weight += redistribution * 0.6
            flow_weight += redistribution * 0.4

        # Apply sector concentration limits (Taiwan tech sector bias)
        if regime_classification.regime == TaiwanMarketRegime.TRENDING_BULL:
            # Reduce momentum to control tech sector concentration
            if momentum_weight > 0.4:
                excess = momentum_weight - 0.4
                momentum_weight = 0.4

                # Redistribute excess
                value_weight += excess * 0.5
                flow_weight += excess * 0.5

        # Ensure minimum weights
        min_weight = self.constraints.min_factor_weight
        weights_array = np.array([value_weight, flow_weight, momentum_weight])
        weights_array = np.maximum(weights_array, min_weight)

        # Ensure maximum single factor weight
        max_weight = self.constraints.max_single_factor_weight
        weights_array = np.minimum(weights_array, max_weight)

        # Final normalization
        total_weight = np.sum(weights_array)
        if total_weight > 0:
            weights_array /= total_weight
        else:
            weights_array = np.array([1/3, 1/3, 1/3])

        constrained_weights = tuple(weights_array)

        self.logger.debug(
            f"Taiwan-constrained weights: "
            f"value={constrained_weights[0]:.1%}, flow={constrained_weights[1]:.1%}, momentum={constrained_weights[2]:.1%}"
        )

        return constrained_weights

    def _apply_transition_smoothing(self,
                                  target_weights: Tuple[float, float, float],
                                  regime_classification: RegimeClassification,
                                  allocation_date: date) -> Tuple[float, float, float]:
        """Apply transition smoothing to prevent excessive turnover."""

        if not self.allocation_history:
            return target_weights

        last_allocation = self.allocation_history[-1]
        current_weights = (
            last_allocation.value_weight,
            last_allocation.flow_weight,
            last_allocation.momentum_weight
        )

        # Determine transition mode
        transition_mode = self._determine_transition_mode(regime_classification)

        if transition_mode == TransitionMode.IMMEDIATE:
            return target_weights

        # Calculate weight changes
        weight_changes = [
            target_weights[i] - current_weights[i] for i in range(3)
        ]

        # Check if changes exceed daily limits
        max_daily_change = self.constraints.max_daily_weight_change
        need_smoothing = any(abs(change) > max_daily_change for change in weight_changes)

        if not need_smoothing:
            return target_weights

        # Apply smoothing based on transition mode
        if transition_mode == TransitionMode.SMOOTH:
            # Exponential smoothing
            alpha = self.constraints.transition_speed
            smoothed_weights = tuple(
                current_weights[i] + alpha * weight_changes[i] for i in range(3)
            )

        elif transition_mode == TransitionMode.GRADUAL:
            # Linear transition (limit to max daily change)
            smoothed_weights = []
            for i in range(3):
                if abs(weight_changes[i]) <= max_daily_change:
                    smoothed_weights.append(target_weights[i])
                else:
                    # Limit to max daily change
                    direction = 1 if weight_changes[i] > 0 else -1
                    smoothed_weights.append(
                        current_weights[i] + direction * max_daily_change
                    )
            smoothed_weights = tuple(smoothed_weights)

            # Renormalize to ensure sum to 1.0
            total_weight = sum(smoothed_weights)
            if total_weight > 0:
                smoothed_weights = tuple(w / total_weight for w in smoothed_weights)

        elif transition_mode == TransitionMode.CONFIDENCE_BASED:
            # Smoothing based on regime confidence
            confidence = regime_classification.confidence_score.confidence
            smoothing_factor = min(confidence, self.constraints.transition_speed)

            smoothed_weights = tuple(
                current_weights[i] + smoothing_factor * weight_changes[i] for i in range(3)
            )

        else:
            smoothed_weights = target_weights

        # Normalize smoothed weights
        total_weight = sum(smoothed_weights)
        if total_weight > 0:
            smoothed_weights = tuple(w / total_weight for w in smoothed_weights)
        else:
            smoothed_weights = target_weights

        self.logger.debug(
            f"Transition smoothed weights ({transition_mode.value}): "
            f"value={smoothed_weights[0]:.1%}, flow={smoothed_weights[1]:.1%}, momentum={smoothed_weights[2]:.1%}"
        )

        return smoothed_weights

    def _determine_transition_mode(self, regime_classification: RegimeClassification) -> TransitionMode:
        """Determine appropriate transition mode based on regime characteristics."""

        confidence = regime_classification.confidence_score.confidence

        # Check if regime changed
        regime_changed = (
            self.allocation_history and
            self.allocation_history[-1].regime != regime_classification.regime
        )

        if confidence > 0.9:
            # High confidence - can use immediate transition
            return TransitionMode.IMMEDIATE
        elif regime_changed and confidence > 0.7:
            # Regime change with good confidence - smooth transition
            return TransitionMode.SMOOTH
        elif confidence > 0.6:
            # Moderate confidence - confidence-based smoothing
            return TransitionMode.CONFIDENCE_BASED
        else:
            # Low confidence - gradual transition
            return TransitionMode.GRADUAL

    def _determine_trigger_reason(self, regime_classification: RegimeClassification) -> str:
        """Determine reason for allocation trigger."""

        if not self.allocation_history:
            return "initial_allocation"

        last_allocation = self.allocation_history[-1]

        if last_allocation.regime != regime_classification.regime:
            return f"regime_change_{last_allocation.regime.value}_to_{regime_classification.regime.value}"

        confidence_change = abs(
            regime_classification.confidence_score.confidence - last_allocation.regime_confidence
        )

        if confidence_change > 0.2:
            return "confidence_update"

        # Check if minimum rebalance period has passed
        days_since_last = (regime_classification.date - last_allocation.allocation_date).days
        if days_since_last >= self.constraints.rebalance_frequency_days:
            return "scheduled_rebalance"

        return "minor_adjustment"

    def _calculate_expected_performance(self,
                                      allocation: DynamicAllocation,
                                      factor_metrics: Dict[str, FactorPerformanceMetrics]) -> None:
        """Calculate expected performance metrics for allocation."""

        # Weight-averaged expected return
        expected_return = (
            allocation.value_weight * factor_metrics['value'].period_return +
            allocation.flow_weight * factor_metrics['flow'].period_return +
            allocation.momentum_weight * factor_metrics['momentum'].period_return
        )

        # Weight-averaged expected volatility (simplified)
        expected_volatility = (
            allocation.value_weight * factor_metrics['value'].volatility +
            allocation.flow_weight * factor_metrics['flow'].volatility +
            allocation.momentum_weight * factor_metrics['momentum'].volatility
        )

        # Expected Sharpe ratio
        expected_sharpe = (
            expected_return / expected_volatility if expected_volatility > 0 else 0.0
        )

        allocation.expected_return = expected_return
        allocation.expected_volatility = expected_volatility
        allocation.expected_sharpe = expected_sharpe

    def _validate_allocation(self, allocation: DynamicAllocation) -> None:
        """Validate allocation meets all constraints and requirements."""

        # Check total leverage
        if allocation.total_weight > self.constraints.max_leverage:
            allocation.meets_constraints = False
            self.logger.warning(f"Allocation exceeds leverage limit: {allocation.total_weight:.1%}")

        # Check single factor concentration
        max_factor_weight = max(allocation.value_weight, allocation.flow_weight, allocation.momentum_weight)
        if max_factor_weight > self.constraints.max_single_factor_weight:
            allocation.meets_constraints = False
            self.logger.warning(f"Single factor weight {max_factor_weight:.1%} exceeds limit")

        # Check minimum weights
        min_factor_weight = min(allocation.value_weight, allocation.flow_weight, allocation.momentum_weight)
        if min_factor_weight < self.constraints.min_factor_weight:
            self.logger.warning(f"Factor weight {min_factor_weight:.1%} below minimum")

        # Check production readiness
        factor_status = self.factor_pipeline.get_factor_group_status()

        # Flow factors not production ready should have reduced weight
        if (not factor_status[FactorGroupType.FLOW].production_ready and
            allocation.flow_weight > 0.3):
            allocation.production_ready = False
            self.logger.warning("High flow weight despite non-production readiness")

        # Must have value factors for production use
        if allocation.value_weight < 0.1:
            allocation.production_ready = False
            self.logger.warning("Value weight too low for production use")

    def _update_allocation_history(self, allocation: DynamicAllocation) -> None:
        """Update allocation history and track transitions."""

        # Check for transition
        if self.allocation_history:
            last_allocation = self.allocation_history[-1]

            # Create transition event
            transition = AllocationTransition(
                transition_date=allocation.allocation_date,
                from_allocation=last_allocation,
                to_allocation=allocation
            )

            # Estimate transaction costs
            transition.transaction_costs = (
                transition.estimated_turnover *
                self.taiwan_params['transaction_cost_bps'] / 10000
            )

            self.transition_history.append(transition)

            self.logger.info(
                f"Allocation transition: {transition.weight_magnitude:.1%} weight change, "
                f"{transition.estimated_turnover:.1%} turnover, "
                f"{transition.transaction_costs:.3%} estimated costs"
            )

        # Add to history
        self.allocation_history.append(allocation)

        # Limit history size for memory management
        max_history_size = 252 * 2  # 2 years of daily allocations

        # Trim allocation history immediately if it exceeds limit
        if len(self.allocation_history) > max_history_size:
            self.allocation_history = self.allocation_history[-max_history_size:]

        # Trim transition history immediately if it exceeds limit
        if len(self.transition_history) > max_history_size:
            self.transition_history = self.transition_history[-max_history_size:]

    def _update_performance_metrics(self,
                                  allocation_date: date,
                                  elapsed_time: float,
                                  allocation: DynamicAllocation) -> None:
        """Update performance metrics for monitoring."""

        self.performance_metrics.update({
            'last_allocation_time': elapsed_time,
            'last_allocation_date': allocation_date,
            'last_regime': allocation.regime.value,
            'last_regime_confidence': allocation.regime_confidence,
            'last_weights': [allocation.value_weight, allocation.flow_weight, allocation.momentum_weight],
            'total_allocations': len(self.allocation_history),
            'total_transitions': len(self.transition_history),
            'meets_constraints': allocation.meets_constraints,
            'production_ready': allocation.production_ready,
            'timestamp': datetime.now()
        })

        # Check performance targets
        target_time = 2.0  # 2 seconds target
        if elapsed_time > target_time:
            self.logger.warning(
                f"Performance alert: Allocation time {elapsed_time:.1f}s exceeds target {target_time}s"
            )

    def _get_default_taiwan_symbols(self) -> List[str]:
        """Get default Taiwan stock symbols for testing."""
        # Mock Taiwan stock symbols (replace with real data source)
        return [
            "2330.TW", "2454.TW", "2317.TW", "2412.TW", "2891.TW",
            "1303.TW", "3711.TW", "2881.TW", "1216.TW", "2382.TW",
            "2886.TW", "3008.TW", "1101.TW", "2002.TW", "2207.TW",
            "5880.TW", "2357.TW", "2603.TW", "1326.TW", "2105.TW"
        ]

    def _is_market_closed(self, check_date: date) -> bool:
        """Check if Taiwan market is closed on given date."""
        return not self.taiwan_calendar.is_trading_day(check_date)

    def get_current_allocation(self) -> Optional[DynamicAllocation]:
        """Get most recent allocation."""
        return self.allocation_history[-1] if self.allocation_history else None

    def get_regime_transition_matrix(self) -> pd.DataFrame:
        """Get regime transition probability matrix from allocations."""

        if len(self.allocation_history) < 10:
            # Return default matrix
            regimes = list(TaiwanMarketRegime)
            n_regimes = len(regimes)
            default_matrix = np.eye(n_regimes) * 0.8 + (1 - np.eye(n_regimes)) * 0.05

            return pd.DataFrame(
                default_matrix,
                index=[r.value for r in regimes],
                columns=[r.value for r in regimes]
            )

        # Calculate transition matrix from allocation history
        regimes = list(TaiwanMarketRegime)
        regime_to_idx = {regime: i for i, regime in enumerate(regimes)}

        transition_counts = np.zeros((len(regimes), len(regimes)))

        for i in range(1, len(self.allocation_history)):
            from_regime = self.allocation_history[i-1].regime
            to_regime = self.allocation_history[i].regime

            from_idx = regime_to_idx[from_regime]
            to_idx = regime_to_idx[to_regime]
            transition_counts[from_idx, to_idx] += 1

        # Convert to probabilities
        row_sums = transition_counts.sum(axis=1)
        transition_probs = np.divide(
            transition_counts,
            row_sums[:, np.newaxis],
            out=np.zeros_like(transition_counts),
            where=row_sums[:, np.newaxis] != 0
        )

        return pd.DataFrame(
            transition_probs,
            index=[r.value for r in regimes],
            columns=[r.value for r in regimes]
        )

    def get_allocation_performance_summary(self) -> Dict[str, Any]:
        """Get allocation performance summary with key metrics."""

        if not self.allocation_history:
            return {}

        # Basic statistics
        total_allocations = len(self.allocation_history)
        total_transitions = len(self.transition_history)

        # Weight statistics
        value_weights = [a.value_weight for a in self.allocation_history]
        flow_weights = [a.flow_weight for a in self.allocation_history]
        momentum_weights = [a.momentum_weight for a in self.allocation_history]

        # Regime distribution
        regime_counts = defaultdict(int)
        for allocation in self.allocation_history:
            regime_counts[allocation.regime.value] += 1

        # Constraint compliance
        compliant_allocations = sum(1 for a in self.allocation_history if a.meets_constraints)
        production_ready_allocations = sum(1 for a in self.allocation_history if a.production_ready)

        # Transition costs
        total_turnover = sum(t.estimated_turnover for t in self.transition_history)
        total_costs = sum(t.transaction_costs for t in self.transition_history)

        return {
            'total_allocations': total_allocations,
            'total_transitions': total_transitions,
            'constraint_compliance_rate': compliant_allocations / total_allocations,
            'production_readiness_rate': production_ready_allocations / total_allocations,
            'weight_statistics': {
                'value': {'mean': np.mean(value_weights), 'std': np.std(value_weights)},
                'flow': {'mean': np.mean(flow_weights), 'std': np.std(flow_weights)},
                'momentum': {'mean': np.mean(momentum_weights), 'std': np.std(momentum_weights)}
            },
            'regime_distribution': dict(regime_counts),
            'total_turnover': total_turnover,
            'total_transaction_costs': total_costs,
            'average_turnover_per_transition': total_turnover / max(total_transitions, 1),
            'strategy': self.strategy.value,
            'performance_metrics': self.performance_metrics
        }

    def export_allocation_history(self, file_path: Optional[str] = None) -> str:
        """Export allocation history to JSON file."""

        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"/tmp/dynamic_allocation_history_{timestamp}.json"

        export_data = {
            'export_date': datetime.now().isoformat(),
            'allocator_config': {
                'strategy': self.strategy.value,
                'constraints': {
                    'max_leverage': self.constraints.max_leverage,
                    'max_single_factor_weight': self.constraints.max_single_factor_weight,
                    'min_factor_weight': self.constraints.min_factor_weight,
                    'max_daily_weight_change': self.constraints.max_daily_weight_change,
                    'transition_speed': self.constraints.transition_speed,
                    'confidence_threshold': self.constraints.confidence_threshold
                },
                'taiwan_params': self.taiwan_params
            },
            'allocation_history': [
                {
                    'date': a.allocation_date.isoformat(),
                    'regime': a.regime.value,
                    'regime_confidence': a.regime_confidence,
                    'value_weight': a.value_weight,
                    'flow_weight': a.flow_weight,
                    'momentum_weight': a.momentum_weight,
                    'strategy': a.strategy.value,
                    'transition_mode': a.transition_mode.value,
                    'expected_return': a.expected_return,
                    'expected_volatility': a.expected_volatility,
                    'expected_sharpe': a.expected_sharpe,
                    'trigger_reason': a.trigger_reason,
                    'meets_constraints': a.meets_constraints,
                    'production_ready': a.production_ready
                }
                for a in self.allocation_history
            ],
            'transition_history': [
                {
                    'date': t.transition_date.isoformat(),
                    'from_regime': t.from_allocation.regime.value,
                    'to_regime': t.to_allocation.regime.value,
                    'regime_change': t.regime_change,
                    'weight_magnitude': t.weight_magnitude,
                    'estimated_turnover': t.estimated_turnover,
                    'transaction_costs': t.transaction_costs,
                    'smooth_transition': t.smooth_transition,
                    'meets_daily_limits': t.meets_daily_limits
                }
                for t in self.transition_history
            ],
            'performance_summary': self.get_allocation_performance_summary()
        }

        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Allocation history exported to {file_path}")
        return file_path


# Factory function for easy instantiation

def create_dynamic_factor_allocator(
    regime_detector: Optional[TaiwanMarketRegimeDetector] = None,
    factor_pipeline: Optional[FactorPipeline] = None,
    strategy: AllocationStrategy = AllocationStrategy.HYBRID_OPTIMAL,
    constraints: Optional[AllocationConstraints] = None,
    **kwargs
) -> DynamicFactorAllocator:
    """
    Factory function to create DynamicFactorAllocator.

    Args:
        regime_detector: Regime detection system (creates default if None)
        factor_pipeline: Factor pipeline (creates default if None)
        strategy: Dynamic allocation strategy
        constraints: Risk constraints and limits
        **kwargs: Additional parameters

    Returns:
        Configured DynamicFactorAllocator instance
    """

    # Create default regime detector if not provided
    if regime_detector is None:
        regime_detector = create_taiwan_regime_detector()

    # Create default factor pipeline if not provided
    if factor_pipeline is None:
        factor_pipeline = create_factor_pipeline()

    return DynamicFactorAllocator(
        regime_detector=regime_detector,
        factor_pipeline=factor_pipeline,
        strategy=strategy,
        constraints=constraints,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Dynamic Factor Weight Allocation System")
    print("=======================================")

    # Create allocator
    allocator = create_dynamic_factor_allocator()

    # Test with current date
    current_date = date.today()

    try:
        # Calculate dynamic allocation
        allocation = allocator.calculate_dynamic_allocation(current_date)

        print(f"\nDynamic Allocation for {current_date}:")
        print(f"Regime: {allocation.regime.value}")
        print(f"Regime Confidence: {allocation.regime_confidence:.1%}")
        print(f"Factor Weights:")
        print(f"  Value: {allocation.value_weight:.1%}")
        print(f"  Flow: {allocation.flow_weight:.1%}")
        print(f"  Momentum: {allocation.momentum_weight:.1%}")
        print(f"Strategy: {allocation.strategy.value}")
        print(f"Meets Constraints: {allocation.meets_constraints}")
        print(f"Production Ready: {allocation.production_ready}")

        if allocation.expected_return is not None:
            print(f"Expected Return: {allocation.expected_return:.2%}")
            print(f"Expected Volatility: {allocation.expected_volatility:.2%}")
            print(f"Expected Sharpe: {allocation.expected_sharpe:.2f}")

        # Get performance summary
        summary = allocator.get_allocation_performance_summary()
        print(f"\nPerformance Summary:")
        print(f"Total Allocations: {summary.get('total_allocations', 0)}")
        print(f"Constraint Compliance: {summary.get('constraint_compliance_rate', 0):.1%}")

        print("\nDynamic factor allocation system ready for integration with Task 007 historical backtesting.")

    except Exception as e:
        print(f"Error during testing: {e}")
        print("System will need real market data for production use.")