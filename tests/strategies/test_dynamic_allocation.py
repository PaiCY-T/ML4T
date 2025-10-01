"""
Comprehensive tests for Dynamic Factor Weight Allocation System - Task #006
GitHub Issue #79

Test coverage:
1. DynamicFactorAllocator initialization and configuration
2. Regime-aware weight calculation with confidence scoring
3. Performance-based allocation adjustments
4. Risk-adjusted weighting with volatility and correlation awareness
5. Taiwan market constraint application and validation
6. Transition management and smoothing during regime changes
7. Integration with Tasks 004 and 005 components
8. Performance requirements validation
9. Edge cases and error handling
10. Export and persistence functionality
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Any

# Import the system under test
from src.strategies.dynamic_allocation import (
    DynamicFactorAllocator, AllocationStrategy, TransitionMode, RiskConstraintType,
    AllocationConstraints, FactorPerformanceMetrics, DynamicAllocation, AllocationTransition,
    create_dynamic_factor_allocator
)

# Import dependencies that will be mocked for testing
from src.market.regime_detection import (
    TaiwanMarketRegimeDetector, RegimeClassification, TaiwanMarketRegime,
    RegimeConfidenceScore
)
from src.strategies.factor_combination import FactorWeight, FactorCombinationMethod
from src.factors.factor_integration import (
    FactorPipeline, IntegratedFactorMetrics, FactorGroupType, FactorGroupStatus
)


class TestAllocationConstraints:
    """Test allocation constraints validation and behavior."""

    def test_default_constraints_are_valid(self):
        """Test that default constraints pass validation."""
        constraints = AllocationConstraints()
        assert constraints.validate_constraints()

    def test_constraint_validation_edge_cases(self):
        """Test constraint validation with edge case values."""
        # Test valid boundaries
        constraints = AllocationConstraints(
            max_leverage=0.5,
            max_single_factor_weight=0.1,
            min_factor_weight=0.01,
            max_daily_weight_change=0.01,
            transition_speed=0.05,
            confidence_threshold=0.5
        )
        assert constraints.validate_constraints()

        # Test invalid values
        invalid_constraints = AllocationConstraints(max_leverage=0.3)  # Too low
        assert not invalid_constraints.validate_constraints()

        invalid_constraints = AllocationConstraints(max_single_factor_weight=0.05)  # Too low
        assert not invalid_constraints.validate_constraints()

        invalid_constraints = AllocationConstraints(confidence_threshold=0.3)  # Too low
        assert not invalid_constraints.validate_constraints()

    def test_constraint_parameter_ranges(self):
        """Test that constraint parameters are within expected ranges."""
        constraints = AllocationConstraints()

        assert 0.5 <= constraints.max_leverage <= 2.0
        assert 0.1 <= constraints.max_single_factor_weight <= 0.8
        assert 0.01 <= constraints.min_factor_weight <= 0.2
        assert 0.01 <= constraints.max_daily_weight_change <= 0.2
        assert 0.05 <= constraints.transition_speed <= 1.0
        assert 0.5 <= constraints.confidence_threshold <= 0.95


class TestFactorPerformanceMetrics:
    """Test factor performance metrics calculation and validation."""

    def test_performance_metrics_initialization(self):
        """Test performance metrics object initialization."""
        test_date = date(2024, 1, 15)
        metrics = FactorPerformanceMetrics(
            factor_name="value",
            measurement_date=test_date
        )

        assert metrics.factor_name == "value"
        assert metrics.measurement_date == test_date
        assert metrics.period_return == 0.0
        assert metrics.volatility == 0.0
        assert metrics.data_completeness == 1.0

    def test_performance_metrics_with_values(self):
        """Test performance metrics with realistic values."""
        metrics = FactorPerformanceMetrics(
            factor_name="momentum",
            measurement_date=date(2024, 1, 15),
            period_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.67,
            hit_rate=0.55,
            data_completeness=0.85
        )

        assert metrics.period_return == 0.12
        assert metrics.volatility == 0.18
        assert metrics.sharpe_ratio == 0.67
        assert metrics.hit_rate == 0.55
        assert metrics.data_completeness == 0.85


class TestDynamicAllocation:
    """Test dynamic allocation object and validation."""

    def test_allocation_initialization_and_normalization(self):
        """Test allocation initialization with weight normalization."""
        constraints = AllocationConstraints()
        allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.TRENDING_BULL,
            regime_confidence=0.85,
            value_weight=0.5,
            flow_weight=0.3,
            momentum_weight=0.2,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )

        # Weights should be normalized to sum to 1.0
        assert abs(allocation.total_weight - 1.0) < 1e-10
        assert allocation.value_weight + allocation.flow_weight + allocation.momentum_weight == 1.0

    def test_allocation_weight_validation(self):
        """Test allocation weight validation rules."""
        constraints = AllocationConstraints()

        # Test negative weight rejection
        with pytest.raises(ValueError, match="All factor weights must be non-negative"):
            DynamicAllocation(
                allocation_date=date(2024, 1, 15),
                regime=TaiwanMarketRegime.MEAN_REVERTING,
                regime_confidence=0.7,
                value_weight=-0.1,
                flow_weight=0.6,
                momentum_weight=0.5,
                strategy=AllocationStrategy.HYBRID_OPTIMAL,
                transition_mode=TransitionMode.SMOOTH,
                constraints_applied=constraints
            )

        # Test zero total weight rejection
        with pytest.raises(ValueError, match="Total weight cannot be zero"):
            DynamicAllocation(
                allocation_date=date(2024, 1, 15),
                regime=TaiwanMarketRegime.HIGH_VOLATILITY,
                regime_confidence=0.8,
                value_weight=0.0,
                flow_weight=0.0,
                momentum_weight=0.0,
                strategy=AllocationStrategy.RISK_ADJUSTED,
                transition_mode=TransitionMode.GRADUAL,
                constraints_applied=constraints
            )

    def test_allocation_balance_assessment(self):
        """Test allocation balance assessment."""
        constraints = AllocationConstraints(max_single_factor_weight=0.5)

        # Balanced allocation
        balanced_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.RECOVERY,
            regime_confidence=0.75,
            value_weight=0.4,
            flow_weight=0.3,
            momentum_weight=0.3,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )
        assert balanced_allocation.is_balanced

        # Imbalanced allocation
        imbalanced_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.TRENDING_BULL,
            regime_confidence=0.9,
            value_weight=0.1,
            flow_weight=0.1,
            momentum_weight=0.8,
            strategy=AllocationStrategy.REGIME_PURE,
            transition_mode=TransitionMode.IMMEDIATE,
            constraints_applied=constraints
        )
        assert not imbalanced_allocation.is_balanced

    def test_allocation_to_factor_weight_conversion(self):
        """Test conversion to FactorWeight object."""
        constraints = AllocationConstraints()
        allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.MEAN_REVERTING,
            regime_confidence=0.8,
            value_weight=0.5,
            flow_weight=0.3,
            momentum_weight=0.2,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints,
            trigger_reason="regime_change"
        )

        factor_weight = allocation.to_factor_weight()

        assert isinstance(factor_weight, FactorWeight)
        assert factor_weight.value_weight == allocation.value_weight
        assert factor_weight.flow_weight == allocation.flow_weight
        assert factor_weight.momentum_weight == allocation.momentum_weight
        assert factor_weight.weight_date == allocation.allocation_date
        assert factor_weight.regime == allocation.regime
        assert factor_weight.confidence_score == allocation.regime_confidence
        assert factor_weight.rebalance_trigger == allocation.trigger_reason


class TestAllocationTransition:
    """Test allocation transition tracking and analysis."""

    def test_transition_calculation(self):
        """Test transition metrics calculation."""
        constraints = AllocationConstraints()

        from_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 14),
            regime=TaiwanMarketRegime.MEAN_REVERTING,
            regime_confidence=0.7,
            value_weight=0.5,
            flow_weight=0.3,
            momentum_weight=0.2,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )

        to_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.TRENDING_BULL,
            regime_confidence=0.85,
            value_weight=0.2,
            flow_weight=0.2,
            momentum_weight=0.6,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )

        transition = AllocationTransition(
            transition_date=date(2024, 1, 15),
            from_allocation=from_allocation,
            to_allocation=to_allocation
        )

        # Check regime change detection
        assert transition.regime_change

        # Check confidence change
        assert transition.confidence_change == (0.85 - 0.7)

        # Check weight magnitude calculation
        expected_magnitude = (
            abs(0.2 - 0.5) +  # value change
            abs(0.2 - 0.3) +  # flow change
            abs(0.6 - 0.2)    # momentum change
        )
        assert abs(transition.weight_magnitude - expected_magnitude) < 1e-10

        # Check turnover estimation
        assert transition.estimated_turnover == transition.weight_magnitude / 2.0

    def test_transition_daily_limits_check(self):
        """Test daily change limits validation."""
        constraints = AllocationConstraints(max_daily_weight_change=0.1)

        from_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 14),
            regime=TaiwanMarketRegime.MEAN_REVERTING,
            regime_confidence=0.7,
            value_weight=0.4,
            flow_weight=0.3,
            momentum_weight=0.3,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )

        # Small change - should meet limits
        small_change_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.MEAN_REVERTING,
            regime_confidence=0.75,
            value_weight=0.45,
            flow_weight=0.28,
            momentum_weight=0.27,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )

        small_transition = AllocationTransition(
            transition_date=date(2024, 1, 15),
            from_allocation=from_allocation,
            to_allocation=small_change_allocation
        )
        assert small_transition.meets_daily_limits

        # Large change - should exceed limits
        large_change_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.TRENDING_BULL,
            regime_confidence=0.9,
            value_weight=0.1,
            flow_weight=0.1,
            momentum_weight=0.8,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )

        large_transition = AllocationTransition(
            transition_date=date(2024, 1, 15),
            from_allocation=from_allocation,
            to_allocation=large_change_allocation
        )
        assert not large_transition.meets_daily_limits


class TestDynamicFactorAllocator:
    """Test the main DynamicFactorAllocator class."""

    @pytest.fixture
    def mock_regime_detector(self):
        """Create mock regime detector for testing."""
        detector = Mock(spec=TaiwanMarketRegimeDetector)
        return detector

    @pytest.fixture
    def mock_factor_pipeline(self):
        """Create mock factor pipeline for testing."""
        pipeline = Mock(spec=FactorPipeline)
        return pipeline

    @pytest.fixture
    def mock_regime_classification(self):
        """Create mock regime classification for testing."""
        classification = Mock(spec=RegimeClassification)
        classification.regime = TaiwanMarketRegime.TRENDING_BULL
        classification.confidence_score = Mock(spec=RegimeConfidenceScore)
        classification.confidence_score.confidence = 0.85
        classification.date = date(2024, 1, 15)
        return classification

    @pytest.fixture
    def mock_integrated_metrics(self):
        """Create mock integrated factor metrics for testing."""
        metrics = {}
        for symbol in ["2330.TW", "2454.TW", "2317.TW"]:
            mock_metric = Mock(spec=IntegratedFactorMetrics)
            mock_metric.normalized_value_score = np.random.normal(0, 1)
            mock_metric.normalized_flow_score = np.random.normal(0, 1)
            mock_metric.normalized_momentum_score = np.random.normal(0, 1)
            mock_metric.total_data_completeness = 0.85
            mock_metric.date = date(2024, 1, 15)
            metrics[symbol] = mock_metric
        return metrics

    @pytest.fixture
    def mock_factor_status(self):
        """Create mock factor group status for testing."""
        status = {}
        for factor_type in [FactorGroupType.VALUE, FactorGroupType.FLOW, FactorGroupType.MOMENTUM]:
            mock_status = Mock(spec=FactorGroupStatus)
            mock_status.production_ready = (factor_type != FactorGroupType.FLOW)  # Flow not ready
            status[factor_type] = mock_status
        return status

    def test_allocator_initialization(self, mock_regime_detector, mock_factor_pipeline):
        """Test DynamicFactorAllocator initialization."""
        constraints = AllocationConstraints()

        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            constraints=constraints
        )

        assert allocator.regime_detector == mock_regime_detector
        assert allocator.factor_pipeline == mock_factor_pipeline
        assert allocator.strategy == AllocationStrategy.HYBRID_OPTIMAL
        assert allocator.constraints == constraints
        assert len(allocator.allocation_history) == 0
        assert len(allocator.transition_history) == 0

    def test_allocator_initialization_with_invalid_constraints(self, mock_regime_detector, mock_factor_pipeline):
        """Test allocator initialization with invalid constraints."""
        invalid_constraints = AllocationConstraints(max_leverage=0.3)  # Invalid

        with pytest.raises(ValueError, match="Invalid allocation constraints provided"):
            DynamicFactorAllocator(
                regime_detector=mock_regime_detector,
                factor_pipeline=mock_factor_pipeline,
                constraints=invalid_constraints
            )

    def test_regime_base_weights_calculation(self, mock_regime_detector, mock_factor_pipeline):
        """Test regime-specific base weight calculation."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Test different regimes
        confidence_score = Mock(spec=RegimeConfidenceScore)
        confidence_score.confidence = 0.8

        # Trending bull should favor momentum
        bull_weights = allocator._calculate_regime_base_weights(
            TaiwanMarketRegime.TRENDING_BULL, confidence_score
        )
        assert bull_weights[2] > bull_weights[0]  # momentum > value
        assert bull_weights[2] > bull_weights[1]  # momentum > flow

        # Trending bear should favor value
        bear_weights = allocator._calculate_regime_base_weights(
            TaiwanMarketRegime.TRENDING_BEAR, confidence_score
        )
        assert bear_weights[0] > bear_weights[1]  # value > flow
        assert bear_weights[0] > bear_weights[2]  # value > momentum

        # Mean reverting should favor value and flow
        mean_rev_weights = allocator._calculate_regime_base_weights(
            TaiwanMarketRegime.MEAN_REVERTING, confidence_score
        )
        assert mean_rev_weights[0] > mean_rev_weights[2]  # value > momentum
        assert mean_rev_weights[1] > mean_rev_weights[2]  # flow > momentum

    def test_regime_base_weights_with_low_confidence(self, mock_regime_detector, mock_factor_pipeline):
        """Test regime base weights with low confidence."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Low confidence should blend toward equal weights
        low_confidence_score = Mock(spec=RegimeConfidenceScore)
        low_confidence_score.confidence = 0.4  # Below threshold

        weights = allocator._calculate_regime_base_weights(
            TaiwanMarketRegime.TRENDING_BULL, low_confidence_score
        )

        # Should be closer to equal weights than pure regime weights
        equal_weight = 1/3
        assert abs(weights[0] - equal_weight) < 0.2  # More generous threshold
        assert abs(weights[1] - equal_weight) < 0.2
        assert abs(weights[2] - equal_weight) < 0.2

    def test_performance_adjustments(self, mock_regime_detector, mock_factor_pipeline):
        """Test performance-based weight adjustments."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline,
            strategy=AllocationStrategy.PERFORMANCE_WEIGHTED
        )

        base_weights = (0.4, 0.3, 0.3)

        # Create factor metrics with different performance
        factor_metrics = {
            'value': FactorPerformanceMetrics(
                factor_name='value',
                measurement_date=date(2024, 1, 15),
                sharpe_ratio=1.2,
                hit_rate=0.65,
                volatility=0.15,
                data_completeness=0.9
            ),
            'flow': FactorPerformanceMetrics(
                factor_name='flow',
                measurement_date=date(2024, 1, 15),
                sharpe_ratio=0.8,
                hit_rate=0.55,
                volatility=0.20,
                data_completeness=0.8
            ),
            'momentum': FactorPerformanceMetrics(
                factor_name='momentum',
                measurement_date=date(2024, 1, 15),
                sharpe_ratio=0.6,
                hit_rate=0.50,
                volatility=0.25,
                data_completeness=0.85
            )
        }

        adjusted_weights = allocator._apply_performance_adjustments(
            base_weights, factor_metrics, date(2024, 1, 15)
        )

        # Value has best performance, should get higher weight
        assert adjusted_weights[0] > base_weights[0]  # Value weight increased

    def test_risk_adjustments(self, mock_regime_detector, mock_factor_pipeline):
        """Test risk-based weight adjustments."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline,
            strategy=AllocationStrategy.RISK_ADJUSTED
        )

        performance_weights = (0.4, 0.3, 0.3)

        # Create factor metrics with different risk profiles
        factor_metrics = {
            'value': FactorPerformanceMetrics(
                factor_name='value',
                measurement_date=date(2024, 1, 15),
                volatility=0.10,  # Low volatility
                correlation_to_peers=0.5  # Moderate correlation
            ),
            'flow': FactorPerformanceMetrics(
                factor_name='flow',
                measurement_date=date(2024, 1, 15),
                volatility=0.30,  # High volatility
                correlation_to_peers=0.8  # High correlation
            ),
            'momentum': FactorPerformanceMetrics(
                factor_name='momentum',
                measurement_date=date(2024, 1, 15),
                volatility=0.20,  # Medium volatility
                correlation_to_peers=0.6  # Medium correlation
            )
        }

        adjusted_weights = allocator._apply_risk_adjustments(
            performance_weights, factor_metrics, date(2024, 1, 15)
        )

        # Value has best risk profile, should get higher weight
        assert adjusted_weights[0] > performance_weights[0]  # Value weight increased
        # Flow has worst risk profile, should get lower weight
        assert adjusted_weights[1] < performance_weights[1]  # Flow weight decreased

    def test_taiwan_constraints_application(self, mock_regime_detector, mock_factor_pipeline,
                                          mock_regime_classification, mock_factor_status):
        """Test Taiwan market constraint application."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Mock factor pipeline status method
        mock_factor_pipeline.get_factor_group_status.return_value = mock_factor_status

        risk_weights = (0.4, 0.4, 0.2)  # Flow weight should be reduced

        constrained_weights = allocator._apply_taiwan_constraints(
            risk_weights, mock_regime_classification, date(2024, 1, 15)
        )

        # Flow weight should be reduced due to non-production readiness
        assert constrained_weights[1] < risk_weights[1]
        # Value weight should be increased (redistribution)
        assert constrained_weights[0] > risk_weights[0]

    def test_transition_mode_determination(self, mock_regime_detector, mock_factor_pipeline):
        """Test transition mode determination based on regime characteristics."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # High confidence should give immediate transition
        high_confidence_classification = Mock(spec=RegimeClassification)
        high_confidence_classification.confidence_score = Mock(spec=RegimeConfidenceScore)
        high_confidence_classification.confidence_score.confidence = 0.95

        mode = allocator._determine_transition_mode(high_confidence_classification)
        assert mode == TransitionMode.IMMEDIATE

        # Low confidence should give gradual transition
        low_confidence_classification = Mock(spec=RegimeClassification)
        low_confidence_classification.confidence_score = Mock(spec=RegimeConfidenceScore)
        low_confidence_classification.confidence_score.confidence = 0.5

        mode = allocator._determine_transition_mode(low_confidence_classification)
        assert mode == TransitionMode.GRADUAL

    def test_transition_smoothing_application(self, mock_regime_detector, mock_factor_pipeline,
                                            mock_regime_classification):
        """Test transition smoothing to prevent excessive turnover."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Add previous allocation to history
        constraints = AllocationConstraints(max_daily_weight_change=0.05)
        previous_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 14),
            regime=TaiwanMarketRegime.MEAN_REVERTING,
            regime_confidence=0.7,
            value_weight=0.5,
            flow_weight=0.3,
            momentum_weight=0.2,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )
        allocator.allocation_history.append(previous_allocation)

        # Target weights with large changes
        target_weights = (0.2, 0.2, 0.6)  # Large momentum shift

        # Mock transition mode determination
        with patch.object(allocator, '_determine_transition_mode', return_value=TransitionMode.GRADUAL):
            smoothed_weights = allocator._apply_transition_smoothing(
                target_weights, mock_regime_classification, date(2024, 1, 15)
            )

        # Changes should be limited by daily constraints (but may be slightly larger due to normalization)
        value_change = abs(smoothed_weights[0] - previous_allocation.value_weight)
        flow_change = abs(smoothed_weights[1] - previous_allocation.flow_weight)
        momentum_change = abs(smoothed_weights[2] - previous_allocation.momentum_weight)

        # Allow small tolerance for normalization effects
        tolerance = 0.02  # 2% tolerance for normalization
        assert value_change <= constraints.max_daily_weight_change + tolerance
        assert flow_change <= constraints.max_daily_weight_change + tolerance
        assert momentum_change <= constraints.max_daily_weight_change + tolerance

    def test_full_allocation_calculation(self, mock_regime_detector, mock_factor_pipeline,
                                       mock_regime_classification, mock_integrated_metrics,
                                       mock_factor_status):
        """Test complete allocation calculation workflow."""
        # Setup mocks
        mock_regime_detector.detect_current_regime.return_value = mock_regime_classification
        mock_factor_pipeline.calculate_integrated_factors.return_value = mock_integrated_metrics
        mock_factor_pipeline.get_factor_group_status.return_value = mock_factor_status

        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Mock Taiwan calendar
        with patch.object(allocator, '_is_market_closed', return_value=False):
            allocation = allocator.calculate_dynamic_allocation(
                allocation_date=date(2024, 1, 15),
                symbols=["2330.TW", "2454.TW"],
                taiex_data=None,
                market_data=None
            )

        # Verify allocation properties
        assert isinstance(allocation, DynamicAllocation)
        assert allocation.allocation_date == date(2024, 1, 15)
        assert allocation.regime == TaiwanMarketRegime.TRENDING_BULL
        assert allocation.regime_confidence == 0.85
        assert abs(allocation.total_weight - 1.0) < 1e-10
        assert allocation.strategy == AllocationStrategy.HYBRID_OPTIMAL
        assert allocation.expected_return is not None
        assert allocation.expected_volatility is not None
        assert allocation.expected_sharpe is not None

        # Verify methods were called
        mock_regime_detector.detect_current_regime.assert_called_once()
        mock_factor_pipeline.calculate_integrated_factors.assert_called_once()
        mock_factor_pipeline.get_factor_group_status.assert_called()

        # Verify allocation is in history
        assert len(allocator.allocation_history) == 1
        assert allocator.allocation_history[0] == allocation

    def test_allocation_validation(self, mock_regime_detector, mock_factor_pipeline,
                                 mock_factor_status):
        """Test allocation validation logic."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Mock factor pipeline status method
        mock_factor_pipeline.get_factor_group_status.return_value = mock_factor_status

        # Test valid allocation
        constraints = AllocationConstraints()
        valid_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.TRENDING_BULL,
            regime_confidence=0.85,
            value_weight=0.4,
            flow_weight=0.2,  # Reduced due to non-production readiness
            momentum_weight=0.4,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )

        allocator._validate_allocation(valid_allocation)
        assert valid_allocation.meets_constraints
        assert valid_allocation.production_ready

        # Test allocation exceeding single factor limit
        constraints_tight = AllocationConstraints(max_single_factor_weight=0.3)
        excessive_allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.TRENDING_BULL,
            regime_confidence=0.85,
            value_weight=0.1,
            flow_weight=0.1,
            momentum_weight=0.8,  # Exceeds 30% limit
            strategy=AllocationStrategy.REGIME_PURE,
            transition_mode=TransitionMode.IMMEDIATE,
            constraints_applied=constraints_tight
        )

        allocator._validate_allocation(excessive_allocation)
        assert not excessive_allocation.meets_constraints

    def test_performance_requirements(self, mock_regime_detector, mock_factor_pipeline,
                                    mock_regime_classification, mock_integrated_metrics,
                                    mock_factor_status):
        """Test that performance requirements are met."""
        # Setup mocks
        mock_regime_detector.detect_current_regime.return_value = mock_regime_classification
        mock_factor_pipeline.calculate_integrated_factors.return_value = mock_integrated_metrics
        mock_factor_pipeline.get_factor_group_status.return_value = mock_factor_status

        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Test weight calculation performance (<2 seconds)
        start_time = datetime.now()
        with patch.object(allocator, '_is_market_closed', return_value=False):
            allocation = allocator.calculate_dynamic_allocation(
                allocation_date=date(2024, 1, 15),
                symbols=["2330.TW"] * 50  # Moderate symbol list
            )
        end_time = datetime.now()

        elapsed_time = (end_time - start_time).total_seconds()
        assert elapsed_time < 2.0  # Performance requirement

        # Verify performance metrics are tracked
        assert 'last_allocation_time' in allocator.performance_metrics
        assert allocator.performance_metrics['last_allocation_time'] > 0

    def test_memory_usage_control(self, mock_regime_detector, mock_factor_pipeline):
        """Test memory usage control through history limits."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Simulate large allocation history using the update method
        constraints = AllocationConstraints()
        max_expected_size = 252 * 2  # 2 years as defined in implementation

        # Add more allocations than the limit to test trimming
        num_allocations = max_expected_size + 100

        for i in range(num_allocations):
            allocation = DynamicAllocation(
                allocation_date=date(2024, 1, 1) + timedelta(days=i),
                regime=TaiwanMarketRegime.MEAN_REVERTING,
                regime_confidence=0.7,
                value_weight=0.4,
                flow_weight=0.3,
                momentum_weight=0.3,
                strategy=AllocationStrategy.HYBRID_OPTIMAL,
                transition_mode=TransitionMode.SMOOTH,
                constraints_applied=constraints
            )
            # Use the actual update method which handles memory management
            allocator._update_allocation_history(allocation)

        # History should be limited to prevent memory issues
        assert len(allocator.allocation_history) == max_expected_size

    def test_allocation_export_functionality(self, mock_regime_detector, mock_factor_pipeline):
        """Test allocation history export functionality."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Add some allocation history
        constraints = AllocationConstraints()
        allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.TRENDING_BULL,
            regime_confidence=0.85,
            value_weight=0.3,
            flow_weight=0.3,
            momentum_weight=0.4,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )
        allocator.allocation_history.append(allocation)

        # Test export with default file path
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                file_path = allocator.export_allocation_history()

                # Verify file operations
                mock_file.assert_called_once()
                mock_json_dump.assert_called_once()

                # Verify export data structure
                export_data = mock_json_dump.call_args[0][0]
                assert 'export_date' in export_data
                assert 'allocator_config' in export_data
                assert 'allocation_history' in export_data
                assert 'performance_summary' in export_data
                assert len(export_data['allocation_history']) == 1

    def test_regime_transition_matrix_calculation(self, mock_regime_detector, mock_factor_pipeline):
        """Test regime transition matrix calculation from allocation history."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Add allocation history with regime transitions
        constraints = AllocationConstraints()
        regimes = [
            TaiwanMarketRegime.MEAN_REVERTING,
            TaiwanMarketRegime.TRENDING_BULL,
            TaiwanMarketRegime.TRENDING_BULL,
            TaiwanMarketRegime.HIGH_VOLATILITY,
            TaiwanMarketRegime.RECOVERY
        ]

        for i, regime in enumerate(regimes):
            allocation = DynamicAllocation(
                allocation_date=date(2024, 1, 1) + timedelta(days=i),
                regime=regime,
                regime_confidence=0.8,
                value_weight=0.4,
                flow_weight=0.3,
                momentum_weight=0.3,
                strategy=AllocationStrategy.HYBRID_OPTIMAL,
                transition_mode=TransitionMode.SMOOTH,
                constraints_applied=constraints
            )
            allocator.allocation_history.append(allocation)

        transition_matrix = allocator.get_regime_transition_matrix()

        # Verify matrix properties
        assert isinstance(transition_matrix, pd.DataFrame)
        assert transition_matrix.shape[0] == len(TaiwanMarketRegime)
        assert transition_matrix.shape[1] == len(TaiwanMarketRegime)

        # Verify row sums (should be close to 1.0 for rows with data)
        row_sums = transition_matrix.sum(axis=1)
        for regime_name, row_sum in row_sums.items():
            if row_sum > 0:  # Only check rows with actual transitions
                assert abs(row_sum - 1.0) < 1e-10

    def test_error_handling_and_edge_cases(self, mock_regime_detector, mock_factor_pipeline):
        """Test error handling and edge cases."""
        allocator = DynamicFactorAllocator(
            regime_detector=mock_regime_detector,
            factor_pipeline=mock_factor_pipeline
        )

        # Test error in regime detection
        mock_regime_detector.detect_current_regime.side_effect = Exception("Regime detection failed")

        with pytest.raises(RuntimeError, match="Dynamic allocation calculation failed"):
            allocator.calculate_dynamic_allocation(date(2024, 1, 15))

        # Reset mock for next test
        mock_regime_detector.detect_current_regime.side_effect = None

        # Test error in factor metrics calculation
        mock_factor_pipeline.calculate_integrated_factors.side_effect = Exception("Factor calculation failed")

        # Should handle error gracefully and use default metrics
        mock_regime_classification = Mock(spec=RegimeClassification)
        mock_regime_classification.regime = TaiwanMarketRegime.MEAN_REVERTING
        mock_regime_classification.confidence_score = Mock(spec=RegimeConfidenceScore)
        mock_regime_classification.confidence_score.confidence = 0.7
        mock_regime_classification.date = date(2024, 1, 15)
        mock_regime_detector.detect_current_regime.return_value = mock_regime_classification

        mock_factor_status = {
            FactorGroupType.VALUE: Mock(production_ready=True),
            FactorGroupType.FLOW: Mock(production_ready=False),
            FactorGroupType.MOMENTUM: Mock(production_ready=True)
        }
        mock_factor_pipeline.get_factor_group_status.return_value = mock_factor_status

        with patch.object(allocator, '_is_market_closed', return_value=False):
            # Should not raise exception, should use default factor metrics
            allocation = allocator.calculate_dynamic_allocation(date(2024, 1, 15))
            assert isinstance(allocation, DynamicAllocation)


class TestFactoryFunction:
    """Test the factory function for creating allocators."""

    def test_create_dynamic_factor_allocator_with_defaults(self):
        """Test factory function with default parameters."""
        with patch('src.strategies.dynamic_allocation.create_taiwan_regime_detector') as mock_create_detector:
            with patch('src.strategies.dynamic_allocation.create_factor_pipeline') as mock_create_pipeline:
                mock_detector = Mock(spec=TaiwanMarketRegimeDetector)
                mock_pipeline = Mock(spec=FactorPipeline)
                mock_create_detector.return_value = mock_detector
                mock_create_pipeline.return_value = mock_pipeline

                allocator = create_dynamic_factor_allocator()

                assert isinstance(allocator, DynamicFactorAllocator)
                assert allocator.regime_detector == mock_detector
                assert allocator.factor_pipeline == mock_pipeline
                assert allocator.strategy == AllocationStrategy.HYBRID_OPTIMAL

                mock_create_detector.assert_called_once()
                mock_create_pipeline.assert_called_once()

    def test_create_dynamic_factor_allocator_with_custom_parameters(self):
        """Test factory function with custom parameters."""
        custom_detector = Mock(spec=TaiwanMarketRegimeDetector)
        custom_pipeline = Mock(spec=FactorPipeline)
        custom_constraints = AllocationConstraints(max_leverage=1.5)

        allocator = create_dynamic_factor_allocator(
            regime_detector=custom_detector,
            factor_pipeline=custom_pipeline,
            strategy=AllocationStrategy.PERFORMANCE_WEIGHTED,
            constraints=custom_constraints
        )

        assert allocator.regime_detector == custom_detector
        assert allocator.factor_pipeline == custom_pipeline
        assert allocator.strategy == AllocationStrategy.PERFORMANCE_WEIGHTED
        assert allocator.constraints == custom_constraints


class TestIntegrationWithTasks004And005:
    """Test integration with factor combination (Task 004) and regime detection (Task 005)."""

    def test_factor_weight_integration(self):
        """Test integration with FactorWeight from Task 004."""
        constraints = AllocationConstraints()
        allocation = DynamicAllocation(
            allocation_date=date(2024, 1, 15),
            regime=TaiwanMarketRegime.TRENDING_BULL,
            regime_confidence=0.85,
            value_weight=0.3,
            flow_weight=0.3,
            momentum_weight=0.4,
            strategy=AllocationStrategy.HYBRID_OPTIMAL,
            transition_mode=TransitionMode.SMOOTH,
            constraints_applied=constraints
        )

        factor_weight = allocation.to_factor_weight()

        # Test that FactorWeight can be used with factor combination strategies
        assert hasattr(factor_weight, 'value_weight')
        assert hasattr(factor_weight, 'flow_weight')
        assert hasattr(factor_weight, 'momentum_weight')
        assert hasattr(factor_weight, 'regime')
        assert hasattr(factor_weight, 'confidence_score')

    def test_regime_classification_integration(self):
        """Test integration with RegimeClassification from Task 005."""
        # This would test actual integration if the regime detection system was available
        # For now, we verify the interface compatibility

        mock_classification = Mock(spec=RegimeClassification)
        mock_classification.regime = TaiwanMarketRegime.TRENDING_BULL
        mock_classification.confidence_score = Mock(spec=RegimeConfidenceScore)
        mock_classification.confidence_score.confidence = 0.85
        mock_classification.date = date(2024, 1, 15)

        # Verify expected attributes are present
        assert hasattr(mock_classification, 'regime')
        assert hasattr(mock_classification, 'confidence_score')
        assert hasattr(mock_classification, 'date')
        assert hasattr(mock_classification.confidence_score, 'confidence')

        # Test regime enum compatibility
        assert isinstance(mock_classification.regime, TaiwanMarketRegime)
        regime_values = [regime.value for regime in TaiwanMarketRegime]
        expected_regimes = ['trending_bull', 'trending_bear', 'mean_reverting', 'high_volatility', 'recovery']
        assert all(regime in expected_regimes for regime in regime_values)


def mock_open(file_content=""):
    """Mock file open for testing."""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open(read_data=file_content)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])