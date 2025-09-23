"""
Tests for execution timing optimization components.

This module tests the execution timing algorithms including TWAP, VWAP,
adaptive strategies, and the execution optimizer.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, patch

from src.trading.costs.timing import (
    ExecutionStrategy,
    TradingSession,
    UrgencyLevel,
    ExecutionParameters,
    ExecutionSlice,
    ExecutionSchedule,
    TWAPStrategy,
    VWAPStrategy,
    AdaptiveStrategy,
    ExecutionOptimizer,
    create_execution_optimizer
)
from src.trading.costs.market_impact import TaiwanMarketImpactModel
from src.trading.costs.liquidity import LiquidityMetrics, LiquidityTier


class TestExecutionParameters:
    """Test execution parameter configurations."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = ExecutionParameters()
        
        assert params.strategy == ExecutionStrategy.TWAP
        assert params.urgency == UrgencyLevel.MEDIUM
        assert params.max_execution_time_hours == 6.0
        assert params.max_participation_rate == 0.20
        assert params.target_participation_rate == 0.10
        assert params.min_slice_size == 100
        assert params.max_market_impact_bps == 50.0
        assert params.timing_risk_aversion == 0.5
        assert params.avoid_opening_minutes == 15
        assert params.avoid_closing_minutes == 15
        assert params.lunch_break_trading is False
        assert params.market_condition_adjustment is True
        assert params.volume_forecast_window == 20
        
        # Check that start_time and end_time are set
        assert params.start_time is not None
        assert params.end_time is not None
        assert params.end_time > params.start_time
    
    def test_parameter_customization(self):
        """Test custom parameter configuration."""
        start_time = datetime(2024, 6, 15, 10, 0)
        end_time = datetime(2024, 6, 15, 14, 0)
        
        params = ExecutionParameters(
            strategy=ExecutionStrategy.VWAP,
            urgency=UrgencyLevel.HIGH,
            start_time=start_time,
            end_time=end_time,
            max_participation_rate=0.15,
            target_participation_rate=0.08,
            avoid_opening_minutes=10,
            lunch_break_trading=True
        )
        
        assert params.strategy == ExecutionStrategy.VWAP
        assert params.urgency == UrgencyLevel.HIGH
        assert params.start_time == start_time
        assert params.end_time == end_time
        assert params.max_participation_rate == 0.15
        assert params.target_participation_rate == 0.08
        assert params.avoid_opening_minutes == 10
        assert params.lunch_break_trading is True


class TestExecutionSlice:
    """Test execution slice data structure."""
    
    def test_slice_creation(self):
        """Test execution slice creation."""
        slice = ExecutionSlice(
            slice_id=1,
            scheduled_time=datetime(2024, 6, 15, 10, 30),
            target_shares=1000,
            target_participation_rate=0.05,
            estimated_price=100.0,
            estimated_volume=20000,
            estimated_impact_bps=15.0,
            session=TradingSession.MORNING
        )
        
        assert slice.slice_id == 1
        assert slice.target_shares == 1000
        assert slice.target_participation_rate == 0.05
        assert slice.estimated_price == 100.0
        assert slice.estimated_volume == 20000
        assert slice.estimated_impact_bps == 15.0
        assert slice.session == TradingSession.MORNING
        assert slice.executed_shares is None  # Not executed yet


class TestExecutionSchedule:
    """Test execution schedule functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_slices = [
            ExecutionSlice(
                slice_id=0,
                scheduled_time=datetime(2024, 6, 15, 9, 30),
                target_shares=500,
                target_participation_rate=0.05,
                estimated_price=100.0,
                estimated_volume=10000,
                estimated_impact_bps=10.0,
                session=TradingSession.MORNING
            ),
            ExecutionSlice(
                slice_id=1,
                scheduled_time=datetime(2024, 6, 15, 10, 0),
                target_shares=500,
                target_participation_rate=0.05,
                estimated_price=100.0,
                estimated_volume=10000,
                estimated_impact_bps=10.0,
                session=TradingSession.MORNING
            )
        ]
        
        self.schedule = ExecutionSchedule(
            order_id="TEST_001",
            symbol="TEST.TW",
            total_shares=1000,
            strategy=ExecutionStrategy.TWAP,
            slices=self.sample_slices,
            total_execution_time=timedelta(hours=2),
            estimated_total_impact_bps=20.0,
            estimated_average_price=100.0,
            timing_risk_score=0.3,
            market_impact_risk_score=0.2,
            implementation_shortfall_bps=20.0,
            schedule_created_at=datetime.now(),
            last_updated_at=datetime.now()
        )
    
    def test_schedule_creation(self):
        """Test execution schedule creation."""
        assert self.schedule.order_id == "TEST_001"
        assert self.schedule.symbol == "TEST.TW"
        assert self.schedule.total_shares == 1000
        assert self.schedule.strategy == ExecutionStrategy.TWAP
        assert len(self.schedule.slices) == 2
        assert self.schedule.estimated_total_impact_bps == 20.0
    
    def test_get_active_slices(self):
        """Test getting active slices for execution."""
        # No slices should be active before start time
        current_time = datetime(2024, 6, 15, 9, 0)
        active_slices = self.schedule.get_active_slices(current_time)
        assert len(active_slices) == 0
        
        # First slice should be active
        current_time = datetime(2024, 6, 15, 9, 30)
        active_slices = self.schedule.get_active_slices(current_time)
        assert len(active_slices) == 1
        assert active_slices[0].slice_id == 0
        
        # Both slices should be active
        current_time = datetime(2024, 6, 15, 10, 30)
        active_slices = self.schedule.get_active_slices(current_time)
        assert len(active_slices) == 2
        
        # Mark first slice as executed
        self.schedule.slices[0].executed_shares = 500
        active_slices = self.schedule.get_active_slices(current_time)
        assert len(active_slices) == 1
        assert active_slices[0].slice_id == 1
    
    def test_completion_percentage(self):
        """Test execution completion percentage calculation."""
        # No execution yet
        assert self.schedule.get_completion_percentage() == 0.0
        
        # Partial execution
        self.schedule.slices[0].executed_shares = 500
        assert self.schedule.get_completion_percentage() == 50.0
        
        # Complete execution
        self.schedule.slices[1].executed_shares = 500
        assert self.schedule.get_completion_percentage() == 100.0


class TestTWAPStrategy:
    """Test TWAP execution strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = TWAPStrategy()
        self.market_data = {
            'current_price': 100.0,
            'volatility': 0.25,
            'avg_daily_volume': 500_000
        }
        self.liquidity_metrics = LiquidityMetrics(
            symbol="TEST.TW",
            date=date.today(),
            adv_20d=500_000,
            adv_60d=500_000,
            lav_20d=500_000,
            current_volume=400_000,
            volume_ratio=0.8,
            liquidity_tier=LiquidityTier.LIQUID,
            liquidity_score=0.75
        )
    
    def test_basic_twap_schedule(self):
        """Test basic TWAP schedule generation."""
        parameters = ExecutionParameters(
            strategy=ExecutionStrategy.TWAP,
            start_time=datetime(2024, 6, 15, 9, 30),
            end_time=datetime(2024, 6, 15, 11, 30),
            max_participation_rate=0.10
        )
        
        schedule = self.strategy.generate_schedule(
            symbol="TEST.TW",
            total_shares=50_000,
            parameters=parameters,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        assert isinstance(schedule, ExecutionSchedule)
        assert schedule.symbol == "TEST.TW"
        assert schedule.total_shares == 50_000
        assert schedule.strategy == ExecutionStrategy.TWAP
        assert len(schedule.slices) > 0
        
        # Check that slices sum to total shares (approximately)
        total_slice_shares = sum(slice.target_shares for slice in schedule.slices)
        assert abs(total_slice_shares - 50_000) < 1000  # Allow some rounding
        
        # Check participation rates
        for slice in schedule.slices:
            assert slice.target_participation_rate <= parameters.max_participation_rate * 1.1  # Small buffer
            assert slice.estimated_impact_bps > 0
            assert slice.scheduled_time >= parameters.start_time
            assert slice.scheduled_time <= parameters.end_time
    
    def test_twap_timing_distribution(self):
        """Test TWAP timing distribution."""
        parameters = ExecutionParameters(
            strategy=ExecutionStrategy.TWAP,
            start_time=datetime(2024, 6, 15, 9, 30),
            end_time=datetime(2024, 6, 15, 12, 0),  # 2.5 hours
            avoid_opening_minutes=15,
            avoid_closing_minutes=15
        )
        
        schedule = self.strategy.generate_schedule(
            symbol="TEST.TW",
            total_shares=30_000,
            parameters=parameters,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        # Check time distribution
        slice_times = [slice.scheduled_time for slice in schedule.slices]
        
        # Should avoid opening/closing periods
        earliest_time = min(slice_times)
        latest_time = max(slice_times)
        
        assert earliest_time >= parameters.start_time + timedelta(minutes=parameters.avoid_opening_minutes)
        assert latest_time <= parameters.end_time - timedelta(minutes=parameters.avoid_closing_minutes)
        
        # Should be roughly evenly distributed
        if len(schedule.slices) > 1:
            time_intervals = []
            for i in range(1, len(slice_times)):
                interval = (slice_times[i] - slice_times[i-1]).total_seconds() / 60  # minutes
                time_intervals.append(interval)
            
            # Intervals should be roughly similar (within 50% of average)
            avg_interval = np.mean(time_intervals)
            for interval in time_intervals:
                assert 0.5 * avg_interval <= interval <= 1.5 * avg_interval
    
    def test_twap_session_handling(self):
        """Test TWAP session handling."""
        parameters = ExecutionParameters(
            strategy=ExecutionStrategy.TWAP,
            start_time=datetime(2024, 6, 15, 9, 0),   # Market start
            end_time=datetime(2024, 6, 15, 13, 30),   # Market end
            lunch_break_trading=False
        )
        
        schedule = self.strategy.generate_schedule(
            symbol="TEST.TW",
            total_shares=20_000,
            parameters=parameters,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        # All slices should be in morning session (Taiwan market)
        for slice in schedule.slices:
            assert slice.session == TradingSession.MORNING
            # Should be within market hours (9:00-12:00 for Taiwan)
            slice_time = slice.scheduled_time.time()
            assert time(9, 0) <= slice_time <= time(12, 0)


class TestVWAPStrategy:
    """Test VWAP execution strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = VWAPStrategy()
        self.market_data = {
            'current_price': 100.0,
            'volatility': 0.25,
            'avg_daily_volume': 500_000,
            'intraday_volume_pattern': {
                '09:30': 1.5,  # Higher volume at open
                '10:00': 1.2,
                '10:30': 1.0,
                '11:00': 0.8,
                '11:30': 0.9
            }
        }
        self.liquidity_metrics = LiquidityMetrics(
            symbol="TEST.TW",
            date=date.today(),
            adv_20d=500_000,
            adv_60d=500_000,
            lav_20d=500_000,
            current_volume=400_000,
            volume_ratio=0.8,
            liquidity_tier=LiquidityTier.LIQUID,
            liquidity_score=0.75
        )
    
    def test_basic_vwap_schedule(self):
        """Test basic VWAP schedule generation."""
        parameters = ExecutionParameters(
            strategy=ExecutionStrategy.VWAP,
            start_time=datetime(2024, 6, 15, 9, 30),
            end_time=datetime(2024, 6, 15, 11, 30),
            max_participation_rate=0.10
        )
        
        schedule = self.strategy.generate_schedule(
            symbol="TEST.TW",
            total_shares=50_000,
            parameters=parameters,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        assert isinstance(schedule, ExecutionSchedule)
        assert schedule.strategy == ExecutionStrategy.VWAP
        assert len(schedule.slices) > 0
        
        # Check volume-weighted distribution
        total_slice_shares = sum(slice.target_shares for slice in schedule.slices)
        assert abs(total_slice_shares - 50_000) < 2000  # Allow more variance for VWAP
        
        # Slices should follow volume pattern
        if len(schedule.slices) > 1:
            # Find slices during high volume periods
            high_volume_slices = [
                s for s in schedule.slices 
                if s.scheduled_time.time() >= time(9, 30) and s.scheduled_time.time() <= time(10, 0)
            ]
            low_volume_slices = [
                s for s in schedule.slices 
                if s.scheduled_time.time() >= time(11, 0) and s.scheduled_time.time() <= time(11, 30)
            ]
            
            if high_volume_slices and low_volume_slices:
                avg_high_volume_shares = np.mean([s.target_shares for s in high_volume_slices])
                avg_low_volume_shares = np.mean([s.target_shares for s in low_volume_slices])
                
                # High volume periods should have more shares
                assert avg_high_volume_shares >= avg_low_volume_shares
    
    def test_vwap_volume_forecast(self):
        """Test VWAP volume forecasting."""
        parameters = ExecutionParameters(
            strategy=ExecutionStrategy.VWAP,
            start_time=datetime(2024, 6, 15, 9, 30),
            end_time=datetime(2024, 6, 15, 11, 0)
        )
        
        # Test with explicit volume pattern
        market_data_with_pattern = self.market_data.copy()
        market_data_with_pattern['intraday_volume_pattern'] = {
            '09:30': 2.0,  # Very high volume at open
            '10:00': 1.0,  # Normal volume
            '10:30': 0.5   # Low volume
        }
        
        schedule = self.strategy.generate_schedule(
            symbol="TEST.TW",
            total_shares=30_000,
            parameters=parameters,
            market_data=market_data_with_pattern,
            liquidity_metrics=self.liquidity_metrics
        )
        
        # Find slices in different volume periods
        opening_slices = [
            s for s in schedule.slices 
            if s.scheduled_time.time() >= time(9, 30) and s.scheduled_time.time() < time(9, 45)
        ]
        mid_slices = [
            s for s in schedule.slices 
            if s.scheduled_time.time() >= time(10, 0) and s.scheduled_time.time() < time(10, 15)
        ]
        
        if opening_slices and mid_slices:
            # Opening period should have higher estimated volume
            avg_opening_volume = np.mean([s.estimated_volume for s in opening_slices])
            avg_mid_volume = np.mean([s.estimated_volume for s in mid_slices])
            assert avg_opening_volume > avg_mid_volume


class TestAdaptiveStrategy:
    """Test adaptive execution strategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.impact_model = TaiwanMarketImpactModel()
        self.strategy = AdaptiveStrategy(self.impact_model)
        self.market_data = {
            'current_price': 100.0,
            'volatility': 0.25,
            'avg_daily_volume': 500_000
        }
        self.liquidity_metrics = LiquidityMetrics(
            symbol="TEST.TW",
            date=date.today(),
            adv_20d=500_000,
            adv_60d=500_000,
            lav_20d=500_000,
            current_volume=400_000,
            volume_ratio=0.8,
            liquidity_tier=LiquidityTier.LIQUID,
            liquidity_score=0.75
        )
    
    def test_adaptive_schedule_generation(self):
        """Test adaptive schedule generation."""
        parameters = ExecutionParameters(
            strategy=ExecutionStrategy.ADAPTIVE,
            start_time=datetime(2024, 6, 15, 9, 30),
            end_time=datetime(2024, 6, 15, 11, 30),
            max_participation_rate=0.10
        )
        
        schedule = self.strategy.generate_schedule(
            symbol="TEST.TW",
            total_shares=40_000,
            parameters=parameters,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        assert isinstance(schedule, ExecutionSchedule)
        assert schedule.strategy == ExecutionStrategy.ADAPTIVE
        assert len(schedule.slices) > 0
        
        # Adaptive strategy should optimize impact
        for slice in schedule.slices:
            assert slice.estimated_impact_bps > 0
            # Impact should be calculated using the detailed model
            assert hasattr(slice, 'estimated_impact_bps')
    
    def test_adaptive_vs_twap_comparison(self):
        """Test adaptive strategy vs TWAP comparison."""
        parameters = ExecutionParameters(
            start_time=datetime(2024, 6, 15, 9, 30),
            end_time=datetime(2024, 6, 15, 11, 30),
            max_participation_rate=0.10
        )
        
        # Generate TWAP schedule
        twap_strategy = TWAPStrategy()
        twap_parameters = ExecutionParameters(strategy=ExecutionStrategy.TWAP, **parameters.__dict__)
        twap_schedule = twap_strategy.generate_schedule(
            symbol="TEST.TW",
            total_shares=40_000,
            parameters=twap_parameters,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        # Generate adaptive schedule
        adaptive_parameters = ExecutionParameters(strategy=ExecutionStrategy.ADAPTIVE, **parameters.__dict__)
        adaptive_schedule = self.strategy.generate_schedule(
            symbol="TEST.TW",
            total_shares=40_000,
            parameters=adaptive_parameters,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        # Both should be valid schedules
        assert len(twap_schedule.slices) > 0
        assert len(adaptive_schedule.slices) > 0
        
        # Adaptive should potentially have different risk characteristics
        # (May not always be better due to model complexity, but should be different)
        assert adaptive_schedule.timing_risk_score != twap_schedule.timing_risk_score or \
               adaptive_schedule.market_impact_risk_score != twap_schedule.market_impact_risk_score


class TestExecutionOptimizer:
    """Test execution optimizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = create_execution_optimizer()
        self.market_data = {
            'current_price': 100.0,
            'volatility': 0.25,
            'avg_daily_volume': 500_000
        }
        self.liquidity_metrics = LiquidityMetrics(
            symbol="TEST.TW",
            date=date.today(),
            adv_20d=500_000,
            adv_60d=500_000,
            lav_20d=500_000,
            current_volume=400_000,
            volume_ratio=0.8,
            liquidity_tier=LiquidityTier.LIQUID,
            liquidity_score=0.75
        )
    
    def test_strategy_recommendation(self):
        """Test strategy recommendation logic."""
        # Small order, liquid stock -> Should recommend ADAPTIVE or TWAP
        strategy, params = self.optimizer.recommend_strategy(
            symbol="TEST.TW",
            total_shares=5_000,  # Small order
            urgency=UrgencyLevel.MEDIUM,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        assert strategy in [ExecutionStrategy.TWAP, ExecutionStrategy.ADAPTIVE]
        assert isinstance(params, ExecutionParameters)
        assert params.strategy == strategy
        
        # Large order -> Should recommend VWAP
        large_liquidity = self.liquidity_metrics
        large_liquidity.adv_20d = 100_000  # Smaller ADV to make order relatively large
        
        strategy_large, params_large = self.optimizer.recommend_strategy(
            symbol="TEST.TW",
            total_shares=50_000,  # Large relative to ADV
            urgency=UrgencyLevel.MEDIUM,
            market_data=self.market_data,
            liquidity_metrics=large_liquidity
        )
        
        assert strategy_large == ExecutionStrategy.VWAP
        assert params_large.max_participation_rate <= 0.15  # Conservative for large orders
        
        # Urgent order -> Should recommend TWAP with aggressive parameters
        strategy_urgent, params_urgent = self.optimizer.recommend_strategy(
            symbol="TEST.TW",
            total_shares=10_000,
            urgency=UrgencyLevel.URGENT,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        assert strategy_urgent == ExecutionStrategy.TWAP
        assert params_urgent.max_execution_time_hours <= 2.0  # Short execution time
        assert params_urgent.max_market_impact_bps >= 50.0  # Accept higher impact
    
    def test_optimal_schedule_generation(self):
        """Test optimal schedule generation."""
        schedule = self.optimizer.generate_optimal_schedule(
            symbol="TEST.TW",
            total_shares=30_000,
            urgency=UrgencyLevel.MEDIUM,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        assert isinstance(schedule, ExecutionSchedule)
        assert schedule.symbol == "TEST.TW"
        assert schedule.total_shares == 30_000
        assert len(schedule.slices) > 0
        
        # Check schedule quality
        assert schedule.estimated_total_impact_bps > 0
        assert schedule.timing_risk_score >= 0
        assert schedule.market_impact_risk_score >= 0
        
        # Check slice properties
        for slice in schedule.slices:
            assert slice.target_shares > 0
            assert slice.estimated_price > 0
            assert slice.estimated_volume > 0
            assert slice.target_participation_rate > 0
    
    def test_strategy_comparison(self):
        """Test strategy comparison functionality."""
        comparison = self.optimizer.compare_strategies(
            symbol="TEST.TW",
            total_shares=25_000,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics
        )
        
        assert isinstance(comparison, dict)
        assert len(comparison) > 0
        
        # Should have multiple strategies
        expected_strategies = [ExecutionStrategy.TWAP, ExecutionStrategy.VWAP, ExecutionStrategy.ADAPTIVE]
        for strategy in expected_strategies:
            if strategy in comparison:
                schedule = comparison[strategy]
                assert isinstance(schedule, ExecutionSchedule)
                assert schedule.strategy == strategy
                assert len(schedule.slices) > 0
        
        # Compare characteristics
        if len(comparison) > 1:
            strategies = list(comparison.keys())
            schedule1 = comparison[strategies[0]]
            schedule2 = comparison[strategies[1]]
            
            # Different strategies should have different characteristics
            assert (schedule1.estimated_total_impact_bps != schedule2.estimated_total_impact_bps or
                   schedule1.timing_risk_score != schedule2.timing_risk_score or
                   len(schedule1.slices) != len(schedule2.slices))
    
    def test_custom_parameters(self):
        """Test custom parameter usage."""
        custom_params = ExecutionParameters(
            strategy=ExecutionStrategy.TWAP,
            max_participation_rate=0.05,  # Very conservative
            target_participation_rate=0.03,
            max_execution_time_hours=8.0,  # Long execution
            avoid_opening_minutes=30,  # Avoid opening longer
            avoid_closing_minutes=30
        )
        
        schedule = self.optimizer.generate_optimal_schedule(
            symbol="TEST.TW",
            total_shares=20_000,
            urgency=UrgencyLevel.LOW,
            market_data=self.market_data,
            liquidity_metrics=self.liquidity_metrics,
            custom_parameters=custom_params
        )
        
        assert schedule.strategy == ExecutionStrategy.TWAP
        
        # Should respect custom parameters
        for slice in schedule.slices:
            assert slice.target_participation_rate <= 0.06  # Small buffer above 0.05
        
        # Should avoid opening/closing periods as specified
        first_slice_time = min(s.scheduled_time for s in schedule.slices).time()
        last_slice_time = max(s.scheduled_time for s in schedule.slices).time()
        
        assert first_slice_time >= time(10, 0)  # After 9:30 + 30 min buffer
        assert last_slice_time <= time(11, 30)  # Before 12:00 - 30 min buffer


class TestFactoryFunctions:
    """Test factory functions for optimizer creation."""
    
    def test_create_execution_optimizer_default(self):
        """Test default optimizer creation."""
        optimizer = create_execution_optimizer()
        
        assert isinstance(optimizer, ExecutionOptimizer)
        assert isinstance(optimizer.impact_model, TaiwanMarketImpactModel)
        assert ExecutionStrategy.TWAP in optimizer.strategies
        assert ExecutionStrategy.VWAP in optimizer.strategies
        assert ExecutionStrategy.ADAPTIVE in optimizer.strategies
    
    def test_create_execution_optimizer_custom(self):
        """Test custom optimizer creation."""
        custom_impact_model = TaiwanMarketImpactModel()
        optimizer = create_execution_optimizer(custom_impact_model)
        
        assert isinstance(optimizer, ExecutionOptimizer)
        assert optimizer.impact_model is custom_impact_model


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = create_execution_optimizer()
        self.twap_strategy = TWAPStrategy()
    
    def test_very_small_order(self):
        """Test handling of very small orders."""
        market_data = {'current_price': 100.0, 'volatility': 0.25, 'avg_daily_volume': 500_000}
        liquidity_metrics = LiquidityMetrics(
            symbol="TEST.TW", date=date.today(), adv_20d=500_000, adv_60d=500_000, lav_20d=500_000,
            current_volume=400_000, volume_ratio=0.8, liquidity_tier=LiquidityTier.LIQUID, liquidity_score=0.75
        )
        
        schedule = self.optimizer.generate_optimal_schedule(
            symbol="TEST.TW",
            total_shares=100,  # Very small order
            urgency=UrgencyLevel.MEDIUM,
            market_data=market_data,
            liquidity_metrics=liquidity_metrics
        )
        
        assert isinstance(schedule, ExecutionSchedule)
        assert schedule.total_shares == 100
        assert len(schedule.slices) >= 1  # Should have at least one slice
    
    def test_very_large_order(self):
        """Test handling of very large orders."""
        market_data = {'current_price': 100.0, 'volatility': 0.25, 'avg_daily_volume': 100_000}
        liquidity_metrics = LiquidityMetrics(
            symbol="TEST.TW", date=date.today(), adv_20d=100_000, adv_60d=100_000, lav_20d=100_000,
            current_volume=80_000, volume_ratio=0.8, liquidity_tier=LiquidityTier.MODERATE, liquidity_score=0.5
        )
        
        schedule = self.optimizer.generate_optimal_schedule(
            symbol="TEST.TW",
            total_shares=500_000,  # 5x daily volume
            urgency=UrgencyLevel.LOW,
            market_data=market_data,
            liquidity_metrics=liquidity_metrics
        )
        
        assert isinstance(schedule, ExecutionSchedule)
        assert schedule.total_shares == 500_000
        
        # Should spread over long time and use conservative participation rates
        assert schedule.total_execution_time >= timedelta(hours=4)
        for slice in schedule.slices:
            assert slice.target_participation_rate <= 0.15  # Conservative
    
    def test_illiquid_stock(self):
        """Test handling of illiquid stocks."""
        market_data = {'current_price': 50.0, 'volatility': 0.40, 'avg_daily_volume': 5_000}
        illiquid_metrics = LiquidityMetrics(
            symbol="ILLIQUID.TW", date=date.today(), adv_20d=5_000, adv_60d=5_000, lav_20d=5_000,
            current_volume=3_000, volume_ratio=0.6, liquidity_tier=LiquidityTier.ILLIQUID, liquidity_score=0.2
        )
        
        strategy, params = self.optimizer.recommend_strategy(
            symbol="ILLIQUID.TW",
            total_shares=10_000,  # Large relative to volume
            urgency=UrgencyLevel.MEDIUM,
            market_data=market_data,
            liquidity_metrics=illiquid_metrics
        )
        
        # Should recommend VWAP for large order in illiquid stock
        assert strategy == ExecutionStrategy.VWAP
        assert params.max_participation_rate <= 0.10  # Conservative participation
        assert params.max_execution_time_hours >= 4.0  # Longer execution time
    
    def test_zero_volume_periods(self):
        """Test handling of zero volume periods."""
        parameters = ExecutionParameters(
            strategy=ExecutionStrategy.TWAP,
            start_time=datetime(2024, 6, 15, 9, 30),
            end_time=datetime(2024, 6, 15, 11, 30)
        )
        
        # Market data with zero volume
        zero_volume_data = {
            'current_price': 100.0,
            'volatility': 0.25,
            'avg_daily_volume': 0  # Zero volume
        }
        
        zero_volume_metrics = LiquidityMetrics(
            symbol="ZERO.TW", date=date.today(), adv_20d=0, adv_60d=0, lav_20d=0,
            current_volume=0, volume_ratio=0, liquidity_tier=LiquidityTier.VERY_ILLIQUID, liquidity_score=0.0
        )
        
        # Should handle gracefully without crashing
        schedule = self.twap_strategy.generate_schedule(
            symbol="ZERO.TW",
            total_shares=1_000,
            parameters=parameters,
            market_data=zero_volume_data,
            liquidity_metrics=zero_volume_metrics
        )
        
        assert isinstance(schedule, ExecutionSchedule)
        # May have high impact estimates due to zero volume
        assert schedule.estimated_total_impact_bps >= 0


if __name__ == "__main__":
    pytest.main([__file__])