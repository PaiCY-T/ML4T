"""
Execution timing optimization for Taiwan equity markets.

This module provides sophisticated execution timing algorithms including TWAP
(Time-Weighted Average Price), VWAP (Volume-Weighted Average Price), and
adaptive timing strategies optimized for Taiwan market microstructure.

Key Features:
- TWAP execution with Taiwan market sessions
- VWAP execution with historical volume patterns
- Adaptive timing based on market impact forecasts
- Implementation Shortfall optimization
- Participation rate optimization
- Real-time market condition adjustments
"""

from datetime import datetime, date, time, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from .market_impact import TaiwanMarketImpactModel, ImpactCalculationResult
from .liquidity import LiquidityAnalyzer, LiquidityMetrics

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Types of execution strategies."""
    TWAP = "twap"                    # Time-Weighted Average Price
    VWAP = "vwap"                    # Volume-Weighted Average Price
    IMPLEMENTATION_SHORTFALL = "is"   # Implementation Shortfall
    PARTICIPATION_RATE = "pov"        # Percent of Volume
    ADAPTIVE = "adaptive"             # Adaptive strategy
    MARKET_ON_CLOSE = "moc"          # Market on Close
    ICEBERG = "iceberg"              # Iceberg orders


class TradingSession(Enum):
    """Taiwan trading sessions."""
    MORNING = "morning"       # 09:00-12:00
    AFTERNOON = "afternoon"   # 13:30-13:30


class UrgencyLevel(Enum):
    """Trade urgency levels."""
    LOW = "low"              # Patient execution over days
    MEDIUM = "medium"        # Normal execution within day
    HIGH = "high"            # Aggressive execution within hours
    URGENT = "urgent"        # Immediate execution


@dataclass
class ExecutionParameters:
    """Parameters for execution timing optimization."""
    
    # Strategy parameters
    strategy: ExecutionStrategy = ExecutionStrategy.TWAP
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    
    # Timing constraints
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_execution_time_hours: float = 6.0
    
    # Volume constraints
    max_participation_rate: float = 0.20      # Max 20% of volume
    target_participation_rate: float = 0.10   # Target 10% of volume
    min_slice_size: int = 100                 # Minimum order slice
    max_slice_size: Optional[int] = None      # Maximum order slice
    
    # Risk constraints
    max_market_impact_bps: float = 50.0       # Maximum acceptable impact
    timing_risk_aversion: float = 0.5         # Risk aversion parameter
    
    # Taiwan-specific parameters
    avoid_opening_minutes: int = 15           # Avoid first 15 minutes
    avoid_closing_minutes: int = 15           # Avoid last 15 minutes
    lunch_break_trading: bool = False         # Trade during lunch break
    
    # Adaptive parameters
    market_condition_adjustment: bool = True   # Adjust for market conditions
    volume_forecast_window: int = 20          # Days for volume forecasting
    
    def __post_init__(self):
        """Set default timing if not provided."""
        if self.start_time is None:
            # Default to market open + buffer
            today = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
            self.start_time = today
        
        if self.end_time is None:
            # Default to market close - buffer
            self.end_time = self.start_time + timedelta(hours=self.max_execution_time_hours)


@dataclass
class ExecutionSlice:
    """Individual execution slice within a strategy."""
    slice_id: int
    scheduled_time: datetime
    target_shares: float
    target_participation_rate: float
    estimated_price: float
    estimated_volume: float
    estimated_impact_bps: float
    session: TradingSession
    
    # Execution results (filled after execution)
    executed_shares: Optional[float] = None
    executed_price: Optional[float] = None
    actual_participation_rate: Optional[float] = None
    actual_impact_bps: Optional[float] = None
    execution_timestamp: Optional[datetime] = None


@dataclass
class ExecutionSchedule:
    """Complete execution schedule for an order."""
    order_id: str
    symbol: str
    total_shares: float
    strategy: ExecutionStrategy
    
    # Schedule details
    slices: List[ExecutionSlice]
    total_execution_time: timedelta
    estimated_total_impact_bps: float
    estimated_average_price: float
    
    # Risk metrics
    timing_risk_score: float
    market_impact_risk_score: float
    implementation_shortfall_bps: float
    
    # Performance tracking
    schedule_created_at: datetime
    last_updated_at: datetime
    
    def get_active_slices(self, current_time: datetime) -> List[ExecutionSlice]:
        """Get slices that should be executed now."""
        return [
            slice for slice in self.slices
            if (slice.scheduled_time <= current_time and 
                slice.executed_shares is None)
        ]
    
    def get_completion_percentage(self) -> float:
        """Get execution completion percentage."""
        executed_shares = sum(
            slice.executed_shares or 0 for slice in self.slices
        )
        return (executed_shares / abs(self.total_shares)) * 100 if self.total_shares != 0 else 0


class BaseExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def generate_schedule(
        self,
        symbol: str,
        total_shares: float,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics
    ) -> ExecutionSchedule:
        """Generate execution schedule."""
        pass
    
    def _get_trading_sessions(
        self, 
        start_time: datetime, 
        end_time: datetime,
        include_lunch: bool = False
    ) -> List[Tuple[datetime, datetime, TradingSession]]:
        """Get trading sessions within time range."""
        sessions = []
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            # Morning session: 09:00-12:00
            morning_start = datetime.combine(current_date, time(9, 0))
            morning_end = datetime.combine(current_date, time(12, 0))
            
            if self._sessions_overlap(start_time, end_time, morning_start, morning_end):
                session_start = max(start_time, morning_start)
                session_end = min(end_time, morning_end)
                sessions.append((session_start, session_end, TradingSession.MORNING))
            
            # Afternoon session: 13:30-13:30 (Taiwan has no afternoon session typically)
            # But we'll include for completeness in case of extended hours
            if include_lunch:
                afternoon_start = datetime.combine(current_date, time(13, 30))
                afternoon_end = datetime.combine(current_date, time(13, 30))  # Same time = no afternoon
                
                if afternoon_start != afternoon_end and self._sessions_overlap(start_time, end_time, afternoon_start, afternoon_end):
                    session_start = max(start_time, afternoon_start)
                    session_end = min(end_time, afternoon_end)
                    sessions.append((session_start, session_end, TradingSession.AFTERNOON))
            
            current_date += timedelta(days=1)
        
        return sessions
    
    def _sessions_overlap(
        self, 
        start1: datetime, 
        end1: datetime, 
        start2: datetime, 
        end2: datetime
    ) -> bool:
        """Check if two time periods overlap."""
        return start1 < end2 and start2 < end1
    
    def _estimate_slice_impact(
        self,
        slice_shares: float,
        estimated_volume: float,
        volatility: float,
        session: TradingSession
    ) -> float:
        """Estimate market impact for an execution slice."""
        if estimated_volume <= 0:
            return 50.0  # High impact for zero volume
        
        participation_rate = abs(slice_shares) / estimated_volume
        
        # Base impact model (simplified)
        base_impact = 20 * np.sqrt(participation_rate)  # Square root scaling
        
        # Session adjustments
        session_multipliers = {
            TradingSession.MORNING: 1.2,    # Higher impact in morning
            TradingSession.AFTERNOON: 1.0   # Normal impact
        }
        
        session_mult = session_multipliers.get(session, 1.0)
        
        # Volatility adjustment
        vol_mult = 1 + volatility
        
        return base_impact * session_mult * vol_mult


class TWAPStrategy(BaseExecutionStrategy):
    """Time-Weighted Average Price execution strategy."""
    
    def __init__(self):
        super().__init__("TWAP")
    
    def generate_schedule(
        self,
        symbol: str,
        total_shares: float,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics
    ) -> ExecutionSchedule:
        """
        Generate TWAP execution schedule.
        
        Distributes order evenly across available trading time.
        """
        # Get trading sessions
        sessions = self._get_trading_sessions(
            parameters.start_time,
            parameters.end_time,
            parameters.lunch_break_trading
        )
        
        if not sessions:
            raise ValueError("No trading sessions found in specified time range")
        
        # Calculate total trading minutes
        total_minutes = sum(
            (session_end - session_start).total_seconds() / 60
            for session_start, session_end, _ in sessions
        )
        
        # Apply opening/closing avoidance
        effective_minutes = total_minutes - (
            (parameters.avoid_opening_minutes + parameters.avoid_closing_minutes) * len(sessions)
        )
        
        # Calculate slice interval
        min_slices = max(1, int(abs(total_shares) / parameters.max_slice_size)) if parameters.max_slice_size else 1
        max_slices = min(100, int(effective_minutes / 5))  # Max one slice per 5 minutes
        target_slices = min(max_slices, max(min_slices, int(effective_minutes / 30)))  # Target one slice per 30 minutes
        
        slice_interval_minutes = effective_minutes / target_slices
        shares_per_slice = total_shares / target_slices
        
        # Generate execution slices
        slices = []
        slice_id = 0
        current_time = parameters.start_time
        
        for session_start, session_end, session_type in sessions:
            # Adjust session times for buffers
            effective_start = session_start + timedelta(minutes=parameters.avoid_opening_minutes)
            effective_end = session_end - timedelta(minutes=parameters.avoid_closing_minutes)
            
            if effective_start >= effective_end:
                continue  # Skip sessions that are too short
            
            # Generate slices for this session
            slice_time = max(effective_start, current_time)
            
            while slice_time <= effective_end and slice_id < target_slices:
                # Estimate market conditions at slice time
                estimated_volume = self._estimate_volume_at_time(
                    slice_time, market_data, liquidity_metrics
                )
                
                estimated_price = market_data.get('current_price', 100.0)
                
                # Calculate participation rate
                participation_rate = abs(shares_per_slice) / estimated_volume if estimated_volume > 0 else 0.5
                participation_rate = min(participation_rate, parameters.max_participation_rate)
                
                # Adjust slice size if participation rate too high
                adjusted_shares = shares_per_slice
                if participation_rate > parameters.max_participation_rate:
                    adjusted_shares = shares_per_slice * (parameters.max_participation_rate / participation_rate)
                
                # Estimate impact
                impact_bps = self._estimate_slice_impact(
                    adjusted_shares,
                    estimated_volume,
                    market_data.get('volatility', 0.25),
                    session_type
                )
                
                slice = ExecutionSlice(
                    slice_id=slice_id,
                    scheduled_time=slice_time,
                    target_shares=adjusted_shares,
                    target_participation_rate=participation_rate,
                    estimated_price=estimated_price,
                    estimated_volume=estimated_volume,
                    estimated_impact_bps=impact_bps,
                    session=session_type
                )
                
                slices.append(slice)
                slice_id += 1
                slice_time += timedelta(minutes=slice_interval_minutes)
            
            current_time = session_end
        
        # Calculate schedule metrics
        total_impact = np.mean([slice.estimated_impact_bps for slice in slices])
        avg_price = np.mean([slice.estimated_price for slice in slices])
        
        return ExecutionSchedule(
            order_id=f"{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            total_shares=total_shares,
            strategy=ExecutionStrategy.TWAP,
            slices=slices,
            total_execution_time=parameters.end_time - parameters.start_time,
            estimated_total_impact_bps=total_impact,
            estimated_average_price=avg_price,
            timing_risk_score=self._calculate_timing_risk(slices),
            market_impact_risk_score=total_impact / 100,  # Normalize to 0-1 scale
            implementation_shortfall_bps=total_impact,  # Simplified for TWAP
            schedule_created_at=datetime.now(),
            last_updated_at=datetime.now()
        )
    
    def _estimate_volume_at_time(
        self,
        target_time: datetime,
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics
    ) -> float:
        """Estimate trading volume at specific time."""
        # Use intraday volume pattern if available
        if 'intraday_volume_pattern' in market_data:
            hour = target_time.hour
            minute = target_time.minute
            pattern = market_data['intraday_volume_pattern']
            
            # Get volume factor for this time
            time_key = f"{hour:02d}:{minute//15*15:02d}"  # Round to 15-minute intervals
            volume_factor = pattern.get(time_key, 1.0)
        else:
            # Default Taiwan market volume pattern
            volume_factor = self._get_default_volume_pattern(target_time)
        
        # Base volume per minute
        trading_minutes_per_day = 270  # 4.5 hours * 60 minutes
        base_volume_per_minute = liquidity_metrics.adv_20d / trading_minutes_per_day
        
        return base_volume_per_minute * volume_factor
    
    def _get_default_volume_pattern(self, target_time: datetime) -> float:
        """Get default intraday volume pattern for Taiwan market."""
        hour = target_time.hour
        minute = target_time.minute
        time_decimal = hour + minute / 60
        
        # Taiwan market volume pattern (simplified)
        if 9.0 <= time_decimal < 9.5:      # Market open
            return 2.0
        elif 9.5 <= time_decimal < 11.0:   # Mid morning
            return 1.2
        elif 11.0 <= time_decimal < 12.0:  # Pre-lunch
            return 0.8
        else:  # Default
            return 1.0
    
    def _calculate_timing_risk(self, slices: List[ExecutionSlice]) -> float:
        """Calculate timing risk score for execution schedule."""
        if not slices:
            return 1.0
        
        # Risk factors
        time_spread = (slices[-1].scheduled_time - slices[0].scheduled_time).total_seconds() / 3600  # Hours
        avg_participation = np.mean([slice.target_participation_rate for slice in slices])
        
        # Normalize risk factors
        time_risk = min(1.0, time_spread / 6)  # 6 hours = max risk
        participation_risk = min(1.0, avg_participation / 0.2)  # 20% participation = max risk
        
        return (time_risk + participation_risk) / 2


class VWAPStrategy(BaseExecutionStrategy):
    """Volume-Weighted Average Price execution strategy."""
    
    def __init__(self):
        super().__init__("VWAP")
    
    def generate_schedule(
        self,
        symbol: str,
        total_shares: float,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics
    ) -> ExecutionSchedule:
        """
        Generate VWAP execution schedule.
        
        Distributes order proportionally to expected volume pattern.
        """
        # Get trading sessions
        sessions = self._get_trading_sessions(
            parameters.start_time,
            parameters.end_time,
            parameters.lunch_break_trading
        )
        
        if not sessions:
            raise ValueError("No trading sessions found in specified time range")
        
        # Build volume forecast
        volume_forecast = self._build_volume_forecast(
            sessions, market_data, liquidity_metrics, parameters
        )
        
        total_forecast_volume = sum(period['volume'] for period in volume_forecast)
        
        # Generate execution slices based on volume distribution
        slices = []
        slice_id = 0
        
        for period in volume_forecast:
            if period['volume'] <= 0:
                continue
            
            # Calculate shares for this period based on volume proportion
            volume_proportion = period['volume'] / total_forecast_volume
            period_shares = total_shares * volume_proportion
            
            # Respect participation rate limits
            max_period_shares = period['volume'] * parameters.max_participation_rate
            adjusted_shares = min(abs(period_shares), max_period_shares) * np.sign(period_shares)
            
            if abs(adjusted_shares) < parameters.min_slice_size:
                continue
            
            # Calculate participation rate
            participation_rate = abs(adjusted_shares) / period['volume']
            
            # Estimate impact
            impact_bps = self._estimate_slice_impact(
                adjusted_shares,
                period['volume'],
                market_data.get('volatility', 0.25),
                period['session']
            )
            
            slice = ExecutionSlice(
                slice_id=slice_id,
                scheduled_time=period['time'],
                target_shares=adjusted_shares,
                target_participation_rate=participation_rate,
                estimated_price=market_data.get('current_price', 100.0),
                estimated_volume=period['volume'],
                estimated_impact_bps=impact_bps,
                session=period['session']
            )
            
            slices.append(slice)
            slice_id += 1
        
        # Calculate schedule metrics
        total_impact = np.mean([slice.estimated_impact_bps for slice in slices])
        avg_price = np.mean([slice.estimated_price for slice in slices])
        
        return ExecutionSchedule(
            order_id=f"{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            total_shares=total_shares,
            strategy=ExecutionStrategy.VWAP,
            slices=slices,
            total_execution_time=parameters.end_time - parameters.start_time,
            estimated_total_impact_bps=total_impact,
            estimated_average_price=avg_price,
            timing_risk_score=self._calculate_vwap_risk(slices, volume_forecast),
            market_impact_risk_score=total_impact / 100,
            implementation_shortfall_bps=total_impact,
            schedule_created_at=datetime.now(),
            last_updated_at=datetime.now()
        )
    
    def _build_volume_forecast(
        self,
        sessions: List[Tuple[datetime, datetime, TradingSession]],
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics,
        parameters: ExecutionParameters
    ) -> List[Dict[str, Any]]:
        """Build volume forecast for execution periods."""
        forecast = []
        
        for session_start, session_end, session_type in sessions:
            # Create 15-minute periods within session
            current_time = session_start + timedelta(minutes=parameters.avoid_opening_minutes)
            session_end_adj = session_end - timedelta(minutes=parameters.avoid_closing_minutes)
            
            while current_time < session_end_adj:
                period_end = min(current_time + timedelta(minutes=15), session_end_adj)
                
                # Estimate volume for this period
                period_volume = self._estimate_period_volume(
                    current_time, period_end, market_data, liquidity_metrics
                )
                
                forecast.append({
                    'time': current_time,
                    'end_time': period_end,
                    'volume': period_volume,
                    'session': session_type
                })
                
                current_time = period_end
        
        return forecast
    
    def _estimate_period_volume(
        self,
        start_time: datetime,
        end_time: datetime,
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics
    ) -> float:
        """Estimate volume for a specific time period."""
        period_minutes = (end_time - start_time).total_seconds() / 60
        
        # Get volume pattern factor
        mid_time = start_time + (end_time - start_time) / 2
        volume_factor = self._get_default_volume_pattern(mid_time)
        
        # Base volume per minute
        trading_minutes_per_day = 270
        base_volume_per_minute = liquidity_metrics.adv_20d / trading_minutes_per_day
        
        return base_volume_per_minute * volume_factor * period_minutes
    
    def _calculate_vwap_risk(
        self, 
        slices: List[ExecutionSlice], 
        volume_forecast: List[Dict[str, Any]]
    ) -> float:
        """Calculate VWAP-specific risk score."""
        if not slices or not volume_forecast:
            return 1.0
        
        # Calculate volume tracking error
        total_forecast_volume = sum(p['volume'] for p in volume_forecast)
        total_slice_volume = sum(slice.estimated_volume for slice in slices)
        
        volume_tracking_error = abs(total_slice_volume - total_forecast_volume) / total_forecast_volume
        
        # Calculate participation rate variance
        participation_rates = [slice.target_participation_rate for slice in slices]
        participation_variance = np.var(participation_rates) if len(participation_rates) > 1 else 0
        
        return min(1.0, volume_tracking_error + participation_variance)


class AdaptiveStrategy(BaseExecutionStrategy):
    """Adaptive execution strategy that adjusts to market conditions."""
    
    def __init__(self, impact_model: TaiwanMarketImpactModel):
        super().__init__("Adaptive")
        self.impact_model = impact_model
    
    def generate_schedule(
        self,
        symbol: str,
        total_shares: float,
        parameters: ExecutionParameters,
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics
    ) -> ExecutionSchedule:
        """
        Generate adaptive execution schedule.
        
        Optimizes between market impact and timing risk.
        """
        # Start with TWAP baseline
        twap_strategy = TWAPStrategy()
        baseline_schedule = twap_strategy.generate_schedule(
            symbol, total_shares, parameters, market_data, liquidity_metrics
        )
        
        # Optimize slices based on market impact forecasts
        optimized_slices = self._optimize_slice_timing(
            baseline_schedule.slices, market_data, liquidity_metrics, parameters
        )
        
        # Recalculate metrics
        total_impact = np.mean([slice.estimated_impact_bps for slice in optimized_slices])
        avg_price = np.mean([slice.estimated_price for slice in optimized_slices])
        
        return ExecutionSchedule(
            order_id=f"{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            total_shares=total_shares,
            strategy=ExecutionStrategy.ADAPTIVE,
            slices=optimized_slices,
            total_execution_time=parameters.end_time - parameters.start_time,
            estimated_total_impact_bps=total_impact,
            estimated_average_price=avg_price,
            timing_risk_score=self._calculate_adaptive_risk(optimized_slices),
            market_impact_risk_score=total_impact / 100,
            implementation_shortfall_bps=total_impact,
            schedule_created_at=datetime.now(),
            last_updated_at=datetime.now()
        )
    
    def _optimize_slice_timing(
        self,
        baseline_slices: List[ExecutionSlice],
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics,
        parameters: ExecutionParameters
    ) -> List[ExecutionSlice]:
        """Optimize slice timing based on market impact forecasts."""
        optimized_slices = []
        
        for slice in baseline_slices:
            # Calculate detailed impact for this slice
            impact_result = self.impact_model.calculate_impact(
                symbol=slice.estimated_price,  # Placeholder
                order_size=slice.target_shares,
                price=slice.estimated_price,
                avg_daily_volume=liquidity_metrics.adv_20d,
                volatility=market_data.get('volatility', 0.25),
                timestamp=slice.scheduled_time
            )
            
            # Adjust slice based on impact analysis
            optimized_slice = ExecutionSlice(
                slice_id=slice.slice_id,
                scheduled_time=slice.scheduled_time,
                target_shares=slice.target_shares,
                target_participation_rate=slice.target_participation_rate,
                estimated_price=slice.estimated_price,
                estimated_volume=slice.estimated_volume,
                estimated_impact_bps=impact_result.total_impact_bps,
                session=slice.session
            )
            
            optimized_slices.append(optimized_slice)
        
        return optimized_slices
    
    def _calculate_adaptive_risk(self, slices: List[ExecutionSlice]) -> float:
        """Calculate risk score for adaptive strategy."""
        if not slices:
            return 1.0
        
        # Consider impact variance as risk factor
        impact_values = [slice.estimated_impact_bps for slice in slices]
        impact_variance = np.var(impact_values) if len(impact_values) > 1 else 0
        
        # Normalize variance to 0-1 scale
        return min(1.0, impact_variance / 100)  # 100 bps variance = max risk


class ExecutionOptimizer:
    """
    Execution timing optimizer that selects and configures optimal strategies.
    
    Analyzes market conditions and order characteristics to recommend
    the best execution approach.
    """
    
    def __init__(self, impact_model: TaiwanMarketImpactModel):
        self.impact_model = impact_model
        self.strategies = {
            ExecutionStrategy.TWAP: TWAPStrategy(),
            ExecutionStrategy.VWAP: VWAPStrategy(),
            ExecutionStrategy.ADAPTIVE: AdaptiveStrategy(impact_model)
        }
        self.logger = logging.getLogger(f"{__name__}.ExecutionOptimizer")
    
    def recommend_strategy(
        self,
        symbol: str,
        total_shares: float,
        urgency: UrgencyLevel,
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics
    ) -> Tuple[ExecutionStrategy, ExecutionParameters]:
        """
        Recommend optimal execution strategy and parameters.
        
        Args:
            symbol: Stock symbol
            total_shares: Total shares to trade
            urgency: Trade urgency level
            market_data: Current market data
            liquidity_metrics: Liquidity analysis
            
        Returns:
            Tuple of (recommended_strategy, optimized_parameters)
        """
        # Calculate order characteristics
        participation_rate = abs(total_shares) / liquidity_metrics.adv_20d
        liquidity_score = liquidity_metrics.liquidity_score
        
        # Determine strategy based on conditions
        if urgency == UrgencyLevel.URGENT:
            strategy = ExecutionStrategy.TWAP  # Fast execution
            max_participation = 0.25
            max_execution_hours = 1.0
        elif participation_rate > 0.15 or liquidity_score < 0.4:
            strategy = ExecutionStrategy.VWAP  # Follow volume for large/illiquid orders
            max_participation = 0.10
            max_execution_hours = 6.0
        elif liquidity_score > 0.7 and participation_rate < 0.05:
            strategy = ExecutionStrategy.ADAPTIVE  # Optimize for liquid, small orders
            max_participation = 0.20
            max_execution_hours = 4.0
        else:
            strategy = ExecutionStrategy.TWAP  # Default balanced approach
            max_participation = 0.15
            max_execution_hours = 4.0
        
        # Create optimized parameters
        parameters = ExecutionParameters(
            strategy=strategy,
            urgency=urgency,
            max_participation_rate=max_participation,
            target_participation_rate=max_participation * 0.7,  # 70% of max
            max_execution_time_hours=max_execution_hours,
            max_market_impact_bps=50.0 if urgency != UrgencyLevel.URGENT else 100.0,
            market_condition_adjustment=True
        )
        
        return strategy, parameters
    
    def generate_optimal_schedule(
        self,
        symbol: str,
        total_shares: float,
        urgency: UrgencyLevel,
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics,
        custom_parameters: Optional[ExecutionParameters] = None
    ) -> ExecutionSchedule:
        """
        Generate optimal execution schedule.
        
        Args:
            symbol: Stock symbol
            total_shares: Total shares to trade
            urgency: Trade urgency level
            market_data: Current market data
            liquidity_metrics: Liquidity analysis
            custom_parameters: Override default parameters
            
        Returns:
            Optimized ExecutionSchedule
        """
        if custom_parameters is None:
            strategy, parameters = self.recommend_strategy(
                symbol, total_shares, urgency, market_data, liquidity_metrics
            )
        else:
            strategy = custom_parameters.strategy
            parameters = custom_parameters
        
        # Generate schedule using selected strategy
        strategy_impl = self.strategies[strategy]
        
        return strategy_impl.generate_schedule(
            symbol=symbol,
            total_shares=total_shares,
            parameters=parameters,
            market_data=market_data,
            liquidity_metrics=liquidity_metrics
        )
    
    def compare_strategies(
        self,
        symbol: str,
        total_shares: float,
        market_data: Dict[str, Any],
        liquidity_metrics: LiquidityMetrics
    ) -> Dict[ExecutionStrategy, ExecutionSchedule]:
        """
        Compare different execution strategies for an order.
        
        Returns:
            Dictionary mapping strategies to their schedules
        """
        results = {}
        base_parameters = ExecutionParameters()
        
        for strategy_type, strategy_impl in self.strategies.items():
            try:
                parameters = ExecutionParameters(strategy=strategy_type)
                schedule = strategy_impl.generate_schedule(
                    symbol=symbol,
                    total_shares=total_shares,
                    parameters=parameters,
                    market_data=market_data,
                    liquidity_metrics=liquidity_metrics
                )
                results[strategy_type] = schedule
            except Exception as e:
                self.logger.warning(f"Failed to generate {strategy_type.value} schedule: {e}")
        
        return results


# Factory function
def create_execution_optimizer(impact_model: Optional[TaiwanMarketImpactModel] = None) -> ExecutionOptimizer:
    """Create execution optimizer with Taiwan market configuration."""
    if impact_model is None:
        from .market_impact import create_taiwan_impact_model
        impact_model = create_taiwan_impact_model()
    
    return ExecutionOptimizer(impact_model)


# Example usage and testing
if __name__ == "__main__":
    print("Taiwan Execution Timing Optimization Demo")
    
    # Create sample market data
    sample_market_data = {
        'current_price': 500.0,
        'volatility': 0.25,
        'avg_daily_volume': 500_000
    }
    
    # Create sample liquidity metrics
    from .liquidity import LiquidityMetrics, LiquidityTier
    sample_liquidity = LiquidityMetrics(
        symbol="2330.TW",
        date=date.today(),
        adv_20d=500_000,
        adv_60d=480_000,
        lav_20d=520_000,
        current_volume=400_000,
        volume_ratio=0.8,
        liquidity_tier=LiquidityTier.LIQUID,
        liquidity_score=0.75
    )
    
    # Create optimizer
    optimizer = create_execution_optimizer()
    
    # Get strategy recommendation
    strategy, parameters = optimizer.recommend_strategy(
        symbol="2330.TW",
        total_shares=50_000,  # 50K shares
        urgency=UrgencyLevel.MEDIUM,
        market_data=sample_market_data,
        liquidity_metrics=sample_liquidity
    )
    
    print(f"\nRecommended Strategy: {strategy.value}")
    print(f"Max Participation Rate: {parameters.max_participation_rate:.1%}")
    print(f"Max Execution Time: {parameters.max_execution_time_hours:.1f} hours")
    
    # Generate optimal schedule
    schedule = optimizer.generate_optimal_schedule(
        symbol="2330.TW",
        total_shares=50_000,
        urgency=UrgencyLevel.MEDIUM,
        market_data=sample_market_data,
        liquidity_metrics=sample_liquidity
    )
    
    print(f"\nExecution Schedule:")
    print(f"Strategy: {schedule.strategy.value}")
    print(f"Number of Slices: {len(schedule.slices)}")
    print(f"Total Execution Time: {schedule.total_execution_time}")
    print(f"Estimated Impact: {schedule.estimated_total_impact_bps:.1f} bps")
    print(f"Implementation Shortfall: {schedule.implementation_shortfall_bps:.1f} bps")
    
    # Show first few slices
    print(f"\nFirst 3 Execution Slices:")
    for i, slice in enumerate(schedule.slices[:3]):
        print(f"  Slice {slice.slice_id}: {slice.scheduled_time.strftime('%H:%M')} - "
              f"{slice.target_shares:,.0f} shares ({slice.target_participation_rate:.1%} participation, "
              f"{slice.estimated_impact_bps:.1f} bps impact)")
    
    # Compare strategies
    print(f"\nStrategy Comparison:")
    comparison = optimizer.compare_strategies(
        symbol="2330.TW",
        total_shares=50_000,
        market_data=sample_market_data,
        liquidity_metrics=sample_liquidity
    )
    
    for strategy_type, strategy_schedule in comparison.items():
        print(f"  {strategy_type.value}: {strategy_schedule.estimated_total_impact_bps:.1f} bps impact, "
              f"{len(strategy_schedule.slices)} slices, "
              f"{strategy_schedule.timing_risk_score:.2f} timing risk")