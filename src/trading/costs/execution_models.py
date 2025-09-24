"""
Execution cost simulation and models for Taiwan market.

This module implements execution cost simulation including slippage modeling,
timing costs, settlement costs, and realistic execution simulation with
market timing considerations.

Key Features:
- Execution cost simulation with market timing
- Slippage and market impact modeling  
- T+2 settlement cost integration
- TWAP/VWAP execution strategies
- Market session timing optimization
- Realistic execution delays and partial fills
"""

from datetime import datetime, date, time, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from decimal import Decimal
import random

from .cost_models import TradeInfo, TradeDirection, TradeCostBreakdown
from .taiwan_microstructure import TaiwanMarketStructure, TradingSession
from ...data.core.temporal import TemporalStore, DataType

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategy types."""
    MARKET = "market"           # Immediate market order
    LIMIT = "limit"             # Limit order
    TWAP = "twap"              # Time-weighted average price
    VWAP = "vwap"              # Volume-weighted average price
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ADAPTIVE = "adaptive"       # Adaptive algorithm


class FillType(Enum):
    """Order fill types."""
    FULL_FILL = "full_fill"
    PARTIAL_FILL = "partial_fill"
    NO_FILL = "no_fill"


@dataclass
class ExecutionFill:
    """Single execution fill."""
    timestamp: datetime
    price: float
    quantity: int
    fill_type: FillType
    market_session: TradingSession
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost_bps: float = 0.0


@dataclass
class ExecutionResult:
    """Complete execution result."""
    symbol: str
    original_order_size: int
    original_price_reference: float
    strategy: ExecutionStrategy
    start_time: datetime
    end_time: datetime
    
    # Execution details
    fills: List[ExecutionFill] = field(default_factory=list)
    total_filled: int = 0
    remaining_quantity: int = 0
    average_fill_price: float = 0.0
    
    # Cost analysis
    total_slippage_bps: float = 0.0
    total_market_impact_bps: float = 0.0
    total_timing_cost_bps: float = 0.0
    total_execution_cost_bps: float = 0.0
    execution_efficiency_score: float = 0.0
    
    def __post_init__(self):
        """Calculate execution metrics."""
        if self.fills:
            total_value = sum(fill.price * abs(fill.quantity) for fill in self.fills)
            total_quantity = sum(abs(fill.quantity) for fill in self.fills)
            
            if total_quantity > 0:
                self.average_fill_price = total_value / total_quantity
                self.total_filled = total_quantity
                self.remaining_quantity = abs(self.original_order_size) - total_quantity
                
                # Calculate weighted average costs
                weights = [abs(fill.quantity) for fill in self.fills]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    self.total_slippage_bps = sum(
                        fill.slippage_bps * (abs(fill.quantity) / total_weight)
                        for fill in self.fills
                    )
                    self.total_market_impact_bps = sum(
                        fill.market_impact_bps * (abs(fill.quantity) / total_weight)
                        for fill in self.fills
                    )
                    self.total_timing_cost_bps = sum(
                        fill.timing_cost_bps * (abs(fill.quantity) / total_weight)
                        for fill in self.fills
                    )
                
                self.total_execution_cost_bps = (
                    self.total_slippage_bps + 
                    self.total_market_impact_bps + 
                    self.total_timing_cost_bps
                )
                
                # Execution efficiency (fill rate and cost efficiency)
                fill_rate = self.total_filled / abs(self.original_order_size)
                cost_efficiency = max(0, 100 - self.total_execution_cost_bps)
                self.execution_efficiency_score = (fill_rate * 0.6 + cost_efficiency / 100 * 0.4) * 100


class SlippageModel:
    """
    Slippage modeling for Taiwan market execution.
    
    Models realistic slippage based on order size, market conditions,
    and execution timing.
    """
    
    def __init__(self):
        # Taiwan market slippage parameters
        self.base_slippage_bps = 3.0        # Base slippage
        self.size_sensitivity = 0.8         # Size impact on slippage
        self.volatility_multiplier = 2.0    # Volatility impact
        self.session_multipliers = {
            TradingSession.MORNING: 1.2,     # Higher slippage at open
            TradingSession.LUNCH_BREAK: 2.0, # Higher during low liquidity
            TradingSession.AFTERNOON: 1.0,   # Normal afternoon
            TradingSession.CLOSED: 3.0       # Very high when closed
        }
        
        logger.info("Taiwan slippage model initialized")
    
    def calculate_slippage(
        self,
        order_size: int,
        avg_daily_volume: float,
        volatility: float,
        market_session: TradingSession,
        reference_price: float,
        execution_price: float
    ) -> float:
        """
        Calculate execution slippage in basis points.
        
        Args:
            order_size: Order size in shares
            avg_daily_volume: Average daily volume
            volatility: Historical volatility
            market_session: Current trading session
            reference_price: Reference price (e.g., decision price)
            execution_price: Actual execution price
            
        Returns:
            Slippage in basis points
        """
        # Participation rate impact
        participation_rate = abs(order_size) / avg_daily_volume
        size_impact = self.size_sensitivity * (participation_rate ** 0.5)
        
        # Volatility impact
        vol_impact = self.volatility_multiplier * volatility
        
        # Session impact
        session_multiplier = self.session_multipliers.get(market_session, 1.0)
        
        # Base model slippage
        model_slippage_bps = (
            self.base_slippage_bps * 
            (1 + size_impact) * 
            (1 + vol_impact) * 
            session_multiplier
        )
        
        # Actual slippage from price difference
        if reference_price > 0:
            actual_slippage_bps = abs(execution_price - reference_price) / reference_price * 10000
            
            # Return the larger of model prediction or actual
            return max(model_slippage_bps, actual_slippage_bps)
        
        return model_slippage_bps
    
    def simulate_execution_price(
        self,
        reference_price: float,
        order_size: int,
        avg_daily_volume: float,
        volatility: float,
        market_session: TradingSession,
        direction: TradeDirection
    ) -> float:
        """
        Simulate realistic execution price with slippage.
        
        Args:
            reference_price: Reference price
            order_size: Order size
            avg_daily_volume: Average daily volume
            volatility: Volatility
            market_session: Trading session
            direction: Buy or sell
            
        Returns:
            Simulated execution price
        """
        # Calculate expected slippage
        expected_slippage_bps = self.calculate_slippage(
            order_size, avg_daily_volume, volatility, 
            market_session, reference_price, reference_price
        )
        
        # Add random component (normal distribution around expected)
        slippage_std = expected_slippage_bps * 0.3
        actual_slippage_bps = np.random.normal(expected_slippage_bps, slippage_std)
        actual_slippage_bps = max(0, actual_slippage_bps)  # No negative slippage
        
        # Apply direction (buy = higher price, sell = lower price)
        direction_multiplier = 1 if direction == TradeDirection.BUY else -1
        price_impact = reference_price * (actual_slippage_bps / 10000) * direction_multiplier
        
        return reference_price + price_impact


class TimingCostModel:
    """
    Timing cost model for execution delays.
    
    Models opportunity cost of delayed execution relative to
    decision time, accounting for market movement.
    """
    
    def __init__(self):
        # Taiwan market timing parameters
        self.base_timing_cost_per_minute = 0.5  # bps per minute delay
        self.volatility_scaling = 1.5            # Volatility impact on timing cost
        self.momentum_persistence = 0.3          # Price momentum persistence
        
        logger.info("Taiwan timing cost model initialized")
    
    def calculate_timing_cost(
        self,
        decision_price: float,
        execution_price: float,
        delay_minutes: float,
        volatility: float,
        market_direction: float = 0.0  # -1 to 1, market momentum
    ) -> float:
        """
        Calculate timing cost in basis points.
        
        Args:
            decision_price: Price at decision time
            execution_price: Actual execution price
            delay_minutes: Execution delay in minutes
            volatility: Historical volatility
            market_direction: Market momentum (-1 to 1)
            
        Returns:
            Timing cost in basis points
        """
        if delay_minutes <= 0 or decision_price <= 0:
            return 0.0
        
        # Base timing cost
        base_cost_bps = delay_minutes * self.base_timing_cost_per_minute
        
        # Volatility adjustment
        vol_adjustment = 1 + (volatility * self.volatility_scaling)
        
        # Momentum adjustment
        momentum_adjustment = 1 + abs(market_direction) * self.momentum_persistence
        
        # Calculate expected timing cost
        expected_timing_cost_bps = base_cost_bps * vol_adjustment * momentum_adjustment
        
        # Actual timing cost from price movement
        if execution_price != decision_price:
            actual_timing_cost_bps = abs(execution_price - decision_price) / decision_price * 10000
            
            # Attribute portion to timing (rest might be slippage/impact)
            timing_attribution = min(1.0, delay_minutes / 30.0)  # Full attribution after 30 min
            attributed_timing_cost = actual_timing_cost_bps * timing_attribution
            
            return max(expected_timing_cost_bps, attributed_timing_cost)
        
        return expected_timing_cost_bps


class SettlementCostModel:
    """
    T+2 settlement cost model for Taiwan market.
    
    Models financing costs and settlement risks associated
    with Taiwan's T+2 settlement cycle.
    """
    
    def __init__(self):
        # Taiwan settlement parameters
        self.settlement_days = 2
        self.funding_rate_annual = 0.02  # 2% annual funding rate
        self.settlement_risk_premium = 0.5  # 0.5 bps per transaction
        
        logger.info("Taiwan settlement cost model initialized")
    
    def calculate_settlement_cost(
        self,
        trade_value: float,
        trade_date: date,
        funding_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate settlement-related costs.
        
        Args:
            trade_value: Trade value in TWD
            trade_date: Trade date
            funding_rate: Custom funding rate (optional)
            
        Returns:
            Dictionary with settlement cost components
        """
        if funding_rate is None:
            funding_rate = self.funding_rate_annual
        
        # Daily funding rate
        daily_funding_rate = funding_rate / 365
        
        # Financing cost for T+2 settlement
        financing_cost = trade_value * daily_funding_rate * self.settlement_days
        
        # Settlement risk premium
        risk_premium = trade_value * (self.settlement_risk_premium / 10000)
        
        # Calculate settlement date
        settlement_date = trade_date + timedelta(days=self.settlement_days)
        # Skip weekends (simplified)
        while settlement_date.weekday() >= 5:
            settlement_date += timedelta(days=1)
        
        return {
            'financing_cost': financing_cost,
            'settlement_risk_premium': risk_premium,
            'total_settlement_cost': financing_cost + risk_premium,
            'settlement_date': settlement_date,
            'funding_days': (settlement_date - trade_date).days
        }


class ExecutionCostSimulator:
    """
    Comprehensive execution cost simulator for Taiwan market.
    
    Simulates realistic order execution with various strategies,
    market conditions, and timing constraints.
    """
    
    def __init__(self, temporal_store: Optional[TemporalStore] = None):
        self.temporal_store = temporal_store
        self.market_structure = TaiwanMarketStructure(temporal_store)
        self.slippage_model = SlippageModel()
        self.timing_model = TimingCostModel()
        self.settlement_model = SettlementCostModel()
        
        # Execution parameters
        self.max_participation_rate = 0.15  # Max 15% of volume per interval
        self.fill_probability_base = 0.8     # Base fill probability
        
        logger.info("Taiwan execution cost simulator initialized")
    
    def simulate_execution(
        self,
        symbol: str,
        order_size: int,
        reference_price: float,
        strategy: ExecutionStrategy,
        start_time: datetime,
        avg_daily_volume: float,
        volatility: float,
        max_execution_time_minutes: int = 240,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Simulate order execution with specified strategy.
        
        Args:
            symbol: Stock symbol
            order_size: Order size (positive for buy, negative for sell)
            reference_price: Reference price at decision time
            strategy: Execution strategy
            start_time: Execution start time
            avg_daily_volume: Average daily volume
            volatility: Historical volatility
            max_execution_time_minutes: Maximum execution time
            market_conditions: Additional market conditions
            
        Returns:
            Execution result with fills and cost analysis
        """
        direction = TradeDirection.BUY if order_size > 0 else TradeDirection.SELL
        remaining_quantity = abs(order_size)
        current_time = start_time
        fills = []
        
        # Strategy-specific parameters
        if strategy == ExecutionStrategy.MARKET:
            execution_intervals = [1]  # Execute immediately
            interval_duration = 1
        elif strategy == ExecutionStrategy.TWAP:
            execution_intervals = self._calculate_twap_schedule(
                remaining_quantity, max_execution_time_minutes
            )
            interval_duration = max_execution_time_minutes / len(execution_intervals)
        elif strategy == ExecutionStrategy.VWAP:
            execution_intervals = self._calculate_vwap_schedule(
                remaining_quantity, avg_daily_volume, max_execution_time_minutes
            )
            interval_duration = max_execution_time_minutes / len(execution_intervals)
        else:
            # Default to TWAP-like execution
            execution_intervals = self._calculate_twap_schedule(
                remaining_quantity, max_execution_time_minutes
            )
            interval_duration = max_execution_time_minutes / len(execution_intervals)
        
        # Execute order in intervals
        for interval_quantity in execution_intervals:
            if remaining_quantity <= 0:
                break
            
            # Determine current market session
            market_session = self.market_structure.get_current_session(current_time)
            
            # Skip execution during market closure
            if market_session == TradingSession.CLOSED:
                current_time += timedelta(minutes=interval_duration)
                continue
            
            # Calculate execution quantity for this interval
            actual_quantity = min(interval_quantity, remaining_quantity)
            
            # Simulate fill probability
            fill_probability = self._calculate_fill_probability(
                actual_quantity, avg_daily_volume, market_session, strategy
            )
            
            if random.random() < fill_probability:
                # Execute fill
                execution_price = self.slippage_model.simulate_execution_price(
                    reference_price, actual_quantity, avg_daily_volume,
                    volatility, market_session, direction
                )
                
                # Calculate costs
                slippage_bps = self.slippage_model.calculate_slippage(
                    actual_quantity, avg_daily_volume, volatility,
                    market_session, reference_price, execution_price
                )
                
                # Market impact (simplified)
                participation_rate = actual_quantity / avg_daily_volume
                market_impact_bps = min(20.0, participation_rate * 100 * volatility * 10000)
                
                # Timing cost
                delay_minutes = (current_time - start_time).total_seconds() / 60
                timing_cost_bps = self.timing_model.calculate_timing_cost(
                    reference_price, execution_price, delay_minutes, volatility
                )
                
                # Create fill
                fill = ExecutionFill(
                    timestamp=current_time,
                    price=execution_price,
                    quantity=actual_quantity if direction == TradeDirection.BUY else -actual_quantity,
                    fill_type=FillType.FULL_FILL if actual_quantity == interval_quantity else FillType.PARTIAL_FILL,
                    market_session=market_session,
                    slippage_bps=slippage_bps,
                    market_impact_bps=market_impact_bps,
                    timing_cost_bps=timing_cost_bps
                )
                
                fills.append(fill)
                remaining_quantity -= actual_quantity
                
                logger.debug(f"Fill executed: {actual_quantity} shares at {execution_price:.2f}")
            
            # Move to next interval
            current_time += timedelta(minutes=interval_duration)
            
            # Check if we've exceeded maximum execution time
            if (current_time - start_time).total_seconds() / 60 > max_execution_time_minutes:
                break
        
        # Create execution result
        result = ExecutionResult(
            symbol=symbol,
            original_order_size=order_size,
            original_price_reference=reference_price,
            strategy=strategy,
            start_time=start_time,
            end_time=current_time,
            fills=fills
        )
        
        logger.info(f"Execution simulation completed: {result.total_filled}/{abs(order_size)} shares filled")
        
        return result
    
    def _calculate_twap_schedule(
        self, 
        total_quantity: int, 
        duration_minutes: int
    ) -> List[int]:
        """Calculate TWAP execution schedule."""
        num_intervals = min(duration_minutes // 5, 48)  # 5-minute intervals, max 48
        if num_intervals <= 0:
            return [total_quantity]
        
        base_quantity = total_quantity // num_intervals
        remainder = total_quantity % num_intervals
        
        schedule = [base_quantity] * num_intervals
        # Distribute remainder across first intervals
        for i in range(remainder):
            schedule[i] += 1
        
        return schedule
    
    def _calculate_vwap_schedule(
        self, 
        total_quantity: int, 
        avg_daily_volume: float,
        duration_minutes: int
    ) -> List[int]:
        """Calculate VWAP execution schedule."""
        # Simplified VWAP schedule - would typically use historical volume patterns
        # For now, use TWAP with slight volume weighting
        
        num_intervals = min(duration_minutes // 10, 24)  # 10-minute intervals
        if num_intervals <= 0:
            return [total_quantity]
        
        # Simple volume pattern (higher in morning, lower at lunch)
        volume_weights = []
        for i in range(num_intervals):
            hour_of_day = 9 + (i * 10) / 60  # Assuming start at 9 AM
            if 9 <= hour_of_day <= 11:
                weight = 1.2  # Higher morning volume
            elif 11 < hour_of_day <= 12:
                weight = 0.8  # Lower pre-lunch volume
            else:
                weight = 1.0  # Normal volume
            volume_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(volume_weights)
        normalized_weights = [w / total_weight for w in volume_weights]
        
        # Calculate quantities
        schedule = [int(total_quantity * weight) for weight in normalized_weights]
        
        # Adjust for rounding
        scheduled_total = sum(schedule)
        difference = total_quantity - scheduled_total
        if difference != 0:
            schedule[0] += difference
        
        return schedule
    
    def _calculate_fill_probability(
        self,
        quantity: int,
        avg_daily_volume: float,
        market_session: TradingSession,
        strategy: ExecutionStrategy
    ) -> float:
        """Calculate probability of fill for a given quantity."""
        # Base probability
        base_prob = self.fill_probability_base
        
        # Size impact
        participation_rate = quantity / avg_daily_volume
        if participation_rate > self.max_participation_rate:
            size_penalty = (participation_rate - self.max_participation_rate) * 2
            base_prob *= (1 - size_penalty)
        
        # Session impact
        session_multipliers = {
            TradingSession.MORNING: 0.95,
            TradingSession.LUNCH_BREAK: 0.5,
            TradingSession.AFTERNOON: 1.0,
            TradingSession.CLOSED: 0.0
        }
        
        session_multiplier = session_multipliers.get(market_session, 0.8)
        base_prob *= session_multiplier
        
        # Strategy impact
        strategy_multipliers = {
            ExecutionStrategy.MARKET: 1.0,
            ExecutionStrategy.LIMIT: 0.7,
            ExecutionStrategy.TWAP: 0.9,
            ExecutionStrategy.VWAP: 0.85
        }
        
        strategy_multiplier = strategy_multipliers.get(strategy, 0.8)
        base_prob *= strategy_multiplier
        
        return max(0.0, min(1.0, base_prob))
    
    def analyze_execution_quality(
        self, 
        execution_result: ExecutionResult,
        benchmark_strategy: ExecutionStrategy = ExecutionStrategy.TWAP
    ) -> Dict[str, Any]:
        """
        Analyze execution quality against benchmarks.
        
        Args:
            execution_result: Execution result to analyze
            benchmark_strategy: Strategy to benchmark against
            
        Returns:
            Execution quality analysis
        """
        # Fill rate analysis
        fill_rate = execution_result.total_filled / abs(execution_result.original_order_size)
        
        # Cost analysis
        total_cost_bps = execution_result.total_execution_cost_bps
        
        # Timing analysis
        execution_duration = (execution_result.end_time - execution_result.start_time).total_seconds() / 60
        
        # Price improvement analysis
        if execution_result.fills:
            price_improvement = (
                execution_result.average_fill_price - execution_result.original_price_reference
            ) / execution_result.original_price_reference * 10000
        else:
            price_improvement = 0.0
        
        return {
            'symbol': execution_result.symbol,
            'strategy': execution_result.strategy.value,
            'execution_summary': {
                'fill_rate': fill_rate,
                'total_filled': execution_result.total_filled,
                'remaining_quantity': execution_result.remaining_quantity,
                'execution_duration_minutes': execution_duration,
                'number_of_fills': len(execution_result.fills)
            },
            'cost_analysis': {
                'total_cost_bps': total_cost_bps,
                'slippage_bps': execution_result.total_slippage_bps,
                'market_impact_bps': execution_result.total_market_impact_bps,
                'timing_cost_bps': execution_result.total_timing_cost_bps
            },
            'price_analysis': {
                'reference_price': execution_result.original_price_reference,
                'average_fill_price': execution_result.average_fill_price,
                'price_improvement_bps': price_improvement
            },
            'quality_metrics': {
                'execution_efficiency_score': execution_result.execution_efficiency_score,
                'cost_efficiency': max(0, 100 - total_cost_bps),
                'fill_efficiency': fill_rate * 100,
                'overall_grade': self._calculate_execution_grade(execution_result)
            }
        }
    
    def _calculate_execution_grade(self, execution_result: ExecutionResult) -> str:
        """Calculate execution grade (A-F)."""
        score = execution_result.execution_efficiency_score
        
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'


# Example usage and testing
if __name__ == "__main__":
    print("Taiwan Execution Cost Simulation Demo")
    
    # Initialize simulator
    simulator = ExecutionCostSimulator()
    
    # Sample execution simulation
    result = simulator.simulate_execution(
        symbol="2330.TW",
        order_size=5000,  # Buy 5000 shares
        reference_price=500.0,
        strategy=ExecutionStrategy.TWAP,
        start_time=datetime.now().replace(hour=9, minute=30),
        avg_daily_volume=100000,
        volatility=0.25,
        max_execution_time_minutes=120  # 2 hours
    )
    
    print(f"\nExecution Result for {result.symbol}:")
    print(f"Strategy: {result.strategy.value}")
    print(f"Fill Rate: {result.total_filled}/{abs(result.original_order_size)} ({result.total_filled/abs(result.original_order_size)*100:.1f}%)")
    print(f"Average Fill Price: NT${result.average_fill_price:.2f}")
    print(f"Total Execution Cost: {result.total_execution_cost_bps:.2f} bps")
    print(f"Efficiency Score: {result.execution_efficiency_score:.1f}")
    
    # Analyze execution quality
    quality_analysis = simulator.analyze_execution_quality(result)
    print(f"\nExecution Grade: {quality_analysis['quality_metrics']['overall_grade']}")
    print(f"Cost Efficiency: {quality_analysis['quality_metrics']['cost_efficiency']:.1f}%")
    print(f"Fill Efficiency: {quality_analysis['quality_metrics']['fill_efficiency']:.1f}%")