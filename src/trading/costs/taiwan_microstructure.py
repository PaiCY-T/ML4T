"""
Taiwan market microstructure models.

This module implements market microstructure models specifically for the Taiwan
Stock Exchange (TSE) and Taipei Exchange (TPEx), including bid-ask spread
modeling, market impact analysis, and tick size handling.

Key Features:
- Taiwan-specific bid-ask spread modeling
- Market impact models with temporary/permanent components
- Tick size handling for different price levels
- Session-based trading patterns
- Market depth and liquidity analysis
"""

from datetime import datetime, date, time, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from decimal import Decimal

from ...data.core.temporal import TemporalStore, DataType, TemporalValue

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Taiwan market trading sessions."""
    MORNING = "morning"      # 09:00-12:00
    LUNCH_BREAK = "lunch"    # 12:00-13:30
    AFTERNOON = "afternoon"  # 13:30-13:30 (special session if any)
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


class MarketDepthLevel(Enum):
    """Market depth analysis levels."""
    LEVEL_1 = "level_1"  # Best bid/ask
    LEVEL_5 = "level_5"  # Top 5 levels
    LEVEL_10 = "level_10"  # Top 10 levels


@dataclass
class TickSizeSchedule:
    """Taiwan tick size schedule based on price levels."""
    price_ranges: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.01, 10.0, 0.01),      # NT$0.01 - NT$10.00: tick = NT$0.01
        (10.01, 50.0, 0.05),     # NT$10.01 - NT$50.00: tick = NT$0.05
        (50.01, 100.0, 0.10),    # NT$50.01 - NT$100.00: tick = NT$0.10
        (100.01, 500.0, 0.50),   # NT$100.01 - NT$500.00: tick = NT$0.50
        (500.01, 1000.0, 1.00),  # NT$500.01 - NT$1000.00: tick = NT$1.00
        (1000.01, float('inf'), 5.00)  # > NT$1000.00: tick = NT$5.00
    ])
    
    def get_tick_size(self, price: float) -> float:
        """Get tick size for a given price."""
        for min_price, max_price, tick_size in self.price_ranges:
            if min_price <= price <= max_price:
                return tick_size
        return 5.00  # Default for very high prices


@dataclass
class BidAskSpreadData:
    """Bid-ask spread data structure."""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    spread: float = field(init=False)
    spread_bps: float = field(init=False)
    mid_price: float = field(init=False)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.spread = self.ask_price - self.bid_price
        self.mid_price = (self.bid_price + self.ask_price) / 2
        self.spread_bps = (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0.0


@dataclass
class MarketImpactData:
    """Market impact analysis data."""
    symbol: str
    order_size: float
    avg_daily_volume: float
    participation_rate: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    decay_half_life_minutes: float = 30.0  # Temporary impact decay


class TaiwanTickSizeModel:
    """
    Taiwan market tick size model.
    
    Handles tick size calculations based on Taiwan Stock Exchange
    and Taipei Exchange tick size schedules.
    """
    
    def __init__(self):
        self.tse_schedule = TickSizeSchedule()
        self.tpex_schedule = TickSizeSchedule()  # Same as TSE for most stocks
        
        logger.info("Taiwan tick size model initialized")
    
    def get_tick_size(self, price: float, exchange: str = "TSE") -> float:
        """
        Get tick size for a given price and exchange.
        
        Args:
            price: Stock price in TWD
            exchange: Exchange code ("TSE" or "TPEx")
            
        Returns:
            Tick size in TWD
        """
        if exchange.upper() == "TPEX":
            return self.tpex_schedule.get_tick_size(price)
        else:
            return self.tse_schedule.get_tick_size(price)
    
    def round_to_tick(self, price: float, exchange: str = "TSE") -> float:
        """Round price to nearest valid tick."""
        tick_size = self.get_tick_size(price, exchange)
        return round(price / tick_size) * tick_size
    
    def get_min_spread_bps(self, price: float, exchange: str = "TSE") -> float:
        """Get minimum possible spread in basis points."""
        tick_size = self.get_tick_size(price, exchange)
        return (tick_size / price) * 10000 if price > 0 else 0.0


class BidAskSpreadModel:
    """
    Taiwan market bid-ask spread prediction model.
    
    Predicts bid-ask spreads based on market conditions, volatility,
    volume, and time-of-day patterns specific to Taiwan market.
    """
    
    def __init__(self, temporal_store: Optional[TemporalStore] = None):
        self.temporal_store = temporal_store
        self.tick_model = TaiwanTickSizeModel()
        
        # Model parameters calibrated for Taiwan market
        self.base_spread_factor = 0.15  # Base spread as % of tick size
        self.volatility_sensitivity = 3.0
        self.volume_sensitivity = -0.5  # Higher volume = lower spread
        self.time_of_day_factors = {
            'morning_open': 1.5,    # 09:00-09:30
            'morning_mid': 1.0,     # 09:30-11:30
            'morning_close': 1.2,   # 11:30-12:00
            'afternoon_open': 1.3,  # 13:30-14:00
            'afternoon_close': 1.4  # After 14:00
        }
        
        logger.info("Taiwan bid-ask spread model initialized")
    
    def predict_spread(
        self,
        symbol: str,
        price: float,
        volatility: float,
        volume_ratio: float = 1.0,
        timestamp: Optional[datetime] = None,
        exchange: str = "TSE"
    ) -> float:
        """
        Predict bid-ask spread for a stock.
        
        Args:
            symbol: Stock symbol
            price: Current price
            volatility: Historical volatility
            volume_ratio: Current volume vs average
            timestamp: Time for session-based adjustments
            exchange: Exchange code
            
        Returns:
            Predicted spread in TWD
        """
        # Get minimum tick-based spread
        tick_size = self.tick_model.get_tick_size(price, exchange)
        min_spread = tick_size
        
        # Base spread calculation
        base_spread = tick_size * self.base_spread_factor
        
        # Volatility adjustment
        vol_adjustment = 1 + (volatility * self.volatility_sensitivity)
        
        # Volume adjustment
        volume_adjustment = volume_ratio ** self.volume_sensitivity
        
        # Time-of-day adjustment
        time_adjustment = self._get_time_of_day_factor(timestamp)
        
        # Calculate predicted spread
        predicted_spread = (
            base_spread * 
            vol_adjustment * 
            volume_adjustment * 
            time_adjustment
        )
        
        # Ensure spread is at least one tick
        return max(predicted_spread, min_spread)
    
    def predict_spread_bps(
        self,
        symbol: str,
        price: float,
        volatility: float,
        volume_ratio: float = 1.0,
        timestamp: Optional[datetime] = None,
        exchange: str = "TSE"
    ) -> float:
        """Predict spread in basis points."""
        spread_twd = self.predict_spread(
            symbol, price, volatility, volume_ratio, timestamp, exchange
        )
        return (spread_twd / price) * 10000 if price > 0 else 0.0
    
    def _get_time_of_day_factor(self, timestamp: Optional[datetime]) -> float:
        """Get time-of-day factor for spread prediction."""
        if timestamp is None:
            return 1.0
        
        time_only = timestamp.time()
        
        # Morning session
        if time(9, 0) <= time_only <= time(9, 30):
            return self.time_of_day_factors['morning_open']
        elif time(9, 30) < time_only <= time(11, 30):
            return self.time_of_day_factors['morning_mid']
        elif time(11, 30) < time_only <= time(12, 0):
            return self.time_of_day_factors['morning_close']
        
        # Afternoon session
        elif time(13, 30) <= time_only <= time(14, 0):
            return self.time_of_day_factors['afternoon_open']
        elif time(14, 0) < time_only <= time(13, 30):  # Note: TSE closes at 13:30
            return self.time_of_day_factors['afternoon_close']
        
        # Outside trading hours
        else:
            return 1.5  # Higher spread for illiquid periods
    
    def analyze_historical_spreads(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """Analyze historical spread patterns."""
        if self.temporal_store is None:
            raise ValueError("TemporalStore required for historical analysis")
        
        # This would query historical bid-ask data
        # For now, return placeholder analysis
        return {
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'avg_spread_bps': 12.5,
            'spread_volatility': 8.2,
            'time_of_day_pattern': self.time_of_day_factors,
            'min_spread_bps': 2.1,
            'max_spread_bps': 45.8
        }


class MarketImpactModel:
    """
    Taiwan market impact model.
    
    Models temporary and permanent market impact based on order size
    relative to average daily volume, with Taiwan market characteristics.
    """
    
    def __init__(self):
        # Taiwan market calibrated parameters
        self.temp_impact_coeff = 0.4     # Temporary impact coefficient
        self.perm_impact_coeff = 0.25    # Permanent impact coefficient  
        self.size_exponent = 0.6         # Non-linear size relationship
        self.volatility_multiplier = 1.5 # Volatility scaling
        self.decay_half_life = 20.0      # Temporary impact decay (minutes)
        
        # Taiwan market liquidity thresholds
        self.liquidity_thresholds = {
            'very_liquid': 0.01,    # < 1% of ADV
            'liquid': 0.05,         # 1-5% of ADV
            'illiquid': 0.20,       # 5-20% of ADV
            'very_illiquid': 1.0    # > 20% of ADV
        }
        
        logger.info("Taiwan market impact model initialized")
    
    def calculate_impact(
        self,
        order_size: float,
        avg_daily_volume: float,
        volatility: float,
        price: float,
        direction: int = 1  # 1 for buy, -1 for sell
    ) -> MarketImpactData:
        """
        Calculate market impact for an order.
        
        Args:
            order_size: Order size in shares
            avg_daily_volume: Average daily volume in shares
            volatility: Historical volatility (annualized)
            price: Stock price
            direction: Order direction (1=buy, -1=sell)
            
        Returns:
            Market impact analysis
        """
        # Calculate participation rate
        participation_rate = abs(order_size) / avg_daily_volume
        
        # Temporary impact (mean-reverting)
        temp_impact_bps = (
            self.temp_impact_coeff * 
            (participation_rate ** self.size_exponent) * 
            volatility * 
            self.volatility_multiplier * 
            10000
        )
        
        # Permanent impact (information-driven)
        perm_impact_bps = (
            self.perm_impact_coeff * 
            (participation_rate ** 0.5) * 
            volatility * 
            10000
        )
        
        # Apply direction
        temp_impact_bps *= direction
        perm_impact_bps *= direction
        
        total_impact_bps = abs(temp_impact_bps) + abs(perm_impact_bps)
        
        return MarketImpactData(
            symbol="",  # To be filled by caller
            order_size=order_size,
            avg_daily_volume=avg_daily_volume,
            participation_rate=participation_rate,
            temporary_impact_bps=temp_impact_bps,
            permanent_impact_bps=perm_impact_bps,
            total_impact_bps=total_impact_bps,
            decay_half_life_minutes=self.decay_half_life
        )
    
    def calculate_impact_decay(
        self,
        initial_impact_bps: float,
        minutes_elapsed: float
    ) -> float:
        """
        Calculate temporary impact decay over time.
        
        Args:
            initial_impact_bps: Initial temporary impact in bps
            minutes_elapsed: Time elapsed since order
            
        Returns:
            Remaining impact in bps
        """
        decay_factor = 0.5 ** (minutes_elapsed / self.decay_half_life)
        return initial_impact_bps * decay_factor
    
    def get_liquidity_category(self, participation_rate: float) -> str:
        """Categorize order by liquidity impact."""
        if participation_rate <= self.liquidity_thresholds['very_liquid']:
            return 'very_liquid'
        elif participation_rate <= self.liquidity_thresholds['liquid']:
            return 'liquid'
        elif participation_rate <= self.liquidity_thresholds['illiquid']:
            return 'illiquid'
        else:
            return 'very_illiquid'
    
    def estimate_execution_time(
        self,
        order_size: float,
        avg_daily_volume: float,
        max_participation_rate: float = 0.1
    ) -> float:
        """
        Estimate execution time to stay within participation rate limits.
        
        Args:
            order_size: Total order size
            avg_daily_volume: Average daily volume
            max_participation_rate: Maximum allowed participation rate
            
        Returns:
            Estimated execution time in minutes
        """
        if order_size <= 0 or avg_daily_volume <= 0:
            return 0.0
        
        max_order_per_interval = avg_daily_volume * max_participation_rate
        
        if order_size <= max_order_per_interval:
            return 1.0  # Can execute immediately
        
        # Estimate based on trading session length (4 hours = 240 minutes)
        trading_minutes = 240
        required_intervals = order_size / max_order_per_interval
        
        return min(required_intervals, trading_minutes)


class TaiwanMarketStructure:
    """
    Comprehensive Taiwan market microstructure model.
    
    Integrates tick size, spread, and market impact models to provide
    complete market structure analysis for Taiwan stocks.
    """
    
    def __init__(self, temporal_store: Optional[TemporalStore] = None):
        self.temporal_store = temporal_store
        self.tick_model = TaiwanTickSizeModel()
        self.spread_model = BidAskSpreadModel(temporal_store)
        self.impact_model = MarketImpactModel()
        
        # Taiwan market characteristics
        self.trading_sessions = {
            'morning': (time(9, 0), time(12, 0)),
            'afternoon': (time(13, 30), time(13, 30))  # Note: TSE typically closes at 13:30
        }
        
        self.settlement_cycle = 2  # T+2 settlement
        
        logger.info("Taiwan market structure model initialized")
    
    def get_current_session(self, timestamp: Optional[datetime] = None) -> TradingSession:
        """Determine current trading session."""
        if timestamp is None:
            timestamp = datetime.now()
        
        time_only = timestamp.time()
        
        if time(9, 0) <= time_only <= time(12, 0):
            return TradingSession.MORNING
        elif time(12, 0) < time_only < time(13, 30):
            return TradingSession.LUNCH_BREAK
        elif time(13, 30) <= time_only <= time(13, 30):  # Special afternoon session if any
            return TradingSession.AFTERNOON
        else:
            return TradingSession.CLOSED
    
    def analyze_trading_costs(
        self,
        symbol: str,
        order_size: float,
        price: float,
        avg_daily_volume: float,
        volatility: float,
        timestamp: Optional[datetime] = None,
        exchange: str = "TSE"
    ) -> Dict[str, Any]:
        """
        Comprehensive trading cost analysis.
        
        Args:
            symbol: Stock symbol
            order_size: Order size in shares
            price: Stock price
            avg_daily_volume: Average daily volume
            volatility: Historical volatility
            timestamp: Execution timestamp
            exchange: Exchange code
            
        Returns:
            Complete cost analysis
        """
        # Tick size analysis
        tick_size = self.tick_model.get_tick_size(price, exchange)
        min_spread_bps = self.tick_model.get_min_spread_bps(price, exchange)
        
        # Spread analysis
        volume_ratio = 1.0  # Default, could be calculated from current volume
        predicted_spread = self.spread_model.predict_spread(
            symbol, price, volatility, volume_ratio, timestamp, exchange
        )
        predicted_spread_bps = self.spread_model.predict_spread_bps(
            symbol, price, volatility, volume_ratio, timestamp, exchange
        )
        
        # Market impact analysis
        direction = 1 if order_size > 0 else -1
        impact_data = self.impact_model.calculate_impact(
            abs(order_size), avg_daily_volume, volatility, price, direction
        )
        
        # Trading session analysis
        current_session = self.get_current_session(timestamp)
        
        # Liquidity analysis
        liquidity_category = self.impact_model.get_liquidity_category(
            impact_data.participation_rate
        )
        
        # Execution time estimate
        execution_time = self.impact_model.estimate_execution_time(
            abs(order_size), avg_daily_volume
        )
        
        return {
            'symbol': symbol,
            'analysis_timestamp': timestamp or datetime.now(),
            'order_size': order_size,
            'price': price,
            'exchange': exchange,
            
            # Tick size analysis
            'tick_size': tick_size,
            'min_spread_bps': min_spread_bps,
            
            # Spread analysis
            'predicted_spread_twd': predicted_spread,
            'predicted_spread_bps': predicted_spread_bps,
            
            # Market impact
            'participation_rate': impact_data.participation_rate,
            'temporary_impact_bps': impact_data.temporary_impact_bps,
            'permanent_impact_bps': impact_data.permanent_impact_bps,
            'total_impact_bps': impact_data.total_impact_bps,
            'impact_decay_half_life': impact_data.decay_half_life_minutes,
            
            # Trading context
            'trading_session': current_session.value,
            'liquidity_category': liquidity_category,
            'estimated_execution_time_minutes': execution_time,
            
            # Cost summary
            'total_microstructure_cost_bps': predicted_spread_bps / 2 + impact_data.total_impact_bps,
            'settlement_date': self._calculate_settlement_date(timestamp)
        }
    
    def _calculate_settlement_date(self, trade_date: Optional[datetime]) -> date:
        """Calculate T+2 settlement date."""
        if trade_date is None:
            trade_date = datetime.now()
        
        settlement_date = trade_date.date() + timedelta(days=self.settlement_cycle)
        
        # Skip weekends (simplified - should use trading calendar)
        while settlement_date.weekday() >= 5:
            settlement_date += timedelta(days=1)
        
        return settlement_date
    
    def benchmark_against_market(
        self,
        symbol: str,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Benchmark trading costs against market averages."""
        # This would typically compare against historical averages
        # For now, provide simple benchmarking logic
        
        spread_bps = analysis_results['predicted_spread_bps']
        impact_bps = analysis_results['total_impact_bps']
        
        # Taiwan market benchmarks (approximate)
        benchmarks = {
            'avg_spread_bps': 15.0,
            'avg_impact_bps': 8.0,
            'liquid_spread_bps': 8.0,
            'liquid_impact_bps': 4.0
        }
        
        return {
            'symbol': symbol,
            'spread_vs_avg': spread_bps / benchmarks['avg_spread_bps'],
            'impact_vs_avg': impact_bps / benchmarks['avg_impact_bps'],
            'spread_vs_liquid': spread_bps / benchmarks['liquid_spread_bps'],
            'impact_vs_liquid': impact_bps / benchmarks['liquid_impact_bps'],
            'overall_cost_percentile': min(100, max(0, 
                50 + 30 * (spread_bps + impact_bps - 23) / 15  # Rough percentile estimate
            )),
            'benchmarks': benchmarks
        }


# Example usage and testing
if __name__ == "__main__":
    print("Taiwan Market Microstructure Models Demo")
    
    # Initialize models
    market_structure = TaiwanMarketStructure()
    
    # Sample analysis
    analysis = market_structure.analyze_trading_costs(
        symbol="2330.TW",  # TSMC
        order_size=1000,
        price=500.0,
        avg_daily_volume=50000,
        volatility=0.25,
        timestamp=datetime.now(),
        exchange="TSE"
    )
    
    print(f"\nMarket Structure Analysis for {analysis['symbol']}:")
    print(f"Tick Size: NT${analysis['tick_size']}")
    print(f"Predicted Spread: {analysis['predicted_spread_bps']:.2f} bps")
    print(f"Market Impact: {analysis['total_impact_bps']:.2f} bps")
    print(f"Total Microstructure Cost: {analysis['total_microstructure_cost_bps']:.2f} bps")
    print(f"Liquidity Category: {analysis['liquidity_category']}")
    print(f"Estimated Execution Time: {analysis['estimated_execution_time_minutes']:.1f} minutes")