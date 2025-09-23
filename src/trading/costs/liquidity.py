"""
Liquidity analysis and capacity modeling for Taiwan equity markets.

This module provides comprehensive liquidity analysis including Average Daily Volume
(ADV) calculations, liquidity-adjusted volume estimates, capacity constraints, and
real-time liquidity monitoring for Taiwan Stock Exchange and Taipei Exchange.

Key Features:
- Multi-timeframe ADV calculation with adjustments
- Liquidity-adjusted volume (LAV) estimation
- Dynamic capacity constraint modeling
- Real-time liquidity monitoring and alerts
- Cross-sectional liquidity analysis
- Taiwan market microstructure integration
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LiquidityTier(Enum):
    """Liquidity classification tiers."""
    VERY_LIQUID = "very_liquid"    # Top tier (ADV > 1M shares)
    LIQUID = "liquid"              # Good liquidity (ADV 100K-1M)
    MODERATE = "moderate"          # Moderate liquidity (ADV 10K-100K)
    ILLIQUID = "illiquid"          # Low liquidity (ADV 1K-10K)
    VERY_ILLIQUID = "very_illiquid"  # Very low liquidity (ADV < 1K)


class LiquidityRegime(Enum):
    """Market liquidity regimes."""
    NORMAL = "normal"
    DRY_UP = "dry_up"       # Liquidity drying up
    CRISIS = "crisis"       # Liquidity crisis
    ABUNDANT = "abundant"   # High liquidity


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics for a security."""
    symbol: str
    date: date
    
    # Volume metrics
    adv_20d: float          # 20-day average daily volume
    adv_60d: float          # 60-day average daily volume
    lav_20d: float          # 20-day liquidity-adjusted volume
    current_volume: float   # Today's volume
    volume_ratio: float     # Current volume / ADV ratio
    
    # Turnover metrics
    shares_outstanding: Optional[float] = None
    daily_turnover_rate: Optional[float] = None
    avg_turnover_rate: Optional[float] = None
    
    # Capacity metrics
    daily_capacity_shares: float = 0.0
    daily_capacity_twd: float = 0.0
    strategy_capacity_shares: float = 0.0
    strategy_capacity_twd: float = 0.0
    
    # Classification
    liquidity_tier: LiquidityTier = LiquidityTier.MODERATE
    liquidity_score: float = 0.5  # 0-1 scale
    
    # Market microstructure
    avg_trade_size: Optional[float] = None
    trade_frequency: Optional[float] = None  # Trades per minute
    zero_volume_days_pct: float = 0.0  # % of days with zero volume
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.shares_outstanding and self.current_volume > 0:
            self.daily_turnover_rate = self.current_volume / self.shares_outstanding
        
        if self.shares_outstanding and self.adv_20d > 0:
            self.avg_turnover_rate = self.adv_20d / self.shares_outstanding


@dataclass
class CapacityConstraints:
    """Capacity constraints for trading strategies."""
    symbol: str
    
    # Volume-based constraints
    max_participation_rate: float = 0.10      # Max 10% of ADV
    max_daily_volume_shares: float = 0.0
    max_daily_volume_twd: float = 0.0
    
    # Time-based constraints
    min_execution_days: float = 1.0          # Minimum days to execute
    max_execution_days: float = 5.0          # Maximum days to execute
    
    # Market impact constraints
    max_impact_bps: float = 50.0             # Maximum acceptable impact
    impact_budget_bps: float = 30.0          # Impact budget
    
    # Position constraints
    max_position_shares: float = 0.0
    max_position_twd: float = 0.0
    max_position_pct_outstanding: float = 0.05  # Max 5% of shares outstanding
    
    # Risk constraints
    concentration_limit_pct: float = 0.20    # Max 20% of portfolio
    liquidity_buffer_factor: float = 0.8     # Use 80% of calculated capacity


@dataclass
class LiquidityAlert:
    """Liquidity alert for risk monitoring."""
    symbol: str
    timestamp: datetime
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    current_value: float
    threshold_value: float
    recommended_action: str


class LiquidityAnalyzer:
    """
    Comprehensive liquidity analyzer for Taiwan equity markets.
    
    Provides real-time liquidity analysis, capacity modeling, and
    risk monitoring for trading strategies.
    """
    
    def __init__(
        self,
        adv_window: int = 20,
        lav_threshold: float = 0.3,
        capacity_threshold: float = 0.10,
        min_liquidity_score: float = 0.3
    ):
        """
        Initialize liquidity analyzer.
        
        Args:
            adv_window: Window for ADV calculation (days)
            lav_threshold: Threshold for LAV calculation (fraction of ADV)
            capacity_threshold: Maximum participation rate
            min_liquidity_score: Minimum acceptable liquidity score
        """
        self.adv_window = adv_window
        self.lav_threshold = lav_threshold
        self.capacity_threshold = capacity_threshold
        self.min_liquidity_score = min_liquidity_score
        self.logger = logging.getLogger(f"{__name__}.LiquidityAnalyzer")
        
        # Cache for performance
        self._liquidity_cache = {}
        self._capacity_cache = {}
    
    def calculate_liquidity_metrics(
        self,
        symbol: str,
        volume_data: pd.Series,
        price_data: pd.Series,
        market_data: Optional[Dict[str, Any]] = None
    ) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity metrics.
        
        Args:
            symbol: Stock symbol
            volume_data: Time series of daily volumes
            price_data: Time series of daily prices
            market_data: Additional market context
            
        Returns:
            LiquidityMetrics with comprehensive analysis
        """
        # Ensure data is aligned and sorted
        volume_data = volume_data.sort_index().dropna()
        price_data = price_data.sort_index().dropna()
        
        if len(volume_data) < self.adv_window:
            raise ValueError(f"Insufficient data: need at least {self.adv_window} days")
        
        # Calculate ADV for different windows
        adv_20d = volume_data.tail(20).mean()
        adv_60d = volume_data.tail(60).mean() if len(volume_data) >= 60 else adv_20d
        
        # Calculate LAV (excluding low-volume days)
        lav_threshold_volume = adv_20d * self.lav_threshold
        high_volume_days = volume_data[volume_data >= lav_threshold_volume]
        lav_20d = high_volume_days.tail(20).mean() if len(high_volume_days) > 0 else adv_20d
        
        # Current metrics
        current_volume = volume_data.iloc[-1]
        volume_ratio = current_volume / adv_20d if adv_20d > 0 else 0.0
        
        # Zero volume analysis
        zero_volume_days = (volume_data.tail(20) == 0).sum()
        zero_volume_days_pct = zero_volume_days / min(20, len(volume_data))
        
        # Estimate market microstructure metrics
        avg_trade_size = self._estimate_avg_trade_size(symbol, volume_data, market_data)
        trade_frequency = self._estimate_trade_frequency(symbol, volume_data, market_data)
        
        # Calculate liquidity score and tier
        liquidity_score = self._calculate_liquidity_score(
            adv_20d, lav_20d, zero_volume_days_pct, volume_ratio
        )
        liquidity_tier = self._classify_liquidity_tier(adv_20d, liquidity_score)
        
        # Calculate capacity metrics
        current_price = price_data.iloc[-1]
        daily_capacity_shares = lav_20d * self.capacity_threshold
        daily_capacity_twd = daily_capacity_shares * current_price
        
        # Get shares outstanding if available
        shares_outstanding = market_data.get('shares_outstanding') if market_data else None
        
        return LiquidityMetrics(
            symbol=symbol,
            date=volume_data.index[-1].date() if hasattr(volume_data.index[-1], 'date') else date.today(),
            adv_20d=adv_20d,
            adv_60d=adv_60d,
            lav_20d=lav_20d,
            current_volume=current_volume,
            volume_ratio=volume_ratio,
            shares_outstanding=shares_outstanding,
            daily_capacity_shares=daily_capacity_shares,
            daily_capacity_twd=daily_capacity_twd,
            liquidity_tier=liquidity_tier,
            liquidity_score=liquidity_score,
            avg_trade_size=avg_trade_size,
            trade_frequency=trade_frequency,
            zero_volume_days_pct=zero_volume_days_pct
        )
    
    def calculate_capacity_constraints(
        self,
        symbol: str,
        liquidity_metrics: LiquidityMetrics,
        strategy_horizon_days: int = 1,
        max_impact_bps: float = 50.0,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> CapacityConstraints:
        """
        Calculate capacity constraints for a trading strategy.
        
        Args:
            symbol: Stock symbol
            liquidity_metrics: Pre-calculated liquidity metrics
            strategy_horizon_days: Strategy execution timeframe
            max_impact_bps: Maximum acceptable market impact
            custom_constraints: Override default constraints
            
        Returns:
            CapacityConstraints for the strategy
        """
        # Base participation rate (adjusted for liquidity tier)
        tier_adjustments = {
            LiquidityTier.VERY_LIQUID: 1.2,    # Can trade more aggressively
            LiquidityTier.LIQUID: 1.0,
            LiquidityTier.MODERATE: 0.8,
            LiquidityTier.ILLIQUID: 0.5,
            LiquidityTier.VERY_ILLIQUID: 0.3
        }
        
        adjusted_participation = (
            self.capacity_threshold * 
            tier_adjustments.get(liquidity_metrics.liquidity_tier, 1.0)
        )
        
        # Calculate volume constraints
        max_daily_volume_shares = liquidity_metrics.lav_20d * adjusted_participation
        
        # Strategy capacity over multiple days
        strategy_capacity_shares = max_daily_volume_shares * strategy_horizon_days
        
        # Apply liquidity buffer
        buffer_factor = 0.8  # Use 80% of calculated capacity
        max_daily_volume_shares *= buffer_factor
        strategy_capacity_shares *= buffer_factor
        
        # Convert to TWD values
        current_price = liquidity_metrics.daily_capacity_twd / liquidity_metrics.daily_capacity_shares
        max_daily_volume_twd = max_daily_volume_shares * current_price
        strategy_capacity_twd = strategy_capacity_shares * current_price
        
        # Position constraints based on shares outstanding
        max_position_shares = float('inf')  # Default: no limit
        if liquidity_metrics.shares_outstanding:
            max_position_shares = liquidity_metrics.shares_outstanding * 0.05  # 5% limit
        
        # Execution time constraints
        min_execution_days = max(1.0, strategy_capacity_shares / max_daily_volume_shares)
        max_execution_days = min(10.0, min_execution_days * 3)  # Up to 3x minimum time
        
        constraints = CapacityConstraints(
            symbol=symbol,
            max_participation_rate=adjusted_participation,
            max_daily_volume_shares=max_daily_volume_shares,
            max_daily_volume_twd=max_daily_volume_twd,
            min_execution_days=min_execution_days,
            max_execution_days=max_execution_days,
            max_impact_bps=max_impact_bps,
            impact_budget_bps=max_impact_bps * 0.6,  # 60% of max impact as budget
            max_position_shares=max_position_shares,
            max_position_twd=max_position_shares * current_price,
            liquidity_buffer_factor=buffer_factor
        )
        
        # Apply custom constraint overrides
        if custom_constraints:
            for key, value in custom_constraints.items():
                if hasattr(constraints, key):
                    setattr(constraints, key, value)
        
        # Update strategy capacity in metrics
        liquidity_metrics.strategy_capacity_shares = strategy_capacity_shares
        liquidity_metrics.strategy_capacity_twd = strategy_capacity_twd
        
        return constraints
    
    def monitor_liquidity_alerts(
        self,
        symbols: List[str],
        current_positions: Dict[str, float],
        liquidity_metrics: Dict[str, LiquidityMetrics],
        capacity_constraints: Dict[str, CapacityConstraints]
    ) -> List[LiquidityAlert]:
        """
        Monitor for liquidity alerts and risk conditions.
        
        Args:
            symbols: List of symbols to monitor
            current_positions: Current position sizes
            liquidity_metrics: Current liquidity metrics
            capacity_constraints: Current capacity constraints
            
        Returns:
            List of liquidity alerts
        """
        alerts = []
        timestamp = datetime.now()
        
        for symbol in symbols:
            metrics = liquidity_metrics.get(symbol)
            constraints = capacity_constraints.get(symbol)
            position = current_positions.get(symbol, 0.0)
            
            if not metrics or not constraints:
                continue
            
            # Check liquidity score
            if metrics.liquidity_score < self.min_liquidity_score:
                alerts.append(LiquidityAlert(
                    symbol=symbol,
                    timestamp=timestamp,
                    alert_type="low_liquidity_score",
                    severity="high",
                    message=f"Liquidity score {metrics.liquidity_score:.2f} below threshold",
                    current_value=metrics.liquidity_score,
                    threshold_value=self.min_liquidity_score,
                    recommended_action="Reduce position size or extend execution time"
                ))
            
            # Check position vs capacity
            if abs(position) > constraints.max_position_shares:
                alerts.append(LiquidityAlert(
                    symbol=symbol,
                    timestamp=timestamp,
                    alert_type="position_exceeds_capacity",
                    severity="critical",
                    message=f"Position {position:,.0f} exceeds capacity {constraints.max_position_shares:,.0f}",
                    current_value=abs(position),
                    threshold_value=constraints.max_position_shares,
                    recommended_action="Reduce position immediately"
                ))
            
            # Check volume ratio anomaly
            if metrics.volume_ratio < 0.1:  # Very low volume day
                alerts.append(LiquidityAlert(
                    symbol=symbol,
                    timestamp=timestamp,
                    alert_type="low_volume_day",
                    severity="medium",
                    message=f"Volume ratio {metrics.volume_ratio:.2f} indicates low trading activity",
                    current_value=metrics.volume_ratio,
                    threshold_value=0.1,
                    recommended_action="Delay non-urgent trades"
                ))
            
            # Check for liquidity tier degradation
            if metrics.liquidity_tier in [LiquidityTier.ILLIQUID, LiquidityTier.VERY_ILLIQUID]:
                alerts.append(LiquidityAlert(
                    symbol=symbol,
                    timestamp=timestamp,
                    alert_type="illiquid_stock",
                    severity="high",
                    message=f"Stock classified as {metrics.liquidity_tier.value}",
                    current_value=metrics.adv_20d,
                    threshold_value=10000,  # 10K ADV threshold
                    recommended_action="Use smaller order sizes and longer execution times"
                ))
        
        return alerts
    
    def optimize_execution_schedule(
        self,
        symbol: str,
        target_shares: float,
        liquidity_metrics: LiquidityMetrics,
        constraints: CapacityConstraints,
        max_execution_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize execution schedule based on liquidity constraints.
        
        Args:
            symbol: Stock symbol
            target_shares: Target number of shares to trade
            liquidity_metrics: Current liquidity metrics
            constraints: Capacity constraints
            max_execution_days: Maximum days for execution
            
        Returns:
            Optimized execution schedule
        """
        if max_execution_days is None:
            max_execution_days = int(constraints.max_execution_days)
        
        # Calculate daily volumes to stay within constraints
        daily_volume_limit = constraints.max_daily_volume_shares
        abs_target = abs(target_shares)
        
        # Minimum execution days needed
        min_days_needed = np.ceil(abs_target / daily_volume_limit)
        execution_days = min(max(min_days_needed, 1), max_execution_days)
        
        # Calculate daily schedule
        daily_target = abs_target / execution_days
        
        # Adjust for market patterns (front-load slightly)
        schedule = []
        remaining_shares = abs_target
        
        for day in range(int(execution_days)):
            if day == execution_days - 1:  # Last day
                daily_shares = remaining_shares
            else:
                # Front-load slightly (105% of average for first few days)
                if day < 2:
                    daily_shares = min(daily_target * 1.05, daily_volume_limit, remaining_shares)
                else:
                    daily_shares = min(daily_target, daily_volume_limit, remaining_shares)
            
            schedule.append({
                'day': day + 1,
                'target_shares': daily_shares,
                'participation_rate': daily_shares / liquidity_metrics.lav_20d,
                'estimated_impact_bps': self._estimate_daily_impact(
                    daily_shares, liquidity_metrics
                )
            })
            
            remaining_shares -= daily_shares
            if remaining_shares <= 0:
                break
        
        # Calculate execution statistics
        total_estimated_impact = sum(day['estimated_impact_bps'] for day in schedule)
        avg_daily_participation = np.mean([day['participation_rate'] for day in schedule])
        
        return {
            'symbol': symbol,
            'target_shares': target_shares,
            'execution_days': len(schedule),
            'daily_schedule': schedule,
            'total_estimated_impact_bps': total_estimated_impact,
            'avg_daily_participation_rate': avg_daily_participation,
            'max_daily_participation_rate': max([day['participation_rate'] for day in schedule]),
            'feasible': all(day['participation_rate'] <= constraints.max_participation_rate * 1.1 for day in schedule)
        }
    
    def _calculate_liquidity_score(
        self,
        adv: float,
        lav: float,
        zero_volume_pct: float,
        volume_ratio: float
    ) -> float:
        """Calculate composite liquidity score (0-1)."""
        
        # Volume component (0-0.4)
        volume_score = min(0.4, np.log10(max(adv, 1)) / 6)  # Log scale up to 1M shares
        
        # LAV adjustment component (0-0.2)
        lav_ratio = lav / adv if adv > 0 else 0
        lav_score = min(0.2, lav_ratio * 0.4)
        
        # Consistency component (0-0.2)
        consistency_score = min(0.2, (1 - zero_volume_pct) * 0.2)
        
        # Current activity component (0-0.2)
        activity_score = min(0.2, min(volume_ratio, 2.0) * 0.1)  # Cap at 2x normal volume
        
        return volume_score + lav_score + consistency_score + activity_score
    
    def _classify_liquidity_tier(self, adv: float, liquidity_score: float) -> LiquidityTier:
        """Classify stock into liquidity tier."""
        
        if adv >= 1_000_000 and liquidity_score >= 0.8:
            return LiquidityTier.VERY_LIQUID
        elif adv >= 100_000 and liquidity_score >= 0.6:
            return LiquidityTier.LIQUID
        elif adv >= 10_000 and liquidity_score >= 0.4:
            return LiquidityTier.MODERATE
        elif adv >= 1_000:
            return LiquidityTier.ILLIQUID
        else:
            return LiquidityTier.VERY_ILLIQUID
    
    def _estimate_avg_trade_size(
        self,
        symbol: str,
        volume_data: pd.Series,
        market_data: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        """Estimate average trade size."""
        if market_data and 'avg_trade_size' in market_data:
            return market_data['avg_trade_size']
        
        # Heuristic: estimate based on volume patterns
        avg_volume = volume_data.tail(20).mean()
        
        # Taiwan market typical trade sizes
        if avg_volume > 500_000:
            return 5_000  # Large cap
        elif avg_volume > 50_000:
            return 2_000  # Mid cap
        else:
            return 500   # Small cap
    
    def _estimate_trade_frequency(
        self,
        symbol: str,
        volume_data: pd.Series,
        market_data: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        """Estimate trade frequency (trades per minute)."""
        if market_data and 'trade_frequency' in market_data:
            return market_data['trade_frequency']
        
        # Heuristic: estimate based on volume and trade size
        avg_volume = volume_data.tail(20).mean()
        avg_trade_size = self._estimate_avg_trade_size(symbol, volume_data, market_data) or 1000
        
        # Estimate trades per day, then per minute (390 minutes in Taiwan trading day)
        trades_per_day = avg_volume / avg_trade_size
        trades_per_minute = trades_per_day / 390
        
        return max(0.1, trades_per_minute)  # At least 1 trade per 10 minutes
    
    def _estimate_daily_impact(self, daily_shares: float, metrics: LiquidityMetrics) -> float:
        """Estimate market impact for daily volume."""
        participation_rate = daily_shares / metrics.lav_20d if metrics.lav_20d > 0 else 0
        
        # Simple square root model for quick estimation
        base_impact_bps = 20  # Base impact for Taiwan market
        impact_bps = base_impact_bps * np.sqrt(participation_rate)
        
        # Adjust for liquidity tier
        tier_multipliers = {
            LiquidityTier.VERY_LIQUID: 0.6,
            LiquidityTier.LIQUID: 0.8,
            LiquidityTier.MODERATE: 1.0,
            LiquidityTier.ILLIQUID: 1.5,
            LiquidityTier.VERY_ILLIQUID: 2.5
        }
        
        multiplier = tier_multipliers.get(metrics.liquidity_tier, 1.0)
        return impact_bps * multiplier


# Factory functions
def create_liquidity_analyzer(
    conservative: bool = False,
    custom_params: Optional[Dict[str, Any]] = None
) -> LiquidityAnalyzer:
    """
    Create liquidity analyzer with Taiwan market defaults.
    
    Args:
        conservative: Use conservative parameters for capacity
        custom_params: Override specific parameters
        
    Returns:
        Configured LiquidityAnalyzer
    """
    if conservative:
        params = {
            'adv_window': 30,           # Longer window
            'capacity_threshold': 0.05,  # Lower participation rate
            'min_liquidity_score': 0.4   # Higher minimum score
        }
    else:
        params = {
            'adv_window': 20,
            'capacity_threshold': 0.10,
            'min_liquidity_score': 0.3
        }
    
    # Override with custom parameters
    if custom_params:
        params.update(custom_params)
    
    return LiquidityAnalyzer(**params)


# Example usage and testing
if __name__ == "__main__":
    print("Taiwan Liquidity Analysis Demo")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    np.random.seed(42)
    
    # Simulate volume data with realistic patterns
    base_volume = 100_000
    volumes = base_volume * (1 + np.random.normal(0, 0.3, 60))
    volumes = np.maximum(volumes, 0)  # No negative volumes
    
    volume_series = pd.Series(volumes, index=dates)
    price_series = pd.Series(100 + np.cumsum(np.random.normal(0, 2, 60)), index=dates)
    
    # Create analyzer
    analyzer = create_liquidity_analyzer()
    
    # Calculate metrics
    metrics = analyzer.calculate_liquidity_metrics(
        symbol="2330.TW",
        volume_data=volume_series,
        price_data=price_series,
        market_data={'shares_outstanding': 25_930_000_000}  # TSMC shares outstanding
    )
    
    print(f"\nLiquidity Analysis for {metrics.symbol}:")
    print(f"20-day ADV: {metrics.adv_20d:,.0f} shares")
    print(f"LAV: {metrics.lav_20d:,.0f} shares")
    print(f"Liquidity Tier: {metrics.liquidity_tier.value}")
    print(f"Liquidity Score: {metrics.liquidity_score:.2f}")
    print(f"Daily Capacity: {metrics.daily_capacity_shares:,.0f} shares (NT${metrics.daily_capacity_twd:,.0f})")
    
    # Calculate constraints
    constraints = analyzer.calculate_capacity_constraints(
        symbol="2330.TW",
        liquidity_metrics=metrics,
        strategy_horizon_days=3
    )
    
    print(f"\nCapacity Constraints:")
    print(f"Max Daily Volume: {constraints.max_daily_volume_shares:,.0f} shares")
    print(f"Max Participation Rate: {constraints.max_participation_rate:.2%}")
    print(f"Strategy Capacity (3 days): {metrics.strategy_capacity_shares:,.0f} shares")
    
    # Test execution schedule
    target_position = 500_000  # 500K shares
    schedule = analyzer.optimize_execution_schedule(
        symbol="2330.TW",
        target_shares=target_position,
        liquidity_metrics=metrics,
        constraints=constraints
    )
    
    print(f"\nExecution Schedule for {target_position:,} shares:")
    print(f"Execution Days: {schedule['execution_days']}")
    print(f"Feasible: {schedule['feasible']}")
    print(f"Total Estimated Impact: {schedule['total_estimated_impact_bps']:.1f} bps")
    
    for day_info in schedule['daily_schedule']:
        print(f"  Day {day_info['day']}: {day_info['target_shares']:,.0f} shares "
              f"({day_info['participation_rate']:.1%} participation, "
              f"{day_info['estimated_impact_bps']:.1f} bps impact)")