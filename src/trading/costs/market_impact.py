"""
Market impact modeling for Taiwan equity markets.

This module implements sophisticated market impact models that capture both temporary
and permanent price effects of trades, specifically calibrated for Taiwan Stock
Exchange and Taipei Exchange characteristics.

Key Features:
- Temporary and permanent market impact decomposition
- Non-linear impact functions based on participation rates
- Taiwan market microstructure calibration
- Intraday impact decay modeling
- Cross-sectional impact analysis
- Integration with real-time market data
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


class ImpactComponent(Enum):
    """Types of market impact components."""
    TEMPORARY = "temporary"
    PERMANENT = "permanent"
    TOTAL = "total"


class ImpactRegime(Enum):
    """Market impact regimes based on conditions."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    STRESSED = "stressed"
    ILLIQUID = "illiquid"


@dataclass
class MarketImpactParameters:
    """Market impact model parameters for Taiwan market."""
    
    # Temporary impact parameters
    temp_impact_coeff: float = 0.35      # Base temporary impact coefficient
    temp_decay_rate: float = 0.6         # Decay rate (higher = faster decay)
    temp_size_exponent: float = 0.65     # Size penalty exponent
    temp_vol_multiplier: float = 1.8     # Volatility amplification
    
    # Permanent impact parameters  
    perm_impact_coeff: float = 0.20      # Base permanent impact coefficient
    perm_size_exponent: float = 0.5      # Square root scaling
    perm_vol_multiplier: float = 1.2     # Volatility amplification
    
    # Liquidity adjustments
    liquidity_penalty_threshold: float = 0.10  # 10% of ADV threshold
    liquidity_penalty_factor: float = 1.5      # Penalty multiplier
    
    # Taiwan-specific adjustments
    session_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'morning_open': 1.3,     # Higher impact at open
        'morning_mid': 1.0,      # Normal morning trading
        'afternoon_open': 1.2,   # Lunch break resumption
        'afternoon_mid': 1.0,    # Normal afternoon trading
        'close': 1.4             # Higher impact at close
    })
    
    # Circuit breaker adjustments
    circuit_breaker_multiplier: float = 2.0  # Impact during high volatility
    
    # Market cap tier adjustments
    market_cap_adjustments: Dict[str, float] = field(default_factory=lambda: {
        'large_cap': 0.8,        # Lower impact for large caps
        'mid_cap': 1.0,          # Baseline
        'small_cap': 1.4         # Higher impact for small caps
    })


@dataclass
class ImpactCalculationResult:
    """Result of market impact calculation."""
    symbol: str
    timestamp: datetime
    order_size: float
    participation_rate: float
    
    # Impact components (in basis points)
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    
    # Impact values (in TWD)
    temporary_impact_twd: float
    permanent_impact_twd: float
    total_impact_twd: float
    
    # Decay characteristics
    decay_half_life_minutes: float
    
    # Market context
    regime: ImpactRegime
    session: str
    market_cap_tier: str
    
    # Confidence metrics
    model_confidence: float = 0.95  # Model confidence level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'order_size': self.order_size,
            'participation_rate': self.participation_rate,
            'temporary_impact_bps': self.temporary_impact_bps,
            'permanent_impact_bps': self.permanent_impact_bps,
            'total_impact_bps': self.total_impact_bps,
            'temporary_impact_twd': self.temporary_impact_twd,
            'permanent_impact_twd': self.permanent_impact_twd,
            'total_impact_twd': self.total_impact_twd,
            'decay_half_life_minutes': self.decay_half_life_minutes,
            'regime': self.regime.value,
            'session': self.session,
            'market_cap_tier': self.market_cap_tier,
            'model_confidence': self.model_confidence
        }


class BaseMarketImpactModel(ABC):
    """Abstract base class for market impact models."""
    
    def __init__(self, name: str, parameters: Optional[MarketImpactParameters] = None):
        self.name = name
        self.parameters = parameters or MarketImpactParameters()
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def calculate_impact(
        self,
        symbol: str,
        order_size: float,
        price: float,
        avg_daily_volume: float,
        volatility: float,
        timestamp: Optional[datetime] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ImpactCalculationResult:
        """Calculate market impact for a trade."""
        pass
    
    @abstractmethod
    def get_decay_function(self, impact_result: ImpactCalculationResult) -> Callable[[float], float]:
        """Get temporary impact decay function."""
        pass
    
    def _determine_regime(
        self, 
        volatility: float, 
        participation_rate: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ImpactRegime:
        """Determine market impact regime."""
        
        # High volatility regime
        if volatility > 0.4:  # 40% annual volatility
            return ImpactRegime.VOLATILE
        
        # Stressed market conditions
        if market_data and market_data.get('vix_equivalent', 0) > 30:
            return ImpactRegime.STRESSED
        
        # Illiquid conditions (high participation rate)
        if participation_rate > 0.15:  # > 15% of ADV
            return ImpactRegime.ILLIQUID
        
        return ImpactRegime.NORMAL
    
    def _get_session_type(self, timestamp: datetime) -> str:
        """Determine trading session type."""
        time_of_day = timestamp.time()
        
        # Taiwan market hours: 09:00-13:30 with 12:00-13:30 lunch break
        if time(9, 0) <= time_of_day <= time(9, 30):
            return 'morning_open'
        elif time(9, 30) <= time_of_day <= time(12, 0):
            return 'morning_mid'
        elif time(13, 30) <= time_of_day <= time(14, 0):
            return 'afternoon_open'
        elif time(14, 0) <= time_of_day <= time(13, 25):
            return 'afternoon_mid'
        elif time(13, 25) <= time_of_day <= time(13, 30):
            return 'close'
        else:
            return 'closed'
    
    def _estimate_market_cap_tier(self, symbol: str, price: float) -> str:
        """Estimate market cap tier based on symbol and price."""
        # Taiwan market cap tiers (simplified heuristic)
        # Large cap: Top 50 stocks (TSMC, etc.)
        # Mid cap: Next 150 stocks
        # Small cap: Remaining stocks
        
        large_cap_symbols = [
            '2330.TW', '2317.TW', '2454.TW', '2881.TW', '2882.TW',
            '6505.TW', '2412.TW', '3711.TW', '2891.TW', '2886.TW'
        ]
        
        if symbol in large_cap_symbols or price > 200:
            return 'large_cap'
        elif price > 50:
            return 'mid_cap'
        else:
            return 'small_cap'


class TaiwanMarketImpactModel(BaseMarketImpactModel):
    """
    Advanced market impact model calibrated for Taiwan equity markets.
    
    Implements the Almgren-Chriss framework with Taiwan-specific adjustments
    for temporary and permanent impact components.
    """
    
    def __init__(self, parameters: Optional[MarketImpactParameters] = None):
        super().__init__("TaiwanMarketImpactModel", parameters)
    
    def calculate_impact(
        self,
        symbol: str,
        order_size: float,
        price: float,
        avg_daily_volume: float,
        volatility: float,
        timestamp: Optional[datetime] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ImpactCalculationResult:
        """
        Calculate market impact using Taiwan-calibrated model.
        
        Args:
            symbol: Stock symbol (e.g., '2330.TW')
            order_size: Order size in shares (positive for buy, negative for sell)
            price: Current stock price in TWD
            avg_daily_volume: Average daily volume in shares
            volatility: Annualized volatility (e.g., 0.25 for 25%)
            timestamp: Trade timestamp
            market_data: Additional market context
            
        Returns:
            ImpactCalculationResult with detailed impact analysis
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate participation rate
        participation_rate = abs(order_size) / avg_daily_volume
        
        # Determine market regime and context
        regime = self._determine_regime(volatility, participation_rate, market_data)
        session = self._get_session_type(timestamp)
        market_cap_tier = self._estimate_market_cap_tier(symbol, price)
        
        # Calculate temporary impact
        temp_impact_bps = self._calculate_temporary_impact(
            participation_rate, volatility, regime, session, market_cap_tier
        )
        
        # Calculate permanent impact
        perm_impact_bps = self._calculate_permanent_impact(
            participation_rate, volatility, regime, market_cap_tier
        )
        
        # Calculate total impact
        total_impact_bps = temp_impact_bps + perm_impact_bps
        
        # Convert to TWD values
        trade_value = abs(order_size) * price
        temp_impact_twd = (temp_impact_bps / 10000) * trade_value
        perm_impact_twd = (perm_impact_bps / 10000) * trade_value
        total_impact_twd = temp_impact_twd + perm_impact_twd
        
        # Calculate decay characteristics
        decay_half_life = self._calculate_decay_half_life(
            participation_rate, volatility, regime
        )
        
        return ImpactCalculationResult(
            symbol=symbol,
            timestamp=timestamp,
            order_size=order_size,
            participation_rate=participation_rate,
            temporary_impact_bps=temp_impact_bps,
            permanent_impact_bps=perm_impact_bps,
            total_impact_bps=total_impact_bps,
            temporary_impact_twd=temp_impact_twd,
            permanent_impact_twd=perm_impact_twd,
            total_impact_twd=total_impact_twd,
            decay_half_life_minutes=decay_half_life,
            regime=regime,
            session=session,
            market_cap_tier=market_cap_tier
        )
    
    def _calculate_temporary_impact(
        self,
        participation_rate: float,
        volatility: float,
        regime: ImpactRegime,
        session: str,
        market_cap_tier: str
    ) -> float:
        """Calculate temporary market impact in basis points."""
        params = self.parameters
        
        # Base temporary impact
        base_impact = (
            params.temp_impact_coeff * 
            (participation_rate ** params.temp_size_exponent) *
            10000  # Convert to basis points
        )
        
        # Volatility adjustment
        vol_adjustment = 1 + (params.temp_vol_multiplier - 1) * volatility
        
        # Session adjustment
        session_mult = params.session_multipliers.get(session, 1.0)
        
        # Market cap adjustment
        cap_mult = params.market_cap_adjustments.get(market_cap_tier, 1.0)
        
        # Regime adjustment
        regime_mult = self._get_regime_multiplier(regime, 'temporary')
        
        # Liquidity penalty for large orders
        liquidity_mult = 1.0
        if participation_rate > params.liquidity_penalty_threshold:
            excess_participation = participation_rate - params.liquidity_penalty_threshold
            liquidity_mult = 1 + excess_participation * params.liquidity_penalty_factor
        
        return base_impact * vol_adjustment * session_mult * cap_mult * regime_mult * liquidity_mult
    
    def _calculate_permanent_impact(
        self,
        participation_rate: float,
        volatility: float,
        regime: ImpactRegime,
        market_cap_tier: str
    ) -> float:
        """Calculate permanent market impact in basis points."""
        params = self.parameters
        
        # Base permanent impact (square root scaling)
        base_impact = (
            params.perm_impact_coeff * 
            (participation_rate ** params.perm_size_exponent) *
            10000  # Convert to basis points
        )
        
        # Volatility adjustment (weaker than temporary)
        vol_adjustment = 1 + (params.perm_vol_multiplier - 1) * volatility
        
        # Market cap adjustment
        cap_mult = params.market_cap_adjustments.get(market_cap_tier, 1.0)
        
        # Regime adjustment
        regime_mult = self._get_regime_multiplier(regime, 'permanent')
        
        return base_impact * vol_adjustment * cap_mult * regime_mult
    
    def _get_regime_multiplier(self, regime: ImpactRegime, impact_type: str) -> float:
        """Get regime-specific impact multipliers."""
        multipliers = {
            ImpactRegime.NORMAL: {'temporary': 1.0, 'permanent': 1.0},
            ImpactRegime.VOLATILE: {'temporary': 1.5, 'permanent': 1.2},
            ImpactRegime.STRESSED: {'temporary': 2.0, 'permanent': 1.5},
            ImpactRegime.ILLIQUID: {'temporary': 1.8, 'permanent': 1.3}
        }
        
        return multipliers.get(regime, {}).get(impact_type, 1.0)
    
    def _calculate_decay_half_life(
        self,
        participation_rate: float,
        volatility: float,
        regime: ImpactRegime
    ) -> float:
        """Calculate temporary impact decay half-life in minutes."""
        
        # Base decay half-life (30 minutes for Taiwan market)
        base_half_life = 30.0
        
        # Larger orders take longer to decay
        size_factor = 1 + participation_rate * 2  # Up to 3x longer for 100% participation
        
        # Higher volatility leads to faster decay
        vol_factor = 1 / (1 + volatility)
        
        # Regime adjustments
        regime_factors = {
            ImpactRegime.NORMAL: 1.0,
            ImpactRegime.VOLATILE: 0.7,     # Faster decay in volatile markets
            ImpactRegime.STRESSED: 1.5,     # Slower decay in stressed markets
            ImpactRegime.ILLIQUID: 2.0      # Much slower decay in illiquid markets
        }
        
        regime_factor = regime_factors.get(regime, 1.0)
        
        return base_half_life * size_factor * vol_factor * regime_factor
    
    def get_decay_function(self, impact_result: ImpactCalculationResult) -> Callable[[float], float]:
        """
        Get exponential decay function for temporary impact.
        
        Args:
            impact_result: Impact calculation result
            
        Returns:
            Function that takes time in minutes and returns impact fraction remaining
        """
        half_life = impact_result.decay_half_life_minutes
        decay_rate = np.log(2) / half_life
        
        def decay_function(time_minutes: float) -> float:
            """
            Calculate remaining temporary impact fraction.
            
            Args:
                time_minutes: Time elapsed since trade in minutes
                
            Returns:
                Fraction of temporary impact remaining (0-1)
            """
            if time_minutes <= 0:
                return 1.0
            
            return np.exp(-decay_rate * time_minutes)
        
        return decay_function
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters for inspection."""
        return {
            'model_name': self.name,
            'temp_impact_coeff': self.parameters.temp_impact_coeff,
            'temp_decay_rate': self.parameters.temp_decay_rate,
            'temp_size_exponent': self.parameters.temp_size_exponent,
            'temp_vol_multiplier': self.parameters.temp_vol_multiplier,
            'perm_impact_coeff': self.parameters.perm_impact_coeff,
            'perm_size_exponent': self.parameters.perm_size_exponent,
            'perm_vol_multiplier': self.parameters.perm_vol_multiplier,
            'liquidity_penalty_threshold': self.parameters.liquidity_penalty_threshold,
            'liquidity_penalty_factor': self.parameters.liquidity_penalty_factor
        }


class PortfolioImpactAnalyzer:
    """
    Analyze aggregate market impact for portfolio transactions.
    
    Handles portfolio-level impact analysis including:
    - Cross-asset impact correlation
    - Timing optimization across trades
    - Aggregate capacity constraints
    """
    
    def __init__(self, impact_model: BaseMarketImpactModel):
        self.impact_model = impact_model
        self.logger = logging.getLogger(f"{__name__}.PortfolioImpactAnalyzer")
    
    def calculate_portfolio_impact(
        self,
        trades: List[Dict[str, Any]],
        timing_spread_minutes: float = 60.0
    ) -> Dict[str, Any]:
        """
        Calculate aggregate impact for a portfolio of trades.
        
        Args:
            trades: List of trade dictionaries with required fields
            timing_spread_minutes: Time spread for trade execution
            
        Returns:
            Portfolio impact analysis results
        """
        results = []
        total_impact_twd = 0.0
        total_trade_value = 0.0
        
        for trade in trades:
            # Calculate individual trade impact
            impact_result = self.impact_model.calculate_impact(
                symbol=trade['symbol'],
                order_size=trade['order_size'],
                price=trade['price'],
                avg_daily_volume=trade['avg_daily_volume'],
                volatility=trade['volatility'],
                timestamp=trade.get('timestamp'),
                market_data=trade.get('market_data')
            )
            
            results.append(impact_result)
            total_impact_twd += impact_result.total_impact_twd
            total_trade_value += abs(trade['order_size']) * trade['price']
        
        # Calculate portfolio-level metrics
        portfolio_impact_bps = (total_impact_twd / total_trade_value) * 10000 if total_trade_value > 0 else 0.0
        
        # Analyze timing optimization potential
        timing_optimization = self._analyze_timing_optimization(results, timing_spread_minutes)
        
        return {
            'individual_impacts': [result.to_dict() for result in results],
            'portfolio_summary': {
                'total_impact_twd': total_impact_twd,
                'total_trade_value': total_trade_value,
                'portfolio_impact_bps': portfolio_impact_bps,
                'trade_count': len(trades),
                'avg_impact_bps': np.mean([r.total_impact_bps for r in results])
            },
            'timing_optimization': timing_optimization
        }
    
    def _analyze_timing_optimization(
        self,
        results: List[ImpactCalculationResult],
        timing_spread_minutes: float
    ) -> Dict[str, Any]:
        """Analyze potential for timing optimization."""
        
        # Sort trades by impact (highest first)
        sorted_results = sorted(results, key=lambda x: x.total_impact_bps, reverse=True)
        
        # Calculate optimal timing spread
        high_impact_trades = [r for r in sorted_results if r.total_impact_bps > 50]  # > 50 bps
        optimal_spread = min(timing_spread_minutes, len(high_impact_trades) * 15)  # Max 15 min per trade
        
        # Estimate impact reduction from timing optimization
        base_total_impact = sum(r.total_impact_twd for r in results)
        
        # Assume 10-20% reduction from optimal timing
        timing_reduction_pct = min(0.20, len(high_impact_trades) * 0.05)
        optimized_impact = base_total_impact * (1 - timing_reduction_pct)
        
        return {
            'high_impact_trade_count': len(high_impact_trades),
            'recommended_timing_spread_minutes': optimal_spread,
            'estimated_impact_reduction_pct': timing_reduction_pct,
            'base_total_impact_twd': base_total_impact,
            'optimized_total_impact_twd': optimized_impact,
            'potential_savings_twd': base_total_impact - optimized_impact
        }


# Factory functions for common use cases
def create_taiwan_impact_model(
    conservative: bool = False,
    custom_params: Optional[Dict[str, Any]] = None
) -> TaiwanMarketImpactModel:
    """
    Create Taiwan market impact model with default or custom parameters.
    
    Args:
        conservative: Use conservative (higher impact) parameters
        custom_params: Override specific parameters
        
    Returns:
        Configured Taiwan market impact model
    """
    if conservative:
        params = MarketImpactParameters(
            temp_impact_coeff=0.45,      # Higher temporary impact
            perm_impact_coeff=0.30,      # Higher permanent impact
            temp_size_exponent=0.7,      # More penalty for size
            liquidity_penalty_factor=2.0  # Higher liquidity penalty
        )
    else:
        params = MarketImpactParameters()  # Use defaults
    
    # Override with custom parameters
    if custom_params:
        for key, value in custom_params.items():
            if hasattr(params, key):
                setattr(params, key, value)
    
    return TaiwanMarketImpactModel(params)


# Example usage and testing
if __name__ == "__main__":
    print("Taiwan Market Impact Model Demo")
    
    # Create model
    impact_model = create_taiwan_impact_model()
    
    # Sample trade (TSMC)
    result = impact_model.calculate_impact(
        symbol="2330.TW",
        order_size=10000,  # 10K shares
        price=500.0,       # NT$500
        avg_daily_volume=500000,  # 500K ADV
        volatility=0.25,   # 25% annual volatility
        timestamp=datetime.now()
    )
    
    print(f"\nImpact Analysis for {result.symbol}:")
    print(f"Order Size: {result.order_size:,} shares")
    print(f"Participation Rate: {result.participation_rate:.2%}")
    print(f"Temporary Impact: {result.temporary_impact_bps:.1f} bps")
    print(f"Permanent Impact: {result.permanent_impact_bps:.1f} bps")
    print(f"Total Impact: {result.total_impact_bps:.1f} bps")
    print(f"Total Cost: NT${result.total_impact_twd:,.2f}")
    print(f"Decay Half-Life: {result.decay_half_life_minutes:.1f} minutes")
    
    # Test decay function
    decay_fn = impact_model.get_decay_function(result)
    print(f"\nTemporary Impact Decay:")
    for minutes in [0, 15, 30, 60, 120]:
        remaining = decay_fn(minutes)
        print(f"After {minutes} min: {remaining:.1%} remaining")