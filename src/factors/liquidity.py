"""
Liquidity factor calculations for Taiwan market microstructure analysis.

This module implements 4 core liquidity measures:
1. Average Daily Turnover - Volume-adjusted liquidity measures
2. Bid-Ask Spread - Effective and quoted spread calculations  
3. Price Impact - Temporary and permanent price impact estimation
4. Amihud Illiquidity Ratio - Classic illiquidity measurement

All factors are adapted for Taiwan market characteristics:
- 4.5-hour trading session (09:00-13:30)
- Variable tick sizes
- 245 trading days per year
- T+2 settlement cycle
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
import pandas as pd

from .base import FactorResult, FactorMetadata, FactorCategory, FactorFrequency
from .microstructure import MicrostructureFactorCalculator, TaiwanMarketSession, TickSizeStructure

# Import dependencies - will be mocked for testing if not available
try:
    from ..data.pipeline.pit_engine import PITQueryEngine
    from ..data.core.temporal import DataType
except ImportError:
    PITQueryEngine = object
    DataType = object

logger = logging.getLogger(__name__)


class AverageDailyTurnoverCalculator(MicrostructureFactorCalculator):
    """
    Average Daily Turnover Factor Calculator.
    
    Calculates volume-adjusted liquidity measures using:
    - Daily turnover = price * volume  
    - Rolling average turnover over multiple periods
    - Turnover normalized by market cap
    - Taiwan session-adjusted calculations
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="avg_daily_turnover",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Average daily turnover over multiple periods (20D, 60D)",
            lookback_days=252,  # 1 year for robust estimation
            data_requirements=[DataType.OHLCV, DataType.MARKET_CAP],
            taiwan_specific=True,
            expected_ic=0.04,
            expected_turnover=0.15
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate average daily turnover factors."""
        
        # Get price and volume data
        price_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        # Get market cap data for normalization
        market_cap_data = self._get_historical_data(
            symbols, as_of_date, DataType.MARKET_CAP, 60
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_prices = price_data[price_data['symbol'] == symbol].copy()
                symbol_mcap = market_cap_data[market_cap_data['symbol'] == symbol].copy()
                
                if symbol_prices.empty or symbol_mcap.empty:
                    continue
                
                # Calculate daily turnover
                symbol_prices['daily_turnover'] = (
                    symbol_prices['close'] * symbol_prices['volume']
                )
                
                # Calculate rolling averages
                symbol_prices['turnover_20d'] = (
                    symbol_prices['daily_turnover']
                    .rolling(window=20, min_periods=10)
                    .mean()
                )
                
                symbol_prices['turnover_60d'] = (
                    symbol_prices['daily_turnover']
                    .rolling(window=60, min_periods=30)
                    .mean()
                )
                
                # Get latest market cap for normalization
                latest_mcap = symbol_mcap['market_cap'].iloc[-1] if not symbol_mcap.empty else None
                
                # Calculate final factor value
                turnover_20d = symbol_prices['turnover_20d'].iloc[-1]
                turnover_60d = symbol_prices['turnover_60d'].iloc[-1]
                
                if pd.notna(turnover_20d) and pd.notna(turnover_60d) and latest_mcap and latest_mcap > 0:
                    # Composite turnover score: recent vs historical, normalized by market cap
                    turnover_ratio = turnover_20d / turnover_60d  # Recent vs historical
                    normalized_turnover = turnover_20d / (latest_mcap * 1e6)  # Normalize by market cap
                    
                    # Combine metrics with Taiwan market adjustment
                    taiwan_adjustment = self._get_taiwan_liquidity_adjustment(symbol, as_of_date)
                    
                    factor_value = (
                        0.6 * np.log1p(normalized_turnover) +  # Log-transform for stability
                        0.3 * np.tanh(turnover_ratio - 1) +    # Recent vs historical
                        0.1 * taiwan_adjustment                # Taiwan-specific adjustment
                    )
                    
                    factor_values[symbol] = factor_value
                    
            except Exception as e:
                logger.warning(f"Error calculating turnover for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _get_taiwan_liquidity_adjustment(self, symbol: str, as_of_date: date) -> float:
        """Get Taiwan-specific liquidity adjustments."""
        # Placeholder for Taiwan-specific adjustments
        # In practice, would consider:
        # - Market segment (main board vs OTC)
        # - Foreign ownership constraints
        # - Index inclusion status
        return 0.0


class BidAskSpreadCalculator(MicrostructureFactorCalculator):
    """
    Bid-Ask Spread Factor Calculator.
    
    Calculates transaction cost proxies using:
    - Quoted spread: (ask - bid) / midpoint
    - Effective spread: trade impact relative to midpoint
    - Proportional spread adjusted for tick size
    - Taiwan session-weighted calculations
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="bid_ask_spread",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Bid-ask spread measures (quoted and effective spreads)",
            lookback_days=63,  # ~3 months for stable estimation
            data_requirements=[DataType.ORDER_BOOK, DataType.TRADES],
            taiwan_specific=True,
            expected_ic=0.05,
            expected_turnover=0.20
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate bid-ask spread factors."""
        
        # Get order book data
        order_book_data = self._get_order_book_data(
            symbols, as_of_date, self.metadata.lookback_days
        )
        
        # Get trade data for effective spread calculation
        trade_data = self._get_historical_data(
            symbols, as_of_date, DataType.TRADES, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_book = order_book_data[order_book_data['symbol'] == symbol].copy()
                symbol_trades = trade_data[trade_data['symbol'] == symbol].copy()
                
                if symbol_book.empty:
                    continue
                
                # Adjust for Taiwan trading session
                symbol_book = self._adjust_for_taiwan_session(symbol_book)
                symbol_trades = self._adjust_for_taiwan_session(symbol_trades)
                
                # Calculate quoted spread
                symbol_book['midpoint'] = (symbol_book['bid'] + symbol_book['ask']) / 2
                symbol_book['quoted_spread'] = (
                    (symbol_book['ask'] - symbol_book['bid']) / symbol_book['midpoint']
                )
                
                # Calculate time-weighted average quoted spread
                avg_quoted_spread = symbol_book['quoted_spread'].mean()
                
                # Calculate effective spread (if trade data available)
                effective_spread = 0.0
                if not symbol_trades.empty:
                    # Merge trades with order book to get contemporaneous midpoints
                    merged_data = pd.merge_asof(
                        symbol_trades.sort_values('timestamp'),
                        symbol_book[['timestamp', 'midpoint']].sort_values('timestamp'),
                        on='timestamp',
                        direction='backward'
                    )
                    
                    if not merged_data.empty:
                        merged_data['effective_spread'] = (
                            2 * abs(merged_data['price'] - merged_data['midpoint']) / 
                            merged_data['midpoint']
                        )
                        effective_spread = merged_data['effective_spread'].mean()
                
                # Adjust for Taiwan tick size structure
                latest_price = symbol_book['midpoint'].iloc[-1] if not symbol_book.empty else 100
                tick_adjustment = self._get_tick_size_adjustment(latest_price)
                
                # Combine spread measures
                if pd.notna(avg_quoted_spread) and avg_quoted_spread > 0:
                    factor_value = (
                        0.6 * avg_quoted_spread +
                        0.3 * effective_spread + 
                        0.1 * tick_adjustment
                    )
                    
                    factor_values[symbol] = factor_value
                    
            except Exception as e:
                logger.warning(f"Error calculating spread for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _get_tick_size_adjustment(self, price: float) -> float:
        """Get adjustment factor based on tick size relative to price."""
        tick_size = self.tick_structure.get_tick_size(price)
        return (tick_size / price) if price > 0 else 0.0


class PriceImpactCalculator(MicrostructureFactorCalculator):
    """
    Price Impact Factor Calculator.
    
    Estimates market impact using:
    - Temporary impact: immediate price movement per unit volume
    - Permanent impact: lasting price effect
    - Volume-normalized impact measures
    - Taiwan market depth considerations
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="price_impact",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Price impact measures (temporary and permanent)",
            lookback_days=126,  # ~6 months for impact estimation
            data_requirements=[DataType.TRADES, DataType.OHLCV],
            taiwan_specific=True,
            expected_ic=0.03,
            expected_turnover=0.12
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate price impact factors."""
        
        # Get trade and price data
        trade_data = self._get_historical_data(
            symbols, as_of_date, DataType.TRADES, self.metadata.lookback_days
        )
        
        price_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_trades = trade_data[trade_data['symbol'] == symbol].copy()
                symbol_prices = price_data[price_data['symbol'] == symbol].copy()
                
                if symbol_trades.empty or symbol_prices.empty:
                    continue
                
                # Adjust for Taiwan trading session
                symbol_trades = self._adjust_for_taiwan_session(symbol_trades)
                
                # Calculate trade-level impacts
                symbol_trades['trade_impact'] = self._calculate_trade_impact(symbol_trades)
                
                # Calculate volume-weighted price impact
                total_volume = symbol_trades['volume'].sum()
                if total_volume > 0:
                    weighted_impact = (
                        (symbol_trades['trade_impact'] * symbol_trades['volume']).sum() / 
                        total_volume
                    )
                    
                    # Calculate volatility adjustment
                    returns = symbol_prices['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(self.market_session.trading_days_per_year)
                    
                    # Normalize impact by volatility and Taiwan market factors
                    if volatility > 0:
                        normalized_impact = weighted_impact / volatility
                        taiwan_depth_adjustment = self._get_market_depth_adjustment(symbol, as_of_date)
                        
                        factor_value = normalized_impact * (1 + taiwan_depth_adjustment)
                        factor_values[symbol] = factor_value
                
            except Exception as e:
                logger.warning(f"Error calculating price impact for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_trade_impact(self, trades: pd.DataFrame) -> pd.Series:
        """Calculate individual trade impacts."""
        # Simple impact measure: price change per unit volume
        trades = trades.sort_values('timestamp')
        
        price_changes = trades['price'].diff()
        volume_sqrt = np.sqrt(trades['volume'])
        
        # Impact = price_change / sqrt(volume) to account for market impact scaling
        impact = price_changes / volume_sqrt
        return impact.fillna(0)
    
    def _get_market_depth_adjustment(self, symbol: str, as_of_date: date) -> float:
        """Get Taiwan market depth adjustment factor."""
        # Placeholder for market depth considerations
        # In practice would consider:
        # - Average order book depth
        # - Market segment liquidity
        # - Time-of-day effects
        return 0.0


class AmihudIlliquidityCalculator(MicrostructureFactorCalculator):
    """
    Amihud Illiquidity Ratio Calculator.
    
    Calculates the classic Amihud (2002) illiquidity measure:
    - Daily |return| / dollar_volume
    - Rolling averages over multiple periods
    - Taiwan market adjustments for price limits and session hours
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="amihud_illiquidity",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Amihud illiquidity ratio (|return| / dollar_volume)",
            lookback_days=252,  # 1 year for robust estimation
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            expected_ic=0.04,
            expected_turnover=0.18
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate Amihud illiquidity ratio."""
        
        # Get price and volume data
        price_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_data = price_data[price_data['symbol'] == symbol].copy()
                
                if symbol_data.empty or len(symbol_data) < 60:  # Need minimum data
                    continue
                
                # Calculate returns  
                symbol_data['return'] = symbol_data['close'].pct_change()
                
                # Calculate dollar volume (price * volume)
                symbol_data['dollar_volume'] = symbol_data['close'] * symbol_data['volume']
                
                # Filter out zero volume days
                valid_data = symbol_data[
                    (symbol_data['dollar_volume'] > 0) & 
                    (symbol_data['return'].notna())
                ].copy()
                
                if valid_data.empty:
                    continue
                
                # Calculate daily Amihud ratios
                valid_data['amihud_daily'] = (
                    abs(valid_data['return']) / valid_data['dollar_volume'] * 1e6  # Scale by million
                )
                
                # Handle Taiwan price limit days
                price_limit_threshold = 0.095  # 9.5% to catch near-limit days
                is_limit_day = abs(valid_data['return']) > price_limit_threshold
                
                # Adjust for price limit days (reduce impact)
                valid_data.loc[is_limit_day, 'amihud_daily'] *= 0.5
                
                # Calculate rolling averages
                valid_data['amihud_21d'] = (
                    valid_data['amihud_daily']
                    .rolling(window=21, min_periods=10)
                    .mean()
                )
                
                valid_data['amihud_63d'] = (
                    valid_data['amihud_daily']
                    .rolling(window=63, min_periods=30)
                    .mean()
                )
                
                # Final factor: recent vs long-term illiquidity
                recent_illiquidity = valid_data['amihud_21d'].iloc[-1]
                long_term_illiquidity = valid_data['amihud_63d'].iloc[-1]
                
                if pd.notna(recent_illiquidity) and pd.notna(long_term_illiquidity):
                    # Log transform for stability and Taiwan session adjustment
                    taiwan_session_multiplier = (
                        self.market_session.session_minutes / 390  # Adjust vs US 6.5hr session
                    )
                    
                    factor_value = (
                        np.log1p(recent_illiquidity) * taiwan_session_multiplier
                    )
                    
                    factor_values[symbol] = factor_value
                
            except Exception as e:
                logger.warning(f"Error calculating Amihud illiquidity for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )


class LiquidityFactors:
    """Container for all liquidity factor calculators."""
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.calculators = {
            'avg_daily_turnover': AverageDailyTurnoverCalculator(pit_engine),
            'bid_ask_spread': BidAskSpreadCalculator(pit_engine),
            'price_impact': PriceImpactCalculator(pit_engine),
            'amihud_illiquidity': AmihudIlliquidityCalculator(pit_engine)
        }
    
    def get_all_calculators(self) -> Dict[str, MicrostructureFactorCalculator]:
        """Get all liquidity factor calculators."""
        return self.calculators.copy()
    
    def calculate_all_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, FactorResult]:
        """Calculate all liquidity factors."""
        results = {}
        
        for name, calculator in self.calculators.items():
            try:
                result = calculator.calculate(symbols, as_of_date)
                results[name] = result
                logger.info(f"Calculated {name}: {result.coverage:.1%} coverage")
            except Exception as e:
                logger.error(f"Error calculating liquidity factor {name}: {e}")
                continue
        
        return results