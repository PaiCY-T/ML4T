"""
Momentum factor calculations for Taiwan market.

This module implements momentum-based factors including price momentum,
RSI-based momentum, and MACD signal strength optimized for Taiwan market.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

from .base import TechnicalFactorCalculator, FactorResult, FactorMetadata, FactorCategory, FactorFrequency
from .taiwan_adjustments import TaiwanMarketAdjustments
from ..data.core.temporal import DataType

logger = logging.getLogger(__name__)


class PriceMomentumCalculator(TechnicalFactorCalculator):
    """Calculate price momentum factors (1M, 3M, 6M, 12M returns)."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="price_momentum",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="Multi-period price momentum (1M, 3M, 6M, 12M returns)",
            lookback_days=365,  # Need up to 12 months
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=365,
            expected_ic=0.05,
            expected_turnover=0.3
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate price momentum factors."""
        
        # Get historical price data
        prices_df = self._get_ohlcv_data(symbols, as_of_date, self.metadata.lookback_days)
        
        if prices_df.empty:
            return FactorResult(
                factor_name=self.metadata.name,
                date=as_of_date,
                values={},
                metadata=self.metadata
            )
        
        # Extract close prices
        close_prices = prices_df['close'].unstack(level=0)  # symbols as columns
        
        # Apply Taiwan market adjustments
        returns = close_prices.pct_change()
        adjusted_prices, adjusted_returns = self.taiwan_adj.adjust_for_price_limits(
            close_prices, returns
        )
        
        # Calculate momentum factors
        factor_values = {}
        
        for symbol in symbols:
            if symbol not in adjusted_prices.columns:
                continue
            
            symbol_prices = adjusted_prices[symbol].dropna()
            
            if len(symbol_prices) < 22:  # Need at least 1 month of data
                continue
            
            try:
                momentum_factors = self._calculate_momentum_factors(symbol_prices)
                
                # Composite momentum score (weighted average)
                weights = [0.1, 0.2, 0.3, 0.4]  # 1M, 3M, 6M, 12M weights
                valid_factors = [f for f in momentum_factors if not np.isnan(f)]
                valid_weights = weights[:len(valid_factors)]
                
                if valid_factors and sum(valid_weights) > 0:
                    composite_momentum = np.average(valid_factors, weights=valid_weights)
                    factor_values[symbol] = composite_momentum
                
            except Exception as e:
                self.logger.warning(f"Error calculating momentum for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_momentum_factors(self, prices: pd.Series) -> List[float]:
        """Calculate individual momentum factors."""
        factors = []
        periods = [22, 66, 132, 264]  # 1M, 3M, 6M, 12M in trading days
        
        current_price = prices.iloc[-1]
        
        for period in periods:
            if len(prices) >= period:
                past_price = prices.iloc[-period]
                momentum = (current_price / past_price) - 1.0
                factors.append(momentum)
            else:
                factors.append(np.nan)
        
        return factors


class RSIMomentumCalculator(TechnicalFactorCalculator):
    """Calculate RSI-based momentum factors."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="rsi_momentum",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="RSI-based momentum with trend strength",
            lookback_days=100,  # Need sufficient history for RSI
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=100,
            expected_ic=0.03,
            expected_turnover=0.4
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate RSI momentum factors."""
        
        prices_df = self._get_ohlcv_data(symbols, as_of_date, self.metadata.lookback_days)
        
        if prices_df.empty:
            return FactorResult(
                factor_name=self.metadata.name,
                date=as_of_date,
                values={},
                metadata=self.metadata
            )
        
        close_prices = prices_df['close'].unstack(level=0)
        returns = close_prices.pct_change()
        adjusted_prices, _ = self.taiwan_adj.adjust_for_price_limits(close_prices, returns)
        
        factor_values = {}
        
        for symbol in symbols:
            if symbol not in adjusted_prices.columns:
                continue
            
            symbol_prices = adjusted_prices[symbol].dropna()
            
            if len(symbol_prices) < 30:  # Need minimum data for RSI
                continue
            
            try:
                rsi_momentum = self._calculate_rsi_momentum(symbol_prices)
                if not np.isnan(rsi_momentum):
                    factor_values[symbol] = rsi_momentum
                    
            except Exception as e:
                self.logger.warning(f"Error calculating RSI momentum for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_rsi_momentum(self, prices: pd.Series, 
                              rsi_window: int = 14) -> float:
        """Calculate RSI-based momentum factor."""
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices, rsi_window)
        
        if len(rsi.dropna()) < 10:
            return np.nan
        
        current_rsi = rsi.iloc[-1]
        
        # RSI momentum: deviation from neutral (50) with trend consideration
        rsi_deviation = (current_rsi - 50) / 50
        
        # Add RSI trend component (RSI slope over recent period)
        recent_rsi = rsi.tail(5).dropna()
        if len(recent_rsi) >= 3:
            rsi_slope = np.polyfit(range(len(recent_rsi)), recent_rsi.values, 1)[0]
            rsi_trend = rsi_slope / 10  # Normalize
        else:
            rsi_trend = 0
        
        # Composite RSI momentum
        rsi_momentum = 0.7 * rsi_deviation + 0.3 * rsi_trend
        
        return float(rsi_momentum)


class MACDSignalCalculator(TechnicalFactorCalculator):
    """Calculate MACD signal strength factors."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="macd_signal",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="MACD histogram and signal line strength",
            lookback_days=80,  # Need enough data for MACD calculation
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=80,
            expected_ic=0.025,
            expected_turnover=0.5
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate MACD signal strength."""
        
        prices_df = self._get_ohlcv_data(symbols, as_of_date, self.metadata.lookback_days)
        
        if prices_df.empty:
            return FactorResult(
                factor_name=self.metadata.name,
                date=as_of_date,
                values={},
                metadata=self.metadata
            )
        
        close_prices = prices_df['close'].unstack(level=0)
        returns = close_prices.pct_change()
        adjusted_prices, _ = self.taiwan_adj.adjust_for_price_limits(close_prices, returns)
        
        factor_values = {}
        
        for symbol in symbols:
            if symbol not in adjusted_prices.columns:
                continue
            
            symbol_prices = adjusted_prices[symbol].dropna()
            
            if len(symbol_prices) < 50:  # Need minimum data for MACD
                continue
            
            try:
                macd_signal = self._calculate_macd_signal_strength(symbol_prices)
                if not np.isnan(macd_signal):
                    factor_values[symbol] = macd_signal
                    
            except Exception as e:
                self.logger.warning(f"Error calculating MACD signal for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_macd_signal_strength(self, prices: pd.Series,
                                      fast: int = 12, slow: int = 26, 
                                      signal: int = 9) -> float:
        """Calculate MACD signal strength factor."""
        
        # Calculate MACD components
        macd_line, signal_line, histogram = self._calculate_macd(
            prices, fast, slow, signal
        )
        
        if len(histogram.dropna()) < 10:
            return np.nan
        
        # Current signal strength components
        current_histogram = histogram.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        # MACD signal strength metrics
        
        # 1. Histogram strength (normalized by recent volatility)
        recent_histogram = histogram.tail(20).dropna()
        if len(recent_histogram) > 0:
            histogram_vol = recent_histogram.std()
            if histogram_vol > 0:
                histogram_strength = current_histogram / histogram_vol
            else:
                histogram_strength = 0
        else:
            histogram_strength = 0
        
        # 2. MACD line momentum
        recent_macd = macd_line.tail(5).dropna()
        if len(recent_macd) >= 3:
            macd_momentum = np.polyfit(range(len(recent_macd)), recent_macd.values, 1)[0]
        else:
            macd_momentum = 0
        
        # 3. Signal line crossover strength
        crossover_strength = (current_macd - current_signal) / abs(current_signal) if current_signal != 0 else 0
        
        # Composite MACD signal strength
        signal_strength = (
            0.5 * histogram_strength +
            0.3 * macd_momentum * 100 +  # Scale momentum
            0.2 * crossover_strength
        )
        
        return float(signal_strength)