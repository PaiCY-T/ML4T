"""
Mean reversion factor calculations for Taiwan market.

This module implements mean reversion factors including price vs moving averages,
Bollinger Band position, and Z-score reversion signals.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from scipy import stats

from .base import TechnicalFactorCalculator, FactorResult, FactorMetadata, FactorCategory, FactorFrequency
from .taiwan_adjustments import TaiwanMarketAdjustments
from ..data.core.temporal import DataType

logger = logging.getLogger(__name__)


class MovingAverageReversionCalculator(TechnicalFactorCalculator):
    """Calculate price vs moving average reversion factors (20D, 50D, 200D)."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="ma_reversion",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="Price relative to moving averages (20D, 50D, 200D)",
            lookback_days=250,  # Need up to 200-day MA
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=250,
            expected_ic=0.035,
            expected_turnover=0.4
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate moving average reversion factors."""
        
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
            
            if len(symbol_prices) < 210:  # Need enough data for 200-day MA
                continue
            
            try:
                ma_reversion = self._calculate_ma_reversion(symbol_prices)
                if not np.isnan(ma_reversion):
                    factor_values[symbol] = ma_reversion
                    
            except Exception as e:
                self.logger.warning(f"Error calculating MA reversion for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_ma_reversion(self, prices: pd.Series) -> float:
        """Calculate moving average reversion factor."""
        
        current_price = prices.iloc[-1]
        
        # Calculate moving averages
        ma_20 = self._calculate_sma(prices, 20).iloc[-1]
        ma_50 = self._calculate_sma(prices, 50).iloc[-1]
        ma_200 = self._calculate_sma(prices, 200).iloc[-1]
        
        if np.isnan(ma_20) or np.isnan(ma_50) or np.isnan(ma_200):
            return np.nan
        
        # Price relative to moving averages (negative indicates below MA, positive above)
        rel_ma_20 = (current_price / ma_20) - 1.0
        rel_ma_50 = (current_price / ma_50) - 1.0
        rel_ma_200 = (current_price / ma_200) - 1.0
        
        # MA trend strength (is MA trending up or down?)
        ma_20_trend = (ma_20 / self._calculate_sma(prices, 20).iloc[-5]) - 1.0 if len(prices) >= 25 else 0
        ma_50_trend = (ma_50 / self._calculate_sma(prices, 50).iloc[-10]) - 1.0 if len(prices) >= 60 else 0
        
        # Composite reversion signal
        # Negative values indicate potential mean reversion opportunity (price below MA)
        # Weight shorter-term MAs more heavily for reversion signals
        reversion_signal = (
            0.5 * (-rel_ma_20) +  # Negative because we want reversion
            0.3 * (-rel_ma_50) +
            0.2 * (-rel_ma_200) +
            0.1 * ma_20_trend +   # Trend component
            0.05 * ma_50_trend
        )
        
        return float(reversion_signal)


class BollingerBandPositionCalculator(TechnicalFactorCalculator):
    """Calculate Bollinger Band position reversion factors."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="bb_position",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="Position within Bollinger Bands with reversion signals",
            lookback_days=60,  # Need sufficient data for BB calculation
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=60,
            expected_ic=0.04,
            expected_turnover=0.6
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate Bollinger Band position factors."""
        
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
            
            if len(symbol_prices) < 30:  # Need minimum data for BB
                continue
            
            try:
                bb_position = self._calculate_bb_position(symbol_prices)
                if not np.isnan(bb_position):
                    factor_values[symbol] = bb_position
                    
            except Exception as e:
                self.logger.warning(f"Error calculating BB position for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_bb_position(self, prices: pd.Series, 
                              window: int = 20, std_mult: float = 2.0) -> float:
        """Calculate Bollinger Band position factor."""
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self._calculate_bollinger_bands(
            prices, window, std_mult
        )
        
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_middle = middle_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        if np.isnan(current_upper) or np.isnan(current_lower):
            return np.nan
        
        # BB position: -1 (at lower band) to +1 (at upper band)
        band_width = current_upper - current_lower
        if band_width == 0:
            return 0.0
        
        bb_position = (current_price - current_middle) / (band_width / 2)
        
        # Band squeeze indicator (narrow bands suggest low volatility)
        recent_band_width = (upper_band - lower_band).tail(10).mean()
        historical_band_width = (upper_band - lower_band).tail(50).mean()
        
        squeeze_factor = recent_band_width / historical_band_width if historical_band_width > 0 else 1.0
        
        # BB momentum (is price moving toward or away from bands?)
        price_momentum = prices.pct_change(5).iloc[-1] if len(prices) >= 6 else 0
        
        # Reversion signal: extreme positions with low volatility suggest mean reversion
        # Negative values indicate reversion opportunity
        if abs(bb_position) > 0.8:  # Near bands
            reversion_strength = -bb_position * squeeze_factor  # Negative for reversion
        else:
            reversion_strength = bb_position * 0.5  # Weak signal in middle
        
        # Add momentum component
        reversion_signal = 0.7 * reversion_strength + 0.3 * (-price_momentum)
        
        return float(reversion_signal)


class ZScoreReversionCalculator(TechnicalFactorCalculator):
    """Calculate Z-score reversion factors relative to historical mean."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="zscore_reversion",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="Z-score reversion relative to historical price mean",
            lookback_days=252,  # One year of data for stable statistics
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=252,
            expected_ic=0.03,
            expected_turnover=0.5
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate Z-score reversion factors."""
        
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
            
            if len(symbol_prices) < 60:  # Need minimum data for stable statistics
                continue
            
            try:
                zscore_reversion = self._calculate_zscore_reversion(symbol_prices)
                if not np.isnan(zscore_reversion):
                    factor_values[symbol] = zscore_reversion
                    
            except Exception as e:
                self.logger.warning(f"Error calculating Z-score reversion for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_zscore_reversion(self, prices: pd.Series) -> float:
        """Calculate Z-score reversion factor."""
        
        current_price = prices.iloc[-1]
        
        # Calculate Z-score using different lookback periods
        zscore_60d = self._calculate_price_zscore(prices, 60)
        zscore_120d = self._calculate_price_zscore(prices, 120)
        zscore_252d = self._calculate_price_zscore(prices, 252)
        
        # Return-based Z-score (alternative approach)
        returns = prices.pct_change().dropna()
        if len(returns) >= 20:
            recent_return = returns.tail(5).mean()  # 5-day average return
            historical_return_mean = returns.tail(60).mean()
            historical_return_std = returns.tail(60).std()
            
            if historical_return_std > 0:
                return_zscore = (recent_return - historical_return_mean) / historical_return_std
            else:
                return_zscore = 0
        else:
            return_zscore = 0
        
        # Composite Z-score reversion signal
        # Negative values indicate reversion opportunity (extreme positive Z-score)
        valid_zscores = [z for z in [zscore_60d, zscore_120d, zscore_252d] if not np.isnan(z)]
        
        if valid_zscores:
            price_zscore = np.mean(valid_zscores)
            
            # Reversion signal: extreme Z-scores suggest mean reversion
            reversion_signal = -price_zscore  # Negative Z-score becomes positive reversion signal
            
            # Add return momentum component
            reversion_signal = 0.8 * reversion_signal + 0.2 * (-return_zscore)
            
            return float(reversion_signal)
        else:
            return np.nan
    
    def _calculate_price_zscore(self, prices: pd.Series, lookback: int) -> float:
        """Calculate price Z-score for given lookback period."""
        
        if len(prices) < lookback:
            return np.nan
        
        historical_prices = prices.tail(lookback)
        current_price = prices.iloc[-1]
        
        mean_price = historical_prices.mean()
        std_price = historical_prices.std()
        
        if std_price == 0:
            return np.nan
        
        zscore = (current_price - mean_price) / std_price
        return float(zscore)


class ShortTermReversalCalculator(TechnicalFactorCalculator):
    """Calculate short-term reversal indicators."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="short_term_reversal",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="Short-term price reversal patterns (1-5 days)",
            lookback_days=30,  # Short-term focus
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=30,
            expected_ic=0.02,
            expected_turnover=0.8
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate short-term reversal factors."""
        
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
        adjusted_prices, adjusted_returns = self.taiwan_adj.adjust_for_price_limits(
            close_prices, returns
        )
        
        factor_values = {}
        
        for symbol in symbols:
            if symbol not in adjusted_prices.columns:
                continue
            
            symbol_prices = adjusted_prices[symbol].dropna()
            symbol_returns = adjusted_returns[symbol].dropna() if symbol in adjusted_returns.columns else None
            
            if len(symbol_prices) < 10 or symbol_returns is None:
                continue
            
            try:
                reversal_signal = self._calculate_short_term_reversal(
                    symbol_prices, symbol_returns
                )
                if not np.isnan(reversal_signal):
                    factor_values[symbol] = reversal_signal
                    
            except Exception as e:
                self.logger.warning(f"Error calculating short-term reversal for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_short_term_reversal(self, prices: pd.Series, 
                                     returns: pd.Series) -> float:
        """Calculate short-term reversal factor."""
        
        if len(returns) < 5:
            return np.nan
        
        # Recent performance (negative indicates potential for reversal)
        recent_1d = returns.iloc[-1]
        recent_3d = returns.tail(3).sum()
        recent_5d = returns.tail(5).sum()
        
        # Volatility adjustment
        volatility = returns.tail(20).std() if len(returns) >= 20 else returns.std()
        if volatility == 0:
            return np.nan
        
        # Volume-adjusted reversal (if volume data available)
        # For now, using price-based signals only
        
        # Reversal patterns
        # 1. Large recent moves suggest potential reversal
        reversal_1d = -recent_1d / volatility if volatility > 0 else 0
        reversal_3d = -recent_3d / (volatility * np.sqrt(3)) if volatility > 0 else 0
        reversal_5d = -recent_5d / (volatility * np.sqrt(5)) if volatility > 0 else 0
        
        # 2. Consecutive moves in same direction suggest reversal
        consecutive_signal = self._calculate_consecutive_move_reversal(returns)
        
        # Composite short-term reversal signal
        reversal_signal = (
            0.4 * reversal_1d +
            0.3 * reversal_3d +
            0.2 * reversal_5d +
            0.1 * consecutive_signal
        )
        
        return float(reversal_signal)
    
    def _calculate_consecutive_move_reversal(self, returns: pd.Series) -> float:
        """Calculate reversal signal based on consecutive moves."""
        
        if len(returns) < 5:
            return 0.0
        
        recent_returns = returns.tail(5)
        
        # Count consecutive moves in same direction
        positive_streak = 0
        negative_streak = 0
        
        for ret in reversed(recent_returns.values):
            if ret > 0:
                if negative_streak > 0:
                    break
                positive_streak += 1
            elif ret < 0:
                if positive_streak > 0:
                    break
                negative_streak += 1
            else:
                break
        
        # Reversal signal based on streak length
        max_streak = max(positive_streak, negative_streak)
        
        if max_streak >= 3:
            # Strong consecutive moves suggest reversal
            direction = 1 if negative_streak > positive_streak else -1
            reversal_strength = min(max_streak / 5.0, 1.0)  # Cap at 1.0
            return direction * reversal_strength
        
        return 0.0