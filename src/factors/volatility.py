"""
Volatility factor calculations for Taiwan market.

This module implements volatility-based factors including realized volatility,
GARCH volatility forecasts, and Taiwan market volatility indicators.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings

from .base import TechnicalFactorCalculator, FactorResult, FactorMetadata, FactorCategory, FactorFrequency
from .taiwan_adjustments import TaiwanMarketAdjustments
from ..data.core.temporal import DataType

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class RealizedVolatilityCalculator(TechnicalFactorCalculator):
    """Calculate realized volatility factors (5D, 20D, 60D)."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="realized_volatility",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="Multi-period realized volatility (5D, 20D, 60D)",
            lookback_days=80,  # Need sufficient data for 60D calculation
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=80,
            expected_ic=0.04,
            expected_turnover=0.3
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate realized volatility factors."""
        
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
            if symbol not in adjusted_returns.columns:
                continue
            
            symbol_returns = adjusted_returns[symbol].dropna()
            
            if len(symbol_returns) < 10:
                continue
            
            try:
                realized_vol = self._calculate_realized_volatility(symbol_returns)
                if not np.isnan(realized_vol):
                    factor_values[symbol] = realized_vol
                    
            except Exception as e:
                self.logger.warning(f"Error calculating realized volatility for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_realized_volatility(self, returns: pd.Series) -> float:
        """Calculate multi-period realized volatility factor."""
        
        # Calculate realized volatility for different periods
        vol_5d = self._calculate_rolling_volatility(returns, 5)
        vol_20d = self._calculate_rolling_volatility(returns, 20)
        vol_60d = self._calculate_rolling_volatility(returns, 60)
        
        current_vol_5d = vol_5d.iloc[-1] if not vol_5d.empty and not np.isnan(vol_5d.iloc[-1]) else np.nan
        current_vol_20d = vol_20d.iloc[-1] if not vol_20d.empty and not np.isnan(vol_20d.iloc[-1]) else np.nan
        current_vol_60d = vol_60d.iloc[-1] if not vol_60d.empty and not np.isnan(vol_60d.iloc[-1]) else np.nan
        
        # Volatility term structure analysis
        vol_signals = []
        
        # Short-term vs medium-term volatility
        if not np.isnan(current_vol_5d) and not np.isnan(current_vol_20d):
            vol_term_structure_1 = (current_vol_5d / current_vol_20d) - 1.0
            vol_signals.append(vol_term_structure_1)
        
        # Medium-term vs long-term volatility
        if not np.isnan(current_vol_20d) and not np.isnan(current_vol_60d):
            vol_term_structure_2 = (current_vol_20d / current_vol_60d) - 1.0
            vol_signals.append(vol_term_structure_2)
        
        # Volatility level relative to historical percentile
        if len(vol_20d.dropna()) >= 20:
            historical_vol = vol_20d.dropna()
            vol_percentile = stats.percentileofscore(historical_vol.values, current_vol_20d) / 100.0
            # Convert to z-score like measure
            vol_level_signal = (vol_percentile - 0.5) * 2  # Range: -1 to 1
            vol_signals.append(vol_level_signal)
        
        # Volatility clustering (persistence)
        if len(vol_5d.dropna()) >= 5:
            recent_vol_trend = vol_5d.tail(5).mean() / vol_5d.tail(10).head(5).mean() - 1.0
            if not np.isnan(recent_vol_trend):
                vol_signals.append(recent_vol_trend)
        
        # Composite volatility factor
        if vol_signals:
            # Weight term structure more heavily
            weights = [0.3, 0.3, 0.2, 0.2][:len(vol_signals)]
            weights = np.array(weights) / sum(weights)  # Normalize
            volatility_factor = np.average(vol_signals, weights=weights)
            return float(volatility_factor)
        
        return np.nan


class GARCHVolatilityCalculator(TechnicalFactorCalculator):
    """Calculate GARCH volatility forecast factors."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="garch_volatility",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="GARCH(1,1) volatility forecasting",
            lookback_days=252,  # Need sufficient data for GARCH
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=100,  # Minimum for GARCH
            expected_ic=0.035,
            expected_turnover=0.25
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate GARCH volatility factors."""
        
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
            if symbol not in adjusted_returns.columns:
                continue
            
            symbol_returns = adjusted_returns[symbol].dropna()
            
            if len(symbol_returns) < 60:  # Minimum for GARCH
                continue
            
            try:
                garch_factor = self._calculate_garch_factor(symbol_returns)
                if not np.isnan(garch_factor):
                    factor_values[symbol] = garch_factor
                    
            except Exception as e:
                self.logger.warning(f"Error calculating GARCH factor for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_garch_factor(self, returns: pd.Series) -> float:
        """Calculate GARCH-based volatility factor."""
        
        # Simplified GARCH(1,1) implementation
        # For production, consider using arch library
        
        returns_array = returns.values
        returns_array = returns_array[~np.isnan(returns_array)]
        
        if len(returns_array) < 30:
            return np.nan
        
        try:
            # Simplified GARCH estimation
            garch_vol = self._fit_simple_garch(returns_array)
            
            if np.isnan(garch_vol):
                return np.nan
            
            # Compare GARCH forecast to recent realized volatility
            recent_realized_vol = np.std(returns_array[-20:]) * np.sqrt(self.taiwan_adj.TRADING_DAYS_PER_YEAR)
            
            if recent_realized_vol > 0:
                vol_forecast_ratio = garch_vol / recent_realized_vol - 1.0
                return float(vol_forecast_ratio)
            
        except Exception as e:
            self.logger.debug(f"GARCH calculation failed: {e}")
        
        return np.nan
    
    def _fit_simple_garch(self, returns: np.ndarray) -> float:
        """Simplified GARCH(1,1) fitting."""
        
        # Initial parameter estimates
        omega = np.var(returns) * 0.1
        alpha = 0.1
        beta = 0.8
        
        try:
            # Simple method of moments estimation
            # This is a simplified approach - for production use arch library
            
            # Calculate squared returns
            squared_returns = returns ** 2
            mean_squared_return = np.mean(squared_returns)
            
            # Simple volatility clustering measure
            vol_persistence = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            if np.isnan(vol_persistence):
                vol_persistence = 0.5
            
            # Adjust parameters based on persistence
            beta = max(0.1, min(0.95, vol_persistence))
            alpha = max(0.01, min(0.3, 1 - beta - 0.1))
            omega = mean_squared_return * (1 - alpha - beta)
            
            # Forecast next period volatility
            recent_vol = np.var(returns[-10:]) if len(returns) >= 10 else np.var(returns)
            garch_forecast = np.sqrt(omega + alpha * returns[-1]**2 + beta * recent_vol)
            
            # Annualize
            annual_garch_vol = garch_forecast * np.sqrt(self.taiwan_adj.TRADING_DAYS_PER_YEAR)
            
            return float(annual_garch_vol)
            
        except Exception:
            return np.nan


class TaiwanVIXCalculator(TechnicalFactorCalculator):
    """Calculate Taiwan market volatility index equivalent."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="taiwan_vix",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="Taiwan market volatility index proxy",
            lookback_days=60,  # Need data for market volatility
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=60,
            expected_ic=0.05,
            expected_turnover=0.2
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate Taiwan VIX equivalent factors."""
        
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
        
        # Calculate market-wide volatility proxy
        market_volatility = self._calculate_market_volatility(adjusted_returns)
        
        factor_values = {}
        
        for symbol in symbols:
            if symbol not in adjusted_returns.columns:
                continue
            
            symbol_returns = adjusted_returns[symbol].dropna()
            
            if len(symbol_returns) < 20:
                continue
            
            try:
                vix_factor = self._calculate_vix_factor(symbol_returns, market_volatility)
                if not np.isnan(vix_factor):
                    factor_values[symbol] = vix_factor
                    
            except Exception as e:
                self.logger.warning(f"Error calculating VIX factor for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_market_volatility(self, returns_df: pd.DataFrame) -> pd.Series:
        """Calculate market-wide volatility measure."""
        
        # Equal-weighted market return
        market_returns = returns_df.mean(axis=1, skipna=True)
        
        # Rolling market volatility
        market_vol = market_returns.rolling(window=20, min_periods=10).std() * np.sqrt(self.taiwan_adj.TRADING_DAYS_PER_YEAR)
        
        return market_vol
    
    def _calculate_vix_factor(self, returns: pd.Series, 
                            market_volatility: pd.Series) -> float:
        """Calculate VIX-like factor for individual stock."""
        
        if len(returns) < 20 or market_volatility.empty:
            return np.nan
        
        # Stock's volatility relative to market
        stock_vol = self._calculate_rolling_volatility(returns, 20)
        
        if stock_vol.empty:
            return np.nan
        
        current_stock_vol = stock_vol.iloc[-1]
        
        # Get corresponding market volatility
        aligned_market_vol = market_volatility.reindex(stock_vol.index, method='ffill')
        current_market_vol = aligned_market_vol.iloc[-1]
        
        if np.isnan(current_stock_vol) or np.isnan(current_market_vol) or current_market_vol == 0:
            return np.nan
        
        # Beta-adjusted volatility
        # Calculate beta using recent data
        stock_returns_recent = returns.tail(60)
        market_returns_recent = returns_df.mean(axis=1, skipna=True).reindex(stock_returns_recent.index)
        
        if len(stock_returns_recent) >= 20 and len(market_returns_recent) >= 20:
            aligned_data = pd.DataFrame({
                'stock': stock_returns_recent,
                'market': market_returns_recent
            }).dropna()
            
            if len(aligned_data) >= 20:
                beta = np.cov(aligned_data['stock'], aligned_data['market'])[0, 1] / np.var(aligned_data['market'])
                if np.isnan(beta):
                    beta = 1.0
            else:
                beta = 1.0
        else:
            beta = 1.0
        
        # Idiosyncratic volatility
        systematic_vol = abs(beta) * current_market_vol
        idiosyncratic_vol = np.sqrt(max(0, current_stock_vol**2 - systematic_vol**2))
        
        # VIX-like factor: combines systematic risk and idiosyncratic risk
        vix_factor = (
            0.6 * (current_market_vol / 0.2 - 1.0) +  # Market volatility component (normalized by 20%)
            0.4 * (idiosyncratic_vol / 0.3 - 1.0)      # Idiosyncratic component (normalized by 30%)
        )
        
        return float(vix_factor)


class VolatilityRiskPremiumCalculator(TechnicalFactorCalculator):
    """Calculate volatility risk premium factors."""
    
    def __init__(self, pit_engine, taiwan_adjustments: TaiwanMarketAdjustments):
        metadata = FactorMetadata(
            name="vol_risk_premium",
            category=FactorCategory.TECHNICAL,
            frequency=FactorFrequency.DAILY,
            description="Volatility risk premium and term structure",
            lookback_days=100,  # Need sufficient data
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=100,
            expected_ic=0.025,
            expected_turnover=0.3
        )
        super().__init__(pit_engine, metadata)
        self.taiwan_adj = taiwan_adjustments
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict] = None) -> FactorResult:
        """Calculate volatility risk premium factors."""
        
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
            if symbol not in adjusted_returns.columns:
                continue
            
            symbol_returns = adjusted_returns[symbol].dropna()
            
            if len(symbol_returns) < 30:
                continue
            
            try:
                vol_risk_premium = self._calculate_vol_risk_premium(symbol_returns)
                if not np.isnan(vol_risk_premium):
                    factor_values[symbol] = vol_risk_premium
                    
            except Exception as e:
                self.logger.warning(f"Error calculating vol risk premium for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_vol_risk_premium(self, returns: pd.Series) -> float:
        """Calculate volatility risk premium factor."""
        
        # Volatility term structure proxy
        vol_short = self._calculate_rolling_volatility(returns, 5)
        vol_medium = self._calculate_rolling_volatility(returns, 20)
        vol_long = self._calculate_rolling_volatility(returns, 60)
        
        if vol_short.empty or vol_medium.empty or vol_long.empty:
            return np.nan
        
        current_vol_short = vol_short.iloc[-1]
        current_vol_medium = vol_medium.iloc[-1]
        current_vol_long = vol_long.iloc[-1]
        
        if any(np.isnan([current_vol_short, current_vol_medium, current_vol_long])):
            return np.nan
        
        # Term structure slope
        vol_term_structure = (current_vol_long - current_vol_short) / current_vol_short
        
        # Volatility of volatility (vol clustering)
        vol_changes = vol_short.pct_change().dropna()
        vol_of_vol = vol_changes.std() if len(vol_changes) > 10 else 0
        
        # Risk premium proxy: compensation for volatility uncertainty
        # High vol-of-vol and steep term structure suggest higher risk premium
        risk_premium = (
            0.6 * vol_term_structure +
            0.4 * vol_of_vol * 10  # Scale vol-of-vol
        )
        
        return float(risk_premium)