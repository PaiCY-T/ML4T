"""
Volume pattern factor calculations for Taiwan market microstructure analysis.

This module implements 4 core volume pattern measures:
1. Volume-Weighted Momentum - Price momentum weighted by volume
2. Volume Breakout Indicators - Volume surge identification  
3. Relative Volume - Volume relative to historical patterns
4. Volume-Price Correlation - Volume-price relationship analysis

All factors are adapted for Taiwan market characteristics:
- 4.5-hour trading session (09:00-13:30)
- Market segment considerations (main board vs OTC)
- Taiwan holiday and seasonality patterns
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from .base import FactorResult, FactorMetadata, FactorCategory, FactorFrequency
from .microstructure import MicrostructureFactorCalculator, TaiwanMarketSession

# Import dependencies - will be mocked for testing if not available
try:
    from ..data.pipeline.pit_engine import PITQueryEngine
    from ..data.core.temporal import DataType
except ImportError:
    PITQueryEngine = object
    DataType = object

logger = logging.getLogger(__name__)


class VolumeWeightedMomentumCalculator(MicrostructureFactorCalculator):
    """
    Volume-Weighted Momentum Factor Calculator.
    
    Calculates momentum signals weighted by trading volume:
    - VWAP-adjusted momentum over multiple periods
    - Volume-weighted returns calculation
    - On-balance volume (OBV) trends
    - Taiwan session volume pattern normalization
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="volume_weighted_momentum",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Price momentum weighted by trading volume (VWAP-based)",
            lookback_days=252,  # 1 year for momentum calculation
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            expected_ic=0.03,
            expected_turnover=0.25
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate volume-weighted momentum factors."""
        
        # Get price and volume data
        ohlcv_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_data = ohlcv_data[ohlcv_data['symbol'] == symbol].copy()
                
                if symbol_data.empty or len(symbol_data) < 60:
                    continue
                
                # Sort by date
                symbol_data = symbol_data.sort_values('date')
                
                # Calculate VWAP (Volume Weighted Average Price)
                symbol_data['typical_price'] = (
                    symbol_data['high'] + symbol_data['low'] + symbol_data['close']
                ) / 3
                
                symbol_data['vwap'] = self._calculate_vwap(symbol_data)
                
                # Calculate volume-weighted returns
                symbol_data['vw_return'] = self._calculate_volume_weighted_returns(symbol_data)
                
                # Calculate On-Balance Volume (OBV)
                symbol_data['obv'] = self._calculate_obv(symbol_data)
                symbol_data['obv_ma'] = symbol_data['obv'].rolling(window=20, min_periods=10).mean()
                
                # Calculate momentum over multiple periods
                momentum_periods = [21, 63, 126]  # ~1M, 3M, 6M
                momentum_scores = []
                
                for period in momentum_periods:
                    if len(symbol_data) >= period:
                        # VWAP-based momentum
                        current_vwap = symbol_data['vwap'].iloc[-1]
                        past_vwap = symbol_data['vwap'].iloc[-period]
                        
                        if pd.notna(current_vwap) and pd.notna(past_vwap) and past_vwap > 0:
                            vwap_momentum = (current_vwap / past_vwap) - 1
                            
                            # Volume-weighted return momentum
                            vw_returns = symbol_data['vw_return'].iloc[-period:].dropna()
                            if len(vw_returns) > 0:
                                cumulative_vw_return = (1 + vw_returns).prod() - 1
                                
                                # OBV trend
                                obv_current = symbol_data['obv'].iloc[-1]
                                obv_past = symbol_data['obv'].iloc[-period]
                                obv_trend = (obv_current - obv_past) / abs(obv_past) if obv_past != 0 else 0
                                
                                # Combine signals with period weighting
                                period_weight = 1.0 / period  # Recent periods get higher weight
                                momentum_score = (
                                    0.5 * vwap_momentum +
                                    0.3 * cumulative_vw_return +
                                    0.2 * np.tanh(obv_trend)  # Bounded OBV trend
                                ) * period_weight
                                
                                momentum_scores.append(momentum_score)
                
                # Combine momentum across periods
                if momentum_scores:
                    # Taiwan market adjustment for session length and volatility
                    taiwan_adjustment = self._get_taiwan_volume_adjustment(symbol_data)
                    
                    final_momentum = np.mean(momentum_scores) * (1 + taiwan_adjustment)
                    factor_values[symbol] = final_momentum
                
            except Exception as e:
                logger.warning(f"Error calculating volume-weighted momentum for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = data['typical_price']
        volume = data['volume']
        
        # Rolling VWAP over 20 days
        price_volume = (typical_price * volume).rolling(window=20, min_periods=5).sum()
        total_volume = volume.rolling(window=20, min_periods=5).sum()
        
        return price_volume / total_volume
    
    def _calculate_volume_weighted_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume-weighted daily returns."""
        returns = data['close'].pct_change()
        volume = data['volume']
        
        # Normalize volume for weighting
        volume_normalized = volume / volume.rolling(window=20, min_periods=5).mean()
        
        # Weight returns by relative volume
        return returns * np.sqrt(volume_normalized)  # Square root to reduce noise
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = data['close'].diff()
        volume = data['volume']
        
        obv_change = np.where(price_change > 0, volume, 
                             np.where(price_change < 0, -volume, 0))
        
        return pd.Series(obv_change, index=data.index).cumsum()
    
    def _get_taiwan_volume_adjustment(self, data: pd.DataFrame) -> float:
        """Get Taiwan-specific volume pattern adjustment."""
        # Adjust for Taiwan market session (4.5 hours vs global 6.5 hours)
        session_adjustment = (4.5 / 6.5) - 1  # Negative adjustment for shorter session
        
        # Adjust for volume concentration patterns in Taiwan market
        recent_volume = data['volume'].tail(10).mean()
        historical_volume = data['volume'].mean()
        
        volume_concentration = (recent_volume / historical_volume - 1) * 0.1 if historical_volume > 0 else 0
        
        return session_adjustment + volume_concentration


class VolumeBreakoutCalculator(MicrostructureFactorCalculator):
    """
    Volume Breakout Indicator Calculator.
    
    Identifies volume surge patterns:
    - Volume spikes relative to historical norms
    - Breakout confirmation signals
    - Volume-price breakout coordination
    - Taiwan market opening/closing pattern analysis
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="volume_breakout",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Volume breakout and surge identification indicators",
            lookback_days=126,  # ~6 months for breakout analysis
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            expected_ic=0.04,
            expected_turnover=0.30
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate volume breakout indicators."""
        
        # Get OHLCV data
        ohlcv_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_data = ohlcv_data[ohlcv_data['symbol'] == symbol].copy()
                
                if symbol_data.empty or len(symbol_data) < 60:
                    continue
                
                # Sort by date
                symbol_data = symbol_data.sort_values('date')
                
                # Calculate volume statistics
                symbol_data['volume_ma20'] = symbol_data['volume'].rolling(window=20).mean()
                symbol_data['volume_ma60'] = symbol_data['volume'].rolling(window=60).mean()
                symbol_data['volume_std20'] = symbol_data['volume'].rolling(window=20).std()
                
                # Calculate volume ratio and z-score
                symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_ma20']
                symbol_data['volume_zscore'] = (
                    (symbol_data['volume'] - symbol_data['volume_ma20']) / 
                    symbol_data['volume_std20']
                )
                
                # Price movement metrics
                symbol_data['price_change_pct'] = symbol_data['close'].pct_change()
                symbol_data['high_low_range'] = (
                    (symbol_data['high'] - symbol_data['low']) / symbol_data['close']
                )
                
                # Calculate breakout scores
                recent_data = symbol_data.tail(5)  # Last 5 days
                
                if len(recent_data) >= 5:
                    # Volume surge detection
                    max_volume_ratio = recent_data['volume_ratio'].max()
                    max_volume_zscore = recent_data['volume_zscore'].max()
                    
                    # Price breakout coordination  
                    price_momentum = recent_data['price_change_pct'].sum()
                    avg_range = recent_data['high_low_range'].mean()
                    
                    # Volume-price coordination score
                    volume_price_corr = self._calculate_volume_price_correlation(symbol_data.tail(20))
                    
                    # Taiwan-specific breakout patterns
                    taiwan_pattern_score = self._analyze_taiwan_breakout_patterns(recent_data)
                    
                    # Combine breakout indicators
                    breakout_score = (
                        0.3 * np.tanh(max_volume_ratio - 1.5) +      # Volume surge above 1.5x normal
                        0.2 * np.tanh(max_volume_zscore / 2) +       # Statistical significance
                        0.2 * np.tanh(abs(price_momentum) * 10) +    # Price movement coordination
                        0.2 * volume_price_corr +                   # Volume-price coordination  
                        0.1 * taiwan_pattern_score                  # Taiwan-specific patterns
                    )
                    
                    factor_values[symbol] = breakout_score
                
            except Exception as e:
                logger.warning(f"Error calculating volume breakout for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_volume_price_correlation(self, data: pd.DataFrame) -> float:
        """Calculate volume-price correlation over recent period."""
        if len(data) < 10:
            return 0.0
        
        volume_changes = data['volume'].pct_change().dropna()
        price_changes = data['close'].pct_change().dropna()
        
        if len(volume_changes) == len(price_changes) and len(volume_changes) >= 10:
            try:
                correlation, _ = stats.pearsonr(volume_changes, abs(price_changes))
                return correlation if not np.isnan(correlation) else 0.0
            except:
                return 0.0
        
        return 0.0
    
    def _analyze_taiwan_breakout_patterns(self, recent_data: pd.DataFrame) -> float:
        """Analyze Taiwan-specific breakout patterns."""
        # Look for patterns specific to Taiwan market:
        # 1. Opening volume surges (first hour effects)
        # 2. Pre-close volume spikes 
        # 3. Monthly/quarterly rebalancing effects
        
        # Simplified pattern: volume consistency during breakouts
        volume_consistency = 1.0 - recent_data['volume'].std() / recent_data['volume'].mean()
        return np.clip(volume_consistency, -0.5, 0.5)


class RelativeVolumeCalculator(MicrostructureFactorCalculator):
    """
    Relative Volume Factor Calculator.
    
    Measures volume relative to historical patterns:
    - Volume percentiles relative to history
    - Seasonal volume adjustments
    - Day-of-week and time-of-day patterns
    - Taiwan market calendar adjustments
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="relative_volume",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Volume relative to historical patterns and seasonal trends",
            lookback_days=252,  # 1 year for seasonal patterns
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            expected_ic=0.02,
            expected_turnover=0.20
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate relative volume factors."""
        
        # Get OHLCV data
        ohlcv_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_data = ohlcv_data[ohlcv_data['symbol'] == symbol].copy()
                
                if symbol_data.empty or len(symbol_data) < 60:
                    continue
                
                # Sort by date and add date features
                symbol_data = symbol_data.sort_values('date')
                symbol_data['day_of_week'] = pd.to_datetime(symbol_data['date']).dt.dayofweek
                symbol_data['month'] = pd.to_datetime(symbol_data['date']).dt.month
                
                # Calculate volume percentiles
                current_volume = symbol_data['volume'].iloc[-1]
                historical_volumes = symbol_data['volume'].dropna()
                
                if len(historical_volumes) >= 60:
                    volume_percentile = (
                        (historical_volumes < current_volume).sum() / len(historical_volumes)
                    )
                    
                    # Seasonal adjustments
                    seasonal_factor = self._calculate_seasonal_adjustment(
                        symbol_data, as_of_date
                    )
                    
                    # Day-of-week adjustment
                    dow_factor = self._calculate_dow_adjustment(symbol_data, as_of_date)
                    
                    # Taiwan market specific adjustments
                    taiwan_calendar_factor = self._calculate_taiwan_calendar_adjustment(as_of_date)
                    
                    # Calculate rolling relative volume metrics
                    symbol_data['volume_ma10'] = symbol_data['volume'].rolling(window=10).mean()
                    symbol_data['volume_ma30'] = symbol_data['volume'].rolling(window=30).mean()
                    
                    recent_vs_short = (
                        symbol_data['volume'].tail(3).mean() / symbol_data['volume_ma10'].iloc[-1] 
                        if symbol_data['volume_ma10'].iloc[-1] > 0 else 1
                    )
                    
                    recent_vs_long = (
                        symbol_data['volume_ma10'].iloc[-1] / symbol_data['volume_ma30'].iloc[-1]
                        if symbol_data['volume_ma30'].iloc[-1] > 0 else 1
                    )
                    
                    # Combine relative volume measures
                    factor_value = (
                        0.4 * (volume_percentile - 0.5) * 2 +        # Center around 0, scale to [-1,1]
                        0.2 * np.tanh(recent_vs_short - 1) +         # Recent vs short-term
                        0.2 * np.tanh(recent_vs_long - 1) +          # Short vs long-term
                        0.1 * seasonal_factor +                     # Seasonal adjustment
                        0.05 * dow_factor +                         # Day-of-week
                        0.05 * taiwan_calendar_factor               # Taiwan calendar
                    )
                    
                    factor_values[symbol] = factor_value
                
            except Exception as e:
                logger.warning(f"Error calculating relative volume for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_seasonal_adjustment(self, data: pd.DataFrame, as_of_date: date) -> float:
        """Calculate seasonal volume patterns."""
        current_month = as_of_date.month
        
        # Group by month and calculate average volume
        monthly_volumes = data.groupby('month')['volume'].mean()
        
        if len(monthly_volumes) >= 6:  # Need reasonable sample
            overall_avg = monthly_volumes.mean()
            current_month_avg = monthly_volumes.get(current_month, overall_avg)
            
            seasonal_factor = (current_month_avg / overall_avg - 1) if overall_avg > 0 else 0
            return np.clip(seasonal_factor, -0.3, 0.3)
        
        return 0.0
    
    def _calculate_dow_adjustment(self, data: pd.DataFrame, as_of_date: date) -> float:
        """Calculate day-of-week volume patterns."""
        current_dow = as_of_date.weekday()  # Monday=0, Sunday=6
        
        # Group by day of week
        dow_volumes = data.groupby('day_of_week')['volume'].mean()
        
        if len(dow_volumes) >= 3:
            overall_avg = dow_volumes.mean()
            current_dow_avg = dow_volumes.get(current_dow, overall_avg)
            
            dow_factor = (current_dow_avg / overall_avg - 1) if overall_avg > 0 else 0
            return np.clip(dow_factor, -0.2, 0.2)
        
        return 0.0
    
    def _calculate_taiwan_calendar_adjustment(self, as_of_date: date) -> float:
        """Calculate Taiwan market calendar effects."""
        # Taiwan-specific calendar effects:
        # 1. Lunar New Year period (lower volume)
        # 2. Quarter-end rebalancing (higher volume)
        # 3. Typhoon season volume impacts
        
        month = as_of_date.month
        day = as_of_date.day
        
        # Quarter-end effect (positive)
        if month in [3, 6, 9, 12] and day >= 25:
            return 0.1
        
        # Lunar New Year period effect (negative) - approximate
        if month in [1, 2] and day <= 20:
            return -0.1
        
        return 0.0


class VolumePriceCorrelationCalculator(MicrostructureFactorCalculator):
    """
    Volume-Price Correlation Factor Calculator.
    
    Analyzes volume-price relationships:
    - Correlation between volume and |price_change|
    - Volume confirmation of price movements
    - Volume-weighted price trend strength
    - Taiwan market volume clustering patterns
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="volume_price_correlation", 
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Volume-price relationship and trend confirmation analysis",
            lookback_days=126,  # ~6 months for correlation analysis
            data_requirements=[DataType.OHLCV],
            taiwan_specific=True,
            expected_ic=0.03,
            expected_turnover=0.15
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate volume-price correlation factors."""
        
        # Get OHLCV data
        ohlcv_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_data = ohlcv_data[ohlcv_data['symbol'] == symbol].copy()
                
                if symbol_data.empty or len(symbol_data) < 60:
                    continue
                
                # Sort by date
                symbol_data = symbol_data.sort_values('date')
                
                # Calculate price changes and volume changes
                symbol_data['price_change'] = symbol_data['close'].pct_change()
                symbol_data['abs_price_change'] = abs(symbol_data['price_change'])
                symbol_data['volume_change'] = symbol_data['volume'].pct_change()
                
                # Calculate volume-price correlations over different windows
                correlations = []
                
                for window in [20, 40, 60]:  # Different time horizons
                    if len(symbol_data) >= window:
                        recent_data = symbol_data.tail(window)
                        
                        # Volume vs absolute price change correlation
                        vol_price_corr = self._safe_correlation(
                            recent_data['volume'], 
                            recent_data['abs_price_change']
                        )
                        
                        # Volume change vs price change correlation  
                        vol_change_corr = self._safe_correlation(
                            recent_data['volume_change'].dropna(),
                            recent_data['abs_price_change'].dropna()
                        )
                        
                        # Weight by window (shorter windows get more weight)
                        weight = 1.0 / window
                        correlations.append((vol_price_corr + vol_change_corr) * weight)
                
                # Calculate trend confirmation score
                trend_confirmation = self._calculate_trend_confirmation(symbol_data)
                
                # Volume clustering analysis (Taiwan-specific)
                clustering_score = self._analyze_volume_clustering(symbol_data)
                
                # Price-volume coordination during moves
                coordination_score = self._calculate_move_coordination(symbol_data)
                
                # Combine all volume-price relationship measures
                if correlations:
                    avg_correlation = np.mean(correlations)
                    
                    factor_value = (
                        0.4 * avg_correlation +
                        0.3 * trend_confirmation + 
                        0.2 * clustering_score +
                        0.1 * coordination_score
                    )
                    
                    factor_values[symbol] = factor_value
                
            except Exception as e:
                logger.warning(f"Error calculating volume-price correlation for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _safe_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Safely calculate correlation with error handling."""
        try:
            # Remove NaN values
            combined = pd.concat([x, y], axis=1).dropna()
            if len(combined) >= 10:  # Minimum sample size
                correlation, _ = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
                return correlation if not np.isnan(correlation) else 0.0
        except:
            pass
        return 0.0
    
    def _calculate_trend_confirmation(self, data: pd.DataFrame) -> float:
        """Calculate how well volume confirms price trends."""
        if len(data) < 20:
            return 0.0
        
        recent_data = data.tail(20)
        
        # Calculate trend strength
        price_trend = recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1
        
        # Calculate volume support for the trend
        if price_trend > 0.02:  # Uptrend
            # Volume should be higher during up days
            up_days = recent_data[recent_data['price_change'] > 0]
            down_days = recent_data[recent_data['price_change'] < 0]
            
            if len(up_days) > 0 and len(down_days) > 0:
                up_volume_avg = up_days['volume'].mean()
                down_volume_avg = down_days['volume'].mean()
                confirmation = (up_volume_avg / down_volume_avg - 1) * 0.5
                return np.clip(confirmation, -0.5, 0.5)
                
        elif price_trend < -0.02:  # Downtrend  
            # Volume should be higher during down days
            up_days = recent_data[recent_data['price_change'] > 0]
            down_days = recent_data[recent_data['price_change'] < 0]
            
            if len(up_days) > 0 and len(down_days) > 0:
                up_volume_avg = up_days['volume'].mean()
                down_volume_avg = down_days['volume'].mean()
                confirmation = (down_volume_avg / up_volume_avg - 1) * 0.5
                return np.clip(confirmation, -0.5, 0.5)
        
        return 0.0
    
    def _analyze_volume_clustering(self, data: pd.DataFrame) -> float:
        """Analyze volume clustering patterns (Taiwan-specific)."""
        if len(data) < 30:
            return 0.0
        
        recent_volumes = data.tail(20)['volume'].values
        
        # Calculate coefficient of variation
        volume_cv = np.std(recent_volumes) / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 0
        
        # Taiwan market tends to have clustered volume patterns
        # Lower CV indicates more consistent volume (positive for Taiwan market)
        clustering_score = -np.tanh(volume_cv - 0.5)  # Invert and bound
        
        return clustering_score
    
    def _calculate_move_coordination(self, data: pd.DataFrame) -> float:
        """Calculate price-volume coordination during significant moves."""
        if len(data) < 20:
            return 0.0
        
        recent_data = data.tail(20)
        
        # Identify significant price moves (>2% daily change)
        big_moves = recent_data[abs(recent_data['price_change']) > 0.02]
        
        if len(big_moves) < 3:
            return 0.0
        
        # Calculate average volume during big moves vs normal days
        normal_days = recent_data[abs(recent_data['price_change']) <= 0.02]
        
        if len(normal_days) > 0:
            big_move_volume = big_moves['volume'].mean()
            normal_volume = normal_days['volume'].mean()
            
            coordination = (big_move_volume / normal_volume - 1) * 0.3 if normal_volume > 0 else 0
            return np.clip(coordination, -0.3, 0.3)
        
        return 0.0


class VolumePatternFactors:
    """Container for all volume pattern factor calculators."""
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.calculators = {
            'volume_weighted_momentum': VolumeWeightedMomentumCalculator(pit_engine),
            'volume_breakout': VolumeBreakoutCalculator(pit_engine),
            'relative_volume': RelativeVolumeCalculator(pit_engine),
            'volume_price_correlation': VolumePriceCorrelationCalculator(pit_engine)
        }
    
    def get_all_calculators(self) -> Dict[str, MicrostructureFactorCalculator]:
        """Get all volume pattern factor calculators."""
        return self.calculators.copy()
    
    def calculate_all_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, FactorResult]:
        """Calculate all volume pattern factors."""
        results = {}
        
        for name, calculator in self.calculators.items():
            try:
                result = calculator.calculate(symbols, as_of_date)
                results[name] = result
                logger.info(f"Calculated {name}: {result.coverage:.1%} coverage")
            except Exception as e:
                logger.error(f"Error calculating volume pattern factor {name}: {e}")
                continue
        
        return results