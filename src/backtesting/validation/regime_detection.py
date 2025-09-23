"""
Regime Detection and Stability Testing for Walk-Forward Validation.

This module implements regime detection algorithms and stability testing
for validating model performance across different market regimes.

Key Features:
- Market regime detection (bull, bear, volatile, stable)
- Hidden Markov Model for regime identification  
- Stability testing across regime transitions
- Taiwan market-specific regime characteristics
- Performance attribution by regime
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import warnings

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.models.taiwan_market import TaiwanTradingCalendar, TaiwanMarketCode
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"                  # Strong upward trend, low volatility
    BEAR = "bear"                  # Strong downward trend, moderate volatility  
    VOLATILE_UP = "volatile_up"    # Upward trend, high volatility
    VOLATILE_DOWN = "volatile_down" # Downward trend, high volatility
    SIDEWAYS = "sideways"          # No clear trend, low volatility
    CRISIS = "crisis"              # Extreme volatility, high uncertainty
    UNKNOWN = "unknown"            # Regime not determined


class RegimeIndicator(Enum):
    """Indicators used for regime detection."""
    RETURN = "return"              # Price returns
    VOLATILITY = "volatility"      # Rolling volatility
    VOLUME = "volume"              # Trading volume
    VIX = "vix"                   # Volatility index
    MOMENTUM = "momentum"          # Price momentum
    CORRELATION = "correlation"    # Cross-asset correlation


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Detection parameters
    lookback_window: int = 252     # Days for regime detection (1 year)
    min_regime_length: int = 20    # Minimum regime duration in days
    stability_window: int = 60     # Days for stability testing
    
    # Regime thresholds
    volatility_threshold_low: float = 0.15   # Low volatility threshold
    volatility_threshold_high: float = 0.30  # High volatility threshold
    return_threshold_positive: float = 0.05  # Positive return threshold
    return_threshold_negative: float = -0.05 # Negative return threshold
    
    # HMM parameters
    n_regimes: int = 4             # Number of hidden states
    covariance_type: str = "full"  # HMM covariance type
    max_iter: int = 100           # Maximum HMM iterations
    random_state: int = 42        # Random seed
    
    # Taiwan market specifics
    use_taiwan_benchmark: bool = True  # Use TAIEX as benchmark
    consider_foreign_flows: bool = True # Consider foreign investment flows
    adjust_for_lunar_new_year: bool = True # Adjust for CNY effects
    
    def __post_init__(self):
        """Validate configuration."""
        if self.lookback_window <= 0:
            raise ValueError("Lookback window must be positive")
        if self.n_regimes < 2:
            raise ValueError("Number of regimes must be >= 2")


@dataclass  
class RegimeState:
    """State information for a market regime."""
    regime: MarketRegime
    start_date: date
    end_date: Optional[date]
    confidence: float              # Confidence in regime classification
    
    # Regime characteristics
    mean_return: float
    volatility: float
    skewness: float
    kurtosis: float
    max_drawdown: float
    
    # Taiwan market specifics
    foreign_flow: Optional[float] = None
    sector_rotation: Optional[Dict[str, float]] = None
    
    def duration_days(self) -> int:
        """Calculate regime duration in days."""
        if self.end_date is None:
            return 0
        return (self.end_date - self.start_date).days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime': self.regime.value,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'confidence': self.confidence,
            'duration_days': self.duration_days(),
            'mean_return': self.mean_return,
            'volatility': self.volatility,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'max_drawdown': self.max_drawdown,
            'foreign_flow': self.foreign_flow,
            'sector_rotation': self.sector_rotation
        }


@dataclass
class StabilityTestResult:
    """Results from model stability testing across regimes."""
    test_name: str
    regime_from: MarketRegime
    regime_to: MarketRegime
    test_statistic: float
    p_value: float
    is_stable: bool
    confidence_level: float = 0.05
    
    # Performance metrics
    performance_before: Optional[Dict[str, float]] = None
    performance_after: Optional[Dict[str, float]] = None
    performance_change: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'regime_from': self.regime_from.value,
            'regime_to': self.regime_to.value,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'is_stable': self.is_stable,
            'confidence_level': self.confidence_level,
            'performance_before': self.performance_before,
            'performance_after': self.performance_after,
            'performance_change': self.performance_change
        }


class RegimeDetector:
    """
    Market regime detection using Hidden Markov Models and statistical indicators.
    
    Implements multiple approaches for regime identification:
    - Rule-based classification using return/volatility thresholds
    - Hidden Markov Model with multiple features
    - Taiwan market-specific regime characteristics
    """
    
    def __init__(
        self,
        config: RegimeConfig,
        temporal_store: TemporalStore,
        pit_engine: Optional[PointInTimeEngine] = None,
        benchmark_symbol: str = "^TWII"  # TAIEX index
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine or PointInTimeEngine(temporal_store)
        self.benchmark_symbol = benchmark_symbol
        
        # Initialize HMM model
        self.hmm_model = None
        self.regime_history: List[RegimeState] = []
        
        logger.info(f"RegimeDetector initialized with {config.n_regimes} regimes")
    
    def detect_regimes(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[RegimeState]:
        """
        Detect market regimes for the given period.
        
        Args:
            start_date: Start date for regime detection
            end_date: End date for regime detection
            symbols: Optional list of symbols (uses benchmark if None)
            
        Returns:
            List of regime states with timing and characteristics
        """
        logger.info(f"Detecting regimes from {start_date} to {end_date}")
        
        # Use benchmark symbol if no symbols provided
        if symbols is None:
            symbols = [self.benchmark_symbol]
        
        # Get market data for regime detection
        market_data = self._get_market_data(symbols, start_date, end_date)
        
        if market_data.empty:
            logger.warning("No market data available for regime detection")
            return []
        
        # Calculate regime indicators
        indicators = self._calculate_regime_indicators(market_data)
        
        # Detect regimes using HMM
        regime_sequence = self._detect_regimes_hmm(indicators)
        
        # Convert to regime states
        regime_states = self._convert_to_regime_states(
            regime_sequence, indicators, start_date
        )
        
        # Apply Taiwan market adjustments
        if self.config.adjust_for_lunar_new_year:
            regime_states = self._adjust_for_taiwan_calendar(regime_states)
        
        self.regime_history = regime_states
        logger.info(f"Detected {len(regime_states)} regime periods")
        
        return regime_states
    
    def classify_current_regime(
        self,
        as_of_date: date,
        symbols: Optional[List[str]] = None
    ) -> Optional[RegimeState]:
        """
        Classify the current market regime as of the given date.
        
        Args:
            as_of_date: Date for regime classification
            symbols: Optional list of symbols
            
        Returns:
            Current regime state or None if cannot be determined
        """
        # Look back to get sufficient data
        start_date = as_of_date - timedelta(days=self.config.lookback_window)
        
        try:
            # Get recent data
            market_data = self._get_market_data(
                symbols or [self.benchmark_symbol], 
                start_date, 
                as_of_date
            )
            
            if len(market_data) < self.config.min_regime_length:
                return None
            
            # Calculate indicators for recent period
            indicators = self._calculate_regime_indicators(market_data)
            
            # Get the most recent regime classification
            if self.hmm_model is not None:
                recent_features = indicators.iloc[-self.config.min_regime_length:].values
                regime_probs = self.hmm_model.predict_proba(recent_features)
                current_regime_idx = np.argmax(regime_probs[-1])
                confidence = np.max(regime_probs[-1])
            else:
                # Fallback to rule-based classification
                current_regime_idx, confidence = self._classify_regime_rule_based(
                    indicators.iloc[-1]
                )
            
            # Create regime state
            regime_type = self._map_regime_index_to_type(current_regime_idx)
            
            regime_state = RegimeState(
                regime=regime_type,
                start_date=as_of_date,
                end_date=None,  # Current regime
                confidence=confidence,
                mean_return=float(indicators['return'].iloc[-self.config.min_regime_length:].mean()),
                volatility=float(indicators['volatility'].iloc[-self.config.min_regime_length:].mean()),
                skewness=float(indicators['return'].iloc[-self.config.min_regime_length:].skew()),
                kurtosis=float(indicators['return'].iloc[-self.config.min_regime_length:].kurtosis()),
                max_drawdown=self._calculate_max_drawdown(
                    market_data['price'].iloc[-self.config.min_regime_length:]
                )
            )
            
            return regime_state
            
        except Exception as e:
            logger.error(f"Failed to classify current regime: {e}")
            return None
    
    def _get_market_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Get market data for regime detection."""
        try:
            # Query price and volume data
            query = PITQuery(
                symbols=symbols,
                as_of_date=end_date,
                data_types=[DataType.PRICE, DataType.VOLUME],
                start_date=start_date,
                end_date=end_date
            )
            
            data = self.pit_engine.query(query)
            
            # Convert to DataFrame
            df_list = []
            for symbol in symbols:
                if symbol in data:
                    symbol_data = []
                    for price_val, volume_val in zip(
                        data[symbol].get(DataType.PRICE, []),
                        data[symbol].get(DataType.VOLUME, [])
                    ):
                        symbol_data.append({
                            'date': price_val.value_date,
                            'symbol': symbol,
                            'price': float(price_val.value),
                            'volume': float(volume_val.value) if volume_val else 0
                        })
                    
                    if symbol_data:
                        df_list.append(pd.DataFrame(symbol_data))
            
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                df['date'] = pd.to_datetime(df['date'])
                
                # For multiple symbols, use equal-weighted average or primary symbol
                if len(symbols) == 1:
                    return df.set_index('date').sort_index()
                else:
                    # Equal-weighted average for multiple symbols
                    grouped = df.groupby('date').agg({
                        'price': 'mean',
                        'volume': 'sum'
                    }).sort_index()
                    return grouped
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return pd.DataFrame()
    
    def _calculate_regime_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators used for regime detection."""
        if market_data.empty:
            return pd.DataFrame()
        
        indicators = pd.DataFrame(index=market_data.index)
        
        # Price returns
        indicators['return'] = market_data['price'].pct_change()
        
        # Rolling volatility (20-day)
        indicators['volatility'] = indicators['return'].rolling(20).std() * np.sqrt(252)
        
        # Volume indicators
        indicators['volume'] = market_data['volume']
        indicators['volume_ma'] = indicators['volume'].rolling(20).mean()
        indicators['volume_ratio'] = indicators['volume'] / indicators['volume_ma']
        
        # Momentum indicators
        indicators['momentum_10'] = market_data['price'].pct_change(10)
        indicators['momentum_20'] = market_data['price'].pct_change(20)
        
        # Trend indicators
        indicators['sma_20'] = market_data['price'].rolling(20).mean()
        indicators['sma_50'] = market_data['price'].rolling(50).mean()
        indicators['trend'] = (indicators['sma_20'] - indicators['sma_50']) / indicators['sma_50']
        
        # Volatility regimes
        indicators['vol_regime'] = pd.cut(
            indicators['volatility'],
            bins=[0, self.config.volatility_threshold_low, 
                  self.config.volatility_threshold_high, np.inf],
            labels=['low', 'medium', 'high']
        )
        
        # Drop NaN values
        indicators = indicators.dropna()
        
        return indicators
    
    def _detect_regimes_hmm(self, indicators: pd.DataFrame) -> np.ndarray:
        """Detect regimes using Hidden Markov Model."""
        if len(indicators) < self.config.min_regime_length:
            return np.array([])
        
        # Select features for HMM
        feature_cols = ['return', 'volatility', 'momentum_10', 'trend', 'volume_ratio']
        features = indicators[feature_cols].values
        
        # Handle missing values
        if np.any(np.isnan(features)):
            features = pd.DataFrame(features).fillna(method='ffill').values
        
        try:
            # Fit Gaussian Mixture Model (simpler than full HMM)
            self.hmm_model = GaussianMixture(
                n_components=self.config.n_regimes,
                covariance_type=self.config.covariance_type,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            )
            
            regime_sequence = self.hmm_model.fit_predict(features)
            
            # Apply minimum regime length constraint
            regime_sequence = self._smooth_regime_sequence(
                regime_sequence, self.config.min_regime_length
            )
            
            return regime_sequence
            
        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}")
            # Fallback to rule-based classification
            return self._detect_regimes_rule_based(indicators)
    
    def _detect_regimes_rule_based(self, indicators: pd.DataFrame) -> np.ndarray:
        """Fallback rule-based regime detection."""
        regimes = []
        
        for _, row in indicators.iterrows():
            regime_idx, _ = self._classify_regime_rule_based(row)
            regimes.append(regime_idx)
        
        return np.array(regimes)
    
    def _classify_regime_rule_based(self, indicators: pd.Series) -> Tuple[int, float]:
        """Classify regime based on simple rules."""
        return_val = indicators['return']
        volatility = indicators['volatility']
        
        confidence = 0.7  # Default confidence for rule-based
        
        # Rule-based classification
        if volatility > self.config.volatility_threshold_high:
            if return_val > self.config.return_threshold_positive:
                return 2, confidence  # Volatile up
            elif return_val < self.config.return_threshold_negative:
                return 3, confidence  # Volatile down / Crisis
            else:
                return 3, confidence  # Crisis
        elif volatility < self.config.volatility_threshold_low:
            if return_val > self.config.return_threshold_positive:
                return 0, confidence  # Bull
            elif return_val < self.config.return_threshold_negative:
                return 1, confidence  # Bear
            else:
                return 4, confidence  # Sideways (if implemented)
        else:
            # Medium volatility
            if return_val > self.config.return_threshold_positive:
                return 0, confidence  # Bull
            elif return_val < self.config.return_threshold_negative:
                return 1, confidence  # Bear
            else:
                return 4, confidence  # Sideways
    
    def _smooth_regime_sequence(
        self, 
        regime_sequence: np.ndarray, 
        min_length: int
    ) -> np.ndarray:
        """Smooth regime sequence to enforce minimum regime length."""
        if len(regime_sequence) == 0:
            return regime_sequence
        
        smoothed = regime_sequence.copy()
        
        # Find regime changes
        changes = np.where(np.diff(regime_sequence) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(regime_sequence)]])
        
        # Merge short regimes with adjacent regimes
        for i, (start, end) in enumerate(zip(starts, ends)):
            if end - start < min_length:
                # Find the most common adjacent regime
                if i > 0:
                    prev_regime = regime_sequence[starts[i-1]]
                else:
                    prev_regime = None
                
                if i < len(starts) - 1:
                    next_regime = regime_sequence[starts[i+1]]
                else:
                    next_regime = None
                
                # Merge with previous or next regime
                if prev_regime is not None:
                    smoothed[start:end] = prev_regime
                elif next_regime is not None:
                    smoothed[start:end] = next_regime
        
        return smoothed
    
    def _convert_to_regime_states(
        self,
        regime_sequence: np.ndarray,
        indicators: pd.DataFrame,
        start_date: date
    ) -> List[RegimeState]:
        """Convert regime sequence to RegimeState objects."""
        if len(regime_sequence) == 0:
            return []
        
        regime_states = []
        dates = indicators.index.date
        
        # Find regime boundaries
        changes = np.where(np.diff(regime_sequence) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(regime_sequence)]])
        
        for start_idx, end_idx in zip(starts, ends):
            regime_idx = regime_sequence[start_idx]
            regime_type = self._map_regime_index_to_type(regime_idx)
            
            # Calculate regime characteristics
            regime_indicators = indicators.iloc[start_idx:end_idx]
            regime_returns = regime_indicators['return']
            
            regime_state = RegimeState(
                regime=regime_type,
                start_date=dates[start_idx],
                end_date=dates[end_idx-1] if end_idx < len(dates) else dates[-1],
                confidence=0.8,  # Default confidence
                mean_return=float(regime_returns.mean()),
                volatility=float(regime_returns.std()) * np.sqrt(252),
                skewness=float(regime_returns.skew()),
                kurtosis=float(regime_returns.kurtosis()),
                max_drawdown=self._calculate_max_drawdown(
                    indicators['price'].iloc[start_idx:end_idx] if 'price' in indicators.columns else None
                )
            )
            
            regime_states.append(regime_state)
        
        return regime_states
    
    def _map_regime_index_to_type(self, regime_idx: int) -> MarketRegime:
        """Map regime index to MarketRegime enum."""
        mapping = {
            0: MarketRegime.BULL,
            1: MarketRegime.BEAR,
            2: MarketRegime.VOLATILE_UP,
            3: MarketRegime.VOLATILE_DOWN,
            4: MarketRegime.SIDEWAYS
        }
        return mapping.get(regime_idx, MarketRegime.UNKNOWN)
    
    def _calculate_max_drawdown(self, prices: Optional[pd.Series]) -> float:
        """Calculate maximum drawdown for a price series."""
        if prices is None or len(prices) == 0:
            return 0.0
        
        # Calculate cumulative returns
        if isinstance(prices.iloc[0], (int, float)):
            cumulative = prices / prices.iloc[0]
        else:
            return 0.0
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        return float(drawdown.min())
    
    def _adjust_for_taiwan_calendar(
        self, 
        regime_states: List[RegimeState]
    ) -> List[RegimeState]:
        """Adjust regime detection for Taiwan market calendar events."""
        # This would implement Taiwan-specific adjustments
        # For now, return unchanged
        return regime_states


class StabilityTester:
    """
    Test model stability across regime transitions.
    
    Implements statistical tests to validate that model performance
    remains stable across different market regimes.
    """
    
    def __init__(self, confidence_level: float = 0.05):
        self.confidence_level = confidence_level
    
    def test_performance_stability(
        self,
        regime_states: List[RegimeState],
        performance_data: Dict[date, Dict[str, float]],
        metrics: List[str] = None
    ) -> List[StabilityTestResult]:
        """
        Test performance stability across regime transitions.
        
        Args:
            regime_states: List of detected regime states
            performance_data: Performance metrics by date
            metrics: List of metrics to test
            
        Returns:
            List of stability test results
        """
        if metrics is None:
            metrics = ['return', 'sharpe_ratio', 'max_drawdown']
        
        test_results = []
        
        # Test stability between consecutive regimes
        for i in range(len(regime_states) - 1):
            regime_from = regime_states[i]
            regime_to = regime_states[i + 1]
            
            for metric in metrics:
                result = self._test_regime_transition_stability(
                    regime_from, regime_to, performance_data, metric
                )
                if result:
                    test_results.append(result)
        
        return test_results
    
    def _test_regime_transition_stability(
        self,
        regime_from: RegimeState,
        regime_to: RegimeState,
        performance_data: Dict[date, Dict[str, float]],
        metric: str
    ) -> Optional[StabilityTestResult]:
        """Test stability between two regimes for a specific metric."""
        try:
            # Extract performance data for each regime
            perf_before = self._extract_regime_performance(
                regime_from, performance_data, metric
            )
            perf_after = self._extract_regime_performance(
                regime_to, performance_data, metric
            )
            
            if len(perf_before) < 5 or len(perf_after) < 5:
                return None  # Insufficient data
            
            # Perform two-sample t-test
            t_stat, p_value = stats.ttest_ind(perf_before, perf_after)
            
            is_stable = p_value > self.confidence_level
            
            result = StabilityTestResult(
                test_name=f"{metric}_stability_t_test",
                regime_from=regime_from.regime,
                regime_to=regime_to.regime,
                test_statistic=float(t_stat),
                p_value=float(p_value),
                is_stable=is_stable,
                confidence_level=self.confidence_level,
                performance_before={'mean': np.mean(perf_before), 'std': np.std(perf_before)},
                performance_after={'mean': np.mean(perf_after), 'std': np.std(perf_after)},
                performance_change={
                    'mean_change': np.mean(perf_after) - np.mean(perf_before),
                    'std_change': np.std(perf_after) - np.std(perf_before)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Stability test failed for {metric}: {e}")
            return None
    
    def _extract_regime_performance(
        self,
        regime_state: RegimeState,
        performance_data: Dict[date, Dict[str, float]],
        metric: str
    ) -> List[float]:
        """Extract performance data for a specific regime period."""
        regime_performance = []
        
        current_date = regime_state.start_date
        end_date = regime_state.end_date or date.today()
        
        while current_date <= end_date:
            if current_date in performance_data:
                if metric in performance_data[current_date]:
                    regime_performance.append(performance_data[current_date][metric])
            current_date += timedelta(days=1)
        
        return regime_performance


# Utility functions
def create_default_regime_config(**kwargs) -> RegimeConfig:
    """Create default regime detection configuration."""
    return RegimeConfig(**kwargs)


def create_taiwan_regime_config(**kwargs) -> RegimeConfig:
    """Create regime configuration optimized for Taiwan market."""
    taiwan_defaults = {
        'use_taiwan_benchmark': True,
        'consider_foreign_flows': True,
        'adjust_for_lunar_new_year': True,
        'volatility_threshold_low': 0.12,   # Lower for Taiwan market
        'volatility_threshold_high': 0.25,  # Adjusted for Taiwan
        'lookback_window': 180,             # Shorter for more responsive detection
    }
    taiwan_defaults.update(kwargs)
    return RegimeConfig(**taiwan_defaults)


# Example usage
if __name__ == "__main__":
    print("Regime Detection and Stability Testing demo")
    
    # Create Taiwan-optimized config
    config = create_taiwan_regime_config()
    print(f"Taiwan regime config: {config}")
    
    print("In actual usage, initialize with TemporalStore and run regime detection")