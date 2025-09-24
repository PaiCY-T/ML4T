"""
Information Coefficient Performance Tester for Feature Selection.

Tests feature performance against Information Coefficient targets,
validates predictive power, and ensures statistical significance.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ICTestType(Enum):
    """Information Coefficient test types."""
    SPEARMAN = "spearman"      # Spearman rank correlation (default)
    PEARSON = "pearson"        # Pearson linear correlation  
    KENDALL = "kendall"        # Kendall tau correlation
    COMPOSITE = "composite"    # Weighted average of all


class PerformanceLevel(Enum):
    """Performance level classification."""
    EXCELLENT = "excellent"    # IC > 0.1, highly significant
    GOOD = "good"             # IC > 0.05, significant
    ACCEPTABLE = "acceptable"  # IC > 0.02, weakly significant  
    WEAK = "weak"             # IC > 0.01, not significant
    POOR = "poor"             # IC <= 0.01 or negative


@dataclass
class ICTestConfig:
    """Configuration for IC performance testing."""
    
    # Performance thresholds
    excellent_ic_threshold: float = 0.10      # Excellent IC threshold
    good_ic_threshold: float = 0.05           # Good IC threshold  
    acceptable_ic_threshold: float = 0.02     # Acceptable IC threshold
    weak_ic_threshold: float = 0.01           # Weak IC threshold
    
    # Statistical significance
    significance_level: float = 0.05          # p-value threshold
    min_observations: int = 60                # Minimum observations for test
    confidence_level: float = 0.95            # Confidence level for intervals
    
    # Test parameters
    ic_test_type: ICTestType = ICTestType.SPEARMAN
    rolling_window: int = 252                 # Rolling window for IC calculation
    min_periods: int = 63                     # Minimum periods for rolling calc
    
    # Return calculation
    forward_return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 21])  # Days
    return_calculation: str = "simple"         # "simple" or "log"
    winsorize_returns: bool = True            # Winsorize extreme returns
    winsorize_percentiles: Tuple[float, float] = (0.01, 0.99)
    
    # Cross-validation
    use_time_series_cv: bool = True           # Use time series cross-validation
    cv_folds: int = 5                        # Number of CV folds
    cv_gap: int = 21                         # Gap between train and test (days)
    
    # Regime analysis  
    analyze_regimes: bool = True              # Analyze performance by market regime
    regime_indicators: List[str] = field(default_factory=lambda: ["bull", "bear", "sideways"])
    
    # Taiwan market specific
    taiwan_adjustments: bool = True           # Apply Taiwan market adjustments
    trading_days_per_year: int = 250          # Taiwan trading days
    price_limit_adjustment: bool = True       # Adjust for 10% price limits


@dataclass
class ICTestResult:
    """Result of IC performance test."""
    feature_name: str
    test_type: ICTestType
    
    # Main IC statistics
    ic_value: float
    ic_std: float
    ic_t_stat: float
    ic_p_value: float
    
    # Performance classification
    performance_level: PerformanceLevel
    is_significant: bool
    
    # Time series statistics
    ic_time_series: Optional[pd.Series] = None
    rolling_ic_stats: Optional[Dict[str, float]] = None
    
    # Multi-horizon results
    horizon_results: Optional[Dict[int, Dict[str, float]]] = None
    
    # Cross-validation results
    cv_ic_mean: Optional[float] = None
    cv_ic_std: Optional[float] = None
    cv_stability: Optional[float] = None
    
    # Regime analysis
    regime_performance: Optional[Dict[str, Dict[str, float]]] = None
    
    # Additional metrics
    hit_rate: Optional[float] = None          # Directional accuracy
    information_ratio: Optional[float] = None  # IC / IC_std
    max_ic: Optional[float] = None
    min_ic: Optional[float] = None
    
    # Diagnostics
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    test_timestamp: datetime = field(default_factory=datetime.now)


class ICPerformanceTester:
    """
    Information Coefficient Performance Tester.
    
    Comprehensive testing framework for evaluating feature predictive power:
    1. IC calculation with multiple correlation methods
    2. Statistical significance testing
    3. Rolling IC analysis for stability
    4. Multi-horizon performance evaluation
    5. Cross-validation for robustness
    6. Market regime analysis
    7. Taiwan market-specific adjustments
    """
    
    def __init__(self, config: Optional[ICTestConfig] = None):
        """Initialize IC performance tester.
        
        Args:
            config: Configuration for IC testing
        """
        self.config = config or ICTestConfig()
        self.test_results: Dict[str, ICTestResult] = {}
        
        logger.info("IC Performance Tester initialized")
        logger.info(f"Performance thresholds: Excellent>{self.config.excellent_ic_threshold:.3f}, "
                   f"Good>{self.config.good_ic_threshold:.3f}, "
                   f"Acceptable>{self.config.acceptable_ic_threshold:.3f}")
    
    def test_features(
        self,
        features: List[str],
        feature_data: pd.DataFrame,
        price_data: pd.DataFrame,
        dates: Optional[pd.Series] = None,
        market_data: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.Series] = None
    ) -> Dict[str, ICTestResult]:
        """
        Test features for IC performance.
        
        Args:
            features: List of feature names to test
            feature_data: Feature data with MultiIndex (date, symbol) or DatetimeIndex
            price_data: Price data for return calculation (same index as feature_data)
            dates: Date series if not in index
            market_data: Market data for context
            regime_data: Market regime classification
            
        Returns:
            Dictionary mapping feature names to IC test results
        """
        logger.info(f"Testing {len(features)} features for IC performance")
        
        # Prepare data
        prepared_data = self._prepare_data(feature_data, price_data, dates)
        feature_matrix = prepared_data['features']
        return_matrix = prepared_data['returns']
        
        results = {}
        
        for feature in features:
            if feature not in feature_matrix.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue
            
            logger.debug(f"Testing feature: {feature}")
            
            try:
                # Extract feature and return data
                feature_series = feature_matrix[feature]
                
                # Test for each return horizon
                horizon_results = {}
                for horizon in self.config.forward_return_periods:
                    if f'return_{horizon}d' in return_matrix.columns:
                        return_series = return_matrix[f'return_{horizon}d']
                        
                        # Align data
                        aligned_data = self._align_data(feature_series, return_series)
                        
                        if len(aligned_data) >= self.config.min_observations:
                            ic_result = self._calculate_ic(
                                aligned_data['feature'],
                                aligned_data['return'],
                                horizon,
                                regime_data
                            )
                            horizon_results[horizon] = ic_result
                        else:
                            logger.warning(f"Insufficient data for {feature} at {horizon}d horizon")
                
                if horizon_results:
                    # Create comprehensive test result
                    test_result = self._create_test_result(feature, horizon_results)
                    
                    # Add cross-validation if requested
                    if self.config.use_time_series_cv and len(feature_series.dropna()) >= self.config.min_observations * 2:
                        cv_results = self._perform_cross_validation(
                            feature_series, return_matrix, feature
                        )
                        test_result.cv_ic_mean = cv_results['mean_ic']
                        test_result.cv_ic_std = cv_results['std_ic']
                        test_result.cv_stability = cv_results['stability']
                    
                    results[feature] = test_result
                else:
                    logger.warning(f"No valid horizon results for feature {feature}")
                    
            except Exception as e:
                logger.error(f"Error testing feature {feature}: {e}")
                
                # Create error result
                error_result = ICTestResult(
                    feature_name=feature,
                    test_type=self.config.ic_test_type,
                    ic_value=0.0,
                    ic_std=0.0,
                    ic_t_stat=0.0,
                    ic_p_value=1.0,
                    performance_level=PerformanceLevel.POOR,
                    is_significant=False,
                    warnings=[f"Testing error: {str(e)}"],
                    recommendations=["Check feature data quality", "Verify data alignment"]
                )
                
                results[feature] = error_result
        
        self.test_results = results
        
        # Log summary
        self._log_test_summary(results)
        
        return results
    
    def _prepare_data(
        self,
        feature_data: pd.DataFrame,
        price_data: pd.DataFrame,
        dates: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """Prepare and align feature and price data."""
        
        logger.debug("Preparing data for IC testing")
        
        # Ensure datetime index
        if not isinstance(feature_data.index, pd.DatetimeIndex):
            if dates is not None:
                feature_data = feature_data.set_index(dates)
            else:
                raise ValueError("Feature data must have datetime index or provide dates")
        
        if not isinstance(price_data.index, pd.DatetimeIndex):
            if dates is not None:
                price_data = price_data.set_index(dates)
            else:
                raise ValueError("Price data must have datetime index or provide dates")
        
        # Calculate forward returns
        return_data = {}
        
        for horizon in self.config.forward_return_periods:
            if self.config.return_calculation == "simple":
                returns = price_data.pct_change(periods=horizon).shift(-horizon)
            else:  # log returns
                returns = np.log(price_data / price_data.shift(horizon)).shift(-horizon)
            
            # Winsorize returns if requested
            if self.config.winsorize_returns:
                returns = self._winsorize_returns(returns)
            
            # Taiwan price limit adjustment
            if self.config.taiwan_adjustments and self.config.price_limit_adjustment:
                returns = self._adjust_for_price_limits(returns)
            
            return_data[f'return_{horizon}d'] = returns
        
        return_matrix = pd.DataFrame(return_data, index=feature_data.index)
        
        logger.debug(f"Prepared data: {len(feature_data)} feature observations, {len(return_matrix.columns)} return horizons")
        
        return {
            'features': feature_data,
            'returns': return_matrix
        }
    
    def _winsorize_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Winsorize extreme returns."""
        
        winsorized_returns = returns.copy()
        
        for col in returns.columns:
            series = returns[col].dropna()
            if len(series) > 0:
                lower_bound = series.quantile(self.config.winsorize_percentiles[0])
                upper_bound = series.quantile(self.config.winsorize_percentiles[1])
                
                winsorized_returns[col] = returns[col].clip(lower=lower_bound, upper=upper_bound)
        
        return winsorized_returns
    
    def _adjust_for_price_limits(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Adjust returns for Taiwan 10% daily price limits."""
        
        # Cap returns at ±9.5% to account for price limits
        price_limit = 0.095
        adjusted_returns = returns.clip(lower=-price_limit, upper=price_limit)
        
        return adjusted_returns
    
    def _align_data(self, feature_series: pd.Series, return_series: pd.Series) -> pd.DataFrame:
        """Align feature and return data."""
        
        # Find common dates
        common_dates = feature_series.index.intersection(return_series.index)
        
        # Align and remove NaN
        aligned_feature = feature_series.reindex(common_dates)
        aligned_return = return_series.reindex(common_dates)
        
        # Create aligned DataFrame and drop NaN
        aligned_data = pd.DataFrame({
            'feature': aligned_feature,
            'return': aligned_return
        }).dropna()
        
        return aligned_data
    
    def _calculate_ic(
        self,
        feature_series: pd.Series,
        return_series: pd.Series,
        horizon: int,
        regime_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Calculate Information Coefficient and related statistics."""
        
        # Main IC calculation
        if self.config.ic_test_type == ICTestType.SPEARMAN:
            ic, p_value = stats.spearmanr(feature_series, return_series)
        elif self.config.ic_test_type == ICTestType.PEARSON:
            ic, p_value = stats.pearsonr(feature_series, return_series)
        elif self.config.ic_test_type == ICTestType.KENDALL:
            ic, p_value = stats.kendalltau(feature_series, return_series)
        elif self.config.ic_test_type == ICTestType.COMPOSITE:
            # Weighted average of all correlation types
            spearman_ic, _ = stats.spearmanr(feature_series, return_series)
            pearson_ic, _ = stats.pearsonr(feature_series, return_series) 
            kendall_ic, _ = stats.kendalltau(feature_series, return_series)
            
            # Weight: Spearman 50%, Pearson 35%, Kendall 15%
            ic = 0.50 * spearman_ic + 0.35 * pearson_ic + 0.15 * kendall_ic
            
            # Use Spearman p-value for significance
            _, p_value = stats.spearmanr(feature_series, return_series)
        else:
            raise ValueError(f"Unknown IC test type: {self.config.ic_test_type}")
        
        # Handle NaN results
        if np.isnan(ic):
            ic = 0.0
        if np.isnan(p_value):
            p_value = 1.0
        
        # Calculate rolling IC for stability analysis
        rolling_ic_series = None
        rolling_stats = {}
        
        if len(feature_series) >= self.config.rolling_window:
            aligned_data = pd.DataFrame({
                'feature': feature_series,
                'return': return_series
            })
            
            rolling_ic_list = []
            for i in range(self.config.rolling_window - 1, len(aligned_data)):
                window_data = aligned_data.iloc[i - self.config.rolling_window + 1:i + 1]
                
                if len(window_data.dropna()) >= self.config.min_periods:
                    window_feature = window_data['feature'].dropna()
                    window_return = window_data['return'].dropna()
                    
                    # Align within window
                    common_idx = window_feature.index.intersection(window_return.index)
                    if len(common_idx) >= self.config.min_periods:
                        window_feature = window_feature.loc[common_idx]
                        window_return = window_return.loc[common_idx]
                        
                        if self.config.ic_test_type == ICTestType.SPEARMAN:
                            window_ic, _ = stats.spearmanr(window_feature, window_return)
                        elif self.config.ic_test_type == ICTestType.PEARSON:
                            window_ic, _ = stats.pearsonr(window_feature, window_return)
                        elif self.config.ic_test_type == ICTestType.KENDALL:
                            window_ic, _ = stats.kendalltau(window_feature, window_return)
                        else:  # COMPOSITE
                            spear_ic, _ = stats.spearmanr(window_feature, window_return)
                            pears_ic, _ = stats.pearsonr(window_feature, window_return)
                            kend_ic, _ = stats.kendalltau(window_feature, window_return)
                            window_ic = 0.50 * spear_ic + 0.35 * pears_ic + 0.15 * kend_ic
                        
                        rolling_ic_list.append(window_ic if not np.isnan(window_ic) else 0.0)
                    else:
                        rolling_ic_list.append(0.0)
                else:
                    rolling_ic_list.append(0.0)
            
            if rolling_ic_list:
                rolling_ic_series = pd.Series(rolling_ic_list, index=aligned_data.index[self.config.rolling_window-1:])
                rolling_stats = {
                    'mean': np.mean(rolling_ic_list),
                    'std': np.std(rolling_ic_list),
                    'min': np.min(rolling_ic_list),
                    'max': np.max(rolling_ic_list),
                    'stability': 1.0 - (np.std(rolling_ic_list) / (abs(np.mean(rolling_ic_list)) + 1e-8))  # Stability score
                }
        
        # Calculate additional metrics
        hit_rate = self._calculate_hit_rate(feature_series, return_series)
        ic_std = rolling_stats.get('std', 0.0)
        information_ratio = ic / (ic_std + 1e-8)  # Avoid division by zero
        
        # T-statistic for significance
        n_obs = len(feature_series)
        t_stat = ic * np.sqrt((n_obs - 2) / (1 - ic**2 + 1e-8))
        
        # Regime analysis
        regime_performance = None
        if regime_data is not None and self.config.analyze_regimes:
            regime_performance = self._analyze_regime_performance(
                feature_series, return_series, regime_data
            )
        
        return {
            'ic': ic,
            'p_value': p_value,
            't_stat': t_stat,
            'rolling_ic_series': rolling_ic_series,
            'rolling_stats': rolling_stats,
            'hit_rate': hit_rate,
            'information_ratio': information_ratio,
            'regime_performance': regime_performance,
            'n_observations': n_obs
        }
    
    def _calculate_hit_rate(self, feature_series: pd.Series, return_series: pd.Series) -> float:
        """Calculate hit rate (directional accuracy)."""
        
        # Determine feature and return directions
        feature_direction = np.sign(feature_series - feature_series.median())
        return_direction = np.sign(return_series)
        
        # Calculate hit rate
        correct_predictions = (feature_direction == return_direction).sum()
        total_predictions = len(feature_direction)
        
        hit_rate = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return hit_rate
    
    def _analyze_regime_performance(
        self,
        feature_series: pd.Series,
        return_series: pd.Series,
        regime_data: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Analyze IC performance by market regime."""
        
        regime_performance = {}
        
        # Align regime data
        common_dates = feature_series.index.intersection(regime_data.index)
        aligned_regimes = regime_data.reindex(common_dates)
        aligned_features = feature_series.reindex(common_dates)
        aligned_returns = return_series.reindex(common_dates)
        
        # Remove NaN
        valid_mask = ~(aligned_regimes.isna() | aligned_features.isna() | aligned_returns.isna())
        aligned_regimes = aligned_regimes[valid_mask]
        aligned_features = aligned_features[valid_mask]
        aligned_returns = aligned_returns[valid_mask]
        
        # Analyze each regime
        for regime in aligned_regimes.unique():
            regime_mask = aligned_regimes == regime
            
            if regime_mask.sum() >= 10:  # Need minimum observations
                regime_features = aligned_features[regime_mask]
                regime_returns = aligned_returns[regime_mask]
                
                # Calculate regime-specific IC
                regime_ic, regime_p = stats.spearmanr(regime_features, regime_returns)
                regime_hit_rate = self._calculate_hit_rate(regime_features, regime_returns)
                
                regime_performance[str(regime)] = {
                    'ic': regime_ic if not np.isnan(regime_ic) else 0.0,
                    'p_value': regime_p if not np.isnan(regime_p) else 1.0,
                    'hit_rate': regime_hit_rate,
                    'observations': len(regime_features)
                }
        
        return regime_performance
    
    def _create_test_result(
        self,
        feature_name: str,
        horizon_results: Dict[int, Dict[str, Any]]
    ) -> ICTestResult:
        """Create comprehensive test result from horizon results."""
        
        # Use primary horizon (usually 1 day) for main statistics
        primary_horizon = min(horizon_results.keys())
        primary_result = horizon_results[primary_horizon]
        
        # Determine performance level
        ic_value = primary_result['ic']
        p_value = primary_result['p_value']
        is_significant = p_value < self.config.significance_level
        
        if abs(ic_value) >= self.config.excellent_ic_threshold and is_significant:
            performance_level = PerformanceLevel.EXCELLENT
        elif abs(ic_value) >= self.config.good_ic_threshold and is_significant:
            performance_level = PerformanceLevel.GOOD
        elif abs(ic_value) >= self.config.acceptable_ic_threshold:
            performance_level = PerformanceLevel.ACCEPTABLE
        elif abs(ic_value) >= self.config.weak_ic_threshold:
            performance_level = PerformanceLevel.WEAK
        else:
            performance_level = PerformanceLevel.POOR
        
        # Generate warnings and recommendations
        warnings = []
        recommendations = []
        
        if not is_significant:
            warnings.append(f"IC not statistically significant (p-value: {p_value:.3f})")
        
        if abs(ic_value) < self.config.acceptable_ic_threshold:
            warnings.append(f"Low IC value: {ic_value:.4f}")
            recommendations.append("Consider feature transformation or combination")
        
        # Check rolling IC stability
        rolling_stats = primary_result.get('rolling_stats', {})
        if rolling_stats and rolling_stats.get('stability', 1.0) < 0.5:
            warnings.append("Low IC stability over time")
            recommendations.append("Investigate regime-dependent performance")
        
        # Multi-horizon summary
        horizon_summary = {}
        for horizon, result in horizon_results.items():
            horizon_summary[horizon] = {
                'ic': result['ic'],
                'p_value': result['p_value'],
                'hit_rate': result['hit_rate'],
                'information_ratio': result['information_ratio']
            }
        
        return ICTestResult(
            feature_name=feature_name,
            test_type=self.config.ic_test_type,
            ic_value=ic_value,
            ic_std=rolling_stats.get('std', 0.0),
            ic_t_stat=primary_result['t_stat'],
            ic_p_value=p_value,
            performance_level=performance_level,
            is_significant=is_significant,
            ic_time_series=primary_result.get('rolling_ic_series'),
            rolling_ic_stats=rolling_stats,
            horizon_results=horizon_summary,
            regime_performance=primary_result.get('regime_performance'),
            hit_rate=primary_result['hit_rate'],
            information_ratio=primary_result['information_ratio'],
            max_ic=rolling_stats.get('max', ic_value),
            min_ic=rolling_stats.get('min', ic_value),
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _perform_cross_validation(
        self,
        feature_series: pd.Series,
        return_matrix: pd.DataFrame,
        feature_name: str
    ) -> Dict[str, float]:
        """Perform time series cross-validation for IC stability."""
        
        logger.debug(f"Performing cross-validation for {feature_name}")
        
        # Use primary return horizon
        primary_horizon = min(self.config.forward_return_periods)
        return_col = f'return_{primary_horizon}d'
        
        if return_col not in return_matrix.columns:
            return {'mean_ic': 0.0, 'std_ic': 0.0, 'stability': 0.0}
        
        return_series = return_matrix[return_col]
        
        # Align data
        aligned_data = self._align_data(feature_series, return_series)
        
        if len(aligned_data) < self.config.min_observations * 2:
            return {'mean_ic': 0.0, 'std_ic': 0.0, 'stability': 0.0}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds, gap=self.config.cv_gap)
        
        cv_ics = []
        
        for train_idx, test_idx in tscv.split(aligned_data):
            train_data = aligned_data.iloc[train_idx]
            test_data = aligned_data.iloc[test_idx]
            
            if len(train_data) >= self.config.min_periods and len(test_data) >= 10:
                # Calculate IC on test set
                test_ic, _ = stats.spearmanr(test_data['feature'], test_data['return'])
                if not np.isnan(test_ic):
                    cv_ics.append(test_ic)
        
        if cv_ics:
            mean_ic = np.mean(cv_ics)
            std_ic = np.std(cv_ics)
            stability = 1.0 - (std_ic / (abs(mean_ic) + 1e-8))
        else:
            mean_ic = std_ic = stability = 0.0
        
        return {
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'stability': max(0.0, stability)  # Ensure non-negative
        }
    
    def _log_test_summary(self, results: Dict[str, ICTestResult]) -> None:
        """Log IC testing summary."""
        
        if not results:
            return
        
        total_features = len(results)
        
        # Count by performance level
        performance_counts = {level: 0 for level in PerformanceLevel}
        for result in results.values():
            performance_counts[result.performance_level] += 1
        
        # Count significant features
        significant_count = sum(1 for result in results.values() if result.is_significant)
        
        # Calculate average IC
        avg_ic = np.mean([abs(result.ic_value) for result in results.values()])
        
        logger.info(f"IC Performance Testing Summary:")
        logger.info(f"  Total Features Tested: {total_features}")
        logger.info(f"  Average |IC|: {avg_ic:.4f}")
        logger.info(f"  Statistically Significant: {significant_count} ({significant_count/total_features:.1%})")
        logger.info(f"  Performance Distribution:")
        
        for level, count in performance_counts.items():
            if count > 0:
                logger.info(f"    {level.value.title()}: {count} ({count/total_features:.1%})")
        
        # Log top performers
        excellent_features = [
            name for name, result in results.items()
            if result.performance_level == PerformanceLevel.EXCELLENT
        ]
        
        if excellent_features:
            logger.info(f"  Excellent Features: {excellent_features[:5]}")
    
    def get_high_performance_features(
        self,
        results: Optional[Dict[str, ICTestResult]] = None,
        min_performance: PerformanceLevel = PerformanceLevel.ACCEPTABLE,
        require_significance: bool = True
    ) -> List[str]:
        """Get features meeting performance criteria."""
        
        if results is None:
            results = self.test_results
        
        high_performance = []
        
        performance_order = {
            PerformanceLevel.EXCELLENT: 5,
            PerformanceLevel.GOOD: 4,
            PerformanceLevel.ACCEPTABLE: 3,
            PerformanceLevel.WEAK: 2,
            PerformanceLevel.POOR: 1
        }
        
        min_level = performance_order[min_performance]
        
        for name, result in results.items():
            if performance_order[result.performance_level] >= min_level:
                if not require_significance or result.is_significant:
                    high_performance.append(name)
        
        # Sort by IC value (descending)
        high_performance.sort(
            key=lambda name: abs(results[name].ic_value),
            reverse=True
        )
        
        logger.info(f"Found {len(high_performance)} high-performance features")
        
        return high_performance
    
    def get_performance_summary(
        self,
        results: Optional[Dict[str, ICTestResult]] = None
    ) -> pd.DataFrame:
        """Get performance summary as DataFrame."""
        
        if results is None:
            results = self.test_results
        
        summary_data = []
        
        for name, result in results.items():
            summary_data.append({
                'feature': name,
                'ic_value': result.ic_value,
                'abs_ic': abs(result.ic_value),
                'p_value': result.ic_p_value,
                'is_significant': result.is_significant,
                'performance_level': result.performance_level.value,
                'hit_rate': result.hit_rate,
                'information_ratio': result.information_ratio,
                'cv_ic_mean': result.cv_ic_mean,
                'cv_stability': result.cv_stability,
                'warning_count': len(result.warnings) if result.warnings else 0
            })
        
        df = pd.DataFrame(summary_data)
        
        # Sort by absolute IC value
        if not df.empty:
            df = df.sort_values('abs_ic', ascending=False)
        
        return df
    
    def generate_performance_report(
        self,
        results: Optional[Dict[str, ICTestResult]] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate comprehensive performance report."""
        
        if results is None:
            results = self.test_results
        
        report_data = []
        
        for name, result in results.items():
            # Base report row
            base_row = {
                'Feature': name,
                'IC_Value': result.ic_value,
                'Abs_IC': abs(result.ic_value),
                'P_Value': result.ic_p_value,
                'T_Stat': result.ic_t_stat,
                'Is_Significant': result.is_significant,
                'Performance_Level': result.performance_level.value,
                'Hit_Rate': result.hit_rate,
                'Information_Ratio': result.information_ratio,
                'IC_Std': result.ic_std,
                'CV_IC_Mean': result.cv_ic_mean,
                'CV_IC_Std': result.cv_ic_std,
                'CV_Stability': result.cv_stability,
                'Warnings': '; '.join(result.warnings) if result.warnings else '',
                'Recommendations': '; '.join(result.recommendations) if result.recommendations else '',
                'Test_Timestamp': result.test_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add horizon-specific results
            if result.horizon_results:
                for horizon, horizon_data in result.horizon_results.items():
                    row = base_row.copy()
                    row['Return_Horizon'] = f"{horizon}d"
                    row['Horizon_IC'] = horizon_data['ic']
                    row['Horizon_P_Value'] = horizon_data['p_value']
                    row['Horizon_Hit_Rate'] = horizon_data['hit_rate']
                    row['Horizon_IR'] = horizon_data['information_ratio']
                    
                    report_data.append(row)
            else:
                # No horizon data, add base row
                base_row['Return_Horizon'] = '1d'
                report_data.append(base_row)
        
        report_df = pd.DataFrame(report_data)
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            logger.info(f"IC performance report saved to {output_path}")
        
        return report_df