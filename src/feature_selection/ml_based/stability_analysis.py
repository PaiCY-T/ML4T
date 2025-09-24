"""
Feature Stability Analysis Across Time Periods and Market Regimes

Implementation of comprehensive stability scoring for feature selection
with Taiwan market regime awareness and time-series considerations.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


@dataclass
class StabilityConfig:
    """Configuration for stability analysis."""
    
    # Time period analysis
    time_windows: List[int] = field(default_factory=lambda: [252, 504, 756])  # 1Y, 2Y, 3Y trading days
    rolling_window: int = 252  # Rolling analysis window
    min_window_size: int = 126  # Minimum window size (6 months)
    
    # Market regime detection
    regime_detection_method: str = 'volatility'  # 'volatility', 'returns', 'vix', 'custom'
    volatility_threshold: float = 0.02  # Daily volatility threshold for regime changes
    regime_min_duration: int = 20  # Minimum regime duration in days
    
    # Stability metrics
    stability_metrics: List[str] = field(default_factory=lambda: [
        'psi',          # Population Stability Index
        'correlation',  # Cross-period correlation
        'distribution', # Distribution stability (KS test)
        'importance',   # Feature importance stability
        'performance'   # Predictive performance stability
    ])
    
    # PSI calculation parameters
    psi_bins: int = 10
    psi_threshold: float = 0.1  # PSI > 0.1 indicates instability
    
    # Performance parameters
    ic_window: int = 63  # Quarter for IC calculation
    min_ic_threshold: float = 0.02
    stability_threshold: float = 0.7  # Overall stability score threshold
    
    # LightGBM parameters for stability testing
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Smaller for stability testing
        'max_depth': 4,
        'learning_rate': 0.15,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # Computation parameters
    n_estimators: int = 50  # Faster for stability testing
    parallel_analysis: bool = True
    cache_results: bool = True


class StabilityAnalyzer:
    """
    Comprehensive Feature Stability Analysis.
    
    Analyzes feature stability across:
    1. Different time periods
    2. Market regimes (bull/bear/volatile)
    3. Rolling windows
    4. Distribution shifts
    5. Predictive performance consistency
    """
    
    def __init__(self, config: Optional[StabilityConfig] = None):
        """Initialize stability analyzer.
        
        Args:
            config: Configuration for stability analysis
        """
        self.config = config or StabilityConfig()
        self.stability_scores_: Optional[pd.DataFrame] = None
        self.regime_labels_: Optional[pd.Series] = None
        self.time_analysis_: Optional[Dict[str, pd.DataFrame]] = None
        self.regime_analysis_: Optional[Dict[str, pd.DataFrame]] = None
        self.rolling_analysis_: Optional[pd.DataFrame] = None
        self._cache: Dict[str, Any] = {}
        
        logger.info("Stability Analyzer initialized")
    
    def analyze_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Comprehensive stability analysis across time periods and market regimes.
        
        Args:
            X: Feature matrix with MultiIndex (date, symbol)
            y: Target variable (forward returns)
            dates: Date series for time analysis
            market_data: Market data for regime detection
            feature_names: Specific features to analyze (all if None)
            
        Returns:
            DataFrame with comprehensive stability scores
        """
        logger.info(f"Starting stability analysis for {X.shape[1]} features")
        
        # Prepare data
        X_clean, y_clean, dates_clean = self._prepare_data(X, y, dates)
        
        # Select features to analyze
        features_to_analyze = feature_names or list(X_clean.columns)
        
        # Market regime detection
        if market_data is not None:
            self.regime_labels_ = self._detect_market_regimes(market_data, dates_clean)
        else:
            # Use returns-based regime detection
            self.regime_labels_ = self._detect_regimes_from_returns(y_clean, dates_clean)
        
        # Initialize results dictionary
        stability_results = {}
        
        # 1. Time-based stability analysis
        logger.info("Analyzing time-based stability")
        time_stability = self._analyze_time_stability(X_clean, y_clean, dates_clean, features_to_analyze)
        stability_results.update(time_stability)
        
        # 2. Market regime stability analysis
        logger.info("Analyzing regime-based stability")
        regime_stability = self._analyze_regime_stability(X_clean, y_clean, features_to_analyze)
        stability_results.update(regime_stability)
        
        # 3. Rolling window analysis
        logger.info("Analyzing rolling stability")
        rolling_stability = self._analyze_rolling_stability(X_clean, y_clean, dates_clean, features_to_analyze)
        stability_results.update(rolling_stability)
        
        # 4. Distribution stability analysis
        logger.info("Analyzing distribution stability")
        distribution_stability = self._analyze_distribution_stability(X_clean, dates_clean, features_to_analyze)
        stability_results.update(distribution_stability)
        
        # 5. Performance stability analysis
        logger.info("Analyzing performance stability")
        performance_stability = self._analyze_performance_stability(X_clean, y_clean, dates_clean, features_to_analyze)
        stability_results.update(performance_stability)
        
        # Combine all stability metrics
        self.stability_scores_ = self._combine_stability_metrics(stability_results, features_to_analyze)
        
        logger.info("Stability analysis completed")
        
        return self.stability_scores_
    
    def _prepare_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        dates: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare and clean data for stability analysis."""
        
        # Remove NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        # Extract dates
        if dates is not None:
            dates_clean = dates[valid_mask].copy()
        else:
            # Try to extract from MultiIndex
            if isinstance(X.index, pd.MultiIndex) and 'date' in X.index.names:
                dates_clean = pd.Series(X_clean.index.get_level_values('date'))
            else:
                # Create synthetic dates
                dates_clean = pd.Series(pd.date_range('2020-01-01', periods=len(X_clean), freq='D'))
        
        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        logger.info(f"Data prepared: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        logger.info(f"Date range: {dates_clean.min()} to {dates_clean.max()}")
        
        return X_clean, y_clean, dates_clean
    
    def _detect_market_regimes(self, market_data: pd.DataFrame, dates: pd.Series) -> pd.Series:
        """Detect market regimes from market data."""
        
        if self.config.regime_detection_method == 'volatility':
            return self._detect_volatility_regimes(market_data, dates)
        elif self.config.regime_detection_method == 'returns':
            return self._detect_return_regimes(market_data, dates)
        else:
            logger.warning(f"Unknown regime detection method: {self.config.regime_detection_method}")
            return self._detect_regimes_from_returns(market_data.iloc[:, 0], dates)
    
    def _detect_volatility_regimes(self, market_data: pd.DataFrame, dates: pd.Series) -> pd.Series:
        """Detect market regimes based on volatility."""
        
        # Calculate rolling volatility
        returns = market_data.pct_change().fillna(0)
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # Use first column if multiple columns
        if len(rolling_vol.columns) > 1:
            rolling_vol = rolling_vol.iloc[:, 0]
        
        # Define regimes based on volatility quantiles
        vol_25 = rolling_vol.quantile(0.25)
        vol_75 = rolling_vol.quantile(0.75)
        
        regimes = pd.Series(index=dates, dtype='category')
        regimes[rolling_vol <= vol_25] = 'low_vol'
        regimes[(rolling_vol > vol_25) & (rolling_vol <= vol_75)] = 'normal_vol'
        regimes[rolling_vol > vol_75] = 'high_vol'
        
        # Apply minimum duration constraint
        regimes = self._smooth_regimes(regimes, self.config.regime_min_duration)
        
        return regimes
    
    def _detect_return_regimes(self, market_data: pd.DataFrame, dates: pd.Series) -> pd.Series:
        """Detect market regimes based on returns."""
        
        returns = market_data.pct_change().fillna(0)
        
        # Use first column if multiple columns
        if len(returns.columns) > 1:
            returns = returns.iloc[:, 0]
        
        # Calculate rolling returns
        rolling_returns = returns.rolling(window=60).mean()  # ~3 months
        
        # Define regimes based on return quantiles
        ret_33 = rolling_returns.quantile(0.33)
        ret_67 = rolling_returns.quantile(0.67)
        
        regimes = pd.Series(index=dates, dtype='category')
        regimes[rolling_returns <= ret_33] = 'bear'
        regimes[(rolling_returns > ret_33) & (rolling_returns <= ret_67)] = 'neutral'
        regimes[rolling_returns > ret_67] = 'bull'
        
        # Apply minimum duration constraint
        regimes = self._smooth_regimes(regimes, self.config.regime_min_duration)
        
        return regimes
    
    def _detect_regimes_from_returns(self, returns: pd.Series, dates: pd.Series) -> pd.Series:
        """Detect regimes from return series (fallback method)."""
        
        # Calculate rolling statistics
        rolling_mean = returns.rolling(window=60).mean()
        rolling_vol = returns.rolling(window=60).std()
        
        # Simple regime classification
        regimes = pd.Series(index=dates, dtype='category')
        
        # High volatility regime
        high_vol_mask = rolling_vol > rolling_vol.quantile(0.75)
        regimes[high_vol_mask] = 'volatile'
        
        # Bull/Bear based on returns
        remaining_mask = ~high_vol_mask
        bull_mask = remaining_mask & (rolling_mean > rolling_mean.quantile(0.6))
        bear_mask = remaining_mask & (rolling_mean < rolling_mean.quantile(0.4))
        
        regimes[bull_mask] = 'bull'
        regimes[bear_mask] = 'bear'
        regimes[remaining_mask & ~bull_mask & ~bear_mask] = 'neutral'
        
        # Apply minimum duration constraint
        regimes = self._smooth_regimes(regimes, self.config.regime_min_duration)
        
        return regimes
    
    def _smooth_regimes(self, regimes: pd.Series, min_duration: int) -> pd.Series:
        """Smooth regime transitions to enforce minimum duration."""
        
        smoothed = regimes.copy()
        
        # Forward pass
        current_regime = None
        regime_start = 0
        
        for i, regime in enumerate(regimes):
            if regime != current_regime:
                if current_regime is not None and (i - regime_start) < min_duration:
                    # Extend previous regime
                    smoothed.iloc[regime_start:i] = current_regime
                current_regime = regime
                regime_start = i
        
        return smoothed
    
    def _analyze_time_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        features: List[str]
    ) -> Dict[str, np.ndarray]:
        """Analyze feature stability across different time windows."""
        
        stability_results = {}
        
        for window in self.config.time_windows:
            logger.debug(f"Analyzing {window}-day time window")
            
            if len(dates) < window * 2:  # Need at least 2 periods
                continue
            
            # Split data into time periods
            mid_point = len(dates) // 2
            
            # Early period
            early_mask = dates <= dates.iloc[mid_point]
            X_early = X[early_mask]
            y_early = y[early_mask]
            
            # Late period  
            late_mask = dates > dates.iloc[mid_point]
            X_late = X[late_mask]
            y_late = y[late_mask]
            
            # Calculate cross-period stability
            cross_period_stability = []
            
            for feature in features:
                # Feature correlation across periods
                if feature in X_early.columns and feature in X_late.columns:
                    early_vals = X_early[feature].values
                    late_vals = X_late[feature].values
                    
                    # Normalize for comparison
                    early_norm = (early_vals - np.mean(early_vals)) / (np.std(early_vals) + 1e-8)
                    late_norm = (late_vals - np.mean(late_vals)) / (np.std(late_vals) + 1e-8)
                    
                    # Sample same number of points
                    min_len = min(len(early_norm), len(late_norm))
                    if min_len > 100:  # Minimum sample size
                        early_sample = np.random.choice(early_norm, min_len, replace=False)
                        late_sample = np.random.choice(late_norm, min_len, replace=False)
                        
                        correlation = np.corrcoef(early_sample, late_sample)[0, 1]
                        stability_score = max(0, correlation)  # Non-negative
                    else:
                        stability_score = 0.0
                else:
                    stability_score = 0.0
                
                cross_period_stability.append(stability_score)
            
            stability_results[f'time_stability_{window}d'] = np.array(cross_period_stability)
        
        return stability_results
    
    def _analyze_regime_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str]
    ) -> Dict[str, np.ndarray]:
        """Analyze feature stability across market regimes."""
        
        if self.regime_labels_ is None:
            return {}
        
        stability_results = {}
        unique_regimes = self.regime_labels_.dropna().unique()
        
        logger.debug(f"Analyzing stability across {len(unique_regimes)} regimes: {unique_regimes}")
        
        # Analyze feature importance stability across regimes
        regime_importances = {}
        
        for regime in unique_regimes:
            regime_mask = self.regime_labels_ == regime
            
            if regime_mask.sum() < 50:  # Skip regimes with too few observations
                continue
            
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            # Train model on regime data
            if len(X_regime) > 0 and len(y_regime) > 0:
                try:
                    model = lgb.LGBMRegressor(
                        n_estimators=self.config.n_estimators,
                        **self.config.lgb_params
                    )
                    
                    model.fit(X_regime[features], y_regime)
                    
                    # Store feature importances for this regime
                    regime_importances[regime] = pd.Series(
                        model.feature_importances_,
                        index=features
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to train model for regime {regime}: {e}")
                    continue
        
        # Calculate importance stability across regimes
        if len(regime_importances) >= 2:
            importance_correlations = []
            
            regime_names = list(regime_importances.keys())
            for i, regime1 in enumerate(regime_names):
                correlations_for_regime = []
                for j, regime2 in enumerate(regime_names):
                    if i != j:
                        corr = regime_importances[regime1].corr(regime_importances[regime2])
                        correlations_for_regime.append(corr if not np.isnan(corr) else 0.0)
                
                avg_correlation = np.mean(correlations_for_regime) if correlations_for_regime else 0.0
                importance_correlations.append(avg_correlation)
            
            # Average importance stability across all regime pairs
            regime_stability = []
            for feature in features:
                feature_correlations = []
                for i, regime1 in enumerate(regime_names):
                    for j, regime2 in enumerate(regime_names):
                        if i < j:  # Avoid double counting
                            imp1 = regime_importances[regime1][feature]
                            imp2 = regime_importances[regime2][feature]
                            # Normalized importance comparison
                            max_imp = max(imp1, imp2)
                            if max_imp > 0:
                                stability = 1 - abs(imp1 - imp2) / max_imp
                            else:
                                stability = 1.0
                            feature_correlations.append(stability)
                
                regime_stability.append(np.mean(feature_correlations) if feature_correlations else 0.0)
            
            stability_results['regime_importance_stability'] = np.array(regime_stability)
        
        return stability_results
    
    def _analyze_rolling_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        features: List[str]
    ) -> Dict[str, np.ndarray]:
        """Analyze feature stability using rolling window analysis."""
        
        stability_results = {}
        
        if len(dates) < self.config.rolling_window * 2:
            logger.warning("Insufficient data for rolling stability analysis")
            return stability_results
        
        # Calculate rolling statistics
        rolling_means = []
        rolling_stds = []
        rolling_ics = []
        
        window_size = self.config.rolling_window
        step_size = window_size // 4  # Overlapping windows
        
        for start_idx in range(0, len(dates) - window_size, step_size):
            end_idx = start_idx + window_size
            
            X_window = X.iloc[start_idx:end_idx][features]
            y_window = y.iloc[start_idx:end_idx]
            
            # Calculate window statistics
            window_means = X_window.mean()
            window_stds = X_window.std()
            
            # Calculate IC for each feature in window
            window_ics = []
            for feature in features:
                ic = pd.Series(X_window[feature]).corr(pd.Series(y_window), method='spearman')
                window_ics.append(ic if not np.isnan(ic) else 0.0)
            
            rolling_means.append(window_means.values)
            rolling_stds.append(window_stds.values)
            rolling_ics.append(np.array(window_ics))
        
        if len(rolling_means) >= 2:
            rolling_means = np.array(rolling_means)
            rolling_stds = np.array(rolling_stds)
            rolling_ics = np.array(rolling_ics)
            
            # Calculate mean stability (coefficient of variation)
            mean_stability = []
            for i in range(len(features)):
                feature_means = rolling_means[:, i]
                if np.std(feature_means) > 0:
                    cv = np.std(feature_means) / (np.mean(np.abs(feature_means)) + 1e-8)
                    stability = np.exp(-cv)  # Convert to stability score
                else:
                    stability = 1.0
                mean_stability.append(stability)
            
            # Calculate IC stability
            ic_stability = []
            for i in range(len(features)):
                feature_ics = rolling_ics[:, i]
                if len(feature_ics) > 1:
                    ic_std = np.std(feature_ics)
                    ic_mean = np.mean(np.abs(feature_ics))
                    if ic_mean > 0:
                        ic_cv = ic_std / (ic_mean + 1e-8)
                        stability = np.exp(-ic_cv)
                    else:
                        stability = 0.0
                else:
                    stability = 0.0
                ic_stability.append(stability)
            
            stability_results['rolling_mean_stability'] = np.array(mean_stability)
            stability_results['rolling_ic_stability'] = np.array(ic_stability)
        
        return stability_results
    
    def _analyze_distribution_stability(
        self,
        X: pd.DataFrame,
        dates: pd.Series,
        features: List[str]
    ) -> Dict[str, np.ndarray]:
        """Analyze feature distribution stability using PSI and KS tests."""
        
        stability_results = {}
        
        # Split data into periods for comparison
        mid_point = len(dates) // 2
        
        X_early = X.iloc[:mid_point][features]
        X_late = X.iloc[mid_point:][features]
        
        psi_scores = []
        ks_pvalues = []
        
        for feature in features:
            # PSI calculation
            psi_score = self._calculate_psi(X_early[feature], X_late[feature])
            psi_scores.append(psi_score)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = ks_2samp(X_early[feature].dropna(), X_late[feature].dropna())
            ks_pvalues.append(ks_pvalue)
        
        # Convert to stability scores (higher is more stable)
        psi_stability = np.exp(-np.array(psi_scores))  # PSI closer to 0 is more stable
        ks_stability = np.array(ks_pvalues)  # Higher p-value means more stable
        
        stability_results['psi_stability'] = psi_stability
        stability_results['ks_stability'] = ks_stability
        
        return stability_results
    
    def _calculate_psi(self, expected: pd.Series, actual: pd.Series) -> float:
        """Calculate Population Stability Index (PSI)."""
        
        try:
            # Remove NaN values
            expected_clean = expected.dropna()
            actual_clean = actual.dropna()
            
            if len(expected_clean) == 0 or len(actual_clean) == 0:
                return 1.0  # Maximum instability
            
            # Create bins based on expected distribution
            bins = np.percentile(expected_clean, np.linspace(0, 100, self.config.psi_bins + 1))
            bins = np.unique(bins)  # Remove duplicates
            
            if len(bins) < 2:
                return 0.0  # Cannot calculate PSI
            
            # Calculate expected and actual proportions
            expected_counts, _ = np.histogram(expected_clean, bins=bins)
            actual_counts, _ = np.histogram(actual_clean, bins=bins)
            
            # Convert to proportions
            expected_props = expected_counts / len(expected_clean)
            actual_props = actual_counts / len(actual_clean)
            
            # Avoid division by zero
            expected_props = np.where(expected_props == 0, 1e-8, expected_props)
            actual_props = np.where(actual_props == 0, 1e-8, actual_props)
            
            # Calculate PSI
            psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
            
            return psi
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 1.0  # Maximum instability for errors
    
    def _analyze_performance_stability(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        features: List[str]
    ) -> Dict[str, np.ndarray]:
        """Analyze predictive performance stability across time."""
        
        stability_results = {}
        
        # Rolling IC analysis
        ic_window = self.config.ic_window
        rolling_ics = []
        
        for start_idx in range(0, len(dates) - ic_window, ic_window // 2):
            end_idx = start_idx + ic_window
            
            X_window = X.iloc[start_idx:end_idx][features]
            y_window = y.iloc[start_idx:end_idx]
            
            # Calculate IC for each feature
            window_ics = []
            for feature in features:
                ic = pd.Series(X_window[feature]).corr(pd.Series(y_window), method='spearman')
                window_ics.append(ic if not np.isnan(ic) else 0.0)
            
            rolling_ics.append(window_ics)
        
        if len(rolling_ics) >= 2:
            rolling_ics = np.array(rolling_ics)
            
            # Calculate IC stability (information ratio-like metric)
            ic_means = np.mean(rolling_ics, axis=0)
            ic_stds = np.std(rolling_ics, axis=0)
            
            # Information ratio for each feature
            information_ratios = []
            for i in range(len(features)):
                if ic_stds[i] > 0:
                    ir = np.abs(ic_means[i]) / ic_stds[i]
                else:
                    ir = 0.0
                information_ratios.append(ir)
            
            # Convert to stability score (sigmoid transform)
            performance_stability = 2 / (1 + np.exp(-np.array(information_ratios))) - 1
            
            stability_results['performance_stability'] = performance_stability
        
        return stability_results
    
    def _combine_stability_metrics(
        self,
        stability_results: Dict[str, np.ndarray],
        features: List[str]
    ) -> pd.DataFrame:
        """Combine all stability metrics into comprehensive scores."""
        
        # Initialize results DataFrame
        stability_df = pd.DataFrame(index=features)
        
        # Add individual metrics
        for metric_name, metric_values in stability_results.items():
            if len(metric_values) == len(features):
                stability_df[metric_name] = metric_values
            else:
                logger.warning(f"Metric {metric_name} has incorrect length: {len(metric_values)} vs {len(features)}")
        
        # Calculate composite stability score
        # Weight different stability aspects
        weights = {
            'time_stability': 0.25,
            'regime_stability': 0.20,
            'rolling_stability': 0.20,
            'distribution_stability': 0.15,
            'performance_stability': 0.20
        }
        
        composite_scores = np.zeros(len(features))
        total_weight = 0.0
        
        # Time stability (average across different windows)
        time_cols = [col for col in stability_df.columns if col.startswith('time_stability_')]
        if time_cols:
            time_avg = stability_df[time_cols].mean(axis=1).fillna(0)
            composite_scores += weights['time_stability'] * time_avg.values
            total_weight += weights['time_stability']
        
        # Regime stability
        if 'regime_importance_stability' in stability_df.columns:
            composite_scores += weights['regime_stability'] * stability_df['regime_importance_stability'].fillna(0).values
            total_weight += weights['regime_stability']
        
        # Rolling stability
        rolling_cols = [col for col in stability_df.columns if col.startswith('rolling_')]
        if rolling_cols:
            rolling_avg = stability_df[rolling_cols].mean(axis=1).fillna(0)
            composite_scores += weights['rolling_stability'] * rolling_avg.values
            total_weight += weights['rolling_stability']
        
        # Distribution stability
        dist_cols = ['psi_stability', 'ks_stability']
        available_dist_cols = [col for col in dist_cols if col in stability_df.columns]
        if available_dist_cols:
            dist_avg = stability_df[available_dist_cols].mean(axis=1).fillna(0)
            composite_scores += weights['distribution_stability'] * dist_avg.values
            total_weight += weights['distribution_stability']
        
        # Performance stability
        if 'performance_stability' in stability_df.columns:
            composite_scores += weights['performance_stability'] * stability_df['performance_stability'].fillna(0).values
            total_weight += weights['performance_stability']
        
        # Normalize composite score
        if total_weight > 0:
            composite_scores = composite_scores / total_weight
        
        stability_df['composite_stability'] = composite_scores
        
        # Add feature ranking based on composite score
        stability_df['stability_rank'] = stability_df['composite_stability'].rank(ascending=False)
        
        # Sort by composite stability (descending)
        stability_df = stability_df.sort_values('composite_stability', ascending=False)
        
        logger.info(f"Stability analysis completed. Top stable feature: {stability_df.index[0]}")
        
        return stability_df
    
    def get_stable_features(
        self,
        n_features: Optional[int] = None,
        stability_threshold: Optional[float] = None
    ) -> List[str]:
        """Get list of stable features based on criteria."""
        
        if self.stability_scores_ is None:
            raise ValueError("Stability analysis has not been performed yet")
        
        # Apply stability threshold filter
        threshold = stability_threshold or self.config.stability_threshold
        stable_features = self.stability_scores_[
            self.stability_scores_['composite_stability'] >= threshold
        ]
        
        # Limit number of features if specified
        if n_features is not None:
            stable_features = stable_features.head(n_features)
        
        return stable_features.index.tolist()
    
    def plot_stability_analysis(self, save_path: Optional[str] = None) -> None:
        """Plot comprehensive stability analysis results."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.stability_scores_ is None:
                raise ValueError("No stability analysis results available")
            
            # Create comprehensive plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Composite stability scores
            top_features = self.stability_scores_.head(20)
            axes[0, 0].barh(range(len(top_features)), top_features['composite_stability'])
            axes[0, 0].set_yticks(range(len(top_features)))
            axes[0, 0].set_yticklabels(top_features.index, fontsize=8)
            axes[0, 0].set_xlabel('Composite Stability Score')
            axes[0, 0].set_title('Top 20 Most Stable Features')
            axes[0, 0].invert_yaxis()
            
            # Plot 2: Stability components heatmap
            stability_components = self.stability_scores_.select_dtypes(include=[np.number]).head(20)
            sns.heatmap(stability_components.T, annot=False, cmap='RdYlGn', 
                       ax=axes[0, 1], cbar_kws={'label': 'Stability Score'})
            axes[0, 1].set_title('Stability Components Heatmap')
            axes[0, 1].set_xlabel('Features (Top 20)')
            
            # Plot 3: Stability distribution
            axes[1, 0].hist(self.stability_scores_['composite_stability'], bins=30, 
                           alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].axvline(self.config.stability_threshold, color='red', 
                              linestyle='--', label=f'Threshold: {self.config.stability_threshold}')
            axes[1, 0].set_xlabel('Composite Stability Score')
            axes[1, 0].set_ylabel('Number of Features')
            axes[1, 0].set_title('Stability Score Distribution')
            axes[1, 0].legend()
            
            # Plot 4: Market regimes (if available)
            if self.regime_labels_ is not None:
                regime_counts = self.regime_labels_.value_counts()
                axes[1, 1].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
                axes[1, 1].set_title('Market Regime Distribution')
            else:
                axes[1, 1].text(0.5, 0.5, 'No regime data available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Market Regimes')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Stability analysis plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Error creating stability plot: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of stability analysis."""
        
        if self.stability_scores_ is None:
            return {"status": "Stability analysis not performed"}
        
        # Calculate summary statistics
        stable_count = (self.stability_scores_['composite_stability'] >= self.config.stability_threshold).sum()
        
        summary = {
            "total_features_analyzed": len(self.stability_scores_),
            "stable_features": stable_count,
            "stability_rate": stable_count / len(self.stability_scores_),
            "mean_stability": self.stability_scores_['composite_stability'].mean(),
            "median_stability": self.stability_scores_['composite_stability'].median(),
            "stability_std": self.stability_scores_['composite_stability'].std(),
            "top_10_stable_features": self.stability_scores_.head(10).index.tolist(),
            "market_regimes": self.regime_labels_.value_counts().to_dict() if self.regime_labels_ is not None else None,
            "config": {
                "stability_threshold": self.config.stability_threshold,
                "time_windows": self.config.time_windows,
                "rolling_window": self.config.rolling_window,
                "regime_detection_method": self.config.regime_detection_method
            }
        }
        
        return summary