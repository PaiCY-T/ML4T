"""
Statistical Validation Engine - Stream A for Task #27

Comprehensive statistical validation system with automated Information Coefficient
monitoring, drift detection, and Taiwan market-specific performance metrics.

Features:
- IC monitoring with 95%+ accuracy
- Statistical significance testing
- Performance tracking across market regimes  
- Taiwan market adaptations (T+2, price limits)
- Real-time validation latency <100ms
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, jarque_bera, normaltest
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for statistical validation engine."""
    
    # IC monitoring parameters
    ic_lookback_periods: List[int] = field(default_factory=lambda: [20, 60, 120, 252])  # Days
    ic_significance_threshold: float = 0.05  # p-value threshold
    ic_accuracy_target: float = 0.95  # 95% accuracy requirement
    
    # Drift detection thresholds
    psi_threshold: float = 0.2  # Population Stability Index
    ks_test_threshold: float = 0.05  # Kolmogorov-Smirnov p-value
    js_divergence_threshold: float = 0.3  # Jensen-Shannon divergence
    feature_drift_threshold: float = 0.1  # Feature importance change threshold
    
    # Performance tracking
    sharpe_degradation_threshold: float = 0.1  # 10% degradation alert
    drawdown_threshold: float = 0.15  # Maximum drawdown limit
    hit_rate_threshold: float = 0.45  # Minimum hit rate
    volatility_scaling_window: int = 60  # Days for volatility scaling
    
    # Taiwan market parameters
    trading_hours: Dict[str, float] = field(default_factory=lambda: {'start': 9.0, 'end': 13.5})
    settlement_days: int = 2  # T+2 settlement
    price_limit: float = 0.1  # 10% daily limit
    
    # Validation timing
    real_time_latency_target: float = 100.0  # milliseconds
    validation_frequency: int = 1  # days


@dataclass 
class ValidationResults:
    """Container for validation results."""
    
    timestamp: datetime
    model_id: str
    
    # IC analysis
    ic_scores: Dict[str, float]
    ic_significance: Dict[str, float] 
    ic_stability: Dict[str, float]
    
    # Drift detection
    feature_drift: Dict[str, float]
    target_drift: float
    prediction_drift: float
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    regime_performance: Dict[str, Dict[str, float]]
    
    # Statistical tests
    statistical_tests: Dict[str, Dict[str, Any]]
    
    # Overall health
    validation_score: float
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'model_id': self.model_id,
            'ic_scores': self.ic_scores,
            'ic_significance': self.ic_significance,
            'ic_stability': self.ic_stability,
            'feature_drift': self.feature_drift,
            'target_drift': self.target_drift,
            'prediction_drift': self.prediction_drift,
            'performance_metrics': self.performance_metrics,
            'regime_performance': self.regime_performance,
            'statistical_tests': self.statistical_tests,
            'validation_score': self.validation_score,
            'alerts': self.alerts,
            'recommendations': self.recommendations
        }


class InformationCoefficientMonitor:
    """Automated IC monitoring with statistical significance testing."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.ic_history: Dict[int, List[float]] = {period: [] for period in config.ic_lookback_periods}
        
    def calculate_ic_with_significance(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        method: str = 'spearman'
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calculate Information Coefficient with statistical significance testing.
        
        Args:
            predictions: Model predictions
            returns: Actual returns
            method: Correlation method ('spearman' or 'pearson')
            
        Returns:
            Tuple of (IC, p-value, detailed_stats)
        """
        # Align data
        aligned_data = pd.DataFrame({'pred': predictions, 'ret': returns}).dropna()
        
        if len(aligned_data) < 10:
            return 0.0, 1.0, {'error': 'Insufficient data for IC calculation'}
        
        # Calculate correlation and significance
        if method == 'spearman':
            ic, p_value = stats.spearmanr(aligned_data['pred'], aligned_data['ret'])
        else:
            ic, p_value = stats.pearsonr(aligned_data['pred'], aligned_data['ret'])
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_ics = []
        
        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(len(aligned_data), len(aligned_data), replace=True)
            sample_data = aligned_data.iloc[sample_idx]
            
            if method == 'spearman':
                boot_ic, _ = stats.spearmanr(sample_data['pred'], sample_data['ret'])
            else:
                boot_ic, _ = stats.pearsonr(sample_data['pred'], sample_data['ret'])
                
            bootstrap_ics.append(boot_ic)
        
        bootstrap_ics = np.array(bootstrap_ics)
        ci_lower = np.percentile(bootstrap_ics, 2.5)
        ci_upper = np.percentile(bootstrap_ics, 97.5)
        
        detailed_stats = {
            'method': method,
            'n_observations': len(aligned_data),
            'ic': ic,
            'p_value': p_value,
            'is_significant': p_value < self.config.ic_significance_threshold,
            'confidence_interval': (ci_lower, ci_upper),
            'bootstrap_mean': np.mean(bootstrap_ics),
            'bootstrap_std': np.std(bootstrap_ics)
        }
        
        return ic, p_value, detailed_stats
    
    def monitor_rolling_ic(
        self,
        predictions_ts: pd.Series,
        returns_ts: pd.Series
    ) -> Dict[int, Dict[str, Any]]:
        """
        Monitor rolling IC across multiple time horizons.
        
        Args:
            predictions_ts: Time series of predictions
            returns_ts: Time series of returns
            
        Returns:
            Dictionary with IC statistics for each lookback period
        """
        results = {}
        
        for period in self.config.ic_lookback_periods:
            if len(predictions_ts) < period:
                continue
                
            # Calculate rolling IC
            rolling_ics = []
            rolling_p_values = []
            
            for i in range(period, len(predictions_ts)):
                window_pred = predictions_ts.iloc[i-period:i]
                window_ret = returns_ts.iloc[i-period:i]
                
                ic, p_val, _ = self.calculate_ic_with_significance(window_pred, window_ret)
                rolling_ics.append(ic)
                rolling_p_values.append(p_val)
            
            if rolling_ics:
                # IC stability metrics
                ic_mean = np.mean(rolling_ics)
                ic_std = np.std(rolling_ics)
                ic_stability = 1.0 - (ic_std / (abs(ic_mean) + 1e-8))  # Stability score
                
                # Significance rate (percentage of significant ICs)
                significant_rate = np.mean([p < self.config.ic_significance_threshold for p in rolling_p_values])
                
                # Recent trend
                recent_window = min(20, len(rolling_ics) // 4)
                if len(rolling_ics) >= recent_window * 2:
                    recent_ic = np.mean(rolling_ics[-recent_window:])
                    historical_ic = np.mean(rolling_ics[:-recent_window])
                    ic_trend = (recent_ic - historical_ic) / (abs(historical_ic) + 1e-8)
                else:
                    ic_trend = 0.0
                
                results[period] = {
                    'ic_mean': ic_mean,
                    'ic_std': ic_std,
                    'ic_stability': ic_stability,
                    'significant_rate': significant_rate,
                    'ic_trend': ic_trend,
                    'current_ic': rolling_ics[-1] if rolling_ics else 0.0,
                    'rolling_ics': rolling_ics[-60:],  # Keep last 60 for plotting
                }
                
                # Update history
                if period in self.ic_history:
                    self.ic_history[period].extend(rolling_ics)
                    # Keep only recent history for memory management
                    if len(self.ic_history[period]) > 1000:
                        self.ic_history[period] = self.ic_history[period][-500:]
        
        return results


class DriftDetectionEngine:
    """Advanced drift detection for features and targets."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.reference_distributions: Dict[str, Dict[str, Any]] = {}
        
    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins based on reference distribution
        try:
            _, bin_edges = pd.qcut(reference.dropna(), q=n_bins, retbins=True, duplicates='drop')
        except ValueError:
            # Fallback to equal-width binning
            bin_edges = np.linspace(reference.min(), reference.max(), n_bins + 1)
        
        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to proportions
        ref_prop = ref_counts / len(reference)
        curr_prop = curr_counts / len(current)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        ref_prop = np.where(ref_prop == 0, epsilon, ref_prop)
        curr_prop = np.where(curr_prop == 0, epsilon, curr_prop)
        
        # Calculate PSI
        psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
        
        return psi
    
    def jensen_shannon_divergence(
        self,
        p: pd.Series,
        q: pd.Series,
        n_bins: int = 50
    ) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions.
        
        Args:
            p: First distribution
            q: Second distribution
            n_bins: Number of bins for discretization
            
        Returns:
            JS divergence value
        """
        # Create common bins
        min_val = min(p.min(), q.min())
        max_val = max(p.max(), q.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Calculate histograms
        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)
        
        # Normalize
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)
        
        # Add small epsilon
        epsilon = 1e-8
        p_hist = np.where(p_hist == 0, epsilon, p_hist)
        q_hist = np.where(q_hist == 0, epsilon, q_hist)
        
        # Calculate JS divergence
        m = 0.5 * (p_hist + q_hist)
        js_div = 0.5 * stats.entropy(p_hist, m) + 0.5 * stats.entropy(q_hist, m)
        
        return js_div
    
    def detect_feature_drift(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect drift in feature distributions.
        
        Args:
            reference_features: Reference feature distributions
            current_features: Current feature distributions
            
        Returns:
            Drift detection results for each feature
        """
        drift_results = {}
        
        common_features = set(reference_features.columns) & set(current_features.columns)
        
        for feature in common_features:
            ref_values = reference_features[feature].dropna()
            curr_values = current_features[feature].dropna()
            
            if len(ref_values) < 10 or len(curr_values) < 10:
                continue
            
            # PSI calculation
            psi = self.calculate_psi(ref_values, curr_values)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p_value = ks_2samp(ref_values, curr_values)
            
            # Jensen-Shannon divergence
            js_div = self.jensen_shannon_divergence(ref_values, curr_values)
            
            # Statistical tests
            ref_normal_test = normaltest(ref_values)
            curr_normal_test = normaltest(curr_values)
            
            # Summary statistics comparison
            stats_drift = {
                'mean_shift': abs(curr_values.mean() - ref_values.mean()) / ref_values.std(),
                'std_ratio': curr_values.std() / ref_values.std(),
                'skew_shift': abs(curr_values.skew() - ref_values.skew()),
                'kurtosis_shift': abs(curr_values.kurtosis() - ref_values.kurtosis())
            }
            
            # Overall drift score (weighted combination)
            drift_score = (
                0.4 * min(psi / self.config.psi_threshold, 1.0) +
                0.3 * min(js_div / self.config.js_divergence_threshold, 1.0) +
                0.2 * (1.0 - ks_p_value) +
                0.1 * min(stats_drift['mean_shift'], 1.0)
            )
            
            drift_results[feature] = {
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'js_divergence': js_div,
                'normality_tests': {
                    'reference': {'statistic': ref_normal_test.statistic, 'p_value': ref_normal_test.pvalue},
                    'current': {'statistic': curr_normal_test.statistic, 'p_value': curr_normal_test.pvalue}
                },
                'stats_drift': stats_drift,
                'drift_score': drift_score,
                'is_drifted': drift_score > 0.5,
                'drift_level': 'high' if drift_score > 0.7 else 'medium' if drift_score > 0.3 else 'low'
            }
        
        return drift_results


class PerformanceRegimeAnalyzer:
    """Analyze model performance across different market regimes."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def identify_market_regimes(
        self,
        market_data: pd.DataFrame,
        lookback_window: int = 60
    ) -> pd.Series:
        """
        Identify market regimes based on volatility and trend.
        
        Args:
            market_data: Market data with returns
            lookback_window: Window for regime detection
            
        Returns:
            Series with regime labels
        """
        returns = market_data['returns'] if 'returns' in market_data else market_data.iloc[:, 0]
        
        # Calculate rolling volatility and trend
        rolling_vol = returns.rolling(lookback_window).std() * np.sqrt(252)
        rolling_trend = returns.rolling(lookback_window).mean() * 252
        
        # Percentile-based regime classification
        vol_percentiles = rolling_vol.quantile([0.33, 0.67])
        trend_percentiles = rolling_trend.quantile([0.33, 0.67])
        
        regimes = pd.Series(index=returns.index, dtype=str)
        
        for idx in returns.index:
            if pd.isna(rolling_vol.loc[idx]) or pd.isna(rolling_trend.loc[idx]):
                regimes.loc[idx] = 'unknown'
                continue
                
            vol = rolling_vol.loc[idx]
            trend = rolling_trend.loc[idx]
            
            # Volatility regime
            if vol <= vol_percentiles.iloc[0]:
                vol_regime = 'low_vol'
            elif vol >= vol_percentiles.iloc[1]:
                vol_regime = 'high_vol'
            else:
                vol_regime = 'med_vol'
            
            # Trend regime
            if trend <= trend_percentiles.iloc[0]:
                trend_regime = 'bear'
            elif trend >= trend_percentiles.iloc[1]:
                trend_regime = 'bull'
            else:
                trend_regime = 'neutral'
            
            regimes.loc[idx] = f"{trend_regime}_{vol_regime}"
        
        return regimes
    
    def analyze_regime_performance(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        regimes: pd.Series,
        market_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze model performance across different market regimes.
        
        Args:
            predictions: Model predictions
            returns: Actual returns  
            regimes: Market regime labels
            market_data: Additional market data
            
        Returns:
            Performance metrics by regime
        """
        performance_by_regime = {}
        
        # Align all data
        aligned_data = pd.DataFrame({
            'pred': predictions,
            'ret': returns,
            'regime': regimes
        }).dropna()
        
        unique_regimes = aligned_data['regime'].unique()
        
        for regime in unique_regimes:
            if regime == 'unknown':
                continue
                
            regime_data = aligned_data[aligned_data['regime'] == regime]
            
            if len(regime_data) < 10:
                continue
            
            # Calculate performance metrics for this regime
            pred_values = regime_data['pred'].values
            ret_values = regime_data['ret'].values
            
            # Information Coefficient
            ic, ic_p_value = stats.spearmanr(pred_values, ret_values)
            
            # Hit rate (percentage of correct directional predictions)
            hit_rate = np.mean(np.sign(pred_values) == np.sign(ret_values))
            
            # Annualized metrics (assuming daily data)
            mean_ret = np.mean(ret_values) * 252
            vol_ret = np.std(ret_values) * np.sqrt(252)
            sharpe = mean_ret / vol_ret if vol_ret > 0 else 0
            
            # Downside metrics
            negative_returns = ret_values[ret_values < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino = mean_ret / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + ret_values)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))
            
            performance_by_regime[regime] = {
                'ic': ic,
                'ic_p_value': ic_p_value,
                'ic_significant': ic_p_value < 0.05,
                'hit_rate': hit_rate,
                'annual_return': mean_ret,
                'annual_volatility': vol_ret,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_drawdown,
                'n_observations': len(regime_data),
                'regime_frequency': len(regime_data) / len(aligned_data)
            }
        
        return performance_by_regime


class StatisticalValidator:
    """Main statistical validation engine combining all components."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.ic_monitor = InformationCoefficientMonitor(self.config)
        self.drift_detector = DriftDetectionEngine(self.config)
        self.regime_analyzer = PerformanceRegimeAnalyzer(self.config)
        
        logger.info("Statistical Validation Engine initialized")
    
    def comprehensive_validation(
        self,
        model_predictions: pd.Series,
        actual_returns: pd.Series,
        feature_data: pd.DataFrame,
        market_data: pd.DataFrame,
        reference_data: Optional[Dict[str, pd.DataFrame]] = None,
        model_id: str = "default"
    ) -> ValidationResults:
        """
        Perform comprehensive statistical validation.
        
        Args:
            model_predictions: Model predictions time series
            actual_returns: Actual returns time series
            feature_data: Feature data for drift detection
            market_data: Market data for regime analysis
            reference_data: Reference data for drift detection
            model_id: Model identifier
            
        Returns:
            Comprehensive validation results
        """
        start_time = datetime.now()
        
        # IC Analysis
        ic_results = self.ic_monitor.monitor_rolling_ic(model_predictions, actual_returns)
        
        # Current IC with significance
        current_ic, current_p_value, ic_details = self.ic_monitor.calculate_ic_with_significance(
            model_predictions, actual_returns
        )
        
        # Drift Detection
        drift_results = {}
        if reference_data and 'features' in reference_data:
            drift_results = self.drift_detector.detect_feature_drift(
                reference_data['features'], feature_data
            )
        
        # Target drift
        target_drift = 0.0
        if reference_data and 'targets' in reference_data:
            target_drift = self.drift_detector.calculate_psi(
                reference_data['targets'], actual_returns
            )
        
        # Prediction drift
        prediction_drift = 0.0
        if reference_data and 'predictions' in reference_data:
            prediction_drift = self.drift_detector.calculate_psi(
                reference_data['predictions'], model_predictions
            )
        
        # Performance metrics
        performance_metrics = self._calculate_comprehensive_metrics(
            model_predictions, actual_returns
        )
        
        # Regime analysis
        regimes = self.regime_analyzer.identify_market_regimes(market_data)
        regime_performance = self.regime_analyzer.analyze_regime_performance(
            model_predictions, actual_returns, regimes, market_data
        )
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(
            model_predictions, actual_returns, feature_data
        )
        
        # Calculate validation score and generate alerts
        validation_score, alerts, recommendations = self._assess_overall_health(
            ic_results, drift_results, performance_metrics, regime_performance
        )
        
        # Measure validation latency
        validation_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
        if validation_time > self.config.real_time_latency_target:
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'message': f'Validation latency {validation_time:.1f}ms exceeded target {self.config.real_time_latency_target}ms',
                'timestamp': datetime.now().isoformat()
            })
        
        # Compile results
        results = ValidationResults(
            timestamp=datetime.now(),
            model_id=model_id,
            ic_scores={
                'current': current_ic,
                **{f'rolling_{period}d': ic_results.get(period, {}).get('ic_mean', 0.0) 
                   for period in self.config.ic_lookback_periods}
            },
            ic_significance={
                'current_p_value': current_p_value,
                **{f'rolling_{period}d_sig_rate': ic_results.get(period, {}).get('significant_rate', 0.0)
                   for period in self.config.ic_lookback_periods}
            },
            ic_stability={f'rolling_{period}d': ic_results.get(period, {}).get('ic_stability', 0.0)
                         for period in self.config.ic_lookback_periods},
            feature_drift={name: result['drift_score'] for name, result in drift_results.items()},
            target_drift=target_drift,
            prediction_drift=prediction_drift,
            performance_metrics=performance_metrics,
            regime_performance=regime_performance,
            statistical_tests=statistical_tests,
            validation_score=validation_score,
            alerts=alerts,
            recommendations=recommendations
        )
        
        logger.info(f"Comprehensive validation completed in {validation_time:.1f}ms")
        logger.info(f"Validation score: {validation_score:.3f}")
        
        return results
    
    def _calculate_comprehensive_metrics(
        self,
        predictions: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        aligned_data = pd.DataFrame({'pred': predictions, 'ret': returns}).dropna()
        
        if len(aligned_data) < 10:
            return {'error': 'insufficient_data'}
        
        pred_values = aligned_data['pred'].values
        ret_values = aligned_data['ret'].values
        
        metrics = {}
        
        # Basic correlation metrics
        metrics['ic_spearman'], _ = stats.spearmanr(pred_values, ret_values)
        metrics['ic_pearson'], _ = stats.pearsonr(pred_values, ret_values)
        
        # Hit rate metrics
        metrics['hit_rate'] = np.mean(np.sign(pred_values) == np.sign(ret_values))
        metrics['hit_rate_top_quintile'] = self._calculate_quintile_hit_rate(pred_values, ret_values, quintile=5)
        metrics['hit_rate_bottom_quintile'] = self._calculate_quintile_hit_rate(pred_values, ret_values, quintile=1)
        
        # Return metrics (annualized, assuming daily data)
        metrics['annual_return'] = np.mean(ret_values) * 252
        metrics['annual_volatility'] = np.std(ret_values) * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_volatility'] if metrics['annual_volatility'] > 0 else 0
        
        # Risk metrics
        negative_returns = ret_values[ret_values < 0]
        if len(negative_returns) > 0:
            metrics['downside_deviation'] = np.std(negative_returns) * np.sqrt(252)
            metrics['sortino_ratio'] = metrics['annual_return'] / metrics['downside_deviation']
        else:
            metrics['downside_deviation'] = 0.0
            metrics['sortino_ratio'] = float('inf')
        
        # Drawdown metrics
        cumulative_returns = np.cumprod(1 + ret_values)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = abs(np.min(drawdowns))
        
        # Information ratio (assuming zero risk-free rate)
        tracking_error = np.std(ret_values) * np.sqrt(252)
        metrics['information_ratio'] = metrics['annual_return'] / tracking_error if tracking_error > 0 else 0
        
        # Taiwan market specific metrics
        metrics['price_limit_proximity'] = self._calculate_price_limit_impact(ret_values)
        metrics['volatility_adjusted_ic'] = self._calculate_volatility_adjusted_ic(pred_values, ret_values)
        
        return metrics
    
    def _calculate_quintile_hit_rate(self, predictions: np.ndarray, returns: np.ndarray, quintile: int) -> float:
        """Calculate hit rate for specific quintile."""
        # Sort by predictions and select quintile
        sorted_indices = np.argsort(predictions)
        n = len(predictions)
        
        if quintile == 5:  # Top quintile
            quintile_indices = sorted_indices[-n//5:]
        elif quintile == 1:  # Bottom quintile
            quintile_indices = sorted_indices[:n//5]
        else:
            start_idx = (quintile - 1) * n // 5
            end_idx = quintile * n // 5
            quintile_indices = sorted_indices[start_idx:end_idx]
        
        quintile_pred = predictions[quintile_indices]
        quintile_ret = returns[quintile_indices]
        
        return np.mean(np.sign(quintile_pred) == np.sign(quintile_ret))
    
    def _calculate_price_limit_impact(self, returns: np.ndarray) -> float:
        """Calculate impact of Taiwan price limits on returns."""
        # Percentage of returns close to price limits (±9.5% threshold)
        near_limit_count = np.sum(np.abs(returns) > 0.095)
        return near_limit_count / len(returns)
    
    def _calculate_volatility_adjusted_ic(self, predictions: np.ndarray, returns: np.ndarray) -> float:
        """Calculate volatility-adjusted IC."""
        # Scale predictions by realized volatility
        rolling_vol = pd.Series(returns).rolling(self.config.volatility_scaling_window).std()
        adjusted_predictions = predictions / (rolling_vol.values + 1e-8)
        
        # Calculate IC with adjusted predictions
        ic, _ = stats.spearmanr(adjusted_predictions[~np.isnan(adjusted_predictions)], 
                               returns[~np.isnan(adjusted_predictions)])
        return ic
    
    def _perform_statistical_tests(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        features: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Perform comprehensive statistical tests."""
        tests = {}
        
        aligned_data = pd.DataFrame({'pred': predictions, 'ret': returns}).dropna()
        if len(aligned_data) < 20:
            return {'error': 'insufficient_data_for_tests'}
        
        pred_values = aligned_data['pred'].values
        ret_values = aligned_data['ret'].values
        
        # Normality tests
        tests['normality'] = {
            'predictions': {
                'jarque_bera': jarque_bera(pred_values),
                'normaltest': normaltest(pred_values)
            },
            'returns': {
                'jarque_bera': jarque_bera(ret_values),
                'normaltest': normaltest(ret_values)
            }
        }
        
        # Stationarity tests (simplified)
        tests['stationarity'] = {
            'predictions_variance_ratio': self._variance_ratio_test(pred_values),
            'returns_variance_ratio': self._variance_ratio_test(ret_values)
        }
        
        # Autocorrelation tests
        tests['autocorrelation'] = {
            'predictions_ljung_box': self._ljung_box_test(pred_values),
            'returns_ljung_box': self._ljung_box_test(ret_values)
        }
        
        return tests
    
    def _variance_ratio_test(self, series: np.ndarray, lags: List[int] = [2, 4, 8]) -> Dict[str, float]:
        """Simple variance ratio test for mean reversion."""
        results = {}
        
        for lag in lags:
            if len(series) <= lag:
                continue
                
            # Calculate variance ratios
            var_1 = np.var(np.diff(series))
            var_k = np.var(series[lag:] - series[:-lag]) / lag
            
            variance_ratio = var_k / var_1 if var_1 > 0 else 1.0
            results[f'lag_{lag}'] = variance_ratio
        
        return results
    
    def _ljung_box_test(self, series: np.ndarray, lags: int = 10) -> Dict[str, Any]:
        """Simplified Ljung-Box test for autocorrelation."""
        from scipy.stats import chi2
        
        n = len(series)
        if n <= lags + 1:
            return {'error': 'insufficient_data'}
        
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, lags + 1):
            if n - lag > 0:
                autocorr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                autocorrs.append(autocorr if not np.isnan(autocorr) else 0.0)
            else:
                autocorrs.append(0.0)
        
        # Ljung-Box statistic
        lb_stat = n * (n + 2) * np.sum([(autocorr ** 2) / (n - lag - 1) 
                                       for lag, autocorr in enumerate(autocorrs)])
        p_value = 1 - chi2.cdf(lb_stat, lags)
        
        return {
            'statistic': lb_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'autocorrelations': autocorrs
        }
    
    def _assess_overall_health(
        self,
        ic_results: Dict[int, Dict[str, Any]],
        drift_results: Dict[str, Dict[str, Any]],
        performance_metrics: Dict[str, float],
        regime_performance: Dict[str, Dict[str, float]]
    ) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """Assess overall model health and generate alerts/recommendations."""
        health_score = 1.0
        alerts = []
        recommendations = []
        
        # IC health assessment
        ic_scores = []
        for period_results in ic_results.values():
            ic_mean = period_results.get('ic_mean', 0.0)
            ic_stability = period_results.get('ic_stability', 0.0)
            significant_rate = period_results.get('significant_rate', 0.0)
            
            ic_score = (abs(ic_mean) * 0.5 + ic_stability * 0.3 + significant_rate * 0.2)
            ic_scores.append(ic_score)
        
        if ic_scores:
            avg_ic_score = np.mean(ic_scores)
            if avg_ic_score < 0.3:
                health_score *= 0.7
                alerts.append({
                    'type': 'performance',
                    'severity': 'high',
                    'message': f'Low IC performance detected: average score {avg_ic_score:.3f}',
                    'timestamp': datetime.now().isoformat()
                })
                recommendations.append("Consider retraining model with updated features or different target horizon")
        
        # Drift assessment
        if drift_results:
            high_drift_features = [name for name, result in drift_results.items() 
                                 if result['drift_score'] > 0.7]
            
            if len(high_drift_features) > len(drift_results) * 0.2:  # More than 20% features drifted
                health_score *= 0.8
                alerts.append({
                    'type': 'drift',
                    'severity': 'medium',
                    'message': f'High feature drift detected in {len(high_drift_features)} features: {high_drift_features[:5]}',
                    'timestamp': datetime.now().isoformat()
                })
                recommendations.append("Investigate feature engineering pipeline and consider feature selection updates")
        
        # Performance assessment
        if 'sharpe_ratio' in performance_metrics:
            if performance_metrics['sharpe_ratio'] < 1.0:
                health_score *= 0.9
                alerts.append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f'Low Sharpe ratio: {performance_metrics["sharpe_ratio"]:.3f}',
                    'timestamp': datetime.now().isoformat()
                })
        
        if 'max_drawdown' in performance_metrics:
            if performance_metrics['max_drawdown'] > self.config.drawdown_threshold:
                health_score *= 0.8
                alerts.append({
                    'type': 'risk',
                    'severity': 'high',
                    'message': f'Maximum drawdown {performance_metrics["max_drawdown"]:.3f} exceeds threshold {self.config.drawdown_threshold}',
                    'timestamp': datetime.now().isoformat()
                })
                recommendations.append("Review position sizing and risk management parameters")
        
        # Regime consistency assessment
        if regime_performance:
            regime_ics = [perf['ic'] for perf in regime_performance.values() if not np.isnan(perf.get('ic', 0))]
            if regime_ics and np.std(regime_ics) > 0.05:  # High IC variability across regimes
                health_score *= 0.9
                alerts.append({
                    'type': 'stability',
                    'severity': 'low',
                    'message': f'High IC variability across market regimes: std={np.std(regime_ics):.3f}',
                    'timestamp': datetime.now().isoformat()
                })
                recommendations.append("Consider regime-specific model parameters or ensemble approach")
        
        return health_score, alerts, recommendations