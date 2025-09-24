"""
Feature Quality Assessment - Task #28 Stream C

Comprehensive feature quality metrics for Taiwan stock market features including:
- Statistical validation (normality, stationarity, outliers)
- Financial metrics (autocorrelation, volatility, regime detection)
- Taiwan compliance checks (T+2 settlement, price limits, trading hours)
- Predictive power assessment (information ratio, correlation with returns)
- Data integrity validation (missing values, infinite values, consistency)

Used by FeatureSelector to ensure only high-quality features are selected.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import warnings
from pathlib import Path

# Statistical imports
from scipy import stats
from scipy.stats import jarque_bera, shapiro, anderson
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import psutil

# Import project modules  
from .taiwan_config import TaiwanMarketConfig

logger = logging.getLogger(__name__)


class FeatureQualityMetrics:
    """
    Comprehensive feature quality assessment for Taiwan stock market.
    
    Evaluates features across multiple dimensions:
    1. Statistical properties (normality, stationarity, outliers)
    2. Financial characteristics (autocorrelation, volatility patterns)  
    3. Taiwan market compliance (settlement, trading hours, price limits)
    4. Predictive potential (information content, signal-to-noise ratio)
    5. Data integrity (completeness, consistency, validity)
    """
    
    def __init__(
        self,
        taiwan_config: Optional[TaiwanMarketConfig] = None,
        significance_level: float = 0.05,
        outlier_threshold: float = 3.0,
        min_observations: int = 252,
        memory_limit_gb: float = 4.0
    ):
        """
        Initialize quality metrics calculator.
        
        Args:
            taiwan_config: Taiwan market configuration
            significance_level: Statistical significance level (default 0.05)
            outlier_threshold: Z-score threshold for outlier detection
            min_observations: Minimum number of observations for valid metrics
            memory_limit_gb: Memory limit for processing
        """
        self.taiwan_config = taiwan_config or TaiwanMarketConfig()
        self.significance_level = significance_level
        self.outlier_threshold = outlier_threshold
        self.min_observations = min_observations
        self.memory_limit_gb = memory_limit_gb
        
        # Quality thresholds
        self.quality_thresholds = {
            'missing_data_max_pct': 5.0,      # Max 5% missing data
            'outlier_max_pct': 1.0,           # Max 1% outliers
            'stationarity_p_value': 0.05,     # ADF test p-value threshold
            'autocorr_max_lag1': 0.8,         # Max lag-1 autocorrelation
            'min_unique_values': 10,          # Min unique values for continuous features
            'skewness_max_abs': 5.0,          # Max absolute skewness
            'kurtosis_max': 20.0,             # Max kurtosis
            'volatility_min_ratio': 0.001,    # Min volatility ratio
            'information_ratio_min': 0.05     # Min information ratio
        }
        
        # Cache for expensive calculations
        self._calculation_cache = {}
        
    def _check_memory_usage(self, stage: str) -> Dict[str, float]:
        """Monitor memory usage during quality assessment."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = {
            'stage': stage,
            'rss_gb': memory_info.rss / 1024 / 1024 / 1024,
            'vms_gb': memory_info.vms / 1024 / 1024 / 1024,
            'percent': psutil.virtual_memory().percent
        }
        
        if memory_usage['rss_gb'] > self.memory_limit_gb:
            logger.warning(
                f"Memory usage ({memory_usage['rss_gb']:.2f}GB) exceeds limit "
                f"({self.memory_limit_gb}GB) at stage: {stage}"
            )
            
        return memory_usage
    
    def calculate_basic_statistics(self, feature_data: pd.Series) -> Dict[str, Any]:
        """
        Calculate basic statistical properties of a feature.
        
        Args:
            feature_data: Feature values as pandas Series
            
        Returns:
            Dictionary with basic statistics
        """
        # Remove infinite and NaN values for calculations
        clean_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) == 0:
            return {'error': 'No valid data points'}
            
        try:
            stats_dict = {
                'count': len(feature_data),
                'valid_count': len(clean_data),
                'missing_count': len(feature_data) - len(clean_data),
                'missing_pct': (len(feature_data) - len(clean_data)) / len(feature_data) * 100,
                
                # Central tendency
                'mean': float(clean_data.mean()),
                'median': float(clean_data.median()),
                'mode': float(clean_data.mode().iloc[0]) if len(clean_data.mode()) > 0 else np.nan,
                
                # Dispersion
                'std': float(clean_data.std()),
                'var': float(clean_data.var()),
                'min': float(clean_data.min()),
                'max': float(clean_data.max()),
                'range': float(clean_data.max() - clean_data.min()),
                'iqr': float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
                
                # Distribution shape
                'skewness': float(clean_data.skew()),
                'kurtosis': float(clean_data.kurtosis()),
                'unique_values': len(clean_data.unique()),
                'unique_ratio': len(clean_data.unique()) / len(clean_data),
                
                # Percentiles
                'p1': float(clean_data.quantile(0.01)),
                'p5': float(clean_data.quantile(0.05)),
                'p25': float(clean_data.quantile(0.25)),
                'p75': float(clean_data.quantile(0.75)),
                'p95': float(clean_data.quantile(0.95)),
                'p99': float(clean_data.quantile(0.99)),
            }
            
        except Exception as e:
            logger.error(f"Error calculating basic statistics: {str(e)}")
            return {'error': str(e)}
            
        return stats_dict
    
    def test_normality(self, feature_data: pd.Series) -> Dict[str, Any]:
        """
        Test feature data for normality using multiple tests.
        
        Args:
            feature_data: Feature values
            
        Returns:
            Normality test results
        """
        clean_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) < 8:  # Minimum for Shapiro-Wilk
            return {'error': 'Insufficient data for normality tests'}
            
        results = {}
        
        try:
            # Jarque-Bera test (good for large samples)
            if len(clean_data) >= 2000:
                jb_stat, jb_p = jarque_bera(clean_data)
                results['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_p),
                    'is_normal': jb_p > self.significance_level
                }
            
            # Shapiro-Wilk test (good for smaller samples)
            if len(clean_data) <= 5000:  # Shapiro-Wilk limitation
                sw_stat, sw_p = shapiro(clean_data)
                results['shapiro_wilk'] = {
                    'statistic': float(sw_stat),
                    'p_value': float(sw_p),
                    'is_normal': sw_p > self.significance_level
                }
            
            # Anderson-Darling test
            ad_result = anderson(clean_data, dist='norm')
            results['anderson_darling'] = {
                'statistic': float(ad_result.statistic),
                'critical_values': ad_result.critical_values.tolist(),
                'significance_levels': ad_result.significance_levels.tolist(),
                'is_normal': ad_result.statistic < ad_result.critical_values[2]  # 5% level
            }
            
        except Exception as e:
            logger.error(f"Error in normality testing: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def test_stationarity(self, feature_data: pd.Series) -> Dict[str, Any]:
        """
        Test feature data for stationarity using ADF and KPSS tests.
        
        Args:
            feature_data: Feature values (time series)
            
        Returns:
            Stationarity test results
        """
        clean_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) < 12:  # Minimum for ADF test
            return {'error': 'Insufficient data for stationarity tests'}
            
        results = {}
        
        try:
            # Augmented Dickey-Fuller test (null: unit root/non-stationary)
            adf_result = adfuller(clean_data, autolag='AIC')
            results['adf'] = {
                'statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'critical_values': {
                    '1%': float(adf_result[4]['1%']),
                    '5%': float(adf_result[4]['5%']),
                    '10%': float(adf_result[4]['10%'])
                },
                'is_stationary': adf_result[1] < self.significance_level
            }
            
            # KPSS test (null: stationary)  
            if len(clean_data) >= 20:  # KPSS needs more data
                kpss_result = kpss(clean_data, regression='c')
                results['kpss'] = {
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                    'is_stationary': kpss_result[1] > self.significance_level
                }
                
        except Exception as e:
            logger.error(f"Error in stationarity testing: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def detect_outliers(self, feature_data: pd.Series) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods.
        
        Args:
            feature_data: Feature values
            
        Returns:
            Outlier detection results
        """
        clean_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) < 10:
            return {'error': 'Insufficient data for outlier detection'}
            
        results = {}
        
        try:
            # Z-score method
            z_scores = np.abs(stats.zscore(clean_data))
            z_outliers = z_scores > self.outlier_threshold
            
            results['z_score'] = {
                'threshold': self.outlier_threshold,
                'outlier_count': int(z_outliers.sum()),
                'outlier_pct': float(z_outliers.sum() / len(clean_data) * 100),
                'max_z_score': float(z_scores.max()),
                'outlier_indices': clean_data.index[z_outliers].tolist() if z_outliers.any() else []
            }
            
            # IQR method
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = (clean_data < lower_bound) | (clean_data > upper_bound)
            
            results['iqr'] = {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_count': int(iqr_outliers.sum()),
                'outlier_pct': float(iqr_outliers.sum() / len(clean_data) * 100),
                'outlier_indices': clean_data.index[iqr_outliers].tolist() if iqr_outliers.any() else []
            }
            
            # Modified Z-score (more robust)
            median = clean_data.median()
            mad = np.median(np.abs(clean_data - median))  # Median Absolute Deviation
            if mad > 0:
                modified_z_scores = 0.6745 * (clean_data - median) / mad
                modified_z_outliers = np.abs(modified_z_scores) > self.outlier_threshold
                
                results['modified_z_score'] = {
                    'threshold': self.outlier_threshold,
                    'outlier_count': int(modified_z_outliers.sum()),
                    'outlier_pct': float(modified_z_outliers.sum() / len(clean_data) * 100),
                    'max_modified_z_score': float(np.abs(modified_z_scores).max()),
                    'outlier_indices': clean_data.index[modified_z_outliers].tolist() if modified_z_outliers.any() else []
                }
                
        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def calculate_autocorrelation(self, feature_data: pd.Series, max_lags: int = 10) -> Dict[str, Any]:
        """
        Calculate autocorrelation and test for serial correlation.
        
        Args:
            feature_data: Feature values (time series)
            max_lags: Maximum number of lags to test
            
        Returns:
            Autocorrelation analysis results
        """
        clean_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) < max_lags * 2:
            return {'error': 'Insufficient data for autocorrelation analysis'}
            
        results = {}
        
        try:
            # Calculate autocorrelation for each lag
            autocorrs = []
            for lag in range(1, max_lags + 1):
                if len(clean_data) > lag:
                    corr = clean_data.autocorr(lag=lag)
                    autocorrs.append(corr if not pd.isna(corr) else 0.0)
                else:
                    autocorrs.append(0.0)
                    
            results['autocorrelations'] = {
                f'lag_{i+1}': float(autocorrs[i]) for i in range(len(autocorrs))
            }
            
            # Ljung-Box test for autocorrelation
            if len(clean_data) > 20:
                lb_result = acorr_ljungbox(clean_data, lags=min(max_lags, len(clean_data)//4))
                results['ljung_box'] = {
                    'statistics': lb_result['lb_stat'].tolist() if hasattr(lb_result['lb_stat'], 'tolist') else [float(lb_result['lb_stat'])],
                    'p_values': lb_result['lb_pvalue'].tolist() if hasattr(lb_result['lb_pvalue'], 'tolist') else [float(lb_result['lb_pvalue'])],
                    'has_autocorr': bool((lb_result['lb_pvalue'] < self.significance_level).any())
                }
                
            # Summary metrics
            results['summary'] = {
                'lag1_autocorr': float(autocorrs[0]) if autocorrs else 0.0,
                'max_autocorr': float(max(np.abs(autocorrs))) if autocorrs else 0.0,
                'mean_autocorr': float(np.mean(np.abs(autocorrs))) if autocorrs else 0.0,
                'high_autocorr': float(max(np.abs(autocorrs))) > self.quality_thresholds['autocorr_max_lag1'] if autocorrs else False
            }
            
        except Exception as e:
            logger.error(f"Error in autocorrelation calculation: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def assess_financial_properties(self, feature_data: pd.Series) -> Dict[str, Any]:
        """
        Assess financial time series properties.
        
        Args:
            feature_data: Feature values
            
        Returns:
            Financial properties assessment
        """
        clean_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) < self.min_observations:
            return {'error': f'Insufficient data for financial assessment (need {self.min_observations})'}
            
        results = {}
        
        try:
            # Volatility analysis
            returns = clean_data.pct_change().dropna()
            if len(returns) > 1:
                results['volatility'] = {
                    'daily_volatility': float(returns.std()),
                    'annualized_volatility': float(returns.std() * np.sqrt(252)),
                    'volatility_of_volatility': float(returns.rolling(window=21).std().std()) if len(returns) > 21 else np.nan,
                    'max_drawdown': float((clean_data / clean_data.expanding().max() - 1).min()),
                    'volatility_ratio': float(returns.std() / clean_data.mean()) if clean_data.mean() != 0 else np.inf
                }
                
            # Regime detection (simple volatility regimes)
            if len(returns) > 60:  # Need at least ~3 months
                vol_window = min(21, len(returns) // 3)
                rolling_vol = returns.rolling(window=vol_window).std()
                vol_median = rolling_vol.median()
                
                high_vol_periods = (rolling_vol > vol_median * 1.5).sum()
                low_vol_periods = (rolling_vol < vol_median * 0.5).sum()
                
                results['regimes'] = {
                    'high_volatility_periods': int(high_vol_periods),
                    'low_volatility_periods': int(low_vol_periods),
                    'regime_changes': int(((rolling_vol > vol_median * 1.5).astype(int).diff() != 0).sum()),
                    'volatility_persistence': float(rolling_vol.autocorr(lag=1)) if len(rolling_vol.dropna()) > 1 else np.nan
                }
                
            # Trend analysis
            if len(clean_data) > 20:
                # Simple linear trend
                x = np.arange(len(clean_data))
                trend_slope, trend_intercept, trend_r, trend_p, _ = stats.linregress(x, clean_data)
                
                results['trend'] = {
                    'slope': float(trend_slope),
                    'r_squared': float(trend_r ** 2),
                    'p_value': float(trend_p),
                    'is_trending': trend_p < self.significance_level,
                    'direction': 'up' if trend_slope > 0 else 'down' if trend_slope < 0 else 'flat'
                }
                
        except Exception as e:
            logger.error(f"Error in financial properties assessment: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def validate_taiwan_compliance(self, feature_data: pd.Series, feature_name: str = "") -> Dict[str, Any]:
        """
        Validate feature compliance with Taiwan market characteristics.
        
        Args:
            feature_data: Feature values
            feature_name: Name of the feature for context
            
        Returns:
            Taiwan compliance validation results
        """
        results = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'checks_performed': []
        }
        
        try:
            # Check for T+2 settlement compliance
            results['checks_performed'].append('t2_settlement')
            if any(term in feature_name.lower() for term in ['realtime', 'instant', 'immediate']):
                results['violations'].append('Feature name suggests T+2 settlement violation')
                results['compliant'] = False
                
            # Check for trading hours compliance
            results['checks_performed'].append('trading_hours')
            if any(term in feature_name.lower() for term in ['overnight', 'after_hours', 'extended']):
                results['violations'].append('Feature may use non-trading hours data')
                results['compliant'] = False
                
            # Check for reasonable value ranges (market-specific)
            if isinstance(feature_data.index, pd.MultiIndex):
                # Panel data - check across time and stocks
                date_index = feature_data.index.get_level_values(0)
                trading_dates = [d for d in pd.to_datetime(date_index).unique() 
                               if self.taiwan_config.is_trading_day(pd.Timestamp(d))]
                
                results['checks_performed'].append('trading_calendar')
                non_trading_ratio = 1 - len(trading_dates) / len(pd.to_datetime(date_index).unique())
                if non_trading_ratio > 0.1:  # More than 10% non-trading dates
                    results['warnings'].append(f'High non-trading date ratio: {non_trading_ratio:.1%}')
                    
            # Check for price limit compliance (if price-related feature)
            results['checks_performed'].append('price_limits')
            if any(term in feature_name.lower() for term in ['price', 'close', 'high', 'low', 'open']):
                # Calculate daily changes if we have time series data
                clean_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean_data) > 1:
                    daily_changes = clean_data.pct_change().dropna()
                    extreme_changes = daily_changes[np.abs(daily_changes) > 0.12]  # Above 12% (Taiwan limit is 10%)
                    
                    if len(extreme_changes) > 0:
                        results['warnings'].append(
                            f'Found {len(extreme_changes)} extreme price changes above 12%'
                        )
                        
            # Check for currency compliance
            results['checks_performed'].append('currency_compliance')
            if 'currency' in feature_name.lower() or 'fx' in feature_name.lower():
                if 'twd' not in feature_name.lower() and 'ntd' not in feature_name.lower():
                    results['warnings'].append('Currency feature may not be TWD-based')
                    
            # Check for data completeness on trading days
            results['checks_performed'].append('data_completeness')
            missing_ratio = feature_data.isnull().sum() / len(feature_data)
            if missing_ratio > 0.05:  # More than 5% missing
                results['warnings'].append(f'High missing data ratio: {missing_ratio:.1%}')
                
        except Exception as e:
            logger.error(f"Error in Taiwan compliance validation: {str(e)}")
            results['violations'].append(f'Validation error: {str(e)}')
            results['compliant'] = False
            
        return results
    
    def calculate_information_content(
        self, 
        feature_data: pd.Series, 
        target_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Assess information content and predictive potential of feature.
        
        Args:
            feature_data: Feature values
            target_data: Target variable (optional)
            
        Returns:
            Information content metrics
        """
        results = {}
        
        try:
            clean_feature = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Entropy-based information content
            if len(clean_feature) > 0:
                # Discretize continuous feature for entropy calculation
                n_bins = min(50, len(clean_feature.unique()))
                if n_bins > 1:
                    binned_feature = pd.cut(clean_feature, bins=n_bins, duplicates='drop')
                    value_counts = binned_feature.value_counts()
                    probabilities = value_counts / len(binned_feature)
                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    
                    results['entropy'] = {
                        'entropy_bits': float(entropy),
                        'max_entropy_bits': float(np.log2(n_bins)),
                        'normalized_entropy': float(entropy / np.log2(n_bins)) if n_bins > 1 else 0.0
                    }
                    
            # Signal-to-noise ratio
            signal_std = clean_feature.std()
            noise_estimate = np.std(clean_feature.diff().dropna()) / np.sqrt(2)  # Noise from first differences
            if noise_estimate > 0:
                snr = signal_std / noise_estimate
                results['signal_to_noise'] = {
                    'ratio': float(snr),
                    'ratio_db': float(20 * np.log10(snr)) if snr > 0 else -np.inf
                }
                
            # Predictive relationship with target (if provided)
            if target_data is not None:
                aligned_data = pd.concat([feature_data, target_data], axis=1).dropna()
                if len(aligned_data) > 10:
                    feature_col = aligned_data.iloc[:, 0]
                    target_col = aligned_data.iloc[:, 1]
                    
                    # Correlation
                    pearson_corr, pearson_p = stats.pearsonr(feature_col, target_col)
                    spearman_corr, spearman_p = stats.spearmanr(feature_col, target_col)
                    
                    results['predictive_power'] = {
                        'pearson_correlation': float(pearson_corr),
                        'pearson_p_value': float(pearson_p),
                        'spearman_correlation': float(spearman_corr),
                        'spearman_p_value': float(spearman_p),
                        'significant_correlation': min(pearson_p, spearman_p) < self.significance_level
                    }
                    
                    # Information ratio (if target represents returns)
                    if target_col.std() > 0:
                        information_ratio = feature_col.mean() / target_col.std()
                        results['information_ratio'] = float(information_ratio)
                        
        except Exception as e:
            logger.error(f"Error in information content calculation: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def validate_feature_quality(
        self, 
        feature_data: pd.Series, 
        feature_name: str = "",
        target_data: Optional[pd.Series] = None,
        comprehensive: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive feature quality validation.
        
        Args:
            feature_data: Feature values
            feature_name: Feature name for context
            target_data: Optional target for predictive assessment
            comprehensive: Whether to run all tests (slower but thorough)
            
        Returns:
            Complete quality assessment results
        """
        logger.info(f"Validating feature quality: {feature_name}")
        start_time = datetime.now()
        self._check_memory_usage("quality_validation_start")
        
        # Initialize results
        quality_results = {
            'feature_name': feature_name,
            'overall_quality_score': 0.0,
            'quality_issues': [],
            'quality_warnings': [],
            'taiwan_compliant': True,
            'recommendation': 'unknown',
            'processing_time_seconds': 0.0
        }
        
        try:
            # Basic statistics (always run)
            quality_results['basic_statistics'] = self.calculate_basic_statistics(feature_data)
            
            # Check for critical issues that would disqualify the feature
            basic_stats = quality_results['basic_statistics']
            if 'error' in basic_stats:
                quality_results['overall_quality_score'] = 0.0
                quality_results['quality_issues'].append('Basic statistics calculation failed')
                quality_results['recommendation'] = 'reject'
                return quality_results
                
            # Data completeness check
            missing_pct = basic_stats.get('missing_pct', 0)
            if missing_pct > self.quality_thresholds['missing_data_max_pct']:
                quality_results['quality_issues'].append(f'High missing data: {missing_pct:.1f}%')
                
            # Unique values check
            unique_count = basic_stats.get('unique_values', 0)
            if unique_count < self.quality_thresholds['min_unique_values']:
                quality_results['quality_issues'].append(f'Too few unique values: {unique_count}')
                
            # Outlier detection
            outlier_results = self.detect_outliers(feature_data)
            quality_results['outliers'] = outlier_results
            if 'z_score' in outlier_results:
                outlier_pct = outlier_results['z_score'].get('outlier_pct', 0)
                if outlier_pct > self.quality_thresholds['outlier_max_pct']:
                    quality_results['quality_warnings'].append(f'High outlier rate: {outlier_pct:.1f}%')
                    
            # Taiwan compliance validation
            taiwan_results = self.validate_taiwan_compliance(feature_data, feature_name)
            quality_results['taiwan_compliance'] = taiwan_results
            quality_results['taiwan_compliant'] = taiwan_results['compliant']
            if not taiwan_results['compliant']:
                quality_results['quality_issues'].extend(taiwan_results['violations'])
            quality_results['quality_warnings'].extend(taiwan_results['warnings'])
            
            # Comprehensive tests (optional)
            if comprehensive:
                # Normality tests
                quality_results['normality'] = self.test_normality(feature_data)
                
                # Stationarity tests (for time series)
                if len(feature_data) >= 20:
                    quality_results['stationarity'] = self.test_stationarity(feature_data)
                    
                # Autocorrelation analysis
                quality_results['autocorrelation'] = self.calculate_autocorrelation(feature_data)
                
                # Financial properties
                quality_results['financial_properties'] = self.assess_financial_properties(feature_data)
                
                # Information content
                quality_results['information_content'] = self.calculate_information_content(
                    feature_data, target_data
                )
                
            # Calculate overall quality score (0-100)
            score_components = []
            
            # Data completeness (0-25 points)
            completeness_score = max(0, 25 - missing_pct)
            score_components.append(completeness_score)
            
            # Statistical properties (0-25 points)
            stat_score = 25
            if abs(basic_stats.get('skewness', 0)) > self.quality_thresholds['skewness_max_abs']:
                stat_score -= 10
            if basic_stats.get('kurtosis', 0) > self.quality_thresholds['kurtosis_max']:
                stat_score -= 10
            score_components.append(max(0, stat_score))
            
            # Taiwan compliance (0-25 points)
            compliance_score = 25 if taiwan_results['compliant'] else 0
            compliance_score -= len(taiwan_results['warnings']) * 2
            score_components.append(max(0, compliance_score))
            
            # Information content (0-25 points)
            info_score = 20  # Base score
            if unique_count >= self.quality_thresholds['min_unique_values']:
                info_score += 5
            if target_data is not None and 'predictive_power' in quality_results.get('information_content', {}):
                pred_power = quality_results['information_content']['predictive_power']
                if pred_power.get('significant_correlation', False):
                    info_score += 5
            score_components.append(min(25, info_score))
            
            quality_results['overall_quality_score'] = float(np.mean(score_components))
            
            # Generate recommendation
            if quality_results['overall_quality_score'] >= 80 and quality_results['taiwan_compliant']:
                quality_results['recommendation'] = 'accept'
            elif quality_results['overall_quality_score'] >= 60 and len(quality_results['quality_issues']) == 0:
                quality_results['recommendation'] = 'conditional_accept'
            else:
                quality_results['recommendation'] = 'reject'
                
        except Exception as e:
            logger.error(f"Error in feature quality validation: {str(e)}")
            quality_results['quality_issues'].append(f'Validation error: {str(e)}')
            quality_results['overall_quality_score'] = 0.0
            quality_results['recommendation'] = 'reject'
            
        # Record processing time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        quality_results['processing_time_seconds'] = elapsed_time
        
        self._check_memory_usage("quality_validation_end")
        
        logger.info(
            f"Quality validation completed for {feature_name}: "
            f"score={quality_results['overall_quality_score']:.1f}, "
            f"recommendation={quality_results['recommendation']}"
        )
        
        return quality_results
    
    def batch_validate_features(
        self, 
        features_df: pd.DataFrame, 
        target_data: Optional[pd.Series] = None,
        comprehensive: bool = False,
        n_jobs: int = 1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate quality of multiple features in batch.
        
        Args:
            features_df: DataFrame with features to validate
            target_data: Optional target variable
            comprehensive: Whether to run comprehensive tests
            n_jobs: Number of parallel jobs (currently sequential)
            
        Returns:
            Dictionary mapping feature names to quality results
        """
        logger.info(f"Batch validating {len(features_df.columns)} features")
        start_time = datetime.now()
        
        results = {}
        
        for i, feature_name in enumerate(features_df.columns):
            logger.info(f"Processing feature {i+1}/{len(features_df.columns)}: {feature_name}")
            
            feature_data = features_df[feature_name]
            feature_results = self.validate_feature_quality(
                feature_data, 
                feature_name, 
                target_data, 
                comprehensive
            )
            
            results[feature_name] = feature_results
            
            # Log progress every 10 features
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(features_df.columns)} features")
                
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Batch validation completed in {elapsed_time:.1f}s")
        
        return results
    
    def generate_quality_report(
        self, 
        validation_results: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality assessment report.
        
        Args:
            validation_results: Results from batch_validate_features
            output_path: Optional path to save report
            
        Returns:
            Quality report summary
        """
        report = {
            'summary': {
                'total_features': len(validation_results),
                'timestamp': datetime.now().isoformat(),
                'taiwan_market_config': str(self.taiwan_config)
            },
            'recommendations': {
                'accept': [],
                'conditional_accept': [],
                'reject': []
            },
            'quality_statistics': {},
            'common_issues': {},
            'taiwan_compliance_summary': {}
        }
        
        # Analyze results
        quality_scores = []
        recommendations = []
        taiwan_compliant = []
        
        for feature_name, results in validation_results.items():
            quality_scores.append(results.get('overall_quality_score', 0))
            recommendations.append(results.get('recommendation', 'unknown'))
            taiwan_compliant.append(results.get('taiwan_compliant', False))
            
            # Categorize by recommendation
            recommendation = results.get('recommendation', 'unknown')
            if recommendation in report['recommendations']:
                report['recommendations'][recommendation].append(feature_name)
                
        # Calculate summary statistics
        report['quality_statistics'] = {
            'mean_quality_score': float(np.mean(quality_scores)),
            'median_quality_score': float(np.median(quality_scores)),
            'min_quality_score': float(np.min(quality_scores)),
            'max_quality_score': float(np.max(quality_scores)),
            'std_quality_score': float(np.std(quality_scores))
        }
        
        # Taiwan compliance summary
        report['taiwan_compliance_summary'] = {
            'compliant_features': sum(taiwan_compliant),
            'compliant_percentage': sum(taiwan_compliant) / len(taiwan_compliant) * 100 if taiwan_compliant else 0,
            'non_compliant_features': len(taiwan_compliant) - sum(taiwan_compliant)
        }
        
        # Recommendation summary
        for rec_type in ['accept', 'conditional_accept', 'reject']:
            count = len(report['recommendations'][rec_type])
            percentage = count / len(validation_results) * 100 if validation_results else 0
            report['summary'][f'{rec_type}_count'] = count
            report['summary'][f'{rec_type}_percentage'] = percentage
            
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"Quality report saved to {output_path}")
            
        return report


def create_quality_metrics_calculator(
    taiwan_config: Optional[TaiwanMarketConfig] = None,
    **kwargs
) -> FeatureQualityMetrics:
    """
    Factory function to create a properly configured FeatureQualityMetrics.
    
    Args:
        taiwan_config: Taiwan market configuration
        **kwargs: Additional configuration options
        
    Returns:
        Configured FeatureQualityMetrics instance
    """
    # Default configuration
    config = {
        'significance_level': 0.05,
        'outlier_threshold': 3.0,
        'min_observations': 252,
        'memory_limit_gb': 4.0
    }
    
    # Apply overrides
    config.update(kwargs)
    
    # Create metrics calculator
    calculator = FeatureQualityMetrics(
        taiwan_config=taiwan_config or TaiwanMarketConfig(),
        **config
    )
    
    logger.info("Created FeatureQualityMetrics calculator")
    return calculator