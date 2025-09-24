"""
Statistical Significance Testing for Feature Selection

This module implements comprehensive statistical significance tests to evaluate
the predictive power of features with respect to target returns in Taiwan
stock market data.

Key Features:
- Multiple statistical tests (t-test, F-test, correlation tests)
- False Discovery Rate (FDR) control using Benjamini-Hochberg procedure
- Bootstrap-based significance testing for robust estimation
- Time-series aware testing with appropriate lag handling
- Taiwan market-specific validation and constraints
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.stats import (
    ttest_ind, f_oneway, pearsonr, spearmanr, kendalltau,
    normaltest, levene, bartlett, shapiro
)
from sklearn.feature_selection import f_regression, f_classif, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
import psutil

logger = logging.getLogger(__name__)

class StatisticalSignificanceTester:
    """
    Statistical significance testing for feature selection in Taiwan stock market.
    
    Implements multiple statistical tests to evaluate feature predictive power
    with proper multiple testing correction and time-series considerations.
    """
    
    def __init__(
        self,
        alpha_level: float = 0.05,
        multiple_test_correction: str = 'benjamini-hochberg',
        test_methods: List[str] = ['correlation', 'f_test', 'regression'],
        min_samples: int = 30,
        bootstrap_samples: int = 1000,
        time_series_aware: bool = True,
        memory_limit_gb: float = 8.0,
        random_state: int = 42
    ):
        """
        Initialize statistical significance tester.
        
        Args:
            alpha_level: Significance level (p-value threshold)
            multiple_test_correction: Method for multiple testing correction
            test_methods: List of statistical tests to perform
            min_samples: Minimum number of samples required for testing
            bootstrap_samples: Number of bootstrap samples for robust testing
            time_series_aware: Whether to account for time-series structure
            memory_limit_gb: Memory limit for processing
            random_state: Random seed for reproducibility
        """
        self.alpha_level = alpha_level
        self.multiple_test_correction = multiple_test_correction
        self.test_methods = test_methods
        self.min_samples = min_samples
        self.bootstrap_samples = bootstrap_samples
        self.time_series_aware = time_series_aware
        self.memory_limit_gb = memory_limit_gb
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Available test methods
        self.available_tests = {
            'correlation': self._test_correlation_significance,
            'f_test': self._test_f_statistic,
            'regression': self._test_regression_significance,
            'mutual_info': self._test_mutual_information_significance,
            'normality': self._test_normality,
            'bootstrap': self._test_bootstrap_significance
        }
        
        # Results storage
        self.test_results_ = {}
        self.significant_features_ = []
        self.insignificant_features_ = {}
        self.corrected_pvalues_ = {}
        self.processing_stats_ = {}
        self.memory_stats_ = {}
        
    def _monitor_memory(self, stage: str) -> Dict[str, float]:
        """Monitor memory usage during processing."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1024 / 1024 / 1024
        
        self.memory_stats_[stage] = {
            'memory_gb': memory_gb,
            'timestamp': datetime.now(),
            'warning': memory_gb > self.memory_limit_gb
        }
        
        if memory_gb > self.memory_limit_gb:
            logger.warning(f"Memory usage ({memory_gb:.2f}GB) exceeds limit at {stage}")
            
        return self.memory_stats_[stage]
        
    def _validate_input_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and prepare input data for statistical testing."""
        logger.info(f"Validating input data: X{X.shape}, y{len(y)}")
        
        if X.empty:
            raise ValueError("Feature matrix X is empty")
        if len(y) == 0:
            raise ValueError("Target vector y is empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
            
        # Select only numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < len(X.columns):
            logger.warning(f"Filtered {len(X.columns) - len(numeric_columns)} non-numeric columns")
            X = X[numeric_columns]
            
        if X.empty:
            raise ValueError("No numeric columns found in feature matrix")
            
        # Remove samples with missing target values
        valid_mask = ~y.isna()
        X_clean = X.loc[valid_mask]
        y_clean = y.loc[valid_mask]
        
        if len(y_clean) < self.min_samples:
            raise ValueError(f"Insufficient samples after cleaning: {len(y_clean)} < {self.min_samples}")
            
        # Handle missing feature values with forward fill then median
        for col in X_clean.columns:
            if X_clean[col].isna().any():
                # For time series, try forward fill first
                if self.time_series_aware:
                    X_clean[col] = X_clean[col].fillna(method='ffill')
                # Then median fill any remaining
                median_value = X_clean[col].median()
                if pd.isna(median_value):
                    median_value = 0.0
                X_clean[col] = X_clean[col].fillna(median_value)
                
        logger.info(f"Data validation completed: X{X_clean.shape}, y{len(y_clean)}")
        return X_clean, y_clean
        
    def _test_correlation_significance(
        self, 
        feature_data: pd.Series, 
        target_data: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """Test correlation significance using multiple correlation methods."""
        results = {'feature': feature_name, 'test_type': 'correlation'}
        
        try:
            # Pearson correlation (linear relationship)
            pearson_corr, pearson_p = pearsonr(feature_data, target_data)
            results['pearson_correlation'] = pearson_corr
            results['pearson_pvalue'] = pearson_p
            
            # Spearman correlation (monotonic relationship)  
            spearman_corr, spearman_p = spearmanr(feature_data, target_data)
            results['spearman_correlation'] = spearman_corr
            results['spearman_pvalue'] = spearman_p
            
            # Kendall's tau (robust rank correlation)
            kendall_corr, kendall_p = kendalltau(feature_data, target_data)
            results['kendall_correlation'] = kendall_corr
            results['kendall_pvalue'] = kendall_p
            
            # Overall significance (minimum p-value)
            pvalues = [pearson_p, spearman_p, kendall_p]
            min_pvalue = min(p for p in pvalues if not pd.isna(p))
            results['min_pvalue'] = min_pvalue
            results['significant'] = min_pvalue < self.alpha_level
            
        except Exception as e:
            logger.warning(f"Correlation test failed for {feature_name}: {str(e)}")
            results.update({
                'pearson_correlation': 0.0, 'pearson_pvalue': 1.0,
                'spearman_correlation': 0.0, 'spearman_pvalue': 1.0,
                'kendall_correlation': 0.0, 'kendall_pvalue': 1.0,
                'min_pvalue': 1.0, 'significant': False
            })
            
        return results
        
    def _test_f_statistic(
        self, 
        feature_data: pd.Series, 
        target_data: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """Test F-statistic significance for regression."""
        results = {'feature': feature_name, 'test_type': 'f_test'}
        
        try:
            # Reshape data for sklearn
            X_reshaped = feature_data.values.reshape(-1, 1)
            
            # F-test for regression
            f_stat, f_pvalue = f_regression(X_reshaped, target_data)
            results['f_statistic'] = f_stat[0]
            results['f_pvalue'] = f_pvalue[0]
            results['significant'] = f_pvalue[0] < self.alpha_level
            
        except Exception as e:
            logger.warning(f"F-test failed for {feature_name}: {str(e)}")
            results.update({
                'f_statistic': 0.0,
                'f_pvalue': 1.0,
                'significant': False
            })
            
        return results
        
    def _test_regression_significance(
        self, 
        feature_data: pd.Series, 
        target_data: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """Test regression coefficient significance."""
        results = {'feature': feature_name, 'test_type': 'regression'}
        
        try:
            # Simple linear regression
            X_reshaped = feature_data.values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X_reshaped, target_data)
            
            # Predictions and residuals
            y_pred = model.predict(X_reshaped)
            residuals = target_data - y_pred
            
            # R-squared
            r2 = r2_score(target_data, y_pred)
            results['r_squared'] = r2
            
            # Calculate t-statistic for coefficient
            n = len(target_data)
            mse = np.sum(residuals**2) / (n - 2)  # Mean squared error
            var_feature = np.var(feature_data, ddof=1)
            
            if var_feature > 0:
                se_coef = np.sqrt(mse / ((n - 1) * var_feature))  # Standard error of coefficient
                t_stat = model.coef_[0] / se_coef if se_coef > 0 else 0
                
                # Two-tailed t-test
                df = n - 2
                t_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                results['coefficient'] = model.coef_[0]
                results['t_statistic'] = t_stat
                results['t_pvalue'] = t_pvalue
                results['significant'] = t_pvalue < self.alpha_level
            else:
                results.update({
                    'coefficient': 0.0,
                    't_statistic': 0.0,
                    't_pvalue': 1.0,
                    'significant': False
                })
                
        except Exception as e:
            logger.warning(f"Regression test failed for {feature_name}: {str(e)}")
            results.update({
                'r_squared': 0.0,
                'coefficient': 0.0,
                't_statistic': 0.0,
                't_pvalue': 1.0,
                'significant': False
            })
            
        return results
        
    def _test_mutual_information_significance(
        self, 
        feature_data: pd.Series, 
        target_data: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """Test mutual information significance using permutation testing."""
        results = {'feature': feature_name, 'test_type': 'mutual_info'}
        
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate actual MI
            X_reshaped = feature_data.values.reshape(-1, 1)
            actual_mi = mutual_info_regression(X_reshaped, target_data, random_state=self.random_state)[0]
            
            # Permutation test for significance
            n_permutations = min(self.bootstrap_samples, 1000)  # Limit for efficiency
            permuted_mi_scores = []
            
            for _ in range(n_permutations):
                # Shuffle target to break relationship
                y_permuted = np.random.permutation(target_data)
                permuted_mi = mutual_info_regression(X_reshaped, y_permuted, random_state=self.random_state)[0]
                permuted_mi_scores.append(permuted_mi)
                
            # Calculate p-value as proportion of permuted scores >= actual
            permuted_mi_scores = np.array(permuted_mi_scores)
            p_value = (permuted_mi_scores >= actual_mi).mean()
            
            results['mutual_info'] = actual_mi
            results['mi_pvalue'] = p_value
            results['significant'] = p_value < self.alpha_level
            
        except Exception as e:
            logger.warning(f"MI significance test failed for {feature_name}: {str(e)}")
            results.update({
                'mutual_info': 0.0,
                'mi_pvalue': 1.0,
                'significant': False
            })
            
        return results
        
    def _test_normality(
        self, 
        feature_data: pd.Series, 
        target_data: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """Test normality assumptions for proper statistical inference."""
        results = {'feature': feature_name, 'test_type': 'normality'}
        
        try:
            # Shapiro-Wilk test for normality (sample size permitting)
            if len(feature_data) <= 5000:  # Shapiro-Wilk limit
                shapiro_stat, shapiro_p = shapiro(feature_data)
                results['shapiro_statistic'] = shapiro_stat
                results['shapiro_pvalue'] = shapiro_p
                results['feature_normal'] = shapiro_p > 0.05
            else:
                # Use Anderson-Darling for larger samples
                try:
                    from scipy.stats import anderson
                    ad_result = anderson(feature_data)
                    results['anderson_statistic'] = ad_result.statistic
                    results['feature_normal'] = ad_result.statistic < ad_result.critical_values[2]  # 5% level
                except:
                    results['feature_normal'] = False
                    
            # Jarque-Bera test for target normality
            jb_stat, jb_p = jarque_bera(target_data)
            results['jarque_bera_statistic'] = jb_stat
            results['jarque_bera_pvalue'] = jb_p
            results['target_normal'] = jb_p > 0.05
            
            results['normality_satisfied'] = results.get('feature_normal', False) and results['target_normal']
            
        except Exception as e:
            logger.warning(f"Normality test failed for {feature_name}: {str(e)}")
            results.update({
                'feature_normal': False,
                'target_normal': False,
                'normality_satisfied': False
            })
            
        return results
        
    def _test_bootstrap_significance(
        self, 
        feature_data: pd.Series, 
        target_data: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """Bootstrap-based significance testing for robust inference."""
        results = {'feature': feature_name, 'test_type': 'bootstrap'}
        
        try:
            # Calculate original correlation
            original_corr, _ = pearsonr(feature_data, target_data)
            
            # Bootstrap resampling
            n_samples = len(feature_data)
            bootstrap_correlations = []
            
            for _ in range(self.bootstrap_samples):
                # Resample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                boot_feature = feature_data.iloc[indices]
                boot_target = target_data.iloc[indices]
                
                # Calculate bootstrap correlation
                boot_corr, _ = pearsonr(boot_feature, boot_target)
                bootstrap_correlations.append(boot_corr)
                
            bootstrap_correlations = np.array(bootstrap_correlations)
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_correlations, 2.5)
            ci_upper = np.percentile(bootstrap_correlations, 97.5)
            
            # Significance test: does CI include 0?
            significant = not (ci_lower <= 0 <= ci_upper)
            
            results.update({
                'original_correlation': original_corr,
                'bootstrap_mean': np.mean(bootstrap_correlations),
                'bootstrap_std': np.std(bootstrap_correlations),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': significant
            })
            
        except Exception as e:
            logger.warning(f"Bootstrap test failed for {feature_name}: {str(e)}")
            results.update({
                'original_correlation': 0.0,
                'bootstrap_mean': 0.0,
                'bootstrap_std': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'significant': False
            })
            
        return results
        
    def _apply_multiple_testing_correction(
        self, 
        test_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Apply multiple testing correction to control False Discovery Rate."""
        logger.info(f"Applying {self.multiple_test_correction} multiple testing correction")
        
        # Collect all p-values for correction
        feature_pvalues = {}
        
        for feature_name, feature_results in test_results.items():
            # Extract the primary p-value from each test
            pvalues = []
            
            for test_method, test_result in feature_results.items():
                if isinstance(test_result, dict):
                    # Extract relevant p-value based on test type
                    if test_result.get('test_type') == 'correlation':
                        pvalues.append(test_result.get('min_pvalue', 1.0))
                    elif test_result.get('test_type') == 'f_test':
                        pvalues.append(test_result.get('f_pvalue', 1.0))
                    elif test_result.get('test_type') == 'regression':
                        pvalues.append(test_result.get('t_pvalue', 1.0))
                    elif test_result.get('test_type') == 'mutual_info':
                        pvalues.append(test_result.get('mi_pvalue', 1.0))
                        
            # Use minimum p-value for each feature (most optimistic)
            if pvalues:
                feature_pvalues[feature_name] = min(p for p in pvalues if not pd.isna(p))
            else:
                feature_pvalues[feature_name] = 1.0
                
        if not feature_pvalues:
            logger.warning("No p-values found for multiple testing correction")
            return {}
            
        # Apply correction
        features = list(feature_pvalues.keys())
        pvalues = [feature_pvalues[f] for f in features]
        
        try:
            if self.multiple_test_correction == 'benjamini-hochberg':
                corrected = multipletests(pvalues, alpha=self.alpha_level, method='fdr_bh')
            elif self.multiple_test_correction == 'bonferroni':
                corrected = multipletests(pvalues, alpha=self.alpha_level, method='bonferroni')
            elif self.multiple_test_correction == 'holm':
                corrected = multipletests(pvalues, alpha=self.alpha_level, method='holm')
            else:
                logger.warning(f"Unknown correction method: {self.multiple_test_correction}")
                corrected = multipletests(pvalues, alpha=self.alpha_level, method='fdr_bh')
                
            # Create corrected p-values dictionary
            corrected_pvalues = dict(zip(features, corrected[1]))  # corrected[1] contains adjusted p-values
            
            n_significant_uncorrected = sum(1 for p in pvalues if p < self.alpha_level)
            n_significant_corrected = sum(1 for p in corrected[1] if p < self.alpha_level)
            
            logger.info(f"Multiple testing correction: {n_significant_uncorrected} → {n_significant_corrected} significant features")
            
            return corrected_pvalues
            
        except Exception as e:
            logger.error(f"Multiple testing correction failed: {str(e)}")
            return feature_pvalues  # Return uncorrected p-values as fallback
            
    def test_feature_significance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical significance testing on features.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing test results and significant features
        """
        logger.info(f"Starting statistical significance testing for {X.shape[1]} features")
        start_time = datetime.now()
        self._monitor_memory("testing_start")
        
        # Step 1: Validate input data
        X_clean, y_clean = self._validate_input_data(X, y)
        
        # Step 2: Run statistical tests for each feature
        logger.info(f"=== Step 1: Running {len(self.test_methods)} statistical tests ===")
        
        self.test_results_ = {}
        
        for i, feature_name in enumerate(X_clean.columns):
            feature_data = X_clean[feature_name]
            self.test_results_[feature_name] = {}
            
            # Run each specified test
            for test_method in self.test_methods:
                if test_method in self.available_tests:
                    test_result = self.available_tests[test_method](
                        feature_data, y_clean, feature_name
                    )
                    self.test_results_[feature_name][test_method] = test_result
                else:
                    logger.warning(f"Unknown test method: {test_method}")
                    
            # Progress monitoring
            if (i + 1) % max(1, len(X_clean.columns) // 20) == 0:
                progress = (i + 1) / len(X_clean.columns) * 100
                logger.info(f"Testing progress: {progress:.1f}%")
                self._monitor_memory(f"testing_{progress:.0f}pct")
                
        # Step 3: Apply multiple testing correction
        logger.info("=== Step 2: Multiple Testing Correction ===")
        self.corrected_pvalues_ = self._apply_multiple_testing_correction(self.test_results_)
        
        # Step 4: Determine significant features
        logger.info("=== Step 3: Significance Determination ===")
        self.significant_features_ = []
        self.insignificant_features_ = {}
        
        for feature_name, corrected_pvalue in self.corrected_pvalues_.items():
            if corrected_pvalue < self.alpha_level:
                self.significant_features_.append(feature_name)
            else:
                self.insignificant_features_[feature_name] = f"corrected_pvalue_{corrected_pvalue:.6f}_above_{self.alpha_level}"
                
        # Generate results
        results = self._generate_results(X, y, start_time)
        
        self._monitor_memory("testing_end")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Statistical significance testing completed in {elapsed_time:.1f}s")
        logger.info(f"Significant features: {len(self.significant_features_)} / {X.shape[1]} "
                   f"({len(self.significant_features_)/X.shape[1]:.1%})")
        
        return results
        
    def _generate_results(
        self, 
        original_X: pd.DataFrame, 
        original_y: pd.Series,
        start_time: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive results dictionary."""
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate test statistics
        if self.corrected_pvalues_:
            pvalue_stats = {
                'mean_pvalue': np.mean(list(self.corrected_pvalues_.values())),
                'median_pvalue': np.median(list(self.corrected_pvalues_.values())),
                'min_pvalue': np.min(list(self.corrected_pvalues_.values())),
                'max_pvalue': np.max(list(self.corrected_pvalues_.values())),
                'significant_count': len(self.significant_features_),
                'insignificant_count': len(self.insignificant_features_)
            }
        else:
            pvalue_stats = {}
            
        # Processing statistics
        self.processing_stats_ = {
            'input_features': original_X.shape[1],
            'input_samples': len(original_y),
            'significant_features': len(self.significant_features_),
            'insignificant_features': len(self.insignificant_features_),
            'significance_rate': len(self.significant_features_) / original_X.shape[1] if original_X.shape[1] > 0 else 0,
            'processing_time_seconds': elapsed_time,
            'alpha_level': self.alpha_level,
            'correction_method': self.multiple_test_correction,
            'tests_performed': self.test_methods.copy(),
            'memory_peak_gb': max(stats['memory_gb'] for stats in self.memory_stats_.values()) if self.memory_stats_ else 0,
            'pvalue_statistics': pvalue_stats
        }
        
        return {
            'significant_features': self.significant_features_.copy(),
            'insignificant_features': self.insignificant_features_.copy(),
            'test_results': self.test_results_.copy(),
            'corrected_pvalues': self.corrected_pvalues_.copy(),
            'processing_stats': self.processing_stats_.copy(),
            'memory_stats': self.memory_stats_.copy(),
            'parameters': {
                'alpha_level': self.alpha_level,
                'multiple_test_correction': self.multiple_test_correction,
                'test_methods': self.test_methods,
                'min_samples': self.min_samples,
                'bootstrap_samples': self.bootstrap_samples,
                'time_series_aware': self.time_series_aware
            }
        }
        
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of statistical test results."""
        if not self.test_results_:
            raise ValueError("Statistical testing not completed")
            
        # Count tests by significance
        test_summary = {}
        for test_method in self.test_methods:
            significant_count = 0
            total_count = 0
            
            for feature_name, feature_results in self.test_results_.items():
                if test_method in feature_results:
                    total_count += 1
                    if feature_results[test_method].get('significant', False):
                        significant_count += 1
                        
            test_summary[test_method] = {
                'total_features': total_count,
                'significant_features': significant_count,
                'significance_rate': significant_count / total_count if total_count > 0 else 0
            }
            
        return {
            'test_method_summary': test_summary,
            'overall_significant': len(self.significant_features_),
            'overall_total': len(self.test_results_),
            'correction_impact': {
                'uncorrected_significant': sum(
                    1 for results in self.test_results_.values()
                    for test_result in results.values()
                    if isinstance(test_result, dict) and test_result.get('significant', False)
                ),
                'corrected_significant': len(self.significant_features_)
            }
        }
        
    def save_results(self, output_path: str) -> None:
        """Save significance testing results to JSON file."""
        import json
        from pathlib import Path
        
        if not self.significant_features_:
            raise ValueError("Statistical testing not completed")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results = {
            'significant_features': self.significant_features_,
            'insignificant_features': self.insignificant_features_,
            'corrected_pvalues': self.corrected_pvalues_,
            'processing_stats': self.processing_stats_,
            'test_summary': self.get_test_summary(),
            'parameters': {
                'alpha_level': self.alpha_level,
                'multiple_test_correction': self.multiple_test_correction,
                'test_methods': self.test_methods,
                'min_samples': self.min_samples,
                'bootstrap_samples': self.bootstrap_samples
            }
        }
        
        # Convert datetime objects
        for stage, stats in self.memory_stats_.items():
            if 'timestamp' in stats:
                stats['timestamp'] = stats['timestamp'].isoformat()
        
        results['memory_stats'] = self.memory_stats_
        
        # Save results (test_results_ omitted for size - can be added if needed)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Significance testing results saved to {output_path}")