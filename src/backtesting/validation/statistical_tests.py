"""
Statistical Testing Framework for Walk-Forward Validation.

This module implements comprehensive statistical significance testing for
walk-forward validation results, specifically designed for Taiwan market
quantitative trading strategies.

Key Features:
- Diebold-Mariano Test for forecast accuracy comparison
- Hansen SPA Test for model selection with multiple comparisons
- White Reality Check for data mining bias correction
- Bootstrap Confidence Intervals for robust statistical inference
- Integration with Taiwan market validation cycles
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import warnings
from decimal import Decimal

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.models.taiwan_market import TaiwanMarketCode, TaiwanTradingCalendar
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel
from .walk_forward import ValidationWindow, ValidationResult
from ..metrics.performance import PerformanceMetrics, PerformanceConfig

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of statistical tests."""
    DIEBOLD_MARIANO = "diebold_mariano"
    HANSEN_SPA = "hansen_spa"
    WHITE_REALITY_CHECK = "white_reality_check"
    BOOTSTRAP_CI = "bootstrap_ci"
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"


class BootstrapMethod(Enum):
    """Bootstrap sampling methods."""
    STATIONARY = "stationary"        # Stationary bootstrap
    CIRCULAR = "circular"            # Circular block bootstrap
    MOVING_BLOCK = "moving_block"    # Moving block bootstrap
    WILD = "wild"                    # Wild bootstrap


@dataclass
class StatisticalTestConfig:
    """Configuration for statistical testing."""
    # Significance levels
    alpha_level: float = 0.05  # 5% significance level
    confidence_level: float = 0.95  # 95% confidence level
    
    # Bootstrap parameters
    bootstrap_iterations: int = 10000
    bootstrap_method: BootstrapMethod = BootstrapMethod.STATIONARY
    block_length: Optional[int] = None  # Auto-calculated if None
    
    # Multiple comparison correction
    multiple_comparison_method: str = "fdr_bh"  # FDR Benjamini-Hochberg
    enable_bonferroni: bool = False
    
    # Test-specific parameters
    diebold_mariano_lags: int = 1
    hansen_spa_bootstrap_iterations: int = 5000
    white_rc_bootstrap_iterations: int = 5000
    
    # Statistical power analysis
    enable_power_analysis: bool = True
    minimum_power: float = 0.8
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.alpha_level < 1:
            raise ValueError("Alpha level must be between 0 and 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.bootstrap_iterations <= 0:
            raise ValueError("Bootstrap iterations must be positive")


@dataclass
class TestResult:
    """Result from a statistical test."""
    test_type: TestType
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Additional test-specific results
    degrees_of_freedom: Optional[int] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None
    
    # Metadata
    test_date: datetime = field(default_factory=datetime.now)
    sample_size: int = 0
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    
    # Decision flags
    is_significant: bool = field(init=False)
    reject_null: bool = field(init=False)
    
    def __post_init__(self):
        """Calculate decision flags."""
        # These will be set by the test method based on alpha level
        self.is_significant = False
        self.reject_null = False
    
    def set_significance(self, alpha_level: float) -> None:
        """Set significance flags based on alpha level."""
        self.is_significant = self.p_value < alpha_level
        self.reject_null = self.is_significant
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_type': self.test_type.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'critical_value': self.critical_value,
            'confidence_interval': self.confidence_interval,
            'degrees_of_freedom': self.degrees_of_freedom,
            'effect_size': self.effect_size,
            'power': self.power,
            'test_date': self.test_date.isoformat(),
            'sample_size': self.sample_size,
            'null_hypothesis': self.null_hypothesis,
            'alternative_hypothesis': self.alternative_hypothesis,
            'is_significant': self.is_significant,
            'reject_null': self.reject_null
        }


@dataclass
class MultipleTestResult:
    """Results from multiple comparison testing."""
    individual_results: List[TestResult]
    adjusted_p_values: List[float]
    significant_tests: List[bool]
    method: str
    family_wise_error_rate: float
    false_discovery_rate: float
    
    def get_significant_results(self) -> List[TestResult]:
        """Get only significant test results."""
        return [
            result for result, is_sig in zip(self.individual_results, self.significant_tests)
            if is_sig
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'individual_results': [result.to_dict() for result in self.individual_results],
            'adjusted_p_values': self.adjusted_p_values,
            'significant_tests': self.significant_tests,
            'method': self.method,
            'family_wise_error_rate': self.family_wise_error_rate,
            'false_discovery_rate': self.false_discovery_rate,
            'num_significant': sum(self.significant_tests),
            'total_tests': len(self.individual_results)
        }


class StatisticalTestEngine:
    """
    Core statistical testing engine for validation results.
    
    Implements comprehensive statistical significance testing including:
    - Diebold-Mariano Test for forecast accuracy comparison
    - Hansen SPA Test for model selection
    - White Reality Check for data mining bias
    - Bootstrap confidence intervals
    """
    
    def __init__(self, config: StatisticalTestConfig):
        self.config = config
        
        # Initialize random state for reproducible results
        self.random_state = np.random.RandomState(42)
        
        logger.info("StatisticalTestEngine initialized")
    
    def diebold_mariano_test(
        self,
        forecast_errors_1: Union[pd.Series, np.ndarray],
        forecast_errors_2: Union[pd.Series, np.ndarray],
        alternative: str = "two-sided"
    ) -> TestResult:
        """
        Diebold-Mariano test for forecast accuracy comparison.
        
        Tests whether two forecasts have equal accuracy.
        
        Args:
            forecast_errors_1: Forecast errors from first model
            forecast_errors_2: Forecast errors from second model
            alternative: "two-sided", "greater", or "less"
            
        Returns:
            Statistical test result
        """
        logger.info("Running Diebold-Mariano test")
        
        # Convert to numpy arrays
        e1 = np.asarray(forecast_errors_1)
        e2 = np.asarray(forecast_errors_2)
        
        if len(e1) != len(e2):
            raise ValueError("Forecast error series must have same length")
        
        if len(e1) < 10:
            raise ValueError("Insufficient observations for Diebold-Mariano test")
        
        # Calculate loss differential (squared errors)
        d = e1**2 - e2**2
        
        # Calculate mean loss differential
        d_mean = np.mean(d)
        
        # Calculate long-run variance using Newey-West estimator
        n = len(d)
        lags = self.config.diebold_mariano_lags
        
        # Calculate autocovariances
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0
        
        for j in range(1, lags + 1):
            if j < n:
                gamma_j = np.cov(d[:-j], d[j:])[0, 1]
                gamma_sum += 2 * (1 - j / (lags + 1)) * gamma_j
        
        # Long-run variance
        long_run_var = gamma_0 + gamma_sum
        
        if long_run_var <= 0:
            logger.warning("Non-positive long-run variance, using sample variance")
            long_run_var = np.var(d, ddof=1)
        
        # Calculate DM statistic
        dm_stat = d_mean / np.sqrt(long_run_var / n)
        
        # Calculate p-value based on alternative hypothesis
        if alternative == "two-sided":
            p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
            null_hyp = "Forecasts have equal accuracy"
            alt_hyp = "Forecasts have different accuracy"
        elif alternative == "greater":
            p_value = 1 - norm.cdf(dm_stat)
            null_hyp = "Forecast 1 is not more accurate than forecast 2"
            alt_hyp = "Forecast 1 is more accurate than forecast 2"
        elif alternative == "less":
            p_value = norm.cdf(dm_stat)
            null_hyp = "Forecast 1 is not less accurate than forecast 2"
            alt_hyp = "Forecast 1 is less accurate than forecast 2"
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        # Calculate critical value
        if alternative == "two-sided":
            critical_value = norm.ppf(1 - self.config.alpha_level / 2)
        else:
            critical_value = norm.ppf(1 - self.config.alpha_level)
        
        # Calculate effect size (standardized mean difference)
        effect_size = d_mean / np.sqrt(long_run_var)
        
        result = TestResult(
            test_type=TestType.DIEBOLD_MARIANO,
            statistic=dm_stat,
            p_value=p_value,
            critical_value=critical_value,
            effect_size=effect_size,
            sample_size=n,
            null_hypothesis=null_hyp,
            alternative_hypothesis=alt_hyp
        )
        
        result.set_significance(self.config.alpha_level)
        
        logger.info(f"Diebold-Mariano test completed: DM={dm_stat:.4f}, p={p_value:.4f}")
        return result
    
    def hansen_spa_test(
        self,
        benchmark_returns: Union[pd.Series, np.ndarray],
        model_returns: List[Union[pd.Series, np.ndarray]],
        loss_function: Optional[Callable] = None
    ) -> TestResult:
        """
        Hansen Superior Predictive Ability (SPA) test.
        
        Tests whether any model significantly outperforms the benchmark.
        
        Args:
            benchmark_returns: Benchmark model returns
            model_returns: List of alternative model returns
            loss_function: Loss function (default: negative returns)
            
        Returns:
            Statistical test result
        """
        logger.info(f"Running Hansen SPA test with {len(model_returns)} models")
        
        if loss_function is None:
            loss_function = lambda x: -x  # Negative returns (higher is better)
        
        # Convert to numpy arrays
        benchmark = np.asarray(benchmark_returns)
        models = [np.asarray(model) for model in model_returns]
        
        # Check all series have same length
        n = len(benchmark)
        if not all(len(model) == n for model in models):
            raise ValueError("All return series must have same length")
        
        if n < 20:
            raise ValueError("Insufficient observations for Hansen SPA test")
        
        # Calculate loss differentials
        benchmark_loss = loss_function(benchmark)
        
        loss_diffs = []
        for model in models:
            model_loss = loss_function(model)
            diff = benchmark_loss - model_loss  # Positive if model is better
            loss_diffs.append(diff)
        
        loss_diffs = np.array(loss_diffs).T  # T x k matrix
        
        # Calculate sample means of loss differentials
        d_bar = np.mean(loss_diffs, axis=0)
        
        # Calculate studentized test statistic
        # Variance-covariance matrix
        omega = np.cov(loss_diffs.T)
        
        # Ensure positive definite
        if np.linalg.det(omega) <= 0:
            omega += np.eye(len(models)) * 1e-8
        
        # Calculate t-statistics for each model
        sqrt_n = np.sqrt(n)
        var_d = np.diag(omega)
        
        # Avoid division by zero
        std_d = np.sqrt(np.maximum(var_d, 1e-10))
        t_stats = sqrt_n * d_bar / std_d
        
        # SPA test statistic (max of positive t-statistics)
        spa_stat = np.max(np.maximum(t_stats, 0))
        
        # Bootstrap p-value calculation
        bootstrap_stats = []
        
        for _ in range(self.config.hansen_spa_bootstrap_iterations):
            # Resample residuals
            indices = self.random_state.choice(n, size=n, replace=True)
            bootstrap_diffs = loss_diffs[indices]
            
            # Center the bootstrap sample
            bootstrap_diffs -= np.mean(bootstrap_diffs, axis=0)
            
            # Calculate bootstrap statistics
            d_boot = np.mean(bootstrap_diffs, axis=0)
            omega_boot = np.cov(bootstrap_diffs.T)
            
            # Ensure positive definite
            if np.linalg.det(omega_boot) <= 0:
                omega_boot += np.eye(len(models)) * 1e-8
            
            var_d_boot = np.diag(omega_boot)
            std_d_boot = np.sqrt(np.maximum(var_d_boot, 1e-10))
            
            t_boot = sqrt_n * d_boot / std_d_boot
            spa_boot = np.max(np.maximum(t_boot, 0))
            
            bootstrap_stats.append(spa_boot)
        
        # Calculate p-value
        bootstrap_stats = np.array(bootstrap_stats)
        p_value = np.mean(bootstrap_stats >= spa_stat)
        
        result = TestResult(
            test_type=TestType.HANSEN_SPA,
            statistic=spa_stat,
            p_value=p_value,
            sample_size=n,
            null_hypothesis="No model is superior to the benchmark",
            alternative_hypothesis="At least one model is superior to the benchmark"
        )
        
        result.set_significance(self.config.alpha_level)
        
        logger.info(f"Hansen SPA test completed: SPA={spa_stat:.4f}, p={p_value:.4f}")
        return result
    
    def white_reality_check(
        self,
        benchmark_returns: Union[pd.Series, np.ndarray],
        model_returns: List[Union[pd.Series, np.ndarray]],
        loss_function: Optional[Callable] = None
    ) -> TestResult:
        """
        White Reality Check test for data mining bias.
        
        Tests whether the best performing model significantly outperforms
        the benchmark, correcting for data mining bias.
        
        Args:
            benchmark_returns: Benchmark model returns
            model_returns: List of alternative model returns
            loss_function: Loss function (default: negative returns)
            
        Returns:
            Statistical test result
        """
        logger.info(f"Running White Reality Check with {len(model_returns)} models")
        
        if loss_function is None:
            loss_function = lambda x: -x  # Negative returns (higher is better)
        
        # Convert to numpy arrays
        benchmark = np.asarray(benchmark_returns)
        models = [np.asarray(model) for model in model_returns]
        
        # Check all series have same length
        n = len(benchmark)
        if not all(len(model) == n for model in models):
            raise ValueError("All return series must have same length")
        
        if n < 20:
            raise ValueError("Insufficient observations for White Reality Check")
        
        # Calculate performance differentials
        benchmark_perf = np.mean(loss_function(benchmark))
        
        performance_diffs = []
        for model in models:
            model_perf = np.mean(loss_function(model))
            diff = model_perf - benchmark_perf  # Positive if model is better
            performance_diffs.append(diff)
        
        # Find best performing model
        best_performance = np.max(performance_diffs)
        best_model_idx = np.argmax(performance_diffs)
        
        # Bootstrap test for the best model
        best_model_returns = models[best_model_idx]
        
        # Calculate actual loss differential series
        benchmark_losses = loss_function(benchmark)
        best_model_losses = loss_function(best_model_returns)
        loss_diff_series = best_model_losses - benchmark_losses
        
        # Bootstrap distribution under null hypothesis
        bootstrap_max_perfs = []
        
        for _ in range(self.config.white_rc_bootstrap_iterations):
            # Bootstrap under null (no predictive ability)
            bootstrap_indices = self.random_state.choice(n, size=n, replace=True)
            
            # Center the loss differences (null hypothesis)
            centered_diffs = loss_diff_series - np.mean(loss_diff_series)
            bootstrap_diffs = centered_diffs[bootstrap_indices]
            
            # For each model, calculate bootstrap performance
            bootstrap_model_perfs = []
            for i, model in enumerate(models):
                if i == best_model_idx:
                    # Use the centered bootstrap sample for best model
                    bootstrap_perf = np.mean(bootstrap_diffs)
                else:
                    # For other models, use original calculation with bootstrap
                    model_losses = loss_function(model)
                    bench_losses = loss_function(benchmark)
                    other_diffs = model_losses - bench_losses
                    centered_other = other_diffs - np.mean(other_diffs)
                    bootstrap_perf = np.mean(centered_other[bootstrap_indices])
                
                bootstrap_model_perfs.append(bootstrap_perf)
            
            # Max performance in this bootstrap sample
            max_bootstrap_perf = np.max(bootstrap_model_perfs)
            bootstrap_max_perfs.append(max_bootstrap_perf)
        
        # Calculate p-value
        bootstrap_max_perfs = np.array(bootstrap_max_perfs)
        p_value = np.mean(bootstrap_max_perfs >= best_performance)
        
        result = TestResult(
            test_type=TestType.WHITE_REALITY_CHECK,
            statistic=best_performance,
            p_value=p_value,
            sample_size=n,
            null_hypothesis="Best model does not outperform benchmark (data mining bias)",
            alternative_hypothesis="Best model significantly outperforms benchmark"
        )
        
        result.set_significance(self.config.alpha_level)
        
        logger.info(f"White Reality Check completed: Best={best_performance:.4f}, p={p_value:.4f}")
        return result
    
    def bootstrap_confidence_interval(
        self,
        data: Union[pd.Series, np.ndarray],
        statistic_func: Callable,
        confidence_level: Optional[float] = None
    ) -> TestResult:
        """
        Bootstrap confidence interval for a statistic.
        
        Args:
            data: Data series
            statistic_func: Function to calculate statistic
            confidence_level: Confidence level (default: from config)
            
        Returns:
            Test result with confidence interval
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level
        
        logger.info(f"Computing {confidence_level:.0%} bootstrap confidence interval")
        
        data = np.asarray(data)
        n = len(data)
        
        if n < 10:
            raise ValueError("Insufficient observations for bootstrap")
        
        # Calculate original statistic
        original_stat = statistic_func(data)
        
        # Bootstrap distribution
        bootstrap_stats = []
        
        for _ in range(self.config.bootstrap_iterations):
            if self.config.bootstrap_method == BootstrapMethod.STATIONARY:
                # Stationary bootstrap with geometric block lengths
                bootstrap_sample = self._stationary_bootstrap(data)
            elif self.config.bootstrap_method == BootstrapMethod.CIRCULAR:
                # Circular block bootstrap
                bootstrap_sample = self._circular_block_bootstrap(data)
            elif self.config.bootstrap_method == BootstrapMethod.MOVING_BLOCK:
                # Moving block bootstrap
                bootstrap_sample = self._moving_block_bootstrap(data)
            else:
                # Simple bootstrap
                indices = self.random_state.choice(n, size=n, replace=True)
                bootstrap_sample = data[indices]
            
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        # Calculate bias-corrected and accelerated (BCa) interval if possible
        try:
            bca_ci = self._calculate_bca_interval(
                data, statistic_func, bootstrap_stats, original_stat, confidence_level
            )
            confidence_interval = bca_ci
        except Exception as e:
            logger.warning(f"BCa interval calculation failed, using percentile: {e}")
            confidence_interval = (ci_lower, ci_upper)
        
        result = TestResult(
            test_type=TestType.BOOTSTRAP_CI,
            statistic=original_stat,
            p_value=np.nan,  # Not applicable for CI
            confidence_interval=confidence_interval,
            sample_size=n,
            null_hypothesis="",
            alternative_hypothesis=""
        )
        
        logger.info(f"Bootstrap CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        return result
    
    def _stationary_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """Stationary bootstrap with geometric block lengths."""
        n = len(data)
        
        # Optimal block length (if not specified)
        if self.config.block_length is None:
            block_length = max(1, int(n ** (1/3)))
        else:
            block_length = self.config.block_length
        
        # Generate geometric block lengths
        p = 1.0 / block_length  # Probability of block end
        
        bootstrap_sample = []
        pos = 0
        
        while len(bootstrap_sample) < n:
            # Random starting position
            start = self.random_state.randint(0, n)
            
            # Geometric block length
            block_len = self.random_state.geometric(p)
            
            # Extract block (with wrapping)
            for i in range(block_len):
                if len(bootstrap_sample) >= n:
                    break
                idx = (start + i) % n
                bootstrap_sample.append(data[idx])
        
        return np.array(bootstrap_sample[:n])
    
    def _circular_block_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """Circular block bootstrap."""
        n = len(data)
        
        if self.config.block_length is None:
            block_length = max(1, int(n ** (1/3)))
        else:
            block_length = self.config.block_length
        
        num_blocks = int(np.ceil(n / block_length))
        
        bootstrap_sample = []
        
        for _ in range(num_blocks):
            # Random starting position
            start = self.random_state.randint(0, n)
            
            # Extract block with circular wrap
            for i in range(block_length):
                if len(bootstrap_sample) >= n:
                    break
                idx = (start + i) % n
                bootstrap_sample.append(data[idx])
        
        return np.array(bootstrap_sample[:n])
    
    def _moving_block_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """Moving block bootstrap."""
        n = len(data)
        
        if self.config.block_length is None:
            block_length = max(1, int(n ** (1/3)))
        else:
            block_length = self.config.block_length
        
        num_blocks = int(np.ceil(n / block_length))
        
        bootstrap_sample = []
        
        for _ in range(num_blocks):
            # Random starting position (ensuring full block fits)
            max_start = max(0, n - block_length)
            start = self.random_state.randint(0, max_start + 1)
            
            # Extract block
            for i in range(block_length):
                if len(bootstrap_sample) >= n:
                    break
                idx = start + i
                if idx < n:
                    bootstrap_sample.append(data[idx])
        
        return np.array(bootstrap_sample[:n])
    
    def _calculate_bca_interval(
        self,
        data: np.ndarray,
        statistic_func: Callable,
        bootstrap_stats: np.ndarray,
        original_stat: float,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate bias-corrected and accelerated confidence interval."""
        n = len(data)
        alpha = 1 - confidence_level
        
        # Bias correction
        prop_less = np.mean(bootstrap_stats < original_stat)
        if prop_less == 0:
            z0 = -np.inf
        elif prop_less == 1:
            z0 = np.inf
        else:
            z0 = norm.ppf(prop_less)
        
        # Acceleration parameter using jackknife
        jackknife_stats = []
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stat = statistic_func(jackknife_sample)
            jackknife_stats.append(jackknife_stat)
        
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)
        
        # Acceleration
        numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
        
        if denominator == 0:
            a = 0
        else:
            a = numerator / denominator
        
        # BCa quantiles
        z_alpha_2 = norm.ppf(alpha / 2)
        z_1_alpha_2 = norm.ppf(1 - alpha / 2)
        
        alpha1 = norm.cdf(z0 + (z0 + z_alpha_2) / (1 - a * (z0 + z_alpha_2)))
        alpha2 = norm.cdf(z0 + (z0 + z_1_alpha_2) / (1 - a * (z0 + z_1_alpha_2)))
        
        # Ensure valid percentiles
        alpha1 = max(0, min(1, alpha1))
        alpha2 = max(0, min(1, alpha2))
        
        # Calculate BCa interval
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha1)
        ci_upper = np.percentile(bootstrap_stats, 100 * alpha2)
        
        return (ci_lower, ci_upper)
    
    def multiple_testing_correction(
        self,
        test_results: List[TestResult],
        method: str = "fdr_bh"
    ) -> MultipleTestResult:
        """
        Apply multiple testing correction to a list of test results.
        
        Args:
            test_results: List of individual test results
            method: Correction method ("bonferroni", "fdr_bh", "fdr_by")
            
        Returns:
            Multiple testing result with adjusted p-values
        """
        logger.info(f"Applying {method} multiple testing correction to {len(test_results)} tests")
        
        p_values = [result.p_value for result in test_results]
        n_tests = len(p_values)
        
        if method == "bonferroni":
            adjusted_p_values = [min(1.0, p * n_tests) for p in p_values]
            family_wise_error_rate = self.config.alpha_level
            false_discovery_rate = np.nan
            
        elif method == "fdr_bh":
            # Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values)
            sorted_p_values = np.array(p_values)[sorted_indices]
            
            # Calculate adjusted p-values
            adjusted_sorted = []
            for i in range(n_tests):
                rank = i + 1
                adjusted_p = min(1.0, sorted_p_values[i] * n_tests / rank)
                adjusted_sorted.append(adjusted_p)
            
            # Ensure monotonicity
            for i in range(n_tests - 2, -1, -1):
                adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
            
            # Reorder to original order
            adjusted_p_values = [0] * n_tests
            for i, orig_idx in enumerate(sorted_indices):
                adjusted_p_values[orig_idx] = adjusted_sorted[i]
            
            family_wise_error_rate = np.nan
            false_discovery_rate = self.config.alpha_level
            
        else:
            raise ValueError(f"Unknown multiple testing correction method: {method}")
        
        # Determine which tests are significant
        significant_tests = [adj_p < self.config.alpha_level for adj_p in adjusted_p_values]
        
        return MultipleTestResult(
            individual_results=test_results,
            adjusted_p_values=adjusted_p_values,
            significant_tests=significant_tests,
            method=method,
            family_wise_error_rate=family_wise_error_rate,
            false_discovery_rate=false_discovery_rate
        )


# Utility functions for common statistics
def sharpe_ratio_statistic(returns: np.ndarray, risk_free_rate: float = 0.01) -> float:
    """Calculate Sharpe ratio for bootstrap testing."""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def information_ratio_statistic(
    returns: np.ndarray, 
    benchmark_returns: np.ndarray
) -> float:
    """Calculate Information ratio for bootstrap testing."""
    if len(returns) != len(benchmark_returns) or len(returns) == 0:
        return 0.0
    
    excess_returns = returns - benchmark_returns
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def create_default_statistical_config(**overrides) -> StatisticalTestConfig:
    """Create default statistical test configuration with Taiwan market settings."""
    config = StatisticalTestConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    return config


# Example usage and testing
if __name__ == "__main__":
    print("Statistical Testing Framework demo")
    
    # Create sample data
    np.random.seed(42)
    n_periods = 252
    
    # Generate sample returns
    model1_returns = np.random.normal(0.001, 0.02, n_periods)
    model2_returns = np.random.normal(0.0008, 0.018, n_periods)
    benchmark_returns = np.random.normal(0.0005, 0.015, n_periods)
    
    # Create configuration
    config = create_default_statistical_config(
        bootstrap_iterations=1000,
        hansen_spa_bootstrap_iterations=1000
    )
    
    # Initialize test engine
    test_engine = StatisticalTestEngine(config)
    
    # Run Diebold-Mariano test
    dm_result = test_engine.diebold_mariano_test(
        model1_returns - benchmark_returns,
        model2_returns - benchmark_returns
    )
    
    print(f"Diebold-Mariano Test:")
    print(f"  Statistic: {dm_result.statistic:.4f}")
    print(f"  P-value: {dm_result.p_value:.4f}")
    print(f"  Significant: {dm_result.is_significant}")
    
    # Run Hansen SPA test
    spa_result = test_engine.hansen_spa_test(
        benchmark_returns,
        [model1_returns, model2_returns]
    )
    
    print(f"\nHansen SPA Test:")
    print(f"  Statistic: {spa_result.statistic:.4f}")
    print(f"  P-value: {spa_result.p_value:.4f}")
    print(f"  Significant: {spa_result.is_significant}")
    
    # Bootstrap confidence interval for Sharpe ratio
    def sharpe_func(data):
        return sharpe_ratio_statistic(data, 0.01)
    
    ci_result = test_engine.bootstrap_confidence_interval(
        model1_returns, sharpe_func, 0.95
    )
    
    print(f"\nBootstrap Confidence Interval (Sharpe Ratio):")
    print(f"  Statistic: {ci_result.statistic:.4f}")
    print(f"  95% CI: [{ci_result.confidence_interval[0]:.4f}, {ci_result.confidence_interval[1]:.4f}]")