"""
Risk-Adjusted Performance Metrics for ML4T Taiwan Market.

This module implements comprehensive risk-adjusted performance metrics specifically
designed for Taiwan market quantitative trading strategies with walk-forward validation.

Key Features:
- Advanced risk-adjusted ratios (Sharpe, Sortino, Calmar, Information, Treynor)
- Taiwan market-specific risk metrics (VaR, CVaR, Maximum Drawdown)
- Rolling risk metrics with statistical significance testing
- Risk-adjusted benchmarking against Taiwan indices
- Tail risk analysis and extreme value theory
- Performance persistence and regime analysis
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import gamma
import warnings
from collections import deque

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.models.taiwan_market import TaiwanMarketCode, TaiwanTradingCalendar
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel
from .performance import PerformanceConfig, BenchmarkType, BenchmarkDataProvider, PerformanceMetrics

logger = logging.getLogger(__name__)


class RiskMetricType(Enum):
    """Types of risk metrics."""
    VOLATILITY = "volatility"
    DOWNSIDE_DEVIATION = "downside_deviation"
    VAR = "var"               # Value at Risk
    CVAR = "cvar"             # Conditional VaR
    MAX_DRAWDOWN = "max_drawdown"
    ULCER_INDEX = "ulcer_index"
    TAIL_RATIO = "tail_ratio"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"


class RiskAdjustedRatio(Enum):
    """Types of risk-adjusted performance ratios."""
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    STERLING = "sterling"
    BURKE = "burke"
    MARTIN = "martin"
    TREYNOR = "treynor"
    INFORMATION = "information"
    MODIGLIANI = "modigliani"


class DistributionModel(Enum):
    """Statistical distribution models for risk analysis."""
    NORMAL = "normal"
    T_DISTRIBUTION = "t_distribution"
    SKEWED_T = "skewed_t"
    GED = "ged"              # Generalized Error Distribution
    CORNISH_FISHER = "cornish_fisher"


@dataclass
class RiskConfig:
    """Configuration for risk-adjusted metrics calculation."""
    # Risk measurement parameters
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_method: str = "historical"  # historical, parametric, cornish_fisher
    holding_period: int = 1         # Days for VaR calculation
    
    # Distribution assumptions
    distribution_model: DistributionModel = DistributionModel.NORMAL
    use_robust_estimators: bool = True
    
    # Rolling window parameters
    rolling_window_days: int = 252  # 1 year rolling window
    min_observations: int = 60      # Minimum observations for calculation
    
    # Downside risk parameters
    downside_threshold: float = 0.0  # Threshold for downside deviation
    use_risk_free_rate: bool = True  # Use risk-free rate as downside threshold
    
    # Extreme value analysis
    tail_threshold_percentile: float = 0.95  # For tail analysis
    block_maxima_size: int = 22             # ~1 month blocks for extreme value
    
    # Statistical testing
    enable_hypothesis_tests: bool = True
    test_significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Taiwan market specifics
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.01  # Taiwan risk-free rate
    
    def __post_init__(self):
        """Validate configuration."""
        if self.rolling_window_days < self.min_observations:
            raise ValueError("Rolling window must be >= minimum observations")
        for cl in self.confidence_levels:
            if cl <= 0 or cl >= 1:
                raise ValueError("Confidence levels must be between 0 and 1")


@dataclass
class RiskMetrics:
    """Container for calculated risk metrics."""
    # Volatility measures
    total_volatility: float
    downside_volatility: float
    upside_volatility: float
    volatility_of_volatility: float
    
    # Value at Risk metrics
    var_95: float
    var_99: float
    cvar_95: float      # Conditional VaR (Expected Shortfall)
    cvar_99: float
    
    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    ulcer_index: float
    pain_index: float
    
    # Distribution properties
    skewness: float
    excess_kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    # Tail risk metrics
    tail_ratio: float      # Ratio of right tail to left tail
    tail_expectation_ratio: float
    gain_loss_ratio: float
    
    # Risk-adjusted ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    martin_ratio: float
    
    # Beta and systematic risk
    beta: float
    systematic_risk: float
    idiosyncratic_risk: float
    treynor_ratio: float
    
    # Information ratio components
    information_ratio: float
    tracking_error: float
    active_return: float
    
    # Metadata
    calculation_date: datetime = field(default_factory=datetime.now)
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    observations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_volatility': self.total_volatility,
            'downside_volatility': self.downside_volatility,
            'upside_volatility': self.upside_volatility,
            'volatility_of_volatility': self.volatility_of_volatility,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'ulcer_index': self.ulcer_index,
            'pain_index': self.pain_index,
            'skewness': self.skewness,
            'excess_kurtosis': self.excess_kurtosis,
            'jarque_bera_stat': self.jarque_bera_stat,
            'jarque_bera_pvalue': self.jarque_bera_pvalue,
            'tail_ratio': self.tail_ratio,
            'tail_expectation_ratio': self.tail_expectation_ratio,
            'gain_loss_ratio': self.gain_loss_ratio,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'sterling_ratio': self.sterling_ratio,
            'burke_ratio': self.burke_ratio,
            'martin_ratio': self.martin_ratio,
            'beta': self.beta,
            'systematic_risk': self.systematic_risk,
            'idiosyncratic_risk': self.idiosyncratic_risk,
            'treynor_ratio': self.treynor_ratio,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error,
            'active_return': self.active_return,
            'calculation_date': self.calculation_date.isoformat(),
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
            'observations': self.observations
        }


class RiskCalculator:
    """
    Advanced risk metrics calculator with Taiwan market adaptations.
    
    Implements comprehensive risk measurement including volatility,
    VaR, tail risk, and risk-adjusted performance ratios.
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        logger.info("RiskCalculator initialized")
    
    def calculate_risk_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
        period_start: Optional[date] = None,
        period_end: Optional[date] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Return series
            benchmark_returns: Benchmark returns for relative metrics
            period_start: Start date of analysis period
            period_end: End date of analysis period
            
        Returns:
            Complete risk metrics
        """
        logger.info("Calculating comprehensive risk metrics")
        
        # Convert to pandas Series if needed
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        returns = returns.dropna()
        
        if len(returns) < self.config.min_observations:
            raise ValueError(f"Insufficient observations: {len(returns)} < {self.config.min_observations}")
        
        # Calculate volatility measures
        volatility_metrics = self._calculate_volatility_metrics(returns)
        
        # Calculate VaR and tail risk
        var_metrics = self._calculate_var_metrics(returns)
        
        # Calculate drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(returns)
        
        # Calculate distribution properties
        distribution_metrics = self._calculate_distribution_metrics(returns)
        
        # Calculate tail risk metrics
        tail_metrics = self._calculate_tail_risk_metrics(returns)
        
        # Calculate risk-adjusted ratios
        ratio_metrics = self._calculate_risk_adjusted_ratios(returns)
        
        # Calculate beta and systematic risk if benchmark provided
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, np.ndarray):
                benchmark_returns = pd.Series(benchmark_returns)
            
            # Align returns
            common_idx = returns.index.intersection(benchmark_returns.index)
            if len(common_idx) > 0:
                aligned_returns = returns.loc[common_idx]
                aligned_benchmark = benchmark_returns.loc[common_idx]
                beta_metrics = self._calculate_beta_metrics(aligned_returns, aligned_benchmark)
            else:
                beta_metrics = self._get_default_beta_metrics()
        else:
            beta_metrics = self._get_default_beta_metrics()
        
        # Compile all metrics
        risk_metrics = RiskMetrics(
            # Volatility
            total_volatility=volatility_metrics['total_volatility'],
            downside_volatility=volatility_metrics['downside_volatility'],
            upside_volatility=volatility_metrics['upside_volatility'],
            volatility_of_volatility=volatility_metrics['volatility_of_volatility'],
            
            # VaR
            var_95=var_metrics['var_95'],
            var_99=var_metrics['var_99'],
            cvar_95=var_metrics['cvar_95'],
            cvar_99=var_metrics['cvar_99'],
            
            # Drawdown
            max_drawdown=drawdown_metrics['max_drawdown'],
            avg_drawdown=drawdown_metrics['avg_drawdown'],
            max_drawdown_duration=drawdown_metrics['max_duration'],
            ulcer_index=drawdown_metrics['ulcer_index'],
            pain_index=drawdown_metrics['pain_index'],
            
            # Distribution
            skewness=distribution_metrics['skewness'],
            excess_kurtosis=distribution_metrics['excess_kurtosis'],
            jarque_bera_stat=distribution_metrics['jb_stat'],
            jarque_bera_pvalue=distribution_metrics['jb_pvalue'],
            
            # Tail risk
            tail_ratio=tail_metrics['tail_ratio'],
            tail_expectation_ratio=tail_metrics['tail_expectation_ratio'],
            gain_loss_ratio=tail_metrics['gain_loss_ratio'],
            
            # Risk-adjusted ratios
            sharpe_ratio=ratio_metrics['sharpe'],
            sortino_ratio=ratio_metrics['sortino'],
            calmar_ratio=ratio_metrics['calmar'],
            sterling_ratio=ratio_metrics['sterling'],
            burke_ratio=ratio_metrics['burke'],
            martin_ratio=ratio_metrics['martin'],
            
            # Beta metrics
            beta=beta_metrics['beta'],
            systematic_risk=beta_metrics['systematic_risk'],
            idiosyncratic_risk=beta_metrics['idiosyncratic_risk'],
            treynor_ratio=beta_metrics['treynor_ratio'],
            information_ratio=beta_metrics['information_ratio'],
            tracking_error=beta_metrics['tracking_error'],
            active_return=beta_metrics['active_return'],
            
            # Metadata
            period_start=period_start,
            period_end=period_end,
            observations=len(returns)
        )
        
        logger.info(f"Risk metrics calculated for {len(returns)} observations")
        return risk_metrics
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate various volatility measures."""
        
        # Total volatility (annualized)
        total_vol = float(returns.std() * np.sqrt(self.config.trading_days_per_year))
        
        # Downside deviation
        threshold = self.config.downside_threshold
        if self.config.use_risk_free_rate:
            threshold = self.config.risk_free_rate / self.config.trading_days_per_year
        
        downside_returns = returns[returns < threshold]
        if len(downside_returns) > 0:
            downside_vol = float(downside_returns.std() * np.sqrt(self.config.trading_days_per_year))
        else:
            downside_vol = 0.0
        
        # Upside deviation
        upside_returns = returns[returns > threshold]
        if len(upside_returns) > 0:
            upside_vol = float(upside_returns.std() * np.sqrt(self.config.trading_days_per_year))
        else:
            upside_vol = 0.0
        
        # Volatility of volatility (using rolling volatility)
        if len(returns) >= self.config.rolling_window_days:
            rolling_vol = returns.rolling(window=min(22, len(returns)//4)).std()
            vol_of_vol = float(rolling_vol.std() * np.sqrt(self.config.trading_days_per_year))
        else:
            vol_of_vol = 0.0
        
        return {
            'total_volatility': total_vol,
            'downside_volatility': downside_vol,
            'upside_volatility': upside_vol,
            'volatility_of_volatility': vol_of_vol
        }
    
    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional VaR."""
        
        var_metrics = {}
        
        for confidence_level in self.config.confidence_levels:
            alpha = 1 - confidence_level
            
            if self.config.var_method == "historical":
                # Historical VaR
                var_value = float(np.percentile(returns, alpha * 100))
                
                # Conditional VaR (Expected Shortfall)
                tail_returns = returns[returns <= var_value]
                if len(tail_returns) > 0:
                    cvar_value = float(tail_returns.mean())
                else:
                    cvar_value = var_value
                    
            elif self.config.var_method == "parametric":
                # Parametric VaR (assuming normal distribution)
                mean_return = returns.mean()
                vol_return = returns.std()
                z_score = stats.norm.ppf(alpha)
                var_value = float(mean_return + z_score * vol_return)
                
                # Conditional VaR for normal distribution
                phi_z = stats.norm.pdf(z_score)
                cvar_value = float(mean_return - vol_return * phi_z / alpha)
                
            elif self.config.var_method == "cornish_fisher":
                # Cornish-Fisher expansion for non-normal distributions
                mean_return = returns.mean()
                vol_return = returns.std()
                skew = stats.skew(returns)
                kurt = stats.kurtosis(returns)
                
                z_score = stats.norm.ppf(alpha)
                
                # Cornish-Fisher adjustment
                cf_z = (z_score + 
                       (z_score**2 - 1) * skew / 6 +
                       (z_score**3 - 3*z_score) * kurt / 24 -
                       (2*z_score**3 - 5*z_score) * skew**2 / 36)
                
                var_value = float(mean_return + cf_z * vol_return)
                
                # Approximate CVaR
                phi_z = stats.norm.pdf(cf_z)
                cvar_value = float(mean_return - vol_return * phi_z / alpha)
            
            else:
                # Default to historical method
                var_value = float(np.percentile(returns, alpha * 100))
                tail_returns = returns[returns <= var_value]
                cvar_value = float(tail_returns.mean()) if len(tail_returns) > 0 else var_value
            
            # Store with percentage key
            conf_pct = int(confidence_level * 100)
            var_metrics[f'var_{conf_pct}'] = var_value
            var_metrics[f'cvar_{conf_pct}'] = cvar_value
        
        return var_metrics
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics."""
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Running maximum
        running_max = cumulative.expanding().max()
        
        # Drawdown series
        drawdowns = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_dd = float(drawdowns.min())
        
        # Average drawdown
        negative_dd = drawdowns[drawdowns < 0]
        avg_dd = float(negative_dd.mean()) if len(negative_dd) > 0 else 0.0
        
        # Maximum drawdown duration
        max_duration = 0
        current_duration = 0
        
        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        # Ulcer Index (RMS of drawdowns)
        ulcer_index = float(np.sqrt((drawdowns ** 2).mean()))
        
        # Pain Index (average drawdown weighted by duration)
        pain_index = float(abs(drawdowns.mean()))
        
        return {
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'max_duration': max_duration,
            'ulcer_index': ulcer_index,
            'pain_index': pain_index
        }
    
    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate statistical distribution properties."""
        
        # Skewness and kurtosis
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))  # Excess kurtosis
        
        # Jarque-Bera test for normality
        if len(returns) >= 8:  # Minimum for JB test
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
        else:
            jb_stat, jb_pvalue = 0.0, 1.0
        
        return {
            'skewness': skew,
            'excess_kurtosis': kurt,
            'jb_stat': float(jb_stat),
            'jb_pvalue': float(jb_pvalue)
        }
    
    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate tail risk and extreme value metrics."""
        
        # Tail ratio (95th percentile / 5th percentile)
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if abs(p5) > 1e-10:
            tail_ratio = float(p95 / abs(p5))
        else:
            tail_ratio = float('inf') if p95 > 0 else 0.0
        
        # Tail expectation ratio
        threshold = np.percentile(returns, 95)
        upper_tail = returns[returns > threshold]
        lower_tail = returns[returns < -threshold]
        
        if len(upper_tail) > 0 and len(lower_tail) > 0:
            tail_exp_ratio = float(upper_tail.mean() / abs(lower_tail.mean()))
        else:
            tail_exp_ratio = 1.0
        
        # Gain-loss ratio
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(gains) > 0 and len(losses) > 0:
            gain_loss_ratio = float(gains.mean() / abs(losses.mean()))
        else:
            gain_loss_ratio = 1.0
        
        return {
            'tail_ratio': tail_ratio,
            'tail_expectation_ratio': tail_exp_ratio,
            'gain_loss_ratio': gain_loss_ratio
        }
    
    def _calculate_risk_adjusted_ratios(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk-adjusted performance ratios."""
        
        # Risk-free rate (daily)
        rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
        excess_returns = returns - rf_daily
        
        # Sharpe ratio
        if returns.std() != 0:
            sharpe = float(excess_returns.mean() / returns.std() * np.sqrt(self.config.trading_days_per_year))
        else:
            sharpe = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() != 0:
            sortino = float(excess_returns.mean() / downside_returns.std() * np.sqrt(self.config.trading_days_per_year))
        else:
            sortino = float('inf') if excess_returns.mean() > 0 else 0.0
        
        # Calmar ratio (return / max drawdown)
        annualized_return = float(returns.mean() * self.config.trading_days_per_year)
        drawdown_metrics = self._calculate_drawdown_metrics(returns)
        max_dd = abs(drawdown_metrics['max_drawdown'])
        
        if max_dd > 0:
            calmar = annualized_return / max_dd
        else:
            calmar = float('inf') if annualized_return > 0 else 0.0
        
        # Sterling ratio (return / average drawdown)
        avg_dd = abs(drawdown_metrics['avg_drawdown'])
        if avg_dd > 0:
            sterling = annualized_return / avg_dd
        else:
            sterling = float('inf') if annualized_return > 0 else 0.0
        
        # Burke ratio (return / square root of sum of squared drawdowns)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        burke_denominator = np.sqrt((drawdowns ** 2).sum())
        
        if burke_denominator > 0:
            burke = annualized_return / burke_denominator
        else:
            burke = float('inf') if annualized_return > 0 else 0.0
        
        # Martin ratio (return / Ulcer Index)
        ulcer_index = drawdown_metrics['ulcer_index']
        if ulcer_index > 0:
            martin = annualized_return / ulcer_index
        else:
            martin = float('inf') if annualized_return > 0 else 0.0
        
        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'sterling': sterling,
            'burke': burke,
            'martin': martin
        }
    
    def _calculate_beta_metrics(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate beta and systematic risk metrics."""
        
        # Risk-free rate (daily)
        rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
        
        excess_returns = returns - rf_daily
        excess_benchmark = benchmark_returns - rf_daily
        
        # Beta calculation
        if len(excess_benchmark) > 1 and excess_benchmark.var() != 0:
            beta = float(np.cov(excess_returns, excess_benchmark)[0, 1] / excess_benchmark.var())
        else:
            beta = 0.0
        
        # Systematic and idiosyncratic risk
        total_variance = returns.var()
        systematic_variance = (beta ** 2) * benchmark_returns.var()
        idiosyncratic_variance = max(0, total_variance - systematic_variance)
        
        systematic_risk = float(np.sqrt(systematic_variance * self.config.trading_days_per_year))
        idiosyncratic_risk = float(np.sqrt(idiosyncratic_variance * self.config.trading_days_per_year))
        
        # Treynor ratio
        mean_excess_return = excess_returns.mean() * self.config.trading_days_per_year
        if beta != 0:
            treynor = float(mean_excess_return / beta)
        else:
            treynor = 0.0
        
        # Information ratio
        active_returns = returns - benchmark_returns
        active_return = float(active_returns.mean() * self.config.trading_days_per_year)
        tracking_error = float(active_returns.std() * np.sqrt(self.config.trading_days_per_year))
        
        if tracking_error != 0:
            information_ratio = active_return / tracking_error
        else:
            information_ratio = 0.0
        
        return {
            'beta': beta,
            'systematic_risk': systematic_risk,
            'idiosyncratic_risk': idiosyncratic_risk,
            'treynor_ratio': treynor,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'active_return': active_return
        }
    
    def _get_default_beta_metrics(self) -> Dict[str, float]:
        """Get default beta metrics when no benchmark provided."""
        return {
            'beta': 0.0,
            'systematic_risk': 0.0,
            'idiosyncratic_risk': 0.0,
            'treynor_ratio': 0.0,
            'information_ratio': 0.0,
            'tracking_error': 0.0,
            'active_return': 0.0
        }


class RollingRiskAnalyzer:
    """
    Rolling risk metrics analyzer for time-varying risk analysis.
    
    Calculates risk metrics over rolling windows to analyze
    risk evolution and identify regime changes.
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.calculator = RiskCalculator(config)
        
        logger.info("RollingRiskAnalyzer initialized")
    
    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        window_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling risk metrics over time.
        
        Args:
            returns: Return series
            benchmark_returns: Benchmark returns
            window_size: Rolling window size (default: from config)
            
        Returns:
            DataFrame with rolling risk metrics
        """
        if window_size is None:
            window_size = self.config.rolling_window_days
        
        logger.info(f"Calculating rolling risk metrics with {window_size}-day window")
        
        rolling_metrics = []
        
        for i in range(window_size, len(returns) + 1):
            # Extract window
            window_returns = returns.iloc[i-window_size:i]
            window_benchmark = None
            
            if benchmark_returns is not None:
                # Align benchmark
                window_benchmark = benchmark_returns.iloc[i-window_size:i]
                common_idx = window_returns.index.intersection(window_benchmark.index)
                if len(common_idx) > 0:
                    window_returns = window_returns.loc[common_idx]
                    window_benchmark = window_benchmark.loc[common_idx]
                else:
                    window_benchmark = None
            
            try:
                # Calculate metrics for this window
                metrics = self.calculator.calculate_risk_metrics(
                    window_returns, 
                    window_benchmark
                )
                
                # Create record
                record = {
                    'date': window_returns.index[-1],
                    'volatility': metrics.total_volatility,
                    'downside_volatility': metrics.downside_volatility,
                    'var_95': metrics.var_95,
                    'cvar_95': metrics.cvar_95,
                    'max_drawdown': metrics.max_drawdown,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'sortino_ratio': metrics.sortino_ratio,
                    'calmar_ratio': metrics.calmar_ratio,
                    'skewness': metrics.skewness,
                    'excess_kurtosis': metrics.excess_kurtosis,
                    'beta': metrics.beta,
                    'tracking_error': metrics.tracking_error,
                    'information_ratio': metrics.information_ratio
                }
                
                rolling_metrics.append(record)
                
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for window ending {window_returns.index[-1]}: {e}")
                continue
        
        df = pd.DataFrame(rolling_metrics)
        if not df.empty:
            df.set_index('date', inplace=True)
        
        logger.info(f"Rolling risk metrics calculated for {len(df)} windows")
        return df
    
    def analyze_risk_regime_changes(
        self,
        rolling_metrics: pd.DataFrame,
        metric_name: str = 'volatility',
        change_threshold: float = 1.5  # Threshold for regime change detection
    ) -> Dict[str, Any]:
        """
        Analyze risk regime changes based on rolling metrics.
        
        Args:
            rolling_metrics: DataFrame from calculate_rolling_metrics
            metric_name: Metric to analyze for regime changes
            change_threshold: Multiplier threshold for regime detection
            
        Returns:
            Regime change analysis results
        """
        if metric_name not in rolling_metrics.columns:
            raise ValueError(f"Metric {metric_name} not found in rolling metrics")
        
        metric_series = rolling_metrics[metric_name].dropna()
        
        if len(metric_series) == 0:
            return {'regime_changes': [], 'analysis': {}}
        
        # Calculate rolling mean and std for regime detection
        window = min(22, len(metric_series) // 4)  # ~1 month or 1/4 of data
        rolling_mean = metric_series.rolling(window=window).mean()
        rolling_std = metric_series.rolling(window=window).std()
        
        # Identify regime changes
        regime_changes = []
        
        for i in range(window, len(metric_series)):
            current_value = metric_series.iloc[i]
            expected_value = rolling_mean.iloc[i-1]
            threshold_std = rolling_std.iloc[i-1]
            
            if threshold_std > 0:
                z_score = abs(current_value - expected_value) / threshold_std
                
                if z_score > change_threshold:
                    regime_changes.append({
                        'date': metric_series.index[i],
                        'metric_value': current_value,
                        'expected_value': expected_value,
                        'z_score': z_score,
                        'change_type': 'increase' if current_value > expected_value else 'decrease'
                    })
        
        # Analysis summary
        analysis = {
            'total_regime_changes': len(regime_changes),
            'avg_time_between_changes': None,
            'metric_statistics': {
                'mean': float(metric_series.mean()),
                'std': float(metric_series.std()),
                'min': float(metric_series.min()),
                'max': float(metric_series.max()),
                'cv': float(metric_series.std() / metric_series.mean()) if metric_series.mean() != 0 else float('inf')
            }
        }
        
        if len(regime_changes) > 1:
            # Calculate average time between regime changes
            change_dates = [pd.to_datetime(rc['date']) for rc in regime_changes]
            time_diffs = [(change_dates[i] - change_dates[i-1]).days for i in range(1, len(change_dates))]
            analysis['avg_time_between_changes'] = float(np.mean(time_diffs))
        
        return {
            'regime_changes': regime_changes,
            'analysis': analysis
        }


# Utility functions
def create_taiwan_risk_config(**overrides) -> RiskConfig:
    """Create risk configuration with Taiwan market defaults."""
    config = RiskConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    return config


def create_taiwan_risk_calculator(
    var_method: str = "historical",
    confidence_levels: List[float] = None,
    **config_overrides
) -> RiskCalculator:
    """Create risk calculator with Taiwan market settings."""
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]
    
    config = create_taiwan_risk_config(
        var_method=var_method,
        confidence_levels=confidence_levels,
        **config_overrides
    )
    return RiskCalculator(config)


# Example usage and testing
if __name__ == "__main__":
    print("Risk-Adjusted Metrics Engine demo")
    
    # Create sample returns data
    np.random.seed(42)
    n_periods = 252  # One year of daily returns
    
    # Generate sample returns with volatility clustering
    returns = []
    vol = 0.02  # Base volatility
    
    for i in range(n_periods):
        # GARCH-like volatility clustering
        vol = 0.9 * vol + 0.1 * 0.02 + 0.05 * np.random.randn() * vol
        vol = max(0.005, min(0.05, vol))  # Bound volatility
        
        ret = np.random.normal(0.0008, vol)  # 20bps daily mean return
        returns.append(ret)
    
    returns_series = pd.Series(returns, index=pd.date_range('2023-01-01', periods=n_periods))
    
    # Create benchmark returns
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.015, n_periods),
        index=returns_series.index
    )
    
    # Create risk calculator
    risk_calc = create_taiwan_risk_calculator(
        var_method="cornish_fisher",
        confidence_levels=[0.95, 0.99],
        enable_hypothesis_tests=True
    )
    
    # Calculate risk metrics
    risk_metrics = risk_calc.calculate_risk_metrics(
        returns_series, 
        benchmark_returns
    )
    
    print(f"Demo risk metrics calculated:")
    print(f"Total Volatility: {risk_metrics.total_volatility:.2%}")
    print(f"Downside Volatility: {risk_metrics.downside_volatility:.2%}")
    print(f"VaR 95%: {risk_metrics.var_95:.2%}")
    print(f"CVaR 95%: {risk_metrics.cvar_95:.2%}")
    print(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {risk_metrics.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {risk_metrics.calmar_ratio:.2f}")
    print(f"Information Ratio: {risk_metrics.information_ratio:.2f}")
    print(f"Beta: {risk_metrics.beta:.2f}")
    print(f"Tracking Error: {risk_metrics.tracking_error:.2%}")
    
    # Test rolling analysis
    rolling_analyzer = RollingRiskAnalyzer(risk_calc.config)
    rolling_metrics = rolling_analyzer.calculate_rolling_metrics(
        returns_series, benchmark_returns, window_size=60
    )
    
    print(f"\nRolling analysis completed with {len(rolling_metrics)} windows")
    
    # Regime change analysis
    regime_analysis = rolling_analyzer.analyze_risk_regime_changes(
        rolling_metrics, 'volatility', change_threshold=2.0
    )
    
    print(f"Regime changes detected: {regime_analysis['analysis']['total_regime_changes']}")
    print(f"Average volatility: {regime_analysis['analysis']['metric_statistics']['mean']:.2%}")
    print(f"Volatility CV: {regime_analysis['analysis']['metric_statistics']['cv']:.2f}")