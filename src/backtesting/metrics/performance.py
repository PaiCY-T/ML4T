"""
Performance Metrics Engine for ML4T Walk-Forward Validation.

This module implements comprehensive performance metrics calculation for
portfolio and strategy validation, designed specifically for Taiwan market
quantitative trading strategies with walk-forward validation integration.

Key Features:
- Real-time performance metric calculation
- Taiwan market benchmark integration (TAIEX, TPEx)
- Risk-adjusted performance metrics (Sharpe, Information Ratio, Sortino)
- Maximum drawdown analysis with statistical significance
- Performance attribution support
- Walk-forward validation integration
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
import warnings
from decimal import Decimal

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.models.taiwan_market import TaiwanMarketCode, TaiwanTradingCalendar
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel
from ..validation.walk_forward import ValidationWindow, ValidationResult

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    RETURN = "return"
    RISK = "risk"  
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN = "drawdown"
    ATTRIBUTION = "attribution"
    BENCHMARK = "benchmark"


class BenchmarkType(Enum):
    """Taiwan market benchmark types."""
    TAIEX = "TAIEX"           # Taiwan Capitalization Weighted Index
    TPEx = "TPEx"             # Taipei Exchange Index
    MSCI_TAIWAN = "MSCI_TW"   # MSCI Taiwan Index
    FTSE_TAIWAN = "FTSE_TW"   # FTSE Taiwan Index
    CUSTOM = "CUSTOM"         # User-defined benchmark


class FrequencyType(Enum):
    """Frequency types for metric calculation."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass
class PerformanceConfig:
    """Configuration for performance metrics calculation."""
    # Risk-free rate and benchmarks
    risk_free_rate: float = 0.01  # 1% annual Taiwan risk-free rate
    benchmark_type: BenchmarkType = BenchmarkType.TAIEX
    custom_benchmark_symbol: Optional[str] = None
    
    # Calculation parameters
    trading_days_per_year: int = 252  # Taiwan market trading days
    confidence_level: float = 0.95
    
    # Performance targets
    target_sharpe_ratio: float = 2.0
    target_information_ratio: float = 0.8
    max_drawdown_threshold: float = 0.15  # 15% max drawdown
    
    # Statistical testing
    enable_statistical_tests: bool = True
    bootstrap_iterations: int = 1000
    
    # Calculation frequency
    calculation_frequency: FrequencyType = FrequencyType.DAILY
    
    def __post_init__(self):
        """Validate configuration."""
        if self.target_sharpe_ratio <= 0:
            raise ValueError("Target Sharpe ratio must be positive")
        if self.target_information_ratio <= 0:
            raise ValueError("Target Information ratio must be positive")
        if self.max_drawdown_threshold <= 0 or self.max_drawdown_threshold >= 1:
            raise ValueError("Max drawdown threshold must be between 0 and 1")


@dataclass
class PerformanceMetrics:
    """Container for calculated performance metrics."""
    # Basic returns
    total_return: float
    annualized_return: float
    volatility: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int  # days
    current_drawdown: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk at 95%
    cvar_95: float  # Conditional VaR at 95%
    
    # Benchmark comparison
    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    
    # Statistical significance
    sharpe_pvalue: Optional[float] = None
    ir_pvalue: Optional[float] = None
    alpha_pvalue: Optional[float] = None
    
    # Metadata
    calculation_date: datetime = field(default_factory=datetime.now)
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    observations: int = 0
    
    def meets_targets(self, config: PerformanceConfig) -> Dict[str, bool]:
        """Check if performance meets target thresholds."""
        return {
            'sharpe_target': self.sharpe_ratio >= config.target_sharpe_ratio,
            'ir_target': self.information_ratio >= config.target_information_ratio,
            'drawdown_target': abs(self.max_drawdown) <= config.max_drawdown_threshold,
            'positive_alpha': self.alpha > 0,
            'positive_returns': self.total_return > 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'current_drawdown': self.current_drawdown,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'alpha': self.alpha,
            'beta': self.beta,
            'correlation': self.correlation,
            'tracking_error': self.tracking_error,
            'sharpe_pvalue': self.sharpe_pvalue,
            'ir_pvalue': self.ir_pvalue,
            'alpha_pvalue': self.alpha_pvalue,
            'calculation_date': self.calculation_date.isoformat(),
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
            'observations': self.observations
        }


class BenchmarkDataProvider:
    """Provider for Taiwan market benchmark data."""
    
    def __init__(self, temporal_store: TemporalStore, pit_engine: PointInTimeEngine):
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        
        # Taiwan benchmark symbols mapping
        self.benchmark_symbols = {
            BenchmarkType.TAIEX: "IX0001.TW",     # TAIEX
            BenchmarkType.TPEx: "IX0043.TWO",     # TPEx Index
            BenchmarkType.MSCI_TAIWAN: "MSCI_TW", # MSCI Taiwan
            BenchmarkType.FTSE_TAIWAN: "FTSE_TW"  # FTSE Taiwan
        }
        
        # Cache for benchmark data
        self._benchmark_cache: Dict[str, Dict[date, float]] = {}
        
        logger.info("BenchmarkDataProvider initialized")
    
    def get_benchmark_returns(
        self,
        benchmark_type: BenchmarkType,
        start_date: date,
        end_date: date,
        custom_symbol: Optional[str] = None
    ) -> pd.Series:
        """
        Get benchmark returns for the specified period.
        
        Args:
            benchmark_type: Type of benchmark
            start_date: Start date for returns
            end_date: End date for returns
            custom_symbol: Custom benchmark symbol if CUSTOM type
            
        Returns:
            Series of benchmark returns indexed by date
        """
        # Determine symbol
        if benchmark_type == BenchmarkType.CUSTOM:
            if not custom_symbol:
                raise ValueError("Custom symbol required for CUSTOM benchmark type")
            symbol = custom_symbol
        else:
            symbol = self.benchmark_symbols.get(benchmark_type)
            if not symbol:
                raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self._benchmark_cache:
            data = self._benchmark_cache[cache_key]
            return pd.Series(data, name=f"{benchmark_type.value}_returns")
        
        try:
            # Query benchmark price data
            query = PITQuery(
                symbols=[symbol],
                as_of_date=end_date,
                data_types=[DataType.PRICE],
                start_date=start_date,
                end_date=end_date
            )
            
            price_data = self.pit_engine.query(query)
            
            if symbol not in price_data or len(price_data[symbol]) == 0:
                logger.warning(f"No benchmark data available for {symbol}")
                # Return zero returns series
                date_range = pd.date_range(start_date, end_date, freq='D')
                return pd.Series(0.0, index=date_range, name=f"{benchmark_type.value}_returns")
            
            # Convert to DataFrame and calculate returns
            records = []
            for temporal_value in price_data[symbol]:
                records.append({
                    'date': temporal_value.value_date,
                    'price': float(temporal_value.value)
                })
            
            df = pd.DataFrame(records).set_index('date').sort_index()
            returns = df['price'].pct_change().dropna()
            
            # Cache the results
            self._benchmark_cache[cache_key] = returns.to_dict()
            
            logger.debug(f"Retrieved {len(returns)} benchmark returns for {symbol}")
            return returns.rename(f"{benchmark_type.value}_returns")
            
        except Exception as e:
            logger.error(f"Failed to retrieve benchmark data for {symbol}: {e}")
            # Return zero returns series as fallback
            date_range = pd.date_range(start_date, end_date, freq='D')
            return pd.Series(0.0, index=date_range, name=f"{benchmark_type.value}_returns")
    
    def get_risk_free_rate(
        self,
        start_date: date,
        end_date: date,
        annual_rate: float = 0.01
    ) -> pd.Series:
        """
        Get risk-free rate series for the period.
        
        Args:
            start_date: Start date
            end_date: End date 
            annual_rate: Annual risk-free rate (default 1% for Taiwan)
            
        Returns:
            Series of daily risk-free rates
        """
        # Convert annual rate to daily
        daily_rate = annual_rate / 252  # 252 trading days per year
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        return pd.Series(daily_rate, index=date_range, name='risk_free_rate')


class PerformanceCalculator:
    """
    Core performance metrics calculation engine.
    
    Implements comprehensive performance analytics including:
    - Risk-adjusted performance metrics
    - Drawdown analysis
    - Statistical significance testing
    - Benchmark comparison
    """
    
    def __init__(
        self,
        config: PerformanceConfig,
        benchmark_provider: BenchmarkDataProvider
    ):
        self.config = config
        self.benchmark_provider = benchmark_provider
        
        logger.info("PerformanceCalculator initialized")
    
    def calculate_metrics(
        self,
        returns: Union[pd.Series, np.ndarray, List[float]],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        period_start: Optional[date] = None,
        period_end: Optional[date] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Portfolio/strategy returns
            benchmark_returns: Benchmark returns for comparison
            period_start: Start date of the performance period
            period_end: End date of the performance period
            
        Returns:
            Complete performance metrics
        """
        logger.info("Calculating performance metrics")
        
        # Convert to pandas Series if needed
        if isinstance(returns, (np.ndarray, list)):
            returns = pd.Series(returns)
        
        if len(returns) == 0:
            raise ValueError("Returns series cannot be empty")
        
        # Remove any NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns after removing NaN values")
        
        # Get benchmark returns if not provided
        if benchmark_returns is None and period_start and period_end:
            benchmark_returns = self.benchmark_provider.get_benchmark_returns(
                self.config.benchmark_type,
                period_start,
                period_end,
                self.config.custom_benchmark_symbol
            )
        
        # Align benchmark returns with portfolio returns
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, (np.ndarray, list)):
                benchmark_returns = pd.Series(benchmark_returns)
            
            # Align indices
            common_dates = returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                returns = returns.loc[common_dates]
                benchmark_returns = benchmark_returns.loc[common_dates]
            else:
                logger.warning("No common dates between returns and benchmark")
                benchmark_returns = None
        
        # Calculate basic metrics
        total_return = self._calculate_total_return(returns)
        annualized_return = self._calculate_annualized_return(returns)
        volatility = self._calculate_volatility(returns)
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        
        # Calculate drawdown metrics
        dd_metrics = self._calculate_drawdown_metrics(returns)
        
        # Calculate distribution metrics
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        var_95 = self._calculate_var(returns, self.config.confidence_level)
        cvar_95 = self._calculate_cvar(returns, self.config.confidence_level)
        
        # Calculate benchmark comparison metrics
        if benchmark_returns is not None:
            alpha, beta = self._calculate_alpha_beta(returns, benchmark_returns)
            correlation = self._calculate_correlation(returns, benchmark_returns)
            tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
            information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
        else:
            alpha = beta = correlation = tracking_error = information_ratio = 0.0
        
        # Calculate statistical significance if enabled
        sharpe_pvalue = ir_pvalue = alpha_pvalue = None
        if self.config.enable_statistical_tests:
            sharpe_pvalue = self._test_sharpe_significance(returns)
            if benchmark_returns is not None:
                ir_pvalue = self._test_information_ratio_significance(returns, benchmark_returns)
                alpha_pvalue = self._test_alpha_significance(returns, benchmark_returns)
        
        # Create metrics object
        metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            max_drawdown=dd_metrics['max_drawdown'],
            max_drawdown_duration=dd_metrics['max_duration'],
            current_drawdown=dd_metrics['current_drawdown'],
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            tracking_error=tracking_error,
            sharpe_pvalue=sharpe_pvalue,
            ir_pvalue=ir_pvalue,
            alpha_pvalue=alpha_pvalue,
            period_start=period_start,
            period_end=period_end,
            observations=len(returns)
        )
        
        logger.info(f"Performance metrics calculated for {len(returns)} observations")
        return metrics
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total cumulative return."""
        return float((1 + returns).prod() - 1)
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        total_return = self._calculate_total_return(returns)
        n_periods = len(returns)
        if n_periods == 0:
            return 0.0
        
        periods_per_year = self.config.trading_days_per_year
        annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
        return float(annualized)
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) <= 1:
            return 0.0
        
        vol = returns.std() * np.sqrt(self.config.trading_days_per_year)
        return float(vol)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - (self.config.risk_free_rate / self.config.trading_days_per_year)
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(self.config.trading_days_per_year)
        return float(sharpe)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - (self.config.risk_free_rate / self.config.trading_days_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(self.config.trading_days_per_year)
        return float(sortino)
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        annualized_return = self._calculate_annualized_return(returns)
        max_dd = abs(self._calculate_drawdown_metrics(returns)['max_drawdown'])
        
        if max_dd == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return float(annualized_return / max_dd)
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics."""
        if len(returns) == 0:
            return {
                'max_drawdown': 0.0,
                'max_duration': 0,
                'current_drawdown': 0.0
            }
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = float(drawdown.min())
        
        # Current drawdown
        current_drawdown = float(drawdown.iloc[-1])
        
        # Calculate drawdown duration
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_duration': max_duration,
            'current_drawdown': current_drawdown
        }
    
    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0.0
        return float(stats.skew(returns))
    
    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 4:
            return 0.0
        return float(stats.kurtosis(returns))
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        alpha = 1 - confidence_level
        var = float(np.percentile(returns, alpha * 100))
        return var
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        cvar = float(returns[returns <= var].mean())
        return cvar if not np.isnan(cvar) else 0.0
    
    def _calculate_alpha_beta(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate alpha and beta vs benchmark."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0, 0.0
        
        # Remove risk-free rate
        rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
        excess_returns = returns - rf_daily
        excess_benchmark = benchmark_returns - rf_daily
        
        # Calculate beta
        covariance = np.cov(excess_returns, excess_benchmark)[0, 1]
        benchmark_variance = np.var(excess_benchmark)
        
        if benchmark_variance == 0:
            beta = 0.0
            alpha = float(excess_returns.mean())
        else:
            beta = float(covariance / benchmark_variance)
            alpha = float(excess_returns.mean() - beta * excess_benchmark.mean())
        
        # Annualize alpha
        alpha_annualized = alpha * self.config.trading_days_per_year
        
        return alpha_annualized, beta
    
    def _calculate_correlation(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate correlation with benchmark."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        correlation = float(np.corrcoef(returns, benchmark_returns)[0, 1])
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_tracking_error(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate tracking error vs benchmark."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        excess_returns = returns - benchmark_returns
        tracking_error = float(excess_returns.std() * np.sqrt(self.config.trading_days_per_year))
        return tracking_error
    
    def _calculate_information_ratio(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate Information ratio."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        excess_returns = returns - benchmark_returns
        if excess_returns.std() == 0:
            return 0.0
        
        ir = float(excess_returns.mean() / excess_returns.std() * np.sqrt(self.config.trading_days_per_year))
        return ir
    
    def _test_sharpe_significance(self, returns: pd.Series) -> float:
        """Test statistical significance of Sharpe ratio."""
        if len(returns) < 10:
            return 1.0  # Not significant
        
        try:
            # Use t-test for mean being different from risk-free rate
            rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
            excess_returns = returns - rf_daily
            
            t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
            return float(p_value)
            
        except Exception as e:
            logger.warning(f"Failed to test Sharpe significance: {e}")
            return 1.0
    
    def _test_information_ratio_significance(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Test statistical significance of Information ratio."""
        if len(returns) != len(benchmark_returns) or len(returns) < 10:
            return 1.0
        
        try:
            excess_returns = returns - benchmark_returns
            t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
            return float(p_value)
            
        except Exception as e:
            logger.warning(f"Failed to test IR significance: {e}")
            return 1.0
    
    def _test_alpha_significance(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Test statistical significance of alpha."""
        if len(returns) != len(benchmark_returns) or len(returns) < 10:
            return 1.0
        
        try:
            # Simple alpha test using excess returns
            rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
            excess_portfolio = returns - rf_daily
            excess_benchmark = benchmark_returns - rf_daily
            
            # Calculate alpha residuals (simplified)
            alpha_residuals = excess_portfolio - excess_benchmark
            t_stat, p_value = stats.ttest_1samp(alpha_residuals, 0)
            return float(p_value)
            
        except Exception as e:
            logger.warning(f"Failed to test alpha significance: {e}")
            return 1.0


class WalkForwardPerformanceAnalyzer:
    """
    Walk-forward validation performance analyzer.
    
    Integrates with walk-forward validation results to calculate
    performance metrics across validation windows and provide
    aggregated performance analysis.
    """
    
    def __init__(
        self,
        config: PerformanceConfig,
        temporal_store: TemporalStore,
        pit_engine: PointInTimeEngine
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        
        # Initialize benchmark provider and calculator
        self.benchmark_provider = BenchmarkDataProvider(temporal_store, pit_engine)
        self.calculator = PerformanceCalculator(config, self.benchmark_provider)
        
        logger.info("WalkForwardPerformanceAnalyzer initialized")
    
    def analyze_validation_result(
        self,
        validation_result: ValidationResult,
        returns_by_window: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Analyze performance across all validation windows.
        
        Args:
            validation_result: Walk-forward validation results
            returns_by_window: Returns for each validation window
            
        Returns:
            Comprehensive performance analysis
        """
        logger.info(f"Analyzing performance for {len(validation_result.windows)} windows")
        
        window_metrics = {}
        all_returns = []
        
        # Calculate metrics for each window
        for window in validation_result.windows:
            if window.window_id in returns_by_window:
                window_returns = returns_by_window[window.window_id]
                
                try:
                    metrics = self.calculator.calculate_metrics(
                        window_returns,
                        period_start=window.test_start,
                        period_end=window.test_end
                    )
                    window_metrics[window.window_id] = metrics
                    all_returns.extend(window_returns.tolist())
                    
                except Exception as e:
                    logger.error(f"Failed to calculate metrics for window {window.window_id}: {e}")
                    continue
        
        # Calculate overall metrics
        if all_returns:
            overall_returns = pd.Series(all_returns)
            overall_metrics = self.calculator.calculate_metrics(overall_returns)
        else:
            logger.warning("No valid returns found for overall metrics calculation")
            overall_metrics = None
        
        # Calculate cross-window statistics
        cross_window_stats = self._calculate_cross_window_statistics(window_metrics)
        
        # Performance consistency analysis
        consistency_metrics = self._analyze_performance_consistency(window_metrics)
        
        # Compile results
        analysis = {
            'overall_metrics': overall_metrics.to_dict() if overall_metrics else None,
            'window_metrics': {wid: metrics.to_dict() for wid, metrics in window_metrics.items()},
            'cross_window_statistics': cross_window_stats,
            'consistency_analysis': consistency_metrics,
            'validation_summary': {
                'total_windows': len(validation_result.windows),
                'successful_windows': len(window_metrics),
                'failed_windows': len(validation_result.windows) - len(window_metrics),
                'success_rate': len(window_metrics) / len(validation_result.windows) if validation_result.windows else 0,
                'total_observations': sum(len(returns) for returns in returns_by_window.values())
            }
        }
        
        logger.info(f"Performance analysis completed for {len(window_metrics)} windows")
        return analysis
    
    def _calculate_cross_window_statistics(
        self, 
        window_metrics: Dict[str, PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Calculate statistics across validation windows."""
        if not window_metrics:
            return {}
        
        # Extract key metrics across windows
        sharpe_ratios = [m.sharpe_ratio for m in window_metrics.values()]
        information_ratios = [m.information_ratio for m in window_metrics.values()]
        max_drawdowns = [m.max_drawdown for m in window_metrics.values()]
        total_returns = [m.total_return for m in window_metrics.values()]
        
        stats_dict = {}
        
        # Calculate statistics for each metric
        for metric_name, values in [
            ('sharpe_ratio', sharpe_ratios),
            ('information_ratio', information_ratios),
            ('max_drawdown', max_drawdowns),
            ('total_return', total_returns)
        ]:
            if values:
                stats_dict[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75))
                }
        
        return stats_dict
    
    def _analyze_performance_consistency(
        self,
        window_metrics: Dict[str, PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Analyze consistency of performance across windows."""
        if not window_metrics:
            return {}
        
        # Performance targets achievement
        target_achievements = {}
        for window_id, metrics in window_metrics.items():
            achievements = metrics.meets_targets(self.config)
            target_achievements[window_id] = achievements
        
        # Calculate achievement rates
        achievement_rates = {}
        if target_achievements:
            all_targets = list(target_achievements.values())[0].keys()
            for target in all_targets:
                achievement_rate = sum(
                    1 for achievements in target_achievements.values()
                    if achievements.get(target, False)
                ) / len(target_achievements)
                achievement_rates[target] = achievement_rate
        
        # Consistency metrics
        sharpe_ratios = [m.sharpe_ratio for m in window_metrics.values()]
        consistency_metrics = {}
        
        if len(sharpe_ratios) > 1:
            # Coefficient of variation for Sharpe ratios
            cv_sharpe = np.std(sharpe_ratios) / abs(np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) != 0 else float('inf')
            
            # Percentage of windows with positive Sharpe
            positive_sharpe_rate = sum(1 for sr in sharpe_ratios if sr > 0) / len(sharpe_ratios)
            
            consistency_metrics = {
                'sharpe_coefficient_of_variation': float(cv_sharpe),
                'positive_sharpe_rate': positive_sharpe_rate,
                'achievement_rates': achievement_rates
            }
        
        return consistency_metrics
    
    def run_performance_statistical_tests(
        self,
        validation_result: ValidationResult,
        window_returns: Dict[str, pd.Series],
        benchmark_name: str = "TAIEX"
    ) -> Dict[str, Any]:
        """
        Run statistical tests on performance metrics across validation windows.
        
        Args:
            validation_result: Walk-forward validation results
            window_returns: Returns for each validation window
            benchmark_name: Benchmark to compare against
            
        Returns:
            Statistical test results for performance metrics
        """
        logger.info(f"Running performance statistical tests against {benchmark_name}")
        
        try:
            # Import statistical testing components
            from ..validation.statistical_tests import (
                StatisticalTestEngine, create_default_statistical_config,
                TestType, sharpe_ratio_statistic, information_ratio_statistic
            )
            from ..validation.benchmarks import (
                TaiwanBenchmarkManager, create_default_benchmark_config
            )
            
            # Initialize statistical testing
            stat_config = create_default_statistical_config()
            test_engine = StatisticalTestEngine(stat_config)
            
            # Initialize benchmark manager
            bench_config = create_default_benchmark_config()
            benchmark_manager = TaiwanBenchmarkManager(
                bench_config, self.temporal_store, self.pit_engine
            )
            
            # Get validation period
            overall_start = min(window.test_start for window in validation_result.windows)
            overall_end = max(window.test_end for window in validation_result.windows)
            
            # Get benchmark returns
            benchmark_returns = benchmark_manager.get_benchmark_returns(
                benchmark_name, overall_start, overall_end, list(window_returns.keys())
            )
            
            # Collect performance metrics across windows
            window_sharpe_ratios = []
            window_information_ratios = []
            window_max_drawdowns = []
            window_total_returns = []
            
            # Calculate metrics for each window
            for window in validation_result.windows:
                if window.window_id in window_returns:
                    returns = window_returns[window.window_id]
                    
                    if len(returns) > 0:
                        # Get corresponding benchmark returns
                        window_benchmark = benchmark_returns[
                            benchmark_returns.index >= pd.Timestamp(window.test_start)
                        ][
                            benchmark_returns.index <= pd.Timestamp(window.test_end)
                        ]
                        
                        # Align lengths
                        min_length = min(len(returns), len(window_benchmark))
                        if min_length > 5:  # Minimum observations
                            aligned_returns = returns.iloc[:min_length]
                            aligned_benchmark = window_benchmark.iloc[:min_length]
                            
                            # Calculate window metrics
                            metrics = self.calculator.calculate_metrics(
                                aligned_returns,
                                aligned_benchmark,
                                window.test_start,
                                window.test_end
                            )
                            
                            window_sharpe_ratios.append(metrics.sharpe_ratio)
                            window_information_ratios.append(metrics.information_ratio)
                            window_max_drawdowns.append(metrics.max_drawdown)
                            window_total_returns.append(metrics.total_return)
            
            # Run statistical tests on performance metrics
            statistical_results = {}
            
            if len(window_sharpe_ratios) >= 3:  # Minimum windows for testing
                # Test if Sharpe ratios are significantly different from zero
                sharpe_array = np.array(window_sharpe_ratios)
                
                try:
                    from scipy.stats import ttest_1samp
                    
                    # T-test for Sharpe ratio significance
                    t_stat, p_value = ttest_1samp(sharpe_array, 0)
                    statistical_results['sharpe_ratio_ttest'] = {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'is_significant': p_value < 0.05,
                        'mean_sharpe': float(np.mean(sharpe_array)),
                        'std_sharpe': float(np.std(sharpe_array)),
                        'num_windows': len(sharpe_array)
                    }
                    
                    # Bootstrap confidence interval for mean Sharpe ratio
                    sharpe_ci = test_engine.bootstrap_confidence_interval(
                        sharpe_array,
                        lambda x: np.mean(x)
                    )
                    statistical_results['sharpe_ratio_ci'] = sharpe_ci.to_dict()
                    
                except Exception as e:
                    logger.warning(f"Sharpe ratio statistical tests failed: {e}")
            
            if len(window_information_ratios) >= 3:
                # Test Information ratios
                ir_array = np.array(window_information_ratios)
                
                try:
                    # T-test for Information ratio significance
                    t_stat, p_value = ttest_1samp(ir_array, 0)
                    statistical_results['information_ratio_ttest'] = {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'is_significant': p_value < 0.05,
                        'mean_ir': float(np.mean(ir_array)),
                        'std_ir': float(np.std(ir_array)),
                        'num_windows': len(ir_array)
                    }
                    
                    # Bootstrap confidence interval for mean Information ratio
                    ir_ci = test_engine.bootstrap_confidence_interval(
                        ir_array,
                        lambda x: np.mean(x)
                    )
                    statistical_results['information_ratio_ci'] = ir_ci.to_dict()
                    
                except Exception as e:
                    logger.warning(f"Information ratio statistical tests failed: {e}")
            
            # Test consistency of performance across windows
            if len(window_total_returns) >= 5:
                returns_array = np.array(window_total_returns)
                
                try:
                    # Test for consistency (percentage of positive windows)
                    positive_windows = np.sum(returns_array > 0)
                    total_windows = len(returns_array)
                    positive_rate = positive_windows / total_windows
                    
                    # Binomial test for hit rate
                    from scipy.stats import binom_test
                    hit_rate_p_value = binom_test(positive_windows, total_windows, 0.5)
                    
                    statistical_results['consistency_test'] = {
                        'positive_windows': int(positive_windows),
                        'total_windows': int(total_windows),
                        'positive_rate': float(positive_rate),
                        'p_value': float(hit_rate_p_value),
                        'is_significant': hit_rate_p_value < 0.05,
                        'null_hypothesis': 'Hit rate = 50%'
                    }
                    
                except Exception as e:
                    logger.warning(f"Consistency test failed: {e}")
            
            # Cross-window correlation analysis
            if len(window_sharpe_ratios) >= 5:
                try:
                    # Test for serial correlation in Sharpe ratios (stability)
                    sharpe_series = pd.Series(window_sharpe_ratios)
                    autocorr_1 = sharpe_series.autocorr(lag=1)
                    
                    # Ljung-Box test for serial correlation
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_result = acorr_ljungbox(sharpe_series, lags=1, return_df=True)
                    
                    statistical_results['serial_correlation_test'] = {
                        'autocorrelation_lag1': float(autocorr_1) if not np.isnan(autocorr_1) else 0.0,
                        'ljung_box_statistic': float(lb_result['lb_stat'].iloc[0]),
                        'ljung_box_p_value': float(lb_result['lb_pvalue'].iloc[0]),
                        'is_serially_correlated': float(lb_result['lb_pvalue'].iloc[0]) < 0.05
                    }
                    
                except Exception as e:
                    logger.warning(f"Serial correlation test failed: {e}")
            
            # Multiple comparisons across different benchmarks
            try:
                all_benchmarks = benchmark_manager.get_all_benchmark_returns(
                    overall_start, overall_end, list(window_returns.keys())
                )
                
                benchmark_test_results = []
                
                for bench_name, bench_returns in all_benchmarks.items():
                    if bench_name != benchmark_name and len(bench_returns) >= 20:
                        # Collect all model returns
                        all_model_returns = []
                        for returns in window_returns.values():
                            all_model_returns.extend(returns.tolist())
                        
                        if len(all_model_returns) >= 20:
                            # Align lengths
                            min_length = min(len(all_model_returns), len(bench_returns))
                            aligned_model = np.array(all_model_returns[:min_length])
                            aligned_bench = np.array(bench_returns.tolist()[:min_length])
                            
                            # Run Diebold-Mariano test
                            try:
                                dm_result = test_engine.diebold_mariano_test(
                                    aligned_model, aligned_bench
                                )
                                benchmark_test_results.append({
                                    'benchmark': bench_name,
                                    'test_result': dm_result.to_dict()
                                })
                                
                            except Exception as e:
                                logger.warning(f"Diebold-Mariano test failed for {bench_name}: {e}")
                
                if benchmark_test_results:
                    # Apply multiple testing correction
                    test_results = [result['test_result'] for result in benchmark_test_results]
                    multiple_test_result = test_engine.multiple_testing_correction(
                        [TestResult(
                            test_type=TestType.DIEBOLD_MARIANO,
                            statistic=tr['statistic'],
                            p_value=tr['p_value']
                        ) for tr in test_results]
                    )
                    
                    statistical_results['multiple_benchmark_tests'] = {
                        'individual_tests': benchmark_test_results,
                        'multiple_testing_correction': multiple_test_result.to_dict()
                    }
                
            except Exception as e:
                logger.warning(f"Multiple benchmark tests failed: {e}")
            
            # Summary statistics
            statistical_results['summary'] = {
                'total_statistical_tests': len([k for k in statistical_results.keys() if k != 'summary']),
                'significant_tests': len([
                    k for k, v in statistical_results.items() 
                    if k != 'summary' and isinstance(v, dict) and v.get('is_significant', False)
                ]),
                'test_period': {
                    'start': overall_start.isoformat(),
                    'end': overall_end.isoformat()
                },
                'primary_benchmark': benchmark_name,
                'total_validation_windows': len(validation_result.windows)
            }
            
            logger.info(f"Performance statistical testing completed: {len(statistical_results)} test groups")
            return statistical_results
            
        except ImportError as e:
            logger.error(f"Statistical testing modules not available: {e}")
            return {"error": "Statistical testing modules not available"}
            
        except Exception as e:
            logger.error(f"Performance statistical testing failed: {e}")
            return {"error": str(e)}


# Utility functions
def create_default_performance_config(**overrides) -> PerformanceConfig:
    """Create default performance configuration with Taiwan market settings."""
    config = PerformanceConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    return config


def create_taiwan_performance_analyzer(
    temporal_store: TemporalStore,
    pit_engine: PointInTimeEngine,
    benchmark_type: BenchmarkType = BenchmarkType.TAIEX,
    **config_overrides
) -> WalkForwardPerformanceAnalyzer:
    """Create walk-forward performance analyzer for Taiwan market."""
    config = create_default_performance_config(
        benchmark_type=benchmark_type,
        **config_overrides
    )
    return WalkForwardPerformanceAnalyzer(config, temporal_store, pit_engine)


# Example usage and testing
if __name__ == "__main__":
    print("Performance Metrics Engine demo")
    
    # Create sample returns data
    np.random.seed(42)
    n_periods = 252  # One year of daily returns
    
    # Generate sample returns with some positive drift
    returns = np.random.normal(0.0008, 0.02, n_periods)  # 0.08% daily mean, 2% daily vol
    returns_series = pd.Series(returns, index=pd.date_range('2023-01-01', periods=n_periods))
    
    # Create configuration
    config = create_default_performance_config(
        target_sharpe_ratio=1.5,
        enable_statistical_tests=True
    )
    
    # Create mock benchmark provider
    class MockBenchmarkProvider:
        def get_benchmark_returns(self, *args, **kwargs):
            benchmark_returns = np.random.normal(0.0005, 0.015, n_periods)
            return pd.Series(benchmark_returns, index=returns_series.index)
    
    # Calculate performance metrics
    calculator = PerformanceCalculator(config, MockBenchmarkProvider())
    metrics = calculator.calculate_metrics(returns_series)
    
    print(f"Demo metrics calculated:")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Information Ratio: {metrics.information_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Alpha: {metrics.alpha:.2%}")
    print(f"Beta: {metrics.beta:.2f}")
    
    # Check target achievement
    targets = metrics.meets_targets(config)
    print(f"\nTarget Achievement:")
    for target, achieved in targets.items():
        print(f"  {target}: {'✓' if achieved else '✗'}")