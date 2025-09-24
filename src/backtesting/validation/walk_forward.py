"""
Walk-Forward Validation Engine for ML4T Taiwan Market.

This module implements a sophisticated walk-forward validation framework with:
- 156-week training / 26-week testing periods
- Taiwan market calendar integration
- Zero look-ahead bias prevention
- Rolling window advancement with purge periods
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.models.taiwan_market import (
    TaiwanTradingCalendar, TaiwanSettlement, create_taiwan_trading_calendar,
    TradingStatus
)
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """Walk-forward window types."""
    EXPANDING = "expanding"  # Growing training window
    SLIDING = "sliding"      # Fixed-size sliding window
    ANCHORED = "anchored"    # Fixed start, growing end


class ValidationStatus(Enum):
    """Validation window status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    train_weeks: int = 156  # 3 years
    test_weeks: int = 26    # 6 months
    purge_weeks: int = 2    # Gap between train/test
    rebalance_weeks: int = 4  # Monthly rebalancing
    window_type: WindowType = WindowType.SLIDING
    min_history_weeks: int = 260  # 5 years minimum
    
    # Taiwan market specifics
    use_taiwan_calendar: bool = True
    settlement_lag_days: int = 2  # T+2 settlement
    handle_lunar_new_year: bool = True
    
    # Bias prevention
    bias_check_level: BiasCheckLevel = BiasCheckLevel.STRICT
    validate_data_lags: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.train_weeks <= 0:
            raise ValueError("Training weeks must be positive")
        if self.test_weeks <= 0:
            raise ValueError("Testing weeks must be positive")
        if self.purge_weeks < 0:
            raise ValueError("Purge weeks cannot be negative")
        if self.rebalance_weeks <= 0:
            raise ValueError("Rebalance weeks must be positive")


@dataclass
class ValidationWindow:
    """A single validation window with train/test periods."""
    window_id: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    purge_start: date
    purge_end: date
    
    # Metadata
    window_number: int
    total_train_days: int
    total_test_days: int
    purge_days: int
    trading_days_train: int
    trading_days_test: int
    
    # Status tracking
    status: ValidationStatus = ValidationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'window_id': self.window_id,
            'train_start': self.train_start.isoformat(),
            'train_end': self.train_end.isoformat(),
            'test_start': self.test_start.isoformat(),
            'test_end': self.test_end.isoformat(),
            'purge_start': self.purge_start.isoformat(),
            'purge_end': self.purge_end.isoformat(),
            'window_number': self.window_number,
            'total_train_days': self.total_train_days,
            'total_test_days': self.total_test_days,
            'purge_days': self.purge_days,
            'trading_days_train': self.trading_days_train,
            'trading_days_test': self.trading_days_test,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }


@dataclass
class ValidationResult:
    """Results from walk-forward validation."""
    config: WalkForwardConfig
    windows: List[ValidationWindow]
    total_windows: int
    successful_windows: int
    failed_windows: int
    total_runtime_seconds: float
    
    # Performance metrics (to be filled by metrics module)
    performance_summary: Optional[Dict[str, Any]] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    
    # Statistical testing results (to be filled by statistical testing module)
    statistical_tests: Optional[Dict[str, Any]] = None
    benchmark_comparisons: Optional[Dict[str, Any]] = None
    significance_results: Optional[Dict[str, Any]] = None
    
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_windows == 0:
            return 0.0
        return self.successful_windows / self.total_windows
    
    def get_window_by_id(self, window_id: str) -> Optional[ValidationWindow]:
        """Get validation window by ID."""
        for window in self.windows:
            if window.window_id == window_id:
                return window
        return None


class WalkForwardSplitter:
    """
    Core walk-forward validation engine with Taiwan market integration.
    
    Features:
    - 156-week training / 26-week testing framework
    - Taiwan market calendar and settlement handling
    - Zero look-ahead bias prevention
    - Multiple window types (expanding, sliding, anchored)
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
        temporal_store: TemporalStore,
        pit_engine: Optional[PointInTimeEngine] = None,
        taiwan_calendar: Optional[TaiwanTradingCalendar] = None
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine or PointInTimeEngine(temporal_store)
        
        # Initialize Taiwan calendar if needed
        if config.use_taiwan_calendar:
            self.taiwan_calendar = taiwan_calendar or create_taiwan_trading_calendar()
        else:
            self.taiwan_calendar = None
            
        # Settlement handler for T+2 timing
        self.settlement = TaiwanSettlement()
        
        logger.info(f"WalkForwardSplitter initialized with config: {config}")
    
    def generate_windows(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[ValidationWindow]:
        """
        Generate walk-forward validation windows.
        
        Args:
            start_date: Start date for validation period
            end_date: End date for validation period
            symbols: Optional list of symbols to validate data availability
            
        Returns:
            List of validation windows
            
        Raises:
            ValueError: If insufficient data or invalid date range
        """
        logger.info(f"Generating walk-forward windows from {start_date} to {end_date}")
        
        # Validate input parameters
        self._validate_date_range(start_date, end_date)
        
        # Calculate actual start date with minimum history requirement
        actual_start = self._calculate_actual_start_date(start_date)
        
        windows = []
        window_number = 1
        current_date = actual_start
        
        while True:
            # Calculate window dates
            window = self._create_validation_window(
                current_date, window_number, symbols
            )
            
            if window is None or window.test_end > end_date:
                break
                
            windows.append(window)
            window_number += 1
            
            # Advance by rebalancing frequency
            current_date = self._advance_window(current_date)
        
        logger.info(f"Generated {len(windows)} validation windows")
        return windows
    
    def validate_window(
        self,
        window: ValidationWindow,
        symbols: List[str],
        data_types: List[DataType] = None
    ) -> bool:
        """
        Validate data availability and quality for a window.
        
        Args:
            window: Validation window to check
            symbols: List of symbols to validate
            data_types: Data types to check (default: PRICE, VOLUME)
            
        Returns:
            True if window has sufficient data quality
        """
        if data_types is None:
            data_types = [DataType.PRICE, DataType.VOLUME]
        
        try:
            # Check training period data availability
            train_query = PITQuery(
                symbols=symbols,
                as_of_date=window.train_end,
                data_types=data_types,
                start_date=window.train_start,
                end_date=window.train_end
            )
            
            train_available = self.pit_engine.check_data_availability(train_query)
            if not train_available:
                window.error_message = "Insufficient training data"
                return False
            
            # Check test period data availability
            test_query = PITQuery(
                symbols=symbols,
                as_of_date=window.test_end,
                data_types=data_types,
                start_date=window.test_start,
                end_date=window.test_end
            )
            
            test_available = self.pit_engine.check_data_availability(test_query)
            if not test_available:
                window.error_message = "Insufficient test data"
                return False
            
            # Validate no look-ahead bias
            bias_detected = self._check_lookhead_bias(window, symbols, data_types)
            if bias_detected:
                window.error_message = "Look-ahead bias detected"
                return False
            
            logger.debug(f"Window {window.window_id} validation passed")
            return True
            
        except Exception as e:
            window.error_message = f"Validation error: {str(e)}"
            logger.error(f"Window validation failed: {e}")
            return False
    
    def _validate_date_range(self, start_date: date, end_date: date) -> None:
        """Validate that date range is sufficient for walk-forward validation."""
        if end_date <= start_date:
            raise ValueError("End date must be after start date")
        
        total_days = (end_date - start_date).days
        min_required_days = (
            self.config.min_history_weeks * 7 +
            self.config.train_weeks * 7 +
            self.config.test_weeks * 7 +
            self.config.purge_weeks * 7
        )
        
        if total_days < min_required_days:
            raise ValueError(
                f"Insufficient data period. Need {min_required_days} days, got {total_days}"
            )
    
    def _calculate_actual_start_date(self, requested_start: date) -> date:
        """Calculate actual start date including minimum history requirement."""
        min_history_days = self.config.min_history_weeks * 7
        actual_start = requested_start + timedelta(days=min_history_days)
        
        # Adjust for Taiwan trading calendar if enabled
        if self.config.use_taiwan_calendar and self.taiwan_calendar:
            actual_start = self._adjust_for_trading_calendar(actual_start)
        
        return actual_start
    
    def _create_validation_window(
        self,
        current_date: date,
        window_number: int,
        symbols: Optional[List[str]] = None
    ) -> Optional[ValidationWindow]:
        """Create a single validation window."""
        try:
            # Calculate base dates
            if self.config.window_type == WindowType.EXPANDING:
                # Expanding window: fixed start, growing training period
                train_start = current_date - timedelta(
                    weeks=self.config.min_history_weeks + 
                    (window_number - 1) * self.config.rebalance_weeks
                )
            else:
                # Sliding window: fixed-size training period
                train_start = current_date - timedelta(weeks=self.config.train_weeks)
            
            train_end = current_date
            purge_start = train_end
            purge_end = purge_start + timedelta(weeks=self.config.purge_weeks)
            test_start = purge_end
            test_end = test_start + timedelta(weeks=self.config.test_weeks)
            
            # Adjust dates for Taiwan trading calendar
            if self.config.use_taiwan_calendar:
                train_start, train_end = self._adjust_for_trading_calendar(train_start, train_end)
                test_start, test_end = self._adjust_for_trading_calendar(test_start, test_end)
                purge_start, purge_end = purge_start, purge_end  # Purge period can include non-trading days
            
            # Handle T+2 settlement lag
            if self.config.settlement_lag_days > 0:
                test_start = self.settlement.adjust_for_settlement_lag(
                    test_start, self.config.settlement_lag_days
                )
            
            # Calculate trading days
            trading_days_train = self._count_trading_days(train_start, train_end)
            trading_days_test = self._count_trading_days(test_start, test_end)
            
            # Create window
            window = ValidationWindow(
                window_id=f"wf_{window_number:04d}_{train_start.strftime('%Y%m%d')}_{test_start.strftime('%Y%m%d')}",
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_start=purge_start,
                purge_end=purge_end,
                window_number=window_number,
                total_train_days=(train_end - train_start).days,
                total_test_days=(test_end - test_start).days,
                purge_days=(purge_end - purge_start).days,
                trading_days_train=trading_days_train,
                trading_days_test=trading_days_test
            )
            
            logger.debug(f"Created window {window.window_id}")
            return window
            
        except Exception as e:
            logger.error(f"Failed to create validation window: {e}")
            return None
    
    def _advance_window(self, current_date: date) -> date:
        """Advance the window by rebalancing frequency."""
        next_date = current_date + timedelta(weeks=self.config.rebalance_weeks)
        
        # Adjust for trading calendar if enabled
        if self.config.use_taiwan_calendar and self.taiwan_calendar:
            next_date = self._adjust_for_trading_calendar(next_date)
        
        return next_date
    
    def _adjust_for_trading_calendar(
        self, 
        start_date: date, 
        end_date: Optional[date] = None
    ) -> Union[date, Tuple[date, date]]:
        """Adjust dates for Taiwan trading calendar."""
        if not self.taiwan_calendar:
            return start_date if end_date is None else (start_date, end_date)
        
        # Find next trading day for start date
        adjusted_start = start_date
        while not self.taiwan_calendar.is_trading_day(adjusted_start):
            adjusted_start += timedelta(days=1)
        
        if end_date is None:
            return adjusted_start
        
        # Find previous trading day for end date
        adjusted_end = end_date
        while not self.taiwan_calendar.is_trading_day(adjusted_end):
            adjusted_end -= timedelta(days=1)
        
        return adjusted_start, adjusted_end
    
    def _count_trading_days(self, start_date: date, end_date: date) -> int:
        """Count trading days between two dates."""
        if not self.taiwan_calendar:
            # Approximate: 5 trading days per week
            total_days = (end_date - start_date).days
            return int(total_days * 5 / 7)
        
        count = 0
        current = start_date
        while current <= end_date:
            if self.taiwan_calendar.is_trading_day(current):
                count += 1
            current += timedelta(days=1)
        
        return count
    
    def _check_lookhead_bias(
        self,
        window: ValidationWindow,
        symbols: List[str],
        data_types: List[DataType]
    ) -> bool:
        """Check for look-ahead bias in validation window."""
        if self.config.bias_check_level == BiasCheckLevel.NONE:
            return False
        
        try:
            # Query for data that should not be available at train_end
            future_query = PITQuery(
                symbols=symbols,
                as_of_date=window.train_end,
                data_types=data_types,
                start_date=window.test_start,
                end_date=window.test_end
            )
            
            # This should return no data if bias checking is working
            future_data = self.pit_engine.query(
                future_query,
                bias_check_level=self.config.bias_check_level
            )
            
            # If we get any data from the future, there's bias
            has_future_data = any(
                len(symbol_data) > 0 
                for symbol_data in future_data.values()
            )
            
            if has_future_data:
                logger.warning(f"Look-ahead bias detected in window {window.window_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Bias check failed for window {window.window_id}: {e}")
            # Conservative approach: assume bias if check fails
            return True


class WalkForwardValidator:
    """
    High-level walk-forward validation orchestrator.
    
    Coordinates the entire validation process including data validation,
    window generation, and result compilation.
    """
    
    def __init__(
        self,
        splitter: WalkForwardSplitter,
        symbols: List[str],
        data_types: List[DataType] = None,
        enable_statistical_testing: bool = True
    ):
        self.splitter = splitter
        self.symbols = symbols
        self.data_types = data_types or [DataType.PRICE, DataType.VOLUME]
        self.enable_statistical_testing = enable_statistical_testing
        
        # Initialize statistical testing components if enabled
        self._statistical_engine = None
        self._benchmark_manager = None
        
        if enable_statistical_testing:
            try:
                from .statistical_tests import StatisticalTestEngine, create_default_statistical_config
                from .benchmarks import TaiwanBenchmarkManager, create_default_benchmark_config
                
                # Initialize statistical testing
                stat_config = create_default_statistical_config()
                self._statistical_engine = StatisticalTestEngine(stat_config)
                
                # Initialize benchmark manager
                bench_config = create_default_benchmark_config()
                self._benchmark_manager = TaiwanBenchmarkManager(
                    bench_config, 
                    self.splitter.temporal_store, 
                    self.splitter.pit_engine
                )
                
                logger.info("Statistical testing enabled for walk-forward validation")
                
            except ImportError as e:
                logger.warning(f"Statistical testing modules not available: {e}")
                self.enable_statistical_testing = False
        
    def run_validation(
        self,
        start_date: date,
        end_date: date,
        validate_windows: bool = True
    ) -> ValidationResult:
        """
        Run complete walk-forward validation.
        
        Args:
            start_date: Start date for validation
            end_date: End date for validation
            validate_windows: Whether to validate data availability
            
        Returns:
            Complete validation results
        """
        start_time = datetime.now()
        logger.info(f"Starting walk-forward validation for {len(self.symbols)} symbols")
        
        try:
            # Generate validation windows
            windows = self.splitter.generate_windows(start_date, end_date, self.symbols)
            
            successful_windows = 0
            failed_windows = 0
            
            if validate_windows:
                # Validate each window
                for window in windows:
                    window.status = ValidationStatus.IN_PROGRESS
                    
                    is_valid = self.splitter.validate_window(
                        window, self.symbols, self.data_types
                    )
                    
                    if is_valid:
                        window.status = ValidationStatus.COMPLETED
                        window.completed_at = datetime.now()
                        successful_windows += 1
                    else:
                        window.status = ValidationStatus.FAILED
                        failed_windows += 1
            else:
                # Mark all windows as completed without validation
                for window in windows:
                    window.status = ValidationStatus.COMPLETED
                    window.completed_at = datetime.now()
                successful_windows = len(windows)
            
            # Compile results
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            result = ValidationResult(
                config=self.splitter.config,
                windows=windows,
                total_windows=len(windows),
                successful_windows=successful_windows,
                failed_windows=failed_windows,
                total_runtime_seconds=runtime
            )
            
            logger.info(
                f"Walk-forward validation completed: {successful_windows}/{len(windows)} "
                f"windows successful in {runtime:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}")
            raise
    
    def run_statistical_tests(
        self,
        validation_result: ValidationResult,
        model_returns: Dict[str, pd.Series],
        benchmark_name: str = "TAIEX"
    ) -> ValidationResult:
        """
        Run statistical significance tests on validation results.
        
        Args:
            validation_result: Completed validation result
            model_returns: Dictionary mapping window_id to model returns
            benchmark_name: Benchmark to compare against
            
        Returns:
            Updated validation result with statistical tests
        """
        if not self.enable_statistical_testing:
            logger.warning("Statistical testing is disabled")
            return validation_result
        
        if not self._statistical_engine or not self._benchmark_manager:
            logger.error("Statistical testing components not initialized")
            return validation_result
        
        logger.info(f"Running statistical tests against {benchmark_name} benchmark")
        
        try:
            # Get benchmark returns for the validation period
            overall_start = min(window.test_start for window in validation_result.windows)
            overall_end = max(window.test_end for window in validation_result.windows)
            
            benchmark_returns = self._benchmark_manager.get_benchmark_returns(
                benchmark_name, overall_start, overall_end, self.symbols
            )
            
            # Run statistical tests
            statistical_tests = {}
            benchmark_comparisons = {}
            significance_results = {}
            
            # Collect all model returns
            all_model_returns = []
            all_benchmark_returns = []
            
            for window_id, returns in model_returns.items():
                if len(returns) > 0:
                    all_model_returns.extend(returns.tolist())
                    
                    # Get corresponding benchmark returns
                    window = validation_result.get_window_by_id(window_id)
                    if window:
                        window_benchmark = benchmark_returns[
                            benchmark_returns.index >= pd.Timestamp(window.test_start)
                        ][
                            benchmark_returns.index <= pd.Timestamp(window.test_end)
                        ]
                        all_benchmark_returns.extend(window_benchmark.tolist())
            
            if len(all_model_returns) > 0 and len(all_benchmark_returns) > 0:
                # Ensure same length
                min_length = min(len(all_model_returns), len(all_benchmark_returns))
                all_model_returns = all_model_returns[:min_length]
                all_benchmark_returns = all_benchmark_returns[:min_length]
                
                # Run Diebold-Mariano test
                if min_length >= 10:
                    try:
                        dm_result = self._statistical_engine.diebold_mariano_test(
                            np.array(all_model_returns),
                            np.array(all_benchmark_returns)
                        )
                        statistical_tests['diebold_mariano'] = dm_result.to_dict()
                        
                    except Exception as e:
                        logger.warning(f"Diebold-Mariano test failed: {e}")
                
                # Run Hansen SPA test (if we have multiple model variants)
                if len(model_returns) > 1:
                    try:
                        model_series_list = [returns for returns in model_returns.values() if len(returns) > 0]
                        if len(model_series_list) > 1:
                            spa_result = self._statistical_engine.hansen_spa_test(
                                np.array(all_benchmark_returns),
                                [np.array(model_series.tolist()) for model_series in model_series_list]
                            )
                            statistical_tests['hansen_spa'] = spa_result.to_dict()
                            
                    except Exception as e:
                        logger.warning(f"Hansen SPA test failed: {e}")
                
                # Run White Reality Check
                if len(model_returns) > 1:
                    try:
                        wrc_result = self._statistical_engine.white_reality_check(
                            np.array(all_benchmark_returns),
                            [np.array(model_series.tolist()) for model_series in model_returns.values() if len(model_series) > 0]
                        )
                        statistical_tests['white_reality_check'] = wrc_result.to_dict()
                        
                    except Exception as e:
                        logger.warning(f"White Reality Check failed: {e}")
                
                # Calculate bootstrap confidence intervals for key metrics
                try:
                    from .statistical_tests import sharpe_ratio_statistic, information_ratio_statistic
                    
                    # Sharpe ratio CI
                    sharpe_ci = self._statistical_engine.bootstrap_confidence_interval(
                        np.array(all_model_returns),
                        lambda x: sharpe_ratio_statistic(x, 0.01)
                    )
                    statistical_tests['sharpe_ratio_ci'] = sharpe_ci.to_dict()
                    
                    # Information ratio CI
                    ir_ci = self._statistical_engine.bootstrap_confidence_interval(
                        np.array(all_model_returns) - np.array(all_benchmark_returns),
                        lambda x: np.mean(x) / np.std(x) * np.sqrt(252) if np.std(x) > 0 else 0
                    )
                    statistical_tests['information_ratio_ci'] = ir_ci.to_dict()
                    
                except Exception as e:
                    logger.warning(f"Bootstrap confidence intervals failed: {e}")
            
            # Get additional benchmark comparisons
            try:
                all_benchmarks = self._benchmark_manager.get_all_benchmark_returns(
                    overall_start, overall_end, self.symbols
                )
                
                for bench_name, bench_returns in all_benchmarks.items():
                    if bench_name != benchmark_name and len(bench_returns) > 0:
                        # Calculate correlation and other comparison metrics
                        if len(all_model_returns) > 0:
                            aligned_bench = bench_returns[:len(all_model_returns)]
                            if len(aligned_bench) > 1:
                                correlation = np.corrcoef(all_model_returns, aligned_bench)[0, 1]
                                benchmark_comparisons[bench_name] = {
                                    'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                                    'mean_return': float(np.mean(aligned_bench)),
                                    'volatility': float(np.std(aligned_bench))
                                }
                
            except Exception as e:
                logger.warning(f"Benchmark comparisons failed: {e}")
            
            # Summarize significance results
            significance_results = {
                'num_significant_tests': sum(
                    1 for test_result in statistical_tests.values()
                    if isinstance(test_result, dict) and test_result.get('is_significant', False)
                ),
                'total_tests': len(statistical_tests),
                'primary_benchmark': benchmark_name,
                'test_period': {
                    'start': overall_start.isoformat(),
                    'end': overall_end.isoformat(),
                    'total_observations': len(all_model_returns)
                }
            }
            
            # Update validation result
            validation_result.statistical_tests = statistical_tests
            validation_result.benchmark_comparisons = benchmark_comparisons
            validation_result.significance_results = significance_results
            
            logger.info(f"Statistical testing completed: {len(statistical_tests)} tests run")
            
        except Exception as e:
            logger.error(f"Statistical testing failed: {e}")
            # Don't fail the entire validation, just log the error
            validation_result.statistical_tests = {"error": str(e)}
        
        return validation_result


# Example usage and testing functions
def create_default_config(**kwargs) -> WalkForwardConfig:
    """Create default walk-forward configuration with overrides."""
    config = WalkForwardConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    return config


def demo_walk_forward_validation():
    """Demonstration of walk-forward validation usage."""
    # This would be called with actual TemporalStore and symbols
    print("Walk-forward validation demo - requires actual data stores")
    
    config = create_default_config(
        train_weeks=52,  # 1 year for demo
        test_weeks=13,   # 3 months for demo
        rebalance_weeks=4
    )
    
    print(f"Demo config: {config}")
    print("In actual usage, initialize with TemporalStore and run validation")


if __name__ == "__main__":
    demo_walk_forward_validation()