"""
Time-Series Cross-Validation with Purged K-Fold for ML4T.

This module implements advanced time-series cross-validation techniques specifically
designed for financial data, preventing look-ahead bias and handling temporal dependencies.

Key Features:
- Purged K-Fold cross-validation with gap periods
- Embargo periods for feature engineering lags
- Taiwan market-specific timing constraints
- Statistical significance testing for fold results
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Iterator, Generator
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.models.taiwan_market import TaiwanTradingCalendar, TaiwanSettlement
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel

logger = logging.getLogger(__name__)


class CVSplitType(Enum):
    """Cross-validation split types."""
    PURGED_KFOLD = "purged_kfold"
    TIME_SERIES_SPLIT = "time_series_split"
    EXPANDING_WINDOW = "expanding_window"
    BLOCKED_TIME_SERIES = "blocked_time_series"


class PurgeMethod(Enum):
    """Methods for purging overlapping observations."""
    PERCENTAGE = "percentage"  # Purge percentage of observations
    DAYS = "days"             # Purge fixed number of days
    OBSERVATIONS = "observations"  # Purge fixed number of observations


@dataclass
class CVConfig:
    """Configuration for time-series cross-validation."""
    n_splits: int = 5
    purge_method: PurgeMethod = PurgeMethod.PERCENTAGE
    purge_pct: float = 0.02  # 2% of observations to purge
    purge_days: int = 5      # Alternative: fixed days
    purge_observations: int = 100  # Alternative: fixed observations
    
    embargo_pct: float = 0.01  # 1% embargo period
    embargo_days: int = 2      # Alternative: fixed days
    embargo_observations: int = 50  # Alternative: fixed observations
    
    # Taiwan market specifics
    respect_taiwan_calendar: bool = True
    settlement_lag_days: int = 2
    
    # Validation settings
    min_train_size: int = 1000  # Minimum training observations
    min_test_size: int = 100    # Minimum test observations
    allow_incomplete_folds: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_splits <= 1:
            raise ValueError("Number of splits must be > 1")
        if self.purge_pct < 0 or self.purge_pct >= 1:
            raise ValueError("Purge percentage must be between 0 and 1")
        if self.embargo_pct < 0 or self.embargo_pct >= 1:
            raise ValueError("Embargo percentage must be between 0 and 1")


@dataclass
class CVFold:
    """A single cross-validation fold."""
    fold_number: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purge_indices: np.ndarray
    embargo_indices: np.ndarray
    
    # Temporal information
    train_start_date: date
    train_end_date: date
    test_start_date: date
    test_end_date: date
    
    # Metadata
    train_size: int
    test_size: int
    purge_size: int
    embargo_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fold to dictionary."""
        return {
            'fold_number': self.fold_number,
            'train_indices': self.train_indices.tolist(),
            'test_indices': self.test_indices.tolist(),
            'purge_indices': self.purge_indices.tolist(),
            'embargo_indices': self.embargo_indices.tolist(),
            'train_start_date': self.train_start_date.isoformat(),
            'train_end_date': self.train_end_date.isoformat(),
            'test_start_date': self.test_start_date.isoformat(),
            'test_end_date': self.test_end_date.isoformat(),
            'train_size': self.train_size,
            'test_size': self.test_size,
            'purge_size': self.purge_size,
            'embargo_size': self.embargo_size
        }


@dataclass
class CrossValidationResult:
    """Results from cross-validation."""
    config: CVConfig
    folds: List[CVFold]
    total_folds: int
    successful_folds: int
    total_observations: int
    
    # Performance metrics per fold
    fold_scores: Optional[Dict[int, Dict[str, float]]] = None
    aggregated_scores: Optional[Dict[str, float]] = None
    
    # Statistical tests
    statistical_tests: Optional[Dict[str, Any]] = None
    
    def get_fold_coverage(self) -> Dict[str, float]:
        """Calculate data coverage statistics."""
        if self.total_observations == 0:
            return {}
        
        total_train_obs = sum(fold.train_size for fold in self.folds)
        total_test_obs = sum(fold.test_size for fold in self.folds)
        total_purged_obs = sum(fold.purge_size for fold in self.folds)
        
        return {
            'train_coverage': total_train_obs / (self.total_folds * self.total_observations),
            'test_coverage': total_test_obs / (self.total_folds * self.total_observations),
            'purged_coverage': total_purged_obs / (self.total_folds * self.total_observations),
            'avg_train_size': total_train_obs / self.total_folds if self.total_folds > 0 else 0,
            'avg_test_size': total_test_obs / self.total_folds if self.total_folds > 0 else 0
        }


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series.
    
    Implements the methodology from "Advances in Financial Machine Learning" by
    Marcos López de Prado, with Taiwan market adaptations.
    
    Key features:
    - Temporal ordering preservation
    - Purge periods to prevent leakage
    - Embargo periods for feature lags
    - Taiwan market calendar integration
    """
    
    def __init__(
        self,
        config: CVConfig,
        taiwan_calendar: Optional[TaiwanTradingCalendar] = None
    ):
        self.config = config
        self.taiwan_calendar = taiwan_calendar
        self.settlement = TaiwanSettlement() if config.respect_taiwan_calendar else None
        
        logger.info(f"PurgedKFold initialized with {config.n_splits} splits")
    
    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[date]]] = None
    ) -> Iterator[CVFold]:
        """
        Generate purged K-fold splits.
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            groups: Group labels (optional)
            dates: Date index for temporal ordering
            
        Yields:
            CVFold objects with train/test/purge/embargo indices
        """
        n_samples = len(X)
        
        if n_samples < self.config.min_train_size + self.config.min_test_size:
            raise ValueError(
                f"Insufficient samples: {n_samples} < "
                f"{self.config.min_train_size + self.config.min_test_size}"
            )
        
        # Convert dates to date objects if provided
        if dates is not None:
            date_index = self._convert_to_dates(dates)
        else:
            # Create synthetic date index
            base_date = date.today() - timedelta(days=n_samples)
            date_index = [base_date + timedelta(days=i) for i in range(n_samples)]
        
        # Calculate test size and positions
        test_size = n_samples // self.config.n_splits
        if test_size < self.config.min_test_size:
            if not self.config.allow_incomplete_folds:
                raise ValueError(
                    f"Test size {test_size} < minimum {self.config.min_test_size}"
                )
            warnings.warn(f"Test size {test_size} is below minimum {self.config.min_test_size}")
        
        # Generate folds
        for fold_num in range(self.config.n_splits):
            fold = self._create_fold(
                fold_num, n_samples, test_size, date_index
            )
            
            if fold is not None:
                yield fold
    
    def _create_fold(
        self,
        fold_num: int,
        n_samples: int,
        test_size: int,
        date_index: List[date]
    ) -> Optional[CVFold]:
        """Create a single cross-validation fold."""
        try:
            # Calculate test period boundaries
            test_start_idx = fold_num * test_size
            test_end_idx = min(test_start_idx + test_size, n_samples)
            
            if test_end_idx - test_start_idx < self.config.min_test_size:
                if not self.config.allow_incomplete_folds:
                    return None
            
            # Test indices
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            # Calculate purge and embargo sizes
            purge_size = self._calculate_purge_size(n_samples, test_end_idx - test_start_idx)
            embargo_size = self._calculate_embargo_size(n_samples, test_end_idx - test_start_idx)
            
            # Purge indices (observations to remove due to overlap)
            purge_start = max(0, test_start_idx - embargo_size)
            purge_end = min(n_samples, test_end_idx + purge_size)
            purge_indices = np.arange(purge_start, purge_end)
            
            # Embargo indices (subset of purge for feature lag)
            embargo_start = max(0, test_start_idx - embargo_size)
            embargo_end = test_start_idx
            embargo_indices = np.arange(embargo_start, embargo_end)
            
            # Training indices (everything not in purge period)
            all_indices = np.arange(n_samples)
            train_indices = np.setdiff1d(all_indices, purge_indices)
            
            if len(train_indices) < self.config.min_train_size:
                if not self.config.allow_incomplete_folds:
                    return None
                warnings.warn(
                    f"Fold {fold_num}: train size {len(train_indices)} < "
                    f"minimum {self.config.min_train_size}"
                )
            
            # Get date boundaries
            train_dates = [date_index[i] for i in train_indices]
            test_dates = [date_index[i] for i in test_indices]
            
            # Adjust for Taiwan market calendar if enabled
            if self.config.respect_taiwan_calendar and self.taiwan_calendar:
                train_dates, test_dates = self._adjust_for_market_calendar(
                    train_dates, test_dates
                )
            
            # Create fold object
            fold = CVFold(
                fold_number=fold_num,
                train_indices=train_indices,
                test_indices=test_indices,
                purge_indices=purge_indices,
                embargo_indices=embargo_indices,
                train_start_date=min(train_dates) if train_dates else date_index[0],
                train_end_date=max(train_dates) if train_dates else date_index[0],
                test_start_date=min(test_dates),
                test_end_date=max(test_dates),
                train_size=len(train_indices),
                test_size=len(test_indices),
                purge_size=len(purge_indices),
                embargo_size=len(embargo_indices)
            )
            
            logger.debug(f"Created fold {fold_num}: train={len(train_indices)}, test={len(test_indices)}")
            return fold
            
        except Exception as e:
            logger.error(f"Failed to create fold {fold_num}: {e}")
            return None
    
    def _calculate_purge_size(self, n_samples: int, test_size: int) -> int:
        """Calculate number of observations to purge."""
        if self.config.purge_method == PurgeMethod.PERCENTAGE:
            return int(n_samples * self.config.purge_pct)
        elif self.config.purge_method == PurgeMethod.OBSERVATIONS:
            return self.config.purge_observations
        elif self.config.purge_method == PurgeMethod.DAYS:
            # Approximate: assume daily observations
            return self.config.purge_days
        else:
            raise ValueError(f"Unknown purge method: {self.config.purge_method}")
    
    def _calculate_embargo_size(self, n_samples: int, test_size: int) -> int:
        """Calculate embargo period size."""
        # Similar logic to purge but for embargo
        return int(n_samples * self.config.embargo_pct)
    
    def _convert_to_dates(self, dates: Union[pd.DatetimeIndex, List[date]]) -> List[date]:
        """Convert various date formats to list of date objects."""
        if isinstance(dates, pd.DatetimeIndex):
            return [d.date() for d in dates]
        elif isinstance(dates, list) and len(dates) > 0:
            if isinstance(dates[0], date):
                return dates
            elif isinstance(dates[0], datetime):
                return [d.date() for d in dates]
            else:
                raise ValueError("Unsupported date format in list")
        else:
            raise ValueError("Unsupported date format")
    
    def _adjust_for_market_calendar(
        self,
        train_dates: List[date],
        test_dates: List[date]
    ) -> Tuple[List[date], List[date]]:
        """Adjust dates for Taiwan market calendar."""
        if not self.taiwan_calendar:
            return train_dates, test_dates
        
        # Filter to trading days only
        train_trading = [d for d in train_dates if self.taiwan_calendar.is_trading_day(d)]
        test_trading = [d for d in test_dates if self.taiwan_calendar.is_trading_day(d)]
        
        return train_trading, test_trading


class TimeSeriesSplit:
    """
    Time series cross-validation with expanding windows.
    
    Simpler alternative to PurgedKFold for cases where temporal
    ordering is more important than sophisticated purging.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size
    
    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series splits."""
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Test indices
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            test_indices = indices[test_start:test_end]
            
            # Train indices
            train_end = test_start - self.gap
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            
            train_indices = indices[train_start:train_end]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class CrossValidationRunner:
    """
    High-level cross-validation orchestrator.
    
    Coordinates cross-validation execution, statistical testing,
    and result compilation.
    """
    
    def __init__(
        self,
        cv_splitter: Union[PurgedKFold, TimeSeriesSplit],
        scoring_functions: Optional[Dict[str, callable]] = None
    ):
        self.cv_splitter = cv_splitter
        self.scoring_functions = scoring_functions or {}
        
    def run_cross_validation(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model: Any,  # Any sklearn-compatible model
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[date]]] = None
    ) -> CrossValidationResult:
        """
        Run complete cross-validation with scoring and statistical tests.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Model to validate (must have fit/predict methods)
            groups: Group labels
            dates: Date index
            
        Returns:
            Complete cross-validation results
        """
        logger.info("Starting cross-validation")
        
        folds = []
        fold_scores = {}
        successful_folds = 0
        
        # Generate folds and collect scores
        if isinstance(self.cv_splitter, PurgedKFold):
            fold_iter = self.cv_splitter.split(X, y, groups, dates)
        else:
            fold_iter = self.cv_splitter.split(X, y, groups)
        
        for fold in fold_iter:
            if isinstance(fold, CVFold):
                # PurgedKFold returns CVFold objects
                train_idx, test_idx = fold.train_indices, fold.test_indices
                fold_obj = fold
            else:
                # TimeSeriesSplit returns tuples
                train_idx, test_idx = fold
                fold_obj = None
            
            try:
                # Train and predict
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[test_idx])
                
                # Calculate scores
                scores = {}
                for score_name, score_func in self.scoring_functions.items():
                    scores[score_name] = score_func(y[test_idx], y_pred)
                
                fold_number = len(folds)
                fold_scores[fold_number] = scores
                successful_folds += 1
                
                if fold_obj is not None:
                    folds.append(fold_obj)
                
                logger.debug(f"Fold {fold_number} completed with scores: {scores}")
                
            except Exception as e:
                logger.error(f"Fold {len(folds)} failed: {e}")
                continue
        
        # Calculate aggregated scores
        aggregated_scores = self._aggregate_scores(fold_scores)
        
        # Run statistical tests
        statistical_tests = self._run_statistical_tests(fold_scores)
        
        # Create result object
        config = getattr(self.cv_splitter, 'config', None)
        result = CrossValidationResult(
            config=config,
            folds=folds,
            total_folds=len(folds),
            successful_folds=successful_folds,
            total_observations=len(X),
            fold_scores=fold_scores,
            aggregated_scores=aggregated_scores,
            statistical_tests=statistical_tests
        )
        
        logger.info(f"Cross-validation completed: {successful_folds} successful folds")
        return result
    
    def _aggregate_scores(self, fold_scores: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate scores across folds."""
        if not fold_scores:
            return {}
        
        # Get all metric names
        metric_names = set()
        for scores in fold_scores.values():
            metric_names.update(scores.keys())
        
        aggregated = {}
        for metric in metric_names:
            values = [scores.get(metric, np.nan) for scores in fold_scores.values()]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
                aggregated[f"{metric}_min"] = np.min(values)
                aggregated[f"{metric}_max"] = np.max(values)
                aggregated[f"{metric}_median"] = np.median(values)
        
        return aggregated
    
    def _run_statistical_tests(self, fold_scores: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """Run statistical significance tests on fold scores."""
        if len(fold_scores) < 2:
            return {}
        
        tests = {}
        
        # Get all metric names
        metric_names = set()
        for scores in fold_scores.values():
            metric_names.update(scores.keys())
        
        for metric in metric_names:
            values = [scores.get(metric, np.nan) for scores in fold_scores.values()]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) >= 2:
                # Test for normality
                _, normality_p = stats.shapiro(values) if len(values) <= 5000 else (None, 0.05)
                
                # One-sample t-test against zero
                t_stat, t_p = stats.ttest_1samp(values, 0)
                
                tests[metric] = {
                    'n_folds': len(values),
                    'normality_p_value': normality_p,
                    'is_normal': normality_p > 0.05 if normality_p else False,
                    't_statistic': t_stat,
                    't_p_value': t_p,
                    'is_significant': t_p < 0.05,
                    'confidence_interval_95': stats.t.interval(
                        0.95, len(values)-1, loc=np.mean(values), 
                        scale=stats.sem(values)
                    ) if len(values) > 1 else None
                }
        
        return tests


# Utility functions for creating common configurations
def create_taiwan_purged_kfold(
    n_splits: int = 5,
    purge_pct: float = 0.02,
    embargo_pct: float = 0.01,
    **kwargs
) -> PurgedKFold:
    """Create PurgedKFold configured for Taiwan market."""
    config = CVConfig(
        n_splits=n_splits,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
        respect_taiwan_calendar=True,
        settlement_lag_days=2,
        **kwargs
    )
    return PurgedKFold(config)


def create_simple_time_series_split(
    n_splits: int = 5,
    gap: int = 5,
    **kwargs
) -> TimeSeriesSplit:
    """Create simple time series split."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap, **kwargs)


# Example usage
if __name__ == "__main__":
    # Demo of purged K-fold usage
    print("Time-series cross-validation demo")
    
    # Create sample data
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = np.random.randn(n_samples)
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create purged K-fold
    cv = create_taiwan_purged_kfold(n_splits=5)
    
    # Generate splits
    for i, fold in enumerate(cv.split(X, y, dates=dates)):
        print(f"Fold {i}: train={len(fold.train_indices)}, test={len(fold.test_indices)}")
        if i >= 2:  # Just show first few
            break