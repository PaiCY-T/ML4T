"""
Time-Series Cross-Validation Framework for ML Model Training

This module provides sophisticated time-series cross-validation capabilities
that integrate with the Task #23 walk-forward validation framework,
specifically designed for Taiwan market ML model validation.
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Callable
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator

from ..backtesting.validation.walk_forward import (
    WalkForwardSplitter, WalkForwardConfig, ValidationWindow, 
    WindowType, ValidationStatus
)
from ..data.core.temporal import TemporalStore, DataType
from ..data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel

logger = logging.getLogger(__name__)


class CVSplitType(Enum):
    """Cross-validation split types."""
    PURGED_GROUP_TIME_SERIES = "purged_group_ts"
    WALK_FORWARD_EXPANDING = "walk_forward_expanding" 
    WALK_FORWARD_SLIDING = "walk_forward_sliding"
    BLOCKED_TIME_SERIES = "blocked_ts"
    GAP_WALK_FORWARD = "gap_walk_forward"


@dataclass
class TimeSeriesCVConfig:
    """Configuration for time-series cross-validation."""
    
    # CV strategy configuration
    cv_type: CVSplitType = CVSplitType.PURGED_GROUP_TIME_SERIES
    n_splits: int = 5
    test_size_days: int = 60  # 3 months test period
    gap_days: int = 5  # Gap between train and test to prevent leakage
    
    # Walk-forward specific configuration
    train_period_days: int = 504  # ~2 years (156 weeks * 7 days / 7 * 5)
    rebalance_frequency_days: int = 20  # Monthly rebalancing
    min_train_size_days: int = 252  # 1 year minimum
    
    # Taiwan market specific configuration
    use_taiwan_calendar: bool = True
    settlement_lag_days: int = 2  # T+2 settlement
    handle_market_holidays: bool = True
    trading_days_only: bool = True
    
    # Bias prevention configuration
    embargo_period_days: int = 2  # Additional embargo period
    purge_overlapping_samples: bool = True
    strict_temporal_order: bool = True
    
    # Performance configuration
    enable_parallel_cv: bool = False  # Parallel CV folds
    memory_efficient_mode: bool = True
    max_memory_gb: float = 8.0


@dataclass 
class CVFoldResult:
    """Results from a single CV fold."""
    
    fold_id: int
    train_start: date
    train_end: date
    test_start: date  
    test_end: date
    gap_start: date
    gap_end: date
    
    # Data statistics
    n_train_samples: int
    n_test_samples: int
    n_train_trading_days: int
    n_test_trading_days: int
    
    # Performance metrics (filled during evaluation)
    fold_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[pd.DataFrame] = None
    
    # Validation information
    data_quality_score: float = 1.0
    bias_check_passed: bool = True
    validation_warnings: List[str] = field(default_factory=list)


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Time-series cross-validation with purged gaps and group-based splitting.
    
    This is the primary CV splitter for Taiwan market ML models, implementing:
    - Purged gaps to prevent look-ahead bias
    - Group-based splitting by time periods
    - Taiwan market calendar awareness
    - Strict temporal ordering
    """
    
    def __init__(
        self,
        config: TimeSeriesCVConfig,
        temporal_store: Optional[TemporalStore] = None,
        pit_engine: Optional[PointInTimeEngine] = None
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        
        # Initialize Taiwan calendar if needed
        if config.use_taiwan_calendar:
            from ..data.models.taiwan_market import create_taiwan_trading_calendar
            self.taiwan_calendar = create_taiwan_trading_calendar()
        else:
            self.taiwan_calendar = None
        
        logger.info(f"PurgedGroupTimeSeriesSplit initialized: {config.n_splits} splits")
    
    def split(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None, 
        groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purged gaps.
        
        Args:
            X: Feature matrix with DatetimeIndex
            y: Target variable (optional)
            groups: Group labels (optional, uses dates if None)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if not isinstance(X.index, (pd.DatetimeIndex, pd.MultiIndex)):
            raise ValueError("X must have DatetimeIndex or MultiIndex with date level")
        
        # Extract date index
        if isinstance(X.index, pd.MultiIndex):
            date_index = X.index.get_level_values(0)
        else:
            date_index = X.index
        
        # Create CV splits based on configuration
        if self.config.cv_type == CVSplitType.PURGED_GROUP_TIME_SERIES:
            yield from self._purged_group_split(X, date_index)
        elif self.config.cv_type == CVSplitType.WALK_FORWARD_EXPANDING:
            yield from self._walk_forward_split(X, date_index, expanding=True)
        elif self.config.cv_type == CVSplitType.WALK_FORWARD_SLIDING:
            yield from self._walk_forward_split(X, date_index, expanding=False)
        else:
            raise ValueError(f"Unsupported CV type: {self.config.cv_type}")
    
    def _purged_group_split(
        self, 
        X: pd.DataFrame, 
        date_index: pd.DatetimeIndex
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged group-based time series splits."""
        unique_dates = sorted(date_index.unique())
        n_dates = len(unique_dates)
        
        if n_dates < self.config.n_splits + 2:
            raise ValueError(f"Insufficient data: need at least {self.config.n_splits + 2} unique dates")
        
        # Calculate split points
        test_size = max(1, self.config.test_size_days)
        gap_size = self.config.gap_days + self.config.embargo_period_days
        
        for split_idx in range(self.config.n_splits):
            # Calculate split boundaries
            test_end_idx = n_dates - 1 - split_idx * self.config.rebalance_frequency_days
            test_start_idx = max(0, test_end_idx - test_size + 1)
            gap_start_idx = max(0, test_start_idx - gap_size)
            train_end_idx = gap_start_idx - 1
            
            if train_end_idx < self.config.min_train_size_days:
                logger.warning(f"Split {split_idx} has insufficient training data, skipping")
                continue
            
            # Convert to actual dates
            test_start_date = unique_dates[test_start_idx]
            test_end_date = unique_dates[test_end_idx]
            train_start_date = unique_dates[0]
            train_end_date = unique_dates[train_end_idx]
            
            # Create index masks
            train_mask = (date_index >= train_start_date) & (date_index <= train_end_date)
            test_mask = (date_index >= test_start_date) & (date_index <= test_end_date)
            
            # Apply Taiwan calendar filtering if enabled
            if self.config.trading_days_only and self.taiwan_calendar:
                train_mask &= date_index.to_series().apply(self.taiwan_calendar.is_trading_day)
                test_mask &= date_index.to_series().apply(self.taiwan_calendar.is_trading_day)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                logger.debug(f"Split {split_idx}: Train {len(train_indices)} samples, Test {len(test_indices)} samples")
                yield train_indices, test_indices
    
    def _walk_forward_split(
        self,
        X: pd.DataFrame,
        date_index: pd.DatetimeIndex, 
        expanding: bool = True
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits (expanding or sliding window)."""
        unique_dates = sorted(date_index.unique())
        
        # Use walk-forward splitter if available
        if self.temporal_store:
            wf_config = WalkForwardConfig(
                train_weeks=int(self.config.train_period_days / 7),
                test_weeks=int(self.config.test_size_days / 7),
                purge_weeks=int(self.config.gap_days / 7),
                rebalance_weeks=int(self.config.rebalance_frequency_days / 7),
                window_type=WindowType.EXPANDING if expanding else WindowType.SLIDING
            )
            
            wf_splitter = WalkForwardSplitter(
                config=wf_config,
                temporal_store=self.temporal_store,
                pit_engine=self.pit_engine
            )
            
            # Generate validation windows
            start_date = unique_dates[0].date()
            end_date = unique_dates[-1].date()
            
            try:
                windows = wf_splitter.generate_windows(start_date, end_date)
                
                for window in windows[:self.config.n_splits]:  # Limit to requested splits
                    # Convert window dates to indices
                    train_mask = (
                        (date_index >= pd.Timestamp(window.train_start)) & 
                        (date_index <= pd.Timestamp(window.train_end))
                    )
                    test_mask = (
                        (date_index >= pd.Timestamp(window.test_start)) & 
                        (date_index <= pd.Timestamp(window.test_end))
                    )
                    
                    train_indices = np.where(train_mask)[0]
                    test_indices = np.where(test_mask)[0]
                    
                    if len(train_indices) > 0 and len(test_indices) > 0:
                        yield train_indices, test_indices
                        
            except Exception as e:
                logger.error(f"Walk-forward splitting failed: {e}")
                # Fall back to simple time-based splitting
                yield from self._purged_group_split(X, date_index)
        else:
            # Simple walk-forward without temporal store
            yield from self._purged_group_split(X, date_index)
    
    def get_n_splits(
        self, 
        X: Optional[pd.DataFrame] = None, 
        y: Optional[pd.Series] = None, 
        groups: Optional[pd.Series] = None
    ) -> int:
        """Return the number of splitting iterations."""
        return self.config.n_splits


class TimeSeriesCrossValidator:
    """
    High-level time-series cross-validation orchestrator for ML models.
    
    This class coordinates the entire CV process including:
    - Data preparation and validation
    - CV split generation with bias prevention
    - Model training and evaluation across folds
    - Results aggregation and analysis
    """
    
    def __init__(
        self,
        config: TimeSeriesCVConfig,
        temporal_store: Optional[TemporalStore] = None,
        pit_engine: Optional[PointInTimeEngine] = None
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        
        # Initialize CV splitter
        self.cv_splitter = PurgedGroupTimeSeriesSplit(
            config=config,
            temporal_store=temporal_store,
            pit_engine=pit_engine
        )
        
        logger.info(f"TimeSeriesCrossValidator initialized with {config.cv_type.value} strategy")
    
    def cross_validate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: Optional[Dict[str, Callable]] = None,
        fit_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation on a model.
        
        Args:
            model: Sklearn-compatible model to validate
            X: Feature matrix
            y: Target variable  
            scoring: Dictionary of scoring functions
            fit_params: Parameters to pass to model.fit()
            
        Returns:
            Dictionary containing CV results and metrics
        """
        logger.info(f"Starting time-series CV with {self.config.n_splits} folds")
        
        if scoring is None:
            scoring = self._get_default_scoring()
        
        if fit_params is None:
            fit_params = {}
        
        # Validate input data
        self._validate_input_data(X, y)
        
        # Initialize results storage
        cv_results = {
            'fold_results': [],
            'scores': {metric: [] for metric in scoring.keys()},
            'fit_times': [],
            'score_times': [],
            'total_cv_time': 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Generate CV splits
            for fold_idx, (train_idx, test_idx) in enumerate(self.cv_splitter.split(X, y)):
                logger.info(f"Processing fold {fold_idx + 1}/{self.config.n_splits}")
                
                # Create fold data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Create fold result object
                fold_result = self._create_fold_result(
                    fold_idx, train_idx, test_idx, X, y
                )
                
                # Validate fold data quality
                if not self._validate_fold_data(X_train, X_test, y_train, y_test, fold_result):
                    logger.warning(f"Fold {fold_idx} failed validation, skipping")
                    continue
                
                # Train model
                fit_start = datetime.now()
                try:
                    model.fit(X_train, y_train, **fit_params)
                except Exception as e:
                    logger.error(f"Model training failed in fold {fold_idx}: {e}")
                    fold_result.validation_warnings.append(f"Training failed: {str(e)}")
                    cv_results['fold_results'].append(fold_result)
                    continue
                
                fit_time = (datetime.now() - fit_start).total_seconds()
                cv_results['fit_times'].append(fit_time)
                
                # Generate predictions and score
                score_start = datetime.now()
                try:
                    y_pred = model.predict(X_test)
                    
                    # Calculate scores
                    fold_scores = {}
                    for metric_name, metric_func in scoring.items():
                        score = metric_func(y_test, y_pred)
                        fold_scores[metric_name] = score
                        cv_results['scores'][metric_name].append(score)
                    
                    fold_result.fold_metrics = fold_scores
                    
                    # Get feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        fold_result.feature_importance = pd.DataFrame({
                            'feature': X_train.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                    
                except Exception as e:
                    logger.error(f"Model prediction/scoring failed in fold {fold_idx}: {e}")
                    fold_result.validation_warnings.append(f"Scoring failed: {str(e)}")
                
                score_time = (datetime.now() - score_start).total_seconds()
                cv_results['score_times'].append(score_time)
                
                cv_results['fold_results'].append(fold_result)
                
                logger.debug(f"Fold {fold_idx} completed in {fit_time + score_time:.1f}s")
        
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            raise
        
        # Calculate summary statistics
        cv_results['total_cv_time'] = (datetime.now() - start_time).total_seconds()
        cv_results['summary'] = self._calculate_cv_summary(cv_results)
        
        logger.info(f"Time-series CV completed in {cv_results['total_cv_time']:.1f}s")
        
        return cv_results
    
    def _get_default_scoring(self) -> Dict[str, Callable]:
        """Get default scoring functions for Taiwan market models."""
        def information_coefficient(y_true, y_pred):
            """Calculate information coefficient (rank correlation)."""
            return pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        
        def sharpe_ratio(y_true, y_pred):
            """Calculate Sharpe ratio approximation."""
            returns = pd.Series(y_pred)
            if returns.std() > 0:
                return returns.mean() / returns.std() * np.sqrt(252)
            return 0.0
        
        def hit_rate(y_true, y_pred):
            """Calculate directional accuracy."""
            return np.mean((np.sign(y_true) == np.sign(y_pred)))
        
        return {
            'ic': information_coefficient,
            'sharpe_ratio': sharpe_ratio,
            'hit_rate': hit_rate,
            'rmse': lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred) ** 2))
        }
    
    def _validate_input_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data for time-series CV."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if not isinstance(X.index, (pd.DatetimeIndex, pd.MultiIndex)):
            raise ValueError("X must have DatetimeIndex or MultiIndex with date level")
        
        if X.isnull().any().any():
            logger.warning("Found NaN values in features")
        
        if y.isnull().any():
            logger.warning("Found NaN values in target")
    
    def _create_fold_result(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        X: pd.DataFrame,
        y: pd.Series
    ) -> CVFoldResult:
        """Create fold result object with metadata."""
        # Extract dates from indices
        if isinstance(X.index, pd.MultiIndex):
            train_dates = X.index[train_idx].get_level_values(0)
            test_dates = X.index[test_idx].get_level_values(0)
        else:
            train_dates = X.index[train_idx]
            test_dates = X.index[test_idx]
        
        # Calculate trading days if calendar available
        n_train_trading_days = len(train_idx)
        n_test_trading_days = len(test_idx)
        
        if self.cv_splitter.taiwan_calendar:
            n_train_trading_days = sum(
                self.cv_splitter.taiwan_calendar.is_trading_day(d.date()) 
                for d in train_dates.unique()
            )
            n_test_trading_days = sum(
                self.cv_splitter.taiwan_calendar.is_trading_day(d.date())
                for d in test_dates.unique()  
            )
        
        return CVFoldResult(
            fold_id=fold_idx,
            train_start=train_dates.min().date(),
            train_end=train_dates.max().date(),
            test_start=test_dates.min().date(),
            test_end=test_dates.max().date(),
            gap_start=train_dates.max().date(),
            gap_end=test_dates.min().date(),
            n_train_samples=len(train_idx),
            n_test_samples=len(test_idx),
            n_train_trading_days=n_train_trading_days,
            n_test_trading_days=n_test_trading_days
        )
    
    def _validate_fold_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        fold_result: CVFoldResult
    ) -> bool:
        """Validate data quality for a single fold."""
        is_valid = True
        
        # Check minimum sample sizes
        if len(X_train) < 100:  # Minimum training samples
            fold_result.validation_warnings.append("Insufficient training samples")
            fold_result.data_quality_score *= 0.8
            is_valid = False
        
        if len(X_test) < 10:  # Minimum test samples
            fold_result.validation_warnings.append("Insufficient test samples")
            fold_result.data_quality_score *= 0.8  
            is_valid = False
        
        # Check temporal ordering
        if self.config.strict_temporal_order:
            train_max_date = X_train.index.max()
            test_min_date = X_test.index.min()
            
            if isinstance(train_max_date, tuple):  # MultiIndex
                train_max_date = train_max_date[0]
                test_min_date = test_min_date[0]
            
            if train_max_date >= test_min_date:
                fold_result.validation_warnings.append("Temporal order violation detected")
                fold_result.bias_check_passed = False
                fold_result.data_quality_score *= 0.5
        
        # Check for data leakage using PIT engine if available
        if self.pit_engine and self.config.purge_overlapping_samples:
            # Simplified bias check - full implementation would use PIT queries
            gap_days = (test_min_date - train_max_date).days
            if gap_days < self.config.gap_days:
                fold_result.validation_warnings.append(f"Insufficient gap: {gap_days} < {self.config.gap_days}")
                fold_result.bias_check_passed = False
                fold_result.data_quality_score *= 0.7
        
        return is_valid and fold_result.data_quality_score > 0.5
    
    def _calculate_cv_summary(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from CV results."""
        summary = {
            'n_folds_completed': len(cv_results['fold_results']),
            'n_folds_requested': self.config.n_splits,
            'success_rate': 0.0,
            'mean_scores': {},
            'std_scores': {},
            'score_ranges': {},
            'timing': {
                'mean_fit_time': np.mean(cv_results['fit_times']) if cv_results['fit_times'] else 0.0,
                'mean_score_time': np.mean(cv_results['score_times']) if cv_results['score_times'] else 0.0,
                'total_time': cv_results['total_cv_time']
            }
        }
        
        if summary['n_folds_completed'] > 0:
            summary['success_rate'] = summary['n_folds_completed'] / summary['n_folds_requested']
            
            # Calculate score statistics
            for metric, scores in cv_results['scores'].items():
                if scores:  # Only if we have scores for this metric
                    summary['mean_scores'][metric] = np.mean(scores)
                    summary['std_scores'][metric] = np.std(scores)
                    summary['score_ranges'][metric] = {
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'range': np.max(scores) - np.min(scores)
                    }
        
        return summary


# Utility functions

def create_taiwan_cv_config(**kwargs) -> TimeSeriesCVConfig:
    """Create CV configuration optimized for Taiwan market."""
    config = TimeSeriesCVConfig()
    
    # Taiwan-specific defaults
    config.cv_type = CVSplitType.PURGED_GROUP_TIME_SERIES
    config.test_size_days = 60  # 3 months
    config.gap_days = 7  # 1 week gap + T+2 settlement
    config.embargo_period_days = 2  # Additional Taiwan T+2 settlement
    config.trading_days_only = True
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    return config


def run_model_cv(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_type: str = 'purged_group_ts',
    n_splits: int = 5
) -> Dict[str, Any]:
    """Quick utility function for running time-series CV."""
    config = create_taiwan_cv_config(
        cv_type=CVSplitType(cv_type),
        n_splits=n_splits
    )
    
    validator = TimeSeriesCrossValidator(config)
    return validator.cross_validate_model(model, X, y)


if __name__ == "__main__":
    # Demo CV setup
    print("Time-Series Cross-Validation Framework for Taiwan Market ML Models")
    print("Integrates with Task #23 walk-forward validation framework")
    
    # Example configuration
    config = create_taiwan_cv_config()
    print(f"Default CV configuration: {config.cv_type.value}, {config.n_splits} splits")