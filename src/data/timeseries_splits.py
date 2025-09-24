"""
Time Series Splitting - Task #28 Stream B
Proper temporal splitting implementation preventing lookahead bias.

CRITICAL: This module ensures time-series integrity through:
1. Temporal ordering (NO SHUFFLING EVER)
2. Proper train/validation/test splits 
3. Taiwan market calendar awareness
4. Panel data handling (multiple stocks)
5. Future data leakage prevention

Expert Analysis Integration:
- First 80% of time periods for training
- Last 20% of time periods for testing  
- Maintains temporal consistency across all stocks
- Respects Taiwan market trading calendar
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from abc import ABC, abstractmethod
import warnings

# Import Taiwan market configuration
try:
    from ..features.taiwan_config import TaiwanMarketConfig
except ImportError:
    # Fallback for standalone usage
    TaiwanMarketConfig = None

logger = logging.getLogger(__name__)


class TimeSeriesSplitter(ABC):
    """
    Abstract base class for time-series aware data splitting.
    
    Prevents lookahead bias by maintaining temporal ordering
    and never shuffling time-series data.
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: Optional[float] = None,
        gap_days: int = 0,
        taiwan_market: bool = True
    ):
        """
        Initialize time series splitter.
        
        Args:
            test_size: Fraction of data for test set (0.2 = 20%)
            validation_size: Optional validation set size (if None, only train/test)
            gap_days: Gap between train/test to prevent leakage
            taiwan_market: Use Taiwan market calendar
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if validation_size is not None and not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
            
        if validation_size is not None and test_size + validation_size >= 1:
            raise ValueError("test_size + validation_size must be < 1")
            
        self.test_size = test_size
        self.validation_size = validation_size
        self.gap_days = gap_days
        self.taiwan_market = taiwan_market
        
        # Initialize Taiwan market configuration if needed
        if taiwan_market and TaiwanMarketConfig:
            self.market_config = TaiwanMarketConfig()
        else:
            self.market_config = None
    
    @abstractmethod
    def split(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            X: Feature data with temporal index
            y: Optional target data
            
        Returns:
            Dictionary with split data: {'train': (X_train, y_train), ...}
        """
        pass
    
    def _validate_temporal_data(self, X: pd.DataFrame) -> None:
        """Validate that data has proper temporal structure."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # Check for temporal index
        if isinstance(X.index, pd.MultiIndex):
            # Panel data: first level should be dates
            date_level = X.index.get_level_values(0)
            try:
                pd.to_datetime(date_level)
            except (ValueError, TypeError):
                logger.warning("First level of MultiIndex may not be dates")
        else:
            # Single time series: index should be dates
            try:
                pd.to_datetime(X.index)
            except (ValueError, TypeError):
                logger.warning("Index may not be dates")
    
    def _get_trading_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Filter dates to trading days only."""
        if self.market_config is None:
            # No market calendar available, assume all business days are trading days
            return dates[dates.weekday < 5]  # Monday=0, Sunday=6
        
        trading_dates = []
        for date in dates:
            if self.market_config.is_trading_day(pd.Timestamp(date)):
                trading_dates.append(date)
                
        return pd.DatetimeIndex(trading_dates)
    
    def _create_gap(
        self, 
        train_end_date: pd.Timestamp,
        test_start_candidates: pd.DatetimeIndex
    ) -> pd.Timestamp:
        """Create gap between train and test to prevent leakage."""
        if self.gap_days <= 0:
            return test_start_candidates[0]
        
        # Find first test date after gap
        gap_end = train_end_date + timedelta(days=self.gap_days)
        
        valid_test_dates = test_start_candidates[test_start_candidates >= gap_end]
        if len(valid_test_dates) == 0:
            logger.warning(f"Gap of {self.gap_days} days may be too large")
            return test_start_candidates[0]
            
        return valid_test_dates[0]


class PanelDataSplitter(TimeSeriesSplitter):
    """
    Time-series splitter for panel data (multiple stocks over time).
    
    Handles MultiIndex DataFrames with (date, stock_id) structure.
    Splits by time periods while maintaining all stocks in each period.
    """
    
    def split(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """
        Split panel data by time periods.
        
        CRITICAL: Maintains temporal order, no shuffling.
        
        Args:
            X: DataFrame with MultiIndex (date, stock_id)
            y: Optional target with same index
            
        Returns:
            Dictionary with split data
        """
        logger.info("Starting panel data temporal split (NO SHUFFLING)")
        self._validate_temporal_data(X)
        
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("Panel data requires MultiIndex (date, stock_id)")
        
        # Extract unique dates and sort them
        dates = pd.to_datetime(X.index.get_level_values(0).unique()).sort_values()
        n_dates = len(dates)
        
        logger.info(f"Panel data: {n_dates} unique dates, {len(X)} total observations")
        
        # Filter to trading days
        trading_dates = self._get_trading_dates(dates)
        n_trading_dates = len(trading_dates)
        
        if n_trading_dates < n_dates:
            logger.info(f"Filtered to {n_trading_dates} trading days")
            dates = trading_dates
            n_dates = n_trading_dates
        
        # Calculate split indices based on time (not observations)
        if self.validation_size is not None:
            # Three-way split: train / validation / test
            train_end_idx = int(n_dates * (1 - self.test_size - self.validation_size))
            val_end_idx = int(n_dates * (1 - self.test_size))
            
            train_dates = dates[:train_end_idx]
            val_dates = dates[train_end_idx:val_end_idx]
            test_dates = dates[val_end_idx:]
            
            logger.info(
                f"Three-way temporal split: "
                f"train={len(train_dates)} dates ({dates[0]} to {train_dates[-1]}), "
                f"val={len(val_dates)} dates ({val_dates[0]} to {val_dates[-1]}), "
                f"test={len(test_dates)} dates ({test_dates[0]} to {dates[-1]})"
            )
            
        else:
            # Two-way split: train / test
            train_end_idx = int(n_dates * (1 - self.test_size))
            
            train_dates = dates[:train_end_idx]
            test_dates = dates[train_end_idx:]
            val_dates = pd.DatetimeIndex([])  # Empty
            
            logger.info(
                f"Two-way temporal split: "
                f"train={len(train_dates)} dates ({dates[0]} to {train_dates[-1]}), "
                f"test={len(test_dates)} dates ({test_dates[0]} to {dates[-1]})"
            )
        
        # Apply gap if specified
        if self.gap_days > 0 and len(test_dates) > 0:
            original_test_start = test_dates[0]
            adjusted_test_start = self._create_gap(train_dates[-1], test_dates)
            
            if adjusted_test_start != original_test_start:
                test_dates = test_dates[test_dates >= adjusted_test_start]
                logger.info(
                    f"Applied {self.gap_days}-day gap: test now starts {adjusted_test_start}"
                )
        
        # Create boolean masks for each split
        date_level = X.index.get_level_values(0)
        train_mask = pd.to_datetime(date_level).isin(train_dates)
        test_mask = pd.to_datetime(date_level).isin(test_dates)
        
        # Split the data
        X_train = X[train_mask].copy()
        X_test = X[test_mask].copy()
        
        splits = {
            'train': (X_train, y[train_mask].copy() if y is not None else None),
            'test': (X_test, y[test_mask].copy() if y is not None else None)
        }
        
        if self.validation_size is not None and len(val_dates) > 0:
            val_mask = pd.to_datetime(date_level).isin(val_dates)
            X_val = X[val_mask].copy()
            splits['validation'] = (X_val, y[val_mask].copy() if y is not None else None)
        
        # Validation: ensure no temporal overlap
        self._validate_temporal_splits(splits)
        
        # Log split statistics
        for split_name, (X_split, y_split) in splits.items():
            logger.info(f"{split_name}: {len(X_split)} observations")
        
        return splits
    
    def _validate_temporal_splits(
        self, 
        splits: Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]
    ) -> None:
        """Validate that splits have no temporal overlap."""
        all_dates = {}
        
        for split_name, (X_split, _) in splits.items():
            if len(X_split) > 0:
                dates = pd.to_datetime(X_split.index.get_level_values(0).unique())
                all_dates[split_name] = set(dates)
        
        # Check for overlaps
        for split1 in all_dates:
            for split2 in all_dates:
                if split1 < split2:  # Only check each pair once
                    overlap = all_dates[split1] & all_dates[split2]
                    if overlap:
                        raise ValueError(
                            f"Temporal overlap detected between {split1} and {split2}: "
                            f"{len(overlap)} dates"
                        )
        
        logger.info("Temporal split validation passed: no overlaps detected")


class SingleSeriesSplitter(TimeSeriesSplitter):
    """
    Time-series splitter for single time series data.
    
    Handles DataFrames with regular DatetimeIndex.
    """
    
    def split(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]:
        """
        Split single time series by time periods.
        
        Args:
            X: DataFrame with DatetimeIndex
            y: Optional target with same index
            
        Returns:
            Dictionary with split data
        """
        logger.info("Starting single series temporal split (NO SHUFFLING)")
        self._validate_temporal_data(X)
        
        # Ensure index is datetime
        if not isinstance(X.index, pd.DatetimeIndex):
            X.index = pd.to_datetime(X.index)
        
        # Sort by date (should already be sorted, but ensure)
        X = X.sort_index()
        if y is not None:
            y = y.sort_index()
        
        n_obs = len(X)
        logger.info(f"Single series: {n_obs} observations from {X.index[0]} to {X.index[-1]}")
        
        # Calculate split points
        if self.validation_size is not None:
            # Three-way split
            train_end_idx = int(n_obs * (1 - self.test_size - self.validation_size))
            val_end_idx = int(n_obs * (1 - self.test_size))
            
            X_train = X.iloc[:train_end_idx].copy()
            X_val = X.iloc[train_end_idx:val_end_idx].copy()
            X_test = X.iloc[val_end_idx:].copy()
            
            splits = {
                'train': (X_train, y.iloc[:train_end_idx].copy() if y is not None else None),
                'validation': (X_val, y.iloc[train_end_idx:val_end_idx].copy() if y is not None else None),
                'test': (X_test, y.iloc[val_end_idx:].copy() if y is not None else None)
            }
            
        else:
            # Two-way split
            train_end_idx = int(n_obs * (1 - self.test_size))
            
            X_train = X.iloc[:train_end_idx].copy()
            X_test = X.iloc[train_end_idx:].copy()
            
            splits = {
                'train': (X_train, y.iloc[:train_end_idx].copy() if y is not None else None),
                'test': (X_test, y.iloc[train_end_idx:].copy() if y is not None else None)
            }
        
        # Log split statistics
        for split_name, (X_split, _) in splits.items():
            logger.info(
                f"{split_name}: {len(X_split)} observations "
                f"({X_split.index[0]} to {X_split.index[-1]})"
            )
        
        return splits


class WalkForwardSplitter(TimeSeriesSplitter):
    """
    Walk-forward time series splitter for robust validation.
    
    Creates multiple train/test splits moving forward in time,
    useful for validating model stability over different periods.
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        n_splits: int = 5,
        gap_days: int = 0,
        taiwan_market: bool = True,
        expanding_window: bool = False
    ):
        """
        Initialize walk-forward splitter.
        
        Args:
            test_size: Size of each test set
            n_splits: Number of walk-forward splits
            gap_days: Gap between train/test
            taiwan_market: Use Taiwan market calendar
            expanding_window: If True, use expanding window; if False, rolling window
        """
        super().__init__(test_size=test_size, gap_days=gap_days, taiwan_market=taiwan_market)
        self.n_splits = n_splits
        self.expanding_window = expanding_window
    
    def split(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Dict[str, List[Tuple[pd.DataFrame, Optional[pd.Series]]]]:
        """
        Create walk-forward splits.
        
        Returns:
            Dictionary with lists of splits for train/test
        """
        logger.info(f"Starting walk-forward split with {self.n_splits} folds")
        self._validate_temporal_data(X)
        
        # Handle both panel data and single series
        if isinstance(X.index, pd.MultiIndex):
            dates = pd.to_datetime(X.index.get_level_values(0).unique()).sort_values()
        else:
            dates = pd.to_datetime(X.index.unique()).sort_values()
        
        dates = self._get_trading_dates(dates)
        n_dates = len(dates)
        
        # Calculate window sizes
        test_window = int(n_dates * self.test_size)
        min_train_window = int(n_dates * 0.3)  # Minimum 30% for training
        
        if test_window * self.n_splits > n_dates * 0.7:
            logger.warning("Walk-forward splits may have insufficient data")
        
        train_splits = []
        test_splits = []
        
        for i in range(self.n_splits):
            # Calculate test period
            test_end_idx = n_dates - i * (test_window // self.n_splits)
            test_start_idx = test_end_idx - test_window
            
            if test_start_idx < min_train_window:
                logger.warning(f"Insufficient data for split {i+1}, skipping")
                continue
            
            test_dates_split = dates[test_start_idx:test_end_idx]
            
            # Calculate train period
            if self.expanding_window:
                # Expanding: use all data before test period
                train_dates_split = dates[:test_start_idx]
            else:
                # Rolling: use fixed window before test period
                train_window = min(test_start_idx, test_window * 2)  # 2x test size
                train_start_idx = max(0, test_start_idx - train_window)
                train_dates_split = dates[train_start_idx:test_start_idx]
            
            # Apply gap
            if self.gap_days > 0:
                gap_date = train_dates_split[-1] + timedelta(days=self.gap_days)
                test_dates_split = test_dates_split[test_dates_split >= gap_date]
            
            # Create masks and split data
            if isinstance(X.index, pd.MultiIndex):
                date_level = pd.to_datetime(X.index.get_level_values(0))
                train_mask = date_level.isin(train_dates_split)
                test_mask = date_level.isin(test_dates_split)
            else:
                train_mask = pd.to_datetime(X.index).isin(train_dates_split)
                test_mask = pd.to_datetime(X.index).isin(test_dates_split)
            
            X_train_fold = X[train_mask].copy()
            X_test_fold = X[test_mask].copy()
            
            train_splits.append((
                X_train_fold, 
                y[train_mask].copy() if y is not None else None
            ))
            test_splits.append((
                X_test_fold,
                y[test_mask].copy() if y is not None else None
            ))
            
            logger.info(
                f"Fold {i+1}: train={len(X_train_fold)} obs, test={len(X_test_fold)} obs"
            )
        
        return {
            'train': train_splits,
            'test': test_splits
        }


def create_time_series_splitter(
    splitter_type: str = 'panel',
    test_size: float = 0.2,
    validation_size: Optional[float] = None,
    gap_days: int = 0,
    taiwan_market: bool = True,
    **kwargs
) -> TimeSeriesSplitter:
    """
    Factory function to create appropriate time series splitter.
    
    Args:
        splitter_type: 'panel', 'single', or 'walk_forward'
        test_size: Fraction for test set
        validation_size: Optional validation set fraction
        gap_days: Gap between train/test
        taiwan_market: Use Taiwan market calendar
        **kwargs: Additional arguments for specific splitters
        
    Returns:
        Configured TimeSeriesSplitter instance
    """
    if splitter_type == 'panel':
        return PanelDataSplitter(
            test_size=test_size,
            validation_size=validation_size,
            gap_days=gap_days,
            taiwan_market=taiwan_market
        )
    elif splitter_type == 'single':
        return SingleSeriesSplitter(
            test_size=test_size,
            validation_size=validation_size,
            gap_days=gap_days,
            taiwan_market=taiwan_market
        )
    elif splitter_type == 'walk_forward':
        return WalkForwardSplitter(
            test_size=test_size,
            gap_days=gap_days,
            taiwan_market=taiwan_market,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown splitter_type: {splitter_type}")


def validate_temporal_split_integrity(
    splits: Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]],
    original_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate temporal split integrity to prevent data leakage.
    
    Args:
        splits: Split data from splitter
        original_data: Original data before splitting
        
    Returns:
        Validation results
    """
    validation_results = {
        'passed': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    try:
        # Check data continuity and coverage
        total_split_rows = sum(len(X) for X, _ in splits.values())
        original_rows = len(original_data)
        
        coverage = total_split_rows / original_rows
        validation_results['statistics']['coverage'] = coverage
        
        if coverage < 0.95:
            validation_results['warnings'].append(
                f"Low data coverage: {coverage:.1%} of original data"
            )
        elif coverage > 1.01:
            validation_results['errors'].append(
                f"Data duplication detected: {coverage:.1%} coverage"
            )
            validation_results['passed'] = False
        
        # Check temporal ordering
        if isinstance(original_data.index, pd.MultiIndex):
            date_col = 0  # First level is dates
        else:
            date_col = None
        
        split_dates = {}
        for split_name, (X_split, _) in splits.items():
            if len(X_split) > 0:
                if date_col is not None:
                    dates = pd.to_datetime(X_split.index.get_level_values(date_col))
                else:
                    dates = pd.to_datetime(X_split.index)
                
                split_dates[split_name] = {
                    'min_date': dates.min(),
                    'max_date': dates.max(),
                    'unique_dates': len(dates.unique())
                }
        
        validation_results['statistics']['split_dates'] = split_dates
        
        # Check for temporal overlaps
        overlaps_found = False
        split_names = list(split_dates.keys())
        
        for i, split1 in enumerate(split_names):
            for split2 in split_names[i+1:]:
                dates1 = set(pd.to_datetime(splits[split1][0].index.get_level_values(0) 
                                          if isinstance(splits[split1][0].index, pd.MultiIndex)
                                          else splits[split1][0].index))
                dates2 = set(pd.to_datetime(splits[split2][0].index.get_level_values(0)
                                          if isinstance(splits[split2][0].index, pd.MultiIndex) 
                                          else splits[split2][0].index))
                
                overlap = dates1 & dates2
                if overlap:
                    validation_results['errors'].append(
                        f"Temporal overlap between {split1} and {split2}: {len(overlap)} dates"
                    )
                    overlaps_found = True
        
        if overlaps_found:
            validation_results['passed'] = False
        else:
            validation_results['statistics']['temporal_integrity'] = 'no_overlaps'
        
        # Check chronological order (train before test)
        if 'train' in split_dates and 'test' in split_dates:
            if split_dates['train']['max_date'] >= split_dates['test']['min_date']:
                validation_results['errors'].append(
                    "Temporal order violation: train data extends into test period"
                )
                validation_results['passed'] = False
            else:
                validation_results['statistics']['chronological_order'] = 'correct'
        
    except Exception as e:
        validation_results['errors'].append(f"Validation error: {str(e)}")
        validation_results['passed'] = False
    
    return validation_results