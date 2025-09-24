"""
Model validation module for ML4T Taiwan market models.

This module provides sophisticated time-series cross-validation capabilities
that integrate with walk-forward validation and prevent data leakage.
"""

from .timeseries_cv import (
    TimeSeriesCrossValidator,
    PurgedGroupTimeSeriesSplit,
    TimeSeriesCVConfig,
    CVFoldResult,
    CVSplitType,
    create_taiwan_cv_config,
    run_model_cv
)

__all__ = [
    'TimeSeriesCrossValidator',
    'PurgedGroupTimeSeriesSplit', 
    'TimeSeriesCVConfig',
    'CVFoldResult',
    'CVSplitType',
    'create_taiwan_cv_config',
    'run_model_cv'
]