"""
Backtesting validation framework for ML4T.

This module provides walk-forward validation, time-series cross-validation,
regime detection, and Taiwan market-specific validation components.
"""

from .walk_forward import (
    WalkForwardSplitter,
    WalkForwardConfig,
    ValidationWindow,
    ValidationResult
)
from .time_series_cv import (
    PurgedKFold,
    TimeSeriesSplit,
    CrossValidationResult
)
from .taiwan_specific import (
    TaiwanValidationConfig,
    TaiwanMarketValidator,
    SettlementValidator
)
from .regime_detection import (
    RegimeDetector,
    RegimeConfig,
    MarketRegime,
    RegimeState,
    StabilityTester,
    StabilityTestResult
)
from .statistical_tests import (
    StatisticalTestEngine,
    StatisticalTestConfig,
    TestResult,
    TestType,
    MultipleTestResult,
    create_default_statistical_config
)
from .benchmarks import (
    TaiwanBenchmarkManager,
    BenchmarkDefinition,
    BenchmarkCalculator,
    BenchmarkConfig,
    BenchmarkCategory,
    create_default_benchmark_config
)

__all__ = [
    'WalkForwardSplitter',
    'WalkForwardConfig', 
    'ValidationWindow',
    'ValidationResult',
    'PurgedKFold',
    'TimeSeriesSplit',
    'CrossValidationResult',
    'TaiwanValidationConfig',
    'TaiwanMarketValidator',
    'SettlementValidator',
    'RegimeDetector',
    'RegimeConfig',
    'MarketRegime',
    'RegimeState',
    'StabilityTester',
    'StabilityTestResult',
    'StatisticalTestEngine',
    'StatisticalTestConfig',
    'TestResult',
    'TestType',
    'MultipleTestResult',
    'create_default_statistical_config',
    'TaiwanBenchmarkManager',
    'BenchmarkDefinition',
    'BenchmarkCalculator',
    'BenchmarkConfig',
    'BenchmarkCategory',
    'create_default_benchmark_config'
]