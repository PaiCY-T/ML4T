"""
ML4T-Alpha Integration Module.

This module provides seamless integration between the enhanced FinLab data pipeline
and ML4T-Alpha backtesting framework, enabling reliable data flow for trading
strategy development and backtesting operations.

Components:
- ml4t_data_interface: Core data interface for ML4T-Alpha compatibility
- format_converters: Data format standardization and conversion utilities
- streaming_engine: Real-time data streaming capabilities
- backtest_optimizer: Performance optimization for backtesting workflows
- pit_integration: Point-in-time data access with bias prevention
"""

__version__ = "1.0.0"

from .ml4t_data_interface import (
    ML4TDataInterface,
    ML4TDataConfig,
    create_ml4t_interface
)
from .format_converters import (
    FinLabToML4TConverter,
    OpenFEDataAdapter,
    BacktestDataFormatter
)
from .streaming_engine import (
    ML4TStreamingEngine,
    StreamingConfig,
    create_streaming_engine
)
from .backtest_optimizer import (
    BacktestOptimizer,
    OptimizationConfig,
    create_backtest_optimizer
)

__all__ = [
    'ML4TDataInterface',
    'ML4TDataConfig',
    'create_ml4t_interface',
    'FinLabToML4TConverter',
    'OpenFEDataAdapter',
    'BacktestDataFormatter',
    'ML4TStreamingEngine',
    'StreamingConfig',
    'create_streaming_engine',
    'BacktestOptimizer',
    'OptimizationConfig',
    'create_backtest_optimizer'
]