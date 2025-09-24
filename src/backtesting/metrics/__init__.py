"""
Performance Metrics Package for ML4T Walk-Forward Validation.

This package provides comprehensive performance measurement and attribution analysis
for Taiwan market quantitative trading strategies with walk-forward validation.

Key Modules:
- performance: Core performance metrics calculation (Sharpe, Information Ratio, etc.)
- attribution: Factor-based performance attribution analysis
- risk_adjusted: Advanced risk-adjusted performance metrics and VaR analysis

Key Features:
- Real-time performance metric calculation
- Taiwan market benchmark integration (TAIEX, TPEx)
- Statistical significance testing
- Factor-based attribution analysis
- Comprehensive risk measurement
"""

from .performance import (
    PerformanceCalculator,
    BenchmarkDataProvider,
    WalkForwardPerformanceAnalyzer,
    PerformanceConfig,
    PerformanceMetrics,
    BenchmarkType,
    MetricType,
    FrequencyType,
    create_default_performance_config,
    create_taiwan_performance_analyzer
)

from .attribution import (
    PerformanceAttributor,
    TaiwanFactorModel,
    AttributionResult,
    FactorExposure,
    FactorReturn,
    AttributionFactorType,
    AttributionMethod,
    create_taiwan_attribution_engine
)

from .risk_adjusted import (
    RiskCalculator,
    RollingRiskAnalyzer,
    RiskMetrics,
    RiskConfig,
    RiskMetricType,
    RiskAdjustedRatio,
    DistributionModel,
    create_taiwan_risk_config,
    create_taiwan_risk_calculator
)

__all__ = [
    # Performance metrics
    'PerformanceCalculator',
    'BenchmarkDataProvider', 
    'WalkForwardPerformanceAnalyzer',
    'PerformanceConfig',
    'PerformanceMetrics',
    'BenchmarkType',
    'MetricType',
    'FrequencyType',
    'create_default_performance_config',
    'create_taiwan_performance_analyzer',
    
    # Attribution analysis
    'PerformanceAttributor',
    'TaiwanFactorModel',
    'AttributionResult',
    'FactorExposure',
    'FactorReturn',
    'AttributionFactorType',
    'AttributionMethod',
    'create_taiwan_attribution_engine',
    
    # Risk-adjusted metrics
    'RiskCalculator',
    'RollingRiskAnalyzer',
    'RiskMetrics',
    'RiskConfig',
    'RiskMetricType',
    'RiskAdjustedRatio',
    'DistributionModel',
    'create_taiwan_risk_config',
    'create_taiwan_risk_calculator'
]

# Package version and metadata
__version__ = "1.0.0"
__author__ = "ML4T Team"
__description__ = "Performance metrics and attribution analysis for Taiwan market strategies"