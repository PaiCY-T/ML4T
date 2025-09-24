"""
Validation Module - Task #27 Stream A & Previous Components
Statistical validation engine and temporal consistency validation for Taiwan market ML pipeline.

This module provides:
- StatisticalValidator: Comprehensive statistical validation with IC monitoring
- InformationCoefficientMonitor: Automated IC tracking with 95%+ accuracy
- DriftDetectionEngine: Advanced drift detection algorithms
- PerformanceRegimeAnalyzer: Market regime-specific performance analysis
- TaiwanMarketValidator: Taiwan market-specific validations
- TemporalConsistencyValidator: Comprehensive temporal validation
- Data leakage detection and pipeline integrity validation

Key Components:
- statistical_validator: Main statistical validation engine (Task #27 Stream A)
- taiwan_market_validator: Taiwan market-specific extensions
- temporal_checks: Temporal validation implementation (Task #28 Stream B)
"""

from .statistical_validator import (
    ValidationConfig,
    ValidationResults,
    InformationCoefficientMonitor,
    DriftDetectionEngine,
    PerformanceRegimeAnalyzer,
    StatisticalValidator
)

from .taiwan_market_validator import (
    TaiwanMarketConfig,
    TaiwanMarketValidator,
    TaiwanSettlementValidator,
    PriceLimitValidator,
    MarketStructureValidator
)

from .temporal_checks import (
    TemporalConsistencyValidator,
    validate_pipeline_temporal_integrity
)

__all__ = [
    # Statistical Validation Engine (Task #27 Stream A)
    'ValidationConfig',
    'ValidationResults', 
    'InformationCoefficientMonitor',
    'DriftDetectionEngine',
    'PerformanceRegimeAnalyzer',
    'StatisticalValidator',
    
    # Taiwan Market Validation Extensions
    'TaiwanMarketConfig',
    'TaiwanMarketValidator',
    'TaiwanSettlementValidator',
    'PriceLimitValidator',
    'MarketStructureValidator',
    
    # Temporal Validation (Task #28 Stream B)
    'TemporalConsistencyValidator',
    'validate_pipeline_temporal_integrity'
]