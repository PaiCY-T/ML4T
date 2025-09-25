"""
Validation Module - Task #27 Stream A, #28 Stream B & Issue #57
Statistical validation engine, temporal consistency validation, and comprehensive
data validation framework for Taiwan market ML pipeline.

This module provides:
- StatisticalValidator: Comprehensive statistical validation with IC monitoring
- InformationCoefficientMonitor: Automated IC tracking with 95%+ accuracy
- DriftDetectionEngine: Advanced drift detection algorithms
- PerformanceRegimeAnalyzer: Market regime-specific performance analysis
- TaiwanMarketValidator: Taiwan market-specific validations
- TemporalConsistencyValidator: Comprehensive temporal validation
- DataValidationFramework: Comprehensive data quality assurance (Issue #57)
- Data leakage detection and pipeline integrity validation

Key Components:
- statistical_validator: Main statistical validation engine (Task #27 Stream A)
- taiwan_market_validator: Taiwan market-specific extensions
- temporal_checks: Temporal validation implementation (Task #28 Stream B)
- data_validation_framework: Comprehensive data validation for FinLab datasets (Issue #57)
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

from .data_validation_framework import (
    DataValidationFramework,
    FinLabSchemaValidator,
    PointInTimeValidator,
    StatisticalOutlierDetector,
    DataQualityScorer,
    DataLineageTracker,
    ValidationReporter,
    ValidationConfig as DataValidationConfig,
    ValidationSeverity,
    DataType,
    SchemaField,
    ValidationResult,
    DataQualityMetrics,
    DataLineageRecord
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
    'validate_pipeline_temporal_integrity',

    # Data Validation Framework (Issue #57)
    'DataValidationFramework',
    'FinLabSchemaValidator',
    'PointInTimeValidator',
    'StatisticalOutlierDetector',
    'DataQualityScorer',
    'DataLineageTracker',
    'ValidationReporter',
    'DataValidationConfig',
    'ValidationSeverity',
    'DataType',
    'SchemaField',
    'ValidationResult',
    'DataQualityMetrics',
    'DataLineageRecord'
]