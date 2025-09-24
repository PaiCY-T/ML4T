"""
Business Logic Validation Framework for ML4T Taiwan Equity Alpha.

This module provides comprehensive business logic validation for model predictions,
trading strategies, and portfolio construction with Taiwan market-specific rules.

Key Components:
- Taiwan regulatory compliance validation
- Trading strategy coherence checks
- Economic intuition scoring
- Sector neutrality analysis
- Risk management integration

Author: ML4T Team
Date: 2025-09-24
"""

from .regulatory_validator import (
    RegulatoryValidator,
    TaiwanRegulatoryConfig,
    ComplianceIssue,
    ComplianceSeverity,
    create_standard_regulatory_validator,
    create_strict_regulatory_validator
)

from .strategy_coherence import (
    StrategyCoherenceValidator,
    CoherenceConfig,
    CoherenceCheck,
    CoherenceResult,
    TradingStrategyType
)

from .economic_intuition import (
    EconomicIntuitionScorer,
    IntuitionConfig,
    IntuitionScore,
    EconomicSignal,
    create_standard_intuition_scorer
)

from .sector_analysis import (
    SectorNeutralityAnalyzer,
    SectorConfig,
    SectorExposure,
    StyleExposure,
    NeutralityResult
)

from .risk_integration import (
    RiskValidator,
    RiskConfig,
    RiskConstraint,
    RiskViolation,
    PositionRiskCheck
)

from .business_validator import (
    BusinessLogicValidator,
    ValidationConfig,
    ValidationResult,
    create_comprehensive_validator
)

from .backtesting_integration import (
    BacktestingValidator,
    BacktestValidationConfig,
    BacktestValidationResult,
    BacktestValidationPhase,
    ValidationTiming,
    create_backtesting_validator,
    create_production_validator
)

__all__ = [
    # Main validator
    'BusinessLogicValidator',
    'ValidationConfig',
    'ValidationResult',
    'create_comprehensive_validator',
    
    # Backtesting integration
    'BacktestingValidator',
    'BacktestValidationConfig',
    'BacktestValidationResult',
    'BacktestValidationPhase',
    'ValidationTiming',
    'create_backtesting_validator',
    'create_production_validator',
    
    # Regulatory compliance
    'RegulatoryValidator',
    'TaiwanRegulatoryConfig',
    'ComplianceIssue',
    'ComplianceSeverity',
    'create_standard_regulatory_validator',
    'create_strict_regulatory_validator',
    
    # Strategy coherence
    'StrategyCoherenceValidator',
    'CoherenceConfig',
    'CoherenceCheck',
    'CoherenceResult',
    'TradingStrategyType',
    
    # Economic intuition
    'EconomicIntuitionScorer',
    'IntuitionConfig',
    'IntuitionScore',
    'EconomicSignal',
    'create_standard_intuition_scorer',
    
    # Sector analysis
    'SectorNeutralityAnalyzer',
    'SectorConfig',
    'SectorExposure',
    'StyleExposure',
    'NeutralityResult',
    
    # Risk integration
    'RiskValidator',
    'RiskConfig',
    'RiskConstraint',
    'RiskViolation',
    'PositionRiskCheck'
]

__version__ = "1.0.0"