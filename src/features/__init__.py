"""
ML4T Features Module

OpenFE integration for automated feature engineering with Taiwan market specifics.
Expert-validated time-series handling to prevent lookahead bias.

Stream C additions:
- Feature selection algorithms with correlation filtering and importance ranking
- Quality assessment metrics with statistical validation  
- Taiwan market compliance validation for generated features
"""

__version__ = "1.0.0"

# Stream A: Foundation (OpenFE wrapper and Taiwan config)
from .openfe_wrapper import FeatureGenerator
from .taiwan_config import TaiwanMarketConfig

# Stream C: Feature Selection and Quality Assessment
from .feature_selection import FeatureSelector, create_feature_selector
from .quality_metrics import FeatureQualityMetrics, create_quality_metrics_calculator
from .taiwan_compliance import TaiwanComplianceValidator, create_taiwan_compliance_validator

__all__ = [
    # Stream A exports
    'FeatureGenerator', 
    'TaiwanMarketConfig',
    
    # Stream C exports  
    'FeatureSelector', 
    'create_feature_selector',
    'FeatureQualityMetrics',
    'create_quality_metrics_calculator', 
    'TaiwanComplianceValidator',
    'create_taiwan_compliance_validator'
]