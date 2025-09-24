"""
Domain Validation & Integration (Stream C) for Feature Selection.

This module provides Taiwan market-specific feature validation,
economic intuition scoring, and integration with the LightGBM pipeline.

Stream C Components:
- Taiwan market compliance validation
- Economic intuition and business logic validation
- Information Coefficient performance testing
- LightGBM pipeline integration
- Comprehensive quality assurance
"""

from .taiwan_compliance import TaiwanMarketComplianceValidator
from .economic_intuition import EconomicIntuitionScorer
from .business_logic import BusinessLogicValidator
from .ic_performance_tester import ICPerformanceTester
from .domain_integration_pipeline import DomainValidationPipeline

__all__ = [
    'TaiwanMarketComplianceValidator',
    'EconomicIntuitionScorer', 
    'BusinessLogicValidator',
    'ICPerformanceTester',
    'DomainValidationPipeline'
]

__version__ = "1.0.0"
__author__ = "ML4T Team"
__description__ = "Stream C: Domain Validation & Integration for Taiwan Market Feature Selection"