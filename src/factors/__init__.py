"""
Factor calculation module for Taiwan market ML pipeline.

This module provides comprehensive factor calculation capabilities
for technical, fundamental, and market microstructure factors
tailored to the Taiwan market.
"""

try:
    from .base import FactorEngine, FactorCalculator, FactorResult
    from .technical import TechnicalFactors
    from .taiwan_adjustments import TaiwanMarketAdjustments

    __all__ = [
        'FactorEngine',
        'FactorCalculator', 
        'FactorResult',
        'TechnicalFactors',
        'TaiwanMarketAdjustments'
    ]
except ImportError as e:
    # Graceful degradation for testing or missing dependencies
    print(f"Warning: Some factor modules could not be imported: {e}")
    __all__ = []