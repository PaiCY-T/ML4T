"""
Factor calculation module for Taiwan market ML pipeline.

This module provides comprehensive factor calculation capabilities
for technical, fundamental, and market microstructure factors
tailored to the Taiwan market.
"""

try:
    from .base import FactorEngine, FactorCalculator, FactorResult, FactorMetadata, FactorCategory, FactorFrequency
    from .technical import TechnicalFactors
    from .taiwan_adjustments import TaiwanMarketAdjustments
    
    # Fundamental factors
    from .fundamental import FundamentalFactorCalculator, FundamentalFactors, FinancialStatement, MarketData
    from .value import ValueFactors, PERatioCalculator, PBRatioCalculator, EVEBITDACalculator, PriceSalesCalculator
    from .quality import QualityFactors, ROEROACalculator, DebtEquityCalculator, OperatingMarginCalculator, EarningsQualityCalculator
    from .growth import GrowthFactors, RevenueGrowthCalculator, EarningsGrowthCalculator, BookValueGrowthCalculator, AnalystRevisionCalculator
    from .taiwan_financials import TaiwanFinancialDataHandler, TaiwanFinancialStatement, TaiwanFinancialMetadata

    __all__ = [
        # Base classes
        'FactorEngine',
        'FactorCalculator', 
        'FactorResult',
        'FactorMetadata',
        'FactorCategory', 
        'FactorFrequency',
        
        # Technical factors (Stream A - already implemented)
        'TechnicalFactors',
        
        # Fundamental factors (Stream B - newly implemented)
        'FundamentalFactorCalculator',
        'FundamentalFactors',
        'FinancialStatement',
        'MarketData',
        
        # Value factors
        'ValueFactors',
        'PERatioCalculator',
        'PBRatioCalculator',
        'EVEBITDACalculator',
        'PriceSalesCalculator',
        
        # Quality factors
        'QualityFactors',
        'ROEROACalculator',
        'DebtEquityCalculator',
        'OperatingMarginCalculator',
        'EarningsQualityCalculator',
        
        # Growth factors
        'GrowthFactors',
        'RevenueGrowthCalculator',
        'EarningsGrowthCalculator',
        'BookValueGrowthCalculator',
        'AnalystRevisionCalculator',
        
        # Taiwan-specific components
        'TaiwanMarketAdjustments',
        'TaiwanFinancialDataHandler',
        'TaiwanFinancialStatement',
        'TaiwanFinancialMetadata'
    ]
except ImportError as e:
    # Graceful degradation for testing or missing dependencies
    print(f"Warning: Some factor modules could not be imported: {e}")
    __all__ = []