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
    
    # Microstructure factors (Stream C - newly implemented)
    from .microstructure import MicrostructureFactorCalculator, MicrostructureFactors, TaiwanMarketSession, TickSizeStructure
    from .microstructure import ForeignFlowData, MarginTradingData, IndexCompositionData, CrossStraitSentiment
    
    # Liquidity factors
    from .liquidity import LiquidityFactors, AverageDailyTurnoverCalculator, BidAskSpreadCalculator
    from .liquidity import PriceImpactCalculator, AmihudIlliquidityCalculator
    
    # Volume pattern factors
    from .volume_patterns import VolumePatternFactors, VolumeWeightedMomentumCalculator, VolumeBreakoutCalculator
    from .volume_patterns import RelativeVolumeCalculator, VolumePriceCorrelationCalculator
    
    # Taiwan-specific factors
    from .taiwan_specific import TaiwanSpecificFactors, ForeignFlowImpactCalculator, MarginTradingRatioCalculator
    from .taiwan_specific import IndexInclusionEffectCalculator, CrossStraitSentimentCalculator, TaiwanMarketStructure
    
    # Supporting modules
    from .tick_data_handler import TickDataHandler, TickData, TickDataCleaner, IntradayMetricsCalculator
    from .tick_data_handler import TaiwanSessionFilter, IntradayMetrics, TickDataAggregator
    from .foreign_flows import ForeignFlowAnalyzer, FlowAnalysisResult, FlowDirection, FlowIntensity
    from .foreign_flows import ForeignInstitutionProfile, ForeignFlowRegimeDetector, ForeignFlowForecaster

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
        
        # Fundamental factors (Stream B - already implemented)
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
        
        # Microstructure factors (Stream C - newly implemented)
        'MicrostructureFactorCalculator',
        'MicrostructureFactors',
        'TaiwanMarketSession',
        'TickSizeStructure',
        'ForeignFlowData',
        'MarginTradingData', 
        'IndexCompositionData',
        'CrossStraitSentiment',
        
        # Liquidity factors
        'LiquidityFactors',
        'AverageDailyTurnoverCalculator',
        'BidAskSpreadCalculator',
        'PriceImpactCalculator',
        'AmihudIlliquidityCalculator',
        
        # Volume pattern factors
        'VolumePatternFactors',
        'VolumeWeightedMomentumCalculator',
        'VolumeBreakoutCalculator',
        'RelativeVolumeCalculator',
        'VolumePriceCorrelationCalculator',
        
        # Taiwan-specific factors
        'TaiwanSpecificFactors',
        'ForeignFlowImpactCalculator',
        'MarginTradingRatioCalculator',
        'IndexInclusionEffectCalculator',
        'CrossStraitSentimentCalculator',
        'TaiwanMarketStructure',
        
        # Supporting modules
        'TickDataHandler',
        'TickData',
        'TickDataCleaner',
        'IntradayMetricsCalculator',
        'TaiwanSessionFilter',
        'IntradayMetrics',
        'TickDataAggregator',
        'ForeignFlowAnalyzer',
        'FlowAnalysisResult',
        'FlowDirection',
        'FlowIntensity',
        'ForeignInstitutionProfile',
        'ForeignFlowRegimeDetector',
        'ForeignFlowForecaster',
        
        # Taiwan-specific components (original)
        'TaiwanMarketAdjustments',
        'TaiwanFinancialDataHandler',
        'TaiwanFinancialStatement',
        'TaiwanFinancialMetadata'
    ]
except ImportError as e:
    # Graceful degradation for testing or missing dependencies
    print(f"Warning: Some factor modules could not be imported: {e}")
    __all__ = []