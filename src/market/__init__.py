"""
Market Analysis Module - Task #005
GitHub Issue #78

Market analysis components for Taiwan market regime detection and pattern analysis.
Implements statistically rigorous regime classification with confidence scoring.

Key Components:
1. TaiwanMarketRegimeDetector: Core regime detection with statistical validation
2. RegimeTransitionAnalyzer: Regime change detection and persistence modeling
3. MarketVolatilityAnalyzer: Taiwan-specific volatility pattern analysis
4. CorrelationBreakdownDetector: Cross-asset correlation monitoring

Integration Context:
- Provides regime signals for Task 004 factor combination strategies
- Addresses statistical rigor concerns identified in Task 002
- Optimized for Taiwan market characteristics and constraints
- Supports real-time regime classification for dynamic factor allocation
"""

from .regime_detection import (
    TaiwanMarketRegimeDetector,
    TaiwanMarketRegime,
    RegimeConfidenceScore,
    RegimeClassification,
    RegimeTransitionEvent,
    create_taiwan_regime_detector
)

__all__ = [
    'TaiwanMarketRegimeDetector',
    'TaiwanMarketRegime',
    'RegimeConfidenceScore',
    'RegimeClassification',
    'RegimeTransitionEvent',
    'create_taiwan_regime_detector'
]