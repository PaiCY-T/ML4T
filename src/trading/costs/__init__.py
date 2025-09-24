"""
Transaction cost modeling for Taiwan market.

This package provides comprehensive transaction cost analysis including:
- Linear and non-linear cost models
- Taiwan market microstructure modeling
- Execution cost simulation
- Market impact analysis
- Cost attribution and optimization
- Real-time cost estimation
- Backtesting integration
"""

from .cost_models import (
    BaseCostModel,
    LinearCostModel, 
    NonLinearCostModel,
    TaiwanCostCalculator,
    TradeInfo,
    TradeDirection,
    TradeCostBreakdown,
    CostModelFactory
)

from .market_impact import (
    TaiwanMarketImpactModel,
    MarketImpactParameters,
    ImpactCalculationResult,
    create_taiwan_impact_model
)

from .attribution import (
    CostAttributor,
    CostBreakdownAttribution,
    PortfolioCostAttribution,
    CostAttributionMethod,
    create_taiwan_cost_attributor
)

from .integration import (
    RealTimeCostEstimator,
    CostEstimationRequest,
    CostEstimationResponse,
    PortfolioRebalancingAnalyzer,
    BacktestingCostIntegrator,
    create_taiwan_backtesting_integration
)

from .optimization import (
    CostOptimizationEngine,
    OptimizationResult,
    OptimizationObjective,
    ExecutionStrategy,
    OptimizationConstraints,
    create_taiwan_cost_optimization_system
)

# Legacy imports (if they exist)
try:
    from .taiwan_microstructure import (
        TaiwanMarketStructure,
        BidAskSpreadModel,
        TaiwanTickSizeModel
    )
except ImportError:
    pass

try:
    from .execution_models import (
        ExecutionCostSimulator,
        SlippageModel,
        TimingCostModel,
        SettlementCostModel
    )
except ImportError:
    pass

__version__ = "1.1.0"
__all__ = [
    # Core cost models
    'BaseCostModel',
    'LinearCostModel',
    'NonLinearCostModel', 
    'TaiwanCostCalculator',
    'TradeInfo',
    'TradeDirection',
    'TradeCostBreakdown',
    'CostModelFactory',
    
    # Market impact
    'TaiwanMarketImpactModel',
    'MarketImpactParameters',
    'ImpactCalculationResult',
    'create_taiwan_impact_model',
    
    # Cost attribution
    'CostAttributor',
    'CostBreakdownAttribution',
    'PortfolioCostAttribution',
    'CostAttributionMethod',
    'create_taiwan_cost_attributor',
    
    # Integration
    'RealTimeCostEstimator',
    'CostEstimationRequest',
    'CostEstimationResponse',
    'PortfolioRebalancingAnalyzer',
    'BacktestingCostIntegrator',
    'create_taiwan_backtesting_integration',
    
    # Optimization
    'CostOptimizationEngine',
    'OptimizationResult',
    'OptimizationObjective',
    'ExecutionStrategy',
    'OptimizationConstraints',
    'create_taiwan_cost_optimization_system'
]