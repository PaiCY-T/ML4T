"""
Trading strategies for monthly rebalancing.

This package provides factor-based trading strategies optimized for
Taiwan market characteristics and monthly rebalancing cycles.
"""

# Import only if modules are available to avoid dependency issues
try:
    from .monthly_rebalance import MonthlyFactorStrategy, FactorSignals, PortfolioConstruction
    MONTHLY_REBALANCE_AVAILABLE = True
except ImportError:
    MONTHLY_REBALANCE_AVAILABLE = False

try:
    from .factor_combination import (
        FactorCombinationStrategy, EqualWeightStrategy, SmartBetaStrategy,
        FactorPortfolioConstructor, FactorCombinationMethod, PortfolioObjective,
        create_equal_weight_strategy, create_smart_beta_strategy,
        create_portfolio_constructor
    )
    FACTOR_COMBINATION_AVAILABLE = True
except ImportError:
    FACTOR_COMBINATION_AVAILABLE = False

# Define available exports based on successful imports
__all__ = []

if MONTHLY_REBALANCE_AVAILABLE:
    __all__.extend([
        'MonthlyFactorStrategy',
        'FactorSignals',
        'PortfolioConstruction'
    ])

if FACTOR_COMBINATION_AVAILABLE:
    __all__.extend([
        'FactorCombinationStrategy',
        'EqualWeightStrategy',
        'SmartBetaStrategy',
        'FactorPortfolioConstructor',
        'FactorCombinationMethod',
        'PortfolioObjective',
        'create_equal_weight_strategy',
        'create_smart_beta_strategy',
        'create_portfolio_constructor'
    ])