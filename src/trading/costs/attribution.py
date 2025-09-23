"""
Cost Attribution Framework for Transaction Cost Analysis.

This module provides comprehensive cost attribution and breakdown capabilities
for Taiwan market trading, integrating with existing cost models and performance
attribution systems to provide detailed transaction cost analysis.

Key Features:
- Component-level cost breakdown (taxes, commissions, impact)
- Performance attribution with transaction costs
- Risk-adjusted return calculations with cost adjustments
- Benchmark comparison with cost considerations
- Integration with backtesting and optimization systems
- Real-time cost estimation and analysis
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

from .cost_models import (
    BaseCostModel, TradeCostBreakdown, TradeInfo, CostComponent, 
    TradeDirection, TaiwanCostCalculator, CostModelFactory
)
from .market_impact import (
    TaiwanMarketImpactModel, ImpactCalculationResult, 
    create_taiwan_impact_model
)
try:
    from ...backtesting.metrics.attribution import (
        PerformanceAttributor, AttributionResult, AttributionMethod
    )
except ImportError:
    # Fallback for development
    PerformanceAttributor = Any
    AttributionResult = Any
    AttributionMethod = Any

logger = logging.getLogger(__name__)


class CostAttributionMethod(Enum):
    """Cost attribution calculation methods."""
    COMPONENT_BASED = "component"      # Component-level attribution
    FACTOR_BASED = "factor"           # Factor-based attribution
    IMPACT_BASED = "impact"           # Market impact attribution
    INTEGRATED = "integrated"         # Full integrated attribution


class CostAttributionLevel(Enum):
    """Levels of cost attribution analysis."""
    TRADE = "trade"                   # Individual trade level
    PORTFOLIO = "portfolio"           # Portfolio level
    FACTOR = "factor"                 # Factor level
    BENCHMARK = "benchmark"           # Benchmark comparison


@dataclass
class CostBreakdownAttribution:
    """Detailed cost breakdown with attribution analysis."""
    # Basic cost information
    symbol: str
    trade_date: date
    direction: TradeDirection
    quantity: float
    price: float
    trade_value: float
    
    # Cost component breakdown (in TWD)
    regulatory_costs: Dict[str, float] = field(default_factory=dict)
    market_costs: Dict[str, float] = field(default_factory=dict)
    opportunity_costs: Dict[str, float] = field(default_factory=dict)
    
    # Cost in basis points
    regulatory_costs_bps: Dict[str, float] = field(default_factory=dict)
    market_costs_bps: Dict[str, float] = field(default_factory=dict)
    opportunity_costs_bps: Dict[str, float] = field(default_factory=dict)
    
    # Attribution to performance
    cost_impact_on_return: float = 0.0    # Direct impact on return
    risk_adjusted_cost_impact: float = 0.0  # Risk-adjusted impact
    
    # Benchmark comparison
    vs_benchmark_cost_diff: float = 0.0   # Cost difference vs benchmark
    vs_benchmark_efficiency: float = 0.0  # Cost efficiency ratio
    
    # Metadata
    model_used: str = ""
    confidence_score: float = 0.95
    attribution_method: str = ""
    
    def total_cost_twd(self) -> float:
        """Calculate total cost in TWD."""
        regulatory_total = sum(self.regulatory_costs.values())
        market_total = sum(self.market_costs.values())
        opportunity_total = sum(self.opportunity_costs.values())
        return regulatory_total + market_total + opportunity_total
    
    def total_cost_bps(self) -> float:
        """Calculate total cost in basis points."""
        if self.trade_value <= 0:
            return 0.0
        return (self.total_cost_twd() / self.trade_value) * 10000
    
    def cost_efficiency_score(self) -> float:
        """Calculate cost efficiency score (higher is better)."""
        total_bps = self.total_cost_bps()
        if total_bps <= 0:
            return 1.0
        # Efficiency score: inverse of cost with normalization
        return max(0.0, min(1.0, 1.0 - (total_bps / 100.0)))  # Normalize against 100bps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'symbol': self.symbol,
            'trade_date': self.trade_date.isoformat(),
            'direction': self.direction.value,
            'quantity': self.quantity,
            'price': self.price,
            'trade_value': self.trade_value,
            'regulatory_costs': self.regulatory_costs,
            'market_costs': self.market_costs,
            'opportunity_costs': self.opportunity_costs,
            'regulatory_costs_bps': self.regulatory_costs_bps,
            'market_costs_bps': self.market_costs_bps,
            'opportunity_costs_bps': self.opportunity_costs_bps,
            'total_cost_twd': self.total_cost_twd(),
            'total_cost_bps': self.total_cost_bps(),
            'cost_efficiency_score': self.cost_efficiency_score(),
            'cost_impact_on_return': self.cost_impact_on_return,
            'risk_adjusted_cost_impact': self.risk_adjusted_cost_impact,
            'vs_benchmark_cost_diff': self.vs_benchmark_cost_diff,
            'vs_benchmark_efficiency': self.vs_benchmark_efficiency,
            'model_used': self.model_used,
            'confidence_score': self.confidence_score,
            'attribution_method': self.attribution_method
        }


@dataclass
class PortfolioCostAttribution:
    """Portfolio-level cost attribution analysis."""
    # Portfolio identification
    portfolio_id: str
    period_start: date
    period_end: date
    
    # Trade-level attributions
    trade_attributions: List[CostBreakdownAttribution] = field(default_factory=list)
    
    # Portfolio-level metrics
    total_trade_value: float = 0.0
    total_cost_twd: float = 0.0
    weighted_avg_cost_bps: float = 0.0
    
    # Cost component aggregations
    total_regulatory_costs: Dict[str, float] = field(default_factory=dict)
    total_market_costs: Dict[str, float] = field(default_factory=dict)
    total_opportunity_costs: Dict[str, float] = field(default_factory=dict)
    
    # Performance impact
    total_return_impact: float = 0.0      # Total impact on portfolio return
    risk_adjusted_impact: float = 0.0     # Risk-adjusted impact
    
    # Benchmark comparison
    benchmark_cost_difference: float = 0.0  # Total difference vs benchmark
    relative_cost_efficiency: float = 0.0   # Relative efficiency score
    
    # Factor attribution
    factor_cost_attribution: Dict[str, float] = field(default_factory=dict)
    
    # Statistical measures
    cost_volatility: float = 0.0          # Volatility of costs
    cost_skewness: float = 0.0            # Skewness of cost distribution
    worst_cost_trade_bps: float = 0.0     # Worst single trade cost
    
    # Metadata
    attribution_method: CostAttributionMethod = CostAttributionMethod.INTEGRATED
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self._calculate_portfolio_metrics()
    
    def _calculate_portfolio_metrics(self):
        """Calculate portfolio-level metrics from trade attributions."""
        if not self.trade_attributions:
            return
        
        # Aggregate values
        self.total_trade_value = sum(attr.trade_value for attr in self.trade_attributions)
        self.total_cost_twd = sum(attr.total_cost_twd() for attr in self.trade_attributions)
        
        if self.total_trade_value > 0:
            self.weighted_avg_cost_bps = (self.total_cost_twd / self.total_trade_value) * 10000
        
        # Aggregate cost components
        for attr in self.trade_attributions:
            for component, cost in attr.regulatory_costs.items():
                self.total_regulatory_costs[component] = (
                    self.total_regulatory_costs.get(component, 0.0) + cost
                )
            for component, cost in attr.market_costs.items():
                self.total_market_costs[component] = (
                    self.total_market_costs.get(component, 0.0) + cost
                )
            for component, cost in attr.opportunity_costs.items():
                self.total_opportunity_costs[component] = (
                    self.total_opportunity_costs.get(component, 0.0) + cost
                )
        
        # Performance impact aggregation
        self.total_return_impact = sum(attr.cost_impact_on_return for attr in self.trade_attributions)
        self.risk_adjusted_impact = sum(attr.risk_adjusted_cost_impact for attr in self.trade_attributions)
        
        # Benchmark comparison aggregation
        self.benchmark_cost_difference = sum(attr.vs_benchmark_cost_diff for attr in self.trade_attributions)
        if len(self.trade_attributions) > 0:
            self.relative_cost_efficiency = np.mean([attr.vs_benchmark_efficiency for attr in self.trade_attributions])
        
        # Statistical measures
        cost_bps_list = [attr.total_cost_bps() for attr in self.trade_attributions]
        if cost_bps_list:
            self.cost_volatility = float(np.std(cost_bps_list))
            self.cost_skewness = float(pd.Series(cost_bps_list).skew())
            self.worst_cost_trade_bps = float(np.max(cost_bps_list))
    
    def get_cost_breakdown_summary(self) -> Dict[str, Any]:
        """Get summary of cost breakdown by component."""
        total_regulatory = sum(self.total_regulatory_costs.values())
        total_market = sum(self.total_market_costs.values())
        total_opportunity = sum(self.total_opportunity_costs.values())
        total_cost = total_regulatory + total_market + total_opportunity
        
        return {
            'regulatory_costs': {
                'total_twd': total_regulatory,
                'percentage': (total_regulatory / total_cost * 100) if total_cost > 0 else 0,
                'components': self.total_regulatory_costs
            },
            'market_costs': {
                'total_twd': total_market,
                'percentage': (total_market / total_cost * 100) if total_cost > 0 else 0,
                'components': self.total_market_costs
            },
            'opportunity_costs': {
                'total_twd': total_opportunity,
                'percentage': (total_opportunity / total_cost * 100) if total_cost > 0 else 0,
                'components': self.total_opportunity_costs
            },
            'total_cost_twd': total_cost,
            'total_cost_bps': self.weighted_avg_cost_bps
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'portfolio_id': self.portfolio_id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_trade_value': self.total_trade_value,
            'total_cost_twd': self.total_cost_twd,
            'weighted_avg_cost_bps': self.weighted_avg_cost_bps,
            'cost_breakdown_summary': self.get_cost_breakdown_summary(),
            'total_return_impact': self.total_return_impact,
            'risk_adjusted_impact': self.risk_adjusted_impact,
            'benchmark_cost_difference': self.benchmark_cost_difference,
            'relative_cost_efficiency': self.relative_cost_efficiency,
            'factor_cost_attribution': self.factor_cost_attribution,
            'cost_volatility': self.cost_volatility,
            'cost_skewness': self.cost_skewness,
            'worst_cost_trade_bps': self.worst_cost_trade_bps,
            'trade_count': len(self.trade_attributions),
            'attribution_method': self.attribution_method.value,
            'calculation_timestamp': self.calculation_timestamp.isoformat()
        }


class CostAttributor:
    """
    Main cost attribution engine for transaction cost analysis.
    
    Provides comprehensive cost attribution capabilities including component
    breakdown, performance impact analysis, and benchmark comparison.
    """
    
    def __init__(
        self,
        cost_model: BaseCostModel,
        impact_model: TaiwanMarketImpactModel,
        benchmark_cost_model: Optional[BaseCostModel] = None,
        attribution_method: CostAttributionMethod = CostAttributionMethod.INTEGRATED
    ):
        self.cost_model = cost_model
        self.impact_model = impact_model
        self.benchmark_cost_model = benchmark_cost_model or CostModelFactory.create_conservative_model()
        self.attribution_method = attribution_method
        
        # Taiwan cost calculator for regulatory costs
        self.taiwan_calculator = TaiwanCostCalculator()
        
        logger.info(f"CostAttributor initialized with method: {attribution_method.value}")
    
    def attribute_trade_costs(
        self,
        trade: TradeInfo,
        market_data: Optional[Dict[str, Any]] = None,
        benchmark_comparison: bool = True,
        performance_context: Optional[Dict[str, Any]] = None
    ) -> CostBreakdownAttribution:
        """
        Perform comprehensive cost attribution for a single trade.
        
        Args:
            trade: Trade information
            market_data: Additional market context
            benchmark_comparison: Whether to compare against benchmark
            performance_context: Portfolio performance context
            
        Returns:
            Detailed cost attribution breakdown
        """
        logger.debug(f"Attributing costs for {trade.symbol} trade")
        
        # Calculate base cost breakdown
        cost_breakdown = self.cost_model.calculate_cost(trade)
        
        # Calculate market impact
        impact_result = self.impact_model.calculate_impact(
            symbol=trade.symbol,
            order_size=trade.quantity,
            price=trade.price,
            avg_daily_volume=trade.daily_volume or 100000,  # Default volume
            volatility=trade.volatility or 0.25,
            timestamp=datetime.combine(trade.trade_date, datetime.min.time()),
            market_data=market_data
        )
        
        # Create attribution result
        attribution = CostBreakdownAttribution(
            symbol=trade.symbol,
            trade_date=trade.trade_date,
            direction=trade.direction,
            quantity=trade.quantity,
            price=trade.price,
            trade_value=trade.trade_value,
            model_used=self.cost_model.name,
            attribution_method=self.attribution_method.value
        )
        
        # Populate regulatory costs
        attribution.regulatory_costs = {
            'commission': cost_breakdown.commission,
            'transaction_tax': cost_breakdown.transaction_tax,
            'exchange_fee': cost_breakdown.exchange_fee,
            'settlement_fee': cost_breakdown.settlement_fee,
            'custody_fee': cost_breakdown.custody_fee
        }
        
        # Convert to basis points
        if trade.trade_value > 0:
            for component, cost in attribution.regulatory_costs.items():
                attribution.regulatory_costs_bps[component] = (cost / trade.trade_value) * 10000
        
        # Populate market costs
        attribution.market_costs = {
            'market_impact': impact_result.total_impact_twd,
            'bid_ask_spread': cost_breakdown.bid_ask_spread_cost,
            'timing_cost': cost_breakdown.timing_cost,
            'slippage': cost_breakdown.slippage
        }
        
        # Convert to basis points
        if trade.trade_value > 0:
            for component, cost in attribution.market_costs.items():
                attribution.market_costs_bps[component] = (cost / trade.trade_value) * 10000
        
        # Calculate opportunity costs
        attribution.opportunity_costs = self._calculate_opportunity_costs(
            trade, impact_result, market_data
        )
        
        if trade.trade_value > 0:
            for component, cost in attribution.opportunity_costs.items():
                attribution.opportunity_costs_bps[component] = (cost / trade.trade_value) * 10000
        
        # Calculate performance impact
        attribution.cost_impact_on_return = self._calculate_performance_impact(
            attribution, performance_context
        )
        
        attribution.risk_adjusted_cost_impact = self._calculate_risk_adjusted_impact(
            attribution, performance_context
        )
        
        # Benchmark comparison
        if benchmark_comparison:
            benchmark_costs = self._calculate_benchmark_comparison(trade, attribution)
            attribution.vs_benchmark_cost_diff = benchmark_costs['cost_difference']
            attribution.vs_benchmark_efficiency = benchmark_costs['efficiency_ratio']
        
        # Set confidence score based on data quality
        attribution.confidence_score = self._calculate_confidence_score(trade, market_data)
        
        logger.debug(f"Cost attribution completed. Total cost: {attribution.total_cost_bps():.2f} bps")
        return attribution
    
    def attribute_portfolio_costs(
        self,
        trades: List[TradeInfo],
        portfolio_id: str,
        period_start: date,
        period_end: date,
        performance_data: Optional[Dict[str, Any]] = None,
        factor_exposures: Optional[Dict[str, float]] = None
    ) -> PortfolioCostAttribution:
        """
        Perform portfolio-level cost attribution analysis.
        
        Args:
            trades: List of trades to analyze
            portfolio_id: Portfolio identifier
            period_start: Analysis period start
            period_end: Analysis period end
            performance_data: Portfolio performance context
            factor_exposures: Factor exposure data
            
        Returns:
            Portfolio cost attribution analysis
        """
        logger.info(f"Starting portfolio cost attribution for {len(trades)} trades")
        
        # Calculate individual trade attributions
        trade_attributions = []
        for trade in trades:
            try:
                attribution = self.attribute_trade_costs(
                    trade, 
                    performance_context=performance_data
                )
                trade_attributions.append(attribution)
            except Exception as e:
                logger.warning(f"Failed to attribute costs for trade {trade.symbol}: {e}")
        
        # Create portfolio attribution
        portfolio_attribution = PortfolioCostAttribution(
            portfolio_id=portfolio_id,
            period_start=period_start,
            period_end=period_end,
            trade_attributions=trade_attributions,
            attribution_method=self.attribution_method
        )
        
        # Calculate factor attribution if factor exposures provided
        if factor_exposures:
            portfolio_attribution.factor_cost_attribution = self._calculate_factor_cost_attribution(
                trade_attributions, factor_exposures
            )
        
        logger.info(f"Portfolio cost attribution completed. Total cost: {portfolio_attribution.weighted_avg_cost_bps:.2f} bps")
        return portfolio_attribution
    
    def _calculate_opportunity_costs(
        self,
        trade: TradeInfo,
        impact_result: ImpactCalculationResult,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate opportunity costs from delayed or suboptimal execution."""
        opportunity_costs = {}
        
        # Execution delay cost
        if trade.execution_delay_seconds and trade.execution_delay_seconds > 0:
            # Use volatility-based delay cost estimation
            delay_minutes = trade.execution_delay_seconds / 60
            vol_annual = trade.volatility or 0.25
            vol_minute = vol_annual * np.sqrt(1 / (252 * 390))  # Annual to minute volatility
            
            delay_cost_bps = delay_minutes * vol_minute * 10000 * 0.2  # 20% of vol impact
            delay_cost_twd = (delay_cost_bps / 10000) * trade.trade_value
            opportunity_costs['execution_delay'] = delay_cost_twd
        else:
            opportunity_costs['execution_delay'] = 0.0
        
        # Missed alpha opportunity (if trade was delayed)
        if market_data and 'expected_alpha' in market_data:
            expected_alpha = market_data['expected_alpha']
            if abs(expected_alpha) > 0.001:  # > 10bps expected alpha
                alpha_cost = abs(expected_alpha) * trade.trade_value
                opportunity_costs['missed_alpha'] = alpha_cost
            else:
                opportunity_costs['missed_alpha'] = 0.0
        else:
            opportunity_costs['missed_alpha'] = 0.0
        
        # Capacity constraint cost (if order size was limited)
        if trade.order_size_vs_avg and trade.order_size_vs_avg > 0.1:  # > 10% of ADV
            # Capacity constraint increased impact
            excess_impact = impact_result.total_impact_twd * 0.5  # 50% penalty for large orders
            opportunity_costs['capacity_constraint'] = excess_impact
        else:
            opportunity_costs['capacity_constraint'] = 0.0
        
        return opportunity_costs
    
    def _calculate_performance_impact(
        self,
        attribution: CostBreakdownAttribution,
        performance_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate direct impact of costs on portfolio performance."""
        total_cost = attribution.total_cost_twd()
        
        # Direct impact is negative of total costs as percentage of trade value
        if attribution.trade_value > 0:
            direct_impact = -total_cost / attribution.trade_value
        else:
            direct_impact = 0.0
        
        # Adjust for portfolio context if available
        if performance_context and 'portfolio_value' in performance_context:
            portfolio_value = performance_context['portfolio_value']
            if portfolio_value > 0:
                # Scale by portfolio weight
                portfolio_impact = (total_cost / portfolio_value)
                return -portfolio_impact  # Negative because costs reduce returns
        
        return direct_impact
    
    def _calculate_risk_adjusted_impact(
        self,
        attribution: CostBreakdownAttribution,
        performance_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate risk-adjusted impact of costs."""
        direct_impact = attribution.cost_impact_on_return
        
        # Adjust for risk if volatility information available
        if performance_context and 'portfolio_volatility' in performance_context:
            portfolio_vol = performance_context['portfolio_volatility']
            if portfolio_vol > 0:
                # Risk-adjusted impact (Sharpe-like adjustment)
                risk_adjusted = direct_impact / portfolio_vol
                return risk_adjusted
        
        return direct_impact
    
    def _calculate_benchmark_comparison(
        self,
        trade: TradeInfo,
        attribution: CostBreakdownAttribution
    ) -> Dict[str, float]:
        """Calculate benchmark cost comparison."""
        try:
            # Calculate benchmark costs using conservative model
            benchmark_cost = self.benchmark_cost_model.calculate_cost(trade)
            
            # Cost difference
            actual_cost = attribution.total_cost_twd()
            benchmark_cost_total = benchmark_cost.total_cost
            cost_difference = actual_cost - benchmark_cost_total
            
            # Efficiency ratio
            if benchmark_cost_total > 0:
                efficiency_ratio = benchmark_cost_total / actual_cost
            else:
                efficiency_ratio = 1.0
            
            return {
                'cost_difference': cost_difference,
                'efficiency_ratio': efficiency_ratio
            }
            
        except Exception as e:
            logger.debug(f"Benchmark comparison failed: {e}")
            return {'cost_difference': 0.0, 'efficiency_ratio': 1.0}
    
    def _calculate_factor_cost_attribution(
        self,
        trade_attributions: List[CostBreakdownAttribution],
        factor_exposures: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate factor-based cost attribution."""
        factor_attribution = {}
        
        # Group trades by cost characteristics
        total_cost = sum(attr.total_cost_twd() for attr in trade_attributions)
        
        if total_cost <= 0:
            return factor_attribution
        
        # Attribute costs to factors based on exposures
        for factor_name, exposure in factor_exposures.items():
            if abs(exposure) > 0.01:  # Only consider significant exposures
                # Simplified factor attribution
                factor_cost_contribution = total_cost * abs(exposure) * 0.1  # 10% factor impact
                factor_attribution[factor_name] = factor_cost_contribution
        
        return factor_attribution
    
    def _calculate_confidence_score(
        self,
        trade: TradeInfo,
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for cost attribution."""
        confidence = 0.95  # Base confidence
        
        # Reduce confidence for missing data
        if trade.daily_volume is None:
            confidence -= 0.1
        if trade.volatility is None:
            confidence -= 0.1
        if trade.bid_ask_spread is None:
            confidence -= 0.05
        if market_data is None:
            confidence -= 0.05
        
        # Reduce confidence for unusual conditions
        if trade.order_size_vs_avg and trade.order_size_vs_avg > 0.2:  # Very large order
            confidence -= 0.1
        
        return max(0.5, confidence)  # Minimum 50% confidence


class CostAttributionReporter:
    """Generate reports and visualizations for cost attribution analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CostAttributionReporter")
    
    def generate_trade_report(
        self,
        attribution: CostBreakdownAttribution
    ) -> Dict[str, Any]:
        """Generate detailed report for a single trade attribution."""
        return {
            'trade_summary': {
                'symbol': attribution.symbol,
                'trade_date': attribution.trade_date.isoformat(),
                'direction': attribution.direction.value,
                'quantity': attribution.quantity,
                'price': attribution.price,
                'trade_value': attribution.trade_value
            },
            'cost_breakdown': {
                'total_cost_twd': attribution.total_cost_twd(),
                'total_cost_bps': attribution.total_cost_bps(),
                'regulatory_costs': attribution.regulatory_costs,
                'market_costs': attribution.market_costs,
                'opportunity_costs': attribution.opportunity_costs
            },
            'cost_breakdown_bps': {
                'regulatory_costs_bps': attribution.regulatory_costs_bps,
                'market_costs_bps': attribution.market_costs_bps,
                'opportunity_costs_bps': attribution.opportunity_costs_bps
            },
            'performance_impact': {
                'cost_impact_on_return': attribution.cost_impact_on_return,
                'risk_adjusted_cost_impact': attribution.risk_adjusted_cost_impact
            },
            'benchmark_comparison': {
                'vs_benchmark_cost_diff': attribution.vs_benchmark_cost_diff,
                'vs_benchmark_efficiency': attribution.vs_benchmark_efficiency
            },
            'quality_metrics': {
                'cost_efficiency_score': attribution.cost_efficiency_score(),
                'confidence_score': attribution.confidence_score,
                'model_used': attribution.model_used
            }
        }
    
    def generate_portfolio_report(
        self,
        portfolio_attribution: PortfolioCostAttribution
    ) -> Dict[str, Any]:
        """Generate comprehensive portfolio cost attribution report."""
        cost_breakdown = portfolio_attribution.get_cost_breakdown_summary()
        
        return {
            'portfolio_summary': {
                'portfolio_id': portfolio_attribution.portfolio_id,
                'period_start': portfolio_attribution.period_start.isoformat(),
                'period_end': portfolio_attribution.period_end.isoformat(),
                'trade_count': len(portfolio_attribution.trade_attributions),
                'total_trade_value': portfolio_attribution.total_trade_value,
                'total_cost_twd': portfolio_attribution.total_cost_twd,
                'weighted_avg_cost_bps': portfolio_attribution.weighted_avg_cost_bps
            },
            'cost_breakdown': cost_breakdown,
            'performance_impact': {
                'total_return_impact': portfolio_attribution.total_return_impact,
                'risk_adjusted_impact': portfolio_attribution.risk_adjusted_impact,
                'benchmark_cost_difference': portfolio_attribution.benchmark_cost_difference,
                'relative_cost_efficiency': portfolio_attribution.relative_cost_efficiency
            },
            'risk_metrics': {
                'cost_volatility': portfolio_attribution.cost_volatility,
                'cost_skewness': portfolio_attribution.cost_skewness,
                'worst_cost_trade_bps': portfolio_attribution.worst_cost_trade_bps
            },
            'factor_attribution': portfolio_attribution.factor_cost_attribution,
            'trade_details': [attr.to_dict() for attr in portfolio_attribution.trade_attributions]
        }


# Factory functions for common use cases
def create_taiwan_cost_attributor(
    conservative_benchmark: bool = False,
    custom_models: Optional[Dict[str, Any]] = None
) -> CostAttributor:
    """
    Create cost attributor for Taiwan market with default models.
    
    Args:
        conservative_benchmark: Use conservative models for benchmark
        custom_models: Custom model configurations
        
    Returns:
        Configured cost attributor for Taiwan market
    """
    # Create cost models
    if custom_models and 'cost_model' in custom_models:
        cost_model = custom_models['cost_model']
    else:
        cost_model = CostModelFactory.create_nonlinear_model()
    
    # Create impact model
    if custom_models and 'impact_model' in custom_models:
        impact_model = custom_models['impact_model']
    else:
        impact_model = create_taiwan_impact_model()
    
    # Create benchmark model
    if conservative_benchmark:
        benchmark_model = CostModelFactory.create_conservative_model()
    else:
        benchmark_model = CostModelFactory.create_linear_model()
    
    return CostAttributor(
        cost_model=cost_model,
        impact_model=impact_model,
        benchmark_cost_model=benchmark_model,
        attribution_method=CostAttributionMethod.INTEGRATED
    )


# Example usage and testing
if __name__ == "__main__":
    print("Cost Attribution Framework Demo")
    
    # Create cost attributor
    attributor = create_taiwan_cost_attributor()
    
    # Sample trade
    sample_trade = TradeInfo(
        symbol="2330.TW",
        trade_date=date.today(),
        direction=TradeDirection.BUY,
        quantity=5000,
        price=500.0,
        daily_volume=500000,
        volatility=0.25,
        bid_ask_spread=0.1,
        order_size_vs_avg=0.01,
        execution_delay_seconds=30
    )
    
    # Perform cost attribution
    attribution = attributor.attribute_trade_costs(sample_trade)
    
    # Generate report
    reporter = CostAttributionReporter()
    report = reporter.generate_trade_report(attribution)
    
    print(f"\nCost Attribution for {attribution.symbol}:")
    print(f"Total Cost: {attribution.total_cost_bps():.2f} bps")
    print(f"Cost Efficiency Score: {attribution.cost_efficiency_score():.3f}")
    print(f"Performance Impact: {attribution.cost_impact_on_return:.6f}")
    print(f"Confidence Score: {attribution.confidence_score:.2f}")
    
    print("\nCost Breakdown (bps):")
    for category, costs in [
        ("Regulatory", attribution.regulatory_costs_bps),
        ("Market", attribution.market_costs_bps),
        ("Opportunity", attribution.opportunity_costs_bps)
    ]:
        print(f"  {category}:")
        for component, cost_bps in costs.items():
            print(f"    {component}: {cost_bps:.2f} bps")