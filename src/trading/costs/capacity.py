"""
Strategy capacity modeling based on market impact and liquidity constraints.

This module provides comprehensive capacity analysis for trading strategies,
integrating market impact models with liquidity constraints to determine
optimal position sizes and strategy limits for Taiwan equity markets.

Key Features:
- Strategy-level capacity modeling with impact thresholds
- Cross-asset capacity allocation and optimization
- Dynamic capacity adjustment based on market conditions
- Multi-timeframe capacity analysis (intraday to monthly)
- Risk-adjusted capacity limits with stress testing
- Integration with portfolio optimization systems
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from .market_impact import TaiwanMarketImpactModel, ImpactCalculationResult, create_taiwan_impact_model
from .liquidity import LiquidityAnalyzer, LiquidityMetrics, CapacityConstraints, create_liquidity_analyzer

logger = logging.getLogger(__name__)


class CapacityType(Enum):
    """Types of capacity analysis."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"


class CapacityRegime(Enum):
    """Capacity regimes based on market conditions."""
    NORMAL = "normal"
    CONSTRAINED = "constrained"
    SEVERELY_CONSTRAINED = "severely_constrained"
    UNCONSTRAINED = "unconstrained"


@dataclass
class StrategyCapacityParameters:
    """Parameters for strategy capacity modeling."""
    
    # Impact thresholds
    max_impact_bps: float = 50.0          # Maximum acceptable total impact
    target_impact_bps: float = 30.0       # Target impact level
    impact_budget_daily_bps: float = 20.0  # Daily impact budget
    
    # Risk management
    stress_test_multiplier: float = 1.5    # Stress test capacity reduction
    confidence_level: float = 0.95        # Confidence level for capacity estimates
    diversification_benefit: float = 0.8   # Benefit from trading multiple stocks
    
    # Temporal constraints
    min_holding_period_days: int = 1       # Minimum holding period
    max_holding_period_days: int = 30      # Maximum holding period
    rebalancing_frequency_days: int = 7    # Rebalancing frequency
    
    # Portfolio constraints
    max_concentration_pct: float = 0.20    # Max position as % of portfolio
    max_turnover_annual: float = 2.0       # Maximum annual turnover
    
    # Market microstructure
    market_hours_per_day: float = 4.5      # Taiwan market hours (09:00-13:30)
    trading_days_per_year: int = 252       # Trading days per year


@dataclass
class CapacityAnalysisResult:
    """Result of strategy capacity analysis."""
    strategy_name: str
    analysis_date: date
    capacity_type: CapacityType
    
    # Capacity limits
    max_position_twd: float
    max_position_shares: Dict[str, float]
    max_daily_turnover_twd: float
    max_portfolio_size_twd: float
    
    # Impact analysis
    estimated_impact_bps: float
    impact_budget_remaining_bps: float
    impact_utilization_pct: float
    
    # Risk metrics
    capacity_utilization_pct: float
    stress_test_capacity_twd: float
    confidence_interval: Tuple[float, float]
    
    # Constraints binding
    binding_constraints: List[str]
    capacity_regime: CapacityRegime
    
    # Recommendations
    recommended_max_position_pct: float
    recommended_rebalancing_frequency: int
    optimization_suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'strategy_name': self.strategy_name,
            'analysis_date': self.analysis_date.isoformat(),
            'capacity_type': self.capacity_type.value,
            'max_position_twd': self.max_position_twd,
            'max_position_shares': self.max_position_shares,
            'max_daily_turnover_twd': self.max_daily_turnover_twd,
            'max_portfolio_size_twd': self.max_portfolio_size_twd,
            'estimated_impact_bps': self.estimated_impact_bps,
            'impact_budget_remaining_bps': self.impact_budget_remaining_bps,
            'impact_utilization_pct': self.impact_utilization_pct,
            'capacity_utilization_pct': self.capacity_utilization_pct,
            'stress_test_capacity_twd': self.stress_test_capacity_twd,
            'confidence_interval': self.confidence_interval,
            'binding_constraints': self.binding_constraints,
            'capacity_regime': self.capacity_regime.value,
            'recommended_max_position_pct': self.recommended_max_position_pct,
            'recommended_rebalancing_frequency': self.recommended_rebalancing_frequency,
            'optimization_suggestions': self.optimization_suggestions
        }


@dataclass
class PortfolioCapacityAllocation:
    """Capacity allocation across portfolio positions."""
    total_capacity_twd: float
    allocations: Dict[str, Dict[str, float]]  # symbol -> {capacity_twd, weight, impact_bps}
    diversification_benefit_twd: float
    utilization_pct: float
    
    # Risk metrics
    concentration_risk_score: float
    liquidity_risk_score: float
    impact_risk_score: float
    
    # Optimization metrics
    efficiency_score: float  # Capacity utilization efficiency
    risk_adjusted_capacity: float


class StrategyCapacityAnalyzer:
    """
    Comprehensive strategy capacity analyzer.
    
    Integrates market impact models and liquidity analysis to determine
    optimal strategy capacity and position limits.
    """
    
    def __init__(
        self,
        impact_model: Optional[TaiwanMarketImpactModel] = None,
        liquidity_analyzer: Optional[LiquidityAnalyzer] = None,
        parameters: Optional[StrategyCapacityParameters] = None
    ):
        """
        Initialize capacity analyzer.
        
        Args:
            impact_model: Market impact model
            liquidity_analyzer: Liquidity analyzer
            parameters: Capacity analysis parameters
        """
        self.impact_model = impact_model or create_taiwan_impact_model()
        self.liquidity_analyzer = liquidity_analyzer or create_liquidity_analyzer()
        self.parameters = parameters or StrategyCapacityParameters()
        self.logger = logging.getLogger(f"{__name__}.StrategyCapacityAnalyzer")
        
        # Cache for performance
        self._capacity_cache = {}
        self._impact_cache = {}
    
    def analyze_strategy_capacity(
        self,
        strategy_name: str,
        universe: List[str],
        market_data: Dict[str, Dict[str, Any]],
        current_positions: Optional[Dict[str, float]] = None,
        capacity_type: CapacityType = CapacityType.STRATEGY
    ) -> CapacityAnalysisResult:
        """
        Analyze capacity for a trading strategy.
        
        Args:
            strategy_name: Name of the strategy
            universe: List of symbols in trading universe
            market_data: Market data for each symbol
            current_positions: Current positions (optional)
            capacity_type: Type of capacity analysis
            
        Returns:
            CapacityAnalysisResult with detailed analysis
        """
        if current_positions is None:
            current_positions = {}
        
        # Calculate individual stock capacities
        stock_capacities = {}
        total_liquidity_impact = 0.0
        binding_constraints = []
        
        for symbol in universe:
            symbol_data = market_data.get(symbol, {})
            if not symbol_data:
                continue
            
            # Calculate liquidity metrics
            try:
                liquidity_metrics = self._get_liquidity_metrics(symbol, symbol_data)
                capacity_constraints = self.liquidity_analyzer.calculate_capacity_constraints(
                    symbol=symbol,
                    liquidity_metrics=liquidity_metrics,
                    strategy_horizon_days=self.parameters.rebalancing_frequency_days,
                    max_impact_bps=self.parameters.max_impact_bps
                )
                
                stock_capacities[symbol] = {
                    'liquidity_metrics': liquidity_metrics,
                    'constraints': capacity_constraints,
                    'max_position_shares': capacity_constraints.max_position_shares,
                    'max_position_twd': capacity_constraints.max_position_twd,
                    'impact_per_share': self._calculate_impact_per_share(symbol, symbol_data)
                }
                
                # Check for binding constraints
                if liquidity_metrics.liquidity_score < 0.3:
                    binding_constraints.append(f"{symbol}: Low liquidity score")
                
                if capacity_constraints.max_participation_rate > 0.15:
                    binding_constraints.append(f"{symbol}: High participation rate required")
                
                total_liquidity_impact += stock_capacities[symbol]['impact_per_share'] * 1000  # Per 1K shares
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze capacity for {symbol}: {e}")
                continue
        
        # Calculate portfolio-level capacity
        portfolio_capacity = self._calculate_portfolio_capacity(
            stock_capacities, capacity_type
        )
        
        # Determine capacity regime
        capacity_regime = self._determine_capacity_regime(
            portfolio_capacity, total_liquidity_impact, len(binding_constraints)
        )
        
        # Calculate stress test capacity
        stress_capacity = portfolio_capacity['max_portfolio_size_twd'] / self.parameters.stress_test_multiplier
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            stock_capacities, portfolio_capacity, binding_constraints
        )
        
        # Calculate confidence interval
        capacity_volatility = self._estimate_capacity_volatility(stock_capacities)
        confidence_interval = self._calculate_confidence_interval(
            portfolio_capacity['max_portfolio_size_twd'], 
            capacity_volatility
        )
        
        return CapacityAnalysisResult(
            strategy_name=strategy_name,
            analysis_date=date.today(),
            capacity_type=capacity_type,
            max_position_twd=portfolio_capacity['max_position_twd'],
            max_position_shares=portfolio_capacity['max_position_shares'],
            max_daily_turnover_twd=portfolio_capacity['max_daily_turnover_twd'],
            max_portfolio_size_twd=portfolio_capacity['max_portfolio_size_twd'],
            estimated_impact_bps=portfolio_capacity['estimated_impact_bps'],
            impact_budget_remaining_bps=max(0, self.parameters.target_impact_bps - portfolio_capacity['estimated_impact_bps']),
            impact_utilization_pct=(portfolio_capacity['estimated_impact_bps'] / self.parameters.max_impact_bps) * 100,
            capacity_utilization_pct=portfolio_capacity['utilization_pct'],
            stress_test_capacity_twd=stress_capacity,
            confidence_interval=confidence_interval,
            binding_constraints=binding_constraints,
            capacity_regime=capacity_regime,
            recommended_max_position_pct=self._recommend_position_size(capacity_regime),
            recommended_rebalancing_frequency=self._recommend_rebalancing_frequency(capacity_regime),
            optimization_suggestions=optimization_suggestions
        )
    
    def optimize_portfolio_capacity_allocation(
        self,
        target_weights: Dict[str, float],
        market_data: Dict[str, Dict[str, Any]],
        total_portfolio_value: float,
        constraints: Optional[Dict[str, Any]] = None
    ) -> PortfolioCapacityAllocation:
        """
        Optimize capacity allocation across portfolio positions.
        
        Args:
            target_weights: Target weights for each symbol
            market_data: Market data for each symbol
            total_portfolio_value: Total portfolio value
            constraints: Additional constraints
            
        Returns:
            PortfolioCapacityAllocation with optimization results
        """
        if constraints is None:
            constraints = {}
        
        # Calculate individual capacity limits
        symbol_capacities = {}
        total_unconstrained_capacity = 0.0
        
        for symbol, target_weight in target_weights.items():
            symbol_data = market_data.get(symbol, {})
            if not symbol_data:
                continue
            
            # Calculate symbol-specific capacity
            liquidity_metrics = self._get_liquidity_metrics(symbol, symbol_data)
            capacity_constraints = self.liquidity_analyzer.calculate_capacity_constraints(
                symbol=symbol,
                liquidity_metrics=liquidity_metrics
            )
            
            # Calculate capacity-constrained weight
            max_position_value = min(
                capacity_constraints.max_position_twd,
                total_portfolio_value * self.parameters.max_concentration_pct
            )
            
            capacity_weight = max_position_value / total_portfolio_value
            constrained_weight = min(target_weight, capacity_weight)
            
            # Calculate impact for this position
            position_shares = (constrained_weight * total_portfolio_value) / symbol_data.get('price', 100)
            impact_result = self.impact_model.calculate_impact(
                symbol=symbol,
                order_size=position_shares,
                price=symbol_data.get('price', 100),
                avg_daily_volume=liquidity_metrics.adv_20d,
                volatility=symbol_data.get('volatility', 0.25)
            )
            
            symbol_capacities[symbol] = {
                'target_weight': target_weight,
                'capacity_weight': capacity_weight,
                'constrained_weight': constrained_weight,
                'capacity_twd': max_position_value,
                'impact_bps': impact_result.total_impact_bps,
                'liquidity_score': liquidity_metrics.liquidity_score
            }
            
            total_unconstrained_capacity += max_position_value
        
        # Apply diversification benefit
        diversification_benefit = total_unconstrained_capacity * (self.parameters.diversification_benefit - 1)
        effective_capacity = total_unconstrained_capacity + diversification_benefit
        
        # Calculate allocation efficiency
        total_target_value = sum(w * total_portfolio_value for w in target_weights.values())
        total_allocated_value = sum(sc['capacity_twd'] for sc in symbol_capacities.values())
        utilization_pct = (total_allocated_value / effective_capacity) * 100
        
        # Calculate risk scores
        concentration_risk = self._calculate_concentration_risk(symbol_capacities, total_allocated_value)
        liquidity_risk = self._calculate_liquidity_risk(symbol_capacities)
        impact_risk = self._calculate_impact_risk(symbol_capacities)
        
        # Calculate efficiency score
        efficiency_score = self._calculate_allocation_efficiency(
            symbol_capacities, target_weights, total_portfolio_value
        )
        
        # Risk-adjusted capacity
        risk_adjustment_factor = 1 - (concentration_risk + liquidity_risk + impact_risk) / 3
        risk_adjusted_capacity = effective_capacity * risk_adjustment_factor
        
        return PortfolioCapacityAllocation(
            total_capacity_twd=effective_capacity,
            allocations={symbol: {
                'capacity_twd': data['capacity_twd'],
                'weight': data['constrained_weight'],
                'impact_bps': data['impact_bps']
            } for symbol, data in symbol_capacities.items()},
            diversification_benefit_twd=diversification_benefit,
            utilization_pct=utilization_pct,
            concentration_risk_score=concentration_risk,
            liquidity_risk_score=liquidity_risk,
            impact_risk_score=impact_risk,
            efficiency_score=efficiency_score,
            risk_adjusted_capacity=risk_adjusted_capacity
        )
    
    def stress_test_capacity(
        self,
        base_capacity: CapacityAnalysisResult,
        stress_scenarios: List[Dict[str, Any]],
        market_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Stress test strategy capacity under adverse scenarios.
        
        Args:
            base_capacity: Base capacity analysis
            stress_scenarios: List of stress test scenarios
            market_data: Current market data
            
        Returns:
            Stress test results
        """
        stress_results = {}
        
        for i, scenario in enumerate(stress_scenarios):
            scenario_name = scenario.get('name', f'Scenario_{i+1}')
            
            # Apply stress conditions
            stressed_data = self._apply_stress_scenario(market_data, scenario)
            
            # Recalculate capacity under stress
            try:
                stressed_capacity = self.analyze_strategy_capacity(
                    strategy_name=f"{base_capacity.strategy_name}_stressed",
                    universe=list(stressed_data.keys()),
                    market_data=stressed_data,
                    capacity_type=base_capacity.capacity_type
                )
                
                # Calculate capacity reduction
                capacity_reduction_pct = (
                    (base_capacity.max_portfolio_size_twd - stressed_capacity.max_portfolio_size_twd) /
                    base_capacity.max_portfolio_size_twd * 100
                )
                
                stress_results[scenario_name] = {
                    'stressed_capacity': stressed_capacity,
                    'capacity_reduction_pct': capacity_reduction_pct,
                    'impact_increase_bps': stressed_capacity.estimated_impact_bps - base_capacity.estimated_impact_bps,
                    'new_binding_constraints': [
                        c for c in stressed_capacity.binding_constraints 
                        if c not in base_capacity.binding_constraints
                    ],
                    'regime_change': stressed_capacity.capacity_regime != base_capacity.capacity_regime
                }
                
            except Exception as e:
                self.logger.error(f"Stress test failed for {scenario_name}: {e}")
                stress_results[scenario_name] = {'error': str(e)}
        
        # Calculate overall stress test metrics
        successful_tests = [r for r in stress_results.values() if 'error' not in r]
        
        if successful_tests:
            avg_capacity_reduction = np.mean([r['capacity_reduction_pct'] for r in successful_tests])
            max_capacity_reduction = max([r['capacity_reduction_pct'] for r in successful_tests])
            
            stress_summary = {
                'total_scenarios': len(stress_scenarios),
                'successful_tests': len(successful_tests),
                'avg_capacity_reduction_pct': avg_capacity_reduction,
                'max_capacity_reduction_pct': max_capacity_reduction,
                'scenarios_with_regime_change': sum(1 for r in successful_tests if r['regime_change']),
                'stress_adjusted_capacity_twd': base_capacity.max_portfolio_size_twd * (1 - max_capacity_reduction / 100)
            }
        else:
            stress_summary = {'error': 'All stress tests failed'}
        
        return {
            'base_capacity': base_capacity.to_dict(),
            'stress_scenarios': stress_results,
            'stress_summary': stress_summary
        }
    
    def _get_liquidity_metrics(self, symbol: str, symbol_data: Dict[str, Any]) -> LiquidityMetrics:
        """Get or calculate liquidity metrics for a symbol."""
        # Use cached metrics if available
        cache_key = f"{symbol}_{symbol_data.get('data_date', date.today())}"
        if cache_key in self._capacity_cache:
            return self._capacity_cache[cache_key]
        
        # Create mock volume/price data if not provided
        if 'volume_data' in symbol_data and 'price_data' in symbol_data:
            volume_data = symbol_data['volume_data']
            price_data = symbol_data['price_data']
        else:
            # Generate reasonable defaults
            adv = symbol_data.get('avg_daily_volume', 100_000)
            price = symbol_data.get('price', 100)
            
            # Create mock time series
            dates = pd.date_range(end=date.today(), periods=60, freq='D')
            volumes = pd.Series([adv * (1 + np.random.normal(0, 0.2)) for _ in range(60)], index=dates)
            prices = pd.Series([price * (1 + np.random.normal(0, 0.02)) for _ in range(60)], index=dates)
            
            volume_data = volumes
            price_data = prices
        
        metrics = self.liquidity_analyzer.calculate_liquidity_metrics(
            symbol=symbol,
            volume_data=volume_data,
            price_data=price_data,
            market_data=symbol_data
        )
        
        # Cache the result
        self._capacity_cache[cache_key] = metrics
        return metrics
    
    def _calculate_impact_per_share(self, symbol: str, symbol_data: Dict[str, Any]) -> float:
        """Calculate market impact per share for estimation."""
        # Use a standard 1000 share order for impact estimation
        try:
            impact_result = self.impact_model.calculate_impact(
                symbol=symbol,
                order_size=1000,
                price=symbol_data.get('price', 100),
                avg_daily_volume=symbol_data.get('avg_daily_volume', 100_000),
                volatility=symbol_data.get('volatility', 0.25)
            )
            return impact_result.total_impact_bps / 1000  # Impact per share in bps
        except Exception as e:
            self.logger.warning(f"Failed to calculate impact for {symbol}: {e}")
            return 0.05  # Default 0.05 bps per share
    
    def _calculate_portfolio_capacity(
        self, 
        stock_capacities: Dict[str, Dict[str, Any]], 
        capacity_type: CapacityType
    ) -> Dict[str, Any]:
        """Calculate portfolio-level capacity metrics."""
        
        total_capacity_twd = sum(sc['max_position_twd'] for sc in stock_capacities.values())
        
        # Apply diversification benefit
        if len(stock_capacities) > 1:
            diversification_factor = min(1.2, 1 + 0.05 * np.sqrt(len(stock_capacities)))
            total_capacity_twd *= diversification_factor
        
        # Calculate position limits
        max_position_shares = {symbol: data['max_position_shares'] for symbol, data in stock_capacities.items()}
        
        # Estimate daily turnover capacity
        daily_turnover_factor = {
            CapacityType.DAILY: 1.0,
            CapacityType.WEEKLY: 0.2,  # 20% per day
            CapacityType.MONTHLY: 0.05,  # 5% per day
            CapacityType.STRATEGY: 1.0 / self.parameters.rebalancing_frequency_days,
            CapacityType.PORTFOLIO: 0.1  # 10% per day
        }
        
        max_daily_turnover_twd = total_capacity_twd * daily_turnover_factor.get(capacity_type, 0.1)
        
        # Estimate portfolio impact
        avg_impact_per_stock = np.mean([sc['impact_per_share'] for sc in stock_capacities.values()])
        portfolio_impact_bps = avg_impact_per_stock * len(stock_capacities) * 1000  # Rough estimate
        
        # Apply impact budget constraint
        if portfolio_impact_bps > self.parameters.max_impact_bps:
            capacity_reduction_factor = self.parameters.max_impact_bps / portfolio_impact_bps
            total_capacity_twd *= capacity_reduction_factor
            max_daily_turnover_twd *= capacity_reduction_factor
            portfolio_impact_bps = self.parameters.max_impact_bps
        
        return {
            'max_portfolio_size_twd': total_capacity_twd,
            'max_position_twd': max(sc['max_position_twd'] for sc in stock_capacities.values()),
            'max_position_shares': max_position_shares,
            'max_daily_turnover_twd': max_daily_turnover_twd,
            'estimated_impact_bps': portfolio_impact_bps,
            'utilization_pct': 75.0,  # Assume 75% utilization for safety
            'stock_count': len(stock_capacities)
        }
    
    def _determine_capacity_regime(
        self, 
        portfolio_capacity: Dict[str, Any], 
        total_impact: float, 
        constraint_count: int
    ) -> CapacityRegime:
        """Determine capacity regime based on analysis."""
        
        if constraint_count > len(portfolio_capacity.get('max_position_shares', {})) * 0.5:
            return CapacityRegime.SEVERELY_CONSTRAINED
        elif total_impact > self.parameters.max_impact_bps * 0.8:
            return CapacityRegime.CONSTRAINED
        elif portfolio_capacity['max_portfolio_size_twd'] > 1_000_000_000:  # > 1B TWD
            return CapacityRegime.UNCONSTRAINED
        else:
            return CapacityRegime.NORMAL
    
    def _generate_optimization_suggestions(
        self,
        stock_capacities: Dict[str, Dict[str, Any]],
        portfolio_capacity: Dict[str, Any],
        binding_constraints: List[str]
    ) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Check for liquidity concentration
        illiquid_stocks = [
            symbol for symbol, data in stock_capacities.items()
            if data['liquidity_metrics'].liquidity_score < 0.4
        ]
        
        if len(illiquid_stocks) > portfolio_capacity['stock_count'] * 0.3:
            suggestions.append("Consider reducing allocation to illiquid stocks to improve capacity")
        
        # Check for impact concentration
        high_impact_stocks = [
            symbol for symbol, data in stock_capacities.items()
            if data['impact_per_share'] * 1000 > 20  # > 20 bps per 1K shares
        ]
        
        if high_impact_stocks:
            suggestions.append("Consider smaller position sizes or longer execution times for high-impact stocks")
        
        # Check overall impact utilization
        if portfolio_capacity['estimated_impact_bps'] > self.parameters.target_impact_bps:
            suggestions.append("Consider reducing overall portfolio size to stay within impact budget")
        
        # Check for binding constraints
        if len(binding_constraints) > 0:
            suggestions.append("Review binding constraints and consider alternative assets or execution strategies")
        
        return suggestions or ["Capacity analysis shows no immediate optimization needs"]
    
    def _recommend_position_size(self, regime: CapacityRegime) -> float:
        """Recommend maximum position size as percentage of portfolio."""
        recommendations = {
            CapacityRegime.UNCONSTRAINED: 0.20,  # 20%
            CapacityRegime.NORMAL: 0.15,         # 15%
            CapacityRegime.CONSTRAINED: 0.10,    # 10%
            CapacityRegime.SEVERELY_CONSTRAINED: 0.05  # 5%
        }
        return recommendations.get(regime, 0.10)
    
    def _recommend_rebalancing_frequency(self, regime: CapacityRegime) -> int:
        """Recommend rebalancing frequency in days."""
        recommendations = {
            CapacityRegime.UNCONSTRAINED: 5,     # Weekly
            CapacityRegime.NORMAL: 7,            # Weekly
            CapacityRegime.CONSTRAINED: 14,      # Bi-weekly
            CapacityRegime.SEVERELY_CONSTRAINED: 30  # Monthly
        }
        return recommendations.get(regime, 7)
    
    def _calculate_concentration_risk(self, symbol_capacities: Dict, total_value: float) -> float:
        """Calculate concentration risk score (0-1)."""
        weights = [data['capacity_twd'] / total_value for data in symbol_capacities.values()]
        max_weight = max(weights) if weights else 0
        
        # High concentration if any single position > 20%
        return min(1.0, max_weight / 0.20)
    
    def _calculate_liquidity_risk(self, symbol_capacities: Dict) -> float:
        """Calculate liquidity risk score (0-1)."""
        liquidity_scores = [data['liquidity_score'] for data in symbol_capacities.values()]
        avg_liquidity = np.mean(liquidity_scores) if liquidity_scores else 0.5
        
        # Risk increases as average liquidity decreases
        return max(0.0, (0.5 - avg_liquidity) / 0.5)
    
    def _calculate_impact_risk(self, symbol_capacities: Dict) -> float:
        """Calculate impact risk score (0-1)."""
        impact_scores = [data['impact_bps'] for data in symbol_capacities.values()]
        avg_impact = np.mean(impact_scores) if impact_scores else 20
        
        # Risk increases with higher impact
        return min(1.0, avg_impact / 100)  # Normalize to 100 bps
    
    def _calculate_allocation_efficiency(
        self, 
        symbol_capacities: Dict, 
        target_weights: Dict[str, float], 
        total_value: float
    ) -> float:
        """Calculate allocation efficiency score (0-1)."""
        efficiency_scores = []
        
        for symbol, target_weight in target_weights.items():
            if symbol in symbol_capacities:
                actual_weight = symbol_capacities[symbol]['constrained_weight']
                efficiency = min(1.0, actual_weight / target_weight) if target_weight > 0 else 1.0
                efficiency_scores.append(efficiency)
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.0
    
    def _estimate_capacity_volatility(self, stock_capacities: Dict) -> float:
        """Estimate capacity volatility for confidence intervals."""
        # Simplified volatility estimation
        capacity_values = [data['max_position_twd'] for data in stock_capacities.values()]
        
        if len(capacity_values) > 1:
            return np.std(capacity_values) / np.mean(capacity_values)
        else:
            return 0.2  # Default 20% volatility
    
    def _calculate_confidence_interval(
        self, 
        base_capacity: float, 
        volatility: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for capacity estimate."""
        confidence_factor = 1.96  # 95% confidence
        margin = base_capacity * volatility * confidence_factor
        
        return (
            max(0, base_capacity - margin),
            base_capacity + margin
        )
    
    def _apply_stress_scenario(
        self, 
        market_data: Dict[str, Dict[str, Any]], 
        scenario: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Apply stress scenario to market data."""
        stressed_data = {}
        
        for symbol, data in market_data.items():
            stressed_data[symbol] = data.copy()
            
            # Apply volatility stress
            if 'volatility_multiplier' in scenario:
                stressed_data[symbol]['volatility'] = (
                    data.get('volatility', 0.25) * scenario['volatility_multiplier']
                )
            
            # Apply liquidity stress (reduce ADV)
            if 'liquidity_reduction' in scenario:
                stressed_data[symbol]['avg_daily_volume'] = (
                    data.get('avg_daily_volume', 100_000) * (1 - scenario['liquidity_reduction'])
                )
            
            # Apply spread widening
            if 'spread_multiplier' in scenario:
                stressed_data[symbol]['spread'] = (
                    data.get('spread', 0.001) * scenario['spread_multiplier']
                )
        
        return stressed_data


# Factory functions
def create_capacity_analyzer(
    conservative: bool = False,
    custom_params: Optional[Dict[str, Any]] = None
) -> StrategyCapacityAnalyzer:
    """
    Create strategy capacity analyzer with default parameters.
    
    Args:
        conservative: Use conservative parameters
        custom_params: Override specific parameters
        
    Returns:
        Configured StrategyCapacityAnalyzer
    """
    if conservative:
        params = StrategyCapacityParameters(
            max_impact_bps=30.0,      # Lower impact tolerance
            target_impact_bps=20.0,
            max_concentration_pct=0.15,  # Lower concentration
            stress_test_multiplier=2.0   # Higher stress factor
        )
    else:
        params = StrategyCapacityParameters()
    
    # Override with custom parameters
    if custom_params:
        for key, value in custom_params.items():
            if hasattr(params, key):
                setattr(params, key, value)
    
    return StrategyCapacityAnalyzer(parameters=params)


# Example usage and testing
if __name__ == "__main__":
    print("Taiwan Strategy Capacity Analysis Demo")
    
    # Create analyzer
    analyzer = create_capacity_analyzer()
    
    # Sample universe and market data
    universe = ['2330.TW', '2317.TW', '2454.TW']
    market_data = {
        '2330.TW': {  # TSMC
            'price': 500.0,
            'avg_daily_volume': 500_000,
            'volatility': 0.25,
            'shares_outstanding': 25_930_000_000
        },
        '2317.TW': {  # Hon Hai
            'price': 100.0,
            'avg_daily_volume': 300_000,
            'volatility': 0.30,
            'shares_outstanding': 13_800_000_000
        },
        '2454.TW': {  # MediaTek
            'price': 800.0,
            'avg_daily_volume': 200_000,
            'volatility': 0.35,
            'shares_outstanding': 1_593_000_000
        }
    }
    
    # Analyze strategy capacity
    capacity_result = analyzer.analyze_strategy_capacity(
        strategy_name="Taiwan Large Cap Strategy",
        universe=universe,
        market_data=market_data,
        capacity_type=CapacityType.STRATEGY
    )
    
    print(f"\nCapacity Analysis for {capacity_result.strategy_name}:")
    print(f"Max Portfolio Size: NT${capacity_result.max_portfolio_size_twd:,.0f}")
    print(f"Max Daily Turnover: NT${capacity_result.max_daily_turnover_twd:,.0f}")
    print(f"Estimated Impact: {capacity_result.estimated_impact_bps:.1f} bps")
    print(f"Capacity Regime: {capacity_result.capacity_regime.value}")
    print(f"Stress Test Capacity: NT${capacity_result.stress_test_capacity_twd:,.0f}")
    
    if capacity_result.binding_constraints:
        print(f"\nBinding Constraints:")
        for constraint in capacity_result.binding_constraints:
            print(f"  - {constraint}")
    
    print(f"\nOptimization Suggestions:")
    for suggestion in capacity_result.optimization_suggestions:
        print(f"  - {suggestion}")
    
    # Test portfolio capacity allocation
    target_weights = {'2330.TW': 0.5, '2317.TW': 0.3, '2454.TW': 0.2}
    allocation = analyzer.optimize_portfolio_capacity_allocation(
        target_weights=target_weights,
        market_data=market_data,
        total_portfolio_value=100_000_000  # 100M TWD
    )
    
    print(f"\nPortfolio Capacity Allocation:")
    print(f"Total Capacity: NT${allocation.total_capacity_twd:,.0f}")
    print(f"Utilization: {allocation.utilization_pct:.1f}%")
    print(f"Efficiency Score: {allocation.efficiency_score:.2f}")
    
    for symbol, alloc in allocation.allocations.items():
        print(f"  {symbol}: NT${alloc['capacity_twd']:,.0f} ({alloc['weight']:.1%} weight, {alloc['impact_bps']:.1f} bps impact)")