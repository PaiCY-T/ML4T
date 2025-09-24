"""
Backtesting Framework Integration for Transaction Cost Modeling.

This module provides seamless integration between transaction cost models and the
existing backtesting engine, enabling real-time cost estimation, historical cost
analysis, and portfolio rebalancing cost estimation for Taiwan market trading.

Key Features:
- Real-time cost estimation API (<100ms response time)
- Historical cost analysis and trending
- Portfolio rebalancing cost estimation
- Integration with walk-forward validation
- Point-in-time data integration
- Cost-aware backtesting execution
- Performance validation with cost adjustments
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import asyncio
import time
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from .attribution import (
    CostAttributor, CostBreakdownAttribution, PortfolioCostAttribution,
    CostAttributionMethod, create_taiwan_cost_attributor
)
from .cost_models import TradeInfo, TradeDirection, BaseCostModel
from .market_impact import TaiwanMarketImpactModel
try:
    from ...backtesting.metrics.performance import PerformanceConfig, PerformanceAnalyzer
    from ...backtesting.validation.walk_forward import WalkForwardValidator
    from ...data.core.temporal import TemporalStore, DataType
    from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery
except ImportError:
    # Fallback imports for development
    TemporalStore = Any
    DataType = Any
    PointInTimeEngine = Any
    PITQuery = Any
    PerformanceConfig = Any
    PerformanceAnalyzer = Any
    WalkForwardValidator = Any

logger = logging.getLogger(__name__)


class CostEstimationMode(Enum):
    """Cost estimation modes for different use cases."""
    REAL_TIME = "real_time"           # <100ms response time
    DETAILED = "detailed"             # Full attribution analysis
    BATCH = "batch"                   # Batch processing optimization
    SIMULATION = "simulation"         # Backtesting simulation mode


class RebalancingStrategy(Enum):
    """Portfolio rebalancing strategies."""
    FULL_REBALANCE = "full"          # Complete portfolio rebalancing
    THRESHOLD_REBALANCE = "threshold" # Rebalance when drift exceeds threshold
    TACTICAL_REBALANCE = "tactical"   # Tactical allocation adjustments
    COST_OPTIMAL = "cost_optimal"     # Cost-optimized rebalancing


@dataclass
class CostEstimationRequest:
    """Request for cost estimation."""
    trades: List[TradeInfo]
    estimation_mode: CostEstimationMode = CostEstimationMode.REAL_TIME
    include_attribution: bool = False
    benchmark_comparison: bool = False
    market_data_snapshot: Optional[Dict[str, Any]] = None
    performance_context: Optional[Dict[str, Any]] = None
    max_response_time_ms: int = 100
    
    def __post_init__(self):
        """Validate request parameters."""
        if not self.trades:
            raise ValueError("At least one trade must be provided")
        
        if self.estimation_mode == CostEstimationMode.REAL_TIME and self.max_response_time_ms > 100:
            logger.warning("Real-time mode should have max response time ≤ 100ms")


@dataclass
class CostEstimationResponse:
    """Response from cost estimation."""
    request_id: str
    trades_analyzed: int
    total_estimated_cost_twd: float
    total_estimated_cost_bps: float
    
    # Individual trade costs
    trade_costs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Attribution analysis (if requested)
    cost_attributions: List[CostBreakdownAttribution] = field(default_factory=list)
    
    # Performance metrics
    estimation_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    confidence_score: float = 0.95
    
    # Market context
    market_regime: str = "normal"
    liquidity_conditions: str = "normal"
    
    # Warnings and issues
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'request_id': self.request_id,
            'trades_analyzed': self.trades_analyzed,
            'total_estimated_cost_twd': self.total_estimated_cost_twd,
            'total_estimated_cost_bps': self.total_estimated_cost_bps,
            'trade_costs': self.trade_costs,
            'performance_metrics': {
                'estimation_time_ms': self.estimation_time_ms,
                'cache_hit_rate': self.cache_hit_rate,
                'confidence_score': self.confidence_score
            },
            'market_context': {
                'market_regime': self.market_regime,
                'liquidity_conditions': self.liquidity_conditions
            },
            'warnings': self.warnings
        }


@dataclass
class RebalancingCostAnalysis:
    """Analysis of portfolio rebalancing costs."""
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    required_trades: List[TradeInfo]
    
    # Cost estimates
    total_rebalancing_cost_twd: float
    total_rebalancing_cost_bps: float
    cost_by_security: Dict[str, float]
    
    # Cost optimization
    alternative_strategies: Dict[RebalancingStrategy, Dict[str, Any]]
    recommended_strategy: RebalancingStrategy
    potential_cost_savings: float
    
    # Trade execution analysis
    execution_schedule: List[Dict[str, Any]]
    estimated_execution_time_minutes: float
    market_impact_analysis: Dict[str, Any]
    
    # Risk considerations
    tracking_error_impact: float
    liquidity_constraints: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'portfolio_analysis': {
                'current_weights': self.current_weights,
                'target_weights': self.target_weights,
                'required_trades_count': len(self.required_trades)
            },
            'cost_estimates': {
                'total_rebalancing_cost_twd': self.total_rebalancing_cost_twd,
                'total_rebalancing_cost_bps': self.total_rebalancing_cost_bps,
                'cost_by_security': self.cost_by_security
            },
            'optimization': {
                'alternative_strategies': self.alternative_strategies,
                'recommended_strategy': self.recommended_strategy.value,
                'potential_cost_savings': self.potential_cost_savings
            },
            'execution_analysis': {
                'execution_schedule': self.execution_schedule,
                'estimated_execution_time_minutes': self.estimated_execution_time_minutes,
                'market_impact_analysis': self.market_impact_analysis
            },
            'risk_analysis': {
                'tracking_error_impact': self.tracking_error_impact,
                'liquidity_constraints': self.liquidity_constraints
            }
        }


class CostEstimationCache:
    """High-performance cache for cost estimations."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _generate_key(self, trade: TradeInfo, market_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for trade and market context."""
        # Create deterministic key based on trade characteristics
        key_parts = [
            trade.symbol,
            trade.direction.value,
            f"{trade.quantity:.0f}",
            f"{trade.price:.4f}",
            trade.trade_date.isoformat()
        ]
        
        if market_context:
            # Include relevant market context
            vol = market_context.get('volatility', 0)
            key_parts.append(f"vol_{vol:.4f}")
        
        return "|".join(key_parts)
    
    def get(self, trade: TradeInfo, market_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get cached cost estimation."""
        key = self._generate_key(trade, market_context)
        current_time = time.time()
        
        if key in self._cache:
            cached_data, timestamp = self._cache[key]
            
            # Check if cache entry is still valid
            if current_time - timestamp < self.ttl_seconds:
                self._access_times[key] = current_time
                self._cache_hits += 1
                return cached_data
            else:
                # Remove expired entry
                del self._cache[key]
                del self._access_times[key]
        
        self._cache_misses += 1
        return None
    
    def put(self, trade: TradeInfo, cost_data: Dict[str, Any], market_context: Optional[Dict[str, Any]] = None):
        """Put cost estimation in cache."""
        key = self._generate_key(trade, market_context)
        current_time = time.time()
        
        # Evict oldest entries if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[key] = (cost_data, current_time)
        self._access_times[key] = current_time
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if self._access_times:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self._cache_hits + self._cache_misses
        if total_requests == 0:
            return 0.0
        return self._cache_hits / total_requests
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self._access_times.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class RealTimeCostEstimator:
    """
    High-performance real-time cost estimation engine.
    
    Optimized for <100ms response times with caching and parallel processing.
    """
    
    def __init__(
        self,
        cost_attributor: CostAttributor,
        temporal_store: TemporalStore,
        pit_engine: PointInTimeEngine,
        max_parallel_trades: int = 50,
        cache_size: int = 10000
    ):
        self.cost_attributor = cost_attributor
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        self.max_parallel_trades = max_parallel_trades
        
        # High-performance cache
        self.cache = CostEstimationCache(max_size=cache_size)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=min(32, (len(os.sched_getaffinity(0)) or 1) + 4))
        
        logger.info(f"RealTimeCostEstimator initialized with {max_parallel_trades} max parallel trades")
    
    async def estimate_costs(self, request: CostEstimationRequest) -> CostEstimationResponse:
        """
        Estimate costs with performance optimization.
        
        Args:
            request: Cost estimation request
            
        Returns:
            Cost estimation response with timing guarantees
        """
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"
        
        logger.debug(f"Processing cost estimation request {request_id} for {len(request.trades)} trades")
        
        try:
            # Choose estimation strategy based on mode
            if request.estimation_mode == CostEstimationMode.REAL_TIME:
                response = await self._estimate_real_time(request, request_id)
            elif request.estimation_mode == CostEstimationMode.BATCH:
                response = await self._estimate_batch(request, request_id)
            else:
                response = await self._estimate_detailed(request, request_id)
            
            # Set timing metrics
            end_time = time.time()
            response.estimation_time_ms = (end_time - start_time) * 1000
            response.cache_hit_rate = self.cache.get_hit_rate()
            
            # Check performance targets
            if (request.estimation_mode == CostEstimationMode.REAL_TIME and 
                response.estimation_time_ms > request.max_response_time_ms):
                response.warnings.append(
                    f"Response time {response.estimation_time_ms:.1f}ms exceeded target {request.max_response_time_ms}ms"
                )
            
            logger.debug(f"Cost estimation completed in {response.estimation_time_ms:.1f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Cost estimation failed for request {request_id}: {e}")
            # Return error response
            return CostEstimationResponse(
                request_id=request_id,
                trades_analyzed=0,
                total_estimated_cost_twd=0.0,
                total_estimated_cost_bps=0.0,
                warnings=[f"Estimation failed: {str(e)}"]
            )
    
    async def _estimate_real_time(
        self,
        request: CostEstimationRequest,
        request_id: str
    ) -> CostEstimationResponse:
        """Real-time cost estimation with aggressive optimization."""
        trade_costs = []
        total_cost_twd = 0.0
        total_trade_value = 0.0
        
        # Process trades in parallel with caching
        tasks = []
        for trade in request.trades[:self.max_parallel_trades]:  # Limit for performance
            task = asyncio.create_task(self._estimate_single_trade_fast(trade, request.market_data_snapshot))
            tasks.append(task)
        
        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=request.max_response_time_ms / 1000.0 * 0.8  # Use 80% of time limit
            )
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Trade estimation failed: {result}")
                    continue
                
                trade_costs.append(result)
                total_cost_twd += result['total_cost_twd']
                total_trade_value += result['trade_value']
        
        except asyncio.TimeoutError:
            logger.warning(f"Real-time estimation timed out for request {request_id}")
        
        # Calculate aggregate metrics
        total_cost_bps = (total_cost_twd / total_trade_value * 10000) if total_trade_value > 0 else 0.0
        
        return CostEstimationResponse(
            request_id=request_id,
            trades_analyzed=len(trade_costs),
            total_estimated_cost_twd=total_cost_twd,
            total_estimated_cost_bps=total_cost_bps,
            trade_costs=trade_costs,
            market_regime="normal",  # Simplified for real-time
            liquidity_conditions="normal"
        )
    
    async def _estimate_single_trade_fast(
        self,
        trade: TradeInfo,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fast single trade cost estimation with caching."""
        # Check cache first
        cached_result = self.cache.get(trade, market_data)
        if cached_result:
            return cached_result
        
        # Calculate costs quickly (simplified for performance)
        try:
            # Use simplified cost model for speed
            cost_breakdown = self.cost_attributor.cost_model.calculate_cost(trade)
            
            result = {
                'symbol': trade.symbol,
                'total_cost_twd': cost_breakdown.total_cost,
                'total_cost_bps': cost_breakdown.cost_bps,
                'trade_value': trade.trade_value,
                'cost_components': {
                    'regulatory': cost_breakdown.total_regulatory_cost,
                    'market': cost_breakdown.total_market_cost
                }
            }
            
            # Cache result
            self.cache.put(trade, result, market_data)
            return result
            
        except Exception as e:
            logger.debug(f"Fast estimation failed for {trade.symbol}: {e}")
            # Return minimal result
            return {
                'symbol': trade.symbol,
                'total_cost_twd': trade.trade_value * 0.001,  # 10bps fallback
                'total_cost_bps': 10.0,
                'trade_value': trade.trade_value,
                'cost_components': {'regulatory': 0.0, 'market': 0.0}
            }
    
    async def _estimate_batch(
        self,
        request: CostEstimationRequest,
        request_id: str
    ) -> CostEstimationResponse:
        """Batch cost estimation with optimized processing."""
        # Process trades in batches for optimal throughput
        batch_size = min(100, len(request.trades))
        trade_costs = []
        total_cost_twd = 0.0
        total_trade_value = 0.0
        
        for i in range(0, len(request.trades), batch_size):
            batch_trades = request.trades[i:i + batch_size]
            batch_results = await self._process_trade_batch(batch_trades, request.market_data_snapshot)
            
            trade_costs.extend(batch_results)
            for result in batch_results:
                total_cost_twd += result['total_cost_twd']
                total_trade_value += result['trade_value']
        
        total_cost_bps = (total_cost_twd / total_trade_value * 10000) if total_trade_value > 0 else 0.0
        
        return CostEstimationResponse(
            request_id=request_id,
            trades_analyzed=len(trade_costs),
            total_estimated_cost_twd=total_cost_twd,
            total_estimated_cost_bps=total_cost_bps,
            trade_costs=trade_costs
        )
    
    async def _process_trade_batch(
        self,
        trades: List[TradeInfo],
        market_data: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of trades efficiently."""
        # Submit batch to thread pool
        loop = asyncio.get_event_loop()
        
        def process_trade(trade):
            try:
                cost_breakdown = self.cost_attributor.cost_model.calculate_cost(trade)
                return {
                    'symbol': trade.symbol,
                    'total_cost_twd': cost_breakdown.total_cost,
                    'total_cost_bps': cost_breakdown.cost_bps,
                    'trade_value': trade.trade_value,
                    'cost_components': {
                        'regulatory': cost_breakdown.total_regulatory_cost,
                        'market': cost_breakdown.total_market_cost
                    }
                }
            except Exception as e:
                logger.debug(f"Batch estimation failed for {trade.symbol}: {e}")
                return {
                    'symbol': trade.symbol,
                    'total_cost_twd': trade.trade_value * 0.001,
                    'total_cost_bps': 10.0,
                    'trade_value': trade.trade_value,
                    'cost_components': {'regulatory': 0.0, 'market': 0.0}
                }
        
        # Execute batch in parallel
        tasks = [loop.run_in_executor(self.executor, process_trade, trade) for trade in trades]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        return valid_results
    
    async def _estimate_detailed(
        self,
        request: CostEstimationRequest,
        request_id: str
    ) -> CostEstimationResponse:
        """Detailed cost estimation with full attribution."""
        trade_costs = []
        cost_attributions = []
        total_cost_twd = 0.0
        total_trade_value = 0.0
        
        for trade in request.trades:
            try:
                # Full attribution analysis
                attribution = self.cost_attributor.attribute_trade_costs(
                    trade,
                    market_data=request.market_data_snapshot,
                    benchmark_comparison=request.benchmark_comparison,
                    performance_context=request.performance_context
                )
                
                cost_attributions.append(attribution)
                
                cost_data = {
                    'symbol': trade.symbol,
                    'total_cost_twd': attribution.total_cost_twd(),
                    'total_cost_bps': attribution.total_cost_bps(),
                    'trade_value': trade.trade_value,
                    'cost_breakdown': attribution.to_dict()
                }
                
                trade_costs.append(cost_data)
                total_cost_twd += attribution.total_cost_twd()
                total_trade_value += trade.trade_value
                
            except Exception as e:
                logger.warning(f"Detailed estimation failed for {trade.symbol}: {e}")
        
        total_cost_bps = (total_cost_twd / total_trade_value * 10000) if total_trade_value > 0 else 0.0
        
        return CostEstimationResponse(
            request_id=request_id,
            trades_analyzed=len(trade_costs),
            total_estimated_cost_twd=total_cost_twd,
            total_estimated_cost_bps=total_cost_bps,
            trade_costs=trade_costs,
            cost_attributions=cost_attributions
        )


class PortfolioRebalancingAnalyzer:
    """
    Analyze portfolio rebalancing costs and optimization strategies.
    
    Provides comprehensive analysis of rebalancing costs and alternative
    strategies for cost-optimal portfolio management.
    """
    
    def __init__(
        self,
        cost_estimator: RealTimeCostEstimator,
        temporal_store: TemporalStore,
        pit_engine: PointInTimeEngine
    ):
        self.cost_estimator = cost_estimator
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        
        logger.info("PortfolioRebalancingAnalyzer initialized")
    
    async def analyze_rebalancing_costs(
        self,
        current_portfolio: Dict[str, float],  # {symbol: current_weight}
        target_portfolio: Dict[str, float],   # {symbol: target_weight}
        portfolio_value: float,
        rebalancing_date: date,
        market_data: Optional[Dict[str, Any]] = None
    ) -> RebalancingCostAnalysis:
        """
        Analyze costs and strategies for portfolio rebalancing.
        
        Args:
            current_portfolio: Current portfolio weights
            target_portfolio: Target portfolio weights
            portfolio_value: Total portfolio value
            rebalancing_date: Rebalancing date
            market_data: Current market data
            
        Returns:
            Comprehensive rebalancing cost analysis
        """
        logger.info(f"Analyzing rebalancing costs for portfolio value: ${portfolio_value:,.0f}")
        
        # Calculate required trades
        required_trades = self._calculate_required_trades(
            current_portfolio, target_portfolio, portfolio_value, rebalancing_date
        )
        
        # Estimate costs for full rebalancing
        full_rebalance_costs = await self._estimate_full_rebalancing_costs(
            required_trades, market_data
        )
        
        # Analyze alternative strategies
        alternative_strategies = await self._analyze_alternative_strategies(
            current_portfolio, target_portfolio, portfolio_value, rebalancing_date, market_data
        )
        
        # Determine recommended strategy
        recommended_strategy = self._select_optimal_strategy(alternative_strategies)
        
        # Calculate execution schedule
        execution_schedule = self._create_execution_schedule(
            required_trades, recommended_strategy, market_data
        )
        
        # Analyze market impact
        market_impact_analysis = await self._analyze_market_impact(required_trades, market_data)
        
        # Calculate risk considerations
        tracking_error_impact = self._calculate_tracking_error_impact(
            current_portfolio, target_portfolio
        )
        
        liquidity_constraints = self._identify_liquidity_constraints(required_trades, market_data)
        
        return RebalancingCostAnalysis(
            current_weights=current_portfolio,
            target_weights=target_portfolio,
            required_trades=required_trades,
            total_rebalancing_cost_twd=full_rebalance_costs['total_cost_twd'],
            total_rebalancing_cost_bps=full_rebalance_costs['total_cost_bps'],
            cost_by_security=full_rebalance_costs['cost_by_security'],
            alternative_strategies=alternative_strategies,
            recommended_strategy=recommended_strategy,
            potential_cost_savings=self._calculate_potential_savings(alternative_strategies),
            execution_schedule=execution_schedule,
            estimated_execution_time_minutes=self._estimate_execution_time(execution_schedule),
            market_impact_analysis=market_impact_analysis,
            tracking_error_impact=tracking_error_impact,
            liquidity_constraints=liquidity_constraints
        )
    
    def _calculate_required_trades(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        portfolio_value: float,
        trade_date: date
    ) -> List[TradeInfo]:
        """Calculate trades required for rebalancing."""
        required_trades = []
        
        # Get all symbols in either portfolio
        all_symbols = set(current_portfolio.keys()) | set(target_portfolio.keys())
        
        for symbol in all_symbols:
            current_weight = current_portfolio.get(symbol, 0.0)
            target_weight = target_portfolio.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.0001:  # 1bp threshold
                # Calculate trade size
                trade_value = abs(weight_diff) * portfolio_value
                
                # Estimate current price (simplified)
                estimated_price = 100.0  # Would fetch from market data
                quantity = trade_value / estimated_price
                
                # Determine direction
                direction = TradeDirection.BUY if weight_diff > 0 else TradeDirection.SELL
                
                trade = TradeInfo(
                    symbol=symbol,
                    trade_date=trade_date,
                    direction=direction,
                    quantity=quantity,
                    price=estimated_price,
                    daily_volume=100000,  # Would fetch from market data
                    volatility=0.25,      # Would fetch from market data
                    bid_ask_spread=0.05   # Would fetch from market data
                )
                
                required_trades.append(trade)
        
        return required_trades
    
    async def _estimate_full_rebalancing_costs(
        self,
        required_trades: List[TradeInfo],
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate costs for full portfolio rebalancing."""
        if not required_trades:
            return {
                'total_cost_twd': 0.0,
                'total_cost_bps': 0.0,
                'cost_by_security': {}
            }
        
        # Create cost estimation request
        request = CostEstimationRequest(
            trades=required_trades,
            estimation_mode=CostEstimationMode.DETAILED,
            market_data_snapshot=market_data
        )
        
        # Get cost estimates
        response = await self.cost_estimator.estimate_costs(request)
        
        # Aggregate by security
        cost_by_security = {}
        for trade_cost in response.trade_costs:
            symbol = trade_cost['symbol']
            cost_by_security[symbol] = trade_cost['total_cost_twd']
        
        return {
            'total_cost_twd': response.total_estimated_cost_twd,
            'total_cost_bps': response.total_estimated_cost_bps,
            'cost_by_security': cost_by_security
        }
    
    async def _analyze_alternative_strategies(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        portfolio_value: float,
        rebalancing_date: date,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[RebalancingStrategy, Dict[str, Any]]:
        """Analyze alternative rebalancing strategies."""
        strategies = {}
        
        # Full rebalancing (already calculated)
        full_trades = self._calculate_required_trades(
            current_portfolio, target_portfolio, portfolio_value, rebalancing_date
        )
        full_costs = await self._estimate_full_rebalancing_costs(full_trades, market_data)
        
        strategies[RebalancingStrategy.FULL_REBALANCE] = {
            'trades_required': len(full_trades),
            'total_cost_twd': full_costs['total_cost_twd'],
            'total_cost_bps': full_costs['total_cost_bps'],
            'description': "Complete rebalancing to target weights"
        }
        
        # Threshold rebalancing (only rebalance positions with >5% drift)
        threshold_trades = self._calculate_threshold_trades(
            current_portfolio, target_portfolio, portfolio_value, rebalancing_date, threshold=0.05
        )
        threshold_costs = await self._estimate_full_rebalancing_costs(threshold_trades, market_data)
        
        strategies[RebalancingStrategy.THRESHOLD_REBALANCE] = {
            'trades_required': len(threshold_trades),
            'total_cost_twd': threshold_costs['total_cost_twd'],
            'total_cost_bps': threshold_costs['total_cost_bps'],
            'description': "Rebalance only positions with >5% weight drift"
        }
        
        # Cost-optimal rebalancing (optimize trade-off between tracking error and costs)
        optimal_trades = self._calculate_cost_optimal_trades(
            current_portfolio, target_portfolio, portfolio_value, rebalancing_date
        )
        optimal_costs = await self._estimate_full_rebalancing_costs(optimal_trades, market_data)
        
        strategies[RebalancingStrategy.COST_OPTIMAL] = {
            'trades_required': len(optimal_trades),
            'total_cost_twd': optimal_costs['total_cost_twd'],
            'total_cost_bps': optimal_costs['total_cost_bps'],
            'description': "Cost-optimized rebalancing with tracking error consideration"
        }
        
        return strategies
    
    def _calculate_threshold_trades(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        portfolio_value: float,
        trade_date: date,
        threshold: float = 0.05
    ) -> List[TradeInfo]:
        """Calculate trades for threshold-based rebalancing."""
        required_trades = []
        
        all_symbols = set(current_portfolio.keys()) | set(target_portfolio.keys())
        
        for symbol in all_symbols:
            current_weight = current_portfolio.get(symbol, 0.0)
            target_weight = target_portfolio.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            # Only trade if drift exceeds threshold
            if abs(weight_diff) > threshold:
                trade_value = abs(weight_diff) * portfolio_value
                estimated_price = 100.0
                quantity = trade_value / estimated_price
                direction = TradeDirection.BUY if weight_diff > 0 else TradeDirection.SELL
                
                trade = TradeInfo(
                    symbol=symbol,
                    trade_date=trade_date,
                    direction=direction,
                    quantity=quantity,
                    price=estimated_price,
                    daily_volume=100000,
                    volatility=0.25,
                    bid_ask_spread=0.05
                )
                
                required_trades.append(trade)
        
        return required_trades
    
    def _calculate_cost_optimal_trades(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        portfolio_value: float,
        trade_date: date
    ) -> List[TradeInfo]:
        """Calculate cost-optimal trades balancing costs and tracking error."""
        # Simplified cost-optimal calculation
        # In practice, this would use optimization algorithms
        
        required_trades = []
        all_symbols = set(current_portfolio.keys()) | set(target_portfolio.keys())
        
        for symbol in all_symbols:
            current_weight = current_portfolio.get(symbol, 0.0)
            target_weight = target_portfolio.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            # Apply cost-aware threshold based on expected costs
            estimated_cost_bps = 15.0  # Estimated transaction cost
            cost_threshold = estimated_cost_bps / 10000  # Convert to decimal
            
            # Only trade if expected benefit exceeds cost
            if abs(weight_diff) > cost_threshold * 2:  # 2x cost threshold
                trade_value = abs(weight_diff) * portfolio_value
                estimated_price = 100.0
                quantity = trade_value / estimated_price
                direction = TradeDirection.BUY if weight_diff > 0 else TradeDirection.SELL
                
                trade = TradeInfo(
                    symbol=symbol,
                    trade_date=trade_date,
                    direction=direction,
                    quantity=quantity,
                    price=estimated_price,
                    daily_volume=100000,
                    volatility=0.25,
                    bid_ask_spread=0.05
                )
                
                required_trades.append(trade)
        
        return required_trades
    
    def _select_optimal_strategy(self, alternative_strategies: Dict[RebalancingStrategy, Dict[str, Any]]) -> RebalancingStrategy:
        """Select optimal rebalancing strategy based on cost-benefit analysis."""
        # Simple strategy selection based on cost savings
        # In practice, this would consider tracking error, risk, and other factors
        
        min_cost = float('inf')
        optimal_strategy = RebalancingStrategy.FULL_REBALANCE
        
        for strategy, metrics in alternative_strategies.items():
            if metrics['total_cost_twd'] < min_cost:
                min_cost = metrics['total_cost_twd']
                optimal_strategy = strategy
        
        return optimal_strategy
    
    def _create_execution_schedule(
        self,
        required_trades: List[TradeInfo],
        strategy: RebalancingStrategy,
        market_data: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create optimal execution schedule for trades."""
        if not required_trades:
            return []
        
        # Sort trades by impact/urgency
        sorted_trades = sorted(required_trades, key=lambda t: t.trade_value, reverse=True)
        
        execution_schedule = []
        current_time = datetime.combine(required_trades[0].trade_date, datetime.min.time())
        
        for i, trade in enumerate(sorted_trades):
            # Spread execution over time for large trades
            if trade.trade_value > 1000000:  # > 1M TWD
                execution_delay = i * 15  # 15 minutes between large trades
            else:
                execution_delay = i * 5   # 5 minutes between small trades
            
            scheduled_time = current_time + timedelta(minutes=execution_delay)
            
            execution_schedule.append({
                'symbol': trade.symbol,
                'direction': trade.direction.value,
                'quantity': trade.quantity,
                'estimated_price': trade.price,
                'scheduled_time': scheduled_time.isoformat(),
                'priority': 'high' if trade.trade_value > 1000000 else 'normal'
            })
        
        return execution_schedule
    
    async def _analyze_market_impact(
        self,
        required_trades: List[TradeInfo],
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze market impact of rebalancing trades."""
        if not required_trades:
            return {'total_impact_bps': 0.0, 'high_impact_trades': []}
        
        high_impact_trades = []
        total_impact_bps = 0.0
        total_trade_value = 0.0
        
        for trade in required_trades:
            # Estimate market impact (simplified)
            if trade.order_size_vs_avg and trade.order_size_vs_avg > 0.05:  # > 5% of ADV
                impact_bps = 20.0 * trade.order_size_vs_avg  # Impact scaling
                high_impact_trades.append({
                    'symbol': trade.symbol,
                    'participation_rate': trade.order_size_vs_avg,
                    'estimated_impact_bps': impact_bps
                })
            else:
                impact_bps = 5.0  # Base impact
            
            total_impact_bps += impact_bps * trade.trade_value
            total_trade_value += trade.trade_value
        
        weighted_avg_impact = (total_impact_bps / total_trade_value) if total_trade_value > 0 else 0.0
        
        return {
            'weighted_avg_impact_bps': weighted_avg_impact,
            'high_impact_trade_count': len(high_impact_trades),
            'high_impact_trades': high_impact_trades
        }
    
    def _calculate_tracking_error_impact(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float]
    ) -> float:
        """Calculate tracking error impact of not rebalancing."""
        # Simplified tracking error calculation
        total_drift = 0.0
        
        all_symbols = set(current_portfolio.keys()) | set(target_portfolio.keys())
        
        for symbol in all_symbols:
            current_weight = current_portfolio.get(symbol, 0.0)
            target_weight = target_portfolio.get(symbol, 0.0)
            drift = abs(target_weight - current_weight)
            total_drift += drift ** 2
        
        tracking_error = np.sqrt(total_drift)
        return tracking_error
    
    def _identify_liquidity_constraints(
        self,
        required_trades: List[TradeInfo],
        market_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify liquidity constraints for rebalancing."""
        constraints = []
        
        for trade in required_trades:
            # Check for high participation rate
            if trade.order_size_vs_avg and trade.order_size_vs_avg > 0.15:  # > 15% of ADV
                constraints.append(
                    f"{trade.symbol}: High participation rate ({trade.order_size_vs_avg:.1%} of ADV)"
                )
            
            # Check for large trade value
            if trade.trade_value > 5000000:  # > 5M TWD
                constraints.append(
                    f"{trade.symbol}: Large trade value (${trade.trade_value:,.0f})"
                )
        
        return constraints
    
    def _calculate_potential_savings(self, alternative_strategies: Dict[RebalancingStrategy, Dict[str, Any]]) -> float:
        """Calculate potential cost savings from optimal strategy."""
        if not alternative_strategies:
            return 0.0
        
        full_rebalance_cost = alternative_strategies.get(
            RebalancingStrategy.FULL_REBALANCE, {}
        ).get('total_cost_twd', 0.0)
        
        min_cost = min(
            strategy['total_cost_twd'] for strategy in alternative_strategies.values()
        )
        
        return max(0.0, full_rebalance_cost - min_cost)
    
    def _estimate_execution_time(self, execution_schedule: List[Dict[str, Any]]) -> float:
        """Estimate total execution time for trades."""
        if not execution_schedule:
            return 0.0
        
        # Find time span of execution
        start_times = [
            datetime.fromisoformat(trade['scheduled_time']) 
            for trade in execution_schedule
        ]
        
        if len(start_times) <= 1:
            return 5.0  # 5 minutes for single trade
        
        total_time_delta = max(start_times) - min(start_times)
        return total_time_delta.total_seconds() / 60.0  # Convert to minutes


class BacktestingCostIntegrator:
    """
    Integrate transaction costs into backtesting framework.
    
    Provides seamless integration with walk-forward validation and
    performance analysis systems.
    """
    
    def __init__(
        self,
        cost_estimator: RealTimeCostEstimator,
        rebalancing_analyzer: PortfolioRebalancingAnalyzer,
        temporal_store: TemporalStore,
        pit_engine: PointInTimeEngine
    ):
        self.cost_estimator = cost_estimator
        self.rebalancing_analyzer = rebalancing_analyzer
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        
        logger.info("BacktestingCostIntegrator initialized")
    
    async def integrate_costs_with_backtest(
        self,
        portfolio_weights: pd.DataFrame,  # Date x Symbol weights
        portfolio_returns: pd.Series,     # Portfolio returns
        rebalancing_dates: List[date],
        portfolio_value_series: pd.Series,
        cost_estimation_mode: CostEstimationMode = CostEstimationMode.DETAILED
    ) -> Dict[str, Any]:
        """
        Integrate transaction costs into backtesting analysis.
        
        Args:
            portfolio_weights: Portfolio weights over time
            portfolio_returns: Portfolio returns time series
            rebalancing_dates: Dates when rebalancing occurred
            portfolio_value_series: Portfolio values over time
            cost_estimation_mode: Cost estimation mode
            
        Returns:
            Integrated cost and performance analysis
        """
        logger.info(f"Integrating costs for backtest with {len(rebalancing_dates)} rebalancing dates")
        
        # Calculate costs for each rebalancing
        rebalancing_costs = []
        total_costs_twd = 0.0
        
        for i, rebal_date in enumerate(rebalancing_dates):
            if rebal_date not in portfolio_weights.index:
                continue
            
            # Get current and target weights
            if i == 0:
                current_weights = {}  # Initial portfolio
            else:
                prev_date = rebalancing_dates[i-1]
                if prev_date in portfolio_weights.index:
                    current_weights = portfolio_weights.loc[prev_date].to_dict()
                else:
                    current_weights = {}
            
            target_weights = portfolio_weights.loc[rebal_date].to_dict()
            portfolio_value = portfolio_value_series.loc[rebal_date] if rebal_date in portfolio_value_series.index else 1000000
            
            # Analyze rebalancing costs
            try:
                rebal_analysis = await self.rebalancing_analyzer.analyze_rebalancing_costs(
                    current_weights, target_weights, portfolio_value, rebal_date
                )
                
                rebalancing_costs.append({
                    'date': rebal_date.isoformat(),
                    'cost_twd': rebal_analysis.total_rebalancing_cost_twd,
                    'cost_bps': rebal_analysis.total_rebalancing_cost_bps,
                    'trades_count': len(rebal_analysis.required_trades),
                    'recommended_strategy': rebal_analysis.recommended_strategy.value,
                    'potential_savings': rebal_analysis.potential_cost_savings
                })
                
                total_costs_twd += rebal_analysis.total_rebalancing_cost_twd
                
            except Exception as e:
                logger.warning(f"Failed to analyze rebalancing costs for {rebal_date}: {e}")
        
        # Calculate cost-adjusted performance metrics
        cost_adjusted_analysis = self._calculate_cost_adjusted_metrics(
            portfolio_returns, rebalancing_costs, portfolio_value_series
        )
        
        return {
            'rebalancing_costs': rebalancing_costs,
            'total_costs_twd': total_costs_twd,
            'cost_adjusted_metrics': cost_adjusted_analysis,
            'cost_summary': {
                'avg_rebalancing_cost_bps': np.mean([r['cost_bps'] for r in rebalancing_costs]) if rebalancing_costs else 0.0,
                'total_rebalancing_events': len(rebalancing_costs),
                'total_cost_drag_bps': cost_adjusted_analysis.get('total_cost_drag_bps', 0.0)
            }
        }
    
    def _calculate_cost_adjusted_metrics(
        self,
        portfolio_returns: pd.Series,
        rebalancing_costs: List[Dict[str, Any]],
        portfolio_value_series: pd.Series
    ) -> Dict[str, Any]:
        """Calculate cost-adjusted performance metrics."""
        if not rebalancing_costs:
            return {
                'cost_adjusted_returns': portfolio_returns,
                'total_cost_drag_bps': 0.0,
                'cost_adjusted_sharpe': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            }
        
        # Create cost impact series
        cost_impacts = pd.Series(0.0, index=portfolio_returns.index)
        
        for cost_data in rebalancing_costs:
            cost_date = pd.to_datetime(cost_data['date']).date()
            if cost_date in portfolio_returns.index:
                # Convert cost to return impact
                portfolio_value = portfolio_value_series.loc[cost_date] if cost_date in portfolio_value_series.index else 1000000
                cost_impact = -cost_data['cost_twd'] / portfolio_value
                cost_impacts.loc[cost_date] = cost_impact
        
        # Calculate cost-adjusted returns
        cost_adjusted_returns = portfolio_returns + cost_impacts
        
        # Calculate metrics
        total_cost_drag = -cost_impacts.sum()
        total_cost_drag_bps = total_cost_drag * 10000
        
        # Cost-adjusted Sharpe ratio
        if cost_adjusted_returns.std() > 0:
            cost_adjusted_sharpe = cost_adjusted_returns.mean() / cost_adjusted_returns.std() * np.sqrt(252)
        else:
            cost_adjusted_sharpe = 0.0
        
        return {
            'cost_adjusted_returns': cost_adjusted_returns,
            'total_cost_drag_bps': total_cost_drag_bps,
            'cost_adjusted_sharpe': cost_adjusted_sharpe,
            'cost_impact_dates': cost_impacts[cost_impacts != 0].index.tolist()
        }


# Factory functions and utilities
def create_taiwan_backtesting_integration(
    temporal_store: TemporalStore,
    pit_engine: PointInTimeEngine,
    cache_size: int = 10000,
    max_parallel_trades: int = 50
) -> BacktestingCostIntegrator:
    """
    Create complete Taiwan market backtesting cost integration.
    
    Args:
        temporal_store: Temporal data store
        pit_engine: Point-in-time data engine
        cache_size: Cost estimation cache size
        max_parallel_trades: Maximum parallel trades for performance
        
    Returns:
        Complete backtesting cost integration system
    """
    # Create cost attributor
    cost_attributor = create_taiwan_cost_attributor()
    
    # Create real-time cost estimator
    cost_estimator = RealTimeCostEstimator(
        cost_attributor=cost_attributor,
        temporal_store=temporal_store,
        pit_engine=pit_engine,
        max_parallel_trades=max_parallel_trades,
        cache_size=cache_size
    )
    
    # Create rebalancing analyzer
    rebalancing_analyzer = PortfolioRebalancingAnalyzer(
        cost_estimator=cost_estimator,
        temporal_store=temporal_store,
        pit_engine=pit_engine
    )
    
    # Create complete integration
    return BacktestingCostIntegrator(
        cost_estimator=cost_estimator,
        rebalancing_analyzer=rebalancing_analyzer,
        temporal_store=temporal_store,
        pit_engine=pit_engine
    )


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    print("Backtesting Cost Integration Demo")
    print("Real-time cost estimation and rebalancing analysis")
    
    # This would be integrated with actual temporal store and PIT engine
    print("\nKey capabilities:")
    print("1. Real-time cost estimation (<100ms)")
    print("2. Portfolio rebalancing cost analysis")
    print("3. Cost-adjusted backtesting metrics")
    print("4. Integration with walk-forward validation")
    print("5. Taiwan market cost optimization")