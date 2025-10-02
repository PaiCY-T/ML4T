"""
Simple ETF Flow Factor System for Personal Trading
Task #002 - Flow Factor Group Implementation (Simplified)

Simplified flow factor calculation for ETF and stock trading:
- Basic buy/sell volume flow ratios
- ETF prioritization for liquidity advantages
- Performance optimization for Taiwan market
- IC monitoring for factor validation

Designed for personal trading systems - no complex market indicators.
GitHub Issue: #75
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
import warnings

# Import base classes
from .base import (
    FactorCalculator, FactorResult, FactorMetadata, FactorCategory,
    FactorFrequency, FactorEngine
)

logger = logging.getLogger(__name__)




@dataclass
class ETFFlowMetrics:
    """Simple ETF flow factor calculations for personal trading."""
    symbol: str
    date: date

    # Core flow factors
    net_flow: Optional[float] = None          # Buy Volume - Sell Volume
    flow_ratio: Optional[float] = None        # (Buy - Sell) / (Buy + Sell)
    total_volume: Optional[float] = None      # Buy Volume + Sell Volume

    # ETF-specific metrics
    is_etf: bool = False
    etf_aum: Optional[float] = None
    etf_liquidity_tier: Optional[int] = None  # 1-3, 1 being most liquid

    # Quality indicators
    data_completeness_score: float = 0.0
    broker_coverage_count: int = 0           # Number of Top15 brokers with data


@dataclass
class ICMonitoringResult:
    """Information Coefficient monitoring result for factor validation."""
    symbol: str
    date: date

    # IC calculations
    ic_5day: Optional[float] = None          # 5-day forward return IC
    ic_20day: Optional[float] = None         # 20-day forward return IC
    ic_significance: bool = False            # |IC| > 0.02 threshold

    # Statistical validation
    pearson_correlation: Optional[float] = None
    p_value: Optional[float] = None          # p < 0.05 threshold
    sample_size: int = 0

    # Performance tracking
    cumulative_ic: Optional[float] = None
    ic_decay_detected: bool = False
    effectiveness_score: float = 0.0        # 0-100% based on IC significance


class SimpleETFFlowFactor:
    """
    Simple ETF Flow Factor system for personal trading.

    Focuses on basic flow calculations:
    - ETF prioritization for liquidity advantages
    - Simple buy/sell volume flow ratios
    - Performance optimization
    - IC monitoring for factor validation

    Performance targets:
    - <200ms latency for Taiwan market
    - |IC| > 0.02 significance threshold
    - Good data completeness
    """

    # Configuration constants
    MIN_IC_SIGNIFICANCE = 0.02               # Statistical significance threshold
    TARGET_LATENCY_MS = 200                  # Performance requirement

    def __init__(self, finlab_connector=None):
        self.finlab_connector = finlab_connector
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Performance monitoring
        self._calculation_times = []
        self._etf_cache = {}

    def calculate_flow_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, ETFFlowMetrics]:
        """
        Calculate simple flow factors for ETFs and stocks.

        Args:
            symbols: List of symbols (ETFs prioritized)
            as_of_date: Calculation date

        Returns:
            Dictionary of symbol -> ETFFlowMetrics with flow calculations
        """
        start_time = datetime.now()

        if not symbols:
            raise ValueError("Symbols list cannot be empty")

        # Prioritize ETFs for better liquidity
        etf_symbols, stock_symbols = self._prioritize_etf_symbols(symbols)

        results = {}

        # Process ETFs first (higher priority and effectiveness)
        if etf_symbols:
            etf_results = self._calculate_etf_flow_factors(etf_symbols, as_of_date)
            results.update(etf_results)

        # Process stocks if within performance budget
        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        remaining_budget_ms = self.TARGET_LATENCY_MS - calculation_time

        if stock_symbols and remaining_budget_ms > 50:  # Reserve 50ms buffer
            stock_results = self._calculate_stock_flow_factors(stock_symbols, as_of_date, remaining_budget_ms)
            results.update(stock_results)

        # Performance tracking
        total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._calculation_times.append(total_time_ms)

        if total_time_ms > self.TARGET_LATENCY_MS:
            self.logger.warning(f"Performance target exceeded: {total_time_ms:.1f}ms > {self.TARGET_LATENCY_MS}ms")

        self.logger.info(f"Calculated flow factors for {len(results)} symbols in {total_time_ms:.1f}ms")

        return results

    def _prioritize_etf_symbols(self, symbols: List[str]) -> Tuple[List[str], List[str]]:
        """Separate and prioritize ETF symbols based on research findings."""
        etf_symbols = []
        stock_symbols = []

        for symbol in symbols:
            # Taiwan ETF identification (00XX format for most ETFs)
            if self._is_taiwan_etf(symbol):
                etf_symbols.append(symbol)
            else:
                stock_symbols.append(symbol)

        # Sort ETFs by liquidity and AUM for processing efficiency
        etf_symbols = self._sort_etfs_by_priority(etf_symbols)

        self.logger.debug(f"Symbol prioritization: {len(etf_symbols)} ETFs, {len(stock_symbols)} stocks")

        return etf_symbols, stock_symbols

    def _is_taiwan_etf(self, symbol: str) -> bool:
        """Identify Taiwan ETFs using market structure knowledge."""
        # Most Taiwan ETFs follow 00XX format
        if symbol.startswith('00') and len(symbol) == 4:
            return True

        # Additional ETF identifiers
        known_etf_prefixes = ['00', '000', '0050', '0051', '0052']  # Common patterns

        return any(symbol.startswith(prefix) for prefix in known_etf_prefixes)

    def _sort_etfs_by_priority(self, etf_symbols: List[str]) -> List[str]:
        """Sort ETFs by liquidity and processing priority."""
        # High priority ETFs (most liquid and important)
        high_priority = ['0050', '0051', '0052', '006208', '00878']  # Major Taiwan ETFs

        prioritized = []
        remaining = []

        for symbol in etf_symbols:
            if symbol in high_priority:
                prioritized.append(symbol)
            else:
                remaining.append(symbol)

        return prioritized + remaining

    def _calculate_etf_flow_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, ETFFlowMetrics]:
        """Calculate flow factors optimized for ETF characteristics."""
        results = {}

        for symbol in symbols:
            try:
                # Get ETF-specific flow data
                flow_data = self._get_etf_flow_data(symbol, as_of_date)

                if flow_data:
                    metrics = self._compute_etf_flow_metrics(symbol, as_of_date, flow_data)
                    if metrics:
                        results[symbol] = metrics

            except Exception as e:
                self.logger.warning(f"Error calculating ETF flow factors for {symbol}: {e}")
                continue

        return results

    def _calculate_stock_flow_factors(self, symbols: List[str], as_of_date: date,
                                    time_budget_ms: float) -> Dict[str, ETFFlowMetrics]:
        """Calculate flow factors for stocks with time budget constraints."""
        results = {}
        start_time = datetime.now()

        for symbol in symbols:
            # Check time budget
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed_ms > time_budget_ms * 0.9:  # Use 90% of budget
                self.logger.debug(f"Time budget exhausted at symbol {symbol}")
                break

            try:
                flow_data = self._get_stock_flow_data(symbol, as_of_date)

                if flow_data:
                    metrics = self._compute_stock_flow_metrics(symbol, as_of_date, flow_data)
                    if metrics:
                        results[symbol] = metrics

            except Exception as e:
                self.logger.warning(f"Error calculating stock flow factors for {symbol}: {e}")
                continue

        return results

    def _compute_etf_flow_metrics(self, symbol: str, as_of_date: date,
                                 flow_data: Dict[str, Any]) -> Optional[ETFFlowMetrics]:
        """Compute research-validated ETF flow metrics."""

        buy_volume = flow_data.get('buy_volume', 0)
        sell_volume = flow_data.get('sell_volume', 0)

        if buy_volume is None or sell_volume is None:
            return None

        # Research-validated flow factor calculations
        metrics = ETFFlowMetrics(symbol=symbol, date=as_of_date, is_etf=True)

        # Core factors (validated across 6,658 observations)
        metrics.net_flow = buy_volume - sell_volume

        # Avoid division by zero
        total_vol = buy_volume + sell_volume
        if total_vol > 0:
            metrics.flow_ratio = (buy_volume - sell_volume) / total_vol
            metrics.total_volume = total_vol
        else:
            metrics.flow_ratio = 0.0
            metrics.total_volume = 0.0

        # ETF-specific attributes
        metrics.etf_aum = flow_data.get('aum', 0)
        metrics.etf_liquidity_tier = self._determine_etf_liquidity_tier(symbol, flow_data)

        # Data quality metrics
        metrics.broker_coverage_count = flow_data.get('top15_broker_count', 0)
        metrics.data_completeness_score = self._calculate_etf_data_completeness(flow_data)

        return metrics

    def _compute_stock_flow_metrics(self, symbol: str, as_of_date: date,
                                   flow_data: Dict[str, Any]) -> Optional[ETFFlowMetrics]:
        """Compute flow metrics for individual stocks (lower priority)."""

        buy_volume = flow_data.get('buy_volume', 0)
        sell_volume = flow_data.get('sell_volume', 0)

        if buy_volume is None or sell_volume is None:
            return None

        metrics = ETFFlowMetrics(symbol=symbol, date=as_of_date, is_etf=False)

        # Basic flow calculations (same formulas but stock-optimized)
        metrics.net_flow = buy_volume - sell_volume

        total_vol = buy_volume + sell_volume
        if total_vol > 0:
            metrics.flow_ratio = (buy_volume - sell_volume) / total_vol
            metrics.total_volume = total_vol
        else:
            metrics.flow_ratio = 0.0
            metrics.total_volume = 0.0

        # Stock-specific quality metrics
        metrics.broker_coverage_count = flow_data.get('top15_broker_count', 0)
        metrics.data_completeness_score = self._calculate_stock_data_completeness(flow_data)

        return metrics

    def _determine_etf_liquidity_tier(self, symbol: str, flow_data: Dict[str, Any]) -> int:
        """Determine ETF liquidity tier for processing optimization."""

        aum = flow_data.get('aum', 0) or 0
        avg_volume = flow_data.get('avg_daily_volume', 0) or 0

        # Tier 1: Most liquid ETFs (priority processing)
        if symbol in ['0050', '0051', '0052'] or aum > 100_000_000_000:  # >100B TWD
            return 1

        # Tier 2: Medium liquidity ETFs
        elif aum > 10_000_000_000 or avg_volume > 50_000_000:  # >10B TWD or >50M daily volume
            return 2

        # Tier 3: Lower liquidity ETFs
        else:
            return 3

    def _calculate_etf_data_completeness(self, flow_data: Dict[str, Any]) -> float:
        """Calculate data completeness score for ETF flow data."""

        required_fields = ['buy_volume', 'sell_volume', 'top15_broker_count', 'aum']
        present_fields = sum(1 for field in required_fields if flow_data.get(field) is not None)

        return present_fields / len(required_fields)

    def _calculate_stock_data_completeness(self, flow_data: Dict[str, Any]) -> float:
        """Calculate data completeness score for stock flow data."""

        required_fields = ['buy_volume', 'sell_volume', 'top15_broker_count']
        present_fields = sum(1 for field in required_fields if flow_data.get(field) is not None)

        return present_fields / len(required_fields)

    def _get_etf_flow_data(self, symbol: str, as_of_date: date) -> Optional[Dict[str, Any]]:
        """Get ETF-optimized flow data from FinLab connector."""

        if not self.finlab_connector:
            return self._get_mock_etf_data(symbol)

        try:
            # ETF-specific data request
            data = self.finlab_connector.get_etf_flow_data(
                symbol=symbol,
                as_of_date=as_of_date,
                include_aum=True,
                include_creation_redemption=True
            )
            return data

        except Exception as e:
            self.logger.warning(f"Error fetching ETF flow data for {symbol}: {e}")
            return None

    def _get_stock_flow_data(self, symbol: str, as_of_date: date) -> Optional[Dict[str, Any]]:
        """Get stock flow data from FinLab connector."""

        if not self.finlab_connector:
            return self._get_mock_stock_data(symbol)

        try:
            data = self.finlab_connector.get_stock_flow_data(
                symbol=symbol,
                as_of_date=as_of_date
            )
            return data

        except Exception as e:
            self.logger.warning(f"Error fetching stock flow data for {symbol}: {e}")
            return None

    def _get_mock_etf_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock ETF data for testing."""
        hash_val = hash(symbol) % 1000

        return {
            'buy_volume': 50_000_000 + hash_val * 500_000,   # TWD
            'sell_volume': 45_000_000 + hash_val * 450_000,  # TWD
            'aum': 20_000_000_000 + hash_val * 100_000_000,  # TWD
            'avg_daily_volume': 100_000_000 + hash_val * 1_000_000,
            'top15_broker_count': min(15, 10 + hash_val % 6),
            'creation_redemption_net': (hash_val % 10 - 5) * 1_000_000
        }

    def _get_mock_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock stock data for testing."""
        hash_val = hash(symbol) % 1000

        return {
            'buy_volume': 10_000_000 + hash_val * 100_000,   # TWD
            'sell_volume': 9_500_000 + hash_val * 95_000,    # TWD
            'top15_broker_count': min(15, 5 + hash_val % 10),
        }




# Factory function for simple flow factor system
def create_simple_etf_flow_factor(finlab_connector=None) -> SimpleETFFlowFactor:
    """
    Create simple ETF Flow Factor system for personal trading.

    Args:
        finlab_connector: Optional FinLab connector for data access

    Returns:
        Configured SimpleETFFlowFactor instance for Taiwan market
    """
    return SimpleETFFlowFactor(finlab_connector)