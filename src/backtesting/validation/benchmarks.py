"""
Taiwan Market Benchmark Management for Walk-Forward Validation.

This module implements comprehensive benchmark management for Taiwan market
validation, including market benchmarks, sector benchmarks, style benchmarks,
and risk parity benchmarks specifically designed for Taiwan quantitative trading.

Key Features:
- Market Benchmarks: TAIEX, MSCI Taiwan, FTSE Taiwan
- Sector Benchmarks: Technology (TSE), Finance (TFB), Manufacturing
- Style Benchmarks: Growth vs Value, Large vs Small cap
- Risk Parity: Equal-weighted Taiwan universe
- Dynamic benchmark composition and rebalancing
- Historical benchmark performance tracking
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from decimal import Decimal
import json

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.models.taiwan_market import (
    TaiwanMarketCode, TaiwanTradingCalendar, create_taiwan_trading_calendar
)
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel
from .walk_forward import ValidationWindow, ValidationResult

logger = logging.getLogger(__name__)


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""
    MARKET = "market"
    SECTOR = "sector"
    STYLE = "style"
    RISK_PARITY = "risk_parity"
    CUSTOM = "custom"


class StyleType(Enum):
    """Style benchmark types."""
    GROWTH = "growth"
    VALUE = "value"
    LARGE_CAP = "large_cap"
    SMALL_CAP = "small_cap"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"


class RebalanceFrequency(Enum):
    """Benchmark rebalancing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark calculation."""
    # Rebalancing
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    rebalance_day: int = 1  # Day of month/week for rebalancing
    
    # Risk constraints
    max_single_weight: float = 0.1  # 10% max single stock weight
    max_sector_weight: float = 0.3  # 30% max sector weight
    min_market_cap: float = 1e9  # 1B TWD minimum market cap
    min_liquidity_adv: float = 1e6  # 1M TWD average daily volume
    
    # Style factor calculations
    lookback_periods: int = 252  # 1 year for factor calculations
    growth_factors: List[str] = field(default_factory=lambda: ["revenue_growth", "earnings_growth"])
    value_factors: List[str] = field(default_factory=lambda: ["pe_ratio", "pb_ratio", "ev_ebitda"])
    quality_factors: List[str] = field(default_factory=lambda: ["roe", "debt_to_equity", "current_ratio"])
    
    # Performance calculation
    include_dividends: bool = True
    transaction_costs: float = 0.001  # 0.1% transaction costs
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.max_single_weight <= 1:
            raise ValueError("Max single weight must be between 0 and 1")
        if not 0 < self.max_sector_weight <= 1:
            raise ValueError("Max sector weight must be between 0 and 1")


@dataclass
class BenchmarkConstituent:
    """A constituent of a benchmark."""
    symbol: str
    weight: float
    sector: str
    market_cap: float
    
    # Style characteristics
    growth_score: Optional[float] = None
    value_score: Optional[float] = None
    quality_score: Optional[float] = None
    momentum_score: Optional[float] = None
    
    # Metadata
    inclusion_date: Optional[date] = None
    exclusion_date: Optional[date] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'weight': self.weight,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'growth_score': self.growth_score,
            'value_score': self.value_score,
            'quality_score': self.quality_score,
            'momentum_score': self.momentum_score,
            'inclusion_date': self.inclusion_date.isoformat() if self.inclusion_date else None,
            'exclusion_date': self.exclusion_date.isoformat() if self.exclusion_date else None
        }


@dataclass
class BenchmarkDefinition:
    """Definition of a benchmark."""
    name: str
    category: BenchmarkCategory
    description: str
    
    # Composition rules
    universe_filter: Dict[str, Any]
    weighting_scheme: str  # "market_cap", "equal", "custom"
    
    # Style-specific parameters
    style_type: Optional[StyleType] = None
    sector_filter: Optional[List[str]] = None
    
    # Rebalancing
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    
    # Historical tracking
    inception_date: Optional[date] = None
    last_rebalance_date: Optional[date] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'category': self.category.value,
            'description': self.description,
            'universe_filter': self.universe_filter,
            'weighting_scheme': self.weighting_scheme,
            'style_type': self.style_type.value if self.style_type else None,
            'sector_filter': self.sector_filter,
            'rebalance_frequency': self.rebalance_frequency.value,
            'inception_date': self.inception_date.isoformat() if self.inception_date else None,
            'last_rebalance_date': self.last_rebalance_date.isoformat() if self.last_rebalance_date else None
        }


class BenchmarkCalculator:
    """
    Core benchmark calculation engine.
    
    Handles benchmark composition, weighting, and performance calculation
    with Taiwan market specifics and point-in-time constraints.
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        temporal_store: TemporalStore,
        pit_engine: PointInTimeEngine,
        taiwan_calendar: Optional[TaiwanTradingCalendar] = None
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        self.taiwan_calendar = taiwan_calendar or create_taiwan_trading_calendar()
        
        # Cache for benchmark data
        self._constituent_cache: Dict[str, Dict[date, List[BenchmarkConstituent]]] = {}
        self._returns_cache: Dict[str, Dict[date, float]] = {}
        
        logger.info("BenchmarkCalculator initialized")
    
    def calculate_benchmark_constituents(
        self,
        benchmark_def: BenchmarkDefinition,
        as_of_date: date,
        universe_symbols: List[str]
    ) -> List[BenchmarkConstituent]:
        """
        Calculate benchmark constituents as of a specific date.
        
        Args:
            benchmark_def: Benchmark definition
            as_of_date: Point-in-time date for calculation
            universe_symbols: Available universe of symbols
            
        Returns:
            List of benchmark constituents with weights
        """
        logger.info(f"Calculating {benchmark_def.name} constituents as of {as_of_date}")
        
        # Check cache first
        cache_key = f"{benchmark_def.name}_{as_of_date}"
        if benchmark_def.name in self._constituent_cache:
            if as_of_date in self._constituent_cache[benchmark_def.name]:
                return self._constituent_cache[benchmark_def.name][as_of_date]
        
        # Filter universe based on benchmark definition
        eligible_symbols = self._filter_universe(
            benchmark_def, as_of_date, universe_symbols
        )
        
        if not eligible_symbols:
            logger.warning(f"No eligible symbols for {benchmark_def.name} on {as_of_date}")
            return []
        
        # Get market data for eligible symbols
        market_data = self._get_market_data(eligible_symbols, as_of_date)
        
        # Calculate style scores if needed
        if benchmark_def.style_type:
            style_scores = self._calculate_style_scores(
                eligible_symbols, as_of_date, benchmark_def.style_type
            )
        else:
            style_scores = {}
        
        # Create constituents
        constituents = []
        for symbol in eligible_symbols:
            if symbol in market_data:
                data = market_data[symbol]
                
                constituent = BenchmarkConstituent(
                    symbol=symbol,
                    weight=0.0,  # Will be calculated below
                    sector=data.get('sector', 'Unknown'),
                    market_cap=data.get('market_cap', 0.0),
                    growth_score=style_scores.get(symbol, {}).get('growth'),
                    value_score=style_scores.get(symbol, {}).get('value'),
                    quality_score=style_scores.get(symbol, {}).get('quality'),
                    momentum_score=style_scores.get(symbol, {}).get('momentum'),
                    inclusion_date=as_of_date
                )
                constituents.append(constituent)
        
        # Calculate weights
        constituents = self._calculate_weights(constituents, benchmark_def)
        
        # Apply constraints
        constituents = self._apply_weight_constraints(constituents, benchmark_def)
        
        # Cache results
        if benchmark_def.name not in self._constituent_cache:
            self._constituent_cache[benchmark_def.name] = {}
        self._constituent_cache[benchmark_def.name][as_of_date] = constituents
        
        logger.info(f"Calculated {len(constituents)} constituents for {benchmark_def.name}")
        return constituents
    
    def calculate_benchmark_returns(
        self,
        benchmark_def: BenchmarkDefinition,
        start_date: date,
        end_date: date,
        universe_symbols: List[str]
    ) -> pd.Series:
        """
        Calculate benchmark returns over a period.
        
        Args:
            benchmark_def: Benchmark definition
            start_date: Start date for calculation
            end_date: End date for calculation
            universe_symbols: Available universe of symbols
            
        Returns:
            Series of benchmark returns indexed by date
        """
        logger.info(f"Calculating {benchmark_def.name} returns from {start_date} to {end_date}")
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(
            start_date, end_date, benchmark_def.rebalance_frequency
        )
        
        # Initialize results
        benchmark_returns = []
        current_constituents = None
        
        # Get all trading days in the period
        trading_days = self._get_trading_days(start_date, end_date)
        
        for trade_date in trading_days:
            # Check if rebalancing is needed
            if (current_constituents is None or 
                any(rebal_date <= trade_date for rebal_date in rebalance_dates 
                    if rebal_date > (trading_days[trading_days.index(trade_date)-1] if trading_days.index(trade_date) > 0 else start_date))):
                
                # Find the appropriate rebalance date
                rebal_date = trade_date
                for rdate in reversed(rebalance_dates):
                    if rdate <= trade_date:
                        rebal_date = rdate
                        break
                
                # Calculate new constituents
                current_constituents = self.calculate_benchmark_constituents(
                    benchmark_def, rebal_date, universe_symbols
                )
                
                logger.debug(f"Rebalanced {benchmark_def.name} on {rebal_date} with {len(current_constituents)} constituents")
            
            # Calculate daily return
            if current_constituents:
                daily_return = self._calculate_daily_benchmark_return(
                    current_constituents, trade_date
                )
                benchmark_returns.append(daily_return)
            else:
                benchmark_returns.append(0.0)
        
        # Create return series
        returns_series = pd.Series(
            benchmark_returns, 
            index=pd.to_datetime(trading_days),
            name=f"{benchmark_def.name}_returns"
        )
        
        logger.info(f"Calculated {len(benchmark_returns)} daily returns for {benchmark_def.name}")
        return returns_series
    
    def _filter_universe(
        self,
        benchmark_def: BenchmarkDefinition,
        as_of_date: date,
        universe_symbols: List[str]
    ) -> List[str]:
        """Filter universe based on benchmark definition criteria."""
        eligible_symbols = []
        
        for symbol in universe_symbols:
            try:
                # Get basic market data
                query = PITQuery(
                    symbols=[symbol],
                    as_of_date=as_of_date,
                    data_types=[DataType.PRICE, DataType.VOLUME, DataType.MARKET_CAP],
                    start_date=as_of_date - timedelta(days=30),
                    end_date=as_of_date
                )
                
                data = self.pit_engine.query(query)
                
                if symbol not in data or not data[symbol]:
                    continue
                
                # Get latest values
                latest_data = {}
                for temporal_value in data[symbol]:
                    if temporal_value.data_type not in latest_data:
                        latest_data[temporal_value.data_type] = temporal_value
                    elif temporal_value.value_date > latest_data[temporal_value.data_type].value_date:
                        latest_data[temporal_value.data_type] = temporal_value
                
                # Apply filters
                if DataType.MARKET_CAP in latest_data:
                    market_cap = float(latest_data[DataType.MARKET_CAP].value)
                    if market_cap < self.config.min_market_cap:
                        continue
                
                if DataType.VOLUME in latest_data:
                    # Check average daily volume over past 30 days
                    volume_data = [tv for tv in data[symbol] if tv.data_type == DataType.VOLUME]
                    if volume_data:
                        avg_volume = np.mean([float(tv.value) for tv in volume_data[-21:]])  # ~1 month
                        if avg_volume < self.config.min_liquidity_adv:
                            continue
                
                # Apply sector filter if specified
                if benchmark_def.sector_filter:
                    # This would need sector data from the temporal store
                    # For now, assume all symbols pass sector filter
                    pass
                
                eligible_symbols.append(symbol)
                
            except Exception as e:
                logger.warning(f"Failed to filter symbol {symbol}: {e}")
                continue
        
        return eligible_symbols
    
    def _get_market_data(
        self,
        symbols: List[str],
        as_of_date: date
    ) -> Dict[str, Dict[str, Any]]:
        """Get market data for symbols as of date."""
        market_data = {}
        
        for symbol in symbols:
            try:
                query = PITQuery(
                    symbols=[symbol],
                    as_of_date=as_of_date,
                    data_types=[DataType.PRICE, DataType.MARKET_CAP, DataType.VOLUME],
                    start_date=as_of_date - timedelta(days=1),
                    end_date=as_of_date
                )
                
                data = self.pit_engine.query(query)
                
                if symbol in data and data[symbol]:
                    # Extract latest values
                    symbol_data = {}
                    for temporal_value in data[symbol]:
                        if temporal_value.data_type == DataType.MARKET_CAP:
                            symbol_data['market_cap'] = float(temporal_value.value)
                        elif temporal_value.data_type == DataType.VOLUME:
                            symbol_data['volume'] = float(temporal_value.value)
                    
                    # Default sector (would be enhanced with actual sector data)
                    symbol_data['sector'] = self._infer_sector_from_symbol(symbol)
                    
                    market_data[symbol] = symbol_data
                
            except Exception as e:
                logger.warning(f"Failed to get market data for {symbol}: {e}")
                continue
        
        return market_data
    
    def _calculate_style_scores(
        self,
        symbols: List[str],
        as_of_date: date,
        style_type: StyleType
    ) -> Dict[str, Dict[str, float]]:
        """Calculate style scores for symbols."""
        style_scores = {}
        
        for symbol in symbols:
            try:
                scores = {}
                
                if style_type in [StyleType.GROWTH, StyleType.VALUE, StyleType.QUALITY]:
                    # Get fundamental data for style calculation
                    # This is a simplified implementation
                    query = PITQuery(
                        symbols=[symbol],
                        as_of_date=as_of_date,
                        data_types=[DataType.PRICE],  # Would include fundamental data types
                        start_date=as_of_date - timedelta(days=self.config.lookback_periods),
                        end_date=as_of_date
                    )
                    
                    data = self.pit_engine.query(query)
                    
                    if symbol in data and data[symbol]:
                        # Calculate style scores based on available data
                        # This is a placeholder - real implementation would use fundamentals
                        
                        if style_type == StyleType.GROWTH:
                            scores['growth'] = np.random.normal(0, 1)  # Placeholder
                        elif style_type == StyleType.VALUE:
                            scores['value'] = np.random.normal(0, 1)  # Placeholder
                        elif style_type == StyleType.QUALITY:
                            scores['quality'] = np.random.normal(0, 1)  # Placeholder
                        
                        # Momentum score (price-based)
                        if len(data[symbol]) >= 2:
                            prices = [float(tv.value) for tv in data[symbol] if tv.data_type == DataType.PRICE]
                            if len(prices) >= 2:
                                momentum = (prices[-1] / prices[0]) - 1
                                scores['momentum'] = momentum
                
                style_scores[symbol] = scores
                
            except Exception as e:
                logger.warning(f"Failed to calculate style scores for {symbol}: {e}")
                continue
        
        return style_scores
    
    def _calculate_weights(
        self,
        constituents: List[BenchmarkConstituent],
        benchmark_def: BenchmarkDefinition
    ) -> List[BenchmarkConstituent]:
        """Calculate weights for benchmark constituents."""
        if not constituents:
            return constituents
        
        if benchmark_def.weighting_scheme == "equal":
            # Equal weighting
            weight = 1.0 / len(constituents)
            for constituent in constituents:
                constituent.weight = weight
                
        elif benchmark_def.weighting_scheme == "market_cap":
            # Market cap weighting
            total_market_cap = sum(c.market_cap for c in constituents)
            if total_market_cap > 0:
                for constituent in constituents:
                    constituent.weight = constituent.market_cap / total_market_cap
            else:
                # Fallback to equal weighting
                weight = 1.0 / len(constituents)
                for constituent in constituents:
                    constituent.weight = weight
                    
        elif benchmark_def.weighting_scheme == "style":
            # Style-based weighting
            if benchmark_def.style_type == StyleType.GROWTH:
                scores = [c.growth_score or 0 for c in constituents]
            elif benchmark_def.style_type == StyleType.VALUE:
                scores = [c.value_score or 0 for c in constituents]
            elif benchmark_def.style_type == StyleType.QUALITY:
                scores = [c.quality_score or 0 for c in constituents]
            elif benchmark_def.style_type == StyleType.MOMENTUM:
                scores = [c.momentum_score or 0 for c in constituents]
            else:
                scores = [1.0] * len(constituents)
            
            # Convert to weights (higher scores get higher weights)
            min_score = min(scores)
            adjusted_scores = [s - min_score + 1 for s in scores]  # Ensure positive
            total_score = sum(adjusted_scores)
            
            if total_score > 0:
                for i, constituent in enumerate(constituents):
                    constituent.weight = adjusted_scores[i] / total_score
            else:
                # Fallback to equal weighting
                weight = 1.0 / len(constituents)
                for constituent in constituents:
                    constituent.weight = weight
        
        return constituents
    
    def _apply_weight_constraints(
        self,
        constituents: List[BenchmarkConstituent],
        benchmark_def: BenchmarkDefinition
    ) -> List[BenchmarkConstituent]:
        """Apply weight constraints to constituents."""
        if not constituents:
            return constituents
        
        # Apply maximum single weight constraint
        max_weight = self.config.max_single_weight
        total_excess = 0
        constrained_constituents = []
        unconstrained_constituents = []
        
        for constituent in constituents:
            if constituent.weight > max_weight:
                total_excess += constituent.weight - max_weight
                constituent.weight = max_weight
                constrained_constituents.append(constituent)
            else:
                unconstrained_constituents.append(constituent)
        
        # Redistribute excess weight to unconstrained constituents
        if total_excess > 0 and unconstrained_constituents:
            total_unconstrained_weight = sum(c.weight for c in unconstrained_constituents)
            
            if total_unconstrained_weight > 0:
                for constituent in unconstrained_constituents:
                    redistribution = total_excess * (constituent.weight / total_unconstrained_weight)
                    new_weight = constituent.weight + redistribution
                    
                    # Ensure the redistributed weight doesn't exceed max_weight
                    if new_weight > max_weight:
                        excess_again = new_weight - max_weight
                        constituent.weight = max_weight
                        # This excess would need further redistribution in a full implementation
                    else:
                        constituent.weight = new_weight
        
        # Normalize weights to ensure they sum to 1
        total_weight = sum(c.weight for c in constituents)
        if total_weight > 0:
            for constituent in constituents:
                constituent.weight /= total_weight
        
        return constituents
    
    def _generate_rebalance_dates(
        self,
        start_date: date,
        end_date: date,
        frequency: RebalanceFrequency
    ) -> List[date]:
        """Generate rebalance dates for the period."""
        rebalance_dates = []
        current_date = start_date
        
        while current_date <= end_date:
            if frequency == RebalanceFrequency.DAILY:
                rebalance_dates.append(current_date)
                current_date += timedelta(days=1)
            elif frequency == RebalanceFrequency.WEEKLY:
                if current_date.weekday() == self.config.rebalance_day:
                    rebalance_dates.append(current_date)
                current_date += timedelta(days=1)
            elif frequency == RebalanceFrequency.MONTHLY:
                if current_date.day == self.config.rebalance_day:
                    rebalance_dates.append(current_date)
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            elif frequency == RebalanceFrequency.QUARTERLY:
                if current_date.month in [1, 4, 7, 10] and current_date.day == self.config.rebalance_day:
                    rebalance_dates.append(current_date)
                current_date += timedelta(days=1)
            elif frequency == RebalanceFrequency.ANNUALLY:
                if current_date.month == 1 and current_date.day == self.config.rebalance_day:
                    rebalance_dates.append(current_date)
                current_date += timedelta(days=1)
        
        # Filter to trading days only
        trading_rebalance_dates = []
        for rebal_date in rebalance_dates:
            if self.taiwan_calendar.is_trading_day(rebal_date):
                trading_rebalance_dates.append(rebal_date)
            else:
                # Find next trading day
                next_trading_day = rebal_date
                while not self.taiwan_calendar.is_trading_day(next_trading_day):
                    next_trading_day += timedelta(days=1)
                    if next_trading_day > end_date:
                        break
                if next_trading_day <= end_date:
                    trading_rebalance_dates.append(next_trading_day)
        
        return trading_rebalance_dates
    
    def _get_trading_days(self, start_date: date, end_date: date) -> List[date]:
        """Get list of trading days between start and end dates."""
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.taiwan_calendar.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def _calculate_daily_benchmark_return(
        self,
        constituents: List[BenchmarkConstituent],
        trade_date: date
    ) -> float:
        """Calculate daily benchmark return."""
        if not constituents:
            return 0.0
        
        total_return = 0.0
        
        for constituent in constituents:
            try:
                # Get daily return for this constituent
                query = PITQuery(
                    symbols=[constituent.symbol],
                    as_of_date=trade_date,
                    data_types=[DataType.PRICE],
                    start_date=trade_date - timedelta(days=5),  # Get a few days
                    end_date=trade_date
                )
                
                data = self.pit_engine.query(query)
                
                if constituent.symbol in data and len(data[constituent.symbol]) >= 2:
                    prices = [float(tv.value) for tv in data[constituent.symbol] 
                             if tv.data_type == DataType.PRICE]
                    
                    if len(prices) >= 2:
                        daily_return = (prices[-1] / prices[-2]) - 1
                        total_return += constituent.weight * daily_return
                
            except Exception as e:
                logger.warning(f"Failed to calculate return for {constituent.symbol} on {trade_date}: {e}")
                continue
        
        return total_return
    
    def _infer_sector_from_symbol(self, symbol: str) -> str:
        """Infer sector from symbol (simplified implementation)."""
        # This is a placeholder - real implementation would use sector data
        if symbol.startswith("23"):  # TSE Technology stocks often start with 23
            return "Technology"
        elif symbol.startswith("28"):  # TSE Financial stocks often start with 28
            return "Financial"
        elif symbol.startswith("15"):  # Manufacturing
            return "Manufacturing"
        else:
            return "Other"


class TaiwanBenchmarkManager:
    """
    Taiwan market benchmark manager.
    
    Provides predefined Taiwan market benchmarks and manages
    benchmark calculations for walk-forward validation.
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        temporal_store: TemporalStore,
        pit_engine: PointInTimeEngine
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        
        # Initialize benchmark calculator
        self.calculator = BenchmarkCalculator(config, temporal_store, pit_engine)
        
        # Define standard Taiwan benchmarks
        self.standard_benchmarks = self._create_standard_benchmarks()
        
        logger.info("TaiwanBenchmarkManager initialized")
    
    def _create_standard_benchmarks(self) -> Dict[str, BenchmarkDefinition]:
        """Create standard Taiwan market benchmarks."""
        benchmarks = {}
        
        # Market Benchmarks
        benchmarks["TAIEX"] = BenchmarkDefinition(
            name="TAIEX",
            category=BenchmarkCategory.MARKET,
            description="Taiwan Stock Exchange Capitalization Weighted Index",
            universe_filter={"exchange": "TSE", "market_cap_min": 1e9},
            weighting_scheme="market_cap",
            rebalance_frequency=RebalanceFrequency.QUARTERLY
        )
        
        benchmarks["MSCI_Taiwan"] = BenchmarkDefinition(
            name="MSCI_Taiwan",
            category=BenchmarkCategory.MARKET,
            description="MSCI Taiwan Index",
            universe_filter={"exchange": "TSE", "market_cap_min": 5e9, "liquidity_screen": True},
            weighting_scheme="market_cap",
            rebalance_frequency=RebalanceFrequency.QUARTERLY
        )
        
        benchmarks["FTSE_Taiwan"] = BenchmarkDefinition(
            name="FTSE_Taiwan",
            category=BenchmarkCategory.MARKET,
            description="FTSE Taiwan Index",
            universe_filter={"exchange": "TSE", "market_cap_min": 3e9},
            weighting_scheme="market_cap",
            rebalance_frequency=RebalanceFrequency.QUARTERLY
        )
        
        # Sector Benchmarks
        benchmarks["Taiwan_Technology"] = BenchmarkDefinition(
            name="Taiwan_Technology",
            category=BenchmarkCategory.SECTOR,
            description="Taiwan Technology Sector Index",
            universe_filter={"sector": "Technology"},
            weighting_scheme="market_cap",
            sector_filter=["Technology", "Semiconductors", "Electronics"],
            rebalance_frequency=RebalanceFrequency.MONTHLY
        )
        
        benchmarks["Taiwan_Financial"] = BenchmarkDefinition(
            name="Taiwan_Financial",
            category=BenchmarkCategory.SECTOR,
            description="Taiwan Financial Sector Index",
            universe_filter={"sector": "Financial"},
            weighting_scheme="market_cap",
            sector_filter=["Banking", "Insurance", "Securities"],
            rebalance_frequency=RebalanceFrequency.MONTHLY
        )
        
        benchmarks["Taiwan_Manufacturing"] = BenchmarkDefinition(
            name="Taiwan_Manufacturing",
            category=BenchmarkCategory.SECTOR,
            description="Taiwan Manufacturing Sector Index",
            universe_filter={"sector": "Manufacturing"},
            weighting_scheme="market_cap",
            sector_filter=["Manufacturing", "Industrial", "Materials"],
            rebalance_frequency=RebalanceFrequency.MONTHLY
        )
        
        # Style Benchmarks
        benchmarks["Taiwan_Growth"] = BenchmarkDefinition(
            name="Taiwan_Growth",
            category=BenchmarkCategory.STYLE,
            description="Taiwan Growth Style Index",
            universe_filter={"market_cap_min": 1e9},
            weighting_scheme="style",
            style_type=StyleType.GROWTH,
            rebalance_frequency=RebalanceFrequency.QUARTERLY
        )
        
        benchmarks["Taiwan_Value"] = BenchmarkDefinition(
            name="Taiwan_Value",
            category=BenchmarkCategory.STYLE,
            description="Taiwan Value Style Index",
            universe_filter={"market_cap_min": 1e9},
            weighting_scheme="style",
            style_type=StyleType.VALUE,
            rebalance_frequency=RebalanceFrequency.QUARTERLY
        )
        
        benchmarks["Taiwan_Small_Cap"] = BenchmarkDefinition(
            name="Taiwan_Small_Cap",
            category=BenchmarkCategory.STYLE,
            description="Taiwan Small Cap Index",
            universe_filter={"market_cap_min": 1e8, "market_cap_max": 5e9},
            weighting_scheme="market_cap",
            style_type=StyleType.SMALL_CAP,
            rebalance_frequency=RebalanceFrequency.QUARTERLY
        )
        
        benchmarks["Taiwan_Large_Cap"] = BenchmarkDefinition(
            name="Taiwan_Large_Cap",
            category=BenchmarkCategory.STYLE,
            description="Taiwan Large Cap Index",
            universe_filter={"market_cap_min": 50e9},
            weighting_scheme="market_cap",
            style_type=StyleType.LARGE_CAP,
            rebalance_frequency=RebalanceFrequency.QUARTERLY
        )
        
        # Risk Parity Benchmark
        benchmarks["Taiwan_Equal_Weight"] = BenchmarkDefinition(
            name="Taiwan_Equal_Weight",
            category=BenchmarkCategory.RISK_PARITY,
            description="Taiwan Equal-Weighted Universe",
            universe_filter={"market_cap_min": 1e9, "liquidity_min": 1e6},
            weighting_scheme="equal",
            rebalance_frequency=RebalanceFrequency.MONTHLY
        )
        
        return benchmarks
    
    def get_benchmark_returns(
        self,
        benchmark_name: str,
        start_date: date,
        end_date: date,
        universe_symbols: List[str]
    ) -> pd.Series:
        """
        Get benchmark returns for validation.
        
        Args:
            benchmark_name: Name of benchmark
            start_date: Start date for returns
            end_date: End date for returns
            universe_symbols: Available universe of symbols
            
        Returns:
            Series of benchmark returns
        """
        if benchmark_name not in self.standard_benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        benchmark_def = self.standard_benchmarks[benchmark_name]
        
        return self.calculator.calculate_benchmark_returns(
            benchmark_def, start_date, end_date, universe_symbols
        )
    
    def get_all_benchmark_returns(
        self,
        start_date: date,
        end_date: date,
        universe_symbols: List[str],
        benchmark_categories: Optional[List[BenchmarkCategory]] = None
    ) -> Dict[str, pd.Series]:
        """
        Get returns for all benchmarks in specified categories.
        
        Args:
            start_date: Start date for returns
            end_date: End date for returns
            universe_symbols: Available universe of symbols
            benchmark_categories: Categories to include (default: all)
            
        Returns:
            Dictionary of benchmark returns
        """
        if benchmark_categories is None:
            benchmark_categories = list(BenchmarkCategory)
        
        benchmark_returns = {}
        
        for name, benchmark_def in self.standard_benchmarks.items():
            if benchmark_def.category in benchmark_categories:
                try:
                    returns = self.get_benchmark_returns(
                        name, start_date, end_date, universe_symbols
                    )
                    benchmark_returns[name] = returns
                    
                except Exception as e:
                    logger.error(f"Failed to calculate returns for {name}: {e}")
                    continue
        
        return benchmark_returns
    
    def add_custom_benchmark(
        self,
        benchmark_def: BenchmarkDefinition
    ) -> None:
        """Add a custom benchmark definition."""
        self.standard_benchmarks[benchmark_def.name] = benchmark_def
        logger.info(f"Added custom benchmark: {benchmark_def.name}")
    
    def get_benchmark_definition(self, benchmark_name: str) -> Optional[BenchmarkDefinition]:
        """Get benchmark definition by name."""
        return self.standard_benchmarks.get(benchmark_name)
    
    def list_available_benchmarks(
        self,
        category: Optional[BenchmarkCategory] = None
    ) -> List[str]:
        """List available benchmark names, optionally filtered by category."""
        if category is None:
            return list(self.standard_benchmarks.keys())
        
        return [
            name for name, benchmark_def in self.standard_benchmarks.items()
            if benchmark_def.category == category
        ]


# Utility functions
def create_default_benchmark_config(**overrides) -> BenchmarkConfig:
    """Create default benchmark configuration with Taiwan market settings."""
    config = BenchmarkConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    return config


def create_taiwan_benchmark_manager(
    temporal_store: TemporalStore,
    pit_engine: PointInTimeEngine,
    **config_overrides
) -> TaiwanBenchmarkManager:
    """Create Taiwan benchmark manager with default configuration."""
    config = create_default_benchmark_config(**config_overrides)
    return TaiwanBenchmarkManager(config, temporal_store, pit_engine)


# Example usage and testing
if __name__ == "__main__":
    print("Taiwan Benchmark Management demo")
    
    # This would be called with actual TemporalStore and PointInTimeEngine
    print("Demo would initialize with real data stores")
    
    # Create sample benchmark definition
    sample_benchmark = BenchmarkDefinition(
        name="Sample_Taiwan_Benchmark",
        category=BenchmarkCategory.CUSTOM,
        description="Sample Taiwan benchmark for testing",
        universe_filter={"market_cap_min": 1e9},
        weighting_scheme="market_cap",
        rebalance_frequency=RebalanceFrequency.MONTHLY
    )
    
    print(f"Sample benchmark: {sample_benchmark.name}")
    print(f"Category: {sample_benchmark.category.value}")
    print(f"Weighting: {sample_benchmark.weighting_scheme}")
    print(f"Rebalance frequency: {sample_benchmark.rebalance_frequency.value}")