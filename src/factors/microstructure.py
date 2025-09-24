"""
Market microstructure factors for Taiwan market ML pipeline.

This module implements microstructure factors focusing on:
1. Liquidity measures (4 factors)
2. Volume patterns (4 factors) 
3. Taiwan-specific market structure factors (4 factors)

Designed specifically for Taiwan market characteristics:
- Trading hours: 09:00-13:30 TST (4.5 hour session)
- Price limits: ±10% daily movement limits
- Settlement: T+2 cycle
- Foreign ownership: 50% caps with real-time tracking
- Tick size structure: Variable based on price levels
"""

from abc import abstractmethod
from datetime import date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
from decimal import Decimal

from .base import (
    FactorCalculator, FactorResult, FactorMetadata, 
    FactorCategory, FactorFrequency
)

# Import dependencies - will be mocked for testing if not available
try:
    from ..data.pipeline.pit_engine import PITQueryEngine, PITQuery
    from ..data.models.taiwan_market import TaiwanMarketCode, TradingStatus
    from ..data.core.temporal import TemporalValue, DataType
except ImportError:
    # For testing or standalone usage
    PITQueryEngine = object
    PITQuery = object
    TaiwanMarketCode = object
    TradingStatus = object
    TemporalValue = object
    DataType = object

logger = logging.getLogger(__name__)


@dataclass
class TaiwanMarketSession:
    """Taiwan market session information."""
    open_time: str = "09:00"
    close_time: str = "13:30"  
    session_minutes: int = 270  # 4.5 hours
    trading_days_per_year: int = 245
    price_limit: float = 0.10  # ±10%
    settlement_cycle: int = 2  # T+2


@dataclass 
class TickSizeStructure:
    """Taiwan tick size structure based on price levels."""
    
    @staticmethod
    def get_tick_size(price: float) -> float:
        """Get appropriate tick size for given price level."""
        if price < 10:
            return 0.01
        elif price < 50:
            return 0.05
        elif price < 100:
            return 0.10
        elif price < 500:
            return 0.50
        elif price < 1000:
            return 1.00
        else:
            return 5.00


class MicrostructureFactorCalculator(FactorCalculator):
    """Base class for microstructure factor calculations."""
    
    def __init__(self, pit_engine: PITQueryEngine, metadata: FactorMetadata):
        super().__init__(pit_engine, metadata)
        self.market_session = TaiwanMarketSession()
        self.tick_structure = TickSizeStructure()
    
    def _get_tick_data(self, symbols: List[str], as_of_date: date, 
                      lookback_days: int) -> pd.DataFrame:
        """Get high-frequency tick data for microstructure calculations."""
        return self._get_historical_data(
            symbols, as_of_date, DataType.TICK, lookback_days
        )
    
    def _get_order_book_data(self, symbols: List[str], as_of_date: date,
                           lookback_days: int) -> pd.DataFrame:
        """Get order book data for bid-ask analysis."""
        return self._get_historical_data(
            symbols, as_of_date, DataType.ORDER_BOOK, lookback_days
        )
    
    def _get_volume_data(self, symbols: List[str], as_of_date: date,
                        lookback_days: int) -> pd.DataFrame:
        """Get volume data for volume pattern analysis."""
        return self._get_historical_data(
            symbols, as_of_date, DataType.VOLUME, lookback_days
        )
    
    def _adjust_for_taiwan_session(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adjust data for Taiwan trading session (09:00-13:30)."""
        if 'timestamp' in data.columns:
            # Filter for Taiwan trading hours
            data = data.copy()
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['minute'] = pd.to_datetime(data['timestamp']).dt.minute
            
            # Keep only trading session data
            trading_mask = (
                ((data['hour'] == 9) & (data['minute'] >= 0)) |
                ((data['hour'].isin([10, 11, 12]))) |
                ((data['hour'] == 13) & (data['minute'] <= 30))
            )
            
            return data[trading_mask]
        
        return data
    
    def _calculate_daily_turnover(self, price_data: pd.DataFrame, 
                                volume_data: pd.DataFrame) -> pd.Series:
        """Calculate daily turnover (price * volume)."""
        merged = pd.merge(price_data, volume_data, 
                         left_index=True, right_index=True, how='inner')
        return (merged['close'] * merged['volume']).fillna(0)
    
    def _calculate_vwap(self, price_data: pd.DataFrame, 
                       volume_data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        merged = pd.merge(price_data, volume_data,
                         left_index=True, right_index=True, how='inner')
        
        if 'typical_price' in merged.columns:
            typical_price = merged['typical_price']
        else:
            typical_price = (merged['high'] + merged['low'] + merged['close']) / 3
        
        weighted_price = typical_price * merged['volume']
        total_volume = merged['volume']
        
        return (weighted_price.rolling(window=20, min_periods=5).sum() / 
                total_volume.rolling(window=20, min_periods=5).sum())


class MicrostructureFactors:
    """Container for all microstructure factor calculators."""
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.calculators = {}
        self._initialize_calculators()
    
    def _initialize_calculators(self):
        """Initialize all microstructure factor calculators."""
        # Will be populated as we add individual calculator classes
        pass
    
    def get_all_calculators(self) -> Dict[str, MicrostructureFactorCalculator]:
        """Get all microstructure factor calculators."""
        return self.calculators.copy()
    
    def calculate_all_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, FactorResult]:
        """Calculate all microstructure factors."""
        results = {}
        
        for name, calculator in self.calculators.items():
            try:
                result = calculator.calculate(symbols, as_of_date)
                results[name] = result
                logger.info(f"Calculated {name}: {result.coverage:.1%} coverage")
            except Exception as e:
                logger.error(f"Error calculating microstructure factor {name}: {e}")
                continue
        
        return results


# Data classes for Taiwan-specific market data
@dataclass
class ForeignFlowData:
    """Foreign institutional flow data."""
    date: date
    symbol: str
    foreign_buy_value: float
    foreign_sell_value: float
    foreign_net_value: float
    foreign_ownership_pct: float
    foreign_ownership_cap_pct: float = 50.0  # Taiwan standard cap


@dataclass 
class MarginTradingData:
    """Margin trading data."""
    date: date
    symbol: str
    margin_buy_volume: int
    margin_sell_volume: int  
    margin_balance: int
    margin_ratio_pct: float
    maintenance_margin_pct: float


@dataclass
class IndexCompositionData:
    """Taiwan index composition data."""
    date: date
    symbol: str
    taiex_weight: Optional[float] = None
    twse50_weight: Optional[float] = None
    tpex_weight: Optional[float] = None
    is_inclusion_candidate: bool = False
    days_to_rebalance: Optional[int] = None


# Placeholder for sentiment data - would integrate with external providers
@dataclass
class CrossStraitSentiment:
    """Cross-strait relation sentiment data."""
    date: date
    sentiment_score: float  # -1 (very negative) to +1 (very positive) 
    news_count: int
    keyword_mentions: Dict[str, int]
    volatility_impact: float