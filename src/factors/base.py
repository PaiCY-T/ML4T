"""
Base classes for factor calculation in the Taiwan market ML pipeline.

This module provides the foundation for all factor calculations, including
base classes, data structures, and common utilities.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from decimal import Decimal

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


class FactorCategory(Enum):
    """Factor categories."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MICROSTRUCTURE = "microstructure"


class FactorFrequency(Enum):
    """Factor calculation frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class FactorMetadata:
    """Metadata for factor definitions."""
    name: str
    category: FactorCategory
    frequency: FactorFrequency
    description: str
    lookback_days: int
    data_requirements: List[DataType]
    taiwan_specific: bool = False
    min_history_days: int = 252  # Default 1 year
    expected_ic: Optional[float] = None
    expected_turnover: Optional[float] = None


@dataclass
class FactorResult:
    """Result container for factor calculations."""
    factor_name: str
    date: date
    values: Dict[str, float]  # symbol -> factor_value
    metadata: FactorMetadata
    calculation_time: Optional[datetime] = None
    coverage: Optional[float] = None  # Percentage of universe covered
    percentile_ranks: Optional[Dict[str, float]] = None  # symbol -> percentile
    z_scores: Optional[Dict[str, float]] = None  # symbol -> z_score
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.values:
            self._calculate_derived_metrics()
    
    def _calculate_derived_metrics(self):
        """Calculate percentile ranks and z-scores."""
        values_array = np.array(list(self.values.values()))
        
        # Remove NaN values for ranking
        finite_values = values_array[np.isfinite(values_array)]
        if len(finite_values) == 0:
            return
            
        # Calculate percentile ranks
        self.percentile_ranks = {}
        for symbol, value in self.values.items():
            if np.isfinite(value):
                percentile = np.searchsorted(np.sort(finite_values), value) / len(finite_values)
                self.percentile_ranks[symbol] = percentile
        
        # Calculate z-scores
        if len(finite_values) > 1:
            mean_val = np.mean(finite_values)
            std_val = np.std(finite_values)
            
            if std_val > 0:
                self.z_scores = {}
                for symbol, value in self.values.items():
                    if np.isfinite(value):
                        z_score = (value - mean_val) / std_val
                        self.z_scores[symbol] = z_score
        
        # Calculate coverage
        total_symbols = len(self.values)
        valid_values = len(finite_values)
        self.coverage = valid_values / total_symbols if total_symbols > 0 else 0.0


class FactorCalculator(ABC):
    """Abstract base class for factor calculations."""
    
    def __init__(self, pit_engine: PITQueryEngine, metadata: FactorMetadata):
        self.pit_engine = pit_engine
        self.metadata = metadata
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """
        Calculate factor values for given symbols and date.
        
        Args:
            symbols: List of stock symbols
            as_of_date: Calculation date
            universe_data: Optional pre-loaded universe data
            
        Returns:
            FactorResult with calculated values
        """
        pass
    
    def _get_historical_data(self, symbols: List[str], as_of_date: date,
                           data_type: DataType, lookback_days: int) -> pd.DataFrame:
        """
        Get historical data using point-in-time engine.
        
        Args:
            symbols: List of symbols
            as_of_date: As-of date
            data_type: Type of data to retrieve
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with historical data
        """
        start_date = as_of_date - timedelta(days=lookback_days)
        
        query = PITQuery(
            symbols=symbols,
            as_of_date=as_of_date,
            data_types=[data_type],
            start_date=start_date,
            end_date=as_of_date
        )
        
        return self.pit_engine.query(query)
    
    def _handle_missing_data(self, data: pd.DataFrame, 
                           method: str = "forward_fill") -> pd.DataFrame:
        """Handle missing data in factor calculations."""
        if method == "forward_fill":
            return data.fillna(method='ffill')
        elif method == "interpolate":
            return data.interpolate()
        elif method == "drop":
            return data.dropna()
        else:
            return data
    
    def _winsorize(self, values: np.ndarray, 
                  lower_percentile: float = 0.01,
                  upper_percentile: float = 0.99) -> np.ndarray:
        """Winsorize extreme values."""
        lower_bound = np.percentile(values, lower_percentile * 100)
        upper_bound = np.percentile(values, upper_percentile * 100)
        return np.clip(values, lower_bound, upper_bound)


class FactorEngine:
    """
    Main factor calculation engine that orchestrates all factor calculations.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.calculators: Dict[str, FactorCalculator] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_calculator(self, calculator: FactorCalculator):
        """Register a factor calculator."""
        self.calculators[calculator.metadata.name] = calculator
        self.logger.info(f"Registered factor calculator: {calculator.metadata.name}")
    
    def calculate_factor(self, factor_name: str, symbols: List[str], 
                        as_of_date: date, **kwargs) -> FactorResult:
        """Calculate a specific factor."""
        if factor_name not in self.calculators:
            raise ValueError(f"Factor calculator '{factor_name}' not found")
        
        calculator = self.calculators[factor_name]
        return calculator.calculate(symbols, as_of_date, **kwargs)
    
    def calculate_all_factors(self, symbols: List[str], as_of_date: date,
                            categories: Optional[List[FactorCategory]] = None) -> Dict[str, FactorResult]:
        """
        Calculate all registered factors for given symbols and date.
        
        Args:
            symbols: List of symbols
            as_of_date: Calculation date
            categories: Optional filter by factor categories
            
        Returns:
            Dictionary mapping factor names to results
        """
        results = {}
        
        for name, calculator in self.calculators.items():
            if categories and calculator.metadata.category not in categories:
                continue
            
            try:
                start_time = datetime.now()
                result = calculator.calculate(symbols, as_of_date)
                result.calculation_time = datetime.now()
                
                results[name] = result
                
                elapsed = (datetime.now() - start_time).total_seconds()
                self.logger.info(
                    f"Calculated {name}: {result.coverage:.1%} coverage "
                    f"in {elapsed:.2f}s"
                )
                
            except Exception as e:
                self.logger.error(f"Error calculating factor {name}: {e}")
                continue
        
        return results
    
    def get_factor_metadata(self) -> Dict[str, FactorMetadata]:
        """Get metadata for all registered factors."""
        return {name: calc.metadata for name, calc in self.calculators.items()}
    
    def validate_factor_requirements(self, as_of_date: date) -> Dict[str, bool]:
        """Validate that all factor requirements are met."""
        validation_results = {}
        
        for name, calculator in self.calculators.items():
            try:
                # Check minimum history requirements
                min_date = as_of_date - timedelta(days=calculator.metadata.min_history_days)
                
                # Check data availability (simplified check)
                available = True  # Would implement actual data availability check
                
                validation_results[name] = available
                
            except Exception as e:
                self.logger.error(f"Validation failed for {name}: {e}")
                validation_results[name] = False
        
        return validation_results


class TechnicalFactorCalculator(FactorCalculator):
    """Base class for technical factor calculations."""
    
    def __init__(self, pit_engine: PITQueryEngine, metadata: FactorMetadata):
        super().__init__(pit_engine, metadata)
    
    def _get_ohlcv_data(self, symbols: List[str], as_of_date: date, 
                       lookback_days: int) -> pd.DataFrame:
        """Get OHLCV data for technical calculations."""
        return self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, lookback_days
        )
    
    def _calculate_returns(self, prices: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate returns over specified periods."""
        return prices.pct_change(periods=periods)
    
    def _calculate_rolling_volatility(self, returns: pd.Series, 
                                    window: int) -> pd.Series:
        """Calculate rolling volatility."""
        return returns.rolling(window=window, min_periods=window//2).std() * np.sqrt(252)
    
    def _calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=window, min_periods=window//2).mean()
    
    def _calculate_ema(self, prices: pd.Series, span: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=span, adjust=False).mean()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, 
                                 std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self._calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * std_mult)
        lower_band = sma - (std * std_mult)
        return upper_band, sma, lower_band
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram