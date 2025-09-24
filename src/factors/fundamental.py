"""
Fundamental factor calculations for Taiwan market ML pipeline.

This module provides the foundation for fundamental factor calculations including
value, quality, and growth factors with Taiwan market specific adjustments.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
from decimal import Decimal

# Import base classes and dependencies
from .base import (
    FactorCalculator, FactorResult, FactorMetadata, FactorCategory, 
    FactorFrequency
)
from .taiwan_adjustments import TaiwanMarketAdjustments

try:
    from ..data.core.temporal import DataType, TemporalValue
    from ..data.pipeline.pit_engine import PITQueryEngine, PITQuery
    from ..data.models.taiwan_market import TaiwanMarketCode, TradingStatus
except ImportError:
    # For testing or standalone usage
    DataType = object
    TemporalValue = object
    PITQueryEngine = object
    PITQuery = object
    TaiwanMarketCode = object
    TradingStatus = object

logger = logging.getLogger(__name__)


@dataclass
class FinancialStatement:
    """Container for financial statement data."""
    symbol: str
    report_date: date
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    operating_income: Optional[float] = None
    ebitda: Optional[float] = None
    shares_outstanding: Optional[float] = None
    book_value_per_share: Optional[float] = None
    tangible_book_value: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.total_equity and self.shares_outstanding:
            self.book_value_per_share = self.total_equity / self.shares_outstanding


@dataclass  
class MarketData:
    """Container for market data."""
    symbol: str
    date: date
    price: float
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    shares_outstanding: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.price and self.shares_outstanding:
            self.market_cap = self.price * self.shares_outstanding


class FundamentalFactorCalculator(FactorCalculator):
    """
    Base class for fundamental factor calculations with Taiwan market compliance.
    
    This class handles:
    - 60-day reporting lag enforcement for quarterly financial statements
    - 90-day lag for annual reports
    - Taiwan GAAP compliance
    - TWD currency handling
    - Industry sector adjustments
    """
    
    # Taiwan financial reporting requirements
    QUARTERLY_REPORTING_LAG_DAYS = 60  # Q reports available 60 days after quarter end
    ANNUAL_REPORTING_LAG_DAYS = 90     # Annual reports available 90 days after year end
    MIN_FINANCIAL_HISTORY_QUARTERS = 8  # Minimum 2 years of quarterly data
    
    def __init__(self, pit_engine: PITQueryEngine, metadata: FactorMetadata,
                 taiwan_adjustments: Optional[TaiwanMarketAdjustments] = None):
        super().__init__(pit_engine, metadata)
        self.taiwan_adjustments = taiwan_adjustments or TaiwanMarketAdjustments()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _get_financial_data(self, symbols: List[str], as_of_date: date, 
                          quarters_back: int = 8) -> Dict[str, List[FinancialStatement]]:
        """
        Get financial statement data with proper Taiwan reporting lag enforcement.
        
        Args:
            symbols: List of stock symbols
            as_of_date: As-of date for point-in-time access
            quarters_back: Number of quarters of history to retrieve
            
        Returns:
            Dictionary mapping symbols to list of financial statements
        """
        # Enforce Taiwan reporting lag
        financial_cutoff_date = self._get_financial_cutoff_date(as_of_date)
        
        # Calculate lookback period
        start_date = financial_cutoff_date - timedelta(days=quarters_back * 90)
        
        self.logger.debug(f"Getting financial data from {start_date} to {financial_cutoff_date}")
        
        # Query financial statement data
        query = PITQuery(
            symbols=symbols,
            as_of_date=financial_cutoff_date,
            data_types=[DataType.FINANCIAL_STATEMENTS],
            start_date=start_date,
            end_date=financial_cutoff_date
        )
        
        raw_data = self.pit_engine.query(query)
        return self._parse_financial_data(raw_data)
    
    def _get_financial_cutoff_date(self, as_of_date: date) -> date:
        """
        Calculate the cutoff date for financial data availability.
        
        Taiwan financial reporting requirements:
        - Quarterly reports: Available 60 days after quarter end
        - Annual reports: Available 90 days after year end
        """
        # For simplicity, use the more conservative 90-day lag
        # In production, this would be more sophisticated based on report type
        cutoff_date = as_of_date - timedelta(days=self.ANNUAL_REPORTING_LAG_DAYS)
        
        self.logger.debug(f"Financial cutoff date: {cutoff_date} (90 days before {as_of_date})")
        return cutoff_date
    
    def _get_market_data(self, symbols: List[str], as_of_date: date,
                        lookback_days: int = 252) -> Dict[str, MarketData]:
        """
        Get market data (prices, market cap, etc.) for factor calculations.
        
        Args:
            symbols: List of symbols
            as_of_date: As-of date
            lookback_days: Lookback period for price data
            
        Returns:
            Dictionary mapping symbols to market data
        """
        # Get price data
        price_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, lookback_days
        )
        
        # Get shares outstanding and other market data
        market_query = PITQuery(
            symbols=symbols,
            as_of_date=as_of_date,
            data_types=[DataType.SHARES_OUTSTANDING, DataType.MARKET_CAP],
            start_date=as_of_date,
            end_date=as_of_date
        )
        
        market_raw = self.pit_engine.query(market_query)
        
        return self._parse_market_data(price_data, market_raw, as_of_date)
    
    def _parse_financial_data(self, raw_data: pd.DataFrame) -> Dict[str, List[FinancialStatement]]:
        """Parse raw financial data into FinancialStatement objects."""
        financial_data = {}
        
        for symbol in raw_data.get('symbol', pd.Series(dtype=str)).unique():
            if pd.isna(symbol):
                continue
                
            symbol_data = raw_data[raw_data['symbol'] == symbol].copy()
            statements = []
            
            for _, row in symbol_data.iterrows():
                try:
                    statement = FinancialStatement(
                        symbol=symbol,
                        report_date=pd.to_datetime(row.get('report_date')).date(),
                        revenue=self._safe_float(row.get('revenue')),
                        net_income=self._safe_float(row.get('net_income')),
                        total_assets=self._safe_float(row.get('total_assets')),
                        total_equity=self._safe_float(row.get('total_equity')),
                        total_debt=self._safe_float(row.get('total_debt')),
                        operating_income=self._safe_float(row.get('operating_income')),
                        ebitda=self._safe_float(row.get('ebitda')),
                        shares_outstanding=self._safe_float(row.get('shares_outstanding')),
                        operating_cash_flow=self._safe_float(row.get('operating_cash_flow')),
                        free_cash_flow=self._safe_float(row.get('free_cash_flow'))
                    )
                    statements.append(statement)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing financial data for {symbol}: {e}")
                    continue
            
            if statements:
                # Sort by report date (most recent first)
                statements.sort(key=lambda x: x.report_date, reverse=True)
                financial_data[symbol] = statements
        
        return financial_data
    
    def _parse_market_data(self, price_data: pd.DataFrame, market_raw: pd.DataFrame,
                          as_of_date: date) -> Dict[str, MarketData]:
        """Parse raw market data into MarketData objects."""
        market_data = {}
        
        for symbol in price_data.columns:
            try:
                # Get latest available price
                symbol_prices = price_data[symbol].dropna()
                if len(symbol_prices) == 0:
                    continue
                    
                latest_price = symbol_prices.iloc[-1]
                
                # Get shares outstanding and market cap from market_raw
                symbol_market = market_raw[market_raw.get('symbol', '') == symbol]
                shares_outstanding = None
                market_cap = None
                
                if len(symbol_market) > 0:
                    shares_outstanding = self._safe_float(symbol_market.iloc[0].get('shares_outstanding'))
                    market_cap = self._safe_float(symbol_market.iloc[0].get('market_cap'))
                
                market_data[symbol] = MarketData(
                    symbol=symbol,
                    date=as_of_date,
                    price=latest_price,
                    market_cap=market_cap,
                    shares_outstanding=shares_outstanding
                )
                
            except Exception as e:
                self.logger.warning(f"Error parsing market data for {symbol}: {e}")
                continue
        
        return market_data
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float, handling NaN and None."""
        if value is None or pd.isna(value):
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _calculate_growth_rate(self, current: Optional[float], previous: Optional[float]) -> Optional[float]:
        """Calculate growth rate between two values."""
        if current is None or previous is None or previous <= 0:
            return None
        
        return (current - previous) / previous
    
    def _calculate_trailing_metric(self, statements: List[FinancialStatement], 
                                 metric_name: str, quarters: int = 4) -> Optional[float]:
        """
        Calculate trailing sum for a metric (e.g., TTM revenue).
        
        Args:
            statements: List of financial statements (sorted by date desc)
            metric_name: Name of the metric to sum
            quarters: Number of quarters to include
            
        Returns:
            Trailing sum or None if insufficient data
        """
        if len(statements) < quarters:
            return None
        
        total = 0.0
        valid_quarters = 0
        
        for statement in statements[:quarters]:
            value = getattr(statement, metric_name, None)
            if value is not None:
                total += value
                valid_quarters += 1
        
        # Require at least 3 of 4 quarters for TTM calculations
        if valid_quarters < max(1, quarters * 0.75):
            return None
        
        return total
    
    def _get_industry_percentile(self, symbol: str, value: float, 
                               all_values: Dict[str, float]) -> Optional[float]:
        """
        Calculate industry percentile for a factor value.
        
        In production, this would use actual industry classifications.
        For now, it calculates percentile within the universe.
        """
        if not all_values or symbol not in all_values:
            return None
        
        values = list(all_values.values())
        finite_values = [v for v in values if np.isfinite(v)]
        
        if len(finite_values) < 10:  # Need minimum sample size
            return None
        
        return np.searchsorted(np.sort(finite_values), value) / len(finite_values)
    
    def _handle_outliers(self, values: Dict[str, float], 
                        method: str = "winsorize") -> Dict[str, float]:
        """Handle outliers in factor values."""
        if not values:
            return values
        
        value_array = np.array(list(values.values()))
        finite_mask = np.isfinite(value_array)
        
        if not finite_mask.any():
            return values
        
        finite_values = value_array[finite_mask]
        
        if method == "winsorize":
            # Winsorize at 1st and 99th percentiles
            p1 = np.percentile(finite_values, 1)
            p99 = np.percentile(finite_values, 99)
            
            clipped_values = np.clip(value_array, p1, p99)
            
            # Update the dictionary with clipped values
            symbols = list(values.keys())
            return {symbols[i]: clipped_values[i] for i in range(len(symbols))}
        
        elif method == "remove":
            # Remove extreme outliers (beyond 3 standard deviations)
            mean_val = np.mean(finite_values)
            std_val = np.std(finite_values)
            
            filtered_values = {}
            for symbol, value in values.items():
                if np.isfinite(value):
                    z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                    if z_score <= 3.0:
                        filtered_values[symbol] = value
                else:
                    filtered_values[symbol] = value
            
            return filtered_values
        
        return values
    
    def validate_financial_data_quality(self, financial_data: Dict[str, List[FinancialStatement]]) -> Dict[str, Any]:
        """
        Validate the quality of financial data for factor calculations.
        
        Returns:
            Dictionary with validation results and quality metrics
        """
        validation_results = {
            'total_symbols': len(financial_data),
            'symbols_with_data': 0,
            'avg_quarters_available': 0.0,
            'data_quality_score': 0.0,
            'missing_data_symbols': [],
            'insufficient_history_symbols': [],
            'quality_warnings': []
        }
        
        if not financial_data:
            return validation_results
        
        quarters_counts = []
        symbols_with_sufficient_data = 0
        
        for symbol, statements in financial_data.items():
            if not statements:
                validation_results['missing_data_symbols'].append(symbol)
                continue
            
            validation_results['symbols_with_data'] += 1
            quarters_counts.append(len(statements))
            
            if len(statements) >= self.MIN_FINANCIAL_HISTORY_QUARTERS:
                symbols_with_sufficient_data += 1
            else:
                validation_results['insufficient_history_symbols'].append(symbol)
        
        if quarters_counts:
            validation_results['avg_quarters_available'] = np.mean(quarters_counts)
        
        # Calculate data quality score
        if validation_results['total_symbols'] > 0:
            coverage_score = validation_results['symbols_with_data'] / validation_results['total_symbols']
            history_score = symbols_with_sufficient_data / validation_results['total_symbols']
            validation_results['data_quality_score'] = (coverage_score + history_score) / 2
        
        # Generate quality warnings
        if validation_results['data_quality_score'] < 0.8:
            validation_results['quality_warnings'].append(
                f"Low data quality score: {validation_results['data_quality_score']:.2f}"
            )
        
        if len(validation_results['missing_data_symbols']) > 0:
            validation_results['quality_warnings'].append(
                f"{len(validation_results['missing_data_symbols'])} symbols have no financial data"
            )
        
        return validation_results


class FundamentalFactors:
    """
    Orchestrator class for all fundamental factor calculations.
    
    This class manages the calculation of value, quality, and growth factors
    and provides a unified interface for fundamental factor computation.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.taiwan_adjustments = TaiwanMarketAdjustments()
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_fundamental_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, FactorResult]:
        """
        Calculate all 12 fundamental factors for given symbols and date.
        
        Returns:
            Dictionary mapping factor names to FactorResult objects
        """
        results = {}
        
        try:
            # This will be implemented when we create the specific factor modules
            from .value import ValueFactors
            from .quality import QualityFactors  
            from .growth import GrowthFactors
            
            # Calculate value factors (4 factors)
            value_calculator = ValueFactors(self.pit_engine)
            value_results = value_calculator.calculate_all_value_factors(symbols, as_of_date)
            results.update(value_results)
            
            # Calculate quality factors (4 factors)
            quality_calculator = QualityFactors(self.pit_engine)
            quality_results = quality_calculator.calculate_all_quality_factors(symbols, as_of_date)
            results.update(quality_results)
            
            # Calculate growth factors (4 factors)
            growth_calculator = GrowthFactors(self.pit_engine)
            growth_results = growth_calculator.calculate_all_growth_factors(symbols, as_of_date)
            results.update(growth_results)
            
        except ImportError as e:
            self.logger.error(f"Could not import factor modules: {e}")
            # Return empty results for now
            pass
        
        return results