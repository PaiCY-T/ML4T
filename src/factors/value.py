"""
Value factor calculations for Taiwan market ML pipeline.

This module implements 4 value factors:
1. P/E Ratio (Trailing twelve months and forward P/E)
2. P/B Ratio (Price-to-book with industry adjustment)
3. EV/EBITDA (Enterprise value to EBITDA multiple)
4. Price/Sales (Price-to-sales ratio with sector normalization)
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
import pandas as pd

from .base import FactorMetadata, FactorCategory, FactorFrequency, FactorResult
from .fundamental import FundamentalFactorCalculator, FinancialStatement, MarketData
from .taiwan_adjustments import TaiwanMarketAdjustments

try:
    from ..data.core.temporal import DataType
    from ..data.pipeline.pit_engine import PITQueryEngine
except ImportError:
    DataType = object
    PITQueryEngine = object

logger = logging.getLogger(__name__)


class PERatioCalculator(FundamentalFactorCalculator):
    """
    P/E Ratio factor calculator.
    
    Calculates both trailing twelve months (TTM) P/E and forward P/E ratios
    with Taiwan market adjustments.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="pe_ratio_ttm",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Trailing twelve months Price-to-Earnings ratio",
            lookback_days=365,
            data_requirements=[DataType.FINANCIAL_STATEMENTS, DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=365,
            expected_ic=0.03,
            expected_turnover=0.15
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate P/E ratio for given symbols."""
        
        # Get financial and market data
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=4)
        market_data = self._get_market_data(symbols, as_of_date)
        
        pe_values = {}
        
        for symbol in symbols:
            pe_ratio = self._calculate_pe_ratio(symbol, financial_data, market_data)
            if pe_ratio is not None:
                pe_values[symbol] = pe_ratio
        
        # Handle outliers and apply Taiwan market adjustments
        pe_values = self._handle_outliers(pe_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=pe_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_pe_ratio(self, symbol: str, 
                          financial_data: Dict[str, List[FinancialStatement]],
                          market_data: Dict[str, MarketData]) -> Optional[float]:
        """Calculate P/E ratio for a specific symbol."""
        
        if symbol not in financial_data or symbol not in market_data:
            return None
        
        statements = financial_data[symbol]
        market = market_data[symbol]
        
        if len(statements) < 4:  # Need at least 4 quarters for TTM
            return None
        
        # Calculate TTM earnings
        ttm_earnings = self._calculate_trailing_metric(statements, 'net_income', quarters=4)
        
        if ttm_earnings is None or ttm_earnings <= 0:
            return None
        
        # Calculate P/E ratio
        if market.market_cap is not None:
            pe_ratio = market.market_cap / ttm_earnings
        elif market.price is not None and market.shares_outstanding is not None:
            earnings_per_share = ttm_earnings / market.shares_outstanding
            pe_ratio = market.price / earnings_per_share
        else:
            return None
        
        # Apply reasonable bounds for P/E ratios
        if pe_ratio < 0 or pe_ratio > 1000:  # Negative or extremely high P/E
            return None
        
        return pe_ratio


class PBRatioCalculator(FundamentalFactorCalculator):
    """
    P/B Ratio factor calculator with industry adjustment.
    
    Calculates price-to-book ratio and applies industry sector adjustments
    to account for different capital intensity across sectors.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="pb_ratio_adjusted",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Price-to-Book ratio with industry adjustment",
            lookback_days=365,
            data_requirements=[DataType.FINANCIAL_STATEMENTS, DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=365,
            expected_ic=0.025,
            expected_turnover=0.12
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate P/B ratio for given symbols."""
        
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=1)
        market_data = self._get_market_data(symbols, as_of_date)
        
        pb_values = {}
        
        for symbol in symbols:
            pb_ratio = self._calculate_pb_ratio(symbol, financial_data, market_data)
            if pb_ratio is not None:
                pb_values[symbol] = pb_ratio
        
        # Apply industry adjustment
        pb_values = self._apply_industry_adjustment(pb_values)
        
        # Handle outliers
        pb_values = self._handle_outliers(pb_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=pb_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_pb_ratio(self, symbol: str,
                          financial_data: Dict[str, List[FinancialStatement]],
                          market_data: Dict[str, MarketData]) -> Optional[float]:
        """Calculate P/B ratio for a specific symbol."""
        
        if symbol not in financial_data or symbol not in market_data:
            return None
        
        statements = financial_data[symbol]
        market = market_data[symbol]
        
        if not statements:
            return None
        
        # Use most recent book value
        latest_statement = statements[0]
        
        if latest_statement.total_equity is None or latest_statement.total_equity <= 0:
            return None
        
        # Calculate P/B ratio
        if market.market_cap is not None:
            pb_ratio = market.market_cap / latest_statement.total_equity
        elif (market.price is not None and 
              latest_statement.book_value_per_share is not None and
              latest_statement.book_value_per_share > 0):
            pb_ratio = market.price / latest_statement.book_value_per_share
        else:
            return None
        
        # Apply reasonable bounds
        if pb_ratio <= 0 or pb_ratio > 100:
            return None
        
        return pb_ratio
    
    def _apply_industry_adjustment(self, pb_values: Dict[str, float]) -> Dict[str, float]:
        """
        Apply industry adjustment to P/B ratios.
        
        In production, this would use actual industry classifications.
        For now, we calculate relative P/B within the universe.
        """
        if not pb_values:
            return pb_values
        
        # Calculate median P/B ratio
        values = list(pb_values.values())
        finite_values = [v for v in values if np.isfinite(v)]
        
        if len(finite_values) < 10:
            return pb_values
        
        median_pb = np.median(finite_values)
        
        # Adjust P/B ratios relative to median (simple industry adjustment)
        adjusted_values = {}
        for symbol, pb_ratio in pb_values.items():
            if np.isfinite(pb_ratio) and median_pb > 0:
                adjusted_pb = pb_ratio / median_pb
                adjusted_values[symbol] = adjusted_pb
            else:
                adjusted_values[symbol] = pb_ratio
        
        return adjusted_values


class EVEBITDACalculator(FundamentalFactorCalculator):
    """
    EV/EBITDA multiple calculator.
    
    Calculates Enterprise Value to EBITDA ratio, which is useful for
    comparing companies with different capital structures.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="ev_ebitda_ttm",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Enterprise Value to EBITDA multiple (TTM)",
            lookback_days=365,
            data_requirements=[DataType.FINANCIAL_STATEMENTS, DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=365,
            expected_ic=0.028,
            expected_turnover=0.14
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate EV/EBITDA for given symbols."""
        
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=4)
        market_data = self._get_market_data(symbols, as_of_date)
        
        ev_ebitda_values = {}
        
        for symbol in symbols:
            ev_ebitda = self._calculate_ev_ebitda(symbol, financial_data, market_data)
            if ev_ebitda is not None:
                ev_ebitda_values[symbol] = ev_ebitda
        
        # Handle outliers
        ev_ebitda_values = self._handle_outliers(ev_ebitda_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=ev_ebitda_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_ev_ebitda(self, symbol: str,
                           financial_data: Dict[str, List[FinancialStatement]],
                           market_data: Dict[str, MarketData]) -> Optional[float]:
        """Calculate EV/EBITDA for a specific symbol."""
        
        if symbol not in financial_data or symbol not in market_data:
            return None
        
        statements = financial_data[symbol]
        market = market_data[symbol]
        
        if len(statements) < 4:  # Need TTM data
            return None
        
        # Calculate TTM EBITDA
        ttm_ebitda = self._calculate_trailing_metric(statements, 'ebitda', quarters=4)
        
        if ttm_ebitda is None or ttm_ebitda <= 0:
            # Try to calculate EBITDA from available components
            ttm_ebitda = self._estimate_ebitda_from_components(statements)
        
        if ttm_ebitda is None or ttm_ebitda <= 0:
            return None
        
        # Calculate Enterprise Value
        enterprise_value = self._calculate_enterprise_value(symbol, financial_data, market_data)
        
        if enterprise_value is None or enterprise_value <= 0:
            return None
        
        ev_ebitda = enterprise_value / ttm_ebitda
        
        # Apply reasonable bounds
        if ev_ebitda < 0 or ev_ebitda > 500:
            return None
        
        return ev_ebitda
    
    def _estimate_ebitda_from_components(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Estimate EBITDA from operating income if EBITDA is not available."""
        
        # Simple estimation: Operating Income + Depreciation
        # In production, we would have more sophisticated methods
        ttm_operating_income = self._calculate_trailing_metric(statements, 'operating_income', quarters=4)
        
        if ttm_operating_income is None:
            return None
        
        # Rough estimate: assume depreciation is ~5% of operating income for Taiwan companies
        estimated_ebitda = ttm_operating_income * 1.05
        
        return estimated_ebitda
    
    def _calculate_enterprise_value(self, symbol: str,
                                  financial_data: Dict[str, List[FinancialStatement]],
                                  market_data: Dict[str, MarketData]) -> Optional[float]:
        """Calculate Enterprise Value = Market Cap + Total Debt - Cash."""
        
        market = market_data[symbol]
        statements = financial_data[symbol]
        
        if not statements or market.market_cap is None:
            return None
        
        latest_statement = statements[0]
        
        # Enterprise Value = Market Cap + Net Debt
        enterprise_value = market.market_cap
        
        if latest_statement.total_debt is not None:
            enterprise_value += latest_statement.total_debt
        
        # Subtract cash (simplified - would use actual cash balances in production)
        # For now, estimate cash as a portion of current assets
        
        return enterprise_value


class PriceSalesCalculator(FundamentalFactorCalculator):
    """
    Price/Sales ratio calculator with sector normalization.
    
    Calculates price-to-sales ratio and normalizes by sector to account
    for different margin profiles across industries.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="price_sales_normalized",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Price-to-Sales ratio with sector normalization",
            lookback_days=365,
            data_requirements=[DataType.FINANCIAL_STATEMENTS, DataType.OHLCV],
            taiwan_specific=True,
            min_history_days=365,
            expected_ic=0.022,
            expected_turnover=0.11
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate P/S ratio for given symbols."""
        
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=4)
        market_data = self._get_market_data(symbols, as_of_date)
        
        ps_values = {}
        
        for symbol in symbols:
            ps_ratio = self._calculate_ps_ratio(symbol, financial_data, market_data)
            if ps_ratio is not None:
                ps_values[symbol] = ps_ratio
        
        # Apply sector normalization
        ps_values = self._apply_sector_normalization(ps_values)
        
        # Handle outliers
        ps_values = self._handle_outliers(ps_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=ps_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_ps_ratio(self, symbol: str,
                          financial_data: Dict[str, List[FinancialStatement]],
                          market_data: Dict[str, MarketData]) -> Optional[float]:
        """Calculate P/S ratio for a specific symbol."""
        
        if symbol not in financial_data or symbol not in market_data:
            return None
        
        statements = financial_data[symbol]
        market = market_data[symbol]
        
        if len(statements) < 4:  # Need TTM data
            return None
        
        # Calculate TTM revenue
        ttm_revenue = self._calculate_trailing_metric(statements, 'revenue', quarters=4)
        
        if ttm_revenue is None or ttm_revenue <= 0:
            return None
        
        # Calculate P/S ratio
        if market.market_cap is not None:
            ps_ratio = market.market_cap / ttm_revenue
        else:
            return None
        
        # Apply reasonable bounds
        if ps_ratio <= 0 or ps_ratio > 100:
            return None
        
        return ps_ratio
    
    def _apply_sector_normalization(self, ps_values: Dict[str, float]) -> Dict[str, float]:
        """
        Apply sector normalization to P/S ratios.
        
        In production, this would use actual sector classifications.
        For now, we normalize by the median P/S in the universe.
        """
        if not ps_values:
            return ps_values
        
        # Calculate median P/S ratio
        values = list(ps_values.values())
        finite_values = [v for v in values if np.isfinite(v)]
        
        if len(finite_values) < 10:
            return ps_values
        
        median_ps = np.median(finite_values)
        
        # Normalize P/S ratios relative to median
        normalized_values = {}
        for symbol, ps_ratio in ps_values.items():
            if np.isfinite(ps_ratio) and median_ps > 0:
                normalized_ps = ps_ratio / median_ps
                normalized_values[symbol] = normalized_ps
            else:
                normalized_values[symbol] = ps_ratio
        
        return normalized_values


class ValueFactors:
    """
    Orchestrator for all value factor calculations.
    
    This class manages the calculation of the 4 value factors and provides
    a unified interface for value factor computation.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.logger = logging.getLogger(__name__)
        
        # Initialize factor calculators
        self.pe_calculator = PERatioCalculator(pit_engine)
        self.pb_calculator = PBRatioCalculator(pit_engine)
        self.ev_ebitda_calculator = EVEBITDACalculator(pit_engine)
        self.ps_calculator = PriceSalesCalculator(pit_engine)
    
    def calculate_all_value_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, FactorResult]:
        """
        Calculate all 4 value factors for given symbols and date.
        
        Returns:
            Dictionary mapping factor names to FactorResult objects
        """
        results = {}
        
        try:
            # P/E Ratio
            self.logger.info("Calculating P/E ratios...")
            results["pe_ratio_ttm"] = self.pe_calculator.calculate(symbols, as_of_date)
            
            # P/B Ratio
            self.logger.info("Calculating P/B ratios...")
            results["pb_ratio_adjusted"] = self.pb_calculator.calculate(symbols, as_of_date)
            
            # EV/EBITDA
            self.logger.info("Calculating EV/EBITDA ratios...")
            results["ev_ebitda_ttm"] = self.ev_ebitda_calculator.calculate(symbols, as_of_date)
            
            # Price/Sales
            self.logger.info("Calculating P/S ratios...")
            results["price_sales_normalized"] = self.ps_calculator.calculate(symbols, as_of_date)
            
            self.logger.info(f"Completed calculation of {len(results)} value factors")
            
        except Exception as e:
            self.logger.error(f"Error calculating value factors: {e}")
            raise
        
        return results
    
    def get_value_factor_metadata(self) -> Dict[str, FactorMetadata]:
        """Get metadata for all value factors."""
        return {
            "pe_ratio_ttm": self.pe_calculator.metadata,
            "pb_ratio_adjusted": self.pb_calculator.metadata,
            "ev_ebitda_ttm": self.ev_ebitda_calculator.metadata,
            "price_sales_normalized": self.ps_calculator.metadata
        }