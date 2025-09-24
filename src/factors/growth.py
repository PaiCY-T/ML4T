"""
Growth factor calculations for Taiwan market ML pipeline.

This module implements 4 growth factors:
1. Revenue Growth (Quarter-over-quarter and year-over-year growth)
2. Earnings Growth (Earnings growth consistency and momentum)
3. Book Value Growth (Tangible book value growth rates)
4. Analyst Revision (Analyst estimate revision momentum)
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


class RevenueGrowthCalculator(FundamentalFactorCalculator):
    """
    Revenue growth calculator with QoQ and YoY analysis.
    
    Calculates revenue growth rates and evaluates growth consistency
    and acceleration patterns for Taiwan market.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="revenue_growth_momentum",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Revenue growth with momentum and consistency scoring",
            lookback_days=1095,  # 3 years for growth trend analysis
            data_requirements=[DataType.FINANCIAL_STATEMENTS],
            taiwan_specific=True,
            min_history_days=1095,
            expected_ic=0.032,
            expected_turnover=0.20
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate revenue growth momentum for given symbols."""
        
        # Need more history for growth calculations
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=12)
        
        growth_values = {}
        
        for symbol in symbols:
            growth_score = self._calculate_revenue_growth_score(symbol, financial_data)
            if growth_score is not None:
                growth_values[symbol] = growth_score
        
        # Handle outliers
        growth_values = self._handle_outliers(growth_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=growth_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_revenue_growth_score(self, symbol: str,
                                      financial_data: Dict[str, List[FinancialStatement]]) -> Optional[float]:
        """Calculate comprehensive revenue growth score."""
        
        if symbol not in financial_data:
            return None
        
        statements = financial_data[symbol]
        
        if len(statements) < 8:  # Need at least 2 years
            return None
        
        # Calculate YoY growth rate
        yoy_growth = self._calculate_yoy_revenue_growth(statements)
        
        # Calculate growth acceleration
        growth_acceleration = self._calculate_growth_acceleration(statements)
        
        # Calculate growth consistency
        growth_consistency = self._calculate_growth_consistency(statements)
        
        if yoy_growth is None:
            return None
        
        # Combine components into overall growth score
        growth_score = yoy_growth
        
        # Bonus for positive acceleration
        if growth_acceleration is not None and growth_acceleration > 0:
            growth_score += min(0.1, growth_acceleration)
        
        # Bonus for consistency
        if growth_consistency is not None:
            growth_score *= (0.7 + 0.3 * growth_consistency)
        
        # Apply reasonable bounds
        if abs(growth_score) > 2.0:  # Growth > 200% is extremely high
            return None
        
        return growth_score
    
    def _calculate_yoy_revenue_growth(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate year-over-year revenue growth rate."""
        
        if len(statements) < 8:  # Need 2 years of quarterly data
            return None
        
        # Get TTM revenue for current and previous year
        current_ttm = self._calculate_trailing_metric(statements[:4], 'revenue', quarters=4)
        previous_ttm = self._calculate_trailing_metric(statements[4:8], 'revenue', quarters=4)
        
        if current_ttm is None or previous_ttm is None or previous_ttm <= 0:
            return None
        
        yoy_growth = (current_ttm - previous_ttm) / previous_ttm
        
        return yoy_growth
    
    def _calculate_growth_acceleration(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate revenue growth acceleration."""
        
        if len(statements) < 12:  # Need 3 years for acceleration
            return None
        
        # Calculate growth rates for each year
        growth_rates = []
        
        for i in range(3):  # 3 years
            start_idx = i * 4
            end_idx = (i + 1) * 4
            
            if end_idx + 4 <= len(statements):
                current_ttm = self._calculate_trailing_metric(statements[start_idx:end_idx], 'revenue', quarters=4)
                previous_ttm = self._calculate_trailing_metric(statements[end_idx:end_idx+4], 'revenue', quarters=4)
                
                if current_ttm is not None and previous_ttm is not None and previous_ttm > 0:
                    growth_rate = (current_ttm - previous_ttm) / previous_ttm
                    growth_rates.append(growth_rate)
        
        if len(growth_rates) < 2:
            return None
        
        # Calculate acceleration (change in growth rate)
        # Positive acceleration means growth is accelerating
        acceleration = growth_rates[0] - growth_rates[1]  # Most recent - previous
        
        return acceleration
    
    def _calculate_growth_consistency(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate revenue growth consistency score."""
        
        if len(statements) < 8:
            return None
        
        # Calculate quarterly growth rates
        quarterly_growth_rates = []
        
        for i in range(len(statements) - 4):  # Rolling QoQ growth
            current_q = statements[i]
            previous_q = statements[i + 4]  # Same quarter previous year
            
            if (current_q.revenue is not None and 
                previous_q.revenue is not None and 
                previous_q.revenue > 0):
                growth_rate = (current_q.revenue - previous_q.revenue) / previous_q.revenue
                quarterly_growth_rates.append(growth_rate)
        
        if len(quarterly_growth_rates) < 3:
            return None
        
        # Consistency = inverse of coefficient of variation of growth rates
        growth_rates = np.array(quarterly_growth_rates)
        mean_growth = np.mean(growth_rates)
        std_growth = np.std(growth_rates)
        
        if abs(mean_growth) < 1e-6:
            return 0.0
        
        cv = std_growth / abs(mean_growth)
        consistency = 1.0 / (1.0 + cv)
        
        return min(1.0, consistency)


class EarningsGrowthCalculator(FundamentalFactorCalculator):
    """
    Earnings growth calculator with consistency and momentum analysis.
    
    Evaluates earnings growth patterns and quality with focus on
    sustainable growth rather than volatile earnings spikes.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="earnings_growth_quality",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Earnings growth with quality and consistency scoring",
            lookback_days=1095,  # 3 years
            data_requirements=[DataType.FINANCIAL_STATEMENTS],
            taiwan_specific=True,
            min_history_days=1095,
            expected_ic=0.038,
            expected_turnover=0.22
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate earnings growth quality for given symbols."""
        
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=12)
        
        growth_values = {}
        
        for symbol in symbols:
            earnings_growth_score = self._calculate_earnings_growth_score(symbol, financial_data)
            if earnings_growth_score is not None:
                growth_values[symbol] = earnings_growth_score
        
        # Handle outliers
        growth_values = self._handle_outliers(growth_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=growth_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_earnings_growth_score(self, symbol: str,
                                       financial_data: Dict[str, List[FinancialStatement]]) -> Optional[float]:
        """Calculate comprehensive earnings growth score."""
        
        if symbol not in financial_data:
            return None
        
        statements = financial_data[symbol]
        
        if len(statements) < 8:
            return None
        
        # Calculate YoY earnings growth
        yoy_earnings_growth = self._calculate_yoy_earnings_growth(statements)
        
        # Calculate earnings growth sustainability
        growth_sustainability = self._calculate_earnings_sustainability(statements)
        
        # Calculate earnings momentum
        earnings_momentum = self._calculate_earnings_momentum(statements)
        
        if yoy_earnings_growth is None:
            return None
        
        # Combine components
        growth_score = yoy_earnings_growth
        
        # Adjust for sustainability (penalize unsustainable growth)
        if growth_sustainability is not None:
            growth_score *= growth_sustainability
        
        # Add momentum component
        if earnings_momentum is not None:
            growth_score += 0.2 * earnings_momentum
        
        # Apply bounds
        if abs(growth_score) > 3.0:
            return None
        
        return growth_score
    
    def _calculate_yoy_earnings_growth(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate year-over-year earnings growth."""
        
        if len(statements) < 8:
            return None
        
        current_ttm = self._calculate_trailing_metric(statements[:4], 'net_income', quarters=4)
        previous_ttm = self._calculate_trailing_metric(statements[4:8], 'net_income', quarters=4)
        
        if current_ttm is None or previous_ttm is None:
            return None
        
        # Handle negative earnings carefully
        if previous_ttm <= 0:
            # If previous earnings were negative and current are positive
            if current_ttm > 0:
                return 1.0  # Improvement from loss to profit
            else:
                return None  # Both negative, growth not meaningful
        
        growth_rate = (current_ttm - previous_ttm) / previous_ttm
        
        return growth_rate
    
    def _calculate_earnings_sustainability(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate earnings growth sustainability score."""
        
        if len(statements) < 8:
            return None
        
        # Compare earnings growth with revenue growth (sustainable growth should be supported by revenue)
        revenue_growth = self._calculate_yoy_revenue_growth(statements)
        earnings_growth = self._calculate_yoy_earnings_growth(statements)
        
        if revenue_growth is None or earnings_growth is None:
            return 0.5  # Neutral sustainability score
        
        # Sustainable earnings growth should not massively exceed revenue growth
        if revenue_growth <= 0 and earnings_growth > 0.2:
            # Earnings growing fast while revenue declining = unsustainable
            return 0.3
        
        if revenue_growth > 0:
            earnings_to_revenue_ratio = earnings_growth / revenue_growth
            
            # Ideal ratio is between 0.8 and 2.0
            if 0.8 <= earnings_to_revenue_ratio <= 2.0:
                return 1.0
            elif earnings_to_revenue_ratio > 2.0:
                # Earnings growing much faster than revenue (potentially unsustainable)
                return max(0.4, 1.0 / earnings_to_revenue_ratio)
            else:
                # Earnings growing slower than revenue (conservative but sustainable)
                return 0.8
        
        return 0.5
    
    def _calculate_earnings_momentum(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate earnings momentum (recent vs earlier performance)."""
        
        if len(statements) < 8:
            return None
        
        # Compare recent 4 quarters with previous 4 quarters
        recent_ttm = self._calculate_trailing_metric(statements[:4], 'net_income', quarters=4)
        earlier_ttm = self._calculate_trailing_metric(statements[4:8], 'net_income', quarters=4)
        
        if recent_ttm is None or earlier_ttm is None:
            return None
        
        if earlier_ttm <= 0:
            return 1.0 if recent_ttm > 0 else 0.0
        
        momentum = (recent_ttm - earlier_ttm) / earlier_ttm
        
        # Cap momentum at reasonable levels
        return np.clip(momentum, -1.0, 1.0)


class BookValueGrowthCalculator(FundamentalFactorCalculator):
    """
    Book value growth calculator focusing on tangible book value.
    
    Measures the growth in book value per share and tangible book value,
    indicating company's ability to create shareholder value.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="book_value_growth",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Tangible book value growth rate",
            lookback_days=1095,  # 3 years
            data_requirements=[DataType.FINANCIAL_STATEMENTS],
            taiwan_specific=True,
            min_history_days=730,
            expected_ic=0.025,
            expected_turnover=0.15
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate book value growth for given symbols."""
        
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=12)
        
        growth_values = {}
        
        for symbol in symbols:
            bv_growth = self._calculate_book_value_growth(symbol, financial_data)
            if bv_growth is not None:
                growth_values[symbol] = bv_growth
        
        # Handle outliers
        growth_values = self._handle_outliers(growth_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=growth_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_book_value_growth(self, symbol: str,
                                   financial_data: Dict[str, List[FinancialStatement]]) -> Optional[float]:
        """Calculate book value growth rate."""
        
        if symbol not in financial_data:
            return None
        
        statements = financial_data[symbol]
        
        if len(statements) < 8:  # Need 2 years
            return None
        
        # Calculate book value per share growth
        current_bvps = self._get_book_value_per_share(statements[0])
        previous_bvps = self._get_book_value_per_share(statements[4])  # 1 year ago
        
        if current_bvps is None or previous_bvps is None or previous_bvps <= 0:
            # Try tangible book value growth instead
            return self._calculate_tangible_bv_growth(statements)
        
        bv_growth = (current_bvps - previous_bvps) / previous_bvps
        
        # Quality adjustment: prefer organic growth
        organic_growth_factor = self._assess_organic_growth(statements)
        if organic_growth_factor is not None:
            bv_growth *= organic_growth_factor
        
        # Apply reasonable bounds
        if abs(bv_growth) > 1.0:  # BV growth > 100% is very high
            return None
        
        return bv_growth
    
    def _get_book_value_per_share(self, statement: FinancialStatement) -> Optional[float]:
        """Get book value per share from financial statement."""
        
        if statement.book_value_per_share is not None:
            return statement.book_value_per_share
        
        # Calculate from total equity and shares outstanding
        if (statement.total_equity is not None and 
            statement.shares_outstanding is not None and
            statement.shares_outstanding > 0):
            return statement.total_equity / statement.shares_outstanding
        
        return None
    
    def _calculate_tangible_bv_growth(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate tangible book value growth when BVPS is not available."""
        
        if len(statements) < 8:
            return None
        
        current_tbv = self._get_tangible_book_value(statements[0])
        previous_tbv = self._get_tangible_book_value(statements[4])
        
        if current_tbv is None or previous_tbv is None or previous_tbv <= 0:
            return None
        
        tbv_growth = (current_tbv - previous_tbv) / previous_tbv
        
        return tbv_growth
    
    def _get_tangible_book_value(self, statement: FinancialStatement) -> Optional[float]:
        """Get tangible book value (book value minus intangible assets)."""
        
        if statement.tangible_book_value is not None:
            return statement.tangible_book_value
        
        # Estimate as total equity (assuming minimal intangible assets for Taiwan companies)
        if statement.total_equity is not None:
            return statement.total_equity
        
        return None
    
    def _assess_organic_growth(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Assess if book value growth is organic (from retained earnings) vs. external financing."""
        
        if len(statements) < 4:
            return None
        
        # Look at relationship between earnings and book value growth
        ttm_earnings = self._calculate_trailing_metric(statements[:4], 'net_income', quarters=4)
        
        if ttm_earnings is None:
            return 1.0  # Neutral factor
        
        current_equity = statements[0].total_equity
        
        if current_equity is None or current_equity <= 0:
            return 1.0
        
        # If earnings-to-equity ratio is reasonable, growth is likely organic
        earnings_yield = ttm_earnings / current_equity
        
        if earnings_yield > 0.05:  # 5% ROE indicates organic growth capability
            return 1.0
        elif earnings_yield > 0:
            return 0.8  # Modest organic growth
        else:
            return 0.6  # Negative earnings suggest growth from external sources
        
        return 1.0


class AnalystRevisionCalculator(FundamentalFactorCalculator):
    """
    Analyst estimate revision momentum calculator.
    
    Tracks changes in analyst estimates and forecasts to capture
    sentiment and expectation changes in the market.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="analyst_revision_momentum",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Analyst estimate revision momentum",
            lookback_days=365,
            data_requirements=[DataType.ANALYST_ESTIMATES],
            taiwan_specific=True,
            min_history_days=180,
            expected_ic=0.042,
            expected_turnover=0.25
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate analyst revision momentum for given symbols."""
        
        # Get analyst estimate data
        analyst_data = self._get_analyst_data(symbols, as_of_date)
        
        revision_values = {}
        
        for symbol in symbols:
            revision_momentum = self._calculate_revision_momentum(symbol, analyst_data)
            if revision_momentum is not None:
                revision_values[symbol] = revision_momentum
        
        # Handle outliers
        revision_values = self._handle_outliers(revision_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=revision_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _get_analyst_data(self, symbols: List[str], as_of_date: date) -> Dict[str, Any]:
        """
        Get analyst estimate data.
        
        In production, this would access real analyst estimate databases.
        For now, we'll return a simplified structure.
        """
        # This is a placeholder - in production would query actual analyst data
        analyst_data = {}
        
        try:
            # Query analyst estimates with 6 months of history
            start_date = as_of_date - timedelta(days=180)
            
            query = PITQuery(
                symbols=symbols,
                as_of_date=as_of_date,
                data_types=[DataType.ANALYST_ESTIMATES],
                start_date=start_date,
                end_date=as_of_date
            )
            
            raw_data = self.pit_engine.query(query)
            analyst_data = self._parse_analyst_data(raw_data)
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve analyst data: {e}")
            # Return empty data structure
            analyst_data = {}
        
        return analyst_data
    
    def _parse_analyst_data(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Parse raw analyst data into structured format."""
        
        parsed_data = {}
        
        for symbol in raw_data.get('symbol', pd.Series(dtype=str)).unique():
            if pd.isna(symbol):
                continue
            
            symbol_data = raw_data[raw_data['symbol'] == symbol].copy()
            
            if len(symbol_data) == 0:
                continue
            
            # Extract estimate revisions over time
            estimates_history = []
            
            for _, row in symbol_data.iterrows():
                try:
                    estimate_record = {
                        'date': pd.to_datetime(row.get('estimate_date')).date(),
                        'eps_estimate': self._safe_float(row.get('eps_estimate')),
                        'revenue_estimate': self._safe_float(row.get('revenue_estimate')),
                        'num_analysts': int(row.get('num_analysts', 1))
                    }
                    estimates_history.append(estimate_record)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing estimate data for {symbol}: {e}")
                    continue
            
            if estimates_history:
                # Sort by date
                estimates_history.sort(key=lambda x: x['date'], reverse=True)
                parsed_data[symbol] = estimates_history
        
        return parsed_data
    
    def _calculate_revision_momentum(self, symbol: str, 
                                   analyst_data: Dict[str, Any]) -> Optional[float]:
        """Calculate analyst revision momentum."""
        
        if symbol not in analyst_data:
            # If no analyst data available, return neutral score
            return 0.0
        
        estimates = analyst_data[symbol]
        
        if len(estimates) < 2:
            return 0.0
        
        # Calculate EPS estimate revision momentum
        eps_momentum = self._calculate_eps_revision_momentum(estimates)
        
        # Calculate revenue estimate revision momentum
        revenue_momentum = self._calculate_revenue_revision_momentum(estimates)
        
        # Combine EPS and revenue momentum
        if eps_momentum is not None and revenue_momentum is not None:
            revision_momentum = 0.7 * eps_momentum + 0.3 * revenue_momentum
        elif eps_momentum is not None:
            revision_momentum = eps_momentum
        elif revenue_momentum is not None:
            revision_momentum = revenue_momentum
        else:
            revision_momentum = 0.0
        
        # Apply bounds
        revision_momentum = np.clip(revision_momentum, -1.0, 1.0)
        
        return revision_momentum
    
    def _calculate_eps_revision_momentum(self, estimates: List[Dict]) -> Optional[float]:
        """Calculate EPS estimate revision momentum."""
        
        eps_estimates = []
        
        for estimate in estimates:
            if estimate['eps_estimate'] is not None:
                eps_estimates.append({
                    'date': estimate['date'],
                    'value': estimate['eps_estimate']
                })
        
        if len(eps_estimates) < 2:
            return None
        
        # Sort by date (most recent first)
        eps_estimates.sort(key=lambda x: x['date'], reverse=True)
        
        # Calculate trend in estimates
        recent_estimate = eps_estimates[0]['value']
        older_estimate = eps_estimates[-1]['value']
        
        if older_estimate == 0:
            return 1.0 if recent_estimate > 0 else -1.0
        
        revision_trend = (recent_estimate - older_estimate) / abs(older_estimate)
        
        return np.clip(revision_trend, -1.0, 1.0)
    
    def _calculate_revenue_revision_momentum(self, estimates: List[Dict]) -> Optional[float]:
        """Calculate revenue estimate revision momentum."""
        
        revenue_estimates = []
        
        for estimate in estimates:
            if estimate['revenue_estimate'] is not None:
                revenue_estimates.append({
                    'date': estimate['date'],
                    'value': estimate['revenue_estimate']
                })
        
        if len(revenue_estimates) < 2:
            return None
        
        # Sort by date (most recent first)
        revenue_estimates.sort(key=lambda x: x['date'], reverse=True)
        
        # Calculate trend in estimates
        recent_estimate = revenue_estimates[0]['value']
        older_estimate = revenue_estimates[-1]['value']
        
        if older_estimate == 0:
            return 1.0 if recent_estimate > 0 else -1.0
        
        revision_trend = (recent_estimate - older_estimate) / abs(older_estimate)
        
        return np.clip(revision_trend, -1.0, 1.0)


class GrowthFactors:
    """
    Orchestrator for all growth factor calculations.
    
    This class manages the calculation of the 4 growth factors and provides
    a unified interface for growth factor computation.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.logger = logging.getLogger(__name__)
        
        # Initialize factor calculators
        self.revenue_growth_calculator = RevenueGrowthCalculator(pit_engine)
        self.earnings_growth_calculator = EarningsGrowthCalculator(pit_engine)
        self.book_value_growth_calculator = BookValueGrowthCalculator(pit_engine)
        self.analyst_revision_calculator = AnalystRevisionCalculator(pit_engine)
    
    def calculate_all_growth_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, FactorResult]:
        """
        Calculate all 4 growth factors for given symbols and date.
        
        Returns:
            Dictionary mapping factor names to FactorResult objects
        """
        results = {}
        
        try:
            # Revenue Growth
            self.logger.info("Calculating Revenue Growth...")
            results["revenue_growth_momentum"] = self.revenue_growth_calculator.calculate(symbols, as_of_date)
            
            # Earnings Growth
            self.logger.info("Calculating Earnings Growth...")
            results["earnings_growth_quality"] = self.earnings_growth_calculator.calculate(symbols, as_of_date)
            
            # Book Value Growth
            self.logger.info("Calculating Book Value Growth...")
            results["book_value_growth"] = self.book_value_growth_calculator.calculate(symbols, as_of_date)
            
            # Analyst Revision Momentum
            self.logger.info("Calculating Analyst Revision Momentum...")
            results["analyst_revision_momentum"] = self.analyst_revision_calculator.calculate(symbols, as_of_date)
            
            self.logger.info(f"Completed calculation of {len(results)} growth factors")
            
        except Exception as e:
            self.logger.error(f"Error calculating growth factors: {e}")
            raise
        
        return results
    
    def get_growth_factor_metadata(self) -> Dict[str, FactorMetadata]:
        """Get metadata for all growth factors."""
        return {
            "revenue_growth_momentum": self.revenue_growth_calculator.metadata,
            "earnings_growth_quality": self.earnings_growth_calculator.metadata,
            "book_value_growth": self.book_value_growth_calculator.metadata,
            "analyst_revision_momentum": self.analyst_revision_calculator.metadata
        }