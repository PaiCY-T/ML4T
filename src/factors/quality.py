"""
Quality factor calculations for Taiwan market ML pipeline.

This module implements 4 quality factors:
1. ROE/ROA (Return on equity and return on assets)
2. Debt-to-Equity (Financial leverage ratios)
3. Operating Margin (Operating margin stability over time)
4. Earnings Quality (Accruals-based earnings quality score)
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


class ROEROACalculator(FundamentalFactorCalculator):
    """
    Return on Equity (ROE) and Return on Assets (ROA) calculator.
    
    Calculates both ROE and ROA with trend analysis and quality adjustments
    for Taiwan market characteristics.
    """
    
    def __init__(self, pit_engine: PITQueryEngine, factor_type: str = "roe"):
        if factor_type not in ["roe", "roa"]:
            raise ValueError("factor_type must be 'roe' or 'roa'")
        
        self.factor_type = factor_type
        
        metadata = FactorMetadata(
            name=f"{factor_type}_ttm",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description=f"Trailing twelve months {'Return on Equity' if factor_type == 'roe' else 'Return on Assets'}",
            lookback_days=730,  # 2 years for trend analysis
            data_requirements=[DataType.FINANCIAL_STATEMENTS],
            taiwan_specific=True,
            min_history_days=730,
            expected_ic=0.035,
            expected_turnover=0.16
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date, 
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate ROE or ROA for given symbols."""
        
        # Get financial data with more history for trend analysis
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=8)
        
        factor_values = {}
        
        for symbol in symbols:
            if self.factor_type == "roe":
                factor_value = self._calculate_roe(symbol, financial_data)
            else:
                factor_value = self._calculate_roa(symbol, financial_data)
            
            if factor_value is not None:
                factor_values[symbol] = factor_value
        
        # Handle outliers
        factor_values = self._handle_outliers(factor_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_roe(self, symbol: str, 
                      financial_data: Dict[str, List[FinancialStatement]]) -> Optional[float]:
        """Calculate Return on Equity with trend and quality adjustments."""
        
        if symbol not in financial_data:
            return None
        
        statements = financial_data[symbol]
        
        if len(statements) < 4:  # Need at least TTM data
            return None
        
        # Calculate TTM net income
        ttm_net_income = self._calculate_trailing_metric(statements, 'net_income', quarters=4)
        
        if ttm_net_income is None:
            return None
        
        # Calculate average equity (more stable than point-in-time)
        equity_values = []
        for statement in statements[:4]:  # TTM period
            if statement.total_equity is not None and statement.total_equity > 0:
                equity_values.append(statement.total_equity)
        
        if len(equity_values) < 2:
            return None
        
        avg_equity = np.mean(equity_values)
        
        if avg_equity <= 0:
            return None
        
        roe = ttm_net_income / avg_equity
        
        # Quality adjustment: penalize volatile ROE
        roe_trend = self._calculate_roe_trend(statements)
        if roe_trend is not None:
            # Apply stability adjustment (higher stability = higher quality)
            stability_factor = max(0.5, 1.0 - abs(roe_trend))
            roe = roe * stability_factor
        
        # Apply reasonable bounds
        if abs(roe) > 5.0:  # ROE > 500% or < -500% is unrealistic
            return None
        
        return roe
    
    def _calculate_roa(self, symbol: str,
                      financial_data: Dict[str, List[FinancialStatement]]) -> Optional[float]:
        """Calculate Return on Assets with quality adjustments."""
        
        if symbol not in financial_data:
            return None
        
        statements = financial_data[symbol]
        
        if len(statements) < 4:  # Need at least TTM data
            return None
        
        # Calculate TTM net income
        ttm_net_income = self._calculate_trailing_metric(statements, 'net_income', quarters=4)
        
        if ttm_net_income is None:
            return None
        
        # Calculate average total assets
        asset_values = []
        for statement in statements[:4]:  # TTM period
            if statement.total_assets is not None and statement.total_assets > 0:
                asset_values.append(statement.total_assets)
        
        if len(asset_values) < 2:
            return None
        
        avg_assets = np.mean(asset_values)
        
        if avg_assets <= 0:
            return None
        
        roa = ttm_net_income / avg_assets
        
        # Quality adjustment based on asset utilization trend
        asset_turnover_trend = self._calculate_asset_turnover_trend(statements)
        if asset_turnover_trend is not None and asset_turnover_trend > 0:
            # Bonus for improving asset utilization
            roa = roa * (1.0 + min(0.2, asset_turnover_trend))
        
        # Apply reasonable bounds
        if abs(roa) > 2.0:  # ROA > 200% or < -200% is unrealistic
            return None
        
        return roa
    
    def _calculate_roe_trend(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate ROE trend over the past 2 years."""
        
        if len(statements) < 8:  # Need 2 years of quarterly data
            return None
        
        quarterly_roe = []
        
        for i in range(len(statements) - 3):  # Rolling 4-quarter ROE
            ttm_income = 0
            avg_equity = 0
            valid_quarters = 0
            
            for j in range(4):
                statement = statements[i + j]
                if (statement.net_income is not None and 
                    statement.total_equity is not None and 
                    statement.total_equity > 0):
                    ttm_income += statement.net_income
                    avg_equity += statement.total_equity
                    valid_quarters += 1
            
            if valid_quarters >= 3:  # At least 3 valid quarters
                avg_equity = avg_equity / valid_quarters
                roe = ttm_income / avg_equity
                quarterly_roe.append(roe)
        
        if len(quarterly_roe) < 3:
            return None
        
        # Calculate trend (slope of ROE over time)
        x = np.arange(len(quarterly_roe))
        y = np.array(quarterly_roe)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def _calculate_asset_turnover_trend(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate asset turnover trend."""
        
        if len(statements) < 8:
            return None
        
        turnover_ratios = []
        
        for i in range(len(statements) - 3):
            ttm_revenue = 0
            avg_assets = 0
            valid_quarters = 0
            
            for j in range(4):
                statement = statements[i + j]
                if (statement.revenue is not None and 
                    statement.total_assets is not None and 
                    statement.total_assets > 0):
                    ttm_revenue += statement.revenue
                    avg_assets += statement.total_assets
                    valid_quarters += 1
            
            if valid_quarters >= 3:
                avg_assets = avg_assets / valid_quarters
                turnover = ttm_revenue / avg_assets
                turnover_ratios.append(turnover)
        
        if len(turnover_ratios) < 3:
            return None
        
        # Calculate trend
        x = np.arange(len(turnover_ratios))
        y = np.array(turnover_ratios)
        
        slope = np.polyfit(x, y, 1)[0]
        
        return slope


class DebtEquityCalculator(FundamentalFactorCalculator):
    """
    Debt-to-Equity ratio calculator with Taiwan market adjustments.
    
    Calculates financial leverage ratios and adjusts for Taiwan market
    characteristics including conservative leverage preferences.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="debt_equity_ratio",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Debt-to-Equity ratio with stability adjustment",
            lookback_days=730,
            data_requirements=[DataType.FINANCIAL_STATEMENTS],
            taiwan_specific=True,
            min_history_days=365,
            expected_ic=-0.025,  # Negative IC - high debt is typically bad
            expected_turnover=0.10
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate Debt-to-Equity ratio for given symbols."""
        
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=8)
        
        de_values = {}
        
        for symbol in symbols:
            de_ratio = self._calculate_debt_equity_ratio(symbol, financial_data)
            if de_ratio is not None:
                de_values[symbol] = de_ratio
        
        # Handle outliers
        de_values = self._handle_outliers(de_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=de_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_debt_equity_ratio(self, symbol: str,
                                   financial_data: Dict[str, List[FinancialStatement]]) -> Optional[float]:
        """Calculate Debt-to-Equity ratio with trend adjustment."""
        
        if symbol not in financial_data:
            return None
        
        statements = financial_data[symbol]
        
        if not statements:
            return None
        
        latest_statement = statements[0]
        
        if (latest_statement.total_debt is None or 
            latest_statement.total_equity is None or
            latest_statement.total_equity <= 0):
            return None
        
        de_ratio = latest_statement.total_debt / latest_statement.total_equity
        
        # Quality adjustment: penalize rapidly increasing leverage
        leverage_trend = self._calculate_leverage_trend(statements)
        if leverage_trend is not None and leverage_trend > 0:
            # Penalty for increasing leverage (multiply by factor > 1)
            de_ratio = de_ratio * (1.0 + min(0.3, leverage_trend))
        
        # Apply reasonable bounds (D/E > 10 is extremely high)
        if de_ratio < 0 or de_ratio > 10:
            return None
        
        return de_ratio
    
    def _calculate_leverage_trend(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate leverage trend over time."""
        
        if len(statements) < 4:
            return None
        
        leverage_ratios = []
        
        for statement in statements[:4]:  # Last 4 quarters
            if (statement.total_debt is not None and 
                statement.total_equity is not None and
                statement.total_equity > 0):
                leverage = statement.total_debt / statement.total_equity
                leverage_ratios.append(leverage)
        
        if len(leverage_ratios) < 3:
            return None
        
        # Calculate trend
        x = np.arange(len(leverage_ratios))
        y = np.array(leverage_ratios)
        
        slope = np.polyfit(x, y, 1)[0]
        
        return slope


class OperatingMarginCalculator(FundamentalFactorCalculator):
    """
    Operating margin stability calculator.
    
    Calculates operating margin and evaluates stability over time,
    which indicates operational efficiency and management quality.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="operating_margin_stability",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Operating margin with stability adjustment",
            lookback_days=730,
            data_requirements=[DataType.FINANCIAL_STATEMENTS],
            taiwan_specific=True,
            min_history_days=730,
            expected_ic=0.030,
            expected_turnover=0.14
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate operating margin stability for given symbols."""
        
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=8)
        
        margin_values = {}
        
        for symbol in symbols:
            margin_score = self._calculate_operating_margin_score(symbol, financial_data)
            if margin_score is not None:
                margin_values[symbol] = margin_score
        
        # Handle outliers
        margin_values = self._handle_outliers(margin_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=margin_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_operating_margin_score(self, symbol: str,
                                        financial_data: Dict[str, List[FinancialStatement]]) -> Optional[float]:
        """Calculate operating margin stability score."""
        
        if symbol not in financial_data:
            return None
        
        statements = financial_data[symbol]
        
        if len(statements) < 4:
            return None
        
        # Calculate TTM operating margin
        ttm_operating_income = self._calculate_trailing_metric(statements, 'operating_income', quarters=4)
        ttm_revenue = self._calculate_trailing_metric(statements, 'revenue', quarters=4)
        
        if ttm_operating_income is None or ttm_revenue is None or ttm_revenue <= 0:
            return None
        
        ttm_margin = ttm_operating_income / ttm_revenue
        
        # Calculate margin stability
        margin_stability = self._calculate_margin_stability(statements)
        
        if margin_stability is None:
            # If we can't calculate stability, just return the current margin
            return ttm_margin
        
        # Combine current margin with stability (stability ranges from 0 to 1)
        # Higher stability gives bonus to the margin score
        stability_adjusted_margin = ttm_margin * (0.7 + 0.3 * margin_stability)
        
        # Apply reasonable bounds
        if abs(stability_adjusted_margin) > 2.0:  # Operating margin > 200% is unrealistic
            return None
        
        return stability_adjusted_margin
    
    def _calculate_margin_stability(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate operating margin stability over time."""
        
        if len(statements) < 6:  # Need at least 6 quarters for stability calc
            return None
        
        quarterly_margins = []
        
        for i in range(len(statements) - 3):  # Rolling 4-quarter margins
            ttm_operating_income = 0
            ttm_revenue = 0
            valid_quarters = 0
            
            for j in range(4):
                statement = statements[i + j]
                if (statement.operating_income is not None and 
                    statement.revenue is not None and 
                    statement.revenue > 0):
                    ttm_operating_income += statement.operating_income
                    ttm_revenue += statement.revenue
                    valid_quarters += 1
            
            if valid_quarters >= 3 and ttm_revenue > 0:
                margin = ttm_operating_income / ttm_revenue
                quarterly_margins.append(margin)
        
        if len(quarterly_margins) < 3:
            return None
        
        # Calculate stability as inverse of coefficient of variation
        margins = np.array(quarterly_margins)
        mean_margin = np.mean(margins)
        std_margin = np.std(margins)
        
        if abs(mean_margin) < 1e-6:  # Avoid division by zero
            return 0.0
        
        cv = std_margin / abs(mean_margin)  # Coefficient of variation
        
        # Convert to stability score (0 to 1, where 1 is most stable)
        stability = 1.0 / (1.0 + cv)
        
        return min(1.0, stability)


class EarningsQualityCalculator(FundamentalFactorCalculator):
    """
    Earnings quality calculator based on accruals analysis.
    
    Calculates earnings quality score based on the relationship between
    earnings and cash flows, following Sloan (1996) methodology.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="earnings_quality_score",
            category=FactorCategory.FUNDAMENTAL,
            frequency=FactorFrequency.DAILY,
            description="Accruals-based earnings quality score",
            lookback_days=730,
            data_requirements=[DataType.FINANCIAL_STATEMENTS],
            taiwan_specific=True,
            min_history_days=730,
            expected_ic=0.027,
            expected_turnover=0.18
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate earnings quality for given symbols."""
        
        financial_data = self._get_financial_data(symbols, as_of_date, quarters_back=8)
        
        quality_values = {}
        
        for symbol in symbols:
            quality_score = self._calculate_earnings_quality(symbol, financial_data)
            if quality_score is not None:
                quality_values[symbol] = quality_score
        
        # Handle outliers
        quality_values = self._handle_outliers(quality_values, method="winsorize")
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=quality_values,
            metadata=self.metadata,
            calculation_time=datetime.now()
        )
    
    def _calculate_earnings_quality(self, symbol: str,
                                  financial_data: Dict[str, List[FinancialStatement]]) -> Optional[float]:
        """Calculate earnings quality based on accruals analysis."""
        
        if symbol not in financial_data:
            return None
        
        statements = financial_data[symbol]
        
        if len(statements) < 4:
            return None
        
        # Calculate TTM metrics
        ttm_net_income = self._calculate_trailing_metric(statements, 'net_income', quarters=4)
        ttm_operating_cf = self._calculate_trailing_metric(statements, 'operating_cash_flow', quarters=4)
        
        if ttm_net_income is None or ttm_operating_cf is None:
            # Try to estimate quality using available data
            return self._estimate_earnings_quality_simple(statements)
        
        # Calculate total accruals = Net Income - Operating Cash Flow
        total_accruals = ttm_net_income - ttm_operating_cf
        
        # Get average total assets for scaling
        avg_assets = self._get_average_total_assets(statements[:4])
        
        if avg_assets is None or avg_assets <= 0:
            return None
        
        # Scale accruals by total assets
        scaled_accruals = total_accruals / avg_assets
        
        # Quality score = negative of scaled accruals (lower accruals = higher quality)
        # Also add cash flow quality component
        cf_quality = self._calculate_cash_flow_quality(statements)
        
        if cf_quality is not None:
            # Combine accruals quality with cash flow quality
            earnings_quality = -scaled_accruals + 0.3 * cf_quality
        else:
            earnings_quality = -scaled_accruals
        
        # Apply reasonable bounds
        if abs(earnings_quality) > 2.0:
            return None
        
        return earnings_quality
    
    def _estimate_earnings_quality_simple(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Simple earnings quality estimation when cash flow data is not available."""
        
        if len(statements) < 6:
            return None
        
        # Use earnings consistency as a proxy for quality
        earnings_values = []
        revenue_values = []
        
        for statement in statements[:6]:
            if (statement.net_income is not None and 
                statement.revenue is not None and 
                statement.revenue > 0):
                earnings_values.append(statement.net_income)
                revenue_values.append(statement.revenue)
        
        if len(earnings_values) < 4:
            return None
        
        # Calculate earnings-to-revenue ratios for stability
        margin_ratios = [e / r for e, r in zip(earnings_values, revenue_values)]
        
        # Quality = consistency of margins (inverse of coefficient of variation)
        margins = np.array(margin_ratios)
        mean_margin = np.mean(margins)
        std_margin = np.std(margins)
        
        if abs(mean_margin) < 1e-6:
            return 0.0
        
        cv = std_margin / abs(mean_margin)
        quality_score = 1.0 / (1.0 + cv) - 0.5  # Center around 0
        
        return quality_score
    
    def _get_average_total_assets(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate average total assets."""
        
        asset_values = []
        
        for statement in statements:
            if statement.total_assets is not None and statement.total_assets > 0:
                asset_values.append(statement.total_assets)
        
        if not asset_values:
            return None
        
        return np.mean(asset_values)
    
    def _calculate_cash_flow_quality(self, statements: List[FinancialStatement]) -> Optional[float]:
        """Calculate cash flow quality component."""
        
        if len(statements) < 6:
            return None
        
        # Compare operating cash flow trend with earnings trend
        ocf_values = []
        income_values = []
        
        for statement in statements[:6]:
            if (statement.operating_cash_flow is not None and 
                statement.net_income is not None):
                ocf_values.append(statement.operating_cash_flow)
                income_values.append(statement.net_income)
        
        if len(ocf_values) < 4:
            return None
        
        # Calculate correlation between cash flow and earnings
        # Higher correlation = higher quality
        correlation = np.corrcoef(ocf_values, income_values)[0, 1]
        
        if np.isnan(correlation):
            return None
        
        # Convert correlation to quality score
        cf_quality = correlation  # Ranges from -1 to 1
        
        return cf_quality


class QualityFactors:
    """
    Orchestrator for all quality factor calculations.
    
    This class manages the calculation of the 4 quality factors and provides
    a unified interface for quality factor computation.
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.logger = logging.getLogger(__name__)
        
        # Initialize factor calculators
        self.roe_calculator = ROEROACalculator(pit_engine, "roe")
        self.roa_calculator = ROEROACalculator(pit_engine, "roa")
        self.debt_equity_calculator = DebtEquityCalculator(pit_engine)
        self.operating_margin_calculator = OperatingMarginCalculator(pit_engine)
        self.earnings_quality_calculator = EarningsQualityCalculator(pit_engine)
    
    def calculate_all_quality_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, FactorResult]:
        """
        Calculate all 4 quality factors for given symbols and date.
        
        Returns:
            Dictionary mapping factor names to FactorResult objects
        """
        results = {}
        
        try:
            # ROE
            self.logger.info("Calculating ROE...")
            results["roe_ttm"] = self.roe_calculator.calculate(symbols, as_of_date)
            
            # ROA  
            self.logger.info("Calculating ROA...")
            results["roa_ttm"] = self.roa_calculator.calculate(symbols, as_of_date)
            
            # Debt-to-Equity
            self.logger.info("Calculating Debt-to-Equity ratios...")
            results["debt_equity_ratio"] = self.debt_equity_calculator.calculate(symbols, as_of_date)
            
            # Operating Margin Stability
            self.logger.info("Calculating Operating Margin Stability...")
            results["operating_margin_stability"] = self.operating_margin_calculator.calculate(symbols, as_of_date)
            
            # Earnings Quality
            self.logger.info("Calculating Earnings Quality...")
            results["earnings_quality_score"] = self.earnings_quality_calculator.calculate(symbols, as_of_date)
            
            self.logger.info(f"Completed calculation of {len(results)} quality factors")
            
        except Exception as e:
            self.logger.error(f"Error calculating quality factors: {e}")
            raise
        
        return results
    
    def get_quality_factor_metadata(self) -> Dict[str, FactorMetadata]:
        """Get metadata for all quality factors."""
        return {
            "roe_ttm": self.roe_calculator.metadata,
            "roa_ttm": self.roa_calculator.metadata,
            "debt_equity_ratio": self.debt_equity_calculator.metadata,
            "operating_margin_stability": self.operating_margin_calculator.metadata,
            "earnings_quality_score": self.earnings_quality_calculator.metadata
        }