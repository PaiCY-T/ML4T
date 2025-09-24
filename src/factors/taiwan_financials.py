"""
Taiwan-specific financial data handling for fundamental factors.

This module implements Taiwan market compliance for financial data:
- 60-day reporting lag enforcement for quarterly reports
- 90-day lag for annual reports
- Taiwan GAAP compliance
- TWD currency handling
- Seasonal effects (Lunar New Year)
- Taiwan Stock Exchange sector classifications
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
import calendar

from .fundamental import FinancialStatement, MarketData
from .taiwan_adjustments import TaiwanMarketAdjustments

try:
    from ..data.core.temporal import DataType, TemporalValue
    from ..data.pipeline.pit_engine import PITQueryEngine, PITQuery
    from ..data.models.taiwan_market import TaiwanMarketCode, TradingStatus
except ImportError:
    DataType = object
    TemporalValue = object
    PITQueryEngine = object
    PITQuery = object
    TaiwanMarketCode = object
    TradingStatus = object

logger = logging.getLogger(__name__)


class TaiwanReportingPeriod(Enum):
    """Taiwan financial reporting periods."""
    Q1 = "Q1"  # Jan-Mar
    Q2 = "Q2"  # Apr-Jun  
    Q3 = "Q3"  # Jul-Sep
    Q4 = "Q4"  # Oct-Dec
    ANNUAL = "ANNUAL"


class TaiwanSectorClassification(Enum):
    """Taiwan Stock Exchange sector classifications."""
    SEMICONDUCTORS = "SEMICONDUCTORS"
    TECHNOLOGY = "TECHNOLOGY"
    FINANCIAL = "FINANCIAL"
    TRADITIONAL_INDUSTRY = "TRADITIONAL_INDUSTRY"
    PLASTIC_CHEMICAL = "PLASTIC_CHEMICAL"
    FOOD = "FOOD"
    TEXTILE = "TEXTILE"
    ELECTRONICS = "ELECTRONICS"
    CONSTRUCTION = "CONSTRUCTION"
    TRANSPORTATION = "TRANSPORTATION"
    TOURISM = "TOURISM"
    OTHER = "OTHER"


@dataclass
class TaiwanFinancialMetadata:
    """Metadata for Taiwan financial reports."""
    symbol: str
    report_date: date
    reporting_period: TaiwanReportingPeriod
    fiscal_year: int
    fiscal_quarter: int
    gaap_standard: str = "Taiwan-GAAP"
    currency: str = "TWD"
    sector: Optional[TaiwanSectorClassification] = None
    is_consolidated: bool = True
    audit_status: str = "unaudited"  # Most quarterly reports are unaudited
    filing_date: Optional[date] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.report_date:
            self.fiscal_year = self.report_date.year
            month = self.report_date.month
            
            if month <= 3:
                self.fiscal_quarter = 1
                self.reporting_period = TaiwanReportingPeriod.Q1
            elif month <= 6:
                self.fiscal_quarter = 2
                self.reporting_period = TaiwanReportingPeriod.Q2
            elif month <= 9:
                self.fiscal_quarter = 3
                self.reporting_period = TaiwanReportingPeriod.Q3
            else:
                self.fiscal_quarter = 4
                self.reporting_period = TaiwanReportingPeriod.Q4


@dataclass
class TaiwanFinancialStatement(FinancialStatement):
    """Enhanced financial statement with Taiwan-specific fields."""
    
    # Additional Taiwan-specific fields
    metadata: Optional[TaiwanFinancialMetadata] = None
    
    # Taiwan-specific financial items
    comprehensive_income: Optional[float] = None
    retained_earnings: Optional[float] = None
    capital_surplus: Optional[float] = None
    treasury_stock: Optional[float] = None
    
    # Industry-specific metrics
    semiconductor_revenue: Optional[float] = None  # For tech companies
    financial_fee_income: Optional[float] = None    # For financial institutions
    construction_revenue_recognized: Optional[float] = None  # For construction
    
    def to_base_statement(self) -> FinancialStatement:
        """Convert to base FinancialStatement for compatibility."""
        return FinancialStatement(
            symbol=self.symbol,
            report_date=self.report_date,
            revenue=self.revenue,
            net_income=self.net_income,
            total_assets=self.total_assets,
            total_equity=self.total_equity,
            total_debt=self.total_debt,
            operating_income=self.operating_income,
            ebitda=self.ebitda,
            shares_outstanding=self.shares_outstanding,
            book_value_per_share=self.book_value_per_share,
            tangible_book_value=self.tangible_book_value,
            operating_cash_flow=self.operating_cash_flow,
            free_cash_flow=self.free_cash_flow
        )


class TaiwanFinancialDataHandler:
    """
    Handler for Taiwan-specific financial data requirements.
    
    This class ensures compliance with Taiwan financial reporting regulations
    and handles Taiwan market-specific characteristics.
    """
    
    # Taiwan reporting lag requirements (in days)
    QUARTERLY_LAG = 60   # Quarterly reports available 60 days after quarter end
    ANNUAL_LAG = 90      # Annual reports available 90 days after year end
    MONTHLY_LAG = 20     # Monthly revenue available ~20 days after month end
    
    # Taiwan fiscal year (same as calendar year for most companies)
    FISCAL_YEAR_END = 12  # December
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.taiwan_adjustments = TaiwanMarketAdjustments()
        self.logger = logging.getLogger(__name__)
        
        # Cache for sector classifications
        self._sector_cache: Dict[str, TaiwanSectorClassification] = {}
        
    def get_financial_statements(self, symbols: List[str], as_of_date: date,
                               quarters_back: int = 8) -> Dict[str, List[TaiwanFinancialStatement]]:
        """
        Get financial statements with proper Taiwan reporting lag enforcement.
        
        Args:
            symbols: List of Taiwan stock symbols
            as_of_date: As-of date for point-in-time access
            quarters_back: Number of quarters of history to retrieve
            
        Returns:
            Dictionary mapping symbols to list of Taiwan financial statements
        """
        
        # Calculate data availability cutoff with Taiwan reporting lags
        quarterly_cutoff = self._calculate_quarterly_cutoff_date(as_of_date)
        annual_cutoff = self._calculate_annual_cutoff_date(as_of_date)
        
        # Use the more conservative cutoff
        data_cutoff = min(quarterly_cutoff, annual_cutoff)
        
        self.logger.info(f"Taiwan financial data cutoff: {data_cutoff}")
        self.logger.info(f"Quarterly cutoff: {quarterly_cutoff}, Annual cutoff: {annual_cutoff}")
        
        # Calculate lookback period
        start_date = data_cutoff - timedelta(days=quarters_back * 95)  # ~3 months per quarter + buffer
        
        # Query financial data
        try:
            query = PITQuery(
                symbols=symbols,
                as_of_date=data_cutoff,
                data_types=[DataType.FINANCIAL_STATEMENTS],
                start_date=start_date,
                end_date=data_cutoff
            )
            
            raw_data = self.pit_engine.query(query)
            
        except Exception as e:
            self.logger.error(f"Error querying financial data: {e}")
            return {}
        
        # Parse and enhance with Taiwan-specific metadata
        financial_statements = self._parse_taiwan_financial_data(raw_data)
        
        # Apply Taiwan-specific validations
        validated_statements = self._validate_taiwan_financial_data(financial_statements)
        
        return validated_statements
    
    def _calculate_quarterly_cutoff_date(self, as_of_date: date) -> date:
        """Calculate cutoff date for quarterly financial data availability."""
        
        # Find the most recent quarter end before as_of_date
        current_year = as_of_date.year
        current_month = as_of_date.month
        
        # Determine the most recent quarter end
        if current_month >= 10:  # Q3 data (Jul-Sep) available
            quarter_end = date(current_year, 9, 30)
        elif current_month >= 7:   # Q2 data (Apr-Jun) available  
            quarter_end = date(current_year, 6, 30)
        elif current_month >= 4:   # Q1 data (Jan-Mar) available
            quarter_end = date(current_year, 3, 31)
        else:  # Q4 data from previous year available
            quarter_end = date(current_year - 1, 12, 31)
        
        # Add Taiwan quarterly reporting lag
        cutoff_date = quarter_end + timedelta(days=self.QUARTERLY_LAG)
        
        # Ensure cutoff is not after as_of_date
        return min(cutoff_date, as_of_date)
    
    def _calculate_annual_cutoff_date(self, as_of_date: date) -> date:
        """Calculate cutoff date for annual financial data availability."""
        
        # Most recent fiscal year end (December 31 for most Taiwan companies)
        if as_of_date.month >= 4:  # After March, previous year's annual data should be available
            fiscal_year_end = date(as_of_date.year - 1, 12, 31)
        else:  # Before March, use year before that
            fiscal_year_end = date(as_of_date.year - 2, 12, 31)
        
        # Add Taiwan annual reporting lag
        cutoff_date = fiscal_year_end + timedelta(days=self.ANNUAL_LAG)
        
        return min(cutoff_date, as_of_date)
    
    def _parse_taiwan_financial_data(self, raw_data: pd.DataFrame) -> Dict[str, List[TaiwanFinancialStatement]]:
        """Parse raw financial data into Taiwan-enhanced financial statements."""
        
        taiwan_statements = {}
        
        for symbol in raw_data.get('symbol', pd.Series(dtype=str)).unique():
            if pd.isna(symbol):
                continue
            
            symbol_data = raw_data[raw_data['symbol'] == symbol].copy()
            statements = []
            
            for _, row in symbol_data.iterrows():
                try:
                    # Parse base financial data
                    report_date = pd.to_datetime(row.get('report_date')).date()
                    
                    # Create Taiwan metadata
                    metadata = TaiwanFinancialMetadata(
                        symbol=symbol,
                        report_date=report_date,
                        reporting_period=TaiwanReportingPeriod.Q1,  # Will be calculated in __post_init__
                        fiscal_year=report_date.year,
                        fiscal_quarter=1,  # Will be calculated in __post_init__
                        sector=self._get_sector_classification(symbol)
                    )
                    
                    # Create Taiwan financial statement
                    statement = TaiwanFinancialStatement(
                        symbol=symbol,
                        report_date=report_date,
                        revenue=self._safe_float(row.get('revenue')),
                        net_income=self._safe_float(row.get('net_income')),
                        total_assets=self._safe_float(row.get('total_assets')),
                        total_equity=self._safe_float(row.get('total_equity')),
                        total_debt=self._safe_float(row.get('total_debt')),
                        operating_income=self._safe_float(row.get('operating_income')),
                        ebitda=self._safe_float(row.get('ebitda')),
                        shares_outstanding=self._safe_float(row.get('shares_outstanding')),
                        operating_cash_flow=self._safe_float(row.get('operating_cash_flow')),
                        free_cash_flow=self._safe_float(row.get('free_cash_flow')),
                        metadata=metadata,
                        
                        # Taiwan-specific fields
                        comprehensive_income=self._safe_float(row.get('comprehensive_income')),
                        retained_earnings=self._safe_float(row.get('retained_earnings')),
                        capital_surplus=self._safe_float(row.get('capital_surplus')),
                        semiconductor_revenue=self._safe_float(row.get('semiconductor_revenue')),
                        financial_fee_income=self._safe_float(row.get('financial_fee_income'))
                    )
                    
                    statements.append(statement)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing Taiwan financial data for {symbol}: {e}")
                    continue
            
            if statements:
                # Sort by report date (most recent first)
                statements.sort(key=lambda x: x.report_date, reverse=True)
                taiwan_statements[symbol] = statements
        
        return taiwan_statements
    
    def _get_sector_classification(self, symbol: str) -> TaiwanSectorClassification:
        """
        Get Taiwan sector classification for a symbol.
        
        In production, this would query actual sector data.
        For now, we use simple heuristics based on symbol patterns.
        """
        
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]
        
        # Simple heuristic classification based on symbol
        # In production, this would be replaced with actual sector lookup
        
        if symbol.startswith(('2330', '2454', '3008', '6505')):  # TSM, MediaTek, etc.
            sector = TaiwanSectorClassification.SEMICONDUCTORS
        elif symbol.startswith(('2317', '2382', '6176')):  # Hon Hai, Quanta, etc.
            sector = TaiwanSectorClassification.ELECTRONICS
        elif symbol.startswith(('2881', '2882', '2884', '2885')):  # Banks
            sector = TaiwanSectorClassification.FINANCIAL
        elif symbol.startswith(('1101', '1102', '1216')):  # Traditional industry
            sector = TaiwanSectorClassification.TRADITIONAL_INDUSTRY
        else:
            sector = TaiwanSectorClassification.OTHER
        
        self._sector_cache[symbol] = sector
        return sector
    
    def _validate_taiwan_financial_data(self, 
                                      statements: Dict[str, List[TaiwanFinancialStatement]]) -> Dict[str, List[TaiwanFinancialStatement]]:
        """Apply Taiwan-specific validations to financial data."""
        
        validated_statements = {}
        
        for symbol, symbol_statements in statements.items():
            validated_list = []
            
            for statement in symbol_statements:
                # Validate reporting lag compliance
                if not self._validate_reporting_lag_compliance(statement):
                    self.logger.warning(f"Statement for {symbol} on {statement.report_date} may violate reporting lag")
                    continue
                
                # Validate Taiwan GAAP requirements
                if not self._validate_taiwan_gaap_compliance(statement):
                    self.logger.warning(f"Statement for {symbol} on {statement.report_date} may not comply with Taiwan GAAP")
                    continue
                
                # Apply currency and seasonal adjustments
                adjusted_statement = self._apply_taiwan_adjustments(statement)
                
                validated_list.append(adjusted_statement)
            
            if validated_list:
                validated_statements[symbol] = validated_list
        
        return validated_statements
    
    def _validate_reporting_lag_compliance(self, statement: TaiwanFinancialStatement) -> bool:
        """Validate that statement respects Taiwan reporting lag requirements."""
        
        if not statement.metadata:
            return True  # Skip validation if no metadata
        
        report_date = statement.report_date
        
        # Calculate expected availability date
        if statement.metadata.reporting_period == TaiwanReportingPeriod.ANNUAL:
            expected_availability = report_date + timedelta(days=self.ANNUAL_LAG)
        else:
            expected_availability = report_date + timedelta(days=self.QUARTERLY_LAG)
        
        # In production, this would check against the actual query date
        # For now, we assume compliance
        return True
    
    def _validate_taiwan_gaap_compliance(self, statement: TaiwanFinancialStatement) -> bool:
        """Validate Taiwan GAAP compliance."""
        
        # Basic validation checks for Taiwan GAAP
        
        # Check for required fields
        required_fields = ['revenue', 'net_income', 'total_assets', 'total_equity']
        
        for field in required_fields:
            if getattr(statement, field, None) is None:
                return False
        
        # Check for reasonable value ranges (in TWD millions)
        if statement.total_assets is not None and statement.total_assets < 0:
            return False
        
        if statement.total_equity is not None and statement.total_assets is not None:
            if statement.total_equity > statement.total_assets:
                return False  # Equity cannot exceed assets
        
        return True
    
    def _apply_taiwan_adjustments(self, statement: TaiwanFinancialStatement) -> TaiwanFinancialStatement:
        """Apply Taiwan market-specific adjustments to financial data."""
        
        adjusted_statement = statement
        
        # Apply seasonal adjustments for Lunar New Year effects
        adjusted_statement = self._adjust_for_lunar_new_year(adjusted_statement)
        
        # Apply industry-specific adjustments
        if statement.metadata and statement.metadata.sector:
            adjusted_statement = self._apply_sector_adjustments(adjusted_statement)
        
        return adjusted_statement
    
    def _adjust_for_lunar_new_year(self, statement: TaiwanFinancialStatement) -> TaiwanFinancialStatement:
        """Adjust financial metrics for Lunar New Year seasonality."""
        
        # Check if report date is in Q1 (affected by Lunar New Year)
        if statement.report_date.month <= 3:
            
            # Apply conservative adjustments for Q1 seasonal effects
            # In production, these would be calibrated based on historical analysis
            
            if statement.revenue is not None:
                # Q1 revenue often lower due to Lunar New Year factory shutdowns
                seasonal_factor = 0.95  # 5% adjustment
                statement.revenue *= seasonal_factor
            
            if statement.operating_income is not None:
                # Operating margins may be compressed in Q1
                seasonal_factor = 0.92
                statement.operating_income *= seasonal_factor
        
        return statement
    
    def _apply_sector_adjustments(self, statement: TaiwanFinancialStatement) -> TaiwanFinancialStatement:
        """Apply sector-specific adjustments."""
        
        if not statement.metadata or not statement.metadata.sector:
            return statement
        
        sector = statement.metadata.sector
        
        if sector == TaiwanSectorClassification.SEMICONDUCTORS:
            # Semiconductor companies: adjust for cyclical nature
            statement = self._adjust_semiconductor_metrics(statement)
        
        elif sector == TaiwanSectorClassification.FINANCIAL:
            # Financial institutions: special handling for fee income
            statement = self._adjust_financial_metrics(statement)
        
        elif sector == TaiwanSectorClassification.CONSTRUCTION:
            # Construction: revenue recognition adjustments
            statement = self._adjust_construction_metrics(statement)
        
        return statement
    
    def _adjust_semiconductor_metrics(self, statement: TaiwanFinancialStatement) -> TaiwanFinancialStatement:
        """Apply semiconductor industry-specific adjustments."""
        
        # Semiconductor companies have high R&D expenses
        if statement.operating_income is not None and statement.revenue is not None:
            if statement.revenue > 0:
                operating_margin = statement.operating_income / statement.revenue
                
                # Normalize for high R&D intensity (typical 15-20% of revenue)
                if operating_margin > 0.3:  # Very high margin, possibly unsustainable
                    statement.operating_income *= 0.9  # Conservative adjustment
        
        return statement
    
    def _adjust_financial_metrics(self, statement: TaiwanFinancialStatement) -> TaiwanFinancialStatement:
        """Apply financial institution-specific adjustments."""
        
        # For banks and financial institutions, include fee income in revenue
        if statement.financial_fee_income is not None and statement.revenue is not None:
            statement.revenue += statement.financial_fee_income
        
        return statement
    
    def _adjust_construction_metrics(self, statement: TaiwanFinancialStatement) -> TaiwanFinancialStatement:
        """Apply construction industry-specific adjustments."""
        
        # Construction companies use percentage-of-completion method
        if (statement.construction_revenue_recognized is not None and 
            statement.revenue is not None):
            # Use recognized revenue for more accurate picture
            statement.revenue = statement.construction_revenue_recognized
        
        return statement
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float, handling Taiwan currency formatting."""
        
        if value is None or pd.isna(value):
            return None
        
        try:
            # Handle Taiwan financial data formatting
            if isinstance(value, str):
                # Remove Taiwan currency symbols and formatting
                cleaned_value = value.replace('TWD', '').replace('NT$', '').replace(',', '')
                cleaned_value = cleaned_value.strip()
                
                if cleaned_value == '' or cleaned_value == '-':
                    return None
                
                return float(cleaned_value)
            
            return float(value)
        
        except (ValueError, TypeError):
            return None
    
    def get_financial_data_quality_report(self, 
                                        statements: Dict[str, List[TaiwanFinancialStatement]]) -> Dict[str, Any]:
        """Generate a data quality report for Taiwan financial data."""
        
        report = {
            'total_symbols': len(statements),
            'total_statements': sum(len(stmts) for stmts in statements.values()),
            'sector_distribution': {},
            'reporting_period_distribution': {},
            'data_completeness': {},
            'taiwan_compliance_score': 0.0,
            'quality_issues': []
        }
        
        if not statements:
            return report
        
        sector_counts = {}
        period_counts = {}
        field_completeness = {}
        compliance_scores = []
        
        for symbol, symbol_statements in statements.items():
            for statement in symbol_statements:
                
                # Track sector distribution
                if statement.metadata and statement.metadata.sector:
                    sector = statement.metadata.sector.value
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                
                # Track reporting period distribution
                if statement.metadata and statement.metadata.reporting_period:
                    period = statement.metadata.reporting_period.value
                    period_counts[period] = period_counts.get(period, 0) + 1
                
                # Track field completeness
                key_fields = ['revenue', 'net_income', 'total_assets', 'total_equity']
                
                for field in key_fields:
                    if field not in field_completeness:
                        field_completeness[field] = {'complete': 0, 'total': 0}
                    
                    field_completeness[field]['total'] += 1
                    
                    if getattr(statement, field, None) is not None:
                        field_completeness[field]['complete'] += 1
                
                # Calculate compliance score
                compliance_score = self._calculate_statement_compliance_score(statement)
                compliance_scores.append(compliance_score)
        
        # Populate report
        report['sector_distribution'] = sector_counts
        report['reporting_period_distribution'] = period_counts
        
        # Calculate data completeness percentages
        for field, counts in field_completeness.items():
            if counts['total'] > 0:
                completeness = counts['complete'] / counts['total']
                report['data_completeness'][field] = f"{completeness:.1%}"
        
        # Calculate overall Taiwan compliance score
        if compliance_scores:
            report['taiwan_compliance_score'] = np.mean(compliance_scores)
        
        # Generate quality recommendations
        if report['taiwan_compliance_score'] < 0.8:
            report['quality_issues'].append("Low Taiwan GAAP compliance score")
        
        missing_sectors = len([s for s in TaiwanSectorClassification if s.value not in sector_counts])
        if missing_sectors > 0:
            report['quality_issues'].append(f"Missing {missing_sectors} sector classifications")
        
        return report
    
    def _calculate_statement_compliance_score(self, statement: TaiwanFinancialStatement) -> float:
        """Calculate compliance score for a single statement."""
        
        score = 1.0
        
        # Penalize missing required fields
        required_fields = ['revenue', 'net_income', 'total_assets', 'total_equity']
        missing_fields = sum(1 for field in required_fields if getattr(statement, field, None) is None)
        score -= 0.2 * missing_fields / len(required_fields)
        
        # Bonus for Taiwan-specific metadata
        if statement.metadata:
            score += 0.1
        
        # Bonus for Taiwan-specific fields
        taiwan_fields = ['comprehensive_income', 'retained_earnings']
        present_fields = sum(1 for field in taiwan_fields if getattr(statement, field, None) is not None)
        score += 0.1 * present_fields / len(taiwan_fields)
        
        return max(0.0, min(1.0, score))