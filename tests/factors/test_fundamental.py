"""
Tests for fundamental factor calculations.

This test suite validates the implementation of fundamental factors
including value, quality, and growth factors with Taiwan market compliance.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Optional

# Import the modules to test
from src.factors.fundamental import (
    FundamentalFactorCalculator, FundamentalFactors, 
    FinancialStatement, MarketData
)
from src.factors.value import (
    ValueFactors, PERatioCalculator, PBRatioCalculator, 
    EVEBITDACalculator, PriceSalesCalculator
)
from src.factors.quality import (
    QualityFactors, ROEROACalculator, DebtEquityCalculator,
    OperatingMarginCalculator, EarningsQualityCalculator
)
from src.factors.growth import (
    GrowthFactors, RevenueGrowthCalculator, EarningsGrowthCalculator,
    BookValueGrowthCalculator, AnalystRevisionCalculator
)
from src.factors.taiwan_financials import (
    TaiwanFinancialDataHandler, TaiwanFinancialStatement,
    TaiwanFinancialMetadata, TaiwanReportingPeriod, TaiwanSectorClassification
)
from src.factors.base import FactorMetadata, FactorCategory, FactorFrequency, FactorResult


class TestFinancialStatement:
    """Test FinancialStatement data class."""
    
    def test_financial_statement_creation(self):
        """Test creation of financial statement."""
        statement = FinancialStatement(
            symbol="2330.TW",
            report_date=date(2024, 3, 31),
            revenue=100000.0,
            net_income=25000.0,
            total_assets=500000.0,
            total_equity=300000.0,
            shares_outstanding=1000000.0
        )
        
        assert statement.symbol == "2330.TW"
        assert statement.revenue == 100000.0
        assert statement.book_value_per_share == 300.0  # 300000 / 1000
    
    def test_financial_statement_post_init(self):
        """Test post_init calculations."""
        statement = FinancialStatement(
            symbol="2330.TW",
            report_date=date(2024, 3, 31),
            total_equity=300000.0,
            shares_outstanding=1500.0
        )
        
        # Should calculate book value per share
        assert statement.book_value_per_share == 200.0


class TestMarketData:
    """Test MarketData data class."""
    
    def test_market_data_creation(self):
        """Test creation of market data."""
        market_data = MarketData(
            symbol="2330.TW",
            date=date(2024, 6, 30),
            price=500.0,
            shares_outstanding=1000.0
        )
        
        assert market_data.symbol == "2330.TW"
        assert market_data.price == 500.0
        assert market_data.market_cap == 500000.0  # 500 * 1000


class MockPITQueryEngine:
    """Mock PIT query engine for testing."""
    
    def __init__(self):
        self.financial_data = {}
        self.market_data = {}
    
    def query(self, query):
        """Mock query method."""
        # Return mock financial data
        return pd.DataFrame({
            'symbol': ['2330.TW', '2454.TW'],
            'report_date': [date(2024, 3, 31), date(2024, 3, 31)],
            'revenue': [100000.0, 50000.0],
            'net_income': [25000.0, 12500.0],
            'total_assets': [500000.0, 250000.0],
            'total_equity': [300000.0, 150000.0],
            'total_debt': [100000.0, 50000.0],
            'operating_income': [30000.0, 15000.0],
            'shares_outstanding': [1000.0, 500.0],
            'operating_cash_flow': [28000.0, 14000.0]
        })


class TestPERatioCalculator:
    """Test P/E ratio calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = MockPITQueryEngine()
        self.calculator = PERatioCalculator(self.mock_engine)
    
    def test_pe_ratio_metadata(self):
        """Test P/E ratio metadata."""
        metadata = self.calculator.metadata
        
        assert metadata.name == "pe_ratio_ttm"
        assert metadata.category == FactorCategory.FUNDAMENTAL
        assert metadata.frequency == FactorFrequency.DAILY
        assert metadata.taiwan_specific == True
    
    def test_pe_ratio_calculation(self):
        """Test P/E ratio calculation logic."""
        symbols = ["2330.TW"]
        as_of_date = date(2024, 6, 30)
        
        # Mock the data retrieval methods
        financial_data = {
            "2330.TW": [
                FinancialStatement(
                    symbol="2330.TW",
                    report_date=date(2024, 3, 31),
                    net_income=6000.0  # TTM net income
                ),
                FinancialStatement(
                    symbol="2330.TW", 
                    report_date=date(2023, 12, 31),
                    net_income=6500.0
                ),
                FinancialStatement(
                    symbol="2330.TW",
                    report_date=date(2023, 9, 30), 
                    net_income=6200.0
                ),
                FinancialStatement(
                    symbol="2330.TW",
                    report_date=date(2023, 6, 30),
                    net_income=6800.0
                )
            ]
        }
        
        market_data = {
            "2330.TW": MarketData(
                symbol="2330.TW",
                date=as_of_date,
                price=500.0,
                market_cap=500000.0  # 500 * 1000 shares
            )
        }
        
        with patch.object(self.calculator, '_get_financial_data', return_value=financial_data):
            with patch.object(self.calculator, '_get_market_data', return_value=market_data):
                result = self.calculator.calculate(symbols, as_of_date)
        
        assert isinstance(result, FactorResult)
        assert result.factor_name == "pe_ratio_ttm"
        assert "2330.TW" in result.values
        
        # TTM earnings = 6000 + 6500 + 6200 + 6800 = 25500
        # P/E = 500000 / 25500 ≈ 19.6
        pe_ratio = result.values["2330.TW"]
        assert 19.0 < pe_ratio < 20.0


class TestROEROACalculator:
    """Test ROE/ROA calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = MockPITQueryEngine()
        self.roe_calculator = ROEROACalculator(self.mock_engine, "roe")
        self.roa_calculator = ROEROACalculator(self.mock_engine, "roa")
    
    def test_roe_calculator_metadata(self):
        """Test ROE calculator metadata."""
        metadata = self.roe_calculator.metadata
        
        assert metadata.name == "roe_ttm"
        assert metadata.category == FactorCategory.FUNDAMENTAL
        assert metadata.taiwan_specific == True
    
    def test_roa_calculator_metadata(self):
        """Test ROA calculator metadata."""
        metadata = self.roa_calculator.metadata
        
        assert metadata.name == "roa_ttm"
        assert metadata.category == FactorCategory.FUNDAMENTAL
    
    def test_roe_calculation(self):
        """Test ROE calculation."""
        symbols = ["2330.TW"]
        as_of_date = date(2024, 6, 30)
        
        financial_data = {
            "2330.TW": [
                FinancialStatement(
                    symbol="2330.TW",
                    report_date=date(2024, 3, 31),
                    net_income=6000.0,
                    total_equity=300000.0
                ),
                FinancialStatement(
                    symbol="2330.TW",
                    report_date=date(2023, 12, 31),
                    net_income=6500.0,
                    total_equity=295000.0
                ),
                FinancialStatement(
                    symbol="2330.TW",
                    report_date=date(2023, 9, 30),
                    net_income=6200.0,
                    total_equity=290000.0
                ),
                FinancialStatement(
                    symbol="2330.TW",
                    report_date=date(2023, 6, 30),
                    net_income=6800.0,
                    total_equity=285000.0
                )
            ]
        }
        
        with patch.object(self.roe_calculator, '_get_financial_data', return_value=financial_data):
            result = self.roe_calculator.calculate(symbols, as_of_date)
        
        assert isinstance(result, FactorResult)
        assert "2330.TW" in result.values
        
        # TTM earnings = 25500, Average equity ≈ 292500
        # ROE = 25500 / 292500 ≈ 0.087
        roe = result.values["2330.TW"]
        assert 0.08 < roe < 0.10


class TestRevenueGrowthCalculator:
    """Test revenue growth calculation."""
    
    def setup_method(self):
        """Set up test fixtures.""" 
        self.mock_engine = MockPITQueryEngine()
        self.calculator = RevenueGrowthCalculator(self.mock_engine)
    
    def test_revenue_growth_metadata(self):
        """Test revenue growth metadata."""
        metadata = self.calculator.metadata
        
        assert metadata.name == "revenue_growth_momentum"
        assert metadata.category == FactorCategory.FUNDAMENTAL
        assert metadata.lookback_days == 1095  # 3 years
    
    def test_revenue_growth_calculation(self):
        """Test revenue growth calculation."""
        symbols = ["2330.TW"]
        as_of_date = date(2024, 6, 30)
        
        # Create financial data with revenue growth
        financial_data = {
            "2330.TW": []
        }
        
        # Current year TTM: Q2'24, Q1'24, Q4'23, Q3'23
        current_quarters = [25000, 24000, 23000, 22000]  # Total: 94000
        
        # Previous year TTM: Q2'23, Q1'23, Q4'22, Q3'22  
        previous_quarters = [22000, 21000, 20000, 19000]  # Total: 82000
        
        # Create statements for 8 quarters
        for i, revenue in enumerate(current_quarters + previous_quarters):
            quarter_date = date(2024, 6, 30) - timedelta(days=i * 90)
            statement = FinancialStatement(
                symbol="2330.TW",
                report_date=quarter_date,
                revenue=revenue
            )
            financial_data["2330.TW"].append(statement)
        
        with patch.object(self.calculator, '_get_financial_data', return_value=financial_data):
            result = self.calculator.calculate(symbols, as_of_date)
        
        assert isinstance(result, FactorResult)
        assert "2330.TW" in result.values
        
        # YoY growth = (94000 - 82000) / 82000 ≈ 0.146 (14.6%)
        growth = result.values["2330.TW"]
        assert 0.10 < growth < 0.20


class TestTaiwanFinancialDataHandler:
    """Test Taiwan-specific financial data handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = MockPITQueryEngine()
        self.handler = TaiwanFinancialDataHandler(self.mock_engine)
    
    def test_quarterly_cutoff_calculation(self):
        """Test quarterly reporting lag cutoff calculation."""
        # Test date in July (Q2 data should be available)
        test_date = date(2024, 7, 15)
        cutoff = self.handler._calculate_quarterly_cutoff_date(test_date)
        
        # Q2 ended June 30, with 60-day lag = August 29
        # But since test date is July 15, cutoff should be July 15
        expected_cutoff = test_date
        assert cutoff <= expected_cutoff
    
    def test_annual_cutoff_calculation(self):
        """Test annual reporting lag cutoff calculation.""" 
        test_date = date(2024, 5, 15)  # May
        cutoff = self.handler._calculate_annual_cutoff_date(test_date)
        
        # Previous year end: Dec 31, 2023 + 90 days = March 31, 2024
        expected_cutoff = date(2024, 3, 31)
        assert cutoff >= expected_cutoff
    
    def test_sector_classification(self):
        """Test Taiwan sector classification."""
        
        # Test semiconductor classification
        sector = self.handler._get_sector_classification("2330.TW")  # TSM
        assert sector == TaiwanSectorClassification.SEMICONDUCTORS
        
        # Test electronics classification  
        sector = self.handler._get_sector_classification("2317.TW")  # Hon Hai
        assert sector == TaiwanSectorClassification.ELECTRONICS
        
        # Test financial classification
        sector = self.handler._get_sector_classification("2881.TW")  # Fubon Financial
        assert sector == TaiwanSectorClassification.FINANCIAL
    
    def test_taiwan_financial_statement_creation(self):
        """Test Taiwan financial statement creation."""
        metadata = TaiwanFinancialMetadata(
            symbol="2330.TW",
            report_date=date(2024, 3, 31),
            reporting_period=TaiwanReportingPeriod.Q1,
            fiscal_year=2024,
            fiscal_quarter=1
        )
        
        statement = TaiwanFinancialStatement(
            symbol="2330.TW", 
            report_date=date(2024, 3, 31),
            revenue=25000.0,
            net_income=6000.0,
            total_assets=500000.0,
            total_equity=300000.0,
            metadata=metadata,
            comprehensive_income=6200.0,
            retained_earnings=150000.0
        )
        
        assert statement.symbol == "2330.TW"
        assert statement.metadata.sector is None  # Not set in this test
        assert statement.comprehensive_income == 6200.0
        
        # Test conversion to base statement
        base_statement = statement.to_base_statement()
        assert isinstance(base_statement, FinancialStatement)
        assert base_statement.revenue == 25000.0


class TestFundamentalFactorsIntegration:
    """Integration tests for fundamental factors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = MockPITQueryEngine()
        self.fundamental_factors = FundamentalFactors(self.mock_engine)
    
    @patch('src.factors.value.ValueFactors')
    @patch('src.factors.quality.QualityFactors') 
    @patch('src.factors.growth.GrowthFactors')
    def test_calculate_all_fundamental_factors(self, mock_growth, mock_quality, mock_value):
        """Test calculation of all fundamental factors."""
        
        # Mock the factor calculators
        mock_value_instance = Mock()
        mock_value.return_value = mock_value_instance
        mock_value_instance.calculate_all_value_factors.return_value = {
            "pe_ratio_ttm": Mock(spec=FactorResult),
            "pb_ratio_adjusted": Mock(spec=FactorResult)
        }
        
        mock_quality_instance = Mock()
        mock_quality.return_value = mock_quality_instance
        mock_quality_instance.calculate_all_quality_factors.return_value = {
            "roe_ttm": Mock(spec=FactorResult),
            "roa_ttm": Mock(spec=FactorResult)
        }
        
        mock_growth_instance = Mock()
        mock_growth.return_value = mock_growth_instance
        mock_growth_instance.calculate_all_growth_factors.return_value = {
            "revenue_growth_momentum": Mock(spec=FactorResult),
            "earnings_growth_quality": Mock(spec=FactorResult)
        }
        
        # Test the integration
        symbols = ["2330.TW", "2454.TW"]
        as_of_date = date(2024, 6, 30)
        
        results = self.fundamental_factors.calculate_all_fundamental_factors(symbols, as_of_date)
        
        # Should have results from all three factor groups
        assert len(results) == 6
        assert "pe_ratio_ttm" in results
        assert "roe_ttm" in results  
        assert "revenue_growth_momentum" in results
    
    def test_data_quality_validation(self):
        """Test financial data quality validation."""
        
        # Create mock financial data
        financial_data = {
            "2330.TW": [
                FinancialStatement(
                    symbol="2330.TW",
                    report_date=date(2024, 3, 31),
                    revenue=25000.0,
                    net_income=6000.0,
                    total_assets=500000.0,
                    total_equity=300000.0
                ),
                FinancialStatement(
                    symbol="2330.TW", 
                    report_date=date(2023, 12, 31),
                    revenue=24000.0,
                    net_income=5800.0,
                    total_assets=480000.0,
                    total_equity=290000.0
                )
            ],
            "2454.TW": [
                FinancialStatement(
                    symbol="2454.TW",
                    report_date=date(2024, 3, 31), 
                    revenue=12000.0,
                    net_income=None,  # Missing data
                    total_assets=250000.0,
                    total_equity=150000.0
                )
            ]
        }
        
        # Create a test calculator to access validation method
        calculator = FundamentalFactorCalculator(self.mock_engine, Mock())
        validation_results = calculator.validate_financial_data_quality(financial_data)
        
        assert validation_results['total_symbols'] == 2
        assert validation_results['symbols_with_data'] == 2
        assert validation_results['data_quality_score'] > 0.0
        assert len(validation_results['insufficient_history_symbols']) > 0  # 2454.TW has insufficient data


class TestTaiwanComplianceValidation:
    """Test Taiwan market compliance validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = MockPITQueryEngine()
        self.handler = TaiwanFinancialDataHandler(self.mock_engine)
    
    def test_reporting_lag_compliance(self):
        """Test Taiwan reporting lag compliance validation."""
        
        # Create a statement that should comply with reporting lag
        metadata = TaiwanFinancialMetadata(
            symbol="2330.TW",
            report_date=date(2024, 3, 31),  # Q1 end
            reporting_period=TaiwanReportingPeriod.Q1,
            fiscal_year=2024,
            fiscal_quarter=1
        )
        
        statement = TaiwanFinancialStatement(
            symbol="2330.TW",
            report_date=date(2024, 3, 31),
            revenue=25000.0,
            metadata=metadata
        )
        
        # Test validation (should pass)
        is_compliant = self.handler._validate_reporting_lag_compliance(statement)
        assert is_compliant  # Should be True in current implementation
    
    def test_taiwan_gaap_compliance(self):
        """Test Taiwan GAAP compliance validation."""
        
        # Create a compliant statement
        compliant_statement = TaiwanFinancialStatement(
            symbol="2330.TW",
            report_date=date(2024, 3, 31),
            revenue=25000.0,
            net_income=6000.0,
            total_assets=500000.0,
            total_equity=300000.0
        )
        
        is_compliant = self.handler._validate_taiwan_gaap_compliance(compliant_statement)
        assert is_compliant
        
        # Create a non-compliant statement (missing required fields)
        non_compliant_statement = TaiwanFinancialStatement(
            symbol="2330.TW",
            report_date=date(2024, 3, 31),
            revenue=None,  # Missing required field
            net_income=None,  # Missing required field
            total_assets=500000.0,
            total_equity=300000.0
        )
        
        is_compliant = self.handler._validate_taiwan_gaap_compliance(non_compliant_statement)
        assert not is_compliant
    
    def test_lunar_new_year_adjustment(self):
        """Test Lunar New Year seasonality adjustment."""
        
        # Create Q1 statement (affected by Lunar New Year)
        q1_statement = TaiwanFinancialStatement(
            symbol="2330.TW",
            report_date=date(2024, 2, 29),  # Q1
            revenue=20000.0,
            operating_income=5000.0
        )
        
        adjusted_statement = self.handler._adjust_for_lunar_new_year(q1_statement)
        
        # Revenue should be adjusted downward for Q1
        assert adjusted_statement.revenue < 20000.0
        assert adjusted_statement.operating_income < 5000.0
        
        # Q3 statement should not be affected
        q3_statement = TaiwanFinancialStatement(
            symbol="2330.TW",
            report_date=date(2024, 9, 30),  # Q3
            revenue=25000.0,
            operating_income=6000.0
        )
        
        adjusted_q3_statement = self.handler._adjust_for_lunar_new_year(q3_statement)
        
        # Should remain unchanged
        assert adjusted_q3_statement.revenue == 25000.0
        assert adjusted_q3_statement.operating_income == 6000.0
    
    def test_data_quality_report(self):
        """Test Taiwan financial data quality report generation."""
        
        # Create test data with various quality characteristics
        statements = {
            "2330.TW": [
                TaiwanFinancialStatement(
                    symbol="2330.TW",
                    report_date=date(2024, 3, 31),
                    revenue=25000.0,
                    net_income=6000.0,
                    total_assets=500000.0,
                    total_equity=300000.0,
                    metadata=TaiwanFinancialMetadata(
                        symbol="2330.TW",
                        report_date=date(2024, 3, 31),
                        reporting_period=TaiwanReportingPeriod.Q1,
                        fiscal_year=2024,
                        fiscal_quarter=1,
                        sector=TaiwanSectorClassification.SEMICONDUCTORS
                    )
                )
            ],
            "2881.TW": [
                TaiwanFinancialStatement(
                    symbol="2881.TW",
                    report_date=date(2024, 3, 31),
                    revenue=15000.0,
                    net_income=None,  # Missing data
                    total_assets=200000.0,
                    total_equity=100000.0,
                    metadata=TaiwanFinancialMetadata(
                        symbol="2881.TW", 
                        report_date=date(2024, 3, 31),
                        reporting_period=TaiwanReportingPeriod.Q1,
                        fiscal_year=2024,
                        fiscal_quarter=1,
                        sector=TaiwanSectorClassification.FINANCIAL
                    )
                )
            ]
        }
        
        report = self.handler.get_financial_data_quality_report(statements)
        
        assert report['total_symbols'] == 2
        assert report['total_statements'] == 2
        assert 'SEMICONDUCTORS' in report['sector_distribution']
        assert 'FINANCIAL' in report['sector_distribution']
        assert 'Q1' in report['reporting_period_distribution']
        assert 'revenue' in report['data_completeness']
        assert report['taiwan_compliance_score'] > 0.0


if __name__ == "__main__":
    pytest.main([__file__])