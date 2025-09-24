"""
Demonstration of fundamental factors for Taiwan market.

This script shows how to use the fundamental factor system with
Taiwan market compliance and 60-day reporting lag enforcement.
"""

import sys
import os
from datetime import date, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.factors.taiwan_financials import (
    TaiwanFinancialDataHandler, TaiwanFinancialStatement, 
    TaiwanFinancialMetadata, TaiwanReportingPeriod, TaiwanSectorClassification
)
from src.factors.base import FactorMetadata, FactorCategory


class DemoDataGenerator:
    """Generate sample data for demonstration purposes."""
    
    def __init__(self):
        self.taiwan_symbols = [
            "2330.TW",  # Taiwan Semiconductor (TSM)
            "2454.TW",  # MediaTek
            "2317.TW",  # Hon Hai (Foxconn)
            "2881.TW",  # Fubon Financial
            "1101.TW",  # Taiwan Cement
        ]
    
    def generate_sample_financial_data(self) -> Dict[str, List[TaiwanFinancialStatement]]:
        """Generate sample financial statements for demo."""
        
        financial_data = {}
        
        for symbol in self.taiwan_symbols:
            statements = []
            
            # Generate 8 quarters of data (2 years)
            for i in range(8):
                report_date = date(2024, 3, 31) - timedelta(days=i * 90)
                
                # Base values with some growth
                base_revenue = 25000 * (1 + 0.1 * (8-i)/8)  # Growth over time
                base_income = base_revenue * 0.25  # 25% net margin
                base_assets = base_revenue * 5     # 5x asset turnover
                base_equity = base_assets * 0.6    # 60% equity ratio
                
                # Add some company-specific variation
                if "2330" in symbol:  # TSM - semiconductor
                    multiplier = 4.0
                elif "2454" in symbol:  # MediaTek
                    multiplier = 2.0
                elif "2881" in symbol:  # Financial
                    base_income = base_revenue * 0.15  # Lower margins for financial
                    multiplier = 1.0
                else:
                    multiplier = 1.5
                
                # Create Taiwan-specific metadata
                quarter = ((12 - report_date.month) // 3) + 1
                metadata = TaiwanFinancialMetadata(
                    symbol=symbol,
                    report_date=report_date,
                    reporting_period=TaiwanReportingPeriod(f"Q{quarter}"),
                    fiscal_year=report_date.year,
                    fiscal_quarter=quarter,
                    sector=self._get_demo_sector(symbol)
                )
                
                statement = TaiwanFinancialStatement(
                    symbol=symbol,
                    report_date=report_date,
                    revenue=base_revenue * multiplier,
                    net_income=base_income * multiplier,
                    total_assets=base_assets * multiplier,
                    total_equity=base_equity * multiplier,
                    total_debt=(base_assets - base_equity) * multiplier,
                    operating_income=base_income * multiplier * 1.2,
                    ebitda=base_income * multiplier * 1.4,
                    shares_outstanding=1000.0 * multiplier,
                    operating_cash_flow=base_income * multiplier * 1.1,
                    metadata=metadata
                )
                
                statements.append(statement)
            
            financial_data[symbol] = statements
        
        return financial_data
    
    def _get_demo_sector(self, symbol: str) -> TaiwanSectorClassification:
        """Get sector classification for demo symbols."""
        if "2330" in symbol or "2454" in symbol:
            return TaiwanSectorClassification.SEMICONDUCTORS
        elif "2317" in symbol:
            return TaiwanSectorClassification.ELECTRONICS
        elif "2881" in symbol:
            return TaiwanSectorClassification.FINANCIAL
        else:
            return TaiwanSectorClassification.TRADITIONAL_INDUSTRY


def demonstrate_taiwan_compliance():
    """Demonstrate Taiwan market compliance features."""
    
    print("🇹🇼 Taiwan Market Fundamental Factors Demo")
    print("=" * 50)
    
    # Create mock PIT engine for demo
    class MockPITEngine:
        def query(self, query):
            return pd.DataFrame()  # Empty for demo
    
    # Initialize Taiwan financial data handler
    mock_engine = MockPITEngine()
    handler = TaiwanFinancialDataHandler(mock_engine)
    
    # Generate sample data
    data_generator = DemoDataGenerator()
    financial_data = data_generator.generate_sample_financial_data()
    
    print(f"📊 Generated data for {len(financial_data)} Taiwan symbols:")
    for symbol in financial_data.keys():
        quarters = len(financial_data[symbol])
        print(f"  • {symbol}: {quarters} quarterly statements")
    
    print("\n💰 Value Factors Demonstration:")
    print("-" * 30)
    
    # Demonstrate P/E ratio calculation
    symbol = "2330.TW"
    statements = financial_data[symbol]
    
    if len(statements) >= 4:
        # Calculate TTM earnings
        ttm_earnings = sum(s.net_income for s in statements[:4] if s.net_income)
        latest_market_cap = statements[0].total_equity * 2  # Simplified market cap
        
        pe_ratio = latest_market_cap / ttm_earnings if ttm_earnings > 0 else None
        
        print(f"  📈 {symbol} P/E Ratio (TTM): {pe_ratio:.2f}" if pe_ratio else "  ❌ P/E calculation failed")
        
        # Calculate P/B ratio
        latest_equity = statements[0].total_equity
        pb_ratio = latest_market_cap / latest_equity if latest_equity > 0 else None
        
        print(f"  📊 {symbol} P/B Ratio: {pb_ratio:.2f}" if pb_ratio else "  ❌ P/B calculation failed")
    
    print("\n🏆 Quality Factors Demonstration:")
    print("-" * 30)
    
    # Demonstrate ROE calculation
    if len(statements) >= 4:
        ttm_income = sum(s.net_income for s in statements[:4] if s.net_income)
        avg_equity = np.mean([s.total_equity for s in statements[:4] if s.total_equity])
        
        roe = ttm_income / avg_equity if avg_equity > 0 else None
        
        print(f"  💎 {symbol} ROE (TTM): {roe:.1%}" if roe else "  ❌ ROE calculation failed")
        
        # Calculate debt-to-equity
        latest_debt = statements[0].total_debt or 0
        latest_equity = statements[0].total_equity or 1
        de_ratio = latest_debt / latest_equity
        
        print(f"  ⚖️ {symbol} Debt-to-Equity: {de_ratio:.2f}")
    
    print("\n📈 Growth Factors Demonstration:")
    print("-" * 30)
    
    # Demonstrate revenue growth
    if len(statements) >= 8:
        current_ttm_revenue = sum(s.revenue for s in statements[:4] if s.revenue)
        previous_ttm_revenue = sum(s.revenue for s in statements[4:8] if s.revenue)
        
        revenue_growth = (current_ttm_revenue - previous_ttm_revenue) / previous_ttm_revenue if previous_ttm_revenue > 0 else None
        
        print(f"  🚀 {symbol} Revenue Growth (YoY): {revenue_growth:.1%}" if revenue_growth else "  ❌ Revenue growth calculation failed")
        
        # Calculate earnings growth
        current_ttm_earnings = sum(s.net_income for s in statements[:4] if s.net_income)
        previous_ttm_earnings = sum(s.net_income for s in statements[4:8] if s.net_income)
        
        earnings_growth = (current_ttm_earnings - previous_ttm_earnings) / previous_ttm_earnings if previous_ttm_earnings > 0 else None
        
        print(f"  💰 {symbol} Earnings Growth (YoY): {earnings_growth:.1%}" if earnings_growth else "  ❌ Earnings growth calculation failed")
    
    print("\n🏛️ Taiwan Market Compliance:")
    print("-" * 30)
    
    # Test reporting lag compliance
    test_date = date(2024, 6, 30)
    quarterly_cutoff = handler._calculate_quarterly_cutoff_date(test_date)
    annual_cutoff = handler._calculate_annual_cutoff_date(test_date)
    
    print(f"  📅 As of date: {test_date}")
    print(f"  ⏰ Quarterly data cutoff: {quarterly_cutoff} (60-day lag)")
    print(f"  📋 Annual data cutoff: {annual_cutoff} (90-day lag)")
    
    # Test sector classification
    print(f"\n🏭 Sector Classifications:")
    for symbol in financial_data.keys():
        sector = handler._get_sector_classification(symbol)
        print(f"  • {symbol}: {sector.value}")
    
    # Generate data quality report
    print(f"\n📊 Data Quality Report:")
    print("-" * 20)
    
    quality_report = handler.get_financial_data_quality_report(financial_data)
    
    print(f"  📈 Total symbols: {quality_report['total_symbols']}")
    print(f"  📄 Total statements: {quality_report['total_statements']}")
    print(f"  🎯 Taiwan compliance score: {quality_report['taiwan_compliance_score']:.1%}")
    
    print(f"\n🏢 Sector distribution:")
    for sector, count in quality_report['sector_distribution'].items():
        print(f"  • {sector}: {count} statements")
    
    print(f"\n📅 Reporting period distribution:")
    for period, count in quality_report['reporting_period_distribution'].items():
        print(f"  • {period}: {count} statements")
    
    print(f"\n✨ Data completeness:")
    for field, completeness in quality_report['data_completeness'].items():
        print(f"  • {field}: {completeness}")
    
    if quality_report['quality_issues']:
        print(f"\n⚠️ Quality issues:")
        for issue in quality_report['quality_issues']:
            print(f"  • {issue}")
    else:
        print(f"\n✅ No quality issues detected")
    
    print("\n🎉 Demo completed successfully!")
    print("\n📝 Summary of 12 Fundamental Factors:")
    print("   Value Factors (4): P/E, P/B, EV/EBITDA, P/S")
    print("   Quality Factors (4): ROE, ROA, Debt-to-Equity, Operating Margin")
    print("   Growth Factors (4): Revenue Growth, Earnings Growth, Book Value Growth, Analyst Revisions")
    print("\n🇹🇼 Taiwan Market Features:")
    print("   • 60-day quarterly reporting lag enforcement")
    print("   • 90-day annual reporting lag enforcement")
    print("   • Taiwan GAAP compliance validation")
    print("   • Sector-specific adjustments")
    print("   • Lunar New Year seasonality handling")


if __name__ == "__main__":
    demonstrate_taiwan_compliance()