"""
Tests for financial reporting calendar functionality.
"""

import pytest
from datetime import date, datetime
from src.finlab_downloader.scheduler.financial_calendar import (
    FinancialReportingCalendar,
    IndustryType,
    ReportingPeriod,
    ReportingDeadline,
    FinancialStatementSchedule
)


class TestFinancialReportingCalendar:
    """Test cases for FinancialReportingCalendar."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calendar = FinancialReportingCalendar()

    def test_general_company_deadlines(self):
        """Test deadline calculation for general companies."""
        # Q1 deadline: May 1-15
        deadline = self.calendar.get_reporting_deadline(
            IndustryType.GENERAL, ReportingPeriod.Q1, 2024
        )
        assert deadline is not None
        assert deadline.earliest_date == date(2024, 5, 1)
        assert deadline.latest_date == date(2024, 5, 15)

        # Q4 deadline: March 1-31 (next year)
        deadline = self.calendar.get_reporting_deadline(
            IndustryType.GENERAL, ReportingPeriod.Q4, 2024
        )
        assert deadline is not None
        assert deadline.earliest_date == date(2025, 3, 1)
        assert deadline.latest_date == date(2025, 3, 31)

    def test_financial_industry_deadlines(self):
        """Test deadline calculation for financial industry."""
        # Q2 deadline: August 1-31 (different from general)
        deadline = self.calendar.get_reporting_deadline(
            IndustryType.FINANCIAL, ReportingPeriod.Q2, 2024
        )
        assert deadline is not None
        assert deadline.earliest_date == date(2024, 8, 1)
        assert deadline.latest_date == date(2024, 8, 31)

    def test_insurance_industry_deadlines(self):
        """Test deadline calculation for insurance industry."""
        # Q1 deadline: April 1-30 (earlier than general)
        deadline = self.calendar.get_reporting_deadline(
            IndustryType.INSURANCE, ReportingPeriod.Q1, 2024
        )
        assert deadline is not None
        assert deadline.earliest_date == date(2024, 4, 1)
        assert deadline.latest_date == date(2024, 4, 30)

    def test_ky_stock_deadlines(self):
        """Test deadline calculation for KY stocks."""
        # KY stocks only have Q2 and Q4
        q2_deadline = self.calendar.get_reporting_deadline(
            IndustryType.KY_STOCK, ReportingPeriod.Q2, 2024
        )
        assert q2_deadline is not None

        # Q1 should not be available for KY stocks
        q1_deadline = self.calendar.get_reporting_deadline(
            IndustryType.KY_STOCK, ReportingPeriod.Q1, 2024
        )
        assert q1_deadline is None

    def test_get_all_deadlines_for_year(self):
        """Test getting all deadlines for a fiscal year."""
        deadlines = self.calendar.get_all_deadlines_for_year(2024, IndustryType.GENERAL)

        # General companies have 4 reporting periods
        assert len(deadlines) == 4

        # Should be sorted by earliest date
        dates = [d.earliest_date for d in deadlines]
        assert dates == sorted(dates)

    def test_get_next_deadline(self):
        """Test finding next upcoming deadline."""
        # Test from a date in January 2024
        test_date = date(2024, 1, 15)

        next_deadline = self.calendar.get_next_deadline(IndustryType.GENERAL, test_date)
        assert next_deadline is not None
        assert next_deadline.latest_date >= test_date

    def test_create_schedule(self):
        """Test creating a financial statement schedule."""
        schedule = self.calendar.create_schedule(
            company_ticker="2330",
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024,
            priority=1
        )

        assert schedule is not None
        assert schedule.company_ticker == "2330"
        assert schedule.industry_type == IndustryType.GENERAL
        assert schedule.reporting_period == ReportingPeriod.Q1
        assert schedule.fiscal_year == 2024
        assert schedule.earliest_possible == date(2024, 5, 1)
        assert schedule.latest_possible == date(2024, 5, 15)

    def test_get_reporting_periods_for_industry(self):
        """Test getting valid reporting periods for different industries."""
        # General, Financial, Insurance have all 4 periods
        for industry in [IndustryType.GENERAL, IndustryType.FINANCIAL, IndustryType.INSURANCE]:
            periods = self.calendar.get_reporting_periods_for_industry(industry)
            assert len(periods) == 4
            assert ReportingPeriod.Q1 in periods
            assert ReportingPeriod.Q2 in periods
            assert ReportingPeriod.Q3 in periods
            assert ReportingPeriod.Q4 in periods

        # KY stocks only have Q2 and Q4
        ky_periods = self.calendar.get_reporting_periods_for_industry(IndustryType.KY_STOCK)
        assert len(ky_periods) == 2
        assert ReportingPeriod.Q2 in ky_periods
        assert ReportingPeriod.Q4 in ky_periods
        assert ReportingPeriod.Q1 not in ky_periods
        assert ReportingPeriod.Q3 not in ky_periods

    def test_is_reporting_required(self):
        """Test checking if reporting is required."""
        # General companies require all periods
        assert self.calendar.is_reporting_required(IndustryType.GENERAL, ReportingPeriod.Q1)
        assert self.calendar.is_reporting_required(IndustryType.GENERAL, ReportingPeriod.Q2)
        assert self.calendar.is_reporting_required(IndustryType.GENERAL, ReportingPeriod.Q3)
        assert self.calendar.is_reporting_required(IndustryType.GENERAL, ReportingPeriod.Q4)

        # KY stocks don't require Q1 and Q3
        assert not self.calendar.is_reporting_required(IndustryType.KY_STOCK, ReportingPeriod.Q1)
        assert self.calendar.is_reporting_required(IndustryType.KY_STOCK, ReportingPeriod.Q2)
        assert not self.calendar.is_reporting_required(IndustryType.KY_STOCK, ReportingPeriod.Q3)
        assert self.calendar.is_reporting_required(IndustryType.KY_STOCK, ReportingPeriod.Q4)

    def test_get_industry_from_ticker(self):
        """Test industry detection from ticker symbols."""
        # Test KY stock detection
        assert self.calendar.get_industry_from_ticker("1234-KY") == IndustryType.KY_STOCK
        assert self.calendar.get_industry_from_ticker("ABCD-KY") == IndustryType.KY_STOCK

        # Test financial institution detection (simplified)
        assert self.calendar.get_industry_from_ticker("2880") == IndustryType.FINANCIAL
        assert self.calendar.get_industry_from_ticker("2881") == IndustryType.FINANCIAL

        # Test insurance company detection (simplified)
        assert self.calendar.get_industry_from_ticker("2816") == IndustryType.INSURANCE
        assert self.calendar.get_industry_from_ticker("2823") == IndustryType.INSURANCE

        # Test general company (default)
        assert self.calendar.get_industry_from_ticker("2330") == IndustryType.GENERAL
        assert self.calendar.get_industry_from_ticker("2317") == IndustryType.GENERAL


class TestReportingDeadline:
    """Test cases for ReportingDeadline dataclass."""

    def test_contains_date(self):
        """Test date range checking."""
        deadline = ReportingDeadline(
            period=ReportingPeriod.Q1,
            industry=IndustryType.GENERAL,
            earliest_date=date(2024, 5, 1),
            latest_date=date(2024, 5, 15),
            description="Test deadline"
        )

        assert deadline.contains_date(date(2024, 5, 1))  # Earliest
        assert deadline.contains_date(date(2024, 5, 15))  # Latest
        assert deadline.contains_date(date(2024, 5, 8))  # Middle
        assert not deadline.contains_date(date(2024, 4, 30))  # Before
        assert not deadline.contains_date(date(2024, 5, 16))  # After

    def test_days_until_deadline(self):
        """Test calculation of days until deadline."""
        deadline = ReportingDeadline(
            period=ReportingPeriod.Q1,
            industry=IndustryType.GENERAL,
            earliest_date=date(2024, 5, 1),
            latest_date=date(2024, 5, 15),
            description="Test deadline"
        )

        # Test from a specific date
        from_date = date(2024, 5, 10)
        days_until = deadline.days_until_deadline(from_date)
        assert days_until == 5  # 5 days from May 10 to May 15


class TestFinancialStatementSchedule:
    """Test cases for FinancialStatementSchedule dataclass."""

    def test_period_display(self):
        """Test period display formatting."""
        schedule = FinancialStatementSchedule(
            company_ticker="2330",
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024,
            expected_date=date(2024, 5, 7),
            earliest_possible=date(2024, 5, 1),
            latest_possible=date(2024, 5, 15)
        )

        assert schedule.period_display == "2024 Q1"

    def test_is_overdue(self):
        """Test overdue detection."""
        schedule = FinancialStatementSchedule(
            company_ticker="2330",
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024,
            expected_date=date(2024, 5, 7),
            earliest_possible=date(2024, 5, 1),
            latest_possible=date(2024, 5, 15)
        )

        # Not overdue if checked before deadline
        assert not schedule.is_overdue(date(2024, 5, 14))

        # Overdue if checked after deadline
        assert schedule.is_overdue(date(2024, 5, 16))

    def test_days_until_expected(self):
        """Test calculation of days until expected release."""
        schedule = FinancialStatementSchedule(
            company_ticker="2330",
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024,
            expected_date=date(2024, 5, 7),
            earliest_possible=date(2024, 5, 1),
            latest_possible=date(2024, 5, 15)
        )

        # Test from a specific date
        days_until = schedule.days_until_expected(date(2024, 5, 1))
        assert days_until == 6  # 6 days from May 1 to May 7