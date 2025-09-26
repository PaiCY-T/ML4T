"""
Financial Statement Reporting Calendar.

Implements industry-specific financial statement release schedules for Taiwan market.
"""

import logging
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Set
from calendar import monthrange

logger = logging.getLogger(__name__)


class IndustryType(Enum):
    """Industry types with different financial reporting schedules."""
    GENERAL = "general"                    # General companies
    FINANCIAL = "financial"               # Banks, financial services
    INSURANCE = "insurance"               # Insurance companies
    KY_STOCK = "ky_stock"                 # KY stocks (post-2021 rules)


class ReportingPeriod(Enum):
    """Financial reporting periods."""
    Q1 = "Q1"  # First quarter
    Q2 = "Q2"  # Second quarter (semi-annual)
    Q3 = "Q3"  # Third quarter
    Q4 = "Q4"  # Fourth quarter (annual)


@dataclass
class ReportingDeadline:
    """Represents a financial statement reporting deadline."""
    period: ReportingPeriod
    industry: IndustryType
    earliest_date: date
    latest_date: date
    description: str = ""

    def contains_date(self, check_date: date) -> bool:
        """Check if a date falls within the reporting deadline range."""
        return self.earliest_date <= check_date <= self.latest_date

    def days_until_deadline(self, from_date: Optional[date] = None) -> int:
        """Calculate days until the latest deadline."""
        if from_date is None:
            from_date = date.today()
        return (self.latest_date - from_date).days


@dataclass
class FinancialStatementSchedule:
    """Schedule for a specific financial statement release."""
    company_ticker: str
    industry_type: IndustryType
    reporting_period: ReportingPeriod
    fiscal_year: int
    expected_date: date
    earliest_possible: date
    latest_possible: date
    priority: int = 1  # 1=highest, 5=lowest
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def period_display(self) -> str:
        """Human-readable period display."""
        return f"{self.fiscal_year} {self.reporting_period.value}"

    def is_overdue(self, as_of_date: Optional[date] = None) -> bool:
        """Check if the statement is overdue."""
        if as_of_date is None:
            as_of_date = date.today()
        return as_of_date > self.latest_possible

    def days_until_expected(self, from_date: Optional[date] = None) -> int:
        """Calculate days until expected release."""
        if from_date is None:
            from_date = date.today()
        return (self.expected_date - from_date).days


class FinancialReportingCalendar:
    """
    Taiwan Financial Statement Reporting Calendar.

    Implements industry-specific reporting deadlines based on Taiwan regulations:
    - General companies: Q1(5-15), Q2(8-14), Q3(11-14), Q4(3-31)
    - Financial industry: Q1(5-15), Q2(8-31), Q3(11-14), Q4(3-31)
    - Insurance industry: Q1(4-30), Q2(8-31), Q3(10-31), Q4(3-31)
    - KY stocks (post-2021): Q2(8-31)
    """

    # Reporting deadline rules by industry type
    REPORTING_DEADLINES = {
        IndustryType.GENERAL: {
            ReportingPeriod.Q1: (5, 15),   # May 1-15
            ReportingPeriod.Q2: (8, 14),   # August 1-14
            ReportingPeriod.Q3: (11, 14),  # November 1-14
            ReportingPeriod.Q4: (3, 31),   # March 1-31 (next year)
        },
        IndustryType.FINANCIAL: {
            ReportingPeriod.Q1: (5, 15),   # May 1-15
            ReportingPeriod.Q2: (8, 31),   # August 1-31
            ReportingPeriod.Q3: (11, 14),  # November 1-14
            ReportingPeriod.Q4: (3, 31),   # March 1-31 (next year)
        },
        IndustryType.INSURANCE: {
            ReportingPeriod.Q1: (4, 30),   # April 1-30
            ReportingPeriod.Q2: (8, 31),   # August 1-31
            ReportingPeriod.Q3: (10, 31),  # October 1-31
            ReportingPeriod.Q4: (3, 31),   # March 1-31 (next year)
        },
        IndustryType.KY_STOCK: {
            # KY stocks only report semi-annual (Q2) and annual (Q4) post-2021
            ReportingPeriod.Q2: (8, 31),   # August 1-31
            ReportingPeriod.Q4: (3, 31),   # March 1-31 (next year)
        }
    }

    def __init__(self):
        """Initialize the financial reporting calendar."""
        logger.info("Initializing Financial Reporting Calendar")

    def get_reporting_deadline(self,
                              industry: IndustryType,
                              period: ReportingPeriod,
                              fiscal_year: int) -> Optional[ReportingDeadline]:
        """
        Get reporting deadline for specific industry and period.

        Args:
            industry: Industry type
            period: Reporting period
            fiscal_year: Fiscal year

        Returns:
            ReportingDeadline if applicable, None otherwise
        """
        deadlines = self.REPORTING_DEADLINES.get(industry, {})
        if period not in deadlines:
            logger.warning(f"No reporting deadline for {industry.value} {period.value}")
            return None

        start_month, end_day = deadlines[period]

        # Determine the calendar year for the deadline
        if period == ReportingPeriod.Q4:
            # Q4 deadlines are in the following calendar year
            deadline_year = fiscal_year + 1
        else:
            deadline_year = fiscal_year

        # Calculate earliest and latest dates
        earliest_date = date(deadline_year, start_month, 1)

        # Handle end of month properly
        if start_month == end_day:  # Same month
            _, last_day = monthrange(deadline_year, start_month)
            latest_date = date(deadline_year, start_month, min(end_day, last_day))
        else:
            latest_date = date(deadline_year, start_month, end_day)

        description = f"{industry.value.title()} {period.value} FY{fiscal_year}"

        return ReportingDeadline(
            period=period,
            industry=industry,
            earliest_date=earliest_date,
            latest_date=latest_date,
            description=description
        )

    def get_all_deadlines_for_year(self,
                                   fiscal_year: int,
                                   industry: Optional[IndustryType] = None) -> List[ReportingDeadline]:
        """
        Get all reporting deadlines for a fiscal year.

        Args:
            fiscal_year: Fiscal year
            industry: Optional industry filter

        Returns:
            List of reporting deadlines
        """
        deadlines = []
        industries = [industry] if industry else list(IndustryType)

        for ind in industries:
            for period in ReportingPeriod:
                deadline = self.get_reporting_deadline(ind, period, fiscal_year)
                if deadline:
                    deadlines.append(deadline)

        # Sort by earliest date
        deadlines.sort(key=lambda x: x.earliest_date)
        return deadlines

    def get_next_deadline(self,
                         industry: IndustryType,
                         from_date: Optional[date] = None) -> Optional[ReportingDeadline]:
        """
        Get the next upcoming deadline for an industry.

        Args:
            industry: Industry type
            from_date: Reference date (defaults to today)

        Returns:
            Next deadline or None if not found
        """
        if from_date is None:
            from_date = date.today()

        # Check current and next fiscal year
        current_fiscal_year = from_date.year
        years_to_check = [current_fiscal_year, current_fiscal_year + 1]

        upcoming_deadlines = []
        for year in years_to_check:
            deadlines = self.get_all_deadlines_for_year(year, industry)
            for deadline in deadlines:
                if deadline.latest_date >= from_date:
                    upcoming_deadlines.append(deadline)

        if not upcoming_deadlines:
            return None

        # Return the earliest upcoming deadline
        upcoming_deadlines.sort(key=lambda x: x.earliest_date)
        return upcoming_deadlines[0]

    def create_schedule(self,
                       company_ticker: str,
                       industry_type: IndustryType,
                       reporting_period: ReportingPeriod,
                       fiscal_year: int,
                       priority: int = 1) -> Optional[FinancialStatementSchedule]:
        """
        Create a financial statement schedule for a company.

        Args:
            company_ticker: Company ticker symbol
            industry_type: Industry classification
            reporting_period: Reporting period
            fiscal_year: Fiscal year
            priority: Download priority (1=highest, 5=lowest)

        Returns:
            FinancialStatementSchedule or None if not applicable
        """
        deadline = self.get_reporting_deadline(industry_type, reporting_period, fiscal_year)
        if not deadline:
            return None

        # Estimate expected release date (usually mid-range, but can be customized)
        date_range = (deadline.latest_date - deadline.earliest_date).days
        expected_offset = min(7, date_range // 2)  # Usually within first week or mid-range
        expected_date = deadline.earliest_date + timedelta(days=expected_offset)

        return FinancialStatementSchedule(
            company_ticker=company_ticker,
            industry_type=industry_type,
            reporting_period=reporting_period,
            fiscal_year=fiscal_year,
            expected_date=expected_date,
            earliest_possible=deadline.earliest_date,
            latest_possible=deadline.latest_date,
            priority=priority,
            metadata={
                "deadline_description": deadline.description,
                "created_at": datetime.now().isoformat()
            }
        )

    def get_reporting_periods_for_industry(self, industry: IndustryType) -> List[ReportingPeriod]:
        """
        Get valid reporting periods for an industry.

        Args:
            industry: Industry type

        Returns:
            List of valid reporting periods
        """
        return list(self.REPORTING_DEADLINES.get(industry, {}).keys())

    def is_reporting_required(self,
                             industry: IndustryType,
                             period: ReportingPeriod) -> bool:
        """
        Check if reporting is required for industry and period.

        Args:
            industry: Industry type
            period: Reporting period

        Returns:
            True if reporting is required
        """
        return period in self.REPORTING_DEADLINES.get(industry, {})

    def get_industry_from_ticker(self, ticker: str) -> IndustryType:
        """
        Determine industry type from ticker symbol.

        This is a simplified implementation. In practice, this would
        lookup industry classification from a database or API.

        Args:
            ticker: Company ticker symbol

        Returns:
            Industry type (defaults to GENERAL)
        """
        ticker_upper = ticker.upper()

        # Simple heuristics for industry classification
        # In practice, this should use actual industry classification data
        if ticker_upper.endswith('-KY'):
            return IndustryType.KY_STOCK

        # Financial institutions (simplified examples)
        financial_prefixes = ['2880', '2881', '2882', '2883', '2884', '2885', '2886', '2887', '2888', '2889', '2890', '2891', '2892']
        if any(ticker_upper.startswith(prefix) for prefix in financial_prefixes):
            return IndustryType.FINANCIAL

        # Insurance companies (simplified examples)
        insurance_prefixes = ['2816', '2823', '2832', '2833', '2834', '2836', '2845', '2867', '2849']
        if any(ticker_upper.startswith(prefix) for prefix in insurance_prefixes):
            return IndustryType.INSURANCE

        # Default to general
        return IndustryType.GENERAL