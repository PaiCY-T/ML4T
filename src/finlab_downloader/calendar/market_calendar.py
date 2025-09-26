"""
Market Calendar for Taiwan Stock Exchange.

Provides trading day calculations and holiday awareness for financial markets.
"""

import logging
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Dict, Union
import pandas as pd

try:
    # Try to use pandas_market_calendars if available
    import pandas_market_calendars as mcal
    MARKET_CALENDARS_AVAILABLE = True
except ImportError:
    MARKET_CALENDARS_AVAILABLE = False
    logging.warning("pandas_market_calendars not available, using built-in calendar")

logger = logging.getLogger(__name__)


class MarketHoliday(Enum):
    """Types of market holidays."""
    NEW_YEAR = "new_year"
    SPRING_FESTIVAL = "spring_festival"
    PEACE_MEMORIAL_DAY = "peace_memorial_day"
    TOMB_SWEEPING_DAY = "tomb_sweeping_day"
    LABOR_DAY = "labor_day"
    DRAGON_BOAT_FESTIVAL = "dragon_boat_festival"
    MID_AUTUMN_FESTIVAL = "mid_autumn_festival"
    NATIONAL_DAY = "national_day"
    WEEKEND = "weekend"
    SPECIAL_HOLIDAY = "special_holiday"


@dataclass
class HolidayInfo:
    """Information about a market holiday."""
    date: date
    holiday_type: MarketHoliday
    name_en: str
    name_zh: str
    is_trading_day: bool = False
    notes: str = ""


class MarketCalendar:
    """Base class for market calendars."""

    def __init__(self, name: str):
        """
        Initialize market calendar.

        Args:
            name: Calendar name
        """
        self.name = name
        self._holidays_cache: Dict[int, Set[date]] = {}

    def is_trading_day(self, check_date: Union[date, datetime]) -> bool:
        """
        Check if a date is a trading day.

        Args:
            check_date: Date to check

        Returns:
            True if it's a trading day
        """
        if isinstance(check_date, datetime):
            check_date = check_date.date()

        # Weekend check
        if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Holiday check
        return not self.is_holiday(check_date)

    def is_holiday(self, check_date: Union[date, datetime]) -> bool:
        """
        Check if a date is a market holiday.

        Args:
            check_date: Date to check

        Returns:
            True if it's a holiday
        """
        if isinstance(check_date, datetime):
            check_date = check_date.date()

        holidays = self.get_holidays(check_date.year)
        return check_date in holidays

    def get_holidays(self, year: int) -> Set[date]:
        """
        Get all holidays for a year.

        Args:
            year: Year to get holidays for

        Returns:
            Set of holiday dates
        """
        if year not in self._holidays_cache:
            self._holidays_cache[year] = self._calculate_holidays(year)
        return self._holidays_cache[year]

    def _calculate_holidays(self, year: int) -> Set[date]:
        """
        Calculate holidays for a year.

        Must be implemented by subclasses.

        Args:
            year: Year to calculate holidays for

        Returns:
            Set of holiday dates
        """
        raise NotImplementedError("Subclasses must implement _calculate_holidays")

    def next_trading_day(self, from_date: Union[date, datetime]) -> date:
        """
        Get the next trading day after a given date.

        Args:
            from_date: Starting date

        Returns:
            Next trading day
        """
        if isinstance(from_date, datetime):
            from_date = from_date.date()

        current = from_date + timedelta(days=1)
        while not self.is_trading_day(current):
            current += timedelta(days=1)
        return current

    def previous_trading_day(self, from_date: Union[date, datetime]) -> date:
        """
        Get the previous trading day before a given date.

        Args:
            from_date: Starting date

        Returns:
            Previous trading day
        """
        if isinstance(from_date, datetime):
            from_date = from_date.date()

        current = from_date - timedelta(days=1)
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
        return current

    def trading_days_between(self,
                           start_date: Union[date, datetime],
                           end_date: Union[date, datetime],
                           inclusive: bool = True) -> int:
        """
        Count trading days between two dates.

        Args:
            start_date: Start date
            end_date: End date
            inclusive: Include end date in count

        Returns:
            Number of trading days
        """
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        if start_date > end_date:
            return 0

        count = 0
        current = start_date
        while current <= end_date:
            if self.is_trading_day(current):
                if current != end_date or inclusive:
                    count += 1
            current += timedelta(days=1)

        return count

    def adjust_to_trading_day(self,
                            target_date: Union[date, datetime],
                            direction: str = "following") -> date:
        """
        Adjust a date to the nearest trading day.

        Args:
            target_date: Target date
            direction: Adjustment direction ("following", "preceding", "modified_following")

        Returns:
            Adjusted trading day
        """
        if isinstance(target_date, datetime):
            target_date = target_date.date()

        if self.is_trading_day(target_date):
            return target_date

        if direction == "following":
            return self.next_trading_day(target_date)
        elif direction == "preceding":
            return self.previous_trading_day(target_date)
        elif direction == "modified_following":
            # Use following unless it goes to next month
            next_day = self.next_trading_day(target_date)
            if next_day.month != target_date.month:
                return self.previous_trading_day(target_date)
            return next_day
        else:
            raise ValueError(f"Invalid direction: {direction}")


class TaiwanMarketCalendar(MarketCalendar):
    """
    Taiwan Stock Exchange (TWSE) market calendar.

    Implements Taiwan-specific holidays and trading rules.
    """

    # Fixed holidays (month, day)
    FIXED_HOLIDAYS = {
        (1, 1): ("New Year's Day", "元旦"),
        (2, 28): ("Peace Memorial Day", "和平紀念日"),
        (5, 1): ("Labor Day", "勞動節"),
        (10, 10): ("National Day", "國慶日"),
    }

    def __init__(self):
        """Initialize Taiwan market calendar."""
        super().__init__("Taiwan Stock Exchange")
        logger.info("Initializing Taiwan Market Calendar")

    def _calculate_holidays(self, year: int) -> Set[date]:
        """
        Calculate Taiwan market holidays for a year.

        Args:
            year: Year to calculate holidays for

        Returns:
            Set of holiday dates
        """
        holidays = set()

        # Add fixed holidays
        for (month, day), (name_en, name_zh) in self.FIXED_HOLIDAYS.items():
            try:
                holiday_date = date(year, month, day)
                holidays.add(holiday_date)
            except ValueError:
                # Handle invalid dates (e.g., Feb 29 in non-leap year)
                pass

        # Add calculated holidays
        holidays.update(self._calculate_lunar_holidays(year))

        # Add weekend holidays (if fixed holidays fall on weekends, they might be observed on other days)
        holidays.update(self._calculate_weekend_adjustments(year, holidays))

        return holidays

    def _calculate_lunar_holidays(self, year: int) -> Set[date]:
        """
        Calculate lunar calendar based holidays.

        This is a simplified implementation. In practice, you would
        use a proper lunar calendar library or API.

        Args:
            year: Year to calculate for

        Returns:
            Set of lunar holiday dates
        """
        holidays = set()

        # Spring Festival (Chinese New Year) - approximate dates
        # This would need proper lunar calendar calculation
        spring_festival_dates = {
            2023: [(1, 22), (1, 23), (1, 24), (1, 25), (1, 26), (1, 27)],
            2024: [(2, 10), (2, 11), (2, 12), (2, 13), (2, 14)],
            2025: [(1, 29), (1, 30), (1, 31), (2, 1), (2, 2)],
            2026: [(2, 17), (2, 18), (2, 19), (2, 20), (2, 21)],
        }

        if year in spring_festival_dates:
            for month, day in spring_festival_dates[year]:
                try:
                    holidays.add(date(year, month, day))
                except ValueError:
                    pass

        # Tomb Sweeping Day - usually around April 4-6
        tomb_sweeping_dates = {
            2023: (4, 5),
            2024: (4, 4),
            2025: (4, 5),
            2026: (4, 5),
        }

        if year in tomb_sweeping_dates:
            month, day = tomb_sweeping_dates[year]
            try:
                holidays.add(date(year, month, day))
            except ValueError:
                pass

        # Dragon Boat Festival - approximate dates
        dragon_boat_dates = {
            2023: (6, 22),
            2024: (6, 10),
            2025: (5, 31),
            2026: (6, 19),
        }

        if year in dragon_boat_dates:
            month, day = dragon_boat_dates[year]
            try:
                holidays.add(date(year, month, day))
            except ValueError:
                pass

        # Mid-Autumn Festival - approximate dates
        mid_autumn_dates = {
            2023: (9, 29),
            2024: (9, 17),
            2025: (10, 6),
            2026: (9, 25),
        }

        if year in mid_autumn_dates:
            month, day = mid_autumn_dates[year]
            try:
                holidays.add(date(year, month, day))
            except ValueError:
                pass

        return holidays

    def _calculate_weekend_adjustments(self, year: int, holidays: Set[date]) -> Set[date]:
        """
        Calculate weekend adjustments for holidays.

        Args:
            year: Year
            holidays: Existing holidays

        Returns:
            Additional holiday dates due to weekend adjustments
        """
        adjustments = set()

        for holiday in holidays:
            # If holiday falls on weekend, it might be observed on Friday or Monday
            if holiday.weekday() == 5:  # Saturday
                # Observed on Friday
                observed = holiday - timedelta(days=1)
                if observed.year == year:
                    adjustments.add(observed)
            elif holiday.weekday() == 6:  # Sunday
                # Observed on Monday
                observed = holiday + timedelta(days=1)
                if observed.year == year:
                    adjustments.add(observed)

        return adjustments

    def get_holiday_info(self, check_date: date) -> Optional[HolidayInfo]:
        """
        Get detailed information about a holiday.

        Args:
            check_date: Date to check

        Returns:
            HolidayInfo if it's a holiday, None otherwise
        """
        if not self.is_holiday(check_date):
            return None

        # Check fixed holidays
        month_day = (check_date.month, check_date.day)
        if month_day in self.FIXED_HOLIDAYS:
            name_en, name_zh = self.FIXED_HOLIDAYS[month_day]
            holiday_type = self._get_holiday_type_from_name(name_en)
            return HolidayInfo(
                date=check_date,
                holiday_type=holiday_type,
                name_en=name_en,
                name_zh=name_zh,
                is_trading_day=False
            )

        # Weekend
        if check_date.weekday() >= 5:
            return HolidayInfo(
                date=check_date,
                holiday_type=MarketHoliday.WEEKEND,
                name_en="Weekend",
                name_zh="週末",
                is_trading_day=False
            )

        # Other holidays (would need more detailed mapping)
        return HolidayInfo(
            date=check_date,
            holiday_type=MarketHoliday.SPECIAL_HOLIDAY,
            name_en="Market Holiday",
            name_zh="市場假日",
            is_trading_day=False
        )

    def _get_holiday_type_from_name(self, name_en: str) -> MarketHoliday:
        """Map holiday name to holiday type."""
        name_mapping = {
            "New Year's Day": MarketHoliday.NEW_YEAR,
            "Peace Memorial Day": MarketHoliday.PEACE_MEMORIAL_DAY,
            "Labor Day": MarketHoliday.LABOR_DAY,
            "National Day": MarketHoliday.NATIONAL_DAY,
        }
        return name_mapping.get(name_en, MarketHoliday.SPECIAL_HOLIDAY)