"""
Holiday and Trading Day Adjustment Engine.

Provides sophisticated date adjustment logic for financial operations
considering market holidays and trading schedules.
"""

import logging
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Callable, Dict

from .market_calendar import MarketCalendar, TaiwanMarketCalendar

logger = logging.getLogger(__name__)


class AdjustmentRule(Enum):
    """Rules for adjusting dates when they fall on non-trading days."""
    FOLLOWING = "following"                    # Next trading day
    PRECEDING = "preceding"                    # Previous trading day
    MODIFIED_FOLLOWING = "modified_following"  # Following, but not next month
    MODIFIED_PRECEDING = "modified_preceding"  # Preceding, but not previous month
    NEAREST = "nearest"                        # Closest trading day
    NO_ADJUSTMENT = "no_adjustment"            # Keep original date


@dataclass
class AdjustmentResult:
    """Result of a date adjustment operation."""
    original_date: date
    adjusted_date: date
    rule_applied: AdjustmentRule
    days_adjusted: int
    was_holiday: bool
    was_weekend: bool
    notes: str = ""

    @property
    def was_adjusted(self) -> bool:
        """Check if the date was actually adjusted."""
        return self.original_date != self.adjusted_date

    @property
    def adjustment_direction(self) -> str:
        """Get the direction of adjustment."""
        if not self.was_adjusted:
            return "none"
        elif self.days_adjusted > 0:
            return "forward"
        else:
            return "backward"


class BusinessDayAdjuster:
    """Utility for adjusting dates to business/trading days."""

    def __init__(self, market_calendar: Optional[MarketCalendar] = None):
        """
        Initialize business day adjuster.

        Args:
            market_calendar: Market calendar to use (defaults to Taiwan)
        """
        self.market_calendar = market_calendar or TaiwanMarketCalendar()
        logger.info(f"Initialized BusinessDayAdjuster with {self.market_calendar.name}")

    def adjust_date(self,
                   target_date: Union[date, datetime],
                   rule: AdjustmentRule = AdjustmentRule.FOLLOWING) -> AdjustmentResult:
        """
        Adjust a date according to the specified rule.

        Args:
            target_date: Date to adjust
            rule: Adjustment rule to apply

        Returns:
            AdjustmentResult with adjustment details
        """
        if isinstance(target_date, datetime):
            target_date = target_date.date()

        original_date = target_date
        was_holiday = self.market_calendar.is_holiday(target_date)
        was_weekend = target_date.weekday() >= 5

        # If it's already a trading day and no adjustment needed
        if rule == AdjustmentRule.NO_ADJUSTMENT or self.market_calendar.is_trading_day(target_date):
            return AdjustmentResult(
                original_date=original_date,
                adjusted_date=target_date,
                rule_applied=rule,
                days_adjusted=0,
                was_holiday=was_holiday,
                was_weekend=was_weekend,
                notes="No adjustment needed" if self.market_calendar.is_trading_day(target_date) else "No adjustment rule applied"
            )

        # Apply adjustment rule
        if rule == AdjustmentRule.FOLLOWING:
            adjusted_date = self._adjust_following(target_date)
        elif rule == AdjustmentRule.PRECEDING:
            adjusted_date = self._adjust_preceding(target_date)
        elif rule == AdjustmentRule.MODIFIED_FOLLOWING:
            adjusted_date = self._adjust_modified_following(target_date)
        elif rule == AdjustmentRule.MODIFIED_PRECEDING:
            adjusted_date = self._adjust_modified_preceding(target_date)
        elif rule == AdjustmentRule.NEAREST:
            adjusted_date = self._adjust_nearest(target_date)
        else:
            adjusted_date = target_date

        days_adjusted = (adjusted_date - original_date).days
        return AdjustmentResult(
            original_date=original_date,
            adjusted_date=adjusted_date,
            rule_applied=rule,
            days_adjusted=days_adjusted,
            was_holiday=was_holiday,
            was_weekend=was_weekend,
            notes=f"Adjusted by {abs(days_adjusted)} day(s) {rule.value}"
        )

    def _adjust_following(self, target_date: date) -> date:
        """Adjust to the next trading day."""
        return self.market_calendar.next_trading_day(target_date - timedelta(days=1))

    def _adjust_preceding(self, target_date: date) -> date:
        """Adjust to the previous trading day."""
        return self.market_calendar.previous_trading_day(target_date + timedelta(days=1))

    def _adjust_modified_following(self, target_date: date) -> date:
        """Adjust following unless it goes to next month."""
        following = self._adjust_following(target_date)
        if following.month != target_date.month:
            return self._adjust_preceding(target_date)
        return following

    def _adjust_modified_preceding(self, target_date: date) -> date:
        """Adjust preceding unless it goes to previous month."""
        preceding = self._adjust_preceding(target_date)
        if preceding.month != target_date.month:
            return self._adjust_following(target_date)
        return preceding

    def _adjust_nearest(self, target_date: date) -> date:
        """Adjust to the nearest trading day."""
        if self.market_calendar.is_trading_day(target_date):
            return target_date

        following = self._adjust_following(target_date)
        preceding = self._adjust_preceding(target_date)

        days_to_following = (following - target_date).days
        days_to_preceding = (target_date - preceding).days

        if days_to_following <= days_to_preceding:
            return following
        else:
            return preceding

    def adjust_date_range(self,
                         start_date: Union[date, datetime],
                         end_date: Union[date, datetime],
                         rule: AdjustmentRule = AdjustmentRule.FOLLOWING) -> tuple[AdjustmentResult, AdjustmentResult]:
        """
        Adjust both start and end dates of a range.

        Args:
            start_date: Start date
            end_date: End date
            rule: Adjustment rule

        Returns:
            Tuple of (start_adjustment, end_adjustment)
        """
        start_result = self.adjust_date(start_date, rule)
        end_result = self.adjust_date(end_date, rule)
        return start_result, end_result


class HolidayAdjustmentEngine:
    """
    Advanced holiday adjustment engine for financial statement scheduling.

    Provides sophisticated date adjustment capabilities specifically designed
    for financial reporting deadlines and download scheduling.
    """

    def __init__(self, market_calendar: Optional[MarketCalendar] = None):
        """
        Initialize holiday adjustment engine.

        Args:
            market_calendar: Market calendar to use
        """
        self.market_calendar = market_calendar or TaiwanMarketCalendar()
        self.adjuster = BusinessDayAdjuster(self.market_calendar)
        self._adjustment_strategies: Dict[str, Callable] = {
            "conservative": self._conservative_adjustment,
            "aggressive": self._aggressive_adjustment,
            "smart": self._smart_adjustment,
        }
        logger.info("Initialized Holiday Adjustment Engine")

    def adjust_reporting_deadline(self,
                                 deadline_date: Union[date, datetime],
                                 strategy: str = "smart") -> AdjustmentResult:
        """
        Adjust a financial reporting deadline considering market conditions.

        Args:
            deadline_date: Original deadline date
            strategy: Adjustment strategy ("conservative", "aggressive", "smart")

        Returns:
            AdjustmentResult with adjusted deadline
        """
        if isinstance(deadline_date, datetime):
            deadline_date = deadline_date.date()

        if strategy not in self._adjustment_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self._adjustment_strategies.keys())}")

        return self._adjustment_strategies[strategy](deadline_date)

    def _conservative_adjustment(self, deadline_date: date) -> AdjustmentResult:
        """
        Conservative adjustment: always adjust backwards to avoid missing deadlines.

        Args:
            deadline_date: Deadline date

        Returns:
            AdjustmentResult
        """
        if self.market_calendar.is_trading_day(deadline_date):
            return AdjustmentResult(
                original_date=deadline_date,
                adjusted_date=deadline_date,
                rule_applied=AdjustmentRule.NO_ADJUSTMENT,
                days_adjusted=0,
                was_holiday=False,
                was_weekend=False,
                notes="Conservative: No adjustment needed"
            )

        # Adjust to previous trading day to be safe
        result = self.adjuster.adjust_date(deadline_date, AdjustmentRule.PRECEDING)
        result.notes = f"Conservative: {result.notes}"
        return result

    def _aggressive_adjustment(self, deadline_date: date) -> AdjustmentResult:
        """
        Aggressive adjustment: adjust forward to maximize time available.

        Args:
            deadline_date: Deadline date

        Returns:
            AdjustmentResult
        """
        if self.market_calendar.is_trading_day(deadline_date):
            return AdjustmentResult(
                original_date=deadline_date,
                adjusted_date=deadline_date,
                rule_applied=AdjustmentRule.NO_ADJUSTMENT,
                days_adjusted=0,
                was_holiday=False,
                was_weekend=False,
                notes="Aggressive: No adjustment needed"
            )

        # Adjust to next trading day to maximize available time
        result = self.adjuster.adjust_date(deadline_date, AdjustmentRule.FOLLOWING)
        result.notes = f"Aggressive: {result.notes}"
        return result

    def _smart_adjustment(self, deadline_date: date) -> AdjustmentResult:
        """
        Smart adjustment: context-aware adjustment based on date and situation.

        Args:
            deadline_date: Deadline date

        Returns:
            AdjustmentResult
        """
        if self.market_calendar.is_trading_day(deadline_date):
            return AdjustmentResult(
                original_date=deadline_date,
                adjusted_date=deadline_date,
                rule_applied=AdjustmentRule.NO_ADJUSTMENT,
                days_adjusted=0,
                was_holiday=False,
                was_weekend=False,
                notes="Smart: No adjustment needed"
            )

        # Smart logic: consider context
        if deadline_date.weekday() >= 5:  # Weekend
            # If it's weekend, typically adjust forward (companies often publish after weekend)
            result = self.adjuster.adjust_date(deadline_date, AdjustmentRule.MODIFIED_FOLLOWING)
            result.notes = f"Smart (weekend): {result.notes}"
        else:  # Holiday on weekday
            # For holidays, use nearest trading day
            result = self.adjuster.adjust_date(deadline_date, AdjustmentRule.NEAREST)
            result.notes = f"Smart (holiday): {result.notes}"

        return result

    def calculate_download_window(self,
                                 earliest_date: Union[date, datetime],
                                 latest_date: Union[date, datetime],
                                 strategy: str = "smart") -> tuple[AdjustmentResult, AdjustmentResult]:
        """
        Calculate an adjusted download window for financial statements.

        Args:
            earliest_date: Earliest possible release date
            latest_date: Latest possible release date (deadline)
            strategy: Adjustment strategy

        Returns:
            Tuple of (adjusted_earliest, adjusted_latest)
        """
        # Adjust earliest date - usually forward to ensure we don't start too early
        earliest_result = self.adjuster.adjust_date(earliest_date, AdjustmentRule.FOLLOWING)
        earliest_result.notes = f"Download window start: {earliest_result.notes}"

        # Adjust latest date using specified strategy
        latest_result = self.adjust_reporting_deadline(latest_date, strategy)
        latest_result.notes = f"Download window end: {latest_result.notes}"

        return earliest_result, latest_result

    def get_trading_days_in_window(self,
                                  start_date: Union[date, datetime],
                                  end_date: Union[date, datetime]) -> List[date]:
        """
        Get all trading days within a window.

        Args:
            start_date: Window start date
            end_date: Window end date

        Returns:
            List of trading days in the window
        """
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        trading_days = []
        current_date = start_date

        while current_date <= end_date:
            if self.market_calendar.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        return trading_days

    def optimize_schedule_dates(self,
                               target_dates: List[Union[date, datetime]],
                               strategy: str = "smart") -> List[AdjustmentResult]:
        """
        Optimize a list of schedule dates using the adjustment engine.

        Args:
            target_dates: List of target dates
            strategy: Adjustment strategy

        Returns:
            List of adjustment results
        """
        results = []
        for target_date in target_dates:
            result = self.adjust_reporting_deadline(target_date, strategy)
            results.append(result)

        return results