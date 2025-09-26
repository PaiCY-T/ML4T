"""
Timestamp management and comparison algorithms for FinLab API responses.

This module provides sophisticated timestamp comparison capabilities to determine
when data has been updated and needs to be re-downloaded.
"""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TimestampComparison(Enum):
    """Result of timestamp comparison."""
    IDENTICAL = "identical"           # Timestamps are identical
    NEWER = "newer"                  # Remote data is newer
    OLDER = "older"                  # Local data is newer (unusual)
    DIFFERENT = "different"          # Timestamps differ but unclear which is newer
    MISSING_LOCAL = "missing_local"  # No local timestamp
    MISSING_REMOTE = "missing_remote" # No remote timestamp
    INVALID = "invalid"              # Invalid timestamp format


@dataclass
class TimestampInfo:
    """Information about a timestamp."""
    value: Optional[datetime]
    source: str
    confidence: float
    timezone_aware: bool
    format_detected: str


class TimestampManager:
    """
    Advanced timestamp management for incremental downloads.

    Features:
    - Multiple timestamp format support
    - Timezone awareness and conversion
    - Confidence scoring for timestamp reliability
    - Intelligent comparison algorithms
    - Support for different data freshness policies
    """

    def __init__(self, default_timezone: str = 'Asia/Taipei'):
        """
        Initialize timestamp manager.

        Args:
            default_timezone: Default timezone for data (Taiwan market default)
        """
        self.default_timezone = timezone.utc
        try:
            import zoneinfo
            self.default_timezone = zoneinfo.ZoneInfo(default_timezone)
        except ImportError:
            # Fallback to UTC if zoneinfo not available
            logger.warning("zoneinfo not available, using UTC as default timezone")

        # Common timestamp formats for FinLab data
        self.timestamp_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%d',
            '%Y%m%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y'
        ]

    def parse_timestamp(self, timestamp_value: Any, source: str = "unknown") -> TimestampInfo:
        """
        Parse timestamp from various formats.

        Args:
            timestamp_value: Timestamp in various formats
            source: Source of the timestamp for tracking

        Returns:
            TimestampInfo with parsed timestamp and metadata
        """
        if timestamp_value is None:
            return TimestampInfo(
                value=None,
                source=source,
                confidence=0.0,
                timezone_aware=False,
                format_detected="none"
            )

        # Handle different input types
        if isinstance(timestamp_value, datetime):
            return TimestampInfo(
                value=self._ensure_timezone_aware(timestamp_value),
                source=source,
                confidence=1.0,
                timezone_aware=timestamp_value.tzinfo is not None,
                format_detected="datetime_object"
            )

        if isinstance(timestamp_value, date):
            dt = datetime.combine(timestamp_value, datetime.min.time())
            return TimestampInfo(
                value=self._ensure_timezone_aware(dt),
                source=source,
                confidence=0.9,
                timezone_aware=False,
                format_detected="date_object"
            )

        # Handle string timestamps
        if isinstance(timestamp_value, str):
            return self._parse_string_timestamp(timestamp_value, source)

        # Handle numeric timestamps (Unix epoch)
        if isinstance(timestamp_value, (int, float)):
            try:
                dt = datetime.fromtimestamp(timestamp_value, tz=self.default_timezone)
                return TimestampInfo(
                    value=dt,
                    source=source,
                    confidence=0.8,
                    timezone_aware=True,
                    format_detected="unix_timestamp"
                )
            except (ValueError, OSError):
                pass

        # Unable to parse
        return TimestampInfo(
            value=None,
            source=source,
            confidence=0.0,
            timezone_aware=False,
            format_detected="unparseable"
        )

    def _parse_string_timestamp(self, timestamp_str: str, source: str) -> TimestampInfo:
        """Parse string timestamp using various formats."""
        timestamp_str = timestamp_str.strip()

        for fmt in self.timestamp_formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)

                # Check if timezone aware
                timezone_aware = dt.tzinfo is not None
                if not timezone_aware:
                    dt = self._ensure_timezone_aware(dt)

                return TimestampInfo(
                    value=dt,
                    source=source,
                    confidence=0.95,
                    timezone_aware=timezone_aware,
                    format_detected=fmt
                )
            except ValueError:
                continue

        # Try pandas if available for more flexible parsing
        try:
            import pandas as pd
            dt = pd.to_datetime(timestamp_str)
            if pd.isna(dt):
                raise ValueError("Invalid timestamp")

            dt = dt.to_pydatetime()
            return TimestampInfo(
                value=self._ensure_timezone_aware(dt),
                source=source,
                confidence=0.85,
                timezone_aware=dt.tzinfo is not None,
                format_detected="pandas_parser"
            )
        except (ImportError, ValueError):
            pass

        # Failed to parse
        return TimestampInfo(
            value=None,
            source=source,
            confidence=0.0,
            timezone_aware=False,
            format_detected="failed"
        )

    def _ensure_timezone_aware(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone aware."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.default_timezone)
        return dt

    def compare_timestamps(self,
                          local_timestamp: Any,
                          remote_timestamp: Any,
                          tolerance_seconds: int = 0) -> TimestampComparison:
        """
        Compare local and remote timestamps.

        Args:
            local_timestamp: Local timestamp (any format)
            remote_timestamp: Remote timestamp (any format)
            tolerance_seconds: Tolerance in seconds for considering timestamps equal

        Returns:
            TimestampComparison result
        """
        local_info = self.parse_timestamp(local_timestamp, "local")
        remote_info = self.parse_timestamp(remote_timestamp, "remote")

        # Handle missing timestamps
        if local_info.value is None and remote_info.value is None:
            return TimestampComparison.IDENTICAL

        if local_info.value is None:
            return TimestampComparison.MISSING_LOCAL

        if remote_info.value is None:
            return TimestampComparison.MISSING_REMOTE

        # Handle invalid timestamps
        if local_info.confidence == 0.0 or remote_info.confidence == 0.0:
            return TimestampComparison.INVALID

        # Compare timestamps
        try:
            time_diff = (remote_info.value - local_info.value).total_seconds()

            if abs(time_diff) <= tolerance_seconds:
                return TimestampComparison.IDENTICAL
            elif time_diff > 0:
                return TimestampComparison.NEWER
            else:
                return TimestampComparison.OLDER

        except Exception as e:
            logger.error(f"Error comparing timestamps: {e}")
            return TimestampComparison.INVALID

    def needs_update(self,
                    local_timestamp: Any,
                    remote_timestamp: Any,
                    force_update: bool = False,
                    max_age_hours: Optional[int] = None) -> Tuple[bool, str]:
        """
        Determine if data needs to be updated based on timestamps.

        Args:
            local_timestamp: Local timestamp
            remote_timestamp: Remote timestamp
            force_update: Force update regardless of timestamps
            max_age_hours: Maximum age in hours before forcing update

        Returns:
            Tuple of (needs_update, reason)
        """
        if force_update:
            return True, "Forced update requested"

        comparison = self.compare_timestamps(local_timestamp, remote_timestamp)

        if comparison == TimestampComparison.MISSING_LOCAL:
            return True, "No local data found"

        if comparison == TimestampComparison.MISSING_REMOTE:
            return False, "No remote timestamp available"

        if comparison == TimestampComparison.INVALID:
            return True, "Invalid timestamp comparison - updating to be safe"

        if comparison == TimestampComparison.NEWER:
            return True, "Remote data is newer"

        if comparison == TimestampComparison.IDENTICAL:
            # Check max age if specified
            if max_age_hours is not None:
                local_info = self.parse_timestamp(local_timestamp, "local")
                if local_info.value:
                    age_hours = (datetime.utcnow() - local_info.value.replace(tzinfo=timezone.utc)).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        return True, f"Data is older than {max_age_hours} hours"

            return False, "Local data is up to date"

        if comparison == TimestampComparison.OLDER:
            return False, "Local data is newer than remote"

        return True, "Unknown comparison result - updating to be safe"

    def extract_timestamps_from_data(self, data: Any) -> Dict[str, TimestampInfo]:
        """
        Extract timestamps from FinLab data structures.

        Args:
            data: Data structure (DataFrame, dict, etc.)

        Returns:
            Dictionary of extracted timestamps
        """
        timestamps = {}

        if hasattr(data, 'index') and hasattr(data.index, 'max'):
            # Pandas DataFrame/Series with datetime index
            try:
                max_timestamp = data.index.max()
                if max_timestamp:
                    timestamps['data_max_date'] = self.parse_timestamp(max_timestamp, "data_index")

                min_timestamp = data.index.min()
                if min_timestamp:
                    timestamps['data_min_date'] = self.parse_timestamp(min_timestamp, "data_index")
            except Exception:
                pass

        if hasattr(data, 'columns'):
            # Look for timestamp columns
            timestamp_columns = [col for col in data.columns
                               if any(keyword in str(col).lower()
                                     for keyword in ['date', 'time', 'timestamp', 'modified', 'updated'])]

            for col in timestamp_columns:
                try:
                    max_val = data[col].max()
                    if max_val:
                        timestamps[f'column_{col}_max'] = self.parse_timestamp(max_val, f"column_{col}")
                except Exception:
                    pass

        if isinstance(data, dict):
            # Extract timestamps from dictionary
            for key, value in data.items():
                if any(keyword in str(key).lower()
                      for keyword in ['date', 'time', 'timestamp', 'modified', 'updated']):
                    timestamps[key] = self.parse_timestamp(value, f"dict_{key}")

        return timestamps

    def get_data_freshness_score(self, timestamps: Dict[str, TimestampInfo]) -> float:
        """
        Calculate a freshness score for data based on timestamps.

        Args:
            timestamps: Dictionary of timestamps

        Returns:
            Freshness score between 0.0 (stale) and 1.0 (very fresh)
        """
        if not timestamps:
            return 0.0

        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        freshness_scores = []

        for timestamp_info in timestamps.values():
            if timestamp_info.value is None:
                continue

            try:
                age_hours = (now - timestamp_info.value).total_seconds() / 3600

                # Calculate freshness based on age
                if age_hours <= 1:
                    score = 1.0
                elif age_hours <= 24:
                    score = 0.9 - (age_hours - 1) * 0.3 / 23  # 0.9 to 0.6
                elif age_hours <= 168:  # 1 week
                    score = 0.6 - (age_hours - 24) * 0.4 / 144  # 0.6 to 0.2
                else:
                    score = max(0.0, 0.2 - (age_hours - 168) * 0.2 / 168)  # 0.2 to 0.0

                # Weight by confidence
                weighted_score = score * timestamp_info.confidence
                freshness_scores.append(weighted_score)

            except Exception:
                continue

        if not freshness_scores:
            return 0.0

        # Return weighted average
        return sum(freshness_scores) / len(freshness_scores)

    def suggest_update_frequency(self,
                                data_type: str,
                                historical_update_pattern: Optional[List[datetime]] = None) -> timedelta:
        """
        Suggest optimal update frequency based on data type and patterns.

        Args:
            data_type: Type of data (price, fundamental, etc.)
            historical_update_pattern: Historical update timestamps

        Returns:
            Suggested update frequency as timedelta
        """
        # Default frequencies by data type
        default_frequencies = {
            'price': timedelta(hours=1),
            'volume': timedelta(hours=1),
            'fundamental': timedelta(days=1),
            'financial_statement': timedelta(weeks=1),
            'dividend': timedelta(weeks=1),
            'split': timedelta(weeks=1),
            'earnings': timedelta(days=1),
            'news': timedelta(hours=6),
            'technical': timedelta(hours=4)
        }

        base_frequency = default_frequencies.get(data_type.lower(), timedelta(hours=12))

        # Analyze historical pattern if available
        if historical_update_pattern and len(historical_update_pattern) > 2:
            try:
                # Calculate average time between updates
                intervals = []
                for i in range(1, len(historical_update_pattern)):
                    interval = historical_update_pattern[i] - historical_update_pattern[i-1]
                    intervals.append(interval.total_seconds())

                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    pattern_frequency = timedelta(seconds=avg_interval)

                    # Use the shorter of base frequency or pattern frequency
                    # This ensures we don't miss updates but don't over-fetch
                    return min(base_frequency, pattern_frequency)

            except Exception as e:
                logger.warning(f"Error analyzing update pattern: {e}")

        return base_frequency

    def format_timestamp_for_api(self, timestamp: datetime, format_type: str = 'iso') -> str:
        """
        Format timestamp for API requests.

        Args:
            timestamp: Timestamp to format
            format_type: Format type ('iso', 'date', 'epoch')

        Returns:
            Formatted timestamp string
        """
        if format_type == 'iso':
            return timestamp.isoformat()
        elif format_type == 'date':
            return timestamp.strftime('%Y-%m-%d')
        elif format_type == 'epoch':
            return str(int(timestamp.timestamp()))
        else:
            return timestamp.strftime('%Y-%m-%d %H:%M:%S')

    def get_market_hours_adjusted_timestamp(self, timestamp: datetime) -> datetime:
        """
        Adjust timestamp to Taiwan market hours context.

        Args:
            timestamp: Original timestamp

        Returns:
            Market-hours adjusted timestamp
        """
        # Taiwan stock market hours: 9:00 AM - 1:30 PM (13:30)
        market_open = timestamp.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close = timestamp.replace(hour=13, minute=30, second=0, microsecond=0)

        # If before market open, use previous day's close
        if timestamp.time() < market_open.time():
            prev_day = timestamp - timedelta(days=1)
            return prev_day.replace(hour=13, minute=30, second=0, microsecond=0)

        # If after market close, use today's close
        if timestamp.time() > market_close.time():
            return market_close

        # During market hours, use actual timestamp
        return timestamp