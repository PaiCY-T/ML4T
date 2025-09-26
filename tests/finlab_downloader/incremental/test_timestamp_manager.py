"""
Tests for the timestamp management system.
"""

import pytest
from datetime import datetime, date, timezone
import pandas as pd

from src.finlab_downloader.incremental.timestamp_manager import (
    TimestampManager, TimestampComparison, TimestampInfo
)


@pytest.fixture
def timestamp_manager():
    """Create a timestamp manager instance for testing."""
    return TimestampManager()


def test_parse_datetime_object(timestamp_manager):
    """Test parsing datetime objects."""
    dt = datetime(2024, 1, 15, 10, 30, 0)
    info = timestamp_manager.parse_timestamp(dt, "test")

    assert info.value is not None
    assert info.source == "test"
    assert info.confidence == 1.0
    assert info.format_detected == "datetime_object"


def test_parse_date_object(timestamp_manager):
    """Test parsing date objects."""
    d = date(2024, 1, 15)
    info = timestamp_manager.parse_timestamp(d, "test")

    assert info.value is not None
    assert info.source == "test"
    assert info.confidence == 0.9
    assert info.format_detected == "date_object"


def test_parse_string_timestamps(timestamp_manager):
    """Test parsing various string timestamp formats."""
    test_cases = [
        "2024-01-15 10:30:00",
        "2024-01-15T10:30:00",
        "2024-01-15T10:30:00.123456",
        "2024-01-15T10:30:00Z",
        "2024-01-15",
        "20240115",
        "2024/01/15"
    ]

    for timestamp_str in test_cases:
        info = timestamp_manager.parse_timestamp(timestamp_str, "test")
        assert info.value is not None, f"Failed to parse: {timestamp_str}"
        assert info.confidence > 0.8


def test_parse_unix_timestamp(timestamp_manager):
    """Test parsing Unix timestamps."""
    unix_ts = 1705312200  # January 15, 2024
    info = timestamp_manager.parse_timestamp(unix_ts, "test")

    assert info.value is not None
    assert info.confidence == 0.8
    assert info.format_detected == "unix_timestamp"


def test_parse_invalid_timestamp(timestamp_manager):
    """Test parsing invalid timestamps."""
    info = timestamp_manager.parse_timestamp("invalid", "test")

    assert info.value is None
    assert info.confidence == 0.0
    assert info.format_detected == "failed"


def test_compare_timestamps_identical(timestamp_manager):
    """Test comparing identical timestamps."""
    dt1 = datetime(2024, 1, 15, 10, 30, 0)
    dt2 = datetime(2024, 1, 15, 10, 30, 0)

    result = timestamp_manager.compare_timestamps(dt1, dt2)
    assert result == TimestampComparison.IDENTICAL


def test_compare_timestamps_newer(timestamp_manager):
    """Test comparing timestamps where remote is newer."""
    dt1 = datetime(2024, 1, 15, 10, 30, 0)  # local
    dt2 = datetime(2024, 1, 15, 11, 30, 0)  # remote (newer)

    result = timestamp_manager.compare_timestamps(dt1, dt2)
    assert result == TimestampComparison.NEWER


def test_compare_timestamps_older(timestamp_manager):
    """Test comparing timestamps where remote is older."""
    dt1 = datetime(2024, 1, 15, 11, 30, 0)  # local
    dt2 = datetime(2024, 1, 15, 10, 30, 0)  # remote (older)

    result = timestamp_manager.compare_timestamps(dt1, dt2)
    assert result == TimestampComparison.OLDER


def test_compare_timestamps_missing(timestamp_manager):
    """Test comparing with missing timestamps."""
    dt = datetime(2024, 1, 15, 10, 30, 0)

    result1 = timestamp_manager.compare_timestamps(None, dt)
    assert result1 == TimestampComparison.MISSING_LOCAL

    result2 = timestamp_manager.compare_timestamps(dt, None)
    assert result2 == TimestampComparison.MISSING_REMOTE

    result3 = timestamp_manager.compare_timestamps(None, None)
    assert result3 == TimestampComparison.IDENTICAL


def test_needs_update_scenarios(timestamp_manager):
    """Test various update need scenarios."""
    old_dt = datetime(2024, 1, 15, 10, 30, 0)
    new_dt = datetime(2024, 1, 15, 11, 30, 0)

    # Remote is newer - should update
    needs_update, reason = timestamp_manager.needs_update(old_dt, new_dt)
    assert needs_update is True
    assert "newer" in reason.lower()

    # Local is newer - should not update
    needs_update, reason = timestamp_manager.needs_update(new_dt, old_dt)
    assert needs_update is False

    # Force update
    needs_update, reason = timestamp_manager.needs_update(new_dt, old_dt, force_update=True)
    assert needs_update is True
    assert "forced" in reason.lower()

    # No local data - should update
    needs_update, reason = timestamp_manager.needs_update(None, new_dt)
    assert needs_update is True
    assert "no local data" in reason.lower()


def test_extract_timestamps_from_dataframe(timestamp_manager):
    """Test extracting timestamps from pandas DataFrames."""
    # Create a test DataFrame with datetime index
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    df = pd.DataFrame({'value': range(5)}, index=dates)

    timestamps = timestamp_manager.extract_timestamps_from_data(df)

    assert 'data_max_date' in timestamps
    assert 'data_min_date' in timestamps
    assert timestamps['data_max_date'].value is not None
    assert timestamps['data_min_date'].value is not None


def test_extract_timestamps_from_dict(timestamp_manager):
    """Test extracting timestamps from dictionaries."""
    data = {
        'last_updated': '2024-01-15T10:30:00',
        'created_date': '2024-01-01',
        'other_field': 'not a timestamp'
    }

    timestamps = timestamp_manager.extract_timestamps_from_data(data)

    assert 'last_updated' in timestamps
    assert 'created_date' in timestamps
    assert 'other_field' not in timestamps  # Should not extract non-timestamp fields


def test_data_freshness_score(timestamp_manager):
    """Test data freshness scoring."""
    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    # Very fresh data (1 hour old)
    fresh_timestamp = TimestampInfo(
        value=now.replace(hour=now.hour-1),
        source="test",
        confidence=1.0,
        timezone_aware=True,
        format_detected="test"
    )

    # Old data (1 week old)
    old_timestamp = TimestampInfo(
        value=now.replace(day=now.day-7),
        source="test",
        confidence=1.0,
        timezone_aware=True,
        format_detected="test"
    )

    fresh_score = timestamp_manager.get_data_freshness_score({'fresh': fresh_timestamp})
    old_score = timestamp_manager.get_data_freshness_score({'old': old_timestamp})

    assert fresh_score > old_score
    assert 0.0 <= fresh_score <= 1.0
    assert 0.0 <= old_score <= 1.0


def test_suggest_update_frequency(timestamp_manager):
    """Test update frequency suggestions."""
    # Test default frequencies
    price_freq = timestamp_manager.suggest_update_frequency('price')
    fundamental_freq = timestamp_manager.suggest_update_frequency('fundamental')

    assert price_freq.total_seconds() < fundamental_freq.total_seconds()

    # Test with historical pattern
    historical_updates = [
        datetime(2024, 1, 1, 9, 0, 0),
        datetime(2024, 1, 1, 10, 0, 0),
        datetime(2024, 1, 1, 11, 0, 0)
    ]

    freq_with_pattern = timestamp_manager.suggest_update_frequency(
        'price', historical_updates
    )

    # Should be close to 1 hour based on the pattern
    assert abs(freq_with_pattern.total_seconds() - 3600) < 600  # Within 10 minutes


def test_market_hours_adjustment(timestamp_manager):
    """Test market hours timestamp adjustment."""
    # Before market open (8 AM)
    before_open = datetime(2024, 1, 15, 8, 0, 0)
    adjusted1 = timestamp_manager.get_market_hours_adjusted_timestamp(before_open)

    # Should adjust to previous day's close
    assert adjusted1.hour == 13 and adjusted1.minute == 30

    # During market hours (10 AM)
    during_market = datetime(2024, 1, 15, 10, 0, 0)
    adjusted2 = timestamp_manager.get_market_hours_adjusted_timestamp(during_market)

    # Should keep original timestamp
    assert adjusted2 == during_market

    # After market close (3 PM)
    after_close = datetime(2024, 1, 15, 15, 0, 0)
    adjusted3 = timestamp_manager.get_market_hours_adjusted_timestamp(after_close)

    # Should adjust to today's close
    assert adjusted3.hour == 13 and adjusted3.minute == 30
    assert adjusted3.date() == after_close.date()


def test_format_timestamp_for_api(timestamp_manager):
    """Test formatting timestamps for API requests."""
    dt = datetime(2024, 1, 15, 10, 30, 0)

    iso_format = timestamp_manager.format_timestamp_for_api(dt, 'iso')
    date_format = timestamp_manager.format_timestamp_for_api(dt, 'date')
    epoch_format = timestamp_manager.format_timestamp_for_api(dt, 'epoch')

    assert 'T' in iso_format
    assert date_format == '2024-01-15'
    assert epoch_format.isdigit()