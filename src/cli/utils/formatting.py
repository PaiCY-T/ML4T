"""
Output formatting utilities for CLI interface.

Provides consistent formatting for tables, status displays, and other
common CLI output patterns using Rich formatting.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from rich.table import Table
from rich.console import Console
from rich.text import Text


def format_table(data: List[Dict[str, Any]],
                headers: Optional[List[str]] = None,
                title: Optional[str] = None,
                max_rows: Optional[int] = None) -> Table:
    """
    Format data as a Rich table.

    Args:
        data: List of dictionaries to display
        headers: Optional list of column headers (uses keys if None)
        title: Optional table title
        max_rows: Maximum number of rows to display

    Returns:
        Rich Table object
    """
    if not data:
        table = Table(title=title or "No Data")
        table.add_column("Message")
        table.add_row("No data available")
        return table

    # Use first row keys as headers if not provided
    if headers is None:
        headers = list(data[0].keys())

    # Create table
    table = Table(title=title, show_header=True, header_style="bold magenta")

    # Add columns
    for header in headers:
        table.add_column(header)

    # Add rows (with optional limit)
    display_data = data[:max_rows] if max_rows else data
    for row_data in display_data:
        row = [str(row_data.get(header, "")) for header in headers]
        table.add_row(*row)

    # Add truncation notice if needed
    if max_rows and len(data) > max_rows:
        table.add_row(*["..." for _ in headers])
        table.caption = f"Showing {max_rows} of {len(data)} rows"

    return table


def format_status(status: str, healthy: bool = True) -> Text:
    """
    Format status text with appropriate colors.

    Args:
        status: Status string
        healthy: Whether status represents healthy state

    Returns:
        Rich Text object with appropriate styling
    """
    if healthy:
        if status.lower() in ["running", "active", "healthy", "ok", "success", "completed"]:
            return Text(status, style="green")
        elif status.lower() in ["starting", "stopping", "pending", "in_progress"]:
            return Text(status, style="yellow")
        else:
            return Text(status, style="blue")
    else:
        if status.lower() in ["error", "failed", "critical", "down"]:
            return Text(status, style="red")
        elif status.lower() in ["warning", "degraded", "slow"]:
            return Text(status, style="orange3")
        else:
            return Text(status, style="red")


def format_duration(seconds: Union[int, float, timedelta]) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds or timedelta object

    Returns:
        Formatted duration string
    """
    if isinstance(seconds, timedelta):
        total_seconds = seconds.total_seconds()
    else:
        total_seconds = float(seconds)

    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        return f"{minutes:.1f}m"
    elif total_seconds < 86400:
        hours = total_seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = total_seconds / 86400
        return f"{days:.1f}d"


def format_bytes(bytes_value: Union[int, float]) -> str:
    """
    Format byte count in human-readable format.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted bytes string
    """
    if bytes_value < 1024:
        return f"{bytes_value}B"
    elif bytes_value < 1024**2:
        return f"{bytes_value/1024:.1f}KB"
    elif bytes_value < 1024**3:
        return f"{bytes_value/(1024**2):.1f}MB"
    elif bytes_value < 1024**4:
        return f"{bytes_value/(1024**3):.1f}GB"
    else:
        return f"{bytes_value/(1024**4):.1f}TB"


def format_timestamp(timestamp: Union[datetime, str],
                    format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp in consistent format.

    Args:
        timestamp: DateTime object or ISO string
        format_str: Format string for strftime

    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp  # Return as-is if parsing fails

    return timestamp.strftime(format_str)


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format percentage value.

    Args:
        value: Percentage value (0.0 to 1.0)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format number with thousands separators.

    Args:
        value: Number to format
        decimals: Number of decimal places for floats

    Returns:
        Formatted number string
    """
    if isinstance(value, int):
        return f"{value:,}"
    else:
        return f"{value:,.{decimals}f}"


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to specified length.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix