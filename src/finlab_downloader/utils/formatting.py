"""
Formatting utilities for FinLab downloader.

Provides functions for formatting sizes, durations, and progress indicators.
"""

import time
from typing import Optional, Iterator
from datetime import datetime, timedelta


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.2 MB")
    """
    if size_bytes == 0:
        return "0 B"

    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "1h 23m 45s")
    """
    if seconds < 0:
        return "0s"

    # Handle sub-second durations
    if seconds < 1:
        milliseconds = int(seconds * 1000)
        return f"{milliseconds}ms"

    # Convert to integer seconds for simplicity
    total_seconds = int(seconds)

    # Calculate components
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    # Format based on magnitude
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def format_timestamp(dt: Optional[datetime] = None, include_microseconds: bool = False) -> str:
    """
    Format timestamp to ISO string.

    Args:
        dt: Datetime object (uses current time if None)
        include_microseconds: Whether to include microseconds

    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()

    if include_microseconds:
        return dt.isoformat()
    else:
        return dt.replace(microsecond=0).isoformat()


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format percentage value.

    Args:
        value: Percentage value (0.0 to 1.0)
        decimal_places: Number of decimal places

    Returns:
        Formatted percentage string (e.g., "45.2%")
    """
    percentage = value * 100
    return f"{percentage:.{decimal_places}f}%"


def format_number(value: float, decimal_places: int = 2, thousands_sep: bool = True) -> str:
    """
    Format number with specified decimal places and optional thousands separator.

    Args:
        value: Number to format
        decimal_places: Number of decimal places
        thousands_sep: Whether to include thousands separator

    Returns:
        Formatted number string
    """
    if thousands_sep:
        return f"{value:,.{decimal_places}f}"
    else:
        return f"{value:.{decimal_places}f}"


class ProgressBar:
    """Simple text-based progress bar."""

    def __init__(
        self,
        total: int,
        width: int = 50,
        prefix: str = "Progress",
        suffix: str = "Complete",
        fill: str = "█",
        empty: str = "░"
    ):
        """
        Initialize progress bar.

        Args:
            total: Total number of items
            width: Width of progress bar in characters
            prefix: Text before progress bar
            suffix: Text after progress bar
            fill: Character for filled portion
            empty: Character for empty portion
        """
        self.total = total
        self.width = width
        self.prefix = prefix
        self.suffix = suffix
        self.fill = fill
        self.empty = empty
        self.current = 0
        self.start_time = time.time()

    def update(self, current: Optional[int] = None) -> str:
        """
        Update progress bar and return formatted string.

        Args:
            current: Current progress (increments by 1 if None)

        Returns:
            Formatted progress bar string
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1

        # Calculate percentage
        percentage = self.current / self.total if self.total > 0 else 0
        percentage = min(1.0, max(0.0, percentage))

        # Calculate bar components
        filled_length = int(self.width * percentage)
        bar = self.fill * filled_length + self.empty * (self.width - filled_length)

        # Calculate elapsed and estimated time
        elapsed_time = time.time() - self.start_time
        if self.current > 0 and percentage > 0:
            estimated_total = elapsed_time / percentage
            remaining_time = estimated_total - elapsed_time
        else:
            remaining_time = 0

        # Format time strings
        elapsed_str = format_duration(elapsed_time)
        remaining_str = format_duration(remaining_time)

        # Format percentage
        percent_str = format_percentage(percentage, 1)

        return (
            f"{self.prefix}: |{bar}| {self.current}/{self.total} "
            f"({percent_str}) [{elapsed_str}<{remaining_str}] {self.suffix}"
        )

    def finish(self) -> str:
        """
        Mark progress as complete and return final string.

        Returns:
            Final progress bar string
        """
        self.current = self.total
        elapsed_time = time.time() - self.start_time
        elapsed_str = format_duration(elapsed_time)

        bar = self.fill * self.width

        return (
            f"{self.prefix}: |{bar}| {self.total}/{self.total} "
            f"(100.0%) [{elapsed_str}] {self.suffix}"
        )


def progress_bar(
    iterable,
    total: Optional[int] = None,
    prefix: str = "Progress",
    suffix: str = "Complete",
    width: int = 50
) -> Iterator:
    """
    Wrap an iterable with a progress bar.

    Args:
        iterable: Iterable to wrap
        total: Total number of items (auto-detected if None)
        prefix: Text before progress bar
        suffix: Text after progress bar
        width: Width of progress bar

    Yields:
        Items from the iterable
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            # Convert to list to get length
            iterable = list(iterable)
            total = len(iterable)

    progress = ProgressBar(total, width, prefix, suffix)

    for i, item in enumerate(iterable):
        yield item
        # Print progress (using carriage return to overwrite)
        print(f"\r{progress.update(i + 1)}", end="", flush=True)

    # Print final progress and newline
    print(f"\r{progress.finish()}")


def format_table(
    data: list,
    headers: list,
    max_width: Optional[int] = None,
    alignment: Optional[list] = None
) -> str:
    """
    Format data as a simple text table.

    Args:
        data: List of rows (each row is a list of values)
        headers: List of column headers
        max_width: Maximum column width
        alignment: List of alignment characters ('l', 'c', 'r')

    Returns:
        Formatted table string
    """
    if not data or not headers:
        return ""

    # Convert all values to strings
    str_data = [[str(cell) for cell in row] for row in data]
    str_headers = [str(header) for header in headers]

    # Calculate column widths
    col_widths = []
    for i in range(len(headers)):
        max_width_col = max(
            len(str_headers[i]),
            max(len(row[i]) if i < len(row) else 0 for row in str_data)
        )
        if max_width:
            max_width_col = min(max_width_col, max_width)
        col_widths.append(max_width_col)

    # Default alignment
    if alignment is None:
        alignment = ['l'] * len(headers)

    # Format header
    header_row = []
    for i, (header, width, align) in enumerate(zip(str_headers, col_widths, alignment)):
        if len(header) > width:
            header = header[:width-3] + "..."

        if align == 'c':
            header = header.center(width)
        elif align == 'r':
            header = header.rjust(width)
        else:  # 'l' or default
            header = header.ljust(width)

        header_row.append(header)

    # Create separator
    separator = '+'.join('-' * (width + 2) for width in col_widths)
    separator = '+' + separator + '+'

    # Create header line
    header_line = '| ' + ' | '.join(header_row) + ' |'

    # Format data rows
    data_rows = []
    for row in str_data:
        formatted_row = []
        for i, (cell, width, align) in enumerate(zip(row, col_widths, alignment)):
            if i >= len(row):
                cell = ""

            if len(cell) > width:
                cell = cell[:width-3] + "..."

            if align == 'c':
                cell = cell.center(width)
            elif align == 'r':
                cell = cell.rjust(width)
            else:  # 'l' or default
                cell = cell.ljust(width)

            formatted_row.append(cell)

        data_rows.append('| ' + ' | '.join(formatted_row) + ' |')

    # Combine all parts
    table_lines = [separator, header_line, separator] + data_rows + [separator]
    return '\n'.join(table_lines)