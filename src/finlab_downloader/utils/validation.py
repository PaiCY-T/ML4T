"""
Data validation utilities for FinLab downloader.

Provides validation functions for dates, symbols, and data formats.
"""

import re
from datetime import datetime, date, timedelta
from typing import List, Union, Optional, Tuple, Any
import pandas as pd

from ..core.exceptions import ValidationError


def validate_date_range(
    start_date: Union[str, date, datetime],
    end_date: Union[str, date, datetime],
    max_range_days: Optional[int] = None
) -> Tuple[date, date]:
    """
    Validate and normalize date range.

    Args:
        start_date: Start date
        end_date: End date
        max_range_days: Maximum allowed range in days

    Returns:
        Tuple of (start_date, end_date) as date objects

    Raises:
        ValidationError: If dates are invalid
    """
    # Convert to date objects
    start_dt = _parse_date(start_date, "start_date")
    end_dt = _parse_date(end_date, "end_date")

    # Validate range
    if start_dt > end_dt:
        raise ValidationError(
            "Start date cannot be after end date",
            field="date_range",
            value=f"{start_dt} - {end_dt}"
        )

    # Check maximum range if specified
    if max_range_days is not None:
        range_days = (end_dt - start_dt).days
        if range_days > max_range_days:
            raise ValidationError(
                f"Date range exceeds maximum allowed ({max_range_days} days)",
                field="date_range",
                value=range_days,
                expected=f"<= {max_range_days}"
            )

    # Check if dates are in the future
    today = date.today()
    if start_dt > today:
        raise ValidationError(
            "Start date cannot be in the future",
            field="start_date",
            value=start_dt
        )

    if end_dt > today:
        raise ValidationError(
            "End date cannot be in the future",
            field="end_date",
            value=end_dt
        )

    return start_dt, end_dt


def validate_symbols(symbols: List[str], max_symbols: Optional[int] = None) -> List[str]:
    """
    Validate list of symbol identifiers.

    Args:
        symbols: List of symbols to validate
        max_symbols: Maximum number of symbols allowed

    Returns:
        List of validated symbols

    Raises:
        ValidationError: If symbols are invalid
    """
    if not symbols:
        raise ValidationError(
            "Symbol list cannot be empty",
            field="symbols",
            value=symbols
        )

    if not isinstance(symbols, list):
        raise ValidationError(
            "Symbols must be a list",
            field="symbols",
            value=type(symbols).__name__
        )

    # Check maximum number of symbols
    if max_symbols is not None and len(symbols) > max_symbols:
        raise ValidationError(
            f"Too many symbols (maximum {max_symbols} allowed)",
            field="symbols",
            value=len(symbols),
            expected=f"<= {max_symbols}"
        )

    # Validate each symbol
    validated_symbols = []
    for i, symbol in enumerate(symbols):
        try:
            validated_symbol = validate_symbol(symbol)
            validated_symbols.append(validated_symbol)
        except ValidationError as e:
            raise ValidationError(
                f"Invalid symbol at index {i}: {e.message}",
                field=f"symbols[{i}]",
                value=symbol,
                cause=e
            )

    # Check for duplicates
    unique_symbols = list(set(validated_symbols))
    if len(unique_symbols) != len(validated_symbols):
        duplicates = [s for s in validated_symbols if validated_symbols.count(s) > 1]
        raise ValidationError(
            f"Duplicate symbols found: {list(set(duplicates))}",
            field="symbols",
            value=duplicates
        )

    return validated_symbols


def validate_symbol(symbol: str) -> str:
    """
    Validate individual symbol identifier.

    Args:
        symbol: Symbol to validate

    Returns:
        Validated symbol (normalized)

    Raises:
        ValidationError: If symbol is invalid
    """
    if not isinstance(symbol, str):
        raise ValidationError(
            "Symbol must be a string",
            field="symbol",
            value=type(symbol).__name__
        )

    # Remove whitespace and convert to uppercase
    symbol = symbol.strip().upper()

    if not symbol:
        raise ValidationError(
            "Symbol cannot be empty",
            field="symbol",
            value=symbol
        )

    # Check length
    if len(symbol) > 20:
        raise ValidationError(
            "Symbol too long (maximum 20 characters)",
            field="symbol",
            value=symbol,
            expected="<= 20 characters"
        )

    # Check for valid characters (alphanumeric, dots, hyphens)
    if not re.match(r'^[A-Z0-9.-]+$', symbol):
        raise ValidationError(
            "Symbol contains invalid characters (only A-Z, 0-9, ., - allowed)",
            field="symbol",
            value=symbol
        )

    return symbol


def is_valid_symbol(symbol: str) -> bool:
    """
    Check if symbol is valid without raising exceptions.

    Args:
        symbol: Symbol to check

    Returns:
        True if valid, False otherwise
    """
    try:
        validate_symbol(symbol)
        return True
    except ValidationError:
        return False


def validate_data_frame(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Validate pandas DataFrame structure and content.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows

    Returns:
        Validated DataFrame

    Raises:
        ValidationError: If DataFrame is invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            "Data must be a pandas DataFrame",
            field="data",
            value=type(df).__name__
        )

    # Check if empty
    if df.empty:
        raise ValidationError(
            "DataFrame cannot be empty",
            field="data",
            value="empty"
        )

    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(
                f"Missing required columns: {list(missing_columns)}",
                field="columns",
                value=list(df.columns),
                expected=required_columns
            )

    # Check row count
    row_count = len(df)

    if min_rows is not None and row_count < min_rows:
        raise ValidationError(
            f"Not enough rows (minimum {min_rows} required)",
            field="row_count",
            value=row_count,
            expected=f">= {min_rows}"
        )

    if max_rows is not None and row_count > max_rows:
        raise ValidationError(
            f"Too many rows (maximum {max_rows} allowed)",
            field="row_count",
            value=row_count,
            expected=f"<= {max_rows}"
        )

    return df


def validate_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate price data DataFrame.

    Args:
        df: Price data DataFrame

    Returns:
        Validated DataFrame

    Raises:
        ValidationError: If price data is invalid
    """
    # Check required columns for price data
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    df = validate_data_frame(df, required_columns=required_columns)

    # Check for non-negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            if (df[col] < 0).any():
                raise ValidationError(
                    f"Negative values found in {col} column",
                    field=col,
                    value="negative values"
                )

    # Check for non-negative volume
    if 'volume' in df.columns:
        if (df['volume'] < 0).any():
            raise ValidationError(
                "Negative values found in volume column",
                field="volume",
                value="negative values"
            )

    # Check OHLC relationships
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # High should be >= all other prices
        high_check = (
            (df['high'] >= df['open']) &
            (df['high'] >= df['low']) &
            (df['high'] >= df['close'])
        )
        if not high_check.all():
            raise ValidationError(
                "High price should be >= open, low, and close prices",
                field="ohlc_relationship",
                value="invalid high prices"
            )

        # Low should be <= all other prices
        low_check = (
            (df['low'] <= df['open']) &
            (df['low'] <= df['high']) &
            (df['low'] <= df['close'])
        )
        if not low_check.all():
            raise ValidationError(
                "Low price should be <= open, high, and close prices",
                field="ohlc_relationship",
                value="invalid low prices"
            )

    return df


def validate_numeric_range(
    value: Union[int, float],
    field_name: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    allow_zero: bool = True
) -> Union[int, float]:
    """
    Validate numeric value is within specified range.

    Args:
        value: Value to validate
        field_name: Name of the field being validated
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_zero: Whether zero is allowed

    Returns:
        Validated value

    Raises:
        ValidationError: If value is invalid
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field_name} must be a number",
            field=field_name,
            value=type(value).__name__
        )

    if not allow_zero and value == 0:
        raise ValidationError(
            f"{field_name} cannot be zero",
            field=field_name,
            value=value
        )

    if min_value is not None and value < min_value:
        raise ValidationError(
            f"{field_name} must be >= {min_value}",
            field=field_name,
            value=value,
            expected=f">= {min_value}"
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{field_name} must be <= {max_value}",
            field=field_name,
            value=value,
            expected=f"<= {max_value}"
        )

    return value


def _parse_date(date_input: Union[str, date, datetime], field_name: str) -> date:
    """
    Parse date input to date object.

    Args:
        date_input: Date input to parse
        field_name: Name of the field for error reporting

    Returns:
        Parsed date object

    Raises:
        ValidationError: If date cannot be parsed
    """
    if isinstance(date_input, date):
        return date_input
    elif isinstance(date_input, datetime):
        return date_input.date()
    elif isinstance(date_input, str):
        # Try to parse string date
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y%m%d'
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_input, fmt).date()
            except ValueError:
                continue

        raise ValidationError(
            f"Cannot parse date '{date_input}'. Expected format: YYYY-MM-DD",
            field=field_name,
            value=date_input,
            expected="YYYY-MM-DD"
        )
    else:
        raise ValidationError(
            f"{field_name} must be a string, date, or datetime object",
            field=field_name,
            value=type(date_input).__name__
        )