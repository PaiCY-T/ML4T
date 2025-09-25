"""
Data Format Converters for ML4T-Alpha Integration.

This module provides data format standardization and conversion utilities
to ensure seamless compatibility between FinLab data and ML4T-Alpha framework.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import json

from ..data.core.temporal import TemporalValue, DataType
from ..data.models.taiwan_market import TaiwanMarketData

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Supported data formats."""
    ML4T_ALPHA = "ml4t_alpha"
    OPENFE = "openfe"
    PANDAS = "pandas"
    ZIPLINE = "zipline"
    BACKTRADER = "backtrader"
    QUANTLIB = "quantlib"


@dataclass
class ConversionConfig:
    """Configuration for data format conversion."""
    # Output format settings
    target_format: DataFormat = DataFormat.ML4T_ALPHA
    include_metadata: bool = True
    flatten_columns: bool = False

    # Data processing options
    forward_fill_missing: bool = True
    handle_corporate_actions: bool = True
    adjust_for_splits: bool = True
    adjust_for_dividends: bool = True

    # Frequency and resampling
    target_frequency: str = "D"  # Daily by default
    business_day_only: bool = True

    # Quality filters
    min_price_threshold: float = 0.01
    max_price_change_pct: float = 0.5  # 50% daily change limit
    min_volume_threshold: int = 0

    # Column naming conventions
    price_column_prefix: str = ""
    fundamental_column_prefix: str = "fund_"
    technical_column_prefix: str = "tech_"


class FinLabToML4TConverter:
    """
    Converts FinLab temporal data to ML4T-Alpha compatible format.
    """

    def __init__(self, config: ConversionConfig):
        self.config = config
        self.conversion_count = 0
        self.quality_issues = 0

    def convert_price_data(self,
                          temporal_values: List[TemporalValue],
                          symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert FinLab temporal price data to ML4T-Alpha format.

        Args:
            temporal_values: List of temporal values from FinLab
            symbols: Optional symbol filter

        Returns:
            DataFrame in ML4T-Alpha format
        """
        if not temporal_values:
            return pd.DataFrame()

        # Group data by symbol and date
        data_dict = {}
        for tv in temporal_values:
            if symbols and tv.symbol not in symbols:
                continue

            if tv.data_type in [DataType.PRICE, DataType.VOLUME, DataType.MARKET_DATA]:
                key = (tv.symbol, tv.value_date)
                if key not in data_dict:
                    data_dict[key] = {}

                field_name = tv.metadata.get('field', 'unknown') if tv.metadata else 'value'
                data_dict[key][field_name] = float(tv.value) if tv.value is not None else np.nan

        if not data_dict:
            return pd.DataFrame()

        # Convert to DataFrame
        rows = []
        for (symbol, date), fields in data_dict.items():
            row = {'symbol': symbol, 'date': date}
            row.update(fields)
            rows.append(row)

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date'])

        # Apply quality filters
        df = self._apply_quality_filters(df)

        # Handle missing data
        if self.config.forward_fill_missing:
            df = df.groupby('symbol').apply(self._forward_fill_group).reset_index(drop=True)

        # Pivot to ML4T format (dates as index, symbols as columns)
        if self.config.target_format == DataFormat.ML4T_ALPHA:
            df = self._pivot_to_ml4t_format(df)
        elif self.config.target_format == DataFormat.OPENFE:
            df = self._format_for_openfe(df)

        self.conversion_count += 1
        return df

    def convert_fundamental_data(self,
                               temporal_values: List[TemporalValue],
                               symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Convert fundamental data with proper lag handling."""
        if not temporal_values:
            return pd.DataFrame()

        data_dict = {}
        for tv in temporal_values:
            if symbols and tv.symbol not in symbols:
                continue

            if tv.data_type == DataType.FUNDAMENTAL:
                # Use as_of_date for availability (accounting for reporting lag)
                key = (tv.symbol, tv.as_of_date)
                if key not in data_dict:
                    data_dict[key] = {
                        'report_date': tv.value_date,
                        'fiscal_year': tv.metadata.get('fiscal_year') if tv.metadata else None,
                        'fiscal_quarter': tv.metadata.get('fiscal_quarter') if tv.metadata else None
                    }

                field_name = tv.metadata.get('field', 'unknown') if tv.metadata else 'value'
                # Add prefix for fundamental fields
                prefixed_field = f"{self.config.fundamental_column_prefix}{field_name}"
                data_dict[key][prefixed_field] = float(tv.value) if tv.value is not None else np.nan

        if not data_dict:
            return pd.DataFrame()

        # Convert to DataFrame
        rows = []
        for (symbol, as_of_date), fields in data_dict.items():
            row = {'symbol': symbol, 'as_of_date': as_of_date}
            row.update(fields)
            rows.append(row)

        df = pd.DataFrame(rows)
        df['as_of_date'] = pd.to_datetime(df['as_of_date'])
        df = df.sort_values(['symbol', 'as_of_date'])

        # Format for target framework
        if self.config.target_format == DataType.ML4T_ALPHA:
            df = self._pivot_fundamental_data(df)

        return df

    def convert_to_zipline_bundle(self,
                                 price_data: pd.DataFrame,
                                 fundamental_data: Optional[pd.DataFrame] = None,
                                 output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Convert data to Zipline bundle format for ML4T integration.

        Args:
            price_data: Price data DataFrame
            fundamental_data: Optional fundamental data
            output_dir: Output directory for bundle files

        Returns:
            Dictionary mapping data types to file paths
        """
        if output_dir is None:
            output_dir = Path.cwd() / "zipline_bundles" / f"finlab_{datetime.now().strftime('%Y%m%d')}"

        output_dir.mkdir(parents=True, exist_ok=True)

        bundle_files = {}

        # Convert price data to Zipline format
        if not price_data.empty:
            zipline_price = self._convert_to_zipline_price_format(price_data)
            price_path = output_dir / "daily_prices.csv"
            zipline_price.to_csv(price_path)
            bundle_files['prices'] = price_path

            # Create Zipline metadata
            metadata = self._create_zipline_metadata(zipline_price)
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            bundle_files['metadata'] = metadata_path

        # Convert fundamental data if provided
        if fundamental_data is not None and not fundamental_data.empty:
            zipline_fundamental = self._convert_to_zipline_fundamental_format(fundamental_data)
            fundamental_path = output_dir / "fundamentals.csv"
            zipline_fundamental.to_csv(fundamental_path)
            bundle_files['fundamentals'] = fundamental_path

        logger.info(f"Created Zipline bundle in {output_dir}")
        return bundle_files

    def _apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters."""
        initial_rows = len(df)

        # Price threshold filter
        if 'close' in df.columns:
            mask = df['close'] >= self.config.min_price_threshold
            df = df[mask]

        # Volume threshold filter
        if 'volume' in df.columns:
            mask = df['volume'] >= self.config.min_volume_threshold
            df = df[mask]

        # Price change filter
        if 'close' in df.columns:
            df = df.groupby('symbol').apply(self._filter_price_changes).reset_index(drop=True)

        filtered_rows = len(df)
        if filtered_rows < initial_rows:
            self.quality_issues += (initial_rows - filtered_rows)
            logger.debug(f"Filtered {initial_rows - filtered_rows} rows due to quality issues")

        return df

    def _filter_price_changes(self, group: pd.DataFrame) -> pd.DataFrame:
        """Filter extreme price changes within a symbol group."""
        if len(group) < 2:
            return group

        group = group.sort_values('date')
        group['price_change_pct'] = group['close'].pct_change().abs()

        # Keep rows within price change threshold
        mask = (group['price_change_pct'].isna()) | (group['price_change_pct'] <= self.config.max_price_change_pct)
        return group[mask]

    def _forward_fill_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """Forward fill missing data within a symbol group."""
        # Fill price columns
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        for col in price_columns:
            if col in group.columns:
                group[col] = group[col].fillna(method='ffill')

        return group

    def _pivot_to_ml4t_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot data to ML4T-Alpha format with multi-level columns."""
        if 'symbol' not in df.columns or 'date' not in df.columns:
            return df

        # Get data columns (exclude symbol and date)
        data_columns = [col for col in df.columns if col not in ['symbol', 'date']]

        if not data_columns:
            return df

        # Create separate DataFrames for each data type
        result_dfs = {}
        for col in data_columns:
            pivoted = df.pivot(index='date', columns='symbol', values=col)
            result_dfs[col] = pivoted

        # Combine into multi-level DataFrame
        combined_df = pd.concat(result_dfs, axis=1)
        combined_df.index.name = 'date'

        return combined_df

    def _format_for_openfe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format data for openFE compatibility."""
        if self.config.flatten_columns and isinstance(df.columns, pd.MultiIndex):
            # Flatten multi-level columns
            df.columns = [f"{level1}_{level0}" if level0 != '' else level1
                         for level0, level1 in df.columns]

        return df

    def _pivot_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot fundamental data for ML4T format."""
        if 'symbol' not in df.columns or 'as_of_date' not in df.columns:
            return df

        # Get fundamental columns
        fund_columns = [col for col in df.columns
                       if col.startswith(self.config.fundamental_column_prefix)]

        if not fund_columns:
            return df

        # Pivot each fundamental field
        result_dfs = {}
        for col in fund_columns:
            pivoted = df.pivot(index='as_of_date', columns='symbol', values=col)
            result_dfs[col] = pivoted

        # Combine
        combined_df = pd.concat(result_dfs, axis=1)
        combined_df.index.name = 'as_of_date'

        return combined_df

    def _convert_to_zipline_price_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to Zipline bundle price format."""
        # Zipline expects specific column names and format
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if isinstance(df.columns, pd.MultiIndex):
            # Handle multi-level columns from ML4T format
            zipline_data = []

            # Get unique symbols from column level
            symbols = df.columns.get_level_values(1).unique()

            for symbol in symbols:
                symbol_data = df.xs(symbol, axis=1, level=1)
                symbol_data['symbol'] = symbol
                symbol_data = symbol_data.reset_index()
                zipline_data.append(symbol_data)

            zipline_df = pd.concat(zipline_data, ignore_index=True)
        else:
            zipline_df = df.copy()

        # Ensure required columns exist
        for col in required_columns:
            if col not in zipline_df.columns:
                if col == 'volume':
                    zipline_df[col] = 0  # Default volume
                else:
                    zipline_df[col] = zipline_df.get('close', np.nan)  # Use close as fallback

        # Add Zipline-specific columns
        zipline_df['dividend'] = 0.0
        zipline_df['split'] = 1.0

        return zipline_df[['symbol', 'date'] + required_columns + ['dividend', 'split']]

    def _convert_to_zipline_fundamental_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert fundamental data to Zipline format."""
        zipline_df = df.copy()

        # Rename as_of_date to timestamp for Zipline
        if 'as_of_date' in zipline_df.columns:
            zipline_df = zipline_df.rename(columns={'as_of_date': 'timestamp'})

        return zipline_df

    def _create_zipline_metadata(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Create Zipline bundle metadata."""
        symbols = price_data['symbol'].unique().tolist()
        start_date = price_data['date'].min()
        end_date = price_data['date'].max()

        metadata = {
            'bundle_name': 'finlab_taiwan',
            'description': 'FinLab Taiwan Market Data Bundle',
            'symbols': symbols,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'calendar': 'XTAI',  # Taiwan stock exchange calendar
            'frequency': 'daily',
            'created': datetime.now().isoformat()
        }

        return metadata

    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        return {
            'conversion_count': self.conversion_count,
            'quality_issues': self.quality_issues,
            'quality_filter_rate': self.quality_issues / max(self.conversion_count, 1)
        }


class OpenFEDataAdapter:
    """
    Adapter for openFE factor generation compatibility.
    """

    def __init__(self, config: ConversionConfig):
        self.config = config

    def prepare_feature_matrix(self,
                             price_data: pd.DataFrame,
                             fundamental_data: Optional[pd.DataFrame] = None,
                             technical_indicators: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare feature matrix for openFE factor generation.

        Args:
            price_data: Price data DataFrame
            fundamental_data: Optional fundamental data
            technical_indicators: Optional technical indicators

        Returns:
            Combined feature matrix suitable for openFE
        """
        features = []

        # Add price features
        if not price_data.empty:
            price_features = self._extract_price_features(price_data)
            features.append(price_features)

        # Add fundamental features
        if fundamental_data is not None and not fundamental_data.empty:
            fundamental_features = self._extract_fundamental_features(fundamental_data)
            features.append(fundamental_features)

        # Add technical indicators
        if technical_indicators is not None and not technical_indicators.empty:
            tech_features = self._extract_technical_features(technical_indicators)
            features.append(tech_features)

        if not features:
            return pd.DataFrame()

        # Combine all features
        combined_features = pd.concat(features, axis=1)

        # Clean up the feature matrix
        combined_features = self._clean_feature_matrix(combined_features)

        return combined_features

    def _extract_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract price-based features for openFE."""
        features = price_data.copy()

        # Add derived price features
        if isinstance(features.columns, pd.MultiIndex):
            # Handle multi-level columns
            for field in ['close', 'high', 'low', 'volume']:
                if field in features.columns.get_level_values(0):
                    field_data = features.xs(field, axis=1, level=0)

                    # Add returns
                    returns = field_data.pct_change()
                    returns.columns = pd.MultiIndex.from_product([['returns'], returns.columns])
                    features = pd.concat([features, returns], axis=1)

                    # Add volatility (rolling std of returns)
                    if field == 'close':
                        volatility = returns.rolling(window=20).std()
                        volatility.columns = pd.MultiIndex.from_product([['volatility_20d'], volatility.columns.get_level_values(1)])
                        features = pd.concat([features, volatility], axis=1)
        else:
            # Simple column structure
            if 'close' in features.columns:
                # Add returns for each symbol column
                return_cols = [col for col in features.columns if 'close' in col]
                for col in return_cols:
                    symbol = col.replace('close_', '') if 'close_' in col else col
                    features[f'returns_{symbol}'] = features[col].pct_change()
                    features[f'volatility_20d_{symbol}'] = features[f'returns_{symbol}'].rolling(20).std()

        return features

    def _extract_fundamental_features(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """Extract fundamental features for openFE."""
        features = fundamental_data.copy()

        # Add fundamental ratios and derived metrics
        if isinstance(features.columns, pd.MultiIndex):
            # Handle multi-level columns
            symbols = features.columns.get_level_values(1).unique()

            for symbol in symbols:
                symbol_data = features.xs(symbol, axis=1, level=1)

                # Calculate financial ratios
                if 'fund_revenue' in symbol_data.columns and 'fund_total_assets' in symbol_data.columns:
                    asset_turnover = symbol_data['fund_revenue'] / symbol_data['fund_total_assets']
                    features[('asset_turnover', symbol)] = asset_turnover

                if 'fund_net_income' in symbol_data.columns and 'fund_revenue' in symbol_data.columns:
                    profit_margin = symbol_data['fund_net_income'] / symbol_data['fund_revenue']
                    features[('profit_margin', symbol)] = profit_margin

        return features

    def _extract_technical_features(self, technical_data: pd.DataFrame) -> pd.DataFrame:
        """Extract technical indicator features for openFE."""
        return technical_data.copy()

    def _clean_feature_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the feature matrix for openFE."""
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)

        # Forward fill missing values if configured
        if self.config.forward_fill_missing:
            features = features.fillna(method='ffill')

        # Drop columns with too many missing values
        threshold = 0.5  # Drop columns with >50% missing
        features = features.dropna(thresh=int(threshold * len(features)), axis=1)

        return features


class BacktestDataFormatter:
    """
    Formats data for various backtesting frameworks.
    """

    def __init__(self, config: ConversionConfig):
        self.config = config

    def format_for_backtrader(self,
                            price_data: pd.DataFrame,
                            symbol: str) -> pd.DataFrame:
        """Format data for Backtrader framework."""
        if isinstance(price_data.columns, pd.MultiIndex):
            # Extract data for specific symbol
            symbol_data = price_data.xs(symbol, axis=1, level=1)
        else:
            # Assume single symbol data
            symbol_data = price_data.copy()

        # Backtrader expects specific column names
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Rename columns if necessary
        column_mapping = {
            'adj_close': 'close',  # Use adjusted close as close
        }

        symbol_data = symbol_data.rename(columns=column_mapping)

        # Ensure all required columns exist
        for col in required_columns:
            if col not in symbol_data.columns:
                if col == 'volume':
                    symbol_data[col] = 0
                else:
                    symbol_data[col] = symbol_data.get('close', np.nan)

        # Ensure datetime index
        if not isinstance(symbol_data.index, pd.DatetimeIndex):
            symbol_data.index = pd.to_datetime(symbol_data.index)

        return symbol_data[required_columns]

    def format_for_quantlib(self,
                          price_data: pd.DataFrame,
                          symbol: str) -> Dict[str, Any]:
        """Format data for QuantLib framework."""
        if isinstance(price_data.columns, pd.MultiIndex):
            symbol_data = price_data.xs(symbol, axis=1, level=1)
        else:
            symbol_data = price_data.copy()

        # QuantLib format
        quantlib_data = {
            'dates': symbol_data.index.tolist(),
            'prices': symbol_data['close'].tolist(),
            'symbol': symbol
        }

        if 'volume' in symbol_data.columns:
            quantlib_data['volumes'] = symbol_data['volume'].tolist()

        return quantlib_data