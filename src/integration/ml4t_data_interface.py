"""
ML4T-Alpha Data Interface.

This module provides the core data interface for ML4T-Alpha compatibility,
establishing reliable data connectivity between FinLab data pipeline and
ML4T-Alpha backtesting framework.
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Set
from dataclasses import dataclass, field
from decimal import Decimal
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from pathlib import Path

from ..data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
from ..data.pipeline.pit_engine import (
    PointInTimeEngine, PITQuery, DataType, QueryMode, BiasCheckLevel,
    create_optimized_pit_engine
)
from ..data.core.temporal import (
    TemporalValue, TemporalStore, InMemoryTemporalStore
)
from ..data.models.taiwan_market import (
    TaiwanMarketData, TaiwanTradingCalendar, create_taiwan_trading_calendar
)

logger = logging.getLogger(__name__)


@dataclass
class ML4TDataConfig:
    """Configuration for ML4T-Alpha data interface."""
    # FinLab connector configuration
    finlab_config: FinLabConfig

    # Data access configuration
    default_lookback_days: int = 252  # 1 year of trading days
    max_symbols_per_query: int = 100
    enable_cache: bool = True
    cache_size: int = 10000

    # Point-in-time configuration
    enable_pit: bool = True
    bias_check_level: BiasCheckLevel = BiasCheckLevel.STRICT
    max_workers: int = 4

    # Performance optimization
    bulk_query_threshold: int = 50
    enable_streaming: bool = True
    streaming_buffer_size: int = 1000

    # Data quality settings
    min_data_quality_score: float = 80.0
    enable_data_validation: bool = True

    # Storage settings
    data_cache_dir: Optional[Path] = None
    export_format: str = "hdf5"  # "hdf5", "parquet", "csv"

    def __post_init__(self):
        """Initialize derived configurations."""
        if self.data_cache_dir is None:
            self.data_cache_dir = Path.cwd() / "cache" / "ml4t_data"
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)


class ML4TDataInterface:
    """
    Core data interface for ML4T-Alpha compatibility.

    Provides seamless data access between FinLab data pipeline and ML4T-Alpha
    backtesting framework with point-in-time integrity and performance optimization.
    """

    def __init__(self,
                 config: ML4TDataConfig,
                 finlab_connector: Optional[FinLabConnector] = None,
                 pit_engine: Optional[PointInTimeEngine] = None,
                 temporal_store: Optional[TemporalStore] = None):
        self.config = config

        # Initialize temporal store
        if temporal_store is None:
            temporal_store = InMemoryTemporalStore()
        self.temporal_store = temporal_store

        # Initialize FinLab connector
        if finlab_connector is None:
            finlab_connector = FinLabConnector(
                config.finlab_config,
                temporal_store,
                enable_cache=config.enable_cache,
                max_workers=config.max_workers
            )
        self.finlab_connector = finlab_connector

        # Initialize PIT engine
        if pit_engine is None:
            pit_engine = create_optimized_pit_engine(
                connection_params=None,  # Will use in-memory for now
                store_type="memory",
                enable_cache=config.enable_cache,
                max_workers=config.max_workers
            )
        self.pit_engine = pit_engine

        # Trading calendar
        self.trading_calendar = create_taiwan_trading_calendar(2024)

        # Performance metrics
        self.query_count = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
        self.data_quality_issues = 0

        # Connection state
        self._connected = False

        logger.info("ML4T-Alpha data interface initialized")

    def connect(self) -> None:
        """Establish connection to FinLab data source."""
        try:
            self.finlab_connector.connect()
            self._connected = True
            logger.info("ML4T-Alpha data interface connected")
        except Exception as e:
            logger.error(f"Failed to connect ML4T data interface: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from data sources."""
        if self._connected:
            self.finlab_connector.disconnect()
            self._connected = False
            logger.info("ML4T-Alpha data interface disconnected")

    def get_symbols(self, as_of_date: Optional[date] = None) -> List[str]:
        """Get available symbols for ML4T-Alpha backtesting."""
        if not self._connected:
            self.connect()

        return self.finlab_connector.get_available_symbols(as_of_date)

    def get_price_data(self,
                      symbols: List[str],
                      start_date: date,
                      end_date: date,
                      fields: Optional[List[str]] = None,
                      as_of_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get price data formatted for ML4T-Alpha backtesting.

        Args:
            symbols: List of symbols to retrieve
            start_date: Start date for data range
            end_date: End date for data range
            fields: Specific price fields to retrieve
            as_of_date: Point-in-time date for data consistency

        Returns:
            DataFrame with ML4T-Alpha compatible price data
        """
        if not self._connected:
            self.connect()

        start_time = datetime.utcnow()

        if as_of_date is None:
            as_of_date = end_date

        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume', 'adj_close']

        # Use point-in-time engine for temporal consistency
        if self.config.enable_pit:
            return self._get_pit_price_data(symbols, start_date, end_date, fields, as_of_date)
        else:
            return self._get_direct_price_data(symbols, start_date, end_date, fields)

    def _get_pit_price_data(self,
                           symbols: List[str],
                           start_date: date,
                           end_date: date,
                           fields: List[str],
                           as_of_date: date) -> pd.DataFrame:
        """Get price data using point-in-time engine."""
        # Create PIT queries for each date in range
        current_date = start_date
        all_data = []

        while current_date <= end_date:
            # Skip non-trading days
            if current_date not in self.trading_calendar:
                current_date += timedelta(days=1)
                continue

            # Create PIT query
            query = PITQuery(
                symbols=symbols,
                as_of_date=min(current_date, as_of_date),  # Don't look ahead
                data_types=[DataType.PRICE, DataType.VOLUME],
                mode=QueryMode.FAST if len(symbols) < self.config.bulk_query_threshold else QueryMode.BULK,
                bias_check=self.config.bias_check_level
            )

            try:
                result = self.pit_engine.execute_query(query)

                # Convert to DataFrame format
                for symbol in symbols:
                    symbol_data = result.data.get(symbol, {})

                    row_data = {
                        'symbol': symbol,
                        'date': current_date
                    }

                    # Extract price data
                    for field in fields:
                        temporal_value = symbol_data.get(DataType.PRICE)
                        if temporal_value and temporal_value.metadata.get('field') == field:
                            row_data[field] = float(temporal_value.value) if temporal_value.value else np.nan
                        else:
                            row_data[field] = np.nan

                    all_data.append(row_data)

            except Exception as e:
                logger.warning(f"Failed to get PIT data for {current_date}: {e}")

            current_date += timedelta(days=1)

        # Convert to DataFrame and pivot for ML4T format
        df = pd.DataFrame(all_data)
        if df.empty:
            return df

        # Pivot to get symbols as columns and dates as index
        price_data = {}
        for field in fields:
            field_data = df.pivot(index='date', columns='symbol', values=field)
            price_data[field] = field_data

        # Combine into multi-level DataFrame (ML4T format)
        combined_df = pd.concat(price_data, axis=1)
        combined_df.index = pd.to_datetime(combined_df.index)
        combined_df = combined_df.sort_index()

        self.query_count += 1
        self.total_query_time += (datetime.utcnow() - start_time).total_seconds()

        return combined_df

    def _get_direct_price_data(self,
                              symbols: List[str],
                              start_date: date,
                              end_date: date,
                              fields: List[str]) -> pd.DataFrame:
        """Get price data directly from FinLab connector."""
        all_data = []

        def fetch_symbol_data(symbol: str) -> pd.DataFrame:
            try:
                temporal_values = self.finlab_connector.get_price_data(
                    symbol, start_date, end_date, fields
                )

                # Convert to DataFrame
                data_rows = []
                for tv in temporal_values:
                    if tv.metadata and 'field' in tv.metadata:
                        data_rows.append({
                            'symbol': symbol,
                            'date': tv.value_date,
                            'field': tv.metadata['field'],
                            'value': float(tv.value) if tv.value else np.nan
                        })

                return pd.DataFrame(data_rows)

            except Exception as e:
                logger.warning(f"Failed to fetch data for symbol {symbol}: {e}")
                return pd.DataFrame()

        # Parallel execution for multiple symbols
        if len(symbols) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(fetch_symbol_data, symbol) for symbol in symbols]
                for future in as_completed(futures):
                    result_df = future.result()
                    if not result_df.empty:
                        all_data.append(result_df)
        else:
            result_df = fetch_symbol_data(symbols[0])
            if not result_df.empty:
                all_data.append(result_df)

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)

        # Pivot to ML4T format
        pivoted = combined.pivot_table(
            index='date',
            columns=['field', 'symbol'],
            values='value',
            aggfunc='first'
        )

        pivoted.index = pd.to_datetime(pivoted.index)
        return pivoted.sort_index()

    def get_fundamental_data(self,
                           symbols: List[str],
                           start_date: date,
                           end_date: date,
                           fields: Optional[List[str]] = None,
                           as_of_date: Optional[date] = None) -> pd.DataFrame:
        """Get fundamental data for ML4T-Alpha backtesting with proper lag handling."""
        if not self._connected:
            self.connect()

        if as_of_date is None:
            as_of_date = end_date

        if fields is None:
            fields = ['revenue', 'net_income', 'total_assets', 'eps', 'roe', 'pe_ratio']

        all_data = []

        for symbol in symbols:
            try:
                temporal_values = self.finlab_connector.get_fundamental_data(
                    symbol, start_date, end_date, fields
                )

                for tv in temporal_values:
                    # Only include data available as of the query date
                    if tv.as_of_date <= as_of_date:
                        all_data.append({
                            'symbol': symbol,
                            'report_date': tv.value_date,
                            'announce_date': tv.as_of_date,
                            'field': tv.metadata.get('field', 'unknown'),
                            'value': float(tv.value) if tv.value else np.nan,
                            'fiscal_year': tv.metadata.get('fiscal_year'),
                            'fiscal_quarter': tv.metadata.get('fiscal_quarter'),
                            'lag_days': tv.metadata.get('lag_days', 0)
                        })

            except Exception as e:
                logger.warning(f"Failed to get fundamental data for {symbol}: {e}")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Create ML4T compatible format using announce_date as index
        pivoted = df.pivot_table(
            index='announce_date',
            columns=['field', 'symbol'],
            values='value',
            aggfunc='first'
        )

        pivoted.index = pd.to_datetime(pivoted.index)
        return pivoted.sort_index()

    def get_openfe_compatible_data(self,
                                  symbols: List[str],
                                  start_date: date,
                                  end_date: date,
                                  include_fundamentals: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get data in format compatible with openFE for factor generation.

        Returns:
            Dictionary with 'price_data' and optionally 'fundamental_data'
        """
        result = {}

        # Get price data
        price_data = self.get_price_data(symbols, start_date, end_date)
        if not price_data.empty:
            # Flatten multi-level columns for openFE compatibility
            flattened_price = price_data.copy()
            if isinstance(flattened_price.columns, pd.MultiIndex):
                flattened_price.columns = [f"{field}_{symbol}" for field, symbol in flattened_price.columns]
            result['price_data'] = flattened_price

        # Get fundamental data if requested
        if include_fundamentals:
            fundamental_data = self.get_fundamental_data(symbols, start_date, end_date)
            if not fundamental_data.empty:
                flattened_fundamental = fundamental_data.copy()
                if isinstance(flattened_fundamental.columns, pd.MultiIndex):
                    flattened_fundamental.columns = [f"{field}_{symbol}" for field, symbol in flattened_fundamental.columns]
                result['fundamental_data'] = flattened_fundamental

        return result

    def export_backtest_data(self,
                           symbols: List[str],
                           start_date: date,
                           end_date: date,
                           output_path: Optional[Path] = None,
                           format: str = "hdf5") -> Path:
        """
        Export data in format suitable for ML4T-Alpha backtesting.

        Args:
            symbols: Symbols to export
            start_date: Start date
            end_date: End date
            output_path: Output file path (auto-generated if None)
            format: Export format ("hdf5", "parquet", "csv")

        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml4t_backtest_data_{timestamp}.{format}"
            output_path = self.config.data_cache_dir / filename

        # Get comprehensive data
        price_data = self.get_price_data(symbols, start_date, end_date)
        fundamental_data = self.get_fundamental_data(symbols, start_date, end_date)

        if format.lower() == "hdf5":
            with pd.HDFStore(output_path, mode='w') as store:
                if not price_data.empty:
                    store['price_data'] = price_data
                if not fundamental_data.empty:
                    store['fundamental_data'] = fundamental_data

                # Add metadata
                metadata = {
                    'symbols': symbols,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'export_time': datetime.now().isoformat(),
                    'data_quality_score': self.calculate_data_quality_score(symbols, start_date, end_date)
                }
                store.get_storer('price_data').attrs.metadata = metadata

        elif format.lower() == "parquet":
            # Save each dataset separately for Parquet
            if not price_data.empty:
                price_path = output_path.with_suffix('.price.parquet')
                price_data.to_parquet(price_path)
            if not fundamental_data.empty:
                fund_path = output_path.with_suffix('.fundamental.parquet')
                fundamental_data.to_parquet(fund_path)

        elif format.lower() == "csv":
            # Save as CSV with separate files
            if not price_data.empty:
                price_path = output_path.with_suffix('.price.csv')
                price_data.to_csv(price_path)
            if not fundamental_data.empty:
                fund_path = output_path.with_suffix('.fundamental.csv')
                fundamental_data.to_csv(fund_path)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported ML4T backtest data to {output_path}")
        return output_path

    def calculate_data_quality_score(self,
                                   symbols: List[str],
                                   start_date: date,
                                   end_date: date) -> float:
        """Calculate overall data quality score for the dataset."""
        total_score = 0.0
        symbol_count = 0

        for symbol in symbols:
            try:
                # Get quality metrics from FinLab connector
                quality_metrics = self.finlab_connector.get_data_quality_metrics(
                    symbol,
                    lookback_days=(end_date - start_date).days
                )

                symbol_score = quality_metrics.get('overall_quality_score', 0.0)
                total_score += symbol_score
                symbol_count += 1

            except Exception as e:
                logger.warning(f"Failed to calculate quality score for {symbol}: {e}")
                self.data_quality_issues += 1

        return total_score / max(symbol_count, 1)

    def validate_backtest_readiness(self,
                                   symbols: List[str],
                                   start_date: date,
                                   end_date: date) -> Dict[str, Any]:
        """
        Validate that data is ready for ML4T-Alpha backtesting.

        Returns validation report with readiness status and any issues.
        """
        report = {
            'ready': True,
            'issues': [],
            'warnings': [],
            'data_quality_score': 0.0,
            'symbol_coverage': {},
            'date_coverage': {},
            'recommendations': []
        }

        try:
            # Check data quality
            quality_score = self.calculate_data_quality_score(symbols, start_date, end_date)
            report['data_quality_score'] = quality_score

            if quality_score < self.config.min_data_quality_score:
                report['ready'] = False
                report['issues'].append(f"Data quality score {quality_score:.1f} below minimum {self.config.min_data_quality_score}")

            # Check symbol coverage
            for symbol in symbols:
                try:
                    price_data = self.get_price_data([symbol], start_date, end_date)
                    coverage_rate = (price_data[symbol].notna().sum() / len(price_data)) if not price_data.empty else 0.0
                    report['symbol_coverage'][symbol] = coverage_rate

                    if coverage_rate < 0.8:  # 80% coverage threshold
                        report['warnings'].append(f"Low data coverage for {symbol}: {coverage_rate:.1%}")

                except Exception as e:
                    report['issues'].append(f"Failed to check coverage for {symbol}: {e}")
                    report['ready'] = False

            # Generate recommendations
            if report['issues']:
                report['recommendations'].append("Resolve data quality issues before backtesting")
            if report['warnings']:
                report['recommendations'].append("Consider filtering symbols with low data coverage")
            if quality_score < 90:
                report['recommendations'].append("Review data validation rules and source reliability")

        except Exception as e:
            report['ready'] = False
            report['issues'].append(f"Validation failed: {e}")

        return report

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get interface performance statistics."""
        stats = {
            'query_count': self.query_count,
            'total_query_time': self.total_query_time,
            'avg_query_time': self.total_query_time / max(self.query_count, 1),
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(self.query_count, 1),
            'data_quality_issues': self.data_quality_issues,
            'connected': self._connected
        }

        # Add FinLab connector stats
        if hasattr(self.finlab_connector, 'get_performance_stats'):
            finlab_stats = self.finlab_connector.get_performance_stats()
            stats.update({f'finlab_{k}': v for k, v in finlab_stats.items()})

        # Add PIT engine stats
        if hasattr(self.pit_engine, 'get_performance_stats'):
            pit_stats = self.pit_engine.get_performance_stats()
            stats.update({f'pit_{k}': v for k, v in pit_stats.items()})

        return stats

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def create_ml4t_interface(
    finlab_config: Optional[FinLabConfig] = None,
    enable_pit: bool = True,
    enable_cache: bool = True,
    max_workers: int = 4,
    data_cache_dir: Optional[Path] = None,
    **kwargs
) -> ML4TDataInterface:
    """
    Factory function to create ML4T-Alpha data interface.

    Args:
        finlab_config: FinLab configuration (auto-created if None)
        enable_pit: Enable point-in-time data access
        enable_cache: Enable data caching
        max_workers: Maximum worker threads
        data_cache_dir: Data cache directory
        **kwargs: Additional configuration parameters

    Returns:
        Configured ML4TDataInterface instance
    """
    if finlab_config is None:
        from ..data.ingestion.finlab_auth import AuthConfig
        auth_config = AuthConfig()  # Will load from environment
        finlab_config = FinLabConfig(auth_config=auth_config)

    config = ML4TDataConfig(
        finlab_config=finlab_config,
        enable_pit=enable_pit,
        enable_cache=enable_cache,
        max_workers=max_workers,
        data_cache_dir=data_cache_dir,
        **kwargs
    )

    return ML4TDataInterface(config)