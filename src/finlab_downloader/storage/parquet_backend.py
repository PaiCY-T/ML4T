"""
Parquet storage backend with pyarrow for high-performance financial data storage.

Optimized for quantitative trading workflows with columnar storage,
efficient compression, and fast query performance.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Iterator
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs

from .base import (
    BaseStorageBackend, StorageError, PartitionError, MetadataError,
    PartitionKey, StorageStats, CompressionType
)

logger = logging.getLogger(__name__)


@dataclass
class PartitionConfig:
    """Configuration for partitioning strategy."""
    by_year: bool = True
    by_quarter: bool = True
    by_month: bool = False
    by_symbol: bool = False
    max_rows_per_file: int = 1_000_000
    min_rows_per_file: int = 1000


@dataclass
class StorageMetadata:
    """Metadata for stored data."""
    dataset: str
    partition_key: PartitionKey
    created_at: datetime
    updated_at: datetime
    schema_version: str
    row_count: int
    file_size_bytes: int
    compression_type: str
    custom_metadata: Dict[str, Any]


class ParquetStorageBackend(BaseStorageBackend):
    """
    High-performance Parquet storage backend using pyarrow.

    Features:
    - Columnar storage with efficient compression (60-80% size reduction)
    - Date-based partitioning for incremental updates
    - Schema evolution and validation
    - Predicate pushdown for fast queries
    - Metadata management for time-series operations
    """

    def __init__(self,
                 base_path: Union[str, Path],
                 compression: CompressionType = CompressionType.SNAPPY,
                 partition_config: Optional[PartitionConfig] = None):
        """
        Initialize Parquet storage backend.

        Args:
            base_path: Base directory for data storage
            compression: Compression algorithm
            partition_config: Partitioning configuration
        """
        super().__init__(base_path, compression)

        self.partition_config = partition_config or PartitionConfig()
        self.filesystem = fs.LocalFileSystem()

        # Create metadata directory
        self.metadata_path = self.base_path / "_metadata"
        self.metadata_path.mkdir(exist_ok=True)

        # Initialize schema registry
        self.schema_registry = {}
        self._load_schema_registry()

    def store_data(self,
                   data: pd.DataFrame,
                   partition_key: PartitionKey,
                   metadata: Optional[Dict[str, Any]] = None,
                   overwrite: bool = False) -> str:
        """
        Store DataFrame as Parquet with efficient partitioning.

        Args:
            data: DataFrame to store
            partition_key: Partition information
            metadata: Optional metadata
            overwrite: Whether to overwrite existing data

        Returns:
            Path to stored file
        """
        try:
            self.validate_partition_key(partition_key)

            if data.empty:
                logger.warning(f"Empty DataFrame for partition {partition_key}")
                return ""

            # Get partition directory
            partition_path = self.get_partition_path(partition_key)
            partition_path.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"data_{timestamp}.parquet"
            file_path = partition_path / filename

            # Check if file exists and handle overwrite
            if file_path.exists() and not overwrite:
                raise StorageError(f"File already exists: {file_path}")

            # Validate and optimize data
            optimized_data = self._optimize_dataframe(data, partition_key.dataset)

            # Convert to pyarrow table
            table = pa.Table.from_pandas(optimized_data)

            # Determine compression
            compression_str = self.compression.value if self.compression != CompressionType.NONE else None

            # Write parquet file
            pq.write_table(
                table,
                file_path,
                compression=compression_str,
                use_dictionary=True,  # Enable dictionary encoding for string columns
                write_statistics=True,  # Write column statistics for predicate pushdown
                coerce_timestamps='ms'  # Millisecond precision for timestamps
            )

            # Store metadata
            storage_metadata = StorageMetadata(
                dataset=partition_key.dataset,
                partition_key=partition_key,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                schema_version="1.0",
                row_count=len(optimized_data),
                file_size_bytes=file_path.stat().st_size,
                compression_type=compression_str or "none",
                custom_metadata=metadata or {}
            )

            self._store_metadata(storage_metadata, file_path)

            # Update schema registry
            self._update_schema_registry(partition_key.dataset, table.schema)

            logger.info(
                f"Stored {len(optimized_data):,} rows to {file_path} "
                f"({file_path.stat().st_size:,} bytes)"
            )

            return str(file_path)

        except Exception as e:
            logger.error(f"Error storing data for partition {partition_key}: {e}")
            raise StorageError(f"Failed to store data: {e}") from e

    def load_data(self,
                  partition_key: PartitionKey,
                  columns: Optional[List[str]] = None,
                  start_date: Optional[date] = None,
                  end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Load data from Parquet files with optional filtering.

        Args:
            partition_key: Partition to load
            columns: Specific columns to load (column pruning)
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            Loaded DataFrame
        """
        try:
            partition_path = self.get_partition_path(partition_key)

            if not partition_path.exists():
                logger.warning(f"Partition path does not exist: {partition_path}")
                return pd.DataFrame()

            # Find all parquet files in partition
            parquet_files = list(partition_path.glob("*.parquet"))

            if not parquet_files:
                logger.warning(f"No parquet files found in {partition_path}")
                return pd.DataFrame()

            # Load data from all files
            dataframes = []
            for file_path in parquet_files:
                try:
                    # Use pyarrow for efficient loading
                    table = pq.read_table(
                        file_path,
                        columns=columns,
                        use_pandas_metadata=True
                    )

                    df = table.to_pandas()

                    # Apply date filtering if specified
                    if start_date or end_date:
                        df = self._filter_by_date(df, start_date, end_date)

                    if not df.empty:
                        dataframes.append(df)

                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
                    continue

            if not dataframes:
                return pd.DataFrame()

            # Combine all dataframes
            combined_data = pd.concat(dataframes, ignore_index=True)

            # Sort by index if it's a datetime
            if isinstance(combined_data.index, pd.DatetimeIndex):
                combined_data = combined_data.sort_index()

            logger.info(f"Loaded {len(combined_data):,} rows from partition {partition_key}")
            return combined_data

        except Exception as e:
            logger.error(f"Error loading data for partition {partition_key}: {e}")
            raise StorageError(f"Failed to load data: {e}") from e

    def delete_partition(self, partition_key: PartitionKey) -> bool:
        """
        Delete a partition and its metadata.

        Args:
            partition_key: Partition to delete

        Returns:
            True if deleted successfully
        """
        try:
            partition_path = self.get_partition_path(partition_key)

            if not partition_path.exists():
                logger.warning(f"Partition does not exist: {partition_path}")
                return False

            # Delete all files in partition
            for file_path in partition_path.rglob("*"):
                if file_path.is_file():
                    file_path.unlink()

            # Remove directory
            partition_path.rmdir()

            # Delete metadata
            self._delete_metadata(partition_key)

            logger.info(f"Deleted partition: {partition_key}")
            return True

        except Exception as e:
            logger.error(f"Error deleting partition {partition_key}: {e}")
            return False

    def list_partitions(self,
                       dataset: str,
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> List[PartitionKey]:
        """
        List available partitions for a dataset.

        Args:
            dataset: Dataset name
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            List of available partition keys
        """
        partitions = []

        dataset_path = self.base_path / f"dataset={dataset}"
        if not dataset_path.exists():
            return partitions

        try:
            # Walk through directory structure
            for year_dir in dataset_path.iterdir():
                if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                    continue

                year = int(year_dir.name.split("=")[1])

                # Apply date filtering
                if start_date and year < start_date.year:
                    continue
                if end_date and year > end_date.year:
                    continue

                for quarter_dir in year_dir.iterdir():
                    if not quarter_dir.is_dir() or not quarter_dir.name.startswith("quarter="):
                        continue

                    quarter = int(quarter_dir.name.split("=")[1][1])  # Remove 'Q' prefix

                    # Check if partition has data
                    parquet_files = list(quarter_dir.rglob("*.parquet"))
                    if parquet_files:
                        partition_key = PartitionKey(
                            dataset=dataset,
                            year=year,
                            quarter=quarter
                        )
                        partitions.append(partition_key)

        except Exception as e:
            logger.error(f"Error listing partitions for dataset {dataset}: {e}")

        return sorted(partitions, key=lambda p: (p.year, p.quarter or 0))

    def get_storage_stats(self, dataset: Optional[str] = None) -> StorageStats:
        """
        Get storage statistics.

        Args:
            dataset: Specific dataset or None for all

        Returns:
            Storage statistics
        """
        try:
            total_size = 0
            total_files = 0
            partitions_count = 0
            created_times = []

            search_path = self.base_path
            if dataset:
                search_path = search_path / f"dataset={dataset}"

            if search_path.exists():
                for file_path in search_path.rglob("*.parquet"):
                    stat = file_path.stat()
                    total_size += stat.st_size
                    total_files += 1
                    created_times.append(datetime.fromtimestamp(stat.st_ctime))

                # Count partitions
                if dataset:
                    partitions_count = len(self.list_partitions(dataset))
                else:
                    # Count all partitions across datasets
                    for dataset_dir in self.base_path.iterdir():
                        if dataset_dir.is_dir() and dataset_dir.name.startswith("dataset="):
                            dataset_name = dataset_dir.name.split("=")[1]
                            partitions_count += len(self.list_partitions(dataset_name))

            # Calculate compression ratio (estimate)
            compression_ratio = 0.3 if self.compression != CompressionType.NONE else 1.0

            return StorageStats(
                total_size_bytes=total_size,
                total_files=total_files,
                partitions_count=partitions_count,
                compression_ratio=compression_ratio,
                last_accessed=datetime.utcnow(),
                created_at=min(created_times) if created_times else datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            raise StorageError(f"Failed to get storage stats: {e}") from e

    def _optimize_dataframe(self, data: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """
        Optimize DataFrame for storage.

        Args:
            data: DataFrame to optimize
            dataset: Dataset name for schema validation

        Returns:
            Optimized DataFrame
        """
        optimized = data.copy()

        # Ensure datetime index if applicable
        if 'date' in optimized.columns and optimized.index.name != 'date':
            optimized = optimized.set_index('date')

        # Convert datetime columns
        for col in optimized.columns:
            if optimized[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    optimized[col] = pd.to_datetime(optimized[col])
                except (ValueError, TypeError):
                    pass

        # Optimize numeric types
        for col in optimized.select_dtypes(include=['int64']).columns:
            if optimized[col].max() < 2**31 and optimized[col].min() > -2**31:
                optimized[col] = optimized[col].astype('int32')

        for col in optimized.select_dtypes(include=['float64']).columns:
            optimized[col] = optimized[col].astype('float32')

        # Optimize string columns
        for col in optimized.select_dtypes(include=['object']).columns:
            if optimized[col].nunique() / len(optimized) < 0.5:  # High cardinality
                optimized[col] = optimized[col].astype('category')

        return optimized

    def _filter_by_date(self, df: pd.DataFrame, start_date: Optional[date], end_date: Optional[date]) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if start_date is None and end_date is None:
            return df

        # Find date column
        date_col = None
        if isinstance(df.index, pd.DatetimeIndex):
            date_series = df.index
        else:
            # Look for date columns
            date_columns = [col for col in df.columns if 'date' in col.lower()]
            if date_columns:
                date_col = date_columns[0]
                date_series = pd.to_datetime(df[date_col])
            else:
                logger.warning("No date column found for filtering")
                return df

        # Apply filtering
        mask = pd.Series(True, index=df.index)
        if start_date:
            mask &= date_series >= pd.Timestamp(start_date)
        if end_date:
            mask &= date_series <= pd.Timestamp(end_date)

        return df[mask]

    def _store_metadata(self, metadata: StorageMetadata, file_path: Path) -> None:
        """Store metadata for a file."""
        try:
            metadata_file = self.metadata_path / f"{file_path.stem}_metadata.json"
            metadata_dict = {
                'dataset': metadata.dataset,
                'partition_key': {
                    'dataset': metadata.partition_key.dataset,
                    'year': metadata.partition_key.year,
                    'quarter': metadata.partition_key.quarter,
                    'month': metadata.partition_key.month,
                    'symbol': metadata.partition_key.symbol
                },
                'created_at': metadata.created_at.isoformat(),
                'updated_at': metadata.updated_at.isoformat(),
                'schema_version': metadata.schema_version,
                'row_count': metadata.row_count,
                'file_size_bytes': metadata.file_size_bytes,
                'compression_type': metadata.compression_type,
                'custom_metadata': metadata.custom_metadata,
                'file_path': str(file_path)
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Error storing metadata: {e}")

    def _delete_metadata(self, partition_key: PartitionKey) -> None:
        """Delete metadata for a partition."""
        try:
            # Find and delete all metadata files for this partition
            pattern = f"*{partition_key.dataset}*{partition_key.year}*"
            for metadata_file in self.metadata_path.glob(pattern):
                metadata_file.unlink()
        except Exception as e:
            logger.error(f"Error deleting metadata: {e}")

    def _load_schema_registry(self) -> None:
        """Load schema registry from storage."""
        schema_file = self.metadata_path / "schema_registry.json"
        if schema_file.exists():
            try:
                with open(schema_file, 'r') as f:
                    self.schema_registry = json.load(f)
            except Exception as e:
                logger.error(f"Error loading schema registry: {e}")
                self.schema_registry = {}

    def _update_schema_registry(self, dataset: str, schema: pa.Schema) -> None:
        """Update schema registry with new schema."""
        try:
            schema_dict = {
                'fields': [
                    {
                        'name': field.name,
                        'type': str(field.type),
                        'nullable': field.nullable
                    }
                    for field in schema
                ],
                'updated_at': datetime.utcnow().isoformat()
            }

            self.schema_registry[dataset] = schema_dict

            # Save to file
            schema_file = self.metadata_path / "schema_registry.json"
            with open(schema_file, 'w') as f:
                json.dump(self.schema_registry, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating schema registry: {e}")

    def get_dataset_schema(self, dataset: str) -> Optional[Dict[str, Any]]:
        """Get schema for a dataset."""
        return self.schema_registry.get(dataset)

    def optimize_partition(self, partition_key: PartitionKey) -> bool:
        """
        Optimize a partition by consolidating small files.

        Args:
            partition_key: Partition to optimize

        Returns:
            True if optimization was performed
        """
        try:
            partition_path = self.get_partition_path(partition_key)
            parquet_files = list(partition_path.glob("*.parquet"))

            if len(parquet_files) <= 1:
                return False  # Nothing to optimize

            # Load all data
            data = self.load_data(partition_key)
            if data.empty:
                return False

            # Remove old files
            for file_path in parquet_files:
                file_path.unlink()

            # Store consolidated data
            self.store_data(data, partition_key, overwrite=True)

            logger.info(f"Optimized partition {partition_key}: {len(parquet_files)} files -> 1 file")
            return True

        except Exception as e:
            logger.error(f"Error optimizing partition {partition_key}: {e}")
            return False