"""
Base storage backend interface for FinLab data storage.

Defines the abstract interface for storage backends with common functionality
and error handling.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Iterator
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class PartitionError(StorageError):
    """Exception for partition-related operations."""
    pass


class MetadataError(StorageError):
    """Exception for metadata operations."""
    pass


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    SNAPPY = "snappy"
    GZIP = "gzip"
    LZ4 = "lz4"
    BROTLI = "brotli"


@dataclass
class PartitionKey:
    """Represents a partition key for data organization."""
    dataset: str
    year: int
    quarter: Optional[int] = None
    month: Optional[int] = None
    symbol: Optional[str] = None

    def to_path(self) -> str:
        """Convert to file path."""
        path_parts = [f"dataset={self.dataset}", f"year={self.year}"]

        if self.quarter is not None:
            path_parts.append(f"quarter=Q{self.quarter}")

        if self.month is not None:
            path_parts.append(f"month={self.month:02d}")

        if self.symbol is not None:
            path_parts.append(f"symbol={self.symbol}")

        return "/".join(path_parts)

    @classmethod
    def from_date(cls, dataset: str, target_date: Union[date, datetime],
                  symbol: Optional[str] = None) -> 'PartitionKey':
        """Create partition key from date."""
        if isinstance(target_date, datetime):
            target_date = target_date.date()

        quarter = (target_date.month - 1) // 3 + 1

        return cls(
            dataset=dataset,
            year=target_date.year,
            quarter=quarter,
            month=target_date.month,
            symbol=symbol
        )


@dataclass
class StorageStats:
    """Storage statistics and metrics."""
    total_size_bytes: int
    total_files: int
    partitions_count: int
    compression_ratio: float
    last_accessed: datetime
    created_at: datetime


class BaseStorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Defines the interface that all storage backends must implement
    for consistent data storage and retrieval operations.
    """

    def __init__(self,
                 base_path: Union[str, Path],
                 compression: CompressionType = CompressionType.SNAPPY):
        """
        Initialize storage backend.

        Args:
            base_path: Base directory for data storage
            compression: Default compression type
        """
        self.base_path = Path(base_path)
        self.compression = compression
        self.base_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def store_data(self,
                   data: pd.DataFrame,
                   partition_key: PartitionKey,
                   metadata: Optional[Dict[str, Any]] = None,
                   overwrite: bool = False) -> str:
        """
        Store data to the backend.

        Args:
            data: DataFrame to store
            partition_key: Partition information
            metadata: Optional metadata
            overwrite: Whether to overwrite existing data

        Returns:
            File path or identifier of stored data
        """
        pass

    @abstractmethod
    def load_data(self,
                  partition_key: PartitionKey,
                  columns: Optional[List[str]] = None,
                  start_date: Optional[date] = None,
                  end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Load data from the backend.

        Args:
            partition_key: Partition to load
            columns: Specific columns to load
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            Loaded DataFrame
        """
        pass

    @abstractmethod
    def delete_partition(self, partition_key: PartitionKey) -> bool:
        """
        Delete a partition.

        Args:
            partition_key: Partition to delete

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    def list_partitions(self,
                       dataset: str,
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> List[PartitionKey]:
        """
        List available partitions.

        Args:
            dataset: Dataset name
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            List of available partition keys
        """
        pass

    @abstractmethod
    def get_storage_stats(self, dataset: Optional[str] = None) -> StorageStats:
        """
        Get storage statistics.

        Args:
            dataset: Specific dataset or None for all

        Returns:
            Storage statistics
        """
        pass

    def partition_exists(self, partition_key: PartitionKey) -> bool:
        """
        Check if partition exists.

        Args:
            partition_key: Partition to check

        Returns:
            True if partition exists
        """
        try:
            partitions = self.list_partitions(partition_key.dataset)
            return partition_key in partitions
        except Exception:
            return False

    def get_partition_path(self, partition_key: PartitionKey) -> Path:
        """
        Get the file system path for a partition.

        Args:
            partition_key: Partition key

        Returns:
            Path object for the partition
        """
        return self.base_path / partition_key.to_path()

    def cleanup_old_partitions(self,
                              dataset: str,
                              days_to_keep: int = 365) -> int:
        """
        Clean up old partitions.

        Args:
            dataset: Dataset to clean
            days_to_keep: Number of days to keep

        Returns:
            Number of partitions deleted
        """
        cutoff_date = date.today() - pd.Timedelta(days=days_to_keep)
        partitions = self.list_partitions(dataset)

        deleted_count = 0
        for partition in partitions:
            partition_date = date(partition.year, partition.month or 1, 1)
            if partition_date < cutoff_date:
                try:
                    if self.delete_partition(partition):
                        deleted_count += 1
                        logger.info(f"Deleted old partition: {partition}")
                except Exception as e:
                    logger.error(f"Failed to delete partition {partition}: {e}")

        return deleted_count

    def validate_partition_key(self, partition_key: PartitionKey) -> None:
        """
        Validate partition key.

        Args:
            partition_key: Key to validate

        Raises:
            PartitionError: If partition key is invalid
        """
        if not partition_key.dataset:
            raise PartitionError("Dataset name is required")

        if partition_key.year < 1900 or partition_key.year > 2100:
            raise PartitionError(f"Invalid year: {partition_key.year}")

        if partition_key.quarter is not None:
            if not (1 <= partition_key.quarter <= 4):
                raise PartitionError(f"Invalid quarter: {partition_key.quarter}")

        if partition_key.month is not None:
            if not (1 <= partition_key.month <= 12):
                raise PartitionError(f"Invalid month: {partition_key.month}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass