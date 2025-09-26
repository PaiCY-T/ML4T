"""
Storage framework for FinLab data with Parquet backend.

This module provides high-performance, columnar storage for financial data
using Parquet format with pyarrow, optimized for quantitative trading workflows.
"""

from .parquet_backend import ParquetStorageBackend, PartitionConfig, StorageMetadata
from .base import BaseStorageBackend, StorageError, PartitionKey
from .metadata_manager import MetadataManager, TableMetadata
from .query_engine import QueryEngine, TimeSeriesQuery

__all__ = [
    'ParquetStorageBackend',
    'BaseStorageBackend',
    'PartitionConfig',
    'StorageMetadata',
    'StorageError',
    'PartitionKey',
    'MetadataManager',
    'TableMetadata',
    'QueryEngine',
    'TimeSeriesQuery'
]