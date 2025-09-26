"""
Tests for the Parquet storage backend.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

from src.finlab_downloader.storage.parquet_backend import (
    ParquetStorageBackend, PartitionConfig, StorageMetadata
)
from src.finlab_downloader.storage.base import PartitionKey, CompressionType


@pytest.fixture
def temp_storage_path():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def parquet_backend(temp_storage_path):
    """Create Parquet storage backend instance."""
    return ParquetStorageBackend(
        base_path=temp_storage_path,
        compression=CompressionType.SNAPPY
    )


@pytest.fixture
def sample_stock_data():
    """Create sample stock price data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'date': dates,
        'symbol': ['AAPL'] * 100,
        'open': 150 + np.random.randn(100) * 5,
        'high': 152 + np.random.randn(100) * 5,
        'low': 148 + np.random.randn(100) * 5,
        'close': 151 + np.random.randn(100) * 5,
        'volume': np.random.randint(1000000, 10000000, 100)
    })

    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data.set_index('date')


def test_storage_backend_initialization(temp_storage_path):
    """Test storage backend initialization."""
    backend = ParquetStorageBackend(temp_storage_path)

    assert backend.base_path == temp_storage_path
    assert backend.compression == CompressionType.SNAPPY
    assert temp_storage_path.exists()
    assert (temp_storage_path / "_metadata").exists()


def test_store_and_load_data(parquet_backend, sample_stock_data):
    """Test storing and loading data."""
    partition_key = PartitionKey(
        dataset="stock_prices",
        year=2024,
        quarter=1,
        symbol="AAPL"
    )

    # Store data
    file_path = parquet_backend.store_data(
        data=sample_stock_data,
        partition_key=partition_key,
        metadata={"test": "data"},
        overwrite=True
    )

    assert file_path
    assert Path(file_path).exists()

    # Load data
    loaded_data = parquet_backend.load_data(partition_key)

    assert len(loaded_data) == len(sample_stock_data)
    assert list(loaded_data.columns) == list(sample_stock_data.columns)

    # Check data integrity
    pd.testing.assert_frame_equal(
        loaded_data.sort_index(),
        sample_stock_data.sort_index(),
        check_dtype=False
    )


def test_partition_key_validation(parquet_backend, sample_stock_data):
    """Test partition key validation."""
    # Valid partition key
    valid_key = PartitionKey(dataset="test", year=2024, quarter=1)
    parquet_backend.validate_partition_key(valid_key)

    # Invalid year
    invalid_year_key = PartitionKey(dataset="test", year=1800, quarter=1)
    with pytest.raises(Exception):
        parquet_backend.validate_partition_key(invalid_year_key)

    # Invalid quarter
    invalid_quarter_key = PartitionKey(dataset="test", year=2024, quarter=5)
    with pytest.raises(Exception):
        parquet_backend.validate_partition_key(invalid_quarter_key)


def test_column_pruning(parquet_backend, sample_stock_data):
    """Test column pruning functionality."""
    partition_key = PartitionKey(dataset="test", year=2024, quarter=1)

    # Store data
    parquet_backend.store_data(sample_stock_data, partition_key)

    # Load only specific columns
    selected_columns = ['open', 'close', 'volume']
    loaded_data = parquet_backend.load_data(
        partition_key,
        columns=selected_columns
    )

    assert list(loaded_data.columns) == selected_columns
    assert len(loaded_data) == len(sample_stock_data)


def test_date_filtering(parquet_backend, sample_stock_data):
    """Test date-based filtering."""
    partition_key = PartitionKey(dataset="test", year=2024, quarter=1)

    # Store data
    parquet_backend.store_data(sample_stock_data, partition_key)

    # Load with date filter
    start_date = date(2024, 1, 15)
    end_date = date(2024, 1, 25)

    filtered_data = parquet_backend.load_data(
        partition_key,
        start_date=start_date,
        end_date=end_date
    )

    # Check date range
    data_dates = pd.to_datetime(filtered_data.index).date
    assert all(start_date <= d <= end_date for d in data_dates)


def test_list_partitions(parquet_backend, sample_stock_data):
    """Test listing partitions."""
    # Store data in multiple partitions
    partitions = [
        PartitionKey(dataset="stocks", year=2024, quarter=1),
        PartitionKey(dataset="stocks", year=2024, quarter=2),
        PartitionKey(dataset="bonds", year=2024, quarter=1)
    ]

    for partition in partitions:
        parquet_backend.store_data(sample_stock_data.head(10), partition)

    # List all partitions for stocks dataset
    stock_partitions = parquet_backend.list_partitions("stocks")
    assert len(stock_partitions) == 2

    # List partitions for bonds dataset
    bond_partitions = parquet_backend.list_partitions("bonds")
    assert len(bond_partitions) == 1


def test_delete_partition(parquet_backend, sample_stock_data):
    """Test partition deletion."""
    partition_key = PartitionKey(dataset="test", year=2024, quarter=1)

    # Store data
    parquet_backend.store_data(sample_stock_data, partition_key)

    # Verify partition exists
    assert parquet_backend.partition_exists(partition_key)

    # Delete partition
    deleted = parquet_backend.delete_partition(partition_key)
    assert deleted

    # Verify partition no longer exists
    assert not parquet_backend.partition_exists(partition_key)


def test_storage_statistics(parquet_backend, sample_stock_data):
    """Test storage statistics collection."""
    partition_key = PartitionKey(dataset="test", year=2024, quarter=1)

    # Store data
    parquet_backend.store_data(sample_stock_data, partition_key)

    # Get statistics
    stats = parquet_backend.get_storage_stats("test")

    assert stats.total_files >= 1
    assert stats.total_size_bytes > 0
    assert stats.partitions_count >= 1
    assert 0 < stats.compression_ratio <= 1


def test_optimization(parquet_backend, sample_stock_data):
    """Test partition optimization."""
    partition_key = PartitionKey(dataset="test", year=2024, quarter=1)

    # Store multiple small files
    chunk_size = len(sample_stock_data) // 3
    for i in range(3):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < 2 else len(sample_stock_data)
        chunk_data = sample_stock_data.iloc[start_idx:end_idx]
        parquet_backend.store_data(chunk_data, partition_key)

    # Optimize partition
    optimized = parquet_backend.optimize_partition(partition_key)
    assert optimized

    # Verify data integrity after optimization
    loaded_data = parquet_backend.load_data(partition_key)
    assert len(loaded_data) > 0


def test_compression_types(temp_storage_path, sample_stock_data):
    """Test different compression types."""
    compression_types = [
        CompressionType.NONE,
        CompressionType.SNAPPY,
        CompressionType.GZIP
    ]

    for compression in compression_types:
        backend = ParquetStorageBackend(
            base_path=temp_storage_path / compression.value,
            compression=compression
        )

        partition_key = PartitionKey(dataset="test", year=2024, quarter=1)

        # Store and load data
        backend.store_data(sample_stock_data, partition_key)
        loaded_data = backend.load_data(partition_key)

        assert len(loaded_data) == len(sample_stock_data)


def test_error_handling(parquet_backend):
    """Test error handling."""
    # Test loading non-existent partition
    non_existent_key = PartitionKey(dataset="nonexistent", year=2024, quarter=1)
    loaded_data = parquet_backend.load_data(non_existent_key)
    assert loaded_data.empty

    # Test storing empty data
    empty_data = pd.DataFrame()
    partition_key = PartitionKey(dataset="test", year=2024, quarter=1)
    file_path = parquet_backend.store_data(empty_data, partition_key)
    assert file_path == ""


def test_schema_registry(parquet_backend, sample_stock_data):
    """Test schema registry functionality."""
    partition_key = PartitionKey(dataset="test_schema", year=2024, quarter=1)

    # Store data
    parquet_backend.store_data(sample_stock_data, partition_key)

    # Get schema
    schema = parquet_backend.get_dataset_schema("test_schema")
    assert schema is not None
    assert 'fields' in schema
    assert len(schema['fields']) == len(sample_stock_data.columns)


def test_metadata_storage(parquet_backend, sample_stock_data):
    """Test metadata storage and retrieval."""
    partition_key = PartitionKey(dataset="test_metadata", year=2024, quarter=1)
    custom_metadata = {
        "source": "test",
        "quality_score": 0.95,
        "validation_passed": True
    }

    # Store data with metadata
    parquet_backend.store_data(
        sample_stock_data,
        partition_key,
        metadata=custom_metadata
    )

    # Verify metadata files are created
    metadata_path = parquet_backend.metadata_path
    metadata_files = list(metadata_path.glob("*metadata.json"))
    assert len(metadata_files) > 0