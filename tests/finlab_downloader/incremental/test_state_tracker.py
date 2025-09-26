"""
Tests for the state tracking system.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, date
from pathlib import Path

from src.finlab_downloader.incremental.state_tracker import (
    StateTracker, DataState, DownloadCheckpoint
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_state.db"

    yield db_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def state_tracker(temp_db):
    """Create a state tracker instance for testing."""
    return StateTracker(temp_db)


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint for testing."""
    return DownloadCheckpoint(
        dataset_name="test_dataset",
        symbol="2330",
        last_download_date=date.today(),
        last_modification_time=datetime.utcnow(),
        data_hash="abc123",
        version=1,
        state=DataState.COMPLETED,
        metadata={"test": "data"},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


def test_state_tracker_initialization(state_tracker, temp_db):
    """Test state tracker initialization."""
    assert state_tracker.database_path == temp_db
    assert temp_db.parent.exists()

    # Check that database file is created
    stats = state_tracker.get_statistics()
    assert stats['total_checkpoints'] == 0


def test_save_and_get_checkpoint(state_tracker, sample_checkpoint):
    """Test saving and retrieving checkpoints."""
    # Save checkpoint
    state_tracker.save_checkpoint(sample_checkpoint)

    # Retrieve checkpoint
    retrieved = state_tracker.get_checkpoint(
        sample_checkpoint.dataset_name,
        sample_checkpoint.symbol
    )

    assert retrieved is not None
    assert retrieved.dataset_name == sample_checkpoint.dataset_name
    assert retrieved.symbol == sample_checkpoint.symbol
    assert retrieved.data_hash == sample_checkpoint.data_hash
    assert retrieved.state == sample_checkpoint.state


def test_checkpoint_update(state_tracker, sample_checkpoint):
    """Test updating existing checkpoints."""
    # Save initial checkpoint
    state_tracker.save_checkpoint(sample_checkpoint)

    # Update checkpoint
    sample_checkpoint.data_hash = "xyz789"
    sample_checkpoint.version = 2
    state_tracker.save_checkpoint(sample_checkpoint)

    # Verify update
    retrieved = state_tracker.get_checkpoint(
        sample_checkpoint.dataset_name,
        sample_checkpoint.symbol
    )

    assert retrieved.data_hash == "xyz789"
    assert retrieved.version == 2


def test_record_download_attempt(state_tracker):
    """Test recording download attempts."""
    state_tracker.record_download_attempt(
        dataset_name="test_dataset",
        symbol="2330",
        download_date=date.today(),
        modification_time=datetime.utcnow(),
        data_hash="abc123",
        version=1,
        operation_type="download",
        success=True
    )

    history = state_tracker.get_download_history("test_dataset", "2330")
    assert len(history) == 1
    assert history[0]['success'] == True
    assert history[0]['operation_type'] == "download"


def test_data_hash_calculation(state_tracker):
    """Test data hash calculation."""
    test_data = {"key": "value", "number": 123}
    hash1 = state_tracker.calculate_data_hash(test_data)
    hash2 = state_tracker.calculate_data_hash(test_data)

    # Same data should produce same hash
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 produces 64-character hex string

    # Different data should produce different hash
    different_data = {"key": "different", "number": 456}
    hash3 = state_tracker.calculate_data_hash(different_data)
    assert hash1 != hash3


def test_get_checkpoints_by_state(state_tracker):
    """Test filtering checkpoints by state."""
    # Create checkpoints with different states
    checkpoint1 = DownloadCheckpoint(
        dataset_name="dataset1",
        symbol="2330",
        last_download_date=date.today(),
        last_modification_time=datetime.utcnow(),
        data_hash="hash1",
        version=1,
        state=DataState.COMPLETED,
        metadata={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    checkpoint2 = DownloadCheckpoint(
        dataset_name="dataset2",
        symbol="2317",
        last_download_date=date.today(),
        last_modification_time=datetime.utcnow(),
        data_hash="hash2",
        version=1,
        state=DataState.FAILED,
        metadata={},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    state_tracker.save_checkpoint(checkpoint1)
    state_tracker.save_checkpoint(checkpoint2)

    # Test filtering
    completed = state_tracker.get_checkpoints_by_state(DataState.COMPLETED)
    failed = state_tracker.get_checkpoints_by_state(DataState.FAILED)

    assert len(completed) == 1
    assert len(failed) == 1
    assert completed[0].dataset_name == "dataset1"
    assert failed[0].dataset_name == "dataset2"


def test_cleanup_old_history(state_tracker):
    """Test cleanup of old download history."""
    # Record some download attempts
    for i in range(10):
        state_tracker.record_download_attempt(
            dataset_name="test_dataset",
            symbol=f"symbol_{i}",
            download_date=date.today(),
            modification_time=datetime.utcnow(),
            data_hash=f"hash_{i}",
            version=1,
            operation_type="download",
            success=True
        )

    # Verify records exist
    history = state_tracker.get_download_history("test_dataset", limit=20)
    assert len(history) >= 10

    # Cleanup (using 0 days to clean everything)
    deleted_count = state_tracker.cleanup_old_history(days_to_keep=0)
    assert deleted_count > 0


def test_statistics(state_tracker, sample_checkpoint):
    """Test statistics collection."""
    # Save a checkpoint
    state_tracker.save_checkpoint(sample_checkpoint)

    # Record a download attempt
    state_tracker.record_download_attempt(
        dataset_name="test_dataset",
        symbol="2330",
        download_date=date.today(),
        modification_time=datetime.utcnow(),
        data_hash="abc123",
        version=1,
        operation_type="download",
        success=True
    )

    stats = state_tracker.get_statistics()

    assert stats['total_checkpoints'] == 1
    assert stats['total_history_records'] == 1
    assert 'database_size_bytes' in stats
    assert 'checkpoints_by_state' in stats