"""
SQLite-based state tracking system for incremental downloads.

This module implements a robust state tracking database that maintains information
about download history, data versions, timestamps, and checksums for integrity verification.
"""

import sqlite3
import json
import logging
import hashlib
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class DataState(Enum):
    """State of data in the tracking system."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    OUTDATED = "outdated"


@dataclass
class DownloadCheckpoint:
    """Represents a checkpoint for tracking download progress."""
    dataset_name: str
    symbol: Optional[str]
    last_download_date: date
    last_modification_time: datetime
    data_hash: str
    version: int
    state: DataState
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for storage."""
        data = asdict(self)
        # Convert dates and enums to strings for JSON serialization
        data['last_download_date'] = self.last_download_date.isoformat()
        data['last_modification_time'] = self.last_modification_time.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['state'] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DownloadCheckpoint':
        """Create checkpoint from dictionary."""
        return cls(
            dataset_name=data['dataset_name'],
            symbol=data.get('symbol'),
            last_download_date=date.fromisoformat(data['last_download_date']),
            last_modification_time=datetime.fromisoformat(data['last_modification_time']),
            data_hash=data['data_hash'],
            version=data['version'],
            state=DataState(data['state']),
            metadata=data['metadata'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )


class StateTracker:
    """
    SQLite-based state tracking system for incremental downloads.

    Features:
    - Persistent state storage across application restarts
    - Checksum-based integrity verification
    - Version tracking for data evolution
    - Transaction support for atomic operations
    - Conflict resolution and rollback capabilities
    """

    def __init__(self, database_path: Union[str, Path]):
        """
        Initialize state tracker.

        Args:
            database_path: Path to SQLite database file
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"StateTracker initialized with database: {self.database_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create checkpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    symbol TEXT,
                    last_download_date TEXT NOT NULL,
                    last_modification_time TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    state TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(dataset_name, symbol)
                )
            """)

            # Create download history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS download_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    symbol TEXT,
                    download_date TEXT NOT NULL,
                    modification_time TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    operation_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)

            # Create data versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    symbol TEXT,
                    version INTEGER NOT NULL,
                    data_hash TEXT NOT NULL,
                    modification_time TEXT NOT NULL,
                    changes_summary TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(dataset_name, symbol, version)
                )
            """)

            # Create rollback operations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rollback_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL UNIQUE,
                    dataset_name TEXT NOT NULL,
                    symbol TEXT,
                    from_version INTEGER NOT NULL,
                    to_version INTEGER NOT NULL,
                    operation_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_dataset ON checkpoints(dataset_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_symbol ON checkpoints(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_state ON checkpoints(state)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_dataset ON download_history(dataset_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_date ON download_history(download_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_dataset ON data_versions(dataset_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rollback_status ON rollback_operations(status)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get SQLite connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.database_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_checkpoint(self, dataset_name: str, symbol: Optional[str] = None) -> Optional[DownloadCheckpoint]:
        """
        Get the latest checkpoint for a dataset/symbol combination.

        Args:
            dataset_name: Name of the dataset
            symbol: Optional symbol identifier

        Returns:
            DownloadCheckpoint if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if symbol is not None:
                cursor.execute(
                    "SELECT * FROM checkpoints WHERE dataset_name = ? AND symbol = ?",
                    (dataset_name, symbol)
                )
            else:
                cursor.execute(
                    "SELECT * FROM checkpoints WHERE dataset_name = ? AND symbol IS NULL",
                    (dataset_name,)
                )

            row = cursor.fetchone()
            if row:
                return DownloadCheckpoint.from_dict(dict(row))
            return None

    def save_checkpoint(self, checkpoint: DownloadCheckpoint) -> None:
        """
        Save or update a checkpoint.

        Args:
            checkpoint: Checkpoint to save
        """
        checkpoint.updated_at = datetime.utcnow()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if checkpoint exists
            existing = self.get_checkpoint(checkpoint.dataset_name, checkpoint.symbol)

            if existing:
                # Update existing checkpoint
                cursor.execute("""
                    UPDATE checkpoints SET
                        last_download_date = ?,
                        last_modification_time = ?,
                        data_hash = ?,
                        version = ?,
                        state = ?,
                        metadata = ?,
                        updated_at = ?
                    WHERE dataset_name = ? AND symbol = ?
                """, (
                    checkpoint.last_download_date.isoformat(),
                    checkpoint.last_modification_time.isoformat(),
                    checkpoint.data_hash,
                    checkpoint.version,
                    checkpoint.state.value,
                    json.dumps(checkpoint.metadata),
                    checkpoint.updated_at.isoformat(),
                    checkpoint.dataset_name,
                    checkpoint.symbol
                ))
            else:
                # Insert new checkpoint
                cursor.execute("""
                    INSERT INTO checkpoints (
                        dataset_name, symbol, last_download_date, last_modification_time,
                        data_hash, version, state, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint.dataset_name,
                    checkpoint.symbol,
                    checkpoint.last_download_date.isoformat(),
                    checkpoint.last_modification_time.isoformat(),
                    checkpoint.data_hash,
                    checkpoint.version,
                    checkpoint.state.value,
                    json.dumps(checkpoint.metadata),
                    checkpoint.created_at.isoformat(),
                    checkpoint.updated_at.isoformat()
                ))

            conn.commit()
            logger.debug(f"Saved checkpoint for {checkpoint.dataset_name}:{checkpoint.symbol}")

    def record_download_attempt(self,
                              dataset_name: str,
                              symbol: Optional[str],
                              download_date: date,
                              modification_time: datetime,
                              data_hash: str,
                              version: int,
                              operation_type: str,
                              success: bool,
                              error_message: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a download attempt in the history.

        Args:
            dataset_name: Name of the dataset
            symbol: Optional symbol identifier
            download_date: Date of download
            modification_time: Data modification timestamp
            data_hash: Hash of the downloaded data
            version: Version number
            operation_type: Type of operation (download, update, etc.)
            success: Whether the operation was successful
            error_message: Optional error message if failed
            metadata: Additional metadata
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO download_history (
                    dataset_name, symbol, download_date, modification_time,
                    data_hash, version, operation_type, success, error_message,
                    metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_name,
                symbol,
                download_date.isoformat(),
                modification_time.isoformat(),
                data_hash,
                version,
                operation_type,
                success,
                error_message,
                json.dumps(metadata or {}),
                datetime.utcnow().isoformat()
            ))

            conn.commit()

    def get_download_history(self,
                           dataset_name: str,
                           symbol: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get download history for a dataset/symbol.

        Args:
            dataset_name: Name of the dataset
            symbol: Optional symbol identifier
            limit: Maximum number of records to return

        Returns:
            List of download history records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if symbol is not None:
                cursor.execute("""
                    SELECT * FROM download_history
                    WHERE dataset_name = ? AND symbol = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (dataset_name, symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM download_history
                    WHERE dataset_name = ? AND symbol IS NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (dataset_name, limit))

            return [dict(row) for row in cursor.fetchall()]

    def save_data_version(self,
                         dataset_name: str,
                         symbol: Optional[str],
                         version: int,
                         data_hash: str,
                         modification_time: datetime,
                         changes_summary: Optional[str] = None) -> None:
        """
        Save a data version for tracking changes.

        Args:
            dataset_name: Name of the dataset
            symbol: Optional symbol identifier
            version: Version number
            data_hash: Hash of the data
            modification_time: When the data was modified
            changes_summary: Optional summary of changes
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO data_versions (
                    dataset_name, symbol, version, data_hash,
                    modification_time, changes_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_name,
                symbol,
                version,
                data_hash,
                modification_time.isoformat(),
                changes_summary,
                datetime.utcnow().isoformat()
            ))

            conn.commit()

    def get_data_versions(self,
                         dataset_name: str,
                         symbol: Optional[str] = None,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get data versions for a dataset/symbol.

        Args:
            dataset_name: Name of the dataset
            symbol: Optional symbol identifier
            limit: Maximum number of versions to return

        Returns:
            List of data version records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if symbol is not None:
                cursor.execute("""
                    SELECT * FROM data_versions
                    WHERE dataset_name = ? AND symbol = ?
                    ORDER BY version DESC
                    LIMIT ?
                """, (dataset_name, symbol, limit))
            else:
                cursor.execute("""
                    SELECT * FROM data_versions
                    WHERE dataset_name = ? AND symbol IS NULL
                    ORDER BY version DESC
                    LIMIT ?
                """, (dataset_name, limit))

            return [dict(row) for row in cursor.fetchall()]

    def create_rollback_operation(self,
                                operation_id: str,
                                dataset_name: str,
                                symbol: Optional[str],
                                from_version: int,
                                to_version: int,
                                operation_type: str) -> None:
        """
        Create a rollback operation record.

        Args:
            operation_id: Unique identifier for the operation
            dataset_name: Name of the dataset
            symbol: Optional symbol identifier
            from_version: Version to rollback from
            to_version: Version to rollback to
            operation_type: Type of rollback operation
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO rollback_operations (
                    operation_id, dataset_name, symbol, from_version,
                    to_version, operation_type, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                operation_id,
                dataset_name,
                symbol,
                from_version,
                to_version,
                operation_type,
                "pending",
                datetime.utcnow().isoformat()
            ))

            conn.commit()

    def update_rollback_status(self, operation_id: str, status: str) -> None:
        """
        Update the status of a rollback operation.

        Args:
            operation_id: Unique identifier for the operation
            status: New status (pending, in_progress, completed, failed)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            completed_at = datetime.utcnow().isoformat() if status in ["completed", "failed"] else None

            cursor.execute("""
                UPDATE rollback_operations
                SET status = ?, completed_at = ?
                WHERE operation_id = ?
            """, (status, completed_at, operation_id))

            conn.commit()

    def get_checkpoints_by_state(self, state: DataState) -> List[DownloadCheckpoint]:
        """
        Get all checkpoints with a specific state.

        Args:
            state: State to filter by

        Returns:
            List of checkpoints
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM checkpoints WHERE state = ?",
                (state.value,)
            )

            return [DownloadCheckpoint.from_dict(dict(row)) for row in cursor.fetchall()]

    def get_outdated_checkpoints(self, cutoff_date: datetime) -> List[DownloadCheckpoint]:
        """
        Get checkpoints that are older than the cutoff date.

        Args:
            cutoff_date: Cutoff datetime for considering data outdated

        Returns:
            List of outdated checkpoints
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM checkpoints WHERE last_modification_time < ?",
                (cutoff_date.isoformat(),)
            )

            return [DownloadCheckpoint.from_dict(dict(row)) for row in cursor.fetchall()]

    def calculate_data_hash(self, data: Any) -> str:
        """
        Calculate SHA-256 hash of data for integrity verification.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, str):
            content = data.encode('utf-8')
        elif hasattr(data, 'to_json'):
            content = data.to_json().encode('utf-8')
        elif hasattr(data, '__str__'):
            content = str(data).encode('utf-8')
        else:
            content = str(data).encode('utf-8')

        return hashlib.sha256(content).hexdigest()

    def cleanup_old_history(self, days_to_keep: int = 30) -> int:
        """
        Clean up old download history records.

        Args:
            days_to_keep: Number of days of history to keep

        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM download_history WHERE created_at < ?",
                (cutoff_date.isoformat(),)
            )

            deleted_count = cursor.rowcount
            conn.commit()

            logger.info(f"Cleaned up {deleted_count} old download history records")
            return deleted_count

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the state tracking database.

        Returns:
            Dictionary containing various statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Count checkpoints by state
            cursor.execute("SELECT state, COUNT(*) FROM checkpoints GROUP BY state")
            state_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Count total records
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            total_checkpoints = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM download_history")
            total_history = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM data_versions")
            total_versions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM rollback_operations")
            total_rollbacks = cursor.fetchone()[0]

            # Get database size
            db_size = self.database_path.stat().st_size if self.database_path.exists() else 0

            return {
                'database_path': str(self.database_path),
                'database_size_bytes': db_size,
                'total_checkpoints': total_checkpoints,
                'total_history_records': total_history,
                'total_versions': total_versions,
                'total_rollback_operations': total_rollbacks,
                'checkpoints_by_state': state_counts
            }