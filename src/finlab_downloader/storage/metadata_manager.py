"""
Metadata management system for efficient time-series queries and data organization.

Provides comprehensive metadata tracking, indexing, and query optimization
for financial data stored in Parquet format.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Iterator
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TableMetadata:
    """Metadata for a data table/partition."""
    dataset: str
    partition_key: str
    file_path: str
    created_at: datetime
    updated_at: datetime
    row_count: int
    column_count: int
    file_size_bytes: int
    compression_type: str
    schema_version: str
    min_date: Optional[date]
    max_date: Optional[date]
    symbols: List[str]
    columns: List[str]
    data_quality_score: float
    checksum: str
    custom_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['min_date'] = self.min_date.isoformat() if self.min_date else None
        data['max_date'] = self.max_date.isoformat() if self.max_date else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableMetadata':
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['min_date'] = date.fromisoformat(data['min_date']) if data['min_date'] else None
        data['max_date'] = date.fromisoformat(data['max_date']) if data['max_date'] else None
        return cls(**data)


@dataclass
class QueryIndex:
    """Index for efficient metadata queries."""
    index_name: str
    columns: List[str]
    index_type: str  # 'btree', 'hash', 'range'
    created_at: datetime
    last_used: datetime
    usage_count: int
    selectivity: float  # Estimated selectivity (0-1)


class MetadataManager:
    """
    Comprehensive metadata management for financial data storage.

    Features:
    - SQLite-based metadata storage with indexes
    - Schema evolution tracking
    - Data lineage and audit trail
    - Query optimization hints
    - Performance metrics collection
    """

    def __init__(self, storage_path: Union[str, Path]):
        """
        Initialize metadata manager.

        Args:
            storage_path: Path to metadata storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.storage_path / "metadata.db"
        self.indexes_path = self.storage_path / "indexes.json"

        self._init_database()
        self._load_indexes()

    def _init_database(self) -> None:
        """Initialize SQLite database with schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS table_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset TEXT NOT NULL,
                    partition_key TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    compression_type TEXT,
                    schema_version TEXT,
                    min_date TEXT,
                    max_date TEXT,
                    symbols TEXT,  -- JSON array
                    columns TEXT,  -- JSON array
                    data_quality_score REAL,
                    checksum TEXT,
                    custom_metadata TEXT  -- JSON object
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    schema_definition TEXT NOT NULL,  -- JSON
                    created_at TEXT NOT NULL,
                    migration_notes TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_pattern TEXT NOT NULL,
                    dataset TEXT,
                    execution_time_ms REAL NOT NULL,
                    rows_scanned INTEGER,
                    rows_returned INTEGER,
                    cache_hit BOOLEAN,
                    timestamp TEXT NOT NULL
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON table_metadata(dataset)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_date_range ON table_metadata(min_date, max_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON table_metadata(updated_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_partition_key ON table_metadata(partition_key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_schema ON schema_history(dataset, schema_version)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def register_table(self, metadata: TableMetadata) -> None:
        """
        Register table metadata.

        Args:
            metadata: Table metadata to register
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if table already exists
            cursor.execute(
                "SELECT id FROM table_metadata WHERE file_path = ?",
                (metadata.file_path,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing record
                cursor.execute("""
                    UPDATE table_metadata SET
                        dataset = ?, partition_key = ?, updated_at = ?,
                        row_count = ?, column_count = ?, file_size_bytes = ?,
                        compression_type = ?, schema_version = ?,
                        min_date = ?, max_date = ?, symbols = ?, columns = ?,
                        data_quality_score = ?, checksum = ?, custom_metadata = ?
                    WHERE file_path = ?
                """, (
                    metadata.dataset, metadata.partition_key, metadata.updated_at.isoformat(),
                    metadata.row_count, metadata.column_count, metadata.file_size_bytes,
                    metadata.compression_type, metadata.schema_version,
                    metadata.min_date.isoformat() if metadata.min_date else None,
                    metadata.max_date.isoformat() if metadata.max_date else None,
                    json.dumps(metadata.symbols), json.dumps(metadata.columns),
                    metadata.data_quality_score, metadata.checksum,
                    json.dumps(metadata.custom_metadata), metadata.file_path
                ))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO table_metadata (
                        dataset, partition_key, file_path, created_at, updated_at,
                        row_count, column_count, file_size_bytes, compression_type,
                        schema_version, min_date, max_date, symbols, columns,
                        data_quality_score, checksum, custom_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.dataset, metadata.partition_key, metadata.file_path,
                    metadata.created_at.isoformat(), metadata.updated_at.isoformat(),
                    metadata.row_count, metadata.column_count, metadata.file_size_bytes,
                    metadata.compression_type, metadata.schema_version,
                    metadata.min_date.isoformat() if metadata.min_date else None,
                    metadata.max_date.isoformat() if metadata.max_date else None,
                    json.dumps(metadata.symbols), json.dumps(metadata.columns),
                    metadata.data_quality_score, metadata.checksum,
                    json.dumps(metadata.custom_metadata)
                ))

            conn.commit()

    def get_table_metadata(self, file_path: str) -> Optional[TableMetadata]:
        """
        Get metadata for a specific table.

        Args:
            file_path: Path to the table file

        Returns:
            Table metadata or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM table_metadata WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()

            if row:
                return self._row_to_metadata(row)
            return None

    def list_tables(self,
                   dataset: Optional[str] = None,
                   start_date: Optional[date] = None,
                   end_date: Optional[date] = None,
                   symbols: Optional[List[str]] = None) -> List[TableMetadata]:
        """
        List tables matching criteria.

        Args:
            dataset: Filter by dataset
            start_date: Filter by start date
            end_date: Filter by end date
            symbols: Filter by symbols

        Returns:
            List of matching table metadata
        """
        query = "SELECT * FROM table_metadata WHERE 1=1"
        params = []

        if dataset:
            query += " AND dataset = ?"
            params.append(dataset)

        if start_date:
            query += " AND (max_date IS NULL OR max_date >= ?)"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND (min_date IS NULL OR min_date <= ?)"
            params.append(end_date.isoformat())

        query += " ORDER BY dataset, min_date"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                metadata = self._row_to_metadata(row)

                # Filter by symbols if specified
                if symbols and metadata.symbols:
                    if not any(symbol in metadata.symbols for symbol in symbols):
                        continue

                results.append(metadata)

            return results

    def delete_table_metadata(self, file_path: str) -> bool:
        """
        Delete metadata for a table.

        Args:
            file_path: Path to the table file

        Returns:
            True if deleted successfully
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM table_metadata WHERE file_path = ?",
                (file_path,)
            )
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def get_dataset_summary(self, dataset: str) -> Dict[str, Any]:
        """
        Get summary statistics for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Summary statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get basic stats
            cursor.execute("""
                SELECT
                    COUNT(*) as table_count,
                    SUM(row_count) as total_rows,
                    SUM(file_size_bytes) as total_size,
                    AVG(data_quality_score) as avg_quality,
                    MIN(min_date) as earliest_date,
                    MAX(max_date) as latest_date,
                    COUNT(DISTINCT partition_key) as partition_count
                FROM table_metadata WHERE dataset = ?
            """, (dataset,))

            row = cursor.fetchone()
            if not row:
                return {}

            summary = {
                'dataset': dataset,
                'table_count': row['table_count'],
                'total_rows': row['total_rows'] or 0,
                'total_size_bytes': row['total_size'] or 0,
                'average_quality_score': row['avg_quality'] or 0.0,
                'earliest_date': row['earliest_date'],
                'latest_date': row['latest_date'],
                'partition_count': row['partition_count'],
                'generated_at': datetime.utcnow().isoformat()
            }

            # Get compression stats
            cursor.execute("""
                SELECT compression_type, COUNT(*) as count
                FROM table_metadata WHERE dataset = ?
                GROUP BY compression_type
            """, (dataset,))

            compression_stats = {row['compression_type']: row['count'] for row in cursor.fetchall()}
            summary['compression_breakdown'] = compression_stats

            # Get column statistics
            cursor.execute("""
                SELECT columns FROM table_metadata WHERE dataset = ?
            """, (dataset,))

            all_columns = set()
            for row in cursor.fetchall():
                if row['columns']:
                    columns = json.loads(row['columns'])
                    all_columns.update(columns)

            summary['unique_columns'] = sorted(list(all_columns))
            summary['column_count'] = len(all_columns)

            return summary

    def record_query_stats(self,
                          query_pattern: str,
                          dataset: Optional[str],
                          execution_time_ms: float,
                          rows_scanned: int,
                          rows_returned: int,
                          cache_hit: bool = False) -> None:
        """
        Record query performance statistics.

        Args:
            query_pattern: Pattern of the query
            dataset: Dataset queried
            execution_time_ms: Query execution time in milliseconds
            rows_scanned: Number of rows scanned
            rows_returned: Number of rows returned
            cache_hit: Whether query was served from cache
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_stats (
                    query_pattern, dataset, execution_time_ms,
                    rows_scanned, rows_returned, cache_hit, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                query_pattern, dataset, execution_time_ms,
                rows_scanned, rows_returned, cache_hit,
                datetime.utcnow().isoformat()
            ))
            conn.commit()

    def get_query_performance(self, dataset: Optional[str] = None,
                            hours: int = 24) -> Dict[str, Any]:
        """
        Get query performance statistics.

        Args:
            dataset: Filter by dataset
            hours: Look back hours

        Returns:
            Performance statistics
        """
        cutoff_time = datetime.utcnow() - pd.Timedelta(hours=hours)

        query = """
            SELECT
                query_pattern,
                COUNT(*) as query_count,
                AVG(execution_time_ms) as avg_time,
                MAX(execution_time_ms) as max_time,
                SUM(rows_scanned) as total_scanned,
                SUM(rows_returned) as total_returned,
                AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END) as cache_hit_rate
            FROM query_stats
            WHERE timestamp >= ?
        """
        params = [cutoff_time.isoformat()]

        if dataset:
            query += " AND dataset = ?"
            params.append(dataset)

        query += " GROUP BY query_pattern ORDER BY query_count DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return {
                'performance_window_hours': hours,
                'query_patterns': [
                    {
                        'pattern': row['query_pattern'],
                        'count': row['query_count'],
                        'avg_time_ms': row['avg_time'],
                        'max_time_ms': row['max_time'],
                        'total_rows_scanned': row['total_scanned'],
                        'total_rows_returned': row['total_returned'],
                        'cache_hit_rate': row['cache_hit_rate']
                    }
                    for row in rows
                ],
                'generated_at': datetime.utcnow().isoformat()
            }

    def suggest_optimizations(self, dataset: str) -> List[Dict[str, Any]]:
        """
        Suggest query and storage optimizations.

        Args:
            dataset: Dataset to analyze

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Analyze table sizes
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check for very large tables
            cursor.execute("""
                SELECT file_path, row_count, file_size_bytes
                FROM table_metadata
                WHERE dataset = ? AND row_count > 1000000
                ORDER BY row_count DESC
            """, (dataset,))

            large_tables = cursor.fetchall()
            if large_tables:
                suggestions.append({
                    'type': 'partitioning',
                    'priority': 'high',
                    'description': f'Consider partitioning {len(large_tables)} large tables',
                    'affected_tables': [row['file_path'] for row in large_tables[:5]],
                    'estimated_benefit': 'Faster queries with predicate pushdown'
                })

            # Check compression ratios
            cursor.execute("""
                SELECT compression_type, COUNT(*) as count, AVG(file_size_bytes) as avg_size
                FROM table_metadata
                WHERE dataset = ?
                GROUP BY compression_type
            """, (dataset,))

            compression_stats = cursor.fetchall()
            uncompressed_count = sum(row['count'] for row in compression_stats
                                   if row['compression_type'] in [None, 'none'])

            if uncompressed_count > 0:
                suggestions.append({
                    'type': 'compression',
                    'priority': 'medium',
                    'description': f'Enable compression for {uncompressed_count} uncompressed tables',
                    'estimated_benefit': '60-80% storage reduction'
                })

            # Analyze query patterns
            cursor.execute("""
                SELECT query_pattern, AVG(execution_time_ms) as avg_time
                FROM query_stats
                WHERE dataset = ?
                GROUP BY query_pattern
                HAVING avg_time > 1000
                ORDER BY avg_time DESC
            """, (dataset,))

            slow_queries = cursor.fetchall()
            if slow_queries:
                suggestions.append({
                    'type': 'indexing',
                    'priority': 'high',
                    'description': f'Optimize {len(slow_queries)} slow query patterns',
                    'slow_patterns': [row['query_pattern'] for row in slow_queries[:3]],
                    'estimated_benefit': 'Faster query execution'
                })

        return suggestions

    def _row_to_metadata(self, row) -> TableMetadata:
        """Convert database row to TableMetadata."""
        return TableMetadata(
            dataset=row['dataset'],
            partition_key=row['partition_key'],
            file_path=row['file_path'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            row_count=row['row_count'],
            column_count=row['column_count'],
            file_size_bytes=row['file_size_bytes'],
            compression_type=row['compression_type'],
            schema_version=row['schema_version'],
            min_date=date.fromisoformat(row['min_date']) if row['min_date'] else None,
            max_date=date.fromisoformat(row['max_date']) if row['max_date'] else None,
            symbols=json.loads(row['symbols']) if row['symbols'] else [],
            columns=json.loads(row['columns']) if row['columns'] else [],
            data_quality_score=row['data_quality_score'] or 0.0,
            checksum=row['checksum'] or '',
            custom_metadata=json.loads(row['custom_metadata']) if row['custom_metadata'] else {}
        )

    def _load_indexes(self) -> None:
        """Load query indexes from storage."""
        if self.indexes_path.exists():
            try:
                with open(self.indexes_path, 'r') as f:
                    self.indexes = json.load(f)
            except Exception as e:
                logger.error(f"Error loading indexes: {e}")
                self.indexes = {}
        else:
            self.indexes = {}

    def _save_indexes(self) -> None:
        """Save query indexes to storage."""
        try:
            with open(self.indexes_path, 'w') as f:
                json.dump(self.indexes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving indexes: {e}")

    def cleanup_old_metadata(self, days_to_keep: int = 365) -> int:
        """
        Clean up old metadata records.

        Args:
            days_to_keep: Number of days to keep

        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.utcnow() - pd.Timedelta(days=days_to_keep)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Delete old query stats
            cursor.execute(
                "DELETE FROM query_stats WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            query_stats_deleted = cursor.rowcount

            # Delete metadata for non-existent files
            cursor.execute("SELECT file_path FROM table_metadata")
            all_files = cursor.fetchall()

            metadata_deleted = 0
            for row in all_files:
                file_path = Path(row['file_path'])
                if not file_path.exists():
                    cursor.execute(
                        "DELETE FROM table_metadata WHERE file_path = ?",
                        (str(file_path),)
                    )
                    metadata_deleted += cursor.rowcount

            conn.commit()

            logger.info(
                f"Cleaned up metadata: {metadata_deleted} table records, "
                f"{query_stats_deleted} query stats"
            )

            return metadata_deleted + query_stats_deleted