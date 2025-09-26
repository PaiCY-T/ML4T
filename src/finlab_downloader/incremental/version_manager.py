"""
Data versioning strategy for handling updates.

This module provides comprehensive data versioning capabilities to track
changes over time and enable rollback functionality.
"""

import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pickle
import gzip

logger = logging.getLogger(__name__)


class VersioningStrategy(Enum):
    """Data versioning strategies."""
    SNAPSHOT = "snapshot"           # Full snapshots of data
    INCREMENTAL = "incremental"     # Only store changes
    HYBRID = "hybrid"              # Snapshots + incremental changes
    COMPRESSED = "compressed"       # Compressed storage


@dataclass
class DataVersion:
    """Represents a version of data."""
    version_id: str
    dataset_name: str
    symbol: Optional[str]
    version_number: int
    parent_version: Optional[str]
    data_hash: str
    timestamp: datetime
    strategy: VersioningStrategy
    metadata: Dict[str, Any]

    # Storage information
    storage_path: Optional[str] = None
    compressed: bool = False
    size_bytes: int = 0

    # Change information
    changes_summary: Optional[str] = None
    affected_fields: List[str] = None

    def __post_init__(self):
        if self.affected_fields is None:
            self.affected_fields = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['strategy'] = self.strategy.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataVersion':
        """Create from dictionary."""
        return cls(
            version_id=data['version_id'],
            dataset_name=data['dataset_name'],
            symbol=data.get('symbol'),
            version_number=data['version_number'],
            parent_version=data.get('parent_version'),
            data_hash=data['data_hash'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            strategy=VersioningStrategy(data['strategy']),
            metadata=data['metadata'],
            storage_path=data.get('storage_path'),
            compressed=data.get('compressed', False),
            size_bytes=data.get('size_bytes', 0),
            changes_summary=data.get('changes_summary'),
            affected_fields=data.get('affected_fields', [])
        )


class VersionManager:
    """
    Advanced data versioning system for incremental downloads.

    Features:
    - Multiple versioning strategies
    - Efficient storage with compression
    - Change tracking and diff generation
    - Version history management
    - Space optimization through cleanup
    """

    def __init__(self,
                 storage_directory: Union[str, Path],
                 default_strategy: VersioningStrategy = VersioningStrategy.HYBRID,
                 max_versions_per_dataset: int = 100,
                 compression_enabled: bool = True):
        """
        Initialize version manager.

        Args:
            storage_directory: Directory for storing versions
            default_strategy: Default versioning strategy
            max_versions_per_dataset: Maximum versions to keep per dataset
            compression_enabled: Enable compression for storage
        """
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)

        self.default_strategy = default_strategy
        self.max_versions_per_dataset = max_versions_per_dataset
        self.compression_enabled = compression_enabled

        # Version registry
        self.versions: Dict[str, DataVersion] = {}
        self.version_chains: Dict[Tuple[str, Optional[str]], List[str]] = {}

        # Load existing versions
        self._load_version_registry()

    def create_version(self,
                      dataset_name: str,
                      data: Any,
                      symbol: Optional[str] = None,
                      parent_version_id: Optional[str] = None,
                      strategy: Optional[VersioningStrategy] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> DataVersion:
        """
        Create a new version of data.

        Args:
            dataset_name: Name of the dataset
            data: Data to version
            symbol: Optional symbol identifier
            parent_version_id: ID of parent version
            strategy: Versioning strategy to use
            metadata: Additional metadata

        Returns:
            Created DataVersion
        """
        strategy = strategy or self.default_strategy
        metadata = metadata or {}

        # Generate version ID and number
        version_id = self._generate_version_id(dataset_name, symbol)
        version_number = self._get_next_version_number(dataset_name, symbol)

        # Calculate data hash
        data_hash = self._calculate_data_hash(data)

        # Create version object
        version = DataVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            symbol=symbol,
            version_number=version_number,
            parent_version=parent_version_id,
            data_hash=data_hash,
            timestamp=datetime.utcnow(),
            strategy=strategy,
            metadata=metadata
        )

        # Store data based on strategy
        self._store_version_data(version, data)

        # Update registry
        self.versions[version_id] = version
        self._update_version_chain(dataset_name, symbol, version_id)

        # Save registry
        self._save_version_registry()

        # Cleanup old versions if needed
        self._cleanup_old_versions(dataset_name, symbol)

        logger.info(f"Created version {version_id} for {dataset_name}:{symbol}")
        return version

    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get version by ID."""
        return self.versions.get(version_id)

    def get_latest_version(self,
                          dataset_name: str,
                          symbol: Optional[str] = None) -> Optional[DataVersion]:
        """Get the latest version for a dataset/symbol."""
        chain_key = (dataset_name, symbol)
        version_chain = self.version_chains.get(chain_key, [])

        if not version_chain:
            return None

        latest_version_id = version_chain[-1]
        return self.versions.get(latest_version_id)

    def get_version_history(self,
                           dataset_name: str,
                           symbol: Optional[str] = None,
                           limit: Optional[int] = None) -> List[DataVersion]:
        """
        Get version history for a dataset/symbol.

        Args:
            dataset_name: Name of the dataset
            symbol: Optional symbol identifier
            limit: Optional limit on number of versions

        Returns:
            List of versions in chronological order (newest first)
        """
        chain_key = (dataset_name, symbol)
        version_chain = self.version_chains.get(chain_key, [])

        if limit:
            version_chain = version_chain[-limit:]

        versions = []
        for version_id in reversed(version_chain):
            version = self.versions.get(version_id)
            if version:
                versions.append(version)

        return versions

    def load_version_data(self, version_id: str) -> Optional[Any]:
        """
        Load data for a specific version.

        Args:
            version_id: Version ID to load

        Returns:
            Version data or None if not found
        """
        version = self.versions.get(version_id)
        if not version:
            return None

        return self._load_version_data(version)

    def compare_versions(self,
                        version_id1: str,
                        version_id2: str) -> Optional[Dict[str, Any]]:
        """
        Compare two versions and return differences.

        Args:
            version_id1: First version ID
            version_id2: Second version ID

        Returns:
            Comparison result or None if versions not found
        """
        version1 = self.versions.get(version_id1)
        version2 = self.versions.get(version_id2)

        if not version1 or not version2:
            return None

        data1 = self._load_version_data(version1)
        data2 = self._load_version_data(version2)

        if data1 is None or data2 is None:
            return None

        # Perform comparison
        comparison = {
            'version1': version1.to_dict(),
            'version2': version2.to_dict(),
            'hash_match': version1.data_hash == version2.data_hash,
            'size_difference': version2.size_bytes - version1.size_bytes,
            'time_difference': (version2.timestamp - version1.timestamp).total_seconds(),
            'differences': self._calculate_differences(data1, data2)
        }

        return comparison

    def delete_version(self, version_id: str) -> bool:
        """
        Delete a specific version.

        Args:
            version_id: Version ID to delete

        Returns:
            True if successfully deleted
        """
        version = self.versions.get(version_id)
        if not version:
            return False

        try:
            # Remove stored data
            if version.storage_path:
                storage_path = Path(version.storage_path)
                if storage_path.exists():
                    storage_path.unlink()

            # Remove from registry
            del self.versions[version_id]

            # Update version chain
            chain_key = (version.dataset_name, version.symbol)
            if chain_key in self.version_chains:
                self.version_chains[chain_key] = [
                    vid for vid in self.version_chains[chain_key] if vid != version_id
                ]

            # Save registry
            self._save_version_registry()

            logger.info(f"Deleted version {version_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting version {version_id}: {e}")
            return False

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics for all versions."""
        total_versions = len(self.versions)
        total_size = sum(v.size_bytes for v in self.versions.values())
        compressed_count = sum(1 for v in self.versions.values() if v.compressed)

        # Group by dataset
        dataset_stats = {}
        for version in self.versions.values():
            key = f"{version.dataset_name}:{version.symbol}"
            if key not in dataset_stats:
                dataset_stats[key] = {
                    'version_count': 0,
                    'total_size': 0,
                    'latest_version': None
                }

            dataset_stats[key]['version_count'] += 1
            dataset_stats[key]['total_size'] += version.size_bytes

            if (dataset_stats[key]['latest_version'] is None or
                version.timestamp > dataset_stats[key]['latest_version']):
                dataset_stats[key]['latest_version'] = version.timestamp

        return {
            'total_versions': total_versions,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'compressed_versions': compressed_count,
            'compression_ratio': compressed_count / max(total_versions, 1),
            'dataset_statistics': dataset_stats,
            'storage_directory': str(self.storage_directory)
        }

    def _generate_version_id(self, dataset_name: str, symbol: Optional[str]) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        symbol_part = f"_{symbol}" if symbol else ""
        return f"{dataset_name}{symbol_part}_{timestamp}"

    def _get_next_version_number(self, dataset_name: str, symbol: Optional[str]) -> int:
        """Get the next version number for a dataset/symbol."""
        chain_key = (dataset_name, symbol)
        version_chain = self.version_chains.get(chain_key, [])

        if not version_chain:
            return 1

        # Find highest version number
        max_version = 0
        for version_id in version_chain:
            version = self.versions.get(version_id)
            if version:
                max_version = max(max_version, version.version_number)

        return max_version + 1

    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate hash for data."""
        try:
            if hasattr(data, 'to_json'):
                content = data.to_json()
            elif isinstance(data, dict):
                content = json.dumps(data, sort_keys=True, default=str)
            else:
                content = str(data)

            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating hash: {e}")
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()

    def _store_version_data(self, version: DataVersion, data: Any) -> None:
        """Store version data to disk."""
        # Create storage path
        storage_filename = f"{version.version_id}.dat"
        if self.compression_enabled:
            storage_filename += ".gz"

        storage_path = self.storage_directory / storage_filename

        try:
            # Serialize data
            if version.strategy == VersioningStrategy.SNAPSHOT:
                serialized_data = pickle.dumps(data)
            elif version.strategy == VersioningStrategy.INCREMENTAL:
                # For incremental, we would store only changes
                # For now, store as snapshot
                serialized_data = pickle.dumps(data)
            else:
                serialized_data = pickle.dumps(data)

            # Compress if enabled
            if self.compression_enabled:
                with gzip.open(storage_path, 'wb') as f:
                    f.write(serialized_data)
                version.compressed = True
            else:
                with open(storage_path, 'wb') as f:
                    f.write(serialized_data)
                version.compressed = False

            version.storage_path = str(storage_path)
            version.size_bytes = storage_path.stat().st_size

        except Exception as e:
            logger.error(f"Error storing version data: {e}")
            raise

    def _load_version_data(self, version: DataVersion) -> Optional[Any]:
        """Load version data from disk."""
        if not version.storage_path:
            return None

        storage_path = Path(version.storage_path)
        if not storage_path.exists():
            logger.warning(f"Version data file not found: {storage_path}")
            return None

        try:
            if version.compressed:
                with gzip.open(storage_path, 'rb') as f:
                    serialized_data = f.read()
            else:
                with open(storage_path, 'rb') as f:
                    serialized_data = f.read()

            return pickle.loads(serialized_data)

        except Exception as e:
            logger.error(f"Error loading version data: {e}")
            return None

    def _update_version_chain(self,
                             dataset_name: str,
                             symbol: Optional[str],
                             version_id: str) -> None:
        """Update version chain for a dataset/symbol."""
        chain_key = (dataset_name, symbol)
        if chain_key not in self.version_chains:
            self.version_chains[chain_key] = []

        self.version_chains[chain_key].append(version_id)

    def _cleanup_old_versions(self, dataset_name: str, symbol: Optional[str]) -> None:
        """Clean up old versions if exceeding max count."""
        chain_key = (dataset_name, symbol)
        version_chain = self.version_chains.get(chain_key, [])

        if len(version_chain) <= self.max_versions_per_dataset:
            return

        # Remove oldest versions
        versions_to_remove = version_chain[:-self.max_versions_per_dataset]

        for version_id in versions_to_remove:
            self.delete_version(version_id)

        logger.info(f"Cleaned up {len(versions_to_remove)} old versions for {dataset_name}:{symbol}")

    def _load_version_registry(self) -> None:
        """Load version registry from disk."""
        registry_path = self.storage_directory / "version_registry.json"

        if not registry_path.exists():
            return

        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)

            # Load versions
            for version_data in registry_data.get('versions', []):
                version = DataVersion.from_dict(version_data)
                self.versions[version.version_id] = version

            # Load version chains
            for chain_key_str, version_list in registry_data.get('version_chains', {}).items():
                # Parse chain key
                if ':' in chain_key_str:
                    dataset_name, symbol = chain_key_str.split(':', 1)
                    symbol = symbol if symbol != 'None' else None
                else:
                    dataset_name, symbol = chain_key_str, None

                self.version_chains[(dataset_name, symbol)] = version_list

            logger.info(f"Loaded {len(self.versions)} versions from registry")

        except Exception as e:
            logger.error(f"Error loading version registry: {e}")

    def _save_version_registry(self) -> None:
        """Save version registry to disk."""
        registry_path = self.storage_directory / "version_registry.json"

        try:
            # Prepare data for serialization
            registry_data = {
                'versions': [version.to_dict() for version in self.versions.values()],
                'version_chains': {}
            }

            # Convert version chains keys to strings
            for (dataset_name, symbol), version_list in self.version_chains.items():
                chain_key_str = f"{dataset_name}:{symbol}"
                registry_data['version_chains'][chain_key_str] = version_list

            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving version registry: {e}")

    def _calculate_differences(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """Calculate differences between two data objects."""
        try:
            import pandas as pd

            if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
                return self._compare_dataframes(data1, data2)
            elif isinstance(data1, dict) and isinstance(data2, dict):
                return self._compare_dicts(data1, data2)
            else:
                return {
                    'type': 'generic',
                    'equal': data1 == data2,
                    'data1_type': type(data1).__name__,
                    'data2_type': type(data2).__name__
                }
        except Exception as e:
            logger.warning(f"Error calculating differences: {e}")
            return {'error': str(e)}

    def _compare_dataframes(self, df1, df2) -> Dict[str, Any]:
        """Compare two DataFrames."""
        try:
            import pandas as pd

            comparison = {
                'type': 'dataframe',
                'shape1': df1.shape,
                'shape2': df2.shape,
                'columns1': list(df1.columns),
                'columns2': list(df2.columns),
                'equal': df1.equals(df2)
            }

            if not comparison['equal']:
                # Find differences
                common_columns = set(df1.columns) & set(df2.columns)
                comparison['added_columns'] = list(set(df2.columns) - set(df1.columns))
                comparison['removed_columns'] = list(set(df1.columns) - set(df2.columns))

                if common_columns and df1.index.equals(df2.index):
                    changed_columns = []
                    for col in common_columns:
                        if not df1[col].equals(df2[col]):
                            changed_columns.append(col)
                    comparison['changed_columns'] = changed_columns

            return comparison

        except Exception as e:
            return {'error': str(e), 'type': 'dataframe'}

    def _compare_dicts(self, dict1: Dict, dict2: Dict) -> Dict[str, Any]:
        """Compare two dictionaries."""
        comparison = {
            'type': 'dict',
            'keys1': list(dict1.keys()),
            'keys2': list(dict2.keys()),
            'equal': dict1 == dict2
        }

        if not comparison['equal']:
            all_keys = set(dict1.keys()) | set(dict2.keys())
            comparison['added_keys'] = list(set(dict2.keys()) - set(dict1.keys()))
            comparison['removed_keys'] = list(set(dict1.keys()) - set(dict2.keys()))

            changed_keys = []
            for key in set(dict1.keys()) & set(dict2.keys()):
                if dict1[key] != dict2[key]:
                    changed_keys.append(key)
            comparison['changed_keys'] = changed_keys

        return comparison