"""
Main incremental download manager integrating all components.

This module provides the primary interface for incremental download operations,
coordinating state tracking, timestamp management, change detection, versioning,
queue management, conflict resolution, and rollback capabilities.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass

from .state_tracker import StateTracker, DataState, DownloadCheckpoint
from .timestamp_manager import TimestampManager, TimestampComparison
from .change_detector import ChangeDetector, ChangeType, DataChange
from .version_manager import VersionManager, DataVersion, VersioningStrategy
from .queue_manager import IncrementalQueueManager, Priority, DownloadTask
from .conflict_resolver import ConflictResolver, ConflictType, ResolutionStrategy, DataRange
from .rollback_manager import RollbackManager, RollbackType

from ..core.client import FinLabClient
from ..core.dataset import DatasetSpecification

logger = logging.getLogger(__name__)


@dataclass
class IncrementalConfig:
    """Configuration for incremental download manager."""
    storage_directory: str
    max_workers: int = 4
    max_queue_size: int = 1000
    default_timeout: int = 300
    max_versions_per_dataset: int = 100
    compression_enabled: bool = True
    conflict_resolution_strategy: ResolutionStrategy = ResolutionStrategy.LATEST_WINS
    versioning_strategy: VersioningStrategy = VersioningStrategy.HYBRID
    enable_automatic_rollback: bool = True
    rollback_on_error_threshold: float = 0.3


class IncrementalDownloadManager:
    """
    Comprehensive incremental download manager for FinLab data.

    Features:
    - Intelligent timestamp-based updates
    - Automatic change detection
    - Data versioning and rollback
    - Conflict resolution
    - Queue-based processing
    - Integrity verification
    - Performance monitoring
    """

    def __init__(self,
                 config: IncrementalConfig,
                 finlab_client: FinLabClient):
        """
        Initialize incremental download manager.

        Args:
            config: Configuration object
            finlab_client: FinLab API client
        """
        self.config = config
        self.finlab_client = finlab_client

        # Setup storage directories
        storage_path = Path(config.storage_directory)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.state_tracker = StateTracker(storage_path / "state.db")
        self.timestamp_manager = TimestampManager()
        self.change_detector = ChangeDetector()

        self.version_manager = VersionManager(
            storage_directory=storage_path / "versions",
            default_strategy=config.versioning_strategy,
            max_versions_per_dataset=config.max_versions_per_dataset,
            compression_enabled=config.compression_enabled
        )

        self.queue_manager = IncrementalQueueManager(
            max_workers=config.max_workers,
            max_queue_size=config.max_queue_size,
            default_timeout=config.default_timeout
        )

        self.conflict_resolver = ConflictResolver(
            default_strategy=config.conflict_resolution_strategy
        )

        self.rollback_manager = RollbackManager(
            state_tracker=self.state_tracker,
            version_manager=self.version_manager,
            storage_directory=storage_path / "rollbacks"
        )

        # Statistics
        self.download_statistics = {
            'total_downloads': 0,
            'incremental_downloads': 0,
            'full_downloads': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'rollbacks_performed': 0,
            'last_updated': datetime.utcnow()
        }

    def start(self) -> None:
        """Start the incremental download manager."""
        self.queue_manager.start()
        logger.info("Incremental download manager started")

    def stop(self) -> None:
        """Stop the incremental download manager."""
        self.queue_manager.stop()
        logger.info("Incremental download manager stopped")

    def schedule_incremental_download(self,
                                    dataset_spec: DatasetSpecification,
                                    symbol: Optional[str] = None,
                                    priority: Priority = Priority.MEDIUM,
                                    force_update: bool = False,
                                    callback: Optional[Callable] = None) -> str:
        """
        Schedule an incremental download.

        Args:
            dataset_spec: Dataset specification
            symbol: Optional symbol filter
            priority: Download priority
            force_update: Force update regardless of timestamps
            callback: Optional completion callback

        Returns:
            Task ID
        """
        task_id = self.queue_manager.schedule_task(
            dataset_name=dataset_spec.name,
            symbol=symbol,
            priority=priority,
            metadata={
                'dataset_spec': dataset_spec,
                'force_update': force_update,
                'incremental_mode': True
            },
            callback=callback
        )

        logger.info(f"Scheduled incremental download: {dataset_spec.name}:{symbol} (task: {task_id})")
        return task_id

    def execute_incremental_download(self,
                                   dataset_spec: DatasetSpecification,
                                   symbol: Optional[str] = None,
                                   force_update: bool = False) -> Dict[str, Any]:
        """
        Execute an incremental download immediately.

        Args:
            dataset_spec: Dataset specification
            symbol: Optional symbol filter
            force_update: Force update regardless of timestamps

        Returns:
            Download result
        """
        logger.info(f"Executing incremental download: {dataset_spec.name}:{symbol}")

        try:
            # Get current checkpoint
            checkpoint = self.state_tracker.get_checkpoint(dataset_spec.name, symbol)

            # Determine if update is needed
            if not force_update and checkpoint:
                needs_update, reason = self._needs_update(dataset_spec, symbol, checkpoint)
                if not needs_update:
                    logger.info(f"No update needed for {dataset_spec.name}:{symbol}: {reason}")
                    return {
                        'status': 'skipped',
                        'reason': reason,
                        'checkpoint': checkpoint.to_dict()
                    }

            # Download new data
            download_result = self._download_data(dataset_spec, symbol, checkpoint)

            if download_result['status'] == 'success':
                # Process and validate the download
                processed_result = self._process_download(
                    dataset_spec, symbol, download_result, checkpoint
                )

                self.download_statistics['total_downloads'] += 1
                self.download_statistics['incremental_downloads'] += 1
                self.download_statistics['last_updated'] = datetime.utcnow()

                return processed_result
            else:
                # Handle download failure
                return self._handle_download_failure(dataset_spec, symbol, download_result)

        except Exception as e:
            logger.error(f"Error in incremental download: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'dataset': dataset_spec.name,
                'symbol': symbol
            }

    def _needs_update(self,
                     dataset_spec: DatasetSpecification,
                     symbol: Optional[str],
                     checkpoint: DownloadCheckpoint) -> Tuple[bool, str]:
        """Determine if data needs to be updated."""
        try:
            # Get remote timestamp (this would be implemented based on FinLab API)
            remote_timestamp = self._get_remote_timestamp(dataset_spec, symbol)

            if remote_timestamp is None:
                return True, "Cannot determine remote timestamp"

            # Compare timestamps
            needs_update, reason = self.timestamp_manager.needs_update(
                local_timestamp=checkpoint.last_modification_time,
                remote_timestamp=remote_timestamp,
                max_age_hours=24  # Consider data stale after 24 hours
            )

            return needs_update, reason

        except Exception as e:
            logger.warning(f"Error checking update need: {e}")
            return True, f"Error checking timestamps: {e}"

    def _get_remote_timestamp(self,
                            dataset_spec: DatasetSpecification,
                            symbol: Optional[str]) -> Optional[datetime]:
        """Get remote data timestamp (placeholder implementation)."""
        # This would be implemented based on FinLab API capabilities
        # For now, return current time to simulate fresh data
        return datetime.utcnow()

    def _download_data(self,
                      dataset_spec: DatasetSpecification,
                      symbol: Optional[str],
                      checkpoint: Optional[DownloadCheckpoint]) -> Dict[str, Any]:
        """Download data from FinLab API."""
        try:
            download_result = self.finlab_client.download_dataset(dataset_spec)

            if 'data' in download_result:
                return {
                    'status': 'success',
                    'data': download_result['data'],
                    'metadata': download_result.get('metadata', {}),
                    'download_time': datetime.utcnow()
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'No data in download result',
                    'result': download_result
                }

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _process_download(self,
                         dataset_spec: DatasetSpecification,
                         symbol: Optional[str],
                         download_result: Dict[str, Any],
                         checkpoint: Optional[DownloadCheckpoint]) -> Dict[str, Any]:
        """Process downloaded data."""
        try:
            data = download_result['data']
            download_time = download_result['download_time']

            # Detect changes if we have previous data
            changes = []
            if checkpoint:
                # Load previous version for comparison
                previous_version = self.version_manager.get_latest_version(
                    dataset_spec.name, symbol
                )
                if previous_version:
                    previous_data = self.version_manager.load_version_data(
                        previous_version.version_id
                    )
                    if previous_data is not None:
                        changes = self.change_detector.detect_changes(
                            previous_data, data, dataset_spec.data_type.value
                        )

            # Calculate data hash
            data_hash = self.state_tracker.calculate_data_hash(data)

            # Create new version
            version = self.version_manager.create_version(
                dataset_name=dataset_spec.name,
                data=data,
                symbol=symbol,
                parent_version_id=checkpoint.data_hash if checkpoint else None,
                metadata={
                    'download_metadata': download_result.get('metadata', {}),
                    'changes': [change.to_dict() for change in changes],
                    'dataset_spec': {
                        'name': dataset_spec.name,
                        'data_type': dataset_spec.data_type.value
                    }
                }
            )

            # Update checkpoint
            new_checkpoint = DownloadCheckpoint(
                dataset_name=dataset_spec.name,
                symbol=symbol,
                last_download_date=download_time.date(),
                last_modification_time=download_time,
                data_hash=data_hash,
                version=version.version_number,
                state=DataState.COMPLETED,
                metadata={
                    'version_id': version.version_id,
                    'changes_count': len(changes),
                    'significant_changes': any(
                        change.change_type in [ChangeType.NEW_DATA, ChangeType.STRUCTURAL_CHANGE]
                        for change in changes
                    )
                },
                created_at=download_time,
                updated_at=download_time
            )

            self.state_tracker.save_checkpoint(new_checkpoint)

            # Record download in history
            self.state_tracker.record_download_attempt(
                dataset_name=dataset_spec.name,
                symbol=symbol,
                download_date=download_time.date(),
                modification_time=download_time,
                data_hash=data_hash,
                version=version.version_number,
                operation_type="incremental_download",
                success=True,
                metadata={
                    'changes_detected': len(changes),
                    'version_id': version.version_id
                }
            )

            # Check for conflicts
            conflicts = self._check_for_conflicts(dataset_spec.name, symbol, version)

            result = {
                'status': 'success',
                'dataset': dataset_spec.name,
                'symbol': symbol,
                'version_id': version.version_id,
                'data_hash': data_hash,
                'changes_detected': len(changes),
                'changes_summary': self.change_detector.summarize_changes(changes),
                'conflicts_detected': len(conflicts),
                'download_time': download_time.isoformat(),
                'checkpoint': new_checkpoint.to_dict()
            }

            if conflicts:
                result['conflicts'] = [
                    {
                        'conflict_id': c.conflict_id,
                        'type': c.conflict_type.value,
                        'ranges_count': len(c.ranges)
                    }
                    for c in conflicts
                ]
                self.download_statistics['conflicts_detected'] += len(conflicts)

            return result

        except Exception as e:
            logger.error(f"Error processing download: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'dataset': dataset_spec.name,
                'symbol': symbol
            }

    def _check_for_conflicts(self,
                           dataset_name: str,
                           symbol: Optional[str],
                           new_version: DataVersion) -> List:
        """Check for conflicts with the new version."""
        try:
            # Get recent versions for conflict detection
            recent_versions = self.version_manager.get_version_history(
                dataset_name, symbol, limit=5
            )

            if len(recent_versions) < 2:
                return []  # No conflicts possible with single version

            # Create data ranges for conflict detection
            ranges = []
            for version in recent_versions:
                # Extract date range from version metadata or use timestamp
                start_date = version.timestamp.date()
                end_date = start_date  # Single day for simplicity

                data_range = DataRange(
                    dataset_name=dataset_name,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    data_hash=version.data_hash,
                    version=version.version_number,
                    timestamp=version.timestamp,
                    metadata=version.metadata
                )
                ranges.append(data_range)

            # Detect conflicts
            conflicts = self.conflict_resolver.detect_conflicts(ranges)

            # Auto-resolve conflicts if enabled
            for conflict in conflicts:
                try:
                    self.conflict_resolver.resolve_conflict(conflict.conflict_id)
                    self.download_statistics['conflicts_resolved'] += 1
                except Exception as e:
                    logger.warning(f"Auto-resolution failed for conflict {conflict.conflict_id}: {e}")

            return conflicts

        except Exception as e:
            logger.error(f"Error checking for conflicts: {e}")
            return []

    def _handle_download_failure(self,
                               dataset_spec: DatasetSpecification,
                               symbol: Optional[str],
                               download_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle download failure."""
        logger.error(f"Download failed for {dataset_spec.name}:{symbol}: {download_result.get('error')}")

        # Record failed attempt
        self.state_tracker.record_download_attempt(
            dataset_name=dataset_spec.name,
            symbol=symbol,
            download_date=datetime.utcnow().date(),
            modification_time=datetime.utcnow(),
            data_hash="",
            version=0,
            operation_type="incremental_download",
            success=False,
            error_message=download_result.get('error'),
            metadata={'download_result': download_result}
        )

        # Check if automatic rollback should be triggered
        if self.config.enable_automatic_rollback:
            self._consider_automatic_rollback(dataset_spec.name, symbol)

        return {
            'status': 'failed',
            'dataset': dataset_spec.name,
            'symbol': symbol,
            'error': download_result.get('error'),
            'timestamp': datetime.utcnow().isoformat()
        }

    def _consider_automatic_rollback(self, dataset_name: str, symbol: Optional[str]) -> None:
        """Consider if automatic rollback should be triggered."""
        try:
            # Get recent download history
            history = self.state_tracker.get_download_history(
                dataset_name, symbol, limit=10
            )

            if len(history) < 3:
                return  # Not enough history for decision

            # Calculate recent failure rate
            recent_attempts = history[:5]  # Last 5 attempts
            failures = sum(1 for attempt in recent_attempts if not attempt['success'])
            failure_rate = failures / len(recent_attempts)

            if failure_rate >= self.config.rollback_on_error_threshold:
                logger.warning(
                    f"High failure rate ({failure_rate:.1%}) detected for {dataset_name}:{symbol}, "
                    "triggering automatic rollback"
                )

                # Find last successful version
                for attempt in history:
                    if attempt['success']:
                        try:
                            self.rollback_manager.rollback_to_checkpoint(
                                dataset_name=dataset_name,
                                symbol=symbol,
                                reason=f"Automatic rollback due to high failure rate ({failure_rate:.1%})"
                            )
                            self.download_statistics['rollbacks_performed'] += 1
                            break
                        except Exception as e:
                            logger.error(f"Automatic rollback failed: {e}")
                        break

        except Exception as e:
            logger.error(f"Error considering automatic rollback: {e}")

    def get_dataset_status(self, dataset_name: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive status for a dataset."""
        try:
            checkpoint = self.state_tracker.get_checkpoint(dataset_name, symbol)
            latest_version = self.version_manager.get_latest_version(dataset_name, symbol)
            download_history = self.state_tracker.get_download_history(dataset_name, symbol, limit=5)

            # Calculate freshness score
            timestamps = {}
            if checkpoint:
                timestamps['checkpoint'] = self.timestamp_manager.parse_timestamp(
                    checkpoint.last_modification_time, "checkpoint"
                )

            freshness_score = self.timestamp_manager.get_data_freshness_score(timestamps)

            # Get pending conflicts
            conflicts = self.conflict_resolver.get_conflicts(dataset_name, resolved=False)

            return {
                'dataset_name': dataset_name,
                'symbol': symbol,
                'checkpoint': checkpoint.to_dict() if checkpoint else None,
                'latest_version': latest_version.to_dict() if latest_version else None,
                'freshness_score': freshness_score,
                'download_history_count': len(download_history),
                'pending_conflicts': len(conflicts),
                'last_download': download_history[0] if download_history else None,
                'status': checkpoint.state.value if checkpoint else 'no_data'
            }

        except Exception as e:
            logger.error(f"Error getting dataset status: {e}")
            return {
                'dataset_name': dataset_name,
                'symbol': symbol,
                'status': 'error',
                'error': str(e)
            }

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        try:
            return {
                'download_statistics': self.download_statistics,
                'state_tracker': self.state_tracker.get_statistics(),
                'version_manager': self.version_manager.get_storage_statistics(),
                'queue_manager': self.queue_manager.get_statistics(),
                'conflict_resolver': self.conflict_resolver.get_statistics(),
                'rollback_manager': self.rollback_manager.get_statistics(),
                'change_detector': self.change_detector.get_change_statistics(),
                'last_updated': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting comprehensive statistics: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }

    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data across all components."""
        cleanup_results = {}

        try:
            # Cleanup state tracker history
            cleanup_results['state_history'] = self.state_tracker.cleanup_old_history(days_to_keep)

            # Cleanup queue manager completed tasks
            cleanup_results['queue_tasks'] = self.queue_manager.cleanup_completed_tasks(
                max_age_hours=days_to_keep * 24
            )

            # Cleanup conflict resolver
            cleanup_results['conflicts'] = self.conflict_resolver.cleanup_old_conflicts(days_to_keep)

            logger.info(f"Cleanup completed: {cleanup_results}")
            return cleanup_results

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {'error': str(e)}

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()