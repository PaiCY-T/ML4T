"""
Rollback capability for failed incremental updates.

This module provides comprehensive rollback functionality to revert to previous
data states when updates fail or produce invalid results.
"""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .state_tracker import StateTracker, DataState, DownloadCheckpoint
from .version_manager import VersionManager, DataVersion

logger = logging.getLogger(__name__)


class RollbackType(Enum):
    """Types of rollback operations."""
    VERSION_ROLLBACK = "version_rollback"       # Roll back to specific version
    CHECKPOINT_ROLLBACK = "checkpoint_rollback" # Roll back to checkpoint
    TIME_ROLLBACK = "time_rollback"             # Roll back to specific time
    SELECTIVE_ROLLBACK = "selective_rollback"   # Roll back specific datasets/symbols


class RollbackStatus(Enum):
    """Status of rollback operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RollbackOperation:
    """Represents a rollback operation."""
    operation_id: str
    rollback_type: RollbackType
    target_identifier: str  # Version ID, checkpoint, or timestamp
    datasets: List[str]
    symbols: List[Optional[str]]
    reason: str
    status: RollbackStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['rollback_type'] = self.rollback_type.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackOperation':
        """Create from dictionary."""
        return cls(
            operation_id=data['operation_id'],
            rollback_type=RollbackType(data['rollback_type']),
            target_identifier=data['target_identifier'],
            datasets=data['datasets'],
            symbols=data['symbols'],
            reason=data['reason'],
            status=RollbackStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            error_message=data.get('error_message'),
            metadata=data.get('metadata', {})
        )


class RollbackManager:
    """
    Advanced rollback management system for incremental downloads.

    Features:
    - Multiple rollback strategies
    - Atomic rollback operations
    - Rollback history tracking
    - Validation before rollback
    - Recovery from failed rollbacks
    """

    def __init__(self,
                 state_tracker: StateTracker,
                 version_manager: VersionManager,
                 storage_directory: Union[str, Path]):
        """
        Initialize rollback manager.

        Args:
            state_tracker: State tracking system
            version_manager: Version management system
            storage_directory: Directory for rollback operation storage
        """
        self.state_tracker = state_tracker
        self.version_manager = version_manager
        self.storage_directory = Path(storage_directory)
        self.storage_directory.mkdir(parents=True, exist_ok=True)

        # Rollback operation tracking
        self.operations: Dict[str, RollbackOperation] = {}
        self._load_operations()

    def rollback_to_version(self,
                           dataset_name: str,
                           version_id: str,
                           symbol: Optional[str] = None,
                           reason: str = "Manual rollback") -> str:
        """
        Roll back to a specific version.

        Args:
            dataset_name: Name of the dataset
            version_id: Target version ID
            symbol: Optional symbol filter
            reason: Reason for rollback

        Returns:
            Rollback operation ID

        Raises:
            ValueError: If version not found or invalid
        """
        # Validate version exists
        version = self.version_manager.get_version(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")

        if version.dataset_name != dataset_name:
            raise ValueError(f"Version {version_id} is not for dataset {dataset_name}")

        if symbol is not None and version.symbol != symbol:
            raise ValueError(f"Version {version_id} is not for symbol {symbol}")

        # Create rollback operation
        operation = RollbackOperation(
            operation_id=str(uuid.uuid4()),
            rollback_type=RollbackType.VERSION_ROLLBACK,
            target_identifier=version_id,
            datasets=[dataset_name],
            symbols=[symbol],
            reason=reason,
            status=RollbackStatus.PENDING,
            created_at=datetime.utcnow(),
            metadata={
                'target_version': version.to_dict()
            }
        )

        # Register operation
        self.operations[operation.operation_id] = operation
        self._save_operations()

        logger.info(f"Created rollback operation {operation.operation_id} to version {version_id}")
        return operation.operation_id

    def rollback_to_checkpoint(self,
                              dataset_name: str,
                              symbol: Optional[str] = None,
                              checkpoint_date: Optional[datetime] = None,
                              reason: str = "Checkpoint rollback") -> str:
        """
        Roll back to a checkpoint.

        Args:
            dataset_name: Name of the dataset
            symbol: Optional symbol filter
            checkpoint_date: Target checkpoint date (latest if None)
            reason: Reason for rollback

        Returns:
            Rollback operation ID

        Raises:
            ValueError: If checkpoint not found
        """
        # Find target checkpoint
        checkpoint = self.state_tracker.get_checkpoint(dataset_name, symbol)
        if not checkpoint:
            raise ValueError(f"No checkpoint found for {dataset_name}:{symbol}")

        if checkpoint_date and checkpoint.last_modification_time > checkpoint_date:
            # Need to find an older checkpoint - this would require checkpoint history
            raise ValueError("Historical checkpoint rollback not yet implemented")

        # Create rollback operation
        operation = RollbackOperation(
            operation_id=str(uuid.uuid4()),
            rollback_type=RollbackType.CHECKPOINT_ROLLBACK,
            target_identifier=checkpoint.data_hash,
            datasets=[dataset_name],
            symbols=[symbol],
            reason=reason,
            status=RollbackStatus.PENDING,
            created_at=datetime.utcnow(),
            metadata={
                'target_checkpoint': checkpoint.to_dict()
            }
        )

        # Register operation
        self.operations[operation.operation_id] = operation
        self._save_operations()

        logger.info(f"Created rollback operation {operation.operation_id} to checkpoint")
        return operation.operation_id

    def rollback_to_time(self,
                        target_time: datetime,
                        datasets: List[str],
                        symbols: Optional[List[str]] = None,
                        reason: str = "Time-based rollback") -> str:
        """
        Roll back to a specific point in time.

        Args:
            target_time: Target timestamp
            datasets: List of datasets to rollback
            symbols: Optional list of symbols to rollback
            reason: Reason for rollback

        Returns:
            Rollback operation ID
        """
        # Create rollback operation
        operation = RollbackOperation(
            operation_id=str(uuid.uuid4()),
            rollback_type=RollbackType.TIME_ROLLBACK,
            target_identifier=target_time.isoformat(),
            datasets=datasets,
            symbols=symbols or [None] * len(datasets),
            reason=reason,
            status=RollbackStatus.PENDING,
            created_at=datetime.utcnow(),
            metadata={
                'target_time': target_time.isoformat()
            }
        )

        # Register operation
        self.operations[operation.operation_id] = operation
        self._save_operations()

        logger.info(f"Created time-based rollback operation {operation.operation_id}")
        return operation.operation_id

    def execute_rollback(self, operation_id: str) -> bool:
        """
        Execute a rollback operation.

        Args:
            operation_id: ID of operation to execute

        Returns:
            True if successful

        Raises:
            ValueError: If operation not found or invalid
        """
        operation = self.operations.get(operation_id)
        if not operation:
            raise ValueError(f"Rollback operation {operation_id} not found")

        if operation.status != RollbackStatus.PENDING:
            raise ValueError(f"Operation {operation_id} is not in pending status")

        try:
            # Update status
            operation.status = RollbackStatus.IN_PROGRESS
            operation.started_at = datetime.utcnow()
            self._save_operations()

            logger.info(f"Executing rollback operation {operation_id}")

            # Execute based on type
            if operation.rollback_type == RollbackType.VERSION_ROLLBACK:
                success = self._execute_version_rollback(operation)
            elif operation.rollback_type == RollbackType.CHECKPOINT_ROLLBACK:
                success = self._execute_checkpoint_rollback(operation)
            elif operation.rollback_type == RollbackType.TIME_ROLLBACK:
                success = self._execute_time_rollback(operation)
            else:
                raise ValueError(f"Unsupported rollback type: {operation.rollback_type}")

            # Update final status
            operation.status = RollbackStatus.COMPLETED if success else RollbackStatus.FAILED
            operation.completed_at = datetime.utcnow()
            self._save_operations()

            if success:
                logger.info(f"Rollback operation {operation_id} completed successfully")
            else:
                logger.error(f"Rollback operation {operation_id} failed")

            return success

        except Exception as e:
            logger.error(f"Error executing rollback operation {operation_id}: {e}")
            operation.status = RollbackStatus.FAILED
            operation.error_message = str(e)
            operation.completed_at = datetime.utcnow()
            self._save_operations()
            return False

    def cancel_rollback(self, operation_id: str) -> bool:
        """
        Cancel a pending rollback operation.

        Args:
            operation_id: ID of operation to cancel

        Returns:
            True if successfully cancelled
        """
        operation = self.operations.get(operation_id)
        if not operation:
            return False

        if operation.status not in [RollbackStatus.PENDING, RollbackStatus.IN_PROGRESS]:
            return False

        operation.status = RollbackStatus.CANCELLED
        operation.completed_at = datetime.utcnow()
        self._save_operations()

        logger.info(f"Cancelled rollback operation {operation_id}")
        return True

    def get_rollback_history(self, limit: int = 100) -> List[RollbackOperation]:
        """
        Get rollback operation history.

        Args:
            limit: Maximum number of operations to return

        Returns:
            List of rollback operations
        """
        operations = sorted(
            self.operations.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return operations[:limit]

    def validate_rollback_safety(self, operation_id: str) -> Dict[str, Any]:
        """
        Validate that a rollback operation is safe to execute.

        Args:
            operation_id: ID of operation to validate

        Returns:
            Validation result dictionary
        """
        operation = self.operations.get(operation_id)
        if not operation:
            return {
                'safe': False,
                'errors': ['Operation not found'],
                'warnings': []
            }

        errors = []
        warnings = []

        # Check if target exists
        if operation.rollback_type == RollbackType.VERSION_ROLLBACK:
            version = self.version_manager.get_version(operation.target_identifier)
            if not version:
                errors.append(f"Target version {operation.target_identifier} not found")

        # Check for data dependencies
        for i, dataset_name in enumerate(operation.datasets):
            symbol = operation.symbols[i] if i < len(operation.symbols) else None

            # Check if there are newer updates that would be lost
            latest_version = self.version_manager.get_latest_version(dataset_name, symbol)
            if latest_version:
                if operation.rollback_type == RollbackType.VERSION_ROLLBACK:
                    target_version = self.version_manager.get_version(operation.target_identifier)
                    if target_version and latest_version.version_number > target_version.version_number:
                        warnings.append(
                            f"Rolling back {dataset_name}:{symbol} will lose "
                            f"{latest_version.version_number - target_version.version_number} newer versions"
                        )

        # Check for ongoing operations
        pending_ops = [op for op in self.operations.values()
                      if op.status in [RollbackStatus.PENDING, RollbackStatus.IN_PROGRESS]
                      and op.operation_id != operation_id]

        if pending_ops:
            warnings.append(f"There are {len(pending_ops)} other pending/active rollback operations")

        return {
            'safe': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def _execute_version_rollback(self, operation: RollbackOperation) -> bool:
        """Execute version-based rollback."""
        try:
            version_id = operation.target_identifier
            version = self.version_manager.get_version(version_id)

            if not version:
                operation.error_message = f"Target version {version_id} not found"
                return False

            # Load version data
            version_data = self.version_manager.load_version_data(version_id)
            if version_data is None:
                operation.error_message = f"Could not load data for version {version_id}"
                return False

            # Create new checkpoint with rollback data
            checkpoint = DownloadCheckpoint(
                dataset_name=version.dataset_name,
                symbol=version.symbol,
                last_download_date=version.timestamp.date(),
                last_modification_time=version.timestamp,
                data_hash=version.data_hash,
                version=version.version_number,
                state=DataState.COMPLETED,
                metadata={
                    'rollback_operation_id': operation.operation_id,
                    'rollback_from_version': version_id,
                    'rollback_reason': operation.reason
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Save checkpoint
            self.state_tracker.save_checkpoint(checkpoint)

            # Record rollback in history
            self.state_tracker.record_download_attempt(
                dataset_name=version.dataset_name,
                symbol=version.symbol,
                download_date=datetime.utcnow().date(),
                modification_time=version.timestamp,
                data_hash=version.data_hash,
                version=version.version_number,
                operation_type="rollback",
                success=True,
                metadata=operation.metadata
            )

            return True

        except Exception as e:
            operation.error_message = f"Version rollback failed: {e}"
            logger.error(f"Version rollback failed: {e}")
            return False

    def _execute_checkpoint_rollback(self, operation: RollbackOperation) -> bool:
        """Execute checkpoint-based rollback."""
        try:
            # Get target checkpoint from metadata
            target_checkpoint_data = operation.metadata.get('target_checkpoint')
            if not target_checkpoint_data:
                operation.error_message = "Target checkpoint data not found in operation metadata"
                return False

            target_checkpoint = DownloadCheckpoint.from_dict(target_checkpoint_data)

            # Create rollback checkpoint
            rollback_checkpoint = DownloadCheckpoint(
                dataset_name=target_checkpoint.dataset_name,
                symbol=target_checkpoint.symbol,
                last_download_date=target_checkpoint.last_download_date,
                last_modification_time=target_checkpoint.last_modification_time,
                data_hash=target_checkpoint.data_hash,
                version=target_checkpoint.version,
                state=DataState.COMPLETED,
                metadata={
                    'rollback_operation_id': operation.operation_id,
                    'rollback_from_checkpoint': target_checkpoint.data_hash,
                    'rollback_reason': operation.reason,
                    'original_checkpoint': target_checkpoint_data
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Save checkpoint
            self.state_tracker.save_checkpoint(rollback_checkpoint)

            # Record rollback in history
            self.state_tracker.record_download_attempt(
                dataset_name=target_checkpoint.dataset_name,
                symbol=target_checkpoint.symbol,
                download_date=datetime.utcnow().date(),
                modification_time=target_checkpoint.last_modification_time,
                data_hash=target_checkpoint.data_hash,
                version=target_checkpoint.version,
                operation_type="checkpoint_rollback",
                success=True,
                metadata=operation.metadata
            )

            return True

        except Exception as e:
            operation.error_message = f"Checkpoint rollback failed: {e}"
            logger.error(f"Checkpoint rollback failed: {e}")
            return False

    def _execute_time_rollback(self, operation: RollbackOperation) -> bool:
        """Execute time-based rollback."""
        try:
            target_time = datetime.fromisoformat(operation.target_identifier)
            success_count = 0
            total_count = len(operation.datasets)

            for i, dataset_name in enumerate(operation.datasets):
                symbol = operation.symbols[i] if i < len(operation.symbols) else None

                try:
                    # Find version at or before target time
                    versions = self.version_manager.get_version_history(dataset_name, symbol)
                    target_version = None

                    for version in reversed(versions):  # Start from oldest
                        if version.timestamp <= target_time:
                            target_version = version
                        else:
                            break

                    if not target_version:
                        logger.warning(f"No version found before {target_time} for {dataset_name}:{symbol}")
                        continue

                    # Create checkpoint for this version
                    checkpoint = DownloadCheckpoint(
                        dataset_name=dataset_name,
                        symbol=symbol,
                        last_download_date=target_version.timestamp.date(),
                        last_modification_time=target_version.timestamp,
                        data_hash=target_version.data_hash,
                        version=target_version.version_number,
                        state=DataState.COMPLETED,
                        metadata={
                            'rollback_operation_id': operation.operation_id,
                            'rollback_target_time': target_time.isoformat(),
                            'rollback_reason': operation.reason
                        },
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )

                    self.state_tracker.save_checkpoint(checkpoint)
                    success_count += 1

                except Exception as e:
                    logger.error(f"Error rolling back {dataset_name}:{symbol}: {e}")

            if success_count == 0:
                operation.error_message = "No datasets were successfully rolled back"
                return False

            if success_count < total_count:
                operation.error_message = f"Only {success_count}/{total_count} datasets were rolled back"

            return success_count > 0

        except Exception as e:
            operation.error_message = f"Time rollback failed: {e}"
            logger.error(f"Time rollback failed: {e}")
            return False

    def _load_operations(self) -> None:
        """Load rollback operations from disk."""
        operations_file = self.storage_directory / "rollback_operations.json"

        if not operations_file.exists():
            return

        try:
            with open(operations_file, 'r') as f:
                operations_data = json.load(f)

            for operation_data in operations_data:
                operation = RollbackOperation.from_dict(operation_data)
                self.operations[operation.operation_id] = operation

            logger.info(f"Loaded {len(self.operations)} rollback operations")

        except Exception as e:
            logger.error(f"Error loading rollback operations: {e}")

    def _save_operations(self) -> None:
        """Save rollback operations to disk."""
        operations_file = self.storage_directory / "rollback_operations.json"

        try:
            operations_data = [op.to_dict() for op in self.operations.values()]

            with open(operations_file, 'w') as f:
                json.dump(operations_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving rollback operations: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get rollback operation statistics."""
        if not self.operations:
            return {
                'total_operations': 0,
                'operations_by_status': {},
                'operations_by_type': {},
                'success_rate': 0.0,
                'average_execution_time': 0.0
            }

        status_counts = {}
        type_counts = {}
        execution_times = []

        for operation in self.operations.values():
            # Count by status
            status = operation.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            # Count by type
            op_type = operation.rollback_type.value
            type_counts[op_type] = type_counts.get(op_type, 0) + 1

            # Calculate execution time
            if operation.started_at and operation.completed_at:
                execution_time = (operation.completed_at - operation.started_at).total_seconds()
                execution_times.append(execution_time)

        completed_count = status_counts.get('completed', 0)
        failed_count = status_counts.get('failed', 0)
        total_finished = completed_count + failed_count

        success_rate = completed_count / total_finished if total_finished > 0 else 0.0
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

        return {
            'total_operations': len(self.operations),
            'operations_by_status': status_counts,
            'operations_by_type': type_counts,
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'storage_directory': str(self.storage_directory)
        }