"""
Incremental download queue management system.

This module provides a sophisticated queue management system for coordinating
incremental downloads with priority handling, scheduling, and resource management.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue, Empty
from concurrent.futures import ThreadPoolExecutor, Future
import uuid

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Download priority levels."""
    CRITICAL = 1    # Real-time data updates
    HIGH = 2        # End-of-day data
    MEDIUM = 3      # Historical backfills
    LOW = 4         # Non-urgent updates


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class DownloadTask:
    """Represents a download task in the queue."""
    task_id: str
    dataset_name: str
    symbol: Optional[str]
    priority: Priority
    scheduled_time: datetime
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None

    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def __lt__(self, other):
        """Priority queue comparison."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.scheduled_time < other.scheduled_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'dataset_name': self.dataset_name,
            'symbol': self.symbol,
            'priority': self.priority.value,
            'scheduled_time': self.scheduled_time.isoformat(),
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }


@dataclass
class QueueStatistics:
    """Queue statistics and performance metrics."""
    total_tasks: int = 0
    pending_tasks: int = 0
    in_progress_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    average_completion_time: float = 0.0
    success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class IncrementalQueueManager:
    """
    Advanced queue management system for incremental downloads.

    Features:
    - Priority-based task scheduling
    - Dependency management
    - Automatic retry logic
    - Resource throttling
    - Performance monitoring
    - Concurrent execution
    """

    def __init__(self,
                 max_workers: int = 4,
                 max_queue_size: int = 1000,
                 default_timeout: int = 300):
        """
        Initialize queue manager.

        Args:
            max_workers: Maximum number of concurrent workers
            max_queue_size: Maximum number of tasks in queue
            default_timeout: Default task timeout in seconds
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout

        # Queue and task management
        self._task_queue = PriorityQueue(maxsize=max_queue_size)
        self._task_registry: Dict[str, DownloadTask] = {}
        self._dependency_graph: Dict[str, List[str]] = {}

        # Thread safety
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        # Worker management
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_futures: Dict[str, Future] = {}

        # Statistics
        self._stats = QueueStatistics()

        # Background worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._running = False

    def start(self) -> None:
        """Start the queue manager."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._worker_thread.start()
        logger.info("Queue manager started")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the queue manager."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        # Cancel pending tasks
        self._cancel_all_pending_tasks()

        # Wait for worker thread to finish
        self._worker_thread.join(timeout=timeout)

        # Shutdown executor
        self._executor.shutdown(wait=True, timeout=timeout)

        logger.info("Queue manager stopped")

    def submit_task(self, task: DownloadTask) -> bool:
        """
        Submit a task to the queue.

        Args:
            task: Task to submit

        Returns:
            True if task was successfully queued
        """
        with self._lock:
            if self._task_queue.qsize() >= self.max_queue_size:
                logger.warning(f"Queue is full, cannot submit task {task.task_id}")
                return False

            # Register task
            self._task_registry[task.task_id] = task

            # Add to dependency graph
            if task.dependencies:
                self._dependency_graph[task.task_id] = task.dependencies.copy()

            # Check if task can be executed immediately
            if self._can_execute_task(task):
                try:
                    self._task_queue.put(task, block=False)
                    self._update_stats()
                    logger.debug(f"Task {task.task_id} queued for execution")
                    return True
                except Exception as e:
                    logger.error(f"Error queuing task {task.task_id}: {e}")
                    return False
            else:
                logger.debug(f"Task {task.task_id} waiting for dependencies")
                return True

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if task was successfully cancelled
        """
        with self._lock:
            task = self._task_registry.get(task_id)
            if not task:
                return False

            if task.status == TaskStatus.IN_PROGRESS:
                # Cancel running future
                future = self._active_futures.get(task_id)
                if future:
                    future.cancel()
                    del self._active_futures[task_id]

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()

            # Remove from dependency graph
            if task_id in self._dependency_graph:
                del self._dependency_graph[task_id]

            self._update_stats()
            logger.info(f"Task {task_id} cancelled")
            return True

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        task = self._task_registry.get(task_id)
        return task.status if task else None

    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a task."""
        task = self._task_registry.get(task_id)
        return task.to_dict() if task else None

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """
        List tasks, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of task information dictionaries
        """
        with self._lock:
            tasks = []
            for task in self._task_registry.values():
                if status is None or task.status == status:
                    tasks.append(task.to_dict())
            return tasks

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            self._update_stats()
            return {
                'total_tasks': self._stats.total_tasks,
                'pending_tasks': self._stats.pending_tasks,
                'in_progress_tasks': self._stats.in_progress_tasks,
                'completed_tasks': self._stats.completed_tasks,
                'failed_tasks': self._stats.failed_tasks,
                'cancelled_tasks': self._stats.cancelled_tasks,
                'queue_size': self._task_queue.qsize(),
                'max_queue_size': self.max_queue_size,
                'active_workers': len(self._active_futures),
                'max_workers': self.max_workers,
                'success_rate': self._stats.success_rate,
                'average_completion_time': self._stats.average_completion_time,
                'last_updated': self._stats.last_updated.isoformat()
            }

    def schedule_task(self,
                     dataset_name: str,
                     symbol: Optional[str] = None,
                     priority: Priority = Priority.MEDIUM,
                     delay_seconds: int = 0,
                     dependencies: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     callback: Optional[Callable] = None) -> str:
        """
        Schedule a download task.

        Args:
            dataset_name: Name of dataset to download
            symbol: Optional symbol filter
            priority: Task priority
            delay_seconds: Delay before execution
            dependencies: List of task IDs this task depends on
            metadata: Additional task metadata
            callback: Optional completion callback

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        scheduled_time = datetime.utcnow() + timedelta(seconds=delay_seconds)

        task = DownloadTask(
            task_id=task_id,
            dataset_name=dataset_name,
            symbol=symbol,
            priority=priority,
            scheduled_time=scheduled_time,
            dependencies=dependencies or [],
            metadata=metadata or {},
            callback=callback,
            timeout_seconds=self.default_timeout
        )

        if self.submit_task(task):
            return task_id
        else:
            raise RuntimeError(f"Failed to schedule task {task_id}")

    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        while not self._stop_event.is_set():
            try:
                # Get next task with timeout
                try:
                    task = self._task_queue.get(timeout=1.0)
                except Empty:
                    continue

                # Check if it's time to execute the task
                if datetime.utcnow() < task.scheduled_time:
                    # Put task back and wait
                    self._task_queue.put(task)
                    time.sleep(0.1)
                    continue

                # Execute task
                self._execute_task(task)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1.0)

    def _execute_task(self, task: DownloadTask) -> None:
        """Execute a download task."""
        with self._lock:
            if task.status != TaskStatus.PENDING:
                return

            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()

        logger.info(f"Executing task {task.task_id}: {task.dataset_name}:{task.symbol}")

        # Submit to thread pool
        future = self._executor.submit(self._run_task, task)
        self._active_futures[task.task_id] = future

        # Handle completion
        future.add_done_callback(lambda f: self._handle_task_completion(task, f))

    def _run_task(self, task: DownloadTask) -> Any:
        """Run the actual download task."""
        try:
            # This would be implemented by the actual downloader
            # For now, simulate work
            time.sleep(1.0)

            # Simulate success/failure based on retry count
            if task.retry_count > 0 and task.retry_count % 2 == 0:
                raise Exception(f"Simulated failure for task {task.task_id}")

            return {"status": "success", "task_id": task.task_id}

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            raise

    def _handle_task_completion(self, task: DownloadTask, future: Future) -> None:
        """Handle task completion or failure."""
        with self._lock:
            # Remove from active futures
            if task.task_id in self._active_futures:
                del self._active_futures[task.task_id]

            try:
                result = future.result()
                # Task succeeded
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()

                logger.info(f"Task {task.task_id} completed successfully")

                # Execute callback if provided
                if task.callback:
                    try:
                        task.callback(task, result)
                    except Exception as e:
                        logger.error(f"Error in task callback: {e}")

            except Exception as e:
                # Task failed
                task.error_message = str(e)

                if task.retry_count < task.max_retries:
                    # Retry the task
                    task.retry_count += 1
                    task.status = TaskStatus.RETRYING
                    task.scheduled_time = datetime.utcnow() + timedelta(seconds=2 ** task.retry_count)

                    logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")

                    # Requeue for retry
                    try:
                        self._task_queue.put(task, block=False)
                    except Exception:
                        task.status = TaskStatus.FAILED
                        task.completed_at = datetime.utcnow()
                        logger.error(f"Failed to requeue task {task.task_id}")
                else:
                    # Max retries exceeded
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.utcnow()
                    logger.error(f"Task {task.task_id} failed permanently after {task.max_retries} retries")

            # Check for dependent tasks that can now be executed
            self._check_dependent_tasks(task.task_id)
            self._update_stats()

    def _can_execute_task(self, task: DownloadTask) -> bool:
        """Check if a task can be executed (all dependencies satisfied)."""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            dep_task = self._task_registry.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False

        return True

    def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """Check for tasks that can now be executed after a task completes."""
        for task_id, task in self._task_registry.items():
            if (task.status == TaskStatus.PENDING and
                completed_task_id in task.dependencies and
                self._can_execute_task(task)):

                try:
                    self._task_queue.put(task, block=False)
                    logger.debug(f"Queued dependent task {task_id}")
                except Exception as e:
                    logger.error(f"Error queuing dependent task {task_id}: {e}")

    def _cancel_all_pending_tasks(self) -> None:
        """Cancel all pending tasks."""
        with self._lock:
            for task in self._task_registry.values():
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = datetime.utcnow()

    def _update_stats(self) -> None:
        """Update queue statistics."""
        status_counts = {status: 0 for status in TaskStatus}
        completion_times = []

        for task in self._task_registry.values():
            status_counts[task.status] += 1

            if task.status == TaskStatus.COMPLETED and task.started_at and task.completed_at:
                completion_time = (task.completed_at - task.started_at).total_seconds()
                completion_times.append(completion_time)

        self._stats.total_tasks = len(self._task_registry)
        self._stats.pending_tasks = status_counts[TaskStatus.PENDING]
        self._stats.in_progress_tasks = status_counts[TaskStatus.IN_PROGRESS]
        self._stats.completed_tasks = status_counts[TaskStatus.COMPLETED]
        self._stats.failed_tasks = status_counts[TaskStatus.FAILED]
        self._stats.cancelled_tasks = status_counts[TaskStatus.CANCELLED]

        if completion_times:
            self._stats.average_completion_time = sum(completion_times) / len(completion_times)

        total_finished = self._stats.completed_tasks + self._stats.failed_tasks
        if total_finished > 0:
            self._stats.success_rate = self._stats.completed_tasks / total_finished

        self._stats.last_updated = datetime.utcnow()

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed tasks.

        Args:
            max_age_hours: Maximum age in hours for keeping completed tasks

        Returns:
            Number of tasks cleaned up
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned_count = 0

        with self._lock:
            tasks_to_remove = []

            for task_id, task in self._task_registry.items():
                if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task.completed_at and task.completed_at < cutoff_time):
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self._task_registry[task_id]
                if task_id in self._dependency_graph:
                    del self._dependency_graph[task_id]
                cleaned_count += 1

            self._update_stats()

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old tasks")

        return cleaned_count