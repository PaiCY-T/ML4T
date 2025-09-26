"""
Automated Scheduling Engine for Financial Statement Downloads.

Provides intelligent scheduling, retry mechanisms, and download orchestration
for financial statement data based on industry-specific reporting cycles.
"""

import logging
import asyncio
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Callable, Any
from pathlib import Path
import json

from ..core.exceptions import SchedulingError, ValidationError
from ..core.dataset import DatasetCatalog, DownloadMethod
from ..config.dataset_schema import DatasetDownloadConfig
from .financial_calendar import (
    FinancialReportingCalendar,
    FinancialStatementSchedule,
    IndustryType,
    ReportingPeriod
)
from ..calendar.holiday_adjustments import HolidayAdjustmentEngine, AdjustmentRule

logger = logging.getLogger(__name__)


class ScheduleStatus(Enum):
    """Status of a scheduled download."""
    PENDING = "pending"                # Waiting to be executed
    SCHEDULED = "scheduled"            # Scheduled for execution
    RUNNING = "running"               # Currently executing
    COMPLETED = "completed"           # Successfully completed
    FAILED = "failed"                 # Failed execution
    RETRYING = "retrying"             # Retrying after failure
    CANCELLED = "cancelled"           # Manually cancelled
    OVERDUE = "overdue"               # Past deadline


class SchedulePriority(Enum):
    """Priority levels for scheduled downloads."""
    CRITICAL = 1    # Must execute immediately
    HIGH = 2        # High priority
    NORMAL = 3      # Normal priority
    LOW = 4         # Low priority
    DEFER = 5       # Can be deferred


@dataclass
class ScheduleRequest:
    """Request to schedule a financial statement download."""
    company_ticker: str
    dataset_names: List[str]
    industry_type: IndustryType
    reporting_period: ReportingPeriod
    fiscal_year: int
    priority: SchedulePriority = SchedulePriority.NORMAL
    earliest_date: Optional[date] = None
    latest_date: Optional[date] = None
    retry_count: int = 3
    retry_delay_minutes: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.company_ticker:
            raise ValueError("company_ticker is required")
        if not self.dataset_names:
            raise ValueError("dataset_names cannot be empty")


@dataclass
class ScheduleResult:
    """Result of a scheduled download operation."""
    request: ScheduleRequest
    schedule_id: str
    status: ScheduleStatus
    scheduled_time: datetime
    actual_start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    downloaded_datasets: List[str] = field(default_factory=list)
    failed_datasets: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate of dataset downloads."""
        total = len(self.downloaded_datasets) + len(self.failed_datasets)
        if total == 0:
            return 0.0
        return len(self.downloaded_datasets) / total

    @property
    def is_completed(self) -> bool:
        """Check if the schedule is completed (successfully or failed)."""
        return self.status in [ScheduleStatus.COMPLETED, ScheduleStatus.FAILED, ScheduleStatus.CANCELLED]

    @property
    def duration_minutes(self) -> float:
        """Get execution duration in minutes."""
        return self.execution_time_seconds / 60.0


@dataclass
class ScheduleTask:
    """Internal task representation for the scheduler."""
    schedule_id: str
    request: ScheduleRequest
    scheduled_time: datetime
    status: ScheduleStatus = ScheduleStatus.PENDING
    retry_count: int = 0
    last_error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def update_status(self, new_status: ScheduleStatus, error: Optional[str] = None):
        """Update task status."""
        self.status = new_status
        self.updated_at = datetime.now()
        if error:
            self.last_error = error


class SchedulerEngine:
    """
    Automated scheduling engine for financial statement downloads.

    Provides:
    - Industry-aware scheduling based on reporting calendars
    - Holiday and trading day adjustments
    - Retry mechanisms with exponential backoff
    - Priority-based execution
    - Download orchestration
    """

    def __init__(self,
                 dataset_catalog: DatasetCatalog,
                 config: DatasetDownloadConfig,
                 data_directory: Optional[Union[str, Path]] = None):
        """
        Initialize scheduler engine.

        Args:
            dataset_catalog: Dataset catalog for validation
            config: Download configuration
            data_directory: Directory for storing schedule data
        """
        self.dataset_catalog = dataset_catalog
        self.config = config
        self.data_directory = Path(data_directory) if data_directory else Path("./schedules")
        self.data_directory.mkdir(parents=True, exist_ok=True)

        # Core components
        self.financial_calendar = FinancialReportingCalendar()
        self.holiday_adjuster = HolidayAdjustmentEngine()

        # Scheduling state
        self._tasks: Dict[str, ScheduleTask] = {}
        self._running_tasks: Set[str] = set()
        self._schedule_counter = 0

        # Event handlers
        self._completion_handlers: List[Callable[[ScheduleResult], None]] = []
        self._error_handlers: List[Callable[[str, str], None]] = []

        logger.info("Initialized Scheduler Engine")

    def schedule_financial_statement_download(self, request: ScheduleRequest) -> str:
        """
        Schedule a financial statement download.

        Args:
            request: Schedule request

        Returns:
            Schedule ID

        Raises:
            SchedulingError: If scheduling fails
        """
        try:
            # Validate request
            self._validate_schedule_request(request)

            # Generate schedule ID
            self._schedule_counter += 1
            schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._schedule_counter:04d}"

            # Calculate optimal scheduling time
            scheduled_time = self._calculate_optimal_schedule_time(request)

            # Create task
            task = ScheduleTask(
                schedule_id=schedule_id,
                request=request,
                scheduled_time=scheduled_time,
                status=ScheduleStatus.SCHEDULED
            )

            # Store task
            self._tasks[schedule_id] = task
            self._save_task_state(task)

            logger.info(f"Scheduled download for {request.company_ticker} {request.reporting_period.value} "
                       f"FY{request.fiscal_year} at {scheduled_time}")

            return schedule_id

        except Exception as e:
            error_msg = f"Failed to schedule download: {e}"
            logger.error(error_msg)
            raise SchedulingError(error_msg, cause=e)

    def schedule_bulk_downloads(self, requests: List[ScheduleRequest]) -> List[str]:
        """
        Schedule multiple downloads with optimization.

        Args:
            requests: List of schedule requests

        Returns:
            List of schedule IDs
        """
        schedule_ids = []

        # Sort requests by priority and timing
        sorted_requests = sorted(requests, key=lambda r: (r.priority.value, r.fiscal_year, r.reporting_period.value))

        for request in sorted_requests:
            try:
                schedule_id = self.schedule_financial_statement_download(request)
                schedule_ids.append(schedule_id)
            except SchedulingError as e:
                logger.error(f"Failed to schedule {request.company_ticker}: {e}")
                continue

        logger.info(f"Scheduled {len(schedule_ids)} out of {len(requests)} downloads")
        return schedule_ids

    async def execute_pending_schedules(self) -> List[ScheduleResult]:
        """
        Execute all pending scheduled downloads.

        Returns:
            List of execution results
        """
        current_time = datetime.now()
        pending_tasks = [
            task for task in self._tasks.values()
            if task.status == ScheduleStatus.SCHEDULED and task.scheduled_time <= current_time
        ]

        if not pending_tasks:
            logger.info("No pending schedules to execute")
            return []

        # Sort by priority and scheduled time
        pending_tasks.sort(key=lambda t: (t.request.priority.value, t.scheduled_time))

        results = []
        for task in pending_tasks:
            if task.schedule_id not in self._running_tasks:
                result = await self._execute_task(task)
                results.append(result)

        return results

    async def _execute_task(self, task: ScheduleTask) -> ScheduleResult:
        """
        Execute a scheduled task.

        Args:
            task: Task to execute

        Returns:
            ScheduleResult
        """
        schedule_id = task.schedule_id
        request = task.request

        # Mark as running
        self._running_tasks.add(schedule_id)
        task.update_status(ScheduleStatus.RUNNING)
        self._save_task_state(task)

        result = ScheduleResult(
            request=request,
            schedule_id=schedule_id,
            status=ScheduleStatus.RUNNING,
            scheduled_time=task.scheduled_time,
            actual_start_time=datetime.now()
        )

        try:
            logger.info(f"Executing schedule {schedule_id} for {request.company_ticker}")

            # Execute downloads for each dataset
            downloaded = []
            failed = []

            for dataset_name in request.dataset_names:
                try:
                    success = await self._download_dataset(request.company_ticker, dataset_name)
                    if success:
                        downloaded.append(dataset_name)
                    else:
                        failed.append(dataset_name)
                except Exception as e:
                    logger.error(f"Failed to download {dataset_name}: {e}")
                    failed.append(dataset_name)

            # Update result
            result.downloaded_datasets = downloaded
            result.failed_datasets = failed
            result.completion_time = datetime.now()
            result.execution_time_seconds = (result.completion_time - result.actual_start_time).total_seconds()

            # Determine final status
            if failed and not downloaded:
                result.status = ScheduleStatus.FAILED
                task.update_status(ScheduleStatus.FAILED, f"All {len(failed)} datasets failed")
            elif failed:
                result.status = ScheduleStatus.COMPLETED
                result.error_message = f"Partial success: {len(failed)} datasets failed"
                task.update_status(ScheduleStatus.COMPLETED)
            else:
                result.status = ScheduleStatus.COMPLETED
                task.update_status(ScheduleStatus.COMPLETED)

            logger.info(f"Completed schedule {schedule_id}: {len(downloaded)} succeeded, {len(failed)} failed")

        except Exception as e:
            error_msg = f"Task execution failed: {e}"
            logger.error(error_msg)
            result.status = ScheduleStatus.FAILED
            result.error_message = error_msg
            result.completion_time = datetime.now()
            task.update_status(ScheduleStatus.FAILED, error_msg)

        finally:
            # Clean up
            self._running_tasks.discard(schedule_id)
            self._save_task_state(task)

            # Notify handlers
            self._notify_completion_handlers(result)

        return result

    async def _download_dataset(self, ticker: str, dataset_name: str) -> bool:
        """
        Download a specific dataset for a company.

        This is a placeholder implementation. In practice, this would
        integrate with the actual FinLab download system.

        Args:
            ticker: Company ticker
            dataset_name: Dataset to download

        Returns:
            True if successful
        """
        try:
            # Simulate download process
            await asyncio.sleep(0.1)  # Simulate API call

            # In practice, this would:
            # 1. Get dataset specification from catalog
            # 2. Call appropriate FinLab API
            # 3. Validate and store data
            # 4. Update download history

            dataset = self.dataset_catalog.get_dataset(dataset_name)
            if not dataset:
                logger.warning(f"Dataset {dataset_name} not found in catalog")
                return False

            logger.debug(f"Downloaded {dataset_name} for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Download failed for {ticker}:{dataset_name}: {e}")
            return False

    def _validate_schedule_request(self, request: ScheduleRequest) -> None:
        """
        Validate a schedule request.

        Args:
            request: Request to validate

        Raises:
            ValidationError: If validation fails
        """
        # Validate datasets exist in catalog
        for dataset_name in request.dataset_names:
            if not self.dataset_catalog.get_dataset(dataset_name):
                raise ValidationError(f"Dataset '{dataset_name}' not found in catalog")

        # Validate reporting period for industry
        valid_periods = self.financial_calendar.get_reporting_periods_for_industry(request.industry_type)
        if request.reporting_period not in valid_periods:
            raise ValidationError(
                f"Reporting period {request.reporting_period.value} not valid for {request.industry_type.value}"
            )

        # Validate fiscal year is reasonable
        current_year = datetime.now().year
        if not (current_year - 5 <= request.fiscal_year <= current_year + 2):
            raise ValidationError(f"Fiscal year {request.fiscal_year} is outside reasonable range")

    def _calculate_optimal_schedule_time(self, request: ScheduleRequest) -> datetime:
        """
        Calculate optimal scheduling time for a request.

        Args:
            request: Schedule request

        Returns:
            Optimal scheduling datetime
        """
        # Get financial statement schedule
        fs_schedule = self.financial_calendar.create_schedule(
            request.company_ticker,
            request.industry_type,
            request.reporting_period,
            request.fiscal_year,
            request.priority.value
        )

        if not fs_schedule:
            # Fallback to immediate scheduling
            return datetime.now()

        # Use provided dates or calculated dates
        earliest = request.earliest_date or fs_schedule.earliest_possible
        latest = request.latest_date or fs_schedule.latest_possible

        # Adjust for holidays
        earliest_adj, latest_adj = self.holiday_adjuster.calculate_download_window(earliest, latest)

        # Calculate optimal time within window
        # For high priority, schedule closer to earliest
        # For low priority, schedule closer to expected date
        if request.priority in [SchedulePriority.CRITICAL, SchedulePriority.HIGH]:
            target_date = earliest_adj.adjusted_date
        else:
            target_date = fs_schedule.expected_date

        # Convert to datetime (schedule for early morning to avoid conflicts)
        return datetime.combine(target_date, datetime.min.time().replace(hour=6))

    def _save_task_state(self, task: ScheduleTask) -> None:
        """Save task state to disk."""
        try:
            task_file = self.data_directory / f"{task.schedule_id}.json"
            task_data = {
                "schedule_id": task.schedule_id,
                "company_ticker": task.request.company_ticker,
                "dataset_names": task.request.dataset_names,
                "industry_type": task.request.industry_type.value,
                "reporting_period": task.request.reporting_period.value,
                "fiscal_year": task.request.fiscal_year,
                "priority": task.request.priority.value,
                "scheduled_time": task.scheduled_time.isoformat(),
                "status": task.status.value,
                "retry_count": task.retry_count,
                "last_error": task.last_error,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat()
            }

            with open(task_file, 'w') as f:
                json.dump(task_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save task state for {task.schedule_id}: {e}")

    def add_completion_handler(self, handler: Callable[[ScheduleResult], None]) -> None:
        """Add a completion event handler."""
        self._completion_handlers.append(handler)

    def add_error_handler(self, handler: Callable[[str, str], None]) -> None:
        """Add an error event handler."""
        self._error_handlers.append(handler)

    def _notify_completion_handlers(self, result: ScheduleResult) -> None:
        """Notify completion handlers."""
        for handler in self._completion_handlers:
            try:
                handler(result)
            except Exception as e:
                logger.error(f"Completion handler failed: {e}")

    def get_schedule_status(self, schedule_id: str) -> Optional[ScheduleTask]:
        """Get status of a specific schedule."""
        return self._tasks.get(schedule_id)

    def list_schedules(self, status_filter: Optional[ScheduleStatus] = None) -> List[ScheduleTask]:
        """List all schedules with optional status filter."""
        if status_filter:
            return [task for task in self._tasks.values() if task.status == status_filter]
        return list(self._tasks.values())

    def cancel_schedule(self, schedule_id: str) -> bool:
        """Cancel a scheduled download."""
        task = self._tasks.get(schedule_id)
        if not task:
            return False

        if task.status in [ScheduleStatus.COMPLETED, ScheduleStatus.FAILED, ScheduleStatus.CANCELLED]:
            return False

        if task.status == ScheduleStatus.RUNNING:
            logger.warning(f"Cannot cancel running schedule {schedule_id}")
            return False

        task.update_status(ScheduleStatus.CANCELLED)
        self._save_task_state(task)
        logger.info(f"Cancelled schedule {schedule_id}")
        return True