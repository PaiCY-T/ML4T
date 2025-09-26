"""
Tests for scheduler engine functionality.
"""

import pytest
import asyncio
from datetime import datetime, date
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from src.finlab_downloader.core.dataset import DatasetCatalog, DatasetSpecification, DownloadMethod, DataType
from src.finlab_downloader.config.dataset_schema import DatasetDownloadConfig
from src.finlab_downloader.scheduler.scheduler_engine import (
    SchedulerEngine,
    ScheduleRequest,
    ScheduleResult,
    SchedulePriority,
    ScheduleStatus
)
from src.finlab_downloader.scheduler.financial_calendar import IndustryType, ReportingPeriod


class TestSchedulerEngine:
    """Test cases for SchedulerEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock dataset catalog
        self.mock_datasets = [
            DatasetSpecification(
                name="營業收入",
                download_method=DownloadMethod.FINANCIAL_STATEMENT,
                download_key="營業收入",
                data_type=DataType.FLOAT,
                category="Financial Statements"
            ),
            DatasetSpecification(
                name="營業成本",
                download_method=DownloadMethod.FINANCIAL_STATEMENT,
                download_key="營業成本",
                data_type=DataType.FLOAT,
                category="Financial Statements"
            )
        ]
        self.dataset_catalog = DatasetCatalog(self.mock_datasets)

        # Create mock configuration
        self.mock_config = {
            "datasets": {
                "catalog_file": "test.csv",
                "download": {
                    "batch_size": 50,
                    "parallel_downloads": 2
                }
            }
        }
        self.download_config = DatasetDownloadConfig(self.mock_config)

        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_directory = Path(self.temp_dir)

        # Initialize scheduler engine
        self.scheduler = SchedulerEngine(
            dataset_catalog=self.dataset_catalog,
            config=self.download_config,
            data_directory=self.data_directory
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schedule_financial_statement_download(self):
        """Test scheduling a financial statement download."""
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入", "營業成本"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024,
            priority=SchedulePriority.NORMAL
        )

        schedule_id = self.scheduler.schedule_financial_statement_download(request)

        assert schedule_id is not None
        assert schedule_id.startswith("schedule_")

        # Check that task was stored
        task = self.scheduler.get_schedule_status(schedule_id)
        assert task is not None
        assert task.request.company_ticker == "2330"
        assert task.status == ScheduleStatus.SCHEDULED

    def test_schedule_bulk_downloads(self):
        """Test scheduling multiple downloads."""
        requests = [
            ScheduleRequest(
                company_ticker="2330",
                dataset_names=["營業收入"],
                industry_type=IndustryType.GENERAL,
                reporting_period=ReportingPeriod.Q1,
                fiscal_year=2024,
                priority=SchedulePriority.HIGH
            ),
            ScheduleRequest(
                company_ticker="2317",
                dataset_names=["營業收入"],
                industry_type=IndustryType.GENERAL,
                reporting_period=ReportingPeriod.Q1,
                fiscal_year=2024,
                priority=SchedulePriority.LOW
            )
        ]

        schedule_ids = self.scheduler.schedule_bulk_downloads(requests)

        assert len(schedule_ids) == 2
        assert all(sid.startswith("schedule_") for sid in schedule_ids)

    def test_validate_schedule_request_invalid_dataset(self):
        """Test validation with invalid dataset."""
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["invalid_dataset"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )

        with pytest.raises(Exception):  # Should raise ValidationError
            self.scheduler.schedule_financial_statement_download(request)

    def test_validate_schedule_request_invalid_period_for_industry(self):
        """Test validation with invalid period for KY stock."""
        request = ScheduleRequest(
            company_ticker="1234-KY",
            dataset_names=["營業收入"],
            industry_type=IndustryType.KY_STOCK,
            reporting_period=ReportingPeriod.Q1,  # Q1 not valid for KY stocks
            fiscal_year=2024
        )

        with pytest.raises(Exception):  # Should raise ValidationError
            self.scheduler.schedule_financial_statement_download(request)

    @pytest.mark.asyncio
    async def test_execute_pending_schedules_empty(self):
        """Test executing when no schedules are pending."""
        results = await self.scheduler.execute_pending_schedules()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_execute_task_success(self):
        """Test successful task execution."""
        # Schedule a download
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )

        schedule_id = self.scheduler.schedule_financial_statement_download(request)

        # Mock the download method to succeed
        with patch.object(self.scheduler, '_download_dataset', return_value=True):
            # Get the task and execute it directly (bypass scheduling time check)
            task = self.scheduler._tasks[schedule_id]
            result = await self.scheduler._execute_task(task)

            assert result.status == ScheduleStatus.COMPLETED
            assert len(result.downloaded_datasets) == 1
            assert len(result.failed_datasets) == 0
            assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_execute_task_partial_failure(self):
        """Test task execution with partial failures."""
        # Schedule a download with multiple datasets
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入", "營業成本"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )

        schedule_id = self.scheduler.schedule_financial_statement_download(request)

        # Mock the download method to succeed for first, fail for second
        async def mock_download(ticker, dataset_name):
            return dataset_name == "營業收入"

        with patch.object(self.scheduler, '_download_dataset', side_effect=mock_download):
            task = self.scheduler._tasks[schedule_id]
            result = await self.scheduler._execute_task(task)

            assert result.status == ScheduleStatus.COMPLETED
            assert len(result.downloaded_datasets) == 1
            assert len(result.failed_datasets) == 1
            assert result.success_rate == 0.5
            assert "營業收入" in result.downloaded_datasets
            assert "營業成本" in result.failed_datasets

    def test_list_schedules(self):
        """Test listing schedules with and without filters."""
        # Create some schedules
        request1 = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )

        request2 = ScheduleRequest(
            company_ticker="2317",
            dataset_names=["營業收入"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q2,
            fiscal_year=2024
        )

        self.scheduler.schedule_financial_statement_download(request1)
        self.scheduler.schedule_financial_statement_download(request2)

        # List all schedules
        all_schedules = self.scheduler.list_schedules()
        assert len(all_schedules) == 2

        # List with status filter
        scheduled_only = self.scheduler.list_schedules(ScheduleStatus.SCHEDULED)
        assert len(scheduled_only) == 2

        completed_only = self.scheduler.list_schedules(ScheduleStatus.COMPLETED)
        assert len(completed_only) == 0

    def test_cancel_schedule(self):
        """Test cancelling a scheduled download."""
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )

        schedule_id = self.scheduler.schedule_financial_statement_download(request)

        # Cancel the schedule
        success = self.scheduler.cancel_schedule(schedule_id)
        assert success

        # Check status
        task = self.scheduler.get_schedule_status(schedule_id)
        assert task.status == ScheduleStatus.CANCELLED

        # Try to cancel again (should fail)
        success = self.scheduler.cancel_schedule(schedule_id)
        assert not success

    def test_cancel_nonexistent_schedule(self):
        """Test cancelling a non-existent schedule."""
        success = self.scheduler.cancel_schedule("invalid_id")
        assert not success

    def test_calculate_optimal_schedule_time(self):
        """Test calculation of optimal scheduling time."""
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024,
            priority=SchedulePriority.HIGH
        )

        scheduled_time = self.scheduler._calculate_optimal_schedule_time(request)

        assert isinstance(scheduled_time, datetime)
        # High priority should be scheduled early (6 AM)
        assert scheduled_time.hour == 6

    def test_task_state_persistence(self):
        """Test that task state is persisted to disk."""
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )

        schedule_id = self.scheduler.schedule_financial_statement_download(request)

        # Check that task file was created
        task_file = self.data_directory / f"{schedule_id}.json"
        assert task_file.exists()

        # Check file contents
        import json
        with open(task_file) as f:
            task_data = json.load(f)

        assert task_data["company_ticker"] == "2330"
        assert task_data["status"] == "scheduled"


class TestScheduleRequest:
    """Test cases for ScheduleRequest dataclass."""

    def test_schedule_request_validation(self):
        """Test schedule request validation."""
        # Valid request
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )
        assert request.company_ticker == "2330"

        # Invalid request - empty ticker
        with pytest.raises(ValueError):
            ScheduleRequest(
                company_ticker="",
                dataset_names=["營業收入"],
                industry_type=IndustryType.GENERAL,
                reporting_period=ReportingPeriod.Q1,
                fiscal_year=2024
            )

        # Invalid request - empty datasets
        with pytest.raises(ValueError):
            ScheduleRequest(
                company_ticker="2330",
                dataset_names=[],
                industry_type=IndustryType.GENERAL,
                reporting_period=ReportingPeriod.Q1,
                fiscal_year=2024
            )


class TestScheduleResult:
    """Test cases for ScheduleResult dataclass."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入", "營業成本"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )

        result = ScheduleResult(
            request=request,
            schedule_id="test_123",
            status=ScheduleStatus.COMPLETED,
            scheduled_time=datetime.now(),
            downloaded_datasets=["營業收入"],
            failed_datasets=["營業成本"]
        )

        assert result.success_rate == 0.5

        # Test with no datasets
        result.downloaded_datasets = []
        result.failed_datasets = []
        assert result.success_rate == 0.0

    def test_is_completed(self):
        """Test completion status checking."""
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )

        result = ScheduleResult(
            request=request,
            schedule_id="test_123",
            status=ScheduleStatus.COMPLETED,
            scheduled_time=datetime.now()
        )

        assert result.is_completed

        result.status = ScheduleStatus.RUNNING
        assert not result.is_completed

    def test_duration_calculation(self):
        """Test duration calculation."""
        request = ScheduleRequest(
            company_ticker="2330",
            dataset_names=["營業收入"],
            industry_type=IndustryType.GENERAL,
            reporting_period=ReportingPeriod.Q1,
            fiscal_year=2024
        )

        result = ScheduleResult(
            request=request,
            schedule_id="test_123",
            status=ScheduleStatus.COMPLETED,
            scheduled_time=datetime.now(),
            execution_time_seconds=120.0
        )

        assert result.duration_minutes == 2.0