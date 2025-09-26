"""
Integration layer between scheduler and existing FinLab downloader systems.

Provides seamless integration with dataset catalog, configuration management,
and CLI commands for the financial statement scheduler.
"""

import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml

from ..core.dataset import DatasetCatalog, DownloadMethod
from ..core.exceptions import ConfigurationError, SchedulingError
from ..config.manager import ConfigurationManager
from ..config.dataset_schema import DatasetDownloadConfig, DatasetValidationConfig
from .scheduler_engine import (
    SchedulerEngine,
    ScheduleRequest,
    ScheduleResult,
    SchedulePriority
)
from .financial_calendar import IndustryType, ReportingPeriod

logger = logging.getLogger(__name__)


class SchedulerIntegration:
    """
    Integration layer for the financial statement scheduler.

    Provides high-level interface for integrating scheduler with existing
    FinLab downloader components and configuration systems.
    """

    def __init__(self,
                 config_manager: ConfigurationManager,
                 dataset_catalog: Optional[DatasetCatalog] = None):
        """
        Initialize scheduler integration.

        Args:
            config_manager: Configuration manager instance
            dataset_catalog: Optional dataset catalog (will be loaded if not provided)
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()

        # Initialize dataset catalog
        if dataset_catalog:
            self.dataset_catalog = dataset_catalog
        else:
            self.dataset_catalog = self._load_dataset_catalog()

        # Initialize configuration objects
        self.download_config = DatasetDownloadConfig(self.config)
        self.validation_config = DatasetValidationConfig(self.config)

        # Initialize scheduler engine
        data_directory = Path(self.config.get("general", {}).get("data_directory", "./data")) / "schedules"
        self.scheduler = SchedulerEngine(
            dataset_catalog=self.dataset_catalog,
            config=self.download_config,
            data_directory=data_directory
        )

        logger.info("Initialized Scheduler Integration")

    def _load_dataset_catalog(self) -> DatasetCatalog:
        """Load dataset catalog from configuration."""
        catalog_file = self.download_config.catalog_file

        # Try different locations for catalog file
        catalog_paths = [
            Path(catalog_file),
            Path(self.config.get("general", {}).get("data_directory", "./data")) / catalog_file,
            Path("./example") / catalog_file,
            Path("./") / catalog_file
        ]

        for catalog_path in catalog_paths:
            if catalog_path.exists():
                logger.info(f"Loading dataset catalog from {catalog_path}")
                return DatasetCatalog.from_csv(catalog_path)

        raise ConfigurationError(f"Dataset catalog not found. Tried: {[str(p) for p in catalog_paths]}")

    def schedule_company_financials(self,
                                   company_ticker: str,
                                   reporting_periods: List[ReportingPeriod],
                                   fiscal_year: int,
                                   industry_type: Optional[IndustryType] = None,
                                   priority: SchedulePriority = SchedulePriority.NORMAL,
                                   dataset_filter: Optional[List[str]] = None) -> List[str]:
        """
        Schedule financial statement downloads for a company.

        Args:
            company_ticker: Company ticker symbol
            reporting_periods: List of reporting periods to schedule
            fiscal_year: Fiscal year
            industry_type: Industry classification (auto-detected if not provided)
            priority: Schedule priority
            dataset_filter: Optional list of specific datasets to download

        Returns:
            List of schedule IDs
        """
        try:
            # Auto-detect industry type if not provided
            if industry_type is None:
                industry_type = self.scheduler.financial_calendar.get_industry_from_ticker(company_ticker)
                logger.info(f"Auto-detected industry type for {company_ticker}: {industry_type.value}")

            # Get relevant financial statement datasets
            financial_datasets = self._get_financial_statement_datasets(dataset_filter)

            if not financial_datasets:
                raise SchedulingError("No financial statement datasets found to schedule")

            schedule_ids = []
            for period in reporting_periods:
                # Check if reporting is required for this industry/period combination
                if not self.scheduler.financial_calendar.is_reporting_required(industry_type, period):
                    logger.info(f"Skipping {period.value} for {industry_type.value} - not required")
                    continue

                request = ScheduleRequest(
                    company_ticker=company_ticker,
                    dataset_names=financial_datasets,
                    industry_type=industry_type,
                    reporting_period=period,
                    fiscal_year=fiscal_year,
                    priority=priority,
                    metadata={
                        "auto_detected_industry": industry_type is None,
                        "total_datasets": len(financial_datasets),
                        "integration_version": "1.0"
                    }
                )

                schedule_id = self.scheduler.schedule_financial_statement_download(request)
                schedule_ids.append(schedule_id)

            logger.info(f"Scheduled {len(schedule_ids)} downloads for {company_ticker} FY{fiscal_year}")
            return schedule_ids

        except Exception as e:
            error_msg = f"Failed to schedule financials for {company_ticker}: {e}"
            logger.error(error_msg)
            raise SchedulingError(error_msg, cause=e)

    def schedule_bulk_companies(self,
                               company_tickers: List[str],
                               reporting_periods: List[ReportingPeriod],
                               fiscal_year: int,
                               industry_mapping: Optional[Dict[str, IndustryType]] = None,
                               priority: SchedulePriority = SchedulePriority.NORMAL) -> Dict[str, List[str]]:
        """
        Schedule financial statement downloads for multiple companies.

        Args:
            company_tickers: List of company ticker symbols
            reporting_periods: List of reporting periods to schedule
            fiscal_year: Fiscal year
            industry_mapping: Optional mapping of ticker to industry type
            priority: Schedule priority

        Returns:
            Dictionary mapping ticker to list of schedule IDs
        """
        results = {}
        industry_mapping = industry_mapping or {}

        for ticker in company_tickers:
            try:
                industry_type = industry_mapping.get(ticker)
                schedule_ids = self.schedule_company_financials(
                    company_ticker=ticker,
                    reporting_periods=reporting_periods,
                    fiscal_year=fiscal_year,
                    industry_type=industry_type,
                    priority=priority
                )
                results[ticker] = schedule_ids
            except SchedulingError as e:
                logger.error(f"Failed to schedule {ticker}: {e}")
                results[ticker] = []

        total_scheduled = sum(len(ids) for ids in results.values())
        logger.info(f"Bulk scheduling completed: {total_scheduled} schedules for {len(company_tickers)} companies")
        return results

    def _get_financial_statement_datasets(self, dataset_filter: Optional[List[str]] = None) -> List[str]:
        """
        Get list of financial statement datasets.

        Args:
            dataset_filter: Optional filter for specific datasets

        Returns:
            List of dataset names
        """
        # Get all financial statement datasets from catalog
        financial_datasets = self.dataset_catalog.list_datasets(
            download_method=DownloadMethod.FINANCIAL_STATEMENT
        )

        dataset_names = [ds.name for ds in financial_datasets]

        # Apply filter if provided
        if dataset_filter:
            # Filter to only include requested datasets that exist
            filtered_names = []
            for name in dataset_filter:
                if name in dataset_names:
                    filtered_names.append(name)
                else:
                    logger.warning(f"Requested dataset '{name}' not found in financial statements")
            dataset_names = filtered_names

        # Apply category-based filtering from configuration
        if self.download_config.is_category_enabled("financial_statements"):
            logger.info(f"Found {len(dataset_names)} financial statement datasets")
        else:
            logger.warning("Financial statements category is disabled in configuration")
            dataset_names = []

        return dataset_names

    def create_quarterly_schedule(self,
                                company_ticker: str,
                                fiscal_year: int,
                                industry_type: Optional[IndustryType] = None) -> List[str]:
        """
        Create a complete quarterly schedule for a company.

        Args:
            company_ticker: Company ticker symbol
            fiscal_year: Fiscal year
            industry_type: Industry classification

        Returns:
            List of schedule IDs
        """
        # Determine all applicable reporting periods
        if industry_type is None:
            industry_type = self.scheduler.financial_calendar.get_industry_from_ticker(company_ticker)

        valid_periods = self.scheduler.financial_calendar.get_reporting_periods_for_industry(industry_type)

        return self.schedule_company_financials(
            company_ticker=company_ticker,
            reporting_periods=valid_periods,
            fiscal_year=fiscal_year,
            industry_type=industry_type,
            priority=SchedulePriority.NORMAL
        )

    def get_schedule_summary(self) -> Dict[str, Any]:
        """
        Get summary of all schedules.

        Returns:
            Summary statistics and information
        """
        all_schedules = self.scheduler.list_schedules()

        summary = {
            "total_schedules": len(all_schedules),
            "by_status": {},
            "by_industry": {},
            "by_period": {},
            "by_priority": {},
            "upcoming_deadlines": [],
            "overdue_schedules": []
        }

        # Count by status
        for task in all_schedules:
            status = task.status.value
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1

            # Count by industry
            industry = task.request.industry_type.value
            summary["by_industry"][industry] = summary["by_industry"].get(industry, 0) + 1

            # Count by period
            period = task.request.reporting_period.value
            summary["by_period"][period] = summary["by_period"].get(period, 0) + 1

            # Count by priority
            priority = task.request.priority.value
            summary["by_priority"][priority] = summary["by_priority"].get(priority, 0) + 1

            # Check for upcoming deadlines (next 7 days)
            if task.scheduled_time > datetime.now():
                days_until = (task.scheduled_time.date() - date.today()).days
                if days_until <= 7:
                    summary["upcoming_deadlines"].append({
                        "schedule_id": task.schedule_id,
                        "company": task.request.company_ticker,
                        "period": task.request.reporting_period.value,
                        "scheduled_time": task.scheduled_time.isoformat(),
                        "days_until": days_until
                    })

            # Check for overdue schedules
            if task.scheduled_time < datetime.now() and task.status.value in ["pending", "scheduled"]:
                summary["overdue_schedules"].append({
                    "schedule_id": task.schedule_id,
                    "company": task.request.company_ticker,
                    "period": task.request.reporting_period.value,
                    "scheduled_time": task.scheduled_time.isoformat(),
                    "days_overdue": (datetime.now() - task.scheduled_time).days
                })

        return summary

    def export_schedule_config(self, output_path: Union[str, Path]) -> None:
        """
        Export current schedule configuration to YAML file.

        Args:
            output_path: Path to save configuration
        """
        config_data = {
            "scheduler_config": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "dataset_catalog": {
                    "file": self.download_config.catalog_file,
                    "total_datasets": self.dataset_catalog.count(),
                    "financial_datasets": len(self._get_financial_statement_datasets())
                },
                "industry_rules": {
                    industry.value: {
                        "valid_periods": [p.value for p in self.scheduler.financial_calendar.get_reporting_periods_for_industry(industry)]
                    }
                    for industry in IndustryType
                },
                "schedule_summary": self.get_schedule_summary()
            }
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

        logger.info(f"Exported schedule configuration to {output_path}")

    async def run_scheduled_downloads(self) -> List[ScheduleResult]:
        """
        Execute all pending scheduled downloads.

        Returns:
            List of execution results
        """
        logger.info("Starting scheduled download execution")
        results = await self.scheduler.execute_pending_schedules()

        # Log summary
        successful = len([r for r in results if r.status.value == "completed"])
        failed = len([r for r in results if r.status.value == "failed"])

        logger.info(f"Download execution completed: {successful} successful, {failed} failed")
        return results

    def add_completion_callback(self, callback: callable) -> None:
        """Add callback for schedule completion events."""
        self.scheduler.add_completion_handler(callback)