"""
Dataset specification and catalog management for FinLab downloader.

Handles parsing of dataset definitions and provides search and filtering capabilities.
"""

import csv
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Set
from pathlib import Path
from enum import Enum

from .exceptions import ConfigurationError, ValidationError


class DataType(Enum):
    """Supported data types for datasets."""
    FLOAT = "float"
    INT = "int"
    STRING = "str"
    BOOLEAN = "bool"
    DATETIME = "datetime"


class DownloadMethod(Enum):
    """Supported download methods."""
    ETL = "etl"
    FINANCIAL_STATEMENT = "financial_statement"
    FUNDAMENTAL_FEATURES = "fundamental_features"
    INSTITUTIONAL_INVESTORS = "institutional_investors_trading_summary"
    MARGIN_TRANSACTIONS = "margin_transactions"
    MONTHLY_REVENUE = "monthly_revenue"
    PRICE_EARNING_RATIO = "price_earning_ratio"
    QUALITY_FACTOR = "quality_factor_z_score"
    ROTC_MONTHLY_REVENUE = "rotc_monthly_revenue"
    TW_BUSINESS_INDICATORS = "tw_business_indicators"


@dataclass
class DatasetSpecification:
    """Specification for a single dataset."""

    # Core identification
    name: str
    # Download configuration
    download_method: DownloadMethod
    download_key: str
    data_type: DataType

    # Optional identification
    english_name: Optional[str] = None
    chinese_name: Optional[str] = None

    # Metadata
    category: Optional[str] = None
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)

    # Validation
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        """Post-initialization validation and normalization."""
        # Determine language names if not explicitly set
        if self.english_name is None and self.chinese_name is None:
            # Try to determine based on name content
            if self._contains_chinese(self.name):
                self.chinese_name = self.name
                self.english_name = self._generate_english_name()
            else:
                self.english_name = self.name
                self.chinese_name = None
        elif self.english_name is None:
            self.english_name = self._generate_english_name()
        elif self.chinese_name is None:
            self.chinese_name = self.name if self._contains_chinese(self.name) else None

        # Set category if not provided
        if self.category is None:
            self.category = self._determine_category(self.download_method)

    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _generate_english_name(self) -> str:
        """Generate a basic English name from the key."""
        # Use the download key as a fallback English name
        return self.download_key.replace('_', ' ').replace(':', ' ').title()

    @staticmethod
    def _determine_category(download_method: DownloadMethod) -> str:
        """Determine dataset category from download method."""
        category_mapping = {
            DownloadMethod.ETL: "Market Data",
            DownloadMethod.FINANCIAL_STATEMENT: "Financial Statements",
            DownloadMethod.FUNDAMENTAL_FEATURES: "Fundamental Analysis",
            DownloadMethod.INSTITUTIONAL_INVESTORS: "Institutional Trading",
            DownloadMethod.MARGIN_TRANSACTIONS: "Margin Trading",
            DownloadMethod.MONTHLY_REVENUE: "Financial Performance",
            DownloadMethod.PRICE_EARNING_RATIO: "Valuation Metrics",
            DownloadMethod.QUALITY_FACTOR: "Quality Factors",
            DownloadMethod.ROTC_MONTHLY_REVENUE: "Revenue Analysis",
            DownloadMethod.TW_BUSINESS_INDICATORS: "Economic Indicators"
        }

        return category_mapping.get(download_method, "Other")

    @property
    def display_name(self) -> str:
        """Get the primary display name (Chinese if available, otherwise English)."""
        return self.chinese_name or self.english_name or self.name

    @property
    def search_terms(self) -> Set[str]:
        """Get all searchable terms for this dataset."""
        terms = {self.name.lower()}

        if self.english_name:
            terms.add(self.english_name.lower())
            terms.update(self.english_name.lower().split())

        if self.chinese_name:
            terms.add(self.chinese_name.lower())

        terms.add(self.download_key.lower())
        terms.update(self.tags)

        if self.category:
            terms.add(self.category.lower())

        return terms

    def matches_search(self, query: str) -> bool:
        """Check if this dataset matches a search query."""
        query_lower = query.lower()

        # Exact matches
        if query_lower in self.search_terms:
            return True

        # Partial matches
        for term in self.search_terms:
            if query_lower in term or term in query_lower:
                return True

        return False

    def validate_value(self, value: Any) -> Any:
        """
        Validate a value against this dataset's specification.

        Args:
            value: Value to validate

        Returns:
            Validated and converted value

        Raises:
            ValidationError: If validation fails
        """
        # Handle null values
        if value is None or (isinstance(value, str) and value.strip() == ""):
            if not self.nullable:
                raise ValidationError(
                    f"Dataset '{self.name}' does not allow null values",
                    field=self.name,
                    value=value
                )
            return None

        # Type conversion and validation
        try:
            if self.data_type == DataType.FLOAT:
                converted = float(value)
            elif self.data_type == DataType.INT:
                converted = int(float(value))  # Handle "1.0" -> 1
            elif self.data_type == DataType.STRING:
                converted = str(value)
            elif self.data_type == DataType.BOOLEAN:
                if isinstance(value, bool):
                    converted = value
                elif isinstance(value, str):
                    converted = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    converted = bool(value)
            else:
                converted = value  # For datetime and other types

        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"Cannot convert value '{value}' to {self.data_type.value} for dataset '{self.name}'",
                field=self.name,
                value=value,
                expected_type=self.data_type.value,
                cause=e
            )

        # Range validation for numeric types
        if self.data_type in (DataType.FLOAT, DataType.INT):
            if self.min_value is not None and converted < self.min_value:
                raise ValidationError(
                    f"Value {converted} is below minimum {self.min_value} for dataset '{self.name}'",
                    field=self.name,
                    value=converted,
                    min_value=self.min_value
                )

            if self.max_value is not None and converted > self.max_value:
                raise ValidationError(
                    f"Value {converted} exceeds maximum {self.max_value} for dataset '{self.name}'",
                    field=self.name,
                    value=converted,
                    max_value=self.max_value
                )

        return converted


class DatasetParser:
    """Parser for FinLab dataset CSV files."""

    @staticmethod
    def parse_csv(csv_path: Union[str, Path]) -> List[DatasetSpecification]:
        """
        Parse dataset specifications from CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            List of dataset specifications

        Raises:
            ConfigurationError: If CSV cannot be parsed
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise ConfigurationError(
                f"Dataset CSV file not found: {csv_path}",
                file_path=str(csv_path)
            )

        datasets = []

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Handle BOM if present
                content = f.read()
                if content.startswith('\ufeff'):
                    content = content[1:]

                reader = csv.DictReader(content.splitlines())

                for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                    try:
                        dataset = DatasetParser._parse_row(row)
                        datasets.append(dataset)
                    except Exception as e:
                        raise ConfigurationError(
                            f"Error parsing CSV row {row_num}: {e}",
                            file_path=str(csv_path),
                            row=row_num,
                            cause=e
                        )

        except IOError as e:
            raise ConfigurationError(
                f"Cannot read dataset CSV file: {e}",
                file_path=str(csv_path),
                cause=e
            )

        return datasets

    @staticmethod
    def _parse_row(row: Dict[str, str]) -> DatasetSpecification:
        """Parse a single CSV row into a DatasetSpecification."""
        # Expected columns: 資料集名稱, 下載方式及key, 數據類型
        name = row.get('資料集名稱', '').strip()
        download_spec = row.get('下載方式及key', '').strip()
        data_type_str = row.get('數據類型', '').strip()

        if not name:
            raise ValueError("Dataset name is required")

        if not download_spec:
            raise ValueError("Download specification is required")

        # Parse download specification
        download_method, download_key = DatasetParser._parse_download_spec(download_spec)

        # Parse data type
        try:
            data_type = DataType(data_type_str.lower())
        except ValueError:
            raise ValueError(f"Unsupported data type: {data_type_str}")

        # Determine category from download method
        category = DatasetParser._determine_category(download_method)

        return DatasetSpecification(
            name=name,
            download_method=download_method,
            download_key=download_key,
            data_type=data_type,
            category=category
        )

    @staticmethod
    def _parse_download_spec(download_spec: str) -> tuple[DownloadMethod, str]:
        """Parse download specification into method and key."""
        if ':' not in download_spec:
            raise ValueError(f"Invalid download specification format: {download_spec}")

        parts = download_spec.split(':', 1)
        method_str = parts[0].strip()
        key = parts[1].strip()

        try:
            method = DownloadMethod(method_str)
        except ValueError:
            raise ValueError(f"Unsupported download method: {method_str}")

        return method, key

    @staticmethod
    def _determine_category(download_method: DownloadMethod) -> str:
        """Determine dataset category from download method."""
        category_mapping = {
            DownloadMethod.ETL: "Market Data",
            DownloadMethod.FINANCIAL_STATEMENT: "Financial Statements",
            DownloadMethod.FUNDAMENTAL_FEATURES: "Fundamental Analysis",
            DownloadMethod.INSTITUTIONAL_INVESTORS: "Institutional Trading",
            DownloadMethod.MARGIN_TRANSACTIONS: "Margin Trading",
            DownloadMethod.MONTHLY_REVENUE: "Financial Performance",
            DownloadMethod.PRICE_EARNING_RATIO: "Valuation Metrics",
            DownloadMethod.QUALITY_FACTOR: "Quality Factors",
            DownloadMethod.ROTC_MONTHLY_REVENUE: "Revenue Analysis",
            DownloadMethod.TW_BUSINESS_INDICATORS: "Economic Indicators"
        }

        return category_mapping.get(download_method, "Other")


class DatasetCatalog:
    """Catalog for managing available datasets with search and filtering."""

    def __init__(self, datasets: Optional[List[DatasetSpecification]] = None):
        """
        Initialize dataset catalog.

        Args:
            datasets: List of dataset specifications
        """
        self._datasets: Dict[str, DatasetSpecification] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._download_methods: Dict[DownloadMethod, Set[str]] = {}

        if datasets:
            self.add_datasets(datasets)

    def add_dataset(self, dataset: DatasetSpecification) -> None:
        """Add a dataset to the catalog."""
        self._datasets[dataset.name] = dataset

        # Update category index
        if dataset.category:
            if dataset.category not in self._categories:
                self._categories[dataset.category] = set()
            self._categories[dataset.category].add(dataset.name)

        # Update download method index
        if dataset.download_method not in self._download_methods:
            self._download_methods[dataset.download_method] = set()
        self._download_methods[dataset.download_method].add(dataset.name)

    def add_datasets(self, datasets: List[DatasetSpecification]) -> None:
        """Add multiple datasets to the catalog."""
        for dataset in datasets:
            self.add_dataset(dataset)

    def get_dataset(self, name: str) -> Optional[DatasetSpecification]:
        """Get dataset by name."""
        return self._datasets.get(name)

    def list_datasets(self, category: Optional[str] = None,
                     download_method: Optional[DownloadMethod] = None) -> List[DatasetSpecification]:
        """
        List datasets with optional filtering.

        Args:
            category: Filter by category
            download_method: Filter by download method

        Returns:
            List of matching datasets
        """
        if category and download_method:
            # Both filters
            category_datasets = self._categories.get(category, set())
            method_datasets = self._download_methods.get(download_method, set())
            matching_names = category_datasets.intersection(method_datasets)
        elif category:
            # Category filter only
            matching_names = self._categories.get(category, set())
        elif download_method:
            # Download method filter only
            matching_names = self._download_methods.get(download_method, set())
        else:
            # No filters
            matching_names = set(self._datasets.keys())

        return [self._datasets[name] for name in matching_names]

    def search(self, query: str, category: Optional[str] = None,
              download_method: Optional[DownloadMethod] = None) -> List[DatasetSpecification]:
        """
        Search datasets by query with optional filtering.

        Args:
            query: Search query
            category: Filter by category
            download_method: Filter by download method

        Returns:
            List of matching datasets
        """
        # Start with filtered datasets
        candidates = self.list_datasets(category=category, download_method=download_method)

        # Apply search filter
        return [dataset for dataset in candidates if dataset.matches_search(query)]

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return sorted(self._categories.keys())

    def get_download_methods(self) -> List[DownloadMethod]:
        """Get list of all download methods."""
        return sorted(self._download_methods.keys(), key=lambda x: x.value)

    def count(self) -> int:
        """Get total number of datasets."""
        return len(self._datasets)

    def count_by_category(self) -> Dict[str, int]:
        """Get dataset count by category."""
        return {category: len(datasets) for category, datasets in self._categories.items()}

    def count_by_download_method(self) -> Dict[DownloadMethod, int]:
        """Get dataset count by download method."""
        return {method: len(datasets) for method, datasets in self._download_methods.items()}

    @classmethod
    def from_csv(cls, csv_path: Union[str, Path]) -> 'DatasetCatalog':
        """
        Create catalog from CSV file.

        Args:
            csv_path: Path to the CSV file

        Returns:
            DatasetCatalog instance
        """
        datasets = DatasetParser.parse_csv(csv_path)
        return cls(datasets)

    def to_dict(self) -> Dict[str, Any]:
        """Convert catalog to dictionary format."""
        return {
            'datasets': {
                name: {
                    'name': ds.name,
                    'english_name': ds.english_name,
                    'chinese_name': ds.chinese_name,
                    'download_method': ds.download_method.value,
                    'download_key': ds.download_key,
                    'data_type': ds.data_type.value,
                    'category': ds.category,
                    'description': ds.description,
                    'tags': list(ds.tags),
                    'nullable': ds.nullable,
                    'min_value': ds.min_value,
                    'max_value': ds.max_value
                }
                for name, ds in self._datasets.items()
            },
            'categories': {cat: list(datasets) for cat, datasets in self._categories.items()},
            'download_methods': {method.value: list(datasets)
                               for method, datasets in self._download_methods.items()},
            'statistics': {
                'total_datasets': self.count(),
                'categories_count': len(self._categories),
                'download_methods_count': len(self._download_methods)
            }
        }