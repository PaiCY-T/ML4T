"""
Tests for dataset parsing and catalog functionality.

Tests the core dataset functionality including CSV parsing, validation, and search.
"""

import pytest
import tempfile
import csv
from pathlib import Path
from unittest.mock import patch

from src.finlab_downloader.core.dataset import (
    DatasetSpecification, DatasetParser, DatasetCatalog,
    DataType, DownloadMethod
)
from src.finlab_downloader.core.validation import (
    DataTypeValidator, BusinessLogicValidator, ValidationContext,
    create_taiwan_market_context
)
from src.finlab_downloader.core.exceptions import ConfigurationError, ValidationError


class TestDatasetSpecification:
    """Test DatasetSpecification class."""

    def test_basic_creation(self):
        """Test basic dataset specification creation."""
        spec = DatasetSpecification(
            name="test_dataset",
            download_method=DownloadMethod.ETL,
            download_key="etl:test_key",
            data_type=DataType.FLOAT
        )

        assert spec.name == "test_dataset"
        assert spec.download_method == DownloadMethod.ETL
        assert spec.download_key == "etl:test_key"
        assert spec.data_type == DataType.FLOAT
        assert spec.category == "Market Data"
        assert spec.nullable is True

    def test_chinese_name_detection(self):
        """Test Chinese name detection and processing."""
        spec = DatasetSpecification(
            name="營業收入",
            download_method=DownloadMethod.FINANCIAL_STATEMENT,
            download_key="financial_statement:營業收入",
            data_type=DataType.FLOAT
        )

        assert spec.chinese_name == "營業收入"
        assert spec.english_name == "Financial Statement 營業收入"
        assert spec.display_name == "營業收入"

    def test_english_name_processing(self):
        """Test English name processing."""
        spec = DatasetSpecification(
            name="adj_close",
            download_method=DownloadMethod.ETL,
            download_key="etl:adj_close",
            data_type=DataType.FLOAT
        )

        assert spec.english_name == "adj_close"
        assert spec.chinese_name is None
        assert spec.display_name == "adj_close"

    def test_search_terms(self):
        """Test search terms generation."""
        spec = DatasetSpecification(
            name="營業收入",
            english_name="Operating Revenue",
            chinese_name="營業收入",
            download_method=DownloadMethod.FINANCIAL_STATEMENT,
            download_key="financial_statement:營業收入",
            data_type=DataType.FLOAT,
            category="Financial Statements",
            tags={"revenue", "income"}
        )

        search_terms = spec.search_terms
        assert "營業收入" in search_terms
        assert "operating revenue" in search_terms
        assert "financial_statement:營業收入" in search_terms
        assert "financial statements" in search_terms
        assert "revenue" in search_terms
        assert "income" in search_terms

    def test_matches_search(self):
        """Test search matching functionality."""
        spec = DatasetSpecification(
            name="營業收入",
            english_name="Operating Revenue",
            download_method=DownloadMethod.FINANCIAL_STATEMENT,
            download_key="financial_statement:營業收入",
            data_type=DataType.FLOAT
        )

        assert spec.matches_search("營業收入")
        assert spec.matches_search("operating")
        assert spec.matches_search("revenue")
        assert spec.matches_search("financial")
        assert not spec.matches_search("unrelated")

    def test_value_validation_float(self):
        """Test float value validation."""
        spec = DatasetSpecification(
            name="test_float",
            download_method=DownloadMethod.ETL,
            download_key="etl:test",
            data_type=DataType.FLOAT,
            min_value=0.0,
            max_value=100.0
        )

        assert spec.validate_value("50.5") == 50.5
        assert spec.validate_value("0") == 0.0
        assert spec.validate_value("100.0") == 100.0

        with pytest.raises(ValidationError):
            spec.validate_value("-1")

        with pytest.raises(ValidationError):
            spec.validate_value("101")

    def test_value_validation_int(self):
        """Test integer value validation."""
        spec = DatasetSpecification(
            name="test_int",
            download_method=DownloadMethod.ETL,
            download_key="etl:test",
            data_type=DataType.INT
        )

        assert spec.validate_value("42") == 42
        assert spec.validate_value("0") == 0
        assert spec.validate_value("-5") == -5
        assert spec.validate_value("1.0") == 1  # Should convert

        with pytest.raises(ValidationError):
            spec.validate_value("not_a_number")

    def test_value_validation_nullable(self):
        """Test nullable value validation."""
        nullable_spec = DatasetSpecification(
            name="nullable_field",
            download_method=DownloadMethod.ETL,
            download_key="etl:test",
            data_type=DataType.FLOAT,
            nullable=True
        )

        non_nullable_spec = DatasetSpecification(
            name="non_nullable_field",
            download_method=DownloadMethod.ETL,
            download_key="etl:test",
            data_type=DataType.FLOAT,
            nullable=False
        )

        assert nullable_spec.validate_value(None) is None
        assert nullable_spec.validate_value("") is None

        with pytest.raises(ValidationError):
            non_nullable_spec.validate_value(None)

        with pytest.raises(ValidationError):
            non_nullable_spec.validate_value("")


class TestDatasetParser:
    """Test DatasetParser class."""

    def create_test_csv(self, data):
        """Create a temporary CSV file with test data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8')
        writer = csv.writer(temp_file)
        writer.writerows(data)
        temp_file.close()
        return temp_file.name

    def test_parse_basic_csv(self):
        """Test parsing basic CSV data."""
        test_data = [
            ['資料集名稱', '下載方式及key', '數據類型'],
            ['adj_close', 'etl:adj_close', 'float'],
            ['營業收入', 'financial_statement:營業收入', 'float'],
            ['buy', 'etl:broker_transactions:top15_buy', 'int']
        ]

        csv_file = self.create_test_csv(test_data)
        try:
            datasets = DatasetParser.parse_csv(csv_file)

            assert len(datasets) == 3

            # Check first dataset
            assert datasets[0].name == "adj_close"
            assert datasets[0].download_method == DownloadMethod.ETL
            assert datasets[0].download_key == "adj_close"
            assert datasets[0].data_type == DataType.FLOAT
            assert datasets[0].category == "Market Data"

            # Check second dataset
            assert datasets[1].name == "營業收入"
            assert datasets[1].download_method == DownloadMethod.FINANCIAL_STATEMENT
            assert datasets[1].download_key == "營業收入"
            assert datasets[1].data_type == DataType.FLOAT
            assert datasets[1].category == "Financial Statements"

            # Check third dataset
            assert datasets[2].name == "buy"
            assert datasets[2].download_method == DownloadMethod.ETL
            assert datasets[2].download_key == "broker_transactions:top15_buy"
            assert datasets[2].data_type == DataType.INT

        finally:
            Path(csv_file).unlink()

    def test_parse_csv_with_bom(self):
        """Test parsing CSV with BOM (Byte Order Mark)."""
        test_data = [
            ['資料集名稱', '下載方式及key', '數據類型'],
            ['test_data', 'etl:test', 'float']
        ]

        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8-sig')
        writer = csv.writer(temp_file)
        writer.writerows(test_data)
        temp_file.close()

        try:
            datasets = DatasetParser.parse_csv(temp_file.name)
            assert len(datasets) == 1
            assert datasets[0].name == "test_data"

        finally:
            Path(temp_file.name).unlink()

    def test_parse_csv_file_not_found(self):
        """Test parsing non-existent CSV file."""
        with pytest.raises(ConfigurationError) as exc_info:
            DatasetParser.parse_csv("non_existent_file.csv")

        assert "not found" in str(exc_info.value)

    def test_parse_invalid_download_spec(self):
        """Test parsing invalid download specification."""
        test_data = [
            ['資料集名稱', '下載方式及key', '數據類型'],
            ['invalid_data', 'invalid_spec_without_colon', 'float']
        ]

        csv_file = self.create_test_csv(test_data)
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                DatasetParser.parse_csv(csv_file)

            assert "Error parsing CSV row" in str(exc_info.value)

        finally:
            Path(csv_file).unlink()

    def test_parse_invalid_data_type(self):
        """Test parsing invalid data type."""
        test_data = [
            ['資料集名稱', '下載方式及key', '數據類型'],
            ['test_data', 'etl:test', 'invalid_type']
        ]

        csv_file = self.create_test_csv(test_data)
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                DatasetParser.parse_csv(csv_file)

            assert "Error parsing CSV row" in str(exc_info.value)

        finally:
            Path(csv_file).unlink()


class TestDatasetCatalog:
    """Test DatasetCatalog class."""

    def create_sample_datasets(self):
        """Create sample datasets for testing."""
        return [
            DatasetSpecification(
                name="adj_close",
                download_method=DownloadMethod.ETL,
                download_key="adj_close",
                data_type=DataType.FLOAT,
                category="Market Data"
            ),
            DatasetSpecification(
                name="營業收入",
                download_method=DownloadMethod.FINANCIAL_STATEMENT,
                download_key="營業收入",
                data_type=DataType.FLOAT,
                category="Financial Statements"
            ),
            DatasetSpecification(
                name="buy",
                download_method=DownloadMethod.ETL,
                download_key="broker_transactions:top15_buy",
                data_type=DataType.INT,
                category="Market Data"
            )
        ]

    def test_catalog_creation(self):
        """Test catalog creation and basic operations."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        assert catalog.count() == 3
        assert len(catalog.get_categories()) == 2
        assert len(catalog.get_download_methods()) == 2

    def test_get_dataset(self):
        """Test getting dataset by name."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        dataset = catalog.get_dataset("adj_close")
        assert dataset is not None
        assert dataset.name == "adj_close"

        assert catalog.get_dataset("non_existent") is None

    def test_list_datasets_no_filter(self):
        """Test listing all datasets without filters."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        all_datasets = catalog.list_datasets()
        assert len(all_datasets) == 3

    def test_list_datasets_by_category(self):
        """Test listing datasets filtered by category."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        market_data = catalog.list_datasets(category="Market Data")
        assert len(market_data) == 2

        financial_statements = catalog.list_datasets(category="Financial Statements")
        assert len(financial_statements) == 1

    def test_list_datasets_by_download_method(self):
        """Test listing datasets filtered by download method."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        etl_datasets = catalog.list_datasets(download_method=DownloadMethod.ETL)
        assert len(etl_datasets) == 2

        financial_datasets = catalog.list_datasets(download_method=DownloadMethod.FINANCIAL_STATEMENT)
        assert len(financial_datasets) == 1

    def test_list_datasets_combined_filters(self):
        """Test listing datasets with combined filters."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        filtered = catalog.list_datasets(
            category="Market Data",
            download_method=DownloadMethod.ETL
        )
        assert len(filtered) == 2

    def test_search_datasets(self):
        """Test searching datasets."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        # Search by name
        results = catalog.search("adj_close")
        assert len(results) == 1
        assert results[0].name == "adj_close"

        # Search by Chinese name
        results = catalog.search("營業收入")
        assert len(results) == 1
        assert results[0].name == "營業收入"

        # Search by partial match
        results = catalog.search("close")
        assert len(results) == 1

        # Search with no results
        results = catalog.search("nonexistent")
        assert len(results) == 0

    def test_search_with_filters(self):
        """Test searching with additional filters."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        # Search within category
        results = catalog.search("adj", category="Market Data")
        assert len(results) == 1

        # Search within download method
        results = catalog.search("buy", download_method=DownloadMethod.ETL)
        assert len(results) == 1

    def test_count_statistics(self):
        """Test count statistics."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        category_counts = catalog.count_by_category()
        assert category_counts["Market Data"] == 2
        assert category_counts["Financial Statements"] == 1

        method_counts = catalog.count_by_download_method()
        assert method_counts[DownloadMethod.ETL] == 2
        assert method_counts[DownloadMethod.FINANCIAL_STATEMENT] == 1

    def test_to_dict(self):
        """Test converting catalog to dictionary."""
        datasets = self.create_sample_datasets()
        catalog = DatasetCatalog(datasets)

        catalog_dict = catalog.to_dict()

        assert "datasets" in catalog_dict
        assert "categories" in catalog_dict
        assert "download_methods" in catalog_dict
        assert "statistics" in catalog_dict

        assert len(catalog_dict["datasets"]) == 3
        assert catalog_dict["statistics"]["total_datasets"] == 3


class TestDataTypeValidator:
    """Test DataTypeValidator class."""

    def test_convert_to_float(self):
        """Test float conversion."""
        assert DataTypeValidator._convert_to_float(42) == 42.0
        assert DataTypeValidator._convert_to_float("3.14") == 3.14
        assert DataTypeValidator._convert_to_float("50%") == 0.5
        assert DataTypeValidator._convert_to_float("1,234.56") == 1234.56

    def test_convert_to_int(self):
        """Test integer conversion."""
        assert DataTypeValidator._convert_to_int(42) == 42
        assert DataTypeValidator._convert_to_int(42.0) == 42
        assert DataTypeValidator._convert_to_int("42") == 42
        assert DataTypeValidator._convert_to_int("1,234") == 1234

        with pytest.raises(ValueError):
            DataTypeValidator._convert_to_int(42.5)

    def test_convert_to_boolean(self):
        """Test boolean conversion."""
        assert DataTypeValidator._convert_to_boolean(True) is True
        assert DataTypeValidator._convert_to_boolean("true") is True
        assert DataTypeValidator._convert_to_boolean("1") is True
        assert DataTypeValidator._convert_to_boolean("yes") is True

        assert DataTypeValidator._convert_to_boolean(False) is False
        assert DataTypeValidator._convert_to_boolean("false") is False
        assert DataTypeValidator._convert_to_boolean("0") is False
        assert DataTypeValidator._convert_to_boolean("no") is False

    def test_validate_and_convert(self):
        """Test validation and conversion with business logic."""
        spec = DatasetSpecification(
            name="test_price",
            download_method=DownloadMethod.ETL,
            download_key="etl:test",
            data_type=DataType.FLOAT,
            min_value=0.0
        )

        result = DataTypeValidator.validate_and_convert("50.5", DataType.FLOAT, spec)
        assert result == 50.5

        with pytest.raises(ValidationError):
            DataTypeValidator.validate_and_convert("-1", DataType.FLOAT, spec)


class TestValidationContext:
    """Test ValidationContext class."""

    def test_context_creation(self):
        """Test creating validation context."""
        context = ValidationContext()
        assert len(context.rules) == 0
        assert len(context.type_mappings) == 0

    def test_add_rules(self):
        """Test adding validation rules."""
        context = ValidationContext()

        from src.finlab_downloader.core.validation import ValidationRule
        rule = ValidationRule(
            name="test_rule",
            validator=lambda x: x > 0,
            error_message="Value must be positive"
        )

        context.add_rule("test_dataset", rule)
        rules = context.get_rules("test_dataset")
        assert len(rules) == 1
        assert rules[0].name == "test_rule"

    def test_taiwan_market_context(self):
        """Test Taiwan market validation context."""
        context = create_taiwan_market_context()

        # Should have rules for various patterns
        stock_rules = context.get_rules("stock_code")
        assert len(stock_rules) > 0

        price_rules = context.get_rules("price")
        assert len(price_rules) > 0


if __name__ == "__main__":
    pytest.main([__file__])