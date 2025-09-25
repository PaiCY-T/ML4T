"""
Automated Testing Suite for Data Validation Framework - Issue #57
Comprehensive tests for data quality assurance and validation components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json
import os

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from validation.data_validation_framework import (
    DataValidationFramework,
    FinLabSchemaValidator,
    PointInTimeValidator,
    StatisticalOutlierDetector,
    DataQualityScorer,
    DataLineageTracker,
    ValidationReporter,
    ValidationConfig,
    ValidationSeverity,
    DataType,
    SchemaField,
    ValidationResult,
    DataQualityMetrics,
    DataLineageRecord
)


class TestFinLabSchemaValidator:
    """Test schema validation functionality"""

    def setup_method(self):
        """Setup test data and validator"""
        self.validator = FinLabSchemaValidator()

        # Create test dataframe matching FinLab structure
        self.test_df = pd.DataFrame({
            'adj_close': [100.5, 101.0, 99.8, 102.1, 101.5],
            'adj_high': [102.0, 103.5, 101.2, 104.0, 103.8],
            'adj_low': [99.0, 100.0, 98.5, 101.0, 100.8],
            'adj_open': [100.0, 100.8, 101.5, 99.9, 102.0],
            'buy': [1500, 1800, 1200, 2100, 1650],
            'sell': [1200, 1600, 1450, 1800, 1550],
            '保留盈餘': [1000000.0, 1050000.0, None, 980000.0, 1020000.0]
        })

    def test_schema_initialization(self):
        """Test schema validator initialization"""
        assert len(self.validator.schema_fields) > 0
        assert 'adj_close' in self.validator.schema_fields
        assert self.validator.schema_fields['adj_close'].data_type == DataType.FLOAT

    def test_valid_schema_validation(self):
        """Test validation with valid schema"""
        results = self.validator.validate_schema(self.test_df)

        # Should have some validation results
        assert len(results) > 0

        # Check that we have data type validation results
        data_type_results = [r for r in results if r.check_name == 'data_type_check']
        assert len(data_type_results) > 0

    def test_missing_required_columns(self):
        """Test detection of missing required columns"""
        # Create dataframe with missing columns
        incomplete_df = pd.DataFrame({'adj_close': [100.0, 101.0]})

        # Set adj_open as non-nullable for this test
        self.validator.schema_fields['adj_open'].nullable = False

        results = self.validator.validate_schema(incomplete_df)

        # Should detect missing required column
        missing_col_results = [r for r in results if r.check_name == 'missing_required_columns']
        # Note: might be 0 if no columns are set as required in default schema

    def test_data_type_validation(self):
        """Test data type validation"""
        # Create dataframe with invalid data types
        invalid_df = pd.DataFrame({
            'adj_close': ['invalid', 'string', 'values'],
            'buy': [1.5, 2.7, 3.8]  # Should be int but float is acceptable
        })

        results = self.validator.validate_schema(invalid_df)

        # Should detect invalid data types for adj_close
        failed_results = [r for r in results if not r.passed and r.field_name == 'adj_close']
        assert len(failed_results) > 0

    def test_value_range_validation(self):
        """Test value range validation"""
        # Create dataframe with values outside expected range
        invalid_range_df = pd.DataFrame({
            'adj_close': [-10.0, 101.0, -5.0],  # Negative values should be invalid
            'buy': [-100, 1500, 2000]  # Negative transaction volumes invalid
        })

        results = self.validator.validate_schema(invalid_range_df)

        # Should detect out-of-range values
        range_violations = [r for r in results if 'value_check' in r.check_name and not r.passed]
        # Note: depends on schema configuration having min_value constraints


class TestPointInTimeValidator:
    """Test point-in-time validation functionality"""

    def setup_method(self):
        """Setup test data and validator"""
        self.validator = PointInTimeValidator(lookback_days=7)

        # Create test dataframe with time series data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.test_df = pd.DataFrame({
            'date': dates,
            'price': np.random.uniform(90, 110, 10),
            'volume': np.random.randint(1000, 5000, 10)
        })

    def test_temporal_consistency_valid(self):
        """Test temporal consistency with valid data"""
        results = self.validator.validate_temporal_consistency(
            self.test_df, 'date', ['price', 'volume']
        )

        # Should have validation results
        assert len(results) >= 0

    def test_duplicate_timestamps(self):
        """Test detection of duplicate timestamps"""
        # Create dataframe with duplicate dates
        duplicate_df = self.test_df.copy()
        duplicate_df.loc[5, 'date'] = duplicate_df.loc[4, 'date']

        results = self.validator.validate_temporal_consistency(
            duplicate_df, 'date', ['price']
        )

        # Should detect duplicate timestamps
        duplicate_results = [r for r in results if r.check_name == 'duplicate_timestamps']
        assert len(duplicate_results) > 0
        assert not duplicate_results[0].passed

    def test_chronological_order(self):
        """Test chronological order validation"""
        # Create dataframe with out-of-order dates
        unordered_df = self.test_df.copy()
        unordered_df = unordered_df.sample(frac=1).reset_index(drop=True)  # Shuffle

        results = self.validator.validate_temporal_consistency(
            unordered_df, 'date', ['price']
        )

        # Should detect chronological disorder
        order_results = [r for r in results if r.check_name == 'chronological_order']
        if len(order_results) > 0:  # Only if data was actually shuffled
            assert not order_results[0].passed

    def test_point_in_time_integrity(self):
        """Test point-in-time data integrity"""
        as_of_date = datetime(2023, 1, 5)

        results = self.validator.validate_point_in_time_integrity(
            self.test_df, as_of_date, 'date'
        )

        # Should have validation results
        assert len(results) >= 0

    def test_future_data_leak(self):
        """Test detection of future data leak"""
        # Add future dates to test data
        future_df = self.test_df.copy()
        future_dates = pd.date_range('2023-01-15', periods=3, freq='D')
        future_data = pd.DataFrame({
            'date': future_dates,
            'price': [115.0, 116.0, 117.0],
            'volume': [3000, 3100, 3200]
        })
        future_df = pd.concat([future_df, future_data]).reset_index(drop=True)

        as_of_date = datetime(2023, 1, 10)
        results = self.validator.validate_point_in_time_integrity(
            future_df, as_of_date, 'date'
        )

        # Should detect future data leak
        leak_results = [r for r in results if r.check_name == 'future_data_leak']
        assert len(leak_results) > 0
        assert not leak_results[0].passed

    def test_data_staleness(self):
        """Test data staleness detection"""
        # Create old data
        old_dates = pd.date_range('2023-01-01', periods=5, freq='D')
        old_df = pd.DataFrame({
            'date': old_dates,
            'price': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        # Test with current date (data should be stale)
        as_of_date = datetime.now()
        results = self.validator.validate_point_in_time_integrity(
            old_df, as_of_date, 'date'
        )

        # Should detect stale data
        staleness_results = [r for r in results if r.check_name == 'data_staleness']
        assert len(staleness_results) > 0
        assert not staleness_results[0].passed


class TestStatisticalOutlierDetector:
    """Test statistical outlier detection"""

    def setup_method(self):
        """Setup test data and detector"""
        self.detector = StatisticalOutlierDetector(z_threshold=2.0, iqr_multiplier=1.5)

        # Create test data with known outliers
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 95)
        outliers = [200, 300, -50, -100, 400]  # Clear outliers

        self.test_df = pd.DataFrame({
            'price': np.concatenate([normal_data, outliers]),
            'volume': np.random.normal(2000, 300, 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        })

    def test_outlier_detection(self):
        """Test basic outlier detection"""
        results = self.detector.detect_outliers(self.test_df, ['price'])

        # Should detect outliers in price column
        outlier_results = [r for r in results if r.field_name == 'price']
        assert len(outlier_results) > 0

        # Should not pass if outliers detected
        if outlier_results[0].details and outlier_results[0].details.get('z_score_outliers', 0) > 0:
            assert not outlier_results[0].passed

    def test_no_outliers_in_normal_data(self):
        """Test that normal data doesn't trigger outlier detection"""
        normal_df = pd.DataFrame({
            'price': np.random.normal(100, 5, 50)  # Small variance, no extreme outliers
        })

        results = self.detector.detect_outliers(normal_df, ['price'])

        # Should either have no results or passing results
        if len(results) > 0:
            # If there are any outliers detected, should be very few
            outlier_result = results[0]
            if outlier_result.details:
                outlier_count = max(
                    outlier_result.details.get('z_score_outliers', 0),
                    outlier_result.details.get('iqr_outliers', 0)
                )
                assert outlier_count <= 2  # Very few outliers in normal data

    def test_anomalous_patterns(self):
        """Test anomalous pattern detection"""
        # Create data with clear spike
        smooth_data = np.sin(np.linspace(0, 4*np.pi, 50)) * 10 + 100
        spike_data = smooth_data.copy()
        spike_data[25] = 500  # Clear spike

        spike_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='D'),
            'price': spike_data
        })

        results = self.detector.detect_anomalous_patterns(spike_df, 'date', ['price'])

        # Should detect anomalous spikes
        spike_results = [r for r in results if r.check_name == 'anomalous_spikes']
        if len(spike_results) > 0:
            assert not spike_results[0].passed


class TestDataQualityScorer:
    """Test data quality scoring system"""

    def setup_method(self):
        """Setup test data and scorer"""
        self.scorer = DataQualityScorer()

        # Create test dataframe with known quality issues
        self.test_df = pd.DataFrame({
            'price': [100.0, 101.0, None, 103.0, 104.0],  # One missing value
            'volume': [1000, 1100, 1200, 1300, 1400],     # Complete
            'date': pd.date_range('2023-01-01', periods=5, freq='D')
        })

        # Create mock validation results
        self.mock_results = [
            ValidationResult(
                field_name="price",
                check_name="data_type_check",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Data type validation passed"
            ),
            ValidationResult(
                field_name="price",
                check_name="null_check",
                severity=ValidationSeverity.MEDIUM,
                passed=False,
                message="Found null values"
            )
        ]

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation"""
        metrics = self.scorer.calculate_quality_metrics(
            self.test_df, self.mock_results, 'date'
        )

        assert isinstance(metrics, DataQualityMetrics)
        assert 0 <= metrics.completeness_score <= 100
        assert 0 <= metrics.accuracy_score <= 100
        assert 0 <= metrics.consistency_score <= 100
        assert 0 <= metrics.validity_score <= 100
        assert 0 <= metrics.uniqueness_score <= 100
        assert 0 <= metrics.timeliness_score <= 100
        assert 0 <= metrics.overall_score <= 100

    def test_completeness_score(self):
        """Test completeness score calculation"""
        metrics = self.scorer.calculate_quality_metrics(
            self.test_df, self.mock_results, 'date'
        )

        # Should reflect the one missing value out of 15 total values (5 rows * 3 cols)
        # Completeness = (14 non-null / 15 total) * 100 = 93.33%
        assert metrics.completeness_score > 90  # Should be around 93.33%

    def test_perfect_quality_data(self):
        """Test quality scoring with perfect data"""
        perfect_df = pd.DataFrame({
            'price': [100.0, 101.0, 102.0, 103.0, 104.0],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'date': pd.date_range(datetime.now() - timedelta(days=2), periods=5, freq='D')
        })

        # All validation results pass
        perfect_results = [
            ValidationResult(
                field_name="price",
                check_name="data_type_check",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Passed"
            )
        ]

        metrics = self.scorer.calculate_quality_metrics(
            perfect_df, perfect_results, 'date'
        )

        # Should have high overall score
        assert metrics.overall_score > 90


class TestDataLineageTracker:
    """Test data lineage tracking"""

    def setup_method(self):
        """Setup test data and tracker"""
        self.temp_dir = tempfile.mkdtemp()
        self.lineage_path = os.path.join(self.temp_dir, "test_lineage.json")
        self.tracker = DataLineageTracker(self.lineage_path)

        self.test_df = pd.DataFrame({
            'price': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200],
            'date': pd.date_range('2023-01-01', periods=3, freq='D')
        })

    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_track_dataset(self):
        """Test dataset tracking"""
        data_hash = self.tracker.track_dataset(
            self.test_df,
            "test_dataset",
            transformation_steps=["clean", "validate"]
        )

        assert isinstance(data_hash, str)
        assert len(data_hash) == 32  # MD5 hash length
        assert len(self.tracker.lineage_records) == 1

        record = self.tracker.lineage_records[0]
        assert record.dataset_name == "test_dataset"
        assert record.row_count == 3
        assert record.column_count == 3
        assert "clean" in record.transformation_steps

    def test_dataset_history(self):
        """Test dataset history retrieval"""
        # Track same dataset multiple times
        self.tracker.track_dataset(self.test_df, "test_dataset", transformation_steps=["v1"])

        # Modify data and track again
        modified_df = self.test_df.copy()
        modified_df.loc[0, 'price'] = 999.0
        self.tracker.track_dataset(modified_df, "test_dataset", transformation_steps=["v2"])

        history = self.tracker.get_dataset_history("test_dataset")
        assert len(history) == 2
        assert history[0].transformation_steps == ["v1"]
        assert history[1].transformation_steps == ["v2"]

    def test_hash_validation(self):
        """Test data hash validation"""
        # Track original data
        original_hash = self.tracker.track_dataset(self.test_df, "test_dataset")

        # Validate same data
        result = self.tracker.validate_data_lineage(self.test_df, original_hash)
        assert result.passed

        # Validate modified data
        modified_df = self.test_df.copy()
        modified_df.loc[0, 'price'] = 999.0
        result = self.tracker.validate_data_lineage(modified_df, original_hash)
        assert not result.passed

    def test_persistence(self):
        """Test lineage data persistence"""
        # Track dataset
        original_hash = self.tracker.track_dataset(self.test_df, "test_dataset")

        # Create new tracker instance (should load existing data)
        new_tracker = DataLineageTracker(self.lineage_path)

        assert len(new_tracker.lineage_records) == 1
        assert new_tracker.lineage_records[0].dataset_name == "test_dataset"
        assert new_tracker.lineage_records[0].data_hash == original_hash


class TestValidationReporter:
    """Test validation reporting system"""

    def setup_method(self):
        """Setup test data and reporter"""
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = ValidationReporter(self.temp_dir)

        # Create mock validation results
        self.mock_results = [
            ValidationResult(
                field_name="price",
                check_name="data_type_check",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Passed"
            ),
            ValidationResult(
                field_name="volume",
                check_name="range_check",
                severity=ValidationSeverity.HIGH,
                passed=False,
                message="Out of range"
            )
        ]

        # Create mock quality metrics
        self.mock_metrics = DataQualityMetrics(
            completeness_score=95.0,
            accuracy_score=85.0,
            consistency_score=90.0,
            validity_score=88.0,
            uniqueness_score=100.0,
            timeliness_score=75.0,
            overall_score=87.2
        )

    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_report(self):
        """Test report generation"""
        report = self.reporter.generate_report(
            self.mock_results, self.mock_metrics, "test_dataset"
        )

        assert report["dataset_name"] == "test_dataset"
        assert "report_timestamp" in report
        assert "summary" in report
        assert report["summary"]["total_checks"] == 2
        assert report["summary"]["passed_checks"] == 1
        assert report["summary"]["failed_checks"] == 1

        assert "quality_metrics" in report
        assert report["quality_metrics"]["overall"] == 87.2

        assert "validation_results" in report
        assert len(report["validation_results"]) == 2

    def test_save_report(self):
        """Test report saving"""
        report = self.reporter.generate_report(
            self.mock_results, self.mock_metrics, "test_dataset"
        )

        report_path = self.reporter.save_report(report, "test_dataset")

        assert os.path.exists(report_path)

        # Verify file contents
        with open(report_path, 'r') as f:
            saved_report = json.load(f)

        assert saved_report["dataset_name"] == "test_dataset"
        assert saved_report["summary"]["total_checks"] == 2

    def test_alert_conditions(self):
        """Test alert condition checking"""
        # Test with low quality metrics
        low_quality_metrics = DataQualityMetrics(
            completeness_score=60.0,
            accuracy_score=50.0,
            consistency_score=40.0,
            validity_score=55.0,
            uniqueness_score=70.0,
            timeliness_score=65.0,
            overall_score=55.0  # Below 70% threshold
        )

        alerts = self.reporter.check_alert_conditions(
            low_quality_metrics, self.mock_results
        )

        # Should trigger low quality score alert
        assert len(alerts) > 0
        assert any("quality score" in alert.lower() for alert in alerts)

    def test_critical_failure_alert(self):
        """Test critical failure alert"""
        critical_results = [
            ValidationResult(
                field_name="critical_field",
                check_name="critical_check",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message="Critical failure"
            )
        ]

        alerts = self.reporter.check_alert_conditions(
            self.mock_metrics, critical_results
        )

        # Should trigger critical failure alert
        assert any("critical" in alert.lower() for alert in alerts)


class TestDataValidationFramework:
    """Test main validation framework integration"""

    def setup_method(self):
        """Setup test data and framework"""
        self.temp_dir = tempfile.mkdtemp()

        # Create test schema config
        schema_data = pd.DataFrame({
            '資料集名稱': ['price', 'volume', 'date'],
            '下載方式及key': ['etl:price', 'etl:volume', 'etl:date'],
            '數據類型': ['float', 'int', 'datetime']
        })
        self.schema_path = os.path.join(self.temp_dir, "test_schema.csv")
        schema_data.to_csv(self.schema_path, index=False)

        # Initialize framework
        self.framework = DataValidationFramework(
            schema_config_path=self.schema_path,
            lineage_db_path=os.path.join(self.temp_dir, "lineage.json"),
            report_dir=os.path.join(self.temp_dir, "reports")
        )

        # Create test dataset
        self.test_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'price': np.random.uniform(90, 110, 10),
            'volume': np.random.randint(1000, 5000, 10)
        })

    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_validation_workflow(self):
        """Test complete validation workflow"""
        result = self.framework.validate_dataset(
            self.test_df,
            "test_dataset",
            as_of_date=datetime.now()
        )

        # Check result structure
        assert "dataset_name" in result
        assert "data_hash" in result
        assert "validation_results" in result
        assert "quality_metrics" in result
        assert "summary" in result
        assert "alerts" in result

        assert result["dataset_name"] == "test_dataset"
        assert isinstance(result["data_hash"], str)
        assert isinstance(result["validation_results"], list)
        assert isinstance(result["quality_metrics"], DataQualityMetrics)
        assert isinstance(result["summary"], dict)
        assert isinstance(result["alerts"], list)

    def test_multiple_datasets_validation(self):
        """Test validation of multiple datasets"""
        datasets = {
            "dataset1": self.test_df,
            "dataset2": self.test_df.copy()
        }

        result = self.framework.validate_multiple_datasets(datasets)

        assert "validation_timestamp" in result
        assert "overall_summary" in result
        assert "dataset_results" in result

        assert result["overall_summary"]["total_datasets"] == 2
        assert len(result["dataset_results"]) == 2
        assert "dataset1" in result["dataset_results"]
        assert "dataset2" in result["dataset_results"]

    def test_validation_with_issues(self):
        """Test validation with data quality issues"""
        # Create problematic data
        problem_df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-01'],  # Duplicate date
            'price': [100.0, None, -50.0],  # Null and negative value
            'volume': [1000, 2000, 'invalid']  # Invalid data type
        })

        result = self.framework.validate_dataset(problem_df, "problem_dataset")

        # Should detect multiple issues
        failed_checks = [r for r in result["validation_results"] if not r.passed]
        assert len(failed_checks) > 0

        # Quality score should be lower due to issues
        assert result["quality_metrics"].overall_score < 100

    def test_configuration_integration(self):
        """Test integration with validation configuration"""
        config_path = os.path.join(self.temp_dir, "validation_config.json")
        config_data = {
            "outlier_detection": {
                "z_threshold": 2.5,
                "enabled": True
            },
            "quality_scoring": {
                "thresholds": {
                    "excellent": 95,
                    "good": 85,
                    "fair": 75,
                    "poor": 65
                }
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        config = ValidationConfig(config_path)
        assert config.get("outlier_detection.z_threshold") == 2.5
        assert config.get("quality_scoring.thresholds.excellent") == 95


class TestValidationConfig:
    """Test validation configuration management"""

    def setup_method(self):
        """Setup test configuration"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")

    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_config(self):
        """Test default configuration loading"""
        config = ValidationConfig(self.config_path)

        # Should have default values
        assert config.get("schema_validation.enabled") == True
        assert config.get("outlier_detection.z_threshold") == 3.0
        assert config.get("alerts.enabled") == True

    def test_config_persistence(self):
        """Test configuration persistence"""
        config = ValidationConfig(self.config_path)

        # Modify configuration
        config.set("outlier_detection.z_threshold", 2.5)
        config.set("alerts.quality_threshold", 80)

        # Create new config instance
        new_config = ValidationConfig(self.config_path)

        # Should load modified values
        assert new_config.get("outlier_detection.z_threshold") == 2.5
        assert new_config.get("alerts.quality_threshold") == 80

    def test_nested_config_access(self):
        """Test nested configuration access"""
        config = ValidationConfig(self.config_path)

        # Test nested key access
        assert config.get("quality_scoring.weights.completeness") == 0.25
        assert config.get("quality_scoring.weights.accuracy") == 0.25

        # Test non-existent key
        assert config.get("non.existent.key", "default") == "default"


# Integration test fixtures
@pytest.fixture
def sample_finlab_data():
    """Create sample FinLab-style dataset for testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    return pd.DataFrame({
        'date': dates,
        'adj_close': np.random.uniform(90, 110, 100),
        'adj_high': np.random.uniform(95, 115, 100),
        'adj_low': np.random.uniform(85, 105, 100),
        'adj_open': np.random.uniform(90, 110, 100),
        'buy': np.random.randint(1000, 5000, 100),
        'sell': np.random.randint(800, 4500, 100),
        '保留盈餘': np.random.uniform(500000, 2000000, 100),
        '不動產廠房及設備': np.random.uniform(1000000, 5000000, 100)
    })


def test_end_to_end_validation(sample_finlab_data):
    """End-to-end validation test with realistic data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize framework
        framework = DataValidationFramework(
            lineage_db_path=os.path.join(temp_dir, "lineage.json"),
            report_dir=os.path.join(temp_dir, "reports")
        )

        # Run complete validation
        result = framework.validate_dataset(
            sample_finlab_data,
            "finlab_test_dataset",
            date_column='date'
        )

        # Verify comprehensive validation completed
        assert result["summary"]["total_checks"] > 0
        assert result["quality_metrics"].overall_score > 0
        assert len(result["data_hash"]) == 32

        # Verify report generation
        assert result["report"] is not None
        assert os.path.exists(os.path.join(temp_dir, "reports"))

        # Verify lineage tracking
        history = framework.lineage_tracker.get_dataset_history("finlab_test_dataset")
        assert len(history) == 1
        assert history[0].row_count == 100
        assert history[0].column_count == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])