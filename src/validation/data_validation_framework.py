"""
Data Validation Framework - Issue #57
Comprehensive data quality assurance and point-in-time integrity checks for FinLab datasets.

This module provides:
- Schema validation for all FinLab datasets
- Point-in-time data integrity checks
- Data quality metrics and scoring system
- Statistical outlier detection
- Validation reporting and alerting
- Data lineage tracking
- Automated data quality testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path
import hashlib
import warnings


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and pandas types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        return super().default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    else:
        return obj

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class DataType(Enum):
    """Supported data types for validation"""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


@dataclass
class SchemaField:
    """Schema definition for a single field"""
    name: str
    data_type: DataType
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None
    pattern: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    field_name: str
    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataQualityMetrics:
    """Data quality metrics for a dataset"""
    completeness_score: float  # % of non-null values
    accuracy_score: float     # % of values within expected ranges
    consistency_score: float  # % of consistent values across time
    validity_score: float     # % of values matching schema
    uniqueness_score: float   # % of unique values where expected
    timeliness_score: float   # % of recent data points
    overall_score: float      # Weighted average of all scores

    def to_dict(self) -> Dict[str, float]:
        return {
            'completeness': self.completeness_score,
            'accuracy': self.accuracy_score,
            'consistency': self.consistency_score,
            'validity': self.validity_score,
            'uniqueness': self.uniqueness_score,
            'timeliness': self.timeliness_score,
            'overall': self.overall_score
        }


@dataclass
class DataLineageRecord:
    """Data lineage tracking record"""
    dataset_name: str
    source_system: str
    extraction_method: str
    extraction_timestamp: datetime
    data_hash: str
    row_count: int
    column_count: int
    file_size: Optional[int] = None
    transformation_steps: List[str] = field(default_factory=list)
    quality_score: Optional[float] = None


class FinLabSchemaValidator:
    """Schema validation for FinLab datasets"""

    def __init__(self, schema_config_path: Optional[str] = None):
        self.schema_fields: Dict[str, SchemaField] = {}
        if schema_config_path:
            self.load_schema_config(schema_config_path)
        else:
            self._initialize_default_schema()

    def _initialize_default_schema(self):
        """Initialize default schema based on FinLab database structure"""
        # Price data fields
        price_fields = ['adj_close', 'adj_high', 'adj_low', 'adj_open']
        for field in price_fields:
            self.schema_fields[field] = SchemaField(
                name=field,
                data_type=DataType.FLOAT,
                nullable=False,
                min_value=0.0,
                description=f"Adjusted {field.split('_')[1]} price"
            )

        # Transaction volume fields
        transaction_fields = ['buy', 'sell']
        for field in transaction_fields:
            self.schema_fields[field] = SchemaField(
                name=field,
                data_type=DataType.INT,
                nullable=True,
                min_value=0,
                description=f"Top 15 broker {field} transactions"
            )

        # Financial statement fields (all float, nullable)
        financial_fields = [
            '一年內到期長期負債', '不動產廠房及設備', '使用權資產', '保留盈餘',
            '停業單位損益', '償還公司債', '償還長期借款', '共同控制下前手權益'
        ]
        for field in financial_fields:
            self.schema_fields[field] = SchemaField(
                name=field,
                data_type=DataType.FLOAT,
                nullable=True,
                description=f"Financial statement: {field}"
            )

    def load_schema_config(self, config_path: str):
        """Load schema configuration from CSV file"""
        try:
            schema_df = pd.read_csv(config_path)
            for _, row in schema_df.iterrows():
                field_name = row['資料集名稱']
                data_type_str = row['數據類型']

                # Map string to DataType enum
                data_type_map = {
                    'float': DataType.FLOAT,
                    'int': DataType.INT,
                    'string': DataType.STRING,
                    'datetime': DataType.DATETIME,
                    'boolean': DataType.BOOLEAN
                }

                data_type = data_type_map.get(data_type_str, DataType.FLOAT)

                # Set min_value based on data type
                min_value = 0.0 if data_type in [DataType.FLOAT, DataType.INT] else None

                self.schema_fields[field_name] = SchemaField(
                    name=field_name,
                    data_type=data_type,
                    nullable=True,
                    min_value=min_value,
                    description=row.get('下載方式及key', '')
                )

        except Exception as e:
            logger.error(f"Failed to load schema config: {e}")
            self._initialize_default_schema()

    def validate_schema(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate dataframe against schema"""
        results = []

        # Check for missing required columns
        required_columns = [name for name, field in self.schema_fields.items()
                          if not field.nullable]
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            results.append(ValidationResult(
                field_name="schema",
                check_name="missing_required_columns",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Missing required columns: {missing_columns}",
                details={"missing_columns": list(missing_columns)}
            ))

        # Check data types for existing columns
        for col in df.columns:
            if col in self.schema_fields:
                field_schema = self.schema_fields[col]
                results.extend(self._validate_field(df[col], field_schema))

        return results

    def _validate_field(self, series: pd.Series, schema: SchemaField) -> List[ValidationResult]:
        """Validate a single field against its schema"""
        results = []
        field_name = schema.name

        # Check nullability
        if not schema.nullable and series.isnull().any():
            null_count = series.isnull().sum()
            results.append(ValidationResult(
                field_name=field_name,
                check_name="null_check",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Found {null_count} null values in non-nullable field",
                details={"null_count": null_count, "total_count": len(series)}
            ))

        # Check data type compatibility
        non_null_series = series.dropna()
        if len(non_null_series) > 0:
            if schema.data_type == DataType.FLOAT:
                try:
                    pd.to_numeric(non_null_series, errors='raise')
                    results.append(ValidationResult(
                        field_name=field_name,
                        check_name="data_type_check",
                        severity=ValidationSeverity.INFO,
                        passed=True,
                        message="Data type validation passed"
                    ))
                except:
                    results.append(ValidationResult(
                        field_name=field_name,
                        check_name="data_type_check",
                        severity=ValidationSeverity.HIGH,
                        passed=False,
                        message="Values cannot be converted to float"
                    ))

            # Check value ranges
            if schema.min_value is not None or schema.max_value is not None:
                numeric_series = pd.to_numeric(non_null_series, errors='coerce')

                if schema.min_value is not None:
                    violations = numeric_series < schema.min_value
                    if violations.any():
                        violation_count = violations.sum()
                        results.append(ValidationResult(
                            field_name=field_name,
                            check_name="min_value_check",
                            severity=ValidationSeverity.MEDIUM,
                            passed=False,
                            message=f"Found {violation_count} values below minimum {schema.min_value}",
                            details={"violation_count": violation_count, "min_value": schema.min_value}
                        ))

                if schema.max_value is not None:
                    violations = numeric_series > schema.max_value
                    if violations.any():
                        violation_count = violations.sum()
                        results.append(ValidationResult(
                            field_name=field_name,
                            check_name="max_value_check",
                            severity=ValidationSeverity.MEDIUM,
                            passed=False,
                            message=f"Found {violation_count} values above maximum {schema.max_value}",
                            details={"violation_count": violation_count, "max_value": schema.max_value}
                        ))

        return results


class PointInTimeValidator:
    """Point-in-time data integrity validation"""

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days

    def validate_temporal_consistency(self, df: pd.DataFrame,
                                    date_column: str = 'date',
                                    value_columns: List[str] = None) -> List[ValidationResult]:
        """Validate temporal consistency of data"""
        results = []

        if date_column not in df.columns:
            results.append(ValidationResult(
                field_name=date_column,
                check_name="temporal_date_column_missing",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Date column '{date_column}' not found in dataset"
            ))
            return results

        # Check for duplicate timestamps
        duplicate_dates = df[date_column].duplicated().sum()
        if duplicate_dates > 0:
            results.append(ValidationResult(
                field_name=date_column,
                check_name="duplicate_timestamps",
                severity=ValidationSeverity.HIGH,
                passed=False,
                message=f"Found {duplicate_dates} duplicate timestamps",
                details={"duplicate_count": duplicate_dates}
            ))

        # Check for chronological order
        df_sorted = df.sort_values(date_column)
        if not df[date_column].equals(df_sorted[date_column]):
            results.append(ValidationResult(
                field_name=date_column,
                check_name="chronological_order",
                severity=ValidationSeverity.MEDIUM,
                passed=False,
                message="Data is not in chronological order"
            ))

        # Check for gaps in time series
        if value_columns:
            for col in value_columns:
                if col in df.columns:
                    gaps = self._detect_time_gaps(df, date_column, col)
                    if gaps:
                        results.append(ValidationResult(
                            field_name=col,
                            check_name="time_series_gaps",
                            severity=ValidationSeverity.MEDIUM,
                            passed=False,
                            message=f"Found {len(gaps)} gaps in time series",
                            details={"gaps": gaps[:10]}  # Limit to first 10 gaps
                        ))

        return results

    def _detect_time_gaps(self, df: pd.DataFrame, date_col: str, value_col: str,
                         max_gap_days: int = 7) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in time series data"""
        df_sorted = df.sort_values(date_col)
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])

        gaps = []
        prev_date = None

        for current_date in df_sorted[date_col]:
            if prev_date is not None:
                gap_days = (current_date - prev_date).days
                if gap_days > max_gap_days:
                    gaps.append((prev_date, current_date))
            prev_date = current_date

        return gaps

    def validate_point_in_time_integrity(self, df: pd.DataFrame,
                                       as_of_date: datetime,
                                       date_column: str = 'date') -> List[ValidationResult]:
        """Validate point-in-time data integrity"""
        results = []

        # Check that no future data is present
        future_data = df[pd.to_datetime(df[date_column]) > as_of_date]
        if not future_data.empty:
            results.append(ValidationResult(
                field_name=date_column,
                check_name="future_data_leak",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message=f"Found {len(future_data)} records with future dates",
                details={"future_records": len(future_data), "as_of_date": as_of_date.isoformat()}
            ))

        # Check data freshness
        latest_date = pd.to_datetime(df[date_column]).max()
        days_old = (as_of_date - latest_date).days

        if days_old > self.lookback_days:
            results.append(ValidationResult(
                field_name=date_column,
                check_name="data_staleness",
                severity=ValidationSeverity.HIGH,
                passed=False,
                message=f"Latest data is {days_old} days old, exceeds threshold of {self.lookback_days} days",
                details={"days_old": days_old, "latest_date": latest_date.isoformat()}
            ))

        return results


class StatisticalOutlierDetector:
    """Statistical outlier detection for data inconsistencies"""

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect_outliers(self, df: pd.DataFrame,
                       numeric_columns: List[str] = None) -> List[ValidationResult]:
        """Detect statistical outliers in numeric columns"""
        results = []

        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_columns:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) < 10:  # Skip if too few data points
                continue

            # Z-score based outlier detection
            z_scores = np.abs((series - series.mean()) / series.std())
            z_outliers = z_scores > self.z_threshold
            z_outlier_count = z_outliers.sum()

            # IQR based outlier detection
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            iqr_outliers = (series < lower_bound) | (series > upper_bound)
            iqr_outlier_count = iqr_outliers.sum()

            # Report outliers if found
            if z_outlier_count > 0 or iqr_outlier_count > 0:
                severity = ValidationSeverity.HIGH if max(z_outlier_count, iqr_outlier_count) > len(series) * 0.05 else ValidationSeverity.MEDIUM

                results.append(ValidationResult(
                    field_name=col,
                    check_name="statistical_outliers",
                    severity=severity,
                    passed=z_outlier_count == 0 and iqr_outlier_count == 0,
                    message=f"Found outliers: {z_outlier_count} (Z-score), {iqr_outlier_count} (IQR)",
                    details={
                        "z_score_outliers": z_outlier_count,
                        "iqr_outliers": iqr_outlier_count,
                        "total_records": len(series),
                        "outlier_percentage": max(z_outlier_count, iqr_outlier_count) / len(series) * 100
                    }
                ))

        return results

    def detect_anomalous_patterns(self, df: pd.DataFrame,
                                 date_column: str = 'date',
                                 value_columns: List[str] = None) -> List[ValidationResult]:
        """Detect anomalous patterns in time series data"""
        results = []

        if value_columns is None:
            value_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in value_columns:
            if col not in df.columns or col == date_column:
                continue

            # Sort by date
            df_sorted = df.sort_values(date_column)
            series = df_sorted[col].dropna()

            if len(series) < 30:  # Need sufficient data for pattern detection
                continue

            # Detect sudden spikes
            rolling_mean = series.rolling(window=7, center=True).mean()
            rolling_std = series.rolling(window=7, center=True).std()

            spikes = np.abs(series - rolling_mean) > 3 * rolling_std
            spike_count = spikes.sum()

            if spike_count > 0:
                results.append(ValidationResult(
                    field_name=col,
                    check_name="anomalous_spikes",
                    severity=ValidationSeverity.MEDIUM,
                    passed=False,
                    message=f"Found {spike_count} anomalous spikes in time series",
                    details={
                        "spike_count": spike_count,
                        "spike_percentage": spike_count / len(series) * 100
                    }
                ))

        return results


class DataQualityScorer:
    """Data quality metrics and scoring system"""

    def __init__(self):
        self.weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.20,
            'validity': 0.15,
            'uniqueness': 0.10,
            'timeliness': 0.05
        }

    def calculate_quality_metrics(self, df: pd.DataFrame,
                                validation_results: List[ValidationResult],
                                date_column: str = 'date',
                                unique_columns: List[str] = None) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics"""

        # Completeness: percentage of non-null values
        total_cells = df.size
        non_null_cells = df.count().sum()
        completeness = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0

        # Accuracy: percentage of values passing validation checks
        accuracy_checks = [r for r in validation_results
                         if r.check_name in ['min_value_check', 'max_value_check', 'data_type_check']]
        accuracy = (len([r for r in accuracy_checks if r.passed]) / len(accuracy_checks) * 100
                   if accuracy_checks else 100)

        # Consistency: temporal consistency score
        consistency_checks = [r for r in validation_results
                            if r.check_name in ['temporal_consistency', 'chronological_order']]
        consistency = (len([r for r in consistency_checks if r.passed]) / len(consistency_checks) * 100
                      if consistency_checks else 100)

        # Validity: schema compliance score
        validity_checks = [r for r in validation_results
                         if r.check_name in ['schema', 'null_check']]
        validity = (len([r for r in validity_checks if r.passed]) / len(validity_checks) * 100
                   if validity_checks else 100)

        # Uniqueness: uniqueness where expected
        uniqueness = 100  # Default to 100% if no unique columns specified
        if unique_columns:
            unique_scores = []
            for col in unique_columns:
                if col in df.columns:
                    unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
                    unique_scores.append(unique_ratio * 100)
            uniqueness = np.mean(unique_scores) if unique_scores else 100

        # Timeliness: recency of data
        timeliness = 100  # Default to 100%
        if date_column in df.columns:
            latest_date = pd.to_datetime(df[date_column]).max()
            days_old = (datetime.now() - latest_date).days
            timeliness = max(0, 100 - (days_old * 2))  # Lose 2% per day

        # Calculate overall score
        overall = (
            completeness * self.weights['completeness'] +
            accuracy * self.weights['accuracy'] +
            consistency * self.weights['consistency'] +
            validity * self.weights['validity'] +
            uniqueness * self.weights['uniqueness'] +
            timeliness * self.weights['timeliness']
        )

        return DataQualityMetrics(
            completeness_score=completeness,
            accuracy_score=accuracy,
            consistency_score=consistency,
            validity_score=validity,
            uniqueness_score=uniqueness,
            timeliness_score=timeliness,
            overall_score=overall
        )


class DataLineageTracker:
    """Data lineage and provenance tracking"""

    def __init__(self, lineage_db_path: str = "data_lineage.json"):
        self.lineage_db_path = Path(lineage_db_path)
        self.lineage_records: List[DataLineageRecord] = []
        self.load_lineage_db()

    def load_lineage_db(self):
        """Load existing lineage records"""
        if self.lineage_db_path.exists():
            try:
                with open(self.lineage_db_path, 'r') as f:
                    data = json.load(f)
                    self.lineage_records = []
                    for record in data:
                        # Convert string timestamp back to datetime
                        if isinstance(record['extraction_timestamp'], str):
                            record['extraction_timestamp'] = datetime.fromisoformat(record['extraction_timestamp'])
                        self.lineage_records.append(DataLineageRecord(**record))
            except Exception as e:
                logger.error(f"Failed to load lineage database: {e}")
                self.lineage_records = []

    def save_lineage_db(self):
        """Save lineage records to database"""
        try:
            data = []
            for record in self.lineage_records:
                record_dict = {
                    'dataset_name': record.dataset_name,
                    'source_system': record.source_system,
                    'extraction_method': record.extraction_method,
                    'extraction_timestamp': record.extraction_timestamp.isoformat(),
                    'data_hash': record.data_hash,
                    'row_count': record.row_count,
                    'column_count': record.column_count,
                    'file_size': record.file_size,
                    'transformation_steps': record.transformation_steps,
                    'quality_score': record.quality_score
                }
                data.append(record_dict)

            with open(self.lineage_db_path, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)

        except Exception as e:
            logger.error(f"Failed to save lineage database: {e}")

    def track_dataset(self, df: pd.DataFrame, dataset_name: str,
                     source_system: str = "finlab",
                     extraction_method: str = "etl",
                     transformation_steps: List[str] = None,
                     quality_score: Optional[float] = None) -> str:
        """Track a dataset and return its hash"""

        # Calculate data hash
        data_string = df.to_string()
        data_hash = hashlib.md5(data_string.encode()).hexdigest()

        # Create lineage record
        record = DataLineageRecord(
            dataset_name=dataset_name,
            source_system=source_system,
            extraction_method=extraction_method,
            extraction_timestamp=datetime.now(),
            data_hash=data_hash,
            row_count=len(df),
            column_count=len(df.columns),
            transformation_steps=transformation_steps or [],
            quality_score=quality_score
        )

        # Add to records
        self.lineage_records.append(record)
        self.save_lineage_db()

        return data_hash

    def get_dataset_history(self, dataset_name: str) -> List[DataLineageRecord]:
        """Get history of a dataset"""
        return [record for record in self.lineage_records
                if record.dataset_name == dataset_name]

    def validate_data_lineage(self, df: pd.DataFrame, expected_hash: str) -> ValidationResult:
        """Validate data against expected hash"""
        data_string = df.to_string()
        actual_hash = hashlib.md5(data_string.encode()).hexdigest()

        return ValidationResult(
            field_name="data_lineage",
            check_name="hash_validation",
            severity=ValidationSeverity.HIGH,
            passed=actual_hash == expected_hash,
            message="Data hash validation" + (" passed" if actual_hash == expected_hash else " failed"),
            details={
                "expected_hash": expected_hash,
                "actual_hash": actual_hash
            }
        )


class ValidationReporter:
    """Validation reporting and alerting system"""

    def __init__(self, report_dir: str = "validation_reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)

    def generate_report(self, validation_results: List[ValidationResult],
                       quality_metrics: DataQualityMetrics,
                       dataset_name: str) -> Dict[str, Any]:
        """Generate comprehensive validation report"""

        # Categorize results by severity
        results_by_severity = {}
        for severity in ValidationSeverity:
            results_by_severity[severity.value] = [
                r for r in validation_results if r.severity == severity
            ]

        # Count pass/fail
        total_checks = len(validation_results)
        passed_checks = len([r for r in validation_results if r.passed])
        failed_checks = total_checks - passed_checks

        report = {
            "dataset_name": dataset_name,
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "pass_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 100
            },
            "quality_metrics": quality_metrics.to_dict(),
            "results_by_severity": {
                severity: len(results) for severity, results in results_by_severity.items()
            },
            "validation_results": [
                {
                    "field_name": r.field_name,
                    "check_name": r.check_name,
                    "severity": r.severity.value,
                    "passed": bool(r.passed),
                    "message": r.message,
                    "details": convert_numpy_types(r.details) if r.details else None,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in validation_results
            ]
        }

        return report

    def save_report(self, report: Dict[str, Any], dataset_name: str) -> str:
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_validation_report_{timestamp}.json"
        filepath = self.report_dir / filename

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

        return str(filepath)

    def check_alert_conditions(self, quality_metrics: DataQualityMetrics,
                             validation_results: List[ValidationResult]) -> List[str]:
        """Check conditions that should trigger alerts"""
        alerts = []

        # Overall quality score too low
        if quality_metrics.overall_score < 70:
            alerts.append(f"Overall data quality score is {quality_metrics.overall_score:.1f}%, below threshold of 70%")

        # Critical validation failures
        critical_failures = [r for r in validation_results
                           if r.severity == ValidationSeverity.CRITICAL and not r.passed]
        if critical_failures:
            alerts.append(f"Found {len(critical_failures)} critical validation failures")

        # High number of failed checks
        failed_checks = len([r for r in validation_results if not r.passed])
        total_checks = len(validation_results)
        if total_checks > 0 and (failed_checks / total_checks) > 0.3:
            alerts.append(f"High failure rate: {failed_checks}/{total_checks} checks failed")

        return alerts


class DataValidationFramework:
    """Main data validation framework orchestrator"""

    def __init__(self, schema_config_path: Optional[str] = None,
                 lineage_db_path: str = "data_lineage.json",
                 report_dir: str = "validation_reports"):

        self.schema_validator = FinLabSchemaValidator(schema_config_path)
        self.pit_validator = PointInTimeValidator()
        self.outlier_detector = StatisticalOutlierDetector()
        self.quality_scorer = DataQualityScorer()
        self.lineage_tracker = DataLineageTracker(lineage_db_path)
        self.reporter = ValidationReporter(report_dir)

    def validate_dataset(self, df: pd.DataFrame,
                        dataset_name: str,
                        as_of_date: Optional[datetime] = None,
                        date_column: str = 'date',
                        unique_columns: List[str] = None,
                        generate_report: bool = True) -> Dict[str, Any]:
        """Complete dataset validation"""

        if as_of_date is None:
            as_of_date = datetime.now()

        # Collect all validation results
        validation_results = []

        # 1. Schema validation
        logger.info(f"Running schema validation for {dataset_name}")
        schema_results = self.schema_validator.validate_schema(df)
        validation_results.extend(schema_results)

        # 2. Point-in-time validation
        logger.info(f"Running point-in-time validation for {dataset_name}")
        pit_results = self.pit_validator.validate_point_in_time_integrity(
            df, as_of_date, date_column
        )
        validation_results.extend(pit_results)

        # 3. Temporal consistency validation
        temporal_results = self.pit_validator.validate_temporal_consistency(
            df, date_column, df.select_dtypes(include=[np.number]).columns.tolist()
        )
        validation_results.extend(temporal_results)

        # 4. Statistical outlier detection
        logger.info(f"Running outlier detection for {dataset_name}")
        outlier_results = self.outlier_detector.detect_outliers(df)
        validation_results.extend(outlier_results)

        # 5. Anomalous pattern detection
        pattern_results = self.outlier_detector.detect_anomalous_patterns(
            df, date_column
        )
        validation_results.extend(pattern_results)

        # 6. Calculate quality metrics
        logger.info(f"Calculating quality metrics for {dataset_name}")
        quality_metrics = self.quality_scorer.calculate_quality_metrics(
            df, validation_results, date_column, unique_columns
        )

        # 7. Track data lineage
        logger.info(f"Tracking data lineage for {dataset_name}")
        data_hash = self.lineage_tracker.track_dataset(
            df, dataset_name, quality_score=quality_metrics.overall_score
        )

        # 8. Generate report
        report = None
        if generate_report:
            logger.info(f"Generating validation report for {dataset_name}")
            report = self.reporter.generate_report(
                validation_results, quality_metrics, dataset_name
            )
            report_path = self.reporter.save_report(report, dataset_name)
            logger.info(f"Validation report saved to {report_path}")

        # 9. Check alert conditions
        alerts = self.reporter.check_alert_conditions(quality_metrics, validation_results)
        if alerts:
            for alert in alerts:
                logger.warning(f"ALERT: {alert}")

        return {
            "dataset_name": dataset_name,
            "data_hash": data_hash,
            "validation_results": validation_results,
            "quality_metrics": quality_metrics,
            "report": report,
            "alerts": alerts,
            "summary": {
                "total_checks": len(validation_results),
                "passed_checks": len([r for r in validation_results if r.passed]),
                "failed_checks": len([r for r in validation_results if not r.passed]),
                "overall_quality_score": quality_metrics.overall_score
            }
        }

    def validate_multiple_datasets(self, datasets: Dict[str, pd.DataFrame],
                                 as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Validate multiple datasets"""
        results = {}

        for dataset_name, df in datasets.items():
            logger.info(f"Validating dataset: {dataset_name}")
            results[dataset_name] = self.validate_dataset(
                df, dataset_name, as_of_date
            )

        # Generate summary across all datasets
        overall_summary = {
            "total_datasets": len(datasets),
            "datasets_validated": len(results),
            "average_quality_score": np.mean([
                result["quality_metrics"].overall_score for result in results.values()
            ]),
            "total_alerts": sum([len(result["alerts"]) for result in results.values()])
        }

        return {
            "validation_timestamp": datetime.now().isoformat(),
            "overall_summary": overall_summary,
            "dataset_results": results
        }


# Configuration management
class ValidationConfig:
    """Configuration management for validation framework"""

    def __init__(self, config_path: str = "validation_config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load validation configuration"""
        default_config = {
            "schema_validation": {
                "enabled": True,
                "strict_mode": False
            },
            "outlier_detection": {
                "z_threshold": 3.0,
                "iqr_multiplier": 1.5,
                "enabled": True
            },
            "quality_scoring": {
                "weights": {
                    "completeness": 0.25,
                    "accuracy": 0.25,
                    "consistency": 0.20,
                    "validity": 0.15,
                    "uniqueness": 0.10,
                    "timeliness": 0.05
                },
                "thresholds": {
                    "excellent": 90,
                    "good": 80,
                    "fair": 70,
                    "poor": 60
                }
            },
            "alerts": {
                "enabled": True,
                "critical_threshold": 0.1,
                "quality_threshold": 70
            },
            "reporting": {
                "auto_generate": True,
                "report_dir": "validation_reports"
            }
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                logger.error(f"Failed to load config, using defaults: {e}")

        return default_config

    def save_config(self):
        """Save current configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2, cls=NumpyEncoder)

    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save_config()