"""
Data quality checker for comprehensive quality assessment and reporting.

Provides holistic data quality analysis including completeness, accuracy,
consistency, validity, and timeliness checks.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

from .financial_validators import ValidationResult, ValidationSeverity, ValidationCategory

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Data quality dimensions."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"


@dataclass
class QualityMetric:
    """Individual quality metric."""
    dimension: QualityDimension
    metric_name: str
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    passed_threshold: bool
    threshold: float


@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    dataset_name: str
    analysis_date: datetime
    total_rows: int
    total_columns: int
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    metrics: List[QualityMetric]
    issues: List[ValidationResult]
    recommendations: List[str]
    metadata: Dict[str, Any]


class DataQualityChecker:
    """
    Comprehensive data quality assessment tool.

    Evaluates data across multiple quality dimensions and provides
    actionable insights for data improvement.
    """

    def __init__(self, quality_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize quality checker.

        Args:
            quality_thresholds: Custom quality thresholds by dimension
        """
        self.quality_thresholds = quality_thresholds or {
            'completeness': 0.95,
            'accuracy': 0.90,
            'consistency': 0.95,
            'validity': 0.90,
            'timeliness': 0.85,
            'uniqueness': 0.98
        }

    def assess_quality(self, data: pd.DataFrame, dataset_name: str = "unknown") -> QualityReport:
        """
        Perform comprehensive quality assessment.

        Args:
            data: DataFrame to assess
            dataset_name: Name of the dataset

        Returns:
            Quality report
        """
        logger.info(f"Starting quality assessment for dataset: {dataset_name}")

        metrics = []
        issues = []

        # Completeness assessment
        completeness_metrics, completeness_issues = self._assess_completeness(data)
        metrics.extend(completeness_metrics)
        issues.extend(completeness_issues)

        # Accuracy assessment
        accuracy_metrics, accuracy_issues = self._assess_accuracy(data)
        metrics.extend(accuracy_metrics)
        issues.extend(accuracy_issues)

        # Consistency assessment
        consistency_metrics, consistency_issues = self._assess_consistency(data)
        metrics.extend(consistency_metrics)
        issues.extend(consistency_issues)

        # Validity assessment
        validity_metrics, validity_issues = self._assess_validity(data)
        metrics.extend(validity_metrics)
        issues.extend(validity_issues)

        # Timeliness assessment
        timeliness_metrics, timeliness_issues = self._assess_timeliness(data)
        metrics.extend(timeliness_metrics)
        issues.extend(timeliness_issues)

        # Uniqueness assessment
        uniqueness_metrics, uniqueness_issues = self._assess_uniqueness(data)
        metrics.extend(uniqueness_metrics)
        issues.extend(uniqueness_issues)

        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(metrics)

        # Calculate overall score
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, issues)

        # Create metadata
        metadata = {
            'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime']).columns)
        }

        report = QualityReport(
            dataset_name=dataset_name,
            analysis_date=datetime.utcnow(),
            total_rows=len(data),
            total_columns=len(data.columns),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations,
            metadata=metadata
        )

        logger.info(f"Quality assessment completed. Overall score: {overall_score:.3f}")
        return report

    def _assess_completeness(self, data: pd.DataFrame) -> Tuple[List[QualityMetric], List[ValidationResult]]:
        """Assess data completeness."""
        metrics = []
        issues = []

        # Overall completeness
        total_cells = data.size
        non_null_cells = data.count().sum()
        completeness_score = non_null_cells / total_cells if total_cells > 0 else 0

        metrics.append(QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            metric_name="overall_completeness",
            score=completeness_score,
            details={
                'total_cells': total_cells,
                'non_null_cells': non_null_cells,
                'null_cells': total_cells - non_null_cells
            },
            passed_threshold=completeness_score >= self.quality_thresholds['completeness'],
            threshold=self.quality_thresholds['completeness']
        ))

        # Column-wise completeness
        for column in data.columns:
            null_count = data[column].isnull().sum()
            col_completeness = 1 - (null_count / len(data))

            metrics.append(QualityMetric(
                dimension=QualityDimension.COMPLETENESS,
                metric_name=f"completeness_{column}",
                score=col_completeness,
                details={
                    'column': column,
                    'null_count': null_count,
                    'total_count': len(data),
                    'null_percentage': null_count / len(data) * 100
                },
                passed_threshold=col_completeness >= self.quality_thresholds['completeness'],
                threshold=self.quality_thresholds['completeness']
            ))

            # Create issue if completeness is low
            if col_completeness < self.quality_thresholds['completeness']:
                issues.append(ValidationResult(
                    rule_name="column_completeness",
                    category=ValidationCategory.COMPLETENESS,
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Column '{column}' has low completeness: {col_completeness:.3f}",
                    affected_rows=null_count,
                    affected_columns=[column],
                    details={'completeness_score': col_completeness, 'threshold': self.quality_thresholds['completeness']},
                    timestamp=datetime.utcnow()
                ))

        return metrics, issues

    def _assess_accuracy(self, data: pd.DataFrame) -> Tuple[List[QualityMetric], List[ValidationResult]]:
        """Assess data accuracy."""
        metrics = []
        issues = []

        # Numeric accuracy checks
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            series = data[column].dropna()
            if len(series) == 0:
                continue

            # Check for outliers using IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            accuracy_score = 1 - (outliers / len(series))

            metrics.append(QualityMetric(
                dimension=QualityDimension.ACCURACY,
                metric_name=f"accuracy_{column}",
                score=accuracy_score,
                details={
                    'column': column,
                    'outliers': outliers,
                    'total_values': len(series),
                    'outlier_percentage': outliers / len(series) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                },
                passed_threshold=accuracy_score >= self.quality_thresholds['accuracy'],
                threshold=self.quality_thresholds['accuracy']
            ))

            # Create issue if accuracy is low
            if accuracy_score < self.quality_thresholds['accuracy']:
                issues.append(ValidationResult(
                    rule_name="numeric_accuracy",
                    category=ValidationCategory.RANGE,
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Column '{column}' has potential accuracy issues: {outliers} outliers",
                    affected_rows=outliers,
                    affected_columns=[column],
                    details={'accuracy_score': accuracy_score, 'outliers': outliers},
                    timestamp=datetime.utcnow()
                ))

        # String accuracy checks
        string_columns = data.select_dtypes(include=['object']).columns

        for column in string_columns:
            series = data[column].dropna()
            if len(series) == 0:
                continue

            # Check for inconsistent formatting
            str_series = series.astype(str)

            # Check for leading/trailing whitespace
            whitespace_issues = (str_series != str_series.str.strip()).sum()

            # Check for mixed case issues (if data should be consistent)
            case_consistency = self._check_case_consistency(str_series)

            accuracy_score = 1 - ((whitespace_issues + case_consistency['inconsistent']) / len(series))

            metrics.append(QualityMetric(
                dimension=QualityDimension.ACCURACY,
                metric_name=f"string_accuracy_{column}",
                score=accuracy_score,
                details={
                    'column': column,
                    'whitespace_issues': whitespace_issues,
                    'case_issues': case_consistency['inconsistent'],
                    'total_values': len(series)
                },
                passed_threshold=accuracy_score >= self.quality_thresholds['accuracy'],
                threshold=self.quality_thresholds['accuracy']
            ))

        return metrics, issues

    def _assess_consistency(self, data: pd.DataFrame) -> Tuple[List[QualityMetric], List[ValidationResult]]:
        """Assess data consistency."""
        metrics = []
        issues = []

        # Cross-field consistency checks
        # OHLC consistency
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_data = data[['open', 'high', 'low', 'close']].dropna()

            if len(ohlc_data) > 0:
                inconsistent = (
                    (ohlc_data['high'] < ohlc_data['open']) |
                    (ohlc_data['high'] < ohlc_data['close']) |
                    (ohlc_data['low'] > ohlc_data['open']) |
                    (ohlc_data['low'] > ohlc_data['close'])
                ).sum()

                consistency_score = 1 - (inconsistent / len(ohlc_data))

                metrics.append(QualityMetric(
                    dimension=QualityDimension.CONSISTENCY,
                    metric_name="ohlc_consistency",
                    score=consistency_score,
                    details={
                        'inconsistent_rows': inconsistent,
                        'total_rows': len(ohlc_data),
                        'inconsistency_rate': inconsistent / len(ohlc_data)
                    },
                    passed_threshold=consistency_score >= self.quality_thresholds['consistency'],
                    threshold=self.quality_thresholds['consistency']
                ))

                if consistency_score < self.quality_thresholds['consistency']:
                    issues.append(ValidationResult(
                        rule_name="ohlc_consistency",
                        category=ValidationCategory.CONSISTENCY,
                        severity=ValidationSeverity.ERROR,
                        passed=False,
                        message=f"OHLC data inconsistency: {inconsistent} problematic rows",
                        affected_rows=inconsistent,
                        affected_columns=['open', 'high', 'low', 'close'],
                        details={'consistency_score': consistency_score},
                        timestamp=datetime.utcnow()
                    ))

        # Date sequence consistency
        date_columns = [col for col in data.columns if 'date' in col.lower()]
        for date_col in date_columns:
            if data[date_col].dtype == 'datetime64[ns]':
                date_series = data[date_col].dropna().sort_values()

                if len(date_series) > 1:
                    # Check for reasonable date progression
                    date_diffs = date_series.diff().dropna()

                    # Most differences should be within reasonable range (e.g., 1-7 days for daily data)
                    reasonable_diffs = ((date_diffs >= pd.Timedelta(days=1)) &
                                      (date_diffs <= pd.Timedelta(days=7))).sum()

                    consistency_score = reasonable_diffs / len(date_diffs) if len(date_diffs) > 0 else 1.0

                    metrics.append(QualityMetric(
                        dimension=QualityDimension.CONSISTENCY,
                        metric_name=f"date_sequence_{date_col}",
                        score=consistency_score,
                        details={
                            'column': date_col,
                            'reasonable_progressions': reasonable_diffs,
                            'total_progressions': len(date_diffs),
                            'avg_diff_days': date_diffs.mean().days if len(date_diffs) > 0 else 0
                        },
                        passed_threshold=consistency_score >= self.quality_thresholds['consistency'],
                        threshold=self.quality_thresholds['consistency']
                    ))

        return metrics, issues

    def _assess_validity(self, data: pd.DataFrame) -> Tuple[List[QualityMetric], List[ValidationResult]]:
        """Assess data validity."""
        metrics = []
        issues = []

        # Data type validity
        for column in data.columns:
            series = data[column].dropna()
            if len(series) == 0:
                continue

            valid_count = 0
            total_count = len(series)

            # Check if numeric columns contain valid numbers
            if pd.api.types.is_numeric_dtype(data[column]):
                valid_count = (~pd.isna(pd.to_numeric(series, errors='coerce'))).sum()

            # Check if string columns are reasonable
            elif pd.api.types.is_string_dtype(data[column]) or pd.api.types.is_object_dtype(data[column]):
                str_series = series.astype(str)
                # Valid strings should not be too long and should not contain control characters
                valid_strings = (
                    (str_series.str.len() <= 1000) &  # Reasonable length
                    (~str_series.str.contains('[\x00-\x1f\x7f-\x9f]', regex=True, na=False))  # No control chars
                )
                valid_count = valid_strings.sum()

            # Check datetime validity
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                valid_count = total_count  # pandas datetime is already validated

            validity_score = valid_count / total_count if total_count > 0 else 0

            metrics.append(QualityMetric(
                dimension=QualityDimension.VALIDITY,
                metric_name=f"validity_{column}",
                score=validity_score,
                details={
                    'column': column,
                    'valid_values': valid_count,
                    'total_values': total_count,
                    'invalid_values': total_count - valid_count
                },
                passed_threshold=validity_score >= self.quality_thresholds['validity'],
                threshold=self.quality_thresholds['validity']
            ))

            if validity_score < self.quality_thresholds['validity']:
                issues.append(ValidationResult(
                    rule_name="data_validity",
                    category=ValidationCategory.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Column '{column}' has validity issues: {total_count - valid_count} invalid values",
                    affected_rows=total_count - valid_count,
                    affected_columns=[column],
                    details={'validity_score': validity_score},
                    timestamp=datetime.utcnow()
                ))

        return metrics, issues

    def _assess_timeliness(self, data: pd.DataFrame) -> Tuple[List[QualityMetric], List[ValidationResult]]:
        """Assess data timeliness."""
        metrics = []
        issues = []

        # Find date columns
        date_columns = []
        for column in data.columns:
            if 'date' in column.lower() or pd.api.types.is_datetime64_any_dtype(data[column]):
                date_columns.append(column)

        for date_col in date_columns:
            try:
                date_series = pd.to_datetime(data[date_col]).dropna()
                if len(date_series) == 0:
                    continue

                # Calculate data age
                latest_date = date_series.max()
                current_date = pd.Timestamp.now()
                data_age_days = (current_date - latest_date).days

                # Score based on age (fresher data gets higher score)
                if data_age_days <= 1:
                    timeliness_score = 1.0
                elif data_age_days <= 7:
                    timeliness_score = 0.9
                elif data_age_days <= 30:
                    timeliness_score = 0.7
                elif data_age_days <= 90:
                    timeliness_score = 0.5
                else:
                    timeliness_score = 0.2

                metrics.append(QualityMetric(
                    dimension=QualityDimension.TIMELINESS,
                    metric_name=f"timeliness_{date_col}",
                    score=timeliness_score,
                    details={
                        'column': date_col,
                        'latest_date': latest_date.isoformat(),
                        'data_age_days': data_age_days,
                        'earliest_date': date_series.min().isoformat()
                    },
                    passed_threshold=timeliness_score >= self.quality_thresholds['timeliness'],
                    threshold=self.quality_thresholds['timeliness']
                ))

                if timeliness_score < self.quality_thresholds['timeliness']:
                    issues.append(ValidationResult(
                        rule_name="data_timeliness",
                        category=ValidationCategory.BUSINESS_LOGIC,
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Data in column '{date_col}' may be stale: {data_age_days} days old",
                        affected_rows=0,
                        affected_columns=[date_col],
                        details={'data_age_days': data_age_days, 'timeliness_score': timeliness_score},
                        timestamp=datetime.utcnow()
                    ))

            except Exception as e:
                logger.warning(f"Error assessing timeliness for column {date_col}: {e}")

        return metrics, issues

    def _assess_uniqueness(self, data: pd.DataFrame) -> Tuple[List[QualityMetric], List[ValidationResult]]:
        """Assess data uniqueness."""
        metrics = []
        issues = []

        # Check for key columns that should be unique
        potential_key_columns = []
        for column in data.columns:
            if any(keyword in column.lower() for keyword in ['id', 'key', 'symbol']):
                potential_key_columns.append(column)

        # Also check combinations that might form composite keys
        if 'date' in data.columns and 'symbol' in data.columns:
            # Check date-symbol combination uniqueness
            combined_series = data[['date', 'symbol']].apply(lambda x: f"{x['date']}_{x['symbol']}", axis=1)
            duplicates = combined_series.duplicated().sum()
            uniqueness_score = 1 - (duplicates / len(combined_series))

            metrics.append(QualityMetric(
                dimension=QualityDimension.UNIQUENESS,
                metric_name="date_symbol_uniqueness",
                score=uniqueness_score,
                details={
                    'columns': ['date', 'symbol'],
                    'duplicates': duplicates,
                    'total_rows': len(combined_series),
                    'duplicate_rate': duplicates / len(combined_series)
                },
                passed_threshold=uniqueness_score >= self.quality_thresholds['uniqueness'],
                threshold=self.quality_thresholds['uniqueness']
            ))

            if uniqueness_score < self.quality_thresholds['uniqueness']:
                issues.append(ValidationResult(
                    rule_name="composite_key_uniqueness",
                    category=ValidationCategory.UNIQUENESS,
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Date-Symbol combination has {duplicates} duplicates",
                    affected_rows=duplicates,
                    affected_columns=['date', 'symbol'],
                    details={'uniqueness_score': uniqueness_score},
                    timestamp=datetime.utcnow()
                ))

        # Check individual key columns
        for column in potential_key_columns:
            if column in data.columns:
                series = data[column].dropna()
                if len(series) == 0:
                    continue

                duplicates = series.duplicated().sum()
                uniqueness_score = 1 - (duplicates / len(series))

                metrics.append(QualityMetric(
                    dimension=QualityDimension.UNIQUENESS,
                    metric_name=f"uniqueness_{column}",
                    score=uniqueness_score,
                    details={
                        'column': column,
                        'duplicates': duplicates,
                        'total_values': len(series),
                        'unique_values': series.nunique()
                    },
                    passed_threshold=uniqueness_score >= self.quality_thresholds['uniqueness'],
                    threshold=self.quality_thresholds['uniqueness']
                ))

        return metrics, issues

    def _check_case_consistency(self, series: pd.Series) -> Dict[str, int]:
        """Check case consistency in string series."""
        if len(series) == 0:
            return {'consistent': 0, 'inconsistent': 0}

        # Sample a subset for performance
        sample_size = min(1000, len(series))
        sample = series.sample(sample_size) if len(series) > sample_size else series

        # Check if most values follow the same case pattern
        lower_count = (sample == sample.str.lower()).sum()
        upper_count = (sample == sample.str.upper()).sum()
        title_count = (sample == sample.str.title()).sum()

        total_consistent = max(lower_count, upper_count, title_count)
        inconsistent = sample_size - total_consistent

        return {
            'consistent': total_consistent,
            'inconsistent': inconsistent
        }

    def _calculate_dimension_scores(self, metrics: List[QualityMetric]) -> Dict[QualityDimension, float]:
        """Calculate average scores by dimension."""
        dimension_scores = {}

        for dimension in QualityDimension:
            dimension_metrics = [m for m in metrics if m.dimension == dimension]
            if dimension_metrics:
                avg_score = sum(m.score for m in dimension_metrics) / len(dimension_metrics)
                dimension_scores[dimension] = avg_score
            else:
                dimension_scores[dimension] = 1.0  # No issues found

        return dimension_scores

    def _generate_recommendations(self, metrics: List[QualityMetric], issues: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on quality assessment."""
        recommendations = []

        # Group issues by severity
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        warning_issues = [i for i in issues if i.severity == ValidationSeverity.WARNING]

        if critical_issues:
            recommendations.append(f"URGENT: Address {len(critical_issues)} critical data quality issues immediately")

        if error_issues:
            recommendations.append(f"Fix {len(error_issues)} data errors to improve reliability")

        # Specific recommendations based on common issues
        completeness_issues = [m for m in metrics if m.dimension == QualityDimension.COMPLETENESS and not m.passed_threshold]
        if completeness_issues:
            recommendations.append("Improve data collection processes to reduce missing values")

        accuracy_issues = [m for m in metrics if m.dimension == QualityDimension.ACCURACY and not m.passed_threshold]
        if accuracy_issues:
            recommendations.append("Implement data validation rules to catch outliers and inconsistencies")

        consistency_issues = [m for m in metrics if m.dimension == QualityDimension.CONSISTENCY and not m.passed_threshold]
        if consistency_issues:
            recommendations.append("Standardize data formats and implement cross-field validation")

        validity_issues = [m for m in metrics if m.dimension == QualityDimension.VALIDITY and not m.passed_threshold]
        if validity_issues:
            recommendations.append("Add input validation and data type constraints")

        timeliness_issues = [m for m in metrics if m.dimension == QualityDimension.TIMELINESS and not m.passed_threshold]
        if timeliness_issues:
            recommendations.append("Implement more frequent data updates or refresh schedules")

        uniqueness_issues = [m for m in metrics if m.dimension == QualityDimension.UNIQUENESS and not m.passed_threshold]
        if uniqueness_issues:
            recommendations.append("Add unique constraints and duplicate detection mechanisms")

        if not recommendations:
            recommendations.append("Data quality is excellent! Continue monitoring to maintain standards")

        return recommendations