#!/usr/bin/env python3
"""
Data Validation Framework Demo - Issue #57
Demonstration of comprehensive data quality assurance for FinLab datasets.

This script demonstrates:
- Schema validation for FinLab data structure
- Point-in-time data integrity checks
- Data quality metrics and scoring
- Statistical outlier detection
- Validation reporting and alerting
- Data lineage tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup paths
project_root = Path(__file__).parent
src_path = project_root / "src"

import sys
sys.path.insert(0, str(src_path))

from validation.data_validation_framework import (
    DataValidationFramework,
    ValidationSeverity,
    ValidationConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_finlab_data(with_issues=False):
    """Create sample FinLab dataset for demonstration"""
    np.random.seed(42)

    # Generate 90 days of data
    dates = pd.date_range('2023-01-01', periods=90, freq='D')
    n_records = len(dates)

    # Base price data
    base_price = 100.0
    price_series = []
    for i in range(n_records):
        if i == 0:
            price = base_price
        else:
            # Random walk with slight upward trend
            change = np.random.normal(0.001, 0.02)  # 0.1% average daily return, 2% volatility
            price = price_series[-1] * (1 + change)
        price_series.append(max(price, 0.1))  # Prevent negative prices

    data = {
        'date': dates,
        'adj_close': price_series,
        'adj_high': [p * np.random.uniform(1.00, 1.05) for p in price_series],
        'adj_low': [p * np.random.uniform(0.95, 1.00) for p in price_series],
        'adj_open': [price_series[max(0, i-1)] * np.random.uniform(0.98, 1.02) for i in range(n_records)],
        'buy': np.random.randint(1000, 8000, n_records),
        'sell': np.random.randint(800, 7500, n_records),
        '保留盈餘': np.random.uniform(800000, 2500000, n_records),
        '不動產廠房及設備': np.random.uniform(2000000, 8000000, n_records),
        '使用權資產': np.random.uniform(100000, 500000, n_records),
        '一年內到期長期負債': np.random.uniform(50000, 800000, n_records),
        '停業單位損益': np.random.uniform(-100000, 50000, n_records),
        '償還公司債': np.random.uniform(0, 200000, n_records)
    }

    df = pd.DataFrame(data)

    if with_issues:
        # Introduce various data quality issues for demonstration

        # 1. Missing values
        df.loc[10:12, '保留盈餘'] = None
        df.loc[25, 'adj_close'] = None

        # 2. Outliers
        df.loc[30, 'adj_close'] = df.loc[30, 'adj_close'] * 5  # Price spike
        df.loc[45, 'buy'] = 50000  # Volume spike
        df.loc[60, '保留盈餘'] = -1000000  # Anomalous negative retained earnings

        # 3. Data type issues (convert some numeric to string)
        df.loc[20, 'buy'] = 'invalid_data'
        df.loc[35, 'adj_high'] = 'N/A'

        # 4. Temporal issues
        df.loc[15, 'date'] = df.loc[10, 'date']  # Duplicate date

        # 5. Logical inconsistencies
        df.loc[40, 'adj_low'] = df.loc[40, 'adj_high'] * 1.1  # Low > High

        # 6. Future data leak
        df.loc[80:82, 'date'] = pd.date_range('2025-01-01', periods=3, freq='D')

        logger.info("Created sample data with intentional quality issues for demonstration")
    else:
        logger.info("Created clean sample FinLab dataset")

    return df


def demonstrate_schema_validation(framework, df):
    """Demonstrate schema validation capabilities"""
    print("\n" + "="*60)
    print("SCHEMA VALIDATION DEMONSTRATION")
    print("="*60)

    # Get just the schema validation results
    schema_results = framework.schema_validator.validate_schema(df)

    print(f"\nTotal schema validation checks: {len(schema_results)}")

    # Group results by field
    results_by_field = {}
    for result in schema_results:
        if result.field_name not in results_by_field:
            results_by_field[result.field_name] = []
        results_by_field[result.field_name].append(result)

    for field_name, field_results in results_by_field.items():
        print(f"\n📊 Field: {field_name}")
        for result in field_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {status} {result.check_name}: {result.message}")
            if result.details:
                for key, value in result.details.items():
                    print(f"    {key}: {value}")


def demonstrate_temporal_validation(framework, df):
    """Demonstrate temporal validation capabilities"""
    print("\n" + "="*60)
    print("TEMPORAL VALIDATION DEMONSTRATION")
    print("="*60)

    # Point-in-time validation
    as_of_date = datetime.now()
    pit_results = framework.pit_validator.validate_point_in_time_integrity(
        df, as_of_date, 'date'
    )

    print(f"\nPoint-in-time validation (as of {as_of_date.strftime('%Y-%m-%d')}):")
    for result in pit_results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status} {result.check_name}: {result.message}")
        if result.details:
            for key, value in result.details.items():
                print(f"    {key}: {value}")

    # Temporal consistency validation
    temporal_results = framework.pit_validator.validate_temporal_consistency(
        df, 'date', ['adj_close', 'buy', 'sell']
    )

    print(f"\nTemporal consistency validation:")
    for result in temporal_results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status} {result.check_name}: {result.message}")
        if result.details:
            for key, value in result.details.items():
                print(f"    {key}: {value}")


def demonstrate_outlier_detection(framework, df):
    """Demonstrate statistical outlier detection"""
    print("\n" + "="*60)
    print("STATISTICAL OUTLIER DETECTION DEMONSTRATION")
    print("="*60)

    # Detect outliers in numeric columns
    outlier_results = framework.outlier_detector.detect_outliers(df)

    if outlier_results:
        print(f"\nOutlier detection results:")
        for result in outlier_results:
            status = "✅ PASS" if result.passed else "⚠️ OUTLIERS"
            print(f"  {status} {result.field_name}: {result.message}")
            if result.details:
                print(f"    Z-score outliers: {result.details.get('z_score_outliers', 0)}")
                print(f"    IQR outliers: {result.details.get('iqr_outliers', 0)}")
                print(f"    Outlier percentage: {result.details.get('outlier_percentage', 0):.2f}%")

    # Detect anomalous patterns
    pattern_results = framework.outlier_detector.detect_anomalous_patterns(
        df, 'date', ['adj_close', 'buy']
    )

    if pattern_results:
        print(f"\nAnomalous pattern detection:")
        for result in pattern_results:
            status = "✅ PASS" if result.passed else "⚠️ ANOMALY"
            print(f"  {status} {result.field_name}: {result.message}")
            if result.details:
                print(f"    Spike count: {result.details.get('spike_count', 0)}")
                print(f"    Spike percentage: {result.details.get('spike_percentage', 0):.2f}%")


def demonstrate_quality_scoring(quality_metrics):
    """Demonstrate data quality scoring"""
    print("\n" + "="*60)
    print("DATA QUALITY SCORING DEMONSTRATION")
    print("="*60)

    print(f"\n📊 DATA QUALITY METRICS:")
    print(f"  Completeness Score:  {quality_metrics.completeness_score:.1f}%")
    print(f"  Accuracy Score:      {quality_metrics.accuracy_score:.1f}%")
    print(f"  Consistency Score:   {quality_metrics.consistency_score:.1f}%")
    print(f"  Validity Score:      {quality_metrics.validity_score:.1f}%")
    print(f"  Uniqueness Score:    {quality_metrics.uniqueness_score:.1f}%")
    print(f"  Timeliness Score:    {quality_metrics.timeliness_score:.1f}%")
    print(f"  ─────────────────────────────────")
    print(f"  Overall Score:       {quality_metrics.overall_score:.1f}%")

    # Provide quality interpretation
    if quality_metrics.overall_score >= 90:
        quality_level = "EXCELLENT"
        emoji = "🟢"
    elif quality_metrics.overall_score >= 80:
        quality_level = "GOOD"
        emoji = "🟡"
    elif quality_metrics.overall_score >= 70:
        quality_level = "FAIR"
        emoji = "🟠"
    else:
        quality_level = "POOR"
        emoji = "🔴"

    print(f"\n{emoji} Quality Level: {quality_level}")


def demonstrate_lineage_tracking(framework, df):
    """Demonstrate data lineage tracking"""
    print("\n" + "="*60)
    print("DATA LINEAGE TRACKING DEMONSTRATION")
    print("="*60)

    # Track the dataset
    data_hash = framework.lineage_tracker.track_dataset(
        df,
        "demo_finlab_dataset",
        source_system="finlab_api",
        extraction_method="daily_batch",
        transformation_steps=["cleaning", "validation", "enrichment"],
        quality_score=85.5
    )

    print(f"\n📋 Dataset tracked with hash: {data_hash}")

    # Get dataset history
    history = framework.lineage_tracker.get_dataset_history("demo_finlab_dataset")

    print(f"\n📈 Dataset History:")
    for i, record in enumerate(history, 1):
        print(f"  Version {i}:")
        print(f"    Timestamp: {record.extraction_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    Rows: {record.row_count}, Columns: {record.column_count}")
        print(f"    Quality Score: {record.quality_score}")
        print(f"    Transformations: {', '.join(record.transformation_steps)}")

    # Validate data integrity
    validation_result = framework.lineage_tracker.validate_data_lineage(df, data_hash)
    status = "✅ VALID" if validation_result.passed else "❌ INVALID"
    print(f"\n🔍 Data Integrity Check: {status}")
    print(f"  {validation_result.message}")


def demonstrate_reporting_and_alerts(framework, validation_result):
    """Demonstrate reporting and alerting system"""
    print("\n" + "="*60)
    print("VALIDATION REPORTING & ALERTING DEMONSTRATION")
    print("="*60)

    # Display summary
    summary = validation_result["summary"]
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"  Total Checks:    {summary['total_checks']}")
    print(f"  Passed Checks:   {summary['passed_checks']}")
    print(f"  Failed Checks:   {summary['failed_checks']}")
    print(f"  Pass Rate:       {summary['passed_checks']/summary['total_checks']*100:.1f}%")

    # Display alerts
    alerts = validation_result["alerts"]
    if alerts:
        print(f"\n🚨 ALERTS ({len(alerts)}):")
        for i, alert in enumerate(alerts, 1):
            print(f"  {i}. {alert}")
    else:
        print(f"\n✅ No alerts triggered")

    # Categorize validation results by severity
    results_by_severity = {}
    for result in validation_result["validation_results"]:
        severity = result.severity.value
        if severity not in results_by_severity:
            results_by_severity[severity] = []
        results_by_severity[severity].append(result)

    print(f"\n📋 RESULTS BY SEVERITY:")
    severity_order = ['critical', 'high', 'medium', 'low', 'info']
    for severity in severity_order:
        if severity in results_by_severity:
            count = len(results_by_severity[severity])
            failed_count = len([r for r in results_by_severity[severity] if not r.passed])
            print(f"  {severity.upper():8}: {count:3} checks ({failed_count} failed)")


def main():
    """Main demonstration function"""
    print("="*80)
    print("DATA VALIDATION FRAMEWORK DEMONSTRATION - ISSUE #57")
    print("Comprehensive Data Quality Assurance for FinLab Datasets")
    print("="*80)

    # Initialize validation framework
    print("\n🚀 Initializing Data Validation Framework...")
    schema_config_path = project_root / "example" / "finlab_database_cleaned.csv"

    framework = DataValidationFramework(
        schema_config_path=str(schema_config_path) if schema_config_path.exists() else None,
        lineage_db_path="demo_lineage.json",
        report_dir="demo_reports"
    )

    print("✅ Framework initialized successfully!")

    # Create sample datasets
    print("\n📋 Creating sample datasets...")
    clean_data = create_sample_finlab_data(with_issues=False)
    problematic_data = create_sample_finlab_data(with_issues=True)

    print(f"Clean dataset: {len(clean_data)} records, {len(clean_data.columns)} columns")
    print(f"Problematic dataset: {len(problematic_data)} records, {len(problematic_data.columns)} columns")

    # Demonstrate with clean data first
    print("\n" + "="*80)
    print("DEMONSTRATION 1: CLEAN DATA VALIDATION")
    print("="*80)

    clean_result = framework.validate_dataset(
        clean_data,
        "clean_finlab_dataset",
        as_of_date=datetime.now(),
        date_column='date',
        generate_report=True
    )

    print(f"\n✅ Clean data validation completed!")
    demonstrate_quality_scoring(clean_result["quality_metrics"])
    demonstrate_lineage_tracking(framework, clean_data)
    demonstrate_reporting_and_alerts(framework, clean_result)

    # Demonstrate with problematic data
    print("\n" + "="*80)
    print("DEMONSTRATION 2: PROBLEMATIC DATA VALIDATION")
    print("="*80)

    problematic_result = framework.validate_dataset(
        problematic_data,
        "problematic_finlab_dataset",
        as_of_date=datetime.now(),
        date_column='date',
        generate_report=True
    )

    print(f"\n⚠️ Problematic data validation completed!")
    demonstrate_schema_validation(framework, problematic_data)
    demonstrate_temporal_validation(framework, problematic_data)
    demonstrate_outlier_detection(framework, problematic_data)
    demonstrate_quality_scoring(problematic_result["quality_metrics"])
    demonstrate_reporting_and_alerts(framework, problematic_result)

    # Multiple dataset validation demonstration
    print("\n" + "="*80)
    print("DEMONSTRATION 3: MULTIPLE DATASET VALIDATION")
    print("="*80)

    datasets = {
        "dataset_a": clean_data,
        "dataset_b": problematic_data,
        "dataset_c": create_sample_finlab_data(with_issues=False)
    }

    multi_result = framework.validate_multiple_datasets(datasets)

    print(f"\n📊 MULTI-DATASET VALIDATION SUMMARY:")
    summary = multi_result["overall_summary"]
    print(f"  Total datasets:          {summary['total_datasets']}")
    print(f"  Datasets validated:      {summary['datasets_validated']}")
    print(f"  Average quality score:   {summary['average_quality_score']:.1f}%")
    print(f"  Total alerts:            {summary['total_alerts']}")

    print(f"\n📋 INDIVIDUAL DATASET RESULTS:")
    for dataset_name, result in multi_result["dataset_results"].items():
        quality_score = result["quality_metrics"].overall_score
        alert_count = len(result["alerts"])
        emoji = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 70 else "🔴"
        print(f"  {emoji} {dataset_name:20}: {quality_score:5.1f}% quality, {alert_count} alerts")

    # Configuration demonstration
    print("\n" + "="*80)
    print("DEMONSTRATION 4: CONFIGURATION MANAGEMENT")
    print("="*80)

    config = ValidationConfig("demo_validation_config.json")

    print(f"\n⚙️ CONFIGURATION SETTINGS:")
    print(f"  Schema validation enabled:   {config.get('schema_validation.enabled')}")
    print(f"  Outlier Z-threshold:         {config.get('outlier_detection.z_threshold')}")
    print(f"  Quality threshold:           {config.get('alerts.quality_threshold')}")
    print(f"  Auto-generate reports:       {config.get('reporting.auto_generate')}")

    # Modify configuration
    config.set("outlier_detection.z_threshold", 2.5)
    config.set("alerts.quality_threshold", 75)

    print(f"\n✏️ Configuration updated:")
    print(f"  New outlier Z-threshold:     {config.get('outlier_detection.z_threshold')}")
    print(f"  New quality threshold:       {config.get('alerts.quality_threshold')}")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)

    print(f"\n📁 Generated files:")
    print(f"  Lineage database:     demo_lineage.json")
    print(f"  Configuration file:   demo_validation_config.json")
    print(f"  Validation reports:   demo_reports/")

    print(f"\n📊 Key takeaways:")
    print(f"  • Clean data achieved {clean_result['quality_metrics'].overall_score:.1f}% quality score")
    print(f"  • Problematic data achieved {problematic_result['quality_metrics'].overall_score:.1f}% quality score")
    print(f"  • Framework detected {problematic_result['summary']['failed_checks']} data quality issues")
    print(f"  • {len(problematic_result['alerts'])} alerts were triggered for the problematic dataset")

    print(f"\n🔧 The Data Validation Framework is ready for ML4T-Alpha integration!")


if __name__ == "__main__":
    main()