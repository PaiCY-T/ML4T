"""
Reporting Package for ML4T Walk-Forward Validation.

This package provides comprehensive reporting capabilities for walk-forward validation
results, including automated report generation with statistical significance testing
and visualization for Taiwan market quantitative trading strategies.

Key Modules:
- validation_reports: Automated validation report generation with multiple output formats

Key Features:
- Automated report generation (HTML, JSON, Markdown)
- Statistical significance testing
- Performance target achievement analysis
- Risk analysis and regime detection
- Benchmark comparisons with Taiwan indices
- Executive summaries and detailed technical appendices
"""

from .validation_reports import (
    ValidationReportGenerator,
    ReportConfig,
    ReportData,
    ReportFormat,
    ReportSection,
    create_validation_report_config,
    generate_taiwan_validation_report
)

__all__ = [
    'ValidationReportGenerator',
    'ReportConfig',
    'ReportData',
    'ReportFormat',
    'ReportSection',
    'create_validation_report_config',
    'generate_taiwan_validation_report'
]

# Package version and metadata
__version__ = "1.0.0"
__author__ = "ML4T Team"
__description__ = "Automated reporting for Taiwan market validation results"