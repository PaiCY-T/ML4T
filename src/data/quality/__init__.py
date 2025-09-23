"""
Data quality validation and monitoring module.

This module provides comprehensive data quality checks, anomaly detection,
monitoring, alerting, and reporting for the Taiwan market point-in-time data system.

Components:
- validation_engine: Core validation framework with plugin architecture
- taiwan_validators: Taiwan market-specific validation rules
- rules_engine: Configurable validation rules system
- monitor: Real-time quality monitoring with <10ms latency
- alerting: Multi-channel alerting system (email, Slack, webhook)
- metrics: Quality scoring algorithms and SLA tracking
- dashboard: Web-based quality dashboard with visualizations
- reporting: Automated quality reports and SLA tracking
"""

# Core validation components (Stream A)
from .validation_engine import (
    ValidationEngine, ValidationRegistry, ValidationPlugin,
    ValidationContext, ValidationOutput, ValidationResult, ValidationPriority
)
from .taiwan_validators import (
    TaiwanPriceLimitValidator, TaiwanVolumeValidator, TaiwanTradingHoursValidator,
    TaiwanSettlementValidator, TaiwanFundamentalLagValidator, create_taiwan_validators
)
from .rules_engine import (
    RulesEngine, ValidationRule, RuleCondition, RuleOperator, RuleAction,
    RulesBasedValidator, create_taiwan_market_rules, create_rules_engine_with_taiwan_rules
)
from .validators import (
    QualityIssue, SeverityLevel, QualityCheckType
)

# Monitoring and alerting components (Stream B)
from .monitor import (
    QualityMonitor, QualityMetrics, MonitoringThresholds, AlertLevel,
    RealtimeQualityStream, create_taiwan_market_monitor, setup_monitoring_pipeline
)
from .alerting import (
    AlertManager, AlertMessage, AlertRule, AlertChannel, AlertStatus,
    ChannelConfig, create_taiwan_alert_manager, setup_monitoring_alerts
)
from .metrics import (
    QualityScoreCalculator, TrendAnalyzer, SLATracker, QualityMetricsAggregator,
    QualityTrend, SLAMetric, SLAResult, SLAStatus, MetricType,
    create_taiwan_metrics_system
)
from .dashboard import (
    QualityDashboard, DashboardData, create_taiwan_dashboard,
    setup_complete_monitoring_system
)
from .reporting import (
    ReportGenerator, ReportScheduler, setup_automated_reporting, generate_manual_report
)

__all__ = [
    # Core validation
    'ValidationEngine', 'ValidationRegistry', 'ValidationPlugin',
    'ValidationContext', 'ValidationOutput', 'ValidationResult', 'ValidationPriority',
    'TaiwanPriceLimitValidator', 'TaiwanVolumeValidator', 'TaiwanTradingHoursValidator',
    'TaiwanSettlementValidator', 'TaiwanFundamentalLagValidator', 'create_taiwan_validators',
    'RulesEngine', 'ValidationRule', 'RuleCondition', 'RuleOperator', 'RuleAction',
    'RulesBasedValidator', 'create_taiwan_market_rules', 'create_rules_engine_with_taiwan_rules',
    'QualityIssue', 'SeverityLevel', 'QualityCheckType',
    
    # Monitoring and alerting
    'QualityMonitor', 'QualityMetrics', 'MonitoringThresholds', 'AlertLevel',
    'RealtimeQualityStream', 'create_taiwan_market_monitor', 'setup_monitoring_pipeline',
    'AlertManager', 'AlertMessage', 'AlertRule', 'AlertChannel', 'AlertStatus',
    'ChannelConfig', 'create_taiwan_alert_manager', 'setup_monitoring_alerts',
    'QualityScoreCalculator', 'TrendAnalyzer', 'SLATracker', 'QualityMetricsAggregator',
    'QualityTrend', 'SLAMetric', 'SLAResult', 'SLAStatus', 'MetricType',
    'create_taiwan_metrics_system',
    'QualityDashboard', 'DashboardData', 'create_taiwan_dashboard',
    'setup_complete_monitoring_system',
    'ReportGenerator', 'ReportScheduler', 'setup_automated_reporting', 'generate_manual_report'
]