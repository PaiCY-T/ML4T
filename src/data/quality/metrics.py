"""
Quality Metrics and Scoring Algorithms.

This module provides comprehensive quality metrics calculation, scoring algorithms,
trend analysis, and SLA tracking for the Taiwan market data quality system.
Includes advanced statistical analysis and anomaly detection.
"""

import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import threading

from .monitor import QualityMetrics, AlertLevel
from .validators import QualityIssue, SeverityLevel, QualityCheckType
from ..core.temporal import DataType

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of quality metrics."""
    SCORE = "score"                    # Overall quality score
    COMPLETENESS = "completeness"      # Data completeness percentage
    ACCURACY = "accuracy"              # Data accuracy rate
    TIMELINESS = "timeliness"         # Data freshness metrics
    CONSISTENCY = "consistency"        # Internal consistency metrics
    VALIDITY = "validity"             # Format and range validity
    LATENCY = "latency"               # Processing latency
    THROUGHPUT = "throughput"         # Processing throughput
    AVAILABILITY = "availability"      # System availability
    COVERAGE = "coverage"             # Symbol/data type coverage


class SLAStatus(Enum):
    """SLA compliance status."""
    MEETING = "meeting"
    AT_RISK = "at_risk" 
    BREACHED = "breached"
    UNKNOWN = "unknown"


@dataclass
class QualityTrend:
    """Quality trend analysis."""
    metric_type: MetricType
    symbol: Optional[str]
    data_type: Optional[DataType]
    period_start: datetime
    period_end: datetime
    trend_direction: str  # "improving", "stable", "degrading"
    trend_strength: float  # -1.0 to 1.0
    current_value: float
    average_value: float
    min_value: float
    max_value: float
    standard_deviation: float
    data_points: int
    confidence_level: float


@dataclass
class SLAMetric:
    """SLA metric definition and tracking."""
    name: str
    description: str
    target_value: float
    threshold_warning: float
    threshold_critical: float
    measurement_window_hours: int = 24
    calculation_method: str = "average"  # "average", "min", "max", "p95", "p99"
    taiwan_market_specific: bool = False
    enabled: bool = True


@dataclass
class SLAResult:
    """SLA compliance result."""
    metric_name: str
    measurement_period: Tuple[datetime, datetime]
    target_value: float
    actual_value: float
    status: SLAStatus
    compliance_percentage: float
    breach_duration_minutes: int = 0
    last_breach_time: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)


class QualityScoreCalculator:
    """Advanced quality score calculation engine."""
    
    def __init__(self):
        self.weights = {
            SeverityLevel.CRITICAL: -50.0,
            SeverityLevel.ERROR: -25.0,
            SeverityLevel.WARNING: -10.0,
            SeverityLevel.INFO: -2.0
        }
        
        self.check_type_weights = {
            QualityCheckType.VALIDITY: 1.5,      # High importance
            QualityCheckType.ACCURACY: 1.4,
            QualityCheckType.CONSISTENCY: 1.3,
            QualityCheckType.COMPLETENESS: 1.2,
            QualityCheckType.TIMELINESS: 1.1,
            QualityCheckType.TEMPORAL: 1.0,
            QualityCheckType.BUSINESS_RULES: 1.3,  # Taiwan market rules
            QualityCheckType.ANOMALY: 0.8
        }
        
        # Taiwan market specific weights
        self.taiwan_multipliers = {
            'price_limit_violation': 2.0,
            'settlement_violation': 1.8,
            'trading_hours_violation': 1.5,
            'volume_anomaly': 1.2
        }
    
    def calculate_score(self, 
                       issues: List[QualityIssue],
                       base_score: float = 100.0,
                       taiwan_market: bool = False) -> float:
        """Calculate comprehensive quality score."""
        score = base_score
        
        if not issues:
            return score
        
        # Group issues by severity and type
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for issue in issues:
            severity_counts[issue.severity] += 1
            type_counts[issue.check_type] += 1
            
            # Base penalty
            penalty = self.weights[issue.severity]
            
            # Apply check type weight
            penalty *= self.check_type_weights.get(issue.check_type, 1.0)
            
            # Apply Taiwan market multipliers
            if taiwan_market:
                for taiwan_key, multiplier in self.taiwan_multipliers.items():
                    if taiwan_key in (issue.description or "").lower():
                        penalty *= multiplier
                        break
            
            score += penalty  # penalty is negative
        
        # Additional penalties for multiple issues
        total_issues = len(issues)
        if total_issues > 10:
            score -= (total_issues - 10) * 2  # Escalating penalty
        
        # Critical issue penalties
        critical_count = severity_counts[SeverityLevel.CRITICAL]
        if critical_count > 0:
            score -= critical_count * 20  # Extra penalty for critical issues
        
        return max(0.0, min(100.0, score))
    
    def calculate_weighted_score(self,
                               validation_results: List,
                               execution_time_ms: float,
                               taiwan_market: bool = False) -> Tuple[float, Dict[str, Any]]:
        """Calculate weighted quality score with detailed breakdown."""
        # Extract all issues
        all_issues = []
        for result in validation_results:
            if hasattr(result, 'issues'):
                all_issues.extend(result.issues)
        
        # Base quality score
        quality_score = self.calculate_score(all_issues, taiwan_market=taiwan_market)
        
        # Performance penalty
        latency_penalty = 0.0
        if execution_time_ms > 10.0:  # Target <10ms
            latency_penalty = min(20.0, (execution_time_ms - 10.0) * 2)
        
        # Final score
        final_score = max(0.0, quality_score - latency_penalty)
        
        # Detailed breakdown
        breakdown = {
            'base_quality_score': quality_score,
            'latency_penalty': latency_penalty,
            'final_score': final_score,
            'issue_count': len(all_issues),
            'execution_time_ms': execution_time_ms,
            'taiwan_market': taiwan_market,
            'severity_breakdown': {
                'critical': len([i for i in all_issues if i.severity == SeverityLevel.CRITICAL]),
                'error': len([i for i in all_issues if i.severity == SeverityLevel.ERROR]),
                'warning': len([i for i in all_issues if i.severity == SeverityLevel.WARNING]),
                'info': len([i for i in all_issues if i.severity == SeverityLevel.INFO])
            }
        }
        
        return final_score, breakdown


class TrendAnalyzer:
    """Statistical trend analysis for quality metrics."""
    
    def __init__(self, min_data_points: int = 10):
        self.min_data_points = min_data_points
    
    def analyze_trend(self, 
                     data_points: List[Tuple[datetime, float]],
                     metric_type: MetricType,
                     symbol: Optional[str] = None,
                     data_type: Optional[DataType] = None) -> Optional[QualityTrend]:
        """Analyze trend in quality metrics."""
        if len(data_points) < self.min_data_points:
            return None
        
        # Sort by timestamp
        data_points = sorted(data_points, key=lambda x: x[0])
        
        timestamps = [dp[0] for dp in data_points]
        values = [dp[1] for dp in data_points]
        
        # Convert timestamps to numeric for regression
        start_time = timestamps[0]
        numeric_times = [(ts - start_time).total_seconds() for ts in timestamps]
        
        # Calculate basic statistics
        current_value = values[-1]
        average_value = statistics.mean(values)
        min_value = min(values)
        max_value = max(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(numeric_times, values)
        
        # Determine trend direction and strength
        trend_direction = "stable"
        trend_strength = r_value  # Correlation coefficient
        
        if abs(slope) > 0.001 and p_value < 0.05:  # Significant trend
            if slope > 0:
                trend_direction = "improving" if metric_type == MetricType.SCORE else "increasing"
            else:
                trend_direction = "degrading" if metric_type == MetricType.SCORE else "decreasing"
        
        # Confidence level based on R-squared and p-value
        confidence_level = (r_value ** 2) * (1 - p_value)
        
        return QualityTrend(
            metric_type=metric_type,
            symbol=symbol,
            data_type=data_type,
            period_start=timestamps[0],
            period_end=timestamps[-1],
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            current_value=current_value,
            average_value=average_value,
            min_value=min_value,
            max_value=max_value,
            standard_deviation=std_dev,
            data_points=len(data_points),
            confidence_level=confidence_level
        )
    
    def detect_anomalies(self, 
                        values: List[float],
                        method: str = "iqr") -> List[int]:
        """Detect anomalies in metric values."""
        if len(values) < 10:
            return []
        
        anomaly_indices = []
        
        if method == "iqr":
            # Interquartile Range method
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    anomaly_indices.append(i)
        
        elif method == "zscore":
            # Z-score method
            mean_val = np.mean(values)
            std_val = np.std(values)
            threshold = 3.0  # 3 standard deviations
            
            for i, value in enumerate(values):
                zscore = abs((value - mean_val) / std_val) if std_val > 0 else 0
                if zscore > threshold:
                    anomaly_indices.append(i)
        
        return anomaly_indices


class SLATracker:
    """SLA (Service Level Agreement) tracking and monitoring."""
    
    def __init__(self):
        self.sla_definitions: Dict[str, SLAMetric] = {}
        self.measurement_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.breach_history: Dict[str, List[Tuple[datetime, datetime]]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Initialize Taiwan market SLAs
        self._initialize_taiwan_slas()
    
    def _initialize_taiwan_slas(self) -> None:
        """Initialize Taiwan market specific SLAs."""
        taiwan_slas = [
            SLAMetric(
                name="quality_score_average",
                description="Average quality score must be >95",
                target_value=95.0,
                threshold_warning=93.0,
                threshold_critical=90.0,
                measurement_window_hours=24,
                calculation_method="average",
                taiwan_market_specific=True
            ),
            SLAMetric(
                name="validation_latency_p95",
                description="95th percentile validation latency <10ms",
                target_value=10.0,
                threshold_warning=8.0,
                threshold_critical=12.0,
                measurement_window_hours=1,
                calculation_method="p95",
                taiwan_market_specific=False
            ),
            SLAMetric(
                name="error_rate",
                description="Error rate must be <1%",
                target_value=0.01,
                threshold_warning=0.005,
                threshold_critical=0.02,
                measurement_window_hours=24,
                calculation_method="average",
                taiwan_market_specific=False
            ),
            SLAMetric(
                name="system_availability",
                description="System availability must be >99.9%",
                target_value=0.999,
                threshold_warning=0.995,
                threshold_critical=0.99,
                measurement_window_hours=24,
                calculation_method="average",
                taiwan_market_specific=True
            ),
            SLAMetric(
                name="taiwan_price_violations",
                description="Taiwan price limit violations <5 per day",
                target_value=5.0,
                threshold_warning=3.0,
                threshold_critical=10.0,
                measurement_window_hours=24,
                calculation_method="sum",
                taiwan_market_specific=True
            )
        ]
        
        for sla in taiwan_slas:
            self.add_sla_metric(sla)
    
    def add_sla_metric(self, sla_metric: SLAMetric) -> None:
        """Add SLA metric definition."""
        with self._lock:
            self.sla_definitions[sla_metric.name] = sla_metric
        logger.info(f"Added SLA metric: {sla_metric.name}")
    
    def record_measurement(self, 
                          metric_name: str, 
                          value: float, 
                          timestamp: Optional[datetime] = None) -> None:
        """Record a measurement for SLA tracking."""
        if metric_name not in self.sla_definitions:
            logger.warning(f"Unknown SLA metric: {metric_name}")
            return
        
        timestamp = timestamp or datetime.utcnow()
        
        with self._lock:
            self.measurement_history[metric_name].append((timestamp, value))
    
    def calculate_sla_compliance(self, 
                                metric_name: str,
                                end_time: Optional[datetime] = None) -> Optional[SLAResult]:
        """Calculate SLA compliance for a metric."""
        if metric_name not in self.sla_definitions:
            return None
        
        sla_metric = self.sla_definitions[metric_name]
        end_time = end_time or datetime.utcnow()
        start_time = end_time - timedelta(hours=sla_metric.measurement_window_hours)
        
        with self._lock:
            # Get measurements in window
            measurements = [
                (ts, value) for ts, value in self.measurement_history[metric_name]
                if start_time <= ts <= end_time
            ]
        
        if not measurements:
            return SLAResult(
                metric_name=metric_name,
                measurement_period=(start_time, end_time),
                target_value=sla_metric.target_value,
                actual_value=0.0,
                status=SLAStatus.UNKNOWN,
                compliance_percentage=0.0
            )
        
        values = [value for _, value in measurements]
        
        # Calculate actual value based on method
        if sla_metric.calculation_method == "average":
            actual_value = statistics.mean(values)
        elif sla_metric.calculation_method == "min":
            actual_value = min(values)
        elif sla_metric.calculation_method == "max":
            actual_value = max(values)
        elif sla_metric.calculation_method == "p95":
            actual_value = np.percentile(values, 95)
        elif sla_metric.calculation_method == "p99":
            actual_value = np.percentile(values, 99)
        elif sla_metric.calculation_method == "sum":
            actual_value = sum(values)
        else:
            actual_value = statistics.mean(values)
        
        # Determine compliance status
        status = SLAStatus.MEETING
        if actual_value < sla_metric.threshold_critical:
            status = SLAStatus.BREACHED
        elif actual_value < sla_metric.threshold_warning:
            status = SLAStatus.AT_RISK
        
        # Calculate compliance percentage
        if sla_metric.target_value > 0:
            compliance_percentage = min(100.0, (actual_value / sla_metric.target_value) * 100)
        else:
            compliance_percentage = 100.0 if status == SLAStatus.MEETING else 0.0
        
        # Check for breaches
        breach_duration = 0
        last_breach = None
        
        if status == SLAStatus.BREACHED:
            # Find continuous breach period
            breach_start = None
            for ts, value in reversed(measurements):
                if value >= sla_metric.threshold_critical:
                    break
                if breach_start is None:
                    breach_start = ts
                last_breach = ts
            
            if breach_start and last_breach:
                breach_duration = int((breach_start - last_breach).total_seconds() / 60)
        
        return SLAResult(
            metric_name=metric_name,
            measurement_period=(start_time, end_time),
            target_value=sla_metric.target_value,
            actual_value=actual_value,
            status=status,
            compliance_percentage=compliance_percentage,
            breach_duration_minutes=breach_duration,
            last_breach_time=last_breach
        )
    
    def get_all_sla_results(self, 
                           end_time: Optional[datetime] = None) -> List[SLAResult]:
        """Get SLA compliance results for all metrics."""
        results = []
        
        with self._lock:
            for metric_name in self.sla_definitions.keys():
                result = self.calculate_sla_compliance(metric_name, end_time)
                if result:
                    results.append(result)
        
        return results
    
    def get_taiwan_specific_slas(self) -> List[SLAResult]:
        """Get Taiwan market specific SLA results."""
        results = []
        
        with self._lock:
            for metric_name, sla_metric in self.sla_definitions.items():
                if sla_metric.taiwan_market_specific:
                    result = self.calculate_sla_compliance(metric_name)
                    if result:
                        results.append(result)
        
        return results
    
    def get_sla_summary(self) -> Dict[str, Any]:
        """Get comprehensive SLA summary."""
        all_results = self.get_all_sla_results()
        
        total_slas = len(all_results)
        meeting_count = len([r for r in all_results if r.status == SLAStatus.MEETING])
        at_risk_count = len([r for r in all_results if r.status == SLAStatus.AT_RISK])
        breached_count = len([r for r in all_results if r.status == SLAStatus.BREACHED])
        
        overall_compliance = (meeting_count / max(total_slas, 1)) * 100
        
        return {
            'total_slas': total_slas,
            'overall_compliance_percentage': overall_compliance,
            'status_distribution': {
                'meeting': meeting_count,
                'at_risk': at_risk_count,
                'breached': breached_count
            },
            'breached_slas': [
                {
                    'metric': r.metric_name,
                    'target': r.target_value,
                    'actual': r.actual_value,
                    'breach_duration_minutes': r.breach_duration_minutes
                }
                for r in all_results if r.status == SLAStatus.BREACHED
            ],
            'taiwan_specific_compliance': self._calculate_taiwan_compliance()
        }
    
    def _calculate_taiwan_compliance(self) -> float:
        """Calculate compliance for Taiwan specific SLAs."""
        taiwan_results = self.get_taiwan_specific_slas()
        
        if not taiwan_results:
            return 100.0
        
        meeting_count = len([r for r in taiwan_results if r.status == SLAStatus.MEETING])
        return (meeting_count / len(taiwan_results)) * 100


class QualityMetricsAggregator:
    """Aggregates and processes quality metrics from multiple sources."""
    
    def __init__(self):
        self.score_calculator = QualityScoreCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.sla_tracker = SLATracker()
        
        # Storage for aggregated metrics
        self.metrics_history: deque[QualityMetrics] = deque(maxlen=100000)
        self.symbol_aggregates: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.data_type_aggregates: Dict[DataType, Dict[str, float]] = defaultdict(dict)
        
        self._lock = threading.RLock()
    
    def process_quality_metrics(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Process and aggregate quality metrics."""
        with self._lock:
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Update SLA tracking
            self._update_sla_measurements(metrics)
            
            # Update aggregates
            self._update_aggregates(metrics)
            
            # Generate comprehensive analysis
            analysis = self._generate_analysis(metrics)
            
            return analysis
    
    def _update_sla_measurements(self, metrics: QualityMetrics) -> None:
        """Update SLA measurements from quality metrics."""
        timestamp = metrics.timestamp
        
        # Record basic metrics
        self.sla_tracker.record_measurement("quality_score_average", metrics.quality_score, timestamp)
        self.sla_tracker.record_measurement("validation_latency_p95", metrics.validation_latency_ms, timestamp)
        
        # Calculate error rate
        if metrics.validation_count > 0:
            error_rate = (metrics.error_count + metrics.critical_count) / metrics.validation_count
            self.sla_tracker.record_measurement("error_rate", error_rate, timestamp)
        
        # System availability (assume 100% if processing metrics)
        self.sla_tracker.record_measurement("system_availability", 1.0, timestamp)
        
        # Taiwan specific metrics
        if 'price_violation' in metrics.metadata:
            violation_count = metrics.metadata.get('price_violation_count', 0)
            self.sla_tracker.record_measurement("taiwan_price_violations", violation_count, timestamp)
    
    def _update_aggregates(self, metrics: QualityMetrics) -> None:
        """Update symbol and data type aggregates."""
        # Symbol aggregates
        symbol_agg = self.symbol_aggregates[metrics.symbol]
        self._update_aggregate_dict(symbol_agg, metrics)
        
        # Data type aggregates
        dt_agg = self.data_type_aggregates[metrics.data_type]
        self._update_aggregate_dict(dt_agg, metrics)
    
    def _update_aggregate_dict(self, agg_dict: Dict[str, float], metrics: QualityMetrics) -> None:
        """Update aggregate dictionary with new metrics."""
        # Rolling averages (simple implementation)
        alpha = 0.1  # Smoothing factor
        
        current_score = agg_dict.get('avg_quality_score', metrics.quality_score)
        agg_dict['avg_quality_score'] = alpha * metrics.quality_score + (1 - alpha) * current_score
        
        current_latency = agg_dict.get('avg_latency_ms', metrics.validation_latency_ms)
        agg_dict['avg_latency_ms'] = alpha * metrics.validation_latency_ms + (1 - alpha) * current_latency
        
        # Update counts
        agg_dict['total_validations'] = agg_dict.get('total_validations', 0) + metrics.validation_count
        agg_dict['total_errors'] = agg_dict.get('total_errors', 0) + metrics.error_count + metrics.critical_count
        
        # Calculate error rate
        if agg_dict['total_validations'] > 0:
            agg_dict['error_rate'] = agg_dict['total_errors'] / agg_dict['total_validations']
        
        agg_dict['last_update'] = metrics.timestamp.timestamp()
    
    def _generate_analysis(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate comprehensive analysis from metrics."""
        recent_metrics = self._get_recent_metrics(hours=1)
        
        analysis = {
            'current_metrics': {
                'quality_score': metrics.quality_score,
                'latency_ms': metrics.validation_latency_ms,
                'success_rate': metrics.success_rate,
                'alert_level': metrics.alert_level.value
            },
            'recent_trends': self._analyze_recent_trends(recent_metrics),
            'sla_status': self.sla_tracker.get_sla_summary(),
            'symbol_performance': self._get_top_symbols_analysis(),
            'data_type_performance': self._get_data_type_analysis(),
            'anomalies': self._detect_recent_anomalies(recent_metrics)
        }
        
        return analysis
    
    def _get_recent_metrics(self, hours: int = 1) -> List[QualityMetrics]:
        """Get metrics from recent time window."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff]
    
    def _analyze_recent_trends(self, recent_metrics: List[QualityMetrics]) -> Dict[str, Any]:
        """Analyze trends in recent metrics."""
        if len(recent_metrics) < 10:
            return {"insufficient_data": True}
        
        # Extract data points
        score_points = [(m.timestamp, m.quality_score) for m in recent_metrics]
        latency_points = [(m.timestamp, m.validation_latency_ms) for m in recent_metrics]
        
        # Analyze trends
        score_trend = self.trend_analyzer.analyze_trend(score_points, MetricType.SCORE)
        latency_trend = self.trend_analyzer.analyze_trend(latency_points, MetricType.LATENCY)
        
        return {
            'quality_score_trend': {
                'direction': score_trend.trend_direction if score_trend else 'unknown',
                'strength': score_trend.trend_strength if score_trend else 0.0,
                'confidence': score_trend.confidence_level if score_trend else 0.0
            },
            'latency_trend': {
                'direction': latency_trend.trend_direction if latency_trend else 'unknown',
                'strength': latency_trend.trend_strength if latency_trend else 0.0,
                'confidence': latency_trend.confidence_level if latency_trend else 0.0
            }
        }
    
    def _get_top_symbols_analysis(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get analysis of top performing symbols."""
        with self._lock:
            symbol_data = []
            
            for symbol, agg in self.symbol_aggregates.items():
                symbol_data.append({
                    'symbol': symbol,
                    'avg_quality_score': agg.get('avg_quality_score', 0),
                    'avg_latency_ms': agg.get('avg_latency_ms', 0),
                    'error_rate': agg.get('error_rate', 0),
                    'total_validations': agg.get('total_validations', 0)
                })
            
            # Sort by quality score
            symbol_data.sort(key=lambda x: x['avg_quality_score'], reverse=True)
            
            return symbol_data[:limit]
    
    def _get_data_type_analysis(self) -> List[Dict[str, Any]]:
        """Get analysis by data type."""
        with self._lock:
            analysis = []
            
            for data_type, agg in self.data_type_aggregates.items():
                analysis.append({
                    'data_type': data_type.value,
                    'avg_quality_score': agg.get('avg_quality_score', 0),
                    'avg_latency_ms': agg.get('avg_latency_ms', 0),
                    'error_rate': agg.get('error_rate', 0),
                    'total_validations': agg.get('total_validations', 0)
                })
            
            return analysis
    
    def _detect_recent_anomalies(self, recent_metrics: List[QualityMetrics]) -> List[Dict[str, Any]]:
        """Detect anomalies in recent metrics."""
        if len(recent_metrics) < 20:
            return []
        
        # Extract values for anomaly detection
        scores = [m.quality_score for m in recent_metrics]
        latencies = [m.validation_latency_ms for m in recent_metrics]
        
        # Detect anomalies
        score_anomalies = self.trend_analyzer.detect_anomalies(scores, method="iqr")
        latency_anomalies = self.trend_analyzer.detect_anomalies(latencies, method="iqr")
        
        anomalies = []
        
        for idx in score_anomalies:
            if idx < len(recent_metrics):
                anomalies.append({
                    'type': 'quality_score',
                    'timestamp': recent_metrics[idx].timestamp.isoformat(),
                    'value': recent_metrics[idx].quality_score,
                    'symbol': recent_metrics[idx].symbol
                })
        
        for idx in latency_anomalies:
            if idx < len(recent_metrics):
                anomalies.append({
                    'type': 'latency',
                    'timestamp': recent_metrics[idx].timestamp.isoformat(),
                    'value': recent_metrics[idx].validation_latency_ms,
                    'symbol': recent_metrics[idx].symbol
                })
        
        return anomalies
    
    def generate_quality_report(self, 
                               hours: int = 24,
                               include_trends: bool = True,
                               include_sla: bool = True) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get metrics in time window
        window_metrics = [
            m for m in self.metrics_history 
            if start_time <= m.timestamp <= end_time
        ]
        
        if not window_metrics:
            return {"error": "No metrics available for the specified time window"}
        
        # Calculate summary statistics
        scores = [m.quality_score for m in window_metrics]
        latencies = [m.validation_latency_ms for m in window_metrics]
        
        report = {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': hours
            },
            'summary_statistics': {
                'total_validations': len(window_metrics),
                'avg_quality_score': statistics.mean(scores),
                'min_quality_score': min(scores),
                'max_quality_score': max(scores),
                'avg_latency_ms': statistics.mean(latencies),
                'max_latency_ms': max(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'unique_symbols': len(set(m.symbol for m in window_metrics)),
                'data_types_processed': len(set(m.data_type for m in window_metrics))
            }
        }
        
        if include_trends:
            score_points = [(m.timestamp, m.quality_score) for m in window_metrics]
            trend = self.trend_analyzer.analyze_trend(score_points, MetricType.SCORE)
            
            report['trend_analysis'] = {
                'quality_score_trend': trend.trend_direction if trend else 'unknown',
                'trend_strength': trend.trend_strength if trend else 0.0,
                'confidence_level': trend.confidence_level if trend else 0.0
            }
        
        if include_sla:
            report['sla_compliance'] = self.sla_tracker.get_sla_summary()
        
        return report


# Utility functions

def create_taiwan_metrics_system() -> QualityMetricsAggregator:
    """Create pre-configured metrics system for Taiwan market."""
    aggregator = QualityMetricsAggregator()
    
    # Configure Taiwan specific settings
    aggregator.score_calculator.taiwan_multipliers.update({
        'tse_violation': 2.5,
        'otc_violation': 2.0,
        'futures_violation': 1.8
    })
    
    logger.info("Taiwan market metrics system created")
    return aggregator