"""
Comprehensive Monitoring and Logging Framework for Data Pipeline.

This module provides real-time monitoring, performance tracking, alerting,
and comprehensive logging for the FinLab data integration pipeline.
"""

import logging
import time
import threading
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from ..core.temporal import DataType
from .data_validation import ValidationReport, ValidationSeverity

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Performance metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class PerformanceMetric:
    """Performance metric representation."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


@dataclass
class Alert:
    """System alert representation."""
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()


@dataclass
class PipelineStatus:
    """Pipeline component status."""
    component: str
    status: str  # healthy, degraded, failed, unknown
    last_update: datetime = field(default_factory=datetime.utcnow)
    message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == "healthy"

    def is_stale(self, max_age_minutes: int = 10) -> bool:
        """Check if status is stale."""
        age = datetime.utcnow() - self.last_update
        return age.total_seconds() > (max_age_minutes * 60)


class PerformanceTracker:
    """Real-time performance tracking and analysis."""

    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.timers: Dict[str, float] = {}
        self._lock = threading.Lock()

    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        with self._lock:
            self.metrics_history[metric.name].append(metric)

    def start_timer(self, name: str) -> None:
        """Start a timing measurement."""
        self.timers[name] = time.time()

    def end_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """End a timing measurement and record metric."""
        if name not in self.timers:
            logger.warning(f"Timer {name} not started")
            return 0.0

        duration = time.time() - self.timers[name]
        del self.timers[name]

        metric = PerformanceMetric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMING,
            tags=tags or {},
            unit="seconds"
        )
        self.record_metric(metric)
        return duration

    def record_counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags or {}
        )
        self.record_metric(metric)

    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = "") -> None:
        """Record a gauge metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags or {},
            unit=unit
        )
        self.record_metric(metric)

    def get_metric_stats(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get statistics for a metric over a time window."""
        with self._lock:
            if name not in self.metrics_history:
                return {"error": f"Metric {name} not found"}

            cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            recent_metrics = [
                m for m in self.metrics_history[name]
                if m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                return {"error": f"No recent data for {name}"}

            values = [m.value for m in recent_metrics]

            stats = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "window_minutes": window_minutes
            }

            if len(values) > 1:
                stats["stddev"] = statistics.stdev(values)
                stats["p50"] = statistics.median(values)

                # Calculate percentiles if enough data
                if len(values) >= 10:
                    sorted_values = sorted(values)
                    stats["p95"] = sorted_values[int(0.95 * len(sorted_values))]
                    stats["p99"] = sorted_values[int(0.99 * len(sorted_values))]

            return stats

    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get all metrics from recent time period."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent = []

        with self._lock:
            for metric_name, metric_deque in self.metrics_history.items():
                for metric in metric_deque:
                    if metric.timestamp >= cutoff_time:
                        recent.append(metric)

        return sorted(recent, key=lambda m: m.timestamp, reverse=True)


class AlertManager:
    """Centralized alert management and notification."""

    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        self._lock = threading.Lock()
        self.alert_counts = defaultdict(int)

    def add_alert_handler(self, level: AlertLevel, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler for a specific level."""
        self.alert_handlers[level].append(handler)

    def raise_alert(self, level: AlertLevel, title: str, message: str,
                   component: str, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Raise a new alert."""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            component=component,
            metadata=metadata or {}
        )

        with self._lock:
            self.alerts.append(alert)
            self.alert_counts[level] += 1

        # Trigger handlers
        for handler in self.alert_handlers[level]:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.log(
            self._get_log_level(level),
            f"[{component}] {title}: {message}"
        )

        return alert

    def _get_log_level(self, alert_level: AlertLevel) -> int:
        """Convert alert level to logging level."""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        return mapping[alert_level]

    def get_recent_alerts(self, minutes: int = 60, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get recent alerts, optionally filtered by level."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self._lock:
            recent = [
                alert for alert in self.alerts
                if alert.timestamp >= cutoff_time and
                (level is None or alert.level == level)
            ]

        return sorted(recent, key=lambda a: a.timestamp, reverse=True)

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alerts over time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]

        summary = {
            "total_alerts": len(recent_alerts),
            "by_level": defaultdict(int),
            "by_component": defaultdict(int),
            "resolved_count": 0,
            "unresolved_count": 0
        }

        for alert in recent_alerts:
            summary["by_level"][alert.level.value] += 1
            summary["by_component"][alert.component] += 1

            if alert.resolved:
                summary["resolved_count"] += 1
            else:
                summary["unresolved_count"] += 1

        return dict(summary)  # Convert defaultdict to regular dict


class PipelineMonitor:
    """Comprehensive pipeline monitoring system."""

    def __init__(self,
                 log_dir: Optional[Path] = None,
                 alert_thresholds: Optional[Dict[str, Dict[str, Any]]] = None):

        self.performance_tracker = PerformanceTracker()
        self.alert_manager = AlertManager()
        self.component_status: Dict[str, PipelineStatus] = {}
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "update_duration": {"warning": 300, "error": 600},  # seconds
            "error_rate": {"warning": 0.05, "error": 0.10},    # percentage
            "validation_failures": {"warning": 10, "error": 50}, # count
            "data_lag": {"warning": 2, "error": 5}             # days
        }

        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Setup structured logging
        self._setup_logging()

        logger.info("Pipeline monitor initialized")

    def _setup_logging(self) -> None:
        """Setup structured logging configuration."""
        log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File handler for pipeline logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Add handler to pipeline loggers
        pipeline_logger = logging.getLogger('src.data.pipeline')
        pipeline_logger.addHandler(file_handler)
        pipeline_logger.setLevel(logging.INFO)

    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Pipeline monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            self._stop_event.set()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=10)
            logger.info("Pipeline monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_system_health()
                self._check_alert_thresholds()
                self._cleanup_old_data()

                # Sleep for monitoring interval
                self._stop_event.wait(60)  # Check every minute

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self._stop_event.wait(10)  # Brief pause on error

    def _check_system_health(self) -> None:
        """Check overall system health."""
        current_time = datetime.utcnow()

        for component, status in self.component_status.items():
            if status.is_stale():
                self.alert_manager.raise_alert(
                    AlertLevel.WARNING,
                    f"Component {component} status stale",
                    f"No status update for {component} in over 10 minutes",
                    component,
                    {"last_update": status.last_update.isoformat()}
                )

    def _check_alert_thresholds(self) -> None:
        """Check performance metrics against alert thresholds."""
        # Check update duration
        duration_stats = self.performance_tracker.get_metric_stats("update_duration", 60)
        if "mean" in duration_stats:
            mean_duration = duration_stats["mean"]
            thresholds = self.alert_thresholds["update_duration"]

            if mean_duration > thresholds["error"]:
                self.alert_manager.raise_alert(
                    AlertLevel.ERROR,
                    "High update duration",
                    f"Average update time {mean_duration:.1f}s exceeds threshold {thresholds['error']}s",
                    "incremental_updater",
                    {"mean_duration": mean_duration, "threshold": thresholds["error"]}
                )
            elif mean_duration > thresholds["warning"]:
                self.alert_manager.raise_alert(
                    AlertLevel.WARNING,
                    "Elevated update duration",
                    f"Average update time {mean_duration:.1f}s exceeds warning threshold {thresholds['warning']}s",
                    "incremental_updater",
                    {"mean_duration": mean_duration, "threshold": thresholds["warning"]}
                )

    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        # This runs periodically to prevent memory bloat
        cutoff_time = datetime.utcnow() - timedelta(days=7)

        # Clean up old log files
        for log_file in self.log_dir.glob("pipeline_*.log"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_time:
                    log_file.unlink()
                    logger.info(f"Cleaned up old log file: {log_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up log file {log_file}: {e}")

    def update_component_status(self,
                               component: str,
                               status: str,
                               message: str = "",
                               metrics: Optional[Dict[str, float]] = None) -> None:
        """Update status for a pipeline component."""
        self.component_status[component] = PipelineStatus(
            component=component,
            status=status,
            message=message,
            metrics=metrics or {}
        )

        logger.info(f"Component {component} status: {status} - {message}")

    def track_update_operation(self,
                              symbol: str,
                              records_processed: int,
                              duration: float,
                              errors: int = 0) -> None:
        """Track an update operation."""
        # Record performance metrics
        self.performance_tracker.record_counter("records_processed", records_processed)
        self.performance_tracker.record_gauge("update_duration", duration, {"symbol": symbol}, "seconds")

        if errors > 0:
            self.performance_tracker.record_counter("update_errors", errors, {"symbol": symbol})

        # Calculate error rate
        error_rate = errors / max(records_processed, 1)
        self.performance_tracker.record_gauge("error_rate", error_rate, {"symbol": symbol}, "percentage")

        # Check for alerts
        if error_rate > self.alert_thresholds["error_rate"]["error"]:
            self.alert_manager.raise_alert(
                AlertLevel.ERROR,
                "High error rate during update",
                f"Symbol {symbol}: {error_rate:.2%} error rate ({errors}/{records_processed})",
                "incremental_updater",
                {"symbol": symbol, "error_rate": error_rate, "errors": errors, "total": records_processed}
            )

    def track_validation_results(self, report: ValidationReport) -> None:
        """Track validation results and generate alerts."""
        # Record validation metrics
        self.performance_tracker.record_gauge(
            "validation_quality_score",
            report.quality_score,
            {"symbol": report.symbol}
        )

        self.performance_tracker.record_counter(
            "validation_failures",
            report.failed_validations,
            {"symbol": report.symbol}
        )

        # Check for critical validation issues
        if report.critical_issues_count > 0:
            self.alert_manager.raise_alert(
                AlertLevel.CRITICAL,
                "Critical data validation issues",
                f"Symbol {report.symbol}: {report.critical_issues_count} critical validation failures",
                "data_validator",
                {"symbol": report.symbol, "critical_issues": report.critical_issues_count}
            )

        # Check quality score threshold
        if report.quality_score < 70:
            self.alert_manager.raise_alert(
                AlertLevel.WARNING,
                "Low data quality score",
                f"Symbol {report.symbol}: Quality score {report.quality_score:.1f}% below threshold",
                "data_validator",
                {"symbol": report.symbol, "quality_score": report.quality_score}
            )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        recent_metrics = self.performance_tracker.get_recent_metrics(60)
        recent_alerts = self.alert_manager.get_recent_alerts(60)
        alert_summary = self.alert_manager.get_alert_summary(24)

        # Component health summary
        healthy_components = sum(1 for s in self.component_status.values() if s.is_healthy())
        total_components = len(self.component_status)

        # Performance summary
        update_stats = self.performance_tracker.get_metric_stats("update_duration", 60)
        error_stats = self.performance_tracker.get_metric_stats("update_errors", 60)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": {
                "healthy_components": healthy_components,
                "total_components": total_components,
                "health_percentage": (healthy_components / max(total_components, 1)) * 100
            },
            "performance": {
                "update_duration": update_stats,
                "error_count": error_stats,
                "recent_metrics_count": len(recent_metrics)
            },
            "alerts": {
                "recent_count": len(recent_alerts),
                "summary": alert_summary
            },
            "components": {
                name: {
                    "status": status.status,
                    "last_update": status.last_update.isoformat(),
                    "message": status.message,
                    "metrics": status.metrics
                }
                for name, status in self.component_status.items()
            }
        }

    def export_metrics_json(self, minutes: int = 60) -> str:
        """Export recent metrics as JSON."""
        recent_metrics = self.performance_tracker.get_recent_metrics(minutes)
        metrics_data = [m.to_dict() for m in recent_metrics]
        return json.dumps(metrics_data, indent=2)

    def export_alerts_json(self, hours: int = 24) -> str:
        """Export recent alerts as JSON."""
        recent_alerts = self.alert_manager.get_recent_alerts(hours * 60)
        alerts_data = [a.to_dict() for a in recent_alerts]
        return json.dumps(alerts_data, indent=2)

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Global monitor instance
pipeline_monitor = PipelineMonitor()


# Decorator for timing operations
def monitor_performance(operation_name: str, component: str = "unknown"):
    """Decorator to automatically monitor operation performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            pipeline_monitor.performance_tracker.start_timer(operation_name)

            try:
                result = func(*args, **kwargs)
                pipeline_monitor.update_component_status(component, "healthy", f"{operation_name} completed")
                return result

            except Exception as e:
                pipeline_monitor.alert_manager.raise_alert(
                    AlertLevel.ERROR,
                    f"{operation_name} failed",
                    str(e),
                    component,
                    {"function": func.__name__, "error": str(e)}
                )
                raise

            finally:
                duration = pipeline_monitor.performance_tracker.end_timer(operation_name, {"component": component})

        return wrapper
    return decorator