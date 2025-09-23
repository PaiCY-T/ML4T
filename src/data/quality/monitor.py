"""
Real-time Data Quality Monitoring System.

This module provides real-time monitoring of data quality with <10ms latency,
designed for the Taiwan market with specific monitoring rules and thresholds.
Integrates with the validation engine to provide continuous quality oversight.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
import threading
from concurrent.futures import ThreadPoolExecutor

from .validation_engine import (
    ValidationEngine, ValidationContext, ValidationOutput, ValidationResult
)
from .validators import QualityIssue, SeverityLevel, QualityCheckType
from ..core.temporal import TemporalValue, DataType

logger = logging.getLogger(__name__)


class MonitoringStatus(Enum):
    """Monitoring system status."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class AlertLevel(Enum):
    """Alert urgency levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QualityMetrics:
    """Real-time quality metrics."""
    timestamp: datetime
    symbol: str
    data_type: DataType
    quality_score: float  # 0-100
    validation_count: int
    passed_validations: int
    warning_count: int
    error_count: int
    critical_count: int
    validation_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        if self.validation_count == 0:
            return 0.0
        return (self.passed_validations / self.validation_count) * 100
    
    @property
    def alert_level(self) -> AlertLevel:
        """Determine alert level based on metrics."""
        if self.critical_count > 0:
            return AlertLevel.CRITICAL
        elif self.error_count > 0:
            return AlertLevel.HIGH
        elif self.warning_count > 0:
            return AlertLevel.MEDIUM
        elif self.quality_score < 95:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW


@dataclass
class MonitoringThresholds:
    """Configurable monitoring thresholds."""
    quality_score_warning: float = 90.0
    quality_score_critical: float = 85.0
    latency_warning_ms: float = 8.0
    latency_critical_ms: float = 10.0
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.10  # 10%
    taiwan_specific: Dict[str, Any] = field(default_factory=lambda: {
        'price_limit_buffer': 0.02,  # 2% buffer above 10% limit
        'volume_spike_threshold': 5.0,  # 5x average volume
        'trading_hours_tolerance_minutes': 5
    })


class QualityMonitor:
    """Real-time data quality monitoring system."""
    
    def __init__(self, 
                 validation_engine: ValidationEngine,
                 thresholds: Optional[MonitoringThresholds] = None,
                 history_size: int = 10000,
                 metrics_window_seconds: int = 300):  # 5 minute windows
        self.validation_engine = validation_engine
        self.thresholds = thresholds or MonitoringThresholds()
        self.history_size = history_size
        self.metrics_window_seconds = metrics_window_seconds
        
        # System state
        self.status = MonitoringStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None
        
        # Metrics storage (thread-safe)
        self._metrics_lock = threading.RLock()
        self._quality_history: deque[QualityMetrics] = deque(maxlen=history_size)
        self._symbol_metrics: Dict[str, QualityMetrics] = {}
        self._data_type_metrics: Dict[DataType, QualityMetrics] = {}
        
        # Real-time monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._validation_callbacks: List[Callable[[QualityMetrics], None]] = []
        self._alert_callbacks: List[Callable[[QualityMetrics, str], None]] = []
        
        # Performance tracking
        self.total_validations = 0
        self.total_latency_ms = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Taiwan market specific tracking
        self._taiwan_trading_hours = {
            'start': time(9, 0),   # 09:00 TST
            'end': time(13, 30)    # 13:30 TST
        }
        self._taiwan_holidays: Set[date] = set()
        
        logger.info("Quality monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.status in [MonitoringStatus.ACTIVE, MonitoringStatus.PAUSED]:
            logger.warning("Monitor already running")
            return
        
        self.status = MonitoringStatus.ACTIVE
        self.start_time = datetime.utcnow()
        logger.info("Real-time quality monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
        
        self.status = MonitoringStatus.STOPPED
        logger.info("Real-time quality monitoring stopped")
    
    def pause_monitoring(self) -> None:
        """Pause monitoring temporarily."""
        self.status = MonitoringStatus.PAUSED
        logger.info("Real-time quality monitoring paused")
    
    def resume_monitoring(self) -> None:
        """Resume monitoring."""
        self.status = MonitoringStatus.ACTIVE
        logger.info("Real-time quality monitoring resumed")
    
    async def monitor_validation(self, value: TemporalValue, 
                                context: Optional[ValidationContext] = None) -> QualityMetrics:
        """Monitor a single validation with <10ms latency target."""
        if self.status != MonitoringStatus.ACTIVE:
            return None
        
        start_time = time.perf_counter()
        
        try:
            # Perform validation
            validation_results = await self.validation_engine.validate_value(value, context)
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Generate quality metrics
            metrics = self._calculate_quality_metrics(value, validation_results, latency_ms)
            
            # Update monitoring state
            with self._metrics_lock:
                self._update_metrics(metrics)
                self.total_validations += 1
                self.total_latency_ms += latency_ms
                self.last_update_time = datetime.utcnow()
            
            # Check thresholds and trigger alerts
            self._check_thresholds(metrics)
            
            # Notify callbacks
            for callback in self._validation_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Validation callback error: {e}")
            
            # Taiwan market specific checks
            if self._is_taiwan_market_hours(value.value_date):
                self._taiwan_market_checks(metrics)
            
            logger.debug(f"Monitored validation for {value.symbol} in {latency_ms:.2f}ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            self.status = MonitoringStatus.ERROR
            raise
    
    async def monitor_batch(self, values: List[TemporalValue],
                           contexts: Optional[List[ValidationContext]] = None) -> List[QualityMetrics]:
        """Monitor batch validation efficiently."""
        if self.status != MonitoringStatus.ACTIVE:
            return []
        
        start_time = time.perf_counter()
        
        try:
            # Perform batch validation
            validation_results = await self.validation_engine.validate_batch(values, contexts)
            
            # Calculate per-value metrics
            metrics_list = []
            for value in values:
                if value in validation_results:
                    results = validation_results[value]
                    # Estimate latency per value
                    batch_latency = (time.perf_counter() - start_time) * 1000
                    per_value_latency = batch_latency / len(values)
                    
                    metrics = self._calculate_quality_metrics(value, results, per_value_latency)
                    metrics_list.append(metrics)
                    
                    # Update monitoring state
                    with self._metrics_lock:
                        self._update_metrics(metrics)
            
            # Update global stats
            with self._metrics_lock:
                self.total_validations += len(values)
                self.total_latency_ms += (time.perf_counter() - start_time) * 1000
                self.last_update_time = datetime.utcnow()
            
            logger.debug(f"Monitored batch of {len(values)} validations")
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Batch monitoring error: {e}")
            self.status = MonitoringStatus.ERROR
            raise
    
    def _calculate_quality_metrics(self, value: TemporalValue, 
                                 validation_results: List[ValidationOutput],
                                 latency_ms: float) -> QualityMetrics:
        """Calculate quality metrics from validation results."""
        validation_count = len(validation_results)
        passed_count = sum(1 for r in validation_results if r.result == ValidationResult.PASS)
        warning_count = sum(1 for r in validation_results if r.result == ValidationResult.WARNING)
        error_count = sum(len([i for i in r.issues if i.severity == SeverityLevel.ERROR]) 
                         for r in validation_results)
        critical_count = sum(len([i for i in r.issues if i.severity == SeverityLevel.CRITICAL]) 
                           for r in validation_results)
        
        # Calculate quality score (0-100)
        quality_score = self._calculate_quality_score(validation_results)
        
        return QualityMetrics(
            timestamp=datetime.utcnow(),
            symbol=value.symbol or "",
            data_type=value.data_type,
            quality_score=quality_score,
            validation_count=validation_count,
            passed_validations=passed_count,
            warning_count=warning_count,
            error_count=error_count,
            critical_count=critical_count,
            validation_latency_ms=latency_ms,
            metadata={
                'value_date': value.value_date.isoformat(),
                'as_of_date': value.as_of_date.isoformat(),
                'validation_results_count': validation_count
            }
        )
    
    def _calculate_quality_score(self, validation_results: List[ValidationOutput]) -> float:
        """Calculate quality score (0-100) based on validation results."""
        if not validation_results:
            return 100.0
        
        base_score = 100.0
        
        for result in validation_results:
            if result.result == ValidationResult.FAIL:
                base_score -= 50.0  # Fail is severe
            elif result.result == ValidationResult.WARNING:
                base_score -= 10.0  # Warning is moderate
            
            # Additional penalties for issues
            for issue in result.issues:
                if issue.severity == SeverityLevel.CRITICAL:
                    base_score -= 30.0
                elif issue.severity == SeverityLevel.ERROR:
                    base_score -= 15.0
                elif issue.severity == SeverityLevel.WARNING:
                    base_score -= 5.0
        
        return max(0.0, base_score)
    
    def _update_metrics(self, metrics: QualityMetrics) -> None:
        """Update internal metrics storage (thread-safe)."""
        # Add to history
        self._quality_history.append(metrics)
        
        # Update per-symbol metrics
        self._symbol_metrics[metrics.symbol] = metrics
        
        # Update per-data-type metrics
        self._data_type_metrics[metrics.data_type] = metrics
    
    def _check_thresholds(self, metrics: QualityMetrics) -> None:
        """Check metrics against thresholds and trigger alerts."""
        alerts = []
        
        # Quality score thresholds
        if metrics.quality_score < self.thresholds.quality_score_critical:
            alerts.append(f"Critical quality score: {metrics.quality_score:.1f}")
        elif metrics.quality_score < self.thresholds.quality_score_warning:
            alerts.append(f"Low quality score: {metrics.quality_score:.1f}")
        
        # Latency thresholds
        if metrics.validation_latency_ms > self.thresholds.latency_critical_ms:
            alerts.append(f"Critical latency: {metrics.validation_latency_ms:.2f}ms")
        elif metrics.validation_latency_ms > self.thresholds.latency_warning_ms:
            alerts.append(f"High latency: {metrics.validation_latency_ms:.2f}ms")
        
        # Error rate thresholds
        error_rate = (metrics.error_count + metrics.critical_count) / max(metrics.validation_count, 1)
        if error_rate > self.thresholds.error_rate_critical:
            alerts.append(f"Critical error rate: {error_rate:.1%}")
        elif error_rate > self.thresholds.error_rate_warning:
            alerts.append(f"High error rate: {error_rate:.1%}")
        
        # Trigger alerts
        for alert_message in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(metrics, alert_message)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def _taiwan_market_checks(self, metrics: QualityMetrics) -> None:
        """Perform Taiwan market specific checks."""
        taiwan_config = self.thresholds.taiwan_specific
        
        # Check if during trading hours
        current_time = datetime.utcnow() + timedelta(hours=8)  # Convert to TST
        current_time_only = current_time.time()
        
        if not (self._taiwan_trading_hours['start'] <= current_time_only <= self._taiwan_trading_hours['end']):
            # Data outside trading hours - should be minimal
            if metrics.validation_count > 0:
                logger.info(f"Data validation outside trading hours: {metrics.symbol}")
        
        # Taiwan specific metadata checks
        if 'price_change_pct' in metrics.metadata:
            price_change = metrics.metadata['price_change_pct']
            limit_with_buffer = 0.10 + taiwan_config['price_limit_buffer']
            
            if price_change > limit_with_buffer:
                logger.warning(f"Price change {price_change:.2%} exceeds Taiwan limit with buffer")
    
    def _is_taiwan_market_hours(self, check_date: date) -> bool:
        """Check if date is during Taiwan market hours."""
        # Check if it's a weekday and not a holiday
        if check_date.weekday() >= 5:  # Weekend
            return False
        
        if check_date in self._taiwan_holidays:
            return False
        
        return True
    
    def add_validation_callback(self, callback: Callable[[QualityMetrics], None]) -> None:
        """Add callback for validation events."""
        self._validation_callbacks.append(callback)
        logger.info(f"Added validation callback: {callback.__name__}")
    
    def add_alert_callback(self, callback: Callable[[QualityMetrics, str], None]) -> None:
        """Add callback for alert events."""
        self._alert_callbacks.append(callback)
        logger.info(f"Added alert callback: {callback.__name__}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        with self._metrics_lock:
            avg_latency = self.total_latency_ms / max(self.total_validations, 1)
            
            # Calculate recent metrics (last 5 minutes)
            recent_cutoff = datetime.utcnow() - timedelta(seconds=self.metrics_window_seconds)
            recent_metrics = [m for m in self._quality_history if m.timestamp >= recent_cutoff]
            
            recent_quality_scores = [m.quality_score for m in recent_metrics]
            recent_latencies = [m.validation_latency_ms for m in recent_metrics]
            
            return {
                'status': self.status.value,
                'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds() 
                                if self.start_time else 0,
                'total_validations': self.total_validations,
                'average_latency_ms': avg_latency,
                'last_update': self.last_update_time.isoformat() 
                              if self.last_update_time else None,
                'recent_metrics': {
                    'count': len(recent_metrics),
                    'avg_quality_score': sum(recent_quality_scores) / len(recent_quality_scores)
                                        if recent_quality_scores else 0,
                    'avg_latency_ms': sum(recent_latencies) / len(recent_latencies)
                                     if recent_latencies else 0,
                    'min_quality_score': min(recent_quality_scores) if recent_quality_scores else 0,
                    'max_latency_ms': max(recent_latencies) if recent_latencies else 0
                },
                'symbol_count': len(self._symbol_metrics),
                'data_types_monitored': len(self._data_type_metrics),
                'thresholds': {
                    'quality_score_warning': self.thresholds.quality_score_warning,
                    'quality_score_critical': self.thresholds.quality_score_critical,
                    'latency_warning_ms': self.thresholds.latency_warning_ms,
                    'latency_critical_ms': self.thresholds.latency_critical_ms
                }
            }
    
    def get_symbol_metrics(self, symbol: str) -> Optional[QualityMetrics]:
        """Get latest metrics for a specific symbol."""
        with self._metrics_lock:
            return self._symbol_metrics.get(symbol)
    
    def get_data_type_metrics(self, data_type: DataType) -> Optional[QualityMetrics]:
        """Get latest metrics for a specific data type."""
        with self._metrics_lock:
            return self._data_type_metrics.get(data_type)
    
    def get_quality_history(self, 
                           symbol: Optional[str] = None,
                           data_type: Optional[DataType] = None,
                           limit: Optional[int] = None) -> List[QualityMetrics]:
        """Get quality metrics history with optional filtering."""
        with self._metrics_lock:
            history = list(self._quality_history)
            
            # Apply filters
            if symbol:
                history = [m for m in history if m.symbol == symbol]
            
            if data_type:
                history = [m for m in history if m.data_type == data_type]
            
            # Apply limit
            if limit:
                history = history[-limit:]
            
            return history
    
    def reset_metrics(self) -> None:
        """Reset all monitoring metrics."""
        with self._metrics_lock:
            self._quality_history.clear()
            self._symbol_metrics.clear()
            self._data_type_metrics.clear()
            self.total_validations = 0
            self.total_latency_ms = 0.0
            self.cache_hits = 0
            self.cache_misses = 0
            self.last_update_time = None
        
        logger.info("Monitoring metrics reset")
    
    def set_taiwan_holidays(self, holidays: Set[date]) -> None:
        """Set Taiwan market holidays for monitoring."""
        self._taiwan_holidays = holidays.copy()
        logger.info(f"Updated Taiwan holidays: {len(holidays)} dates")
    
    def update_thresholds(self, new_thresholds: MonitoringThresholds) -> None:
        """Update monitoring thresholds."""
        self.thresholds = new_thresholds
        logger.info("Monitoring thresholds updated")


class RealtimeQualityStream:
    """Real-time quality metrics streaming interface."""
    
    def __init__(self, monitor: QualityMonitor):
        self.monitor = monitor
        self._subscribers: Dict[str, Callable[[QualityMetrics], None]] = {}
        self._stream_active = False
        
    def subscribe(self, subscriber_id: str, callback: Callable[[QualityMetrics], None]) -> None:
        """Subscribe to real-time quality metrics."""
        self._subscribers[subscriber_id] = callback
        self.monitor.add_validation_callback(callback)
        logger.info(f"Subscriber {subscriber_id} added to quality stream")
    
    def unsubscribe(self, subscriber_id: str) -> None:
        """Unsubscribe from real-time quality metrics."""
        if subscriber_id in self._subscribers:
            del self._subscribers[subscriber_id]
            logger.info(f"Subscriber {subscriber_id} removed from quality stream")
    
    def start_stream(self) -> None:
        """Start real-time streaming."""
        self._stream_active = True
        logger.info("Real-time quality stream started")
    
    def stop_stream(self) -> None:
        """Stop real-time streaming."""
        self._stream_active = False
        logger.info("Real-time quality stream stopped")
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Get streaming status."""
        return {
            'active': self._stream_active,
            'subscriber_count': len(self._subscribers),
            'monitor_status': self.monitor.status.value
        }


# Utility functions for monitoring setup

def create_taiwan_market_monitor(validation_engine: ValidationEngine) -> QualityMonitor:
    """Create a pre-configured monitor for Taiwan market."""
    taiwan_thresholds = MonitoringThresholds(
        quality_score_warning=95.0,  # High standards for Taiwan market
        quality_score_critical=90.0,
        latency_warning_ms=8.0,      # Strict latency requirements
        latency_critical_ms=10.0,
        error_rate_warning=0.02,     # 2% error rate warning
        error_rate_critical=0.05,    # 5% error rate critical
        taiwan_specific={
            'price_limit_buffer': 0.01,      # 1% buffer above 10% limit
            'volume_spike_threshold': 5.0,    # 5x average volume
            'trading_hours_tolerance_minutes': 2  # Strict trading hours
        }
    )
    
    monitor = QualityMonitor(
        validation_engine=validation_engine,
        thresholds=taiwan_thresholds,
        history_size=50000,  # Large history for Taiwan market
        metrics_window_seconds=300  # 5 minute windows
    )
    
    # Set Taiwan holidays (example - would be loaded from external source)
    taiwan_holidays = {
        date(2024, 1, 1),   # New Year
        date(2024, 2, 10),  # Lunar New Year (example)
        date(2024, 10, 10), # National Day
        # Add more holidays as needed
    }
    monitor.set_taiwan_holidays(taiwan_holidays)
    
    logger.info("Taiwan market monitor created with specialized configuration")
    return monitor


def setup_monitoring_pipeline(validation_engine: ValidationEngine) -> Tuple[QualityMonitor, RealtimeQualityStream]:
    """Set up complete monitoring pipeline."""
    monitor = create_taiwan_market_monitor(validation_engine)
    stream = RealtimeQualityStream(monitor)
    
    # Start monitoring
    monitor.start_monitoring()
    stream.start_stream()
    
    logger.info("Complete monitoring pipeline established")
    return monitor, stream