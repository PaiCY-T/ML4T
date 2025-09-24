"""
Model Health Monitoring and Alerting System.

Comprehensive monitoring system for LightGBM alpha model in production:
- Real-time performance tracking
- Feature and concept drift detection  
- Automated alerting and notifications
- Model degradation detection
- Taiwan market specific monitoring
"""

import asyncio
import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ..models.lightgbm_alpha import LightGBMAlphaModel
from ..inference.realtime import RealtimePredictor, InferenceMetrics
from ..backtesting.metrics.performance import PerformanceMetrics

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Model health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class MonitoringConfig:
    """Configuration for model health monitoring."""
    
    # Performance thresholds
    min_ic_threshold: float = 0.03  # Minimum acceptable IC
    max_rmse_threshold: float = 0.15  # Maximum acceptable RMSE
    min_sharpe_threshold: float = 1.0  # Minimum Sharpe ratio
    max_drawdown_threshold: float = 0.20  # Maximum drawdown
    
    # Drift detection thresholds
    feature_drift_threshold: float = 0.05  # KL divergence threshold
    prediction_drift_threshold: float = 0.10  # Prediction distribution shift
    concept_drift_threshold: float = 0.15  # Performance degradation threshold
    
    # Monitoring windows
    performance_window_days: int = 30  # Rolling window for performance
    drift_detection_window: int = 1000  # Number of samples for drift detection
    alert_frequency_minutes: int = 30  # Minimum time between similar alerts
    
    # Taiwan market specifics
    market_regime_detection: bool = True
    sector_rotation_monitoring: bool = True
    foreign_flow_tracking: bool = True
    
    # System health
    max_inference_latency_ms: float = 100.0
    min_prediction_rate: float = 0.95  # Success rate threshold
    max_memory_usage_gb: float = 16.0
    
    # Alerting
    enable_email_alerts: bool = True
    enable_slack_alerts: bool = False
    alert_recipients: List[str] = field(default_factory=list)
    
    # Logging
    log_level: str = "INFO"
    metrics_retention_days: int = 90


@dataclass
class Alert:
    """Alert message container."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'recommendation': self.recommendation
        }


@dataclass
class HealthMetrics:
    """Comprehensive model health metrics."""
    
    # Performance metrics
    current_ic: float = 0.0
    rolling_ic_30d: float = 0.0
    current_sharpe: float = 0.0
    rolling_sharpe_30d: float = 0.0
    current_rmse: float = 0.0
    max_drawdown: float = 0.0
    
    # Drift metrics
    feature_drift_score: float = 0.0
    prediction_drift_score: float = 0.0
    concept_drift_score: float = 0.0
    
    # System metrics
    avg_inference_latency_ms: float = 0.0
    prediction_success_rate: float = 0.0
    memory_usage_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Market regime metrics
    current_market_regime: str = "unknown"
    regime_stability: float = 0.0
    sector_rotation_strength: float = 0.0
    
    # Health status
    overall_status: HealthStatus = HealthStatus.HEALTHY
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        score_components = []
        
        # Performance score (40% weight)
        performance_score = 0.0
        if self.current_ic > 0.05:
            performance_score += 25
        elif self.current_ic > 0.03:
            performance_score += 15
        
        if self.current_sharpe > 2.0:
            performance_score += 25
        elif self.current_sharpe > 1.0:
            performance_score += 15
        
        if self.max_drawdown < 0.10:
            performance_score += 25
        elif self.max_drawdown < 0.20:
            performance_score += 15
        
        if self.current_rmse < 0.10:
            performance_score += 25
        elif self.current_rmse < 0.15:
            performance_score += 15
        
        score_components.append(min(performance_score, 40))
        
        # System health score (30% weight)
        system_score = 0.0
        if self.avg_inference_latency_ms < 50:
            system_score += 10
        elif self.avg_inference_latency_ms < 100:
            system_score += 5
        
        if self.prediction_success_rate > 0.99:
            system_score += 10
        elif self.prediction_success_rate > 0.95:
            system_score += 5
        
        if self.memory_usage_gb < 8:
            system_score += 10
        elif self.memory_usage_gb < 16:
            system_score += 5
        
        score_components.append(min(system_score, 30))
        
        # Drift score (30% weight) - lower drift is better
        drift_score = 30.0
        if self.feature_drift_score > 0.10:
            drift_score -= 10
        if self.prediction_drift_score > 0.15:
            drift_score -= 10  
        if self.concept_drift_score > 0.20:
            drift_score -= 10
            
        score_components.append(max(drift_score, 0))
        
        return sum(score_components)


class DriftDetector:
    """Statistical drift detection for model features and predictions."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.reference_distributions = {}
        self.sample_buffer = defaultdict(lambda: deque(maxlen=config.drift_detection_window))
        
    def update_reference_distribution(
        self, 
        name: str, 
        data: np.ndarray
    ) -> None:
        """Update reference distribution for drift detection."""
        self.reference_distributions[name] = {
            'mean': np.mean(data),
            'std': np.std(data),
            'quantiles': np.percentile(data, [25, 50, 75]),
            'samples': data.copy() if len(data) <= 1000 else np.random.choice(data, 1000, replace=False)
        }
        logger.debug(f"Updated reference distribution for {name}")
    
    def detect_feature_drift(
        self, 
        feature_name: str, 
        current_data: np.ndarray
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect feature drift using multiple statistical tests.
        
        Args:
            feature_name: Name of feature to test
            current_data: Current feature data
            
        Returns:
            Tuple of (is_drifted, drift_score, test_details)
        """
        if feature_name not in self.reference_distributions:
            logger.warning(f"No reference distribution for feature {feature_name}")
            return False, 0.0, {}
        
        reference = self.reference_distributions[feature_name]
        test_results = {}
        
        try:
            # Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.ks_2samp(
                reference['samples'], 
                current_data
            )
            test_results['ks_statistic'] = ks_statistic
            test_results['ks_p_value'] = ks_p_value
            
            # Wasserstein distance (Earth Mover's Distance) 
            wasserstein_distance = stats.wasserstein_distance(
                reference['samples'],
                current_data
            )
            test_results['wasserstein_distance'] = wasserstein_distance
            
            # Population Stability Index (PSI)
            psi = self._calculate_psi(reference['samples'], current_data)
            test_results['psi'] = psi
            
            # Combined drift score
            drift_score = (
                ks_statistic * 0.3 + 
                min(wasserstein_distance / (reference['std'] + 1e-8), 1.0) * 0.4 +
                min(psi / 0.25, 1.0) * 0.3
            )
            
            is_drifted = (
                ks_p_value < 0.05 or  # Significant KS test
                psi > 0.25 or  # High PSI indicates drift
                drift_score > self.config.feature_drift_threshold
            )
            
            logger.debug(f"Feature drift test for {feature_name}: score={drift_score:.4f}, drifted={is_drifted}")
            
            return is_drifted, drift_score, test_results
            
        except Exception as e:
            logger.error(f"Feature drift detection failed for {feature_name}: {e}")
            return False, 0.0, {"error": str(e)}
    
    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        reference_predictions: Optional[np.ndarray] = None
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Detect drift in model predictions."""
        if reference_predictions is None:
            if 'predictions' not in self.reference_distributions:
                logger.warning("No reference predictions available")
                return False, 0.0, {}
            reference_predictions = self.reference_distributions['predictions']['samples']
        
        try:
            # Statistical tests on prediction distributions
            ks_statistic, ks_p_value = stats.ks_2samp(
                reference_predictions,
                current_predictions
            )
            
            # Calculate distribution moments
            ref_mean, ref_std = np.mean(reference_predictions), np.std(reference_predictions)
            cur_mean, cur_std = np.mean(current_predictions), np.std(current_predictions)
            
            mean_shift = abs(cur_mean - ref_mean) / (ref_std + 1e-8)
            std_ratio = cur_std / (ref_std + 1e-8)
            
            # Combined drift score
            drift_score = (
                ks_statistic * 0.4 +
                min(mean_shift / 2.0, 1.0) * 0.3 +
                min(abs(std_ratio - 1.0), 1.0) * 0.3
            )
            
            is_drifted = (
                ks_p_value < 0.01 or
                mean_shift > 1.0 or  # Mean shifted by more than 1 std
                std_ratio > 2.0 or std_ratio < 0.5 or  # Variance changed significantly
                drift_score > self.config.prediction_drift_threshold
            )
            
            test_results = {
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'mean_shift': mean_shift,
                'std_ratio': std_ratio,
                'ref_mean': ref_mean,
                'cur_mean': cur_mean,
                'ref_std': ref_std,
                'cur_std': cur_std
            }
            
            return is_drifted, drift_score, test_results
            
        except Exception as e:
            logger.error(f"Prediction drift detection failed: {e}")
            return False, 0.0, {"error": str(e)}
    
    def _calculate_psi(
        self, 
        reference: np.ndarray, 
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        try:
            # Create bins based on reference distribution
            bin_edges = np.histogram_bin_edges(reference, bins=bins)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
            cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)
            
            # Normalize to get proportions
            ref_prop = ref_hist / np.sum(ref_hist)
            cur_prop = cur_hist / np.sum(cur_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            ref_prop = np.maximum(ref_prop, epsilon)
            cur_prop = np.maximum(cur_prop, epsilon)
            
            # Calculate PSI
            psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
            
            return psi
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0


class PerformanceTracker:
    """Track model performance metrics over time."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.performance_history = defaultdict(list)
        self.returns_buffer = deque(maxlen=config.performance_window_days * 252)  # Daily returns
        self.predictions_buffer = deque(maxlen=10000)
        self.actuals_buffer = deque(maxlen=10000)
        
    def update_performance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        returns: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None
    ) -> HealthMetrics:
        """Update performance metrics with new data."""
        timestamp = timestamp or datetime.now()
        
        # Store in buffers
        self.predictions_buffer.extend(predictions)
        self.actuals_buffer.extend(actuals)
        if returns is not None:
            self.returns_buffer.extend(returns)
        
        # Calculate current metrics
        health_metrics = HealthMetrics()
        
        try:
            # Regression metrics
            if len(self.predictions_buffer) > 0 and len(self.actuals_buffer) > 0:
                pred_array = np.array(self.predictions_buffer)
                actual_array = np.array(self.actuals_buffer)
                
                # Ensure same length
                min_len = min(len(pred_array), len(actual_array))
                pred_array = pred_array[-min_len:]
                actual_array = actual_array[-min_len:]
                
                health_metrics.current_rmse = np.sqrt(mean_squared_error(actual_array, pred_array))
                health_metrics.current_ic = np.corrcoef(pred_array, actual_array)[0, 1]
                
                # Rolling metrics
                if min_len >= 30:  # At least 30 observations
                    recent_pred = pred_array[-30:]
                    recent_actual = actual_array[-30:]
                    health_metrics.rolling_ic_30d = np.corrcoef(recent_pred, recent_actual)[0, 1]
            
            # Performance metrics from returns
            if len(self.returns_buffer) > 0:
                returns_array = np.array(self.returns_buffer)
                
                # Sharpe ratio (assuming daily returns)
                if len(returns_array) >= 252:  # At least 1 year of data
                    annual_return = np.mean(returns_array) * 252
                    annual_vol = np.std(returns_array) * np.sqrt(252)
                    health_metrics.current_sharpe = annual_return / (annual_vol + 1e-8)
                    
                    # Rolling Sharpe
                    recent_returns = returns_array[-252:]  # Last year
                    recent_annual_return = np.mean(recent_returns) * 252
                    recent_annual_vol = np.std(recent_returns) * np.sqrt(252)
                    health_metrics.rolling_sharpe_30d = recent_annual_return / (recent_annual_vol + 1e-8)
                
                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                health_metrics.max_drawdown = abs(np.min(drawdown))
            
            # Update timestamp
            health_metrics.last_updated = timestamp
            
            # Store in history
            self.performance_history['ic'].append((timestamp, health_metrics.current_ic))
            self.performance_history['sharpe'].append((timestamp, health_metrics.current_sharpe))
            self.performance_history['rmse'].append((timestamp, health_metrics.current_rmse))
            self.performance_history['drawdown'].append((timestamp, health_metrics.max_drawdown))
            
            # Trim history
            cutoff_date = timestamp - timedelta(days=self.config.metrics_retention_days)
            for metric_name in self.performance_history:
                self.performance_history[metric_name] = [
                    (ts, value) for ts, value in self.performance_history[metric_name]
                    if ts > cutoff_date
                ]
            
            logger.debug(f"Performance updated: IC={health_metrics.current_ic:.4f}, "
                        f"Sharpe={health_metrics.current_sharpe:.2f}, "
                        f"RMSE={health_metrics.current_rmse:.4f}")
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Performance tracking update failed: {e}")
            health_metrics.last_updated = timestamp
            return health_metrics
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, Dict[str, float]]:
        """Get performance trends over specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        trends = {}
        
        for metric_name, history in self.performance_history.items():
            recent_history = [(ts, val) for ts, val in history if ts > cutoff_date]
            
            if len(recent_history) >= 2:
                values = [val for _, val in recent_history]
                
                # Calculate trend
                if len(values) > 1:
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    trends[metric_name] = {
                        'current_value': values[-1],
                        'trend_slope': slope,
                        'trend_r_squared': r_value**2,
                        'trend_significance': p_value,
                        'mean_value': np.mean(values),
                        'std_value': np.std(values),
                        'min_value': np.min(values),
                        'max_value': np.max(values)
                    }
        
        return trends


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.recent_alerts = deque(maxlen=1000)
        self.alert_counts = defaultdict(int)
        self.last_alert_time = defaultdict(lambda: datetime.min)
        
        # Email configuration (would be loaded from environment)
        self.smtp_config = {
            'host': 'smtp.gmail.com',
            'port': 587,
            'username': '',  # Load from environment
            'password': '',  # Load from environment
        }
    
    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        recommendation: str = ""
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            alert_id=f"{metric_name}_{int(time.time())}",
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            recommendation=recommendation
        )
        
        # Check if we should suppress similar recent alerts
        if self._should_suppress_alert(alert):
            logger.debug(f"Suppressed similar alert for {metric_name}")
            return alert
        
        # Store alert
        self.recent_alerts.append(alert)
        self.alert_counts[metric_name] += 1
        self.last_alert_time[metric_name] = alert.timestamp
        
        # Send notifications
        try:
            if self.config.enable_email_alerts and self.config.alert_recipients:
                self._send_email_alert(alert)
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
        
        logger.warning(f"ALERT [{level.value.upper()}] {title}: {message}")
        
        return alert
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if similar alert was sent recently."""
        time_threshold = timedelta(minutes=self.config.alert_frequency_minutes)
        last_time = self.last_alert_time[alert.metric_name]
        
        return (alert.timestamp - last_time) < time_threshold
    
    def _send_email_alert(self, alert: Alert) -> None:
        """Send email alert notification."""
        if not self.smtp_config['username']:
            logger.warning("Email alerts configured but no SMTP credentials")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['username']
            msg['To'] = ', '.join(self.config.alert_recipients)
            msg['Subject'] = f"[ML4T Alert - {alert.level.value.upper()}] {alert.title}"
            
            body = f"""
            ML4T Model Health Alert
            
            Alert Level: {alert.level.value.upper()}
            Metric: {alert.metric_name}
            Current Value: {alert.current_value:.4f}
            Threshold: {alert.threshold_value:.4f}
            Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Message: {alert.message}
            
            Recommendation: {alert.recommendation}
            
            Alert ID: {alert.alert_id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
            server.starttls()
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from recent hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.recent_alerts if alert.timestamp > cutoff]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        recent_alerts = self.get_recent_alerts(24)
        
        return {
            'total_alerts_24h': len(recent_alerts),
            'alerts_by_level': {
                level.value: sum(1 for a in recent_alerts if a.level == level)
                for level in AlertLevel
            },
            'top_alert_metrics': dict(sorted(
                self.alert_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),
            'last_alert_time': max([a.timestamp for a in recent_alerts]) if recent_alerts else None
        }


class ModelHealthMonitor:
    """
    Comprehensive model health monitoring system.
    
    Integrates performance tracking, drift detection, and alerting
    for production LightGBM alpha model.
    """
    
    def __init__(
        self,
        model: LightGBMAlphaModel,
        predictor: Optional[RealtimePredictor] = None,
        config: Optional[MonitoringConfig] = None
    ):
        self.model = model
        self.predictor = predictor
        self.config = config or MonitoringConfig()
        
        # Initialize components
        self.drift_detector = DriftDetector(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Current health state
        self.current_health = HealthMetrics()
        self.monitoring_active = False
        
        logger.info(f"ModelHealthMonitor initialized with config: {self.config}")
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        self.monitoring_active = True
        logger.info("Model health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False
        logger.info("Model health monitoring stopped")
    
    def update_health(
        self,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
        features: Optional[pd.DataFrame] = None,
        returns: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None
    ) -> HealthMetrics:
        """
        Update comprehensive health metrics.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            features: Feature data for drift detection
            returns: Portfolio returns
            timestamp: Timestamp for metrics
            
        Returns:
            Updated health metrics
        """
        if not self.monitoring_active:
            logger.warning("Monitoring not active, skipping health update")
            return self.current_health
        
        timestamp = timestamp or datetime.now()
        
        try:
            # Update performance metrics
            if predictions is not None and actuals is not None:
                self.current_health = self.performance_tracker.update_performance(
                    predictions, actuals, returns, timestamp
                )
            
            # Update inference metrics from predictor
            if self.predictor:
                inference_metrics = self.predictor.get_performance_summary()
                self.current_health.avg_inference_latency_ms = inference_metrics.get('avg_latency_ms', 0.0)
                self.current_health.prediction_success_rate = inference_metrics.get('success_rate', 0.0)
            
            # Detect feature drift
            drift_scores = []
            if features is not None:
                for column in features.columns:
                    try:
                        is_drifted, drift_score, _ = self.drift_detector.detect_feature_drift(
                            column, features[column].values
                        )
                        drift_scores.append(drift_score)
                        
                        if is_drifted:
                            self._create_drift_alert(column, drift_score)
                    
                    except Exception as e:
                        logger.warning(f"Drift detection failed for {column}: {e}")
            
            self.current_health.feature_drift_score = np.mean(drift_scores) if drift_scores else 0.0
            
            # Detect prediction drift
            if predictions is not None:
                try:
                    is_pred_drifted, pred_drift_score, _ = self.drift_detector.detect_prediction_drift(predictions)
                    self.current_health.prediction_drift_score = pred_drift_score
                    
                    if is_pred_drifted:
                        self._create_prediction_drift_alert(pred_drift_score)
                        
                except Exception as e:
                    logger.warning(f"Prediction drift detection failed: {e}")
            
            # Update system health
            try:
                import psutil
                process = psutil.Process()
                self.current_health.memory_usage_gb = process.memory_info().rss / (1024**3)
                self.current_health.cpu_usage_percent = process.cpu_percent()
            except ImportError:
                logger.debug("psutil not available for system metrics")
            
            # Determine overall health status
            self._update_health_status()
            
            # Check thresholds and create alerts
            self._check_performance_thresholds()
            
            self.current_health.last_updated = timestamp
            
            logger.debug(f"Health updated: status={self.current_health.overall_status.value}, "
                        f"score={self.current_health.get_health_score():.1f}")
            
            return self.current_health
            
        except Exception as e:
            logger.error(f"Health update failed: {e}")
            self.current_health.overall_status = HealthStatus.CRITICAL
            return self.current_health
    
    def _update_health_status(self) -> None:
        """Update overall health status based on metrics."""
        health_score = self.current_health.get_health_score()
        
        if health_score >= 80:
            self.current_health.overall_status = HealthStatus.HEALTHY
        elif health_score >= 60:
            self.current_health.overall_status = HealthStatus.WARNING
        elif health_score >= 40:
            self.current_health.overall_status = HealthStatus.DEGRADED
        else:
            self.current_health.overall_status = HealthStatus.CRITICAL
    
    def _check_performance_thresholds(self) -> None:
        """Check performance metrics against thresholds and create alerts."""
        metrics = self.current_health
        
        # IC threshold check
        if metrics.current_ic < self.config.min_ic_threshold:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                title="Low Information Coefficient",
                message=f"Current IC ({metrics.current_ic:.4f}) below threshold ({self.config.min_ic_threshold:.4f})",
                metric_name="information_coefficient",
                current_value=metrics.current_ic,
                threshold_value=self.config.min_ic_threshold,
                recommendation="Consider retraining model or reviewing feature engineering"
            )
        
        # RMSE threshold check
        if metrics.current_rmse > self.config.max_rmse_threshold:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                title="High RMSE",
                message=f"Current RMSE ({metrics.current_rmse:.4f}) above threshold ({self.config.max_rmse_threshold:.4f})",
                metric_name="rmse",
                current_value=metrics.current_rmse,
                threshold_value=self.config.max_rmse_threshold,
                recommendation="Model accuracy degraded, consider retraining"
            )
        
        # Sharpe ratio check
        if metrics.current_sharpe < self.config.min_sharpe_threshold:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                title="Low Sharpe Ratio",
                message=f"Current Sharpe ({metrics.current_sharpe:.2f}) below threshold ({self.config.min_sharpe_threshold:.2f})",
                metric_name="sharpe_ratio",
                current_value=metrics.current_sharpe,
                threshold_value=self.config.min_sharpe_threshold,
                recommendation="Risk-adjusted performance declining"
            )
        
        # Drawdown check
        if metrics.max_drawdown > self.config.max_drawdown_threshold:
            self.alert_manager.create_alert(
                level=AlertLevel.ERROR,
                title="High Drawdown",
                message=f"Max drawdown ({metrics.max_drawdown:.2%}) exceeds threshold ({self.config.max_drawdown_threshold:.2%})",
                metric_name="max_drawdown",
                current_value=metrics.max_drawdown,
                threshold_value=self.config.max_drawdown_threshold,
                recommendation="Significant losses detected, review risk management"
            )
        
        # Inference latency check
        if metrics.avg_inference_latency_ms > self.config.max_inference_latency_ms:
            self.alert_manager.create_alert(
                level=AlertLevel.WARNING,
                title="High Inference Latency",
                message=f"Average latency ({metrics.avg_inference_latency_ms:.1f}ms) above SLA ({self.config.max_inference_latency_ms:.1f}ms)",
                metric_name="inference_latency",
                current_value=metrics.avg_inference_latency_ms,
                threshold_value=self.config.max_inference_latency_ms,
                recommendation="Optimize model or infrastructure for better performance"
            )
    
    def _create_drift_alert(self, feature_name: str, drift_score: float) -> None:
        """Create alert for feature drift."""
        self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title=f"Feature Drift Detected: {feature_name}",
            message=f"Feature {feature_name} shows significant drift (score: {drift_score:.4f})",
            metric_name=f"feature_drift_{feature_name}",
            current_value=drift_score,
            threshold_value=self.config.feature_drift_threshold,
            recommendation="Monitor feature stability and consider retraining if drift persists"
        )
    
    def _create_prediction_drift_alert(self, drift_score: float) -> None:
        """Create alert for prediction drift."""
        self.alert_manager.create_alert(
            level=AlertLevel.WARNING,
            title="Prediction Distribution Drift",
            message=f"Model predictions show distribution shift (score: {drift_score:.4f})",
            metric_name="prediction_drift",
            current_value=drift_score,
            threshold_value=self.config.prediction_drift_threshold,
            recommendation="Model output patterns changed, investigate market regime shifts"
        )
    
    def setup_reference_data(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray
    ) -> None:
        """Set up reference distributions for drift detection."""
        logger.info("Setting up reference distributions for drift detection")
        
        # Set up feature reference distributions
        for column in features.columns:
            self.drift_detector.update_reference_distribution(
                column, features[column].dropna().values
            )
        
        # Set up prediction reference distribution  
        self.drift_detector.update_reference_distribution(
            'predictions', predictions
        )
        
        logger.info(f"Reference distributions set up for {len(features.columns)} features and predictions")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': self.current_health.overall_status.value,
            'health_score': self.current_health.get_health_score(),
            'metrics': {
                'performance': {
                    'current_ic': self.current_health.current_ic,
                    'rolling_ic_30d': self.current_health.rolling_ic_30d,
                    'current_sharpe': self.current_health.current_sharpe,
                    'current_rmse': self.current_health.current_rmse,
                    'max_drawdown': self.current_health.max_drawdown
                },
                'drift': {
                    'feature_drift_score': self.current_health.feature_drift_score,
                    'prediction_drift_score': self.current_health.prediction_drift_score
                },
                'system': {
                    'avg_inference_latency_ms': self.current_health.avg_inference_latency_ms,
                    'prediction_success_rate': self.current_health.prediction_success_rate,
                    'memory_usage_gb': self.current_health.memory_usage_gb,
                    'cpu_usage_percent': self.current_health.cpu_usage_percent
                }
            },
            'alerts': self.alert_manager.get_alert_summary(),
            'performance_trends': self.performance_tracker.get_performance_trends(),
            'monitoring_active': self.monitoring_active
        }
    
    def export_metrics(self, filepath: Path) -> None:
        """Export health metrics to file."""
        report = self.get_health_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Health metrics exported to {filepath}")


# Utility functions for Taiwan market specific monitoring

def detect_market_regime_change(
    returns: np.ndarray,
    window: int = 252
) -> Tuple[str, float]:
    """
    Detect market regime changes for Taiwan market.
    
    Args:
        returns: Daily returns array
        window: Lookback window for regime detection
        
    Returns:
        Tuple of (regime, confidence_score)
    """
    if len(returns) < window:
        return "unknown", 0.0
    
    recent_returns = returns[-window:]
    
    # Simple regime classification based on volatility and trend
    volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
    mean_return = np.mean(recent_returns) * 252  # Annualized
    
    # Taiwan market specific thresholds
    if volatility < 0.15 and mean_return > 0.05:
        return "low_vol_bull", 0.8
    elif volatility < 0.15 and mean_return < -0.05:
        return "low_vol_bear", 0.8
    elif volatility > 0.30 and mean_return > 0.05:
        return "high_vol_bull", 0.7
    elif volatility > 0.30 and mean_return < -0.05:
        return "high_vol_bear", 0.7
    elif volatility > 0.25:
        return "high_volatility", 0.6
    else:
        return "normal", 0.5


async def run_monitoring_loop(
    monitor: ModelHealthMonitor,
    update_interval_seconds: int = 300
) -> None:
    """
    Run continuous monitoring loop.
    
    Args:
        monitor: Model health monitor instance
        update_interval_seconds: Update frequency
    """
    logger.info(f"Starting monitoring loop with {update_interval_seconds}s interval")
    
    monitor.start_monitoring()
    
    try:
        while monitor.monitoring_active:
            # In production, this would:
            # 1. Fetch latest predictions and actuals from database
            # 2. Get recent feature data
            # 3. Calculate returns from portfolio
            # 4. Update health metrics
            
            logger.debug("Monitoring cycle - would fetch live data in production")
            
            # For demo, just log current status
            if monitor.current_health.last_updated:
                age_minutes = (datetime.now() - monitor.current_health.last_updated).total_seconds() / 60
                logger.info(f"Health status: {monitor.current_health.overall_status.value} "
                           f"(last updated {age_minutes:.1f} minutes ago)")
            
            await asyncio.sleep(update_interval_seconds)
            
    except KeyboardInterrupt:
        logger.info("Monitoring loop interrupted")
    finally:
        monitor.stop_monitoring()
        logger.info("Monitoring loop stopped")


# Demo and testing functions
def demo_model_monitoring():
    """Demonstration of model health monitoring."""
    print("Model health monitoring demo")
    
    config = MonitoringConfig(
        min_ic_threshold=0.03,
        max_rmse_threshold=0.15,
        enable_email_alerts=False
    )
    
    print(f"Demo config: {config}")
    print("In actual usage:")
    print("1. Initialize with trained model and predictor")
    print("2. Set up reference distributions")
    print("3. Call update_health() regularly with live data")
    print("4. Monitor alerts and health status")


if __name__ == "__main__":
    demo_model_monitoring()