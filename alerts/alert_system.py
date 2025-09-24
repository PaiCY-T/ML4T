"""
Automated Alert System for ML4T Model Monitoring.

Advanced alerting system with severity-based routing, performance degradation
detection, and automated retraining triggers for Taiwan equity alpha model.

Key Features:
- Multi-channel alerting (email, Slack, SMS, webhook)
- Severity-based alert escalation
- Performance degradation detection
- Automated retraining triggers
- Alert suppression and grouping
- Taiwan market hours awareness
"""

import asyncio
import json
import logging
import smtplib
import ssl
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any, Union
import warnings
import os

import numpy as np
import pandas as pd
import requests
from scipy import stats

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels with escalation rules."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"  # Immediate intervention required


class AlertCategory(Enum):
    """Alert categories for routing and processing."""
    PERFORMANCE = "performance"
    DRIFT = "drift"
    SYSTEM = "system"
    REGULATORY = "regulatory"
    RETRAINING = "retraining"
    MARKET = "market"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


@dataclass
class AlertRule:
    """Definition of an alert rule with conditions and actions."""
    name: str
    category: AlertCategory
    severity: AlertSeverity
    metric_name: str
    condition: str  # e.g., "value > threshold", "trend_slope < -0.1"
    threshold: float
    comparison_window: int = 5  # Number of data points to consider
    cooldown_minutes: int = 30
    channels: List[NotificationChannel] = field(default_factory=list)
    business_hours_only: bool = False
    taiwan_market_hours_only: bool = False
    escalation_minutes: int = 60  # Time before escalating to next severity
    
    def evaluate(self, values: List[float], timestamps: List[datetime] = None) -> bool:
        """Evaluate if alert condition is met."""
        if not values or len(values) < self.comparison_window:
            return False
        
        recent_values = values[-self.comparison_window:]
        current_value = recent_values[-1]
        
        try:
            if self.condition == "value > threshold":
                return current_value > self.threshold
            elif self.condition == "value < threshold":
                return current_value < self.threshold
            elif self.condition == "trending_down":
                if len(recent_values) < 3:
                    return False
                slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                return slope < self.threshold
            elif self.condition == "trending_up":
                if len(recent_values) < 3:
                    return False
                slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                return slope > self.threshold
            elif self.condition == "volatility_spike":
                if len(recent_values) < 5:
                    return False
                std_current = np.std(recent_values[-3:])
                std_baseline = np.std(recent_values[:-3])
                return std_current > std_baseline * self.threshold
            else:
                logger.warning(f"Unknown condition type: {self.condition}")
                return False
                
        except Exception as e:
            logger.error(f"Alert rule evaluation error for {self.name}: {e}")
            return False


@dataclass
class Alert:
    """Alert instance with metadata and tracking."""
    id: str
    rule_name: str
    severity: AlertSeverity
    category: AlertCategory
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    channels: List[NotificationChannel]
    
    # Tracking fields
    sent_channels: Set[NotificationChannel] = field(default_factory=set)
    escalated: bool = False
    acknowledged: bool = False
    resolved: bool = False
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    impact_assessment: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'category': self.category.value,
            'title': self.title,
            'message': self.message,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'timestamp': self.timestamp.isoformat(),
            'channels': [c.value for c in self.channels],
            'sent_channels': [c.value for c in self.sent_channels],
            'escalated': self.escalated,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'context': self.context,
            'recommendation': self.recommendation,
            'impact_assessment': self.impact_assessment
        }


class NotificationHandler(ABC):
    """Abstract base class for notification handlers."""
    
    @abstractmethod
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert."""
        pass


class EmailNotificationHandler(NotificationHandler):
    """Email notification handler."""
    
    def __init__(self, smtp_config: Dict[str, str], recipients: List[str]):
        self.smtp_config = smtp_config
        self.recipients = recipients
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.smtp_config.get('from_email', 'ml4t@example.com')
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[ML4T {alert.severity.value.upper()}] {alert.title}"
            
            # Create HTML email content
            html_content = self._create_html_content(alert)
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_config['host'], self.smtp_config.get('port', 587)) as server:
                server.starttls(context=context)
                if self.smtp_config.get('username'):
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Email notification failed for alert {alert.id}: {e}")
            return False
    
    def _create_html_content(self, alert: Alert) -> str:
        """Create HTML email content."""
        severity_colors = {
            'info': '#3b82f6',
            'warning': '#f59e0b',
            'error': '#ef4444',
            'critical': '#dc2626',
            'emergency': '#991b1b'
        }
        
        color = severity_colors.get(alert.severity.value, '#6b7280')
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: {color}; color: white; padding: 15px; border-radius: 5px;">
                    <h2 style="margin: 0;">ML4T Model Alert - {alert.severity.value.upper()}</h2>
                </div>
                
                <div style="background: #f9f9f9; padding: 20px; margin: 10px 0; border-radius: 5px;">
                    <h3>{alert.title}</h3>
                    <p><strong>Message:</strong> {alert.message}</p>
                    <p><strong>Metric:</strong> {alert.metric_name}</p>
                    <p><strong>Current Value:</strong> {alert.current_value:.4f}</p>
                    <p><strong>Threshold:</strong> {alert.threshold_value:.4f}</p>
                    <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                {f'<div style="background: #e1f5fe; padding: 15px; margin: 10px 0; border-radius: 5px;"><strong>Recommendation:</strong> {alert.recommendation}</div>' if alert.recommendation else ''}
                
                {f'<div style="background: #fff3e0; padding: 15px; margin: 10px 0; border-radius: 5px;"><strong>Impact Assessment:</strong> {alert.impact_assessment}</div>' if alert.impact_assessment else ''}
                
                <div style="margin-top: 20px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 12px; color: #666;">
                    <p>Alert ID: {alert.id}</p>
                    <p>Category: {alert.category.value}</p>
                    <p>This is an automated message from the ML4T monitoring system.</p>
                </div>
            </div>
        </body>
        </html>
        """


class SlackNotificationHandler(NotificationHandler):
    """Slack notification handler."""
    
    def __init__(self, webhook_url: str, channel: str = None):
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            severity_colors = {
                'info': '#36a64f',
                'warning': '#ff9500', 
                'error': '#ff0000',
                'critical': '#8b0000',
                'emergency': '#4b0000'
            }
            
            payload = {
                "channel": self.channel,
                "username": "ML4T Monitor",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [{
                    "color": severity_colors.get(alert.severity.value, "#cccccc"),
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Metric",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": f"{alert.current_value:.4f}",
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": f"{alert.threshold_value:.4f}",
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            "short": True
                        }
                    ],
                    "footer": f"Alert ID: {alert.id}",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            if alert.recommendation:
                payload["attachments"][0]["fields"].append({
                    "title": "Recommendation",
                    "value": alert.recommendation,
                    "short": False
                })
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Slack notification failed for alert {alert.id}: {e}")
            return False


class WebhookNotificationHandler(NotificationHandler):
    """Generic webhook notification handler."""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            payload = alert.to_dict()
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Webhook notification failed for alert {alert.id}: {e}")
            return False


class PerformanceDegradationDetector:
    """Advanced performance degradation detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_stats = {}
        
        # Degradation thresholds
        self.ic_degradation_threshold = config.get('ic_degradation_threshold', 0.02)
        self.sharpe_degradation_threshold = config.get('sharpe_degradation_threshold', 0.3)
        self.consecutive_degradation_periods = config.get('consecutive_periods', 3)
        
    def update_performance(self, metric_name: str, value: float, timestamp: datetime):
        """Update performance metrics."""
        self.performance_history[metric_name].append((timestamp, value))
        
        # Update baseline statistics
        if len(self.performance_history[metric_name]) >= 50:
            values = [v for _, v in self.performance_history[metric_name]]
            self.baseline_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'quantiles': np.percentile(values, [25, 50, 75, 90, 95])
            }
    
    def detect_degradation(self, metric_name: str) -> Dict[str, Any]:
        """Detect performance degradation patterns."""
        if metric_name not in self.performance_history:
            return {'degradation_detected': False}
        
        history = list(self.performance_history[metric_name])
        if len(history) < 10:
            return {'degradation_detected': False}
        
        recent_values = [v for _, v in history[-10:]]
        baseline = self.baseline_stats.get(metric_name)
        
        if not baseline:
            return {'degradation_detected': False}
        
        # Multiple degradation detection methods
        degradation_signals = []
        
        # 1. Consecutive periods below threshold
        if metric_name == 'information_coefficient':
            threshold = baseline['mean'] - self.ic_degradation_threshold
            consecutive_below = self._count_consecutive_below_threshold(recent_values, threshold)
            if consecutive_below >= self.consecutive_degradation_periods:
                degradation_signals.append({
                    'type': 'consecutive_degradation',
                    'description': f'IC below {threshold:.4f} for {consecutive_below} consecutive periods',
                    'severity': 'high'
                })
        
        # 2. Statistical shift detection
        if len(history) >= 50:
            old_values = [v for _, v in history[-50:-10]]
            recent_values_10 = [v for _, v in history[-10:]]
            
            # T-test for mean shift
            try:
                t_stat, p_value = stats.ttest_ind(old_values, recent_values_10)
                if p_value < 0.05 and np.mean(recent_values_10) < np.mean(old_values):
                    degradation_signals.append({
                        'type': 'statistical_shift',
                        'description': f'Significant mean decrease (p={p_value:.4f})',
                        'severity': 'medium'
                    })
            except:
                pass
        
        # 3. Trend analysis
        if len(recent_values) >= 5:
            slope, _, r_value, p_value, _ = stats.linregress(range(len(recent_values)), recent_values)
            if p_value < 0.1 and slope < 0 and abs(r_value) > 0.5:
                degradation_signals.append({
                    'type': 'negative_trend',
                    'description': f'Strong negative trend (slope={slope:.6f}, R²={r_value**2:.3f})',
                    'severity': 'medium'
                })
        
        # 4. Volatility spike
        current_volatility = np.std(recent_values)
        if current_volatility > baseline['std'] * 2:
            degradation_signals.append({
                'type': 'volatility_spike',
                'description': f'Volatility spike: {current_volatility:.4f} vs baseline {baseline["std"]:.4f}',
                'severity': 'low'
            })
        
        # Determine overall degradation status
        degradation_detected = len(degradation_signals) > 0
        severity = 'critical' if any(s['severity'] == 'high' for s in degradation_signals) else 'warning'
        
        return {
            'degradation_detected': degradation_detected,
            'severity': severity,
            'signals': degradation_signals,
            'current_value': recent_values[-1] if recent_values else None,
            'baseline_mean': baseline['mean'],
            'baseline_std': baseline['std'],
            'recommendation': self._generate_degradation_recommendation(degradation_signals)
        }
    
    def _count_consecutive_below_threshold(self, values: List[float], threshold: float) -> int:
        """Count consecutive values below threshold."""
        consecutive = 0
        for value in reversed(values):
            if value < threshold:
                consecutive += 1
            else:
                break
        return consecutive
    
    def _generate_degradation_recommendation(self, signals: List[Dict[str, Any]]) -> str:
        """Generate recommendation based on degradation signals."""
        if not signals:
            return ""
        
        high_severity = any(s['severity'] == 'high' for s in signals)
        
        if high_severity:
            return "URGENT: Significant model degradation detected. Consider immediate retraining or model replacement."
        else:
            return "Model performance declining. Monitor closely and prepare for potential retraining."


class RetrainingTrigger:
    """Automated model retraining trigger system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retraining_conditions = config.get('retraining_conditions', {})
        self.last_retraining = None
        self.min_retraining_interval = timedelta(days=config.get('min_retraining_days', 7))
        
    def should_trigger_retraining(
        self,
        performance_degradation: Dict[str, Any],
        drift_scores: Dict[str, float],
        current_time: datetime = None
    ) -> Dict[str, Any]:
        """Determine if model retraining should be triggered."""
        current_time = current_time or datetime.now()
        
        # Check minimum interval
        if self.last_retraining and (current_time - self.last_retraining) < self.min_retraining_interval:
            return {
                'trigger_retraining': False,
                'reason': 'Minimum retraining interval not elapsed',
                'next_eligible': self.last_retraining + self.min_retraining_interval
            }
        
        triggers = []
        
        # Performance degradation trigger
        if performance_degradation.get('degradation_detected'):
            if performance_degradation.get('severity') == 'critical':
                triggers.append({
                    'type': 'performance_degradation',
                    'severity': 'critical',
                    'description': 'Critical performance degradation detected'
                })
            elif performance_degradation.get('severity') == 'warning':
                signals = performance_degradation.get('signals', [])
                high_severity_signals = [s for s in signals if s['severity'] == 'high']
                if len(high_severity_signals) >= 2:
                    triggers.append({
                        'type': 'multiple_degradation_signals',
                        'severity': 'high',
                        'description': f'{len(high_severity_signals)} high-severity degradation signals'
                    })
        
        # Drift-based triggers
        feature_drift = drift_scores.get('feature_drift_score', 0)
        concept_drift = drift_scores.get('concept_drift_score', 0)
        
        if feature_drift > self.retraining_conditions.get('max_feature_drift', 0.2):
            triggers.append({
                'type': 'feature_drift',
                'severity': 'high',
                'description': f'Feature drift score {feature_drift:.3f} exceeds threshold'
            })
        
        if concept_drift > self.retraining_conditions.get('max_concept_drift', 0.3):
            triggers.append({
                'type': 'concept_drift',
                'severity': 'high',
                'description': f'Concept drift score {concept_drift:.3f} exceeds threshold'
            })
        
        # Determine if retraining should be triggered
        trigger_retraining = False
        if triggers:
            critical_triggers = [t for t in triggers if t['severity'] == 'critical']
            high_triggers = [t for t in triggers if t['severity'] == 'high']
            
            if critical_triggers or len(high_triggers) >= 2:
                trigger_retraining = True
        
        result = {
            'trigger_retraining': trigger_retraining,
            'triggers': triggers,
            'recommendation': self._generate_retraining_recommendation(triggers),
            'estimated_retraining_time': self._estimate_retraining_time()
        }
        
        if trigger_retraining:
            self.last_retraining = current_time
            result['retraining_initiated'] = current_time.isoformat()
        
        return result
    
    def _generate_retraining_recommendation(self, triggers: List[Dict[str, Any]]) -> str:
        """Generate retraining recommendation."""
        if not triggers:
            return "Model performance stable, no retraining required."
        
        critical_triggers = [t for t in triggers if t['severity'] == 'critical']
        if critical_triggers:
            return "URGENT: Initiate emergency model retraining immediately."
        
        high_triggers = [t for t in triggers if t['severity'] == 'high']
        if len(high_triggers) >= 2:
            return "Recommend scheduled model retraining within 24-48 hours."
        elif high_triggers:
            return "Monitor model closely. Prepare for potential retraining."
        
        return "Minor issues detected. Continue monitoring."
    
    def _estimate_retraining_time(self) -> Dict[str, int]:
        """Estimate retraining time requirements."""
        return {
            'data_preparation_hours': 2,
            'model_training_hours': 4,
            'validation_hours': 2,
            'deployment_hours': 1,
            'total_hours': 9
        }


class AlertSystem:
    """Main alert system orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = []
        self.notification_handlers = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=100))
        
        # Components
        self.degradation_detector = PerformanceDegradationDetector(config)
        self.retraining_trigger = RetrainingTrigger(config)
        
        # Taiwan market hours
        self.taiwan_market_open = time(9, 0)  # 09:00
        self.taiwan_market_close = time(13, 30)  # 13:30
        
        # Setup notification handlers
        self._setup_notification_handlers()
        self._setup_default_rules()
    
    def _setup_notification_handlers(self):
        """Setup notification handlers from configuration."""
        if 'email' in self.config:
            self.notification_handlers[NotificationChannel.EMAIL] = EmailNotificationHandler(
                self.config['email']['smtp'],
                self.config['email']['recipients']
            )
        
        if 'slack' in self.config:
            self.notification_handlers[NotificationChannel.SLACK] = SlackNotificationHandler(
                self.config['slack']['webhook_url'],
                self.config['slack'].get('channel')
            )
        
        if 'webhook' in self.config:
            self.notification_handlers[NotificationChannel.WEBHOOK] = WebhookNotificationHandler(
                self.config['webhook']['url'],
                self.config['webhook'].get('headers')
            )
    
    def _setup_default_rules(self):
        """Setup default alert rules for Taiwan equity model."""
        # IC degradation rules
        self.add_rule(AlertRule(
            name="ic_critical_degradation",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.CRITICAL,
            metric_name="information_coefficient",
            condition="value < threshold",
            threshold=0.02,
            comparison_window=3,
            cooldown_minutes=60,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            taiwan_market_hours_only=True
        ))
        
        # Sharpe ratio degradation
        self.add_rule(AlertRule(
            name="sharpe_degradation",
            category=AlertCategory.PERFORMANCE,
            severity=AlertSeverity.WARNING,
            metric_name="sharpe_ratio",
            condition="trending_down",
            threshold=-0.1,  # Negative slope threshold
            comparison_window=7,
            cooldown_minutes=120,
            channels=[NotificationChannel.SLACK]
        ))
        
        # Feature drift alerts
        self.add_rule(AlertRule(
            name="high_feature_drift",
            category=AlertCategory.DRIFT,
            severity=AlertSeverity.WARNING,
            metric_name="feature_drift_score",
            condition="value > threshold",
            threshold=0.15,
            comparison_window=5,
            cooldown_minutes=30,
            channels=[NotificationChannel.SLACK]
        ))
        
        # System performance alerts
        self.add_rule(AlertRule(
            name="high_inference_latency",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.ERROR,
            metric_name="inference_latency_ms",
            condition="value > threshold",
            threshold=100.0,
            comparison_window=3,
            cooldown_minutes=15,
            channels=[NotificationChannel.EMAIL]
        ))
        
        # Retraining trigger alert
        self.add_rule(AlertRule(
            name="retraining_required",
            category=AlertCategory.RETRAINING,
            severity=AlertSeverity.CRITICAL,
            metric_name="retraining_trigger",
            condition="value > threshold",
            threshold=0.5,  # Trigger probability
            comparison_window=1,
            cooldown_minutes=360,  # 6 hours
            channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Alert rule added: {rule.name}")
    
    def update_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Update metric value and check alert conditions."""
        timestamp = timestamp or datetime.now()
        
        # Store in buffer
        self.metrics_buffer[metric_name].append((timestamp, value))
        
        # Update degradation detector
        if metric_name in ['information_coefficient', 'sharpe_ratio', 'hit_rate']:
            self.degradation_detector.update_performance(metric_name, value, timestamp)
        
        # Check all relevant alert rules
        for rule in self.alert_rules:
            if rule.metric_name == metric_name:
                self._check_rule(rule, timestamp)
    
    def _check_rule(self, rule: AlertRule, timestamp: datetime):
        """Check if alert rule conditions are met."""
        if rule.metric_name not in self.metrics_buffer:
            return
        
        # Check market hours constraints
        if rule.taiwan_market_hours_only and not self._is_taiwan_market_hours(timestamp):
            return
        
        # Check business hours constraints
        if rule.business_hours_only and not self._is_business_hours(timestamp):
            return
        
        # Check cooldown
        rule_key = f"{rule.name}_{rule.metric_name}"
        if rule_key in self.active_alerts:
            last_alert_time = self.active_alerts[rule_key].timestamp
            if (timestamp - last_alert_time).total_seconds() < rule.cooldown_minutes * 60:
                return
        
        # Get metric values
        metric_data = list(self.metrics_buffer[rule.metric_name])
        values = [v for _, v in metric_data]
        timestamps = [t for t, _ in metric_data]
        
        # Evaluate rule condition
        if rule.evaluate(values, timestamps):
            self._create_alert(rule, values[-1], timestamp)
    
    def _create_alert(self, rule: AlertRule, current_value: float, timestamp: datetime):
        """Create and process a new alert."""
        alert_id = f"{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Generate context-specific message and recommendation
        message, recommendation, impact = self._generate_alert_content(rule, current_value)
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            category=rule.category,
            title=f"{rule.category.value.title()} Alert: {rule.name.replace('_', ' ').title()}",
            message=message,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=rule.threshold,
            timestamp=timestamp,
            channels=rule.channels.copy(),
            recommendation=recommendation,
            impact_assessment=impact
        )
        
        # Store alert
        rule_key = f"{rule.name}_{rule.metric_name}"
        self.active_alerts[rule_key] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        asyncio.create_task(self._send_notifications(alert))
        
        logger.warning(f"Alert created: {alert.title} (ID: {alert_id})")
        
        return alert
    
    def _generate_alert_content(self, rule: AlertRule, current_value: float) -> tuple:
        """Generate contextual alert message, recommendation, and impact assessment."""
        if rule.metric_name == "information_coefficient":
            message = f"Information Coefficient dropped to {current_value:.4f}, below threshold of {rule.threshold:.4f}"
            recommendation = "Consider immediate model performance review and potential retraining"
            impact = "Model predictive power significantly reduced, affecting trading strategy performance"
            
        elif rule.metric_name == "sharpe_ratio":
            message = f"Sharpe ratio showing negative trend, current value {current_value:.2f}"
            recommendation = "Review risk management parameters and model exposure"
            impact = "Risk-adjusted returns declining, portfolio efficiency compromised"
            
        elif rule.metric_name == "feature_drift_score":
            message = f"Feature drift score elevated at {current_value:.3f}, exceeding {rule.threshold:.3f} threshold"
            recommendation = "Investigate feature data quality and market regime changes"
            impact = "Model inputs changing, predictions may become less reliable"
            
        elif rule.metric_name == "inference_latency_ms":
            message = f"Inference latency spiked to {current_value:.0f}ms, above {rule.threshold:.0f}ms SLA"
            recommendation = "Check system resources and optimize model inference pipeline"
            impact = "Trading latency increased, potential missed opportunities"
            
        else:
            message = f"Metric {rule.metric_name} value {current_value:.4f} triggered alert (threshold: {rule.threshold:.4f})"
            recommendation = "Review metric behavior and investigate underlying causes"
            impact = "System behavior outside expected parameters"
        
        return message, recommendation, impact
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels."""
        for channel in alert.channels:
            if channel in self.notification_handlers:
                try:
                    success = await self.notification_handlers[channel].send_notification(alert)
                    if success:
                        alert.sent_channels.add(channel)
                except Exception as e:
                    logger.error(f"Notification failed for channel {channel.value}: {e}")
        
        logger.info(f"Alert {alert.id} sent to channels: {[c.value for c in alert.sent_channels]}")
    
    def _is_taiwan_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during Taiwan market hours."""
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        current_time = timestamp.time()
        return self.taiwan_market_open <= current_time <= self.taiwan_market_close
    
    def _is_business_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during business hours (9 AM - 6 PM, weekdays)."""
        if timestamp.weekday() >= 5:
            return False
        
        current_time = timestamp.time()
        return time(9, 0) <= current_time <= time(18, 0)
    
    def check_retraining_triggers(self, drift_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """Check if model retraining should be triggered."""
        # Get performance degradation analysis
        degradation_results = {}
        for metric_name in ['information_coefficient', 'sharpe_ratio']:
            if metric_name in self.metrics_buffer:
                degradation_results[metric_name] = self.degradation_detector.detect_degradation(metric_name)
        
        # Combine degradation signals
        overall_degradation = {
            'degradation_detected': any(r.get('degradation_detected', False) for r in degradation_results.values()),
            'severity': 'critical' if any(r.get('severity') == 'critical' for r in degradation_results.values()) else 'warning',
            'signals': []
        }
        
        for metric, result in degradation_results.items():
            if result.get('degradation_detected'):
                for signal in result.get('signals', []):
                    signal['metric'] = metric
                    overall_degradation['signals'].append(signal)
        
        # Check retraining conditions
        drift_scores = drift_scores or {}
        retraining_result = self.retraining_trigger.should_trigger_retraining(
            overall_degradation, drift_scores
        )
        
        # Create retraining alert if needed
        if retraining_result.get('trigger_retraining'):
            self.update_metric('retraining_trigger', 1.0)
        
        return retraining_result
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alert_history if alert.timestamp > cutoff]
        
        return {
            'total_alerts': len(recent_alerts),
            'alerts_by_severity': {
                severity.value: sum(1 for a in recent_alerts if a.severity == severity)
                for severity in AlertSeverity
            },
            'alerts_by_category': {
                category.value: sum(1 for a in recent_alerts if a.category == category)
                for category in AlertCategory
            },
            'active_alerts': len(self.active_alerts),
            'recent_alerts': [alert.to_dict() for alert in recent_alerts[-10:]]  # Last 10
        }
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        for rule_key, alert in self.active_alerts.items():
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.context['acknowledged_by'] = acknowledged_by
                alert.context['acknowledged_at'] = datetime.now().isoformat()
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        for rule_key, alert in self.active_alerts.items():
            if alert.id == alert_id:
                alert.resolved = True
                alert.context['resolved_by'] = resolved_by
                alert.context['resolved_at'] = datetime.now().isoformat()
                del self.active_alerts[rule_key]
                logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
        return False


def create_production_alert_system() -> AlertSystem:
    """Create production-ready alert system with Taiwan market configuration."""
    
    config = {
        # Email configuration
        'email': {
            'smtp': {
                'host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
                'port': int(os.getenv('SMTP_PORT', '587')),
                'username': os.getenv('SMTP_USERNAME', ''),
                'password': os.getenv('SMTP_PASSWORD', ''),
                'from_email': os.getenv('FROM_EMAIL', 'ml4t@example.com')
            },
            'recipients': os.getenv('ALERT_RECIPIENTS', '').split(',')
        },
        
        # Slack configuration
        'slack': {
            'webhook_url': os.getenv('SLACK_WEBHOOK_URL', ''),
            'channel': os.getenv('SLACK_CHANNEL', '#ml4t-alerts')
        },
        
        # Webhook configuration
        'webhook': {
            'url': os.getenv('WEBHOOK_URL', ''),
            'headers': {'Authorization': f"Bearer {os.getenv('WEBHOOK_TOKEN', '')}"}
        },
        
        # Performance degradation thresholds
        'ic_degradation_threshold': 0.02,
        'sharpe_degradation_threshold': 0.3,
        'consecutive_periods': 3,
        
        # Retraining configuration
        'retraining_conditions': {
            'max_feature_drift': 0.2,
            'max_concept_drift': 0.3
        },
        'min_retraining_days': 7
    }
    
    alert_system = AlertSystem(config)
    logger.info("Production alert system created with Taiwan market configuration")
    
    return alert_system


# Demo function
async def demo_alert_system():
    """Demonstrate alert system functionality."""
    print("ML4T Alert System Demo")
    
    # Create demo alert system
    config = {
        'email': {'smtp': {}, 'recipients': []},
        'ic_degradation_threshold': 0.03,
        'consecutive_periods': 3
    }
    
    alert_system = AlertSystem(config)
    
    # Simulate metric updates that trigger alerts
    now = datetime.now()
    
    # Simulate IC degradation
    print("\nSimulating IC degradation...")
    for i in range(5):
        ic_value = 0.04 - (i * 0.008)  # Decreasing IC
        alert_system.update_metric('information_coefficient', ic_value, now + timedelta(minutes=i*10))
        print(f"  IC update {i+1}: {ic_value:.4f}")
    
    # Check for retraining triggers
    print("\nChecking retraining triggers...")
    retraining_result = alert_system.check_retraining_triggers({'feature_drift_score': 0.25})
    print(f"  Retraining required: {retraining_result.get('trigger_retraining', False)}")
    
    # Get alert summary
    summary = alert_system.get_alert_summary()
    print(f"\nAlert Summary:")
    print(f"  Total alerts: {summary['total_alerts']}")
    print(f"  Active alerts: {summary['active_alerts']}")
    
    print("\nAlert system demo completed")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_alert_system())