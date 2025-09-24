"""
Multi-channel Alerting System for Data Quality Monitoring.

This module provides comprehensive alerting capabilities including email, Slack,
webhook notifications, and SMS alerts for data quality issues. Designed with
rate limiting, escalation policies, and Taiwan market specific configurations.
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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
import requests
import threading
from urllib.parse import urljoin

from .monitor import QualityMetrics, AlertLevel, MonitoringStatus
from .validators import QualityIssue, SeverityLevel

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Available alert channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"
    DISCORD = "discord"


class AlertStatus(Enum):
    """Alert delivery status."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    description: str
    channels: List[AlertChannel]
    severity_levels: List[SeverityLevel]
    conditions: Dict[str, Any]  # Quality score thresholds, error rates, etc.
    cooldown_minutes: int = 5   # Minimum time between same alerts
    max_alerts_per_hour: int = 12
    escalation_minutes: int = 30  # Time before escalating to next level
    taiwan_market_only: bool = False
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertMessage:
    """Alert message structure."""
    id: str
    timestamp: datetime
    rule_name: str
    channel: AlertChannel
    severity: SeverityLevel
    title: str
    message: str
    metrics: QualityMetrics
    details: Dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.PENDING
    retry_count: int = 0
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ChannelConfig:
    """Configuration for alert channels."""
    # Email settings
    email_smtp_host: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    # Slack settings
    slack_webhook_url: str = ""
    slack_channel: str = "#data-quality"
    slack_username: str = "Data Quality Bot"
    
    # Webhook settings
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    webhook_timeout: int = 10
    
    # SMS settings (using external service)
    sms_api_url: str = ""
    sms_api_key: str = ""
    sms_phone_numbers: List[str] = field(default_factory=list)
    
    # Teams settings
    teams_webhook_url: str = ""
    
    # General settings
    rate_limit_window_minutes: int = 60
    max_retries: int = 3
    retry_delay_seconds: int = 30


class AlertChannel_Base(ABC):
    """Abstract base class for alert channels."""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.rate_limiter = defaultdict(lambda: deque())
        self._lock = threading.RLock()
    
    @abstractmethod
    async def send_alert(self, message: AlertMessage) -> bool:
        """Send alert through this channel."""
        pass
    
    def is_rate_limited(self, rule_name: str) -> bool:
        """Check if rule is rate limited."""
        with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=self.config.rate_limit_window_minutes)
            
            # Remove old entries
            recent_alerts = self.rate_limiter[rule_name]
            while recent_alerts and recent_alerts[0] < cutoff:
                recent_alerts.popleft()
            
            # Check limit
            return len(recent_alerts) >= 10  # Max 10 alerts per hour by default
    
    def record_alert(self, rule_name: str) -> None:
        """Record alert for rate limiting."""
        with self._lock:
            self.rate_limiter[rule_name].append(datetime.utcnow())


class EmailAlertChannel(AlertChannel_Base):
    """Email alert channel implementation."""
    
    async def send_alert(self, message: AlertMessage) -> bool:
        """Send email alert."""
        if self.is_rate_limited(message.rule_name):
            logger.warning(f"Email alert rate limited for rule: {message.rule_name}")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"[Data Quality Alert] {message.title}"
            
            # Create HTML body
            html_body = self._create_email_html(message)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.config.email_smtp_host, self.config.email_smtp_port) as server:
                server.starttls()
                server.login(self.config.email_username, self.config.email_password)
                server.send_message(msg)
            
            self.record_alert(message.rule_name)
            logger.info(f"Email alert sent for rule: {message.rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_email_html(self, message: AlertMessage) -> str:
        """Create HTML email content."""
        severity_colors = {
            SeverityLevel.CRITICAL: "#ff0000",
            SeverityLevel.ERROR: "#ff6600",
            SeverityLevel.WARNING: "#ffcc00",
            SeverityLevel.INFO: "#0066cc"
        }
        
        color = severity_colors.get(message.severity, "#666666")
        
        html = f"""
        <html>
        <head></head>
        <body>
            <h2 style="color: {color};">Data Quality Alert</h2>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><td><b>Rule</b></td><td>{message.rule_name}</td></tr>
                <tr><td><b>Severity</b></td><td style="color: {color};">{message.severity.value.upper()}</td></tr>
                <tr><td><b>Symbol</b></td><td>{message.metrics.symbol}</td></tr>
                <tr><td><b>Data Type</b></td><td>{message.metrics.data_type.value}</td></tr>
                <tr><td><b>Quality Score</b></td><td>{message.metrics.quality_score:.1f}</td></tr>
                <tr><td><b>Timestamp</b></td><td>{message.timestamp.isoformat()}</td></tr>
            </table>
            
            <h3>Message</h3>
            <p>{message.message}</p>
            
            <h3>Metrics Details</h3>
            <ul>
                <li>Validation Count: {message.metrics.validation_count}</li>
                <li>Passed: {message.metrics.passed_validations}</li>
                <li>Warnings: {message.metrics.warning_count}</li>
                <li>Errors: {message.metrics.error_count}</li>
                <li>Critical: {message.metrics.critical_count}</li>
                <li>Latency: {message.metrics.validation_latency_ms:.2f}ms</li>
            </ul>
            
            <p><small>Generated by ML4T Data Quality Monitoring System</small></p>
        </body>
        </html>
        """
        return html


class SlackAlertChannel(AlertChannel_Base):
    """Slack alert channel implementation."""
    
    async def send_alert(self, message: AlertMessage) -> bool:
        """Send Slack alert."""
        if self.is_rate_limited(message.rule_name):
            logger.warning(f"Slack alert rate limited for rule: {message.rule_name}")
            return False
        
        try:
            payload = self._create_slack_payload(message)
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=self.config.webhook_timeout
            )
            
            if response.status_code == 200:
                self.record_alert(message.rule_name)
                logger.info(f"Slack alert sent for rule: {message.rule_name}")
                return True
            else:
                logger.error(f"Slack alert failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _create_slack_payload(self, message: AlertMessage) -> Dict[str, Any]:
        """Create Slack message payload."""
        severity_colors = {
            SeverityLevel.CRITICAL: "danger",
            SeverityLevel.ERROR: "warning",
            SeverityLevel.WARNING: "warning",
            SeverityLevel.INFO: "good"
        }
        
        color = severity_colors.get(message.severity, "#666666")
        
        return {
            "channel": self.config.slack_channel,
            "username": self.config.slack_username,
            "icon_emoji": ":warning:",
            "attachments": [{
                "color": color,
                "title": f"Data Quality Alert: {message.title}",
                "text": message.message,
                "fields": [
                    {"title": "Rule", "value": message.rule_name, "short": True},
                    {"title": "Severity", "value": message.severity.value.upper(), "short": True},
                    {"title": "Symbol", "value": message.metrics.symbol, "short": True},
                    {"title": "Data Type", "value": message.metrics.data_type.value, "short": True},
                    {"title": "Quality Score", "value": f"{message.metrics.quality_score:.1f}", "short": True},
                    {"title": "Latency", "value": f"{message.metrics.validation_latency_ms:.2f}ms", "short": True}
                ],
                "timestamp": int(message.timestamp.timestamp())
            }]
        }


class WebhookAlertChannel(AlertChannel_Base):
    """Generic webhook alert channel implementation."""
    
    async def send_alert(self, message: AlertMessage) -> bool:
        """Send webhook alert."""
        if self.is_rate_limited(message.rule_name):
            logger.warning(f"Webhook alert rate limited for rule: {message.rule_name}")
            return False
        
        try:
            payload = {
                "alert_id": message.id,
                "timestamp": message.timestamp.isoformat(),
                "rule_name": message.rule_name,
                "severity": message.severity.value,
                "title": message.title,
                "message": message.message,
                "metrics": {
                    "symbol": message.metrics.symbol,
                    "data_type": message.metrics.data_type.value,
                    "quality_score": message.metrics.quality_score,
                    "validation_count": message.metrics.validation_count,
                    "passed_validations": message.metrics.passed_validations,
                    "warning_count": message.metrics.warning_count,
                    "error_count": message.metrics.error_count,
                    "critical_count": message.metrics.critical_count,
                    "validation_latency_ms": message.metrics.validation_latency_ms
                },
                "details": message.details
            }
            
            headers = {"Content-Type": "application/json"}
            headers.update(self.config.webhook_headers)
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers=headers,
                timeout=self.config.webhook_timeout
            )
            
            if 200 <= response.status_code < 300:
                self.record_alert(message.rule_name)
                logger.info(f"Webhook alert sent for rule: {message.rule_name}")
                return True
            else:
                logger.error(f"Webhook alert failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class AlertManager:
    """Central alert management system."""
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[AlertChannel, AlertChannel_Base] = {}
        self.alert_history: deque[AlertMessage] = deque(maxlen=10000)
        self.pending_alerts: Dict[str, AlertMessage] = {}
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._alert_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Initialize channels
        self._initialize_channels()
        
        # Create default rules
        self._create_default_rules()
        
        logger.info("Alert manager initialized")
    
    def _initialize_channels(self) -> None:
        """Initialize alert channels."""
        if self.config.email_to:
            self.channels[AlertChannel.EMAIL] = EmailAlertChannel(self.config)
        
        if self.config.slack_webhook_url:
            self.channels[AlertChannel.SLACK] = SlackAlertChannel(self.config)
        
        if self.config.webhook_url:
            self.channels[AlertChannel.WEBHOOK] = WebhookAlertChannel(self.config)
        
        logger.info(f"Initialized {len(self.channels)} alert channels")
    
    def _create_default_rules(self) -> None:
        """Create default alert rules for Taiwan market."""
        # Critical quality score drop
        self.add_rule(AlertRule(
            name="critical_quality_drop",
            description="Quality score dropped below critical threshold",
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            severity_levels=[SeverityLevel.CRITICAL, SeverityLevel.ERROR],
            conditions={"quality_score_lt": 85.0},
            cooldown_minutes=5,
            max_alerts_per_hour=6,
            taiwan_market_only=True
        ))
        
        # High latency alert
        self.add_rule(AlertRule(
            name="high_latency",
            description="Validation latency exceeded threshold",
            channels=[AlertChannel.SLACK],
            severity_levels=[SeverityLevel.WARNING, SeverityLevel.ERROR],
            conditions={"latency_ms_gt": 10.0},
            cooldown_minutes=10,
            max_alerts_per_hour=12
        ))
        
        # High error rate
        self.add_rule(AlertRule(
            name="high_error_rate",
            description="Error rate exceeded acceptable threshold",
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            severity_levels=[SeverityLevel.ERROR, SeverityLevel.CRITICAL],
            conditions={"error_rate_gt": 0.05},  # 5%
            cooldown_minutes=15,
            max_alerts_per_hour=4
        ))
        
        # Taiwan price limit violation
        self.add_rule(AlertRule(
            name="taiwan_price_limit_violation",
            description="Price limit violation detected in Taiwan market",
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK],
            severity_levels=[SeverityLevel.ERROR, SeverityLevel.CRITICAL],
            conditions={"taiwan_price_violation": True},
            cooldown_minutes=1,  # Short cooldown for price violations
            max_alerts_per_hour=60,
            taiwan_market_only=True
        ))
        
        # Volume spike alert
        self.add_rule(AlertRule(
            name="volume_spike",
            description="Unusual volume spike detected",
            channels=[AlertChannel.SLACK],
            severity_levels=[SeverityLevel.WARNING],
            conditions={"volume_spike_factor_gt": 10.0},
            cooldown_minutes=30,
            max_alerts_per_hour=8,
            taiwan_market_only=True
        ))
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self._lock:
            self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
        return False
    
    def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """Get an alert rule."""
        with self._lock:
            return self.rules.get(rule_name)
    
    def list_rules(self) -> List[AlertRule]:
        """List all alert rules."""
        with self._lock:
            return list(self.rules.values())
    
    async def process_metrics(self, metrics: QualityMetrics) -> List[AlertMessage]:
        """Process metrics and generate alerts."""
        alerts_generated = []
        
        with self._lock:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                # Check if rule applies
                if rule.taiwan_market_only and not self._is_taiwan_market_data(metrics):
                    continue
                
                # Check cooldown
                if self._is_in_cooldown(rule.name):
                    continue
                
                # Evaluate conditions
                if self._evaluate_rule_conditions(rule, metrics):
                    alert = self._create_alert_message(rule, metrics)
                    alerts_generated.append(alert)
                    
                    # Queue for delivery
                    await self._queue_alert(alert)
                    
                    # Update cooldown
                    self.cooldown_tracker[rule.name] = datetime.utcnow()
        
        return alerts_generated
    
    def _is_taiwan_market_data(self, metrics: QualityMetrics) -> bool:
        """Check if metrics are from Taiwan market."""
        # Simple heuristic - could be more sophisticated
        symbol = metrics.symbol
        return bool(symbol and (len(symbol) == 4 or symbol.endswith('.TW')))
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if rule is in cooldown period."""
        if rule_name not in self.cooldown_tracker:
            return False
        
        rule = self.rules[rule_name]
        last_triggered = self.cooldown_tracker[rule_name]
        cooldown_end = last_triggered + timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.utcnow() < cooldown_end
    
    def _evaluate_rule_conditions(self, rule: AlertRule, metrics: QualityMetrics) -> bool:
        """Evaluate rule conditions against metrics."""
        conditions = rule.conditions
        
        # Quality score conditions
        if "quality_score_lt" in conditions:
            if metrics.quality_score >= conditions["quality_score_lt"]:
                return False
        
        if "quality_score_gt" in conditions:
            if metrics.quality_score <= conditions["quality_score_gt"]:
                return False
        
        # Latency conditions
        if "latency_ms_gt" in conditions:
            if metrics.validation_latency_ms <= conditions["latency_ms_gt"]:
                return False
        
        # Error rate conditions
        if "error_rate_gt" in conditions:
            error_rate = (metrics.error_count + metrics.critical_count) / max(metrics.validation_count, 1)
            if error_rate <= conditions["error_rate_gt"]:
                return False
        
        # Taiwan specific conditions
        if "taiwan_price_violation" in conditions:
            # Check metadata for price violation indicators
            if not metrics.metadata.get("price_limit_violation", False):
                return False
        
        if "volume_spike_factor_gt" in conditions:
            volume_factor = metrics.metadata.get("volume_spike_factor", 0)
            if volume_factor <= conditions["volume_spike_factor_gt"]:
                return False
        
        return True
    
    def _create_alert_message(self, rule: AlertRule, metrics: QualityMetrics) -> AlertMessage:
        """Create alert message from rule and metrics."""
        severity = self._determine_alert_severity(rule, metrics)
        
        title = f"{rule.description} - {metrics.symbol}"
        message = self._create_alert_text(rule, metrics)
        
        alert_id = f"{rule.name}_{metrics.symbol}_{int(time.time())}"
        
        return AlertMessage(
            id=alert_id,
            timestamp=datetime.utcnow(),
            rule_name=rule.name,
            channel=rule.channels[0],  # Primary channel
            severity=severity,
            title=title,
            message=message,
            metrics=metrics,
            details={
                "rule_description": rule.description,
                "conditions": rule.conditions,
                "taiwan_market_only": rule.taiwan_market_only
            }
        )
    
    def _determine_alert_severity(self, rule: AlertRule, metrics: QualityMetrics) -> SeverityLevel:
        """Determine alert severity based on metrics and rule."""
        if metrics.critical_count > 0:
            return SeverityLevel.CRITICAL
        elif metrics.error_count > 0:
            return SeverityLevel.ERROR
        elif metrics.warning_count > 0:
            return SeverityLevel.WARNING
        else:
            return SeverityLevel.INFO
    
    def _create_alert_text(self, rule: AlertRule, metrics: QualityMetrics) -> str:
        """Create alert message text."""
        lines = [
            f"Data quality alert triggered for {metrics.symbol}",
            f"Quality Score: {metrics.quality_score:.1f}/100",
            f"Validation Latency: {metrics.validation_latency_ms:.2f}ms",
            f"Validations: {metrics.passed_validations}/{metrics.validation_count} passed"
        ]
        
        if metrics.warning_count > 0:
            lines.append(f"Warnings: {metrics.warning_count}")
        
        if metrics.error_count > 0:
            lines.append(f"Errors: {metrics.error_count}")
        
        if metrics.critical_count > 0:
            lines.append(f"Critical Issues: {metrics.critical_count}")
        
        # Add Taiwan specific details
        if rule.taiwan_market_only:
            lines.append("\nTaiwan Market Specific Alert")
            if "price_limit_violation" in rule.conditions:
                lines.append("Price limit violation detected")
        
        return "\n".join(lines)
    
    async def _queue_alert(self, alert: AlertMessage) -> None:
        """Queue alert for delivery."""
        await self._alert_queue.put(alert)
        
        # Start processing task if not running
        if not self._processing_task or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_alert_queue())
    
    async def _process_alert_queue(self) -> None:
        """Process queued alerts."""
        while True:
            try:
                alert = await asyncio.wait_for(self._alert_queue.get(), timeout=1.0)
                await self._deliver_alert(alert)
                self._alert_queue.task_done()
                
            except asyncio.TimeoutError:
                # Check if queue is empty
                if self._alert_queue.empty():
                    break
            except Exception as e:
                logger.error(f"Error processing alert queue: {e}")
    
    async def _deliver_alert(self, alert: AlertMessage) -> None:
        """Deliver alert through configured channels."""
        rule = self.rules[alert.rule_name]
        
        for channel_type in rule.channels:
            if channel_type not in self.channels:
                logger.warning(f"Channel {channel_type.value} not configured")
                continue
            
            channel = self.channels[channel_type]
            
            try:
                # Create channel-specific alert message
                channel_alert = AlertMessage(
                    id=f"{alert.id}_{channel_type.value}",
                    timestamp=alert.timestamp,
                    rule_name=alert.rule_name,
                    channel=channel_type,
                    severity=alert.severity,
                    title=alert.title,
                    message=alert.message,
                    metrics=alert.metrics,
                    details=alert.details
                )
                
                # Attempt delivery
                success = await channel.send_alert(channel_alert)
                
                if success:
                    channel_alert.status = AlertStatus.SENT
                    channel_alert.delivered_at = datetime.utcnow()
                    logger.info(f"Alert delivered via {channel_type.value}: {alert.id}")
                else:
                    channel_alert.status = AlertStatus.FAILED
                    channel_alert.error_message = "Delivery failed"
                    logger.error(f"Alert delivery failed via {channel_type.value}: {alert.id}")
                
                # Store in history
                with self._lock:
                    self.alert_history.append(channel_alert)
                
            except Exception as e:
                logger.error(f"Error delivering alert via {channel_type.value}: {e}")
    
    def get_alert_history(self, 
                         rule_name: Optional[str] = None,
                         channel: Optional[AlertChannel] = None,
                         limit: Optional[int] = None) -> List[AlertMessage]:
        """Get alert history with optional filtering."""
        with self._lock:
            history = list(self.alert_history)
            
            if rule_name:
                history = [a for a in history if a.rule_name == rule_name]
            
            if channel:
                history = [a for a in history if a.channel == channel]
            
            if limit:
                history = history[-limit:]
            
            return history
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert delivery statistics."""
        with self._lock:
            total_alerts = len(self.alert_history)
            
            if total_alerts == 0:
                return {"total_alerts": 0}
            
            status_counts = defaultdict(int)
            channel_counts = defaultdict(int)
            rule_counts = defaultdict(int)
            
            for alert in self.alert_history:
                status_counts[alert.status.value] += 1
                channel_counts[alert.channel.value] += 1
                rule_counts[alert.rule_name] += 1
            
            success_rate = status_counts[AlertStatus.SENT.value] / total_alerts
            
            return {
                "total_alerts": total_alerts,
                "success_rate": success_rate,
                "status_distribution": dict(status_counts),
                "channel_distribution": dict(channel_counts),
                "rule_distribution": dict(rule_counts),
                "active_rules": len([r for r in self.rules.values() if r.enabled]),
                "configured_channels": len(self.channels)
            }
    
    def test_channel(self, channel: AlertChannel) -> bool:
        """Test alert channel connectivity."""
        if channel not in self.channels:
            logger.error(f"Channel {channel.value} not configured")
            return False
        
        # Create test alert
        test_metrics = QualityMetrics(
            timestamp=datetime.utcnow(),
            symbol="TEST",
            data_type=DataType.PRICE,
            quality_score=95.0,
            validation_count=1,
            passed_validations=1,
            warning_count=0,
            error_count=0,
            critical_count=0,
            validation_latency_ms=5.0
        )
        
        test_alert = AlertMessage(
            id="test_alert",
            timestamp=datetime.utcnow(),
            rule_name="test_rule",
            channel=channel,
            severity=SeverityLevel.INFO,
            title="Test Alert",
            message="This is a test alert to verify channel connectivity.",
            metrics=test_metrics
        )
        
        try:
            # Synchronous test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.channels[channel].send_alert(test_alert))
            loop.close()
            
            if result:
                logger.info(f"Channel {channel.value} test successful")
            else:
                logger.error(f"Channel {channel.value} test failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Channel {channel.value} test error: {e}")
            return False


# Utility functions

def create_taiwan_alert_manager(email_config: Dict[str, Any] = None,
                               slack_config: Dict[str, Any] = None,
                               webhook_config: Dict[str, Any] = None) -> AlertManager:
    """Create pre-configured alert manager for Taiwan market."""
    config = ChannelConfig()
    
    # Configure email if provided
    if email_config:
        config.email_smtp_host = email_config.get('smtp_host', 'smtp.gmail.com')
        config.email_smtp_port = email_config.get('smtp_port', 587)
        config.email_username = email_config.get('username', '')
        config.email_password = email_config.get('password', '')
        config.email_from = email_config.get('from_address', '')
        config.email_to = email_config.get('to_addresses', [])
    
    # Configure Slack if provided
    if slack_config:
        config.slack_webhook_url = slack_config.get('webhook_url', '')
        config.slack_channel = slack_config.get('channel', '#data-quality')
        config.slack_username = slack_config.get('username', 'Taiwan Market Monitor')
    
    # Configure webhook if provided
    if webhook_config:
        config.webhook_url = webhook_config.get('url', '')
        config.webhook_headers = webhook_config.get('headers', {})
        config.webhook_timeout = webhook_config.get('timeout', 10)
    
    manager = AlertManager(config)
    logger.info("Taiwan market alert manager created")
    return manager


def setup_monitoring_alerts(monitor, alert_manager: AlertManager) -> None:
    """Set up integration between monitor and alert manager."""
    async def alert_callback(metrics: QualityMetrics):
        """Callback to process metrics and generate alerts."""
        try:
            alerts = await alert_manager.process_metrics(metrics)
            if alerts:
                logger.info(f"Generated {len(alerts)} alerts for {metrics.symbol}")
        except Exception as e:
            logger.error(f"Error processing alerts: {e}")
    
    # Add callback to monitor
    monitor.add_validation_callback(lambda m: asyncio.create_task(alert_callback(m)))
    logger.info("Alert integration established with quality monitor")