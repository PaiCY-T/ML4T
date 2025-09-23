"""
Data Quality Dashboard with Real-time Visualizations.

This module provides a comprehensive web-based dashboard for monitoring
Taiwan market data quality with real-time charts, alerts, and SLA tracking.
Built with Flask, WebSocket, and interactive visualizations.
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import pandas as pd

from .monitor import QualityMonitor, QualityMetrics, AlertLevel
from .alerting import AlertManager, AlertMessage
from .metrics import QualityMetricsAggregator, SLAResult, MetricType
from ..core.temporal import DataType

logger = logging.getLogger(__name__)


class DashboardData:
    """Thread-safe data container for dashboard."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._lock = threading.RLock()
        
        # Real-time data
        self.quality_metrics: deque[QualityMetrics] = deque(maxlen=max_history)
        self.alert_messages: deque[AlertMessage] = deque(maxlen=max_history)
        self.sla_results: List[SLAResult] = []
        
        # Aggregated data
        self.symbol_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.data_type_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.hourly_summary: deque[Dict[str, Any]] = deque(maxlen=48)  # 48 hours
        
        # Dashboard state
        self.last_update: Optional[datetime] = None
        self.active_alerts: List[AlertMessage] = []
        self.system_status: Dict[str, Any] = {}
    
    def add_quality_metrics(self, metrics: QualityMetrics) -> None:
        """Add quality metrics to dashboard data."""
        with self._lock:
            self.quality_metrics.append(metrics)
            self.last_update = datetime.utcnow()
            self._update_aggregates(metrics)
    
    def add_alert_message(self, alert: AlertMessage) -> None:
        """Add alert message to dashboard data."""
        with self._lock:
            self.alert_messages.append(alert)
            
            # Update active alerts
            if alert.severity in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]:
                self.active_alerts.append(alert)
                
                # Keep only recent active alerts (last hour)
                cutoff = datetime.utcnow() - timedelta(hours=1)
                self.active_alerts = [
                    a for a in self.active_alerts if a.timestamp >= cutoff
                ]
    
    def update_sla_results(self, results: List[SLAResult]) -> None:
        """Update SLA results."""
        with self._lock:
            self.sla_results = results.copy()
    
    def update_system_status(self, status: Dict[str, Any]) -> None:
        """Update system status."""
        with self._lock:
            self.system_status = status.copy()
    
    def _update_aggregates(self, metrics: QualityMetrics) -> None:
        """Update aggregated statistics."""
        # Update symbol stats
        symbol_key = metrics.symbol
        symbol_stats = self.symbol_stats[symbol_key]
        
        self._update_rolling_average(symbol_stats, 'quality_score', metrics.quality_score)
        self._update_rolling_average(symbol_stats, 'latency_ms', metrics.validation_latency_ms)
        symbol_stats['last_update'] = metrics.timestamp.isoformat()
        
        # Update data type stats
        dt_key = metrics.data_type.value
        dt_stats = self.data_type_stats[dt_key]
        
        self._update_rolling_average(dt_stats, 'quality_score', metrics.quality_score)
        self._update_rolling_average(dt_stats, 'latency_ms', metrics.validation_latency_ms)
        dt_stats['last_update'] = metrics.timestamp.isoformat()
    
    def _update_rolling_average(self, stats: Dict[str, float], key: str, new_value: float) -> None:
        """Update rolling average for a statistic."""
        alpha = 0.1  # Smoothing factor
        current_avg = stats.get(key, new_value)
        stats[key] = alpha * new_value + (1 - alpha) * current_avg
    
    def get_recent_metrics(self, minutes: int = 30) -> List[QualityMetrics]:
        """Get metrics from recent time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            return [m for m in self.quality_metrics if m.timestamp >= cutoff]
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get summary data for dashboard."""
        with self._lock:
            recent_metrics = self.get_recent_metrics(30)
            
            if not recent_metrics:
                return {"error": "No recent metrics available"}
            
            # Calculate summary statistics
            scores = [m.quality_score for m in recent_metrics]
            latencies = [m.validation_latency_ms for m in recent_metrics]
            
            return {
                'current_time': datetime.utcnow().isoformat(),
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'metrics_count': len(recent_metrics),
                'average_quality_score': sum(scores) / len(scores),
                'max_latency_ms': max(latencies) if latencies else 0,
                'active_alerts_count': len(self.active_alerts),
                'critical_alerts_count': len([a for a in self.active_alerts 
                                             if a.severity == SeverityLevel.CRITICAL]),
                'symbols_monitored': len(set(m.symbol for m in recent_metrics)),
                'data_types_active': len(set(m.data_type.value for m in recent_metrics)),
                'system_status': self.system_status
            }


class QualityDashboard:
    """Web-based quality monitoring dashboard."""
    
    def __init__(self, 
                 monitor: QualityMonitor,
                 alert_manager: AlertManager,
                 metrics_aggregator: QualityMetricsAggregator,
                 host: str = "0.0.0.0",
                 port: int = 5000):
        self.monitor = monitor
        self.alert_manager = alert_manager
        self.metrics_aggregator = metrics_aggregator
        self.host = host
        self.port = port
        
        # Dashboard data storage
        self.data = DashboardData()
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'taiwan_market_quality_dashboard'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Background update thread
        self._update_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Setup routes and handlers
        self._setup_routes()
        self._setup_websocket_handlers()
        self._setup_monitoring_callbacks()
        
        logger.info("Quality dashboard initialized")
    
    def _setup_routes(self) -> None:
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('quality_dashboard.html')
        
        @self.app.route('/api/summary')
        def api_summary():
            """API endpoint for dashboard summary."""
            return jsonify(self.data.get_dashboard_summary())
        
        @self.app.route('/api/metrics/recent')
        def api_recent_metrics():
            """API endpoint for recent metrics."""
            minutes = request.args.get('minutes', 30, type=int)
            recent_metrics = self.data.get_recent_metrics(minutes)
            
            # Convert to JSON-serializable format
            metrics_data = []
            for m in recent_metrics:
                metrics_data.append({
                    'timestamp': m.timestamp.isoformat(),
                    'symbol': m.symbol,
                    'data_type': m.data_type.value,
                    'quality_score': m.quality_score,
                    'validation_latency_ms': m.validation_latency_ms,
                    'validation_count': m.validation_count,
                    'error_count': m.error_count,
                    'critical_count': m.critical_count,
                    'alert_level': m.alert_level.value
                })
            
            return jsonify(metrics_data)
        
        @self.app.route('/api/alerts/active')
        def api_active_alerts():
            """API endpoint for active alerts."""
            with self.data._lock:
                alerts_data = []
                for alert in self.data.active_alerts:
                    alerts_data.append({
                        'id': alert.id,
                        'timestamp': alert.timestamp.isoformat(),
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'message': alert.message,
                        'symbol': alert.metrics.symbol,
                        'rule_name': alert.rule_name
                    })
                
                return jsonify(alerts_data)
        
        @self.app.route('/api/sla/status')
        def api_sla_status():
            """API endpoint for SLA status."""
            with self.data._lock:
                sla_data = []
                for sla in self.data.sla_results:
                    sla_data.append({
                        'metric_name': sla.metric_name,
                        'target_value': sla.target_value,
                        'actual_value': sla.actual_value,
                        'status': sla.status.value,
                        'compliance_percentage': sla.compliance_percentage,
                        'breach_duration_minutes': sla.breach_duration_minutes
                    })
                
                return jsonify(sla_data)
        
        @self.app.route('/api/charts/quality_timeline')
        def api_quality_timeline():
            """API endpoint for quality score timeline chart."""
            minutes = request.args.get('minutes', 60, type=int)
            recent_metrics = self.data.get_recent_metrics(minutes)
            
            if not recent_metrics:
                return jsonify({'error': 'No data available'})
            
            # Create time series data
            timestamps = [m.timestamp for m in recent_metrics]
            scores = [m.quality_score for m in recent_metrics]
            
            chart_data = {
                'x': [ts.isoformat() for ts in timestamps],
                'y': scores,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Quality Score',
                'line': {'color': '#1f77b4'}
            }
            
            layout = {
                'title': 'Quality Score Timeline',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Quality Score', 'range': [0, 100]},
                'hovermode': 'x unified'
            }
            
            return jsonify({'data': [chart_data], 'layout': layout})
        
        @self.app.route('/api/charts/latency_distribution')
        def api_latency_distribution():
            """API endpoint for latency distribution chart."""
            minutes = request.args.get('minutes', 60, type=int)
            recent_metrics = self.data.get_recent_metrics(minutes)
            
            if not recent_metrics:
                return jsonify({'error': 'No data available'})
            
            latencies = [m.validation_latency_ms for m in recent_metrics]
            
            chart_data = {
                'x': latencies,
                'type': 'histogram',
                'nbinsx': 20,
                'name': 'Latency Distribution',
                'marker': {'color': '#ff7f0e'}
            }
            
            layout = {
                'title': 'Validation Latency Distribution',
                'xaxis': {'title': 'Latency (ms)'},
                'yaxis': {'title': 'Count'},
                'bargap': 0.1
            }
            
            return jsonify({'data': [chart_data], 'layout': layout})
        
        @self.app.route('/api/charts/symbol_performance')
        def api_symbol_performance():
            """API endpoint for symbol performance chart."""
            with self.data._lock:
                symbols = list(self.data.symbol_stats.keys())[:20]  # Top 20 symbols
                
                if not symbols:
                    return jsonify({'error': 'No symbol data available'})
                
                scores = [self.data.symbol_stats[s].get('quality_score', 0) for s in symbols]
                latencies = [self.data.symbol_stats[s].get('latency_ms', 0) for s in symbols]
                
                chart_data = [
                    {
                        'x': symbols,
                        'y': scores,
                        'type': 'bar',
                        'name': 'Quality Score',
                        'yaxis': 'y',
                        'marker': {'color': '#2ca02c'}
                    },
                    {
                        'x': symbols,
                        'y': latencies,
                        'type': 'scatter',
                        'mode': 'markers',
                        'name': 'Latency (ms)',
                        'yaxis': 'y2',
                        'marker': {'color': '#d62728', 'size': 8}
                    }
                ]
                
                layout = {
                    'title': 'Symbol Performance Overview',
                    'xaxis': {'title': 'Symbol'},
                    'yaxis': {'title': 'Quality Score', 'side': 'left', 'range': [0, 100]},
                    'yaxis2': {'title': 'Latency (ms)', 'side': 'right', 'overlaying': 'y'},
                    'hovermode': 'x unified'
                }
                
                return jsonify({'data': chart_data, 'layout': layout})
        
        @self.app.route('/api/charts/taiwan_market_status')
        def api_taiwan_market_status():
            """API endpoint for Taiwan market specific status."""
            # Get Taiwan specific SLA results
            taiwan_slas = [
                sla for sla in self.data.sla_results 
                if 'taiwan' in sla.metric_name.lower()
            ]
            
            if not taiwan_slas:
                return jsonify({'error': 'No Taiwan market SLA data available'})
            
            metric_names = [sla.metric_name.replace('_', ' ').title() for sla in taiwan_slas]
            compliance_percentages = [sla.compliance_percentage for sla in taiwan_slas]
            
            # Color coding based on compliance
            colors = []
            for pct in compliance_percentages:
                if pct >= 95:
                    colors.append('#2ca02c')  # Green
                elif pct >= 85:
                    colors.append('#ff7f0e')  # Orange
                else:
                    colors.append('#d62728')  # Red
            
            chart_data = {
                'x': metric_names,
                'y': compliance_percentages,
                'type': 'bar',
                'name': 'SLA Compliance',
                'marker': {'color': colors}
            }
            
            layout = {
                'title': 'Taiwan Market SLA Compliance',
                'xaxis': {'title': 'SLA Metric'},
                'yaxis': {'title': 'Compliance %', 'range': [0, 100]},
                'showlegend': False
            }
            
            return jsonify({'data': [chart_data], 'layout': layout})
    
    def _setup_websocket_handlers(self) -> None:
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            logger.info("Dashboard client connected")
            emit('status', {'message': 'Connected to Taiwan Market Data Quality Dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info("Dashboard client disconnected")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle client request for data update."""
            summary = self.data.get_dashboard_summary()
            emit('dashboard_update', summary)
    
    def _setup_monitoring_callbacks(self) -> None:
        """Setup callbacks to receive data from monitoring systems."""
        
        def metrics_callback(metrics: QualityMetrics):
            """Callback for new quality metrics."""
            self.data.add_quality_metrics(metrics)
            
            # Emit real-time update to connected clients
            self.socketio.emit('metrics_update', {
                'timestamp': metrics.timestamp.isoformat(),
                'symbol': metrics.symbol,
                'quality_score': metrics.quality_score,
                'latency_ms': metrics.validation_latency_ms,
                'alert_level': metrics.alert_level.value
            })
        
        def alert_callback(metrics: QualityMetrics, message: str):
            """Callback for new alerts."""
            # Note: This is a simplified alert callback
            # In practice, would receive AlertMessage objects
            alert_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': metrics.symbol,
                'message': message,
                'severity': 'warning'  # Simplified
            }
            
            self.socketio.emit('alert_update', alert_data)
        
        # Register callbacks
        self.monitor.add_validation_callback(metrics_callback)
        self.monitor.add_alert_callback(alert_callback)
    
    def start_background_updates(self) -> None:
        """Start background data update thread."""
        if self._update_thread and self._update_thread.is_alive():
            logger.warning("Background update thread already running")
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._background_update_loop, daemon=True)
        self._update_thread.start()
        logger.info("Background update thread started")
    
    def stop_background_updates(self) -> None:
        """Stop background data update thread."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("Background update thread stopped")
    
    def _background_update_loop(self) -> None:
        """Background loop to update dashboard data."""
        while self._running:
            try:
                # Update SLA results
                sla_results = self.metrics_aggregator.sla_tracker.get_all_sla_results()
                self.data.update_sla_results(sla_results)
                
                # Update system status
                monitor_stats = self.monitor.get_current_metrics()
                alert_stats = self.alert_manager.get_alert_statistics()
                
                system_status = {
                    'monitor_status': monitor_stats.get('status', 'unknown'),
                    'total_validations': monitor_stats.get('total_validations', 0),
                    'average_latency_ms': monitor_stats.get('average_latency_ms', 0),
                    'alert_success_rate': alert_stats.get('success_rate', 0),
                    'last_update': datetime.utcnow().isoformat()
                }
                
                self.data.update_system_status(system_status)
                
                # Emit periodic updates to clients
                summary = self.data.get_dashboard_summary()
                self.socketio.emit('periodic_update', summary)
                
                # Sleep for update interval
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in background update loop: {e}")
                time.sleep(10)  # Shorter sleep on error
    
    def run(self, debug: bool = False) -> None:
        """Run the dashboard server."""
        self.start_background_updates()
        
        try:
            logger.info(f"Starting quality dashboard on {self.host}:{self.port}")
            self.socketio.run(
                self.app, 
                host=self.host, 
                port=self.port, 
                debug=debug,
                allow_unsafe_werkzeug=True
            )
        finally:
            self.stop_background_updates()
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL."""
        return f"http://{self.host}:{self.port}"


# HTML Template for Dashboard (simplified version)
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Taiwan Market Data Quality Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; margin-top: 5px; }
        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .alerts-section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px; }
        .alert-item { padding: 10px; margin: 5px 0; border-left: 4px solid #e74c3c; background-color: #fdf2f2; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-online { background-color: #27ae60; }
        .status-warning { background-color: #f39c12; }
        .status-error { background-color: #e74c3c; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Taiwan Market Data Quality Dashboard</h1>
        <p>Real-time monitoring and validation system</p>
        <div id="connection-status">
            <span class="status-indicator status-online"></span>
            <span id="status-text">Connected</span>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value" id="avg-quality-score">--</div>
            <div class="metric-label">Average Quality Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="max-latency">--</div>
            <div class="metric-label">Max Latency (ms)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="active-alerts">--</div>
            <div class="metric-label">Active Alerts</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="symbols-monitored">--</div>
            <div class="metric-label">Symbols Monitored</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-container">
            <div id="quality-timeline-chart"></div>
        </div>
        <div class="chart-container">
            <div id="latency-distribution-chart"></div>
        </div>
        <div class="chart-container">
            <div id="symbol-performance-chart"></div>
        </div>
        <div class="chart-container">
            <div id="taiwan-sla-chart"></div>
        </div>
    </div>

    <div class="alerts-section">
        <h3>Active Alerts</h3>
        <div id="alerts-container">
            <p>No active alerts</p>
        </div>
    </div>

    <script>
        // WebSocket connection
        const socket = io();
        
        socket.on('connect', function() {
            document.getElementById('status-text').textContent = 'Connected';
            document.querySelector('.status-indicator').className = 'status-indicator status-online';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('status-text').textContent = 'Disconnected';
            document.querySelector('.status-indicator').className = 'status-indicator status-error';
        });
        
        // Update dashboard with summary data
        function updateDashboard(data) {
            document.getElementById('avg-quality-score').textContent = data.average_quality_score?.toFixed(1) || '--';
            document.getElementById('max-latency').textContent = data.max_latency_ms?.toFixed(2) || '--';
            document.getElementById('active-alerts').textContent = data.active_alerts_count || '0';
            document.getElementById('symbols-monitored').textContent = data.symbols_monitored || '--';
        }
        
        // Load and update charts
        function loadCharts() {
            // Quality timeline chart
            fetch('/api/charts/quality_timeline')
                .then(response => response.json())
                .then(data => {
                    if (!data.error) {
                        Plotly.newPlot('quality-timeline-chart', data.data, data.layout);
                    }
                });
            
            // Latency distribution chart
            fetch('/api/charts/latency_distribution')
                .then(response => response.json())
                .then(data => {
                    if (!data.error) {
                        Plotly.newPlot('latency-distribution-chart', data.data, data.layout);
                    }
                });
            
            // Symbol performance chart
            fetch('/api/charts/symbol_performance')
                .then(response => response.json())
                .then(data => {
                    if (!data.error) {
                        Plotly.newPlot('symbol-performance-chart', data.data, data.layout);
                    }
                });
            
            // Taiwan SLA chart
            fetch('/api/charts/taiwan_market_status')
                .then(response => response.json())
                .then(data => {
                    if (!data.error) {
                        Plotly.newPlot('taiwan-sla-chart', data.data, data.layout);
                    }
                });
        }
        
        // Load active alerts
        function loadAlerts() {
            fetch('/api/alerts/active')
                .then(response => response.json())
                .then(alerts => {
                    const container = document.getElementById('alerts-container');
                    if (alerts.length === 0) {
                        container.innerHTML = '<p>No active alerts</p>';
                    } else {
                        container.innerHTML = alerts.map(alert => 
                            `<div class="alert-item">
                                <strong>${alert.title}</strong> - ${alert.symbol}<br>
                                <small>${alert.timestamp} | Severity: ${alert.severity}</small>
                            </div>`
                        ).join('');
                    }
                });
        }
        
        // Socket event handlers
        socket.on('dashboard_update', updateDashboard);
        socket.on('periodic_update', function(data) {
            updateDashboard(data);
            loadCharts();
            loadAlerts();
        });
        
        socket.on('metrics_update', function(data) {
            // Real-time metric update - could update specific chart points
            console.log('New metrics:', data);
        });
        
        socket.on('alert_update', function(alert) {
            // Real-time alert update
            console.log('New alert:', alert);
            loadAlerts();
        });
        
        // Initial load
        fetch('/api/summary')
            .then(response => response.json())
            .then(updateDashboard);
        
        loadCharts();
        loadAlerts();
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            fetch('/api/summary')
                .then(response => response.json())
                .then(updateDashboard);
            loadCharts();
            loadAlerts();
        }, 30000);
    </script>
</body>
</html>
"""


def create_dashboard_template_file(templates_dir: str = "templates") -> None:
    """Create the dashboard HTML template file."""
    import os
    
    os.makedirs(templates_dir, exist_ok=True)
    template_path = os.path.join(templates_dir, "quality_dashboard.html")
    
    with open(template_path, 'w') as f:
        f.write(DASHBOARD_HTML_TEMPLATE)
    
    logger.info(f"Dashboard template created at {template_path}")


# Utility functions

def create_taiwan_dashboard(monitor: QualityMonitor,
                           alert_manager: AlertManager,
                           metrics_aggregator: QualityMetricsAggregator,
                           port: int = 5000) -> QualityDashboard:
    """Create pre-configured dashboard for Taiwan market monitoring."""
    dashboard = QualityDashboard(
        monitor=monitor,
        alert_manager=alert_manager,
        metrics_aggregator=metrics_aggregator,
        host="0.0.0.0",
        port=port
    )
    
    # Create template file if needed
    try:
        create_dashboard_template_file()
    except Exception as e:
        logger.warning(f"Could not create template file: {e}")
    
    logger.info(f"Taiwan market dashboard created on port {port}")
    return dashboard


def setup_complete_monitoring_system(validation_engine) -> Tuple[QualityMonitor, AlertManager, QualityMetricsAggregator, QualityDashboard]:
    """Set up complete monitoring system with dashboard."""
    from .monitor import create_taiwan_market_monitor, setup_monitoring_pipeline
    from .alerting import create_taiwan_alert_manager, setup_monitoring_alerts
    from .metrics import create_taiwan_metrics_system
    
    # Create all components
    monitor = create_taiwan_market_monitor(validation_engine)
    alert_manager = create_taiwan_alert_manager()
    metrics_aggregator = create_taiwan_metrics_system()
    
    # Setup integration
    setup_monitoring_alerts(monitor, alert_manager)
    
    # Create dashboard
    dashboard = create_taiwan_dashboard(monitor, alert_manager, metrics_aggregator)
    
    logger.info("Complete Taiwan market monitoring system established")
    return monitor, alert_manager, metrics_aggregator, dashboard