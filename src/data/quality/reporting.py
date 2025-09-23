"""
Automated Quality Reports and SLA Tracking.

This module provides automated generation of quality reports, SLA tracking,
and scheduled delivery of monitoring insights for Taiwan market data quality.
"""

import asyncio
import logging
import os
import smtplib
from datetime import datetime, timedelta, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import schedule
import threading
import time

from .monitor import QualityMonitor, QualityMetrics
from .alerting import AlertManager
from .metrics import QualityMetricsAggregator, SLAResult, MetricType
from ..core.temporal import DataType

logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ReportGenerator:
    """Generate various types of quality reports."""
    
    def __init__(self, 
                 monitor: QualityMonitor,
                 alert_manager: AlertManager,
                 metrics_aggregator: QualityMetricsAggregator):
        self.monitor = monitor
        self.alert_manager = alert_manager
        self.metrics_aggregator = metrics_aggregator
        
        # Report templates
        self.daily_template = self._load_daily_report_template()
        self.weekly_template = self._load_weekly_report_template()
        self.sla_template = self._load_sla_report_template()
    
    def generate_daily_report(self, 
                            report_date: Optional[date] = None,
                            include_charts: bool = True) -> Dict[str, Any]:
        """Generate daily quality report."""
        if report_date is None:
            report_date = date.today() - timedelta(days=1)  # Previous day
        
        start_time = datetime.combine(report_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)
        
        # Get metrics for the day
        daily_metrics = [
            m for m in self.monitor.get_quality_history() 
            if start_time <= m.timestamp < end_time
        ]
        
        if not daily_metrics:
            return {"error": f"No metrics available for {report_date}"}
        
        # Calculate daily statistics
        quality_scores = [m.quality_score for m in daily_metrics]
        latencies = [m.validation_latency_ms for m in daily_metrics]
        
        symbols = list(set(m.symbol for m in daily_metrics))
        data_types = list(set(m.data_type.value for m in daily_metrics))
        
        # Error analysis
        total_errors = sum(m.error_count + m.critical_count for m in daily_metrics)
        total_validations = sum(m.validation_count for m in daily_metrics)
        error_rate = total_errors / max(total_validations, 1)
        
        # SLA compliance
        sla_results = self.metrics_aggregator.sla_tracker.get_all_sla_results(end_time)
        
        # Taiwan market specific analysis
        taiwan_metrics = [m for m in daily_metrics if self._is_taiwan_symbol(m.symbol)]
        taiwan_coverage = len(taiwan_metrics) / len(daily_metrics) if daily_metrics else 0
        
        report_data = {
            'report_date': report_date.isoformat(),
            'generation_time': datetime.utcnow().isoformat(),
            'summary': {
                'total_validations': total_validations,
                'unique_symbols': len(symbols),
                'data_types_processed': len(data_types),
                'avg_quality_score': sum(quality_scores) / len(quality_scores),
                'min_quality_score': min(quality_scores),
                'max_quality_score': max(quality_scores),
                'avg_latency_ms': sum(latencies) / len(latencies),
                'max_latency_ms': max(latencies),
                'p95_latency_ms': pd.Series(latencies).quantile(0.95),
                'error_rate': error_rate,
                'taiwan_market_coverage': taiwan_coverage
            },
            'sla_compliance': {
                'total_slas': len(sla_results),
                'meeting_slas': len([s for s in sla_results if s.status.value == 'meeting']),
                'breached_slas': [
                    {
                        'metric': s.metric_name,
                        'target': s.target_value,
                        'actual': s.actual_value,
                        'compliance': s.compliance_percentage
                    }
                    for s in sla_results if s.status.value == 'breached'
                ]
            },
            'top_symbols': self._get_top_symbols_daily(daily_metrics),
            'data_type_breakdown': self._get_data_type_breakdown(daily_metrics),
            'alerts_summary': self._get_alerts_summary(start_time, end_time),
            'charts': self._generate_daily_charts(daily_metrics) if include_charts else {}
        }
        
        return report_data
    
    def generate_weekly_report(self, 
                             week_ending: Optional[date] = None,
                             include_trends: bool = True) -> Dict[str, Any]:
        """Generate weekly quality report with trends."""
        if week_ending is None:
            week_ending = date.today()
        
        week_start = week_ending - timedelta(days=6)
        start_time = datetime.combine(week_start, datetime.min.time())
        end_time = datetime.combine(week_ending, datetime.min.time()) + timedelta(days=1)
        
        # Get weekly metrics
        weekly_metrics = [
            m for m in self.monitor.get_quality_history() 
            if start_time <= m.timestamp < end_time
        ]
        
        if not weekly_metrics:
            return {"error": f"No metrics available for week ending {week_ending}"}
        
        # Daily aggregations
        daily_aggregates = self._calculate_daily_aggregates(weekly_metrics, week_start, week_ending)
        
        # Weekly summary
        quality_scores = [m.quality_score for m in weekly_metrics]
        latencies = [m.validation_latency_ms for m in weekly_metrics]
        
        # Trend analysis
        trends = {}
        if include_trends and len(daily_aggregates) >= 3:
            trends = self._calculate_weekly_trends(daily_aggregates)
        
        # Taiwan market analysis
        taiwan_analysis = self._get_taiwan_weekly_analysis(weekly_metrics)
        
        report_data = {
            'week_ending': week_ending.isoformat(),
            'week_start': week_start.isoformat(),
            'generation_time': datetime.utcnow().isoformat(),
            'weekly_summary': {
                'total_validations': len(weekly_metrics),
                'avg_daily_validations': len(weekly_metrics) / 7,
                'avg_quality_score': sum(quality_scores) / len(quality_scores),
                'quality_score_stability': pd.Series(quality_scores).std(),
                'avg_latency_ms': sum(latencies) / len(latencies),
                'latency_consistency': pd.Series(latencies).std(),
                'uptime_percentage': self._calculate_uptime(weekly_metrics),
                'unique_symbols': len(set(m.symbol for m in weekly_metrics))
            },
            'daily_breakdown': daily_aggregates,
            'trends': trends,
            'taiwan_market_analysis': taiwan_analysis,
            'sla_weekly_compliance': self._get_weekly_sla_compliance(start_time, end_time),
            'charts': self._generate_weekly_charts(weekly_metrics, daily_aggregates) if include_trends else {}
        }
        
        return report_data
    
    def generate_sla_report(self, 
                           report_period_hours: int = 24) -> Dict[str, Any]:
        """Generate detailed SLA compliance report."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=report_period_hours)
        
        # Get all SLA results
        sla_results = self.metrics_aggregator.sla_tracker.get_all_sla_results(end_time)
        
        # Calculate compliance metrics
        total_slas = len(sla_results)
        meeting_count = len([s for s in sla_results if s.status.value == 'meeting'])
        at_risk_count = len([s for s in sla_results if s.status.value == 'at_risk'])
        breached_count = len([s for s in sla_results if s.status.value == 'breached'])
        
        overall_compliance = (meeting_count / max(total_slas, 1)) * 100
        
        # Taiwan specific SLAs
        taiwan_slas = [s for s in sla_results if 'taiwan' in s.metric_name.lower()]
        taiwan_compliance = (
            len([s for s in taiwan_slas if s.status.value == 'meeting']) / 
            max(len(taiwan_slas), 1) * 100
        )
        
        # Breach analysis
        breached_slas = [s for s in sla_results if s.status.value == 'breached']
        breach_analysis = []
        
        for sla in breached_slas:
            breach_duration_hours = sla.breach_duration_minutes / 60
            impact_score = self._calculate_breach_impact(sla)
            
            breach_analysis.append({
                'metric_name': sla.metric_name,
                'target_value': sla.target_value,
                'actual_value': sla.actual_value,
                'compliance_percentage': sla.compliance_percentage,
                'breach_duration_hours': breach_duration_hours,
                'impact_score': impact_score,
                'recommended_action': self._get_breach_recommendation(sla)
            })
        
        report_data = {
            'report_period_hours': report_period_hours,
            'period_start': start_time.isoformat(),
            'period_end': end_time.isoformat(),
            'generation_time': datetime.utcnow().isoformat(),
            'overall_compliance': {
                'total_slas': total_slas,
                'overall_compliance_percentage': overall_compliance,
                'meeting_count': meeting_count,
                'at_risk_count': at_risk_count,
                'breached_count': breached_count
            },
            'taiwan_market_compliance': {
                'taiwan_specific_slas': len(taiwan_slas),
                'taiwan_compliance_percentage': taiwan_compliance,
                'critical_taiwan_breaches': [
                    s for s in breached_slas if 'taiwan' in s.metric_name.lower()
                ]
            },
            'breach_analysis': breach_analysis,
            'recommendations': self._generate_sla_recommendations(sla_results),
            'compliance_trends': self._get_sla_compliance_trends()
        }
        
        return report_data
    
    def _is_taiwan_symbol(self, symbol: str) -> bool:
        """Check if symbol is Taiwan market."""
        if not symbol:
            return False
        return len(symbol) == 4 or symbol.endswith('.TW') or symbol.endswith('.TWO')
    
    def _get_top_symbols_daily(self, metrics: List[QualityMetrics], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing symbols for the day."""
        symbol_stats = {}
        
        for metric in metrics:
            if metric.symbol not in symbol_stats:
                symbol_stats[metric.symbol] = {
                    'total_score': 0,
                    'count': 0,
                    'total_latency': 0,
                    'errors': 0
                }
            
            stats = symbol_stats[metric.symbol]
            stats['total_score'] += metric.quality_score
            stats['count'] += 1
            stats['total_latency'] += metric.validation_latency_ms
            stats['errors'] += metric.error_count + metric.critical_count
        
        # Calculate averages and sort
        symbol_rankings = []
        for symbol, stats in symbol_stats.items():
            symbol_rankings.append({
                'symbol': symbol,
                'avg_quality_score': stats['total_score'] / stats['count'],
                'avg_latency_ms': stats['total_latency'] / stats['count'],
                'total_errors': stats['errors'],
                'validation_count': stats['count'],
                'is_taiwan': self._is_taiwan_symbol(symbol)
            })
        
        symbol_rankings.sort(key=lambda x: x['avg_quality_score'], reverse=True)
        return symbol_rankings[:limit]
    
    def _get_data_type_breakdown(self, metrics: List[QualityMetrics]) -> List[Dict[str, Any]]:
        """Get breakdown by data type."""
        type_stats = {}
        
        for metric in metrics:
            data_type = metric.data_type.value
            if data_type not in type_stats:
                type_stats[data_type] = {
                    'count': 0,
                    'total_score': 0,
                    'total_latency': 0,
                    'errors': 0
                }
            
            stats = type_stats[data_type]
            stats['count'] += 1
            stats['total_score'] += metric.quality_score
            stats['total_latency'] += metric.validation_latency_ms
            stats['errors'] += metric.error_count + metric.critical_count
        
        breakdown = []
        for data_type, stats in type_stats.items():
            breakdown.append({
                'data_type': data_type,
                'validation_count': stats['count'],
                'avg_quality_score': stats['total_score'] / stats['count'],
                'avg_latency_ms': stats['total_latency'] / stats['count'],
                'total_errors': stats['errors']
            })
        
        return breakdown
    
    def _get_alerts_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get summary of alerts for time period."""
        # Get alert history for period
        all_alerts = self.alert_manager.get_alert_history()
        period_alerts = [
            a for a in all_alerts 
            if start_time <= a.timestamp < end_time
        ]
        
        if not period_alerts:
            return {
                'total_alerts': 0,
                'severity_breakdown': {},
                'top_alert_rules': []
            }
        
        # Severity breakdown
        severity_counts = {}
        rule_counts = {}
        
        for alert in period_alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            rule = alert.rule_name
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        # Top alert rules
        top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_alerts': len(period_alerts),
            'severity_breakdown': severity_counts,
            'top_alert_rules': [{'rule': rule, 'count': count} for rule, count in top_rules]
        }
    
    def _generate_daily_charts(self, metrics: List[QualityMetrics]) -> Dict[str, str]:
        """Generate charts for daily report."""
        charts = {}
        
        # Quality score timeline
        timestamps = [m.timestamp for m in metrics]
        scores = [m.quality_score for m in metrics]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, scores, marker='o', linewidth=2)
        plt.title('Quality Score Timeline')
        plt.xlabel('Time')
        plt.ylabel('Quality Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        charts['quality_timeline'] = img_buffer.getvalue()
        plt.close()
        
        # Latency distribution
        latencies = [m.validation_latency_ms for m in metrics]
        
        plt.figure(figsize=(10, 6))
        plt.hist(latencies, bins=30, alpha=0.7, color='orange')
        plt.title('Validation Latency Distribution')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.axvline(x=10, color='red', linestyle='--', label='10ms Target')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        charts['latency_distribution'] = img_buffer.getvalue()
        plt.close()
        
        return charts
    
    def _calculate_daily_aggregates(self, metrics: List[QualityMetrics], 
                                   start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Calculate daily aggregates for weekly report."""
        daily_data = []
        current_date = start_date
        
        while current_date <= end_date:
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = day_start + timedelta(days=1)
            
            day_metrics = [
                m for m in metrics 
                if day_start <= m.timestamp < day_end
            ]
            
            if day_metrics:
                scores = [m.quality_score for m in day_metrics]
                latencies = [m.validation_latency_ms for m in day_metrics]
                
                daily_data.append({
                    'date': current_date.isoformat(),
                    'validation_count': len(day_metrics),
                    'avg_quality_score': sum(scores) / len(scores),
                    'avg_latency_ms': sum(latencies) / len(latencies),
                    'unique_symbols': len(set(m.symbol for m in day_metrics)),
                    'total_errors': sum(m.error_count + m.critical_count for m in day_metrics)
                })
            else:
                daily_data.append({
                    'date': current_date.isoformat(),
                    'validation_count': 0,
                    'avg_quality_score': 0,
                    'avg_latency_ms': 0,
                    'unique_symbols': 0,
                    'total_errors': 0
                })
            
            current_date += timedelta(days=1)
        
        return daily_data
    
    def _calculate_weekly_trends(self, daily_aggregates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trends from daily aggregates."""
        scores = [d['avg_quality_score'] for d in daily_aggregates if d['avg_quality_score'] > 0]
        latencies = [d['avg_latency_ms'] for d in daily_aggregates if d['avg_latency_ms'] > 0]
        
        if len(scores) < 2:
            return {"insufficient_data": True}
        
        # Simple trend calculation
        score_trend = "stable"
        latency_trend = "stable"
        
        if len(scores) >= 3:
            # Linear trend approximation
            score_diff = scores[-1] - scores[0]
            if score_diff > 2:
                score_trend = "improving"
            elif score_diff < -2:
                score_trend = "degrading"
            
            latency_diff = latencies[-1] - latencies[0]
            if latency_diff > 1:
                latency_trend = "increasing"
            elif latency_diff < -1:
                latency_trend = "improving"
        
        return {
            'quality_score_trend': score_trend,
            'latency_trend': latency_trend,
            'score_variance': pd.Series(scores).var() if scores else 0,
            'latency_variance': pd.Series(latencies).var() if latencies else 0
        }
    
    def _get_taiwan_weekly_analysis(self, metrics: List[QualityMetrics]) -> Dict[str, Any]:
        """Get Taiwan market specific weekly analysis."""
        taiwan_metrics = [m for m in metrics if self._is_taiwan_symbol(m.symbol)]
        
        if not taiwan_metrics:
            return {"no_taiwan_data": True}
        
        taiwan_symbols = set(m.symbol for m in taiwan_metrics)
        taiwan_scores = [m.quality_score for m in taiwan_metrics]
        
        return {
            'taiwan_symbols_count': len(taiwan_symbols),
            'taiwan_validations': len(taiwan_metrics),
            'taiwan_avg_quality': sum(taiwan_scores) / len(taiwan_scores),
            'taiwan_coverage_percentage': len(taiwan_metrics) / len(metrics) * 100,
            'top_taiwan_symbols': [
                s['symbol'] for s in self._get_top_symbols_daily(taiwan_metrics, 5)
                if s['is_taiwan']
            ]
        }
    
    def _calculate_uptime(self, metrics: List[QualityMetrics]) -> float:
        """Calculate system uptime based on metrics availability."""
        if not metrics:
            return 0.0
        
        # Simple uptime calculation: if we have metrics, system was up
        # In practice, would be more sophisticated
        expected_intervals = 7 * 24 * 60 / 5  # 5-minute intervals for a week
        actual_intervals = len(metrics)
        
        return min(100.0, (actual_intervals / expected_intervals) * 100)
    
    def _get_weekly_sla_compliance(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get weekly SLA compliance summary."""
        sla_results = self.metrics_aggregator.sla_tracker.get_all_sla_results(end_time)
        
        return {
            'total_slas': len(sla_results),
            'compliant_slas': len([s for s in sla_results if s.status.value == 'meeting']),
            'breached_slas': len([s for s in sla_results if s.status.value == 'breached']),
            'overall_compliance': (
                len([s for s in sla_results if s.status.value == 'meeting']) / 
                max(len(sla_results), 1) * 100
            )
        }
    
    def _generate_weekly_charts(self, metrics: List[QualityMetrics], 
                               daily_aggregates: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate charts for weekly report."""
        charts = {}
        
        # Daily quality score trend
        dates = [d['date'] for d in daily_aggregates]
        scores = [d['avg_quality_score'] for d in daily_aggregates]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, scores, marker='o', linewidth=2, markersize=6)
        plt.title('Weekly Quality Score Trend')
        plt.xlabel('Date')
        plt.ylabel('Average Quality Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        charts['weekly_trend'] = img_buffer.getvalue()
        plt.close()
        
        return charts
    
    def _calculate_breach_impact(self, sla: SLAResult) -> float:
        """Calculate impact score for SLA breach."""
        # Simple impact calculation
        breach_severity = 1.0 - (sla.compliance_percentage / 100)
        duration_factor = min(1.0, sla.breach_duration_minutes / 60)  # Hours
        
        return breach_severity * duration_factor
    
    def _get_breach_recommendation(self, sla: SLAResult) -> str:
        """Get recommendation for SLA breach."""
        if 'latency' in sla.metric_name.lower():
            return "Review validation algorithms and consider optimization"
        elif 'quality' in sla.metric_name.lower():
            return "Investigate data sources and validation rules"
        elif 'error' in sla.metric_name.lower():
            return "Check system logs and error handling"
        else:
            return "Monitor closely and investigate root cause"
    
    def _generate_sla_recommendations(self, sla_results: List[SLAResult]) -> List[str]:
        """Generate overall recommendations based on SLA results."""
        recommendations = []
        
        breached_slas = [s for s in sla_results if s.status.value == 'breached']
        
        if len(breached_slas) > 2:
            recommendations.append("Multiple SLA breaches detected - perform comprehensive system review")
        
        latency_breaches = [s for s in breached_slas if 'latency' in s.metric_name.lower()]
        if latency_breaches:
            recommendations.append("Performance optimization required for validation latency")
        
        quality_breaches = [s for s in breached_slas if 'quality' in s.metric_name.lower()]
        if quality_breaches:
            recommendations.append("Data quality issues need immediate attention")
        
        taiwan_breaches = [s for s in breached_slas if 'taiwan' in s.metric_name.lower()]
        if taiwan_breaches:
            recommendations.append("Taiwan market specific issues require investigation")
        
        if not recommendations:
            recommendations.append("All SLAs meeting targets - maintain current performance")
        
        return recommendations
    
    def _get_sla_compliance_trends(self) -> Dict[str, Any]:
        """Get SLA compliance trends."""
        # Simplified trend analysis
        # In practice, would analyze historical SLA data
        return {
            "trend_available": False,
            "note": "Historical SLA trend analysis requires longer data collection period"
        }
    
    def _load_daily_report_template(self) -> Template:
        """Load daily report HTML template."""
        template_str = """
        <html>
        <head><title>Daily Quality Report - {{ report_date }}</title></head>
        <body>
            <h1>Taiwan Market Data Quality - Daily Report</h1>
            <h2>{{ report_date }}</h2>
            
            <h3>Executive Summary</h3>
            <ul>
                <li>Total Validations: {{ summary.total_validations }}</li>
                <li>Average Quality Score: {{ "%.1f" | format(summary.avg_quality_score) }}</li>
                <li>Error Rate: {{ "%.2%" | format(summary.error_rate) }}</li>
                <li>Average Latency: {{ "%.2f" | format(summary.avg_latency_ms) }}ms</li>
            </ul>
            
            <h3>SLA Compliance</h3>
            <p>{{ sla_compliance.meeting_slas }}/{{ sla_compliance.total_slas }} SLAs met</p>
            
            {% if sla_compliance.breached_slas %}
            <h4>SLA Breaches</h4>
            <ul>
            {% for breach in sla_compliance.breached_slas %}
                <li>{{ breach.metric }}: {{ "%.1f" | format(breach.actual) }} (target: {{ "%.1f" | format(breach.target) }})</li>
            {% endfor %}
            </ul>
            {% endif %}
        </body>
        </html>
        """
        return Template(template_str)
    
    def _load_weekly_report_template(self) -> Template:
        """Load weekly report HTML template."""
        template_str = """
        <html>
        <head><title>Weekly Quality Report - Week Ending {{ week_ending }}</title></head>
        <body>
            <h1>Taiwan Market Data Quality - Weekly Report</h1>
            <h2>Week Ending {{ week_ending }}</h2>
            
            <h3>Weekly Summary</h3>
            <ul>
                <li>Total Validations: {{ weekly_summary.total_validations }}</li>
                <li>Average Quality Score: {{ "%.1f" | format(weekly_summary.avg_quality_score) }}</li>
                <li>System Uptime: {{ "%.1f" | format(weekly_summary.uptime_percentage) }}%</li>
                <li>Unique Symbols: {{ weekly_summary.unique_symbols }}</li>
            </ul>
            
            <h3>Trends</h3>
            <p>Quality Score Trend: {{ trends.quality_score_trend | default('Unknown') }}</p>
            <p>Latency Trend: {{ trends.latency_trend | default('Unknown') }}</p>
        </body>
        </html>
        """
        return Template(template_str)
    
    def _load_sla_report_template(self) -> Template:
        """Load SLA report HTML template."""
        template_str = """
        <html>
        <head><title>SLA Compliance Report</title></head>
        <body>
            <h1>SLA Compliance Report</h1>
            <h2>{{ period_start }} to {{ period_end }}</h2>
            
            <h3>Overall Compliance</h3>
            <p>{{ "%.1f" | format(overall_compliance.overall_compliance_percentage) }}% 
               ({{ overall_compliance.meeting_count }}/{{ overall_compliance.total_slas }})</p>
            
            {% if breach_analysis %}
            <h3>Breach Analysis</h3>
            <table border="1">
                <tr><th>Metric</th><th>Target</th><th>Actual</th><th>Impact</th></tr>
                {% for breach in breach_analysis %}
                <tr>
                    <td>{{ breach.metric_name }}</td>
                    <td>{{ "%.2f" | format(breach.target_value) }}</td>
                    <td>{{ "%.2f" | format(breach.actual_value) }}</td>
                    <td>{{ "%.2f" | format(breach.impact_score) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </body>
        </html>
        """
        return Template(template_str)


class ReportScheduler:
    """Schedule and deliver automated reports."""
    
    def __init__(self, 
                 report_generator: ReportGenerator,
                 email_config: Optional[Dict[str, str]] = None):
        self.report_generator = report_generator
        self.email_config = email_config or {}
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Setup default schedules
        self._setup_default_schedules()
    
    def _setup_default_schedules(self) -> None:
        """Setup default report schedules."""
        # Daily reports at 8 AM
        schedule.every().day.at("08:00").do(self._generate_and_send_daily_report)
        
        # Weekly reports on Monday at 9 AM
        schedule.every().monday.at("09:00").do(self._generate_and_send_weekly_report)
        
        # SLA reports every 4 hours
        schedule.every(4).hours.do(self._generate_and_send_sla_report)
    
    def start_scheduler(self) -> None:
        """Start the report scheduler."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Report scheduler already running")
            return
        
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Report scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the report scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Report scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)
    
    def _generate_and_send_daily_report(self) -> None:
        """Generate and send daily report."""
        try:
            logger.info("Generating daily quality report")
            report_data = self.report_generator.generate_daily_report()
            
            if "error" not in report_data:
                self._send_report_email(
                    subject=f"Daily Quality Report - {report_data['report_date']}",
                    report_data=report_data,
                    template=self.report_generator.daily_template
                )
                logger.info("Daily report sent successfully")
            else:
                logger.warning(f"Daily report generation failed: {report_data['error']}")
                
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    def _generate_and_send_weekly_report(self) -> None:
        """Generate and send weekly report."""
        try:
            logger.info("Generating weekly quality report")
            report_data = self.report_generator.generate_weekly_report()
            
            if "error" not in report_data:
                self._send_report_email(
                    subject=f"Weekly Quality Report - Week Ending {report_data['week_ending']}",
                    report_data=report_data,
                    template=self.report_generator.weekly_template
                )
                logger.info("Weekly report sent successfully")
            else:
                logger.warning(f"Weekly report generation failed: {report_data['error']}")
                
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
    
    def _generate_and_send_sla_report(self) -> None:
        """Generate and send SLA report."""
        try:
            logger.info("Generating SLA compliance report")
            report_data = self.report_generator.generate_sla_report()
            
            # Only send if there are breaches or significant issues
            if report_data['overall_compliance']['breached_count'] > 0:
                self._send_report_email(
                    subject="SLA Compliance Alert - Action Required",
                    report_data=report_data,
                    template=self.report_generator.sla_template,
                    priority="high"
                )
                logger.info("SLA alert report sent")
            else:
                logger.debug("SLA compliance normal - no alert sent")
                
        except Exception as e:
            logger.error(f"Error generating SLA report: {e}")
    
    def _send_report_email(self, 
                          subject: str, 
                          report_data: Dict[str, Any],
                          template: Template,
                          priority: str = "normal") -> None:
        """Send report via email."""
        if not self.email_config.get('enabled', False):
            logger.debug("Email not configured - report not sent")
            return
        
        try:
            # Render template
            html_content = template.render(**report_data)
            
            # Create email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config.get('from_address', '')
            msg['To'] = ', '.join(self.email_config.get('to_addresses', []))
            
            if priority == "high":
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(
                self.email_config.get('smtp_host', 'localhost'), 
                self.email_config.get('smtp_port', 587)
            ) as server:
                if self.email_config.get('use_tls', True):
                    server.starttls()
                
                if self.email_config.get('username'):
                    server.login(
                        self.email_config['username'], 
                        self.email_config['password']
                    )
                
                server.send_message(msg)
            
            logger.info(f"Report email sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send report email: {e}")


# Utility functions

def setup_automated_reporting(monitor: QualityMonitor,
                             alert_manager: AlertManager,
                             metrics_aggregator: QualityMetricsAggregator,
                             email_config: Optional[Dict[str, str]] = None) -> ReportScheduler:
    """Set up automated reporting system."""
    
    # Create report generator
    report_generator = ReportGenerator(monitor, alert_manager, metrics_aggregator)
    
    # Create and start scheduler
    scheduler = ReportScheduler(report_generator, email_config)
    scheduler.start_scheduler()
    
    logger.info("Automated reporting system established")
    return scheduler


def generate_manual_report(monitor: QualityMonitor,
                          alert_manager: AlertManager,
                          metrics_aggregator: QualityMetricsAggregator,
                          report_type: str = "daily",
                          save_path: Optional[str] = None) -> Dict[str, Any]:
    """Generate a manual report of specified type."""
    
    report_generator = ReportGenerator(monitor, alert_manager, metrics_aggregator)
    
    if report_type == "daily":
        report_data = report_generator.generate_daily_report()
    elif report_type == "weekly":
        report_data = report_generator.generate_weekly_report()
    elif report_type == "sla":
        report_data = report_generator.generate_sla_report()
    else:
        raise ValueError(f"Unknown report type: {report_type}")
    
    if save_path and "error" not in report_data:
        # Save as JSON
        import json
        with open(save_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        logger.info(f"Report saved to {save_path}")
    
    return report_data