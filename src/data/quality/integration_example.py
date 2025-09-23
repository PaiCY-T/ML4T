"""
Example integration of Stream B monitoring system components.

This example demonstrates how to set up and use the complete monitoring
and alerting system for Taiwan market data quality validation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from .validation_engine import ValidationEngine, ValidationContext
from .monitor import QualityMonitor, create_taiwan_market_monitor
from .alerting import AlertManager, create_taiwan_alert_manager 
from .metrics import QualityMetricsAggregator, create_taiwan_metrics_system
from .dashboard import QualityDashboard, create_taiwan_dashboard
from ..core.temporal import TemporalValue, DataType

logger = logging.getLogger(__name__)


async def setup_monitoring_system() -> Dict[str, Any]:
    """
    Set up complete monitoring system for Taiwan market.
    
    Returns:
        Dictionary containing all system components
    """
    # Create mock validation engine (replace with real one)
    validation_engine = MockValidationEngine()
    
    # Create monitoring components
    monitor = create_taiwan_market_monitor(validation_engine)
    
    # Create alerting system (with mock config)
    alert_config = {
        'email': {
            'smtp_host': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'data-quality@example.com',
            'password': 'your_password',
            'from_address': 'data-quality@example.com',
            'to_addresses': ['admin@example.com']
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'channel': '#data-quality',
            'username': 'Taiwan Market Monitor'
        }
    }
    alert_manager = create_taiwan_alert_manager(
        email_config=alert_config.get('email'),
        slack_config=alert_config.get('slack')
    )
    
    # Create metrics aggregator
    metrics_aggregator = create_taiwan_metrics_system()
    
    # Create dashboard (optional)
    dashboard = create_taiwan_dashboard(
        monitor=monitor,
        alert_manager=alert_manager, 
        metrics_aggregator=metrics_aggregator,
        port=5000
    )
    
    # Setup integration callbacks
    def metrics_callback(metrics):
        """Process metrics through aggregator and check alerts."""
        # Process through metrics aggregator
        analysis = metrics_aggregator.process_quality_metrics(metrics)
        
        # Check for alerts
        asyncio.create_task(alert_manager.process_metrics(metrics))
        
        logger.info(f"Processed metrics for {metrics.symbol} - Score: {metrics.quality_score:.1f}")
    
    # Add callback to monitor
    monitor.add_validation_callback(metrics_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    logger.info("Taiwan market monitoring system initialized successfully")
    
    return {
        'monitor': monitor,
        'alert_manager': alert_manager,
        'metrics_aggregator': metrics_aggregator,
        'dashboard': dashboard,
        'validation_engine': validation_engine
    }


class MockValidationEngine:
    """Mock validation engine for demonstration."""
    
    async def validate_value(self, value: TemporalValue, context=None):
        """Mock validation that returns sample results."""
        # Simulate validation processing
        await asyncio.sleep(0.005)  # 5ms processing time
        
        # Return mock validation results
        from .validation_engine import ValidationOutput, ValidationResult
        
        return [
            ValidationOutput(
                validator_name="mock_validator",
                result=ValidationResult.PASS,
                execution_time_ms=5.0,
                issues=[],
                metadata={'mock': True}
            )
        ]
    
    async def validate_batch(self, values, contexts=None):
        """Mock batch validation."""
        results = {}
        for value in values:
            results[value] = await self.validate_value(value)
        return results


async def demo_monitoring_workflow():
    """Demonstrate the monitoring workflow with sample data."""
    print("Setting up Taiwan market monitoring system...")
    
    # Setup system
    system = await setup_monitoring_system()
    monitor = system['monitor']
    alert_manager = system['alert_manager']
    metrics_aggregator = system['metrics_aggregator']
    
    print("✅ System initialized successfully")
    
    # Create sample Taiwan market data
    sample_values = [
        TemporalValue(
            symbol="2330",  # TSMC
            value_date=datetime.now().date(),
            as_of_date=datetime.now().date(),
            data_type=DataType.PRICE,
            value=580.0,
            metadata={'exchange': 'TSE', 'currency': 'TWD'}
        ),
        TemporalValue(
            symbol="2317",  # Hon Hai
            value_date=datetime.now().date(),
            as_of_date=datetime.now().date(),
            data_type=DataType.PRICE,
            value=105.5,
            metadata={'exchange': 'TSE', 'currency': 'TWD'}
        )
    ]
    
    print("Processing sample Taiwan market data...")
    
    # Process sample data through monitoring system
    for value in sample_values:
        metrics = await monitor.monitor_validation(value)
        if metrics:
            print(f"  Processed {value.symbol}: Quality Score {metrics.quality_score:.1f}, "
                  f"Latency {metrics.validation_latency_ms:.2f}ms")
    
    # Get system status
    status = monitor.get_current_metrics()
    sla_summary = metrics_aggregator.sla_tracker.get_sla_summary()
    alert_stats = alert_manager.get_alert_statistics()
    
    print("\n📊 System Status:")
    print(f"  Total Validations: {status.get('total_validations', 0)}")
    print(f"  Average Latency: {status.get('average_latency_ms', 0):.2f}ms")
    print(f"  SLA Compliance: {sla_summary.get('overall_compliance_percentage', 0):.1f}%")
    print(f"  Alert Success Rate: {alert_stats.get('success_rate', 0):.1%}")
    
    # Test alert system
    print("\n🚨 Testing alert system...")
    test_results = {}
    if alert_manager.channels:
        for channel_type in alert_manager.channels.keys():
            result = alert_manager.test_channel(channel_type)
            test_results[channel_type.value] = "✅ Success" if result else "❌ Failed"
    
    for channel, result in test_results.items():
        print(f"  {channel}: {result}")
    
    print("\n✅ Demo completed successfully!")
    
    # Cleanup
    monitor.stop_monitoring()
    
    return system


def create_monitoring_configuration() -> Dict[str, Any]:
    """Create sample monitoring configuration for Taiwan market."""
    return {
        'monitoring': {
            'thresholds': {
                'quality_score_warning': 95.0,
                'quality_score_critical': 90.0,
                'latency_warning_ms': 8.0,
                'latency_critical_ms': 10.0,
                'error_rate_warning': 0.02,
                'error_rate_critical': 0.05
            },
            'taiwan_specific': {
                'price_limit_buffer': 0.01,
                'volume_spike_threshold': 5.0,
                'trading_hours_tolerance_minutes': 2
            }
        },
        'alerting': {
            'email': {
                'enabled': True,
                'smtp_host': 'smtp.example.com',
                'recipients': ['admin@example.com']
            },
            'slack': {
                'enabled': True,
                'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK',
                'channel': '#data-quality'
            },
            'webhook': {
                'enabled': False,
                'url': 'https://your-webhook-endpoint.com/alerts'
            }
        },
        'sla': {
            'quality_score_target': 95.0,
            'latency_p95_target_ms': 10.0,
            'availability_target': 0.999,
            'error_rate_target': 0.01
        },
        'dashboard': {
            'enabled': True,
            'host': '0.0.0.0',
            'port': 5000,
            'refresh_interval_seconds': 30
        }
    }


if __name__ == "__main__":
    # Run demo
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_monitoring_workflow())