"""
Operational Monitoring Integration Layer - Stream C, Task #27.

Integration layer that connects operational monitoring (Stream C) with
statistical validation (Stream A) and business logic validation (Stream B)
to provide unified monitoring and alerting for the ML4T system.

Key Features:
- Unified validation orchestration
- Cross-stream alert correlation
- Integrated performance reporting
- Real-time monitoring coordination
- Taiwan market operational compliance
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import warnings

import numpy as np
import pandas as pd

# Import Stream A components
from ..validation.statistical_validator import (
    StatisticalValidator, ValidationConfig, ValidationResults
)

# Import Stream B components  
from ..validation.business_logic.business_validator import (
    BusinessValidator, ValidationResult, ValidationSeverity
)

# Import Stream C components
from .model_health import ModelHealthMonitor, HealthMetrics, Alert
from .retraining_manager import ModelRetrainingManager, RetrainingEvent
from ..alerts.alert_system import AlertSystem, AlertSeverity as SystemAlertSeverity

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Integration system status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class ValidationStreamStatus(Enum):
    """Status of individual validation streams."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


@dataclass
class StreamHealth:
    """Health status of a validation stream."""
    stream_id: str
    status: ValidationStreamStatus
    last_update: datetime
    validation_score: float
    response_time_ms: float
    error_count: int = 0
    success_rate: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'stream_id': self.stream_id,
            'status': self.status.value,
            'last_update': self.last_update.isoformat(),
            'validation_score': self.validation_score,
            'response_time_ms': self.response_time_ms,
            'error_count': self.error_count,
            'success_rate': self.success_rate
        }


@dataclass
class IntegratedValidationResult:
    """Combined validation result from all streams."""
    timestamp: datetime
    
    # Stream A - Statistical validation
    statistical_score: float
    ic_score: float
    drift_scores: Dict[str, float]
    
    # Stream B - Business logic validation
    business_score: float
    regulatory_compliance: float
    risk_score: float
    
    # Stream C - Operational monitoring
    operational_score: float
    system_health: float
    alert_count: int
    
    # Overall assessment
    overall_score: float
    overall_status: str
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'statistical_score': self.statistical_score,
            'ic_score': self.ic_score,
            'drift_scores': self.drift_scores,
            'business_score': self.business_score,
            'regulatory_compliance': self.regulatory_compliance,
            'risk_score': self.risk_score,
            'operational_score': self.operational_score,
            'system_health': self.system_health,
            'alert_count': self.alert_count,
            'overall_score': self.overall_score,
            'overall_status': self.overall_status,
            'recommendations': self.recommendations
        }


class ValidationOrchestrator:
    """Orchestrate validation across all three streams."""
    
    def __init__(
        self,
        statistical_validator: Optional[StatisticalValidator] = None,
        business_validator: Optional[BusinessValidator] = None,
        health_monitor: Optional[ModelHealthMonitor] = None,
        alert_system: Optional[AlertSystem] = None
    ):
        self.statistical_validator = statistical_validator
        self.business_validator = business_validator  
        self.health_monitor = health_monitor
        self.alert_system = alert_system
        
        # Stream health tracking
        self.stream_health = {
            'stream_a': StreamHealth('stream_a', ValidationStreamStatus.HEALTHY, datetime.now(), 0.0, 0.0),
            'stream_b': StreamHealth('stream_b', ValidationStreamStatus.HEALTHY, datetime.now(), 0.0, 0.0),
            'stream_c': StreamHealth('stream_c', ValidationStreamStatus.HEALTHY, datetime.now(), 0.0, 0.0)
        }
        
        # Validation history
        self.validation_history = deque(maxlen=1000)
        
        # Performance tracking
        self.validation_performance = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'avg_validation_time_ms': 0.0,
            'last_validation_time': None
        }
        
    async def run_integrated_validation(
        self,
        model_data: Dict[str, Any],
        timeout_seconds: float = 30.0
    ) -> IntegratedValidationResult:
        """
        Run integrated validation across all streams with timeout protection.
        
        Args:
            model_data: Model data including predictions, features, portfolio weights, etc.
            timeout_seconds: Maximum time to wait for validation completion
            
        Returns:
            Integrated validation results
        """
        start_time = time.time()
        validation_tasks = []
        
        try:
            # Start all validation streams concurrently
            if self.statistical_validator:
                validation_tasks.append(
                    asyncio.create_task(self._run_statistical_validation(model_data))
                )
            
            if self.business_validator:
                validation_tasks.append(
                    asyncio.create_task(self._run_business_validation(model_data))
                )
            
            if self.health_monitor:
                validation_tasks.append(
                    asyncio.create_task(self._run_operational_validation(model_data))
                )
            
            # Wait for all validations with timeout
            if validation_tasks:
                results = await asyncio.wait_for(
                    asyncio.gather(*validation_tasks, return_exceptions=True),
                    timeout=timeout_seconds
                )
            else:
                results = []
            
            # Process results
            statistical_result = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None
            business_result = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
            operational_result = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None
            
            # Handle exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    stream_names = ['stream_a', 'stream_b', 'stream_c']
                    if i < len(stream_names):
                        self._handle_stream_error(stream_names[i], result)
            
            # Create integrated result
            integrated_result = self._create_integrated_result(
                statistical_result, business_result, operational_result
            )
            
            # Update performance tracking
            validation_time = (time.time() - start_time) * 1000
            self._update_performance_tracking(validation_time, success=True)
            
            # Store in history
            self.validation_history.append(integrated_result)
            
            logger.info(f"Integrated validation completed in {validation_time:.1f}ms")
            
            return integrated_result
            
        except asyncio.TimeoutError:
            logger.error(f"Validation timeout after {timeout_seconds}s")
            self._update_performance_tracking((time.time() - start_time) * 1000, success=False)
            
            # Return degraded result
            return self._create_degraded_result("Validation timeout")
            
        except Exception as e:
            logger.error(f"Integrated validation failed: {e}")
            self._update_performance_tracking((time.time() - start_time) * 1000, success=False)
            
            return self._create_degraded_result(f"Validation error: {str(e)}")
    
    async def _run_statistical_validation(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stream A statistical validation."""
        start_time = time.time()
        
        try:
            if not self.statistical_validator:
                return None
            
            # Extract data for statistical validation
            predictions = model_data.get('predictions')
            actuals = model_data.get('actuals')
            features = model_data.get('features')
            
            if predictions is None:
                raise ValueError("No predictions provided for statistical validation")
            
            # Run statistical validation
            if hasattr(self.statistical_validator, 'validate_model_performance'):
                result = await self._async_wrapper(
                    self.statistical_validator.validate_model_performance,
                    predictions, actuals, features
                )
            else:
                # Fallback for synchronous validator
                result = self.statistical_validator.comprehensive_validation(
                    predictions, actuals, features
                )
            
            # Update stream health
            response_time = (time.time() - start_time) * 1000
            self._update_stream_health('stream_a', result, response_time)
            
            return result
            
        except Exception as e:
            self._handle_stream_error('stream_a', e)
            return None
    
    async def _run_business_validation(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stream B business logic validation."""
        start_time = time.time()
        
        try:
            if not self.business_validator:
                return None
            
            # Extract data for business validation
            portfolio_weights = model_data.get('portfolio_weights')
            predictions = model_data.get('predictions')
            
            if portfolio_weights is None:
                # Create portfolio from predictions for validation
                portfolio_weights = self._create_portfolio_from_predictions(
                    predictions, model_data.get('universe', [])
                )
            
            # Run business validation
            if hasattr(self.business_validator, 'validate_portfolio'):
                result = await self._async_wrapper(
                    self.business_validator.validate_portfolio,
                    portfolio_weights
                )
            else:
                # Fallback for synchronous validator
                result = self.business_validator.comprehensive_validation(portfolio_weights)
            
            # Update stream health
            response_time = (time.time() - start_time) * 1000
            self._update_stream_health('stream_b', result, response_time)
            
            return result
            
        except Exception as e:
            self._handle_stream_error('stream_b', e)
            return None
    
    async def _run_operational_validation(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stream C operational monitoring."""
        start_time = time.time()
        
        try:
            if not self.health_monitor:
                return None
            
            # Update health monitor with latest data
            predictions = model_data.get('predictions')
            actuals = model_data.get('actuals')
            features = model_data.get('features')
            returns = model_data.get('returns')
            
            health_metrics = self.health_monitor.update_health(
                predictions=predictions,
                actuals=actuals,
                features=features,
                returns=returns
            )
            
            # Get health report
            health_report = self.health_monitor.get_health_report()
            
            # Update stream health
            response_time = (time.time() - start_time) * 1000
            self._update_stream_health('stream_c', health_report, response_time)
            
            return health_report
            
        except Exception as e:
            self._handle_stream_error('stream_c', e)
            return None
    
    async def _async_wrapper(self, func: Callable, *args, **kwargs) -> Any:
        """Wrapper to run synchronous functions asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def _create_portfolio_from_predictions(
        self,
        predictions: np.ndarray,
        universe: List[str]
    ) -> pd.DataFrame:
        """Create simple portfolio from predictions for business validation."""
        if predictions is None or len(predictions) == 0:
            return pd.DataFrame()
        
        # Simple rank-based portfolio construction
        n_stocks = len(predictions)
        if len(universe) == 0:
            universe = [f"stock_{i}" for i in range(n_stocks)]
        
        # Top 20% get positive weights, bottom 20% get negative weights
        sorted_indices = np.argsort(predictions)
        weights = np.zeros(n_stocks)
        
        top_n = max(1, n_stocks // 5)
        bottom_n = max(1, n_stocks // 5)
        
        # Long top performers
        weights[sorted_indices[-top_n:]] = 1.0 / top_n
        
        # Short bottom performers
        weights[sorted_indices[:bottom_n]] = -1.0 / bottom_n
        
        return pd.DataFrame({
            'symbol': universe[:n_stocks],
            'weight': weights
        })
    
    def _create_integrated_result(
        self,
        statistical_result: Optional[Dict[str, Any]],
        business_result: Optional[Dict[str, Any]],
        operational_result: Optional[Dict[str, Any]]
    ) -> IntegratedValidationResult:
        """Create integrated validation result from individual stream results."""
        
        # Extract statistical metrics
        statistical_score = 0.0
        ic_score = 0.0
        drift_scores = {}
        
        if statistical_result:
            statistical_score = statistical_result.get('validation_score', 0.0)
            if isinstance(statistical_result.get('ic_scores'), dict):
                ic_score = statistical_result['ic_scores'].get('current', 0.0)
            else:
                ic_score = statistical_result.get('ic_score', 0.0)
            drift_scores = statistical_result.get('drift_scores', {})
            if not isinstance(drift_scores, dict):
                drift_scores = {'feature_drift_score': statistical_result.get('feature_drift_score', 0.0)}
        
        # Extract business metrics
        business_score = 0.0
        regulatory_compliance = 0.0
        risk_score = 0.0
        
        if business_result:
            business_score = business_result.get('overall_score', 0.0)
            regulatory_compliance = business_result.get('regulatory_compliance', 0.0)
            risk_score = business_result.get('risk_management', 0.0)
        
        # Extract operational metrics
        operational_score = 0.0
        system_health = 0.0
        alert_count = 0
        
        if operational_result:
            operational_score = operational_result.get('health_score', 0.0)
            system_health = operational_result['metrics']['system'].get('prediction_success_rate', 0.0) * 100
            alert_count = operational_result.get('alerts', {}).get('total_alerts_24h', 0)
        
        # Calculate overall score (weighted combination)
        weights = {'statistical': 0.4, 'business': 0.35, 'operational': 0.25}
        
        overall_score = (
            statistical_score * weights['statistical'] +
            business_score * weights['business'] +
            operational_score * weights['operational']
        )
        
        # Determine overall status
        if overall_score >= 0.8:
            overall_status = "healthy"
        elif overall_score >= 0.6:
            overall_status = "warning"
        elif overall_score >= 0.4:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        # Generate recommendations
        recommendations = self._generate_integrated_recommendations(
            statistical_result, business_result, operational_result,
            overall_score, overall_status
        )
        
        return IntegratedValidationResult(
            timestamp=datetime.now(),
            statistical_score=statistical_score,
            ic_score=ic_score,
            drift_scores=drift_scores,
            business_score=business_score,
            regulatory_compliance=regulatory_compliance,
            risk_score=risk_score,
            operational_score=operational_score,
            system_health=system_health,
            alert_count=alert_count,
            overall_score=overall_score,
            overall_status=overall_status,
            recommendations=recommendations
        )
    
    def _generate_integrated_recommendations(
        self,
        statistical_result: Optional[Dict[str, Any]],
        business_result: Optional[Dict[str, Any]],
        operational_result: Optional[Dict[str, Any]],
        overall_score: float,
        overall_status: str
    ) -> List[str]:
        """Generate integrated recommendations based on all validation results."""
        recommendations = []
        
        # Statistical recommendations
        if statistical_result:
            stat_recommendations = statistical_result.get('recommendations', [])
            if isinstance(stat_recommendations, list):
                recommendations.extend(stat_recommendations)
        
        # Business recommendations
        if business_result:
            business_warnings = business_result.get('warnings', [])
            if isinstance(business_warnings, list):
                recommendations.extend([f"Business: {w}" for w in business_warnings])
        
        # Operational recommendations
        if operational_result and operational_result.get('alerts'):
            alert_info = operational_result['alerts']
            if alert_info.get('total_alerts_24h', 0) > 5:
                recommendations.append("Operational: High alert volume - investigate system stability")
        
        # Overall status recommendations
        if overall_status == "critical":
            recommendations.insert(0, "CRITICAL: Immediate intervention required - consider emergency retraining")
        elif overall_status == "degraded":
            recommendations.insert(0, "DEGRADED: Schedule model retraining within 24-48 hours")
        elif overall_status == "warning":
            recommendations.insert(0, "WARNING: Monitor closely and prepare for potential retraining")
        
        # Taiwan market specific
        recommendations.append("Ensure compliance with Taiwan market regulations and trading hours")
        
        return recommendations
    
    def _create_degraded_result(self, error_message: str) -> IntegratedValidationResult:
        """Create degraded validation result when validation fails."""
        return IntegratedValidationResult(
            timestamp=datetime.now(),
            statistical_score=0.0,
            ic_score=0.0,
            drift_scores={},
            business_score=0.0,
            regulatory_compliance=0.0,
            risk_score=0.0,
            operational_score=0.0,
            system_health=0.0,
            alert_count=0,
            overall_score=0.0,
            overall_status="critical",
            recommendations=[f"Validation system error: {error_message}"]
        )
    
    def _update_stream_health(
        self,
        stream_id: str,
        result: Optional[Dict[str, Any]],
        response_time_ms: float
    ):
        """Update health status for a validation stream."""
        if stream_id not in self.stream_health:
            return
        
        stream_health = self.stream_health[stream_id]
        stream_health.last_update = datetime.now()
        stream_health.response_time_ms = response_time_ms
        
        if result is None:
            stream_health.status = ValidationStreamStatus.ERROR
            stream_health.error_count += 1
            stream_health.validation_score = 0.0
        else:
            # Extract validation score based on stream type
            if stream_id == 'stream_a':
                stream_health.validation_score = result.get('validation_score', 0.0)
            elif stream_id == 'stream_b':
                stream_health.validation_score = result.get('overall_score', 0.0)
            elif stream_id == 'stream_c':
                stream_health.validation_score = result.get('health_score', 0.0)
            
            # Determine status based on score and response time
            if stream_health.validation_score >= 0.8 and response_time_ms < 200:
                stream_health.status = ValidationStreamStatus.HEALTHY
            elif stream_health.validation_score >= 0.6 or response_time_ms < 500:
                stream_health.status = ValidationStreamStatus.WARNING
            else:
                stream_health.status = ValidationStreamStatus.ERROR
        
        # Update success rate
        total_calls = stream_health.error_count + max(1, self.validation_performance['total_validations'])
        stream_health.success_rate = 1.0 - (stream_health.error_count / total_calls)
    
    def _handle_stream_error(self, stream_id: str, error: Exception):
        """Handle error from a validation stream."""
        logger.error(f"Stream {stream_id} error: {error}")
        
        if stream_id in self.stream_health:
            self.stream_health[stream_id].status = ValidationStreamStatus.ERROR
            self.stream_health[stream_id].error_count += 1
    
    def _update_performance_tracking(self, validation_time_ms: float, success: bool):
        """Update validation performance tracking."""
        self.validation_performance['total_validations'] += 1
        self.validation_performance['last_validation_time'] = datetime.now()
        
        if success:
            self.validation_performance['successful_validations'] += 1
        else:
            self.validation_performance['failed_validations'] += 1
        
        # Update average validation time (exponential moving average)
        alpha = 0.1
        if self.validation_performance['avg_validation_time_ms'] == 0:
            self.validation_performance['avg_validation_time_ms'] = validation_time_ms
        else:
            self.validation_performance['avg_validation_time_ms'] = (
                alpha * validation_time_ms + 
                (1 - alpha) * self.validation_performance['avg_validation_time_ms']
            )
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        # Determine overall integration status
        stream_statuses = [health.status for health in self.stream_health.values()]
        
        if all(status == ValidationStreamStatus.HEALTHY for status in stream_statuses):
            integration_status = IntegrationStatus.ACTIVE
        elif any(status == ValidationStreamStatus.ERROR for status in stream_statuses):
            error_count = sum(1 for status in stream_statuses if status == ValidationStreamStatus.ERROR)
            if error_count >= 2:
                integration_status = IntegrationStatus.CRITICAL
            else:
                integration_status = IntegrationStatus.DEGRADED
        elif any(status == ValidationStreamStatus.WARNING for status in stream_statuses):
            integration_status = IntegrationStatus.DEGRADED
        else:
            integration_status = IntegrationStatus.OFFLINE
        
        return {
            'integration_status': integration_status.value,
            'timestamp': datetime.now().isoformat(),
            'stream_health': {k: v.to_dict() for k, v in self.stream_health.items()},
            'performance': self.validation_performance,
            'recent_validations': len(self.validation_history),
            'latest_validation': self.validation_history[-1].to_dict() if self.validation_history else None
        }
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_validations = [
            v for v in self.validation_history 
            if v.timestamp > cutoff_time
        ]
        
        if not recent_validations:
            return {
                'period_hours': hours,
                'validation_count': 0,
                'avg_overall_score': 0.0,
                'status_distribution': {},
                'trend': 'insufficient_data'
            }
        
        # Calculate metrics
        overall_scores = [v.overall_score for v in recent_validations]
        statuses = [v.overall_status for v in recent_validations]
        
        status_distribution = {}
        for status in statuses:
            status_distribution[status] = status_distribution.get(status, 0) + 1
        
        # Calculate trend
        if len(overall_scores) >= 3:
            recent_trend = np.polyfit(range(len(overall_scores)), overall_scores, 1)[0]
            if recent_trend > 0.01:
                trend = 'improving'
            elif recent_trend < -0.01:
                trend = 'degrading'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'period_hours': hours,
            'validation_count': len(recent_validations),
            'avg_overall_score': np.mean(overall_scores),
            'min_overall_score': np.min(overall_scores),
            'max_overall_score': np.max(overall_scores),
            'status_distribution': status_distribution,
            'trend': trend,
            'latest_score': overall_scores[-1] if overall_scores else 0.0
        }


class OperationalIntegrationManager:
    """Main manager for operational monitoring integration."""
    
    def __init__(
        self,
        statistical_validator: Optional[StatisticalValidator] = None,
        business_validator: Optional[BusinessValidator] = None,
        health_monitor: Optional[ModelHealthMonitor] = None,
        alert_system: Optional[AlertSystem] = None,
        retraining_manager: Optional[ModelRetrainingManager] = None
    ):
        # Initialize orchestrator
        self.orchestrator = ValidationOrchestrator(
            statistical_validator, business_validator, health_monitor, alert_system
        )
        
        self.retraining_manager = retraining_manager
        
        # Integration state
        self.is_running = False
        self.monitoring_task = None
        
        # Callbacks
        self.validation_callbacks = []
        self.alert_callbacks = []
        
    async def start_monitoring(self, update_interval: int = 300):
        """Start continuous integrated monitoring."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(update_interval)
        )
        
        logger.info(f"Integrated monitoring started with {update_interval}s interval")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Integrated monitoring stopped")
    
    async def _monitoring_loop(self, update_interval: int):
        """Main monitoring loop."""
        logger.info(f"Starting monitoring loop with {update_interval}s interval")
        
        while self.is_running:
            try:
                # Get latest model data (would be from production data sources)
                model_data = await self._get_latest_model_data()
                
                if model_data:
                    # Run integrated validation
                    validation_result = await self.orchestrator.run_integrated_validation(model_data)
                    
                    # Process validation result
                    await self._process_validation_result(validation_result)
                    
                    # Check retraining triggers
                    if self.retraining_manager:
                        await self._check_retraining_triggers(validation_result)
                
                # Wait for next interval
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(min(update_interval, 60))  # Back off on error
    
    async def _get_latest_model_data(self) -> Dict[str, Any]:
        """Get latest model data for validation."""
        # In production, this would fetch from:
        # - Model prediction database
        # - Market data feeds
        # - Portfolio management system
        # - Feature engineering pipeline
        
        # For demo/testing, return mock data
        n_samples = 100
        return {
            'predictions': np.random.normal(0.02, 0.1, n_samples),
            'actuals': np.random.normal(0.02, 0.12, n_samples),
            'features': pd.DataFrame({
                'momentum_1m': np.random.normal(0.05, 0.2, n_samples),
                'value_pe': np.random.normal(15.0, 5.0, n_samples),
                'quality_roe': np.random.normal(0.12, 0.05, n_samples)
            }),
            'returns': np.random.normal(0.001, 0.02, n_samples),
            'universe': [f'stock_{i}' for i in range(n_samples)],
            'timestamp': datetime.now()
        }
    
    async def _process_validation_result(self, result: IntegratedValidationResult):
        """Process validation result and trigger callbacks."""
        # Notify callbacks
        for callback in self.validation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Validation callback error: {e}")
        
        # Generate alerts if needed
        if result.overall_status in ['degraded', 'critical']:
            alert_data = {
                'type': 'integrated_validation',
                'severity': result.overall_status,
                'overall_score': result.overall_score,
                'timestamp': result.timestamp,
                'recommendations': result.recommendations
            }
            
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert_data)
                    else:
                        callback(alert_data)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    async def _check_retraining_triggers(self, result: IntegratedValidationResult):
        """Check if retraining should be triggered based on validation results."""
        if not self.retraining_manager:
            return
        
        # Extract drift scores for retraining check
        drift_scores = {
            'feature_drift_score': result.drift_scores.get('feature_drift_score', 0.0),
            'concept_drift_score': result.drift_scores.get('concept_drift_score', 0.0),
            'prediction_drift_score': result.drift_scores.get('prediction_drift_score', 0.0)
        }
        
        # Add performance metrics to retraining manager
        self.retraining_manager.add_performance_sample(
            ic=result.ic_score,
            sharpe=2.0,  # Would be calculated from actual returns
            hit_rate=0.52,  # Would be calculated from predictions vs actuals
            max_drawdown=0.08,  # Would be calculated from returns
            volatility=0.15,  # Would be calculated from returns
            total_return=0.08,  # Would be calculated from cumulative returns
            sample_size=100
        )
        
        # Check retraining triggers
        retraining_result = self.retraining_manager.check_retraining_triggers(drift_scores)
        
        if retraining_result.get('trigger_retraining'):
            logger.warning(f"Retraining triggered: {retraining_result.get('recommendation')}")
            
            # Trigger alert
            alert_data = {
                'type': 'retraining_trigger',
                'severity': 'critical',
                'urgency': retraining_result.get('urgency'),
                'triggers': retraining_result.get('triggers', []),
                'recommendation': retraining_result.get('recommendation'),
                'timestamp': datetime.now()
            }
            
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert_data)
                    else:
                        callback(alert_data)
                except Exception as e:
                    logger.error(f"Retraining alert callback error: {e}")
    
    def add_validation_callback(self, callback: Callable):
        """Add callback for validation results."""
        self.validation_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for alert generation."""
        self.alert_callbacks.append(callback)
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get comprehensive integration summary."""
        orchestrator_status = self.orchestrator.get_integration_status()
        performance_summary = self.orchestrator.get_performance_summary()
        
        return {
            'is_running': self.is_running,
            'monitoring_active': self.monitoring_task is not None and not self.monitoring_task.done(),
            'orchestrator_status': orchestrator_status,
            'performance_summary': performance_summary,
            'components': {
                'statistical_validator': self.orchestrator.statistical_validator is not None,
                'business_validator': self.orchestrator.business_validator is not None,
                'health_monitor': self.orchestrator.health_monitor is not None,
                'alert_system': self.orchestrator.alert_system is not None,
                'retraining_manager': self.retraining_manager is not None
            },
            'callbacks': {
                'validation_callbacks': len(self.validation_callbacks),
                'alert_callbacks': len(self.alert_callbacks)
            }
        }


# Factory function for creating production integration
def create_production_integration(
    model,
    model_path: str,
    data_loader: Callable = None,
    model_trainer: Callable = None,
    model_validator: Callable = None
) -> OperationalIntegrationManager:
    """Create production-ready operational integration system."""
    
    # Create components
    from ..validation.statistical_validator import StatisticalValidator, ValidationConfig
    from ..validation.business_logic.business_validator import BusinessValidator
    from .model_health import ModelHealthMonitor, MonitoringConfig
    from ..alerts.alert_system import create_production_alert_system
    from .retraining_manager import ModelRetrainingManager, RetrainingConfig
    
    # Configuration
    validation_config = ValidationConfig()
    monitoring_config = MonitoringConfig()
    retraining_config = RetrainingConfig()
    
    # Initialize components
    statistical_validator = StatisticalValidator(validation_config)
    business_validator = BusinessValidator()
    health_monitor = ModelHealthMonitor(model, config=monitoring_config)
    alert_system = create_production_alert_system()
    retraining_manager = ModelRetrainingManager(
        config=retraining_config,
        model_path=Path(model_path),
        data_loader=data_loader,
        model_trainer=model_trainer,
        validator=model_validator
    )
    
    # Create integration manager
    integration_manager = OperationalIntegrationManager(
        statistical_validator=statistical_validator,
        business_validator=business_validator,
        health_monitor=health_monitor,
        alert_system=alert_system,
        retraining_manager=retraining_manager
    )
    
    logger.info("Production operational integration system created")
    
    return integration_manager


# Demo function
async def demo_operational_integration():
    """Demonstrate operational integration functionality."""
    print("ML4T Operational Integration Demo")
    
    # Create mock components
    from unittest.mock import Mock
    
    mock_statistical_validator = Mock()
    mock_statistical_validator.comprehensive_validation.return_value = {
        'validation_score': 0.72,
        'ic_score': 0.045,
        'drift_scores': {'feature_drift_score': 0.08},
        'recommendations': ['Monitor feature stability']
    }
    
    mock_business_validator = Mock()
    mock_business_validator.comprehensive_validation.return_value = {
        'overall_score': 0.85,
        'regulatory_compliance': 0.95,
        'risk_management': 0.88,
        'warnings': ['Low volatility exposure']
    }
    
    mock_health_monitor = Mock()
    mock_health_monitor.get_health_report.return_value = {
        'health_score': 78.5,
        'metrics': {'system': {'prediction_success_rate': 0.996}},
        'alerts': {'total_alerts_24h': 2}
    }
    
    # Create integration manager
    integration_manager = OperationalIntegrationManager(
        statistical_validator=mock_statistical_validator,
        business_validator=mock_business_validator,
        health_monitor=mock_health_monitor
    )
    
    # Add callbacks
    validation_results = []
    alerts_received = []
    
    def validation_callback(result):
        validation_results.append(result)
        print(f"Validation result: {result.overall_status} (score: {result.overall_score:.2f})")
    
    def alert_callback(alert):
        alerts_received.append(alert)
        print(f"Alert: {alert['type']} - {alert['severity']}")
    
    integration_manager.add_validation_callback(validation_callback)
    integration_manager.add_alert_callback(alert_callback)
    
    # Test integrated validation
    model_data = {
        'predictions': np.random.normal(0.02, 0.1, 100),
        'actuals': np.random.normal(0.02, 0.12, 100),
        'features': pd.DataFrame({'feature1': np.random.randn(100)}),
        'returns': np.random.normal(0.001, 0.02, 100)
    }
    
    result = await integration_manager.orchestrator.run_integrated_validation(model_data)
    
    print(f"\nIntegration Demo Results:")
    print(f"  Overall Score: {result.overall_score:.3f}")
    print(f"  Overall Status: {result.overall_status}")
    print(f"  Statistical Score: {result.statistical_score:.3f}")
    print(f"  Business Score: {result.business_score:.3f}")
    print(f"  Operational Score: {result.operational_score:.3f}")
    print(f"  Recommendations: {len(result.recommendations)}")
    
    # Get integration summary
    summary = integration_manager.get_integration_summary()
    print(f"\nIntegration Summary:")
    print(f"  Components active: {sum(summary['components'].values())}")
    print(f"  Callbacks registered: {summary['callbacks']['validation_callbacks']}")
    
    print("\nOperational integration demo completed")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_operational_integration())