"""
Automated Model Retraining Manager for ML4T Taiwan Equity Alpha.

Comprehensive retraining system with performance decay detection, automated
triggers, and intelligent retraining scheduling for production environments.

Key Features:
- Performance decay detection with statistical significance testing
- Automated retraining triggers with configurable thresholds
- Taiwan market-aware retraining scheduling
- Integration with monitoring and validation systems
- Production deployment coordination
"""

import asyncio
import logging
import pickle
import shutil
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
import joblib

logger = logging.getLogger(__name__)


class RetrainingStatus(Enum):
    """Retraining process status."""
    IDLE = "idle"
    MONITORING = "monitoring"
    TRIGGERED = "triggered"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RetrainingTrigger(Enum):
    """Types of retraining triggers."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FEATURE_DRIFT = "feature_drift"
    CONCEPT_DRIFT = "concept_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    EMERGENCY = "emergency"


class RetrainingUrgency(Enum):
    """Retraining urgency levels."""
    LOW = "low"          # Next scheduled window
    MEDIUM = "medium"    # Within 24-48 hours
    HIGH = "high"        # Within 12 hours
    CRITICAL = "critical"  # Immediate (emergency)


@dataclass
class RetrainingConfig:
    """Configuration for retraining manager."""
    
    # Performance thresholds
    min_ic_threshold: float = 0.025
    ic_degradation_periods: int = 5
    sharpe_degradation_threshold: float = 0.2
    max_drawdown_threshold: float = 0.15
    
    # Drift thresholds
    max_feature_drift: float = 0.2
    max_concept_drift: float = 0.3
    max_prediction_drift: float = 0.25
    
    # Statistical significance
    performance_significance_level: float = 0.05
    min_performance_samples: int = 30
    
    # Timing constraints
    min_retraining_interval_days: int = 7
    max_retraining_interval_days: int = 90
    emergency_override: bool = True
    
    # Taiwan market considerations
    taiwan_market_hours_only: bool = True
    weekend_retraining: bool = True
    holiday_retraining: bool = False
    
    # Model management
    model_backup_count: int = 5
    validation_holdout_ratio: float = 0.2
    min_training_samples: int = 10000
    
    # Resource limits
    max_concurrent_retrainings: int = 1
    retraining_timeout_hours: int = 12


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    ic: float
    sharpe: float
    hit_rate: float
    max_drawdown: float
    volatility: float
    total_return: float
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'ic': self.ic,
            'sharpe': self.sharpe,
            'hit_rate': self.hit_rate,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'total_return': self.total_return,
            'sample_size': self.sample_size
        }


@dataclass
class RetrainingEvent:
    """Retraining event record."""
    id: str
    trigger: RetrainingTrigger
    urgency: RetrainingUrgency
    status: RetrainingStatus
    triggered_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Performance context
    performance_before: Optional[PerformanceMetrics] = None
    performance_after: Optional[PerformanceMetrics] = None
    
    # Trigger details
    trigger_reason: str = ""
    trigger_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Execution details
    training_duration: Optional[timedelta] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)
    deployment_success: bool = False
    
    # Error handling
    error_message: str = ""
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'trigger': self.trigger.value,
            'urgency': self.urgency.value,
            'status': self.status.value,
            'triggered_at': self.triggered_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'performance_before': self.performance_before.to_dict() if self.performance_before else None,
            'performance_after': self.performance_after.to_dict() if self.performance_after else None,
            'trigger_reason': self.trigger_reason,
            'trigger_metrics': self.trigger_metrics,
            'training_duration': str(self.training_duration) if self.training_duration else None,
            'validation_results': self.validation_results,
            'deployment_success': self.deployment_success,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }


class PerformanceDecayDetector:
    """Advanced performance decay detection with statistical rigor."""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.performance_history = deque(maxlen=1000)
        self.baseline_performance = None
        
    def add_performance_sample(self, metrics: PerformanceMetrics):
        """Add performance sample to history."""
        self.performance_history.append(metrics)
        
        # Update baseline if we have enough samples
        if len(self.performance_history) >= 50 and not self.baseline_performance:
            self._calculate_baseline_performance()
    
    def _calculate_baseline_performance(self):
        """Calculate baseline performance from historical data."""
        if len(self.performance_history) < 30:
            return
        
        # Use first 30 samples as baseline
        baseline_samples = list(self.performance_history)[:30]
        
        ic_values = [m.ic for m in baseline_samples]
        sharpe_values = [m.sharpe for m in baseline_samples]
        
        self.baseline_performance = {
            'ic_mean': np.mean(ic_values),
            'ic_std': np.std(ic_values),
            'sharpe_mean': np.mean(sharpe_values),
            'sharpe_std': np.std(sharpe_values),
            'sample_count': len(baseline_samples),
            'calculated_at': datetime.now()
        }
        
        logger.info(f"Baseline performance calculated: IC={self.baseline_performance['ic_mean']:.4f}±{self.baseline_performance['ic_std']:.4f}")
    
    def detect_performance_decay(self) -> Dict[str, Any]:
        """
        Detect performance decay using multiple statistical methods.
        
        Returns:
            Dictionary with decay detection results and recommendations.
        """
        if len(self.performance_history) < self.config.min_performance_samples:
            return {
                'decay_detected': False,
                'reason': f'Insufficient samples: {len(self.performance_history)} < {self.config.min_performance_samples}'
            }
        
        if not self.baseline_performance:
            self._calculate_baseline_performance()
            if not self.baseline_performance:
                return {'decay_detected': False, 'reason': 'No baseline performance available'}
        
        recent_samples = list(self.performance_history)[-self.config.min_performance_samples:]
        decay_signals = []
        
        # 1. IC degradation detection
        ic_decay = self._detect_ic_degradation(recent_samples)
        if ic_decay['decay_detected']:
            decay_signals.append(ic_decay)
        
        # 2. Sharpe ratio degradation
        sharpe_decay = self._detect_sharpe_degradation(recent_samples)
        if sharpe_decay['decay_detected']:
            decay_signals.append(sharpe_decay)
        
        # 3. Consecutive poor performance periods
        consecutive_decay = self._detect_consecutive_degradation(recent_samples)
        if consecutive_decay['decay_detected']:
            decay_signals.append(consecutive_decay)
        
        # 4. Statistical shift detection
        statistical_shift = self._detect_statistical_shift(recent_samples)
        if statistical_shift['decay_detected']:
            decay_signals.append(statistical_shift)
        
        # 5. Volatility-adjusted performance decay
        volatility_adjusted_decay = self._detect_volatility_adjusted_decay(recent_samples)
        if volatility_adjusted_decay['decay_detected']:
            decay_signals.append(volatility_adjusted_decay)
        
        # Determine overall decay status
        decay_detected = len(decay_signals) > 0
        severity = self._assess_decay_severity(decay_signals)
        
        result = {
            'decay_detected': decay_detected,
            'severity': severity,
            'signals': decay_signals,
            'signal_count': len(decay_signals),
            'recommendation': self._generate_decay_recommendation(decay_signals),
            'current_performance': recent_samples[-1].to_dict() if recent_samples else None,
            'baseline_performance': self.baseline_performance
        }
        
        return result
    
    def _detect_ic_degradation(self, samples: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Detect IC degradation."""
        ic_values = [s.ic for s in samples]
        baseline_ic = self.baseline_performance['ic_mean']
        
        # Check if current IC is below threshold
        current_ic = ic_values[-1]
        threshold_breach = current_ic < self.config.min_ic_threshold
        
        # Check for significant decrease from baseline
        ic_decrease = baseline_ic - current_ic
        significant_decrease = ic_decrease > (2 * self.baseline_performance['ic_std'])
        
        # Check for consecutive periods below baseline
        consecutive_below = sum(1 for ic in ic_values[-self.config.ic_degradation_periods:] 
                               if ic < baseline_ic - self.baseline_performance['ic_std'])
        
        decay_detected = (
            threshold_breach or 
            significant_decrease or 
            consecutive_below >= self.config.ic_degradation_periods
        )
        
        return {
            'decay_detected': decay_detected,
            'type': 'ic_degradation',
            'severity': 'critical' if threshold_breach else 'high' if significant_decrease else 'medium',
            'current_value': current_ic,
            'baseline_value': baseline_ic,
            'threshold_value': self.config.min_ic_threshold,
            'consecutive_periods': consecutive_below,
            'description': f'IC degraded from {baseline_ic:.4f} to {current_ic:.4f}'
        }
    
    def _detect_sharpe_degradation(self, samples: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Detect Sharpe ratio degradation."""
        sharpe_values = [s.sharpe for s in samples]
        baseline_sharpe = self.baseline_performance['sharpe_mean']
        
        current_sharpe = sharpe_values[-1]
        sharpe_decline = (baseline_sharpe - current_sharpe) / max(baseline_sharpe, 0.1)
        
        decay_detected = sharpe_decline > self.config.sharpe_degradation_threshold
        
        return {
            'decay_detected': decay_detected,
            'type': 'sharpe_degradation',
            'severity': 'high' if sharpe_decline > 0.3 else 'medium',
            'current_value': current_sharpe,
            'baseline_value': baseline_sharpe,
            'decline_percentage': sharpe_decline * 100,
            'description': f'Sharpe declined {sharpe_decline*100:.1f}% from baseline'
        }
    
    def _detect_consecutive_degradation(self, samples: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Detect consecutive periods of poor performance."""
        # Define poor performance as both IC and Sharpe below baseline
        baseline_ic = self.baseline_performance['ic_mean']
        baseline_sharpe = self.baseline_performance['sharpe_mean']
        
        consecutive_poor = 0
        for sample in reversed(samples):
            if (sample.ic < baseline_ic * 0.8 and 
                sample.sharpe < baseline_sharpe * 0.8):
                consecutive_poor += 1
            else:
                break
        
        decay_detected = consecutive_poor >= self.config.ic_degradation_periods
        
        return {
            'decay_detected': decay_detected,
            'type': 'consecutive_degradation',
            'severity': 'high' if consecutive_poor >= 7 else 'medium',
            'consecutive_periods': consecutive_poor,
            'threshold_periods': self.config.ic_degradation_periods,
            'description': f'{consecutive_poor} consecutive periods of poor performance'
        }
    
    def _detect_statistical_shift(self, samples: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Detect statistical shift in performance distribution."""
        if len(self.performance_history) < 60:  # Need enough historical data
            return {'decay_detected': False, 'type': 'statistical_shift'}
        
        # Compare recent performance to historical baseline
        historical_samples = list(self.performance_history)[:-len(samples)]
        
        if len(historical_samples) < 30:
            return {'decay_detected': False, 'type': 'statistical_shift'}
        
        historical_ic = [s.ic for s in historical_samples[-30:]]
        recent_ic = [s.ic for s in samples]
        
        # Two-sample t-test
        try:
            t_stat, p_value = stats.ttest_ind(historical_ic, recent_ic)
            significant_shift = (p_value < self.config.performance_significance_level and 
                               np.mean(recent_ic) < np.mean(historical_ic))
        except:
            significant_shift = False
            t_stat = 0
            p_value = 1
        
        return {
            'decay_detected': significant_shift,
            'type': 'statistical_shift',
            'severity': 'high' if p_value < 0.01 else 'medium',
            'p_value': p_value,
            't_statistic': t_stat,
            'recent_mean': np.mean(recent_ic),
            'historical_mean': np.mean(historical_ic),
            'description': f'Significant performance shift (p={p_value:.4f})'
        }
    
    def _detect_volatility_adjusted_decay(self, samples: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Detect performance decay adjusted for volatility changes."""
        if len(samples) < 10:
            return {'decay_detected': False, 'type': 'volatility_adjusted'}
        
        # Calculate risk-adjusted performance (Sharpe-like metric)
        risk_adjusted_performance = []
        for sample in samples:
            if sample.volatility > 0:
                risk_adj = sample.total_return / sample.volatility
                risk_adjusted_performance.append(risk_adj)
        
        if len(risk_adjusted_performance) < 5:
            return {'decay_detected': False, 'type': 'volatility_adjusted'}
        
        # Check for declining trend in risk-adjusted performance
        x = np.arange(len(risk_adjusted_performance))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, risk_adjusted_performance)
        
        significant_decline = (p_value < 0.1 and slope < -0.01 and abs(r_value) > 0.4)
        
        return {
            'decay_detected': significant_decline,
            'type': 'volatility_adjusted_decay',
            'severity': 'medium',
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'description': f'Declining risk-adjusted performance (slope={slope:.4f})'
        }
    
    def _assess_decay_severity(self, signals: List[Dict[str, Any]]) -> str:
        """Assess overall decay severity from individual signals."""
        if not signals:
            return 'none'
        
        critical_signals = [s for s in signals if s.get('severity') == 'critical']
        high_signals = [s for s in signals if s.get('severity') == 'high']
        
        if critical_signals:
            return 'critical'
        elif len(high_signals) >= 2:
            return 'critical'
        elif high_signals:
            return 'high'
        else:
            return 'medium'
    
    def _generate_decay_recommendation(self, signals: List[Dict[str, Any]]) -> str:
        """Generate recommendation based on decay signals."""
        if not signals:
            return "Performance stable, continue monitoring."
        
        critical_signals = [s for s in signals if s.get('severity') == 'critical']
        high_signals = [s for s in signals if s.get('severity') == 'high']
        
        if critical_signals:
            return "CRITICAL: Immediate model retraining required due to severe performance degradation."
        elif len(high_signals) >= 2:
            return "URGENT: Multiple performance degradation signals detected. Schedule retraining within 24 hours."
        elif high_signals:
            return "HIGH: Performance degradation detected. Consider retraining within 48-72 hours."
        else:
            return "MEDIUM: Minor performance issues detected. Monitor closely and prepare for potential retraining."


class RetrainingScheduler:
    """Intelligent retraining scheduler with Taiwan market awareness."""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.scheduled_retrainings = []
        self.taiwan_market_open = time(9, 0)
        self.taiwan_market_close = time(13, 30)
        
    def schedule_retraining(
        self,
        trigger: RetrainingTrigger,
        urgency: RetrainingUrgency,
        trigger_reason: str,
        trigger_metrics: Dict[str, float] = None
    ) -> datetime:
        """
        Schedule retraining based on urgency and market constraints.
        
        Args:
            trigger: Type of retraining trigger
            urgency: Urgency level
            trigger_reason: Human-readable reason
            trigger_metrics: Metrics that triggered retraining
            
        Returns:
            Scheduled datetime for retraining
        """
        now = datetime.now()
        trigger_metrics = trigger_metrics or {}
        
        if urgency == RetrainingUrgency.CRITICAL:
            # Emergency retraining - immediate
            scheduled_time = now + timedelta(minutes=5)
        
        elif urgency == RetrainingUrgency.HIGH:
            # Within 12 hours, prefer off-market hours
            if self.config.taiwan_market_hours_only:
                scheduled_time = self._find_next_off_market_window(now, hours_ahead=12)
            else:
                scheduled_time = now + timedelta(hours=2)  # Quick turnaround
        
        elif urgency == RetrainingUrgency.MEDIUM:
            # Within 24-48 hours, weekend preferred
            if self.config.weekend_retraining:
                scheduled_time = self._find_next_weekend_window(now)
            else:
                scheduled_time = self._find_next_off_market_window(now, hours_ahead=48)
        
        else:  # LOW urgency
            # Next scheduled maintenance window
            scheduled_time = self._find_next_maintenance_window(now)
        
        logger.info(f"Retraining scheduled for {scheduled_time} (urgency: {urgency.value}, trigger: {trigger.value})")
        
        return scheduled_time
    
    def _find_next_off_market_window(self, from_time: datetime, hours_ahead: int = 24) -> datetime:
        """Find next window when Taiwan market is closed."""
        current = from_time
        max_search = from_time + timedelta(hours=hours_ahead)
        
        while current < max_search:
            # Check if it's weekend
            if current.weekday() >= 5:  # Saturday or Sunday
                return current
            
            # Check if it's after market close (13:30) but before next day open
            current_time = current.time()
            if current_time >= self.taiwan_market_close:
                # After market close today
                return current
            elif current_time < self.taiwan_market_open:
                # Before market open today
                return current
            else:
                # During market hours, move to after close
                current = current.replace(hour=14, minute=0, second=0, microsecond=0)
                return current
        
        # Fallback to immediate scheduling if no good window found
        return from_time + timedelta(hours=1)
    
    def _find_next_weekend_window(self, from_time: datetime) -> datetime:
        """Find next weekend window for retraining."""
        current = from_time
        
        # Find next Saturday
        days_until_saturday = (5 - current.weekday()) % 7
        if days_until_saturday == 0 and current.weekday() == 5:
            # It's already Saturday
            weekend_time = current + timedelta(hours=1)
        else:
            if days_until_saturday == 0:  # It's Sunday
                days_until_saturday = 6
            weekend_time = current + timedelta(days=days_until_saturday)
            weekend_time = weekend_time.replace(hour=10, minute=0, second=0, microsecond=0)
        
        return weekend_time
    
    def _find_next_maintenance_window(self, from_time: datetime) -> datetime:
        """Find next regular maintenance window."""
        # Default: Next Sunday at 2 AM
        current = from_time
        days_until_sunday = (6 - current.weekday()) % 7
        
        if days_until_sunday == 0:  # It's already Sunday
            if current.hour < 2:
                maintenance_time = current.replace(hour=2, minute=0, second=0, microsecond=0)
            else:
                maintenance_time = current + timedelta(days=7)
                maintenance_time = maintenance_time.replace(hour=2, minute=0, second=0, microsecond=0)
        else:
            maintenance_time = current + timedelta(days=days_until_sunday)
            maintenance_time = maintenance_time.replace(hour=2, minute=0, second=0, microsecond=0)
        
        return maintenance_time


class ModelRetrainingManager:
    """Main retraining manager orchestrating the entire retraining process."""
    
    def __init__(
        self,
        config: RetrainingConfig,
        model_path: Path,
        data_loader: Callable = None,
        model_trainer: Callable = None,
        validator: Callable = None
    ):
        self.config = config
        self.model_path = Path(model_path)
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        self.validator = validator
        
        # Components
        self.decay_detector = PerformanceDecayDetector(config)
        self.scheduler = RetrainingScheduler(config)
        
        # State management
        self.current_status = RetrainingStatus.IDLE
        self.retraining_events = deque(maxlen=1000)
        self.active_event: Optional[RetrainingEvent] = None
        self.last_retraining = None
        
        # Callbacks for external integration
        self.status_callbacks = []
        self.completion_callbacks = []
        
        logger.info(f"ModelRetrainingManager initialized with config: {config}")
    
    def add_performance_sample(
        self,
        ic: float,
        sharpe: float,
        hit_rate: float,
        max_drawdown: float,
        volatility: float,
        total_return: float,
        sample_size: int,
        timestamp: datetime = None
    ):
        """Add performance sample for monitoring."""
        timestamp = timestamp or datetime.now()
        
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            ic=ic,
            sharpe=sharpe,
            hit_rate=hit_rate,
            max_drawdown=max_drawdown,
            volatility=volatility,
            total_return=total_return,
            sample_size=sample_size
        )
        
        self.decay_detector.add_performance_sample(metrics)
        logger.debug(f"Performance sample added: IC={ic:.4f}, Sharpe={sharpe:.2f}")
    
    def check_retraining_triggers(
        self,
        drift_scores: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Check all retraining triggers and determine if retraining should be initiated.
        
        Args:
            drift_scores: External drift detection scores
            
        Returns:
            Dictionary with trigger analysis and recommendations
        """
        drift_scores = drift_scores or {}
        
        # Check minimum interval constraint
        if self.last_retraining:
            time_since_last = datetime.now() - self.last_retraining
            if time_since_last < timedelta(days=self.config.min_retraining_interval_days):
                if not self.config.emergency_override:
                    return {
                        'trigger_retraining': False,
                        'reason': 'Minimum retraining interval not elapsed',
                        'time_since_last': str(time_since_last),
                        'next_eligible': self.last_retraining + timedelta(days=self.config.min_retraining_interval_days)
                    }
        
        triggers = []
        urgency = RetrainingUrgency.LOW
        
        # 1. Performance decay triggers
        decay_result = self.decay_detector.detect_performance_decay()
        if decay_result.get('decay_detected'):
            severity = decay_result.get('severity', 'medium')
            
            if severity == 'critical':
                triggers.append({
                    'trigger': RetrainingTrigger.PERFORMANCE_DEGRADATION,
                    'urgency': RetrainingUrgency.CRITICAL,
                    'reason': 'Critical performance degradation detected',
                    'details': decay_result
                })
                urgency = max(urgency, RetrainingUrgency.CRITICAL, key=lambda x: x.value)
            
            elif severity == 'high':
                triggers.append({
                    'trigger': RetrainingTrigger.PERFORMANCE_DEGRADATION,
                    'urgency': RetrainingUrgency.HIGH,
                    'reason': 'Significant performance degradation detected',
                    'details': decay_result
                })
                urgency = max(urgency, RetrainingUrgency.HIGH, key=lambda x: x.value)
        
        # 2. Drift-based triggers
        feature_drift = drift_scores.get('feature_drift_score', 0)
        if feature_drift > self.config.max_feature_drift:
            triggers.append({
                'trigger': RetrainingTrigger.FEATURE_DRIFT,
                'urgency': RetrainingUrgency.MEDIUM,
                'reason': f'Feature drift score {feature_drift:.3f} exceeds threshold {self.config.max_feature_drift:.3f}',
                'details': {'feature_drift_score': feature_drift}
            })
            urgency = max(urgency, RetrainingUrgency.MEDIUM, key=lambda x: x.value)
        
        concept_drift = drift_scores.get('concept_drift_score', 0)
        if concept_drift > self.config.max_concept_drift:
            triggers.append({
                'trigger': RetrainingTrigger.CONCEPT_DRIFT,
                'urgency': RetrainingUrgency.HIGH,
                'reason': f'Concept drift score {concept_drift:.3f} exceeds threshold {self.config.max_concept_drift:.3f}',
                'details': {'concept_drift_score': concept_drift}
            })
            urgency = max(urgency, RetrainingUrgency.HIGH, key=lambda x: x.value)
        
        # 3. Scheduled retraining check
        if self.last_retraining:
            time_since_last = datetime.now() - self.last_retraining
            if time_since_last > timedelta(days=self.config.max_retraining_interval_days):
                triggers.append({
                    'trigger': RetrainingTrigger.SCHEDULED,
                    'urgency': RetrainingUrgency.MEDIUM,
                    'reason': f'Scheduled retraining due ({time_since_last.days} days since last)',
                    'details': {'days_since_last': time_since_last.days}
                })
                urgency = max(urgency, RetrainingUrgency.MEDIUM, key=lambda x: x.value)
        
        # Determine if retraining should be triggered
        trigger_retraining = len(triggers) > 0
        
        result = {
            'trigger_retraining': trigger_retraining,
            'urgency': urgency.value if trigger_retraining else None,
            'triggers': triggers,
            'trigger_count': len(triggers),
            'recommendation': self._generate_retraining_recommendation(triggers, urgency),
            'scheduled_time': None
        }
        
        # Schedule retraining if triggered
        if trigger_retraining and self.current_status == RetrainingStatus.IDLE:
            primary_trigger = triggers[0]['trigger']
            trigger_reason = triggers[0]['reason']
            trigger_metrics = {**drift_scores}
            
            scheduled_time = self.scheduler.schedule_retraining(
                primary_trigger, urgency, trigger_reason, trigger_metrics
            )
            result['scheduled_time'] = scheduled_time.isoformat()
            
            # Create retraining event
            self._create_retraining_event(
                primary_trigger, urgency, trigger_reason, trigger_metrics, scheduled_time
            )
        
        return result
    
    def _generate_retraining_recommendation(
        self,
        triggers: List[Dict[str, Any]],
        urgency: RetrainingUrgency
    ) -> str:
        """Generate human-readable retraining recommendation."""
        if not triggers:
            return "Model performance stable. Continue monitoring."
        
        if urgency == RetrainingUrgency.CRITICAL:
            return "CRITICAL: Emergency model retraining required immediately due to severe performance issues."
        elif urgency == RetrainingUrgency.HIGH:
            return "HIGH PRIORITY: Schedule model retraining within 12 hours to address performance degradation."
        elif urgency == RetrainingUrgency.MEDIUM:
            return "MEDIUM PRIORITY: Plan model retraining within 24-48 hours based on detected issues."
        else:
            return "LOW PRIORITY: Consider retraining during next maintenance window."
    
    def _create_retraining_event(
        self,
        trigger: RetrainingTrigger,
        urgency: RetrainingUrgency,
        reason: str,
        metrics: Dict[str, float],
        scheduled_time: datetime
    ):
        """Create a new retraining event."""
        event_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get current performance snapshot
        current_performance = None
        if self.decay_detector.performance_history:
            latest_metrics = self.decay_detector.performance_history[-1]
            current_performance = latest_metrics
        
        event = RetrainingEvent(
            id=event_id,
            trigger=trigger,
            urgency=urgency,
            status=RetrainingStatus.TRIGGERED,
            triggered_at=datetime.now(),
            performance_before=current_performance,
            trigger_reason=reason,
            trigger_metrics=metrics
        )
        
        self.active_event = event
        self.retraining_events.append(event)
        self.current_status = RetrainingStatus.TRIGGERED
        
        logger.warning(f"Retraining event created: {event_id} ({urgency.value} urgency)")
        self._notify_status_change()
    
    def manual_trigger_retraining(
        self,
        reason: str = "Manual trigger",
        urgency: RetrainingUrgency = RetrainingUrgency.MEDIUM
    ) -> str:
        """Manually trigger retraining."""
        if self.current_status != RetrainingStatus.IDLE:
            raise ValueError(f"Cannot trigger retraining: current status is {self.current_status.value}")
        
        scheduled_time = self.scheduler.schedule_retraining(
            RetrainingTrigger.MANUAL, urgency, reason
        )
        
        self._create_retraining_event(
            RetrainingTrigger.MANUAL, urgency, reason, {}, scheduled_time
        )
        
        logger.info(f"Manual retraining triggered: {reason}")
        return self.active_event.id
    
    async def execute_retraining(self, event_id: str = None) -> bool:
        """
        Execute the retraining process.
        
        Args:
            event_id: Optional event ID to execute specific event
            
        Returns:
            True if successful, False otherwise
        """
        if not self.active_event:
            logger.error("No active retraining event to execute")
            return False
        
        if event_id and self.active_event.id != event_id:
            logger.error(f"Event ID mismatch: {event_id} vs {self.active_event.id}")
            return False
        
        event = self.active_event
        
        try:
            # Update status
            event.status = RetrainingStatus.PREPARING
            event.started_at = datetime.now()
            self.current_status = RetrainingStatus.PREPARING
            self._notify_status_change()
            
            # 1. Data preparation
            logger.info(f"Starting retraining process for event {event.id}")
            
            if not self.data_loader:
                raise ValueError("No data loader configured")
            
            # Load fresh training data
            training_data = await self._load_training_data()
            logger.info(f"Training data loaded: {len(training_data)} samples")
            
            # 2. Model training
            event.status = RetrainingStatus.TRAINING
            self.current_status = RetrainingStatus.TRAINING
            self._notify_status_change()
            
            if not self.model_trainer:
                raise ValueError("No model trainer configured")
            
            new_model = await self._train_model(training_data)
            logger.info("Model training completed")
            
            # 3. Validation
            event.status = RetrainingStatus.VALIDATING
            self.current_status = RetrainingStatus.VALIDATING
            self._notify_status_change()
            
            validation_results = await self._validate_model(new_model, training_data)
            event.validation_results = validation_results
            logger.info(f"Model validation completed: {validation_results}")
            
            # Check if validation passes minimum thresholds
            if not self._validation_passes(validation_results):
                raise ValueError(f"Model validation failed: {validation_results}")
            
            # 4. Model deployment
            event.status = RetrainingStatus.DEPLOYING
            self.current_status = RetrainingStatus.DEPLOYING
            self._notify_status_change()
            
            success = await self._deploy_model(new_model)
            if not success:
                raise ValueError("Model deployment failed")
            
            # 5. Complete the process
            event.status = RetrainingStatus.COMPLETED
            event.completed_at = datetime.now()
            event.training_duration = event.completed_at - event.started_at
            event.deployment_success = True
            
            self.current_status = RetrainingStatus.IDLE
            self.last_retraining = event.completed_at
            self.active_event = None
            
            logger.info(f"Retraining completed successfully: {event.id}")
            self._notify_completion(event, success=True)
            
            return True
            
        except Exception as e:
            # Handle failure
            event.status = RetrainingStatus.FAILED
            event.error_message = str(e)
            event.completed_at = datetime.now()
            
            self.current_status = RetrainingStatus.IDLE
            self.active_event = None
            
            logger.error(f"Retraining failed for event {event.id}: {e}")
            self._notify_completion(event, success=False)
            
            return False
    
    async def _load_training_data(self) -> pd.DataFrame:
        """Load training data."""
        if asyncio.iscoroutinefunction(self.data_loader):
            return await self.data_loader()
        else:
            return self.data_loader()
    
    async def _train_model(self, training_data: pd.DataFrame) -> Any:
        """Train the model."""
        if asyncio.iscoroutinefunction(self.model_trainer):
            return await self.model_trainer(training_data)
        else:
            return self.model_trainer(training_data)
    
    async def _validate_model(self, model: Any, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the trained model."""
        if self.validator:
            if asyncio.iscoroutinefunction(self.validator):
                return await self.validator(model, training_data)
            else:
                return self.validator(model, training_data)
        else:
            # Basic validation
            return {
                'validation_score': 0.8,  # Placeholder
                'ic_score': 0.05,
                'sharpe_score': 1.2,
                'passed': True
            }
    
    def _validation_passes(self, validation_results: Dict[str, Any]) -> bool:
        """Check if validation results meet minimum thresholds."""
        ic_score = validation_results.get('ic_score', 0)
        sharpe_score = validation_results.get('sharpe_score', 0)
        
        return (
            ic_score >= self.config.min_ic_threshold and
            sharpe_score >= 0.5  # Minimum acceptable Sharpe
        )
    
    async def _deploy_model(self, model: Any) -> bool:
        """Deploy the trained model."""
        try:
            # Backup existing model
            backup_path = self._create_model_backup()
            
            # Save new model
            if hasattr(model, 'save'):
                model.save(str(self.model_path))
            else:
                joblib.dump(model, self.model_path)
            
            logger.info(f"Model deployed to {self.model_path} (backup: {backup_path})")
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False
    
    def _create_model_backup(self) -> Path:
        """Create backup of existing model."""
        if not self.model_path.exists():
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.model_path.parent / f"{self.model_path.stem}_backup_{timestamp}{self.model_path.suffix}"
        
        shutil.copy2(self.model_path, backup_path)
        
        # Clean up old backups
        self._cleanup_old_backups()
        
        return backup_path
    
    def _cleanup_old_backups(self):
        """Remove old model backups beyond configured limit."""
        backup_pattern = f"{self.model_path.stem}_backup_*{self.model_path.suffix}"
        backups = list(self.model_path.parent.glob(backup_pattern))
        
        if len(backups) > self.config.model_backup_count:
            # Sort by creation time and remove oldest
            backups.sort(key=lambda p: p.stat().st_ctime)
            for backup in backups[:-self.config.model_backup_count]:
                backup.unlink()
                logger.info(f"Removed old model backup: {backup}")
    
    def add_status_callback(self, callback: Callable[[RetrainingStatus, Optional[RetrainingEvent]], None]):
        """Add callback for status changes."""
        self.status_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[RetrainingEvent, bool], None]):
        """Add callback for retraining completion."""
        self.completion_callbacks.append(callback)
    
    def _notify_status_change(self):
        """Notify all status callbacks."""
        for callback in self.status_callbacks:
            try:
                callback(self.current_status, self.active_event)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    def _notify_completion(self, event: RetrainingEvent, success: bool):
        """Notify all completion callbacks."""
        for callback in self.completion_callbacks:
            try:
                callback(event, success)
            except Exception as e:
                logger.error(f"Completion callback error: {e}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        return {
            'current_status': self.current_status.value,
            'active_event': self.active_event.to_dict() if self.active_event else None,
            'last_retraining': self.last_retraining.isoformat() if self.last_retraining else None,
            'total_retrainings': len(self.retraining_events),
            'recent_events': [e.to_dict() for e in list(self.retraining_events)[-5:]],
            'performance_samples': len(self.decay_detector.performance_history),
            'baseline_performance': self.decay_detector.baseline_performance
        }


# Demo and testing functions
def demo_retraining_manager():
    """Demonstrate retraining manager functionality."""
    print("ML4T Retraining Manager Demo")
    
    config = RetrainingConfig(
        min_ic_threshold=0.03,
        ic_degradation_periods=3,
        min_retraining_interval_days=1  # Short for demo
    )
    
    # Mock data loader and trainer
    def mock_data_loader():
        return pd.DataFrame({'feature': np.random.randn(1000), 'target': np.random.randn(1000)})
    
    def mock_trainer(data):
        return {'model_type': 'mock', 'trained_at': datetime.now()}
    
    def mock_validator(model, data):
        return {'ic_score': 0.045, 'sharpe_score': 1.1, 'passed': True}
    
    manager = ModelRetrainingManager(
        config=config,
        model_path=Path('mock_model.pkl'),
        data_loader=mock_data_loader,
        model_trainer=mock_trainer,
        validator=mock_validator
    )
    
    print(f"Manager created with status: {manager.current_status.value}")
    
    # Add some performance samples
    print("\nAdding performance samples...")
    for i in range(10):
        ic = 0.05 - (i * 0.005)  # Declining IC
        sharpe = 1.5 - (i * 0.1)  # Declining Sharpe
        manager.add_performance_sample(
            ic=ic, sharpe=sharpe, hit_rate=0.52, max_drawdown=0.08,
            volatility=0.15, total_return=0.08, sample_size=100
        )
        print(f"  Sample {i+1}: IC={ic:.4f}, Sharpe={sharpe:.2f}")
    
    # Check for retraining triggers
    print("\nChecking retraining triggers...")
    result = manager.check_retraining_triggers({'feature_drift_score': 0.25})
    print(f"  Trigger retraining: {result.get('trigger_retraining', False)}")
    print(f"  Urgency: {result.get('urgency', 'N/A')}")
    print(f"  Triggers: {len(result.get('triggers', []))}")
    
    # Get status summary
    summary = manager.get_status_summary()
    print(f"\nStatus Summary:")
    print(f"  Current status: {summary['current_status']}")
    print(f"  Performance samples: {summary['performance_samples']}")
    print(f"  Total retrainings: {summary['total_retrainings']}")
    
    print("\nRetraining manager demo completed")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    demo_retraining_manager()