"""
Model Performance Metrics and Tracking System

This module provides comprehensive performance tracking capabilities for ML models
in Taiwan equity markets, including financial metrics, statistical validation,
and real-time monitoring capabilities.
"""

import logging
import warnings
import json
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    FINANCIAL = "financial"
    STATISTICAL = "statistical"
    RISK = "risk"
    TAIWAN_SPECIFIC = "taiwan_specific"


class MetricFrequency(Enum):
    """Metric calculation frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    REAL_TIME = "real_time"


@dataclass
class PerformanceMetric:
    """Individual performance metric definition."""
    
    name: str
    metric_type: PerformanceMetricType
    frequency: MetricFrequency
    description: str
    calculation_func: Callable
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    higher_is_better: bool = True
    
    # Taiwan market specific parameters
    use_trading_days_only: bool = True
    adjust_for_settlement: bool = True
    market_hours_only: bool = False


@dataclass
class PerformanceResult:
    """Results from performance metric calculation."""
    
    metric_name: str
    value: float
    timestamp: datetime
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    # Status indicators
    is_target_met: Optional[bool] = None
    warning_triggered: bool = False
    critical_triggered: bool = False
    
    # Additional metadata
    sample_size: Optional[int] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    
    # Taiwan market context
    trading_days_in_period: Optional[int] = None
    market_regime: Optional[str] = None


@dataclass
class ModelPerformanceSnapshot:
    """Complete performance snapshot for a model at a point in time."""
    
    model_id: str
    timestamp: datetime
    evaluation_period: Tuple[datetime, datetime]
    
    # Core metrics
    financial_metrics: Dict[str, PerformanceResult] = field(default_factory=dict)
    statistical_metrics: Dict[str, PerformanceResult] = field(default_factory=dict)
    risk_metrics: Dict[str, PerformanceResult] = field(default_factory=dict)
    taiwan_metrics: Dict[str, PerformanceResult] = field(default_factory=dict)
    
    # Performance summary
    overall_score: Optional[float] = None
    performance_grade: Optional[str] = None  # A, B, C, D, F
    meets_targets: bool = False
    warning_count: int = 0
    critical_count: int = 0
    
    # Model metadata
    model_version: Optional[str] = None
    data_vintage: Optional[str] = None
    feature_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_id': self.model_id,
            'timestamp': self.timestamp.isoformat(),
            'evaluation_period': [
                self.evaluation_period[0].isoformat(),
                self.evaluation_period[1].isoformat()
            ],
            'financial_metrics': {
                k: {
                    'value': v.value,
                    'is_target_met': v.is_target_met,
                    'warning_triggered': v.warning_triggered,
                    'critical_triggered': v.critical_triggered
                } for k, v in self.financial_metrics.items()
            },
            'statistical_metrics': {
                k: {
                    'value': v.value,
                    'is_target_met': v.is_target_met
                } for k, v in self.statistical_metrics.items()
            },
            'risk_metrics': {
                k: {
                    'value': v.value,
                    'is_target_met': v.is_target_met
                } for k, v in self.risk_metrics.items()
            },
            'taiwan_metrics': {
                k: {
                    'value': v.value,
                    'is_target_met': v.is_target_met
                } for k, v in self.taiwan_metrics.items()
            },
            'overall_score': self.overall_score,
            'performance_grade': self.performance_grade,
            'meets_targets': self.meets_targets,
            'warning_count': self.warning_count,
            'critical_count': self.critical_count
        }


class TaiwanMarketMetrics:
    """
    Taiwan market specific performance metrics.
    
    These metrics account for Taiwan market microstructure, trading patterns,
    and regulatory environment.
    """
    
    @staticmethod
    def information_coefficient(
        predictions: pd.Series, 
        returns: pd.Series,
        method: str = 'spearman'
    ) -> float:
        """
        Calculate Information Coefficient (IC).
        
        Args:
            predictions: Model predictions
            returns: Actual forward returns
            method: 'spearman' (rank IC) or 'pearson' (normal IC)
            
        Returns:
            Information coefficient
        """
        if len(predictions) != len(returns):
            raise ValueError("Predictions and returns must have same length")
        
        # Remove NaN values
        valid_mask = ~(predictions.isna() | returns.isna())
        pred_clean = predictions[valid_mask]
        ret_clean = returns[valid_mask]
        
        if len(pred_clean) < 10:
            logger.warning("Insufficient data for IC calculation")
            return 0.0
        
        try:
            if method == 'spearman':
                ic, _ = stats.spearmanr(pred_clean, ret_clean)
            else:
                ic, _ = stats.pearsonr(pred_clean, ret_clean)
                
            return ic if not np.isnan(ic) else 0.0
        except Exception as e:
            logger.error(f"IC calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def information_ratio(
        predictions: pd.Series,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            predictions: Model predictions
            returns: Actual returns
            benchmark_returns: Benchmark returns (if None, uses zero)
            
        Returns:
            Information ratio
        """
        if benchmark_returns is None:
            benchmark_returns = pd.Series(0.0, index=returns.index)
        
        # Calculate excess returns
        excess_returns = returns - benchmark_returns
        
        # Calculate IC time series (rolling)
        ic_series = []
        window_size = min(60, len(predictions) // 4)  # Use 60 days or 25% of data
        
        for i in range(window_size, len(predictions)):
            window_pred = predictions.iloc[i-window_size:i]
            window_ret = excess_returns.iloc[i-window_size:i]
            
            ic = TaiwanMarketMetrics.information_coefficient(window_pred, window_ret)
            ic_series.append(ic)
        
        if not ic_series:
            return 0.0
        
        ic_series = pd.Series(ic_series)
        
        # Information Ratio = mean(IC) / std(IC) * sqrt(252)
        if ic_series.std() > 0:
            ir = ic_series.mean() / ic_series.std() * np.sqrt(252)
            return ir if not np.isnan(ir) else 0.0
        
        return 0.0
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.01,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe ratio for Taiwan market.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annual)
            annualize: Whether to annualize the result
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Convert annual risk-free rate to daily
        if annualize:
            daily_rf = risk_free_rate / 252
        else:
            daily_rf = risk_free_rate
        
        excess_returns = returns - daily_rf
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std()
        
        if annualize:
            sharpe *= np.sqrt(252)
        
        return sharpe if not np.isnan(sharpe) else 0.0
    
    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Return series
            
        Returns:
            Maximum drawdown (positive value)
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown series
        drawdown = (cumulative - running_max) / running_max
        
        # Return maximum drawdown as positive value
        max_dd = abs(drawdown.min())
        
        return max_dd if not np.isnan(max_dd) else 0.0
    
    @staticmethod
    def hit_rate(
        predictions: pd.Series,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate directional accuracy (hit rate).
        
        Args:
            predictions: Model predictions
            returns: Actual returns
            threshold: Threshold for considering a prediction correct
            
        Returns:
            Hit rate (0-1)
        """
        if len(predictions) != len(returns):
            raise ValueError("Predictions and returns must have same length")
        
        # Remove NaN values
        valid_mask = ~(predictions.isna() | returns.isna())
        pred_clean = predictions[valid_mask]
        ret_clean = returns[valid_mask]
        
        if len(pred_clean) == 0:
            return 0.0
        
        # Calculate directional accuracy
        pred_direction = (pred_clean > threshold).astype(int)
        actual_direction = (ret_clean > threshold).astype(int)
        
        hit_rate = (pred_direction == actual_direction).mean()
        
        return hit_rate if not np.isnan(hit_rate) else 0.0
    
    @staticmethod
    def turnover_adjusted_return(
        returns: pd.Series,
        positions: pd.Series,
        transaction_cost: float = 0.003  # 30bps for Taiwan
    ) -> pd.Series:
        """
        Calculate returns adjusted for trading costs.
        
        Args:
            returns: Raw returns
            positions: Position weights
            transaction_cost: Transaction cost rate
            
        Returns:
            Adjusted returns series
        """
        # Calculate position changes (turnover)
        position_changes = positions.diff().abs()
        
        # Calculate transaction costs
        costs = position_changes * transaction_cost
        
        # Adjust returns for costs
        adjusted_returns = returns - costs
        
        return adjusted_returns
    
    @staticmethod
    def taiwan_market_beta(
        returns: pd.Series,
        market_returns: pd.Series,
        window_days: int = 252
    ) -> pd.Series:
        """
        Calculate rolling beta to Taiwan market.
        
        Args:
            returns: Strategy returns
            market_returns: TAIEX returns
            window_days: Rolling window size
            
        Returns:
            Rolling beta series
        """
        # Align series
        aligned_data = pd.DataFrame({
            'strategy': returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < window_days:
            logger.warning(f"Insufficient data for beta calculation: {len(aligned_data)} < {window_days}")
            return pd.Series(1.0, index=returns.index)
        
        # Calculate rolling beta
        betas = []
        for i in range(window_days, len(aligned_data)):
            window_data = aligned_data.iloc[i-window_days:i]
            
            try:
                beta = np.cov(window_data['strategy'], window_data['market'])[0, 1] / np.var(window_data['market'])
                betas.append(beta)
            except:
                betas.append(1.0)  # Default beta
        
        # Create series with proper index
        beta_index = aligned_data.index[window_days:]
        beta_series = pd.Series(betas, index=beta_index)
        
        # Reindex to match original returns
        return beta_series.reindex(returns.index, method='ffill').fillna(1.0)


class ModelPerformanceTracker:
    """
    Comprehensive model performance tracking system.
    
    This class provides real-time monitoring, historical tracking,
    and alert generation for ML model performance.
    """
    
    def __init__(self, model_id: str):
        """
        Initialize performance tracker.
        
        Args:
            model_id: Unique identifier for the model
        """
        self.model_id = model_id
        self.metrics_registry: Dict[str, PerformanceMetric] = {}
        self.performance_history: List[ModelPerformanceSnapshot] = []
        
        # Register default metrics
        self._register_default_metrics()
        
        logger.info(f"ModelPerformanceTracker initialized for model {model_id}")
    
    def _register_default_metrics(self) -> None:
        """Register default performance metrics."""
        
        # Financial metrics
        self.register_metric(PerformanceMetric(
            name="information_coefficient",
            metric_type=PerformanceMetricType.FINANCIAL,
            frequency=MetricFrequency.DAILY,
            description="Information Coefficient (rank correlation with returns)",
            calculation_func=TaiwanMarketMetrics.information_coefficient,
            target_value=0.05,
            threshold_warning=0.03,
            threshold_critical=0.01,
            higher_is_better=True
        ))
        
        self.register_metric(PerformanceMetric(
            name="information_ratio",
            metric_type=PerformanceMetricType.FINANCIAL,
            frequency=MetricFrequency.WEEKLY,
            description="Information Ratio (risk-adjusted IC)",
            calculation_func=TaiwanMarketMetrics.information_ratio,
            target_value=0.8,
            threshold_warning=0.5,
            threshold_critical=0.3,
            higher_is_better=True
        ))
        
        self.register_metric(PerformanceMetric(
            name="sharpe_ratio",
            metric_type=PerformanceMetricType.RISK,
            frequency=MetricFrequency.WEEKLY,
            description="Sharpe Ratio (risk-adjusted returns)",
            calculation_func=TaiwanMarketMetrics.sharpe_ratio,
            target_value=2.0,
            threshold_warning=1.5,
            threshold_critical=1.0,
            higher_is_better=True
        ))
        
        self.register_metric(PerformanceMetric(
            name="maximum_drawdown",
            metric_type=PerformanceMetricType.RISK,
            frequency=MetricFrequency.DAILY,
            description="Maximum Drawdown",
            calculation_func=TaiwanMarketMetrics.maximum_drawdown,
            target_value=0.15,
            threshold_warning=0.20,
            threshold_critical=0.25,
            higher_is_better=False
        ))
        
        self.register_metric(PerformanceMetric(
            name="hit_rate",
            metric_type=PerformanceMetricType.STATISTICAL,
            frequency=MetricFrequency.DAILY,
            description="Directional Accuracy",
            calculation_func=TaiwanMarketMetrics.hit_rate,
            target_value=0.52,
            threshold_warning=0.50,
            threshold_critical=0.48,
            higher_is_better=True
        ))
        
        # Taiwan specific metrics
        self.register_metric(PerformanceMetric(
            name="rmse",
            metric_type=PerformanceMetricType.STATISTICAL,
            frequency=MetricFrequency.DAILY,
            description="Root Mean Square Error",
            calculation_func=lambda pred, actual: np.sqrt(mean_squared_error(actual, pred)),
            threshold_warning=0.05,
            threshold_critical=0.08,
            higher_is_better=False
        ))
    
    def register_metric(self, metric: PerformanceMetric) -> None:
        """Register a new performance metric."""
        self.metrics_registry[metric.name] = metric
        logger.debug(f"Registered metric: {metric.name}")
    
    def calculate_performance(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        positions: Optional[pd.Series] = None,
        evaluation_start: Optional[datetime] = None,
        evaluation_end: Optional[datetime] = None
    ) -> ModelPerformanceSnapshot:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            predictions: Model predictions
            actual_returns: Actual forward returns
            benchmark_returns: Benchmark returns (optional)
            positions: Position sizes (optional)
            evaluation_start: Start of evaluation period
            evaluation_end: End of evaluation period
            
        Returns:
            Complete performance snapshot
        """
        timestamp = datetime.now()
        
        if evaluation_start is None:
            evaluation_start = predictions.index[0] if hasattr(predictions.index[0], 'to_pydatetime') else timestamp
        if evaluation_end is None:
            evaluation_end = predictions.index[-1] if hasattr(predictions.index[-1], 'to_pydatetime') else timestamp
        
        logger.info(f"Calculating performance for period {evaluation_start} to {evaluation_end}")
        
        # Initialize snapshot
        snapshot = ModelPerformanceSnapshot(
            model_id=self.model_id,
            timestamp=timestamp,
            evaluation_period=(evaluation_start, evaluation_end)
        )
        
        # Calculate each registered metric
        for metric_name, metric in self.metrics_registry.items():
            try:
                result = self._calculate_single_metric(
                    metric, predictions, actual_returns, benchmark_returns, positions
                )
                
                # Assign to appropriate category
                if metric.metric_type == PerformanceMetricType.FINANCIAL:
                    snapshot.financial_metrics[metric_name] = result
                elif metric.metric_type == PerformanceMetricType.STATISTICAL:
                    snapshot.statistical_metrics[metric_name] = result
                elif metric.metric_type == PerformanceMetricType.RISK:
                    snapshot.risk_metrics[metric_name] = result
                elif metric.metric_type == PerformanceMetricType.TAIWAN_SPECIFIC:
                    snapshot.taiwan_metrics[metric_name] = result
                
                # Update snapshot flags
                if result.warning_triggered:
                    snapshot.warning_count += 1
                if result.critical_triggered:
                    snapshot.critical_count += 1
                
            except Exception as e:
                logger.error(f"Failed to calculate metric {metric_name}: {e}")
        
        # Calculate overall performance score and grade
        snapshot.overall_score = self._calculate_overall_score(snapshot)
        snapshot.performance_grade = self._assign_performance_grade(snapshot.overall_score)
        snapshot.meets_targets = self._check_targets_met(snapshot)
        
        # Store snapshot
        self.performance_history.append(snapshot)
        
        # Limit history size (keep last 1000 snapshots)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        logger.info(f"Performance calculated: Score {snapshot.overall_score:.3f}, Grade {snapshot.performance_grade}")
        
        return snapshot
    
    def _calculate_single_metric(
        self,
        metric: PerformanceMetric,
        predictions: pd.Series,
        actual_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        positions: Optional[pd.Series] = None
    ) -> PerformanceResult:
        """Calculate a single performance metric."""
        
        # Prepare arguments based on metric requirements
        if metric.name == "information_coefficient":
            value = metric.calculation_func(predictions, actual_returns)
        elif metric.name == "information_ratio":
            value = metric.calculation_func(predictions, actual_returns, benchmark_returns)
        elif metric.name == "sharpe_ratio":
            value = metric.calculation_func(actual_returns)
        elif metric.name == "maximum_drawdown":
            value = metric.calculation_func(actual_returns)
        elif metric.name == "hit_rate":
            value = metric.calculation_func(predictions, actual_returns)
        elif metric.name == "rmse":
            # Align predictions and returns
            aligned = pd.concat([predictions, actual_returns], axis=1).dropna()
            if len(aligned) > 0:
                value = metric.calculation_func(aligned.iloc[:, 0], aligned.iloc[:, 1])
            else:
                value = np.nan
        else:
            # Generic metric calculation
            value = metric.calculation_func(predictions, actual_returns)
        
        # Create result object
        result = PerformanceResult(
            metric_name=metric.name,
            value=value,
            timestamp=datetime.now(),
            sample_size=len(predictions)
        )
        
        # Check targets and thresholds
        if metric.target_value is not None:
            if metric.higher_is_better:
                result.is_target_met = value >= metric.target_value
            else:
                result.is_target_met = value <= metric.target_value
        
        # Check warning thresholds
        if metric.threshold_warning is not None:
            if metric.higher_is_better:
                result.warning_triggered = value < metric.threshold_warning
            else:
                result.warning_triggered = value > metric.threshold_warning
        
        # Check critical thresholds
        if metric.threshold_critical is not None:
            if metric.higher_is_better:
                result.critical_triggered = value < metric.threshold_critical
            else:
                result.critical_triggered = value > metric.threshold_critical
        
        return result
    
    def _calculate_overall_score(self, snapshot: ModelPerformanceSnapshot) -> float:
        """Calculate overall performance score (0-100)."""
        scores = []
        weights = {
            PerformanceMetricType.FINANCIAL: 0.4,
            PerformanceMetricType.STATISTICAL: 0.2,
            PerformanceMetricType.RISK: 0.3,
            PerformanceMetricType.TAIWAN_SPECIFIC: 0.1
        }
        
        # Score financial metrics
        if snapshot.financial_metrics:
            financial_score = np.mean([
                100.0 if result.is_target_met else (50.0 if not result.critical_triggered else 0.0)
                for result in snapshot.financial_metrics.values()
            ])
            scores.append(financial_score * weights[PerformanceMetricType.FINANCIAL])
        
        # Score statistical metrics
        if snapshot.statistical_metrics:
            statistical_score = np.mean([
                100.0 if result.is_target_met else (50.0 if not result.critical_triggered else 0.0)
                for result in snapshot.statistical_metrics.values()
            ])
            scores.append(statistical_score * weights[PerformanceMetricType.STATISTICAL])
        
        # Score risk metrics
        if snapshot.risk_metrics:
            risk_score = np.mean([
                100.0 if result.is_target_met else (50.0 if not result.critical_triggered else 0.0)
                for result in snapshot.risk_metrics.values()
            ])
            scores.append(risk_score * weights[PerformanceMetricType.RISK])
        
        # Score Taiwan specific metrics
        if snapshot.taiwan_metrics:
            taiwan_score = np.mean([
                100.0 if result.is_target_met else (50.0 if not result.critical_triggered else 0.0)
                for result in snapshot.taiwan_metrics.values()
            ])
            scores.append(taiwan_score * weights[PerformanceMetricType.TAIWAN_SPECIFIC])
        
        return np.sum(scores) if scores else 0.0
    
    def _assign_performance_grade(self, overall_score: float) -> str:
        """Assign letter grade based on overall score."""
        if overall_score >= 90:
            return "A"
        elif overall_score >= 80:
            return "B"
        elif overall_score >= 70:
            return "C"
        elif overall_score >= 60:
            return "D"
        else:
            return "F"
    
    def _check_targets_met(self, snapshot: ModelPerformanceSnapshot) -> bool:
        """Check if all critical targets are met."""
        all_metrics = []
        all_metrics.extend(snapshot.financial_metrics.values())
        all_metrics.extend(snapshot.statistical_metrics.values())
        all_metrics.extend(snapshot.risk_metrics.values())
        all_metrics.extend(snapshot.taiwan_metrics.values())
        
        # Must meet targets and have no critical issues
        targets_met = all(
            result.is_target_met for result in all_metrics 
            if result.is_target_met is not None
        )
        no_critical = not any(result.critical_triggered for result in all_metrics)
        
        return targets_met and no_critical
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for the last N days."""
        if not self.performance_history:
            return {"error": "No performance history available"}
        
        # Filter recent snapshots
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_snapshots = [
            s for s in self.performance_history 
            if s.timestamp >= cutoff_date
        ]
        
        if not recent_snapshots:
            return {"error": f"No performance data in last {days} days"}
        
        # Calculate summary statistics
        overall_scores = [s.overall_score for s in recent_snapshots if s.overall_score is not None]
        grades = [s.performance_grade for s in recent_snapshots if s.performance_grade is not None]
        
        summary = {
            "period_days": days,
            "total_snapshots": len(recent_snapshots),
            "mean_overall_score": np.mean(overall_scores) if overall_scores else 0.0,
            "latest_score": recent_snapshots[-1].overall_score,
            "latest_grade": recent_snapshots[-1].performance_grade,
            "targets_met_rate": np.mean([s.meets_targets for s in recent_snapshots]),
            "warning_rate": np.mean([s.warning_count > 0 for s in recent_snapshots]),
            "critical_rate": np.mean([s.critical_count > 0 for s in recent_snapshots])
        }
        
        return summary
    
    def export_performance_history(self, filepath: str) -> None:
        """Export performance history to JSON file."""
        history_data = [snapshot.to_dict() for snapshot in self.performance_history]
        
        with open(filepath, 'w') as f:
            json.dump({
                'model_id': self.model_id,
                'export_timestamp': datetime.now().isoformat(),
                'total_snapshots': len(history_data),
                'performance_history': history_data
            }, f, indent=2)
        
        logger.info(f"Performance history exported to {filepath}")


# Utility functions

def calculate_model_metrics(
    predictions: pd.Series,
    actual_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None
) -> Dict[str, float]:
    """Quick utility to calculate common model metrics."""
    
    metrics = {}
    
    try:
        # Information Coefficient
        metrics['ic'] = TaiwanMarketMetrics.information_coefficient(predictions, actual_returns)
        
        # Sharpe Ratio
        metrics['sharpe_ratio'] = TaiwanMarketMetrics.sharpe_ratio(actual_returns)
        
        # Maximum Drawdown
        metrics['max_drawdown'] = TaiwanMarketMetrics.maximum_drawdown(actual_returns)
        
        # Hit Rate
        metrics['hit_rate'] = TaiwanMarketMetrics.hit_rate(predictions, actual_returns)
        
        # Information Ratio
        metrics['information_ratio'] = TaiwanMarketMetrics.information_ratio(
            predictions, actual_returns, benchmark_returns
        )
        
        # Statistical metrics
        aligned = pd.concat([predictions, actual_returns], axis=1).dropna()
        if len(aligned) > 0:
            metrics['rmse'] = np.sqrt(mean_squared_error(aligned.iloc[:, 1], aligned.iloc[:, 0]))
            metrics['r2'] = r2_score(aligned.iloc[:, 1], aligned.iloc[:, 0])
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    return metrics


if __name__ == "__main__":
    # Demo performance tracking
    print("Model Performance Metrics and Tracking System")
    print("Provides comprehensive performance tracking for Taiwan market ML models")
    
    # Example tracker initialization
    tracker = ModelPerformanceTracker("lightgbm_alpha_v1")
    print(f"Initialized tracker with {len(tracker.metrics_registry)} default metrics")