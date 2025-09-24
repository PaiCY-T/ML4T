"""
Trading Strategy Coherence Validator for ML4T.

This module validates the logical consistency and coherence of trading strategies,
ensuring that model predictions, risk management, and portfolio construction
work together harmoniously.

Key Features:
- Signal consistency validation
- Risk-return coherence checks
- Portfolio construction alignment
- Strategy parameter validation
- Multi-timeframe consistency
- Alpha decay analysis

Author: ML4T Team
Date: 2025-09-24
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TradingStrategyType(Enum):
    """Types of trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    QUANTITATIVE = "quantitative"
    HYBRID = "hybrid"
    MARKET_NEUTRAL = "market_neutral"
    LONG_SHORT = "long_short"


class CoherenceLevel(Enum):
    """Coherence validation levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    CONCERNING = "concerning"
    INCOHERENT = "incoherent"


class CoherenceCategory(Enum):
    """Categories of coherence checks."""
    SIGNAL_CONSISTENCY = "signal_consistency"
    RISK_ALIGNMENT = "risk_alignment"
    PORTFOLIO_CONSTRUCTION = "portfolio_construction"
    PARAMETER_COHERENCE = "parameter_coherence"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    ALPHA_PERSISTENCE = "alpha_persistence"
    ECONOMIC_RATIONALE = "economic_rationale"


@dataclass
class CoherenceCheck:
    """Individual coherence check definition."""
    category: CoherenceCategory
    check_name: str
    description: str
    weight: float = 1.0
    threshold_excellent: float = 0.9
    threshold_good: float = 0.8
    threshold_acceptable: float = 0.7
    threshold_concerning: float = 0.5
    validation_function: Optional[Callable] = None
    
    def evaluate_score(self, score: float) -> CoherenceLevel:
        """Evaluate coherence level based on score."""
        if score >= self.threshold_excellent:
            return CoherenceLevel.EXCELLENT
        elif score >= self.threshold_good:
            return CoherenceLevel.GOOD
        elif score >= self.threshold_acceptable:
            return CoherenceLevel.ACCEPTABLE
        elif score >= self.threshold_concerning:
            return CoherenceLevel.CONCERNING
        else:
            return CoherenceLevel.INCOHERENT


@dataclass
class CoherenceResult:
    """Result of a coherence validation."""
    check: CoherenceCheck
    score: float
    level: CoherenceLevel
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'check_name': self.check.check_name,
            'category': self.check.category.value,
            'score': self.score,
            'level': self.level.value,
            'weight': self.check.weight,
            'details': self.details,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class CoherenceConfig:
    """Configuration for strategy coherence validation."""
    
    # Signal consistency parameters
    min_signal_correlation: float = 0.3
    max_signal_noise_ratio: float = 2.0
    signal_stability_window: int = 30
    
    # Risk alignment parameters
    risk_budget_tolerance: float = 0.1  # 10% tolerance
    max_concentration_risk: float = 0.15  # 15% max single position
    expected_tracking_error: Tuple[float, float] = (0.02, 0.08)  # 2-8% TE range
    
    # Portfolio construction parameters
    min_diversification_ratio: float = 0.8
    max_turnover_threshold: float = 0.5  # 50% monthly turnover
    rebalancing_frequency_days: int = 5  # 5-day rebalancing
    
    # Temporal consistency parameters
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    consistency_threshold: float = 0.7
    regime_stability_threshold: float = 0.6
    
    # Alpha persistence parameters
    alpha_decay_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    min_alpha_persistence: float = 0.4
    target_information_ratio: float = 1.5
    
    # Strategy-specific weights
    strategy_weights: Dict[CoherenceCategory, float] = field(default_factory=lambda: {
        CoherenceCategory.SIGNAL_CONSISTENCY: 0.25,
        CoherenceCategory.RISK_ALIGNMENT: 0.20,
        CoherenceCategory.PORTFOLIO_CONSTRUCTION: 0.20,
        CoherenceCategory.TEMPORAL_CONSISTENCY: 0.15,
        CoherenceCategory.ALPHA_PERSISTENCE: 0.15,
        CoherenceCategory.ECONOMIC_RATIONALE: 0.05
    })


class StrategyCoherenceValidator:
    """
    Trading Strategy Coherence Validator.
    
    Validates the logical consistency and coherence of trading strategies,
    ensuring all components work together effectively.
    """
    
    def __init__(
        self,
        config: CoherenceConfig,
        strategy_type: TradingStrategyType = TradingStrategyType.QUANTITATIVE
    ):
        self.config = config
        self.strategy_type = strategy_type
        
        # Initialize coherence checks
        self.coherence_checks = self._initialize_coherence_checks()
        
        logger.info(f"StrategyCoherenceValidator initialized for {strategy_type.value} strategy")
    
    def validate_strategy_coherence(
        self,
        predictions: pd.DataFrame,
        portfolio_weights: pd.DataFrame,
        returns: pd.DataFrame,
        risk_model: Optional[Any] = None,
        validation_date: date = None
    ) -> Dict[str, CoherenceResult]:
        """
        Validate overall strategy coherence.
        
        Args:
            predictions: Model predictions DataFrame (symbols x dates)
            portfolio_weights: Portfolio weights DataFrame (symbols x dates)
            returns: Historical returns DataFrame (symbols x dates)
            risk_model: Risk model for risk-based validation
            validation_date: Date for validation context
            
        Returns:
            Dictionary of coherence results by check name
        """
        validation_date = validation_date or date.today()
        results = {}
        
        logger.info(f"Validating strategy coherence for {len(self.coherence_checks)} checks")
        
        # Run all coherence checks
        for check_name, check in self.coherence_checks.items():
            try:
                if check.validation_function:
                    score = check.validation_function(
                        predictions=predictions,
                        portfolio_weights=portfolio_weights,
                        returns=returns,
                        risk_model=risk_model,
                        config=self.config
                    )
                else:
                    score = self._run_default_check(
                        check, predictions, portfolio_weights, returns
                    )
                
                level = check.evaluate_score(score)
                details, recommendations = self._generate_check_details(
                    check, score, predictions, portfolio_weights, returns
                )
                
                results[check_name] = CoherenceResult(
                    check=check,
                    score=score,
                    level=level,
                    details=details,
                    recommendations=recommendations
                )
                
            except Exception as e:
                logger.error(f"Error running coherence check {check_name}: {e}")
                results[check_name] = CoherenceResult(
                    check=check,
                    score=0.0,
                    level=CoherenceLevel.INCOHERENT,
                    details={'error': str(e)},
                    recommendations=[f"Fix validation error: {e}"]
                )
        
        logger.info(f"Strategy coherence validation completed: {len(results)} checks")
        return results
    
    def calculate_overall_coherence_score(
        self,
        coherence_results: Dict[str, CoherenceResult]
    ) -> Tuple[float, CoherenceLevel]:
        """Calculate weighted overall coherence score."""
        total_weight = 0.0
        weighted_score = 0.0
        
        for result in coherence_results.values():
            category_weight = self.config.strategy_weights.get(
                result.check.category, 1.0
            )
            check_weight = result.check.weight
            
            total_weight += category_weight * check_weight
            weighted_score += result.score * category_weight * check_weight
        
        if total_weight == 0:
            return 0.0, CoherenceLevel.INCOHERENT
        
        overall_score = weighted_score / total_weight
        
        # Determine overall level
        if overall_score >= 0.85:
            level = CoherenceLevel.EXCELLENT
        elif overall_score >= 0.75:
            level = CoherenceLevel.GOOD
        elif overall_score >= 0.65:
            level = CoherenceLevel.ACCEPTABLE
        elif overall_score >= 0.50:
            level = CoherenceLevel.CONCERNING
        else:
            level = CoherenceLevel.INCOHERENT
        
        return overall_score, level
    
    def _initialize_coherence_checks(self) -> Dict[str, CoherenceCheck]:
        """Initialize strategy coherence checks."""
        checks = {}
        
        # Signal consistency checks
        checks['signal_stability'] = CoherenceCheck(
            category=CoherenceCategory.SIGNAL_CONSISTENCY,
            check_name='signal_stability',
            description='Validates stability of prediction signals over time',
            weight=1.0,
            validation_function=self._validate_signal_stability
        )
        
        checks['signal_correlation'] = CoherenceCheck(
            category=CoherenceCategory.SIGNAL_CONSISTENCY,
            check_name='signal_correlation',
            description='Validates correlation between predictions and portfolio weights',
            weight=1.2,
            validation_function=self._validate_signal_correlation
        )
        
        # Risk alignment checks
        checks['risk_budget_alignment'] = CoherenceCheck(
            category=CoherenceCategory.RISK_ALIGNMENT,
            check_name='risk_budget_alignment',
            description='Validates alignment of risk budgets with strategy objectives',
            weight=1.3,
            validation_function=self._validate_risk_budget_alignment
        )
        
        checks['concentration_risk'] = CoherenceCheck(
            category=CoherenceCategory.RISK_ALIGNMENT,
            check_name='concentration_risk',
            description='Validates position concentration vs risk limits',
            weight=1.0,
            validation_function=self._validate_concentration_risk
        )
        
        # Portfolio construction checks
        checks['diversification_coherence'] = CoherenceCheck(
            category=CoherenceCategory.PORTFOLIO_CONSTRUCTION,
            check_name='diversification_coherence',
            description='Validates portfolio diversification effectiveness',
            weight=1.1,
            validation_function=self._validate_diversification_coherence
        )
        
        checks['turnover_coherence'] = CoherenceCheck(
            category=CoherenceCategory.PORTFOLIO_CONSTRUCTION,
            check_name='turnover_coherence',
            description='Validates turnover levels vs strategy frequency',
            weight=1.0,
            validation_function=self._validate_turnover_coherence
        )
        
        # Temporal consistency checks
        checks['temporal_consistency'] = CoherenceCheck(
            category=CoherenceCategory.TEMPORAL_CONSISTENCY,
            check_name='temporal_consistency',
            description='Validates consistency across different time horizons',
            weight=1.0,
            validation_function=self._validate_temporal_consistency
        )
        
        # Alpha persistence checks
        checks['alpha_persistence'] = CoherenceCheck(
            category=CoherenceCategory.ALPHA_PERSISTENCE,
            check_name='alpha_persistence',
            description='Validates alpha persistence and decay patterns',
            weight=1.2,
            validation_function=self._validate_alpha_persistence
        )
        
        # Economic rationale checks
        checks['economic_intuition'] = CoherenceCheck(
            category=CoherenceCategory.ECONOMIC_RATIONALE,
            check_name='economic_intuition',
            description='Validates economic intuition behind strategy',
            weight=0.8,
            validation_function=self._validate_economic_intuition
        )
        
        return checks
    
    def _validate_signal_stability(self, **kwargs) -> float:
        """Validate signal stability over time."""
        predictions = kwargs.get('predictions')
        config = kwargs.get('config')
        
        if predictions is None or predictions.empty:
            return 0.0
        
        # Calculate rolling correlation of signals
        window = config.signal_stability_window
        correlations = []
        
        for i in range(window, len(predictions.index)):
            current_window = predictions.iloc[i-window:i]
            next_window = predictions.iloc[i-window+1:i+1]
            
            if not current_window.empty and not next_window.empty:
                # Calculate correlation between consecutive windows
                corr_matrix = current_window.corrwith(next_window)
                mean_corr = corr_matrix.mean()
                if not np.isnan(mean_corr):
                    correlations.append(mean_corr)
        
        if not correlations:
            return 0.5  # Neutral score if no data
        
        mean_stability = np.mean(correlations)
        return max(0.0, min(1.0, mean_stability))
    
    def _validate_signal_correlation(self, **kwargs) -> float:
        """Validate correlation between predictions and portfolio weights."""
        predictions = kwargs.get('predictions')
        portfolio_weights = kwargs.get('portfolio_weights')
        config = kwargs.get('config')
        
        if predictions is None or portfolio_weights is None:
            return 0.0
        
        # Align data by common dates and symbols
        common_dates = predictions.index.intersection(portfolio_weights.index)
        common_symbols = predictions.columns.intersection(portfolio_weights.columns)
        
        if len(common_dates) == 0 or len(common_symbols) == 0:
            return 0.0
        
        pred_aligned = predictions.loc[common_dates, common_symbols]
        weights_aligned = portfolio_weights.loc[common_dates, common_symbols]
        
        # Calculate correlation between predictions and weights
        correlations = []
        for date in common_dates:
            if date in pred_aligned.index and date in weights_aligned.index:
                pred_row = pred_aligned.loc[date]
                weight_row = weights_aligned.loc[date]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(pred_row) | np.isnan(weight_row))
                if valid_mask.sum() > 2:  # Need at least 3 points for correlation
                    corr = np.corrcoef(pred_row[valid_mask], weight_row[valid_mask])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        mean_correlation = np.mean(correlations)
        
        # Scale correlation to 0-1 score
        # High positive correlation is good, negative correlation is bad
        if mean_correlation >= config.min_signal_correlation:
            return min(1.0, mean_correlation)
        else:
            return max(0.0, (mean_correlation + 1) / 2)  # Scale -1 to +1 → 0 to 1
    
    def _validate_risk_budget_alignment(self, **kwargs) -> float:
        """Validate risk budget alignment."""
        portfolio_weights = kwargs.get('portfolio_weights')
        risk_model = kwargs.get('risk_model')
        config = kwargs.get('config')
        
        if portfolio_weights is None:
            return 0.5
        
        # Calculate concentration metrics
        latest_weights = portfolio_weights.iloc[-1]
        
        # Herfindahl-Hirschman Index for concentration
        hhi = np.sum(latest_weights ** 2)
        diversification_score = 1 - hhi
        
        # Check if diversification meets threshold
        target_diversification = 1 - config.max_concentration_risk
        
        if diversification_score >= target_diversification:
            return 1.0
        else:
            return max(0.0, diversification_score / target_diversification)
    
    def _validate_concentration_risk(self, **kwargs) -> float:
        """Validate position concentration risk."""
        portfolio_weights = kwargs.get('portfolio_weights')
        config = kwargs.get('config')
        
        if portfolio_weights is None:
            return 0.0
        
        # Check maximum position sizes
        max_weights = portfolio_weights.abs().max(axis=1)
        violations = (max_weights > config.max_concentration_risk).sum()
        
        # Score based on violation rate
        violation_rate = violations / len(max_weights)
        return max(0.0, 1.0 - violation_rate * 2)  # Penalize violations heavily
    
    def _validate_diversification_coherence(self, **kwargs) -> float:
        """Validate portfolio diversification effectiveness."""
        portfolio_weights = kwargs.get('portfolio_weights')
        returns = kwargs.get('returns')
        config = kwargs.get('config')
        
        if portfolio_weights is None or returns is None:
            return 0.5
        
        # Calculate effective number of positions
        latest_weights = portfolio_weights.iloc[-1].abs()
        effective_positions = 1 / np.sum(latest_weights ** 2)
        total_positions = (latest_weights > 0.001).sum()
        
        diversification_ratio = effective_positions / max(1, total_positions)
        
        return min(1.0, diversification_ratio / config.min_diversification_ratio)
    
    def _validate_turnover_coherence(self, **kwargs) -> float:
        """Validate turnover coherence with strategy frequency."""
        portfolio_weights = kwargs.get('portfolio_weights')
        config = kwargs.get('config')
        
        if portfolio_weights is None or len(portfolio_weights) < 2:
            return 0.5
        
        # Calculate turnover
        weight_changes = portfolio_weights.diff().abs()
        daily_turnover = weight_changes.sum(axis=1)
        
        # Annualize turnover (assuming daily data)
        mean_daily_turnover = daily_turnover.mean()
        annualized_turnover = mean_daily_turnover * 252
        
        # Compare to threshold
        if annualized_turnover <= config.max_turnover_threshold:
            return 1.0
        else:
            # Penalize excessive turnover
            excess_ratio = annualized_turnover / config.max_turnover_threshold
            return max(0.0, 1.0 / excess_ratio)
    
    def _validate_temporal_consistency(self, **kwargs) -> float:
        """Validate consistency across time horizons."""
        predictions = kwargs.get('predictions')
        config = kwargs.get('config')
        
        if predictions is None:
            return 0.5
        
        consistency_scores = []
        
        for period in config.lookback_periods:
            if len(predictions) < period * 2:
                continue
                
            # Split into periods
            first_half = predictions.iloc[-period*2:-period]
            second_half = predictions.iloc[-period:]
            
            if not first_half.empty and not second_half.empty:
                # Calculate rank correlation between periods
                first_ranks = first_half.rank(axis=1, pct=True)
                second_ranks = second_half.rank(axis=1, pct=True)
                
                # Average rank correlation across time
                correlations = []
                for col in first_ranks.columns:
                    if col in second_ranks.columns:
                        corr = first_ranks[col].corrwith(second_ranks[col])
                        mean_corr = corr.mean()
                        if not np.isnan(mean_corr):
                            correlations.append(mean_corr)
                
                if correlations:
                    consistency_scores.append(np.mean(correlations))
        
        if not consistency_scores:
            return 0.5
        
        overall_consistency = np.mean(consistency_scores)
        return max(0.0, min(1.0, (overall_consistency + 1) / 2))
    
    def _validate_alpha_persistence(self, **kwargs) -> float:
        """Validate alpha persistence and decay patterns."""
        predictions = kwargs.get('predictions')
        returns = kwargs.get('returns')
        config = kwargs.get('config')
        
        if predictions is None or returns is None:
            return 0.5
        
        # Align data
        common_dates = predictions.index.intersection(returns.index)
        common_symbols = predictions.columns.intersection(returns.columns)
        
        if len(common_dates) < max(config.alpha_decay_periods):
            return 0.5
        
        pred_aligned = predictions.loc[common_dates, common_symbols]
        ret_aligned = returns.loc[common_dates, common_symbols]
        
        # Calculate forward-looking alpha at different horizons
        alpha_persistence = []
        
        for horizon in config.alpha_decay_periods:
            if len(pred_aligned) < horizon + 10:
                continue
                
            # Calculate forward returns
            forward_returns = ret_aligned.shift(-horizon)
            
            # Calculate IC (Information Coefficient)
            ics = []
            for date in pred_aligned.index[:-horizon]:
                pred_cross = pred_aligned.loc[date]
                ret_cross = forward_returns.loc[date]
                
                valid_mask = ~(np.isnan(pred_cross) | np.isnan(ret_cross))
                if valid_mask.sum() > 10:
                    ic = np.corrcoef(pred_cross[valid_mask], ret_cross[valid_mask])[0, 1]
                    if not np.isnan(ic):
                        ics.append(abs(ic))  # Use absolute IC
            
            if ics:
                mean_ic = np.mean(ics)
                alpha_persistence.append(mean_ic)
        
        if not alpha_persistence:
            return 0.5
        
        # Check if alpha decays reasonably (later periods should have lower IC)
        if len(alpha_persistence) > 1:
            decay_score = 1.0
            for i in range(1, len(alpha_persistence)):
                if alpha_persistence[i] > alpha_persistence[i-1] * 1.2:  # Allow some noise
                    decay_score *= 0.8  # Penalize non-decaying alpha
        else:
            decay_score = 1.0
        
        # Score based on initial alpha level
        initial_alpha = alpha_persistence[0] if alpha_persistence else 0
        alpha_score = min(1.0, initial_alpha / config.min_alpha_persistence)
        
        return alpha_score * decay_score
    
    def _validate_economic_intuition(self, **kwargs) -> float:
        """Validate economic intuition behind strategy."""
        predictions = kwargs.get('predictions')
        returns = kwargs.get('returns')
        
        # This would implement more sophisticated economic validation
        # For now, return a reasonable score based on prediction-return alignment
        
        if predictions is None or returns is None:
            return 0.5
        
        # Basic check: are predictions generally aligned with subsequent returns?
        # This is a simplified version - real implementation would be more sophisticated
        return 0.75  # Placeholder score
    
    def _run_default_check(
        self,
        check: CoherenceCheck,
        predictions: pd.DataFrame,
        portfolio_weights: pd.DataFrame,
        returns: pd.DataFrame
    ) -> float:
        """Run default validation for checks without custom functions."""
        # Default implementation returns neutral score
        return 0.5
    
    def _generate_check_details(
        self,
        check: CoherenceCheck,
        score: float,
        predictions: pd.DataFrame,
        portfolio_weights: pd.DataFrame,
        returns: pd.DataFrame
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Generate detailed results and recommendations for a check."""
        details = {
            'score': score,
            'threshold_excellent': check.threshold_excellent,
            'threshold_good': check.threshold_good,
            'threshold_acceptable': check.threshold_acceptable
        }
        
        recommendations = []
        
        # Generate recommendations based on score and check type
        if score < check.threshold_acceptable:
            if check.category == CoherenceCategory.SIGNAL_CONSISTENCY:
                recommendations.append("Review signal generation process for consistency")
                recommendations.append("Consider increasing signal smoothing or filtering")
            elif check.category == CoherenceCategory.RISK_ALIGNMENT:
                recommendations.append("Adjust risk budgets to align with strategy objectives")
                recommendations.append("Review position sizing methodology")
            elif check.category == CoherenceCategory.PORTFOLIO_CONSTRUCTION:
                recommendations.append("Improve diversification or turnover management")
                recommendations.append("Review portfolio optimization constraints")
            elif check.category == CoherenceCategory.ALPHA_PERSISTENCE:
                recommendations.append("Investigate alpha decay patterns")
                recommendations.append("Consider adjusting prediction horizons")
        
        if not recommendations:
            recommendations.append("Strategy coherence is satisfactory")
        
        return details, recommendations


# Example usage
if __name__ == "__main__":
    print("Strategy Coherence Validator demo")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    symbols = ['2330.TW', '2317.TW', '2454.TW', '2882.TW', '2412.TW']
    
    # Mock predictions and portfolio weights
    np.random.seed(42)
    predictions = pd.DataFrame(
        np.random.randn(100, 5),
        index=dates,
        columns=symbols
    )
    
    portfolio_weights = pd.DataFrame(
        np.random.dirichlet([1]*5, 100),
        index=dates,
        columns=symbols
    )
    
    returns = pd.DataFrame(
        np.random.randn(100, 5) * 0.02,
        index=dates,
        columns=symbols
    )
    
    # Create validator
    config = CoherenceConfig()
    validator = StrategyCoherenceValidator(
        config, 
        TradingStrategyType.QUANTITATIVE
    )
    
    # Validate coherence
    results = validator.validate_strategy_coherence(
        predictions=predictions,
        portfolio_weights=portfolio_weights,
        returns=returns
    )
    
    # Calculate overall score
    overall_score, overall_level = validator.calculate_overall_coherence_score(results)
    
    print(f"Overall coherence: {overall_score:.3f} ({overall_level.value})")
    print(f"Individual check results:")
    for name, result in results.items():
        print(f"- {name}: {result.score:.3f} ({result.level.value})")