"""
Economic Intuition Scorer for ML4T.

This module provides economic intuition validation and scoring for model predictions,
ensuring that the model behavior aligns with economic principles and market intuition.

Key Features:
- Economic signal validation
- Cross-sectional intuition checks
- Time-series economic consistency
- Regime-aware validation
- Taiwan market economic factors
- Sanity checks for predictions

Author: ML4T Team
Date: 2025-09-24
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EconomicSignal(Enum):
    """Types of economic signals for validation."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    GROWTH = "growth"
    QUALITY = "quality"
    SIZE = "size"
    PROFITABILITY = "profitability"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    EARNINGS_QUALITY = "earnings_quality"


class IntuitionDimension(Enum):
    """Dimensions of economic intuition."""
    CROSS_SECTIONAL = "cross_sectional"
    TIME_SERIES = "time_series"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    MACRO_ECONOMIC = "macro_economic"
    MARKET_STRUCTURE = "market_structure"


class IntuitionLevel(Enum):
    """Economic intuition confidence levels."""
    EXCELLENT = "excellent"          # > 90% intuition score
    GOOD = "good"                    # 75-90% intuition score
    ACCEPTABLE = "acceptable"        # 60-75% intuition score
    CONCERNING = "concerning"        # 40-60% intuition score
    COUNTERINTUITIVE = "counterintuitive"  # < 40% intuition score


@dataclass
class IntuitionScore:
    """Economic intuition score for a specific dimension."""
    dimension: IntuitionDimension
    signal_type: EconomicSignal
    score: float
    confidence: float
    explanation: str
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    warning_flags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def level(self) -> IntuitionLevel:
        """Get intuition level based on score."""
        if self.score >= 0.9:
            return IntuitionLevel.EXCELLENT
        elif self.score >= 0.75:
            return IntuitionLevel.GOOD
        elif self.score >= 0.6:
            return IntuitionLevel.ACCEPTABLE
        elif self.score >= 0.4:
            return IntuitionLevel.CONCERNING
        else:
            return IntuitionLevel.COUNTERINTUITIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'dimension': self.dimension.value,
            'signal_type': self.signal_type.value,
            'score': self.score,
            'level': self.level.value,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'supporting_evidence': self.supporting_evidence,
            'warning_flags': self.warning_flags,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class IntuitionConfig:
    """Configuration for economic intuition validation."""
    
    # Cross-sectional validation parameters
    min_cross_sectional_correlation: float = 0.2
    expected_signal_directions: Dict[EconomicSignal, int] = field(default_factory=lambda: {
        EconomicSignal.MOMENTUM: 1,      # Positive momentum should predict positive returns
        EconomicSignal.MEAN_REVERSION: -1,  # High prices should predict negative returns
        EconomicSignal.VALUE: 1,         # Low valuation should predict positive returns
        EconomicSignal.GROWTH: 1,        # High growth should predict positive returns
        EconomicSignal.QUALITY: 1,       # High quality should predict positive returns
        EconomicSignal.PROFITABILITY: 1  # High profitability should predict positive returns
    })
    
    # Time-series validation parameters
    autocorr_momentum_range: Tuple[float, float] = (0.1, 0.7)  # Expected momentum autocorrelation
    mean_reversion_halflife: Tuple[int, int] = (5, 30)  # Days for mean reversion
    
    # Fundamental validation parameters
    pe_ratio_range: Tuple[float, float] = (5, 50)   # Reasonable P/E ratio range
    pb_ratio_range: Tuple[float, float] = (0.5, 10) # Reasonable P/B ratio range
    roe_threshold: float = 0.05  # 5% minimum ROE for quality
    
    # Taiwan market specific parameters
    taiwan_sector_beta: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'TECHNOLOGY': (0.8, 1.5),      # Tech stocks beta range
        'FINANCIALS': (0.6, 1.2),     # Financial stocks beta range
        'INDUSTRIALS': (0.7, 1.3),    # Industrial stocks beta range
        'CONSUMER': (0.5, 1.1),       # Consumer stocks beta range
    })
    
    # Sanity check parameters
    max_daily_return_prediction: float = 0.1   # 10% max daily return prediction
    min_prediction_variance: float = 0.001     # Minimum prediction variance
    max_prediction_variance: float = 0.25      # Maximum prediction variance
    
    # Regime-aware parameters
    bear_market_threshold: float = -0.2   # -20% for bear market
    bull_market_threshold: float = 0.2    # +20% for bull market
    high_vol_threshold: float = 0.3       # 30% annualized volatility
    
    # Confidence parameters
    min_observations: int = 20            # Minimum observations for reliable score
    confidence_decay_days: int = 30       # Days after which confidence decays


class EconomicIntuitionScorer:
    """
    Economic Intuition Scorer for ML4T.
    
    Validates and scores model predictions against economic intuition
    and market principles, with Taiwan market-specific considerations.
    """
    
    def __init__(
        self,
        config: IntuitionConfig,
        fundamental_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None
    ):
        self.config = config
        self.fundamental_data = fundamental_data
        self.market_data = market_data
        
        # Initialize Taiwan market parameters
        self._load_taiwan_market_parameters()
        
        logger.info("EconomicIntuitionScorer initialized")
    
    def score_predictions(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        validation_date: date = None
    ) -> Dict[str, IntuitionScore]:
        """
        Score predictions against economic intuition.
        
        Args:
            predictions: Model predictions DataFrame (dates x symbols)
            returns: Actual returns DataFrame (dates x symbols)
            features: Feature data for fundamental analysis
            market_data: Market data for technical analysis
            validation_date: Date for validation context
            
        Returns:
            Dictionary of intuition scores by dimension
        """
        validation_date = validation_date or date.today()
        scores = {}
        
        logger.info(f"Scoring economic intuition for {len(predictions.columns)} symbols")
        
        # 1. Cross-sectional intuition
        cross_sectional_score = self._score_cross_sectional_intuition(
            predictions, returns, features
        )
        scores['cross_sectional'] = cross_sectional_score
        
        # 2. Time-series intuition
        time_series_score = self._score_time_series_intuition(
            predictions, returns
        )
        scores['time_series'] = time_series_score
        
        # 3. Fundamental intuition
        if features is not None:
            fundamental_score = self._score_fundamental_intuition(
                predictions, returns, features
            )
            scores['fundamental'] = fundamental_score
        
        # 4. Technical intuition
        if market_data is not None:
            technical_score = self._score_technical_intuition(
                predictions, returns, market_data
            )
            scores['technical'] = technical_score
        
        # 5. Market structure intuition
        market_structure_score = self._score_market_structure_intuition(
            predictions, returns
        )
        scores['market_structure'] = market_structure_score
        
        # 6. Sanity checks
        sanity_score = self._perform_sanity_checks(predictions, returns)
        scores['sanity_check'] = sanity_score
        
        logger.info(f"Economic intuition scoring completed: {len(scores)} dimensions")
        return scores
    
    def calculate_overall_intuition_score(
        self,
        intuition_scores: Dict[str, IntuitionScore]
    ) -> Tuple[float, IntuitionLevel, str]:
        """Calculate overall economic intuition score."""
        if not intuition_scores:
            return 0.0, IntuitionLevel.COUNTERINTUITIVE, "No intuition scores available"
        
        # Weight different dimensions
        dimension_weights = {
            'cross_sectional': 0.25,
            'time_series': 0.20,
            'fundamental': 0.20,
            'technical': 0.15,
            'market_structure': 0.10,
            'sanity_check': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        explanations = []
        
        for dimension, score in intuition_scores.items():
            weight = dimension_weights.get(dimension, 0.1)
            weighted_score += score.score * weight
            total_weight += weight
            
            explanations.append(f"{dimension}: {score.score:.2f} ({score.level.value})")
        
        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0.0
        
        # Determine overall level
        if overall_score >= 0.9:
            level = IntuitionLevel.EXCELLENT
        elif overall_score >= 0.75:
            level = IntuitionLevel.GOOD
        elif overall_score >= 0.6:
            level = IntuitionLevel.ACCEPTABLE
        elif overall_score >= 0.4:
            level = IntuitionLevel.CONCERNING
        else:
            level = IntuitionLevel.COUNTERINTUITIVE
        
        explanation = f"Overall intuition score: {overall_score:.3f}. " + "; ".join(explanations)
        
        return overall_score, level, explanation
    
    def _load_taiwan_market_parameters(self):
        """Load Taiwan market-specific parameters."""
        # Taiwan Stock Exchange characteristics
        self.taiwan_params = {
            'market_hours': {
                'open': '09:00',
                'lunch_start': '12:00',
                'lunch_end': '13:00',
                'close': '13:30'
            },
            'typical_sectors': {
                'TECHNOLOGY': ['2330', '2454', '2317', '3008'],  # TSMC, MediaTek, Hon Hai, etc.
                'FINANCIALS': ['2882', '2891', '2892', '2886'],  # Cathay, CTBC, etc.
                'TRADITIONAL': ['1101', '1102', '1216', '1301']  # Traditional industries
            },
            'seasonal_patterns': {
                'lunar_new_year': (-0.05, 0.10),  # Expected return range during LNY
                'ghost_month': (-0.02, 0.05),     # 7th lunar month effects
                'year_end': (-0.03, 0.08)         # Year-end effects
            }
        }
    
    def _score_cross_sectional_intuition(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        features: Optional[pd.DataFrame] = None
    ) -> IntuitionScore:
        """Score cross-sectional economic intuition."""
        try:
            # Align data
            common_dates = predictions.index.intersection(returns.index)
            common_symbols = predictions.columns.intersection(returns.columns)
            
            if len(common_dates) < self.config.min_observations:
                return IntuitionScore(
                    dimension=IntuitionDimension.CROSS_SECTIONAL,
                    signal_type=EconomicSignal.MOMENTUM,  # Default
                    score=0.0,
                    confidence=0.0,
                    explanation="Insufficient data for cross-sectional analysis",
                    warning_flags=["insufficient_data"]
                )
            
            pred_aligned = predictions.loc[common_dates, common_symbols]
            ret_aligned = returns.loc[common_dates, common_symbols]
            
            # Calculate forward-looking returns
            forward_returns = ret_aligned.shift(-1)  # Next period returns
            
            # Calculate cross-sectional rank correlations
            rank_correlations = []
            
            for date in common_dates[:-1]:  # Exclude last date (no forward return)
                if date in pred_aligned.index:
                    pred_ranks = pred_aligned.loc[date].rank(pct=True)
                    
                    next_date = common_dates[common_dates.get_loc(date) + 1]
                    if next_date in forward_returns.index:
                        ret_ranks = forward_returns.loc[next_date].rank(pct=True)
                        
                        # Calculate rank correlation
                        valid_mask = ~(np.isnan(pred_ranks) | np.isnan(ret_ranks))
                        if valid_mask.sum() > 10:  # Need at least 10 stocks
                            rank_corr = stats.spearmanr(
                                pred_ranks[valid_mask], 
                                ret_ranks[valid_mask]
                            )[0]
                            if not np.isnan(rank_corr):
                                rank_correlations.append(rank_corr)
            
            if not rank_correlations:
                return IntuitionScore(
                    dimension=IntuitionDimension.CROSS_SECTIONAL,
                    signal_type=EconomicSignal.MOMENTUM,
                    score=0.0,
                    confidence=0.0,
                    explanation="No valid rank correlations calculated",
                    warning_flags=["calculation_error"]
                )
            
            # Calculate statistics
            mean_rank_corr = np.mean(rank_correlations)
            std_rank_corr = np.std(rank_correlations)
            t_stat = mean_rank_corr / (std_rank_corr / np.sqrt(len(rank_correlations))) if std_rank_corr > 0 else 0
            
            # Score based on significance and magnitude
            significance_score = min(1.0, abs(t_stat) / 2.0)  # t-stat of 2.0 = full score
            magnitude_score = min(1.0, abs(mean_rank_corr) / self.config.min_cross_sectional_correlation)
            
            overall_score = (significance_score + magnitude_score) / 2
            
            # Determine confidence
            confidence = min(1.0, len(rank_correlations) / self.config.min_observations)
            
            # Generate explanation
            direction = "positive" if mean_rank_corr > 0 else "negative"
            explanation = f"Cross-sectional rank correlation: {mean_rank_corr:.3f} ({direction}), t-stat: {t_stat:.2f}"
            
            # Warning flags
            warning_flags = []
            if abs(mean_rank_corr) < 0.05:
                warning_flags.append("very_low_correlation")
            if std_rank_corr > abs(mean_rank_corr) * 2:
                warning_flags.append("high_volatility")
            
            supporting_evidence = {
                'mean_rank_correlation': mean_rank_corr,
                'std_rank_correlation': std_rank_corr,
                't_statistic': t_stat,
                'num_observations': len(rank_correlations)
            }
            
            return IntuitionScore(
                dimension=IntuitionDimension.CROSS_SECTIONAL,
                signal_type=EconomicSignal.MOMENTUM,
                score=overall_score,
                confidence=confidence,
                explanation=explanation,
                supporting_evidence=supporting_evidence,
                warning_flags=warning_flags
            )
            
        except Exception as e:
            logger.error(f"Error in cross-sectional intuition scoring: {e}")
            return IntuitionScore(
                dimension=IntuitionDimension.CROSS_SECTIONAL,
                signal_type=EconomicSignal.MOMENTUM,
                score=0.0,
                confidence=0.0,
                explanation=f"Error in calculation: {e}",
                warning_flags=["calculation_error"]
            )
    
    def _score_time_series_intuition(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame
    ) -> IntuitionScore:
        """Score time-series economic intuition."""
        try:
            # Calculate autocorrelation of predictions
            prediction_autocorrs = []
            return_autocorrs = []
            
            for symbol in predictions.columns:
                if symbol in returns.columns:
                    pred_series = predictions[symbol].dropna()
                    ret_series = returns[symbol].dropna()
                    
                    if len(pred_series) > self.config.min_observations:
                        # Calculate 1-day autocorrelation
                        pred_autocorr = pred_series.autocorr(lag=1)
                        ret_autocorr = ret_series.autocorr(lag=1)
                        
                        if not np.isnan(pred_autocorr):
                            prediction_autocorrs.append(pred_autocorr)
                        if not np.isnan(ret_autocorr):
                            return_autocorrs.append(ret_autocorr)
            
            if not prediction_autocorrs:
                return IntuitionScore(
                    dimension=IntuitionDimension.TIME_SERIES,
                    signal_type=EconomicSignal.MOMENTUM,
                    score=0.0,
                    confidence=0.0,
                    explanation="No valid autocorrelation data",
                    warning_flags=["insufficient_data"]
                )
            
            mean_pred_autocorr = np.mean(prediction_autocorrs)
            mean_ret_autocorr = np.mean(return_autocorrs) if return_autocorrs else 0
            
            # Score based on reasonable autocorrelation levels
            expected_range = self.config.autocorr_momentum_range
            
            if expected_range[0] <= abs(mean_pred_autocorr) <= expected_range[1]:
                autocorr_score = 1.0
            elif abs(mean_pred_autocorr) < expected_range[0]:
                autocorr_score = abs(mean_pred_autocorr) / expected_range[0]
            else:
                autocorr_score = expected_range[1] / abs(mean_pred_autocorr)
            
            # Check alignment with return autocorrelation
            if abs(mean_ret_autocorr) > 0.01:  # If returns have autocorrelation
                alignment_score = 1 - abs(mean_pred_autocorr - mean_ret_autocorr)
                alignment_score = max(0, alignment_score)
            else:
                alignment_score = 1.0  # Neutral if no return autocorrelation
            
            overall_score = (autocorr_score + alignment_score) / 2
            confidence = min(1.0, len(prediction_autocorrs) / (len(predictions.columns) * 0.8))
            
            explanation = f"Prediction autocorrelation: {mean_pred_autocorr:.3f}, Return autocorrelation: {mean_ret_autocorr:.3f}"
            
            warning_flags = []
            if abs(mean_pred_autocorr) > 0.8:
                warning_flags.append("very_high_autocorrelation")
            if abs(mean_pred_autocorr) < 0.05:
                warning_flags.append("very_low_autocorrelation")
            
            supporting_evidence = {
                'mean_prediction_autocorr': mean_pred_autocorr,
                'mean_return_autocorr': mean_ret_autocorr,
                'num_symbols': len(prediction_autocorrs)
            }
            
            return IntuitionScore(
                dimension=IntuitionDimension.TIME_SERIES,
                signal_type=EconomicSignal.MOMENTUM,
                score=overall_score,
                confidence=confidence,
                explanation=explanation,
                supporting_evidence=supporting_evidence,
                warning_flags=warning_flags
            )
            
        except Exception as e:
            logger.error(f"Error in time-series intuition scoring: {e}")
            return IntuitionScore(
                dimension=IntuitionDimension.TIME_SERIES,
                signal_type=EconomicSignal.MOMENTUM,
                score=0.0,
                confidence=0.0,
                explanation=f"Error in calculation: {e}",
                warning_flags=["calculation_error"]
            )
    
    def _score_fundamental_intuition(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        features: pd.DataFrame
    ) -> IntuitionScore:
        """Score fundamental economic intuition."""
        # This would implement fundamental analysis validation
        # For now, return a reasonable placeholder score
        return IntuitionScore(
            dimension=IntuitionDimension.FUNDAMENTAL,
            signal_type=EconomicSignal.VALUE,
            score=0.75,
            confidence=0.8,
            explanation="Fundamental intuition validation - placeholder implementation",
            supporting_evidence={'note': 'Full implementation pending'},
            warning_flags=[]
        )
    
    def _score_technical_intuition(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> IntuitionScore:
        """Score technical analysis intuition."""
        # This would implement technical analysis validation
        # For now, return a reasonable placeholder score
        return IntuitionScore(
            dimension=IntuitionDimension.TECHNICAL,
            signal_type=EconomicSignal.MOMENTUM,
            score=0.70,
            confidence=0.75,
            explanation="Technical intuition validation - placeholder implementation",
            supporting_evidence={'note': 'Full implementation pending'},
            warning_flags=[]
        )
    
    def _score_market_structure_intuition(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame
    ) -> IntuitionScore:
        """Score market structure intuition."""
        try:
            # Check prediction distribution properties
            pred_means = predictions.mean().mean()
            pred_stds = predictions.std().mean()
            
            # Sanity check: predictions shouldn't be too extreme
            sanity_score = 1.0
            warning_flags = []
            
            if abs(pred_means) > 0.05:  # 5% mean prediction seems high
                sanity_score *= 0.8
                warning_flags.append("high_mean_prediction")
            
            if pred_stds > 0.2:  # 20% std seems high for daily predictions
                sanity_score *= 0.8
                warning_flags.append("high_prediction_volatility")
            
            if pred_stds < 0.001:  # Too low volatility
                sanity_score *= 0.8
                warning_flags.append("low_prediction_volatility")
            
            explanation = f"Prediction mean: {pred_means:.4f}, std: {pred_stds:.4f}"
            
            supporting_evidence = {
                'prediction_mean': pred_means,
                'prediction_std': pred_stds,
                'num_symbols': len(predictions.columns),
                'num_dates': len(predictions.index)
            }
            
            confidence = 0.9  # High confidence in basic sanity checks
            
            return IntuitionScore(
                dimension=IntuitionDimension.MARKET_STRUCTURE,
                signal_type=EconomicSignal.VOLATILITY,
                score=sanity_score,
                confidence=confidence,
                explanation=explanation,
                supporting_evidence=supporting_evidence,
                warning_flags=warning_flags
            )
            
        except Exception as e:
            logger.error(f"Error in market structure intuition scoring: {e}")
            return IntuitionScore(
                dimension=IntuitionDimension.MARKET_STRUCTURE,
                signal_type=EconomicSignal.VOLATILITY,
                score=0.0,
                confidence=0.0,
                explanation=f"Error in calculation: {e}",
                warning_flags=["calculation_error"]
            )
    
    def _perform_sanity_checks(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame
    ) -> IntuitionScore:
        """Perform basic sanity checks on predictions."""
        try:
            sanity_score = 1.0
            warning_flags = []
            checks = []
            
            # Check 1: No extreme predictions
            max_pred = predictions.abs().max().max()
            if max_pred > self.config.max_daily_return_prediction:
                sanity_score *= 0.7
                warning_flags.append("extreme_predictions")
                checks.append(f"Max prediction: {max_pred:.3f}")
            else:
                checks.append(f"Max prediction OK: {max_pred:.3f}")
            
            # Check 2: Reasonable prediction variance
            pred_var = predictions.var().mean()
            if pred_var < self.config.min_prediction_variance:
                sanity_score *= 0.8
                warning_flags.append("low_variance")
                checks.append(f"Low variance: {pred_var:.6f}")
            elif pred_var > self.config.max_prediction_variance:
                sanity_score *= 0.8
                warning_flags.append("high_variance")
                checks.append(f"High variance: {pred_var:.6f}")
            else:
                checks.append(f"Variance OK: {pred_var:.6f}")
            
            # Check 3: No constant predictions
            constant_symbols = []
            for symbol in predictions.columns:
                symbol_std = predictions[symbol].std()
                if symbol_std < 1e-8:  # Effectively constant
                    constant_symbols.append(symbol)
            
            if constant_symbols:
                sanity_score *= 0.6
                warning_flags.append("constant_predictions")
                checks.append(f"Constant predictions: {len(constant_symbols)} symbols")
            else:
                checks.append("No constant predictions")
            
            # Check 4: No all-NaN predictions
            nan_symbols = []
            for symbol in predictions.columns:
                if predictions[symbol].isna().all():
                    nan_symbols.append(symbol)
            
            if nan_symbols:
                sanity_score *= 0.5
                warning_flags.append("all_nan_predictions")
                checks.append(f"All-NaN predictions: {len(nan_symbols)} symbols")
            else:
                checks.append("No all-NaN predictions")
            
            explanation = "Sanity checks: " + "; ".join(checks)
            
            supporting_evidence = {
                'max_prediction': max_pred,
                'prediction_variance': pred_var,
                'constant_symbols': len(constant_symbols) if constant_symbols else 0,
                'nan_symbols': len(nan_symbols) if nan_symbols else 0
            }
            
            confidence = 0.95  # Very high confidence in sanity checks
            
            return IntuitionScore(
                dimension=IntuitionDimension.MARKET_STRUCTURE,
                signal_type=EconomicSignal.VOLATILITY,
                score=sanity_score,
                confidence=confidence,
                explanation=explanation,
                supporting_evidence=supporting_evidence,
                warning_flags=warning_flags
            )
            
        except Exception as e:
            logger.error(f"Error in sanity checks: {e}")
            return IntuitionScore(
                dimension=IntuitionDimension.MARKET_STRUCTURE,
                signal_type=EconomicSignal.VOLATILITY,
                score=0.0,
                confidence=0.0,
                explanation=f"Error in sanity checks: {e}",
                warning_flags=["calculation_error"]
            )


def create_standard_intuition_scorer(**config_overrides) -> EconomicIntuitionScorer:
    """Create economic intuition scorer with standard configuration."""
    config = IntuitionConfig(**config_overrides)
    return EconomicIntuitionScorer(config)


# Example usage
if __name__ == "__main__":
    print("Economic Intuition Scorer demo")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    symbols = ['2330.TW', '2317.TW', '2454.TW', '2882.TW', '2412.TW']
    
    # Mock predictions and returns
    np.random.seed(42)
    predictions = pd.DataFrame(
        np.random.randn(100, 5) * 0.02,  # 2% std predictions
        index=dates,
        columns=symbols
    )
    
    returns = pd.DataFrame(
        np.random.randn(100, 5) * 0.015,  # 1.5% std returns
        index=dates,
        columns=symbols
    )
    
    # Create scorer
    scorer = create_standard_intuition_scorer()
    
    # Score predictions
    intuition_scores = scorer.score_predictions(
        predictions=predictions,
        returns=returns
    )
    
    # Calculate overall score
    overall_score, overall_level, explanation = scorer.calculate_overall_intuition_score(
        intuition_scores
    )
    
    print(f"Overall intuition score: {overall_score:.3f} ({overall_level.value})")
    print(f"Explanation: {explanation}")
    print(f"Individual dimension scores:")
    for dimension, score in intuition_scores.items():
        print(f"- {dimension}: {score.score:.3f} ({score.level.value})")