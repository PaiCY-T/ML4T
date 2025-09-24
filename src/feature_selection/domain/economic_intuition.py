"""
Economic Intuition Scorer for Feature Selection.

Scores features based on economic intuition, financial theory alignment,
and business logic validity for the Taiwan stock market.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IntuitionCategory(Enum):
    """Economic intuition categories."""
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical" 
    MARKET_MICROSTRUCTURE = "market_microstructure"
    BEHAVIORAL = "behavioral"
    MACRO_ECONOMIC = "macro_economic"
    SECTOR_SPECIFIC = "sector_specific"
    UNKNOWN = "unknown"


class IntuitionScore(Enum):
    """Intuition score levels."""
    EXCELLENT = 5  # Strong economic rationale
    GOOD = 4      # Solid theoretical foundation  
    MODERATE = 3  # Some economic justification
    WEAK = 2      # Limited economic basis
    POOR = 1      # Questionable economic logic
    UNKNOWN = 0   # Cannot assess


@dataclass
class EconomicRationale:
    """Economic rationale for a feature."""
    category: IntuitionCategory
    theory: str
    mechanism: str
    expected_relationship: str  # "positive", "negative", "non_linear", "unknown"
    confidence: float  # 0.0 to 1.0
    references: List[str] = field(default_factory=list)
    taiwan_specific: bool = False
    market_regime_dependent: bool = False


@dataclass
class IntuitionResult:
    """Result of economic intuition scoring."""
    feature_name: str
    intuition_score: IntuitionScore
    category: IntuitionCategory
    rationale: EconomicRationale
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class EconomicIntuitionConfig:
    """Configuration for economic intuition scoring."""
    
    # Scoring weights by category
    category_weights: Dict[IntuitionCategory, float] = field(default_factory=lambda: {
        IntuitionCategory.FUNDAMENTAL: 1.0,
        IntuitionCategory.TECHNICAL: 0.8,
        IntuitionCategory.MARKET_MICROSTRUCTURE: 0.9,
        IntuitionCategory.BEHAVIORAL: 0.7,
        IntuitionCategory.MACRO_ECONOMIC: 0.8,
        IntuitionCategory.SECTOR_SPECIFIC: 0.9,
        IntuitionCategory.UNKNOWN: 0.3
    })
    
    # Taiwan market sector focus
    taiwan_sectors: Dict[str, float] = field(default_factory=lambda: {
        'technology': 0.35,      # Tech sector dominance in Taiwan
        'manufacturing': 0.20,    # Strong manufacturing base
        'financial': 0.15,       # Banking and finance
        'traditional': 0.15,     # Traditional industries  
        'healthcare': 0.08,      # Growing healthcare sector
        'other': 0.07           # Other sectors
    })
    
    # Economic theory mappings
    fundamental_theories: List[str] = field(default_factory=lambda: [
        'dividend_discount_model',
        'dcf_valuation',
        'pe_ratio_theory',
        'roe_dupont_analysis',
        'debt_equity_theory',
        'earnings_quality',
        'value_growth_theory'
    ])
    
    technical_theories: List[str] = field(default_factory=lambda: [
        'momentum_theory',
        'mean_reversion',
        'support_resistance',
        'trend_following',
        'volatility_clustering', 
        'relative_strength',
        'price_volume_relationship'
    ])
    
    # Feature interpretability requirements
    require_interpretation: bool = True
    min_confidence_threshold: float = 0.6
    penalize_black_box: bool = True


class BaseIntuitionEvaluator(ABC):
    """Base class for feature intuition evaluators."""
    
    @abstractmethod
    def evaluate_feature(
        self, 
        feature_name: str, 
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> IntuitionResult:
        """Evaluate feature for economic intuition."""
        pass
    
    @abstractmethod
    def get_category(self) -> IntuitionCategory:
        """Get the category this evaluator handles."""
        pass


class FundamentalIntuitionEvaluator(BaseIntuitionEvaluator):
    """Evaluator for fundamental analysis features."""
    
    def __init__(self, config: EconomicIntuitionConfig):
        self.config = config
        
        # Fundamental feature patterns
        self.valuation_patterns = {
            'pe': ('price_earnings_ratio', 'valuation_multiple', 'negative'),
            'pb': ('price_book_ratio', 'book_value_multiple', 'negative'),
            'peg': ('price_earnings_growth', 'growth_adjusted_valuation', 'negative'),
            'dividend_yield': ('dividend_discount_model', 'income_return', 'positive'),
            'ev_ebitda': ('enterprise_value_multiple', 'operational_valuation', 'negative'),
            'ps': ('price_sales_ratio', 'revenue_multiple', 'negative')
        }
        
        self.profitability_patterns = {
            'roe': ('return_on_equity', 'equity_efficiency', 'positive'),
            'roa': ('return_on_assets', 'asset_efficiency', 'positive'),
            'roic': ('return_on_invested_capital', 'capital_efficiency', 'positive'),
            'gross_margin': ('operational_efficiency', 'cost_control', 'positive'),
            'operating_margin': ('operational_profitability', 'operational_efficiency', 'positive'),
            'net_margin': ('net_profitability', 'overall_efficiency', 'positive')
        }
        
        self.growth_patterns = {
            'revenue_growth': ('top_line_growth', 'business_expansion', 'positive'),
            'earnings_growth': ('bottom_line_growth', 'profit_expansion', 'positive'),
            'eps_growth': ('earnings_per_share_growth', 'shareholder_value', 'positive'),
            'book_value_growth': ('equity_growth', 'balance_sheet_strength', 'positive')
        }
        
        self.leverage_patterns = {
            'debt_equity': ('leverage_ratio', 'financial_risk', 'negative'),
            'debt_ratio': ('debt_burden', 'solvency_risk', 'negative'),
            'interest_coverage': ('debt_service_ability', 'financial_stability', 'positive'),
            'current_ratio': ('liquidity_ratio', 'short_term_solvency', 'positive'),
            'quick_ratio': ('acid_test_ratio', 'immediate_liquidity', 'positive')
        }
    
    def evaluate_feature(
        self, 
        feature_name: str, 
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> IntuitionResult:
        """Evaluate fundamental feature intuition."""
        
        feature_lower = feature_name.lower()
        
        # Check valuation metrics
        for pattern, (theory, mechanism, relationship) in self.valuation_patterns.items():
            if pattern in feature_lower:
                return self._create_fundamental_result(
                    feature_name, pattern, theory, mechanism, relationship,
                    "Valuation multiple based on established financial theory"
                )
        
        # Check profitability metrics
        for pattern, (theory, mechanism, relationship) in self.profitability_patterns.items():
            if pattern in feature_lower:
                return self._create_fundamental_result(
                    feature_name, pattern, theory, mechanism, relationship,
                    "Profitability measure indicating management effectiveness"
                )
        
        # Check growth metrics
        for pattern, (theory, mechanism, relationship) in self.growth_patterns.items():
            if pattern in feature_lower:
                return self._create_fundamental_result(
                    feature_name, pattern, theory, mechanism, relationship,
                    "Growth measure indicating business expansion potential"
                )
        
        # Check leverage metrics
        for pattern, (theory, mechanism, relationship) in self.leverage_patterns.items():
            if pattern in feature_lower:
                return self._create_fundamental_result(
                    feature_name, pattern, theory, mechanism, relationship,
                    "Financial health measure indicating risk and stability"
                )
        
        # Generic fundamental feature
        fundamental_keywords = ['eps', 'revenue', 'assets', 'equity', 'cash', 'fcf']
        if any(keyword in feature_lower for keyword in fundamental_keywords):
            return IntuitionResult(
                feature_name=feature_name,
                intuition_score=IntuitionScore.MODERATE,
                category=IntuitionCategory.FUNDAMENTAL,
                rationale=EconomicRationale(
                    category=IntuitionCategory.FUNDAMENTAL,
                    theory="fundamental_analysis",
                    mechanism="financial_statement_information",
                    expected_relationship="unknown",
                    confidence=0.5
                ),
                recommendations=["Specify exact fundamental relationship", "Add clear economic interpretation"]
            )
        
        # Not a fundamental feature
        return IntuitionResult(
            feature_name=feature_name,
            intuition_score=IntuitionScore.UNKNOWN,
            category=IntuitionCategory.UNKNOWN,
            rationale=EconomicRationale(
                category=IntuitionCategory.UNKNOWN,
                theory="unknown",
                mechanism="not_fundamental",
                expected_relationship="unknown",
                confidence=0.0
            )
        )
    
    def _create_fundamental_result(
        self,
        feature_name: str,
        pattern: str,
        theory: str,
        mechanism: str,
        relationship: str,
        description: str
    ) -> IntuitionResult:
        """Create a fundamental intuition result."""
        
        # Determine score based on pattern recognition and theory strength
        if pattern in ['roe', 'roa', 'pe', 'pb']:  # Core fundamental metrics
            score = IntuitionScore.EXCELLENT
            confidence = 0.9
        elif pattern in ['revenue_growth', 'eps_growth', 'debt_equity']:  # Important metrics
            score = IntuitionScore.GOOD  
            confidence = 0.8
        else:  # Other fundamental metrics
            score = IntuitionScore.MODERATE
            confidence = 0.7
        
        # Check for Taiwan-specific considerations
        taiwan_specific = self._check_taiwan_relevance(feature_name)
        
        rationale = EconomicRationale(
            category=IntuitionCategory.FUNDAMENTAL,
            theory=theory,
            mechanism=mechanism,
            expected_relationship=relationship,
            confidence=confidence,
            references=[f"fundamental_analysis_{theory}"],
            taiwan_specific=taiwan_specific,
            market_regime_dependent=pattern in ['pe', 'pb', 'peg']  # Valuation multiples are regime-dependent
        )
        
        recommendations = []
        if not taiwan_specific:
            recommendations.append("Consider Taiwan market-specific adjustments")
        if pattern in ['pe', 'pb']:
            recommendations.append("Account for Taiwan market valuation premiums/discounts")
        
        return IntuitionResult(
            feature_name=feature_name,
            intuition_score=score,
            category=IntuitionCategory.FUNDAMENTAL,
            rationale=rationale,
            recommendations=recommendations,
            confidence=confidence
        )
    
    def _check_taiwan_relevance(self, feature_name: str) -> bool:
        """Check if feature has Taiwan-specific considerations."""
        taiwan_indicators = ['tw', 'taiwan', 'tse', 'twii', 'local']
        return any(indicator in feature_name.lower() for indicator in taiwan_indicators)
    
    def get_category(self) -> IntuitionCategory:
        """Get evaluator category."""
        return IntuitionCategory.FUNDAMENTAL


class TechnicalIntuitionEvaluator(BaseIntuitionEvaluator):
    """Evaluator for technical analysis features."""
    
    def __init__(self, config: EconomicIntuitionConfig):
        self.config = config
        
        # Technical indicator patterns
        self.momentum_patterns = {
            'rsi': ('relative_strength_index', 'momentum_oscillator', 'mean_reversion'),
            'macd': ('moving_average_convergence_divergence', 'trend_momentum', 'trend_following'),
            'stochastic': ('stochastic_oscillator', 'momentum_position', 'mean_reversion'),
            'williams_r': ('williams_percent_r', 'momentum_oscillator', 'mean_reversion'),
            'momentum': ('price_momentum', 'trend_continuation', 'momentum_theory'),
            'roc': ('rate_of_change', 'momentum_measure', 'momentum_theory')
        }
        
        self.trend_patterns = {
            'ma': ('moving_average', 'trend_identification', 'trend_following'),
            'ema': ('exponential_moving_average', 'trend_identification', 'trend_following'),
            'bollinger': ('bollinger_bands', 'volatility_envelope', 'mean_reversion'),
            'adx': ('average_directional_index', 'trend_strength', 'trend_following'),
            'parabolic_sar': ('parabolic_sar', 'trend_reversal', 'trend_following'),
            'trendline': ('trendline_analysis', 'support_resistance', 'technical_analysis')
        }
        
        self.volume_patterns = {
            'obv': ('on_balance_volume', 'volume_price_relationship', 'volume_analysis'),
            'volume_sma': ('volume_moving_average', 'volume_trend', 'volume_analysis'),
            'vwap': ('volume_weighted_average_price', 'institutional_behavior', 'microstructure'),
            'accumulation_distribution': ('accumulation_distribution_line', 'volume_accumulation', 'volume_analysis'),
            'money_flow': ('money_flow_index', 'volume_momentum', 'volume_analysis')
        }
        
        self.volatility_patterns = {
            'atr': ('average_true_range', 'volatility_measure', 'volatility_analysis'),
            'volatility': ('price_volatility', 'risk_measure', 'volatility_clustering'),
            'std': ('standard_deviation', 'price_dispersion', 'volatility_analysis'),
            'var': ('variance', 'price_variability', 'volatility_analysis'),
            'garch': ('garch_volatility', 'volatility_modeling', 'volatility_clustering')
        }
    
    def evaluate_feature(
        self, 
        feature_name: str, 
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> IntuitionResult:
        """Evaluate technical feature intuition."""
        
        feature_lower = feature_name.lower()
        
        # Check momentum indicators
        for pattern, (theory, mechanism, relationship) in self.momentum_patterns.items():
            if pattern in feature_lower:
                return self._create_technical_result(
                    feature_name, pattern, theory, mechanism, relationship,
                    "Technical momentum indicator based on price action"
                )
        
        # Check trend indicators
        for pattern, (theory, mechanism, relationship) in self.trend_patterns.items():
            if pattern in feature_lower:
                return self._create_technical_result(
                    feature_name, pattern, theory, mechanism, relationship,
                    "Technical trend indicator for direction identification"
                )
        
        # Check volume indicators
        for pattern, (theory, mechanism, relationship) in self.volume_patterns.items():
            if pattern in feature_lower:
                return self._create_technical_result(
                    feature_name, pattern, theory, mechanism, relationship,
                    "Volume-based technical indicator"
                )
        
        # Check volatility indicators
        for pattern, (theory, mechanism, relationship) in self.volatility_patterns.items():
            if pattern in feature_lower:
                return self._create_technical_result(
                    feature_name, pattern, theory, mechanism, relationship,
                    "Volatility-based risk and uncertainty measure"
                )
        
        # Generic technical feature
        technical_keywords = ['price', 'high', 'low', 'close', 'open', 'candle']
        if any(keyword in feature_lower for keyword in technical_keywords):
            return IntuitionResult(
                feature_name=feature_name,
                intuition_score=IntuitionScore.MODERATE,
                category=IntuitionCategory.TECHNICAL,
                rationale=EconomicRationale(
                    category=IntuitionCategory.TECHNICAL,
                    theory="technical_analysis",
                    mechanism="price_action",
                    expected_relationship="unknown",
                    confidence=0.5
                ),
                recommendations=["Specify exact technical relationship", "Add clear signal interpretation"]
            )
        
        # Not a technical feature
        return IntuitionResult(
            feature_name=feature_name,
            intuition_score=IntuitionScore.UNKNOWN,
            category=IntuitionCategory.UNKNOWN,
            rationale=EconomicRationale(
                category=IntuitionCategory.UNKNOWN,
                theory="unknown", 
                mechanism="not_technical",
                expected_relationship="unknown",
                confidence=0.0
            )
        )
    
    def _create_technical_result(
        self,
        feature_name: str,
        pattern: str,
        theory: str,
        mechanism: str,
        relationship: str,
        description: str
    ) -> IntuitionResult:
        """Create a technical intuition result."""
        
        # Determine score based on pattern recognition and theory strength
        if pattern in ['rsi', 'macd', 'ma', 'bollinger']:  # Well-established indicators
            score = IntuitionScore.GOOD
            confidence = 0.8
        elif pattern in ['atr', 'obv', 'vwap']:  # Solid technical indicators
            score = IntuitionScore.MODERATE
            confidence = 0.7
        else:  # Other technical indicators
            score = IntuitionScore.MODERATE
            confidence = 0.6
        
        # Check for Taiwan market considerations
        taiwan_considerations = self._get_taiwan_technical_considerations(pattern)
        
        rationale = EconomicRationale(
            category=IntuitionCategory.TECHNICAL,
            theory=theory,
            mechanism=mechanism,
            expected_relationship=relationship,
            confidence=confidence,
            references=[f"technical_analysis_{theory}"],
            taiwan_specific=bool(taiwan_considerations),
            market_regime_dependent=True  # Technical indicators are generally regime-dependent
        )
        
        recommendations = []
        if taiwan_considerations:
            recommendations.extend(taiwan_considerations)
        recommendations.append("Validate signal stability across different market conditions")
        
        return IntuitionResult(
            feature_name=feature_name,
            intuition_score=score,
            category=IntuitionCategory.TECHNICAL,
            rationale=rationale,
            recommendations=recommendations,
            confidence=confidence
        )
    
    def _get_taiwan_technical_considerations(self, pattern: str) -> List[str]:
        """Get Taiwan-specific considerations for technical patterns."""
        considerations = []
        
        if pattern in ['rsi', 'stochastic', 'williams_r']:
            considerations.append("Consider Taiwan market's high retail participation impact on momentum")
        
        if pattern in ['ma', 'ema']:
            considerations.append("Adjust for Taiwan market's shorter trading hours (4.5 hours)")
            
        if pattern in ['volume_sma', 'obv', 'vwap']:
            considerations.append("Account for Taiwan's concentrated trading volume patterns")
            
        if pattern in ['atr', 'volatility']:
            considerations.append("Consider Taiwan's 10% daily price limits impact on volatility")
        
        return considerations
    
    def get_category(self) -> IntuitionCategory:
        """Get evaluator category."""
        return IntuitionCategory.TECHNICAL


class MarketMicrostructureEvaluator(BaseIntuitionEvaluator):
    """Evaluator for market microstructure features."""
    
    def __init__(self, config: EconomicIntuitionConfig):
        self.config = config
        
        self.microstructure_patterns = {
            'bid_ask_spread': ('market_efficiency', 'liquidity_measure', 'negative'),
            'order_imbalance': ('supply_demand_pressure', 'price_pressure', 'positive'),
            'volume_weighted_price': ('institutional_behavior', 'execution_quality', 'information'),
            'tick_size': ('market_structure', 'price_discovery', 'microstructure'),
            'trade_count': ('market_activity', 'participation_level', 'positive'),
            'average_trade_size': ('institutional_vs_retail', 'market_participation', 'information'),
            'price_impact': ('market_depth', 'liquidity_impact', 'negative'),
            'implementation_shortfall': ('execution_cost', 'trading_efficiency', 'negative')
        }
    
    def evaluate_feature(
        self, 
        feature_name: str, 
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> IntuitionResult:
        """Evaluate microstructure feature intuition."""
        
        feature_lower = feature_name.lower()
        
        # Check microstructure patterns
        for pattern, (theory, mechanism, relationship) in self.microstructure_patterns.items():
            if pattern in feature_lower or pattern.replace('_', '') in feature_lower:
                return self._create_microstructure_result(
                    feature_name, pattern, theory, mechanism, relationship
                )
        
        # Generic microstructure keywords
        microstructure_keywords = ['spread', 'liquidity', 'depth', 'imbalance', 'flow', 'tick']
        if any(keyword in feature_lower for keyword in microstructure_keywords):
            return IntuitionResult(
                feature_name=feature_name,
                intuition_score=IntuitionScore.GOOD,
                category=IntuitionCategory.MARKET_MICROSTRUCTURE,
                rationale=EconomicRationale(
                    category=IntuitionCategory.MARKET_MICROSTRUCTURE,
                    theory="market_microstructure",
                    mechanism="market_structure_effect",
                    expected_relationship="unknown",
                    confidence=0.6,
                    taiwan_specific=True  # Microstructure is market-specific
                ),
                recommendations=["Specify exact microstructure relationship", "Validate against Taiwan market structure"]
            )
        
        return IntuitionResult(
            feature_name=feature_name,
            intuition_score=IntuitionScore.UNKNOWN,
            category=IntuitionCategory.UNKNOWN,
            rationale=EconomicRationale(
                category=IntuitionCategory.UNKNOWN,
                theory="unknown",
                mechanism="not_microstructure",
                expected_relationship="unknown",
                confidence=0.0
            )
        )
    
    def _create_microstructure_result(
        self,
        feature_name: str,
        pattern: str,
        theory: str,
        mechanism: str,
        relationship: str
    ) -> IntuitionResult:
        """Create microstructure intuition result."""
        
        # Microstructure features are highly relevant for Taiwan market
        score = IntuitionScore.GOOD
        confidence = 0.8
        
        taiwan_considerations = [
            "Consider Taiwan's T+2 settlement impact",
            "Account for concentrated market maker presence",
            "Validate against TSE market structure"
        ]
        
        if 'spread' in pattern:
            taiwan_considerations.append("Consider Taiwan's tick size rules impact on spreads")
        
        if 'volume' in pattern:
            taiwan_considerations.append("Account for Taiwan's high retail trading volume")
        
        rationale = EconomicRationale(
            category=IntuitionCategory.MARKET_MICROSTRUCTURE,
            theory=theory,
            mechanism=mechanism,
            expected_relationship=relationship,
            confidence=confidence,
            taiwan_specific=True,
            market_regime_dependent=True
        )
        
        return IntuitionResult(
            feature_name=feature_name,
            intuition_score=score,
            category=IntuitionCategory.MARKET_MICROSTRUCTURE,
            rationale=rationale,
            recommendations=taiwan_considerations,
            confidence=confidence
        )
    
    def get_category(self) -> IntuitionCategory:
        """Get evaluator category."""
        return IntuitionCategory.MARKET_MICROSTRUCTURE


class EconomicIntuitionScorer:
    """
    Economic Intuition Scorer for Feature Selection.
    
    Evaluates features based on:
    1. Economic theory alignment
    2. Financial intuition
    3. Business logic validity
    4. Taiwan market relevance
    """
    
    def __init__(self, config: Optional[EconomicIntuitionConfig] = None):
        """Initialize economic intuition scorer.
        
        Args:
            config: Configuration for economic intuition scoring
        """
        self.config = config or EconomicIntuitionConfig()
        
        # Initialize evaluators
        self.evaluators = [
            FundamentalIntuitionEvaluator(self.config),
            TechnicalIntuitionEvaluator(self.config),
            MarketMicrostructureEvaluator(self.config)
        ]
        
        self.scoring_results: Dict[str, IntuitionResult] = {}
        
        logger.info("Economic Intuition Scorer initialized")
        logger.info(f"Loaded {len(self.evaluators)} category evaluators")
    
    def score_features(
        self,
        features: List[str],
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, IntuitionResult]:
        """
        Score features for economic intuition.
        
        Args:
            features: List of feature names to score
            feature_data: Optional feature data for validation
            market_data: Optional market data for context
            feature_metadata: Optional feature metadata
            
        Returns:
            Dictionary mapping feature names to intuition results
        """
        logger.info(f"Scoring {len(features)} features for economic intuition")
        
        results = {}
        
        for feature in features:
            # Try each evaluator until one recognizes the feature
            feature_result = None
            
            for evaluator in self.evaluators:
                result = evaluator.evaluate_feature(feature, feature_data, market_data)
                
                if result.intuition_score != IntuitionScore.UNKNOWN:
                    feature_result = result
                    break
            
            # If no evaluator recognized it, create unknown result
            if feature_result is None:
                feature_result = IntuitionResult(
                    feature_name=feature,
                    intuition_score=IntuitionScore.UNKNOWN,
                    category=IntuitionCategory.UNKNOWN,
                    rationale=EconomicRationale(
                        category=IntuitionCategory.UNKNOWN,
                        theory="unknown",
                        mechanism="unrecognized_feature",
                        expected_relationship="unknown",
                        confidence=0.0
                    ),
                    warnings=["Feature not recognized by any economic evaluator"],
                    recommendations=["Provide clear economic interpretation", "Consider feature relevance"]
                )
            
            # Apply Taiwan-specific adjustments
            feature_result = self._apply_taiwan_adjustments(feature_result)
            
            # Apply interpretability requirements
            if self.config.require_interpretation:
                feature_result = self._apply_interpretability_check(feature_result)
            
            results[feature] = feature_result
        
        self.scoring_results = results
        
        # Log summary
        self._log_scoring_summary(results)
        
        return results
    
    def _apply_taiwan_adjustments(self, result: IntuitionResult) -> IntuitionResult:
        """Apply Taiwan market-specific adjustments."""
        
        # Boost score for Taiwan-relevant features
        if result.rationale.taiwan_specific:
            if result.intuition_score.value < IntuitionScore.EXCELLENT.value:
                # Upgrade score by one level for Taiwan-specific features
                new_score_value = min(result.intuition_score.value + 1, IntuitionScore.EXCELLENT.value)
                result.intuition_score = IntuitionScore(new_score_value)
                result.confidence = min(result.confidence + 0.1, 1.0)
        
        # Apply sector-specific adjustments
        feature_lower = result.feature_name.lower()
        for sector, weight in self.config.taiwan_sectors.items():
            if sector in feature_lower and weight > 0.2:  # High-weight sectors
                if result.intuition_score.value < IntuitionScore.GOOD.value:
                    result.recommendations.append(f"Feature relevant to Taiwan's {sector} sector (weight: {weight:.1%})")
        
        return result
    
    def _apply_interpretability_check(self, result: IntuitionResult) -> IntuitionResult:
        """Apply interpretability requirements."""
        
        # Penalize black-box features
        if self.config.penalize_black_box:
            blackbox_keywords = ['autoencoder', 'neural', 'deep', 'embedding', 'latent']
            if any(keyword in result.feature_name.lower() for keyword in blackbox_keywords):
                result.warnings.append("Black-box feature may lack economic interpretability")
                if result.intuition_score.value > IntuitionScore.WEAK.value:
                    result.intuition_score = IntuitionScore(result.intuition_score.value - 1)
                    result.confidence = max(result.confidence - 0.2, 0.0)
        
        # Check confidence threshold
        if result.confidence < self.config.min_confidence_threshold:
            result.warnings.append(f"Low confidence ({result.confidence:.2f}) below threshold ({self.config.min_confidence_threshold})")
        
        return result
    
    def _log_scoring_summary(self, results: Dict[str, IntuitionResult]) -> None:
        """Log scoring summary statistics."""
        
        if not results:
            return
        
        # Count by score
        score_counts = {score: 0 for score in IntuitionScore}
        for result in results.values():
            score_counts[result.intuition_score] += 1
        
        # Count by category
        category_counts = {category: 0 for category in IntuitionCategory}
        for result in results.values():
            category_counts[result.category] += 1
        
        # Calculate average confidence
        avg_confidence = np.mean([result.confidence for result in results.values()])
        
        logger.info(f"Economic Intuition Scoring Summary:")
        logger.info(f"  Total Features: {len(results)}")
        logger.info(f"  Average Confidence: {avg_confidence:.2f}")
        logger.info(f"  Score Distribution:")
        for score, count in score_counts.items():
            if count > 0:
                logger.info(f"    {score.name}: {count} ({count/len(results):.1%})")
        
        logger.info(f"  Category Distribution:")
        for category, count in category_counts.items():
            if count > 0:
                logger.info(f"    {category.value}: {count} ({count/len(results):.1%})")
        
        # Log high-quality features
        high_quality_features = [
            name for name, result in results.items()
            if result.intuition_score.value >= IntuitionScore.GOOD.value
            and result.confidence >= self.config.min_confidence_threshold
        ]
        
        logger.info(f"  High-Quality Features: {len(high_quality_features)} ({len(high_quality_features)/len(results):.1%})")
        if high_quality_features:
            logger.info(f"  Top Features: {high_quality_features[:5]}")
    
    def get_high_quality_features(
        self, 
        results: Optional[Dict[str, IntuitionResult]] = None,
        min_score: IntuitionScore = IntuitionScore.GOOD,
        min_confidence: Optional[float] = None
    ) -> List[str]:
        """Get list of high-quality features based on intuition scoring."""
        
        if results is None:
            results = self.scoring_results
        
        if not results:
            return []
        
        min_conf = min_confidence or self.config.min_confidence_threshold
        
        high_quality = []
        for name, result in results.items():
            if (result.intuition_score.value >= min_score.value and 
                result.confidence >= min_conf):
                high_quality.append(name)
        
        logger.info(f"Found {len(high_quality)} high-quality features")
        
        return high_quality
    
    def get_feature_scores(
        self,
        results: Optional[Dict[str, IntuitionResult]] = None
    ) -> pd.DataFrame:
        """Get feature scores as DataFrame."""
        
        if results is None:
            results = self.scoring_results
        
        if not results:
            return pd.DataFrame()
        
        score_data = []
        for name, result in results.items():
            score_data.append({
                'feature': name,
                'intuition_score': result.intuition_score.value,
                'category': result.category.value,
                'confidence': result.confidence,
                'taiwan_specific': result.rationale.taiwan_specific,
                'regime_dependent': result.rationale.market_regime_dependent,
                'theory': result.rationale.theory,
                'expected_relationship': result.rationale.expected_relationship,
                'warning_count': len(result.warnings),
                'recommendation_count': len(result.recommendations)
            })
        
        return pd.DataFrame(score_data)
    
    def generate_intuition_report(
        self,
        results: Optional[Dict[str, IntuitionResult]] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate comprehensive intuition scoring report."""
        
        if results is None:
            results = self.scoring_results
        
        report_data = []
        for name, result in results.items():
            report_data.append({
                'Feature': name,
                'Intuition_Score': result.intuition_score.name,
                'Score_Value': result.intuition_score.value,
                'Category': result.category.value,
                'Theory': result.rationale.theory,
                'Mechanism': result.rationale.mechanism,
                'Expected_Relationship': result.rationale.expected_relationship,
                'Confidence': result.confidence,
                'Taiwan_Specific': result.rationale.taiwan_specific,
                'Regime_Dependent': result.rationale.market_regime_dependent,
                'Warnings': '; '.join(result.warnings),
                'Recommendations': '; '.join(result.recommendations)
            })
        
        report_df = pd.DataFrame(report_data)
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            logger.info(f"Intuition scoring report saved to {output_path}")
        
        return report_df
    
    def get_category_statistics(
        self,
        results: Optional[Dict[str, IntuitionResult]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get statistics by feature category."""
        
        if results is None:
            results = self.scoring_results
        
        if not results:
            return {}
        
        category_stats = {}
        
        for category in IntuitionCategory:
            category_results = [r for r in results.values() if r.category == category]
            
            if category_results:
                scores = [r.intuition_score.value for r in category_results]
                confidences = [r.confidence for r in category_results]
                
                category_stats[category.value] = {
                    'count': len(category_results),
                    'avg_score': np.mean(scores),
                    'avg_confidence': np.mean(confidences),
                    'high_quality_count': len([r for r in category_results if r.intuition_score.value >= IntuitionScore.GOOD.value]),
                    'taiwan_specific_count': len([r for r in category_results if r.rationale.taiwan_specific])
                }
        
        return category_stats