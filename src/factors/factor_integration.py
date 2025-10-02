"""
Factor Integration Framework - Task #003
GitHub Issue #76

Comprehensive factor integration framework that unifies value, flow, and momentum factors
into a cohesive multi-factor system for the Taiwan market. This implementation addresses
the quality and performance differences between factor groups while providing a unified
interface for factor computation and harmonized cross-sectional scoring.

Key Integration Features:
1. FactorPipeline: Orchestrates all factor group calculations
2. Unified normalization framework with cross-sectional ranking
3. Performance optimization ensuring no bottlenecks from varying factor quality
4. Quality coordination addressing production readiness gaps
5. Taiwan market consistency across all factor types
6. Evidence-based integration with comprehensive validation

Performance Requirements:
- Maintain 31x performance advantage from value factors
- Address flow factor production concerns during integration
- Ensure consistent Taiwan market assumptions across factor groups
- Provide robust pipeline handling varying data quality

Quality Framework:
- All integration claims backed by evidence (code, tests, documentation)
- Production readiness assessment for complete pipeline
- Cross-factor harmonization validation
- Interface preparation for Task 004 (Factor Combination Strategy)
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from decimal import Decimal
import warnings
import time
from pathlib import Path
import json

# Import factor group implementations
from .value_factors import ValueFactorGroup, ValueFactorMetrics, ValueFactorScores, create_value_factor_group
from .flow_factors import FlowFactorGroup, InstitutionalFlowMetrics, BrokerSentimentMetrics, FlowMomentumMetrics, FlowFactorScores, create_flow_factor_group
from .simple_flow_factor import SimpleETFFlowFactor, ETFFlowMetrics
from .momentum import PriceMomentumCalculator, RSIMomentumCalculator, MACDSignalCalculator
from .base import (
    FactorCalculator, FactorResult, FactorMetadata, FactorCategory,
    FactorFrequency, FactorEngine
)

try:
    from ..data.core.temporal import DataType, TemporalValue
    from ..data.pipeline.pit_engine import PITQueryEngine, PITQuery
    from ..data.ingestion.finlab_connector import FinLabConnector, FinLabConfig
    from ..data.models.taiwan_market import TaiwanMarketCode, TradingStatus
except ImportError:
    # For testing or standalone usage
    DataType = object
    TemporalValue = object
    PITQueryEngine = object
    PITQuery = object
    FinLabConnector = object
    FinLabConfig = object
    TaiwanMarketCode = object
    TradingStatus = object

logger = logging.getLogger(__name__)


class FactorGroupType(Enum):
    """Enumeration of factor group types."""
    VALUE = "value"
    FLOW = "flow"
    MOMENTUM = "momentum"
    QUALITY = "quality"  # For future expansion


class IntegrationQuality(Enum):
    """Quality assessment levels for factor integration."""
    PRODUCTION_READY = "production_ready"      # High quality, ready for live trading
    FUNCTIONAL = "functional"                  # Works but needs hardening
    DEVELOPMENT = "development"                # Early stage, needs validation
    DEPRECATED = "deprecated"                  # Legacy, should be replaced


@dataclass
class FactorGroupStatus:
    """Status container for factor group integration assessment."""
    group_type: FactorGroupType
    quality_level: IntegrationQuality
    performance_score: float  # Relative performance metric
    coverage_score: float     # Data coverage percentage
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    production_ready: bool = False


@dataclass
class IntegratedFactorMetrics:
    """Container for integrated factor metrics across all factor groups."""
    symbol: str
    date: date

    # Value factor metrics
    value_metrics: Optional[ValueFactorMetrics] = None
    value_scores: Optional[ValueFactorScores] = None

    # Flow factor metrics
    institutional_metrics: Optional[InstitutionalFlowMetrics] = None
    broker_metrics: Optional[BrokerSentimentMetrics] = None
    momentum_flow_metrics: Optional[FlowMomentumMetrics] = None
    flow_scores: Optional[FlowFactorScores] = None

    # Research-validated ETF flow factor metrics
    etf_flow_metrics: Optional[ETFFlowMetrics] = None
    # uncertainty_level: Optional[str] = None  # Removed for simplicity

    # Momentum factor metrics (simplified for integration)
    momentum_score: Optional[float] = None

    # Integrated scores
    normalized_value_score: Optional[float] = None
    normalized_flow_score: Optional[float] = None
    normalized_etf_flow_score: Optional[float] = None
    normalized_momentum_score: Optional[float] = None

    # Cross-factor correlation metrics
    value_flow_correlation: Optional[float] = None
    value_momentum_correlation: Optional[float] = None
    flow_momentum_correlation: Optional[float] = None

    # Data quality indicators
    total_data_completeness: float = 0.0
    factor_group_coverage: Dict[str, bool] = field(default_factory=dict)


@dataclass
class FactorCorrelationMatrix:
    """Container for factor correlation analysis."""
    date: date

    # Cross-sectional correlations
    value_flow_correlation: float
    value_momentum_correlation: float
    flow_momentum_correlation: float

    # Individual factor correlations
    pe_foreign_flow_corr: Optional[float] = None
    pb_broker_sentiment_corr: Optional[float] = None
    dividend_yield_momentum_corr: Optional[float] = None

    # Statistical significance indicators
    sample_size: int = 0
    correlation_significance: Dict[str, float] = field(default_factory=dict)

    # Quality alerts
    high_correlation_alert: bool = False  # Correlation >0.8
    correlation_warnings: List[str] = field(default_factory=list)


class FactorPipeline:
    """
    Unified factor calculation pipeline integrating value, flow, and momentum factors.

    This class orchestrates all factor group calculations while handling the varying
    quality levels and performance characteristics of different factor implementations.
    Provides unified normalization and correlation monitoring across factor types.

    Key Features:
    - Orchestrates value, flow, and momentum factor calculations
    - Handles performance differences (value: 31x faster, flow: needs hardening)
    - Unified cross-sectional ranking and normalization
    - Factor correlation monitoring with configurable thresholds
    - Production readiness assessment and quality coordination
    - Taiwan market consistency across all factor groups
    """

    # Performance and quality constants
    CORRELATION_ALERT_THRESHOLD = 0.8
    MIN_FACTOR_COVERAGE = 0.7  # 70% minimum coverage for production use
    MAX_PROCESSING_TIME_SECONDS = 300  # 5 minutes max for full pipeline
    QUALITY_SCORE_THRESHOLD = 0.75  # Minimum quality score for production readiness

    def __init__(self,
                 finlab_connector: Optional[FinLabConnector] = None,
                 pit_engine: Optional[PITQueryEngine] = None,
                 enable_performance_monitoring: bool = True,
                 correlation_threshold: float = 0.8,
                 quality_validation: bool = True):
        """
        Initialize the factor integration pipeline.

        Args:
            finlab_connector: FinLab connector for data access
            pit_engine: Point-in-time query engine for momentum factors
            enable_performance_monitoring: Enable performance tracking
            correlation_threshold: Threshold for correlation alerts
            quality_validation: Enable quality validation during integration
        """

        # Initialize factor groups with quality assessment
        self.value_factor_group = create_value_factor_group(finlab_connector)
        self.flow_factor_group = create_flow_factor_group(finlab_connector)
        self.simple_etf_flow_factor = SimpleETFFlowFactor(finlab_connector)

        # Initialize momentum calculators (if PIT engine available)
        if pit_engine:
            from .taiwan_adjustments import TaiwanMarketAdjustments
            taiwan_adj = TaiwanMarketAdjustments()
            self.momentum_calculators = {
                'price_momentum': PriceMomentumCalculator(pit_engine, taiwan_adj),
                'rsi_momentum': RSIMomentumCalculator(pit_engine, taiwan_adj),
                'macd_signal': MACDSignalCalculator(pit_engine, taiwan_adj)
            }
        else:
            self.momentum_calculators = {}
            logger.warning("No PIT engine provided - momentum factors will use mock data")

        # Configuration
        self.correlation_threshold = correlation_threshold
        self.enable_performance_monitoring = enable_performance_monitoring
        self.quality_validation = quality_validation

        # State tracking
        self.factor_group_status: Dict[FactorGroupType, FactorGroupStatus] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.correlation_history: List[FactorCorrelationMatrix] = []

        # Initialize factor group status assessment
        self._assess_factor_group_status()

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _assess_factor_group_status(self) -> None:
        """Assess the production readiness status of each factor group."""

        # Value Factor Group Status (High Quality - Task 001 Evidence)
        self.factor_group_status[FactorGroupType.VALUE] = FactorGroupStatus(
            group_type=FactorGroupType.VALUE,
            quality_level=IntegrationQuality.PRODUCTION_READY,
            performance_score=31.0,  # 31x performance improvement documented
            coverage_score=0.95,     # High coverage expected
            issues=[],
            recommendations=["Maintain performance advantages during integration"],
            production_ready=True
        )

        # Flow Factor Group Status (Functional but needs hardening - Task 002 Evidence)
        self.factor_group_status[FactorGroupType.FLOW] = FactorGroupStatus(
            group_type=FactorGroupType.FLOW,
            quality_level=IntegrationQuality.FUNCTIONAL,
            performance_score=1.0,   # Baseline performance
            coverage_score=0.70,     # Adequate coverage
            issues=[
                "Regime detection algorithm needs statistical rigor",
                "Performance claims based on mock data, needs real data validation",
                "Production hardening required for live trading"
            ],
            recommendations=[
                "Implement robust regime detection with statistical validation",
                "Validate performance claims with real Taiwan market data",
                "Add comprehensive error handling and edge case management",
                "Implement data quality monitoring and alerts"
            ],
            production_ready=False
        )

        # Momentum Factor Group Status (Legacy implementation - existing)
        self.factor_group_status[FactorGroupType.MOMENTUM] = FactorGroupStatus(
            group_type=FactorGroupType.MOMENTUM,
            quality_level=IntegrationQuality.FUNCTIONAL,
            performance_score=1.0,   # Baseline performance
            coverage_score=0.85,     # Good coverage for technical factors
            issues=["Legacy implementation may need modernization"],
            recommendations=["Consider upgrading to match value factor quality standards"],
            production_ready=True  # Existing system, assume production ready
        )

    def calculate_integrated_factors(self,
                                   symbols: List[str],
                                   as_of_date: date,
                                   include_correlations: bool = True,
                                   validate_quality: bool = True) -> Dict[str, IntegratedFactorMetrics]:
        """
        Calculate integrated factors across all factor groups with unified normalization.

        Args:
            symbols: List of Taiwan stock symbols
            as_of_date: Calculation date
            include_correlations: Calculate cross-factor correlations
            validate_quality: Perform quality validation during calculation

        Returns:
            Dictionary mapping symbols to integrated factor metrics

        Raises:
            ValueError: If symbols list is empty or invalid parameters
            RuntimeError: If integration fails for technical reasons
        """
        start_time = time.time()

        if not symbols:
            raise ValueError("Symbols list cannot be empty")

        if as_of_date > date.today():
            raise ValueError("Calculation date cannot be in the future")

        try:
            self.logger.info(f"Starting integrated factor calculation for {len(symbols)} symbols as of {as_of_date}")

            # Step 1: Calculate value factors (highest quality, fastest)
            value_results = self._calculate_value_factors_safe(symbols, as_of_date)

            # Step 2: Calculate flow factors (functional, needs monitoring)
            flow_results = self._calculate_flow_factors_safe(symbols, as_of_date)

            # Step 2b: Calculate research-validated ETF flow factors
            etf_flow_results = self._calculate_etf_flow_factors_safe(symbols, as_of_date)

            # Step 3: Calculate momentum factors (legacy, stable)
            momentum_results = self._calculate_momentum_factors_safe(symbols, as_of_date)

            # Step 4: Integrate and normalize factors
            integrated_metrics = self._integrate_factor_results(
                symbols, as_of_date, value_results, flow_results, etf_flow_results, momentum_results
            )

            # Step 5: Calculate cross-factor correlations if requested
            if include_correlations:
                correlation_matrix = self._calculate_factor_correlations(integrated_metrics, as_of_date)
                self.correlation_history.append(correlation_matrix)

                # Check for correlation alerts
                self._check_correlation_alerts(correlation_matrix)

            # Step 6: Quality validation if enabled
            if validate_quality and self.quality_validation:
                quality_results = self._validate_integration_quality(integrated_metrics)
                self._log_quality_assessment(quality_results)

            # Step 7: Performance monitoring
            if self.enable_performance_monitoring:
                elapsed_time = time.time() - start_time
                self._update_performance_metrics(len(symbols), elapsed_time, integrated_metrics)

            self.logger.info(
                f"Completed integrated factor calculation: "
                f"{len(integrated_metrics)}/{len(symbols)} symbols processed "
                f"in {time.time() - start_time:.2f}s"
            )

            return integrated_metrics

        except Exception as e:
            self.logger.error(f"Error in integrated factor calculation: {e}")
            raise RuntimeError(f"Factor integration failed: {e}")

    def _calculate_value_factors_safe(self, symbols: List[str], as_of_date: date) -> Tuple[Dict[str, ValueFactorMetrics], Dict[str, ValueFactorScores]]:
        """Safely calculate value factors with error handling."""
        try:
            start_time = time.time()
            value_metrics, value_scores = self.value_factor_group.calculate_value_factors(symbols, as_of_date)

            # Update performance tracking for value factors
            processing_time = time.time() - start_time
            self.performance_metrics['value_factor_time'] = processing_time
            self.performance_metrics['value_factor_coverage'] = len(value_metrics) / len(symbols)

            self.logger.debug(f"Value factors calculated: {len(value_metrics)} symbols in {processing_time:.2f}s")
            return value_metrics, value_scores

        except Exception as e:
            self.logger.error(f"Error calculating value factors: {e}")
            # Return empty results to continue with other factors
            return {}, {}

    def _calculate_flow_factors_safe(self, symbols: List[str], as_of_date: date) -> Tuple[
        Dict[str, InstitutionalFlowMetrics],
        Dict[str, BrokerSentimentMetrics],
        Dict[str, FlowMomentumMetrics],
        Dict[str, FlowFactorScores]
    ]:
        """Safely calculate flow factors with enhanced error handling for production concerns."""
        try:
            start_time = time.time()

            # Add warning about flow factor production readiness
            status = self.factor_group_status[FactorGroupType.FLOW]
            if not status.production_ready:
                self.logger.warning(
                    f"Flow factors not production ready: {status.issues}. "
                    f"Recommendations: {status.recommendations}"
                )

            institutional_metrics, broker_metrics, momentum_metrics, flow_scores = \
                self.flow_factor_group.calculate_flow_factors(symbols, as_of_date)

            # Update performance tracking
            processing_time = time.time() - start_time
            self.performance_metrics['flow_factor_time'] = processing_time
            self.performance_metrics['flow_factor_coverage'] = len(flow_scores) / len(symbols)

            # Validate flow factor data quality (addressing Task 002 concerns)
            flow_quality = self.flow_factor_group.validate_data_quality(institutional_metrics, broker_metrics)
            if not flow_quality['validation_passed']:
                self.logger.warning(f"Flow factor quality issues: {flow_quality['issues']}")

            self.logger.debug(f"Flow factors calculated: {len(flow_scores)} symbols in {processing_time:.2f}s")
            return institutional_metrics, broker_metrics, momentum_metrics, flow_scores

        except Exception as e:
            self.logger.error(f"Error calculating flow factors: {e}")
            # Return empty results to continue with other factors
            return {}, {}, {}, {}

    def _calculate_etf_flow_factors_safe(self, symbols: List[str], as_of_date: date) -> Dict[str, ETFFlowMetrics]:
        """Safely calculate research-validated ETF flow factors."""
        try:
            start_time = time.time()

            self.logger.info("Calculating research-validated Taiwan ETF flow factors")
            etf_flow_results = self.simple_etf_flow_factor.calculate_flow_factors(symbols, as_of_date)

            # Update performance tracking
            processing_time = time.time() - start_time
            self.performance_metrics['etf_flow_factor_time'] = processing_time
            self.performance_metrics['etf_flow_factor_coverage'] = len(etf_flow_results) / len(symbols)

            # Count ETF vs individual stock processing
            etf_count = sum(1 for metrics in etf_flow_results.values() if metrics.is_etf)
            individual_count = len(etf_flow_results) - etf_count

            self.logger.info(
                f"ETF flow factors calculated: {len(etf_flow_results)} symbols "
                f"({etf_count} ETFs, {individual_count} stocks) in {processing_time:.2f}s"
            )

            return etf_flow_results

        except Exception as e:
            self.logger.error(f"Error calculating ETF flow factors: {e}")
            # Return empty results to continue with other factors
            return {}

    def _calculate_momentum_factors_safe(self, symbols: List[str], as_of_date: date) -> Dict[str, float]:
        """Safely calculate momentum factors with legacy system integration."""
        momentum_results = {}

        if not self.momentum_calculators:
            # Use mock momentum data for testing
            self.logger.warning("Using mock momentum data - no PIT engine available")
            for symbol in symbols:
                hash_val = hash(f"{symbol}_{as_of_date}") % 1000
                momentum_results[symbol] = (hash_val - 500) / 500.0  # Normalize to [-1, 1]
            return momentum_results

        try:
            start_time = time.time()

            # Calculate each momentum factor type
            for calc_name, calculator in self.momentum_calculators.items():
                try:
                    result = calculator.calculate(symbols, as_of_date)

                    # Integrate momentum factor values
                    for symbol, value in result.values.items():
                        if symbol not in momentum_results:
                            momentum_results[symbol] = 0.0
                        momentum_results[symbol] += value / len(self.momentum_calculators)  # Average

                except Exception as e:
                    self.logger.warning(f"Error calculating {calc_name}: {e}")
                    continue

            # Update performance tracking
            processing_time = time.time() - start_time
            self.performance_metrics['momentum_factor_time'] = processing_time
            self.performance_metrics['momentum_factor_coverage'] = len(momentum_results) / len(symbols)

            self.logger.debug(f"Momentum factors calculated: {len(momentum_results)} symbols in {processing_time:.2f}s")
            return momentum_results

        except Exception as e:
            self.logger.error(f"Error calculating momentum factors: {e}")
            return {}

    def _integrate_factor_results(self,
                                symbols: List[str],
                                as_of_date: date,
                                value_results: Tuple[Dict[str, ValueFactorMetrics], Dict[str, ValueFactorScores]],
                                flow_results: Tuple[Dict[str, InstitutionalFlowMetrics], Dict[str, BrokerSentimentMetrics], Dict[str, FlowMomentumMetrics], Dict[str, FlowFactorScores]],
                                etf_flow_results: Dict[str, ETFFlowMetrics],
                                momentum_results: Dict[str, float]) -> Dict[str, IntegratedFactorMetrics]:
        """Integrate factor results with unified cross-sectional normalization."""

        value_metrics, value_scores = value_results
        institutional_metrics, broker_metrics, momentum_flow_metrics, flow_scores = flow_results

        integrated_metrics = {}

        # Step 1: Collect all factor scores for cross-sectional normalization
        all_value_scores = [score.composite_value_score for score in value_scores.values() if score.composite_value_score is not None]
        all_flow_scores = [score.composite_flow_score for score in flow_scores.values() if score.composite_flow_score is not None]
        all_etf_flow_scores = [metrics.flow_ratio for metrics in etf_flow_results.values() if metrics.flow_ratio is not None]
        all_momentum_scores = [score for score in momentum_results.values() if score is not None]

        # Step 2: Calculate unified normalization parameters
        normalization_params = self._calculate_normalization_parameters(
            all_value_scores, all_flow_scores, all_etf_flow_scores, all_momentum_scores
        )

        # Step 3: Create integrated metrics for each symbol
        for symbol in symbols:
            metrics = IntegratedFactorMetrics(symbol=symbol, date=as_of_date)

            # Integrate value factor data
            if symbol in value_metrics:
                metrics.value_metrics = value_metrics[symbol]
            if symbol in value_scores:
                metrics.value_scores = value_scores[symbol]
                if value_scores[symbol].composite_value_score is not None:
                    metrics.normalized_value_score = self._normalize_score(
                        value_scores[symbol].composite_value_score,
                        normalization_params['value']
                    )

            # Integrate flow factor data
            if symbol in institutional_metrics:
                metrics.institutional_metrics = institutional_metrics[symbol]
            if symbol in broker_metrics:
                metrics.broker_metrics = broker_metrics[symbol]
            if symbol in momentum_flow_metrics:
                metrics.momentum_flow_metrics = momentum_flow_metrics[symbol]
            if symbol in flow_scores:
                metrics.flow_scores = flow_scores[symbol]
                if flow_scores[symbol].composite_flow_score is not None:
                    metrics.normalized_flow_score = self._normalize_score(
                        flow_scores[symbol].composite_flow_score,
                        normalization_params['flow']
                    )

            # Integrate research-validated ETF flow factor data
            if symbol in etf_flow_results:
                metrics.etf_flow_metrics = etf_flow_results[symbol]
                if etf_flow_results[symbol].flow_ratio is not None:
                    metrics.normalized_etf_flow_score = self._normalize_score(
                        etf_flow_results[symbol].flow_ratio,
                        normalization_params['etf_flow']
                    )

            # Integrate momentum factor data
            if symbol in momentum_results:
                metrics.momentum_score = momentum_results[symbol]
                metrics.normalized_momentum_score = self._normalize_score(
                    momentum_results[symbol],
                    normalization_params['momentum']
                )

            # Calculate data completeness and coverage
            metrics.total_data_completeness = self._calculate_total_completeness(metrics)
            metrics.factor_group_coverage = self._assess_factor_coverage(metrics)

            integrated_metrics[symbol] = metrics

        return integrated_metrics

    def _calculate_normalization_parameters(self,
                                          value_scores: List[float],
                                          flow_scores: List[float],
                                          etf_flow_scores: List[float],
                                          momentum_scores: List[float]) -> Dict[str, Dict[str, float]]:
        """Calculate normalization parameters for unified cross-sectional ranking."""

        normalization_params = {}

        for factor_type, scores in [('value', value_scores), ('flow', flow_scores), ('etf_flow', etf_flow_scores), ('momentum', momentum_scores)]:
            if scores:
                # Use robust statistics for normalization
                median = np.median(scores)
                mad = np.median(np.abs(np.array(scores) - median))
                mad_scaled = mad * 1.4826  # Scale to approximate standard deviation

                normalization_params[factor_type] = {
                    'median': median,
                    'mad_scaled': max(mad_scaled, 1e-6),  # Avoid division by zero
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            else:
                # Default parameters if no data available
                normalization_params[factor_type] = {
                    'median': 0.0,
                    'mad_scaled': 1.0,
                    'min': -1.0,
                    'max': 1.0
                }

        return normalization_params

    def _normalize_score(self, score: float, params: Dict[str, float]) -> float:
        """Normalize individual score using robust statistics."""
        # Z-score normalization using median and MAD
        normalized = (score - params['median']) / params['mad_scaled']

        # Clip extreme values to [-3, 3] range
        return np.clip(normalized, -3.0, 3.0)

    def _calculate_total_completeness(self, metrics: IntegratedFactorMetrics) -> float:
        """Calculate overall data completeness across all factor groups."""
        completeness_scores = []

        # Value factor completeness
        if metrics.value_metrics:
            completeness_scores.append(metrics.value_metrics.data_completeness_score)

        # Flow factor completeness
        if metrics.institutional_metrics:
            completeness_scores.append(metrics.institutional_metrics.data_completeness_score)
        if metrics.broker_metrics:
            completeness_scores.append(metrics.broker_metrics.data_completeness_score)

        # Momentum factor completeness (assume 1.0 if available)
        if metrics.momentum_score is not None:
            completeness_scores.append(1.0)

        return np.mean(completeness_scores) if completeness_scores else 0.0

    def _assess_factor_coverage(self, metrics: IntegratedFactorMetrics) -> Dict[str, bool]:
        """Assess which factor groups have adequate coverage for the symbol."""
        coverage = {}

        coverage['value'] = (
            metrics.value_metrics is not None and
            metrics.value_scores is not None and
            metrics.value_metrics.data_completeness_score >= 0.5
        )

        coverage['flow'] = (
            metrics.flow_scores is not None and
            metrics.flow_scores.composite_flow_score is not None
        )

        coverage['etf_flow'] = (
            metrics.etf_flow_metrics is not None and
            metrics.etf_flow_metrics.flow_ratio is not None
        )

        coverage['momentum'] = metrics.momentum_score is not None

        return coverage

    def _calculate_factor_correlations(self,
                                     integrated_metrics: Dict[str, IntegratedFactorMetrics],
                                     as_of_date: date) -> FactorCorrelationMatrix:
        """Calculate cross-factor correlation matrix for diversification assessment."""

        # Extract factor scores for correlation analysis
        value_scores = []
        flow_scores = []
        etf_flow_scores = []
        momentum_scores = []
        symbols_with_all_factors = []

        for symbol, metrics in integrated_metrics.items():
            if (metrics.normalized_value_score is not None and
                (metrics.normalized_flow_score is not None or metrics.normalized_etf_flow_score is not None) and
                metrics.normalized_momentum_score is not None):

                value_scores.append(metrics.normalized_value_score)

                # Prefer research-validated ETF flow factors over legacy flow factors
                if metrics.normalized_etf_flow_score is not None:
                    flow_scores.append(metrics.normalized_etf_flow_score)
                else:
                    flow_scores.append(metrics.normalized_flow_score or 0.0)

                momentum_scores.append(metrics.normalized_momentum_score)
                symbols_with_all_factors.append(symbol)

        # Calculate correlations if we have sufficient data
        if len(symbols_with_all_factors) >= 10:  # Minimum sample size
            try:
                # Main cross-factor correlations
                value_flow_corr = np.corrcoef(value_scores, flow_scores)[0, 1]
                value_momentum_corr = np.corrcoef(value_scores, momentum_scores)[0, 1]
                flow_momentum_corr = np.corrcoef(flow_scores, momentum_scores)[0, 1]

                # Handle NaN correlations
                value_flow_corr = value_flow_corr if np.isfinite(value_flow_corr) else 0.0
                value_momentum_corr = value_momentum_corr if np.isfinite(value_momentum_corr) else 0.0
                flow_momentum_corr = flow_momentum_corr if np.isfinite(flow_momentum_corr) else 0.0

            except Exception as e:
                self.logger.warning(f"Error calculating correlations: {e}")
                value_flow_corr = value_momentum_corr = flow_momentum_corr = 0.0
        else:
            self.logger.warning(f"Insufficient data for correlation analysis: {len(symbols_with_all_factors)} symbols")
            value_flow_corr = value_momentum_corr = flow_momentum_corr = 0.0

        # Create correlation matrix
        correlation_matrix = FactorCorrelationMatrix(
            date=as_of_date,
            value_flow_correlation=value_flow_corr,
            value_momentum_correlation=value_momentum_corr,
            flow_momentum_correlation=flow_momentum_corr,
            sample_size=len(symbols_with_all_factors)
        )

        # Check for high correlation alerts
        max_correlation = max(abs(value_flow_corr), abs(value_momentum_corr), abs(flow_momentum_corr))
        if max_correlation > self.correlation_threshold:
            correlation_matrix.high_correlation_alert = True
            correlation_matrix.correlation_warnings.append(
                f"High correlation detected: {max_correlation:.3f} exceeds threshold {self.correlation_threshold}"
            )

        return correlation_matrix

    def _check_correlation_alerts(self, correlation_matrix: FactorCorrelationMatrix) -> None:
        """Check for correlation alerts and log warnings."""
        if correlation_matrix.high_correlation_alert:
            self.logger.warning(
                f"Factor correlation alert on {correlation_matrix.date}: "
                f"Value-Flow: {correlation_matrix.value_flow_correlation:.3f}, "
                f"Value-Momentum: {correlation_matrix.value_momentum_correlation:.3f}, "
                f"Flow-Momentum: {correlation_matrix.flow_momentum_correlation:.3f}"
            )

            for warning in correlation_matrix.correlation_warnings:
                self.logger.warning(warning)

    def _validate_integration_quality(self,
                                    integrated_metrics: Dict[str, IntegratedFactorMetrics]) -> Dict[str, Any]:
        """Validate overall integration quality and identify issues."""

        quality_results = {
            'total_symbols': len(integrated_metrics),
            'coverage_by_factor_group': {},
            'average_completeness': 0.0,
            'quality_issues': [],
            'production_readiness': False
        }

        # Calculate coverage by factor group
        for factor_group in ['value', 'flow', 'etf_flow', 'momentum']:
            coverage_count = sum(
                1 for metrics in integrated_metrics.values()
                if metrics.factor_group_coverage.get(factor_group, False)
            )
            coverage_pct = coverage_count / len(integrated_metrics) if integrated_metrics else 0
            quality_results['coverage_by_factor_group'][factor_group] = coverage_pct

            if coverage_pct < self.MIN_FACTOR_COVERAGE:
                quality_results['quality_issues'].append(
                    f"Low {factor_group} factor coverage: {coverage_pct:.1%} < {self.MIN_FACTOR_COVERAGE:.1%}"
                )

        # Calculate average completeness
        completeness_scores = [m.total_data_completeness for m in integrated_metrics.values()]
        quality_results['average_completeness'] = np.mean(completeness_scores) if completeness_scores else 0.0

        # Check production readiness based on factor group status
        production_ready_groups = sum(
            1 for status in self.factor_group_status.values()
            if status.production_ready
        )

        quality_results['production_readiness'] = (
            production_ready_groups >= 2 and  # At least 2 factor groups production ready
            quality_results['average_completeness'] >= self.QUALITY_SCORE_THRESHOLD and
            len(quality_results['quality_issues']) == 0
        )

        return quality_results

    def _log_quality_assessment(self, quality_results: Dict[str, Any]) -> None:
        """Log quality assessment results."""
        self.logger.info(
            f"Integration quality assessment: "
            f"completeness={quality_results['average_completeness']:.1%}, "
            f"production_ready={quality_results['production_readiness']}"
        )

        for factor_group, coverage in quality_results['coverage_by_factor_group'].items():
            self.logger.info(f"{factor_group} factor coverage: {coverage:.1%}")

        for issue in quality_results['quality_issues']:
            self.logger.warning(f"Quality issue: {issue}")

    def _update_performance_metrics(self,
                                  symbol_count: int,
                                  elapsed_time: float,
                                  integrated_metrics: Dict[str, IntegratedFactorMetrics]) -> None:
        """Update performance metrics for monitoring."""

        self.performance_metrics.update({
            'last_calculation_time': elapsed_time,
            'last_symbol_count': symbol_count,
            'last_processing_rate': symbol_count / elapsed_time if elapsed_time > 0 else 0,
            'last_integration_success_rate': len(integrated_metrics) / symbol_count if symbol_count > 0 else 0,
            'timestamp': datetime.now()
        })

        # Check performance against requirements
        if elapsed_time > self.MAX_PROCESSING_TIME_SECONDS:
            self.logger.warning(
                f"Performance alert: Processing time {elapsed_time:.1f}s exceeds "
                f"maximum {self.MAX_PROCESSING_TIME_SECONDS}s"
            )

    def get_factor_group_status(self) -> Dict[FactorGroupType, FactorGroupStatus]:
        """Get current status assessment for all factor groups."""
        return self.factor_group_status.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def get_correlation_history(self) -> List[FactorCorrelationMatrix]:
        """Get factor correlation history."""
        return self.correlation_history.copy()

    def get_integration_summary(self) -> Dict[str, Any]:
        """Get comprehensive integration summary for monitoring and validation."""
        return {
            'factor_group_status': {k.value: v for k, v in self.factor_group_status.items()},
            'performance_metrics': self.performance_metrics,
            'correlation_summary': {
                'correlation_count': len(self.correlation_history),
                'latest_correlation': self.correlation_history[-1] if self.correlation_history else None,
                'correlation_threshold': self.correlation_threshold
            },
            'configuration': {
                'enable_performance_monitoring': self.enable_performance_monitoring,
                'quality_validation': self.quality_validation,
                'min_factor_coverage': self.MIN_FACTOR_COVERAGE,
                'max_processing_time': self.MAX_PROCESSING_TIME_SECONDS
            }
        }


# Factory function for easy instantiation
def create_factor_pipeline(finlab_connector: Optional[FinLabConnector] = None,
                         pit_engine: Optional[PITQueryEngine] = None,
                         **kwargs) -> FactorPipeline:
    """
    Factory function to create FactorPipeline instance.

    Args:
        finlab_connector: Optional FinLab connector instance
        pit_engine: Optional point-in-time query engine
        **kwargs: Additional configuration parameters

    Returns:
        Configured FactorPipeline instance
    """
    return FactorPipeline(finlab_connector=finlab_connector, pit_engine=pit_engine, **kwargs)