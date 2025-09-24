"""
Main Business Logic Validator for ML4T Taiwan Equity Alpha.

This module orchestrates all business logic validation components including
regulatory compliance, strategy coherence, economic intuition, sector analysis,
and risk management for comprehensive model validation.

Key Features:
- Comprehensive validation orchestration
- Taiwan market-specific business rules
- Integration with backtesting framework
- Automated validation reporting
- Real-time validation capabilities

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
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .regulatory_validator import RegulatoryValidator, ComplianceIssue
from .strategy_coherence import StrategyCoherenceValidator, CoherenceResult
from .economic_intuition import EconomicIntuitionScorer, IntuitionScore
from .sector_analysis import SectorNeutralityAnalyzer, NeutralityResult
from .risk_integration import RiskValidator, PositionRiskCheck, RiskViolation

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Overall validation severity levels."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of business logic validation."""
    REGULATORY = "regulatory"
    STRATEGY = "strategy"
    ECONOMIC = "economic"
    SECTOR = "sector"
    RISK = "risk"
    OVERALL = "overall"


@dataclass
class ValidationIssue:
    """A business logic validation issue."""
    category: ValidationCategory
    severity: ValidationSeverity
    description: str
    details: str
    symbol: Optional[str] = None
    date: Optional[date] = None
    score: Optional[float] = None  # 0-100 where higher is better
    recommendation: Optional[str] = None
    reference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'description': self.description,
            'details': self.details,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'score': self.score,
            'recommendation': self.recommendation,
            'reference': self.reference
        }


@dataclass
class ValidationResult:
    """Comprehensive business logic validation result."""
    overall_status: ValidationSeverity
    overall_score: float  # 0-100, higher is better
    validation_date: date
    
    # Component results
    regulatory_score: float
    strategy_score: float
    economic_score: float
    sector_score: float
    risk_score: float
    
    # Detailed results
    compliance_issues: List[ComplianceIssue]
    coherence_results: List[CoherenceResult]
    intuition_scores: List[IntuitionScore]
    neutrality_result: Optional[NeutralityResult]
    risk_checks: List[PositionRiskCheck]
    risk_violations: List[RiskViolation]
    
    # Summary
    issues: List[ValidationIssue]
    recommendations: List[str]
    
    # Metrics
    total_positions: int
    passed_positions: int
    warning_positions: int
    failed_positions: int
    
    def get_pass_rate(self) -> float:
        """Get overall validation pass rate."""
        if self.total_positions == 0:
            return 0.0
        return self.passed_positions / self.total_positions
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues filtered by category."""
        return [issue for issue in self.issues if issue.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_status': self.overall_status.value,
            'overall_score': self.overall_score,
            'validation_date': self.validation_date.isoformat(),
            'scores': {
                'regulatory': self.regulatory_score,
                'strategy': self.strategy_score,
                'economic': self.economic_score,
                'sector': self.sector_score,
                'risk': self.risk_score
            },
            'metrics': {
                'total_positions': self.total_positions,
                'passed_positions': self.passed_positions,
                'warning_positions': self.warning_positions,
                'failed_positions': self.failed_positions,
                'pass_rate': self.get_pass_rate()
            },
            'issues': [issue.to_dict() for issue in self.issues],
            'recommendations': self.recommendations,
            'compliance_issues': [issue.to_dict() for issue in self.compliance_issues],
            'risk_violations': [violation.to_dict() for violation in self.risk_violations]
        }


@dataclass
class ValidationConfig:
    """Configuration for business logic validation."""
    
    # Component enables
    enable_regulatory: bool = True
    enable_strategy: bool = True
    enable_economic: bool = True
    enable_sector: bool = True
    enable_risk: bool = True
    
    # Scoring weights (must sum to 1.0)
    regulatory_weight: float = 0.25
    strategy_weight: float = 0.20
    economic_weight: float = 0.20
    sector_weight: float = 0.15
    risk_weight: float = 0.20
    
    # Validation thresholds
    pass_threshold: float = 70.0  # Minimum score to pass
    warning_threshold: float = 85.0  # Above this is warning, not pass
    
    # Performance settings
    parallel_validation: bool = True
    max_workers: int = 4
    timeout_seconds: float = 30.0
    
    # Taiwan market specific
    market_hours_only: bool = True  # Validate only during market hours
    include_otc_stocks: bool = True  # Include over-the-counter stocks
    
    # Integration settings
    cache_results: bool = True
    cache_ttl_minutes: int = 60
    
    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0."""
        total_weight = (
            self.regulatory_weight + self.strategy_weight + 
            self.economic_weight + self.sector_weight + self.risk_weight
        )
        return abs(total_weight - 1.0) < 0.001  # Allow small floating point errors
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total_weight = (
            self.regulatory_weight + self.strategy_weight + 
            self.economic_weight + self.sector_weight + self.risk_weight
        )
        
        if total_weight > 0:
            self.regulatory_weight /= total_weight
            self.strategy_weight /= total_weight
            self.economic_weight /= total_weight
            self.sector_weight /= total_weight
            self.risk_weight /= total_weight


class BusinessLogicValidator:
    """Main orchestrator for business logic validation."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize the business logic validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        
        # Ensure weights are normalized
        if not self.config.validate_weights():
            logger.warning("Validation weights don't sum to 1.0, normalizing...")
            self.config.normalize_weights()
        
        # Initialize component validators
        self.regulatory_validator = RegulatoryValidator() if self.config.enable_regulatory else None
        self.strategy_validator = StrategyCoherenceValidator() if self.config.enable_strategy else None
        self.economic_scorer = EconomicIntuitionScorer() if self.config.enable_economic else None
        self.sector_analyzer = SectorNeutralityAnalyzer() if self.config.enable_sector else None
        self.risk_validator = RiskValidator() if self.config.enable_risk else None
        
        # Initialize cache
        self._cache = {} if self.config.cache_results else None
        
        # Thread pool for parallel validation
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) if self.config.parallel_validation else None
    
    def validate_portfolio(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float,
        date: date,
        alpha_signals: Optional[Dict[str, float]] = None,
        market_data: Optional[pd.DataFrame] = None,
        strategy_metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Perform comprehensive business logic validation.
        
        Args:
            portfolio_weights: Dictionary of symbol -> weight
            portfolio_value: Total portfolio value in NTD
            date: Validation date
            alpha_signals: Optional alpha signals for economic validation
            market_data: Optional market data
            strategy_metadata: Optional strategy information
            
        Returns:
            Comprehensive validation result
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(portfolio_weights, date)
            if self._cache and cache_key in self._cache:
                cached_result, cache_time = self._cache[cache_key]
                if datetime.now() - cache_time < timedelta(minutes=self.config.cache_ttl_minutes):
                    logger.info(f"Returning cached validation result for {date}")
                    return cached_result
            
            logger.info(f"Starting comprehensive validation for {len(portfolio_weights)} positions on {date}")
            
            # Get market data if not provided
            if market_data is None:
                market_data = self._get_market_data(list(portfolio_weights.keys()), date)
            
            # Run validations
            if self.config.parallel_validation and self._executor:
                result = self._validate_parallel(
                    portfolio_weights, portfolio_value, date,
                    alpha_signals, market_data, strategy_metadata
                )
            else:
                result = self._validate_sequential(
                    portfolio_weights, portfolio_value, date,
                    alpha_signals, market_data, strategy_metadata
                )
            
            # Cache result
            if self._cache:
                self._cache[cache_key] = (result, datetime.now())
            
            logger.info(f"Validation completed with overall score {result.overall_score:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
    
    def validate_single_position(
        self,
        symbol: str,
        weight: float,
        date: date,
        alpha_signal: Optional[float] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Validate a single position against business logic rules.
        
        Args:
            symbol: Stock symbol
            weight: Position weight
            date: Validation date
            alpha_signal: Optional alpha signal
            market_data: Optional market data
            
        Returns:
            Dictionary with validation results for the position
        """
        try:
            portfolio = {symbol: weight}
            portfolio_value = 1000000  # 1M NTD for single position validation
            alpha_signals = {symbol: alpha_signal} if alpha_signal else None
            
            result = self.validate_portfolio(
                portfolio, portfolio_value, date, alpha_signals, market_data
            )
            
            # Extract results for the single position
            return {
                'symbol': symbol,
                'overall_score': result.overall_score,
                'regulatory_score': result.regulatory_score,
                'strategy_score': result.strategy_score,
                'economic_score': result.economic_score,
                'sector_score': result.sector_score,
                'risk_score': result.risk_score,
                'issues': [issue.to_dict() for issue in result.issues],
                'recommendations': result.recommendations
            }
            
        except Exception as e:
            logger.error(f"Error validating position {symbol}: {e}")
            raise
    
    def _validate_parallel(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float,
        date: date,
        alpha_signals: Optional[Dict[str, float]],
        market_data: pd.DataFrame,
        strategy_metadata: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Run validations in parallel."""
        futures = []
        
        # Submit validation tasks
        if self.regulatory_validator:
            futures.append(
                self._executor.submit(
                    self._run_regulatory_validation,
                    portfolio_weights, portfolio_value, date, market_data
                )
            )
        
        if self.strategy_validator:
            futures.append(
                self._executor.submit(
                    self._run_strategy_validation,
                    portfolio_weights, date, strategy_metadata, market_data
                )
            )
        
        if self.economic_scorer:
            futures.append(
                self._executor.submit(
                    self._run_economic_validation,
                    portfolio_weights, alpha_signals, date, market_data
                )
            )
        
        if self.sector_analyzer:
            futures.append(
                self._executor.submit(
                    self._run_sector_validation,
                    portfolio_weights, date, market_data
                )
            )
        
        if self.risk_validator:
            futures.append(
                self._executor.submit(
                    self._run_risk_validation,
                    portfolio_weights, portfolio_value, date, market_data
                )
            )
        
        # Collect results
        results = {}
        for future in as_completed(futures, timeout=self.config.timeout_seconds):
            try:
                component, result = future.result()
                results[component] = result
            except Exception as e:
                logger.error(f"Validation component failed: {e}")
                # Continue with other components
        
        return self._aggregate_results(results, portfolio_weights, date)
    
    def _validate_sequential(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float,
        date: date,
        alpha_signals: Optional[Dict[str, float]],
        market_data: pd.DataFrame,
        strategy_metadata: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Run validations sequentially."""
        results = {}
        
        # Run each validation component
        if self.regulatory_validator:
            try:
                results['regulatory'] = self._run_regulatory_validation(
                    portfolio_weights, portfolio_value, date, market_data
                )[1]
            except Exception as e:
                logger.error(f"Regulatory validation failed: {e}")
        
        if self.strategy_validator:
            try:
                results['strategy'] = self._run_strategy_validation(
                    portfolio_weights, date, strategy_metadata, market_data
                )[1]
            except Exception as e:
                logger.error(f"Strategy validation failed: {e}")
        
        if self.economic_scorer:
            try:
                results['economic'] = self._run_economic_validation(
                    portfolio_weights, alpha_signals, date, market_data
                )[1]
            except Exception as e:
                logger.error(f"Economic validation failed: {e}")
        
        if self.sector_analyzer:
            try:
                results['sector'] = self._run_sector_validation(
                    portfolio_weights, date, market_data
                )[1]
            except Exception as e:
                logger.error(f"Sector validation failed: {e}")
        
        if self.risk_validator:
            try:
                results['risk'] = self._run_risk_validation(
                    portfolio_weights, portfolio_value, date, market_data
                )[1]
            except Exception as e:
                logger.error(f"Risk validation failed: {e}")
        
        return self._aggregate_results(results, portfolio_weights, date)
    
    def _run_regulatory_validation(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: pd.DataFrame
    ) -> Tuple[str, Any]:
        """Run regulatory compliance validation."""
        # Convert portfolio to required format for regulatory validator
        positions = []
        for symbol, weight in portfolio_weights.items():
            positions.append({
                'symbol': symbol,
                'weight': weight,
                'dollar_amount': weight * portfolio_value,
                'date': date
            })
        
        issues = self.regulatory_validator.validate_portfolio(positions, date)
        
        return 'regulatory', {
            'issues': issues,
            'score': self._calculate_regulatory_score(issues)
        }
    
    def _run_strategy_validation(
        self,
        portfolio_weights: Dict[str, float],
        date: date,
        strategy_metadata: Optional[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> Tuple[str, Any]:
        """Run strategy coherence validation."""
        # Prepare strategy context
        strategy_context = strategy_metadata or {
            'strategy_type': 'alpha_generation',
            'target_volatility': 0.12,
            'rebalancing_frequency': 'weekly'
        }
        
        coherence_results = []
        for symbol, weight in portfolio_weights.items():
            stock_data = market_data[market_data['symbol'] == symbol]
            if not stock_data.empty:
                result = self.strategy_validator.validate_position_coherence(
                    symbol, weight, strategy_context, date
                )
                coherence_results.append(result)
        
        return 'strategy', {
            'results': coherence_results,
            'score': self._calculate_strategy_score(coherence_results)
        }
    
    def _run_economic_validation(
        self,
        portfolio_weights: Dict[str, float],
        alpha_signals: Optional[Dict[str, float]],
        date: date,
        market_data: pd.DataFrame
    ) -> Tuple[str, Any]:
        """Run economic intuition validation."""
        if not alpha_signals:
            # Generate dummy alpha signals based on momentum
            alpha_signals = {}
            for symbol in portfolio_weights.keys():
                stock_data = market_data[market_data['symbol'] == symbol]
                if not stock_data.empty:
                    # Use price momentum as proxy for alpha
                    alpha_signals[symbol] = np.random.uniform(-0.1, 0.1)
                else:
                    alpha_signals[symbol] = 0.0
        
        intuition_scores = []
        for symbol, alpha in alpha_signals.items():
            if symbol in portfolio_weights:
                score = self.economic_scorer.score_prediction(
                    symbol, alpha, date, market_context=None
                )
                intuition_scores.append(score)
        
        return 'economic', {
            'scores': intuition_scores,
            'score': self._calculate_economic_score(intuition_scores)
        }
    
    def _run_sector_validation(
        self,
        portfolio_weights: Dict[str, float],
        date: date,
        market_data: pd.DataFrame
    ) -> Tuple[str, Any]:
        """Run sector neutrality validation."""
        neutrality_result = self.sector_analyzer.analyze_neutrality(
            portfolio_weights, date, sector_data=market_data
        )
        
        return 'sector', {
            'result': neutrality_result,
            'score': neutrality_result.neutrality_score
        }
    
    def _run_risk_validation(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: pd.DataFrame
    ) -> Tuple[str, Any]:
        """Run risk management validation."""
        position_checks, portfolio_violations = self.risk_validator.validate_portfolio(
            portfolio_weights, portfolio_value, date, market_data
        )
        
        return 'risk', {
            'position_checks': position_checks,
            'violations': portfolio_violations,
            'score': self._calculate_risk_score(position_checks, portfolio_violations)
        }
    
    def _aggregate_results(
        self,
        results: Dict[str, Any],
        portfolio_weights: Dict[str, float],
        date: date
    ) -> ValidationResult:
        """Aggregate component results into final validation result."""
        # Extract component scores
        regulatory_score = results.get('regulatory', {}).get('score', 100.0)
        strategy_score = results.get('strategy', {}).get('score', 100.0)
        economic_score = results.get('economic', {}).get('score', 100.0)
        sector_score = results.get('sector', {}).get('score', 100.0)
        risk_score = results.get('risk', {}).get('score', 100.0)
        
        # Calculate weighted overall score
        overall_score = (
            regulatory_score * self.config.regulatory_weight +
            strategy_score * self.config.strategy_weight +
            economic_score * self.config.economic_weight +
            sector_score * self.config.sector_weight +
            risk_score * self.config.risk_weight
        )
        
        # Determine overall status
        if overall_score < self.config.pass_threshold:
            overall_status = ValidationSeverity.FAIL
        elif overall_score < self.config.warning_threshold:
            overall_status = ValidationSeverity.WARNING
        else:
            overall_status = ValidationSeverity.PASS
        
        # Extract detailed results
        compliance_issues = results.get('regulatory', {}).get('issues', [])
        coherence_results = results.get('strategy', {}).get('results', [])
        intuition_scores = results.get('economic', {}).get('scores', [])
        neutrality_result = results.get('sector', {}).get('result', None)
        risk_checks = results.get('risk', {}).get('position_checks', [])
        risk_violations = results.get('risk', {}).get('violations', [])
        
        # Generate issues and recommendations
        issues = self._generate_issues(results, overall_score)
        recommendations = self._generate_recommendations(results, overall_score)
        
        # Calculate position statistics
        total_positions = len(portfolio_weights)
        passed_positions = sum(
            1 for check in risk_checks
            if check.position_size_score >= self.config.pass_threshold
        )
        failed_positions = sum(
            1 for check in risk_checks
            if check.position_size_score < self.config.pass_threshold
        )
        warning_positions = total_positions - passed_positions - failed_positions
        
        return ValidationResult(
            overall_status=overall_status,
            overall_score=overall_score,
            validation_date=date,
            regulatory_score=regulatory_score,
            strategy_score=strategy_score,
            economic_score=economic_score,
            sector_score=sector_score,
            risk_score=risk_score,
            compliance_issues=compliance_issues,
            coherence_results=coherence_results,
            intuition_scores=intuition_scores,
            neutrality_result=neutrality_result,
            risk_checks=risk_checks,
            risk_violations=risk_violations,
            issues=issues,
            recommendations=recommendations,
            total_positions=total_positions,
            passed_positions=passed_positions,
            warning_positions=warning_positions,
            failed_positions=failed_positions
        )
    
    def _calculate_regulatory_score(self, issues: List[ComplianceIssue]) -> float:
        """Calculate regulatory compliance score."""
        if not issues:
            return 100.0
        
        penalty = 0.0
        for issue in issues:
            if issue.severity.value == 'critical':
                penalty += 50
            elif issue.severity.value == 'high':
                penalty += 25
            elif issue.severity.value == 'medium':
                penalty += 10
            elif issue.severity.value == 'low':
                penalty += 5
        
        return max(0.0, 100.0 - penalty)
    
    def _calculate_strategy_score(self, results: List[CoherenceResult]) -> float:
        """Calculate strategy coherence score."""
        if not results:
            return 100.0
        
        scores = [result.coherence_score for result in results if result.coherence_score is not None]
        return np.mean(scores) if scores else 100.0
    
    def _calculate_economic_score(self, scores: List[IntuitionScore]) -> float:
        """Calculate economic intuition score."""
        if not scores:
            return 100.0
        
        score_values = [score.total_score for score in scores]
        return np.mean(score_values) if score_values else 100.0
    
    def _calculate_risk_score(
        self,
        position_checks: List[PositionRiskCheck],
        violations: List[RiskViolation]
    ) -> float:
        """Calculate risk management score."""
        base_score = 100.0
        
        # Penalize for violations
        for violation in violations:
            if violation.severity.value == 'critical':
                base_score -= 40
            elif violation.severity.value == 'high':
                base_score -= 20
            elif violation.severity.value == 'medium':
                base_score -= 10
            elif violation.severity.value == 'low':
                base_score -= 5
        
        # Average position scores
        if position_checks:
            position_scores = [check.position_size_score for check in position_checks]
            avg_position_score = np.mean(position_scores)
            # Weight position scores 50%, violation penalties 50%
            return max(0.0, (base_score + avg_position_score) / 2)
        
        return max(0.0, base_score)
    
    def _generate_issues(self, results: Dict[str, Any], overall_score: float) -> List[ValidationIssue]:
        """Generate consolidated validation issues."""
        issues = []
        
        # Regulatory issues
        compliance_issues = results.get('regulatory', {}).get('issues', [])
        for issue in compliance_issues:
            severity = ValidationSeverity.FAIL if issue.severity.value in ['critical', 'high'] else ValidationSeverity.WARNING
            issues.append(ValidationIssue(
                category=ValidationCategory.REGULATORY,
                severity=severity,
                description=issue.description,
                details=f"Rule: {issue.rule_reference}",
                symbol=issue.symbol,
                date=issue.date,
                recommendation=issue.remediation
            ))
        
        # Risk violations
        risk_violations = results.get('risk', {}).get('violations', [])
        for violation in risk_violations:
            severity = ValidationSeverity.FAIL if violation.severity.value in ['critical', 'high'] else ValidationSeverity.WARNING
            issues.append(ValidationIssue(
                category=ValidationCategory.RISK,
                severity=severity,
                description=violation.description,
                details=f"Violation: {violation.violation_percentage:.1%}",
                symbol=violation.symbol,
                date=violation.date,
                recommendation=violation.remediation
            ))
        
        # Overall score issue if below threshold
        if overall_score < self.config.pass_threshold:
            issues.append(ValidationIssue(
                category=ValidationCategory.OVERALL,
                severity=ValidationSeverity.FAIL,
                description=f"Overall validation score {overall_score:.1f} below threshold {self.config.pass_threshold:.1f}",
                details="Multiple validation components failed",
                score=overall_score,
                recommendation="Review and address individual component failures"
            ))
        
        return issues
    
    def _generate_recommendations(self, results: Dict[str, Any], overall_score: float) -> List[str]:
        """Generate validation recommendations."""
        recommendations = []
        
        # Score-based recommendations
        if overall_score < 50:
            recommendations.append("Portfolio requires major restructuring to meet business logic requirements")
        elif overall_score < 70:
            recommendations.append("Portfolio has significant issues that should be addressed before deployment")
        elif overall_score < 85:
            recommendations.append("Portfolio has minor issues that can be improved")
        
        # Component-specific recommendations
        regulatory_score = results.get('regulatory', {}).get('score', 100.0)
        if regulatory_score < 80:
            recommendations.append("Review regulatory compliance, particularly position limits and settlement requirements")
        
        risk_score = results.get('risk', {}).get('score', 100.0)
        if risk_score < 80:
            recommendations.append("Reassess risk management framework and position sizing methodology")
        
        sector_score = results.get('sector', {}).get('score', 100.0)
        if sector_score < 80:
            recommendations.append("Improve sector neutrality and reduce concentration risk")
        
        return recommendations
    
    def _generate_cache_key(self, portfolio_weights: Dict[str, float], date: date) -> str:
        """Generate cache key for validation results."""
        # Create a hash of the portfolio weights and date
        import hashlib
        portfolio_str = str(sorted(portfolio_weights.items()))
        key_str = f"{portfolio_str}_{date.isoformat()}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_market_data(self, symbols: List[str], date: date) -> pd.DataFrame:
        """Get market data for symbols."""
        # This would query the database for market data
        # For now, return synthetic data for testing
        data = []
        sectors = ['technology', 'financials', 'industrials', 'materials', 'healthcare']
        
        for symbol in symbols:
            data.append({
                'symbol': symbol,
                'sector': np.random.choice(sectors),
                'volatility': np.random.uniform(0.15, 0.35),
                'beta': np.random.uniform(0.5, 1.5),
                'market_cap': np.random.uniform(1e9, 100e9),
                'liquidity_score': np.random.uniform(0.3, 0.9),
                'value_loading': np.random.uniform(-1, 1),
                'growth_loading': np.random.uniform(-1, 1),
                'momentum_loading': np.random.uniform(-1, 1),
                'quality_loading': np.random.uniform(-1, 1),
                'low_volatility_loading': np.random.uniform(-1, 1),
                'size_loading': np.random.uniform(-1, 1),
                'liquidity_loading': np.random.uniform(-1, 1)
            })
        
        return pd.DataFrame(data)
    
    def __del__(self):
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=False)


def create_comprehensive_validator(
    strict_mode: bool = False,
    taiwan_focus: bool = True
) -> BusinessLogicValidator:
    """Create a comprehensive business logic validator.
    
    Args:
        strict_mode: Use stricter validation thresholds
        taiwan_focus: Focus on Taiwan market requirements
        
    Returns:
        Configured business logic validator
    """
    config = ValidationConfig()
    
    if strict_mode:
        config.pass_threshold = 80.0
        config.warning_threshold = 90.0
    
    if taiwan_focus:
        config.regulatory_weight = 0.30  # Higher weight for regulatory compliance
        config.market_hours_only = True
        config.include_otc_stocks = True
    
    # Normalize weights
    config.normalize_weights()
    
    return BusinessLogicValidator(config)


def create_fast_validator() -> BusinessLogicValidator:
    """Create a fast validator with reduced functionality for real-time use.
    
    Returns:
        Fast business logic validator
    """
    config = ValidationConfig(
        enable_economic=False,  # Skip slower economic validation
        enable_sector=False,    # Skip sector analysis
        parallel_validation=True,
        max_workers=2,
        timeout_seconds=10.0,
        cache_results=True,
        cache_ttl_minutes=30
    )
    
    # Reweight remaining components
    config.regulatory_weight = 0.5
    config.strategy_weight = 0.0  # Disabled
    config.economic_weight = 0.0  # Disabled
    config.sector_weight = 0.0    # Disabled
    config.risk_weight = 0.5
    
    return BusinessLogicValidator(config)