"""
Backtesting Framework Integration for Business Logic Validation.

This module integrates the business logic validator with the walk-forward
validation engine and transaction cost modeling from Tasks #23-24.

Key Features:
- Integration with walk-forward validation framework
- Business logic validation during backtesting
- Transaction cost validation integration
- Performance attribution with business rules
- Taiwan market compliance during backtests

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
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .business_validator import BusinessLogicValidator, ValidationResult, ValidationSeverity
from .regulatory_validator import RegulatoryValidator
from .risk_integration import RiskValidator

logger = logging.getLogger(__name__)


class BacktestValidationPhase(Enum):
    """Phases of backtesting where validation occurs."""
    PRE_TRADE = "pre_trade"          # Before portfolio construction
    POSITION_SIZING = "position_sizing"  # During position sizing
    POST_CONSTRUCTION = "post_construction"  # After portfolio built
    POST_EXECUTION = "post_execution"    # After trades executed
    PERIODIC_REVIEW = "periodic_review"  # Regular validation checks


class ValidationTiming(Enum):
    """Timing for validation execution."""
    SYNCHRONOUS = "synchronous"      # Block execution until validation complete
    ASYNCHRONOUS = "asynchronous"    # Non-blocking validation
    BATCH = "batch"                  # Batch validation at end of period


@dataclass
class BacktestValidationConfig:
    """Configuration for backtesting integration."""
    
    # Validation phases to run
    enable_pre_trade: bool = True
    enable_position_sizing: bool = True
    enable_post_construction: bool = True
    enable_post_execution: bool = False  # Optional, can slow backtests
    enable_periodic_review: bool = True
    
    # Validation timing
    validation_timing: ValidationTiming = ValidationTiming.ASYNCHRONOUS
    max_validation_time_ms: float = 5000.0  # 5 second timeout
    
    # Performance settings
    parallel_validation: bool = True
    cache_validation_results: bool = True
    cache_ttl_minutes: int = 30
    
    # Backtesting integration
    fail_on_critical_violations: bool = True
    warn_on_high_violations: bool = True
    skip_validation_on_timeout: bool = True
    
    # Validation frequency
    validate_every_rebalance: bool = True
    validate_every_n_days: Optional[int] = None  # Additional periodic validation
    
    # Taiwan market specific
    respect_market_hours: bool = True
    validate_settlement_constraints: bool = True
    
    # Reporting
    save_validation_history: bool = True
    validation_report_frequency: str = "monthly"  # daily, weekly, monthly
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'enable_pre_trade': self.enable_pre_trade,
            'enable_position_sizing': self.enable_position_sizing,
            'enable_post_construction': self.enable_post_construction,
            'enable_post_execution': self.enable_post_execution,
            'enable_periodic_review': self.enable_periodic_review,
            'validation_timing': self.validation_timing.value,
            'max_validation_time_ms': self.max_validation_time_ms,
            'parallel_validation': self.parallel_validation,
            'fail_on_critical_violations': self.fail_on_critical_violations,
            'warn_on_high_violations': self.warn_on_high_violations
        }


@dataclass
class BacktestValidationResult:
    """Result of validation during backtesting."""
    phase: BacktestValidationPhase
    date: date
    validation_result: ValidationResult
    execution_time_ms: float
    blocked_trades: List[str] = field(default_factory=list)  # Trades blocked due to validation
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def is_critical_failure(self) -> bool:
        """Check if validation result represents critical failure."""
        return self.validation_result.overall_status == ValidationSeverity.CRITICAL
    
    def is_failure(self) -> bool:
        """Check if validation failed."""
        return self.validation_result.overall_status in [ValidationSeverity.FAIL, ValidationSeverity.CRITICAL]
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return (self.validation_result.overall_status == ValidationSeverity.WARNING or 
                len(self.warnings) > 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'phase': self.phase.value,
            'date': self.date.isoformat(),
            'overall_status': self.validation_result.overall_status.value,
            'overall_score': self.validation_result.overall_score,
            'execution_time_ms': self.execution_time_ms,
            'blocked_trades': self.blocked_trades,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'detailed_result': self.validation_result.to_dict()
        }


class BacktestingValidator:
    """Integrates business logic validation with backtesting framework."""
    
    def __init__(
        self,
        business_validator: Optional[BusinessLogicValidator] = None,
        config: Optional[BacktestValidationConfig] = None
    ):
        """Initialize the backtesting validator.
        
        Args:
            business_validator: Business logic validator instance
            config: Backtesting validation configuration
        """
        self.business_validator = business_validator or BusinessLogicValidator()
        self.config = config or BacktestValidationConfig()
        
        # Validation history
        self.validation_history: List[BacktestValidationResult] = []
        
        # Performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'blocked_trades': 0,
            'avg_execution_time_ms': 0.0,
            'timeout_count': 0
        }
        
        # Cache for validation results
        self._cache = {} if self.config.cache_validation_results else None
        
        # Executor for async validation
        self._executor = ThreadPoolExecutor(
            max_workers=2
        ) if self.config.parallel_validation else None
    
    async def validate_pre_trade(
        self,
        alpha_signals: Dict[str, float],
        current_portfolio: Dict[str, float],
        date: date,
        market_data: Optional[pd.DataFrame] = None
    ) -> BacktestValidationResult:
        """Validate before trade generation.
        
        Args:
            alpha_signals: Alpha signals for next period
            current_portfolio: Current portfolio weights
            date: Trading date
            market_data: Optional market data
            
        Returns:
            Validation result for pre-trade phase
        """
        if not self.config.enable_pre_trade:
            return self._create_skip_result(BacktestValidationPhase.PRE_TRADE, date)
        
        start_time = datetime.now()
        
        try:
            # Validate alpha signals for economic intuition
            validation_result = await self._run_validation_with_timeout(
                self._validate_alpha_signals,
                alpha_signals, current_portfolio, date, market_data
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = BacktestValidationResult(
                phase=BacktestValidationPhase.PRE_TRADE,
                date=date,
                validation_result=validation_result,
                execution_time_ms=execution_time
            )
            
            self._update_stats(result)
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Pre-trade validation timeout on {date}")
            return self._create_timeout_result(BacktestValidationPhase.PRE_TRADE, date)
        except Exception as e:
            logger.error(f"Pre-trade validation error on {date}: {e}")
            return self._create_error_result(BacktestValidationPhase.PRE_TRADE, date, str(e))
    
    async def validate_position_sizing(
        self,
        proposed_weights: Dict[str, float],
        alpha_signals: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: Optional[pd.DataFrame] = None
    ) -> BacktestValidationResult:
        """Validate during position sizing phase.
        
        Args:
            proposed_weights: Proposed position weights
            alpha_signals: Alpha signals used for sizing
            portfolio_value: Total portfolio value
            date: Trading date
            market_data: Optional market data
            
        Returns:
            Validation result for position sizing
        """
        if not self.config.enable_position_sizing:
            return self._create_skip_result(BacktestValidationPhase.POSITION_SIZING, date)
        
        start_time = datetime.now()
        
        try:
            # Focus on risk management and regulatory compliance
            validation_result = await self._run_validation_with_timeout(
                self._validate_position_sizing,
                proposed_weights, alpha_signals, portfolio_value, date, market_data
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check for critical violations that should block trades
            blocked_trades = []
            if validation_result.overall_status == ValidationSeverity.CRITICAL:
                # Block trades for positions that violate critical constraints
                critical_issues = [
                    issue for issue in validation_result.issues
                    if issue.severity == ValidationSeverity.CRITICAL
                ]
                blocked_trades = [
                    issue.symbol for issue in critical_issues 
                    if issue.symbol is not None
                ]
            
            result = BacktestValidationResult(
                phase=BacktestValidationPhase.POSITION_SIZING,
                date=date,
                validation_result=validation_result,
                execution_time_ms=execution_time,
                blocked_trades=blocked_trades
            )
            
            self._update_stats(result)
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Position sizing validation timeout on {date}")
            return self._create_timeout_result(BacktestValidationPhase.POSITION_SIZING, date)
        except Exception as e:
            logger.error(f"Position sizing validation error on {date}: {e}")
            return self._create_error_result(BacktestValidationPhase.POSITION_SIZING, date, str(e))
    
    async def validate_post_construction(
        self,
        final_portfolio: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: Optional[pd.DataFrame] = None,
        strategy_metadata: Optional[Dict[str, Any]] = None
    ) -> BacktestValidationResult:
        """Validate after portfolio construction.
        
        Args:
            final_portfolio: Final portfolio weights
            portfolio_value: Total portfolio value
            date: Trading date
            market_data: Optional market data
            strategy_metadata: Optional strategy metadata
            
        Returns:
            Validation result for post-construction
        """
        if not self.config.enable_post_construction:
            return self._create_skip_result(BacktestValidationPhase.POST_CONSTRUCTION, date)
        
        start_time = datetime.now()
        
        try:
            # Comprehensive validation of final portfolio
            validation_result = await self._run_validation_with_timeout(
                self._validate_comprehensive,
                final_portfolio, portfolio_value, date, market_data, strategy_metadata
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = BacktestValidationResult(
                phase=BacktestValidationPhase.POST_CONSTRUCTION,
                date=date,
                validation_result=validation_result,
                execution_time_ms=execution_time,
                recommendations=validation_result.recommendations
            )
            
            self._update_stats(result)
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Post-construction validation timeout on {date}")
            return self._create_timeout_result(BacktestValidationPhase.POST_CONSTRUCTION, date)
        except Exception as e:
            logger.error(f"Post-construction validation error on {date}: {e}")
            return self._create_error_result(BacktestValidationPhase.POST_CONSTRUCTION, date, str(e))
    
    async def validate_post_execution(
        self,
        executed_portfolio: Dict[str, float],
        execution_costs: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: Optional[pd.DataFrame] = None
    ) -> BacktestValidationResult:
        """Validate after trade execution.
        
        Args:
            executed_portfolio: Actual executed portfolio
            execution_costs: Transaction costs by symbol
            portfolio_value: Portfolio value after execution
            date: Trading date
            market_data: Optional market data
            
        Returns:
            Validation result for post-execution
        """
        if not self.config.enable_post_execution:
            return self._create_skip_result(BacktestValidationPhase.POST_EXECUTION, date)
        
        start_time = datetime.now()
        
        try:
            # Validate actual execution against constraints
            validation_result = await self._run_validation_with_timeout(
                self._validate_post_execution,
                executed_portfolio, execution_costs, portfolio_value, date, market_data
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = BacktestValidationResult(
                phase=BacktestValidationPhase.POST_EXECUTION,
                date=date,
                validation_result=validation_result,
                execution_time_ms=execution_time
            )
            
            self._update_stats(result)
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Post-execution validation timeout on {date}")
            return self._create_timeout_result(BacktestValidationPhase.POST_EXECUTION, date)
        except Exception as e:
            logger.error(f"Post-execution validation error on {date}: {e}")
            return self._create_error_result(BacktestValidationPhase.POST_EXECUTION, date, str(e))
    
    def should_block_trades(self, validation_result: BacktestValidationResult) -> bool:
        """Determine if trades should be blocked based on validation result.
        
        Args:
            validation_result: Validation result to check
            
        Returns:
            True if trades should be blocked
        """
        if not self.config.fail_on_critical_violations:
            return False
        
        return validation_result.is_critical_failure()
    
    def get_validation_summary(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Get validation summary for date range.
        
        Args:
            start_date: Start date for summary
            end_date: End date for summary
            
        Returns:
            Validation summary statistics
        """
        # Filter validation history by date range
        period_results = [
            result for result in self.validation_history
            if start_date <= result.date <= end_date
        ]
        
        if not period_results:
            return {
                'period': f"{start_date} to {end_date}",
                'total_validations': 0,
                'summary': 'No validations in period'
            }
        
        # Calculate statistics
        total_validations = len(period_results)
        failed_validations = sum(1 for r in period_results if r.is_failure())
        warning_validations = sum(1 for r in period_results if r.has_warnings())
        blocked_trades_total = sum(len(r.blocked_trades) for r in period_results)
        
        avg_score = np.mean([r.validation_result.overall_score for r in period_results])
        avg_execution_time = np.mean([r.execution_time_ms for r in period_results])
        
        # Phase breakdown
        phase_stats = {}
        for phase in BacktestValidationPhase:
            phase_results = [r for r in period_results if r.phase == phase]
            if phase_results:
                phase_stats[phase.value] = {
                    'count': len(phase_results),
                    'avg_score': np.mean([r.validation_result.overall_score for r in phase_results]),
                    'failure_rate': sum(1 for r in phase_results if r.is_failure()) / len(phase_results)
                }
        
        return {
            'period': f"{start_date} to {end_date}",
            'total_validations': total_validations,
            'failed_validations': failed_validations,
            'warning_validations': warning_validations,
            'blocked_trades_total': blocked_trades_total,
            'success_rate': 1 - (failed_validations / total_validations),
            'avg_validation_score': avg_score,
            'avg_execution_time_ms': avg_execution_time,
            'phase_breakdown': phase_stats,
            'overall_stats': self.validation_stats.copy()
        }
    
    async def _run_validation_with_timeout(
        self,
        validation_func: Callable,
        *args,
        **kwargs
    ) -> ValidationResult:
        """Run validation with timeout protection."""
        timeout_seconds = self.config.max_validation_time_ms / 1000.0
        
        if self.config.validation_timing == ValidationTiming.SYNCHRONOUS:
            # Synchronous validation with timeout
            try:
                return await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self._executor, validation_func, *args, **kwargs
                    ),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise
        else:
            # Asynchronous validation
            return await asyncio.get_event_loop().run_in_executor(
                self._executor, validation_func, *args, **kwargs
            )
    
    def _validate_alpha_signals(
        self,
        alpha_signals: Dict[str, float],
        current_portfolio: Dict[str, float],
        date: date,
        market_data: Optional[pd.DataFrame]
    ) -> ValidationResult:
        """Validate alpha signals for economic intuition."""
        # Create a dummy portfolio based on alpha signals for validation
        # In practice, this would be more sophisticated
        alpha_portfolio = {
            symbol: max(0, alpha * 0.1)  # Simple scaling
            for symbol, alpha in alpha_signals.items()
            if alpha > 0
        }
        
        # Normalize weights
        total_weight = sum(alpha_portfolio.values())
        if total_weight > 0:
            alpha_portfolio = {
                symbol: weight / total_weight
                for symbol, weight in alpha_portfolio.items()
            }
        
        return self.business_validator.validate_portfolio(
            alpha_portfolio,
            portfolio_value=1000000000,  # 1B NTD default
            date=date,
            alpha_signals=alpha_signals,
            market_data=market_data
        )
    
    def _validate_position_sizing(
        self,
        proposed_weights: Dict[str, float],
        alpha_signals: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: Optional[pd.DataFrame]
    ) -> ValidationResult:
        """Validate proposed position sizes."""
        return self.business_validator.validate_portfolio(
            proposed_weights,
            portfolio_value=portfolio_value,
            date=date,
            alpha_signals=alpha_signals,
            market_data=market_data
        )
    
    def _validate_comprehensive(
        self,
        final_portfolio: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: Optional[pd.DataFrame],
        strategy_metadata: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """Comprehensive validation of final portfolio."""
        return self.business_validator.validate_portfolio(
            final_portfolio,
            portfolio_value=portfolio_value,
            date=date,
            market_data=market_data,
            strategy_metadata=strategy_metadata
        )
    
    def _validate_post_execution(
        self,
        executed_portfolio: Dict[str, float],
        execution_costs: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: Optional[pd.DataFrame]
    ) -> ValidationResult:
        """Validate post-execution portfolio."""
        # Add transaction cost validation to strategy metadata
        strategy_metadata = {
            'execution_costs': execution_costs,
            'total_transaction_cost': sum(execution_costs.values()),
            'post_execution_validation': True
        }
        
        return self.business_validator.validate_portfolio(
            executed_portfolio,
            portfolio_value=portfolio_value,
            date=date,
            market_data=market_data,
            strategy_metadata=strategy_metadata
        )
    
    def _create_skip_result(
        self,
        phase: BacktestValidationPhase,
        date: date
    ) -> BacktestValidationResult:
        """Create result for skipped validation phase."""
        # Create a minimal passing validation result
        from .business_validator import ValidationResult
        
        validation_result = ValidationResult(
            overall_status=ValidationSeverity.PASS,
            overall_score=100.0,
            validation_date=date,
            regulatory_score=100.0,
            strategy_score=100.0,
            economic_score=100.0,
            sector_score=100.0,
            risk_score=100.0,
            compliance_issues=[],
            coherence_results=[],
            intuition_scores=[],
            neutrality_result=None,
            risk_checks=[],
            risk_violations=[],
            issues=[],
            recommendations=[],
            total_positions=0,
            passed_positions=0,
            warning_positions=0,
            failed_positions=0
        )
        
        return BacktestValidationResult(
            phase=phase,
            date=date,
            validation_result=validation_result,
            execution_time_ms=0.0,
            warnings=["Validation phase disabled"]
        )
    
    def _create_timeout_result(
        self,
        phase: BacktestValidationPhase,
        date: date
    ) -> BacktestValidationResult:
        """Create result for timed out validation."""
        from .business_validator import ValidationResult, ValidationIssue, ValidationCategory
        
        timeout_issue = ValidationIssue(
            category=ValidationCategory.OVERALL,
            severity=ValidationSeverity.WARNING,
            description=f"Validation timeout in {phase.value} phase",
            details=f"Validation exceeded {self.config.max_validation_time_ms}ms timeout"
        )
        
        validation_result = ValidationResult(
            overall_status=ValidationSeverity.WARNING,
            overall_score=50.0,  # Neutral score for timeout
            validation_date=date,
            regulatory_score=50.0,
            strategy_score=50.0,
            economic_score=50.0,
            sector_score=50.0,
            risk_score=50.0,
            compliance_issues=[],
            coherence_results=[],
            intuition_scores=[],
            neutrality_result=None,
            risk_checks=[],
            risk_violations=[],
            issues=[timeout_issue],
            recommendations=["Increase validation timeout or optimize validation performance"],
            total_positions=0,
            passed_positions=0,
            warning_positions=0,
            failed_positions=0
        )
        
        self.validation_stats['timeout_count'] += 1
        
        return BacktestValidationResult(
            phase=phase,
            date=date,
            validation_result=validation_result,
            execution_time_ms=self.config.max_validation_time_ms,
            warnings=[f"Validation timeout in {phase.value}"]
        )
    
    def _create_error_result(
        self,
        phase: BacktestValidationPhase,
        date: date,
        error_msg: str
    ) -> BacktestValidationResult:
        """Create result for validation error."""
        from .business_validator import ValidationResult, ValidationIssue, ValidationCategory
        
        error_issue = ValidationIssue(
            category=ValidationCategory.OVERALL,
            severity=ValidationSeverity.FAIL,
            description=f"Validation error in {phase.value} phase",
            details=error_msg
        )
        
        validation_result = ValidationResult(
            overall_status=ValidationSeverity.FAIL,
            overall_score=0.0,
            validation_date=date,
            regulatory_score=0.0,
            strategy_score=0.0,
            economic_score=0.0,
            sector_score=0.0,
            risk_score=0.0,
            compliance_issues=[],
            coherence_results=[],
            intuition_scores=[],
            neutrality_result=None,
            risk_checks=[],
            risk_violations=[],
            issues=[error_issue],
            recommendations=["Review validation error and fix underlying issue"],
            total_positions=0,
            passed_positions=0,
            warning_positions=0,
            failed_positions=0
        )
        
        return BacktestValidationResult(
            phase=phase,
            date=date,
            validation_result=validation_result,
            execution_time_ms=0.0,
            warnings=[f"Validation error: {error_msg}"]
        )
    
    def _update_stats(self, result: BacktestValidationResult) -> None:
        """Update validation statistics."""
        self.validation_stats['total_validations'] += 1
        
        if result.is_failure():
            self.validation_stats['failed_validations'] += 1
        
        if result.blocked_trades:
            self.validation_stats['blocked_trades'] += len(result.blocked_trades)
        
        # Update average execution time
        current_avg = self.validation_stats['avg_execution_time_ms']
        total_validations = self.validation_stats['total_validations']
        
        new_avg = ((current_avg * (total_validations - 1)) + result.execution_time_ms) / total_validations
        self.validation_stats['avg_execution_time_ms'] = new_avg
        
        # Store in history if configured
        if self.config.save_validation_history:
            self.validation_history.append(result)
            
            # Limit history size to prevent memory issues
            max_history = 10000
            if len(self.validation_history) > max_history:
                self.validation_history = self.validation_history[-max_history:]
    
    def __del__(self):
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=False)


def create_backtesting_validator(
    strict_mode: bool = False,
    fast_mode: bool = False
) -> BacktestingValidator:
    """Create a backtesting validator with appropriate configuration.
    
    Args:
        strict_mode: Use strict validation settings
        fast_mode: Optimize for speed over completeness
        
    Returns:
        Configured backtesting validator
    """
    # Configure business validator
    if fast_mode:
        from .business_validator import create_fast_validator
        business_validator = create_fast_validator()
    else:
        from .business_validator import create_comprehensive_validator
        business_validator = create_comprehensive_validator(strict_mode=strict_mode)
    
    # Configure backtesting integration
    config = BacktestValidationConfig()
    
    if fast_mode:
        config.enable_post_execution = False
        config.validation_timing = ValidationTiming.ASYNCHRONOUS
        config.max_validation_time_ms = 2000.0  # Shorter timeout
        config.cache_validation_results = True
        config.cache_ttl_minutes = 60
    
    if strict_mode:
        config.fail_on_critical_violations = True
        config.warn_on_high_violations = True
        config.enable_post_execution = True
        config.validate_settlement_constraints = True
    
    return BacktestingValidator(business_validator, config)


def create_production_validator() -> BacktestingValidator:
    """Create a validator optimized for production backtesting.
    
    Returns:
        Production-optimized backtesting validator
    """
    config = BacktestValidationConfig(
        # Enable core validation phases
        enable_pre_trade=True,
        enable_position_sizing=True,
        enable_post_construction=True,
        enable_post_execution=False,  # Skip for performance
        enable_periodic_review=True,
        
        # Optimize for performance
        validation_timing=ValidationTiming.ASYNCHRONOUS,
        max_validation_time_ms=3000.0,
        parallel_validation=True,
        cache_validation_results=True,
        cache_ttl_minutes=30,
        
        # Production safety
        fail_on_critical_violations=True,
        warn_on_high_violations=True,
        skip_validation_on_timeout=True,
        
        # Taiwan market compliance
        respect_market_hours=True,
        validate_settlement_constraints=True,
        
        # Reporting
        save_validation_history=True,
        validation_report_frequency="weekly"
    )
    
    business_validator = BusinessLogicValidator()
    
    return BacktestingValidator(business_validator, config)