"""
Test suite for BacktestingValidator and backtesting integration.

Tests integration between business logic validation and backtesting framework
including validation phases, timing, and performance optimization.

Author: ML4T Team
Date: 2025-09-24
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from datetime import date, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.validation.business_logic.backtesting_integration import (
    BacktestingValidator,
    BacktestValidationConfig,
    BacktestValidationResult,
    BacktestValidationPhase,
    ValidationTiming,
    create_backtesting_validator,
    create_production_validator
)
from src.validation.business_logic.business_validator import (
    BusinessLogicValidator, ValidationResult, ValidationSeverity
)


class TestBacktestValidationConfig:
    """Test backtesting validation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestValidationConfig()
        
        assert config.enable_pre_trade is True
        assert config.enable_position_sizing is True
        assert config.enable_post_construction is True
        assert config.enable_post_execution is False  # Disabled by default
        assert config.enable_periodic_review is True
        
        assert config.validation_timing == ValidationTiming.ASYNCHRONOUS
        assert config.max_validation_time_ms == 5000.0
        assert config.parallel_validation is True
        assert config.fail_on_critical_violations is True
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestValidationConfig(
            enable_post_execution=True,
            validation_timing=ValidationTiming.SYNCHRONOUS,
            max_validation_time_ms=2000.0,
            fail_on_critical_violations=False
        )
        
        assert config.enable_post_execution is True
        assert config.validation_timing == ValidationTiming.SYNCHRONOUS
        assert config.max_validation_time_ms == 2000.0
        assert config.fail_on_critical_violations is False
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = BacktestValidationConfig(
            enable_pre_trade=False,
            max_validation_time_ms=3000.0
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['enable_pre_trade'] is False
        assert config_dict['max_validation_time_ms'] == 3000.0
        assert config_dict['validation_timing'] == 'asynchronous'


class TestBacktestValidationResult:
    """Test backtesting validation result."""
    
    @pytest.fixture
    def mock_validation_result(self):
        """Mock validation result for testing."""
        return Mock(
            overall_status=ValidationSeverity.PASS,
            overall_score=85.0,
            recommendations=["Improve risk controls"]
        )
    
    def test_result_creation(self, mock_validation_result):
        """Test validation result creation."""
        result = BacktestValidationResult(
            phase=BacktestValidationPhase.PRE_TRADE,
            date=date(2025, 9, 24),
            validation_result=mock_validation_result,
            execution_time_ms=150.0,
            blocked_trades=["TSM"],
            warnings=["Low liquidity detected"]
        )
        
        assert result.phase == BacktestValidationPhase.PRE_TRADE
        assert result.date == date(2025, 9, 24)
        assert result.execution_time_ms == 150.0
        assert len(result.blocked_trades) == 1
        assert len(result.warnings) == 1
    
    def test_status_checks(self, mock_validation_result):
        """Test status checking methods."""
        # Test passing result
        mock_validation_result.overall_status = ValidationSeverity.PASS
        result = BacktestValidationResult(
            phase=BacktestValidationPhase.POSITION_SIZING,
            date=date.today(),
            validation_result=mock_validation_result,
            execution_time_ms=100.0
        )
        
        assert result.is_critical_failure() is False
        assert result.is_failure() is False
        assert result.has_warnings() is False
        
        # Test critical failure
        mock_validation_result.overall_status = ValidationSeverity.CRITICAL
        assert result.is_critical_failure() is True
        assert result.is_failure() is True
        
        # Test warning
        mock_validation_result.overall_status = ValidationSeverity.WARNING
        assert result.is_critical_failure() is False
        assert result.is_failure() is False
        assert result.has_warnings() is True
        
        # Test failure
        mock_validation_result.overall_status = ValidationSeverity.FAIL
        assert result.is_critical_failure() is False
        assert result.is_failure() is True
    
    def test_result_serialization(self, mock_validation_result):
        """Test result serialization."""
        mock_validation_result.to_dict.return_value = {'detailed': 'result'}
        
        result = BacktestValidationResult(
            phase=BacktestValidationPhase.POST_CONSTRUCTION,
            date=date(2025, 9, 24),
            validation_result=mock_validation_result,
            execution_time_ms=250.0,
            recommendations=["Reduce concentration"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['phase'] == 'post_construction'
        assert result_dict['date'] == '2025-09-24'
        assert result_dict['overall_score'] == 85.0
        assert result_dict['execution_time_ms'] == 250.0
        assert result_dict['recommendations'] == ["Reduce concentration"]
        assert result_dict['detailed_result'] == {'detailed': 'result'}


class TestBacktestingValidator:
    """Test the main backtesting validator."""
    
    @pytest.fixture
    def mock_business_validator(self):
        """Mock business logic validator."""
        validator = Mock(spec=BusinessLogicValidator)
        validator.validate_portfolio.return_value = Mock(
            overall_status=ValidationSeverity.PASS,
            overall_score=85.0,
            recommendations=["Test recommendation"],
            issues=[]
        )
        return validator
    
    @pytest.fixture
    def sample_portfolio(self):
        """Sample portfolio for testing."""
        return {
            'TSM': 0.15,
            '2330': 0.12,
            '2317': 0.08,
            '2454': 0.05,
            '2882': 0.10
        }
    
    @pytest.fixture
    def sample_alpha_signals(self):
        """Sample alpha signals for testing."""
        return {
            'TSM': 0.08,
            '2330': 0.06,
            '2317': 0.04,
            '2454': 0.02,
            '2882': -0.01
        }
    
    def test_validator_initialization(self, mock_business_validator):
        """Test validator initialization."""
        # Default initialization
        validator = BacktestingValidator()
        assert validator.business_validator is not None
        assert validator.config.enable_pre_trade is True
        assert len(validator.validation_history) == 0
        
        # Custom initialization
        config = BacktestValidationConfig(enable_post_execution=True)
        validator = BacktestingValidator(mock_business_validator, config)
        assert validator.business_validator == mock_business_validator
        assert validator.config.enable_post_execution is True
    
    @pytest.mark.asyncio
    async def test_pre_trade_validation(self, mock_business_validator, sample_alpha_signals, sample_portfolio):
        """Test pre-trade validation phase."""
        validator = BacktestingValidator(mock_business_validator)
        
        result = await validator.validate_pre_trade(
            alpha_signals=sample_alpha_signals,
            current_portfolio=sample_portfolio,
            date=date.today()
        )
        
        assert isinstance(result, BacktestValidationResult)
        assert result.phase == BacktestValidationPhase.PRE_TRADE
        assert result.execution_time_ms > 0
        mock_business_validator.validate_portfolio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_position_sizing_validation(self, mock_business_validator, sample_portfolio, sample_alpha_signals):
        """Test position sizing validation phase."""
        validator = BacktestingValidator(mock_business_validator)
        
        result = await validator.validate_position_sizing(
            proposed_weights=sample_portfolio,
            alpha_signals=sample_alpha_signals,
            portfolio_value=1000000000,
            date=date.today()
        )
        
        assert isinstance(result, BacktestValidationResult)
        assert result.phase == BacktestValidationPhase.POSITION_SIZING
        assert result.execution_time_ms > 0
        mock_business_validator.validate_portfolio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_post_construction_validation(self, mock_business_validator, sample_portfolio):
        """Test post-construction validation phase."""
        validator = BacktestingValidator(mock_business_validator)
        
        result = await validator.validate_post_construction(
            final_portfolio=sample_portfolio,
            portfolio_value=1000000000,
            date=date.today(),
            strategy_metadata={'strategy_type': 'alpha_generation'}
        )
        
        assert isinstance(result, BacktestValidationResult)
        assert result.phase == BacktestValidationPhase.POST_CONSTRUCTION
        assert result.execution_time_ms > 0
        mock_business_validator.validate_portfolio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_post_execution_validation(self, mock_business_validator, sample_portfolio):
        """Test post-execution validation phase."""
        validator = BacktestingValidator(mock_business_validator)
        
        execution_costs = {symbol: 0.001 * weight for symbol, weight in sample_portfolio.items()}
        
        result = await validator.validate_post_execution(
            executed_portfolio=sample_portfolio,
            execution_costs=execution_costs,
            portfolio_value=1000000000,
            date=date.today()
        )
        
        assert isinstance(result, BacktestValidationResult)
        assert result.phase == BacktestValidationPhase.POST_EXECUTION
        assert result.execution_time_ms > 0
        mock_business_validator.validate_portfolio.assert_called_once()
        
        # Check that execution costs are passed in strategy metadata
        call_args = mock_business_validator.validate_portfolio.call_args
        strategy_metadata = call_args[1]['strategy_metadata']
        assert 'execution_costs' in strategy_metadata
        assert 'total_transaction_cost' in strategy_metadata
    
    @pytest.mark.asyncio
    async def test_disabled_phase_validation(self, mock_business_validator):
        """Test validation when phase is disabled."""
        config = BacktestValidationConfig(enable_pre_trade=False)
        validator = BacktestingValidator(mock_business_validator, config)
        
        result = await validator.validate_pre_trade({}, {}, date.today())
        
        assert result.phase == BacktestValidationPhase.PRE_TRADE
        assert result.validation_result.overall_status == ValidationSeverity.PASS
        assert "disabled" in result.warnings[0].lower()
        assert result.execution_time_ms == 0.0
        mock_business_validator.validate_portfolio.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_validation_timeout(self, mock_business_validator):
        """Test validation timeout handling."""
        # Configure short timeout
        config = BacktestValidationConfig(max_validation_time_ms=10.0)  # 10ms timeout
        validator = BacktestingValidator(mock_business_validator, config)
        
        # Mock slow validation
        async def slow_validation(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms delay
            return Mock(overall_status=ValidationSeverity.PASS, overall_score=85.0)
        
        # Replace the validation method with slow version
        validator._validate_alpha_signals = Mock(side_effect=slow_validation)
        
        result = await validator.validate_pre_trade({}, {}, date.today())
        
        assert result.validation_result.overall_status == ValidationSeverity.WARNING
        assert "timeout" in result.warnings[0].lower()
        assert validator.validation_stats['timeout_count'] == 1
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, mock_business_validator):
        """Test validation error handling."""
        validator = BacktestingValidator(mock_business_validator)
        
        # Mock validation that raises exception
        mock_business_validator.validate_portfolio.side_effect = Exception("Test error")
        
        result = await validator.validate_pre_trade({}, {}, date.today())
        
        assert result.validation_result.overall_status == ValidationSeverity.FAIL
        assert result.validation_result.overall_score == 0.0
        assert "error" in result.warnings[0].lower()
    
    def test_should_block_trades(self, mock_business_validator):
        """Test trade blocking logic."""
        validator = BacktestingValidator(mock_business_validator)
        
        # Test critical failure result
        critical_result = BacktestValidationResult(
            phase=BacktestValidationPhase.POSITION_SIZING,
            date=date.today(),
            validation_result=Mock(overall_status=ValidationSeverity.CRITICAL),
            execution_time_ms=100.0
        )
        
        assert validator.should_block_trades(critical_result) is True
        
        # Test non-critical result
        normal_result = BacktestValidationResult(
            phase=BacktestValidationPhase.POSITION_SIZING,
            date=date.today(),
            validation_result=Mock(overall_status=ValidationSeverity.PASS),
            execution_time_ms=100.0
        )
        
        assert validator.should_block_trades(normal_result) is False
        
        # Test with blocking disabled
        config = BacktestValidationConfig(fail_on_critical_violations=False)
        validator = BacktestingValidator(mock_business_validator, config)
        assert validator.should_block_trades(critical_result) is False
    
    def test_validation_statistics(self, mock_business_validator):
        """Test validation statistics tracking."""
        validator = BacktestingValidator(mock_business_validator)
        
        # Create test results
        passing_result = BacktestValidationResult(
            phase=BacktestValidationPhase.PRE_TRADE,
            date=date.today(),
            validation_result=Mock(overall_status=ValidationSeverity.PASS),
            execution_time_ms=100.0
        )
        
        failing_result = BacktestValidationResult(
            phase=BacktestValidationPhase.POSITION_SIZING,
            date=date.today(),
            validation_result=Mock(overall_status=ValidationSeverity.FAIL),
            execution_time_ms=200.0,
            blocked_trades=['TSM', '2330']
        )
        
        # Update stats
        validator._update_stats(passing_result)
        validator._update_stats(failing_result)
        
        stats = validator.validation_stats
        assert stats['total_validations'] == 2
        assert stats['failed_validations'] == 1
        assert stats['blocked_trades'] == 2
        assert stats['avg_execution_time_ms'] == 150.0  # (100 + 200) / 2
    
    def test_validation_summary(self, mock_business_validator):
        """Test validation summary generation."""
        validator = BacktestingValidator(mock_business_validator)
        
        # Create test validation history
        test_date = date.today()
        
        results = [
            BacktestValidationResult(
                phase=BacktestValidationPhase.PRE_TRADE,
                date=test_date,
                validation_result=Mock(overall_status=ValidationSeverity.PASS, overall_score=85.0),
                execution_time_ms=100.0
            ),
            BacktestValidationResult(
                phase=BacktestValidationPhase.POSITION_SIZING,
                date=test_date,
                validation_result=Mock(overall_status=ValidationSeverity.WARNING, overall_score=75.0),
                execution_time_ms=150.0
            ),
            BacktestValidationResult(
                phase=BacktestValidationPhase.POST_CONSTRUCTION,
                date=test_date,
                validation_result=Mock(overall_status=ValidationSeverity.FAIL, overall_score=45.0),
                execution_time_ms=200.0,
                blocked_trades=['BAD_STOCK']
            )
        ]
        
        validator.validation_history = results
        
        summary = validator.get_validation_summary(test_date, test_date)
        
        assert summary['total_validations'] == 3
        assert summary['failed_validations'] == 1
        assert summary['warning_validations'] == 1
        assert summary['blocked_trades_total'] == 1
        assert summary['success_rate'] == pytest.approx(2/3, rel=1e-3)
        assert summary['avg_validation_score'] == pytest.approx(68.33, rel=1e-2)  # (85+75+45)/3
        assert summary['avg_execution_time_ms'] == pytest.approx(150.0, rel=1e-3)
        
        # Check phase breakdown
        assert 'pre_trade' in summary['phase_breakdown']
        assert summary['phase_breakdown']['pre_trade']['count'] == 1
        assert summary['phase_breakdown']['pre_trade']['failure_rate'] == 0.0


class TestValidatorFactories:
    """Test validator factory functions."""
    
    def test_backtesting_validator_creation(self):
        """Test backtesting validator factory."""
        # Standard validator
        validator = create_backtesting_validator()
        assert isinstance(validator, BacktestingValidator)
        assert validator.config.enable_pre_trade is True
        
        # Strict mode
        strict_validator = create_backtesting_validator(strict_mode=True)
        assert strict_validator.config.fail_on_critical_violations is True
        assert strict_validator.config.enable_post_execution is True
        
        # Fast mode
        fast_validator = create_backtesting_validator(fast_mode=True)
        assert fast_validator.config.enable_post_execution is False
        assert fast_validator.config.max_validation_time_ms == 2000.0
        assert fast_validator.config.cache_validation_results is True
    
    def test_production_validator_creation(self):
        """Test production validator factory."""
        validator = create_production_validator()
        
        assert isinstance(validator, BacktestingValidator)
        assert validator.config.enable_pre_trade is True
        assert validator.config.enable_position_sizing is True
        assert validator.config.enable_post_construction is True
        assert validator.config.enable_post_execution is False  # Disabled for performance
        assert validator.config.validation_timing == ValidationTiming.ASYNCHRONOUS
        assert validator.config.parallel_validation is True
        assert validator.config.fail_on_critical_violations is True
        assert validator.config.respect_market_hours is True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_portfolio_validation(self, mock_business_validator):
        """Test validation with empty portfolios."""
        validator = BacktestingValidator(mock_business_validator)
        
        result = await validator.validate_pre_trade(
            alpha_signals={},
            current_portfolio={},
            date=date.today()
        )
        
        assert isinstance(result, BacktestValidationResult)
        # Should still call validator, even with empty data
        mock_business_validator.validate_portfolio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_validation_timing(self, mock_business_validator):
        """Test asynchronous validation timing."""
        config = BacktestValidationConfig(validation_timing=ValidationTiming.ASYNCHRONOUS)
        validator = BacktestingValidator(mock_business_validator, config)
        
        # Should complete without blocking
        result = await validator.validate_position_sizing(
            proposed_weights={'TSM': 0.1},
            alpha_signals={'TSM': 0.05},
            portfolio_value=1000000,
            date=date.today()
        )
        
        assert isinstance(result, BacktestValidationResult)
        assert result.execution_time_ms > 0
    
    def test_validation_history_management(self, mock_business_validator):
        """Test validation history size management."""
        config = BacktestValidationConfig(save_validation_history=True)
        validator = BacktestingValidator(mock_business_validator, config)
        
        # Create many results to test history limiting
        for i in range(15000):  # More than the 10000 limit
            result = BacktestValidationResult(
                phase=BacktestValidationPhase.PRE_TRADE,
                date=date.today(),
                validation_result=Mock(overall_status=ValidationSeverity.PASS, overall_score=85.0),
                execution_time_ms=100.0
            )
            validator._update_stats(result)
        
        # Should limit history size
        assert len(validator.validation_history) == 10000
    
    @pytest.mark.asyncio
    async def test_synchronous_validation_timeout(self, mock_business_validator):
        """Test synchronous validation with timeout."""
        config = BacktestValidationConfig(
            validation_timing=ValidationTiming.SYNCHRONOUS,
            max_validation_time_ms=50.0  # Short timeout
        )
        validator = BacktestingValidator(mock_business_validator, config)
        
        # Mock slow validation
        def slow_validation(*args, **kwargs):
            import time
            time.sleep(0.1)  # 100ms delay
            return Mock(overall_status=ValidationSeverity.PASS, overall_score=85.0)
        
        mock_business_validator.validate_portfolio = slow_validation
        
        result = await validator.validate_pre_trade({}, {}, date.today())
        
        # Should timeout and return warning result
        assert result.validation_result.overall_status == ValidationSeverity.WARNING
        assert "timeout" in result.warnings[0].lower()


if __name__ == "__main__":
    pytest.main([__file__])