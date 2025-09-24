"""
Comprehensive test suite for BusinessLogicValidator.

Tests the main orchestrator for business logic validation including
integration between components and overall validation workflow.

Author: ML4T Team
Date: 2025-09-24
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.validation.business_logic.business_validator import (
    BusinessLogicValidator,
    ValidationConfig,
    ValidationResult,
    ValidationSeverity,
    ValidationCategory,
    ValidationIssue,
    create_comprehensive_validator,
    create_fast_validator
)


class TestValidationConfig:
    """Test configuration class for business logic validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        assert config.enable_regulatory is True
        assert config.enable_strategy is True
        assert config.enable_economic is True
        assert config.enable_sector is True
        assert config.enable_risk is True
        
        assert config.pass_threshold == 70.0
        assert config.warning_threshold == 85.0
        assert config.parallel_validation is True
        
    def test_weight_validation(self):
        """Test weight validation and normalization."""
        config = ValidationConfig(
            regulatory_weight=0.3,
            strategy_weight=0.2,
            economic_weight=0.2,
            sector_weight=0.15,
            risk_weight=0.15
        )
        
        assert config.validate_weights() is True
        
        # Test invalid weights
        config.regulatory_weight = 0.5
        assert config.validate_weights() is False
        
        # Test normalization
        config.normalize_weights()
        assert config.validate_weights() is True
    
    def test_weight_normalization(self):
        """Test automatic weight normalization."""
        config = ValidationConfig(
            regulatory_weight=0.4,
            strategy_weight=0.4,
            economic_weight=0.4,
            sector_weight=0.4,
            risk_weight=0.4
        )
        
        config.normalize_weights()
        
        total_weight = (
            config.regulatory_weight + config.strategy_weight +
            config.economic_weight + config.sector_weight + config.risk_weight
        )
        
        assert abs(total_weight - 1.0) < 0.001


class TestValidationResult:
    """Test validation result class."""
    
    def test_pass_rate_calculation(self):
        """Test pass rate calculation."""
        result = ValidationResult(
            overall_status=ValidationSeverity.PASS,
            overall_score=85.0,
            validation_date=date.today(),
            regulatory_score=90.0,
            strategy_score=80.0,
            economic_score=85.0,
            sector_score=90.0,
            risk_score=85.0,
            compliance_issues=[],
            coherence_results=[],
            intuition_scores=[],
            neutrality_result=None,
            risk_checks=[],
            risk_violations=[],
            issues=[],
            recommendations=[],
            total_positions=10,
            passed_positions=8,
            warning_positions=1,
            failed_positions=1
        )
        
        assert result.get_pass_rate() == 0.8
        
        # Test zero positions
        result.total_positions = 0
        assert result.get_pass_rate() == 0.0
    
    def test_issue_filtering(self):
        """Test issue filtering by severity and category."""
        issues = [
            ValidationIssue(
                ValidationCategory.REGULATORY,
                ValidationSeverity.FAIL,
                "Regulatory issue",
                "Details"
            ),
            ValidationIssue(
                ValidationCategory.RISK,
                ValidationSeverity.WARNING,
                "Risk issue",
                "Details"
            ),
            ValidationIssue(
                ValidationCategory.REGULATORY,
                ValidationSeverity.WARNING,
                "Another regulatory issue",
                "Details"
            )
        ]
        
        result = ValidationResult(
            overall_status=ValidationSeverity.WARNING,
            overall_score=75.0,
            validation_date=date.today(),
            regulatory_score=70.0,
            strategy_score=80.0,
            economic_score=75.0,
            sector_score=80.0,
            risk_score=75.0,
            compliance_issues=[],
            coherence_results=[],
            intuition_scores=[],
            neutrality_result=None,
            risk_checks=[],
            risk_violations=[],
            issues=issues,
            recommendations=[],
            total_positions=5,
            passed_positions=3,
            warning_positions=2,
            failed_positions=0
        )
        
        fail_issues = result.get_issues_by_severity(ValidationSeverity.FAIL)
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        regulatory_issues = result.get_issues_by_category(ValidationCategory.REGULATORY)
        
        assert len(fail_issues) == 1
        assert len(warning_issues) == 2
        assert len(regulatory_issues) == 2
    
    def test_result_serialization(self):
        """Test result serialization to dictionary."""
        result = ValidationResult(
            overall_status=ValidationSeverity.PASS,
            overall_score=85.0,
            validation_date=date(2025, 9, 24),
            regulatory_score=90.0,
            strategy_score=80.0,
            economic_score=85.0,
            sector_score=90.0,
            risk_score=85.0,
            compliance_issues=[],
            coherence_results=[],
            intuition_scores=[],
            neutrality_result=None,
            risk_checks=[],
            risk_violations=[],
            issues=[],
            recommendations=["Improve risk controls"],
            total_positions=5,
            passed_positions=4,
            warning_positions=1,
            failed_positions=0
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['overall_status'] == 'pass'
        assert result_dict['overall_score'] == 85.0
        assert result_dict['validation_date'] == '2025-09-24'
        assert result_dict['scores']['regulatory'] == 90.0
        assert result_dict['metrics']['pass_rate'] == 0.8
        assert result_dict['recommendations'] == ["Improve risk controls"]


class TestBusinessLogicValidator:
    """Test the main business logic validator."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Sample portfolio for testing."""
        return {
            'TSM': 0.15,    # Taiwan Semiconductor
            'TSLA': 0.10,   # Tesla (ADR)
            '2317': 0.08,   # Hon Hai
            '2330': 0.12,   # TSMC
            '2454': 0.05    # MediaTek
        }
    
    @pytest.fixture
    def sample_alpha_signals(self):
        """Sample alpha signals for testing."""
        return {
            'TSM': 0.05,
            'TSLA': -0.02,
            '2317': 0.03,
            '2330': 0.08,
            '2454': 0.01
        }
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock market data for testing."""
        return pd.DataFrame([
            {
                'symbol': 'TSM',
                'sector': 'technology',
                'volatility': 0.25,
                'beta': 1.1,
                'market_cap': 500e9,
                'liquidity_score': 0.9,
                'value_loading': -0.2,
                'growth_loading': 0.8,
                'momentum_loading': 0.3
            },
            {
                'symbol': 'TSLA',
                'sector': 'consumer_discretionary',
                'volatility': 0.45,
                'beta': 1.8,
                'market_cap': 800e9,
                'liquidity_score': 0.85,
                'value_loading': -0.9,
                'growth_loading': 1.2,
                'momentum_loading': -0.1
            },
            {
                'symbol': '2317',
                'sector': 'technology',
                'volatility': 0.30,
                'beta': 1.3,
                'market_cap': 80e9,
                'liquidity_score': 0.7,
                'value_loading': 0.1,
                'growth_loading': 0.4,
                'momentum_loading': 0.2
            },
            {
                'symbol': '2330',
                'sector': 'technology',
                'volatility': 0.22,
                'beta': 1.0,
                'market_cap': 600e9,
                'liquidity_score': 0.95,
                'value_loading': 0.0,
                'growth_loading': 0.6,
                'momentum_loading': 0.4
            },
            {
                'symbol': '2454',
                'sector': 'technology',
                'volatility': 0.28,
                'beta': 1.2,
                'market_cap': 120e9,
                'liquidity_score': 0.8,
                'value_loading': 0.2,
                'growth_loading': 0.5,
                'momentum_loading': 0.1
            }
        ])
    
    def test_validator_initialization(self):
        """Test validator initialization with different configurations."""
        # Default configuration
        validator = BusinessLogicValidator()
        assert validator.config.enable_regulatory is True
        assert validator.regulatory_validator is not None
        assert validator.risk_validator is not None
        
        # Custom configuration
        config = ValidationConfig(
            enable_economic=False,
            enable_sector=False,
            parallel_validation=False
        )
        validator = BusinessLogicValidator(config)
        assert validator.economic_scorer is None
        assert validator.sector_analyzer is None
        assert validator._executor is None
    
    @patch('src.validation.business_logic.business_validator.RegulatoryValidator')
    @patch('src.validation.business_logic.business_validator.RiskValidator')
    def test_sequential_validation(self, mock_risk_validator, mock_regulatory_validator, 
                                 sample_portfolio, mock_market_data):
        """Test sequential validation workflow."""
        # Setup mocks
        mock_reg_instance = Mock()
        mock_reg_instance.validate_portfolio.return_value = []
        mock_regulatory_validator.return_value = mock_reg_instance
        
        mock_risk_instance = Mock()
        mock_risk_instance.validate_portfolio.return_value = ([], [])
        mock_risk_validator.return_value = mock_risk_instance
        
        # Create validator with sequential processing
        config = ValidationConfig(
            parallel_validation=False,
            enable_strategy=False,
            enable_economic=False,
            enable_sector=False
        )
        validator = BusinessLogicValidator(config)
        
        # Run validation
        result = validator.validate_portfolio(
            sample_portfolio,
            portfolio_value=1000000,
            date=date.today(),
            market_data=mock_market_data
        )
        
        # Verify results
        assert isinstance(result, ValidationResult)
        assert result.overall_score >= 0
        assert result.overall_score <= 100
        assert result.validation_date == date.today()
        
        # Verify mocks were called
        mock_reg_instance.validate_portfolio.assert_called_once()
        mock_risk_instance.validate_portfolio.assert_called_once()
    
    @patch('src.validation.business_logic.business_validator.RegulatoryValidator')
    @patch('src.validation.business_logic.business_validator.RiskValidator')
    def test_parallel_validation(self, mock_risk_validator, mock_regulatory_validator,
                               sample_portfolio, mock_market_data):
        """Test parallel validation workflow."""
        # Setup mocks
        mock_reg_instance = Mock()
        mock_reg_instance.validate_portfolio.return_value = []
        mock_regulatory_validator.return_value = mock_reg_instance
        
        mock_risk_instance = Mock()
        mock_risk_instance.validate_portfolio.return_value = ([], [])
        mock_risk_validator.return_value = mock_risk_instance
        
        # Create validator with parallel processing
        config = ValidationConfig(
            parallel_validation=True,
            max_workers=2,
            timeout_seconds=30.0,
            enable_strategy=False,
            enable_economic=False,
            enable_sector=False
        )
        validator = BusinessLogicValidator(config)
        
        # Run validation
        result = validator.validate_portfolio(
            sample_portfolio,
            portfolio_value=1000000,
            date=date.today(),
            market_data=mock_market_data
        )
        
        # Verify results
        assert isinstance(result, ValidationResult)
        assert result.overall_score >= 0
        assert result.overall_score <= 100
        
        # Verify mocks were called
        mock_reg_instance.validate_portfolio.assert_called_once()
        mock_risk_instance.validate_portfolio.assert_called_once()
    
    def test_single_position_validation(self, mock_market_data):
        """Test validation of a single position."""
        config = ValidationConfig(
            enable_strategy=False,
            enable_economic=False,
            enable_sector=False,
            parallel_validation=False
        )
        validator = BusinessLogicValidator(config)
        
        # Mock the regulatory and risk validators
        with patch.object(validator, 'regulatory_validator') as mock_reg:
            with patch.object(validator, 'risk_validator') as mock_risk:
                mock_reg.validate_portfolio.return_value = []
                mock_risk.validate_portfolio.return_value = ([], [])
                
                result = validator.validate_single_position(
                    symbol='TSM',
                    weight=0.15,
                    date=date.today(),
                    alpha_signal=0.05,
                    market_data=mock_market_data
                )
                
                assert isinstance(result, dict)
                assert 'symbol' in result
                assert result['symbol'] == 'TSM'
                assert 'overall_score' in result
                assert 'issues' in result
                assert 'recommendations' in result
    
    def test_caching_functionality(self, sample_portfolio, mock_market_data):
        """Test result caching functionality."""
        config = ValidationConfig(
            cache_results=True,
            cache_ttl_minutes=60,
            enable_strategy=False,
            enable_economic=False,
            enable_sector=False,
            parallel_validation=False
        )
        validator = BusinessLogicValidator(config)
        
        # Mock validators
        with patch.object(validator, 'regulatory_validator') as mock_reg:
            with patch.object(validator, 'risk_validator') as mock_risk:
                mock_reg.validate_portfolio.return_value = []
                mock_risk.validate_portfolio.return_value = ([], [])
                
                # First call
                result1 = validator.validate_portfolio(
                    sample_portfolio,
                    portfolio_value=1000000,
                    date=date.today(),
                    market_data=mock_market_data
                )
                
                # Second call should use cache
                result2 = validator.validate_portfolio(
                    sample_portfolio,
                    portfolio_value=1000000,
                    date=date.today(),
                    market_data=mock_market_data
                )
                
                # Should have same results (from cache)
                assert result1.overall_score == result2.overall_score
                assert result1.validation_date == result2.validation_date
                
                # Validator should only be called once due to caching
                assert mock_reg.validate_portfolio.call_count == 1
                assert mock_risk.validate_portfolio.call_count == 1
    
    def test_score_calculation_and_thresholds(self, sample_portfolio, mock_market_data):
        """Test score calculation and threshold application."""
        config = ValidationConfig(
            pass_threshold=75.0,
            warning_threshold=85.0,
            enable_strategy=False,
            enable_economic=False,
            enable_sector=False,
            parallel_validation=False
        )
        validator = BusinessLogicValidator(config)
        
        # Mock validators to return specific scores
        with patch.object(validator, '_calculate_regulatory_score', return_value=60.0):
            with patch.object(validator, '_calculate_risk_score', return_value=80.0):
                with patch.object(validator, 'regulatory_validator') as mock_reg:
                    with patch.object(validator, 'risk_validator') as mock_risk:
                        mock_reg.validate_portfolio.return_value = []
                        mock_risk.validate_portfolio.return_value = ([], [])
                        
                        result = validator.validate_portfolio(
                            sample_portfolio,
                            portfolio_value=1000000,
                            date=date.today(),
                            market_data=mock_market_data
                        )
                        
                        # With default weights (0.25 reg, 0.20 risk), score should be:
                        # 0.5 * 60 + 0.5 * 80 = 70 (after renormalization)
                        expected_score = 0.5 * 60.0 + 0.5 * 80.0
                        assert abs(result.overall_score - expected_score) < 1.0
                        
                        # Should be FAIL since score < pass_threshold (75)
                        assert result.overall_status == ValidationSeverity.FAIL
    
    def test_issue_generation(self):
        """Test validation issue generation."""
        validator = BusinessLogicValidator()
        
        # Mock results with various issues
        mock_results = {
            'regulatory': {
                'issues': [Mock(severity=Mock(value='high'), description='High severity issue')],
                'score': 60.0
            },
            'risk': {
                'violations': [Mock(
                    severity=Mock(value='critical'),
                    description='Critical risk violation',
                    violation_percentage=0.5,
                    symbol='TSM',
                    date=date.today(),
                    remediation='Fix this issue'
                )],
                'score': 50.0
            }
        }
        
        issues = validator._generate_issues(mock_results, 55.0)
        
        # Should have regulatory and risk issues, plus overall score issue
        assert len(issues) >= 2
        
        # Check for overall score issue
        overall_issues = [i for i in issues if i.category == ValidationCategory.OVERALL]
        assert len(overall_issues) == 1
        assert overall_issues[0].severity == ValidationSeverity.FAIL
    
    def test_recommendation_generation(self):
        """Test validation recommendation generation."""
        validator = BusinessLogicValidator()
        
        # Test different score scenarios
        recommendations_low = validator._generate_recommendations({}, 40.0)
        assert any("major restructuring" in r.lower() for r in recommendations_low)
        
        recommendations_medium = validator._generate_recommendations({}, 65.0)
        assert any("significant issues" in r.lower() for r in recommendations_medium)
        
        recommendations_high = validator._generate_recommendations({}, 80.0)
        assert any("minor issues" in r.lower() for r in recommendations_high)
        
        # Test component-specific recommendations
        mock_results = {
            'regulatory': {'score': 60.0},
            'risk': {'score': 70.0},
            'sector': {'score': 75.0}
        }
        
        recommendations = validator._generate_recommendations(mock_results, 70.0)
        assert any("regulatory compliance" in r.lower() for r in recommendations)
        assert any("risk management" in r.lower() for r in recommendations)


class TestValidatorFactories:
    """Test validator factory functions."""
    
    def test_comprehensive_validator_creation(self):
        """Test comprehensive validator factory."""
        # Standard comprehensive validator
        validator = create_comprehensive_validator()
        
        assert isinstance(validator, BusinessLogicValidator)
        assert validator.config.enable_regulatory is True
        assert validator.config.enable_risk is True
        
        # Strict mode
        strict_validator = create_comprehensive_validator(strict_mode=True)
        assert strict_validator.config.pass_threshold == 80.0
        assert strict_validator.config.warning_threshold == 90.0
        
        # Taiwan focus
        taiwan_validator = create_comprehensive_validator(taiwan_focus=True)
        assert taiwan_validator.config.regulatory_weight == 0.30
        assert taiwan_validator.config.market_hours_only is True
    
    def test_fast_validator_creation(self):
        """Test fast validator factory."""
        validator = create_fast_validator()
        
        assert isinstance(validator, BusinessLogicValidator)
        assert validator.config.enable_economic is False
        assert validator.config.enable_sector is False
        assert validator.config.parallel_validation is True
        assert validator.config.timeout_seconds == 10.0
        assert validator.config.cache_results is True
        
        # Check weight redistribution
        assert validator.config.regulatory_weight == 0.5
        assert validator.config.risk_weight == 0.5
        assert validator.config.economic_weight == 0.0
        assert validator.config.sector_weight == 0.0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_portfolio_validation(self):
        """Test validation with empty portfolio."""
        validator = BusinessLogicValidator()
        
        result = validator.validate_portfolio(
            portfolio_weights={},
            portfolio_value=1000000,
            date=date.today()
        )
        
        assert isinstance(result, ValidationResult)
        assert result.total_positions == 0
        assert result.passed_positions == 0
    
    def test_validation_with_missing_data(self):
        """Test validation when market data is incomplete."""
        validator = BusinessLogicValidator()
        
        portfolio = {'UNKNOWN_STOCK': 0.10}
        
        # Should handle gracefully without crashing
        result = validator.validate_portfolio(
            portfolio_weights=portfolio,
            portfolio_value=1000000,
            date=date.today(),
            market_data=pd.DataFrame()  # Empty market data
        )
        
        assert isinstance(result, ValidationResult)
        assert result.total_positions == 1
    
    def test_extreme_portfolio_weights(self):
        """Test validation with extreme portfolio weights."""
        validator = BusinessLogicValidator()
        
        # Portfolio with very high concentration
        extreme_portfolio = {'TSM': 0.95, 'TSLA': 0.05}
        
        result = validator.validate_portfolio(
            portfolio_weights=extreme_portfolio,
            portfolio_value=1000000,
            date=date.today()
        )
        
        # Should detect concentration issues
        assert isinstance(result, ValidationResult)
        assert len(result.issues) > 0
    
    def test_negative_portfolio_weights(self):
        """Test validation with negative weights (short positions)."""
        validator = BusinessLogicValidator()
        
        # Portfolio with short positions
        short_portfolio = {'TSM': 0.60, 'TSLA': -0.10}  # Net 50% exposure
        
        result = validator.validate_portfolio(
            portfolio_weights=short_portfolio,
            portfolio_value=1000000,
            date=date.today()
        )
        
        assert isinstance(result, ValidationResult)
        # Should handle negative weights in leverage calculation
    
    def test_timeout_handling(self):
        """Test validation timeout handling."""
        config = ValidationConfig(
            parallel_validation=True,
            timeout_seconds=0.001  # Very short timeout
        )
        validator = BusinessLogicValidator(config)
        
        # Mock slow validation
        with patch.object(validator, '_run_regulatory_validation') as mock_reg:
            import time
            mock_reg.side_effect = lambda *args: time.sleep(1)  # Simulate slow validation
            
            try:
                result = validator.validate_portfolio(
                    portfolio_weights={'TSM': 0.10},
                    portfolio_value=1000000,
                    date=date.today()
                )
                # Should complete with partial results or handle timeout gracefully
                assert isinstance(result, ValidationResult)
            except Exception as e:
                # Timeout exceptions are acceptable
                assert "timeout" in str(e).lower() or "time" in str(e).lower()
    
    def test_validator_cleanup(self):
        """Test validator resource cleanup."""
        validator = BusinessLogicValidator(ValidationConfig(parallel_validation=True))
        
        # Validator should have executor
        assert validator._executor is not None
        
        # Cleanup should work without errors
        del validator
        
        # No assertions needed - just shouldn't crash


if __name__ == "__main__":
    pytest.main([__file__])