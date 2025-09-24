"""
Test suite for RiskValidator and risk integration components.

Tests risk constraint validation, position sizing, and risk management
integration for Taiwan equity portfolios.

Author: ML4T Team
Date: 2025-09-24
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import Mock, patch

from src.validation.business_logic.risk_integration import (
    RiskValidator,
    RiskConfig,
    RiskConstraint,
    RiskViolation,
    PositionRiskCheck,
    RiskConstraintType,
    RiskSeverity,
    PositionSizeMethod,
    create_standard_risk_validator,
    create_conservative_risk_validator
)


class TestRiskConfig:
    """Test risk configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RiskConfig()
        
        assert config.max_single_position == 0.05
        assert config.max_sector_concentration == 0.20
        assert config.min_position_size == 0.001
        assert config.max_portfolio_volatility == 0.20
        assert config.max_tracking_error == 0.08
        assert config.max_leverage == 1.0
        assert config.max_portfolio_beta == 1.5
        assert config.var_confidence_level == 0.95
        assert config.position_sizing_method == PositionSizeMethod.VOLATILITY_ADJUSTED
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = RiskConfig(
            max_single_position=0.03,
            max_portfolio_volatility=0.15,
            position_sizing_method=PositionSizeMethod.EQUAL_WEIGHT,
            target_portfolio_volatility=0.10
        )
        
        assert config.max_single_position == 0.03
        assert config.max_portfolio_volatility == 0.15
        assert config.position_sizing_method == PositionSizeMethod.EQUAL_WEIGHT
        assert config.target_portfolio_volatility == 0.10
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = RiskConfig(max_single_position=0.04, max_leverage=0.95)
        
        config_dict = config.to_dict()
        
        assert config_dict['max_single_position'] == 0.04
        assert config_dict['max_leverage'] == 0.95
        assert config_dict['position_sizing_method'] == 'volatility_adjusted'


class TestRiskConstraint:
    """Test risk constraint class."""
    
    def test_constraint_creation(self):
        """Test risk constraint creation."""
        constraint = RiskConstraint(
            constraint_type=RiskConstraintType.SINGLE_STOCK_LIMIT,
            limit_value=0.05,
            current_value=0.03,
            warning_threshold=0.8,
            description="Max 5% per stock",
            enforcement_level=RiskSeverity.HIGH
        )
        
        assert constraint.constraint_type == RiskConstraintType.SINGLE_STOCK_LIMIT
        assert constraint.limit_value == 0.05
        assert constraint.current_value == 0.03
        assert constraint.enforcement_level == RiskSeverity.HIGH
        
    def test_violation_detection(self):
        """Test violation and warning detection."""
        constraint = RiskConstraint(
            RiskConstraintType.POSITION_SIZE,
            limit_value=0.10,
            current_value=0.05,
            warning_threshold=0.8
        )
        
        # Not violated, not warning
        assert constraint.is_violated() is False
        assert constraint.is_warning() is False
        assert constraint.violation_percentage() == 0.0
        
        # At warning level
        constraint.current_value = 0.09  # 90% of limit
        assert constraint.is_violated() is False
        assert constraint.is_warning() is True
        
        # Violated
        constraint.current_value = 0.12
        assert constraint.is_violated() is True
        assert constraint.is_warning() is True
        assert constraint.violation_percentage() == pytest.approx(0.2, rel=1e-3)  # 20% over
        
    def test_constraint_with_no_current_value(self):
        """Test constraint behavior with no current value."""
        constraint = RiskConstraint(
            RiskConstraintType.VALUE_AT_RISK,
            limit_value=0.02
        )
        
        assert constraint.is_violated() is False
        assert constraint.is_warning() is False
        assert constraint.violation_percentage() == 0.0


class TestRiskViolation:
    """Test risk violation class."""
    
    def test_violation_creation(self):
        """Test risk violation creation."""
        constraint = RiskConstraint(
            RiskConstraintType.SINGLE_STOCK_LIMIT,
            limit_value=0.05,
            current_value=0.08
        )
        
        violation = RiskViolation(
            constraint=constraint,
            severity=RiskSeverity.HIGH,
            description="Position exceeds limit",
            violation_amount=0.03,
            violation_percentage=0.6,
            symbol="TSM",
            date=date.today(),
            remediation="Reduce position size"
        )
        
        assert violation.severity == RiskSeverity.HIGH
        assert violation.violation_amount == 0.03
        assert violation.symbol == "TSM"
        assert violation.remediation == "Reduce position size"
        
    def test_violation_serialization(self):
        """Test violation serialization."""
        constraint = RiskConstraint(RiskConstraintType.LEVERAGE_LIMIT, 1.0, 1.2)
        violation = RiskViolation(
            constraint=constraint,
            severity=RiskSeverity.CRITICAL,
            description="Leverage exceeded",
            violation_amount=0.2,
            violation_percentage=0.2,
            date=date(2025, 9, 24)
        )
        
        violation_dict = violation.to_dict()
        
        assert violation_dict['constraint_type'] == 'leverage_limit'
        assert violation_dict['severity'] == 'critical'
        assert violation_dict['violation_amount'] == 0.2
        assert violation_dict['date'] == '2025-09-24'
        assert violation_dict['limit_value'] == 1.0
        assert violation_dict['current_value'] == 1.2


class TestPositionRiskCheck:
    """Test position risk check class."""
    
    def test_position_check_creation(self):
        """Test position risk check creation."""
        violations = [
            RiskViolation(
                constraint=RiskConstraint(RiskConstraintType.POSITION_SIZE, 0.05, 0.08),
                severity=RiskSeverity.MEDIUM,
                description="Position too large",
                violation_amount=0.03,
                violation_percentage=0.6
            )
        ]
        
        check = PositionRiskCheck(
            symbol="2330",
            weight=0.08,
            dollar_amount=80000000,  # 80M NTD
            volatility=0.25,
            beta=1.1,
            sector="technology",
            market_cap=500e9,
            liquidity_score=0.9,
            risk_contribution=0.02,
            position_size_score=75.0,
            violations=violations
        )
        
        assert check.symbol == "2330"
        assert check.weight == 0.08
        assert check.volatility == 0.25
        assert check.position_size_score == 75.0
        assert len(check.violations) == 1
        
    def test_position_check_serialization(self):
        """Test position check serialization."""
        check = PositionRiskCheck(
            symbol="TSM",
            weight=0.12,
            dollar_amount=120000000,
            volatility=0.30,
            beta=1.2,
            sector="technology",
            market_cap=600e9,
            liquidity_score=0.85,
            risk_contribution=0.036,
            position_size_score=80.0
        )
        
        check_dict = check.to_dict()
        
        assert check_dict['symbol'] == "TSM"
        assert check_dict['weight'] == 0.12
        assert check_dict['sector'] == "technology"
        assert check_dict['position_size_score'] == 80.0
        assert check_dict['violations'] == []


class TestRiskValidator:
    """Test the main risk validator."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Sample portfolio for testing."""
        return {
            'TSM': 0.15,
            '2330': 0.12,
            '2317': 0.08,
            '2454': 0.05,
            '2882': 0.10,
            '1303': 0.06,
            '2002': 0.04,
            '3008': 0.03,
            '2412': 0.02,
            '6505': 0.35  # Large position that should trigger violations
        }
    
    @pytest.fixture
    def sample_alpha_signals(self):
        """Sample alpha signals for testing."""
        return {
            'TSM': 0.08,
            '2330': 0.06,
            '2317': 0.04,
            '2454': 0.02,
            '2882': -0.01,
            '1303': 0.03,
            '2002': 0.01,
            '3008': 0.02,
            '2412': 0.05,
            '6505': 0.10
        }
    
    @pytest.fixture
    def mock_market_data(self):
        """Mock market data for testing."""
        symbols = ['TSM', '2330', '2317', '2454', '2882', '1303', '2002', '3008', '2412', '6505']
        sectors = ['technology', 'technology', 'technology', 'technology', 
                  'financials', 'materials', 'consumer_staples', 'industrials',
                  'technology', 'technology']
        
        data = []
        for i, symbol in enumerate(symbols):
            data.append({
                'symbol': symbol,
                'volatility': np.random.uniform(0.15, 0.35),
                'beta': np.random.uniform(0.8, 1.5),
                'sector': sectors[i],
                'market_cap': np.random.uniform(50e9, 500e9),
                'liquidity_score': np.random.uniform(0.4, 0.9)
            })
        
        return pd.DataFrame(data)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        # Default configuration
        validator = RiskValidator()
        assert validator.config.max_single_position == 0.05
        assert len(validator.constraints) > 0
        
        # Custom configuration
        config = RiskConfig(max_single_position=0.03)
        validator = RiskValidator(config)
        assert validator.config.max_single_position == 0.03
        
    def test_portfolio_validation(self, sample_portfolio, mock_market_data):
        """Test comprehensive portfolio validation."""
        validator = RiskValidator()
        
        position_checks, violations = validator.validate_portfolio(
            sample_portfolio,
            portfolio_value=1000000000,  # 1B NTD
            date=date.today(),
            market_data=mock_market_data
        )
        
        # Should have position checks for all stocks
        assert len(position_checks) == len(sample_portfolio)
        
        # Should detect violation for 6505 (35% position > 5% limit)
        position_violations = []
        for check in position_checks:
            if check.symbol == '6505':
                position_violations.extend(check.violations)
        
        large_position_violations = [v for v in violations if 'Position weight' in v.description]
        
        assert len(position_violations) > 0 or len(large_position_violations) > 0
        
        # Should detect sector concentration (technology heavy)
        sector_violations = [v for v in violations if 'concentration' in v.description.lower()]
        tech_weight = sum(w for s, w in sample_portfolio.items() 
                         if mock_market_data[mock_market_data['symbol'] == s]['sector'].iloc[0] == 'technology')
        
        if tech_weight > validator.config.max_sector_concentration:
            assert len(sector_violations) > 0
    
    def test_position_validation(self, mock_market_data):
        """Test individual position validation."""
        validator = RiskValidator()
        
        # Test normal position
        check = validator._validate_position(
            'TSM', 0.04, 1000000000, date.today(), mock_market_data
        )
        
        assert check.symbol == 'TSM'
        assert check.weight == 0.04
        assert len(check.violations) == 0  # Should be within limits
        assert check.position_size_score > 0
        
        # Test oversized position
        large_check = validator._validate_position(
            'TSM', 0.15, 1000000000, date.today(), mock_market_data
        )
        
        assert large_check.weight == 0.15
        # Should have violations for exceeding single position limit
        size_violations = [v for v in large_check.violations 
                          if v.constraint.constraint_type == RiskConstraintType.SINGLE_STOCK_LIMIT]
        assert len(size_violations) > 0
    
    def test_equal_weight_position_sizing(self, sample_alpha_signals, mock_market_data):
        """Test equal weight position sizing."""
        config = RiskConfig(position_sizing_method=PositionSizeMethod.EQUAL_WEIGHT)
        validator = RiskValidator(config)
        
        # Filter to positive alpha signals
        positive_signals = {k: v for k, v in sample_alpha_signals.items() if v > 0}
        
        positions = validator._calculate_equal_weight_positions(positive_signals)
        
        # All positions should be equal (subject to max position constraint)
        expected_weight = min(1.0 / len(positive_signals), validator.config.max_single_position)
        
        for symbol, weight in positions.items():
            assert weight == pytest.approx(expected_weight, rel=1e-3)
    
    def test_volatility_adjusted_position_sizing(self, sample_alpha_signals, mock_market_data):
        """Test volatility-adjusted position sizing."""
        validator = RiskValidator()
        
        positions = validator._calculate_volatility_adjusted_positions(
            sample_alpha_signals, mock_market_data, date.today()
        )
        
        # Should only have positive alpha positions
        for symbol, weight in positions.items():
            assert sample_alpha_signals[symbol] > 0
            assert 0 < weight <= validator.config.max_single_position
        
        # Total weight should not exceed 100%
        total_weight = sum(positions.values())
        assert total_weight <= 1.0
    
    def test_kelly_position_sizing(self, sample_alpha_signals, mock_market_data):
        """Test Kelly criterion position sizing."""
        config = RiskConfig(position_sizing_method=PositionSizeMethod.KELLY_CRITERION)
        validator = RiskValidator(config)
        
        positions = validator._calculate_kelly_positions(
            sample_alpha_signals, mock_market_data, date.today()
        )
        
        # Should respect position limits
        for symbol, weight in positions.items():
            assert 0 < weight <= validator.config.max_single_position
            assert weight >= validator.config.min_position_size
    
    def test_optimal_position_size_calculation(self, sample_alpha_signals, mock_market_data):
        """Test optimal position size calculation workflow."""
        validator = RiskValidator()
        
        positions = validator.calculate_optimal_position_sizes(
            sample_alpha_signals,
            portfolio_value=1000000000,
            date=date.today(),
            market_data=mock_market_data
        )
        
        # Should return a valid portfolio
        assert isinstance(positions, dict)
        assert len(positions) > 0
        
        # All positions should be within limits
        for symbol, weight in positions.items():
            assert 0 < weight <= validator.config.max_single_position
            assert weight >= validator.config.min_position_size
        
        # Total weight should be reasonable
        total_weight = sum(positions.values())
        assert 0 < total_weight <= 1.0
    
    def test_portfolio_metrics_calculation(self, sample_portfolio, mock_market_data):
        """Test portfolio-level risk metrics calculation."""
        validator = RiskValidator()
        
        metrics = validator._calculate_portfolio_metrics(
            sample_portfolio, mock_market_data, date.today()
        )
        
        assert 'volatility' in metrics
        assert 'tracking_error' in metrics
        assert 'beta' in metrics
        assert 'variance' in metrics
        
        # Metrics should be reasonable
        assert 0 < metrics['volatility'] < 1.0  # 0-100% volatility
        assert 0 < metrics['tracking_error'] < 1.0
        assert 0 < metrics['beta'] < 5.0  # Reasonable beta range
        assert metrics['variance'] > 0
    
    def test_sector_concentration_calculation(self, sample_portfolio, mock_market_data):
        """Test sector concentration calculation."""
        validator = RiskValidator()
        
        concentrations = validator._calculate_sector_concentrations(
            sample_portfolio, mock_market_data
        )
        
        # Should have concentrations for sectors present in portfolio
        assert len(concentrations) > 0
        
        # Technology should be concentrated given the sample portfolio
        if 'technology' in concentrations:
            tech_concentration = concentrations['technology']
            # Sum of technology weights should be significant
            tech_symbols = mock_market_data[
                mock_market_data['sector'] == 'technology'
            ]['symbol'].tolist()
            expected_tech_weight = sum(
                abs(sample_portfolio.get(symbol, 0)) for symbol in tech_symbols
            )
            assert tech_concentration == pytest.approx(expected_tech_weight, rel=1e-3)
    
    def test_position_size_score_calculation(self, mock_market_data):
        """Test position size score calculation."""
        validator = RiskValidator()
        
        # Good position (no violations, good liquidity)
        stock_info = mock_market_data[mock_market_data['symbol'] == 'TSM'].iloc[0]
        stock_info['liquidity_score'] = 0.9
        stock_info['volatility'] = 0.20
        
        score = validator._calculate_position_size_score('TSM', 0.03, stock_info, [])
        assert score > 80  # Should be high score
        
        # Bad position (violations, poor liquidity)
        violations = [
            RiskViolation(
                constraint=RiskConstraint(RiskConstraintType.POSITION_SIZE, 0.05, 0.10),
                severity=RiskSeverity.HIGH,
                description="Position too large",
                violation_amount=0.05,
                violation_percentage=1.0
            )
        ]
        
        bad_stock_info = stock_info.copy()
        bad_stock_info['liquidity_score'] = 0.2
        bad_stock_info['volatility'] = 0.50
        
        bad_score = validator._calculate_position_size_score(
            'BAD_STOCK', 0.10, bad_stock_info, violations
        )
        assert bad_score < 50  # Should be low score
    
    def test_constraint_creation(self):
        """Test default constraint creation."""
        validator = RiskValidator()
        
        constraints = validator._create_default_constraints()
        
        # Should have key constraints
        constraint_types = [c.constraint_type for c in constraints]
        assert RiskConstraintType.SINGLE_STOCK_LIMIT in constraint_types
        assert RiskConstraintType.SECTOR_CONCENTRATION in constraint_types
        assert RiskConstraintType.LEVERAGE_LIMIT in constraint_types
        assert RiskConstraintType.VALUE_AT_RISK in constraint_types
        
        # All constraints should have proper configuration
        for constraint in constraints:
            assert constraint.limit_value > 0
            assert constraint.description != ""
            assert constraint.reference != ""


class TestValidatorFactories:
    """Test validator factory functions."""
    
    def test_standard_validator_creation(self):
        """Test standard validator factory."""
        validator = create_standard_risk_validator()
        
        assert isinstance(validator, RiskValidator)
        assert validator.config.max_single_position == 0.05
        assert validator.config.position_sizing_method == PositionSizeMethod.VOLATILITY_ADJUSTED
        
        # Test custom parameters
        custom_validator = create_standard_risk_validator(max_position=0.03, max_vol=0.12)
        assert custom_validator.config.max_single_position == 0.03
        assert custom_validator.config.max_portfolio_volatility == 0.12
        assert custom_validator.config.target_portfolio_volatility == 0.096  # 80% of max_vol
    
    def test_conservative_validator_creation(self):
        """Test conservative validator factory."""
        validator = create_conservative_risk_validator()
        
        assert isinstance(validator, RiskValidator)
        assert validator.config.max_single_position == 0.03  # More conservative
        assert validator.config.max_sector_concentration == 0.15  # More conservative
        assert validator.config.max_portfolio_volatility == 0.10  # More conservative
        assert validator.config.max_leverage == 0.95  # More conservative
        assert validator.config.position_sizing_method == PositionSizeMethod.RISK_PARITY


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_portfolio_validation(self):
        """Test validation with empty portfolio."""
        validator = RiskValidator()
        
        position_checks, violations = validator.validate_portfolio(
            portfolio_weights={},
            portfolio_value=1000000,
            date=date.today(),
            market_data=pd.DataFrame()
        )
        
        assert len(position_checks) == 0
        assert len(violations) == 0  # No violations for empty portfolio
    
    def test_single_stock_portfolio(self):
        """Test validation with single stock portfolio."""
        validator = RiskValidator()
        
        single_portfolio = {'TSM': 1.0}  # 100% in one stock
        mock_data = pd.DataFrame([{
            'symbol': 'TSM',
            'volatility': 0.25,
            'beta': 1.1,
            'sector': 'technology',
            'market_cap': 500e9,
            'liquidity_score': 0.9
        }])
        
        position_checks, violations = validator.validate_portfolio(
            single_portfolio,
            portfolio_value=1000000,
            date=date.today(),
            market_data=mock_data
        )
        
        # Should detect massive single position violation
        assert len(position_checks) == 1
        assert len(position_checks[0].violations) > 0
        
        # Should also detect leverage if total weight > max leverage
        if single_portfolio['TSM'] > validator.config.max_leverage:
            leverage_violations = [v for v in violations 
                                 if 'leverage' in v.description.lower()]
            assert len(leverage_violations) > 0
    
    def test_short_positions(self):
        """Test validation with short positions (negative weights)."""
        validator = RiskValidator()
        
        portfolio_with_shorts = {'TSM': 0.8, 'TSLA': -0.3}  # Net 50% long
        
        mock_data = pd.DataFrame([
            {'symbol': 'TSM', 'volatility': 0.25, 'beta': 1.1, 
             'sector': 'technology', 'market_cap': 500e9, 'liquidity_score': 0.9},
            {'symbol': 'TSLA', 'volatility': 0.45, 'beta': 1.8,
             'sector': 'consumer_discretionary', 'market_cap': 800e9, 'liquidity_score': 0.85}
        ])
        
        position_checks, violations = validator.validate_portfolio(
            portfolio_with_shorts,
            portfolio_value=1000000,
            date=date.today(),
            market_data=mock_data
        )
        
        # Should handle negative weights in calculations
        assert len(position_checks) == 2
        
        # Leverage should account for absolute values: |0.8| + |-0.3| = 1.1
        total_leverage = sum(abs(w) for w in portfolio_with_shorts.values())
        if total_leverage > validator.config.max_leverage:
            leverage_violations = [v for v in violations 
                                 if 'leverage' in v.description.lower()]
            assert len(leverage_violations) > 0
    
    def test_zero_volatility_stocks(self):
        """Test handling of stocks with zero volatility."""
        validator = RiskValidator()
        
        alpha_signals = {'ZERO_VOL_STOCK': 0.05, 'NORMAL_STOCK': 0.03}
        
        mock_data = pd.DataFrame([
            {'symbol': 'ZERO_VOL_STOCK', 'volatility': 0.0, 'beta': 1.0,
             'sector': 'utilities', 'market_cap': 100e9, 'liquidity_score': 0.5},
            {'symbol': 'NORMAL_STOCK', 'volatility': 0.20, 'beta': 1.0,
             'sector': 'industrials', 'market_cap': 50e9, 'liquidity_score': 0.7}
        ])
        
        # Should handle zero volatility gracefully in volatility-adjusted sizing
        positions = validator._calculate_volatility_adjusted_positions(
            alpha_signals, mock_data, date.today()
        )
        
        # Should still generate positions, handling division by zero
        assert len(positions) >= 1  # At least the normal stock should be included
    
    def test_missing_market_data(self):
        """Test handling of missing market data for some stocks."""
        validator = RiskValidator()
        
        portfolio = {'EXISTS': 0.5, 'MISSING': 0.5}
        
        partial_data = pd.DataFrame([{
            'symbol': 'EXISTS',
            'volatility': 0.20,
            'beta': 1.0,
            'sector': 'technology',
            'market_cap': 100e9,
            'liquidity_score': 0.8
        }])
        # 'MISSING' stock has no data
        
        position_checks, violations = validator.validate_portfolio(
            portfolio,
            portfolio_value=1000000,
            date=date.today(),
            market_data=partial_data
        )
        
        # Should create checks for all positions, using defaults for missing data
        assert len(position_checks) == 2
        
        # Check that missing data stock has default values
        missing_check = next((c for c in position_checks if c.symbol == 'MISSING'), None)
        assert missing_check is not None
        assert missing_check.volatility == 0.0  # Default value
        assert missing_check.sector == "unknown"


if __name__ == "__main__":
    pytest.main([__file__])