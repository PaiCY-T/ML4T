"""
Test suite for SectorNeutralityAnalyzer.

Tests sector analysis, style factor exposure analysis, and neutrality
validation for Taiwan equity portfolios.

Author: ML4T Team
Date: 2025-09-24
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import Mock, patch

from src.validation.business_logic.sector_analysis import (
    SectorNeutralityAnalyzer,
    SectorConfig,
    SectorExposure,
    StyleExposure,
    NeutralityResult,
    TaiwanSector,
    StyleFactor,
    ExposureLevel,
    create_standard_sector_analyzer,
    create_strict_sector_analyzer
)


class TestSectorConfig:
    """Test sector analysis configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SectorConfig()
        
        assert config.max_sector_deviation == 0.05
        assert config.max_style_deviation == 0.5
        assert config.max_concentration_hhi == 0.10
        assert config.benchmark_index == "TAIEX"
        assert config.lookback_period_days == 252
        assert config.include_tpex_stocks is True
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = SectorConfig(
            max_sector_deviation=0.03,
            max_style_deviation=0.3,
            benchmark_index="TPEx",
            value_metrics=['price_to_book'],
            growth_metrics=['revenue_growth_1y']
        )
        
        assert config.max_sector_deviation == 0.03
        assert config.max_style_deviation == 0.3
        assert config.benchmark_index == "TPEx"
        assert config.value_metrics == ['price_to_book']
        assert config.growth_metrics == ['revenue_growth_1y']


class TestSectorExposure:
    """Test sector exposure analysis."""
    
    def test_sector_exposure_creation(self):
        """Test sector exposure object creation."""
        exposure = SectorExposure(
            sector=TaiwanSector.TECHNOLOGY,
            portfolio_weight=0.45,
            benchmark_weight=0.35,
            active_weight=0.10,
            relative_exposure=0.286,  # 0.10 / 0.35
            exposure_level=ExposureLevel.HIGH,
            number_of_stocks=15,
            concentration_ratio=0.08
        )
        
        assert exposure.sector == TaiwanSector.TECHNOLOGY
        assert exposure.active_weight == 0.10
        assert exposure.relative_exposure == pytest.approx(0.286, rel=1e-3)
        assert exposure.exposure_level == ExposureLevel.HIGH
        
    def test_sector_exposure_serialization(self):
        """Test sector exposure to dictionary conversion."""
        exposure = SectorExposure(
            sector=TaiwanSector.FINANCIALS,
            portfolio_weight=0.20,
            benchmark_weight=0.25,
            active_weight=-0.05,
            relative_exposure=-0.20,
            exposure_level=ExposureLevel.LOW,
            number_of_stocks=8,
            concentration_ratio=0.12
        )
        
        result_dict = exposure.to_dict()
        
        assert result_dict['sector'] == 'financials'
        assert result_dict['portfolio_weight'] == 0.20
        assert result_dict['benchmark_weight'] == 0.25
        assert result_dict['active_weight'] == -0.05
        assert result_dict['exposure_level'] == 'low'


class TestStyleExposure:
    """Test style factor exposure analysis."""
    
    def test_style_exposure_creation(self):
        """Test style exposure object creation."""
        exposure = StyleExposure(
            factor=StyleFactor.VALUE,
            portfolio_loading=0.3,
            benchmark_loading=0.0,
            active_loading=0.3,
            t_statistic=2.1,
            significance_level=0.036,
            exposure_level=ExposureLevel.HIGH,
            factor_volatility=0.15
        )
        
        assert exposure.factor == StyleFactor.VALUE
        assert exposure.active_loading == 0.3
        assert exposure.t_statistic == 2.1
        assert exposure.significance_level == pytest.approx(0.036, rel=1e-3)
        assert exposure.exposure_level == ExposureLevel.HIGH
        
    def test_style_exposure_serialization(self):
        """Test style exposure to dictionary conversion."""
        exposure = StyleExposure(
            factor=StyleFactor.MOMENTUM,
            portfolio_loading=-0.2,
            benchmark_loading=0.1,
            active_loading=-0.3,
            t_statistic=-1.8,
            significance_level=0.072,
            exposure_level=ExposureLevel.LOW,
            factor_volatility=0.20
        )
        
        result_dict = exposure.to_dict()
        
        assert result_dict['factor'] == 'momentum'
        assert result_dict['portfolio_loading'] == -0.2
        assert result_dict['benchmark_loading'] == 0.1
        assert result_dict['active_loading'] == -0.3
        assert result_dict['exposure_level'] == 'low'


class TestNeutralityResult:
    """Test neutrality analysis result."""
    
    def test_neutrality_result_creation(self):
        """Test neutrality result creation."""
        sector_exposures = [
            SectorExposure(
                TaiwanSector.TECHNOLOGY, 0.4, 0.35, 0.05, 0.143,
                ExposureLevel.NEUTRAL, 10, 0.06
            ),
            SectorExposure(
                TaiwanSector.FINANCIALS, 0.2, 0.25, -0.05, -0.20,
                ExposureLevel.NEUTRAL, 5, 0.08
            )
        ]
        
        style_exposures = [
            StyleExposure(
                StyleFactor.VALUE, 0.1, 0.0, 0.1, 1.2, 0.23,
                ExposureLevel.NEUTRAL, 0.15
            )
        ]
        
        result = NeutralityResult(
            is_sector_neutral=True,
            sector_exposures=sector_exposures,
            style_exposures=style_exposures,
            total_active_risk=0.08,
            sector_contribution_to_risk=0.6,
            style_contribution_to_risk=0.4,
            concentration_hhi=0.06,
            max_sector_deviation=0.05,
            max_style_deviation=0.1,
            neutrality_score=85.0
        )
        
        assert result.is_sector_neutral is True
        assert len(result.sector_exposures) == 2
        assert len(result.style_exposures) == 1
        assert result.neutrality_score == 85.0
        
    def test_violation_detection(self):
        """Test violation detection in neutrality result."""
        config = SectorConfig(
            max_sector_deviation=0.03,
            max_style_deviation=0.2,
            max_concentration_hhi=0.08
        )
        
        sector_exposures = [
            SectorExposure(
                TaiwanSector.TECHNOLOGY, 0.4, 0.35, 0.05, 0.143,  # Violates 3% limit
                ExposureLevel.HIGH, 10, 0.06
            )
        ]
        
        style_exposures = [
            StyleExposure(
                StyleFactor.MOMENTUM, 0.3, 0.0, 0.3, 2.0, 0.05,  # Violates 0.2 limit
                ExposureLevel.HIGH, 0.18
            )
        ]
        
        result = NeutralityResult(
            is_sector_neutral=False,
            sector_exposures=sector_exposures,
            style_exposures=style_exposures,
            total_active_risk=0.10,
            sector_contribution_to_risk=0.7,
            style_contribution_to_risk=0.3,
            concentration_hhi=0.09,  # Violates 8% limit
            max_sector_deviation=0.05,
            max_style_deviation=0.3,
            neutrality_score=60.0
        )
        
        violations = result.get_violations(config)
        
        assert len(violations) >= 3  # Should detect all violations
        
        # Check for specific violation types
        violation_text = ' '.join(violations)
        assert 'technology' in violation_text.lower()
        assert 'momentum' in violation_text.lower()
        assert 'concentration' in violation_text.lower()


class TestSectorNeutralityAnalyzer:
    """Test the main sector neutrality analyzer."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Sample portfolio for testing."""
        return {
            'TSM': 0.15,    # Technology
            '2317': 0.10,   # Technology  
            '2330': 0.20,   # Technology
            '2882': 0.08,   # Financials
            '2454': 0.07,   # Technology
            '1303': 0.05,   # Materials
            '2002': 0.05,   # Consumer Staples
            '3008': 0.05,   # Industrials
            '2412': 0.05,   # Technology
            '6505': 0.20    # Technology - creates concentration
        }
    
    @pytest.fixture
    def mock_sector_data(self):
        """Mock sector classification data."""
        return pd.DataFrame([
            {'symbol': 'TSM', 'sector': 'technology'},
            {'symbol': '2317', 'sector': 'technology'},
            {'symbol': '2330', 'sector': 'technology'},
            {'symbol': '2882', 'sector': 'financials'},
            {'symbol': '2454', 'sector': 'technology'},
            {'symbol': '1303', 'sector': 'materials'},
            {'symbol': '2002', 'sector': 'consumer_staples'},
            {'symbol': '3008', 'sector': 'industrials'},
            {'symbol': '2412', 'sector': 'technology'},
            {'symbol': '6505', 'sector': 'technology'}
        ])
    
    @pytest.fixture
    def mock_style_data(self):
        """Mock style factor data."""
        symbols = ['TSM', '2317', '2330', '2882', '2454', '1303', '2002', '3008', '2412', '6505']
        data = []
        
        for symbol in symbols:
            data.append({
                'symbol': symbol,
                'value_loading': np.random.uniform(-1, 1),
                'growth_loading': np.random.uniform(-1, 1),
                'momentum_loading': np.random.uniform(-1, 1),
                'quality_loading': np.random.uniform(-1, 1),
                'low_volatility_loading': np.random.uniform(-1, 1),
                'size_loading': np.random.uniform(-1, 1),
                'liquidity_loading': np.random.uniform(-1, 1)
            })
        
        return pd.DataFrame(data)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        # Default configuration
        analyzer = SectorNeutralityAnalyzer()
        assert analyzer.config.max_sector_deviation == 0.05
        assert analyzer.config.benchmark_index == "TAIEX"
        
        # Custom configuration
        config = SectorConfig(max_sector_deviation=0.03)
        analyzer = SectorNeutralityAnalyzer(config)
        assert analyzer.config.max_sector_deviation == 0.03
    
    def test_neutrality_analysis(self, sample_portfolio, mock_sector_data, mock_style_data):
        """Test comprehensive neutrality analysis."""
        analyzer = SectorNeutralityAnalyzer()
        
        with patch.object(analyzer, '_get_sector_data', return_value=mock_sector_data):
            with patch.object(analyzer, '_get_style_data', return_value=mock_style_data):
                with patch.object(analyzer, '_get_benchmark_sector_weights') as mock_benchmark:
                    # Mock balanced benchmark weights
                    mock_benchmark.return_value = {
                        sector: 1.0 / len(TaiwanSector) for sector in TaiwanSector
                    }
                    
                    result = analyzer.analyze_neutrality(
                        sample_portfolio,
                        date.today(),
                        mock_sector_data,
                        mock_style_data
                    )
                    
                    assert isinstance(result, NeutralityResult)
                    assert len(result.sector_exposures) == len(TaiwanSector)
                    assert len(result.style_exposures) == len(StyleFactor)
                    assert 0 <= result.neutrality_score <= 100
                    
                    # Should detect technology concentration
                    tech_exposure = next(
                        (exp for exp in result.sector_exposures 
                         if exp.sector == TaiwanSector.TECHNOLOGY), 
                        None
                    )
                    assert tech_exposure is not None
                    assert tech_exposure.portfolio_weight > 0.5  # >50% in tech
    
    def test_sector_weight_calculation(self, sample_portfolio, mock_sector_data):
        """Test portfolio sector weight calculation."""
        analyzer = SectorNeutralityAnalyzer()
        
        sector_weights = analyzer._calculate_portfolio_sector_weights(
            sample_portfolio, mock_sector_data
        )
        
        # Technology should dominate (TSM + 2317 + 2330 + 2454 + 2412 + 6505)
        tech_weight = sector_weights[TaiwanSector.TECHNOLOGY]
        expected_tech_weight = 0.15 + 0.10 + 0.20 + 0.07 + 0.05 + 0.20  # 0.77
        
        assert tech_weight == pytest.approx(expected_tech_weight, rel=1e-3)
        
        # Financials should have one stock
        fin_weight = sector_weights[TaiwanSector.FINANCIALS]
        assert fin_weight == 0.08
    
    def test_factor_loading_calculation(self, sample_portfolio, mock_style_data):
        """Test portfolio factor loading calculation."""
        analyzer = SectorNeutralityAnalyzer()
        
        factor_loadings = analyzer._calculate_portfolio_factor_loadings(
            sample_portfolio, mock_style_data
        )
        
        # Should have loadings for all style factors
        assert len(factor_loadings) == len(StyleFactor)
        
        for factor in StyleFactor:
            assert factor in factor_loadings
            assert isinstance(factor_loadings[factor], (int, float))
    
    def test_concentration_calculation(self, sample_portfolio):
        """Test HHI concentration calculation."""
        analyzer = SectorNeutralityAnalyzer()
        
        hhi = analyzer._calculate_concentration_hhi(sample_portfolio)
        
        # Calculate expected HHI
        expected_hhi = sum(w**2 for w in sample_portfolio.values())
        assert hhi == pytest.approx(expected_hhi, rel=1e-6)
        
        # Test with equal weights (should be 1/n)
        equal_portfolio = {f'stock_{i}': 0.1 for i in range(10)}
        equal_hhi = analyzer._calculate_concentration_hhi(equal_portfolio)
        assert equal_hhi == pytest.approx(0.1, rel=1e-6)
    
    def test_exposure_level_classification(self):
        """Test exposure level classification."""
        analyzer = SectorNeutralityAnalyzer()
        
        # Test different exposure ratios
        assert analyzer._classify_exposure_level(-2.5) == ExposureLevel.VERY_LOW
        assert analyzer._classify_exposure_level(-1.5) == ExposureLevel.LOW
        assert analyzer._classify_exposure_level(0.0) == ExposureLevel.NEUTRAL
        assert analyzer._classify_exposure_level(0.5) == ExposureLevel.NEUTRAL
        assert analyzer._classify_exposure_level(1.5) == ExposureLevel.HIGH
        assert analyzer._classify_exposure_level(2.5) == ExposureLevel.VERY_HIGH
    
    def test_neutrality_score_calculation(self):
        """Test neutrality score calculation."""
        analyzer = SectorNeutralityAnalyzer()
        
        # Create mock exposures
        sector_exposures = [
            Mock(active_weight=0.02),  # Within 5% limit
            Mock(active_weight=-0.03), # Within 5% limit
            Mock(active_weight=0.08)   # Exceeds 5% limit
        ]
        
        style_exposures = [
            Mock(active_loading=0.3),  # Within 0.5 limit
            Mock(active_loading=-0.2), # Within 0.5 limit
            Mock(active_loading=0.7)   # Exceeds 0.5 limit
        ]
        
        score = analyzer._calculate_neutrality_score(
            sector_exposures, style_exposures, 0.08  # Below 0.10 HHI limit
        )
        
        # Score should be penalized for violations but not zero
        assert 0 < score < 100
    
    def test_risk_decomposition(self, sample_portfolio, mock_sector_data, mock_style_data):
        """Test risk decomposition calculation."""
        analyzer = SectorNeutralityAnalyzer()
        
        # Create mock exposures for testing
        sector_exposures = [Mock(active_weight=0.05), Mock(active_weight=-0.03)]
        style_exposures = [Mock(active_loading=0.2), Mock(active_loading=-0.1)]
        
        risk_decomp = analyzer._calculate_risk_decomposition(
            sample_portfolio, sector_exposures, style_exposures, date.today()
        )
        
        assert 'total_risk' in risk_decomp
        assert 'sector_risk' in risk_decomp
        assert 'style_risk' in risk_decomp
        
        assert risk_decomp['total_risk'] >= 0
        assert 0 <= risk_decomp['sector_risk'] <= 1
        assert 0 <= risk_decomp['style_risk'] <= 1
        assert abs(risk_decomp['sector_risk'] + risk_decomp['style_risk'] - 1.0) < 0.01


class TestAnalyzerFactories:
    """Test analyzer factory functions."""
    
    def test_standard_analyzer_creation(self):
        """Test standard analyzer factory."""
        analyzer = create_standard_sector_analyzer()
        
        assert isinstance(analyzer, SectorNeutralityAnalyzer)
        assert analyzer.config.benchmark_index == "TAIEX"
        assert analyzer.config.max_sector_deviation == 0.05
        
        # Test custom parameters
        custom_analyzer = create_standard_sector_analyzer(
            benchmark="TPEx",
            max_sector_dev=0.03
        )
        assert custom_analyzer.config.benchmark_index == "TPEx"
        assert custom_analyzer.config.max_sector_deviation == 0.03
    
    def test_strict_analyzer_creation(self):
        """Test strict analyzer factory."""
        analyzer = create_strict_sector_analyzer()
        
        assert isinstance(analyzer, SectorNeutralityAnalyzer)
        assert analyzer.config.benchmark_index == "TAIEX"
        assert analyzer.config.max_sector_deviation == 0.02  # Stricter
        assert analyzer.config.max_style_deviation == 0.3     # Stricter
        assert analyzer.config.max_concentration_hhi == 0.05  # Stricter
        
        # Test custom benchmark
        custom_analyzer = create_strict_sector_analyzer(benchmark="TPEx")
        assert custom_analyzer.config.benchmark_index == "TPEx"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_portfolio_analysis(self):
        """Test analysis with empty portfolio."""
        analyzer = SectorNeutralityAnalyzer()
        
        with patch.object(analyzer, '_get_sector_data', return_value=pd.DataFrame()):
            with patch.object(analyzer, '_get_style_data', return_value=pd.DataFrame()):
                result = analyzer.analyze_neutrality({}, date.today())
                
                assert isinstance(result, NeutralityResult)
                assert len(result.sector_exposures) == len(TaiwanSector)
                assert len(result.style_exposures) == len(StyleFactor)
                assert result.neutrality_score >= 0
    
    def test_missing_sector_data(self):
        """Test handling of missing sector data."""
        analyzer = SectorNeutralityAnalyzer()
        
        portfolio = {'UNKNOWN_STOCK': 0.5, 'ANOTHER_UNKNOWN': 0.5}
        empty_data = pd.DataFrame()
        
        with patch.object(analyzer, '_get_sector_data', return_value=empty_data):
            with patch.object(analyzer, '_get_style_data', return_value=empty_data):
                result = analyzer.analyze_neutrality(portfolio, date.today())
                
                # Should handle gracefully without crashing
                assert isinstance(result, NeutralityResult)
    
    def test_extreme_concentration(self):
        """Test analysis with extreme portfolio concentration."""
        analyzer = SectorNeutralityAnalyzer()
        
        # Single stock portfolio
        extreme_portfolio = {'SINGLE_STOCK': 1.0}
        
        sector_data = pd.DataFrame([{'symbol': 'SINGLE_STOCK', 'sector': 'technology'}])
        style_data = pd.DataFrame([{
            'symbol': 'SINGLE_STOCK',
            'value_loading': 0.5,
            'growth_loading': -0.3,
            'momentum_loading': 0.2,
            'quality_loading': 0.8,
            'low_volatility_loading': -0.1,
            'size_loading': 0.0,
            'liquidity_loading': 0.4
        }])
        
        with patch.object(analyzer, '_get_sector_data', return_value=sector_data):
            with patch.object(analyzer, '_get_style_data', return_value=style_data):
                result = analyzer.analyze_neutrality(extreme_portfolio, date.today())
                
                assert isinstance(result, NeutralityResult)
                assert result.concentration_hhi == 1.0  # Perfect concentration
                assert result.neutrality_score < 50     # Should be low due to concentration


if __name__ == "__main__":
    pytest.main([__file__])