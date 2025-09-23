"""
Comprehensive test suite for the Statistical Testing Framework.

Tests for statistical tests, benchmark management, and integration
with the walk-forward validation system.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.backtesting.validation.statistical_tests import (
    StatisticalTestEngine,
    StatisticalTestConfig,
    TestResult,
    TestType,
    MultipleTestResult,
    BootstrapMethod,
    create_default_statistical_config,
    sharpe_ratio_statistic,
    information_ratio_statistic
)
from src.backtesting.validation.benchmarks import (
    TaiwanBenchmarkManager,
    BenchmarkDefinition,
    BenchmarkCalculator,
    BenchmarkConfig,
    BenchmarkCategory,
    BenchmarkConstituent,
    StyleType,
    RebalanceFrequency,
    create_default_benchmark_config
)
from src.backtesting.validation.walk_forward import (
    ValidationResult,
    ValidationWindow,
    WalkForwardConfig,
    ValidationStatus
)


class TestStatisticalTestConfig:
    """Test StatisticalTestConfig configuration and validation."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = StatisticalTestConfig()
        assert config.alpha_level == 0.05
        assert config.confidence_level == 0.95
        assert config.bootstrap_iterations == 10000
        assert config.bootstrap_method == BootstrapMethod.STATIONARY
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Test invalid alpha level
        with pytest.raises(ValueError, match="Alpha level must be between 0 and 1"):
            StatisticalTestConfig(alpha_level=1.5)
        
        # Test invalid confidence level
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            StatisticalTestConfig(confidence_level=0)
        
        # Test invalid bootstrap iterations
        with pytest.raises(ValueError, match="Bootstrap iterations must be positive"):
            StatisticalTestConfig(bootstrap_iterations=-100)
    
    def test_create_default_config_with_overrides(self):
        """Test creating default config with overrides."""
        config = create_default_statistical_config(
            alpha_level=0.01,
            bootstrap_iterations=5000
        )
        assert config.alpha_level == 0.01
        assert config.bootstrap_iterations == 5000
        assert config.confidence_level == 0.95  # Default unchanged


class TestTestResult:
    """Test TestResult class functionality."""
    
    def test_test_result_creation(self):
        """Test basic test result creation."""
        result = TestResult(
            test_type=TestType.DIEBOLD_MARIANO,
            statistic=2.5,
            p_value=0.01,
            critical_value=1.96,
            sample_size=100,
            null_hypothesis="No difference in accuracy",
            alternative_hypothesis="Model 1 is more accurate"
        )
        
        assert result.test_type == TestType.DIEBOLD_MARIANO
        assert result.statistic == 2.5
        assert result.p_value == 0.01
        assert result.critical_value == 1.96
        assert result.sample_size == 100
    
    def test_significance_setting(self):
        """Test significance flag setting."""
        result = TestResult(
            test_type=TestType.T_TEST,
            statistic=2.0,
            p_value=0.03
        )
        
        # Test significance at 5% level
        result.set_significance(0.05)
        assert result.is_significant
        assert result.reject_null
        
        # Test significance at 1% level
        result.set_significance(0.01)
        assert not result.is_significant
        assert not result.reject_null
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        result = TestResult(
            test_type=TestType.HANSEN_SPA,
            statistic=1.5,
            p_value=0.12
        )
        result.set_significance(0.05)
        
        result_dict = result.to_dict()
        assert result_dict['test_type'] == 'hansen_spa'
        assert result_dict['statistic'] == 1.5
        assert result_dict['p_value'] == 0.12
        assert result_dict['is_significant'] == False


class TestStatisticalTestEngine:
    """Test StatisticalTestEngine functionality."""
    
    @pytest.fixture
    def test_engine(self):
        """Create test engine with default configuration."""
        config = create_default_statistical_config(bootstrap_iterations=1000)
        return StatisticalTestEngine(config)
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data for testing."""
        np.random.seed(42)
        n_periods = 252
        
        # Model returns (slightly better performance)
        model_returns = np.random.normal(0.001, 0.02, n_periods)
        
        # Benchmark returns
        benchmark_returns = np.random.normal(0.0005, 0.015, n_periods)
        
        # Alternative model returns
        alt_model_returns = np.random.normal(0.0008, 0.018, n_periods)
        
        return model_returns, benchmark_returns, alt_model_returns
    
    def test_diebold_mariano_test(self, test_engine, sample_returns):
        """Test Diebold-Mariano test implementation."""
        model_returns, benchmark_returns, _ = sample_returns
        
        # Calculate forecast errors (negative returns)
        model_errors = -model_returns
        benchmark_errors = -benchmark_returns
        
        result = test_engine.diebold_mariano_test(
            model_errors, benchmark_errors, alternative="two-sided"
        )
        
        assert isinstance(result, TestResult)
        assert result.test_type == TestType.DIEBOLD_MARIANO
        assert isinstance(result.statistic, float)
        assert 0 <= result.p_value <= 1
        assert result.sample_size == len(model_errors)
        assert result.null_hypothesis is not None
        assert result.alternative_hypothesis is not None
    
    def test_diebold_mariano_alternatives(self, test_engine, sample_returns):
        """Test different alternatives for Diebold-Mariano test."""
        model_returns, benchmark_returns, _ = sample_returns
        model_errors = -model_returns
        benchmark_errors = -benchmark_returns
        
        # Test all alternatives
        for alternative in ["two-sided", "greater", "less"]:
            result = test_engine.diebold_mariano_test(
                model_errors, benchmark_errors, alternative=alternative
            )
            assert isinstance(result, TestResult)
            assert 0 <= result.p_value <= 1
    
    def test_diebold_mariano_insufficient_data(self, test_engine):
        """Test Diebold-Mariano test with insufficient data."""
        short_series1 = np.array([0.1, 0.2])
        short_series2 = np.array([0.15, 0.25])
        
        with pytest.raises(ValueError, match="Insufficient observations"):
            test_engine.diebold_mariano_test(short_series1, short_series2)
    
    def test_hansen_spa_test(self, test_engine, sample_returns):
        """Test Hansen SPA test implementation."""
        model_returns, benchmark_returns, alt_model_returns = sample_returns
        
        result = test_engine.hansen_spa_test(
            benchmark_returns,
            [model_returns, alt_model_returns]
        )
        
        assert isinstance(result, TestResult)
        assert result.test_type == TestType.HANSEN_SPA
        assert isinstance(result.statistic, float)
        assert 0 <= result.p_value <= 1
        assert result.sample_size == len(benchmark_returns)
    
    def test_white_reality_check(self, test_engine, sample_returns):
        """Test White Reality Check implementation."""
        model_returns, benchmark_returns, alt_model_returns = sample_returns
        
        result = test_engine.white_reality_check(
            benchmark_returns,
            [model_returns, alt_model_returns]
        )
        
        assert isinstance(result, TestResult)
        assert result.test_type == TestType.WHITE_REALITY_CHECK
        assert isinstance(result.statistic, float)
        assert 0 <= result.p_value <= 1
        assert result.sample_size == len(benchmark_returns)
    
    def test_bootstrap_confidence_interval(self, test_engine, sample_returns):
        """Test bootstrap confidence interval calculation."""
        model_returns, _, _ = sample_returns
        
        # Test with Sharpe ratio
        result = test_engine.bootstrap_confidence_interval(
            model_returns,
            lambda x: sharpe_ratio_statistic(x, 0.01),
            confidence_level=0.95
        )
        
        assert isinstance(result, TestResult)
        assert result.test_type == TestType.BOOTSTRAP_CI
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.confidence_interval[1]
    
    def test_bootstrap_methods(self, test_engine, sample_returns):
        """Test different bootstrap methods."""
        model_returns, _, _ = sample_returns
        
        # Test all bootstrap methods
        for method in BootstrapMethod:
            test_engine.config.bootstrap_method = method
            
            try:
                result = test_engine.bootstrap_confidence_interval(
                    model_returns,
                    lambda x: np.mean(x)
                )
                assert isinstance(result, TestResult)
                assert result.confidence_interval is not None
            except Exception as e:
                pytest.fail(f"Bootstrap method {method} failed: {e}")
    
    def test_multiple_testing_correction(self, test_engine):
        """Test multiple testing correction."""
        # Create sample test results
        test_results = []
        p_values = [0.01, 0.03, 0.08, 0.15, 0.25]
        
        for i, p_val in enumerate(p_values):
            result = TestResult(
                test_type=TestType.T_TEST,
                statistic=2.0 - i * 0.3,
                p_value=p_val
            )
            test_results.append(result)
        
        # Test Bonferroni correction
        bonf_result = test_engine.multiple_testing_correction(
            test_results, method="bonferroni"
        )
        
        assert isinstance(bonf_result, MultipleTestResult)
        assert len(bonf_result.adjusted_p_values) == len(test_results)
        assert len(bonf_result.significant_tests) == len(test_results)
        assert bonf_result.method == "bonferroni"
        
        # Test FDR correction
        fdr_result = test_engine.multiple_testing_correction(
            test_results, method="fdr_bh"
        )
        
        assert isinstance(fdr_result, MultipleTestResult)
        assert fdr_result.method == "fdr_bh"


class TestUtilityFunctions:
    """Test utility functions for statistical tests."""
    
    def test_sharpe_ratio_statistic(self):
        """Test Sharpe ratio calculation."""
        # Test with positive returns
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.008])
        sharpe = sharpe_ratio_statistic(returns, 0.01)
        assert isinstance(sharpe, float)
        
        # Test with empty returns
        empty_returns = np.array([])
        sharpe_empty = sharpe_ratio_statistic(empty_returns, 0.01)
        assert sharpe_empty == 0.0
        
        # Test with zero volatility
        constant_returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe_constant = sharpe_ratio_statistic(constant_returns, 0.01)
        assert sharpe_constant == 0.0
    
    def test_information_ratio_statistic(self):
        """Test Information ratio calculation."""
        returns = np.array([0.01, 0.02, -0.005, 0.015])
        benchmark = np.array([0.008, 0.015, -0.002, 0.012])
        
        ir = information_ratio_statistic(returns, benchmark)
        assert isinstance(ir, float)
        
        # Test with mismatched lengths
        short_benchmark = np.array([0.008, 0.015])
        ir_mismatch = information_ratio_statistic(returns, short_benchmark)
        assert ir_mismatch == 0.0


class TestBenchmarkConfig:
    """Test BenchmarkConfig configuration and validation."""
    
    def test_default_config(self):
        """Test default benchmark configuration."""
        config = BenchmarkConfig()
        assert config.rebalance_frequency == RebalanceFrequency.MONTHLY
        assert config.max_single_weight == 0.1
        assert config.max_sector_weight == 0.3
        assert config.include_dividends == True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid max single weight
        with pytest.raises(ValueError, match="Max single weight must be between 0 and 1"):
            BenchmarkConfig(max_single_weight=1.5)
        
        # Test invalid max sector weight
        with pytest.raises(ValueError, match="Max sector weight must be between 0 and 1"):
            BenchmarkConfig(max_sector_weight=0)


class TestBenchmarkConstituent:
    """Test BenchmarkConstituent class."""
    
    def test_constituent_creation(self):
        """Test benchmark constituent creation."""
        constituent = BenchmarkConstituent(
            symbol="2330.TW",
            weight=0.15,
            sector="Technology",
            market_cap=15e12,
            growth_score=1.2,
            value_score=-0.5
        )
        
        assert constituent.symbol == "2330.TW"
        assert constituent.weight == 0.15
        assert constituent.sector == "Technology"
        assert constituent.market_cap == 15e12
        assert constituent.growth_score == 1.2
        assert constituent.value_score == -0.5
    
    def test_constituent_to_dict(self):
        """Test constituent conversion to dictionary."""
        constituent = BenchmarkConstituent(
            symbol="1301.TW",
            weight=0.08,
            sector="Plastics",
            market_cap=2e11,
            inclusion_date=date(2023, 1, 1)
        )
        
        constituent_dict = constituent.to_dict()
        assert constituent_dict['symbol'] == "1301.TW"
        assert constituent_dict['weight'] == 0.08
        assert constituent_dict['sector'] == "Plastics"
        assert constituent_dict['inclusion_date'] == "2023-01-01"


class TestBenchmarkDefinition:
    """Test BenchmarkDefinition class."""
    
    def test_benchmark_definition_creation(self):
        """Test benchmark definition creation."""
        definition = BenchmarkDefinition(
            name="Taiwan_Tech",
            category=BenchmarkCategory.SECTOR,
            description="Taiwan Technology Sector Index",
            universe_filter={"sector": "Technology"},
            weighting_scheme="market_cap",
            style_type=StyleType.GROWTH,
            rebalance_frequency=RebalanceFrequency.QUARTERLY
        )
        
        assert definition.name == "Taiwan_Tech"
        assert definition.category == BenchmarkCategory.SECTOR
        assert definition.style_type == StyleType.GROWTH
        assert definition.rebalance_frequency == RebalanceFrequency.QUARTERLY
    
    def test_benchmark_definition_to_dict(self):
        """Test benchmark definition conversion to dictionary."""
        definition = BenchmarkDefinition(
            name="Custom_Benchmark",
            category=BenchmarkCategory.CUSTOM,
            description="Custom benchmark for testing",
            universe_filter={"market_cap_min": 1e9},
            weighting_scheme="equal"
        )
        
        definition_dict = definition.to_dict()
        assert definition_dict['name'] == "Custom_Benchmark"
        assert definition_dict['category'] == "custom"
        assert definition_dict['weighting_scheme'] == "equal"


class TestBenchmarkCalculator:
    """Test BenchmarkCalculator functionality."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        mock_temporal_store = Mock()
        mock_pit_engine = Mock()
        mock_taiwan_calendar = Mock()
        
        # Mock calendar to return True for all dates (simplified)
        mock_taiwan_calendar.is_trading_day.return_value = True
        
        return mock_temporal_store, mock_pit_engine, mock_taiwan_calendar
    
    @pytest.fixture
    def benchmark_calculator(self, mock_dependencies):
        """Create benchmark calculator with mocked dependencies."""
        temporal_store, pit_engine, taiwan_calendar = mock_dependencies
        config = create_default_benchmark_config()
        
        return BenchmarkCalculator(
            config, temporal_store, pit_engine, taiwan_calendar
        )
    
    def test_benchmark_calculator_initialization(self, benchmark_calculator):
        """Test benchmark calculator initialization."""
        assert benchmark_calculator.config is not None
        assert benchmark_calculator.temporal_store is not None
        assert benchmark_calculator.pit_engine is not None
        assert benchmark_calculator.taiwan_calendar is not None
    
    @patch('src.backtesting.validation.benchmarks.BenchmarkCalculator._get_market_data')
    @patch('src.backtesting.validation.benchmarks.BenchmarkCalculator._filter_universe')
    def test_calculate_benchmark_constituents(
        self, 
        mock_filter, 
        mock_market_data, 
        benchmark_calculator
    ):
        """Test benchmark constituents calculation."""
        # Setup mocks
        mock_filter.return_value = ["2330.TW", "2317.TW"]
        mock_market_data.return_value = {
            "2330.TW": {
                "market_cap": 15e12,
                "sector": "Technology",
                "volume": 50e6
            },
            "2317.TW": {
                "market_cap": 1e12,
                "sector": "Technology",
                "volume": 10e6
            }
        }
        
        # Create test benchmark definition
        benchmark_def = BenchmarkDefinition(
            name="Test_Benchmark",
            category=BenchmarkCategory.MARKET,
            description="Test benchmark",
            universe_filter={"market_cap_min": 1e9},
            weighting_scheme="market_cap"
        )
        
        # Calculate constituents
        constituents = benchmark_calculator.calculate_benchmark_constituents(
            benchmark_def, date(2023, 6, 1), ["2330.TW", "2317.TW", "1301.TW"]
        )
        
        assert len(constituents) == 2
        assert all(isinstance(c, BenchmarkConstituent) for c in constituents)
        
        # Check weights sum to 1
        total_weight = sum(c.weight for c in constituents)
        assert abs(total_weight - 1.0) < 1e-10


class TestTaiwanBenchmarkManager:
    """Test TaiwanBenchmarkManager functionality."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        mock_temporal_store = Mock()
        mock_pit_engine = Mock()
        return mock_temporal_store, mock_pit_engine
    
    @pytest.fixture
    def benchmark_manager(self, mock_dependencies):
        """Create benchmark manager with mocked dependencies."""
        temporal_store, pit_engine = mock_dependencies
        config = create_default_benchmark_config()
        
        return TaiwanBenchmarkManager(config, temporal_store, pit_engine)
    
    def test_benchmark_manager_initialization(self, benchmark_manager):
        """Test benchmark manager initialization."""
        assert benchmark_manager.config is not None
        assert benchmark_manager.calculator is not None
        assert len(benchmark_manager.standard_benchmarks) > 0
    
    def test_standard_benchmarks_creation(self, benchmark_manager):
        """Test creation of standard Taiwan benchmarks."""
        benchmarks = benchmark_manager.standard_benchmarks
        
        # Check that key benchmarks exist
        expected_benchmarks = [
            "TAIEX", "Taiwan_Technology", "Taiwan_Financial",
            "Taiwan_Growth", "Taiwan_Value", "Taiwan_Equal_Weight"
        ]
        
        for bench_name in expected_benchmarks:
            assert bench_name in benchmarks
            assert isinstance(benchmarks[bench_name], BenchmarkDefinition)
    
    def test_list_available_benchmarks(self, benchmark_manager):
        """Test listing available benchmarks."""
        all_benchmarks = benchmark_manager.list_available_benchmarks()
        assert len(all_benchmarks) > 0
        assert "TAIEX" in all_benchmarks
        
        # Test filtering by category
        market_benchmarks = benchmark_manager.list_available_benchmarks(
            category=BenchmarkCategory.MARKET
        )
        assert "TAIEX" in market_benchmarks
        
        sector_benchmarks = benchmark_manager.list_available_benchmarks(
            category=BenchmarkCategory.SECTOR
        )
        assert "Taiwan_Technology" in sector_benchmarks
    
    def test_add_custom_benchmark(self, benchmark_manager):
        """Test adding custom benchmark."""
        custom_benchmark = BenchmarkDefinition(
            name="Custom_Test",
            category=BenchmarkCategory.CUSTOM,
            description="Custom test benchmark",
            universe_filter={},
            weighting_scheme="equal"
        )
        
        benchmark_manager.add_custom_benchmark(custom_benchmark)
        
        assert "Custom_Test" in benchmark_manager.standard_benchmarks
        assert benchmark_manager.get_benchmark_definition("Custom_Test") == custom_benchmark


class TestWalkForwardIntegration:
    """Test integration with walk-forward validation."""
    
    @pytest.fixture
    def mock_validation_result(self):
        """Create mock validation result."""
        windows = []
        for i in range(3):
            window = ValidationWindow(
                window_id=f"test_window_{i}",
                train_start=date(2023, 1, 1) + timedelta(weeks=i*4),
                train_end=date(2023, 6, 1) + timedelta(weeks=i*4),
                test_start=date(2023, 6, 1) + timedelta(weeks=i*4),
                test_end=date(2023, 9, 1) + timedelta(weeks=i*4),
                purge_start=date(2023, 6, 1) + timedelta(weeks=i*4),
                purge_end=date(2023, 6, 1) + timedelta(weeks=i*4),
                window_number=i+1,
                total_train_days=120,
                total_test_days=60,
                purge_days=10,
                trading_days_train=84,
                trading_days_test=42,
                status=ValidationStatus.COMPLETED
            )
            windows.append(window)
        
        return ValidationResult(
            config=WalkForwardConfig(),
            windows=windows,
            total_windows=3,
            successful_windows=3,
            failed_windows=0,
            total_runtime_seconds=150.0
        )
    
    @pytest.fixture
    def mock_returns_data(self):
        """Create mock returns data."""
        np.random.seed(42)
        returns_data = {}
        
        for i in range(3):
            window_id = f"test_window_{i}"
            returns = pd.Series(
                np.random.normal(0.001, 0.02, 60),
                index=pd.date_range(
                    start=date(2023, 6, 1) + timedelta(weeks=i*4),
                    periods=60,
                    freq='D'
                )
            )
            returns_data[window_id] = returns
        
        return returns_data
    
    def test_validation_result_statistical_fields(self, mock_validation_result):
        """Test that ValidationResult has statistical testing fields."""
        result = mock_validation_result
        
        # Check that statistical testing fields exist
        assert hasattr(result, 'statistical_tests')
        assert hasattr(result, 'benchmark_comparisons')
        assert hasattr(result, 'significance_results')
        
        # Initially should be None
        assert result.statistical_tests is None
        assert result.benchmark_comparisons is None
        assert result.significance_results is None


class TestPerformanceMetrics:
    """Test performance metrics calculation and integration."""
    
    def test_performance_config_statistical_settings(self):
        """Test performance config includes statistical testing settings."""
        from src.backtesting.metrics.performance import create_default_performance_config
        
        config = create_default_performance_config(
            enable_statistical_tests=True,
            bootstrap_iterations=1000
        )
        
        assert config.enable_statistical_tests == True
        assert config.bootstrap_iterations == 1000


if __name__ == "__main__":
    # Run with pytest for comprehensive testing
    pytest.main([__file__, "-v", "--tb=short"])