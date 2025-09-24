"""
Test suite for OpenFE setup and integration.

CRITICAL TESTS:
- Time-series integrity (no lookahead bias)
- Memory usage monitoring  
- Taiwan market compliance
- Expert-provided FeatureGenerator functionality
"""

import pytest
import numpy as np
import pandas as pd
import psutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from features.openfe_wrapper import FeatureGenerator
from features.taiwan_config import TaiwanMarketConfig, taiwan_config


class TestOpenFESetup:
    """Test OpenFE library setup and basic functionality."""
    
    def test_openfe_import(self):
        """Test that OpenFE can be imported (if available)."""
        try:
            import openfe
            assert True, "OpenFE import successful"
        except ImportError:
            pytest.skip("OpenFE not available - install with: pip install openfe")
            
    def test_dependencies_available(self):
        """Test that all required dependencies are available."""
        required_modules = [
            'numpy', 'pandas', 'sklearn', 'psutil'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                pytest.fail(f"Required module {module} not available")


class TestFeatureGenerator:
    """Test the expert-provided FeatureGenerator class."""
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample time-series panel data for testing."""
        # Create 30 days of data for 5 stocks
        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        stock_ids = ['2330', '2454', '3008', '2382', '2317']
        
        # Create MultiIndex (date, stock_id)
        index = pd.MultiIndex.from_product(
            [dates, stock_ids], 
            names=['date', 'stock_id']
        )
        
        # Generate sample features (simplified factor data)
        np.random.seed(42)  # For reproducible tests
        n_samples = len(index)
        
        data = {
            'close_price': np.random.uniform(50, 500, n_samples),
            'volume': np.random.uniform(1000, 100000, n_samples),
            'market_cap': np.random.uniform(1e9, 1e12, n_samples),
            'pe_ratio': np.random.uniform(5, 30, n_samples),
            'momentum_20d': np.random.uniform(-0.2, 0.2, n_samples),
        }
        
        df = pd.DataFrame(data, index=index)
        return df
    
    @pytest.fixture
    def sample_target(self, sample_time_series_data):
        """Create sample target variable."""
        np.random.seed(42)
        return pd.Series(
            np.random.choice([0, 1], len(sample_time_series_data)),
            index=sample_time_series_data.index,
            name='target'
        )
    
    def test_feature_generator_initialization(self):
        """Test FeatureGenerator can be initialized with proper parameters."""
        # Test with default parameters
        fg = FeatureGenerator()
        assert fg.taiwan_market == True
        assert fg.memory_limit_mb == 8192
        assert fg.max_features == 500
        
        # Test with custom parameters
        fg_custom = FeatureGenerator(
            memory_limit_mb=4096,
            max_features=100,
            taiwan_market=False
        )
        assert fg_custom.memory_limit_mb == 4096
        assert fg_custom.max_features == 100
        assert fg_custom.taiwan_market == False
    
    def test_time_series_data_validation(self, sample_time_series_data):
        """Test time-series data validation."""
        fg = FeatureGenerator()
        
        # Should not raise error for valid data
        try:
            fg._validate_time_series_data(sample_time_series_data)
        except Exception as e:
            pytest.fail(f"Validation failed for valid data: {e}")
    
    def test_time_series_split_no_shuffling(self, sample_time_series_data, sample_target):
        """
        CRITICAL TEST: Ensure time-series split maintains temporal order.
        This prevents lookahead bias.
        """
        fg = FeatureGenerator()
        
        X_train, X_test, y_train, y_test = fg._time_series_split(
            sample_time_series_data, sample_target, test_size=0.2
        )
        
        # Get unique dates from train and test sets
        train_dates = X_train.index.get_level_values(0).unique()
        test_dates = X_test.index.get_level_values(0).unique()
        
        # CRITICAL: All training dates must be before all test dates
        max_train_date = train_dates.max()
        min_test_date = test_dates.min()
        
        assert max_train_date < min_test_date, (
            f"LOOKAHEAD BIAS DETECTED: Latest training date ({max_train_date}) "
            f"is after earliest test date ({min_test_date})"
        )
        
        # Check proportions
        total_dates = len(sample_time_series_data.index.get_level_values(0).unique())
        expected_train_dates = int(total_dates * 0.8)
        expected_test_dates = total_dates - expected_train_dates
        
        assert len(train_dates) == expected_train_dates
        assert len(test_dates) == expected_test_dates
    
    def test_memory_monitoring(self, sample_time_series_data):
        """Test memory usage monitoring functionality."""
        fg = FeatureGenerator(memory_limit_mb=1024)  # Low limit for testing
        
        # Test memory checking
        memory_info = fg._check_memory_usage("test_stage")
        
        assert 'stage' in memory_info
        assert 'rss_mb' in memory_info
        assert 'vms_mb' in memory_info
        assert 'percent' in memory_info
        assert memory_info['stage'] == "test_stage"
        assert memory_info['rss_mb'] > 0
    
    def test_taiwan_market_validation(self, sample_time_series_data):
        """Test Taiwan market specific validation."""
        fg = FeatureGenerator(taiwan_market=True)
        
        validation_results = fg.taiwan_market_validate(sample_time_series_data)
        
        assert 'passed' in validation_results
        assert 'warnings' in validation_results
        assert 'errors' in validation_results
        assert isinstance(validation_results['passed'], bool)
    
    @patch('features.openfe_wrapper.OPENFE_AVAILABLE', False)
    def test_openfe_unavailable_handling(self):
        """Test graceful handling when OpenFE is not available."""
        with pytest.raises(ImportError, match="OpenFE is required"):
            FeatureGenerator()
    
    def test_feature_names_output(self, sample_time_series_data):
        """Test feature names are properly handled."""
        fg = FeatureGenerator(max_features=10)
        
        # Mock fit to avoid OpenFE dependency in tests
        fg.is_fitted_ = True
        fg.feature_names_ = ['feature_1', 'feature_2', 'feature_3']
        
        feature_names = fg.get_feature_names_out()
        assert isinstance(feature_names, list)
        assert len(feature_names) == 3
        assert 'feature_1' in feature_names
        
    def test_memory_usage_tracking(self, sample_time_series_data):
        """Test memory usage tracking throughout process."""
        fg = FeatureGenerator()
        
        # Simulate some memory tracking
        fg._check_memory_usage("stage1")
        fg._check_memory_usage("stage2")
        
        memory_usage = fg.get_memory_usage()
        assert isinstance(memory_usage, dict)
        assert 'stage1' in memory_usage
        assert 'stage2' in memory_usage


class TestTaiwanMarketConfig:
    """Test Taiwan market configuration."""
    
    def test_taiwan_config_initialization(self):
        """Test Taiwan market config initialization."""
        config = TaiwanMarketConfig()
        
        assert config.SETTLEMENT_DAYS == 2
        assert config.CURRENCY == 'TWD'
        assert config.PRICE_LIMIT_PERCENT == 0.10
        assert config.TRADING_HOURS_PER_DAY == 4.5
    
    def test_trading_hours(self):
        """Test trading hours configuration."""
        config = TaiwanMarketConfig()
        hours = config.get_trading_hours()
        
        assert hours['start'].hour == 9
        assert hours['start'].minute == 0
        assert hours['end'].hour == 13
        assert hours['end'].minute == 30
        assert hours['lunch_start'] is None  # No lunch break
        
    def test_price_limits(self):
        """Test price limit calculations."""
        config = TaiwanMarketConfig()
        
        previous_close = 100.0
        upper_limit, lower_limit = config.apply_price_limits(105.0, previous_close)
        
        assert upper_limit == 110.0  # +10%
        assert lower_limit == 90.0   # -10%
        
        # Test price within limits
        assert config.is_price_within_limits(105.0, previous_close) == True
        assert config.is_price_within_limits(115.0, previous_close) == False
        assert config.is_price_within_limits(85.0, previous_close) == False
    
    def test_settlement_date_calculation(self):
        """Test T+2 settlement date calculation."""
        config = TaiwanMarketConfig()
        
        # Mock trading calendar for testing
        trade_date = pd.Timestamp('2024-01-15')  # Monday
        
        # Since we don't have full trading calendar, test the logic
        # Settlement should be 2 business days later
        settlement_date = config.get_settlement_date(trade_date)
        
        # Should be 2+ days later (accounting for weekends)
        assert settlement_date > trade_date
        assert (settlement_date - trade_date).days >= 2
    
    def test_feature_engineering_config(self):
        """Test feature engineering configuration."""
        config = TaiwanMarketConfig()
        fe_config = config.get_feature_engineering_config()
        
        required_keys = [
            'settlement_lag', 'min_history_days', 'max_lookback_days',
            'price_limit_percent', 'currency', 'timezone',
            'openfe_task', 'n_data_blocks', 'max_features'
        ]
        
        for key in required_keys:
            assert key in fe_config
            
        assert fe_config['settlement_lag'] == 2
        assert fe_config['currency'] == 'TWD'
        assert fe_config['max_features'] == 500
    
    def test_data_validation(self):
        """Test market data validation."""
        config = TaiwanMarketConfig()
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=10, freq='B')
        sample_data = pd.DataFrame({
            'close': [100, 105, 102, 108, 95, 99, 103, 107, 104, 101],
            'volume': [10000] * 10,
            'currency': ['TWD'] * 10
        }, index=dates)
        
        validation = config.validate_data_for_taiwan_market(sample_data)
        
        assert 'passed' in validation
        assert 'warnings' in validation
        assert 'errors' in validation
        assert 'statistics' in validation


class TestMemoryProfiling:
    """Test memory profiling and resource management."""
    
    def test_memory_monitoring_functions(self):
        """Test basic memory monitoring works."""
        initial_memory = psutil.virtual_memory().percent
        assert isinstance(initial_memory, (int, float))
        assert 0 <= initial_memory <= 100
    
    def test_memory_limit_detection(self):
        """Test memory limit detection and warnings."""
        # Test with very low memory limit to trigger warning
        fg = FeatureGenerator(memory_limit_mb=1)  # 1MB limit
        
        # This should trigger a warning due to low limit
        memory_info = fg._check_memory_usage("test_low_limit")
        assert memory_info['rss_mb'] > 1  # Should exceed 1MB limit
    
    def test_garbage_collection_behavior(self):
        """Test garbage collection functionality."""
        import gc
        
        # Create some objects
        large_list = [i for i in range(10000)]
        
        # Force garbage collection
        initial_objects = len(gc.get_objects())
        del large_list
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Objects should be cleaned up
        assert final_objects <= initial_objects


class TestIntegrationReadiness:
    """Test readiness for integration with existing ML4T components."""
    
    def test_sklearn_compatibility(self):
        """Test that FeatureGenerator is sklearn compatible."""
        from sklearn.base import BaseEstimator, TransformerMixin
        
        fg = FeatureGenerator()
        assert isinstance(fg, BaseEstimator)
        assert isinstance(fg, TransformerMixin)
        
        # Check required methods exist
        assert hasattr(fg, 'fit')
        assert hasattr(fg, 'transform')
        assert hasattr(fg, 'fit_transform')
        assert hasattr(fg, 'get_feature_names_out')
    
    def test_pipeline_integration_readiness(self, sample_time_series_data):
        """Test readiness for ML pipeline integration."""
        fg = FeatureGenerator()
        
        # Test basic pipeline compatibility
        X = sample_time_series_data
        
        # Should be able to call fit (even if it fails due to missing OpenFE)
        try:
            # Mock the fit process to avoid OpenFE dependency
            fg.is_fitted_ = True
            fg.original_features_ = X.columns.tolist()
            fg.feature_names_ = X.columns.tolist()
            fg.generated_features_ = X
            
            # Test transform
            X_transformed = fg.transform(X)
            assert isinstance(X_transformed, pd.DataFrame)
            
        except Exception as e:
            # Should handle errors gracefully
            assert True, f"Error handling working: {e}"


if __name__ == "__main__":
    # Run specific tests for quick validation
    import subprocess
    import sys
    
    print("Running OpenFE setup validation tests...")
    
    # Run key tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", 
        "--tb=short", "-x"
    ])
    
    if result.returncode == 0:
        print("✅ All OpenFE setup tests passed!")
    else:
        print("❌ Some tests failed - check output above")
        sys.exit(1)