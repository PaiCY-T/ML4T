"""
Test Suite for Statistical Selection Engine - Task #29 Stream A

Comprehensive tests for the Statistical Selection Engine including:
- Large feature set processing (500+ features)
- Memory usage validation
- Information preservation tracking
- Taiwan market compliance
- Pipeline integration testing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the statistical selection engine
from src.feature_selection.statistical import StatisticalSelectionEngine
from src.feature_selection.statistical.correlation_analysis import CorrelationAnalyzer
from src.feature_selection.statistical.variance_filter import VarianceFilter
from src.feature_selection.statistical.mutual_info_selector import MutualInfoSelector
from src.feature_selection.statistical.significance_tester import StatisticalSignificanceTester

@pytest.fixture
def sample_feature_data():
    """Generate sample feature data simulating Taiwan stock market features."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 200  # Manageable size for testing
    
    # Generate base data with different correlation structures
    data = {}
    
    # High-information features (correlated with returns)
    for i in range(50):
        data[f'high_info_feature_{i}'] = np.random.randn(n_samples) + np.sin(np.linspace(0, 10, n_samples))
        
    # Medium-information features 
    for i in range(50):
        data[f'medium_info_feature_{i}'] = np.random.randn(n_samples) * 0.5
        
    # Low-information features (mostly noise)
    for i in range(50):
        data[f'low_info_feature_{i}'] = np.random.randn(n_samples) * 0.1
        
    # Highly correlated features (should be filtered out)
    base_correlated = np.random.randn(n_samples)
    for i in range(25):
        data[f'correlated_feature_{i}'] = base_correlated + np.random.randn(n_samples) * 0.05
        
    # Quasi-constant features (should be filtered out)
    for i in range(25):
        data[f'constant_feature_{i}'] = np.ones(n_samples) + np.random.randn(n_samples) * 0.001
        
    df = pd.DataFrame(data)
    return df

@pytest.fixture  
def sample_target_data():
    """Generate sample target returns."""
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate returns with some autocorrelation and volatility clustering
    returns = []
    vol = 0.02
    
    for i in range(n_samples):
        if i > 0:
            vol = 0.9 * vol + 0.1 * abs(returns[i-1])
        ret = np.random.normal(0.001, vol)  # Small positive mean return
        returns.append(ret)
        
    return pd.Series(returns, name='returns')

@pytest.fixture
def large_feature_data():
    """Generate large feature dataset (500+ features) for memory testing."""
    np.random.seed(42)
    n_samples = 500  # Smaller sample size to manage memory
    n_features = 600  # Large feature set
    
    # Generate features with realistic patterns
    features = {}
    
    # Technical indicators (100 features)
    for i in range(100):
        base = np.random.randn(n_samples).cumsum()
        features[f'tech_indicator_{i}'] = base + np.random.randn(n_samples) * 0.1
        
    # Fundamental ratios (100 features) 
    for i in range(100):
        features[f'fundamental_ratio_{i}'] = np.abs(np.random.randn(n_samples)) + 1
        
    # OpenFE generated features (200 features)
    for i in range(200):
        # Mix of polynomial and interaction features
        base1 = np.random.randn(n_samples)
        base2 = np.random.randn(n_samples) 
        features[f'openfe_feature_{i}'] = base1 * base2 + np.random.randn(n_samples) * 0.2
        
    # Cross-sectional features (100 features)
    for i in range(100):
        features[f'cross_sectional_{i}'] = np.random.randn(n_samples)
        
    # Time-series features (100 features)
    for i in range(100):
        ts = np.random.randn(n_samples).cumsum()
        features[f'time_series_{i}'] = np.roll(ts, i % 10)  # Lagged versions
        
    return pd.DataFrame(features)

class TestStatisticalSelectionEngine:
    """Test the main Statistical Selection Engine."""
    
    def test_initialization(self):
        """Test engine initialization with various parameters."""
        # Default initialization
        engine = StatisticalSelectionEngine()
        assert engine.target_feature_count == 75
        assert engine.correlation_threshold == 0.7
        assert engine.vif_threshold == 10.0
        assert not engine.is_fitted_
        
        # Custom initialization
        engine = StatisticalSelectionEngine(
            target_feature_count=50,
            correlation_threshold=0.8,
            memory_limit_gb=4.0
        )
        assert engine.target_feature_count == 50
        assert engine.correlation_threshold == 0.8
        assert engine.memory_limit_gb == 4.0
        
    def test_data_validation(self, sample_feature_data, sample_target_data):
        """Test input data validation."""
        engine = StatisticalSelectionEngine()
        
        # Valid data
        X_valid, y_valid = engine._validate_input_data(sample_feature_data, sample_target_data)
        assert isinstance(X_valid, pd.DataFrame)
        assert isinstance(y_valid, pd.Series)
        assert len(X_valid) == len(y_valid)
        
        # Missing target
        X_valid, y_none = engine._validate_input_data(sample_feature_data, None)
        assert y_none is None
        
        # Empty features
        with pytest.raises(ValueError, match="Feature matrix X is empty"):
            engine._validate_input_data(pd.DataFrame(), sample_target_data)
            
        # Mismatched lengths
        short_target = sample_target_data.iloc[:100]
        with pytest.raises(ValueError, match="X and y must have same length"):
            engine._validate_input_data(sample_feature_data, short_target)
            
    def test_fit_statistical_selection(self, sample_feature_data, sample_target_data):
        """Test the complete statistical selection pipeline."""
        engine = StatisticalSelectionEngine(
            target_feature_count=30,
            min_feature_count=20,
            max_feature_count=50
        )
        
        # Fit the engine
        result = engine.fit_statistical_selection(sample_feature_data, sample_target_data)
        
        # Check that engine was fitted
        assert engine.is_fitted_
        assert result is engine  # Should return self
        
        # Check results
        assert len(engine.selected_features_) >= engine.min_feature_count
        assert len(engine.selected_features_) <= engine.max_feature_count
        assert engine.final_stats_['original_feature_count'] == sample_feature_data.shape[1]
        assert engine.final_stats_['final_feature_count'] == len(engine.selected_features_)
        
        # Check that stages were completed
        assert 'variance_filtering' in engine.stage_results_
        assert 'correlation_analysis' in engine.stage_results_
        assert 'mutual_information' in engine.stage_results_
        assert 'significance_testing' in engine.stage_results_
        
    def test_fit_without_target(self, sample_feature_data):
        """Test pipeline with no target variable."""
        engine = StatisticalSelectionEngine(target_feature_count=30)
        
        # Fit without target
        engine.fit_statistical_selection(sample_feature_data, y=None)
        
        assert engine.is_fitted_
        assert len(engine.selected_features_) > 0
        
        # Check that supervised stages were skipped
        assert engine.stage_results_['mutual_information']['skipped'] == 'no_target_variable'
        assert engine.stage_results_['significance_testing']['skipped'] == 'no_target_variable'
        
    def test_transform(self, sample_feature_data, sample_target_data):
        """Test feature transformation."""
        engine = StatisticalSelectionEngine(target_feature_count=30)
        
        # Fit and transform
        X_transformed = engine.fit_transform(sample_feature_data, sample_target_data)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape[1] == len(engine.selected_features_)
        assert X_transformed.shape[0] == sample_feature_data.shape[0]
        assert list(X_transformed.columns) == engine.selected_features_
        
        # Test transform on new data
        new_data = sample_feature_data.iloc[:100]  # Smaller subset
        X_transformed_new = engine.transform(new_data)
        assert X_transformed_new.shape[1] == len(engine.selected_features_)
        assert X_transformed_new.shape[0] == 100
        
    def test_transform_not_fitted(self, sample_feature_data):
        """Test transform without fitting."""
        engine = StatisticalSelectionEngine()
        
        with pytest.raises(ValueError, match="StatisticalSelectionEngine not fitted"):
            engine.transform(sample_feature_data)
            
    def test_memory_monitoring(self, sample_feature_data, sample_target_data):
        """Test memory usage monitoring."""
        engine = StatisticalSelectionEngine(
            target_feature_count=30,
            memory_limit_gb=0.1  # Very low limit to trigger warnings
        )
        
        with patch('src.feature_selection.statistical.statistical_engine.logger') as mock_logger:
            engine.fit_statistical_selection(sample_feature_data, sample_target_data)
            
            # Should have memory monitoring data
            assert len(engine.memory_usage_) > 0
            assert 'pipeline_start' in engine.memory_usage_
            assert 'pipeline_complete' in engine.memory_usage_
            
            # Check memory stats
            for stage, stats in engine.memory_usage_.items():
                assert 'memory_gb' in stats
                assert 'timestamp' in stats
                
    def test_large_feature_set_processing(self, large_feature_data):
        """Test processing of large feature sets (500+ features)."""
        # Generate corresponding target
        np.random.seed(42)
        target = pd.Series(np.random.randn(len(large_feature_data)), name='returns')
        
        engine = StatisticalSelectionEngine(
            target_feature_count=75,
            memory_limit_gb=16.0  # Higher limit for large data
        )
        
        # This should complete without memory errors
        engine.fit_statistical_selection(large_feature_data, target)
        
        assert engine.is_fitted_
        assert len(engine.selected_features_) <= engine.target_feature_count
        assert engine.final_stats_['original_feature_count'] == large_feature_data.shape[1]
        
    def test_information_preservation(self, sample_feature_data, sample_target_data):
        """Test information preservation calculation."""
        engine = StatisticalSelectionEngine(target_feature_count=30)
        engine.fit_statistical_selection(sample_feature_data, sample_target_data)
        
        # Should have reasonable information preservation
        info_preservation = engine.final_stats_['information_preservation']
        assert 0 <= info_preservation <= 1
        assert info_preservation > 0.1  # Should preserve some information
        
    def test_feature_scoring_combination(self, sample_feature_data, sample_target_data):
        """Test combination of feature scores from different methods."""
        engine = StatisticalSelectionEngine()
        
        # Mock individual scores
        variance_scores = {f'feature_{i}': np.random.rand() * 0.1 for i in range(10)}
        mi_scores = {f'feature_{i}': np.random.rand() * 0.5 for i in range(10)}
        correlation_scores = {f'feature_{i}': np.random.rand() for i in range(10)}
        significance_pvalues = {f'feature_{i}': np.random.rand() for i in range(10)}
        
        combined_scores = engine._combine_feature_scores(
            variance_scores, mi_scores, correlation_scores, significance_pvalues
        )
        
        assert len(combined_scores) == 10
        for feature, score in combined_scores.items():
            assert 0 <= score <= 1
            assert not np.isnan(score)
            
    def test_get_selection_summary(self, sample_feature_data, sample_target_data):
        """Test selection summary generation."""
        engine = StatisticalSelectionEngine(target_feature_count=30)
        engine.fit_statistical_selection(sample_feature_data, sample_target_data)
        
        summary = engine.get_selection_summary()
        
        # Check required fields
        required_fields = [
            'selected_features', 'feature_scores', 'stage_results',
            'elimination_history', 'final_statistics', 'parameters'
        ]
        for field in required_fields:
            assert field in summary
            
        assert len(summary['selected_features']) == len(engine.selected_features_)
        assert summary['final_statistics']['final_feature_count'] == len(engine.selected_features_)
        
    def test_save_detailed_results(self, sample_feature_data, sample_target_data):
        """Test saving of detailed results."""
        engine = StatisticalSelectionEngine(target_feature_count=20)
        engine.fit_statistical_selection(sample_feature_data, sample_target_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            engine.save_detailed_results(temp_dir)
            
            # Check that files were created
            output_path = Path(temp_dir)
            assert (output_path / 'statistical_selection_results.json').exists()
            assert (output_path / 'selected_features.txt').exists()
            assert (output_path / 'feature_scores.csv').exists()
            assert (output_path / 'elimination_history.csv').exists()
            
            # Validate JSON content
            with open(output_path / 'statistical_selection_results.json', 'r') as f:
                results = json.load(f)
                assert 'selected_features' in results
                assert 'final_statistics' in results
                
    def test_get_feature_importance_report(self, sample_feature_data, sample_target_data):
        """Test feature importance report generation."""
        engine = StatisticalSelectionEngine(target_feature_count=30)
        engine.fit_statistical_selection(sample_feature_data, sample_target_data)
        
        report = engine.get_feature_importance_report(top_n=10)
        
        # Check report structure
        assert 'top_features' in report
        assert 'stage_breakdown' in report
        assert 'overall_statistics' in report
        assert 'elimination_reasons' in report
        
        # Check top features
        assert len(report['top_features']) <= 10
        for feature, score in report['top_features']:
            assert feature in engine.feature_scores_
            assert score == engine.feature_scores_[feature]
            
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        engine = StatisticalSelectionEngine()
        
        # Very small dataset
        small_X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        small_y = pd.Series([0.1, 0.2, 0.3])
        
        engine.fit_statistical_selection(small_X, small_y)
        assert engine.is_fitted_
        
        # Dataset with all constant features
        constant_X = pd.DataFrame({
            'const1': [1, 1, 1, 1, 1],
            'const2': [2, 2, 2, 2, 2]
        })
        constant_y = pd.Series([0, 0, 0, 0, 0])
        
        engine2 = StatisticalSelectionEngine()
        engine2.fit_statistical_selection(constant_X, constant_y)
        # Should handle gracefully, may result in empty selection
        
    @pytest.mark.performance
    def test_processing_time_performance(self, large_feature_data):
        """Test that processing completes within reasonable time limits."""
        target = pd.Series(np.random.randn(len(large_feature_data)), name='returns')
        
        engine = StatisticalSelectionEngine(
            target_feature_count=50,
            memory_limit_gb=16.0
        )
        
        start_time = datetime.now()
        engine.fit_statistical_selection(large_feature_data, target)
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete within 30 minutes as per requirements
        assert elapsed_time < 1800  # 30 minutes
        assert engine.final_stats_['processing_time_seconds'] < 1800
        
        # Performance should be logged
        assert 'processing_time_seconds' in engine.final_stats_
        
    def test_multicollinearity_handling(self, sample_feature_data, sample_target_data):
        """Test handling of multicollinear features."""
        # Add highly correlated features
        correlated_data = sample_feature_data.copy()
        base_feature = sample_feature_data.iloc[:, 0]
        
        # Add correlated versions
        for i in range(5):
            correlated_data[f'highly_corr_{i}'] = base_feature + np.random.randn(len(base_feature)) * 0.01
            
        engine = StatisticalSelectionEngine(
            target_feature_count=30,
            correlation_threshold=0.7,  # Should filter correlated features
            vif_threshold=5.0  # Lower VIF threshold
        )
        
        engine.fit_statistical_selection(correlated_data, sample_target_data)
        
        # Should have eliminated some correlated features
        assert len(engine.elimination_history_) > 0
        corr_eliminations = [r for r in engine.elimination_history_.values() 
                            if 'correlation' in r]
        assert len(corr_eliminations) > 0


class TestIntegrationWithExistingSystem:
    """Test integration with existing ML4T components."""
    
    def test_integration_with_taiwan_config(self):
        """Test integration with Taiwan market configuration."""
        # This would test integration with Taiwan-specific configurations
        # For now, just test that the engine can be initialized
        engine = StatisticalSelectionEngine()
        assert engine is not None
        
    def test_feature_naming_compatibility(self, sample_feature_data, sample_target_data):
        """Test that feature names are compatible with downstream systems."""
        engine = StatisticalSelectionEngine(target_feature_count=20)
        engine.fit_statistical_selection(sample_feature_data, sample_target_data)
        
        # Check that selected feature names are valid
        for feature in engine.selected_features_:
            assert isinstance(feature, str)
            assert len(feature) > 0
            assert feature in sample_feature_data.columns
            
    def test_output_format_compatibility(self, sample_feature_data, sample_target_data):
        """Test that output format is compatible with ML pipeline."""
        engine = StatisticalSelectionEngine(target_feature_count=20)
        X_transformed = engine.fit_transform(sample_feature_data, sample_target_data)
        
        # Check output format
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.index.equals(sample_feature_data.index)
        assert all(col in sample_feature_data.columns for col in X_transformed.columns)