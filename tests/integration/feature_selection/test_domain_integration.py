"""
Integration tests for Domain Validation Pipeline (Stream C).

Tests the complete domain validation and integration workflow.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from src.feature_selection.domain import (
    DomainValidationPipeline,
    DomainValidationConfig,
    TaiwanMarketComplianceValidator,
    EconomicIntuitionScorer,
    BusinessLogicValidator,
    ICPerformanceTester
)


class TestDomainValidationPipeline:
    """Test suite for domain validation pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        
        # Create date range
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        n_dates = len(dates)
        n_assets = 50
        
        # Generate feature data
        np.random.seed(42)
        
        # Create sample features with different characteristics
        features = {}
        
        # Fundamental features
        features['pe_ratio'] = np.random.gamma(2, 5, n_dates * n_assets)  # PE ratios
        features['pb_ratio'] = np.random.gamma(1.5, 2, n_dates * n_assets)  # PB ratios
        features['roe'] = np.random.normal(0.1, 0.05, n_dates * n_assets)  # ROE
        features['debt_equity_ratio'] = np.random.gamma(1, 2, n_dates * n_assets)  # D/E ratios
        
        # Technical features  
        features['rsi_14'] = np.random.uniform(20, 80, n_dates * n_assets)  # RSI
        features['ma_20'] = np.random.normal(100, 20, n_dates * n_assets)  # Moving average
        features['volatility_20'] = np.random.gamma(2, 0.01, n_dates * n_assets)  # Volatility
        
        # Volume features
        features['volume_sma_10'] = np.random.lognormal(10, 1, n_dates * n_assets)  # Volume
        features['volume_ratio'] = np.random.gamma(2, 0.5, n_dates * n_assets)  # Volume ratio
        
        # Low-quality features (should be filtered out)
        features['constant_feature'] = np.ones(n_dates * n_assets)  # Constant
        features['random_noise'] = np.random.normal(0, 1, n_dates * n_assets)  # Pure noise
        features['future_return'] = np.random.normal(0, 0.02, n_dates * n_assets)  # Look-ahead bias
        
        # Create MultiIndex DataFrame
        asset_ids = [f'ASSET_{i:03d}' for i in range(n_assets)]
        index = pd.MultiIndex.from_product([dates, asset_ids], names=['date', 'symbol'])
        
        feature_data = pd.DataFrame(features, index=index)
        
        # Generate price data for return calculation
        price_data = pd.DataFrame({
            'close': np.random.lognormal(4, 0.5, len(index))  # Log-normal prices
        }, index=index)
        
        # Add some realistic price movements
        for asset in asset_ids:
            asset_mask = index.get_level_values('symbol') == asset
            asset_prices = price_data.loc[asset_mask, 'close'].values
            
            # Apply random walk
            returns = np.random.normal(0, 0.02, len(asset_prices))
            returns[0] = 0  # First return is zero
            cumulative_returns = np.cumsum(returns)
            price_data.loc[asset_mask, 'close'] = asset_prices[0] * np.exp(cumulative_returns)
        
        return {
            'feature_data': feature_data,
            'price_data': price_data,
            'dates': dates,
            'assets': asset_ids
        }
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        
        # Default configuration
        pipeline = DomainValidationPipeline()
        assert pipeline.config is not None
        assert pipeline.compliance_validator is not None
        assert pipeline.intuition_scorer is not None
        assert pipeline.business_logic_validator is not None
        assert pipeline.ic_tester is not None
        
        # Custom configuration
        config = DomainValidationConfig(
            final_feature_count=50,
            min_feature_count=10
        )
        pipeline = DomainValidationPipeline(config)
        assert pipeline.config.final_feature_count == 50
        assert pipeline.config.min_feature_count == 10
    
    def test_complete_validation_workflow(self, sample_data, temp_output_dir):
        """Test complete validation workflow."""
        
        # Setup
        config = DomainValidationConfig(
            final_feature_count=5,  # Small number for testing
            min_feature_count=2,
            output_dir=temp_output_dir,
            save_intermediate_results=True,
            integrate_with_lightgbm=False  # Disable for simpler testing
        )
        
        pipeline = DomainValidationPipeline(config)
        
        # Get sample data
        feature_data = sample_data['feature_data']
        price_data = sample_data['price_data']
        
        # Reset index for easier handling
        feature_data_reset = feature_data.reset_index().set_index('date')
        price_data_reset = price_data.reset_index().set_index('date')
        
        # Run validation
        features = list(feature_data.columns)
        results = pipeline.validate_and_select_features(
            features=features,
            feature_data=feature_data_reset,
            price_data=price_data_reset
        )
        
        # Verify results structure
        assert isinstance(results.input_features, list)
        assert isinstance(results.final_selected_features, list)
        assert len(results.final_selected_features) <= config.final_feature_count
        assert len(results.final_selected_features) >= 1  # Should select at least one
        
        # Check that all components ran
        assert results.compliance_results is not None
        assert results.intuition_results is not None
        assert results.business_logic_results is not None
        assert results.ic_test_results is not None
        
        # Check final scores
        assert results.final_scores is not None
        assert not results.final_scores.empty
        assert 'composite_score' in results.final_scores.columns
        
        # Check validation statistics
        assert results.validation_statistics is not None
        assert 'input_feature_count' in results.validation_statistics
        assert 'final_feature_count' in results.validation_statistics
        
        # Check that files were saved
        output_path = Path(temp_output_dir)
        assert any(f.name.startswith('domain_selected_features_') for f in output_path.glob('*.json'))
        assert any(f.name.startswith('domain_feature_scores_') for f in output_path.glob('*.csv'))
    
    def test_component_integration(self, sample_data):
        """Test individual component integration."""
        
        feature_data = sample_data['feature_data'].reset_index().set_index('date')
        price_data = sample_data['price_data'].reset_index().set_index('date')
        features = list(feature_data.columns)[:5]  # Test subset
        
        pipeline = DomainValidationPipeline()
        
        # Test Taiwan compliance validator
        compliance_results = pipeline.compliance_validator.validate_features(
            features, feature_data
        )
        assert isinstance(compliance_results, dict)
        assert len(compliance_results) == len(features)
        
        # Test economic intuition scorer
        intuition_results = pipeline.intuition_scorer.score_features(
            features, feature_data
        )
        assert isinstance(intuition_results, dict)
        assert len(intuition_results) == len(features)
        
        # Test business logic validator
        business_results = pipeline.business_logic_validator.validate_features(
            features, feature_data
        )
        assert isinstance(business_results, dict)
        assert len(business_results) == len(features)
        
        # Test IC performance tester
        ic_results = pipeline.ic_tester.test_features(
            features, feature_data, price_data
        )
        assert isinstance(ic_results, dict)
        assert len(ic_results) <= len(features)  # Some features might not have enough data
    
    def test_feature_filtering(self, sample_data):
        """Test that problematic features are filtered out."""
        
        feature_data = sample_data['feature_data'].reset_index().set_index('date')
        price_data = sample_data['price_data'].reset_index().set_index('date')
        
        config = DomainValidationConfig(
            final_feature_count=8,
            integrate_with_lightgbm=False
        )
        
        pipeline = DomainValidationPipeline(config)
        
        features = list(feature_data.columns)
        results = pipeline.validate_and_select_features(
            features=features,
            feature_data=feature_data,
            price_data=price_data
        )
        
        # Check that problematic features are not selected
        selected = results.final_selected_features
        
        # Constant feature should be filtered out
        assert 'constant_feature' not in selected
        
        # Future return (look-ahead bias) should be filtered out
        assert 'future_return' not in selected
        
        # Should prefer features with economic rationale
        economic_features = ['pe_ratio', 'pb_ratio', 'roe', 'rsi_14', 'ma_20']
        selected_economic = [f for f in selected if f in economic_features]
        assert len(selected_economic) > 0
    
    def test_scoring_system(self, sample_data):
        """Test composite scoring system."""
        
        feature_data = sample_data['feature_data'].reset_index().set_index('date')
        price_data = sample_data['price_data'].reset_index().set_index('date')
        
        config = DomainValidationConfig(
            scoring_weights={
                'compliance_score': 0.2,
                'intuition_score': 0.3,
                'business_logic_score': 0.2,
                'ic_performance_score': 0.3
            }
        )
        
        pipeline = DomainValidationPipeline(config)
        
        features = ['pe_ratio', 'rsi_14', 'volume_sma_10']
        results = pipeline.validate_and_select_features(
            features=features,
            feature_data=feature_data,
            price_data=price_data
        )
        
        # Check scoring components
        scores = results.final_scores
        assert not scores.empty
        
        required_columns = [
            'feature', 'compliance_score', 'intuition_score',
            'business_logic_score', 'ic_performance_score', 'composite_score'
        ]
        
        for col in required_columns:
            assert col in scores.columns
        
        # Check that composite scores are calculated correctly
        for _, row in scores.iterrows():
            expected_composite = (
                config.scoring_weights['compliance_score'] * row['compliance_score'] +
                config.scoring_weights['intuition_score'] * row['intuition_score'] +
                config.scoring_weights['business_logic_score'] * row['business_logic_score'] +
                config.scoring_weights['ic_performance_score'] * row['ic_performance_score']
            )
            
            # Allow small floating point differences
            assert abs(row['composite_score'] - expected_composite) < 1e-6
    
    def test_edge_cases(self, temp_output_dir):
        """Test edge cases and error handling."""
        
        config = DomainValidationConfig(output_dir=temp_output_dir)
        pipeline = DomainValidationPipeline(config)
        
        # Test with empty feature list
        with pytest.raises((ValueError, IndexError)):
            pipeline.validate_and_select_features(
                features=[],
                feature_data=pd.DataFrame(),
                price_data=pd.DataFrame()
            )
        
        # Test with single feature
        dates = pd.date_range('2023-01-01', periods=100)
        single_feature_data = pd.DataFrame({
            'test_feature': np.random.normal(0, 1, 100)
        }, index=dates)
        
        price_data = pd.DataFrame({
            'close': np.random.lognormal(4, 0.1, 100)
        }, index=dates)
        
        results = pipeline.validate_and_select_features(
            features=['test_feature'],
            feature_data=single_feature_data,
            price_data=price_data
        )
        
        assert len(results.final_selected_features) <= 1
        assert results.execution_time > 0
    
    def test_get_methods(self, sample_data):
        """Test getter methods."""
        
        feature_data = sample_data['feature_data'].reset_index().set_index('date')
        price_data = sample_data['price_data'].reset_index().set_index('date')
        
        pipeline = DomainValidationPipeline()
        
        # Test before validation (should raise errors)
        with pytest.raises(ValueError):
            pipeline.get_selected_features()
        
        with pytest.raises(ValueError):
            pipeline.get_feature_scores()
        
        with pytest.raises(ValueError):
            pipeline.get_validation_summary()
        
        # Run validation
        features = ['pe_ratio', 'rsi_14', 'volume_sma_10']
        results = pipeline.validate_and_select_features(
            features=features,
            feature_data=feature_data,
            price_data=price_data
        )
        
        # Test after validation
        selected_features = pipeline.get_selected_features()
        assert isinstance(selected_features, list)
        assert len(selected_features) <= len(features)
        
        scores = pipeline.get_feature_scores()
        assert isinstance(scores, pd.DataFrame)
        assert not scores.empty
        
        summary = pipeline.get_validation_summary()
        assert isinstance(summary, dict)
        assert 'input_feature_count' in summary
        
        # Test comprehensive report
        report = pipeline.generate_comprehensive_report()
        assert isinstance(report, pd.DataFrame)
        assert not report.empty
        
        # Test pipeline status
        status = pipeline.get_pipeline_status()
        assert isinstance(status, dict)
        assert 'configuration' in status
        assert 'components' in status
    
    def test_configuration_options(self, sample_data):
        """Test different configuration options."""
        
        feature_data = sample_data['feature_data'].reset_index().set_index('date')
        price_data = sample_data['price_data'].reset_index().set_index('date')
        features = list(feature_data.columns)[:6]
        
        # Test with minimal features
        config_minimal = DomainValidationConfig(
            final_feature_count=2,
            min_feature_count=1
        )
        pipeline_minimal = DomainValidationPipeline(config_minimal)
        
        results_minimal = pipeline_minimal.validate_and_select_features(
            features=features,
            feature_data=feature_data,
            price_data=price_data
        )
        
        assert len(results_minimal.final_selected_features) <= 2
        assert len(results_minimal.final_selected_features) >= 1
        
        # Test with different scoring weights
        config_ic_focused = DomainValidationConfig(
            scoring_weights={
                'compliance_score': 0.1,
                'intuition_score': 0.1, 
                'business_logic_score': 0.1,
                'ic_performance_score': 0.7  # Focus on IC performance
            }
        )
        pipeline_ic_focused = DomainValidationPipeline(config_ic_focused)
        
        results_ic_focused = pipeline_ic_focused.validate_and_select_features(
            features=features,
            feature_data=feature_data,
            price_data=price_data
        )
        
        # Should have different selections due to different weights
        assert isinstance(results_ic_focused.final_selected_features, list)
        
        # Test memory efficient mode
        config_efficient = DomainValidationConfig(
            memory_efficient=True,
            parallel_validation=False
        )
        pipeline_efficient = DomainValidationPipeline(config_efficient)
        
        results_efficient = pipeline_efficient.validate_and_select_features(
            features=features,
            feature_data=feature_data,
            price_data=price_data
        )
        
        assert isinstance(results_efficient.final_selected_features, list)