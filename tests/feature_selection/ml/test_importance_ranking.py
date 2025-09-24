"""
Tests for LightGBM Importance Ranking module.

Test Suite for ML-based feature selection - Importance Ranking component.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

from src.feature_selection.ml_based.importance_ranking import (
    LightGBMImportanceRanker,
    ImportanceRankerConfig
)


class TestImportanceRankerConfig:
    """Test configuration class for importance ranker."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ImportanceRankerConfig()
        
        assert config.cv_folds == 5
        assert config.importance_types == ['gain', 'split']
        assert config.min_importance_threshold == 0.001
        assert config.stability_threshold == 0.7
        assert config.target_horizon == 5
        assert config.min_ic_threshold == 0.05
        assert config.n_estimators == 100
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = ImportanceRankerConfig(
            cv_folds=3,
            min_importance_threshold=0.01,
            use_shap=False
        )
        
        assert config.cv_folds == 3
        assert config.min_importance_threshold == 0.01
        assert config.use_shap == False


class TestLightGBMImportanceRanker:
    """Test LightGBM importance ranker."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Create feature matrix
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)],
            index=pd.MultiIndex.from_product([
                pd.date_range('2020-01-01', periods=n_samples//10, freq='D'),
                [f'stock_{i}' for i in range(10)]
            ], names=['date', 'symbol'])
        )
        
        # Create target with some signal
        y = pd.Series(
            X['feature_0'] * 0.5 + X['feature_1'] * 0.3 + np.random.randn(n_samples) * 0.1,
            index=X.index,
            name='forward_returns'
        )
        
        return X, y
    
    @pytest.fixture
    def ranker(self):
        """Create importance ranker for testing."""
        config = ImportanceRankerConfig(
            cv_folds=3,
            n_estimators=10,  # Small for fast testing
            use_shap=False  # Disable SHAP for speed
        )
        return LightGBMImportanceRanker(config)
    
    def test_initialization(self):
        """Test ranker initialization."""
        ranker = LightGBMImportanceRanker()
        
        assert ranker.config is not None
        assert ranker.feature_importance_ is None
        assert ranker.cv_results_ is None
        assert ranker.stability_scores_ is None
    
    def test_rank_features(self, ranker, sample_data):
        """Test feature ranking functionality."""
        X, y = sample_data
        
        # Rank features
        feature_rankings = ranker.rank_features(X, y)
        
        # Check results
        assert isinstance(feature_rankings, pd.DataFrame)
        assert len(feature_rankings) <= len(X.columns)
        assert 'feature' in feature_rankings.columns
        assert 'composite_score' in feature_rankings.columns
        assert 'importance_gain' in feature_rankings.columns
        assert 'stability_score' in feature_rankings.columns
        
        # Check that composite scores are valid
        assert all(feature_rankings['composite_score'] >= 0)
        assert all(feature_rankings['composite_score'] <= 1)
        
        # Check that features are sorted by composite score
        composite_scores = feature_rankings['composite_score'].values
        assert all(composite_scores[i] >= composite_scores[i+1] 
                  for i in range(len(composite_scores)-1))
    
    def test_cross_validation_importance(self, ranker, sample_data):
        """Test cross-validation importance calculation."""
        X, y = sample_data
        X_clean, y_clean = ranker._prepare_data(X, y)
        
        # Run cross-validation
        cv_importance = ranker._cross_validate_importance(X_clean, y_clean)
        
        # Check results
        assert isinstance(cv_importance, dict)
        assert 'gain' in cv_importance
        assert isinstance(cv_importance['gain'], list)
        assert len(cv_importance['gain']) == ranker.config.cv_folds
        
        # Check CV results stored
        assert ranker.cv_results_ is not None
        assert 'cv_scores' in ranker.cv_results_
        assert 'mean_ic' in ranker.cv_results_
    
    def test_stability_scores_calculation(self, ranker, sample_data):
        """Test stability score calculation."""
        X, y = sample_data
        X_clean, y_clean = ranker._prepare_data(X, y)
        
        # Mock CV importance data
        cv_importance = {'gain': []}
        for fold in range(3):
            fold_importance = pd.DataFrame({
                'feature': X_clean.columns,
                'importance': np.random.rand(len(X_clean.columns)),
                'fold': fold
            })
            cv_importance['gain'].append(fold_importance)
        
        # Calculate stability scores
        stability_scores = ranker._calculate_stability_scores(cv_importance)
        
        # Check results
        assert isinstance(stability_scores, pd.Series)
        assert len(stability_scores) == len(X_clean.columns)
        assert stability_scores.name == 'stability_score'
        assert all(stability_scores >= 0)
        assert all(stability_scores <= 1)
    
    def test_get_top_features(self, ranker, sample_data):
        """Test getting top features."""
        X, y = sample_data
        
        # Rank features first
        ranker.rank_features(X, y)
        
        # Get top features
        top_5 = ranker.get_top_features(5)
        assert isinstance(top_5, list)
        assert len(top_5) == 5
        
        # Get features with stability filter
        stable_features = ranker.get_top_features(10, min_stability=0.5)
        assert isinstance(stable_features, list)
        assert len(stable_features) <= 10
    
    def test_get_top_features_before_ranking(self, ranker):
        """Test that get_top_features raises error before ranking."""
        with pytest.raises(ValueError, match="Features have not been ranked yet"):
            ranker.get_top_features(5)
    
    def test_prepare_data(self, ranker, sample_data):
        """Test data preparation."""
        X, y = sample_data
        
        # Add some NaN values
        X_with_nan = X.copy()
        X_with_nan.iloc[0:10, 0] = np.nan
        y_with_nan = y.copy()
        y_with_nan.iloc[0:5] = np.nan
        
        # Prepare data
        X_clean, y_clean = ranker._prepare_data(X_with_nan, y_with_nan)
        
        # Check that NaN values are removed
        assert not X_clean.isnull().any().any()
        assert not y_clean.isnull().any()
        
        # Check that infinite values are handled
        X_with_inf = X.copy()
        X_with_inf.iloc[0:10, 0] = np.inf
        X_clean_inf, y_clean_inf = ranker._prepare_data(X_with_inf, y)
        
        assert not np.isinf(X_clean_inf.values).any()
    
    def test_sample_weights(self, ranker, sample_data):
        """Test ranking with sample weights."""
        X, y = sample_data
        
        # Create sample weights
        sample_weight = np.random.rand(len(X))
        
        # Rank features with sample weights
        feature_rankings = ranker.rank_features(X, y, sample_weight=sample_weight)
        
        # Check that ranking completed successfully
        assert isinstance(feature_rankings, pd.DataFrame)
        assert len(feature_rankings) > 0
    
    def test_market_regime_labels(self, ranker, sample_data):
        """Test ranking with market regime labels."""
        X, y = sample_data
        
        # Create regime labels
        regime_labels = pd.Series(
            np.random.choice(['bull', 'bear', 'neutral'], size=len(y)),
            index=y.index,
            name='regime'
        )
        
        # Rank features with regime labels
        feature_rankings = ranker.rank_features(X, y, market_regime_labels=regime_labels)
        
        # Check that ranking completed successfully
        assert isinstance(feature_rankings, pd.DataFrame)
        assert len(feature_rankings) > 0
        
        # Check that regime importance columns are present
        regime_columns = [col for col in feature_rankings.columns if 'regime_' in col]
        assert len(regime_columns) > 0
    
    @patch('matplotlib.pyplot.show')
    def test_plot_importance(self, mock_show, ranker, sample_data):
        """Test importance plotting."""
        X, y = sample_data
        
        # Rank features first
        ranker.rank_features(X, y)
        
        # Test plotting (should not raise errors)
        ranker.plot_importance(top_n=10)
        
        # Check that show was called
        mock_show.assert_called_once()
    
    def test_get_summary_stats(self, ranker, sample_data):
        """Test summary statistics."""
        X, y = sample_data
        
        # Test before ranking
        summary_before = ranker.get_summary_stats()
        assert summary_before["status"] == "No ranking performed"
        
        # Rank features
        ranker.rank_features(X, y)
        
        # Test after ranking
        summary_after = ranker.get_summary_stats()
        
        assert "total_features" in summary_after
        assert "significant_features" in summary_after
        assert "mean_stability" in summary_after
        assert "top_10_features" in summary_after
        assert "cv_performance" in summary_after
        assert "config" in summary_after
        
        assert isinstance(summary_after["top_10_features"], list)
        assert len(summary_after["top_10_features"]) <= 10
    
    def test_edge_cases(self, ranker):
        """Test edge cases and error handling."""
        # Empty data
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)
        
        with pytest.raises((ValueError, IndexError)):
            ranker.rank_features(X_empty, y_empty)
        
        # Single feature
        X_single = pd.DataFrame({'feature_0': [1, 2, 3, 4, 5]})
        y_single = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Should handle single feature case
        feature_rankings = ranker.rank_features(X_single, y_single)
        assert len(feature_rankings) == 1
    
    def test_memory_efficiency(self, sample_data):
        """Test memory efficiency features."""
        X, y = sample_data
        
        # Create ranker with memory optimization
        config = ImportanceRankerConfig(
            cv_folds=2,
            n_estimators=5,
            use_shap=False
        )
        ranker = LightGBMImportanceRanker(config)
        
        # Rank features
        feature_rankings = ranker.rank_features(X, y)
        
        # Check that ranking completed successfully
        assert isinstance(feature_rankings, pd.DataFrame)
        assert len(feature_rankings) > 0


class TestIntegration:
    """Integration tests for importance ranking."""
    
    def test_taiwan_market_scenario(self):
        """Test Taiwan market specific scenario."""
        # Create Taiwan market-like data
        np.random.seed(42)
        n_days = 252  # 1 year
        n_stocks = 50  # TSE stocks
        n_features = 30
        
        # Create realistic feature names
        feature_names = (
            [f'price_ma_{w}' for w in [5, 10, 20, 60]] +
            [f'volume_ma_{w}' for w in [5, 10, 20]] +
            [f'rsi_{w}' for w in [14, 21]] +
            ['pe_ratio', 'pb_ratio', 'roe', 'eps_growth'] +
            [f'momentum_{w}' for w in [1, 5, 10, 20]] +
            [f'volatility_{w}' for w in [5, 10, 20]] +
            ['market_cap', 'sector_beta', 'liquidity'] +
            ['taiwan_specific_1', 'taiwan_specific_2', 'taiwan_specific_3']
        )
        
        # Create date index
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
        symbols = [f'TSE_{i:04d}' for i in range(1101, 1101 + n_stocks)]  # Taiwan stock codes
        
        index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=['date', 'symbol']
        )
        
        # Create feature matrix
        X = pd.DataFrame(
            np.random.randn(len(index), len(feature_names)),
            index=index,
            columns=feature_names
        )
        
        # Create target with Taiwan market characteristics
        # - Higher volatility
        # - Sector concentration effects
        # - Regime changes
        base_return = (
            X['price_ma_20'] * 0.3 +
            X['momentum_20'] * 0.2 +
            X['volume_ma_20'] * 0.15 +
            np.random.randn(len(X)) * 0.02  # Higher noise for Taiwan market
        )
        
        y = pd.Series(base_return, index=X.index, name='forward_returns')
        
        # Create importance ranker with Taiwan-optimized config
        config = ImportanceRankerConfig(
            cv_folds=5,
            target_horizon=5,  # 5-day forward returns typical for Taiwan
            min_ic_threshold=0.03,  # Lower threshold for emerging market
            n_estimators=50,
            use_shap=False  # Skip SHAP for speed
        )
        
        ranker = LightGBMImportanceRanker(config)
        
        # Rank features
        feature_rankings = ranker.rank_features(X, y)
        
        # Validate Taiwan market specific results
        assert len(feature_rankings) > 0
        assert len(feature_rankings) <= len(feature_names)
        
        # Check that momentum and price features rank highly (typical for Taiwan)
        top_10_features = feature_rankings.head(10)['feature'].tolist()
        momentum_features = [f for f in top_10_features if 'momentum' in f or 'price_ma' in f]
        
        # Should have at least some momentum/price features in top 10
        assert len(momentum_features) >= 2
        
        # Check stability requirements
        stable_features = ranker.get_top_features(20, min_stability=0.6)
        assert len(stable_features) >= 10  # Should have at least 10 stable features
        
        # Check summary statistics
        summary = ranker.get_summary_stats()
        assert summary['cv_performance'] is not None
        assert summary['mean_stability'] > 0.5  # Reasonable stability threshold
        
        # Validate performance meets Taiwan market standards
        assert summary['cv_performance'] >= 0.02  # Minimum IC for Taiwan market