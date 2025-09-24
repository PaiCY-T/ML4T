"""
Tests for ML Feature Selection Pipeline.

Integration tests for the complete ML-based feature selection framework.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import json
import pickle

from src.feature_selection.ml_based.ml_selection_pipeline import (
    MLFeatureSelectionPipeline,
    MLSelectionConfig
)


class TestMLSelectionConfig:
    """Test configuration class for ML selection pipeline."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MLSelectionConfig()
        
        assert config.selection_strategy == 'comprehensive'
        assert config.target_features == 100
        assert config.min_features == 20
        assert config.max_features == 200
        assert config.ic_threshold == 0.05
        assert config.stability_threshold == 0.7
        assert config.validation_folds == 5
        
        # Check method weights
        assert config.method_weights['importance_ranking'] == 0.3
        assert config.method_weights['rfe'] == 0.25
        assert config.method_weights['forward_backward'] == 0.25
        assert config.method_weights['stability_analysis'] == 0.2
        
        # Check Taiwan market parameters
        assert config.taiwan_market_weight == 0.15
        assert config.sector_balance == True
        assert config.regime_awareness == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_weights = {
            'importance_ranking': 0.4,
            'rfe': 0.3,
            'forward_backward': 0.2,
            'stability_analysis': 0.1
        }
        
        config = MLSelectionConfig(
            selection_strategy='fast',
            target_features=50,
            method_weights=custom_weights,
            sector_balance=False
        )
        
        assert config.selection_strategy == 'fast'
        assert config.target_features == 50
        assert config.method_weights == custom_weights
        assert config.sector_balance == False


class TestMLFeatureSelectionPipeline:
    """Test ML feature selection pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data for testing."""
        np.random.seed(42)
        n_days = 500
        n_stocks = 20
        n_features = 100
        
        # Create realistic feature names
        feature_names = []
        
        # Technical indicators
        for indicator in ['ma', 'rsi', 'macd', 'bb']:
            for window in [5, 10, 20, 50]:
                feature_names.append(f'{indicator}_{window}')
        
        # Fundamental features
        fundamentals = ['pe', 'pb', 'roe', 'eps', 'revenue', 'debt', 'margin']
        feature_names.extend(fundamentals)
        
        # Market features
        market_features = ['beta', 'alpha', 'correlation', 'market_cap', 'volume']
        feature_names.extend(market_features)
        
        # Momentum features
        for window in [1, 5, 10, 20, 60]:
            feature_names.append(f'momentum_{window}')
        
        # Volatility features
        for window in [5, 10, 20, 60]:
            feature_names.append(f'volatility_{window}')
        
        # Sector features
        for sector in ['tech', 'finance', 'manufacturing']:
            feature_names.append(f'sector_{sector}')
        
        # Fill remaining with generic features
        while len(feature_names) < n_features:
            feature_names.append(f'feature_{len(feature_names)}')
        
        # Create date and symbol index
        dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
        symbols = [f'TSE_{i:04d}' for i in range(1101, 1101 + n_stocks)]
        
        index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=['date', 'symbol']
        )
        
        # Create feature matrix with realistic correlations
        np.random.seed(42)
        base_data = np.random.randn(len(index), n_features)
        
        # Add feature correlations and signal
        X = pd.DataFrame(base_data, index=index, columns=feature_names)
        
        # Create correlated technical features
        for i, col in enumerate([c for c in X.columns if 'ma_' in c]):
            if i > 0:
                prev_col = [c for c in X.columns if 'ma_' in c][i-1]
                X[col] = 0.8 * X[prev_col] + 0.2 * X[col]
        
        # Create target with meaningful signal
        signal_features = ['ma_20', 'momentum_20', 'rsi_14', 'pe', 'roe']
        signal_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        target_signal = np.zeros(len(X))
        for feat, weight in zip(signal_features, signal_weights):
            if feat in X.columns:
                target_signal += weight * X[feat].values
        
        # Add regime-dependent effects
        regime_threshold = np.percentile(X['volatility_20'], 75)
        high_vol_mask = X['volatility_20'] > regime_threshold
        target_signal[high_vol_mask] *= 0.5  # Reduced signal in high vol periods
        
        y = pd.Series(
            target_signal + np.random.randn(len(X)) * 0.1,
            index=X.index,
            name='forward_returns'
        )
        
        # Create market data for regime detection
        market_data = pd.DataFrame({
            'market_return': np.random.randn(n_days) * 0.02,
            'market_volatility': np.abs(np.random.randn(n_days) * 0.01),
        }, index=dates)
        
        # Create dates series
        dates_series = pd.Series(
            np.repeat(dates, n_stocks),
            index=X.index,
            name='date'
        )
        
        return X, y, market_data, dates_series
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pipeline_config(self, temp_dir):
        """Create pipeline configuration for testing."""
        # Faster configuration for testing
        config = MLSelectionConfig(
            selection_strategy='comprehensive',
            target_features=20,  # Smaller for testing
            min_features=5,
            max_features=50,
            validation_folds=3,  # Fewer folds for speed
            output_dir=temp_dir,
            save_intermediate_results=True
        )
        
        # Configure components for speed
        config.importance_config.cv_folds = 3
        config.importance_config.n_estimators = 10
        config.importance_config.use_shap = False
        
        config.rfe_config.cv_folds = 3
        config.rfe_config.n_estimators = 10
        config.rfe_config.step = 0.2  # Remove 20% per step
        
        config.fb_config.cv_folds = 3
        config.fb_config.n_estimators = 10
        config.fb_config.patience = 2
        
        config.stability_config.time_windows = [126]  # Only 6 months
        config.stability_config.n_estimators = 10
        
        return config
    
    @pytest.fixture
    def pipeline(self, pipeline_config):
        """Create pipeline for testing."""
        return MLFeatureSelectionPipeline(pipeline_config)
    
    def test_initialization(self, pipeline_config):
        """Test pipeline initialization."""
        pipeline = MLFeatureSelectionPipeline(pipeline_config)
        
        assert pipeline.config == pipeline_config
        assert pipeline.importance_ranker is not None
        assert pipeline.rfe_selector is not None
        assert pipeline.fb_selector is not None
        assert pipeline.stability_analyzer is not None
        
        assert pipeline.selected_features_ == []
        assert pipeline.feature_scores_ is None
        assert pipeline.method_results_ == {}
    
    def test_comprehensive_selection(self, pipeline, sample_data):
        """Test comprehensive selection strategy."""
        X, y, market_data, dates = sample_data
        
        # Run comprehensive selection
        selected_features = pipeline.fit_select(
            X, y, market_data=market_data, dates=dates
        )
        
        # Validate results
        assert isinstance(selected_features, list)
        assert len(selected_features) > 0
        assert len(selected_features) <= pipeline.config.target_features
        
        # Check that all selected features exist in original data
        assert all(f in X.columns for f in selected_features)
        
        # Check that pipeline state is updated
        assert pipeline.selected_features_ == selected_features
        assert pipeline.feature_scores_ is not None
        assert len(pipeline.method_results_) > 0
        assert pipeline.final_validation_ is not None
        
        # Check method results
        assert 'importance' in pipeline.method_results_
        assert 'rfe' in pipeline.method_results_
        assert 'forward_backward' in pipeline.method_results_
        assert 'stability' in pipeline.method_results_
        
        # Check validation results
        validation = pipeline.final_validation_
        assert 'mean_ic' in validation
        assert 'std_ic' in validation
        assert 'selected_feature_count' in validation
        assert validation['selected_feature_count'] == len(selected_features)
    
    def test_fast_selection(self, pipeline_config, sample_data):
        """Test fast selection strategy."""
        X, y, market_data, dates = sample_data
        
        # Configure for fast selection
        pipeline_config.selection_strategy = 'fast'
        pipeline = MLFeatureSelectionPipeline(pipeline_config)
        
        # Run fast selection
        selected_features = pipeline.fit_select(X, y)
        
        # Validate results
        assert isinstance(selected_features, list)
        assert len(selected_features) > 0
        assert len(selected_features) <= pipeline.config.target_features
        
        # Fast selection should complete quickly
        assert pipeline.selected_features_ == selected_features
    
    def test_stability_focused_selection(self, pipeline_config, sample_data):
        """Test stability-focused selection strategy."""
        X, y, market_data, dates = sample_data
        
        # Configure for stability-focused selection
        pipeline_config.selection_strategy = 'stability_focused'
        pipeline = MLFeatureSelectionPipeline(pipeline_config)
        
        # Run stability-focused selection
        selected_features = pipeline.fit_select(
            X, y, market_data=market_data, dates=dates
        )
        
        # Validate results
        assert isinstance(selected_features, list)
        assert len(selected_features) > 0
        assert len(selected_features) <= pipeline.config.target_features
    
    def test_ensemble_selection(self, pipeline, sample_data):
        """Test ensemble selection logic."""
        X, y, market_data, dates = sample_data
        
        # Run selection to populate method results
        pipeline.fit_select(X, y, market_data=market_data, dates=dates)
        
        # Test ensemble selection with mock method results
        candidate_features = X.columns[:50].tolist()  # Use first 50 features
        
        # Mock method results
        pipeline.method_results_ = {
            'importance': {
                'selected_features': candidate_features[:30],
                'rankings': pd.DataFrame({
                    'feature': candidate_features[:30],
                    'composite_score': np.linspace(0.9, 0.1, 30)
                })
            },
            'rfe': {
                'selected_features': candidate_features[:25]
            },
            'forward_backward': {
                'selected_features': candidate_features[:20]
            },
            'stability': {
                'selected_features': candidate_features[:15],
                'stability_scores': pd.DataFrame(
                    {'composite_stability': np.random.rand(15)},
                    index=candidate_features[:15]
                )
            }
        }
        
        # Test ensemble selection
        final_features = pipeline._ensemble_selection(candidate_features)
        
        assert isinstance(final_features, list)
        assert len(final_features) <= pipeline.config.target_features
        assert pipeline.feature_scores_ is not None
    
    def test_sector_balance(self, pipeline, sample_data):
        """Test sector balance functionality."""
        X, y, market_data, dates = sample_data
        
        # Enable sector balance
        pipeline.config.sector_balance = True
        
        # Create mock sorted features with sector info
        sorted_features = [(f, 0.8 - i*0.01) for i, f in enumerate(X.columns[:50])]
        
        # Test sector balance application
        balanced_features = pipeline._apply_sector_balance(sorted_features)
        
        assert isinstance(balanced_features, list)
        assert len(balanced_features) <= pipeline.config.target_features
    
    def test_final_validation(self, pipeline, sample_data):
        """Test final validation functionality."""
        X, y, market_data, dates = sample_data
        
        # Select subset of features
        selected_features = X.columns[:20].tolist()
        X_selected = X[selected_features]
        
        # Run final validation
        pipeline._final_validation(X_selected, y, sample_weight=None)
        
        # Check validation results
        assert pipeline.final_validation_ is not None
        validation = pipeline.final_validation_
        
        assert 'mean_ic' in validation
        assert 'std_ic' in validation
        assert 'mean_rmse' in validation
        assert 'cv_folds' in validation
        assert 'selected_feature_count' in validation
        assert 'ic_threshold_met' in validation
        
        assert validation['cv_folds'] == pipeline.config.validation_folds
        assert validation['selected_feature_count'] == len(selected_features)
        assert isinstance(validation['ic_threshold_met'], bool)
    
    @patch('src.feature_selection.ml_based.ml_selection_pipeline.LightGBMAlphaModel')
    def test_lightgbm_integration(self, mock_lgb_model, pipeline, sample_data):
        """Test LightGBM integration functionality."""
        X, y, market_data, dates = sample_data
        
        # Mock LightGBM model
        mock_model_instance = Mock()
        mock_model_instance.prepare_training_data.return_value = (X, y)
        mock_model_instance.train.return_value = {'best_score': 0.85}
        mock_model_instance.get_feature_importance.return_value = pd.DataFrame({
            'feature': X.columns[:10],
            'importance': np.random.rand(10)
        })
        mock_lgb_model.return_value = mock_model_instance
        
        # Enable LightGBM integration
        pipeline.config.integrate_with_lgb = True
        
        # Select subset of features
        X_selected = X[X.columns[:20]]
        
        # Run LightGBM integration
        pipeline._integrate_with_lightgbm(X_selected, y)
        
        # Check integration results
        assert pipeline.lgb_integration_ is not None
        integration = pipeline.lgb_integration_
        
        assert 'integration_successful' in integration
        assert 'training_stats' in integration
        assert 'feature_importance' in integration
        
        # Verify model was called correctly
        mock_model_instance.prepare_training_data.assert_called_once()
        mock_model_instance.train.assert_called_once()
        mock_model_instance.get_feature_importance.assert_called_once()
    
    def test_save_results(self, pipeline, sample_data, temp_dir):
        """Test results saving functionality."""
        X, y, market_data, dates = sample_data
        
        # Configure to save results
        pipeline.config.save_intermediate_results = True
        pipeline.config.output_dir = temp_dir
        
        # Run selection to generate results
        pipeline.fit_select(X, y)
        
        # Check that files were created
        output_path = Path(temp_dir)
        files = list(output_path.glob('*'))
        
        assert len(files) > 0
        
        # Check for expected file types
        json_files = list(output_path.glob('*.json'))
        csv_files = list(output_path.glob('*.csv'))
        pkl_files = list(output_path.glob('*.pkl'))
        
        assert len(json_files) >= 1  # At least selected_features
        assert len(csv_files) >= 1   # Feature scores
        assert len(pkl_files) >= 1   # Method results
        
        # Verify file contents
        selected_features_file = [f for f in json_files if 'selected_features' in f.name][0]
        with open(selected_features_file, 'r') as f:
            saved_data = json.load(f)
        
        assert 'selected_features' in saved_data
        assert 'config' in saved_data
        assert 'timestamp' in saved_data
    
    def test_transform(self, pipeline, sample_data):
        """Test data transformation functionality."""
        X, y, market_data, dates = sample_data
        
        # Run selection first
        selected_features = pipeline.fit_select(X, y)
        
        # Test transformation
        X_transformed = pipeline.transform(X)
        
        assert isinstance(X_transformed, pd.DataFrame)
        assert list(X_transformed.columns) == selected_features
        assert len(X_transformed) == len(X)
        assert X_transformed.index.equals(X.index)
    
    def test_transform_before_selection(self, pipeline, sample_data):
        """Test that transform raises error before selection."""
        X, y, market_data, dates = sample_data
        
        with pytest.raises(ValueError, match="No features selected"):
            pipeline.transform(X)
    
    def test_transform_missing_features(self, pipeline, sample_data):
        """Test transform with missing features."""
        X, y, market_data, dates = sample_data
        
        # Run selection
        pipeline.fit_select(X, y)
        
        # Create data missing some features
        X_missing = X.drop(columns=[pipeline.selected_features_[0]])
        
        with pytest.raises(ValueError, match="Missing features in input data"):
            pipeline.transform(X_missing)
    
    def test_get_methods(self, pipeline, sample_data):
        """Test getter methods."""
        X, y, market_data, dates = sample_data
        
        # Test before selection
        assert pipeline.get_selected_features() == []
        
        with pytest.raises(ValueError):
            pipeline.get_feature_scores()
        
        with pytest.raises(ValueError):
            pipeline.get_validation_results()
        
        # Run selection
        pipeline.fit_select(X, y)
        
        # Test after selection
        selected_features = pipeline.get_selected_features()
        assert isinstance(selected_features, list)
        assert len(selected_features) > 0
        
        feature_scores = pipeline.get_feature_scores()
        assert isinstance(feature_scores, pd.DataFrame)
        
        method_results = pipeline.get_method_results()
        assert isinstance(method_results, dict)
        
        validation_results = pipeline.get_validation_results()
        assert isinstance(validation_results, dict)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_selection_summary(self, mock_show, pipeline, sample_data):
        """Test selection summary plotting."""
        X, y, market_data, dates = sample_data
        
        # Run selection first
        pipeline.fit_select(X, y)
        
        # Test plotting
        pipeline.plot_selection_summary()
        
        # Verify plot was created
        mock_show.assert_called_once()
    
    def test_get_summary_stats(self, pipeline, sample_data):
        """Test summary statistics."""
        X, y, market_data, dates = sample_data
        
        # Run selection
        pipeline.fit_select(X, y)
        
        # Get summary statistics
        summary = pipeline.get_summary_stats()
        
        assert isinstance(summary, dict)
        assert 'pipeline_config' in summary
        assert 'selection_results' in summary
        assert 'validation_performance' in summary
        assert 'method_contributions' in summary
        
        # Check config section
        config_section = summary['pipeline_config']
        assert 'strategy' in config_section
        assert 'target_features' in config_section
        assert 'ic_threshold' in config_section
        
        # Check selection results
        results_section = summary['selection_results']
        assert 'selected_features' in results_section
        assert 'feature_list' in results_section
        assert 'method_weights' in results_section
        
        # Check method contributions
        contributions = summary['method_contributions']
        assert len(contributions) > 0
        for method, contrib in contributions.items():
            assert 'features_selected' in contrib
            assert 'method_score' in contrib
    
    def test_error_handling(self, pipeline):
        """Test error handling and edge cases."""
        # Test with invalid strategy
        with pytest.raises(ValueError, match="Unknown selection strategy"):
            pipeline.config.selection_strategy = 'invalid_strategy'
            X = pd.DataFrame({'feature_1': [1, 2, 3]})
            y = pd.Series([0.1, 0.2, 0.3])
            pipeline.fit_select(X, y)
        
        # Test with insufficient data
        X_small = pd.DataFrame({'feature_1': [1, 2]})
        y_small = pd.Series([0.1, 0.2])
        
        pipeline.config.selection_strategy = 'fast'
        
        # Should handle small dataset gracefully
        try:
            selected_features = pipeline.fit_select(X_small, y_small)
            # If it succeeds, check reasonable output
            assert isinstance(selected_features, list)
        except (ValueError, IndexError):
            # Expected for very small datasets
            pass


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_taiwan_market(self, temp_dir):
        """Test complete end-to-end Taiwan market scenario."""
        # Create realistic Taiwan market dataset
        np.random.seed(42)
        n_days = 756  # 3 years
        n_stocks = 100  # TSE stocks
        
        # Taiwan stock codes
        symbols = [f'TSE_{i:04d}' for i in range(1101, 1101 + n_stocks)]
        dates = pd.date_range('2021-01-01', periods=n_days, freq='D')
        
        index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=['date', 'symbol']
        )
        
        # Create comprehensive feature set
        feature_names = []
        
        # Technical indicators
        for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'stoch']:
            for window in [5, 10, 20, 60]:
                feature_names.append(f'{indicator}_{window}')
        
        # Volume indicators
        for indicator in ['volume_sma', 'volume_ratio', 'vwap']:
            for window in [5, 10, 20]:
                feature_names.append(f'{indicator}_{window}')
        
        # Fundamental ratios
        fundamentals = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa', 'eps_growth',
                       'revenue_growth', 'debt_equity', 'current_ratio', 'gross_margin']
        feature_names.extend(fundamentals)
        
        # Market microstructure
        microstructure = ['bid_ask_spread', 'order_imbalance', 'tick_size_effect',
                         'intraday_volatility', 'closing_auction_effect']
        feature_names.extend(microstructure)
        
        # Taiwan specific features
        taiwan_specific = ['taiwan_election_effect', 'cny_exchange_rate', 'us_taiwan_relation',
                          'semiconductor_cycle', 'export_orders', 'manufacturing_pmi']
        feature_names.extend(taiwan_specific)
        
        # Sector features
        sectors = ['technology', 'finance', 'manufacturing', 'consumer', 'materials',
                  'energy', 'healthcare', 'utilities', 'telecom']
        for sector in sectors:
            feature_names.extend([f'{sector}_beta', f'{sector}_momentum'])
        
        # Macro features
        macro_features = ['interest_rate', 'inflation_rate', 'gdp_growth', 'unemployment',
                         'money_supply', 'foreign_reserves', 'trade_balance']
        feature_names.extend(macro_features)
        
        # Ensure we have exactly the number we want
        while len(feature_names) < 200:
            feature_names.append(f'feature_{len(feature_names)}')
        feature_names = feature_names[:200]
        
        # Create feature matrix
        X = pd.DataFrame(
            np.random.randn(len(index), len(feature_names)),
            index=index,
            columns=feature_names
        )
        
        # Add realistic feature relationships
        # Technical indicator relationships
        X['sma_20'] = X['sma_10'].rolling(2).mean() + np.random.randn(len(X)) * 0.1
        X['ema_20'] = 0.9 * X['sma_20'] + 0.1 * X['ema_20'].shift(1).fillna(0)
        
        # Fundamental relationships
        X['eps_growth'] = X['revenue_growth'] * 0.8 + np.random.randn(len(X)) * 0.2
        X['roe'] = X['roa'] * (1 + X['debt_equity']) + np.random.randn(len(X)) * 0.1
        
        # Create realistic target variable
        # Taiwan market characteristics: momentum, mean reversion, sector effects
        signal_components = []
        
        # Momentum signal
        momentum_signal = (
            X['sma_20'] * 0.15 +
            X['rsi_14'] * 0.1 +
            X['macd_12'] * 0.1
        )
        signal_components.append(momentum_signal)
        
        # Fundamental signal
        fundamental_signal = (
            -X['pe_ratio'] * 0.08 +  # Lower PE is better
            X['roe'] * 0.12 +
            X['eps_growth'] * 0.1
        )
        signal_components.append(fundamental_signal)
        
        # Sector rotation signal
        sector_signal = (
            X['technology_momentum'] * 0.2 +  # Taiwan tech focus
            X['finance_beta'] * 0.05
        )
        signal_components.append(sector_signal)
        
        # Combine signals with regime-dependent weights
        base_signal = sum(signal_components)
        
        # Add market regime effects
        high_vol_periods = X['intraday_volatility'] > X['intraday_volatility'].quantile(0.8)
        regime_adjustment = np.where(high_vol_periods, 0.5, 1.0)  # Reduced signal in high vol
        
        # Final target
        y = pd.Series(
            base_signal * regime_adjustment + np.random.randn(len(X)) * 0.05,
            index=X.index,
            name='forward_returns'
        )
        
        # Create market data for regime detection
        market_data = pd.DataFrame({
            'tse_index': np.cumsum(np.random.randn(n_days) * 0.02),
            'volatility': np.abs(np.random.randn(n_days) * 0.01),
            'volume': np.random.lognormal(10, 0.5, n_days)
        }, index=dates)
        
        # Create dates series
        dates_series = pd.Series(
            np.repeat(dates, n_stocks),
            index=X.index,
            name='date'
        )
        
        # Configure pipeline for Taiwan market
        config = MLSelectionConfig(
            selection_strategy='comprehensive',
            target_features=50,  # Reasonable for Taiwan market
            min_features=20,
            ic_threshold=0.03,  # Lower threshold for emerging market
            stability_threshold=0.65,  # Slightly lower for Taiwan volatility
            taiwan_market_weight=0.2,  # Higher weight for Taiwan specifics
            sector_balance=True,
            regime_awareness=True,
            output_dir=temp_dir,
            save_intermediate_results=True
        )
        
        # Optimize for Taiwan market characteristics
        config.importance_config.target_horizon = 5  # 1-week typical
        config.importance_config.min_ic_threshold = 0.02
        config.rfe_config.step = 0.15  # More aggressive elimination
        config.fb_config.patience = 5  # More patience for Taiwan volatility
        config.stability_config.regime_detection_method = 'volatility'
        
        # Create pipeline
        pipeline = MLFeatureSelectionPipeline(config)
        
        # Run comprehensive selection
        selected_features = pipeline.fit_select(
            X, y,
            market_data=market_data,
            dates=dates_series,
            feature_groups={
                'technical': [f for f in feature_names if any(t in f for t in ['sma', 'ema', 'rsi', 'macd'])],
                'fundamental': [f for f in feature_names if f in fundamentals],
                'sector': [f for f in feature_names if any(s in f for s in sectors)],
                'taiwan_specific': taiwan_specific
            }
        )
        
        # Comprehensive validation
        assert isinstance(selected_features, list)
        assert 20 <= len(selected_features) <= 50
        
        # Check Taiwan market requirements
        validation = pipeline.get_validation_results()
        assert validation['mean_ic'] >= 0.02  # Taiwan market minimum
        assert validation['ic_threshold_met'] == True
        
        # Check feature diversity (sector balance)
        feature_categories = {
            'technical': sum(1 for f in selected_features if any(t in f for t in ['sma', 'ema', 'rsi', 'macd'])),
            'fundamental': sum(1 for f in selected_features if f in fundamentals),
            'taiwan_specific': sum(1 for f in selected_features if f in taiwan_specific)
        }
        
        # Should have representation from multiple categories
        assert feature_categories['technical'] >= 5  # Taiwan momentum focus
        assert feature_categories['fundamental'] >= 3  # Some fundamentals
        assert feature_categories['taiwan_specific'] >= 1  # Taiwan factors
        
        # Check method contributions
        method_results = pipeline.get_method_results()
        assert len(method_results) == 4  # All methods ran
        
        # Check that results were saved
        output_path = Path(temp_dir)
        saved_files = list(output_path.glob('*'))
        assert len(saved_files) >= 3  # Features, scores, results
        
        # Test transformation
        X_transformed = pipeline.transform(X)
        assert X_transformed.shape[1] == len(selected_features)
        assert X_transformed.shape[0] == X.shape[0]
        
        # Check feature importance makes sense for Taiwan market
        feature_scores = pipeline.get_feature_scores()
        top_features = feature_scores.head(10)['feature'].tolist()
        
        # Should include momentum and fundamental features for Taiwan
        momentum_count = sum(1 for f in top_features if any(m in f for m in ['sma', 'momentum', 'rsi']))
        assert momentum_count >= 3  # Taiwan market momentum importance
        
        # Get comprehensive summary
        summary = pipeline.get_summary_stats()
        
        # Validate summary completeness
        assert summary['pipeline_config']['strategy'] == 'comprehensive'
        assert summary['selection_results']['selected_features'] == len(selected_features)
        assert summary['validation_performance']['mean_ic'] >= 0.02
        
        # Check method contributions
        contributions = summary['method_contributions']
        assert len(contributions) == 4
        for method in ['importance', 'rfe', 'forward_backward', 'stability']:
            assert method in contributions
            assert contributions[method]['features_selected'] > 0
        
        print(f"✅ End-to-end Taiwan market test completed successfully")
        print(f"📊 Selected {len(selected_features)} features with IC = {validation['mean_ic']:.4f}")
        print(f"🏆 Top 5 features: {selected_features[:5]}")