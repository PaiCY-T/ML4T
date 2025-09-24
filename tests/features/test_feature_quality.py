"""
Test Suite for Feature Quality Validation - Task #28 Stream C

Comprehensive tests for feature quality assessment, selection algorithms,
and Taiwan market compliance validation.

Test Coverage:
- Feature selection algorithms (correlation, univariate, model-based)
- Quality metrics calculation (statistical, financial, compliance)
- Taiwan compliance validation (T+2, price limits, trading hours)
- Integration with OpenFE pipeline
- Performance and memory validation
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
import tempfile
import os

# Import modules to test
from src.features.feature_selection import FeatureSelector, create_feature_selector
from src.features.quality_metrics import FeatureQualityMetrics, create_quality_metrics_calculator
from src.features.taiwan_compliance import TaiwanComplianceValidator, create_taiwan_compliance_validator
from src.features.taiwan_config import TaiwanMarketConfig


class TestFeatureSelection:
    """Test suite for FeatureSelector class."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature matrix for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        # Create dates and stocks for panel data
        dates = pd.date_range('2022-01-01', periods=200, freq='B')  # Business days
        stocks = [f'stock_{i:03d}' for i in range(5)]
        
        # Create MultiIndex
        index_tuples = [(date, stock) for date in dates for stock in stocks]
        index = pd.MultiIndex.from_tuples(index_tuples, names=['date', 'stock'])
        
        # Generate features with different characteristics
        features_data = {}
        
        # Good features (normal distribution, low correlation)
        for i in range(20):
            features_data[f'good_feature_{i}'] = np.random.normal(0, 1, n_samples)
            
        # Highly correlated features
        base_corr = np.random.normal(0, 1, n_samples)
        for i in range(5):
            features_data[f'corr_feature_{i}'] = base_corr + np.random.normal(0, 0.1, n_samples)
            
        # Low variance features
        for i in range(5):
            features_data[f'low_var_feature_{i}'] = np.random.normal(0, 0.001, n_samples)
            
        # Features with outliers
        for i in range(5):
            feature = np.random.normal(0, 1, n_samples)
            # Add some outliers
            outlier_indices = np.random.choice(n_samples, size=10, replace=False)
            feature[outlier_indices] = np.random.choice([-10, 10], size=10)
            features_data[f'outlier_feature_{i}'] = feature
            
        # Features with missing data
        for i in range(5):
            feature = np.random.normal(0, 1, n_samples)
            # Add missing values
            missing_indices = np.random.choice(n_samples, size=50, replace=False)
            feature[missing_indices] = np.nan
            features_data[f'missing_feature_{i}'] = feature
            
        # Non-compliant features (Taiwan market violations)
        features_data['realtime_price'] = np.random.normal(100, 10, n_samples)  # T+2 violation
        features_data['overnight_volume'] = np.random.normal(1000, 100, n_samples)  # Trading hours violation
        features_data['insider_flow'] = np.random.normal(0, 1, n_samples)  # Prohibited data
        
        # Create DataFrame
        features_df = pd.DataFrame(features_data, index=index[:n_samples])
        
        return features_df
    
    @pytest.fixture
    def sample_target(self):
        """Create sample target variable."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create target with some signal
        target = np.random.normal(0, 1, n_samples)
        
        # Create dates and stocks for panel data
        dates = pd.date_range('2022-01-01', periods=200, freq='B')
        stocks = [f'stock_{i:03d}' for i in range(5)]
        
        index_tuples = [(date, stock) for date in dates for stock in stocks]
        index = pd.MultiIndex.from_tuples(index_tuples, names=['date', 'stock'])
        
        return pd.Series(target, index=index[:n_samples], name='target')
    
    def test_feature_selector_initialization(self):
        """Test FeatureSelector initialization."""
        # Test default initialization
        selector = FeatureSelector()
        assert selector.target_feature_count == 350
        assert selector.correlation_threshold == 0.95
        assert selector.variance_threshold == 0.01
        assert not selector.is_fitted_
        
        # Test custom initialization
        selector = FeatureSelector(
            target_feature_count=200,
            correlation_threshold=0.9,
            variance_threshold=0.05
        )
        assert selector.target_feature_count == 200
        assert selector.correlation_threshold == 0.9
        assert selector.variance_threshold == 0.05
    
    def test_variance_filtering(self, sample_features):
        """Test low variance feature removal."""
        selector = FeatureSelector(variance_threshold=0.01)
        
        X_filtered, removed = selector.remove_low_variance_features(sample_features)
        
        # Should remove low variance features
        assert len(removed) >= 5  # At least the 5 low variance features
        assert len(X_filtered.columns) < len(sample_features.columns)
        
        # Check that low variance features are actually removed
        low_var_features = [col for col in sample_features.columns if 'low_var' in col]
        for feature in low_var_features:
            assert feature in removed
    
    def test_correlation_filtering(self, sample_features):
        """Test highly correlated feature removal."""
        selector = FeatureSelector(correlation_threshold=0.95)
        
        X_filtered, removed = selector.remove_highly_correlated_features(sample_features)
        
        # Should remove some correlated features
        assert len(removed) >= 3  # Should remove most correlated features
        assert len(X_filtered.columns) < len(sample_features.columns)
        
        # Verify correlation matrix of remaining features
        remaining_corr = X_filtered.corr().abs()
        
        # Check that no remaining feature pairs exceed threshold
        n_features = len(remaining_corr.columns)
        high_corr_count = 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if remaining_corr.iloc[i, j] >= 0.95:
                    high_corr_count += 1
        
        assert high_corr_count == 0  # No high correlations should remain
    
    def test_taiwan_compliance_filtering(self, sample_features):
        """Test Taiwan market compliance filtering."""
        selector = FeatureSelector()
        
        X_filtered, removed = selector.apply_taiwan_market_filters(sample_features)
        
        # Should remove non-compliant features
        non_compliant_features = ['realtime_price', 'overnight_volume', 'insider_flow']
        for feature in non_compliant_features:
            assert feature in removed
        
        # Compliant features should remain
        assert len(X_filtered.columns) > 0
        for feature in X_filtered.columns:
            assert feature not in non_compliant_features
    
    def test_univariate_selection(self, sample_features, sample_target):
        """Test univariate feature selection."""
        selector = FeatureSelector()
        
        X_selected, selected_features, scores = selector.select_features_univariate(
            sample_features, sample_target, k=20, task_type='regression'
        )
        
        assert len(selected_features) == 20
        assert len(scores) == 20
        assert X_selected.shape[1] == 20
        
        # Check that scores are reasonable
        score_values = list(scores.values())
        assert all(score >= 0 for score in score_values)  # F-scores should be non-negative
    
    def test_model_based_selection(self, sample_features, sample_target):
        """Test model-based feature selection."""
        selector = FeatureSelector()
        
        X_selected, selected_features, importances = selector.select_features_model_based(
            sample_features, sample_target, model_type='random_forest', max_features=15
        )
        
        assert len(selected_features) == 15
        assert len(importances) == 15
        assert X_selected.shape[1] == 15
        
        # Check that importances are reasonable
        importance_values = list(importances.values())
        assert all(imp >= 0 for imp in importance_values)  # Importances should be non-negative
    
    def test_full_selection_pipeline(self, sample_features, sample_target):
        """Test complete feature selection pipeline."""
        selector = FeatureSelector(
            target_feature_count=25,
            min_feature_count=20,
            max_feature_count=30
        )
        
        # Fit selector
        selector.fit(sample_features, sample_target)
        
        assert selector.is_fitted_
        assert 20 <= len(selector.selected_features_) <= 30
        assert selector.target_feature_count == 25
        
        # Transform data
        X_transformed = selector.transform(sample_features)
        assert X_transformed.shape[1] == len(selector.selected_features_)
        
        # Test fit_transform
        X_fit_transformed = selector.fit_transform(sample_features, sample_target)
        pd.testing.assert_frame_equal(X_transformed, X_fit_transformed)
    
    def test_selection_summary(self, sample_features, sample_target):
        """Test selection summary generation."""
        selector = FeatureSelector(target_feature_count=20)
        selector.fit(sample_features, sample_target)
        
        summary = selector.get_selection_summary()
        
        assert 'final_selected_features' in summary
        assert 'selection_stages' in summary
        assert 'feature_scores' in summary
        assert 'elimination_reasons' in summary
        
        assert summary['final_selected_features'] == len(selector.selected_features_)
        assert summary['target_achieved'] == (20 <= len(selector.selected_features_) <= 500)
    
    def test_factory_function(self):
        """Test feature selector factory function."""
        selector = create_feature_selector(target_features=300)
        
        assert isinstance(selector, FeatureSelector)
        assert selector.target_feature_count == 300
        assert 180 <= selector.min_feature_count <= 200  # 60% of target
        assert 420 <= selector.max_feature_count <= 450  # 140% of target


class TestFeatureQualityMetrics:
    """Test suite for FeatureQualityMetrics class."""
    
    @pytest.fixture
    def sample_feature_data(self):
        """Create sample feature data for testing."""
        np.random.seed(42)
        
        # Create time series data with MultiIndex
        dates = pd.date_range('2022-01-01', periods=252, freq='B')  # 1 year business days
        stocks = ['2330', '2454', '3008']  # Taiwan stock codes
        
        index_tuples = [(date, stock) for date in dates for stock in stocks]
        index = pd.MultiIndex.from_tuples(index_tuples, names=['date', 'stock'])
        
        n_samples = len(index_tuples)
        
        # Normal feature
        normal_data = np.random.normal(0, 1, n_samples)
        
        # Feature with outliers
        outlier_data = np.random.normal(0, 1, n_samples)
        outlier_indices = np.random.choice(n_samples, size=10, replace=False)
        outlier_data[outlier_indices] = np.random.choice([-5, 5], size=10)
        
        # Feature with missing values
        missing_data = np.random.normal(0, 1, n_samples)
        missing_indices = np.random.choice(n_samples, size=20, replace=False)
        missing_data[missing_indices] = np.nan
        
        # Autocorrelated feature
        autocorr_data = np.zeros(n_samples)
        autocorr_data[0] = np.random.normal()
        for i in range(1, n_samples):
            autocorr_data[i] = 0.5 * autocorr_data[i-1] + np.random.normal(0, 0.8)
        
        return {
            'normal_feature': pd.Series(normal_data, index=index),
            'outlier_feature': pd.Series(outlier_data, index=index),
            'missing_feature': pd.Series(missing_data, index=index),
            'autocorr_feature': pd.Series(autocorr_data, index=index)
        }
    
    @pytest.fixture
    def sample_target_data(self):
        """Create sample target data."""
        np.random.seed(42)
        
        dates = pd.date_range('2022-01-01', periods=252, freq='B')
        stocks = ['2330', '2454', '3008']
        
        index_tuples = [(date, stock) for date in dates for stock in stocks]
        index = pd.MultiIndex.from_tuples(index_tuples, names=['date', 'stock'])
        
        n_samples = len(index_tuples)
        target_data = np.random.normal(0, 0.02, n_samples)  # Return-like data
        
        return pd.Series(target_data, index=index, name='returns')
    
    def test_quality_metrics_initialization(self):
        """Test FeatureQualityMetrics initialization."""
        # Default initialization
        calculator = FeatureQualityMetrics()
        assert calculator.significance_level == 0.05
        assert calculator.outlier_threshold == 3.0
        assert calculator.min_observations == 252
        
        # Custom initialization
        calculator = FeatureQualityMetrics(
            significance_level=0.01,
            outlier_threshold=2.5,
            min_observations=100
        )
        assert calculator.significance_level == 0.01
        assert calculator.outlier_threshold == 2.5
        assert calculator.min_observations == 100
    
    def test_basic_statistics(self, sample_feature_data):
        """Test basic statistics calculation."""
        calculator = FeatureQualityMetrics()
        
        stats = calculator.calculate_basic_statistics(sample_feature_data['normal_feature'])
        
        # Check required fields
        required_fields = ['count', 'valid_count', 'missing_count', 'missing_pct',
                          'mean', 'median', 'std', 'var', 'min', 'max',
                          'skewness', 'kurtosis', 'unique_values']
        
        for field in required_fields:
            assert field in stats
            
        # Check reasonable values for normal data
        assert stats['missing_pct'] < 1  # Should be very little missing data
        assert abs(stats['skewness']) < 2  # Normal data shouldn't be too skewed
        assert stats['kurtosis'] > -2  # Reasonable kurtosis range
    
    def test_normality_testing(self, sample_feature_data):
        """Test normality testing."""
        calculator = FeatureQualityMetrics()
        
        # Test with normal data
        normal_results = calculator.test_normality(sample_feature_data['normal_feature'])
        
        # Should contain at least one normality test
        assert len(normal_results) > 0
        if 'shapiro_wilk' in normal_results:
            assert 'statistic' in normal_results['shapiro_wilk']
            assert 'p_value' in normal_results['shapiro_wilk']
            assert 'is_normal' in normal_results['shapiro_wilk']
    
    def test_stationarity_testing(self, sample_feature_data):
        """Test stationarity testing."""
        calculator = FeatureQualityMetrics()
        
        # Test ADF on normal data
        stationarity_results = calculator.test_stationarity(sample_feature_data['normal_feature'])
        
        if 'adf' in stationarity_results:
            assert 'statistic' in stationarity_results['adf']
            assert 'p_value' in stationarity_results['adf']
            assert 'is_stationary' in stationarity_results['adf']
            assert 'critical_values' in stationarity_results['adf']
    
    def test_outlier_detection(self, sample_feature_data):
        """Test outlier detection."""
        calculator = FeatureQualityMetrics()
        
        # Test with feature containing outliers
        outlier_results = calculator.detect_outliers(sample_feature_data['outlier_feature'])
        
        # Should detect outliers using multiple methods
        assert 'z_score' in outlier_results
        assert 'iqr' in outlier_results
        
        z_results = outlier_results['z_score']
        assert z_results['outlier_count'] > 0  # Should detect the artificial outliers
        assert z_results['outlier_pct'] > 0
    
    def test_autocorrelation_analysis(self, sample_feature_data):
        """Test autocorrelation analysis."""
        calculator = FeatureQualityMetrics()
        
        # Test with autocorrelated data
        autocorr_results = calculator.calculate_autocorrelation(
            sample_feature_data['autocorr_feature']
        )
        
        assert 'autocorrelations' in autocorr_results
        assert 'summary' in autocorr_results
        
        summary = autocorr_results['summary']
        assert 'lag1_autocorr' in summary
        assert 'max_autocorr' in summary
        
        # The artificially autocorrelated feature should show high lag1 correlation
        assert abs(summary['lag1_autocorr']) > 0.3
    
    def test_financial_properties(self, sample_feature_data):
        """Test financial properties assessment."""
        calculator = FeatureQualityMetrics()
        
        financial_results = calculator.assess_financial_properties(
            sample_feature_data['normal_feature']
        )
        
        if 'volatility' in financial_results:
            vol_results = financial_results['volatility']
            assert 'daily_volatility' in vol_results
            assert 'annualized_volatility' in vol_results
            assert vol_results['daily_volatility'] > 0
            assert vol_results['annualized_volatility'] > 0
    
    def test_taiwan_compliance_validation(self, sample_feature_data):
        """Test Taiwan compliance validation."""
        calculator = FeatureQualityMetrics()
        
        # Test compliant feature
        compliant_results = calculator.validate_taiwan_compliance(
            sample_feature_data['normal_feature'],
            'volume_lag2'
        )
        
        assert 'compliant' in compliant_results
        assert 'violations' in compliant_results
        assert 'warnings' in compliant_results
        assert 'checks_performed' in compliant_results
        
        # Test non-compliant feature name
        non_compliant_results = calculator.validate_taiwan_compliance(
            sample_feature_data['normal_feature'],
            'realtime_price'  # Should violate T+2 rule
        )
        
        assert not non_compliant_results['compliant']
        assert len(non_compliant_results['violations']) > 0
    
    def test_information_content(self, sample_feature_data, sample_target_data):
        """Test information content calculation."""
        calculator = FeatureQualityMetrics()
        
        info_results = calculator.calculate_information_content(
            sample_feature_data['normal_feature'],
            sample_target_data
        )
        
        if 'entropy' in info_results:
            entropy_results = info_results['entropy']
            assert 'entropy_bits' in entropy_results
            assert 'normalized_entropy' in entropy_results
            assert 0 <= entropy_results['normalized_entropy'] <= 1
        
        if 'predictive_power' in info_results:
            pred_results = info_results['predictive_power']
            assert 'pearson_correlation' in pred_results
            assert 'spearman_correlation' in pred_results
    
    def test_comprehensive_quality_validation(self, sample_feature_data, sample_target_data):
        """Test comprehensive feature quality validation."""
        calculator = FeatureQualityMetrics()
        
        # Test with normal feature
        quality_results = calculator.validate_feature_quality(
            sample_feature_data['normal_feature'],
            feature_name='good_volume_lag2',
            target_data=sample_target_data,
            comprehensive=True
        )
        
        # Check required fields
        required_fields = ['feature_name', 'overall_quality_score', 'quality_issues',
                          'taiwan_compliant', 'recommendation']
        
        for field in required_fields:
            assert field in quality_results
            
        assert 0 <= quality_results['overall_quality_score'] <= 100
        assert quality_results['recommendation'] in ['accept', 'conditional_accept', 'reject']
    
    def test_batch_validation(self, sample_feature_data, sample_target_data):
        """Test batch feature validation."""
        calculator = FeatureQualityMetrics()
        
        # Create DataFrame from sample data
        features_df = pd.DataFrame(sample_feature_data)
        
        batch_results = calculator.batch_validate_features(
            features_df,
            target_data=sample_target_data,
            comprehensive=False  # Faster for testing
        )
        
        assert len(batch_results) == len(features_df.columns)
        
        for feature_name in features_df.columns:
            assert feature_name in batch_results
            assert 'overall_quality_score' in batch_results[feature_name]
    
    def test_quality_report_generation(self, sample_feature_data, sample_target_data):
        """Test quality report generation."""
        calculator = FeatureQualityMetrics()
        
        features_df = pd.DataFrame(sample_feature_data)
        batch_results = calculator.batch_validate_features(features_df, sample_target_data)
        
        # Generate report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / 'quality_report.json'
            report = calculator.generate_quality_report(batch_results, str(report_path))
            
            # Check report structure
            assert 'summary' in report
            assert 'recommendations' in report
            assert 'quality_statistics' in report
            
            # Check file was created
            assert report_path.exists()
    
    def test_factory_function(self):
        """Test quality metrics factory function."""
        calculator = create_quality_metrics_calculator(significance_level=0.01)
        
        assert isinstance(calculator, FeatureQualityMetrics)
        assert calculator.significance_level == 0.01


class TestTaiwanComplianceValidator:
    """Test suite for TaiwanComplianceValidator class."""
    
    @pytest.fixture
    def sample_panel_data(self):
        """Create sample panel data for compliance testing."""
        # Create proper Taiwan trading dates
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='B')  # Business days
        taiwan_config = TaiwanMarketConfig()
        
        # Filter to actual trading days
        trading_dates = [d for d in dates if taiwan_config.is_trading_day(d)][:100]  # Use first 100
        stocks = ['2330', '2454', '3008', '2382', '2317']  # Taiwan stock codes
        
        index_tuples = [(date, stock) for date in trading_dates for stock in stocks]
        index = pd.MultiIndex.from_tuples(index_tuples, names=['date', 'stock'])
        
        n_samples = len(index_tuples)
        
        # Create features with different compliance characteristics
        return {
            'volume_lag2': pd.Series(np.random.lognormal(8, 1, n_samples), index=index),  # Compliant
            'price_close': pd.Series(np.random.normal(100, 10, n_samples), index=index),  # Compliant  
            'realtime_flow': pd.Series(np.random.normal(0, 1, n_samples), index=index),  # T+2 violation
            'overnight_volume': pd.Series(np.random.normal(1000, 100, n_samples), index=index),  # Hours violation
            'insider_data': pd.Series(np.random.normal(0, 1, n_samples), index=index)  # Prohibited data
        }
    
    def test_compliance_validator_initialization(self):
        """Test TaiwanComplianceValidator initialization."""
        # Default initialization
        validator = TaiwanComplianceValidator()
        assert validator.strict_mode == True
        assert validator.allow_warnings == True
        
        # Custom initialization
        validator = TaiwanComplianceValidator(strict_mode=False, allow_warnings=False)
        assert validator.strict_mode == False
        assert validator.allow_warnings == False
    
    def test_feature_naming_compliance(self, sample_panel_data):
        """Test feature naming compliance checking."""
        validator = TaiwanComplianceValidator()
        
        # Test compliant feature name
        compliant_results = validator._check_feature_naming_compliance('volume_lag2')
        assert compliant_results['compliant']
        assert len(compliant_results['violations']) == 0
        
        # Test non-compliant feature names
        prohibited_results = validator._check_feature_naming_compliance('insider_data')
        assert not prohibited_results['compliant']
        assert len(prohibited_results['violations']) > 0
        
        realtime_results = validator._check_feature_naming_compliance('realtime_flow')
        assert not realtime_results['compliant']
        
        overnight_results = validator._check_feature_naming_compliance('overnight_volume')
        assert not overnight_results['compliant']
    
    def test_temporal_alignment_validation(self, sample_panel_data):
        """Test temporal alignment with Taiwan trading calendar."""
        validator = TaiwanComplianceValidator()
        
        # Test with proper trading day data
        temporal_results = validator._validate_temporal_alignment(
            sample_panel_data['volume_lag2']
        )
        
        assert 'compliant' in temporal_results
        assert 'statistics' in temporal_results
        assert temporal_results['statistics']['trading_dates'] > 0
        
        # The sample data should have good temporal alignment
        if temporal_results['statistics']['total_dates'] > 0:
            trading_ratio = (temporal_results['statistics']['trading_dates'] / 
                           temporal_results['statistics']['total_dates'])
            assert trading_ratio > 0.8  # Most dates should be trading days
    
    def test_settlement_compliance_validation(self, sample_panel_data):
        """Test T+2 settlement compliance validation."""
        validator = TaiwanComplianceValidator()
        
        # Test compliant feature (has lag indication)
        settlement_results = validator._validate_settlement_compliance(
            sample_panel_data['volume_lag2'],
            'volume_lag2'
        )
        
        assert settlement_results['compliant']
        assert settlement_results['settlement_analysis']['is_settlement_sensitive']
        assert settlement_results['settlement_analysis']['has_lag_indication']
        
        # Test non-compliant feature (realtime, no lag)
        non_compliant_results = validator._validate_settlement_compliance(
            sample_panel_data['realtime_flow'],
            'realtime_flow'
        )
        
        assert not non_compliant_results['compliant']
        assert len(non_compliant_results['violations']) > 0
    
    def test_price_limit_compliance_validation(self, sample_panel_data):
        """Test price limit compliance validation."""
        validator = TaiwanComplianceValidator()
        
        # Test price feature
        price_results = validator._validate_price_limit_compliance(
            sample_panel_data['price_close'],
            'price_close'
        )
        
        assert 'price_analysis' in price_results
        assert price_results['price_analysis']['is_price_feature']
        
        # Create feature with extreme price movements to test limits
        extreme_returns = pd.Series([0.15, -0.15, 0.20, -0.20], 
                                  index=sample_panel_data['price_close'].index[:4])
        
        extreme_results = validator._validate_price_limit_compliance(
            extreme_returns,
            'extreme_returns'
        )
        
        # Should detect extreme movements
        if 'price_analysis' in extreme_results and 'extreme_moves_count' in extreme_results['price_analysis']:
            assert extreme_results['price_analysis']['extreme_moves_count'] > 0
    
    def test_cross_sectional_consistency_validation(self, sample_panel_data):
        """Test cross-sectional consistency validation."""
        validator = TaiwanComplianceValidator()
        
        cross_results = validator._validate_cross_sectional_consistency(
            sample_panel_data['volume_lag2']
        )
        
        assert 'cross_sectional_analysis' in cross_results
        assert cross_results['cross_sectional_analysis']['is_panel_data']
        assert cross_results['cross_sectional_analysis']['unique_dates'] > 0
        assert cross_results['cross_sectional_analysis']['unique_stocks'] > 0
    
    def test_comprehensive_compliance_validation(self, sample_panel_data):
        """Test comprehensive compliance validation."""
        validator = TaiwanComplianceValidator(strict_mode=True)
        
        # Test compliant feature
        compliant_results = validator.validate_feature_compliance(
            sample_panel_data['volume_lag2'],
            'volume_lag2',
            comprehensive=True
        )
        
        assert 'overall_compliant' in compliant_results
        assert 'compliance_score' in compliant_results
        assert 'recommendations' in compliant_results
        assert 0 <= compliant_results['compliance_score'] <= 100
        
        # Test non-compliant feature
        non_compliant_results = validator.validate_feature_compliance(
            sample_panel_data['realtime_flow'],
            'realtime_flow',
            comprehensive=True
        )
        
        assert not non_compliant_results['overall_compliant']
        assert len(non_compliant_results['critical_violations']) > 0
        assert non_compliant_results['compliance_score'] < 100
    
    def test_batch_compliance_validation(self, sample_panel_data):
        """Test batch compliance validation."""
        validator = TaiwanComplianceValidator()
        
        features_df = pd.DataFrame(sample_panel_data)
        
        batch_results = validator.batch_validate_compliance(
            features_df,
            comprehensive=False  # Faster for testing
        )
        
        assert len(batch_results) == len(features_df.columns)
        
        # Check that compliant and non-compliant features are identified correctly
        assert batch_results['volume_lag2']['overall_compliant']
        assert not batch_results['realtime_flow']['overall_compliant']
        assert not batch_results['insider_data']['overall_compliant']
    
    def test_compliance_report_generation(self, sample_panel_data):
        """Test compliance report generation."""
        validator = TaiwanComplianceValidator()
        
        features_df = pd.DataFrame(sample_panel_data)
        batch_results = validator.batch_validate_compliance(features_df)
        
        # Generate report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / 'compliance_report.json'
            report = validator.generate_compliance_report(batch_results, str(report_path))
            
            # Check report structure
            assert 'summary' in report
            assert 'compliance_statistics' in report
            assert 'recommendations' in report
            
            # Check statistics
            stats = report['compliance_statistics']
            assert 'compliant_features' in stats
            assert 'compliant_percentage' in stats
            assert 'total_violations' in stats
            
            # Check file was created
            assert report_path.exists()
    
    def test_compliance_summary(self):
        """Test compliance summary generation."""
        validator = TaiwanComplianceValidator()
        
        summary = validator.get_compliance_summary()
        
        assert 'taiwan_market_config' in summary
        assert 'compliance_rules' in summary
        assert 'validator_config' in summary
        
        # Check Taiwan config details
        taiwan_config = summary['taiwan_market_config']
        assert 'settlement_days' in taiwan_config
        assert 'trading_hours' in taiwan_config
        assert 'price_limit_percent' in taiwan_config
    
    def test_factory_function(self):
        """Test compliance validator factory function."""
        validator = create_taiwan_compliance_validator(strict_mode=False)
        
        assert isinstance(validator, TaiwanComplianceValidator)
        assert not validator.strict_mode


class TestIntegration:
    """Integration tests for feature quality validation system."""
    
    @pytest.fixture
    def integration_data(self):
        """Create realistic integration test data."""
        np.random.seed(42)
        
        # Create 6 months of Taiwan trading data
        start_date = date(2023, 1, 1)
        end_date = date(2023, 6, 30)
        
        taiwan_config = TaiwanMarketConfig()
        all_dates = pd.date_range(start_date, end_date, freq='B')
        trading_dates = [d for d in all_dates if taiwan_config.is_trading_day(d)]
        
        # Taiwan stock universe (sample)
        stocks = ['2330', '2454', '3008', '2382', '2317', '2412', '1301', '1303', '2002', '2882']
        
        # Create MultiIndex
        index_tuples = [(date, stock) for date in trading_dates for stock in stocks]
        index = pd.MultiIndex.from_tuples(index_tuples, names=['date', 'stock'])
        
        n_samples = len(index_tuples)
        
        # Create realistic financial features
        features = {}
        
        # Price-related features (compliant)
        features['close_price'] = pd.Series(
            np.random.lognormal(4, 0.3, n_samples),  # Prices around 50-150 TWD
            index=index
        )
        
        features['returns_1d'] = pd.Series(
            np.random.normal(0, 0.02, n_samples),  # Daily returns ~2% vol
            index=index
        )
        
        # Volume features (compliant with T+2)
        features['volume_lag2'] = pd.Series(
            np.random.lognormal(10, 1, n_samples),  # Volume with proper lag
            index=index
        )
        
        features['turnover_lag2'] = pd.Series(
            np.random.lognormal(8, 0.8, n_samples),  # Turnover with proper lag
            index=index
        )
        
        # Technical indicators (compliant)
        features['sma_20'] = pd.Series(
            np.random.lognormal(4, 0.25, n_samples),  # Simple moving average
            index=index
        )
        
        features['volatility_20d'] = pd.Series(
            np.random.gamma(2, 0.01, n_samples),  # 20-day volatility
            index=index
        )
        
        # Non-compliant features (should be filtered out)
        features['realtime_price'] = pd.Series(
            np.random.lognormal(4, 0.3, n_samples),  # T+2 violation
            index=index
        )
        
        features['overnight_volume'] = pd.Series(
            np.random.lognormal(9, 1, n_samples),  # Trading hours violation
            index=index
        )
        
        # Low quality features (should be filtered out)
        features['constant_feature'] = pd.Series(
            np.ones(n_samples) * 1.5,  # No variance
            index=index
        )
        
        features['mostly_missing'] = pd.Series(
            np.random.normal(0, 1, n_samples),
            index=index
        )
        # Make 80% missing
        missing_indices = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
        features['mostly_missing'].iloc[missing_indices] = np.nan
        
        # Create correlated features
        base_feature = np.random.normal(0, 1, n_samples)
        for i in range(3):
            features[f'corr_feature_{i}'] = pd.Series(
                base_feature + np.random.normal(0, 0.05, n_samples),  # Highly correlated
                index=index
            )
            
        # Create target variable (stock returns)
        target = pd.Series(
            np.random.normal(0, 0.015, n_samples),  # Daily returns
            index=index,
            name='forward_returns'
        )
        
        return pd.DataFrame(features), target
    
    def test_end_to_end_feature_selection_pipeline(self, integration_data):
        """Test complete end-to-end feature selection pipeline."""
        features_df, target = integration_data
        
        logger.info(f"Starting with {len(features_df.columns)} features")
        
        # Create feature selector with Taiwan settings
        selector = create_feature_selector(
            target_features=8,  # Target 8 features from ~15 input features
            taiwan_config=TaiwanMarketConfig()
        )
        
        # Run complete selection pipeline
        selected_features_df = selector.fit_transform(features_df, target)
        
        # Validate results
        assert len(selected_features_df.columns) >= 6  # At least 6 features
        assert len(selected_features_df.columns) <= 10  # At most 10 features
        
        # Check that non-compliant features were removed
        non_compliant = ['realtime_price', 'overnight_volume']
        for feature in non_compliant:
            assert feature not in selected_features_df.columns
            
        # Check that low-quality features were removed
        low_quality = ['constant_feature', 'mostly_missing']
        for feature in low_quality:
            assert feature not in selected_features_df.columns
            
        # Check that some correlated features were removed
        corr_features = [col for col in features_df.columns if 'corr_feature' in col]
        remaining_corr = [col for col in selected_features_df.columns if 'corr_feature' in col]
        assert len(remaining_corr) < len(corr_features)  # Some should be removed
        
        # Get selection summary
        summary = selector.get_selection_summary()
        assert summary['target_achieved']  # Should meet target constraints
        
        logger.info(
            f"Selection completed: {len(features_df.columns)} -> "
            f"{len(selected_features_df.columns)} features"
        )
    
    def test_quality_and_compliance_integration(self, integration_data):
        """Test integration of quality metrics and compliance validation."""
        features_df, target = integration_data
        
        # Create quality calculator and compliance validator
        quality_calc = create_quality_metrics_calculator()
        compliance_validator = create_taiwan_compliance_validator(strict_mode=True)
        
        # Run quality assessment
        quality_results = quality_calc.batch_validate_features(
            features_df,
            target_data=target,
            comprehensive=False  # Faster for testing
        )
        
        # Run compliance validation
        compliance_results = compliance_validator.batch_validate_compliance(
            features_df,
            comprehensive=False
        )
        
        # Analyze results
        high_quality_features = []
        compliant_features = []
        
        for feature_name in features_df.columns:
            # Check quality
            quality = quality_results[feature_name]
            if (quality['overall_quality_score'] >= 70 and 
                quality['recommendation'] in ['accept', 'conditional_accept']):
                high_quality_features.append(feature_name)
                
            # Check compliance
            compliance = compliance_results[feature_name]
            if compliance['overall_compliant'] and compliance['compliance_score'] >= 80:
                compliant_features.append(feature_name)
        
        # There should be overlap between high quality and compliant features
        good_features = set(high_quality_features) & set(compliant_features)
        assert len(good_features) >= 4  # Should have several good features
        
        # Known bad features should be filtered out
        bad_features = ['realtime_price', 'overnight_volume', 'constant_feature', 'mostly_missing']
        for bad_feature in bad_features:
            if bad_feature in quality_results:
                # Should have low quality or non-compliant
                quality_score = quality_results[bad_feature]['overall_quality_score']
                compliance_score = compliance_results[bad_feature]['compliance_score']
                assert quality_score < 80 or compliance_score < 80
    
    def test_memory_and_performance_validation(self, integration_data):
        """Test memory usage and performance characteristics."""
        features_df, target = integration_data
        
        # Test with memory monitoring
        selector = FeatureSelector(memory_limit_gb=2.0)  # Restrictive limit for testing
        
        start_time = datetime.now()
        
        # This should complete without exceeding memory limits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore memory warnings for test
            selected_df = selector.fit_transform(features_df, target)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time
        assert processing_time < 60  # Less than 1 minute for test data
        
        # Check memory usage tracking
        memory_usage = selector.get_memory_usage()
        assert len(memory_usage) > 0  # Should have recorded memory usage
        
        # Result should be valid
        assert not selected_df.empty
        assert len(selected_df.columns) > 0


# Performance benchmarks and stress tests
class TestPerformanceAndStress:
    """Performance and stress tests for feature selection system."""
    
    @pytest.mark.slow
    def test_large_scale_feature_selection(self):
        """Test feature selection with large number of features."""
        np.random.seed(42)
        
        # Create large feature set (similar to OpenFE output)
        n_samples = 5000  # 5000 observations
        n_features = 1000  # 1000 features (realistic OpenFE output)
        
        # Create random features
        features_data = np.random.normal(0, 1, (n_samples, n_features))
        feature_names = [f'openfe_feature_{i:04d}' for i in range(n_features)]
        
        # Create simple index
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='B')
        features_df = pd.DataFrame(features_data, columns=feature_names, index=dates)
        
        # Create target
        target = pd.Series(np.random.normal(0, 1, n_samples), index=dates)
        
        # Test feature selection performance
        start_time = datetime.now()
        
        selector = FeatureSelector(
            target_feature_count=300,  # Reduce to 300 features
            memory_limit_gb=8.0
        )
        
        selected_df = selector.fit_transform(features_df, target)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time (< 5 minutes for 1000 features)
        assert processing_time < 300
        
        # Should achieve target reduction
        assert 250 <= len(selected_df.columns) <= 350  # Within reasonable range of target
        
        logger.info(
            f"Large scale test: {n_features} -> {len(selected_df.columns)} features "
            f"in {processing_time:.1f}s"
        )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])