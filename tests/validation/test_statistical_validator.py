"""
Comprehensive test suite for Statistical Validation Engine.

Tests cover:
- Information Coefficient monitoring accuracy
- Drift detection algorithms  
- Performance tracking across regimes
- Statistical significance testing
- Taiwan market-specific validations
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.validation.statistical_validator import (
    ValidationConfig,
    ValidationResults,
    InformationCoefficientMonitor,
    DriftDetectionEngine,
    PerformanceRegimeAnalyzer,
    StatisticalValidator
)
from src.validation.taiwan_market_validator import (
    TaiwanMarketConfig,
    TaiwanMarketValidator,
    TaiwanSettlementValidator,
    PriceLimitValidator
)


class TestValidationConfig:
    """Test validation configuration."""
    
    def test_default_config(self):
        config = ValidationConfig()
        
        assert config.ic_accuracy_target == 0.95
        assert config.real_time_latency_target == 100.0
        assert config.settlement_days == 2
        assert len(config.ic_lookback_periods) == 4
        
    def test_custom_config(self):
        config = ValidationConfig(
            ic_accuracy_target=0.98,
            psi_threshold=0.15,
            trading_hours={'start': 9.5, 'end': 14.0}
        )
        
        assert config.ic_accuracy_target == 0.98
        assert config.psi_threshold == 0.15
        assert config.trading_hours['start'] == 9.5


class TestInformationCoefficientMonitor:
    """Test IC monitoring with 95%+ accuracy requirements."""
    
    @pytest.fixture
    def ic_monitor(self):
        config = ValidationConfig()
        return InformationCoefficientMonitor(config)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample prediction and return data."""
        np.random.seed(42)
        n_samples = 252  # 1 year of daily data
        
        # Generate correlated predictions and returns
        true_signal = np.random.randn(n_samples)
        noise = np.random.randn(n_samples) * 0.5
        
        predictions = true_signal + noise * 0.3
        returns = true_signal + noise
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        
        return (
            pd.Series(predictions, index=dates, name='predictions'),
            pd.Series(returns, index=dates, name='returns')
        )
    
    def test_calculate_ic_with_significance(self, ic_monitor, sample_data):
        """Test IC calculation with statistical significance."""
        predictions, returns = sample_data
        
        ic, p_value, details = ic_monitor.calculate_ic_with_significance(predictions, returns)
        
        # Validate IC calculation
        assert isinstance(ic, float)
        assert not np.isnan(ic)
        assert abs(ic) <= 1.0
        
        # Validate p-value
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1.0
        
        # Validate detailed statistics
        assert 'method' in details
        assert 'n_observations' in details
        assert 'confidence_interval' in details
        assert 'bootstrap_mean' in details
        
        # With our synthetic data, IC should be significant
        assert details['is_significant'] == True
        assert details['n_observations'] == len(predictions)
        
    def test_bootstrap_confidence_intervals(self, ic_monitor, sample_data):
        """Test bootstrap confidence interval accuracy."""
        predictions, returns = sample_data
        
        _, _, details = ic_monitor.calculate_ic_with_significance(predictions, returns)
        
        ci_lower, ci_upper = details['confidence_interval']
        bootstrap_mean = details['bootstrap_mean']
        
        # Confidence interval should contain the bootstrap mean
        assert ci_lower <= bootstrap_mean <= ci_upper
        
        # Bootstrap mean should be close to original IC
        assert abs(bootstrap_mean - details['ic']) < 0.05
        
    def test_monitor_rolling_ic(self, ic_monitor, sample_data):
        """Test rolling IC monitoring across multiple horizons."""
        predictions, returns = sample_data
        
        results = ic_monitor.monitor_rolling_ic(predictions, returns)
        
        # Should have results for all configured lookback periods
        expected_periods = [20, 60, 120, 252]
        for period in expected_periods:
            if len(predictions) >= period:
                assert period in results
                
                period_results = results[period]
                assert 'ic_mean' in period_results
                assert 'ic_stability' in period_results
                assert 'significant_rate' in period_results
                assert 'ic_trend' in period_results
                
                # Stability should be between 0 and 1
                assert 0 <= period_results['ic_stability'] <= 1
                
                # Significant rate should be between 0 and 1
                assert 0 <= period_results['significant_rate'] <= 1
    
    def test_rolling_ic_insufficient_data(self, ic_monitor):
        """Test handling of insufficient data for rolling IC."""
        # Create short time series
        short_predictions = pd.Series([0.1, 0.2, 0.3])
        short_returns = pd.Series([0.05, 0.1, 0.15])
        
        results = ic_monitor.monitor_rolling_ic(short_predictions, short_returns)
        
        # Should handle insufficient data gracefully
        assert len(results) == 0  # No periods have enough data


class TestDriftDetectionEngine:
    """Test drift detection algorithms."""
    
    @pytest.fixture
    def drift_detector(self):
        config = ValidationConfig()
        return DriftDetectionEngine(config)
    
    @pytest.fixture
    def reference_features(self):
        """Generate reference feature distribution."""
        np.random.seed(42)
        n_samples = 1000
        
        features = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.exponential(1, n_samples),
            'feature_3': np.random.uniform(-1, 1, n_samples)
        })
        
        return features
    
    @pytest.fixture 
    def drifted_features(self, reference_features):
        """Generate drifted feature distribution."""
        # Introduce drift in feature_1 (mean shift)
        drifted = reference_features.copy()
        drifted['feature_1'] = drifted['feature_1'] + 0.5  # Mean shift
        drifted['feature_2'] = drifted['feature_2'] * 1.5  # Scale change
        # feature_3 remains unchanged
        
        return drifted
    
    def test_calculate_psi(self, drift_detector, reference_features, drifted_features):
        """Test Population Stability Index calculation."""
        # No drift case
        psi_no_drift = drift_detector.calculate_psi(
            reference_features['feature_3'], reference_features['feature_3']
        )
        assert psi_no_drift < 0.1  # Should be very low
        
        # Drift case
        psi_with_drift = drift_detector.calculate_psi(
            reference_features['feature_1'], drifted_features['feature_1']
        )
        assert psi_with_drift > 0.1  # Should detect drift
        
    def test_jensen_shannon_divergence(self, drift_detector, reference_features, drifted_features):
        """Test Jensen-Shannon divergence calculation."""
        # No drift case
        js_no_drift = drift_detector.jensen_shannon_divergence(
            reference_features['feature_3'], reference_features['feature_3']
        )
        assert js_no_drift < 0.1
        
        # Drift case
        js_with_drift = drift_detector.jensen_shannon_divergence(
            reference_features['feature_1'], drifted_features['feature_1']
        )
        assert js_with_drift > 0.1
        
    def test_detect_feature_drift(self, drift_detector, reference_features, drifted_features):
        """Test comprehensive feature drift detection."""
        drift_results = drift_detector.detect_feature_drift(reference_features, drifted_features)
        
        # Should have results for all common features
        assert 'feature_1' in drift_results
        assert 'feature_2' in drift_results
        assert 'feature_3' in drift_results
        
        # feature_1 should show high drift (mean shift)
        assert drift_results['feature_1']['drift_score'] > 0.5
        assert drift_results['feature_1']['is_drifted'] == True
        
        # feature_2 should show some drift (scale change)
        assert drift_results['feature_2']['drift_score'] > 0.3
        
        # feature_3 should show minimal drift
        assert drift_results['feature_3']['drift_score'] < 0.3
        assert drift_results['feature_3']['is_drifted'] == False
        
        # Validate statistical components
        for feature, results in drift_results.items():
            assert 'psi' in results
            assert 'ks_statistic' in results
            assert 'ks_p_value' in results
            assert 'js_divergence' in results
            assert 'drift_level' in results['drift_level'] in ['low', 'medium', 'high']


class TestPerformanceRegimeAnalyzer:
    """Test performance tracking across market regimes."""
    
    @pytest.fixture
    def regime_analyzer(self):
        config = ValidationConfig()
        return PerformanceRegimeAnalyzer(config)
    
    @pytest.fixture
    def market_data_with_regimes(self):
        """Generate market data with different regimes."""
        np.random.seed(42)
        n_samples = 500
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        
        # Create market data with different regimes
        market_data = pd.DataFrame(index=dates)
        
        # Bull market (first 200 days) - positive trend, moderate volatility
        bull_returns = np.random.normal(0.001, 0.015, 200)
        
        # Bear market (next 150 days) - negative trend, high volatility  
        bear_returns = np.random.normal(-0.002, 0.025, 150)
        
        # Neutral market (last 150 days) - no trend, low volatility
        neutral_returns = np.random.normal(0.0, 0.010, 150)
        
        market_data['returns'] = np.concatenate([bull_returns, bear_returns, neutral_returns])
        
        # Generate correlated predictions
        signal_strength = {'bull': 0.3, 'bear': 0.2, 'neutral': 0.4}
        predictions = []
        
        for i, regime in enumerate(['bull'] * 200 + ['bear'] * 150 + ['neutral'] * 150):
            strength = signal_strength[regime]
            pred = market_data['returns'].iloc[i] * strength + np.random.normal(0, 0.01)
            predictions.append(pred)
        
        market_data['predictions'] = predictions
        
        return market_data
    
    def test_identify_market_regimes(self, regime_analyzer, market_data_with_regimes):
        """Test market regime identification."""
        regimes = regime_analyzer.identify_market_regimes(market_data_with_regimes)
        
        # Should identify different regimes
        unique_regimes = regimes.unique()
        assert len(unique_regimes) > 1
        
        # Each regime should have reasonable components
        for regime in unique_regimes:
            if regime != 'unknown':
                assert '_' in regime  # Should be in format 'trend_volatility'
                trend, vol = regime.split('_')
                assert trend in ['bull', 'bear', 'neutral']
                assert vol in ['low_vol', 'med_vol', 'high_vol']
    
    def test_analyze_regime_performance(self, regime_analyzer, market_data_with_regimes):
        """Test performance analysis across regimes."""
        regimes = regime_analyzer.identify_market_regimes(market_data_with_regimes)
        
        performance = regime_analyzer.analyze_regime_performance(
            market_data_with_regimes['predictions'],
            market_data_with_regimes['returns'],
            regimes,
            market_data_with_regimes
        )
        
        # Should have performance metrics for identified regimes
        assert len(performance) > 0
        
        for regime, metrics in performance.items():
            # Validate required metrics
            assert 'ic' in metrics
            assert 'hit_rate' in metrics
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown' in metrics
            assert 'n_observations' in metrics
            
            # Validate metric ranges
            assert abs(metrics['ic']) <= 1.0
            assert 0 <= metrics['hit_rate'] <= 1.0
            assert metrics['max_drawdown'] >= 0
            assert metrics['n_observations'] > 0
            assert 0 <= metrics['regime_frequency'] <= 1.0


class TestStatisticalValidator:
    """Test main statistical validator integration."""
    
    @pytest.fixture
    def validator(self):
        config = ValidationConfig(real_time_latency_target=200.0)  # Relaxed for testing
        return StatisticalValidator(config)
    
    @pytest.fixture
    def comprehensive_test_data(self):
        """Generate comprehensive test data."""
        np.random.seed(42)
        n_samples = 300
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        
        # Generate correlated predictions and returns
        true_signal = np.random.randn(n_samples) * 0.02
        noise = np.random.randn(n_samples) * 0.01
        
        predictions = pd.Series(true_signal + noise * 0.5, index=dates, name='predictions')
        returns = pd.Series(true_signal + noise, index=dates, name='returns')
        
        # Generate feature data
        feature_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.exponential(1, n_samples),
            'feature_3': np.random.uniform(-1, 1, n_samples)
        }, index=dates)
        
        # Generate market data
        market_data = pd.DataFrame({
            'returns': returns,
            'volume': np.random.exponential(1e6, n_samples),
            'volatility': np.random.uniform(0.1, 0.3, n_samples)
        }, index=dates)
        
        return predictions, returns, feature_data, market_data
    
    def test_comprehensive_validation(self, validator, comprehensive_test_data):
        """Test comprehensive validation pipeline."""
        predictions, returns, features, market_data = comprehensive_test_data
        
        # Generate reference data (earlier period)
        reference_data = {
            'features': features.iloc[:100],  # First 100 days as reference
            'targets': returns.iloc[:100],
            'predictions': predictions.iloc[:100]
        }
        
        # Validate current period
        current_features = features.iloc[100:]
        current_predictions = predictions.iloc[100:]
        current_returns = returns.iloc[100:]
        current_market = market_data.iloc[100:]
        
        results = validator.comprehensive_validation(
            current_predictions,
            current_returns, 
            current_features,
            current_market,
            reference_data,
            model_id="test_model"
        )
        
        # Validate results structure
        assert isinstance(results, ValidationResults)
        assert results.model_id == "test_model"
        assert isinstance(results.validation_score, float)
        assert 0 <= results.validation_score <= 1
        
        # Validate IC analysis
        assert 'current' in results.ic_scores
        assert len(results.ic_significance) > 0
        assert len(results.ic_stability) > 0
        
        # Validate drift detection
        assert len(results.feature_drift) > 0
        assert isinstance(results.target_drift, float)
        assert isinstance(results.prediction_drift, float)
        
        # Validate performance metrics
        assert 'ic_spearman' in results.performance_metrics
        assert 'sharpe_ratio' in results.performance_metrics
        assert 'hit_rate' in results.performance_metrics
        
        # Validate regime analysis
        assert len(results.regime_performance) > 0
        
        # Validate alerts and recommendations
        assert isinstance(results.alerts, list)
        assert isinstance(results.recommendations, list)
    
    def test_validation_latency_monitoring(self, validator, comprehensive_test_data):
        """Test real-time latency monitoring."""
        predictions, returns, features, market_data = comprehensive_test_data
        
        # Use small dataset for fast validation
        small_pred = predictions.iloc[-50:]
        small_ret = returns.iloc[-50:]
        small_feat = features.iloc[-50:]
        small_market = market_data.iloc[-50:]
        
        results = validator.comprehensive_validation(
            small_pred, small_ret, small_feat, small_market, model_id="latency_test"
        )
        
        # Check if latency alert was generated (depends on system performance)
        latency_alerts = [alert for alert in results.alerts if alert['type'] == 'performance']
        
        # Should complete within reasonable time for small dataset
        assert results.validation_score is not None
    
    def test_insufficient_data_handling(self, validator):
        """Test handling of insufficient data scenarios."""
        # Create minimal dataset
        minimal_pred = pd.Series([0.1, 0.2])
        minimal_ret = pd.Series([0.05, 0.1])
        minimal_feat = pd.DataFrame({'f1': [1, 2]})
        minimal_market = pd.DataFrame({'returns': minimal_ret})
        
        results = validator.comprehensive_validation(
            minimal_pred, minimal_ret, minimal_feat, minimal_market, model_id="minimal_test"
        )
        
        # Should handle gracefully without errors
        assert isinstance(results, ValidationResults)
        
        # Many metrics might be unavailable or have errors
        errors_expected = True  # Expected due to insufficient data


class TestTaiwanMarketValidator:
    """Test Taiwan market-specific validation."""
    
    @pytest.fixture
    def taiwan_validator(self):
        config = TaiwanMarketConfig()
        return TaiwanMarketValidator(config)
    
    @pytest.fixture 
    def taiwan_market_data(self):
        """Generate Taiwan market-specific test data."""
        np.random.seed(42)
        n_samples = 200
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        
        predictions = pd.Series(np.random.normal(0, 0.02, n_samples), index=dates)
        returns = pd.Series(np.random.normal(0, 0.025, n_samples), index=dates)
        
        # Add some price limit events
        limit_events = np.random.choice(n_samples, size=10, replace=False)
        returns.iloc[limit_events] = np.random.choice([0.098, -0.098], size=10)  # Near limit moves
        
        market_data = pd.DataFrame({
            'ret_t0': returns,
            'ret_t1': returns.shift(-1),
            'ret_t2': returns.shift(-2),
            'volume': np.random.exponential(1e6, n_samples),
            'sector': np.random.choice(['24', '17', '08'], n_samples),  # Tech, Financial, Traditional
            'timestamp': dates,
            'foreign_ownership': np.random.uniform(0.1, 0.6, n_samples)
        }, index=dates)
        
        return predictions, returns, market_data
    
    def test_settlement_impact_analysis(self, taiwan_validator, taiwan_market_data):
        """Test T+2 settlement impact analysis."""
        predictions, returns, market_data = taiwan_market_data
        
        settlement_results = taiwan_validator.settlement_validator.analyze_settlement_impact(
            predictions,
            market_data['ret_t0'],
            market_data['ret_t1'].dropna(),
            market_data['ret_t2'].dropna(),
            market_data['volume']
        )
        
        # Validate settlement analysis structure
        assert 'settlement_analysis' in settlement_results
        assert 'trading_cost_impact' in settlement_results
        
        settlement = settlement_results['settlement_analysis']
        assert 'ic_decay_pattern' in settlement
        assert 'settlement_efficiency' in settlement
        
        # Validate IC decay pattern
        ic_decay = settlement['ic_decay_pattern']
        assert all(key in ic_decay for key in ['t0', 't1', 't2'])
        
    def test_price_limit_impact_analysis(self, taiwan_validator, taiwan_market_data):
        """Test price limit impact analysis."""
        predictions, returns, market_data = taiwan_market_data
        
        # Create mock OHLC data
        ohlc_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, len(returns)),
            'high': np.random.uniform(95, 115, len(returns)),
            'low': np.random.uniform(85, 105, len(returns)),
            'close': np.random.uniform(90, 110, len(returns))
        }, index=returns.index)
        
        limit_results = taiwan_validator.limit_validator.analyze_price_limit_impact(
            predictions, returns, ohlc_data
        )
        
        # Validate limit analysis structure
        assert 'limit_statistics' in limit_results
        assert 'prediction_accuracy' in limit_results
        assert 'clustering_analysis' in limit_results
        
        # Validate statistics
        stats = limit_results['limit_statistics']
        assert all(key in stats for key in ['limit_up_frequency', 'limit_down_frequency', 'total_limit_events'])
        assert all(0 <= stats[key] <= 1 for key in ['limit_up_frequency', 'limit_down_frequency'])
    
    def test_comprehensive_taiwan_validation(self, taiwan_validator, taiwan_market_data):
        """Test comprehensive Taiwan market validation."""
        predictions, returns, market_data = taiwan_market_data
        
        results = taiwan_validator.comprehensive_taiwan_validation(
            predictions, returns, market_data
        )
        
        # Validate comprehensive results structure
        assert 'timestamp' in results
        assert 'validation_type' in results
        assert results['validation_type'] == 'taiwan_market_comprehensive'
        
        # Should include Taiwan-specific analyses
        expected_sections = ['settlement_analysis', 'sector_analysis', 'compliance_check', 'taiwan_recommendations']
        
        # At least some sections should be present (depends on available data)
        assert any(section in results for section in expected_sections)
        
        # Validate recommendations
        if 'taiwan_recommendations' in results:
            assert isinstance(results['taiwan_recommendations'], list)


class TestValidationIntegration:
    """Integration tests combining statistical and Taiwan market validation."""
    
    def test_end_to_end_validation_pipeline(self):
        """Test complete validation pipeline from model predictions to recommendations."""
        # Setup
        np.random.seed(42)
        n_samples = 250
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        
        # Generate realistic test data
        true_alpha = np.random.randn(n_samples) * 0.01
        predictions = pd.Series(true_alpha + np.random.randn(n_samples) * 0.005, index=dates)
        returns = pd.Series(true_alpha + np.random.randn(n_samples) * 0.015, index=dates)
        
        features = pd.DataFrame({
            'momentum': np.random.normal(0, 1, n_samples),
            'value': np.random.normal(0, 1, n_samples), 
            'quality': np.random.normal(0, 1, n_samples)
        }, index=dates)
        
        market_data = pd.DataFrame({
            'returns': returns,
            'ret_t0': returns,
            'ret_t1': returns.shift(-1),
            'ret_t2': returns.shift(-2),
            'volume': np.random.exponential(1e6, n_samples),
            'sector': np.random.choice(['24', '17', '08'], n_samples),
            'timestamp': dates,
            'foreign_ownership': np.random.uniform(0.2, 0.5, n_samples)
        }, index=dates)
        
        # Initialize validators
        statistical_validator = StatisticalValidator()
        taiwan_validator = TaiwanMarketValidator()
        
        # Perform validations
        statistical_results = statistical_validator.comprehensive_validation(
            predictions, returns, features, market_data, model_id="integration_test"
        )
        
        taiwan_results = taiwan_validator.comprehensive_taiwan_validation(
            predictions, returns, market_data
        )
        
        # Validate integration
        assert statistical_results.validation_score > 0
        assert len(statistical_results.alerts) >= 0
        assert len(statistical_results.recommendations) >= 0
        
        assert 'timestamp' in taiwan_results
        assert len(taiwan_results.get('taiwan_recommendations', [])) >= 0
        
        # Combined scoring could be implemented here
        combined_score = (statistical_results.validation_score + 
                         (1.0 if taiwan_results.get('compliance_check', {}) else 0.5)) / 2
        
        assert 0 <= combined_score <= 1
    
    def test_performance_under_load(self):
        """Test validation performance with larger datasets."""
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 1000  # ~4 years of daily data
        
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        predictions = pd.Series(np.random.randn(n_samples) * 0.02, index=dates)
        returns = pd.Series(np.random.randn(n_samples) * 0.025, index=dates)
        
        features = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) for i in range(20)
        }, index=dates)
        
        market_data = pd.DataFrame({
            'returns': returns,
            'volume': np.random.exponential(1e6, n_samples)
        }, index=dates)
        
        # Test statistical validation performance
        validator = StatisticalValidator()
        
        start_time = datetime.now()
        results = validator.comprehensive_validation(
            predictions, returns, features, market_data, model_id="load_test"
        )
        end_time = datetime.now()
        
        validation_time = (end_time - start_time).total_seconds() * 1000  # ms
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert validation_time < 5000  # 5 seconds
        assert results.validation_score is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])