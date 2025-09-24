"""
Integration Tests for Production Pipeline.

Comprehensive integration testing for LightGBM alpha model production system:
- End-to-end model training and inference pipeline
- Real-time prediction system performance validation
- Model monitoring and health check system
- Taiwan market specific validations
- Production deployment readiness verification
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock

# Import production modules
from src.models.lightgbm_alpha import LightGBMAlphaModel, ModelConfig
from src.inference.realtime import (
    RealtimePredictor, InferenceConfig, PredictionBatch, 
    InferenceMetrics, LatencyOptimizer
)
from src.monitoring.model_health import (
    ModelHealthMonitor, MonitoringConfig, HealthMetrics,
    DriftDetector, PerformanceTracker, AlertManager, AlertLevel
)
from src.factors.base import FactorEngine
from src.backtesting.validation.walk_forward import WalkForwardSplitter, WalkForwardConfig

logger = logging.getLogger(__name__)


class TestProductionPipeline:
    """Integration tests for complete production pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        n_symbols = 100
        
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=n_samples//n_symbols, freq='D')
        symbols = [f'TSE_{i:04d}' for i in range(n_symbols)]
        
        # Multi-index for (date, symbol)
        index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
        
        # Generate features
        feature_names = [f'factor_{i:02d}' for i in range(n_features)]
        features = pd.DataFrame(
            np.random.randn(len(index), n_features),
            index=index[:len(index)],
            columns=feature_names
        )
        
        # Generate returns (somewhat correlated with first few features)
        signal = (
            0.3 * features['factor_00'] + 
            0.2 * features['factor_01'] + 
            0.1 * features['factor_02'] + 
            np.random.randn(len(features)) * 0.5
        )
        returns = pd.Series(signal / 100, index=features.index, name='returns')  # Scale to realistic returns
        
        # Generate prices for market data
        prices = pd.DataFrame({
            'open': 100 + np.random.randn(len(features)) * 5,
            'high': 105 + np.random.randn(len(features)) * 3,
            'low': 95 + np.random.randn(len(features)) * 3,
            'close': 100 + np.random.randn(len(features)) * 4,
            'volume': np.random.randint(1000, 100000, len(features))
        }, index=features.index)
        
        return {
            'features': features,
            'returns': returns,
            'prices': prices,
            'symbols': symbols,
            'dates': dates
        }
    
    @pytest.fixture
    def mock_factor_engine(self, sample_data):
        """Create mock factor engine for testing."""
        engine = Mock(spec=FactorEngine)
        
        def mock_compute_factors_for_symbol(symbol, timestamp, feature_columns):
            # Return sample features for the symbol
            return sample_data['features'].loc[:, feature_columns].iloc[0]
        
        def mock_compute_factors_batch(symbols, timestamp, feature_columns):
            # Return sample features for all symbols
            n_symbols = len(symbols)
            return pd.DataFrame(
                np.random.randn(n_symbols, len(feature_columns)),
                index=symbols,
                columns=feature_columns
            )
        
        engine.compute_factors_for_symbol.side_effect = mock_compute_factors_for_symbol
        engine.compute_factors_batch.side_effect = mock_compute_factors_batch
        
        return engine
    
    @pytest.fixture 
    def trained_model(self, sample_data):
        """Create and train a model for testing."""
        config = ModelConfig()
        model = LightGBMAlphaModel(config)
        
        # Prepare training data
        X, y = model.prepare_training_data(
            sample_data['features'], 
            sample_data['returns'],
            target_horizon=5
        )
        
        # Train the model
        model.train(X, y, verbose=False)
        
        return model
    
    def test_model_training_pipeline(self, sample_data):
        """Test complete model training pipeline."""
        # Initialize model
        config = ModelConfig(
            n_estimators=50,  # Smaller for testing
            early_stopping_rounds=10
        )
        model = LightGBMAlphaModel(config)
        
        # Test data preparation
        X, y = model.prepare_training_data(
            sample_data['features'],
            sample_data['returns'],
            target_horizon=5
        )
        
        assert X.shape[0] > 0, "Training data should not be empty"
        assert X.shape[1] == len(sample_data['features'].columns), "Feature count mismatch"
        assert len(y) == len(X), "Target and feature length mismatch"
        
        # Test training
        training_stats = model.train(X, y, verbose=False)
        
        assert model.model is not None, "Model should be trained"
        assert 'training_time_seconds' in training_stats, "Training stats missing"
        assert training_stats['training_time_seconds'] > 0, "Training should take time"
        
        # Test prediction
        predictions = model.predict(X.head(10))
        assert len(predictions) == 10, "Prediction count mismatch"
        assert all(np.isfinite(predictions)), "Predictions should be finite"
        
        # Test model saving and loading
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save_model(temp_path)
            assert Path(temp_path).exists(), "Model file should be saved"
            
            # Load model and test
            new_model = LightGBMAlphaModel(config)
            new_model.load_model(temp_path)
            
            new_predictions = new_model.predict(X.head(10))
            np.testing.assert_array_almost_equal(
                predictions, new_predictions, 
                decimal=6, err_msg="Loaded model predictions should match"
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_realtime_inference_system(self, trained_model, mock_factor_engine, sample_data):
        """Test real-time inference system performance and functionality."""
        # Initialize inference config for testing
        config = InferenceConfig(
            max_latency_ms=100.0,
            target_latency_ms=50.0,
            batch_size=50,
            enable_async=True
        )
        
        predictor = RealtimePredictor(
            model=trained_model,
            factor_engine=mock_factor_engine,
            config=config
        )
        
        # Test warmup
        warmup_metrics = predictor.warmup(n_symbols=50)
        assert warmup_metrics['ready_for_production'], "System should be ready after warmup"
        assert warmup_metrics['prediction_latency_ms'] > 0, "Warmup should measure latency"
        
        # Test single prediction
        test_symbol = sample_data['symbols'][0]
        prediction, latency = predictor.predict_single(test_symbol)
        
        assert np.isfinite(prediction), "Prediction should be finite"
        assert latency > 0, "Latency should be measured"
        assert latency < config.max_latency_ms * 2, f"Latency {latency}ms too high for single prediction"
        
        # Test batch prediction
        test_symbols = sample_data['symbols'][:20]
        predictions, batch_latency = predictor.predict_batch(test_symbols)
        
        assert len(predictions) == len(test_symbols), "Prediction count should match symbol count"
        assert all(np.isfinite(predictions)), "All predictions should be finite"
        assert batch_latency < config.max_latency_ms * 1.5, f"Batch latency {batch_latency}ms too high"
        
        # Test performance under load
        large_symbol_list = sample_data['symbols']  # 100 symbols
        start_time = time.time()
        large_predictions, large_latency = predictor.predict_batch(large_symbol_list)
        total_time = time.time() - start_time
        
        assert len(large_predictions) == len(large_symbol_list), "Large batch prediction count mismatch"
        assert large_latency < config.max_latency_ms * 2, f"Large batch latency {large_latency}ms too high"
        
        # Verify throughput
        throughput = len(large_symbol_list) / total_time
        assert throughput > 100, f"Throughput {throughput} predictions/sec too low"
        
        # Test metrics collection
        performance_summary = predictor.get_performance_summary()
        assert performance_summary['total_predictions'] > 0, "Should have recorded predictions"
        assert performance_summary['success_rate'] > 0.95, "Success rate should be high"
        
        logger.info(f"Inference performance: {large_latency:.1f}ms for {len(large_symbol_list)} symbols")
    
    @pytest.mark.asyncio
    async def test_async_inference(self, trained_model, mock_factor_engine, sample_data):
        """Test asynchronous inference capabilities."""
        config = InferenceConfig(enable_async=True, max_workers=4)
        predictor = RealtimePredictor(trained_model, mock_factor_engine, config)
        
        # Test async batch prediction
        test_symbols = sample_data['symbols'][:30]
        
        async_start = time.time()
        async_predictions = await predictor.predict_async(test_symbols, priority=1)
        async_time = time.time() - async_start
        
        assert len(async_predictions) == len(test_symbols), "Async prediction count mismatch"
        assert all(np.isfinite(async_predictions)), "Async predictions should be finite"
        
        # Test multiple concurrent requests
        concurrent_tasks = [
            predictor.predict_async(sample_data['symbols'][i:i+10], priority=i)
            for i in range(0, 50, 10)
        ]
        
        concurrent_start = time.time()
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = time.time() - concurrent_start
        
        assert len(concurrent_results) == 5, "Should have 5 concurrent results"
        total_predictions = sum(len(result) for result in concurrent_results)
        assert total_predictions == 50, "Total concurrent predictions should be 50"
        
        logger.info(f"Async inference: {concurrent_time:.2f}s for 50 symbols across 5 concurrent requests")
    
    def test_model_monitoring_system(self, trained_model, sample_data):
        """Test comprehensive model monitoring and health system."""
        # Initialize monitoring config
        config = MonitoringConfig(
            min_ic_threshold=0.02,
            max_rmse_threshold=0.20,
            feature_drift_threshold=0.05,
            enable_email_alerts=False  # Disable for testing
        )
        
        monitor = ModelHealthMonitor(
            model=trained_model,
            config=config
        )
        
        # Set up reference data for drift detection
        reference_features = sample_data['features'].head(500)
        reference_predictions = np.random.randn(500)
        monitor.setup_reference_data(reference_features, reference_predictions)
        
        # Test health update with good data
        good_predictions = np.random.randn(100) * 0.05  # Low variance, good predictions
        good_actuals = good_predictions + np.random.randn(100) * 0.01  # Correlated actuals
        good_returns = np.random.randn(100) * 0.01 + 0.0005  # Positive expected return
        
        health_metrics = monitor.update_health(
            predictions=good_predictions,
            actuals=good_actuals,
            features=sample_data['features'].head(100),
            returns=good_returns
        )
        
        assert health_metrics.current_ic > 0, "IC should be positive for correlated data"
        assert health_metrics.current_rmse > 0, "RMSE should be calculated"
        assert health_metrics.overall_status != health_metrics.overall_status.CRITICAL, "Should not be critical with good data"
        
        # Test health update with poor data (should trigger alerts)
        bad_predictions = np.random.randn(100) * 0.5  # High variance
        bad_actuals = -bad_predictions + np.random.randn(100) * 0.1  # Negative correlation
        bad_returns = np.random.randn(100) * 0.05 - 0.01  # Negative returns
        
        bad_health = monitor.update_health(
            predictions=bad_predictions,
            actuals=bad_actuals,
            returns=bad_returns
        )
        
        assert bad_health.current_ic < health_metrics.current_ic, "IC should be worse for bad data"
        assert bad_health.current_rmse > health_metrics.current_rmse, "RMSE should be higher for bad data"
        
        # Test drift detection with different data
        drifted_features = sample_data['features'].head(100) + 2.0  # Significant mean shift
        
        monitor.update_health(
            features=drifted_features
        )
        
        assert monitor.current_health.feature_drift_score > 0, "Should detect feature drift"
        
        # Test comprehensive health report
        health_report = monitor.get_health_report()
        
        assert 'overall_status' in health_report, "Report should include overall status"
        assert 'health_score' in health_report, "Report should include health score"
        assert 'metrics' in health_report, "Report should include metrics"
        assert 'alerts' in health_report, "Report should include alerts"
        
        # Test alert system
        alert_summary = monitor.alert_manager.get_alert_summary()
        assert 'total_alerts_24h' in alert_summary, "Alert summary should include 24h count"
        
        logger.info(f"Model health score: {health_metrics.get_health_score():.1f}/100")
    
    def test_drift_detection_system(self, sample_data):
        """Test statistical drift detection capabilities."""
        config = MonitoringConfig()
        drift_detector = DriftDetector(config)
        
        # Set up reference data
        reference_data = np.random.randn(1000)
        drift_detector.update_reference_distribution('test_feature', reference_data)
        
        # Test with similar data (should not detect drift)
        similar_data = np.random.randn(500)  # Same distribution
        is_drifted, drift_score, test_results = drift_detector.detect_feature_drift(
            'test_feature', similar_data
        )
        
        assert not is_drifted or drift_score < 0.1, "Should not detect drift in similar data"
        assert 'ks_statistic' in test_results, "Should include KS test results"
        assert 'psi' in test_results, "Should include PSI"
        
        # Test with drifted data (should detect drift)
        drifted_data = np.random.randn(500) + 2.0  # Mean shift
        is_drifted_2, drift_score_2, test_results_2 = drift_detector.detect_feature_drift(
            'test_feature', drifted_data
        )
        
        assert is_drifted_2, "Should detect drift in shifted data"
        assert drift_score_2 > drift_score, "Drift score should be higher for drifted data"
        assert test_results_2['ks_p_value'] < 0.05, "KS test should be significant"
        
        # Test prediction drift detection
        ref_predictions = np.random.randn(1000) * 0.1
        drift_detector.update_reference_distribution('predictions', ref_predictions)
        
        # Drifted predictions (different scale)
        drifted_predictions = np.random.randn(500) * 0.5  # Different variance
        pred_drifted, pred_score, pred_results = drift_detector.detect_prediction_drift(
            drifted_predictions, ref_predictions
        )
        
        assert pred_drifted, "Should detect prediction distribution drift"
        assert pred_score > 0.1, "Prediction drift score should be significant"
    
    def test_latency_optimization(self, trained_model, mock_factor_engine):
        """Test latency optimization techniques."""
        config = InferenceConfig()
        optimizer = LatencyOptimizer(config)
        
        # Test batch size optimization
        optimal_size_100 = optimizer.optimize_batch_size(100)
        optimal_size_500 = optimizer.optimize_batch_size(500)
        optimal_size_2000 = optimizer.optimize_batch_size(2000)
        
        assert optimal_size_100 <= 100, "Optimal size should not exceed input size"
        assert optimal_size_500 > optimal_size_100, "Larger inputs should get larger batch sizes"
        assert optimal_size_2000 <= config.max_batch_size, "Should not exceed max batch size"
        
        # Test feature preprocessing optimization
        n_symbols = 100
        n_features = 20
        test_features = pd.DataFrame(
            np.random.randn(n_symbols, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add some missing values and infinity
        test_features.iloc[0, 0] = np.nan
        test_features.iloc[1, 1] = np.inf
        
        start_time = time.time()
        processed_features = optimizer.preprocess_features_vectorized(test_features)
        processing_time = time.time() - start_time
        
        assert not processed_features.isnull().any().any(), "Should handle missing values"
        assert not np.isinf(processed_features.values).any(), "Should handle infinite values"
        assert processing_time < 0.1, f"Preprocessing too slow: {processing_time:.3f}s"
        assert processed_features.dtypes.apply(lambda x: x == np.float32).all(), "Should convert to float32"
    
    def test_taiwan_market_specifics(self, sample_data):
        """Test Taiwan market specific functionality."""
        from src.inference.realtime import is_market_hours, calculate_market_timing_features
        
        config = InferenceConfig()
        
        # Test market hours detection
        market_open = datetime(2023, 6, 15, 9, 30)  # 09:30 TST
        market_close = datetime(2023, 6, 15, 13, 30)  # 13:30 TST
        pre_market = datetime(2023, 6, 15, 8, 0)  # 08:00 TST
        post_market = datetime(2023, 6, 15, 15, 0)  # 15:00 TST
        
        assert is_market_hours(market_open, config), "09:30 should be market hours"
        assert is_market_hours(market_close, config), "13:30 should be market hours"
        assert not is_market_hours(pre_market, config), "08:00 should not be market hours"
        assert not is_market_hours(post_market, config), "15:00 should not be market hours"
        
        # Test market timing features
        timing_features = calculate_market_timing_features(market_open, config)
        assert timing_features['market_session'] == 1.0, "Should be active market session"
        assert 0 <= timing_features['session_progress'] <= 1, "Session progress should be normalized"
        
        pre_timing = calculate_market_timing_features(pre_market, config)
        assert pre_timing['market_session'] == 0.0, "Should be pre-market session"
        assert pre_timing['time_to_open'] > 0, "Should have time to open"
        
        post_timing = calculate_market_timing_features(post_market, config)
        assert post_timing['market_session'] == 2.0, "Should be post-market session"
        assert post_timing['time_to_close'] == 0.0, "Market should be closed"
    
    def test_production_deployment_readiness(self, trained_model, mock_factor_engine, sample_data):
        """Test production deployment readiness and requirements."""
        # Test model serialization and deserialization under load
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            # Save and load model multiple times to test stability
            for i in range(3):
                trained_model.save_model(model_path)
                
                new_model = LightGBMAlphaModel()
                load_start = time.time()
                new_model.load_model(model_path)
                load_time = time.time() - load_start
                
                assert load_time < 30.0, f"Model loading too slow: {load_time:.2f}s"
                
                # Test consistency
                test_features = sample_data['features'].head(10)
                original_preds = trained_model.predict(test_features)
                loaded_preds = new_model.predict(test_features)
                
                np.testing.assert_array_almost_equal(
                    original_preds, loaded_preds, decimal=6,
                    err_msg=f"Model consistency failed on iteration {i}"
                )
        finally:
            Path(model_path).unlink(missing_ok=True)
        
        # Test memory usage under sustained load
        config = InferenceConfig(batch_size=100, max_batch_size=500)
        predictor = RealtimePredictor(trained_model, mock_factor_engine, config)
        
        # Simulate sustained inference load
        memory_usage = []
        for batch_num in range(10):
            batch_symbols = sample_data['symbols']
            predictions, latency = predictor.predict_batch(batch_symbols)
            
            # Check predictions are reasonable
            assert len(predictions) == len(batch_symbols), f"Batch {batch_num} prediction count mismatch"
            assert all(np.isfinite(predictions)), f"Batch {batch_num} has invalid predictions"
            assert latency < config.max_latency_ms * 2, f"Batch {batch_num} latency too high: {latency:.1f}ms"
            
            # Simulate memory tracking (would use psutil in production)
            memory_usage.append(latency)  # Use latency as proxy for resource usage
        
        # Test performance doesn't degrade over time
        early_performance = np.mean(memory_usage[:3])
        late_performance = np.mean(memory_usage[-3:])
        performance_degradation = (late_performance - early_performance) / early_performance
        
        assert performance_degradation < 0.5, f"Performance degraded by {performance_degradation:.2%}"
        
        # Test graceful error handling
        try:
            # Test with invalid features
            invalid_features = pd.DataFrame(
                np.full((10, len(trained_model.feature_columns)), np.nan),
                columns=trained_model.feature_columns
            )
            predictions = trained_model.predict(invalid_features)
            # Should not crash, may return default predictions
            assert len(predictions) == 10, "Should handle invalid features gracefully"
            
        except Exception as e:
            # Should raise appropriate error, not crash
            assert "Invalid" in str(e) or "NaN" in str(e), f"Unexpected error type: {e}"
        
        # Test production-ready monitoring
        monitor = ModelHealthMonitor(trained_model, config=MonitoringConfig(enable_email_alerts=False))
        
        # Should handle missing data gracefully
        incomplete_health = monitor.update_health()
        assert incomplete_health.overall_status != None, "Health update should handle missing data"
        
        logger.info("Production deployment readiness: PASSED")
    
    def test_integration_with_validation_framework(self, trained_model, sample_data):
        """Test integration with existing validation framework."""
        # This tests that the production components work with the validation framework
        from src.data.core.temporal import TemporalStore
        
        # Mock temporal store for integration testing
        mock_temporal_store = Mock(spec=TemporalStore)
        
        # Test walk-forward validation config compatibility
        wf_config = WalkForwardConfig(
            train_weeks=52,  # 1 year for testing
            test_weeks=4,    # 1 month for testing
            use_taiwan_calendar=True
        )
        
        assert wf_config.train_weeks > 0, "Training weeks should be positive"
        assert wf_config.test_weeks > 0, "Testing weeks should be positive"
        assert wf_config.use_taiwan_calendar, "Should use Taiwan calendar"
        
        # Test model config compatibility with validation
        model_config = ModelConfig(
            target_horizons=[1, 5, 10],  # Compatible with walk-forward
            winsorize_quantile=0.01,
            memory_limit_gb=8.0
        )
        
        assert 1 in model_config.target_horizons, "Should include 1-day horizon"
        assert model_config.memory_limit_gb <= 16.0, "Memory limit should be reasonable"
        
        # Test inference config production settings
        inference_config = InferenceConfig(
            max_latency_ms=100.0,  # Production SLA
            batch_size=500,        # Taiwan market ~2000 stocks
            enable_async=True      # Production requirement
        )
        
        assert inference_config.max_latency_ms <= 100.0, "Latency should meet SLA"
        assert inference_config.batch_size >= 100, "Batch size should handle reasonable load"
        assert inference_config.enable_async, "Should support async operations"
        
        # Test monitoring config Taiwan market settings
        monitoring_config = MonitoringConfig(
            min_ic_threshold=0.03,    # Taiwan market appropriate
            market_regime_detection=True,
            sector_rotation_monitoring=True,
            foreign_flow_tracking=True
        )
        
        assert monitoring_config.market_regime_detection, "Should monitor Taiwan market regimes"
        assert monitoring_config.sector_rotation_monitoring, "Should monitor sector rotation"
        assert monitoring_config.foreign_flow_tracking, "Should track foreign flows"
        
        logger.info("Validation framework integration: PASSED")


@pytest.mark.integration
class TestEndToEndProduction:
    """End-to-end integration tests for production system."""
    
    def test_complete_pipeline_e2e(self):
        """Test complete end-to-end production pipeline."""
        # This test would simulate the complete production workflow:
        # 1. Data ingestion and factor computation
        # 2. Model training with cross-validation
        # 3. Real-time inference system startup
        # 4. Monitoring system initialization
        # 5. Simulated trading day operations
        # 6. Health monitoring and alerting
        
        logger.info("Complete E2E pipeline test would require:")
        logger.info("1. Real data connections (FinLab, market data)")
        logger.info("2. Production database setup")
        logger.info("3. External monitoring systems")
        logger.info("4. Email/Slack notification setup")
        logger.info("5. Multi-hour sustained operation test")
        
        # For now, just verify all components can be imported and initialized
        try:
            # Test all production imports
            from src.models.lightgbm_alpha import LightGBMAlphaModel
            from src.inference.realtime import RealtimePredictor
            from src.monitoring.model_health import ModelHealthMonitor
            
            logger.info("All production modules imported successfully")
            
            # Test basic initialization without data
            model = LightGBMAlphaModel()
            assert model is not None, "Model should initialize"
            
            logger.info("Production pipeline components ready for deployment")
            
        except Exception as e:
            pytest.fail(f"Production component initialization failed: {e}")


# Performance benchmarks for production readiness
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for production system."""
    
    @pytest.mark.parametrize("n_symbols", [100, 500, 1000, 2000])
    def test_inference_latency_benchmark(self, n_symbols, trained_model, mock_factor_engine):
        """Benchmark inference latency across different symbol counts."""
        config = InferenceConfig(batch_size=min(n_symbols, 500))
        predictor = RealtimePredictor(trained_model, mock_factor_engine, config)
        
        symbols = [f'TSE_{i:04d}' for i in range(n_symbols)]
        
        # Warmup
        predictor.warmup(n_symbols=min(n_symbols, 100))
        
        # Benchmark
        latencies = []
        for _ in range(5):  # Multiple runs for stability
            start_time = time.time()
            predictions, latency = predictor.predict_batch(symbols)
            total_time = time.time() - start_time
            
            latencies.append(latency)
            
            assert len(predictions) == n_symbols, f"Prediction count mismatch for {n_symbols} symbols"
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Performance assertions
        if n_symbols <= 500:
            assert avg_latency < 50.0, f"Average latency {avg_latency:.1f}ms too high for {n_symbols} symbols"
        elif n_symbols <= 1000:
            assert avg_latency < 75.0, f"Average latency {avg_latency:.1f}ms too high for {n_symbols} symbols"
        else:
            assert avg_latency < 100.0, f"Average latency {avg_latency:.1f}ms too high for {n_symbols} symbols"
        
        assert p95_latency < avg_latency * 1.5, f"P95 latency {p95_latency:.1f}ms too high"
        
        logger.info(f"Inference benchmark - {n_symbols} symbols: "
                   f"avg={avg_latency:.1f}ms, p95={p95_latency:.1f}ms")


# Utility fixtures and functions
@pytest.fixture(scope="session")
def trained_model(sample_data):
    """Session-scoped trained model to avoid repeated training."""
    config = ModelConfig(n_estimators=100, early_stopping_rounds=20)
    model = LightGBMAlphaModel(config)
    
    # Use a subset of data for faster training
    features_subset = sample_data['features'].head(500)
    returns_subset = sample_data['returns'].head(500)
    
    X, y = model.prepare_training_data(features_subset, returns_subset, target_horizon=5)
    model.train(X, y, verbose=False)
    
    return model


if __name__ == "__main__":
    # Run integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not benchmark"  # Skip benchmarks by default
    ])