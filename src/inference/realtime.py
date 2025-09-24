"""
Real-time Inference System for LightGBM Alpha Model.

High-performance prediction system optimized for Taiwan equity market with:
- Sub-100ms latency for 2000 stocks
- Batch processing optimization
- Memory-efficient feature processing
- Real-time performance monitoring
"""

import asyncio
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from functools import lru_cache

from ..models.lightgbm_alpha import LightGBMAlphaModel, ModelConfig
from ..factors.base import FactorEngine
from ..data.core.temporal import TemporalStore, DataType

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class InferenceConfig:
    """Configuration for real-time inference system."""
    
    # Performance targets
    max_latency_ms: float = 100.0  # Maximum allowable inference latency
    target_latency_ms: float = 50.0  # Target inference latency
    batch_size: int = 500  # Optimal batch size for processing
    max_batch_size: int = 2000  # Maximum batch size (full universe)
    
    # Memory optimization
    feature_cache_size: int = 10000  # LRU cache size for features
    model_cache_ttl_seconds: int = 3600  # Model cache TTL (1 hour)
    enable_gpu: bool = False  # Use GPU acceleration if available
    
    # Threading configuration  
    max_workers: int = 4  # Maximum thread workers
    enable_async: bool = True  # Use async processing
    
    # Taiwan market specifics
    market_open_hour: int = 9  # TST
    market_close_hour: int = 13  # TST (13:30 close)
    market_close_minute: int = 30
    
    # Monitoring
    enable_metrics: bool = True
    metrics_window_size: int = 1000  # Rolling window for metrics
    alert_latency_threshold_ms: float = 150.0  # Alert if latency exceeds


@dataclass 
class PredictionBatch:
    """Container for batch prediction requests."""
    batch_id: str
    symbols: List[str]
    features: pd.DataFrame
    timestamp: datetime
    priority: int = 0  # Higher priority processed first
    
    def __len__(self) -> int:
        return len(self.symbols)


@dataclass
class InferenceMetrics:
    """Real-time inference performance metrics."""
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    
    # Latency metrics
    latencies_ms: List[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Throughput metrics
    predictions_per_second: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    
    # Error tracking
    recent_errors: List[str] = field(default_factory=list)
    error_rate: float = 0.0
    
    def update_latency(self, latency_ms: float) -> None:
        """Update latency metrics with new measurement."""
        self.latencies_ms.append(latency_ms)
        
        # Keep only recent measurements for rolling window
        if len(self.latencies_ms) > 1000:
            self.latencies_ms = self.latencies_ms[-1000:]
        
        # Update derived metrics
        if self.latencies_ms:
            self.avg_latency_ms = np.mean(self.latencies_ms)
            self.p95_latency_ms = np.percentile(self.latencies_ms, 95)
            self.p99_latency_ms = np.percentile(self.latencies_ms, 99)
            self.max_latency_ms = max(self.latencies_ms)
    
    def update_throughput(self) -> None:
        """Update throughput metrics."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > 0:
            self.predictions_per_second = self.successful_predictions / elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'failed_predictions': self.failed_predictions,
            'success_rate': self.successful_predictions / max(self.total_predictions, 1),
            'avg_latency_ms': self.avg_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,  
            'p99_latency_ms': self.p99_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'predictions_per_second': self.predictions_per_second,
            'error_rate': self.error_rate,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }


class LatencyOptimizer:
    """Optimizes inference latency through various techniques."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._feature_cache = {}
        self._model_cache = {}
        
    @lru_cache(maxsize=1000)
    def _cached_feature_computation(
        self, 
        symbol: str, 
        data_hash: str,
        computation_func: Callable
    ) -> np.ndarray:
        """Cache expensive feature computations."""
        return computation_func()
    
    def optimize_batch_size(self, n_symbols: int) -> int:
        """Determine optimal batch size based on number of symbols."""
        if n_symbols <= self.config.batch_size:
            return n_symbols
        elif n_symbols <= self.config.max_batch_size:
            # Find divisor that minimizes remainder
            optimal_size = self.config.batch_size
            for size in range(self.config.batch_size, min(1000, n_symbols), 100):
                if n_symbols % size < n_symbols % optimal_size:
                    optimal_size = size
            return optimal_size
        else:
            return self.config.max_batch_size
    
    def preprocess_features_vectorized(
        self, 
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """Vectorized feature preprocessing for speed."""
        # Use vectorized operations instead of loops
        features_processed = features.copy()
        
        # Handle missing values efficiently
        features_processed = features_processed.fillna(method='ffill').fillna(0)
        
        # Apply any needed transformations vectorized
        numeric_columns = features_processed.select_dtypes(include=[np.number]).columns
        features_processed[numeric_columns] = features_processed[numeric_columns].astype(np.float32)
        
        return features_processed


class RealtimePredictor:
    """
    High-performance real-time prediction system for Taiwan equity alpha.
    
    Features:
    - Sub-100ms latency for up to 2000 stocks
    - Batch processing optimization
    - Asynchronous prediction capabilities
    - Real-time performance monitoring
    - Memory-efficient operations
    """
    
    def __init__(
        self,
        model: LightGBMAlphaModel,
        factor_engine: FactorEngine,
        config: Optional[InferenceConfig] = None
    ):
        self.model = model
        self.factor_engine = factor_engine
        self.config = config or InferenceConfig()
        
        # Performance optimization
        self.optimizer = LatencyOptimizer(self.config)
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Metrics tracking
        self.metrics = InferenceMetrics()
        
        # Prediction queue for batch processing
        self._prediction_queue: asyncio.Queue = asyncio.Queue()
        self._is_running = False
        
        # Precompute model info for optimization
        self._precompute_model_metadata()
        
        logger.info(f"RealtimePredictor initialized with config: {self.config}")
    
    def _precompute_model_metadata(self) -> None:
        """Precompute model metadata for faster inference."""
        if self.model.model is None:
            logger.warning("Model not trained - some optimizations unavailable")
            return
            
        # Cache feature columns for fast validation
        self._feature_columns = set(self.model.feature_columns)
        self._n_features = len(self.model.feature_columns)
        
        # Precompute feature importance for potential feature selection
        if hasattr(self.model, 'feature_importance_'):
            top_features = self.model.get_feature_importance(50)  # Top 50 features
            self._important_features = set(top_features['feature'].tolist())
        else:
            self._important_features = self._feature_columns
            
        logger.debug(f"Precomputed metadata: {self._n_features} features")
    
    def predict_single(
        self,
        symbol: str,
        features: Optional[pd.Series] = None,
        timestamp: Optional[datetime] = None
    ) -> Tuple[float, float]:
        """
        Generate single symbol prediction with latency tracking.
        
        Args:
            symbol: Stock symbol to predict
            features: Precomputed features (optional, will compute if None)
            timestamp: Prediction timestamp (defaults to now)
            
        Returns:
            Tuple of (prediction, latency_ms)
        """
        start_time = time.perf_counter()
        timestamp = timestamp or datetime.now()
        
        try:
            # Get or compute features
            if features is None:
                features = self._get_features_for_symbol(symbol, timestamp)
            
            # Validate features
            if not self._validate_features(features):
                raise ValueError(f"Invalid features for symbol {symbol}")
            
            # Convert to DataFrame for model
            feature_df = pd.DataFrame([features])
            
            # Generate prediction
            prediction = self.model.predict(feature_df)[0]
            
            # Update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.total_predictions += 1
            self.metrics.successful_predictions += 1
            self.metrics.update_latency(latency_ms)
            
            if latency_ms > self.config.alert_latency_threshold_ms:
                logger.warning(f"High latency for {symbol}: {latency_ms:.2f}ms")
            
            return prediction, latency_ms
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.total_predictions += 1
            self.metrics.failed_predictions += 1
            self.metrics.recent_errors.append(str(e))
            
            # Keep only recent errors
            if len(self.metrics.recent_errors) > 100:
                self.metrics.recent_errors = self.metrics.recent_errors[-100:]
            
            logger.error(f"Prediction failed for {symbol}: {e}")
            raise
    
    def predict_batch(
        self,
        symbols: List[str],
        features: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None
    ) -> Tuple[pd.Series, float]:
        """
        Generate batch predictions with latency optimization.
        
        Args:
            symbols: List of symbols to predict
            features: Precomputed features DataFrame (optional)  
            timestamp: Prediction timestamp
            
        Returns:
            Tuple of (predictions Series, total_latency_ms)
        """
        start_time = time.perf_counter()
        timestamp = timestamp or datetime.now()
        
        try:
            # Get or compute features for all symbols
            if features is None:
                features = self._get_features_for_symbols(symbols, timestamp)
            
            # Validate batch features
            if not self._validate_batch_features(features, symbols):
                raise ValueError("Invalid batch features")
            
            # Optimize batch processing
            predictions = []
            batch_size = self.optimizer.optimize_batch_size(len(symbols))
            
            # Process in optimized batches
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                batch_features = features.loc[batch_symbols]
                
                # Generate batch predictions
                batch_preds = self.model.predict(batch_features)
                predictions.extend(batch_preds)
            
            # Convert to Series
            prediction_series = pd.Series(predictions, index=symbols)
            
            # Update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.total_predictions += len(symbols)
            self.metrics.successful_predictions += len(symbols)
            self.metrics.update_latency(latency_ms)
            
            if latency_ms > self.config.max_latency_ms:
                logger.warning(f"Batch latency exceeded target: {latency_ms:.2f}ms for {len(symbols)} symbols")
            
            logger.debug(f"Batch prediction completed: {len(symbols)} symbols in {latency_ms:.2f}ms")
            
            return prediction_series, latency_ms
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.total_predictions += len(symbols)
            self.metrics.failed_predictions += len(symbols)
            self.metrics.recent_errors.append(str(e))
            
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    async def predict_async(
        self,
        symbols: List[str],
        features: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
        priority: int = 0
    ) -> pd.Series:
        """
        Asynchronous batch prediction with priority queueing.
        
        Args:
            symbols: Symbols to predict
            features: Precomputed features (optional)
            timestamp: Prediction timestamp
            priority: Request priority (higher = more urgent)
            
        Returns:
            Predictions Series
        """
        timestamp = timestamp or datetime.now()
        
        # Create prediction batch
        batch = PredictionBatch(
            batch_id=f"batch_{timestamp.strftime('%H%M%S')}_{len(symbols)}",
            symbols=symbols,
            features=features,
            timestamp=timestamp,
            priority=priority
        )
        
        # Queue the batch
        await self._prediction_queue.put(batch)
        
        # Process immediately if not running async processor
        if not self._is_running:
            return await self._process_batch(batch)
        
        # In production, would return future/result
        logger.info(f"Queued batch {batch.batch_id} with {len(symbols)} symbols")
        return await self._process_batch(batch)
    
    async def _process_batch(self, batch: PredictionBatch) -> pd.Series:
        """Process a single prediction batch."""
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            predictions, latency = await loop.run_in_executor(
                self.executor,
                lambda: self.predict_batch(batch.symbols, batch.features, batch.timestamp)
            )
            
            logger.debug(f"Processed batch {batch.batch_id} in {latency:.2f}ms")
            return predictions
            
        except Exception as e:
            logger.error(f"Batch processing failed for {batch.batch_id}: {e}")
            raise
    
    def _get_features_for_symbol(
        self, 
        symbol: str, 
        timestamp: datetime
    ) -> pd.Series:
        """Get features for single symbol."""
        try:
            # Use factor engine to compute features
            features = self.factor_engine.compute_factors_for_symbol(
                symbol, timestamp, self.model.feature_columns
            )
            return features
            
        except Exception as e:
            logger.error(f"Feature computation failed for {symbol}: {e}")
            # Return zero features as fallback
            return pd.Series(0.0, index=self.model.feature_columns)
    
    def _get_features_for_symbols(
        self,
        symbols: List[str],
        timestamp: datetime
    ) -> pd.DataFrame:
        """Get features for multiple symbols efficiently."""
        try:
            # Use factor engine batch computation
            features = self.factor_engine.compute_factors_batch(
                symbols, timestamp, self.model.feature_columns
            )
            
            # Apply preprocessing optimizations
            features = self.optimizer.preprocess_features_vectorized(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Batch feature computation failed: {e}")
            # Return zero features as fallback
            return pd.DataFrame(
                0.0, 
                index=symbols, 
                columns=self.model.feature_columns
            )
    
    def _validate_features(self, features: pd.Series) -> bool:
        """Validate single symbol features."""
        if features is None or len(features) == 0:
            return False
            
        # Check for required features
        missing_features = self._feature_columns - set(features.index)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            return False
            
        # Check for invalid values
        if features.isnull().any() or np.isinf(features).any():
            logger.warning("Features contain NaN or infinite values")
            return False
            
        return True
    
    def _validate_batch_features(
        self, 
        features: pd.DataFrame, 
        symbols: List[str]
    ) -> bool:
        """Validate batch features."""
        if features is None or features.empty:
            return False
            
        # Check dimensions
        if len(features) != len(symbols):
            logger.warning(f"Feature count mismatch: {len(features)} vs {len(symbols)}")
            return False
            
        # Check for required features
        missing_features = self._feature_columns - set(features.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            return False
            
        return True
    
    def warmup(self, n_symbols: int = 100) -> Dict[str, float]:
        """
        Warm up the prediction system with dummy data.
        
        Args:
            n_symbols: Number of dummy symbols for warmup
            
        Returns:
            Warmup performance metrics
        """
        logger.info(f"Starting warmup with {n_symbols} symbols")
        
        # Create dummy symbols and features
        dummy_symbols = [f"DUMMY_{i:04d}" for i in range(n_symbols)]
        dummy_features = pd.DataFrame(
            np.random.randn(n_symbols, len(self.model.feature_columns)),
            index=dummy_symbols,
            columns=self.model.feature_columns
        )
        
        # Perform warmup predictions
        start_time = time.perf_counter()
        predictions, latency = self.predict_batch(dummy_symbols, dummy_features)
        total_time = time.perf_counter() - start_time
        
        warmup_metrics = {
            'warmup_symbols': n_symbols,
            'total_time_ms': total_time * 1000,
            'prediction_latency_ms': latency,
            'throughput_symbols_per_sec': n_symbols / total_time,
            'ready_for_production': latency < self.config.max_latency_ms
        }
        
        logger.info(f"Warmup completed: {warmup_metrics}")
        return warmup_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        self.metrics.update_throughput()
        
        summary = self.metrics.get_summary()
        summary.update({
            'config': {
                'max_latency_ms': self.config.max_latency_ms,
                'target_latency_ms': self.config.target_latency_ms,
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers
            },
            'model_info': {
                'n_features': self._n_features,
                'feature_columns_cached': bool(self._feature_columns),
                'important_features_count': len(self._important_features)
            },
            'health_status': {
                'is_healthy': summary['success_rate'] > 0.95 and summary['avg_latency_ms'] < self.config.max_latency_ms,
                'latency_sla_compliance': summary['p95_latency_ms'] < self.config.max_latency_ms,
                'error_rate_acceptable': summary['error_rate'] < 0.05
            }
        })
        
        return summary
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = InferenceMetrics()
        logger.info("Performance metrics reset")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Utility functions for Taiwan market specific inference

def is_market_hours(timestamp: datetime, config: InferenceConfig) -> bool:
    """Check if timestamp is during Taiwan market trading hours."""
    # Taiwan market: 09:00-13:30 TST
    hour = timestamp.hour
    minute = timestamp.minute
    
    if hour < config.market_open_hour:
        return False
    elif hour > config.market_close_hour:
        return False
    elif hour == config.market_close_hour and minute > config.market_close_minute:
        return False
    
    return True


def calculate_market_timing_features(
    timestamp: datetime,
    config: InferenceConfig
) -> Dict[str, float]:
    """Calculate Taiwan market timing features for inference."""
    hour = timestamp.hour
    minute = timestamp.minute
    
    # Convert to minutes since market open
    market_open_minutes = config.market_open_hour * 60
    current_minutes = hour * 60 + minute
    market_close_minutes = config.market_close_hour * 60 + config.market_close_minute
    
    if current_minutes < market_open_minutes:
        # Pre-market
        time_to_open = market_open_minutes - current_minutes
        return {
            'market_session': 0.0,  # Pre-market
            'session_progress': 0.0,
            'time_to_open': time_to_open,
            'time_to_close': market_close_minutes - market_open_minutes + time_to_open
        }
    elif current_minutes > market_close_minutes:
        # Post-market
        return {
            'market_session': 2.0,  # Post-market
            'session_progress': 1.0,
            'time_to_open': 24 * 60 - current_minutes + market_open_minutes,  # Next day
            'time_to_close': 0.0
        }
    else:
        # During market hours
        minutes_since_open = current_minutes - market_open_minutes
        total_market_minutes = market_close_minutes - market_open_minutes
        progress = minutes_since_open / total_market_minutes
        
        return {
            'market_session': 1.0,  # Active market
            'session_progress': progress,
            'time_to_open': 0.0,
            'time_to_close': market_close_minutes - current_minutes
        }


async def run_inference_server(
    predictor: RealtimePredictor,
    symbols: List[str],
    update_frequency_seconds: int = 60
) -> None:
    """
    Run continuous inference server for Taiwan market.
    
    Args:
        predictor: Real-time predictor instance
        symbols: List of symbols to monitor
        update_frequency_seconds: Update frequency in seconds
    """
    logger.info(f"Starting inference server for {len(symbols)} symbols")
    
    predictor._is_running = True
    
    try:
        while predictor._is_running:
            current_time = datetime.now()
            
            # Only run predictions during or near market hours
            if is_market_hours(current_time, predictor.config):
                try:
                    # Generate predictions for all symbols
                    predictions, latency = predictor.predict_batch(symbols)
                    
                    logger.info(
                        f"Generated {len(predictions)} predictions in {latency:.2f}ms "
                        f"at {current_time.strftime('%H:%M:%S')}"
                    )
                    
                    # Here you would typically:
                    # 1. Store predictions in database
                    # 2. Send to trading system
                    # 3. Update monitoring dashboards
                    
                except Exception as e:
                    logger.error(f"Inference cycle failed: {e}")
            else:
                logger.debug("Outside market hours, skipping prediction cycle")
            
            # Wait for next update
            await asyncio.sleep(update_frequency_seconds)
            
    except KeyboardInterrupt:
        logger.info("Inference server stopping...")
    finally:
        predictor._is_running = False
        logger.info("Inference server stopped")


# Demo and testing functions
def demo_realtime_prediction():
    """Demonstration of real-time prediction capabilities."""
    print("Real-time prediction demo - requires trained model and factor engine")
    
    # This would be used with actual trained model
    config = InferenceConfig(
        max_latency_ms=50.0,
        batch_size=100,
        enable_async=True
    )
    
    print(f"Demo config: {config}")
    print("In actual usage:")
    print("1. Initialize with trained LightGBM model")
    print("2. Set up factor engine for feature computation")
    print("3. Call predict_batch() or predict_async() for real-time inference")


if __name__ == "__main__":
    demo_realtime_prediction()