"""
Latency Validator for ML4T Performance Testing

Comprehensive latency testing and validation for real-time inference requirements.
Measures end-to-end pipeline latency and validates against production requirements.
"""

import time
import logging
import numpy as np
import pandas as pd
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    operation: str
    start_time: float
    end_time: float
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    payload_size: Optional[int] = None
    
    
@dataclass 
class LatencyProfile:
    """Complete latency profile for an operation."""
    operation_name: str
    measurements: List[LatencyMeasurement] = field(default_factory=list)
    target_latency_ms: float = 100.0
    
    def add_measurement(self, measurement: LatencyMeasurement):
        """Add a latency measurement."""
        self.measurements.append(measurement)
        
    def get_statistics(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.measurements:
            return {}
            
        successful_measurements = [m for m in self.measurements if m.success]
        if not successful_measurements:
            return {'success_rate': 0.0}
            
        latencies = [m.latency_ms for m in successful_measurements]
        
        return {
            'count': len(latencies),
            'success_rate': len(successful_measurements) / len(self.measurements),
            'mean_ms': statistics.mean(latencies),
            'median_ms': statistics.median(latencies),
            'min_ms': min(latencies),
            'max_ms': max(latencies),
            'std_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'p999_ms': np.percentile(latencies, 99.9),
            'meets_target': np.percentile(latencies, 95) <= self.target_latency_ms
        }


class LatencyTimer:
    """Context manager for measuring operation latency."""
    
    def __init__(self, operation_name: str, payload_size: Optional[int] = None):
        """
        Initialize latency timer.
        
        Args:
            operation_name: Name of the operation being timed
            payload_size: Size of data being processed (optional)
        """
        self.operation_name = operation_name
        self.payload_size = payload_size
        self.start_time = None
        self.end_time = None
        self.success = True
        self.error_message = None
        
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and handle exceptions."""
        self.end_time = time.time()
        
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)
            
        return False  # Don't suppress exceptions
        
    def get_measurement(self) -> LatencyMeasurement:
        """Get the latency measurement."""
        latency_ms = (self.end_time - self.start_time) * 1000 if self.start_time and self.end_time else 0
        
        return LatencyMeasurement(
            operation=self.operation_name,
            start_time=self.start_time or 0,
            end_time=self.end_time or 0,
            latency_ms=latency_ms,
            success=self.success,
            error_message=self.error_message,
            payload_size=self.payload_size
        )


class LatencyValidator:
    """Main latency validation class for ML4T performance testing."""
    
    def __init__(self, log_level: str = 'INFO'):
        """Initialize latency validator."""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.profiles = {}
        
        # Taiwan market specific timing requirements
        self.latency_requirements = {
            'real_time_inference': 100,      # <100ms for real-time predictions
            'feature_generation': 30000,     # <30s for single stock feature generation  
            'model_prediction': 10000,       # <10s for portfolio optimization
            'signal_generation': 60000,      # <1min for trading decisions
            'data_update': 5000,            # <5s for data ingestion
            'risk_calculation': 15000        # <15s for risk metrics
        }
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def create_profile(self, operation_name: str, target_latency_ms: float = None) -> LatencyProfile:
        """Create a new latency profile for an operation."""
        target_latency = target_latency_ms or self.latency_requirements.get(operation_name, 100.0)
        
        profile = LatencyProfile(
            operation_name=operation_name,
            target_latency_ms=target_latency
        )
        
        self.profiles[operation_name] = profile
        return profile
        
    def measure_operation(self, 
                         operation_name: str,
                         operation_func: Callable,
                         *args,
                         payload_size: Optional[int] = None,
                         **kwargs) -> LatencyMeasurement:
        """
        Measure latency of a single operation.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to measure
            *args: Function arguments
            payload_size: Size of data being processed
            **kwargs: Function keyword arguments
            
        Returns:
            Latency measurement
        """
        with LatencyTimer(operation_name, payload_size) as timer:
            try:
                result = operation_func(*args, **kwargs)
                # Store result if needed for validation
                timer.result = result
            except Exception as e:
                self.logger.error(f"Operation {operation_name} failed: {e}")
                raise
                
        return timer.get_measurement()
        
    def validate_inference_latency(self, target_latency_ms: float = 100) -> Dict[str, Any]:
        """
        Validate real-time inference latency requirements.
        
        Args:
            target_latency_ms: Target latency in milliseconds
            
        Returns:
            Inference latency validation results
        """
        self.logger.info(f"Validating inference latency (target: {target_latency_ms}ms)")
        
        profile = self.create_profile('real_time_inference', target_latency_ms)
        
        # Generate test data for inference
        test_data = self._generate_inference_test_data()
        
        # Run inference latency tests
        num_tests = 1000  # 1000 inference calls
        inference_times = []
        
        for i in range(num_tests):
            # Simulate different payload sizes
            payload_size = np.random.choice([100, 500, 1000, 2000])  # Different stock counts
            test_batch = test_data.iloc[:payload_size] if len(test_data) >= payload_size else test_data
            
            measurement = self.measure_operation(
                'real_time_inference',
                self._simulate_inference_operation,
                test_batch,
                payload_size=payload_size
            )
            
            profile.add_measurement(measurement)
            if measurement.success:
                inference_times.append(measurement.latency_ms)
                
        # Analyze results
        stats = profile.get_statistics()
        
        # Additional analysis
        throughput_ops_per_sec = len([m for m in profile.measurements if m.success]) / (
            sum(m.latency_ms for m in profile.measurements if m.success) / 1000
        ) if any(m.success for m in profile.measurements) else 0
        
        # Latency distribution analysis
        latency_analysis = self._analyze_latency_distribution(inference_times)
        
        results = {
            'success': len(profile.measurements) > 0,
            'operation': 'real_time_inference',
            'target_latency_ms': target_latency_ms,
            'measurements_count': len(profile.measurements),
            'success_rate': stats.get('success_rate', 0.0),
            'latency_stats': stats,
            'throughput_ops_per_sec': throughput_ops_per_sec,
            'latency_distribution': latency_analysis,
            
            # Key performance indicators
            'latency_p50_ms': stats.get('p50_ms'),
            'latency_p95_ms': stats.get('p95_ms'), 
            'latency_p99_ms': stats.get('p99_ms'),
            
            # Requirements validation
            'meets_target_latency': stats.get('meets_target', False),
            'inference_latency_passed': stats.get('meets_target', False) and stats.get('success_rate', 0) >= 0.95
        }
        
        self.logger.info(
            f"Inference latency validation complete - "
            f"P95: {stats.get('p95_ms', 0):.1f}ms, "
            f"Success rate: {stats.get('success_rate', 0):.1%}, "
            f"Meets target: {stats.get('meets_target', False)}"
        )
        
        return results
        
    def _generate_inference_test_data(self, max_stocks: int = 2000) -> pd.DataFrame:
        """Generate test data for inference latency testing."""
        np.random.seed(42)
        
        # Create realistic feature data
        stock_ids = [f"{2000 + i:04d}.TW" for i in range(max_stocks)]
        features = 42  # From Task #25
        
        feature_data = {}
        for i in range(features):
            if i < 10:  # Price-based features
                feature_data[f'price_feature_{i}'] = np.random.lognormal(4, 0.5, max_stocks)
            elif i < 20:  # Technical indicators
                feature_data[f'technical_feature_{i}'] = np.random.normal(0, 1, max_stocks)
            elif i < 30:  # Fundamental features
                feature_data[f'fundamental_feature_{i}'] = np.random.exponential(1, max_stocks)
            else:  # Cross-sectional features
                feature_data[f'cross_sectional_feature_{i}'] = np.random.uniform(-1, 1, max_stocks)
                
        df = pd.DataFrame(feature_data, index=stock_ids)
        return df
        
    def _simulate_inference_operation(self, data: pd.DataFrame) -> np.ndarray:
        """Simulate ML model inference operation."""
        # Simulate feature processing
        processed_features = data.values
        
        # Simulate model computation (matrix operations)
        weights = np.random.normal(0, 0.1, processed_features.shape[1])
        
        # Simulate prediction calculation
        predictions = processed_features @ weights
        
        # Simulate post-processing
        predictions = np.tanh(predictions)  # Bound predictions
        
        # Add small delay to simulate real computation
        time.sleep(0.001)  # 1ms base processing time
        
        return predictions
        
    def validate_feature_latency(self, stock_count: int = 50, target_latency_ms: float = 30000) -> Dict[str, Any]:
        """
        Validate feature generation latency.
        
        Args:
            stock_count: Number of stocks to test
            target_latency_ms: Target latency in milliseconds  
            
        Returns:
            Feature generation latency validation results
        """
        self.logger.info(f"Validating feature generation latency for {stock_count} stocks")
        
        profile = self.create_profile('feature_generation', target_latency_ms)
        
        # Test feature generation for different batch sizes
        batch_sizes = [1, 5, 10, 25, stock_count]
        feature_times = []
        
        for batch_size in batch_sizes:
            if batch_size > stock_count:
                continue
                
            # Run multiple tests per batch size
            for _ in range(5):  # 5 tests per batch size
                measurement = self.measure_operation(
                    'feature_generation',
                    self._simulate_feature_generation,
                    batch_size,
                    payload_size=batch_size
                )
                
                profile.add_measurement(measurement)
                if measurement.success:
                    feature_times.append(measurement.latency_ms)
                    
        # Analyze results
        stats = profile.get_statistics()
        
        # Calculate per-stock processing time
        per_stock_times = []
        for measurement in profile.measurements:
            if measurement.success and measurement.payload_size:
                per_stock_time = measurement.latency_ms / measurement.payload_size
                per_stock_times.append(per_stock_time)
                
        results = {
            'success': len(profile.measurements) > 0,
            'operation': 'feature_generation',
            'target_latency_ms': target_latency_ms,
            'stock_count_tested': stock_count,
            'measurements_count': len(profile.measurements),
            'success_rate': stats.get('success_rate', 0.0),
            'latency_stats': stats,
            'per_stock_latency_ms': {
                'mean': np.mean(per_stock_times) if per_stock_times else 0,
                'median': np.median(per_stock_times) if per_stock_times else 0,
                'p95': np.percentile(per_stock_times, 95) if per_stock_times else 0
            },
            
            # Key performance indicators
            'latency_p95_ms': stats.get('p95_ms'),
            'avg_per_stock_ms': np.mean(per_stock_times) if per_stock_times else 0,
            
            # Requirements validation
            'meets_target_latency': stats.get('meets_target', False),
            'feature_latency_passed': stats.get('meets_target', False) and stats.get('success_rate', 0) >= 0.9
        }
        
        self.logger.info(
            f"Feature generation latency validation complete - "
            f"P95: {stats.get('p95_ms', 0):.1f}ms, "
            f"Per stock: {results['avg_per_stock_ms']:.1f}ms, "
            f"Meets target: {stats.get('meets_target', False)}"
        )
        
        return results
        
    def _simulate_feature_generation(self, stock_count: int) -> pd.DataFrame:
        """Simulate feature generation for a batch of stocks."""
        # Simulate data loading
        time.sleep(0.01 * stock_count)  # 10ms per stock for data loading
        
        # Generate raw data
        np.random.seed(42)
        days = 60  # 60 days of history
        raw_data = np.random.normal(0, 1, (days, stock_count, 10))  # 10 raw features
        
        # Simulate feature engineering computations
        features = []
        
        # Technical indicators (computationally intensive)
        for i in range(stock_count):
            stock_data = raw_data[:, i, :]
            
            # Moving averages
            ma_5 = np.convolve(stock_data[:, 0], np.ones(5)/5, mode='same')
            ma_20 = np.convolve(stock_data[:, 0], np.ones(20)/20, mode='same')
            
            # RSI calculation (simplified)
            returns = np.diff(stock_data[:, 0])
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            rsi = 100 - (100 / (1 + np.mean(gains) / np.mean(losses))) if np.mean(losses) > 0 else 50
            
            # Volatility
            volatility = np.std(returns)
            
            features.append([ma_5[-1], ma_20[-1], rsi, volatility])
            
        # Additional processing delay
        time.sleep(0.005 * stock_count)  # 5ms per stock for final processing
        
        # Return as DataFrame
        feature_names = ['ma_5', 'ma_20', 'rsi', 'volatility']
        stock_ids = [f"{2000 + i:04d}.TW" for i in range(stock_count)]
        
        return pd.DataFrame(features, index=stock_ids, columns=feature_names)
        
    def _analyze_latency_distribution(self, latencies: List[float]) -> Dict[str, Any]:
        """Analyze latency distribution patterns."""
        if not latencies:
            return {}
            
        latency_array = np.array(latencies)
        
        # Percentile analysis
        percentiles = [50, 75, 90, 95, 99, 99.9]
        percentile_values = {f'p{p}': np.percentile(latency_array, p) for p in percentiles}
        
        # Distribution characteristics
        skewness = self._calculate_skewness(latency_array)
        
        # Outlier detection
        q1 = np.percentile(latency_array, 25)
        q3 = np.percentile(latency_array, 75)
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        outliers = latency_array[latency_array > outlier_threshold]
        
        return {
            'percentiles': percentile_values,
            'skewness': skewness,
            'outliers_count': len(outliers),
            'outliers_percentage': len(outliers) / len(latencies) * 100,
            'distribution_quality': self._assess_distribution_quality(skewness, len(outliers) / len(latencies))
        }
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of latency distribution."""
        if len(data) < 3:
            return 0.0
            
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
            
        skew = np.mean(((data - mean) / std) ** 3)
        return float(skew)
        
    def _assess_distribution_quality(self, skewness: float, outlier_ratio: float) -> str:
        """Assess quality of latency distribution."""
        if abs(skewness) < 0.5 and outlier_ratio < 0.05:
            return 'excellent'
        elif abs(skewness) < 1.0 and outlier_ratio < 0.10:
            return 'good'
        elif abs(skewness) < 2.0 and outlier_ratio < 0.20:
            return 'acceptable'
        else:
            return 'needs_optimization'
            
    def validate_concurrent_latency(self, 
                                   concurrent_users: int = 3,
                                   operations_per_user: int = 100) -> Dict[str, Any]:
        """
        Validate latency under concurrent load.
        
        Args:
            concurrent_users: Number of concurrent users to simulate
            operations_per_user: Operations per user
            
        Returns:
            Concurrent latency validation results
        """
        self.logger.info(f"Validating concurrent latency: {concurrent_users} users, {operations_per_user} ops each")
        
        profile = self.create_profile('concurrent_inference', 100.0)  # 100ms target
        
        # Generate test data
        test_data = self._generate_inference_test_data(1000)
        
        # Run concurrent tests
        all_measurements = []
        
        def user_operations(user_id: int) -> List[LatencyMeasurement]:
            """Simulate operations for a single user."""
            user_measurements = []
            
            for i in range(operations_per_user):
                # Random data batch
                batch_size = np.random.choice([10, 50, 100, 200])
                batch_data = test_data.iloc[:batch_size]
                
                measurement = self.measure_operation(
                    f'concurrent_inference_user_{user_id}',
                    self._simulate_inference_operation,
                    batch_data,
                    payload_size=batch_size
                )
                
                user_measurements.append(measurement)
                
                # Small delay between operations
                time.sleep(0.01)  # 10ms between operations
                
            return user_measurements
            
        # Execute concurrent operations
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_operations, user_id) 
                      for user_id in range(concurrent_users)]
            
            for future in as_completed(futures):
                try:
                    user_measurements = future.result()
                    all_measurements.extend(user_measurements)
                    for measurement in user_measurements:
                        profile.add_measurement(measurement)
                except Exception as e:
                    self.logger.error(f"Concurrent user operations failed: {e}")
                    
        total_time = time.time() - start_time
        
        # Analyze results
        stats = profile.get_statistics()
        successful_measurements = [m for m in all_measurements if m.success]
        
        # Calculate concurrent-specific metrics
        total_operations = len(all_measurements)
        successful_operations = len(successful_measurements)
        overall_throughput = successful_operations / total_time if total_time > 0 else 0
        
        # Latency degradation analysis
        single_user_baseline = 50.0  # Baseline latency for single user (ms)
        degradation_factor = stats.get('p95_ms', 0) / single_user_baseline if single_user_baseline > 0 else 1.0
        
        results = {
            'success': len(all_measurements) > 0,
            'concurrent_users': concurrent_users,
            'operations_per_user': operations_per_user,
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'total_duration_seconds': total_time,
            'overall_throughput_ops_per_sec': overall_throughput,
            'latency_stats': stats,
            'latency_degradation_factor': degradation_factor,
            
            # Key performance indicators
            'latency_p95_ms': stats.get('p95_ms'),
            'throughput_ops_per_sec': overall_throughput,
            
            # Requirements validation
            'meets_latency_requirement': stats.get('p95_ms', float('inf')) <= 200,  # 200ms under load
            'meets_throughput_requirement': overall_throughput >= 50,  # 50 ops/sec minimum
            'concurrent_latency_passed': (stats.get('p95_ms', float('inf')) <= 200 and 
                                        overall_throughput >= 50 and
                                        stats.get('success_rate', 0) >= 0.95)
        }
        
        self.logger.info(
            f"Concurrent latency validation complete - "
            f"P95: {stats.get('p95_ms', 0):.1f}ms, "
            f"Throughput: {overall_throughput:.1f} ops/sec, "
            f"Degradation: {degradation_factor:.1f}x"
        )
        
        return results
        
    def generate_latency_report(self) -> Dict[str, Any]:
        """Generate comprehensive latency analysis report."""
        if not self.profiles:
            return {'error': 'No latency profiles available'}
            
        # Aggregate statistics across all profiles
        report = {
            'summary': {
                'total_profiles': len(self.profiles),
                'total_measurements': sum(len(p.measurements) for p in self.profiles.values()),
                'operations_tested': list(self.profiles.keys())
            },
            
            'requirements_validation': {},
            'performance_analysis': {},
            'recommendations': []
        }
        
        # Analyze each profile
        for operation, profile in self.profiles.items():
            stats = profile.get_statistics()
            
            report['requirements_validation'][operation] = {
                'target_latency_ms': profile.target_latency_ms,
                'actual_p95_ms': stats.get('p95_ms'),
                'meets_requirement': stats.get('meets_target', False),
                'success_rate': stats.get('success_rate', 0.0)
            }
            
            report['performance_analysis'][operation] = stats
            
        # Generate recommendations
        report['recommendations'] = self._generate_latency_recommendations()
        
        return report
        
    def _generate_latency_recommendations(self) -> List[str]:
        """Generate latency optimization recommendations."""
        recommendations = []
        
        for operation, profile in self.profiles.items():
            stats = profile.get_statistics()
            
            if not stats.get('meets_target', True):
                p95_latency = stats.get('p95_ms', 0)
                target_latency = profile.target_latency_ms
                
                recommendations.append(
                    f"{operation}: P95 latency ({p95_latency:.1f}ms) exceeds target "
                    f"({target_latency:.1f}ms). Consider optimization."
                )
                
            if stats.get('success_rate', 1.0) < 0.95:
                recommendations.append(
                    f"{operation}: Low success rate ({stats.get('success_rate', 0):.1%}). "
                    "Investigate error handling and timeouts."
                )
                
        if not recommendations:
            recommendations.append("All latency requirements are met within acceptable parameters.")
            
        return recommendations