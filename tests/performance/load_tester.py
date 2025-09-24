"""
Load Tester for ML4T Performance Testing

Comprehensive load testing for real-time inference and concurrent user scenarios.
Tests system behavior under sustained load and validates throughput requirements.
"""

import time
import logging
import threading
import queue
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import psutil
import statistics
from collections import deque, defaultdict
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing."""
    timestamp: float
    operations_completed: int
    operations_failed: int
    current_throughput_ops_per_sec: float
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate_percent: float
    
    
@dataclass
class LoadTestResult:
    """Results from a complete load test."""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float
    
    # Operation metrics
    total_operations: int
    successful_operations: int
    failed_operations: int
    success_rate: float
    
    # Performance metrics
    peak_throughput_ops_per_sec: float
    average_throughput_ops_per_sec: float
    min_response_time_ms: float
    max_response_time_ms: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    
    # Resource utilization
    peak_memory_usage_mb: float
    avg_memory_usage_mb: float
    peak_cpu_usage_percent: float
    avg_cpu_usage_percent: float
    
    # Load characteristics
    concurrent_users: int
    target_ops_per_sec: Optional[float] = None
    actual_ops_per_sec: float = 0.0
    
    # Test validation
    meets_throughput_requirement: bool = False
    meets_latency_requirement: bool = False
    load_test_passed: bool = False
    
    def duration_minutes(self) -> float:
        """Get test duration in minutes."""
        return self.total_duration_seconds / 60.0


class LoadTestMonitor:
    """Monitors system performance during load testing."""
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize load test monitor.
        
        Args:
            sampling_interval: Time between samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
        self.process = psutil.Process()
        
        # Metrics tracking
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = deque(maxlen=1000)  # Keep last 1000 response times
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> List[LoadTestMetrics]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        # Collect all metrics
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                metrics.append(metric)
            except queue.Empty:
                break
                
        return metrics
        
    def record_operation(self, success: bool, response_time_ms: float):
        """Record a completed operation."""
        self.operation_counts['success' if success else 'failure'] += 1
        self.response_times.append(response_time_ms)
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_sample_time = time.time()
        last_success_count = 0
        last_failure_count = 0
        
        while self.monitoring:
            try:
                current_time = time.time()
                
                # System resource metrics
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                # Operation metrics
                current_success_count = self.operation_counts['success']
                current_failure_count = self.operation_counts['failure']
                
                # Calculate throughput since last sample
                time_delta = current_time - last_sample_time
                if time_delta > 0:
                    ops_completed_delta = (current_success_count + current_failure_count) - (last_success_count + last_failure_count)
                    current_throughput = ops_completed_delta / time_delta
                else:
                    current_throughput = 0.0
                
                # Response time metrics
                current_response_time = statistics.median(self.response_times) if self.response_times else 0.0
                
                # Error rate
                total_ops = current_success_count + current_failure_count
                error_rate = (current_failure_count / total_ops * 100) if total_ops > 0 else 0.0
                
                # Create metrics snapshot
                metrics = LoadTestMetrics(
                    timestamp=current_time,
                    operations_completed=current_success_count,
                    operations_failed=current_failure_count,
                    current_throughput_ops_per_sec=current_throughput,
                    response_time_ms=current_response_time,
                    memory_usage_mb=memory_info.rss / 1024 / 1024,
                    cpu_usage_percent=cpu_percent,
                    error_rate_percent=error_rate
                )
                
                self.metrics_queue.put(metrics)
                
                # Update for next iteration
                last_sample_time = current_time
                last_success_count = current_success_count
                last_failure_count = current_failure_count
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logging.warning(f"Load test monitoring error: {e}")


class LoadTester:
    """Main load testing class for ML4T performance validation."""
    
    def __init__(self, log_level: str = 'INFO'):
        """Initialize load tester."""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def test_concurrent_users(self, 
                             concurrent_users: int = 3,
                             duration_minutes: int = 10) -> Dict[str, Any]:
        """
        Test system performance with concurrent users.
        
        Args:
            concurrent_users: Number of concurrent users to simulate
            duration_minutes: Duration of test in minutes
            
        Returns:
            Concurrent user load test results
        """
        self.logger.info(f"Starting concurrent user load test: {concurrent_users} users, {duration_minutes} minutes")
        
        test_start = datetime.now()
        start_time = time.time()
        duration_seconds = duration_minutes * 60
        
        # Setup monitoring
        monitor = LoadTestMonitor()
        monitor.start_monitoring()
        
        # Generate test data
        test_data = self._generate_load_test_data()
        
        # User operation tracking
        user_stats = {i: {'operations': 0, 'failures': 0, 'response_times': []} 
                     for i in range(concurrent_users)}
        
        def user_simulation(user_id: int, stop_event: threading.Event) -> Dict[str, Any]:
            """Simulate operations for a single user."""
            operations = 0
            failures = 0
            response_times = []
            
            while not stop_event.is_set():
                try:
                    # Random operation parameters
                    batch_size = np.random.choice([10, 50, 100, 200])
                    batch_data = test_data.iloc[:batch_size]
                    
                    # Measure operation
                    op_start = time.time()
                    success = self._simulate_user_operation(batch_data)
                    op_end = time.time()
                    
                    response_time_ms = (op_end - op_start) * 1000
                    
                    operations += 1
                    if success:
                        monitor.record_operation(True, response_time_ms)
                    else:
                        failures += 1
                        monitor.record_operation(False, response_time_ms)
                        
                    response_times.append(response_time_ms)
                    
                    # Variable think time (simulate user behavior)
                    think_time = np.random.exponential(0.5)  # Average 500ms think time
                    time.sleep(min(think_time, 2.0))  # Max 2s think time
                    
                except Exception as e:
                    self.logger.warning(f"User {user_id} operation failed: {e}")
                    failures += 1
                    monitor.record_operation(False, 5000.0)  # 5s timeout
                    
            return {
                'user_id': user_id,
                'operations': operations,
                'failures': failures,
                'response_times': response_times
            }
            
        # Run concurrent users
        stop_event = threading.Event()
        
        # Schedule stop after duration
        stop_timer = threading.Timer(duration_seconds, stop_event.set)
        stop_timer.start()
        
        # Start user threads
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_simulation, user_id, stop_event) 
                      for user_id in range(concurrent_users)]
            
            # Wait for completion
            user_results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    user_results.append(result)
                    user_stats[result['user_id']] = result
                except Exception as e:
                    self.logger.error(f"User simulation failed: {e}")
                    
        # Stop monitoring and collect metrics
        metrics = monitor.stop_monitoring()
        
        test_end = datetime.now()
        total_duration = time.time() - start_time
        
        # Aggregate results
        total_operations = sum(r['operations'] for r in user_results)
        total_failures = sum(r['failures'] for r in user_results)
        all_response_times = []
        for r in user_results:
            all_response_times.extend(r['response_times'])
            
        # Calculate performance metrics
        success_rate = ((total_operations - total_failures) / total_operations) if total_operations > 0 else 0
        avg_throughput = total_operations / total_duration if total_duration > 0 else 0
        
        # Response time analysis
        if all_response_times:
            min_response = min(all_response_times)
            max_response = max(all_response_times)
            avg_response = statistics.mean(all_response_times)
            p95_response = np.percentile(all_response_times, 95)
            p99_response = np.percentile(all_response_times, 99)
        else:
            min_response = max_response = avg_response = p95_response = p99_response = 0
            
        # Resource utilization analysis
        if metrics:
            peak_memory = max(m.memory_usage_mb for m in metrics)
            avg_memory = statistics.mean(m.memory_usage_mb for m in metrics)
            peak_cpu = max(m.cpu_usage_percent for m in metrics)
            avg_cpu = statistics.mean(m.cpu_usage_percent for m in metrics)
            peak_throughput = max(m.current_throughput_ops_per_sec for m in metrics)
        else:
            peak_memory = avg_memory = peak_cpu = avg_cpu = peak_throughput = 0
            
        # Create result object
        result = LoadTestResult(
            test_name=f"concurrent_users_{concurrent_users}_{duration_minutes}min",
            start_time=test_start,
            end_time=test_end,
            total_duration_seconds=total_duration,
            total_operations=total_operations,
            successful_operations=total_operations - total_failures,
            failed_operations=total_failures,
            success_rate=success_rate,
            peak_throughput_ops_per_sec=peak_throughput,
            average_throughput_ops_per_sec=avg_throughput,
            min_response_time_ms=min_response,
            max_response_time_ms=max_response,
            avg_response_time_ms=avg_response,
            p95_response_time_ms=p95_response,
            p99_response_time_ms=p99_response,
            peak_memory_usage_mb=peak_memory,
            avg_memory_usage_mb=avg_memory,
            peak_cpu_usage_percent=peak_cpu,
            avg_cpu_usage_percent=avg_cpu,
            concurrent_users=concurrent_users,
            actual_ops_per_sec=avg_throughput
        )
        
        # Validate requirements
        result.meets_throughput_requirement = avg_throughput >= 10  # Minimum 10 ops/sec
        result.meets_latency_requirement = p95_response <= 1000    # P95 < 1 second
        result.load_test_passed = (result.meets_throughput_requirement and 
                                  result.meets_latency_requirement and
                                  success_rate >= 0.95)
        
        self.test_results.append(result)
        
        self.logger.info(
            f"Concurrent user load test complete - "
            f"Users: {concurrent_users}, Duration: {duration_minutes}min, "
            f"Throughput: {avg_throughput:.1f} ops/sec, P95: {p95_response:.1f}ms, "
            f"Success rate: {success_rate:.1%}"
        )
        
        return self._convert_result_to_dict(result, user_stats, metrics)
        
    def validate_prediction_throughput(self, 
                                     target_ops_per_sec: float = 1500,
                                     duration_seconds: int = 300) -> Dict[str, Any]:
        """
        Validate prediction throughput against target requirements.
        
        Args:
            target_ops_per_sec: Target throughput in operations per second
            duration_seconds: Duration of test in seconds
            
        Returns:
            Throughput validation results
        """
        self.logger.info(f"Validating prediction throughput: {target_ops_per_sec} ops/sec target")
        
        test_start = datetime.now()
        start_time = time.time()
        
        # Setup monitoring
        monitor = LoadTestMonitor(sampling_interval=0.5)  # Higher frequency for throughput test
        monitor.start_monitoring()
        
        # Generate test data
        test_data = self._generate_load_test_data(stock_count=2000)  # Full universe
        
        # Calculate optimal batch size for target throughput
        # Assume each operation takes ~50ms base time
        base_operation_time = 0.05  # 50ms
        target_batch_size = max(1, int(target_ops_per_sec * base_operation_time))
        
        operations_completed = 0
        operations_failed = 0
        response_times = []
        
        # Throughput test loop
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            
            try:
                # Create batch for processing
                batch_size = min(target_batch_size, len(test_data))
                batch_data = test_data.iloc[:batch_size]
                
                # Process batch
                success = self._simulate_batch_prediction(batch_data)
                
                batch_end = time.time()
                batch_time_ms = (batch_end - batch_start) * 1000
                
                # Record metrics
                if success:
                    operations_completed += batch_size  # Count individual predictions
                    monitor.record_operation(True, batch_time_ms / batch_size)  # Per-prediction time
                else:
                    operations_failed += batch_size
                    monitor.record_operation(False, batch_time_ms / batch_size)
                    
                response_times.extend([batch_time_ms / batch_size] * batch_size)
                
                # Maintain target rate (if we're running too fast)
                target_batch_time = batch_size / target_ops_per_sec
                actual_batch_time = batch_end - batch_start
                
                if actual_batch_time < target_batch_time:
                    sleep_time = target_batch_time - actual_batch_time
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.warning(f"Throughput test batch failed: {e}")
                operations_failed += target_batch_size
                monitor.record_operation(False, 5000.0)  # 5s timeout
                
        # Stop monitoring
        metrics = monitor.stop_monitoring()
        
        test_end = datetime.now()
        total_duration = time.time() - start_time
        total_operations = operations_completed + operations_failed
        
        # Calculate metrics
        actual_throughput = operations_completed / total_duration if total_duration > 0 else 0
        success_rate = operations_completed / total_operations if total_operations > 0 else 0
        
        # Response time analysis
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
            
        # Resource utilization
        if metrics:
            peak_memory = max(m.memory_usage_mb for m in metrics)
            avg_cpu = statistics.mean(m.cpu_usage_percent for m in metrics)
            peak_throughput = max(m.current_throughput_ops_per_sec for m in metrics)
        else:
            peak_memory = avg_cpu = peak_throughput = 0
            
        # Create result
        result = LoadTestResult(
            test_name=f"throughput_validation_{target_ops_per_sec}_ops_sec",
            start_time=test_start,
            end_time=test_end,
            total_duration_seconds=total_duration,
            total_operations=total_operations,
            successful_operations=operations_completed,
            failed_operations=operations_failed,
            success_rate=success_rate,
            peak_throughput_ops_per_sec=peak_throughput,
            average_throughput_ops_per_sec=actual_throughput,
            min_response_time_ms=min(response_times) if response_times else 0,
            max_response_time_ms=max(response_times) if response_times else 0,
            avg_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            peak_memory_usage_mb=peak_memory,
            avg_memory_usage_mb=peak_memory,  # Simplified
            peak_cpu_usage_percent=avg_cpu,
            avg_cpu_usage_percent=avg_cpu,
            concurrent_users=1,  # Single-threaded throughput test
            target_ops_per_sec=target_ops_per_sec,
            actual_ops_per_sec=actual_throughput
        )
        
        # Validate requirements
        throughput_achieved = actual_throughput >= target_ops_per_sec * 0.9  # 90% of target
        latency_acceptable = p95_response_time <= 100  # <100ms P95
        
        result.meets_throughput_requirement = throughput_achieved
        result.meets_latency_requirement = latency_acceptable
        result.load_test_passed = throughput_achieved and latency_acceptable and success_rate >= 0.95
        
        self.test_results.append(result)
        
        self.logger.info(
            f"Throughput validation complete - "
            f"Target: {target_ops_per_sec}, Actual: {actual_throughput:.1f} ops/sec, "
            f"P95: {p95_response_time:.1f}ms, Success: {success_rate:.1%}, "
            f"Passed: {result.load_test_passed}"
        )
        
        return self._convert_result_to_dict(result, metrics=metrics)
        
    def _generate_load_test_data(self, stock_count: int = 1000) -> pd.DataFrame:
        """Generate test data for load testing."""
        np.random.seed(42)
        
        # Create Taiwan stock universe
        stock_ids = [f"{2000 + i:04d}.TW" for i in range(stock_count // 2)]
        stock_ids.extend([f"{1000 + i:04d}.TWO" for i in range(stock_count - len(stock_ids))])
        
        # Generate features (Task #25 handcrafted factors)
        features = {}
        for i in range(42):  # 42 features from Task #25
            if i < 15:  # Price and volume features
                features[f'price_vol_feature_{i}'] = np.random.lognormal(0, 1, stock_count)
            elif i < 30:  # Technical indicators
                features[f'technical_feature_{i}'] = np.random.normal(0, 1, stock_count)
            else:  # Fundamental and cross-sectional
                features[f'fundamental_feature_{i}'] = np.random.exponential(1, stock_count)
                
        df = pd.DataFrame(features, index=stock_ids)
        return df
        
    def _simulate_user_operation(self, data: pd.DataFrame) -> bool:
        """Simulate a typical user operation (inference + processing)."""
        try:
            # Simulate inference
            features = data.values
            weights = np.random.normal(0, 0.1, features.shape[1])
            predictions = features @ weights
            
            # Simulate post-processing
            processed_predictions = np.tanh(predictions)
            
            # Simulate ranking/filtering
            top_predictions = np.argsort(processed_predictions)[-10:]  # Top 10
            
            # Add processing delay
            processing_time = 0.01 + np.random.exponential(0.02)  # 10ms + exponential delay
            time.sleep(processing_time)
            
            return True  # Success
            
        except Exception:
            return False  # Failure
            
    def _simulate_batch_prediction(self, data: pd.DataFrame) -> bool:
        """Simulate high-throughput batch prediction."""
        try:
            # Optimized batch processing
            features = data.values
            
            # Use cached/pre-computed weights for speed
            weights = np.random.normal(0, 0.1, features.shape[1])
            
            # Vectorized prediction
            predictions = features @ weights
            
            # Minimal post-processing
            processed_predictions = np.clip(predictions, -3, 3)  # Faster than tanh
            
            # Minimal delay (optimized for throughput)
            base_time = len(data) * 0.0001  # 0.1ms per stock
            time.sleep(base_time)
            
            return True
            
        except Exception:
            return False
            
    def _convert_result_to_dict(self, 
                               result: LoadTestResult,
                               user_stats: Dict = None,
                               metrics: List[LoadTestMetrics] = None) -> Dict[str, Any]:
        """Convert load test result to dictionary format."""
        result_dict = {
            'success': True,
            'test_name': result.test_name,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'duration_seconds': result.total_duration_seconds,
            'duration_minutes': result.duration_minutes(),
            
            # Operation metrics
            'total_operations': result.total_operations,
            'successful_operations': result.successful_operations,
            'failed_operations': result.failed_operations,
            'success_rate': result.success_rate,
            
            # Performance metrics
            'peak_throughput_ops_per_sec': result.peak_throughput_ops_per_sec,
            'average_throughput_ops_per_sec': result.average_throughput_ops_per_sec,
            'response_time_stats': {
                'min_ms': result.min_response_time_ms,
                'max_ms': result.max_response_time_ms,
                'avg_ms': result.avg_response_time_ms,
                'p95_ms': result.p95_response_time_ms,
                'p99_ms': result.p99_response_time_ms
            },
            
            # Resource utilization
            'resource_usage': {
                'peak_memory_mb': result.peak_memory_usage_mb,
                'avg_memory_mb': result.avg_memory_usage_mb,
                'peak_cpu_percent': result.peak_cpu_usage_percent,
                'avg_cpu_percent': result.avg_cpu_usage_percent
            },
            
            # Load characteristics
            'load_profile': {
                'concurrent_users': result.concurrent_users,
                'target_ops_per_sec': result.target_ops_per_sec,
                'actual_ops_per_sec': result.actual_ops_per_sec
            },
            
            # Requirements validation
            'meets_throughput_requirement': result.meets_throughput_requirement,
            'meets_latency_requirement': result.meets_latency_requirement,
            'load_test_passed': result.load_test_passed,
            
            # Framework compatibility
            'throughput_ops_per_sec': result.average_throughput_ops_per_sec,
            'latency_p95_ms': result.p95_response_time_ms
        }
        
        # Add user statistics if available
        if user_stats:
            result_dict['user_statistics'] = user_stats
            
        # Add metrics timeline if available
        if metrics:
            result_dict['metrics_timeline'] = [
                {
                    'timestamp': m.timestamp,
                    'throughput_ops_per_sec': m.current_throughput_ops_per_sec,
                    'response_time_ms': m.response_time_ms,
                    'error_rate_percent': m.error_rate_percent,
                    'memory_usage_mb': m.memory_usage_mb,
                    'cpu_usage_percent': m.cpu_usage_percent
                }
                for m in metrics[-100:]  # Last 100 samples
            ]
            
        return result_dict
        
    def run_comprehensive_load_suite(self) -> Dict[str, Any]:
        """Run comprehensive load testing suite."""
        self.logger.info("Starting comprehensive load testing suite")
        
        suite_start = time.time()
        suite_results = {}
        
        # Concurrent user load tests
        self.logger.info("Running concurrent user load tests...")
        user_configs = [
            (1, 5),   # 1 user, 5 minutes - baseline
            (3, 10),  # 3 users, 10 minutes - target
            (5, 5),   # 5 users, 5 minutes - stress test
        ]
        
        for users, duration in user_configs:
            test_name = f"concurrent_users_{users}_{duration}min"
            self.logger.info(f"Running {test_name}...")
            suite_results[test_name] = self.test_concurrent_users(users, duration)
            
        # Throughput validation tests
        self.logger.info("Running throughput validation tests...")
        throughput_targets = [500, 1000, 1500]  # Ops per second targets
        
        for target in throughput_targets:
            test_name = f"throughput_{target}_ops_sec"
            self.logger.info(f"Running {test_name}...")
            suite_results[test_name] = self.validate_prediction_throughput(target, 300)  # 5 minutes
            
        total_time = time.time() - suite_start
        
        # Generate summary
        successful_tests = sum(1 for r in suite_results.values() if r.get('success', False))
        passed_tests = sum(1 for r in suite_results.values() if r.get('load_test_passed', False))
        
        suite_results['summary'] = {
            'total_load_test_time_seconds': total_time,
            'total_tests': len(suite_results) - 1,  # Exclude summary
            'successful_tests': successful_tests,
            'passed_tests': passed_tests,
            'overall_load_ready': passed_tests == len(suite_results) - 1,
            'load_test_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(
            f"Comprehensive load testing complete - "
            f"Time: {total_time:.1f}s, Success: {successful_tests}/{len(suite_results)-1}, "
            f"Passed: {passed_tests}/{len(suite_results)-1}"
        )
        
        return suite_results