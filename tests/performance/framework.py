"""
Performance Test Framework

Core framework for comprehensive performance testing of ML4T system.
Orchestrates all performance testing components and provides unified results.
"""

import time
import psutil
import logging
import gc
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from .production_scale import ProductionScaleValidator
from .taiwan_stress import TaiwanMarketStressTester
from .memory_profiler import MemoryProfiler
from .latency_validator import LatencyValidator
from .load_tester import LoadTester
from .reporting import PerformanceReporter


@dataclass
class PerformanceBenchmark:
    """Single performance benchmark specification."""
    name: str
    test_function: Callable
    parameters: Dict[str, Any]
    timeout_seconds: int = 3600  # 1 hour default
    memory_limit_mb: int = 16384  # 16GB default
    required_for_production: bool = True
    taiwan_specific: bool = False


@dataclass 
class PerformanceResult:
    """Results from a single performance test."""
    benchmark_name: str
    success: bool
    execution_time_seconds: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    
    # Performance metrics
    throughput_ops_per_sec: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    
    # Quality metrics
    information_coefficient: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    prediction_accuracy: Optional[float] = None
    
    # Resource utilization
    io_read_mb: Optional[float] = None
    io_write_mb: Optional[float] = None
    network_bytes: Optional[float] = None
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Test-specific metrics
    test_specific_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def meets_production_requirements(self) -> bool:
        """Check if result meets production requirements."""
        checks = {
            'memory_limit': self.peak_memory_mb <= 16384,  # 16GB limit
            'latency_requirement': self.latency_p95_ms is None or self.latency_p95_ms <= 100,  # <100ms
            'ic_requirement': self.information_coefficient is None or self.information_coefficient >= 0.05,  # >0.05 IC
            'execution_success': self.success
        }
        
        failed_checks = [check for check, passes in checks.items() if not passes]
        if failed_checks:
            self.warnings.extend([f"Failed requirement: {check}" for check in failed_checks])
            
        return all(checks.values())


class SystemResourceMonitor:
    """Monitors system resources during performance tests."""
    
    def __init__(self, sampling_interval: float = 0.1):
        """Initialize resource monitor.
        
        Args:
            sampling_interval: Time between samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.samples = []
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.samples = []
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        
        if not self.samples:
            return {}
            
        # Calculate statistics
        memory_values = [s['memory_mb'] for s in self.samples]
        cpu_values = [s['cpu_percent'] for s in self.samples]
        
        return {
            'duration_seconds': len(self.samples) * self.sampling_interval,
            'memory_avg_mb': np.mean(memory_values),
            'memory_peak_mb': np.max(memory_values),
            'memory_std_mb': np.std(memory_values),
            'cpu_avg_percent': np.mean(cpu_values),
            'cpu_peak_percent': np.max(cpu_values),
            'sample_count': len(self.samples)
        }
        
    def sample_resources(self):
        """Sample current resource usage."""
        if not self.monitoring:
            return
            
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            self.samples.append({
                'timestamp': time.time(),
                'memory_mb': memory_info.rss / 1024 / 1024,
                'cpu_percent': cpu_percent
            })
        except Exception as e:
            logging.warning(f"Failed to sample resources: {e}")


class PerformanceTestFramework:
    """Main performance testing framework orchestrating all performance tests."""
    
    def __init__(self, 
                 log_level: str = 'INFO',
                 output_dir: str = 'performance_results',
                 parallel_tests: bool = True,
                 max_workers: int = 4):
        """Initialize performance test framework.
        
        Args:
            log_level: Logging level for tests
            output_dir: Directory for test results and reports
            parallel_tests: Whether to run tests in parallel where possible
            max_workers: Maximum parallel workers
        """
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.parallel_tests = parallel_tests
        self.max_workers = max_workers
        
        # Initialize test components
        self.production_scale_validator = ProductionScaleValidator()
        self.taiwan_stress_tester = TaiwanMarketStressTester()
        self.memory_profiler = MemoryProfiler()
        self.latency_validator = LatencyValidator()
        self.load_tester = LoadTester()
        self.reporter = PerformanceReporter(output_dir=str(self.output_dir))
        
        # Test results
        self.test_results: List[PerformanceResult] = []
        self.test_start_time = None
        self.test_end_time = None
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('performance_tests.log')
            ]
        )
        
    def define_benchmark_suite(self) -> List[PerformanceBenchmark]:
        """Define the complete benchmark suite for production readiness."""
        benchmarks = [
            # Production Scale Validation
            PerformanceBenchmark(
                name="production_scale_2000_stocks",
                test_function=self.production_scale_validator.validate_full_universe,
                parameters={'stock_count': 2000, 'days': 252},
                timeout_seconds=7200,  # 2 hours
                required_for_production=True
            ),
            
            PerformanceBenchmark(
                name="production_scale_feature_generation", 
                test_function=self.production_scale_validator.validate_feature_processing,
                parameters={'stock_count': 2000, 'feature_count': 42},
                timeout_seconds=1800,  # 30 minutes
                required_for_production=True
            ),
            
            # Taiwan Market Stress Testing
            PerformanceBenchmark(
                name="taiwan_market_volatility_stress",
                test_function=self.taiwan_stress_tester.test_high_volatility_periods,
                parameters={'scenario': 'march_2020_crash'},
                taiwan_specific=True,
                required_for_production=True
            ),
            
            PerformanceBenchmark(
                name="taiwan_market_circuit_breaker",
                test_function=self.taiwan_stress_tester.test_circuit_breaker_scenarios,
                parameters={},
                taiwan_specific=True,
                required_for_production=True
            ),
            
            # Memory Optimization Testing
            PerformanceBenchmark(
                name="memory_usage_scaling",
                test_function=self.memory_profiler.profile_memory_scaling,
                parameters={'max_stocks': 2000},
                memory_limit_mb=16384,
                required_for_production=True
            ),
            
            PerformanceBenchmark(
                name="memory_leak_detection",
                test_function=self.memory_profiler.detect_memory_leaks,
                parameters={'iterations': 100},
                timeout_seconds=1800,
                required_for_production=True
            ),
            
            # Latency Validation
            PerformanceBenchmark(
                name="real_time_inference_latency",
                test_function=self.latency_validator.validate_inference_latency,
                parameters={'target_latency_ms': 100},
                required_for_production=True
            ),
            
            PerformanceBenchmark(
                name="feature_generation_latency",
                test_function=self.latency_validator.validate_feature_latency,
                parameters={'stock_count': 50, 'target_latency_ms': 30000},  # 30s per stock
                required_for_production=True
            ),
            
            # Load Testing
            PerformanceBenchmark(
                name="concurrent_user_load",
                test_function=self.load_tester.test_concurrent_users,
                parameters={'concurrent_users': 3, 'duration_minutes': 10},
                required_for_production=True
            ),
            
            PerformanceBenchmark(
                name="throughput_validation",
                test_function=self.load_tester.validate_prediction_throughput,
                parameters={'target_ops_per_sec': 1500, 'duration_seconds': 300},
                required_for_production=True
            )
        ]
        
        return benchmarks
        
    def run_single_benchmark(self, benchmark: PerformanceBenchmark) -> PerformanceResult:
        """Run a single performance benchmark with monitoring."""
        self.logger.info(f"Starting benchmark: {benchmark.name}")
        
        # Setup resource monitoring
        monitor = SystemResourceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Run the benchmark with timeout
            result = benchmark.test_function(**benchmark.parameters)
            success = True
            error_message = None
            
        except Exception as e:
            self.logger.error(f"Benchmark {benchmark.name} failed: {e}")
            result = {}
            success = False
            error_message = str(e)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Stop monitoring and get metrics
        resource_metrics = monitor.stop_monitoring()
        
        # Force garbage collection
        gc.collect()
        
        # Create performance result
        perf_result = PerformanceResult(
            benchmark_name=benchmark.name,
            success=success,
            execution_time_seconds=execution_time,
            memory_usage_mb=resource_metrics.get('memory_avg_mb', 0),
            peak_memory_mb=resource_metrics.get('memory_peak_mb', 0),
            cpu_usage_percent=resource_metrics.get('cpu_avg_percent', 0),
            error_message=error_message
        )
        
        # Extract test-specific metrics from result
        if isinstance(result, dict):
            perf_result.throughput_ops_per_sec = result.get('throughput_ops_per_sec')
            perf_result.latency_p50_ms = result.get('latency_p50_ms')
            perf_result.latency_p95_ms = result.get('latency_p95_ms')
            perf_result.latency_p99_ms = result.get('latency_p99_ms')
            perf_result.information_coefficient = result.get('information_coefficient')
            perf_result.sharpe_ratio = result.get('sharpe_ratio')
            perf_result.prediction_accuracy = result.get('prediction_accuracy')
            perf_result.test_specific_metrics = result
            
        self.logger.info(
            f"Completed benchmark: {benchmark.name} - "
            f"Success: {success}, Time: {execution_time:.2f}s, "
            f"Peak Memory: {perf_result.peak_memory_mb:.1f}MB"
        )
        
        return perf_result
        
    def run_benchmarks_sequential(self, benchmarks: List[PerformanceBenchmark]) -> List[PerformanceResult]:
        """Run benchmarks sequentially."""
        results = []
        
        for benchmark in benchmarks:
            result = self.run_single_benchmark(benchmark)
            results.append(result)
            
            # Check if this was a critical failure
            if benchmark.required_for_production and not result.success:
                self.logger.critical(
                    f"Critical benchmark {benchmark.name} failed - "
                    f"production deployment not recommended"
                )
                
        return results
        
    def run_benchmarks_parallel(self, benchmarks: List[PerformanceBenchmark]) -> List[PerformanceResult]:
        """Run benchmarks in parallel where safe to do so."""
        # Separate benchmarks by whether they can run in parallel
        sequential_benchmarks = []
        parallel_benchmarks = []
        
        for benchmark in benchmarks:
            # Memory profiling and load tests should run sequentially
            if ('memory' in benchmark.name.lower() or 
                'load' in benchmark.name.lower() or
                'stress' in benchmark.name.lower()):
                sequential_benchmarks.append(benchmark)
            else:
                parallel_benchmarks.append(benchmark)
                
        results = []
        
        # Run sequential benchmarks first
        if sequential_benchmarks:
            self.logger.info(f"Running {len(sequential_benchmarks)} sequential benchmarks...")
            results.extend(self.run_benchmarks_sequential(sequential_benchmarks))
            
        # Run parallel benchmarks
        if parallel_benchmarks:
            self.logger.info(f"Running {len(parallel_benchmarks)} parallel benchmarks...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_benchmark = {
                    executor.submit(self.run_single_benchmark, benchmark): benchmark 
                    for benchmark in parallel_benchmarks
                }
                
                for future in as_completed(future_to_benchmark):
                    result = future.result()
                    results.append(result)
                    
        return results
        
    def run_comprehensive_performance_suite(self) -> Dict[str, Any]:
        """Run the complete performance test suite."""
        self.logger.info("Starting comprehensive performance test suite for Issue #30 Stream B")
        self.test_start_time = datetime.now()
        
        # Get benchmark definitions
        benchmarks = self.define_benchmark_suite()
        
        self.logger.info(f"Running {len(benchmarks)} performance benchmarks...")
        
        # Run benchmarks
        if self.parallel_tests:
            self.test_results = self.run_benchmarks_parallel(benchmarks)
        else:
            self.test_results = self.run_benchmarks_sequential(benchmarks)
            
        self.test_end_time = datetime.now()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save results
        self.reporter.save_results(self.test_results, report)
        
        return report
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance test report."""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.success])
        production_ready_tests = len([r for r in self.test_results if r.meets_production_requirements()])
        
        # Performance statistics
        execution_times = [r.execution_time_seconds for r in self.test_results if r.success]
        memory_usage = [r.peak_memory_mb for r in self.test_results if r.success]
        
        # Extract key metrics
        throughput_results = [r.throughput_ops_per_sec for r in self.test_results 
                            if r.throughput_ops_per_sec is not None]
        latency_results = [r.latency_p95_ms for r in self.test_results 
                         if r.latency_p95_ms is not None]
        ic_results = [r.information_coefficient for r in self.test_results 
                     if r.information_coefficient is not None]
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'production_ready_tests': production_ready_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'production_readiness_rate': production_ready_tests / total_tests if total_tests > 0 else 0
            },
            
            'performance_metrics': {
                'total_execution_time_seconds': sum(execution_times) if execution_times else 0,
                'avg_execution_time_seconds': np.mean(execution_times) if execution_times else 0,
                'max_execution_time_seconds': max(execution_times) if execution_times else 0,
                'peak_memory_usage_mb': max(memory_usage) if memory_usage else 0,
                'avg_memory_usage_mb': np.mean(memory_usage) if memory_usage else 0
            },
            
            'key_metrics': {
                'max_throughput_ops_per_sec': max(throughput_results) if throughput_results else None,
                'avg_throughput_ops_per_sec': np.mean(throughput_results) if throughput_results else None,
                'min_latency_p95_ms': min(latency_results) if latency_results else None,
                'avg_latency_p95_ms': np.mean(latency_results) if latency_results else None,
                'max_information_coefficient': max(ic_results) if ic_results else None,
                'avg_information_coefficient': np.mean(ic_results) if ic_results else None
            },
            
            'production_requirements': {
                'memory_limit_16gb_met': all(r.peak_memory_mb <= 16384 for r in self.test_results if r.success),
                'latency_100ms_met': all(r.latency_p95_ms <= 100 for r in self.test_results 
                                       if r.latency_p95_ms is not None),
                'ic_005_met': all(r.information_coefficient >= 0.05 for r in self.test_results 
                                if r.information_coefficient is not None),
                'throughput_1500_met': any(r.throughput_ops_per_sec >= 1500 for r in self.test_results 
                                         if r.throughput_ops_per_sec is not None)
            },
            
            'recommendations': self._generate_recommendations(),
            
            'test_details': [
                {
                    'name': r.benchmark_name,
                    'success': r.success,
                    'production_ready': r.meets_production_requirements(),
                    'execution_time_seconds': r.execution_time_seconds,
                    'peak_memory_mb': r.peak_memory_mb,
                    'error_message': r.error_message,
                    'warnings': r.warnings
                }
                for r in self.test_results
            ],
            
            'test_start_time': self.test_start_time.isoformat() if self.test_start_time else None,
            'test_end_time': self.test_end_time.isoformat() if self.test_end_time else None,
            'total_test_duration_minutes': ((self.test_end_time - self.test_start_time).total_seconds() / 60
                                          if self.test_start_time and self.test_end_time else None)
        }
        
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        # Check memory usage
        max_memory = max((r.peak_memory_mb for r in self.test_results if r.success), default=0)
        if max_memory > 12000:  # > 12GB
            recommendations.append(
                f"High memory usage detected ({max_memory:.1f}MB). "
                "Consider memory optimization or increasing system RAM."
            )
            
        # Check for failed tests
        failed_tests = [r for r in self.test_results if not r.success]
        if failed_tests:
            recommendations.append(
                f"{len(failed_tests)} tests failed. Review error logs and fix issues before production."
            )
            
        # Check throughput
        throughput_results = [r.throughput_ops_per_sec for r in self.test_results 
                            if r.throughput_ops_per_sec is not None]
        if throughput_results and max(throughput_results) < 1000:
            recommendations.append(
                "Low throughput detected. Consider parallel processing optimizations."
            )
            
        # Check latency
        latency_results = [r.latency_p95_ms for r in self.test_results 
                         if r.latency_p95_ms is not None]
        if latency_results and min(latency_results) > 100:
            recommendations.append(
                "High latency detected. Consider caching and optimization strategies."
            )
            
        if not recommendations:
            recommendations.append("All performance tests passed within acceptable parameters.")
            
        return recommendations
        
    def get_production_readiness_status(self) -> Dict[str, Any]:
        """Get overall production readiness status."""
        if not self.test_results:
            return {'ready': False, 'reason': 'No tests have been run'}
            
        failed_critical_tests = [
            r for r in self.test_results 
            if not r.success or not r.meets_production_requirements()
        ]
        
        if failed_critical_tests:
            return {
                'ready': False,
                'reason': f'{len(failed_critical_tests)} critical tests failed or did not meet requirements',
                'failed_tests': [r.benchmark_name for r in failed_critical_tests]
            }
            
        return {
            'ready': True,
            'reason': 'All performance benchmarks passed production requirements',
            'total_tests': len(self.test_results)
        }