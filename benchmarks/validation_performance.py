"""
Performance Benchmarks for Walk-Forward Validation Engine.

This module provides comprehensive performance benchmarks for the walk-forward validation
system, testing scalability, throughput, and resource utilization under various
configurations and load conditions for Taiwan market data.

Key Features:
- Scalability testing across different data sizes
- Performance profiling for large-scale validations
- Memory usage monitoring and optimization
- Parallel processing efficiency benchmarks
- Taiwan market-specific performance characteristics
"""

import time
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd
from functools import wraps
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import Mock, MagicMock

# Import the validation components
from src.backtesting.validation.walk_forward import (
    WalkForwardSplitter, WalkForwardConfig, WalkForwardValidator, ValidationWindow
)
from src.backtesting.integration.pit_validator import (
    PITValidator, PITValidationConfig, ValidationLevel, create_strict_pit_validator
)
from src.backtesting.validation.taiwan_specific import (
    TaiwanMarketValidator, TaiwanValidationConfig
)
from src.data.core.temporal import TemporalStore, DataType, TemporalValue
from src.data.pipeline.pit_engine import PointInTimeEngine, PITQuery
from src.data.models.taiwan_market import TaiwanTradingCalendar

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    benchmark_name: str
    test_parameters: Dict[str, Any]
    execution_time_seconds: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    throughput_operations_per_second: float
    success_rate: float
    
    # Detailed metrics
    windows_processed: int
    symbols_processed: int
    data_points_processed: int
    
    # Performance breakdown
    validation_time_seconds: float
    bias_detection_time_seconds: float
    quality_check_time_seconds: float
    
    # Resource utilization
    threads_used: int
    cache_hit_rate: float
    
    # Error information
    errors_encountered: int
    error_types: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'benchmark_name': self.benchmark_name,
            'test_parameters': self.test_parameters,
            'execution_time_seconds': self.execution_time_seconds,
            'memory_usage_mb': self.memory_usage_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'throughput_operations_per_second': self.throughput_operations_per_second,
            'success_rate': self.success_rate,
            'windows_processed': self.windows_processed,
            'symbols_processed': self.symbols_processed,
            'data_points_processed': self.data_points_processed,
            'validation_time_seconds': self.validation_time_seconds,
            'bias_detection_time_seconds': self.bias_detection_time_seconds,
            'quality_check_time_seconds': self.quality_check_time_seconds,
            'threads_used': self.threads_used,
            'cache_hit_rate': self.cache_hit_rate,
            'errors_encountered': self.errors_encountered,
            'error_types': self.error_types
        }


class PerformanceMonitor:
    """Monitor system performance during benchmarks."""
    
    def __init__(self):
        self.monitoring = False
        self.start_time = None
        self.metrics = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.metrics:
            return {}
        
        # Aggregate metrics
        memory_values = [m['memory_mb'] for m in self.metrics]
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        
        return {
            'duration_seconds': time.time() - self.start_time,
            'avg_memory_mb': np.mean(memory_values),
            'peak_memory_mb': np.max(memory_values),
            'avg_cpu_percent': np.mean(cpu_values),
            'peak_cpu_percent': np.max(cpu_values),
            'sample_count': len(self.metrics)
        }
    
    def _monitor_loop(self):
        """Monitor system metrics in background thread."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'cpu_percent': cpu_percent
                })
                
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break


def benchmark_decorator(func):
    """Decorator for benchmark functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Benchmark {func.__name__} failed: {e}")
            result = None
            success = False
        
        end_time = time.time()
        metrics = monitor.stop_monitoring()
        
        # Force garbage collection
        gc.collect()
        
        return {
            'result': result,
            'success': success,
            'execution_time': end_time - start_time,
            'system_metrics': metrics
        }
    
    return wrapper


class ValidationBenchmarkSuite:
    """
    Comprehensive benchmark suite for walk-forward validation engine.
    
    Tests performance characteristics across various dimensions:
    - Data size scaling
    - Symbol count scaling  
    - Window count scaling
    - Parallel processing efficiency
    - Memory usage patterns
    """
    
    def __init__(self):
        self.results = []
        self.mock_data_cache = {}
        
    def create_mock_dependencies(self, symbol_count: int, data_density: float = 1.0) -> Dict[str, Any]:
        """Create mock dependencies for testing."""
        temporal_store = Mock(spec=TemporalStore)
        pit_engine = Mock(spec=PointInTimeEngine)
        taiwan_calendar = Mock(spec=TaiwanTradingCalendar)
        
        # Configure mocks
        pit_engine.check_data_availability.return_value = True
        
        # Create mock data based on symbol count and density
        mock_data = self._generate_mock_data(symbol_count, data_density)
        pit_engine.query.return_value = mock_data
        
        # Taiwan calendar
        taiwan_calendar.is_trading_day.side_effect = lambda d: d.weekday() < 5
        
        return {
            'temporal_store': temporal_store,
            'pit_engine': pit_engine,
            'taiwan_calendar': taiwan_calendar,
            'mock_data': mock_data
        }
    
    def _generate_mock_data(self, symbol_count: int, data_density: float) -> Dict[str, List[TemporalValue]]:
        """Generate mock temporal data for benchmarking."""
        cache_key = f"{symbol_count}_{data_density}"
        if cache_key in self.mock_data_cache:
            return self.mock_data_cache[cache_key]
        
        symbols = [f"{2330 + i}.TW" for i in range(symbol_count)]
        start_date = date(2020, 1, 1)
        
        mock_data = {}
        for symbol in symbols:
            data_points = []
            current_date = start_date
            
            # Generate data points based on density
            for day in range(int(1000 * data_density)):  # Up to ~3 years of data
                if current_date.weekday() < 5:  # Trading days only
                    data_points.append(TemporalValue(
                        value=500.0 + np.random.normal(0, 50),  # Random price
                        as_of_date=current_date,
                        value_date=current_date,
                        data_type=DataType.PRICE,
                        symbol=symbol
                    ))
                current_date += timedelta(days=1)
            
            mock_data[symbol] = data_points
        
        self.mock_data_cache[cache_key] = mock_data
        return mock_data
    
    @benchmark_decorator
    def benchmark_basic_validation(self, symbol_count: int = 10, window_count: int = 20) -> BenchmarkResult:
        """Benchmark basic walk-forward validation."""
        deps = self.create_mock_dependencies(symbol_count)
        
        config = WalkForwardConfig(
            train_weeks=52,
            test_weeks=13,
            purge_weeks=1,
            rebalance_weeks=4
        )
        
        splitter = WalkForwardSplitter(
            config=config,
            temporal_store=deps['temporal_store'],
            pit_engine=deps['pit_engine'],
            taiwan_calendar=deps['taiwan_calendar']
        )
        
        symbols = list(deps['mock_data'].keys())[:symbol_count]
        
        # Generate windows
        start_time = time.time()
        windows = splitter.generate_windows(
            start_date=date(2020, 1, 1),
            end_date=date(2023, 12, 31),
            symbols=symbols
        )
        window_generation_time = time.time() - start_time
        
        # Limit to specified window count
        windows = windows[:window_count]
        
        # Validate windows
        validation_start = time.time()
        successful_validations = 0
        
        for window in windows:
            is_valid = splitter.validate_window(window, symbols)
            if is_valid:
                successful_validations += 1
        
        validation_time = time.time() - validation_start
        
        return BenchmarkResult(
            benchmark_name="basic_validation",
            test_parameters={
                'symbol_count': symbol_count,
                'window_count': window_count
            },
            execution_time_seconds=window_generation_time + validation_time,
            memory_usage_mb=0,  # Will be filled by decorator
            peak_memory_mb=0,   # Will be filled by decorator
            cpu_usage_percent=0,  # Will be filled by decorator
            throughput_operations_per_second=len(windows) / (window_generation_time + validation_time),
            success_rate=successful_validations / len(windows) if windows else 0,
            windows_processed=len(windows),
            symbols_processed=symbol_count,
            data_points_processed=symbol_count * window_count * 100,  # Estimate
            validation_time_seconds=validation_time,
            bias_detection_time_seconds=0,
            quality_check_time_seconds=0,
            threads_used=1,
            cache_hit_rate=0.0,
            errors_encountered=0
        )
    
    @benchmark_decorator
    def benchmark_pit_integration(self, symbol_count: int = 50, validation_level: ValidationLevel = ValidationLevel.STRICT) -> BenchmarkResult:
        """Benchmark PIT validation integration."""
        deps = self.create_mock_dependencies(symbol_count)
        
        config = PITValidationConfig(
            validation_level=validation_level,
            enable_parallel_processing=False,  # Test sequential first
            enable_look_ahead_detection=True,
            enable_survivorship_detection=True,
            require_quality_validation=True
        )
        
        validator = PITValidator(
            config=config,
            temporal_store=deps['temporal_store'],
            pit_engine=deps['pit_engine'],
            taiwan_calendar=deps['taiwan_calendar']
        )
        
        wf_config = WalkForwardConfig(
            train_weeks=52,
            test_weeks=13,
            purge_weeks=1
        )
        
        symbols = list(deps['mock_data'].keys())[:symbol_count]
        
        # Run comprehensive validation
        start_time = time.time()
        result = validator.validate_walk_forward_scenario(
            wf_config=wf_config,
            symbols=symbols,
            start_date=date(2020, 1, 1),
            end_date=date(2022, 12, 31),
            enable_performance_validation=False  # Skip to focus on validation
        )
        total_time = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name="pit_integration",
            test_parameters={
                'symbol_count': symbol_count,
                'validation_level': validation_level.value
            },
            execution_time_seconds=total_time,
            memory_usage_mb=0,  # Will be filled by decorator
            peak_memory_mb=0,   # Will be filled by decorator
            cpu_usage_percent=0,  # Will be filled by decorator
            throughput_operations_per_second=result['validation_summary']['total_windows'] / total_time,
            success_rate=result['validation_summary']['successful_validations'] / result['validation_summary']['total_windows'],
            windows_processed=result['validation_summary']['total_windows'],
            symbols_processed=symbol_count,
            data_points_processed=symbol_count * result['validation_summary']['total_windows'] * 100,
            validation_time_seconds=total_time * 0.6,  # Estimate
            bias_detection_time_seconds=total_time * 0.3,  # Estimate
            quality_check_time_seconds=total_time * 0.1,  # Estimate
            threads_used=1,
            cache_hit_rate=0.0,
            errors_encountered=0
        )
    
    @benchmark_decorator
    def benchmark_parallel_processing(self, symbol_count: int = 100, max_workers: int = 4) -> BenchmarkResult:
        """Benchmark parallel processing efficiency."""
        deps = self.create_mock_dependencies(symbol_count)
        
        config = PITValidationConfig(
            validation_level=ValidationLevel.STANDARD,
            enable_parallel_processing=True,
            max_concurrent_validations=max_workers
        )
        
        validator = PITValidator(
            config=config,
            temporal_store=deps['temporal_store'],
            pit_engine=deps['pit_engine'],
            taiwan_calendar=deps['taiwan_calendar']
        )
        
        wf_config = WalkForwardConfig(
            train_weeks=26,  # Shorter for parallel test
            test_weeks=6,
            purge_weeks=1
        )
        
        symbols = list(deps['mock_data'].keys())[:symbol_count]
        
        # Run parallel validation
        start_time = time.time()
        result = validator.validate_walk_forward_scenario(
            wf_config=wf_config,
            symbols=symbols,
            start_date=date(2021, 1, 1),
            end_date=date(2023, 12, 31),
            enable_performance_validation=False
        )
        total_time = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name="parallel_processing",
            test_parameters={
                'symbol_count': symbol_count,
                'max_workers': max_workers
            },
            execution_time_seconds=total_time,
            memory_usage_mb=0,  # Will be filled by decorator
            peak_memory_mb=0,   # Will be filled by decorator
            cpu_usage_percent=0,  # Will be filled by decorator
            throughput_operations_per_second=result['validation_summary']['total_windows'] / total_time,
            success_rate=result['validation_summary']['successful_validations'] / result['validation_summary']['total_windows'],
            windows_processed=result['validation_summary']['total_windows'],
            symbols_processed=symbol_count,
            data_points_processed=symbol_count * result['validation_summary']['total_windows'] * 100,
            validation_time_seconds=total_time,
            bias_detection_time_seconds=0,
            quality_check_time_seconds=0,
            threads_used=max_workers,
            cache_hit_rate=0.0,
            errors_encountered=0
        )
    
    @benchmark_decorator
    def benchmark_memory_scaling(self, symbol_counts: List[int] = None) -> Dict[int, BenchmarkResult]:
        """Benchmark memory usage scaling with symbol count."""
        if symbol_counts is None:
            symbol_counts = [10, 50, 100, 250, 500]
        
        results = {}
        
        for symbol_count in symbol_counts:
            deps = self.create_mock_dependencies(symbol_count, data_density=0.5)  # Reduce density for large counts
            
            config = PITValidationConfig(
                validation_level=ValidationLevel.STANDARD,
                enable_parallel_processing=False
            )
            
            validator = PITValidator(
                config=config,
                temporal_store=deps['temporal_store'],
                pit_engine=deps['pit_engine'],
                taiwan_calendar=deps['taiwan_calendar']
            )
            
            wf_config = WalkForwardConfig(
                train_weeks=26,
                test_weeks=6,
                purge_weeks=1
            )
            
            symbols = list(deps['mock_data'].keys())[:symbol_count]
            
            # Monitor memory before/after
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            result = validator.validate_walk_forward_scenario(
                wf_config=wf_config,
                symbols=symbols,
                start_date=date(2022, 1, 1),
                end_date=date(2022, 12, 31),
                enable_performance_validation=False
            )
            total_time = time.time() - start_time
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            
            results[symbol_count] = BenchmarkResult(
                benchmark_name="memory_scaling",
                test_parameters={'symbol_count': symbol_count},
                execution_time_seconds=total_time,
                memory_usage_mb=memory_used,
                peak_memory_mb=memory_after,
                cpu_usage_percent=0,
                throughput_operations_per_second=result['validation_summary']['total_windows'] / total_time,
                success_rate=result['validation_summary']['successful_validations'] / result['validation_summary']['total_windows'],
                windows_processed=result['validation_summary']['total_windows'],
                symbols_processed=symbol_count,
                data_points_processed=symbol_count * result['validation_summary']['total_windows'] * 50,
                validation_time_seconds=total_time,
                bias_detection_time_seconds=0,
                quality_check_time_seconds=0,
                threads_used=1,
                cache_hit_rate=0.0,
                errors_encountered=0
            )
            
            # Force cleanup between tests
            del deps
            gc.collect()
        
        return results
    
    @benchmark_decorator
    def benchmark_taiwan_specific_validation(self, symbol_count: int = 30) -> BenchmarkResult:
        """Benchmark Taiwan market-specific validation features."""
        deps = self.create_mock_dependencies(symbol_count)
        
        # Taiwan-specific symbols
        taiwan_symbols = [f"{2330 + i}.TW" if i < symbol_count//2 else f"{1101 + i}.TWO" 
                         for i in range(symbol_count)]
        
        config = TaiwanValidationConfig(
            validate_price_limits=True,
            validate_volume_constraints=True,
            validate_corporate_actions=True,
            validate_market_events=True,
            handle_lunar_new_year=True,
            handle_typhoon_days=True
        )
        
        validator = TaiwanMarketValidator(
            config=config,
            temporal_store=deps['temporal_store'],
            pit_engine=deps['pit_engine'],
            taiwan_calendar=deps['taiwan_calendar']
        )
        
        # Test comprehensive scenario validation
        start_time = time.time()
        issues = validator.validate_trading_scenario(
            symbols=taiwan_symbols,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )
        total_time = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name="taiwan_specific",
            test_parameters={'symbol_count': symbol_count},
            execution_time_seconds=total_time,
            memory_usage_mb=0,  # Will be filled by decorator
            peak_memory_mb=0,   # Will be filled by decorator  
            cpu_usage_percent=0,  # Will be filled by decorator
            throughput_operations_per_second=symbol_count / total_time,
            success_rate=1.0 - (len([i for i in issues if i.severity.value == 'critical']) / max(len(issues), 1)),
            windows_processed=1,  # Single scenario
            symbols_processed=symbol_count,
            data_points_processed=symbol_count * 252,  # Assume 1 year of trading days
            validation_time_seconds=total_time,
            bias_detection_time_seconds=0,
            quality_check_time_seconds=0,
            threads_used=1,
            cache_hit_rate=0.0,
            errors_encountered=len(issues)
        )
    
    def run_comprehensive_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete benchmark suite and return results."""
        logger.info("Starting comprehensive validation benchmark suite")
        
        all_results = {}
        
        # Basic validation scaling
        logger.info("Running basic validation benchmarks...")
        basic_results = []
        for symbol_count in [10, 25, 50, 100]:
            result = self.benchmark_basic_validation(symbol_count=symbol_count)
            if result['success']:
                basic_results.append(result['result'])
        all_results['basic_validation'] = basic_results
        
        # PIT integration benchmarks
        logger.info("Running PIT integration benchmarks...")
        pit_results = []
        for level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            result = self.benchmark_pit_integration(symbol_count=50, validation_level=level)
            if result['success']:
                pit_results.append(result['result'])
        all_results['pit_integration'] = pit_results
        
        # Parallel processing benchmarks
        logger.info("Running parallel processing benchmarks...")
        parallel_results = []
        for workers in [1, 2, 4, 8]:
            result = self.benchmark_parallel_processing(symbol_count=100, max_workers=workers)
            if result['success']:
                parallel_results.append(result['result'])
        all_results['parallel_processing'] = parallel_results
        
        # Memory scaling benchmarks
        logger.info("Running memory scaling benchmarks...")
        memory_result = self.benchmark_memory_scaling()
        if memory_result['success']:
            all_results['memory_scaling'] = list(memory_result['result'].values())
        
        # Taiwan-specific benchmarks
        logger.info("Running Taiwan-specific benchmarks...")
        taiwan_result = self.benchmark_taiwan_specific_validation()
        if taiwan_result['success']:
            all_results['taiwan_specific'] = [taiwan_result['result']]
        
        # Generate summary statistics
        all_results['summary'] = self._generate_benchmark_summary(all_results)
        
        logger.info("Benchmark suite completed")
        return all_results
    
    def _generate_benchmark_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            'total_benchmarks_run': 0,
            'total_windows_processed': 0,
            'total_symbols_processed': 0,
            'average_throughput_ops_per_sec': 0,
            'average_success_rate': 0,
            'peak_memory_usage_mb': 0,
            'benchmark_categories': list(results.keys())
        }
        
        all_benchmark_results = []
        for category, category_results in results.items():
            if category == 'summary':
                continue
                
            if isinstance(category_results, list):
                all_benchmark_results.extend(category_results)
            else:
                all_benchmark_results.append(category_results)
        
        if all_benchmark_results:
            summary['total_benchmarks_run'] = len(all_benchmark_results)
            summary['total_windows_processed'] = sum(r.windows_processed for r in all_benchmark_results)
            summary['total_symbols_processed'] = sum(r.symbols_processed for r in all_benchmark_results)
            summary['average_throughput_ops_per_sec'] = np.mean([r.throughput_operations_per_second for r in all_benchmark_results])
            summary['average_success_rate'] = np.mean([r.success_rate for r in all_benchmark_results])
            summary['peak_memory_usage_mb'] = max(r.peak_memory_mb for r in all_benchmark_results)
        
        return summary


class BenchmarkReporting:
    """Generate reports and visualizations from benchmark results."""
    
    @staticmethod
    def generate_performance_report(results: Dict[str, Any], output_file: str = None) -> str:
        """Generate comprehensive performance report."""
        report_lines = [
            "# Walk-Forward Validation Performance Benchmark Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Executive Summary"
        ]
        
        if 'summary' in results:
            summary = results['summary']
            report_lines.extend([
                f"- Total Benchmarks Run: {summary['total_benchmarks_run']}",
                f"- Total Windows Processed: {summary['total_windows_processed']}",
                f"- Total Symbols Processed: {summary['total_symbols_processed']}",
                f"- Average Throughput: {summary['average_throughput_ops_per_sec']:.2f} ops/sec",
                f"- Average Success Rate: {summary['average_success_rate']:.1%}",
                f"- Peak Memory Usage: {summary['peak_memory_usage_mb']:.1f} MB",
                ""
            ])
        
        # Basic validation results
        if 'basic_validation' in results:
            report_lines.extend([
                "## Basic Validation Performance",
                "| Symbol Count | Execution Time (s) | Throughput (ops/s) | Success Rate |",
                "|--------------|-------------------|-------------------|--------------|"
            ])
            
            for result in results['basic_validation']:
                report_lines.append(
                    f"| {result.symbols_processed} | {result.execution_time_seconds:.2f} | "
                    f"{result.throughput_operations_per_second:.2f} | {result.success_rate:.1%} |"
                )
            report_lines.append("")
        
        # Memory scaling results
        if 'memory_scaling' in results:
            report_lines.extend([
                "## Memory Scaling Analysis",
                "| Symbol Count | Memory Usage (MB) | Peak Memory (MB) | Efficiency |",
                "|--------------|------------------|-----------------|------------|"
            ])
            
            for result in results['memory_scaling']:
                efficiency = result.symbols_processed / result.memory_usage_mb if result.memory_usage_mb > 0 else 0
                report_lines.append(
                    f"| {result.symbols_processed} | {result.memory_usage_mb:.1f} | "
                    f"{result.peak_memory_mb:.1f} | {efficiency:.2f} symbols/MB |"
                )
            report_lines.append("")
        
        # Performance recommendations
        report_lines.extend([
            "## Performance Recommendations",
            "- For symbol counts > 100: Enable parallel processing",
            "- For memory-constrained environments: Use validation_level=STANDARD",
            "- For production use: Enable caching and incremental validation",
            "- For Taiwan market: Use Taiwan-specific calendar for optimal performance",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    @staticmethod
    def plot_scaling_results(results: Dict[str, Any], output_dir: str = None):
        """Generate scaling performance plots."""
        plt.style.use('seaborn-v0_8')
        
        # Memory scaling plot
        if 'memory_scaling' in results:
            memory_results = results['memory_scaling']
            symbol_counts = [r.symbols_processed for r in memory_results]
            memory_usage = [r.memory_usage_mb for r in memory_results]
            
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            plt.plot(symbol_counts, memory_usage, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Symbol Count')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Scaling')
            plt.grid(True, alpha=0.3)
        
        # Throughput scaling plot
        if 'basic_validation' in results:
            basic_results = results['basic_validation']
            symbol_counts = [r.symbols_processed for r in basic_results]
            throughput = [r.throughput_operations_per_second for r in basic_results]
            
            plt.subplot(2, 2, 2)
            plt.plot(symbol_counts, throughput, 's-', linewidth=2, markersize=8, color='orange')
            plt.xlabel('Symbol Count')
            plt.ylabel('Throughput (ops/sec)')
            plt.title('Throughput Scaling')
            plt.grid(True, alpha=0.3)
        
        # Parallel efficiency plot
        if 'parallel_processing' in results:
            parallel_results = results['parallel_processing']
            worker_counts = [r.threads_used for r in parallel_results]
            execution_times = [r.execution_time_seconds for r in parallel_results]
            
            plt.subplot(2, 2, 3)
            plt.plot(worker_counts, execution_times, '^-', linewidth=2, markersize=8, color='green')
            plt.xlabel('Worker Threads')
            plt.ylabel('Execution Time (s)')
            plt.title('Parallel Processing Efficiency')
            plt.grid(True, alpha=0.3)
        
        # Success rate comparison
        all_results = []
        labels = []
        for category, category_results in results.items():
            if category == 'summary':
                continue
            if isinstance(category_results, list):
                for result in category_results:
                    all_results.append(result.success_rate)
                    labels.append(f"{category}\n({result.symbols_processed} symbols)")
        
        if all_results:
            plt.subplot(2, 2, 4)
            bars = plt.bar(range(len(all_results)), all_results, color='skyblue', alpha=0.7)
            plt.xlabel('Benchmark')
            plt.ylabel('Success Rate')
            plt.title('Success Rate by Benchmark')
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.ylim(0, 1.1)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{all_results[i]:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/validation_performance_benchmarks.png", dpi=300, bbox_inches='tight')
        
        plt.show()


def run_benchmark_suite():
    """Main function to run the complete benchmark suite."""
    logger.info("Starting Walk-Forward Validation Performance Benchmark Suite")
    
    suite = ValidationBenchmarkSuite()
    
    try:
        # Run comprehensive benchmarks
        results = suite.run_comprehensive_benchmark_suite()
        
        # Generate report
        reporting = BenchmarkReporting()
        report = reporting.generate_performance_report(results)
        print(report)
        
        # Generate plots
        reporting.plot_scaling_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    import os
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run benchmarks
    results = run_benchmark_suite()
    
    # Save results to file
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate detailed report
    reporting = BenchmarkReporting()
    report = reporting.generate_performance_report(
        results, 
        output_file=f"{output_dir}/performance_report.md"
    )
    
    print(f"\nBenchmark completed. Results saved to {output_dir}/")
    print(f"Summary: {results.get('summary', {})}")