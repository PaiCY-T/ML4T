"""
Memory Profiler for ML4T Performance Testing

Advanced memory usage profiling and optimization analysis for production readiness.
Detects memory leaks, analyzes scaling patterns, and validates memory constraints.
"""

import time
import psutil
import logging
import gc
import tracemalloc
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict
import weakref
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot."""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    tracemalloc_current_mb: Optional[float] = None
    tracemalloc_peak_mb: Optional[float] = None
    gc_objects: Optional[int] = None
    
    
@dataclass
class MemoryProfile:
    """Complete memory profile for a test run."""
    test_name: str
    start_time: float
    end_time: float
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    memory_leak_detected: bool = False
    memory_growth_rate_mb_per_sec: float = 0.0
    gc_collections: Dict[int, int] = field(default_factory=dict)
    memory_efficiency_score: float = 0.0
    
    def duration_seconds(self) -> float:
        """Get test duration in seconds."""
        return self.end_time - self.start_time
        
    def memory_growth_mb(self) -> float:
        """Get total memory growth in MB."""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[-1].rss_mb - self.snapshots[0].rss_mb


class MemoryMonitor:
    """Advanced memory monitoring with leak detection."""
    
    def __init__(self, sampling_interval: float = 0.1, enable_tracemalloc: bool = True):
        """
        Initialize memory monitor.
        
        Args:
            sampling_interval: Time between samples in seconds
            enable_tracemalloc: Whether to use Python tracemalloc for detailed tracking
        """
        self.sampling_interval = sampling_interval
        self.enable_tracemalloc = enable_tracemalloc
        self.monitoring = False
        self.monitor_thread = None
        self.snapshots = []
        self.process = psutil.Process()
        
        # Tracemalloc setup
        if self.enable_tracemalloc:
            tracemalloc.start()
            
        # Weak reference tracking for leak detection
        self.tracked_objects = weakref.WeakSet()
        
    def start_monitoring(self, test_name: str) -> None:
        """Start memory monitoring for a test."""
        self.test_name = test_name
        self.monitoring = True
        self.snapshots = []
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> MemoryProfile:
        """Stop monitoring and return memory profile."""
        self.monitoring = False
        end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join()
            
        # Create memory profile
        profile = MemoryProfile(
            test_name=self.test_name,
            start_time=self.start_time,
            end_time=end_time,
            snapshots=self.snapshots.copy()
        )
        
        # Calculate metrics
        self._calculate_profile_metrics(profile)
        
        return profile
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logging.warning(f"Memory monitoring error: {e}")
                
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        # Process memory info
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # System memory info
        system_memory = psutil.virtual_memory()
        
        # Tracemalloc info
        tracemalloc_current = None
        tracemalloc_peak = None
        if self.enable_tracemalloc:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_current = current / 1024 / 1024  # MB
            tracemalloc_peak = peak / 1024 / 1024  # MB
            
        # GC objects count
        gc_objects = len(gc.get_objects()) if hasattr(gc, 'get_objects') else None
        
        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=system_memory.available / 1024 / 1024,
            tracemalloc_current_mb=tracemalloc_current,
            tracemalloc_peak_mb=tracemalloc_peak,
            gc_objects=gc_objects
        )
        
    def _calculate_profile_metrics(self, profile: MemoryProfile) -> None:
        """Calculate derived metrics for memory profile."""
        if not profile.snapshots:
            return
            
        # Peak memory
        profile.peak_memory_mb = max(s.rss_mb for s in profile.snapshots)
        
        # Memory leak detection
        if len(profile.snapshots) >= 10:  # Need enough samples
            # Check for consistent growth
            memory_values = [s.rss_mb for s in profile.snapshots]
            growth_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
            
            profile.memory_growth_rate_mb_per_sec = growth_trend / self.sampling_interval
            
            # Detect leak: consistent growth > 1MB/minute
            if profile.memory_growth_rate_mb_per_sec > 1.0 / 60:
                profile.memory_leak_detected = True
                
        # GC statistics
        profile.gc_collections = {
            gen: gc.get_count()[gen] for gen in range(3)
        } if hasattr(gc, 'get_count') else {}
        
        # Memory efficiency score (lower memory increase = higher score)
        if profile.duration_seconds() > 0:
            memory_growth = profile.memory_growth_mb()
            efficiency_base = max(100 - abs(memory_growth), 0)  # Base score
            profile.memory_efficiency_score = min(efficiency_base / 100.0, 1.0)


class MemoryProfiler:
    """Main memory profiling class for ML4T performance testing."""
    
    def __init__(self, log_level: str = 'INFO'):
        """Initialize memory profiler."""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.profiles = []
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def profile_function(self, 
                        func: Callable, 
                        *args, 
                        test_name: str = None,
                        **kwargs) -> Tuple[Any, MemoryProfile]:
        """
        Profile memory usage of a function call.
        
        Args:
            func: Function to profile
            *args: Function arguments
            test_name: Name for the test
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (function_result, memory_profile)
        """
        test_name = test_name or f"{func.__name__}"
        self.logger.info(f"Starting memory profiling: {test_name}")
        
        # Setup monitoring
        monitor = MemoryMonitor()
        monitor.start_monitoring(test_name)
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            success = True
            
        except Exception as e:
            self.logger.error(f"Function execution failed: {e}")
            result = None
            success = False
            
        finally:
            # Stop monitoring
            profile = monitor.stop_monitoring()
            profile.success = success
            
        self.profiles.append(profile)
        
        self.logger.info(
            f"Memory profiling complete: {test_name} - "
            f"Peak: {profile.peak_memory_mb:.1f}MB, "
            f"Growth: {profile.memory_growth_mb():.1f}MB, "
            f"Leak: {'Yes' if profile.memory_leak_detected else 'No'}"
        )
        
        return result, profile
        
    def profile_memory_scaling(self, max_stocks: int = 2000) -> Dict[str, Any]:
        """
        Profile memory usage scaling with data size.
        
        Args:
            max_stocks: Maximum number of stocks to test
            
        Returns:
            Memory scaling analysis results
        """
        self.logger.info(f"Starting memory scaling analysis up to {max_stocks} stocks")
        
        stock_counts = [10, 50, 100, 250, 500, 1000, max_stocks]
        scaling_results = []
        
        for stock_count in stock_counts:
            self.logger.info(f"Testing memory usage with {stock_count} stocks")
            
            # Profile data generation and basic operations
            result, profile = self.profile_function(
                self._generate_test_data_with_operations,
                stock_count,
                test_name=f"scaling_{stock_count}_stocks"
            )
            
            if hasattr(profile, 'success') and profile.success:
                scaling_results.append({
                    'stock_count': stock_count,
                    'peak_memory_mb': profile.peak_memory_mb,
                    'memory_growth_mb': profile.memory_growth_mb(),
                    'duration_seconds': profile.duration_seconds(),
                    'memory_efficiency_score': profile.memory_efficiency_score,
                    'memory_per_stock_kb': (profile.peak_memory_mb * 1024) / stock_count
                })
            else:
                self.logger.warning(f"Failed to profile {stock_count} stocks")
                
            # Force cleanup between tests
            gc.collect()
            time.sleep(1)  # Allow memory cleanup
            
        # Analyze scaling patterns
        if len(scaling_results) >= 3:
            analysis = self._analyze_memory_scaling(scaling_results)
        else:
            analysis = {'error': 'Insufficient data points for scaling analysis'}
            
        return {
            'success': len(scaling_results) > 0,
            'scaling_results': scaling_results,
            'scaling_analysis': analysis,
            'max_stocks_tested': max_stocks,
            'memory_scaling_passed': self._validate_memory_scaling(scaling_results)
        }
        
    def _generate_test_data_with_operations(self, stock_count: int) -> pd.DataFrame:
        """Generate test data and perform typical operations for memory profiling."""
        # Generate large dataset
        np.random.seed(42)
        days = 252
        features = 42
        
        # Create stock IDs
        stock_ids = [f"{2000 + i:04d}.TW" for i in range(stock_count)]
        
        # Create date range
        dates = pd.bdate_range(start='2023-01-01', periods=days)
        
        # Create MultiIndex
        index = pd.MultiIndex.from_product([dates, stock_ids], names=['date', 'stock_id'])
        
        # Generate features
        n_samples = len(index)
        feature_data = {
            f'feature_{i:02d}': np.random.normal(0, 1, n_samples) 
            for i in range(features)
        }
        
        # Create DataFrame
        df = pd.DataFrame(feature_data, index=index)
        
        # Perform typical operations that might consume memory
        # 1. Groupby operations
        daily_stats = df.groupby('date').agg(['mean', 'std'])
        
        # 2. Cross-sectional operations
        latest_cross_section = df.xs(dates[-1], level='date')
        
        # 3. Rolling calculations
        rolling_means = df.groupby('stock_id').rolling(window=20).mean()
        
        # 4. Correlation matrix (memory intensive)
        if stock_count <= 500:  # Only for smaller datasets
            pivot_data = df.pivot_table(
                values='feature_00', 
                index='date', 
                columns='stock_id'
            )
            correlation_matrix = pivot_data.corr()
        
        return df
        
    def _analyze_memory_scaling(self, scaling_results: List[Dict]) -> Dict[str, Any]:
        """Analyze memory scaling patterns."""
        stock_counts = [r['stock_count'] for r in scaling_results]
        memory_usage = [r['peak_memory_mb'] for r in scaling_results]
        
        # Fit scaling relationship
        # Try linear: memory = a * stocks + b
        linear_fit = np.polyfit(stock_counts, memory_usage, 1)
        linear_slope = linear_fit[0]  # MB per stock
        
        # Try power law: memory = a * stocks^b
        if all(s > 0 and m > 0 for s, m in zip(stock_counts, memory_usage)):
            log_stocks = np.log(stock_counts)
            log_memory = np.log(memory_usage)
            power_fit = np.polyfit(log_stocks, log_memory, 1)
            power_exponent = power_fit[0]
        else:
            power_exponent = None
            
        # Memory efficiency analysis
        memory_per_stock = [r['memory_per_stock_kb'] for r in scaling_results]
        efficiency_trend = np.polyfit(stock_counts, memory_per_stock, 1)[0]  # KB/stock trend
        
        # Predict full universe memory usage
        full_universe_prediction = linear_fit[0] * 2000 + linear_fit[1]  # 2000 stocks
        
        return {
            'linear_slope_mb_per_stock': linear_slope,
            'power_law_exponent': power_exponent,
            'efficiency_trend_kb_per_stock': efficiency_trend,
            'predicted_full_universe_memory_mb': full_universe_prediction,
            'memory_scaling_type': self._classify_scaling_type(power_exponent),
            'scaling_efficiency': 'good' if efficiency_trend < 0.1 else 'needs_optimization'
        }
        
    def _classify_scaling_type(self, power_exponent: Optional[float]) -> str:
        """Classify memory scaling type."""
        if power_exponent is None:
            return 'unknown'
        elif power_exponent < 1.2:
            return 'linear'  # Good scaling
        elif power_exponent < 1.5:
            return 'superlinear'  # Acceptable
        else:
            return 'quadratic'  # Concerning
            
    def _validate_memory_scaling(self, scaling_results: List[Dict]) -> bool:
        """Validate if memory scaling meets production requirements."""
        if not scaling_results:
            return False
            
        # Check if 2000 stocks would fit in 16GB
        max_result = max(scaling_results, key=lambda x: x['stock_count'])
        max_memory = max_result['peak_memory_mb']
        max_stocks = max_result['stock_count']
        
        # Linear extrapolation to 2000 stocks
        if max_stocks > 0:
            predicted_memory = max_memory * (2000 / max_stocks)
            return predicted_memory <= 16384  # 16GB limit
        return False
        
    def detect_memory_leaks(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Detect memory leaks through repeated operations.
        
        Args:
            iterations: Number of iterations to test
            
        Returns:
            Memory leak detection results
        """
        self.logger.info(f"Starting memory leak detection with {iterations} iterations")
        
        # Profile repeated operations
        result, profile = self.profile_function(
            self._repeated_operations_test,
            iterations,
            test_name=f"memory_leak_detection_{iterations}_iterations"
        )
        
        # Additional analysis
        leak_analysis = self._analyze_memory_leak_patterns(profile)
        
        return {
            'success': hasattr(profile, 'success') and profile.success,
            'iterations_tested': iterations,
            'memory_leak_detected': profile.memory_leak_detected,
            'memory_growth_rate_mb_per_sec': profile.memory_growth_rate_mb_per_sec,
            'peak_memory_mb': profile.peak_memory_mb,
            'total_memory_growth_mb': profile.memory_growth_mb(),
            'memory_efficiency_score': profile.memory_efficiency_score,
            'leak_analysis': leak_analysis,
            'memory_leak_test_passed': not profile.memory_leak_detected
        }
        
    def _repeated_operations_test(self, iterations: int) -> None:
        """Perform repeated operations to detect memory leaks."""
        data_cache = []  # Intentionally keep some data to test cleanup
        
        for i in range(iterations):
            # Generate small dataset
            df = pd.DataFrame(
                np.random.normal(0, 1, (1000, 10)),
                columns=[f'col_{j}' for j in range(10)]
            )
            
            # Perform operations
            result = df.describe()
            corr_matrix = df.corr()
            
            # Simulate some objects being kept (potential leak)
            if i % 10 == 0:  # Keep every 10th result
                data_cache.append(result)
                
            # Simulate proper cleanup
            if len(data_cache) > 5:
                data_cache.pop(0)
                
            # Force garbage collection periodically
            if i % 20 == 0:
                gc.collect()
                
    def _analyze_memory_leak_patterns(self, profile: MemoryProfile) -> Dict[str, Any]:
        """Analyze memory usage patterns for leak detection."""
        if len(profile.snapshots) < 10:
            return {'error': 'Insufficient samples for leak analysis'}
            
        # Memory growth analysis
        memory_values = [s.rss_mb for s in profile.snapshots]
        time_values = [(s.timestamp - profile.start_time) for s in profile.snapshots]
        
        # Detect growth phases
        growth_phases = []
        for i in range(1, len(memory_values)):
            growth = memory_values[i] - memory_values[i-1]
            if abs(growth) > 1:  # >1MB change
                growth_phases.append({
                    'time_seconds': time_values[i],
                    'growth_mb': growth,
                    'total_memory_mb': memory_values[i]
                })
        
        # Statistical analysis
        memory_std = np.std(memory_values)
        memory_trend = np.polyfit(time_values, memory_values, 1)[0]  # MB/sec
        
        return {
            'memory_volatility_mb': memory_std,
            'memory_trend_mb_per_sec': memory_trend,
            'growth_phases_count': len(growth_phases),
            'significant_growth_phases': [p for p in growth_phases if p['growth_mb'] > 5],
            'leak_risk_level': self._assess_leak_risk(memory_trend, memory_std)
        }
        
    def _assess_leak_risk(self, trend: float, volatility: float) -> str:
        """Assess memory leak risk level."""
        if trend > 0.1:  # >0.1 MB/sec growth
            return 'high'
        elif trend > 0.05:  # >0.05 MB/sec growth
            return 'medium'
        elif volatility > 10:  # High volatility might indicate issues
            return 'low'
        else:
            return 'minimal'
            
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory analysis report."""
        if not self.profiles:
            return {'error': 'No memory profiles available'}
            
        # Aggregate statistics
        total_profiles = len(self.profiles)
        successful_profiles = [p for p in self.profiles if hasattr(p, 'success') and p.success]
        
        peak_memories = [p.peak_memory_mb for p in successful_profiles]
        memory_growths = [p.memory_growth_mb() for p in successful_profiles]
        efficiency_scores = [p.memory_efficiency_score for p in successful_profiles]
        
        # Leak detection summary
        profiles_with_leaks = [p for p in successful_profiles if p.memory_leak_detected]
        
        report = {
            'summary': {
                'total_profiles': total_profiles,
                'successful_profiles': len(successful_profiles),
                'profiles_with_leaks': len(profiles_with_leaks),
                'peak_memory_usage_mb': max(peak_memories) if peak_memories else 0,
                'avg_memory_usage_mb': np.mean(peak_memories) if peak_memories else 0,
                'avg_memory_growth_mb': np.mean(memory_growths) if memory_growths else 0,
                'avg_efficiency_score': np.mean(efficiency_scores) if efficiency_scores else 0
            },
            
            'memory_requirements_check': {
                'meets_16gb_limit': all(m <= 16384 for m in peak_memories),
                'max_memory_usage_mb': max(peak_memories) if peak_memories else 0,
                'memory_limit_utilization': (max(peak_memories) / 16384) if peak_memories else 0
            },
            
            'leak_detection': {
                'leaks_detected': len(profiles_with_leaks) > 0,
                'leak_profiles': [p.test_name for p in profiles_with_leaks],
                'max_growth_rate_mb_per_sec': max(
                    (p.memory_growth_rate_mb_per_sec for p in successful_profiles), 
                    default=0
                )
            },
            
            'recommendations': self._generate_memory_recommendations(successful_profiles),
            
            'detailed_profiles': [
                {
                    'test_name': p.test_name,
                    'peak_memory_mb': p.peak_memory_mb,
                    'memory_growth_mb': p.memory_growth_mb(),
                    'duration_seconds': p.duration_seconds(),
                    'efficiency_score': p.memory_efficiency_score,
                    'leak_detected': p.memory_leak_detected
                }
                for p in successful_profiles
            ]
        }
        
        return report
        
    def _generate_memory_recommendations(self, profiles: List[MemoryProfile]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if not profiles:
            return ['No profiles available for analysis']
            
        peak_memories = [p.peak_memory_mb for p in profiles]
        max_memory = max(peak_memories)
        
        # Memory usage recommendations
        if max_memory > 12000:  # >12GB
            recommendations.append(
                f"High memory usage detected ({max_memory:.1f}MB). "
                "Consider implementing data chunking or streaming processing."
            )
        elif max_memory > 8000:  # >8GB
            recommendations.append(
                "Moderate memory usage. Monitor memory growth during longer operations."
            )
            
        # Leak recommendations
        leak_profiles = [p for p in profiles if p.memory_leak_detected]
        if leak_profiles:
            recommendations.append(
                f"Memory leaks detected in {len(leak_profiles)} tests. "
                "Review object lifecycle management and garbage collection."
            )
            
        # Efficiency recommendations
        efficiency_scores = [p.memory_efficiency_score for p in profiles]
        avg_efficiency = np.mean(efficiency_scores)
        
        if avg_efficiency < 0.7:
            recommendations.append(
                f"Low memory efficiency (score: {avg_efficiency:.2f}). "
                "Consider optimizing data structures and algorithms."
            )
            
        if not recommendations:
            recommendations.append("Memory usage patterns are within acceptable limits.")
            
        return recommendations