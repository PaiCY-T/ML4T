"""
Failover and Error Recovery Testing

Tests ML4T system resilience, error handling, and recovery mechanisms
under various failure scenarios and edge cases.
"""

import time
import logging
import threading
import random
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import psutil
from collections import defaultdict, deque
from pathlib import Path
import sys
import signal
import subprocess
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


@dataclass
class FailureScenario:
    """Definition of a failure scenario for testing."""
    name: str
    description: str
    failure_type: str
    severity: str  # low, medium, high, critical
    trigger_function: Callable
    expected_recovery_time_seconds: float
    expected_behavior: str
    
    
@dataclass
class RecoveryTestResult:
    """Results from a recovery test scenario."""
    scenario_name: str
    success: bool
    failure_triggered_at: float
    recovery_detected_at: Optional[float]
    recovery_time_seconds: Optional[float]
    expected_recovery_time: float
    system_behavior: str
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recovery_within_sla: bool = False
    

class SystemMonitor:
    """Monitors system health during failover testing."""
    
    def __init__(self, sampling_interval: float = 0.5):
        """Initialize system monitor."""
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.health_checks = []
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start system health monitoring."""
        self.monitoring = True
        self.health_checks = []
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> List[Dict[str, Any]]:
        """Stop monitoring and return health check history."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.health_checks.copy()
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                health_check = self._perform_health_check()
                self.health_checks.append(health_check)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logging.warning(f"Health check failed: {e}")
                
    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        timestamp = time.time()
        
        try:
            # System resources
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()
            
            # System availability
            system_responsive = self._check_system_responsiveness()
            
            # Process status
            process_status = self.process.status()
            thread_count = self.process.num_threads()
            
            return {
                'timestamp': timestamp,
                'system_responsive': system_responsive,
                'memory_mb': memory_info.rss / 1024 / 1024,
                'cpu_percent': cpu_percent,
                'process_status': process_status,
                'thread_count': thread_count,
                'healthy': system_responsive and cpu_percent < 95
            }
            
        except Exception as e:
            return {
                'timestamp': timestamp,
                'system_responsive': False,
                'error': str(e),
                'healthy': False
            }
            
    def _check_system_responsiveness(self) -> bool:
        """Check if system is responsive."""
        try:
            # Simple computation test
            start = time.time()
            result = sum(range(1000))
            end = time.time()
            
            # Should complete in reasonable time
            return (end - start) < 0.1  # 100ms threshold
            
        except Exception:
            return False


class FailoverRecoveryTester:
    """Main class for testing system failover and recovery mechanisms."""
    
    def __init__(self, log_level: str = 'INFO'):
        """Initialize failover recovery tester."""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def define_failure_scenarios(self) -> List[FailureScenario]:
        """Define comprehensive failure scenarios for testing."""
        scenarios = [
            # Memory pressure scenarios
            FailureScenario(
                name="memory_exhaustion",
                description="Simulate memory exhaustion scenario",
                failure_type="resource",
                severity="high",
                trigger_function=self._trigger_memory_exhaustion,
                expected_recovery_time_seconds=30.0,
                expected_behavior="System should free memory and continue processing"
            ),
            
            # CPU overload scenarios  
            FailureScenario(
                name="cpu_overload",
                description="Simulate CPU overload with intensive computation",
                failure_type="resource",
                severity="medium",
                trigger_function=self._trigger_cpu_overload,
                expected_recovery_time_seconds=15.0,
                expected_behavior="System should throttle processing and recover"
            ),
            
            # Data corruption scenarios
            FailureScenario(
                name="data_corruption",
                description="Simulate corrupted input data",
                failure_type="data",
                severity="medium",
                trigger_function=self._trigger_data_corruption,
                expected_recovery_time_seconds=5.0,
                expected_behavior="System should detect corruption and skip bad data"
            ),
            
            # Network/IO failures
            FailureScenario(
                name="io_failure",
                description="Simulate I/O operation failures",
                failure_type="io",
                severity="medium",
                trigger_function=self._trigger_io_failure,
                expected_recovery_time_seconds=10.0,
                expected_behavior="System should retry operations and handle failures gracefully"
            ),
            
            # Exception handling
            FailureScenario(
                name="exception_cascade",
                description="Simulate cascading exceptions",
                failure_type="logic",
                severity="high",
                trigger_function=self._trigger_exception_cascade,
                expected_recovery_time_seconds=20.0,
                expected_behavior="System should contain exceptions and maintain stability"
            ),
            
            # Timeout scenarios
            FailureScenario(
                name="operation_timeout",
                description="Simulate operation timeouts",
                failure_type="timeout",
                severity="medium",
                trigger_function=self._trigger_operation_timeout,
                expected_recovery_time_seconds=30.0,
                expected_behavior="System should timeout gracefully and retry"
            ),
            
            # Taiwan market specific
            FailureScenario(
                name="market_data_gap",
                description="Simulate Taiwan market data gaps",
                failure_type="market_data",
                severity="medium",
                trigger_function=self._trigger_market_data_gap,
                expected_recovery_time_seconds=10.0,
                expected_behavior="System should handle missing data and interpolate if needed"
            ),
            
            # Circuit breaker simulation
            FailureScenario(
                name="circuit_breaker_cascade",
                description="Simulate Taiwan circuit breaker cascade effects",
                failure_type="market_event",
                severity="high",
                trigger_function=self._trigger_circuit_breaker_cascade,
                expected_recovery_time_seconds=60.0,
                expected_behavior="System should handle market halts and resume processing"
            )
        ]
        
        return scenarios
        
    def test_failure_scenario(self, scenario: FailureScenario) -> RecoveryTestResult:
        """
        Test a specific failure scenario and measure recovery.
        
        Args:
            scenario: Failure scenario to test
            
        Returns:
            Recovery test results
        """
        self.logger.info(f"Testing failure scenario: {scenario.name}")
        
        # Setup monitoring
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        test_start = time.time()
        failure_triggered_at = None
        recovery_detected_at = None
        error_messages = []
        warnings = []
        
        try:
            # Baseline system health check
            baseline_health = monitor._perform_health_check()
            if not baseline_health['healthy']:
                warnings.append("System not healthy at test start")
                
            # Trigger failure scenario
            self.logger.info(f"Triggering failure: {scenario.description}")
            failure_start = time.time()
            
            try:
                scenario.trigger_function()
                failure_triggered_at = failure_start
                
            except Exception as e:
                error_messages.append(f"Failure trigger error: {str(e)}")
                
            # Monitor recovery
            recovery_timeout = scenario.expected_recovery_time_seconds * 3  # 3x expected time
            recovery_start = time.time()
            
            while time.time() - recovery_start < recovery_timeout:
                current_health = monitor._perform_health_check()
                
                if current_health['healthy'] and failure_triggered_at:
                    # Recovery detected
                    recovery_detected_at = time.time()
                    break
                    
                time.sleep(1.0)  # Check every second
                
            # Calculate recovery metrics
            if failure_triggered_at and recovery_detected_at:
                recovery_time = recovery_detected_at - failure_triggered_at
                recovery_within_sla = recovery_time <= scenario.expected_recovery_time_seconds
            else:
                recovery_time = None
                recovery_within_sla = False
                
            # Determine system behavior
            health_history = monitor.stop_monitoring()
            system_behavior = self._analyze_system_behavior(health_history, scenario)
            
            # Create result
            result = RecoveryTestResult(
                scenario_name=scenario.name,
                success=recovery_detected_at is not None,
                failure_triggered_at=failure_triggered_at or 0,
                recovery_detected_at=recovery_detected_at,
                recovery_time_seconds=recovery_time,
                expected_recovery_time=scenario.expected_recovery_time_seconds,
                system_behavior=system_behavior,
                error_messages=error_messages,
                warnings=warnings,
                recovery_within_sla=recovery_within_sla,
                metrics={
                    'total_test_time_seconds': time.time() - test_start,
                    'baseline_healthy': baseline_health['healthy'],
                    'health_checks_count': len(health_history)
                }
            )
            
            self.logger.info(
                f"Failure scenario {scenario.name} complete - "
                f"Recovery: {'Success' if result.success else 'Failed'}, "
                f"Time: {recovery_time:.1f}s" if recovery_time else "Time: N/A"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failure scenario test failed: {e}")
            
            monitor.stop_monitoring()
            
            return RecoveryTestResult(
                scenario_name=scenario.name,
                success=False,
                failure_triggered_at=failure_triggered_at or 0,
                recovery_detected_at=None,
                recovery_time_seconds=None,
                expected_recovery_time=scenario.expected_recovery_time_seconds,
                system_behavior="Test execution failed",
                error_messages=error_messages + [str(e)],
                warnings=warnings,
                recovery_within_sla=False
            )
            
    def _trigger_memory_exhaustion(self):
        """Trigger memory exhaustion scenario."""
        self.logger.info("Triggering memory exhaustion...")
        
        # Allocate large amounts of memory
        memory_hogs = []
        try:
            for _ in range(10):  # 10 iterations
                # Allocate ~100MB per iteration
                large_array = np.random.normal(0, 1, (1000, 10000))  # ~100MB
                memory_hogs.append(large_array)
                time.sleep(0.5)  # Gradual allocation
                
            # Hold memory for a period
            time.sleep(2.0)
            
        finally:
            # Clean up to allow recovery
            del memory_hogs
            import gc
            gc.collect()
            
    def _trigger_cpu_overload(self):
        """Trigger CPU overload scenario."""
        self.logger.info("Triggering CPU overload...")
        
        def cpu_intensive_task():
            # CPU-intensive computation
            for _ in range(10000000):  # 10M iterations
                math_result = sum(i * i for i in range(100))
                
        # Start multiple CPU-intensive threads
        threads = []
        for _ in range(psutil.cpu_count()):
            thread = threading.Thread(target=cpu_intensive_task)
            thread.start()
            threads.append(thread)
            
        # Let it run for a period
        time.sleep(5.0)
        
        # Threads will complete naturally
        for thread in threads:
            thread.join(timeout=1.0)
            
    def _trigger_data_corruption(self):
        """Trigger data corruption scenario."""
        self.logger.info("Triggering data corruption...")
        
        # Create corrupted data and try to process it
        corrupted_data = pd.DataFrame({
            'price': [np.inf, np.nan, -1000, 'invalid', None],
            'volume': ['text', np.inf, -100, np.nan, 0],
            'returns': [np.nan, np.inf, 'corrupted', -10, None]
        })
        
        try:
            # Try various operations that should fail gracefully
            result1 = corrupted_data.mean()  # Should handle NaN/inf
            result2 = corrupted_data.fillna(0).sum()  # Should handle after cleanup
            result3 = corrupted_data.astype(str)  # Should handle type conversion
            
        except Exception as e:
            # This is expected - system should recover from data errors
            self.logger.info(f"Data corruption handled: {e}")
            
    def _trigger_io_failure(self):
        """Trigger I/O failure scenario."""
        self.logger.info("Triggering I/O failures...")
        
        # Try to access non-existent files
        fake_paths = [
            "/non/existent/path/data.csv",
            "/tmp/missing_file.json",
            "Z:/invalid/drive/file.txt"
        ]
        
        for path in fake_paths:
            try:
                with open(path, 'r') as f:
                    data = f.read()
            except (FileNotFoundError, OSError, PermissionError) as e:
                # Expected failures - system should handle gracefully
                self.logger.info(f"I/O failure handled: {e}")
                
        # Try to write to read-only locations (if any)
        try:
            with open('/dev/null/readonly', 'w') as f:
                f.write("test")
        except (PermissionError, OSError) as e:
            self.logger.info(f"Write failure handled: {e}")
            
    def _trigger_exception_cascade(self):
        """Trigger cascading exceptions."""
        self.logger.info("Triggering exception cascade...")
        
        def failing_function_1():
            raise ValueError("First level failure")
            
        def failing_function_2():
            try:
                failing_function_1()
            except ValueError:
                raise RuntimeError("Second level failure")
                
        def failing_function_3():
            try:
                failing_function_2()
            except RuntimeError:
                raise SystemError("Third level failure")
                
        try:
            failing_function_3()
        except SystemError as e:
            self.logger.info(f"Exception cascade handled: {e}")
            
        # Trigger multiple concurrent exceptions
        def worker_with_exception(worker_id):
            try:
                if worker_id % 2 == 0:
                    raise ValueError(f"Worker {worker_id} failed")
                else:
                    raise RuntimeError(f"Worker {worker_id} crashed")
            except Exception as e:
                self.logger.info(f"Worker exception handled: {e}")
                
        # Multiple failing workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_with_exception, i) for i in range(10)]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    # Expected - testing exception handling
                    pass
                    
    def _trigger_operation_timeout(self):
        """Trigger operation timeout scenario.""" 
        self.logger.info("Triggering operation timeouts...")
        
        def slow_operation(duration: float):
            """Simulate slow operation."""
            time.sleep(duration)
            return f"Completed after {duration} seconds"
            
        # Test various timeout scenarios
        timeout_scenarios = [
            (2.0, 1.0),  # 2s operation with 1s timeout
            (5.0, 3.0),  # 5s operation with 3s timeout
            (10.0, 2.0)  # 10s operation with 2s timeout
        ]
        
        for operation_time, timeout in timeout_scenarios:
            try:
                # Simulate timeout handling
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(slow_operation, operation_time)
                    
                    try:
                        result = future.result(timeout=timeout)
                        self.logger.info(f"Operation completed: {result}")
                    except TimeoutError:
                        self.logger.info(f"Operation timed out after {timeout}s (expected)")
                        
            except Exception as e:
                self.logger.info(f"Timeout scenario handled: {e}")
                
    def _trigger_market_data_gap(self):
        """Trigger Taiwan market data gap scenario."""
        self.logger.info("Triggering market data gap...")
        
        # Simulate Taiwan market data with gaps
        dates = pd.bdate_range('2024-01-01', '2024-01-31')
        
        # Create data with intentional gaps
        taiwan_stocks = ['2330.TW', '2454.TW', '2882.TW', '1101.TW']
        
        for stock in taiwan_stocks:
            try:
                # Create incomplete data
                incomplete_dates = dates[::3]  # Every 3rd date only
                incomplete_data = pd.DataFrame({
                    'stock_id': stock,
                    'price': np.random.lognormal(4, 0.5, len(incomplete_dates)),
                    'volume': np.random.lognormal(10, 1, len(incomplete_dates))
                }, index=incomplete_dates)
                
                # Try to process with gaps
                filled_data = incomplete_data.resample('D').ffill()  # Forward fill
                interpolated_data = incomplete_data.resample('D').interpolate()
                
            except Exception as e:
                self.logger.info(f"Data gap handling: {e}")
                
    def _trigger_circuit_breaker_cascade(self):
        """Trigger Taiwan circuit breaker cascade scenario."""
        self.logger.info("Triggering circuit breaker cascade...")
        
        # Simulate Taiwan market circuit breaker scenario
        taiwan_stocks = [f"2{330 + i}.TW" for i in range(20)]  # 20 major stocks
        
        # Simulate market crash triggering circuit breakers
        market_crash_returns = np.random.uniform(-0.10, -0.07, len(taiwan_stocks))  # 7-10% drops
        
        circuit_breaker_stocks = []
        
        for i, (stock, return_val) in enumerate(zip(taiwan_stocks, market_crash_returns)):
            try:
                # Simulate trading halt detection
                if abs(return_val) >= 0.075:  # 7.5% circuit breaker threshold
                    circuit_breaker_stocks.append(stock)
                    
                    # Simulate halt processing delay
                    time.sleep(0.1)  # 100ms per halt
                    
                    # Simulate order cancellation processing
                    cancelled_orders = random.randint(100, 1000)
                    
                    self.logger.info(f"Circuit breaker triggered for {stock}: {return_val:.1%}")
                    
            except Exception as e:
                self.logger.info(f"Circuit breaker processing error: {e}")
                
        # Simulate market-wide halt if too many stocks triggered
        if len(circuit_breaker_stocks) >= 10:
            self.logger.info("Market-wide halt triggered")
            time.sleep(2.0)  # Market halt processing time
            
        # Simulate recovery processing
        for stock in circuit_breaker_stocks:
            try:
                # Simulate order book reconstruction
                time.sleep(0.05)  # 50ms per stock recovery
                self.logger.info(f"Trading resumed for {stock}")
                
            except Exception as e:
                self.logger.info(f"Recovery processing error: {e}")
                
    def _analyze_system_behavior(self, 
                                health_history: List[Dict[str, Any]], 
                                scenario: FailureScenario) -> str:
        """Analyze system behavior during failure scenario."""
        if not health_history:
            return "No health data available"
            
        # Count healthy vs unhealthy periods
        healthy_checks = len([h for h in health_history if h.get('healthy', False)])
        total_checks = len(health_history)
        health_ratio = healthy_checks / total_checks if total_checks > 0 else 0
        
        # Analyze response times and resource usage
        response_times = []
        memory_usage = []
        
        for check in health_history:
            if 'memory_mb' in check:
                memory_usage.append(check['memory_mb'])
                
        # Categorize behavior
        if health_ratio >= 0.8:
            behavior = "System remained stable throughout test"
        elif health_ratio >= 0.5:
            behavior = "System experienced temporary degradation but recovered"
        elif health_ratio >= 0.2:
            behavior = "System experienced significant instability"
        else:
            behavior = "System became largely unresponsive"
            
        # Add memory analysis
        if memory_usage:
            max_memory = max(memory_usage)
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            if max_memory > 8000:  # >8GB
                behavior += f" with high memory usage (peak: {max_memory:.0f}MB)"
            elif avg_memory > 4000:  # >4GB average
                behavior += f" with elevated memory usage (avg: {avg_memory:.0f}MB)"
                
        return behavior
        
    def run_comprehensive_failover_suite(self) -> Dict[str, Any]:
        """Run comprehensive failover and recovery testing suite."""
        self.logger.info("Starting comprehensive failover and recovery testing")
        
        suite_start = time.time()
        scenarios = self.define_failure_scenarios()
        
        # Test each failure scenario
        results = {}
        for scenario in scenarios:
            self.logger.info(f"Testing scenario: {scenario.name}")
            
            try:
                result = self.test_failure_scenario(scenario)
                results[scenario.name] = result
                
                # Add recovery delay between tests
                time.sleep(2.0)
                
            except Exception as e:
                self.logger.error(f"Scenario {scenario.name} testing failed: {e}")
                results[scenario.name] = RecoveryTestResult(
                    scenario_name=scenario.name,
                    success=False,
                    failure_triggered_at=0,
                    recovery_detected_at=None,
                    recovery_time_seconds=None,
                    expected_recovery_time=scenario.expected_recovery_time_seconds,
                    system_behavior="Test execution failed",
                    error_messages=[str(e)],
                    recovery_within_sla=False
                )
                
        total_time = time.time() - suite_start
        
        # Generate summary
        total_scenarios = len(results)
        successful_recoveries = len([r for r in results.values() if r.success])
        recoveries_within_sla = len([r for r in results.values() if r.recovery_within_sla])
        
        # Calculate average recovery time
        recovery_times = [r.recovery_time_seconds for r in results.values() 
                         if r.recovery_time_seconds is not None]
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else None
        
        summary = {
            'total_test_time_seconds': total_time,
            'total_scenarios': total_scenarios,
            'successful_recoveries': successful_recoveries,
            'recoveries_within_sla': recoveries_within_sla,
            'success_rate': successful_recoveries / total_scenarios if total_scenarios > 0 else 0,
            'sla_compliance_rate': recoveries_within_sla / total_scenarios if total_scenarios > 0 else 0,
            'avg_recovery_time_seconds': avg_recovery_time,
            'failover_test_timestamp': datetime.now().isoformat(),
            'system_resilience_score': self._calculate_resilience_score(results)
        }
        
        # Convert results to dict format
        results_dict = {}
        for scenario_name, result in results.items():
            results_dict[scenario_name] = {
                'success': result.success,
                'recovery_time_seconds': result.recovery_time_seconds,
                'expected_recovery_time': result.expected_recovery_time,
                'recovery_within_sla': result.recovery_within_sla,
                'system_behavior': result.system_behavior,
                'error_count': len(result.error_messages),
                'warning_count': len(result.warnings)
            }
            
        final_results = {
            'summary': summary,
            'scenarios': results_dict,
            'overall_resilience_passed': summary['success_rate'] >= 0.8 and summary['sla_compliance_rate'] >= 0.7
        }
        
        self.logger.info(
            f"Comprehensive failover testing complete - "
            f"Time: {total_time:.1f}s, Success rate: {summary['success_rate']:.1%}, "
            f"SLA compliance: {summary['sla_compliance_rate']:.1%}"
        )
        
        return final_results
        
    def _calculate_resilience_score(self, results: Dict[str, RecoveryTestResult]) -> float:
        """Calculate overall system resilience score."""
        if not results:
            return 0.0
            
        total_score = 0.0
        scenario_weights = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        total_weight = 0.0
        
        for result in results.values():
            # Determine scenario severity (simplified)
            if 'memory' in result.scenario_name or 'cascade' in result.scenario_name:
                severity = 'high'
            elif 'timeout' in result.scenario_name or 'io' in result.scenario_name:
                severity = 'medium' 
            else:
                severity = 'low'
                
            weight = scenario_weights.get(severity, 0.5)
            
            # Calculate scenario score
            scenario_score = 0.0
            if result.success:
                scenario_score += 0.6  # 60% for successful recovery
                
            if result.recovery_within_sla:
                scenario_score += 0.4  # 40% for meeting SLA
                
            total_score += scenario_score * weight
            total_weight += weight
            
        return total_score / total_weight if total_weight > 0 else 0.0