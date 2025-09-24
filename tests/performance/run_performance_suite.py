"""
ML4T Performance Test Suite Runner

Comprehensive performance testing execution for Issue #30 Stream B.
Orchestrates all performance tests and validates production readiness.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from .framework import PerformanceTestFramework
from .production_scale import ProductionScaleValidator
from .taiwan_stress import TaiwanMarketStressTester
from .memory_profiler import MemoryProfiler
from .latency_validator import LatencyValidator
from .load_tester import LoadTester
from .failover_recovery import FailoverRecoveryTester
from .reporting import PerformanceReporter


def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """Setup comprehensive logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def print_banner():
    """Print test suite banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                   ML4T Performance Test Suite                ║
║                                                              ║
║  Issue #30 - Production Readiness Testing                   ║
║  Stream B: Performance & Load Testing                       ║
║                                                              ║
║  Target Requirements:                                        ║
║  • Latency: <100ms real-time inference                      ║
║  • Memory: <16GB peak usage                                  ║  
║  • Throughput: >1500 predictions/second                     ║
║  • IC Performance: >0.05 Information Coefficient            ║
║  • Taiwan Market: 2000+ stock universe support              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_quick_validation() -> Dict[str, Any]:
    """Run quick validation test suite (for CI/testing)."""
    logger = logging.getLogger(__name__)
    logger.info("Running quick validation test suite...")
    
    # Initialize framework with minimal settings
    framework = PerformanceTestFramework(
        log_level='INFO',
        output_dir='performance_results_quick',
        parallel_tests=False,  # Sequential for reliability
        max_workers=2
    )
    
    # Run subset of critical tests
    results = {}
    
    try:
        # 1. Quick production scale test (small sample)
        logger.info("Running quick production scale validation...")
        production_validator = ProductionScaleValidator()
        results['production_scale_quick'] = production_validator.validate_full_universe(
            stock_count=100,  # Reduced from 2000
            days=63           # ~3 months
        )
        
        # 2. Quick memory profiling
        logger.info("Running quick memory profiling...")
        memory_profiler = MemoryProfiler()
        results['memory_scaling_quick'] = memory_profiler.profile_memory_scaling(
            max_stocks=500    # Reduced from 2000
        )
        
        # 3. Quick latency validation
        logger.info("Running quick latency validation...")
        latency_validator = LatencyValidator()
        results['latency_quick'] = latency_validator.validate_inference_latency(
            target_latency_ms=100
        )
        
        # 4. Quick load test
        logger.info("Running quick load test...")
        load_tester = LoadTester()
        results['load_test_quick'] = load_tester.test_concurrent_users(
            concurrent_users=2,
            duration_minutes=2
        )
        
        success_count = sum(1 for r in results.values() if r.get('success', False))
        
        summary = {
            'test_type': 'quick_validation',
            'total_tests': len(results),
            'successful_tests': success_count,
            'success_rate': success_count / len(results),
            'duration_seconds': 0,  # Will be set by caller
            'recommendation': 'Run full suite for production deployment' if success_count == len(results) else 'Fix issues before full testing'
        }
        
        results['summary'] = summary
        logger.info(f"Quick validation complete - {success_count}/{len(results)} tests passed")
        
        return results
        
    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'summary': {
                'test_type': 'quick_validation',
                'total_tests': 0,
                'successful_tests': 0,
                'success_rate': 0.0
            }
        }


def run_comprehensive_suite() -> Dict[str, Any]:
    """Run comprehensive performance test suite."""
    logger = logging.getLogger(__name__)
    logger.info("Running comprehensive performance test suite...")
    
    # Initialize main framework
    framework = PerformanceTestFramework(
        log_level='INFO',
        output_dir='performance_results_comprehensive',
        parallel_tests=True,
        max_workers=4
    )
    
    # Run comprehensive test suite
    logger.info("Executing comprehensive performance benchmarks...")
    results = framework.run_comprehensive_performance_suite()
    
    return results


def run_taiwan_stress_suite() -> Dict[str, Any]:
    """Run Taiwan market stress testing suite."""
    logger = logging.getLogger(__name__)
    logger.info("Running Taiwan market stress testing...")
    
    stress_tester = TaiwanMarketStressTester()
    results = stress_tester.run_comprehensive_taiwan_stress_suite()
    
    return results


def run_failover_suite() -> Dict[str, Any]:
    """Run failover and recovery testing suite."""
    logger = logging.getLogger(__name__)
    logger.info("Running failover and recovery testing...")
    
    failover_tester = FailoverRecoveryTester()
    results = failover_tester.run_comprehensive_failover_suite()
    
    return results


def run_load_testing_suite() -> Dict[str, Any]:
    """Run comprehensive load testing suite."""
    logger = logging.getLogger(__name__)
    logger.info("Running comprehensive load testing...")
    
    load_tester = LoadTester()
    results = load_tester.run_comprehensive_load_suite()
    
    return results


def validate_production_requirements(results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate results against production requirements."""
    logger = logging.getLogger(__name__)
    logger.info("Validating production requirements...")
    
    validation = {
        'memory_requirement': False,      # <16GB peak memory
        'latency_requirement': False,     # <100ms P95 latency  
        'throughput_requirement': False,  # >1500 ops/sec
        'ic_requirement': False,          # >0.05 IC
        'taiwan_requirement': False,      # 2000+ stocks supported
        'reliability_requirement': False, # >95% success rate
        'recovery_requirement': False     # Recovery within SLA
    }
    
    issues = []
    
    # Check comprehensive suite results
    if 'performance_metrics' in results.get('comprehensive', {}):
        metrics = results['comprehensive']['performance_metrics']
        
        # Memory validation
        peak_memory = metrics.get('peak_memory_usage_mb', 0)
        if peak_memory <= 16384:  # 16GB
            validation['memory_requirement'] = True
        else:
            issues.append(f"Memory usage ({peak_memory:.0f}MB) exceeds 16GB limit")
    
    # Check key metrics
    if 'key_metrics' in results.get('comprehensive', {}):
        key_metrics = results['comprehensive']['key_metrics']
        
        # Latency validation
        min_latency = key_metrics.get('min_latency_p95_ms')
        if min_latency and min_latency <= 100:
            validation['latency_requirement'] = True
        elif min_latency:
            issues.append(f"Latency P95 ({min_latency:.1f}ms) exceeds 100ms target")
            
        # Throughput validation
        max_throughput = key_metrics.get('max_throughput_ops_per_sec')
        if max_throughput and max_throughput >= 1500:
            validation['throughput_requirement'] = True
        elif max_throughput:
            issues.append(f"Throughput ({max_throughput:.0f} ops/sec) below 1500 ops/sec target")
            
        # IC validation
        max_ic = key_metrics.get('max_information_coefficient')
        if max_ic and max_ic >= 0.05:
            validation['ic_requirement'] = True
        elif max_ic:
            issues.append(f"Information Coefficient ({max_ic:.3f}) below 0.05 target")
    
    # Check Taiwan stress results
    taiwan_results = results.get('taiwan_stress', {})
    if taiwan_results.get('summary', {}).get('overall_stress_ready', False):
        validation['taiwan_requirement'] = True
    else:
        issues.append("Taiwan market stress testing failed")
    
    # Check reliability (success rates)
    comprehensive_summary = results.get('comprehensive', {}).get('test_summary', {})
    success_rate = comprehensive_summary.get('success_rate', 0)
    if success_rate >= 0.95:
        validation['reliability_requirement'] = True
    else:
        issues.append(f"Test success rate ({success_rate:.1%}) below 95% requirement")
    
    # Check failover results
    failover_results = results.get('failover', {})
    if failover_results.get('overall_resilience_passed', False):
        validation['recovery_requirement'] = True
    else:
        issues.append("Failover and recovery testing failed")
    
    # Overall validation
    requirements_met = sum(validation.values())
    total_requirements = len(validation)
    overall_ready = requirements_met == total_requirements
    
    return {
        'overall_production_ready': overall_ready,
        'requirements_met': requirements_met,
        'total_requirements': total_requirements,
        'requirements_details': validation,
        'issues_found': issues,
        'recommendation': 'System ready for production deployment' if overall_ready else 'Address issues before production deployment'
    }


def main():
    """Main test suite execution."""
    parser = argparse.ArgumentParser(description='ML4T Performance Test Suite')
    parser.add_argument('--suite', choices=['quick', 'comprehensive', 'taiwan', 'failover', 'load', 'all'],
                       default='comprehensive', help='Test suite to run')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    parser.add_argument('--output-dir', type=str, default='performance_results',
                       help='Output directory for results')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive report')
    parser.add_argument('--validate-requirements', action='store_true',
                       help='Validate against production requirements')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level, args.log_file)
    print_banner()
    
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    suite_start = time.time()
    all_results = {}
    
    try:
        # Run selected test suite(s)
        if args.suite == 'quick':
            logger.info("Running quick validation suite...")
            all_results['quick'] = run_quick_validation()
            
        elif args.suite == 'comprehensive':
            logger.info("Running comprehensive test suite...")
            all_results['comprehensive'] = run_comprehensive_suite()
            
        elif args.suite == 'taiwan':
            logger.info("Running Taiwan stress test suite...")
            all_results['taiwan_stress'] = run_taiwan_stress_suite()
            
        elif args.suite == 'failover':
            logger.info("Running failover test suite...")
            all_results['failover'] = run_failover_suite()
            
        elif args.suite == 'load':
            logger.info("Running load test suite...")
            all_results['load'] = run_load_testing_suite()
            
        elif args.suite == 'all':
            logger.info("Running all test suites...")
            
            # Run in sequence for stability
            all_results['quick'] = run_quick_validation()
            
            if all_results['quick'].get('summary', {}).get('success_rate', 0) >= 0.8:
                all_results['comprehensive'] = run_comprehensive_suite()
                all_results['taiwan_stress'] = run_taiwan_stress_suite()
                all_results['load'] = run_load_testing_suite()
                all_results['failover'] = run_failover_suite()
            else:
                logger.warning("Quick validation failed - skipping comprehensive tests")
        
        suite_duration = time.time() - suite_start
        
        # Update duration in results
        for suite_name, results in all_results.items():
            if isinstance(results, dict) and 'summary' in results:
                results['summary']['duration_seconds'] = suite_duration
        
        # Production requirements validation
        if args.validate_requirements:
            logger.info("Validating production requirements...")
            validation_results = validate_production_requirements(all_results)
            all_results['production_validation'] = validation_results
            
            # Print validation summary
            print("\n" + "="*60)
            print("PRODUCTION READINESS VALIDATION")
            print("="*60)
            print(f"Overall Ready: {'✅ YES' if validation_results['overall_production_ready'] else '❌ NO'}")
            print(f"Requirements Met: {validation_results['requirements_met']}/{validation_results['total_requirements']}")
            
            if validation_results['issues_found']:
                print("\nIssues Found:")
                for issue in validation_results['issues_found']:
                    print(f"  • {issue}")
                    
            print(f"\nRecommendation: {validation_results['recommendation']}")
            print("="*60)
        
        # Generate comprehensive report
        if args.generate_report:
            logger.info("Generating comprehensive report...")
            
            reporter = PerformanceReporter(output_dir=str(output_dir))
            
            # Extract test results for reporting (if available)
            test_results = []
            summary_report = {}
            
            if 'comprehensive' in all_results:
                # This would need to be adapted based on actual framework structure
                comprehensive_results = all_results['comprehensive']
                summary_report = comprehensive_results
                
            # Generate production readiness certificate
            if args.validate_requirements and 'production_validation' in all_results:
                certificate = reporter.generate_production_readiness_certificate(
                    test_results, all_results
                )
                all_results['certificate'] = certificate
                
                print(f"\nCertificate ID: {certificate['certificate_id']}")
                print(f"Status: {certificate['certification_status']}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f'performance_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
            
        logger.info(f"Results saved to: {results_file}")
        
        # Print summary
        print(f"\n🏁 Test suite completed in {suite_duration:.1f} seconds")
        print(f"📊 Results saved to: {results_file}")
        
        if args.generate_report:
            print(f"📑 Reports generated in: {output_dir}")
        
        # Exit code based on success
        if args.validate_requirements:
            exit_code = 0 if all_results.get('production_validation', {}).get('overall_production_ready') else 1
        else:
            # Check if any tests failed
            has_failures = any(
                not result.get('summary', {}).get('success_rate', 1.0) >= 0.8
                for result in all_results.values()
                if isinstance(result, dict)
            )
            exit_code = 1 if has_failures else 0
            
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        sys.exit(2)
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        
        # Save error results
        error_results = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'suite': args.suite,
            'partial_results': all_results
        }
        
        error_file = output_dir / f'performance_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(error_file, 'w') as f:
            json.dump(error_results, f, indent=2, default=str)
            
        logger.info(f"Error details saved to: {error_file}")
        sys.exit(3)


if __name__ == '__main__':
    main()