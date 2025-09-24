"""
OpenFE Performance Validation and Resource Monitoring

Benchmarks memory usage, processing time, and system resources
for OpenFE integration with Taiwan market data.

CRITICAL MONITORING:
- Memory usage during feature generation (target: <8GB)
- Processing time benchmarks (target: <2x baseline)
- Resource utilization and limits
- Feature generation quality metrics
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.openfe_wrapper import FeatureGenerator
from features.taiwan_config import TaiwanMarketConfig


class OpenFEPerformanceMonitor:
    """Performance monitoring for OpenFE integration."""
    
    def __init__(self, log_level: str = 'INFO'):
        """Initialize performance monitor."""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.benchmarks = {}
        self.resource_usage = {}
        self.taiwan_config = TaiwanMarketConfig()
        
    def setup_logging(self, log_level: str) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('openfe_performance.log')
            ]
        )
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_gb': psutil.disk_usage('/').total / (1024**3),
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
        
    def create_benchmark_data(self, 
                            n_stocks: int = 100, 
                            n_days: int = 252, 
                            n_features: int = 42) -> pd.DataFrame:
        """
        Create benchmark dataset mimicking Taiwan market data.
        
        Args:
            n_stocks: Number of stocks (default: 100, full universe: 2000)
            n_days: Number of trading days (default: 1 year)
            n_features: Number of input features (from Task #25)
            
        Returns:
            Benchmark DataFrame with MultiIndex (date, stock_id)
        """
        self.logger.info(f"Creating benchmark data: {n_stocks} stocks, {n_days} days, {n_features} features")
        
        # Generate trading dates (excluding weekends)
        start_date = datetime.now() - timedelta(days=int(n_days * 1.4))  # Buffer for weekends
        dates = pd.bdate_range(start=start_date, periods=n_days)
        
        # Generate stock IDs (Taiwan stock format)
        stock_ids = [f"{2000 + i:04d}" for i in range(n_stocks)]
        
        # Create MultiIndex
        index = pd.MultiIndex.from_product(
            [dates, stock_ids], 
            names=['date', 'stock_id']
        )
        
        # Generate realistic financial features
        np.random.seed(42)  # Reproducible benchmarks
        n_samples = len(index)
        
        # Base financial features (simplified version of 42 factors)
        features = {
            # Price features
            'close_price': np.random.lognormal(4, 0.5, n_samples),
            'high_price': np.random.lognormal(4.1, 0.5, n_samples),
            'low_price': np.random.lognormal(3.9, 0.5, n_samples),
            'volume': np.random.lognormal(10, 1, n_samples),
            
            # Technical indicators
            'rsi_14': np.random.uniform(20, 80, n_samples),
            'macd': np.random.normal(0, 2, n_samples),
            'bollinger_position': np.random.uniform(-1, 1, n_samples),
            'momentum_20d': np.random.normal(0, 0.15, n_samples),
            
            # Fundamental ratios
            'pe_ratio': np.random.lognormal(2.5, 0.5, n_samples),
            'pb_ratio': np.random.lognormal(0.5, 0.3, n_samples),
            'roe': np.random.uniform(0.05, 0.25, n_samples),
            'debt_ratio': np.random.uniform(0.1, 0.8, n_samples),
            
            # Market microstructure
            'bid_ask_spread': np.random.exponential(0.01, n_samples),
            'order_imbalance': np.random.normal(0, 0.1, n_samples),
            'trade_frequency': np.random.poisson(50, n_samples),
            
            # Volatility measures
            'realized_vol_20d': np.random.exponential(0.2, n_samples),
            'garch_vol': np.random.exponential(0.25, n_samples),
            
            # Cross-sectional factors
            'sector_momentum': np.random.normal(0, 0.1, n_samples),
            'market_beta': np.random.uniform(0.5, 2.0, n_samples),
        }
        
        # Add more features to reach target number
        for i in range(len(features), n_features):
            features[f'synthetic_feature_{i:02d}'] = np.random.normal(0, 1, n_samples)
            
        df = pd.DataFrame(features, index=index)
        
        # Apply Taiwan market constraints
        df = self.taiwan_config.apply_taiwan_feature_constraints(df)
        
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        self.logger.info(f"Benchmark data created: {df.shape}, Memory: {memory_usage_mb:.1f}MB")
        
        return df
        
    def monitor_resource_usage(self, stage: str) -> Dict[str, Any]:
        """Monitor system resource usage."""
        process = psutil.Process()
        
        resource_info = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': process.cpu_percent(),
            'memory_rss_mb': process.memory_info().rss / 1024 / 1024,
            'memory_vms_mb': process.memory_info().vms / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'open_files': len(process.open_files()),
            'num_threads': process.num_threads()
        }
        
        self.resource_usage[stage] = resource_info
        return resource_info
        
    def benchmark_feature_generation(self, 
                                   data: pd.DataFrame,
                                   target: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Benchmark OpenFE feature generation performance.
        
        Args:
            data: Input feature data
            target: Optional target variable
            
        Returns:
            Benchmark results dictionary
        """
        self.logger.info("Starting OpenFE feature generation benchmark...")
        
        benchmark_results = {
            'input_shape': data.shape,
            'input_memory_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'start_time': datetime.now().isoformat()
        }
        
        # Monitor initial resources
        self.monitor_resource_usage('benchmark_start')
        
        try:
            # Initialize FeatureGenerator with monitoring settings
            fg = FeatureGenerator(
                n_jobs=-1,
                time_budget=600,  # 10 minutes
                memory_limit_mb=8192,  # 8GB limit
                taiwan_market=True,
                max_features=500
            )
            
            # Monitor after initialization
            self.monitor_resource_usage('after_init')
            
            # Measure fit time
            start_time = time.time()
            
            try:
                fg.fit(data, target)
                fit_successful = True
                
            except ImportError as e:
                self.logger.warning(f"OpenFE not available: {e}")
                fit_successful = False
                
            except Exception as e:
                self.logger.error(f"Feature generation failed: {e}")
                fit_successful = False
                
            fit_time = time.time() - start_time
            
            # Monitor after fit
            self.monitor_resource_usage('after_fit')
            
            # Test transform if fit was successful
            transform_time = 0
            if fit_successful:
                start_time = time.time()
                try:
                    transformed_data = fg.transform(data)
                    transform_time = time.time() - start_time
                    
                    benchmark_results.update({
                        'output_shape': transformed_data.shape,
                        'output_memory_mb': transformed_data.memory_usage(deep=True).sum() / 1024 / 1024,
                        'feature_expansion_ratio': transformed_data.shape[1] / data.shape[1]
                    })
                    
                except Exception as e:
                    self.logger.error(f"Transform failed: {e}")
                    
            # Monitor final resources
            self.monitor_resource_usage('benchmark_end')
            
            # Calculate performance metrics
            benchmark_results.update({
                'fit_successful': fit_successful,
                'fit_time_seconds': fit_time,
                'transform_time_seconds': transform_time,
                'total_time_seconds': fit_time + transform_time,
                'memory_usage': fg.get_memory_usage() if fit_successful else {},
                'end_time': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            benchmark_results['error'] = str(e)
            benchmark_results['fit_successful'] = False
            
        return benchmark_results
        
    def validate_taiwan_compliance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate Taiwan market compliance."""
        self.logger.info("Validating Taiwan market compliance...")
        
        # Use Taiwan config validation
        validation_results = self.taiwan_config.validate_data_for_taiwan_market(data)
        
        # Additional performance-related checks
        memory_usage_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Check if memory usage is within reasonable bounds
        if memory_usage_mb > 1000:  # > 1GB
            validation_results['warnings'].append(
                f"High memory usage: {memory_usage_mb:.1f}MB - consider chunking"
            )
            
        # Estimate full universe memory requirements
        current_stocks = len(data.index.get_level_values(1).unique())
        estimated_full_memory = memory_usage_mb * (2000 / current_stocks)
        
        validation_results['performance_estimates'] = {
            'current_memory_mb': memory_usage_mb,
            'estimated_full_universe_memory_mb': estimated_full_memory,
            'current_stocks': current_stocks,
            'memory_efficient': estimated_full_memory < 8192  # 8GB limit
        }
        
        return validation_results
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'system_info': self.get_system_info(),
            'benchmarks': self.benchmarks.copy(),
            'resource_usage': self.resource_usage.copy(),
            'summary': {}
        }
        
        # Calculate summary statistics
        if self.resource_usage:
            memory_usage_values = [
                stage_info['memory_rss_mb'] 
                for stage_info in self.resource_usage.values()
            ]
            
            report['summary'] = {
                'peak_memory_mb': max(memory_usage_values),
                'min_memory_mb': min(memory_usage_values),
                'memory_increase_mb': max(memory_usage_values) - min(memory_usage_values),
                'stages_monitored': len(self.resource_usage),
                'report_generated': datetime.now().isoformat()
            }
            
        return report
        
    def run_comprehensive_benchmark(self, 
                                  small_test: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            small_test: If True, use small dataset for testing
                       If False, use full-scale dataset
        """
        self.logger.info("Starting comprehensive OpenFE benchmark suite...")
        
        # Determine test scale
        if small_test:
            n_stocks, n_days, n_features = 10, 30, 20  # Small test
            scale = "small_test"
        else:
            n_stocks, n_days, n_features = 100, 252, 42  # Production scale
            scale = "production_scale"
            
        self.logger.info(f"Running {scale} benchmark...")
        
        # Create benchmark data
        data = self.create_benchmark_data(n_stocks, n_days, n_features)
        
        # Create synthetic target
        np.random.seed(42)
        target = pd.Series(
            np.random.choice([0, 1], len(data)),
            index=data.index,
            name='target'
        )
        
        # Run benchmarks
        self.benchmarks[f'feature_generation_{scale}'] = self.benchmark_feature_generation(data, target)
        self.benchmarks[f'taiwan_compliance_{scale}'] = self.validate_taiwan_compliance(data)
        
        # Generate final report
        report = self.generate_performance_report()
        
        # Log summary
        if 'summary' in report and report['summary']:
            summary = report['summary']
            self.logger.info(
                f"Benchmark complete - Peak memory: {summary['peak_memory_mb']:.1f}MB, "
                f"Memory increase: {summary['memory_increase_mb']:.1f}MB"
            )
            
        return report


def main():
    """Main function for running performance validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenFE Performance Validation')
    parser.add_argument('--scale', choices=['small', 'full'], default='small',
                       help='Benchmark scale (small for testing, full for production)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file for benchmark results (JSON)')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = OpenFEPerformanceMonitor(log_level=args.log_level)
    
    # Run benchmark
    report = monitor.run_comprehensive_benchmark(
        small_test=(args.scale == 'small')
    )
    
    # Save results if requested
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Results saved to {args.output_file}")
        
    # Print summary
    print("\n" + "="*60)
    print("OpenFE Performance Benchmark Summary")
    print("="*60)
    
    if 'summary' in report and report['summary']:
        summary = report['summary']
        print(f"Peak Memory Usage: {summary['peak_memory_mb']:.1f} MB")
        print(f"Memory Increase: {summary['memory_increase_mb']:.1f} MB")
        print(f"Monitoring Stages: {summary['stages_monitored']}")
        
    if 'benchmarks' in report:
        for bench_name, bench_results in report['benchmarks'].items():
            if 'fit_successful' in bench_results:
                status = "✅ SUCCESS" if bench_results['fit_successful'] else "❌ FAILED"
                print(f"{bench_name}: {status}")
                
                if 'total_time_seconds' in bench_results:
                    print(f"  Total Time: {bench_results['total_time_seconds']:.2f}s")
                    
    print("="*60)
    
    return report


if __name__ == "__main__":
    report = main()