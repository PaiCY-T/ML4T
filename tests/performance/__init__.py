"""
Performance Testing Module

Comprehensive performance testing framework for ML4T production readiness validation.
Implements Issue #30 Stream B requirements for production-scale performance testing.

Key Features:
- Production-scale validation (2000+ stocks)
- Taiwan market-specific stress testing  
- Memory and latency optimization benchmarking
- Real-time inference load testing
- Comprehensive performance reporting

Target Performance Benchmarks:
- Latency: <100ms real-time inference
- Memory: <16GB peak usage during full pipeline execution
- Throughput: >1500 predictions/second for 2000-stock universe
- IC Performance: >0.05 Information Coefficient maintained
- Feature Processing: <30 minutes for full feature generation cycle
"""

from .framework import PerformanceTestFramework
from .production_scale import ProductionScaleValidator
from .taiwan_stress import TaiwanMarketStressTester
from .memory_profiler import MemoryProfiler
from .latency_validator import LatencyValidator
from .load_tester import LoadTester
from .reporting import PerformanceReporter

__all__ = [
    'PerformanceTestFramework',
    'ProductionScaleValidator', 
    'TaiwanMarketStressTester',
    'MemoryProfiler',
    'LatencyValidator',
    'LoadTester',
    'PerformanceReporter'
]