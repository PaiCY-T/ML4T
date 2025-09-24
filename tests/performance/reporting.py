"""
Performance Reporting Module

Comprehensive reporting and visualization for ML4T performance testing results.
Generates detailed reports, charts, and analysis for production readiness assessment.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from .framework import PerformanceResult


class PerformanceReporter:
    """Comprehensive performance reporting and analysis."""
    
    def __init__(self, output_dir: str = 'performance_results'):
        """
        Initialize performance reporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
        
        # Setup matplotlib for headless operation
        plt.style.use('seaborn-v0_8')
        
    def save_results(self, 
                    test_results: List[PerformanceResult], 
                    summary_report: Dict[str, Any]) -> None:
        """
        Save performance test results and generate reports.
        
        Args:
            test_results: List of performance test results
            summary_report: Summary report data
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results as JSON
        results_file = self.output_dir / f'performance_results_{timestamp}.json'
        self._save_results_json(test_results, summary_report, results_file)
        
        # Generate detailed report
        report_file = self.output_dir / f'performance_report_{timestamp}.md'
        self._generate_markdown_report(test_results, summary_report, report_file)
        
        # Generate visualizations
        charts_dir = self.output_dir / f'charts_{timestamp}'
        charts_dir.mkdir(exist_ok=True)
        self._generate_performance_charts(test_results, summary_report, charts_dir)
        
        # Generate CSV summary
        csv_file = self.output_dir / f'performance_summary_{timestamp}.csv'
        self._generate_csv_summary(test_results, csv_file)
        
        self.logger.info(f"Performance results saved to {self.output_dir}")
        
    def _save_results_json(self, 
                          test_results: List[PerformanceResult],
                          summary_report: Dict[str, Any],
                          output_file: Path) -> None:
        """Save results in JSON format."""
        try:
            # Convert results to serializable format
            serializable_results = []
            for result in test_results:
                result_dict = asdict(result) if hasattr(result, '__dict__') else {
                    'benchmark_name': result.benchmark_name,
                    'success': result.success,
                    'execution_time_seconds': result.execution_time_seconds,
                    'memory_usage_mb': result.memory_usage_mb,
                    'peak_memory_mb': result.peak_memory_mb,
                    'cpu_usage_percent': result.cpu_usage_percent,
                    'throughput_ops_per_sec': result.throughput_ops_per_sec,
                    'latency_p50_ms': result.latency_p50_ms,
                    'latency_p95_ms': result.latency_p95_ms,
                    'latency_p99_ms': result.latency_p99_ms,
                    'information_coefficient': result.information_coefficient,
                    'error_message': result.error_message,
                    'warnings': result.warnings,
                    'test_specific_metrics': result.test_specific_metrics,
                    'meets_production_requirements': result.meets_production_requirements()
                }
                serializable_results.append(result_dict)
                
            data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'ml4t_version': '1.0.0',
                    'test_framework_version': '1.0.0',
                    'issue_number': '#30',
                    'stream': 'B - Performance & Load Testing'
                },
                'test_results': serializable_results,
                'summary_report': summary_report
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {e}")
            
    def _generate_markdown_report(self,
                                 test_results: List[PerformanceResult],
                                 summary_report: Dict[str, Any],
                                 output_file: Path) -> None:
        """Generate comprehensive markdown report."""
        try:
            report_lines = [
                "# ML4T Performance Testing Report",
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
                f"**Issue**: #30 - Production Readiness Testing  ",
                f"**Stream**: B - Performance & Load Testing  ",
                "",
                "## Executive Summary",
                "",
            ]
            
            # Executive summary from summary report
            if 'test_summary' in summary_report:
                summary = summary_report['test_summary']
                report_lines.extend([
                    f"- **Total Tests**: {summary.get('total_tests', 0)}",
                    f"- **Passed Tests**: {summary.get('passed_tests', 0)}",
                    f"- **Success Rate**: {summary.get('success_rate', 0):.1%}",
                    f"- **Production Ready**: {summary.get('production_readiness_rate', 0):.1%}",
                    ""
                ])
                
            # Performance metrics summary
            if 'performance_metrics' in summary_report:
                perf = summary_report['performance_metrics']
                report_lines.extend([
                    "### Key Performance Metrics",
                    "",
                    f"- **Peak Memory Usage**: {perf.get('peak_memory_usage_mb', 0):.1f} MB",
                    f"- **Average Execution Time**: {perf.get('avg_execution_time_seconds', 0):.2f} seconds",
                    f"- **Total Execution Time**: {perf.get('total_execution_time_seconds', 0):.1f} seconds",
                    ""
                ])
                
            # Key metrics
            if 'key_metrics' in summary_report:
                key_metrics = summary_report['key_metrics']
                report_lines.extend([
                    "### Performance Targets Validation",
                    "",
                    "| Metric | Target | Achieved | Status |",
                    "|--------|--------|----------|--------|",
                ])
                
                # Throughput
                max_throughput = key_metrics.get('max_throughput_ops_per_sec')
                if max_throughput:
                    throughput_status = "✅ PASS" if max_throughput >= 1500 else "❌ FAIL"
                    report_lines.append(f"| Throughput | ≥1500 ops/sec | {max_throughput:.0f} ops/sec | {throughput_status} |")
                    
                # Latency
                min_latency = key_metrics.get('min_latency_p95_ms')
                if min_latency:
                    latency_status = "✅ PASS" if min_latency <= 100 else "❌ FAIL"
                    report_lines.append(f"| Latency P95 | ≤100ms | {min_latency:.1f}ms | {latency_status} |")
                    
                # Information Coefficient
                max_ic = key_metrics.get('max_information_coefficient')
                if max_ic:
                    ic_status = "✅ PASS" if max_ic >= 0.05 else "❌ FAIL"
                    report_lines.append(f"| Information Coefficient | ≥0.05 | {max_ic:.3f} | {ic_status} |")
                    
                report_lines.append("")
                
            # Production requirements
            if 'production_requirements' in summary_report:
                prod_req = summary_report['production_requirements']
                report_lines.extend([
                    "### Production Requirements Check",
                    "",
                    "| Requirement | Status | Details |",
                    "|-------------|--------|---------|",
                    f"| Memory Limit (16GB) | {'✅ PASS' if prod_req.get('memory_limit_16gb_met') else '❌ FAIL'} | Peak memory within 16GB limit |",
                    f"| Latency Target (100ms) | {'✅ PASS' if prod_req.get('latency_100ms_met') else '❌ FAIL'} | P95 latency under 100ms |",
                    f"| IC Requirement (0.05) | {'✅ PASS' if prod_req.get('ic_005_met') else '❌ FAIL'} | Information coefficient above 0.05 |",
                    f"| Throughput (1500 ops/sec) | {'✅ PASS' if prod_req.get('throughput_1500_met') else '❌ FAIL'} | Throughput meets target |",
                    ""
                ])
                
            # Individual test results
            report_lines.extend([
                "## Detailed Test Results",
                "",
                "| Test Name | Status | Duration | Memory | Throughput | Latency P95 | IC |",
                "|-----------|--------|----------|--------|------------|-------------|-----|"
            ])
            
            for result in test_results:
                status = "✅ PASS" if result.success and result.meets_production_requirements() else "❌ FAIL"
                throughput = f"{result.throughput_ops_per_sec:.0f}" if result.throughput_ops_per_sec else "N/A"
                latency = f"{result.latency_p95_ms:.1f}ms" if result.latency_p95_ms else "N/A"
                ic = f"{result.information_coefficient:.3f}" if result.information_coefficient else "N/A"
                
                report_lines.append(
                    f"| {result.benchmark_name} | {status} | {result.execution_time_seconds:.1f}s | "
                    f"{result.peak_memory_mb:.0f}MB | {throughput} ops/sec | {latency} | {ic} |"
                )
                
            report_lines.append("")
            
            # Recommendations
            if 'recommendations' in summary_report:
                report_lines.extend([
                    "## Recommendations",
                    ""
                ])
                for i, rec in enumerate(summary_report['recommendations'], 1):
                    report_lines.append(f"{i}. {rec}")
                    
                report_lines.append("")
                
            # Taiwan Market Specific Analysis
            taiwan_results = [r for r in test_results if 'taiwan' in r.benchmark_name.lower()]
            if taiwan_results:
                report_lines.extend([
                    "## Taiwan Market Specific Results",
                    "",
                    "Analysis of Taiwan Stock Exchange (TSE) and Taipei Exchange (TPEx) specific performance:",
                    ""
                ])
                
                for result in taiwan_results:
                    report_lines.extend([
                        f"### {result.benchmark_name}",
                        f"- **Status**: {'✅ Passed' if result.success else '❌ Failed'}",
                        f"- **Duration**: {result.execution_time_seconds:.1f} seconds",
                        f"- **Memory Usage**: {result.peak_memory_mb:.1f} MB",
                    ])
                    
                    if result.error_message:
                        report_lines.append(f"- **Error**: {result.error_message}")
                        
                    if result.warnings:
                        report_lines.append("- **Warnings**:")
                        for warning in result.warnings:
                            report_lines.append(f"  - {warning}")
                            
                    report_lines.append("")
                    
            # Test environment
            report_lines.extend([
                "## Test Environment",
                "",
                "- **Platform**: Linux WSL2",
                "- **Python Version**: 3.x",
                "- **ML4T Version**: 1.0.0",
                "- **Test Date**: " + datetime.now().strftime('%Y-%m-%d'),
                ""
            ])
            
            # Write report
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
                
        except Exception as e:
            self.logger.error(f"Failed to generate markdown report: {e}")
            
    def _generate_performance_charts(self,
                                   test_results: List[PerformanceResult],
                                   summary_report: Dict[str, Any],
                                   output_dir: Path) -> None:
        """Generate performance visualization charts."""
        try:
            # Memory usage chart
            self._create_memory_usage_chart(test_results, output_dir / 'memory_usage.png')
            
            # Execution time chart  
            self._create_execution_time_chart(test_results, output_dir / 'execution_times.png')
            
            # Throughput comparison
            self._create_throughput_chart(test_results, output_dir / 'throughput_comparison.png')
            
            # Latency distribution
            self._create_latency_chart(test_results, output_dir / 'latency_distribution.png')
            
            # Requirements compliance
            self._create_compliance_chart(test_results, output_dir / 'requirements_compliance.png')
            
            # Performance overview dashboard
            self._create_dashboard(test_results, summary_report, output_dir / 'performance_dashboard.png')
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance charts: {e}")
            
    def _create_memory_usage_chart(self, test_results: List[PerformanceResult], output_file: Path):
        """Create memory usage comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        test_names = [r.benchmark_name for r in test_results if r.success]
        memory_usage = [r.peak_memory_mb for r in test_results if r.success]
        
        if test_names and memory_usage:
            bars = ax.bar(range(len(test_names)), memory_usage, 
                         color=['green' if m <= 16384 else 'red' for m in memory_usage])
            
            # Add 16GB limit line
            ax.axhline(y=16384, color='red', linestyle='--', alpha=0.7, label='16GB Production Limit')
            
            ax.set_xlabel('Test Name')
            ax.set_ylabel('Peak Memory Usage (MB)')
            ax.set_title('Memory Usage by Test - ML4T Performance Testing')
            ax.set_xticks(range(len(test_names)))
            ax.set_xticklabels(test_names, rotation=45, ha='right')
            ax.legend()
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                       f'{height:.0f}MB', ha='center', va='bottom')
                       
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
    def _create_execution_time_chart(self, test_results: List[PerformanceResult], output_file: Path):
        """Create execution time comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        test_names = [r.benchmark_name for r in test_results if r.success]
        execution_times = [r.execution_time_seconds for r in test_results if r.success]
        
        if test_names and execution_times:
            bars = ax.bar(range(len(test_names)), execution_times, color='skyblue', alpha=0.7)
            
            ax.set_xlabel('Test Name')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title('Execution Time by Test - ML4T Performance Testing')
            ax.set_xticks(range(len(test_names)))
            ax.set_xticklabels(test_names, rotation=45, ha='right')
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(execution_times)*0.01,
                       f'{height:.1f}s', ha='center', va='bottom')
                       
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
    def _create_throughput_chart(self, test_results: List[PerformanceResult], output_file: Path):
        """Create throughput comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        throughput_results = [r for r in test_results if r.success and r.throughput_ops_per_sec]
        
        if throughput_results:
            test_names = [r.benchmark_name for r in throughput_results]
            throughput_values = [r.throughput_ops_per_sec for r in throughput_results]
            
            bars = ax.bar(range(len(test_names)), throughput_values,
                         color=['green' if t >= 1500 else 'orange' if t >= 1000 else 'red' 
                               for t in throughput_values])
            
            # Add target line
            ax.axhline(y=1500, color='green', linestyle='--', alpha=0.7, label='Target: 1500 ops/sec')
            
            ax.set_xlabel('Test Name')
            ax.set_ylabel('Throughput (ops/sec)')
            ax.set_title('Throughput Performance - ML4T Performance Testing')
            ax.set_xticks(range(len(test_names)))
            ax.set_xticklabels(test_names, rotation=45, ha='right')
            ax.legend()
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(throughput_values)*0.01,
                       f'{height:.0f}', ha='center', va='bottom')
                       
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
    def _create_latency_chart(self, test_results: List[PerformanceResult], output_file: Path):
        """Create latency distribution chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        latency_results = [r for r in test_results if r.success and r.latency_p95_ms]
        
        if latency_results:
            test_names = [r.benchmark_name for r in latency_results]
            p95_latencies = [r.latency_p95_ms for r in latency_results]
            p99_latencies = [r.latency_p99_ms for r in latency_results if r.latency_p99_ms]
            
            # P95 latency chart
            bars1 = ax1.bar(range(len(test_names)), p95_latencies,
                           color=['green' if l <= 100 else 'red' for l in p95_latencies])
            ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100ms Target')
            ax1.set_xlabel('Test Name')
            ax1.set_ylabel('Latency P95 (ms)')
            ax1.set_title('P95 Latency by Test')
            ax1.set_xticks(range(len(test_names)))
            ax1.set_xticklabels(test_names, rotation=45, ha='right')
            ax1.legend()
            
            # P99 latency chart (if available)
            if p99_latencies and len(p99_latencies) == len(test_names):
                bars2 = ax2.bar(range(len(test_names)), p99_latencies, color='lightcoral', alpha=0.7)
                ax2.set_xlabel('Test Name')
                ax2.set_ylabel('Latency P99 (ms)')
                ax2.set_title('P99 Latency by Test')
                ax2.set_xticks(range(len(test_names)))
                ax2.set_xticklabels(test_names, rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, 'P99 Data Not Available', transform=ax2.transAxes,
                        ha='center', va='center', fontsize=12, alpha=0.5)
                        
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
    def _create_compliance_chart(self, test_results: List[PerformanceResult], output_file: Path):
        """Create requirements compliance chart."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compliance analysis
        compliant_tests = [r for r in test_results if r.success and r.meets_production_requirements()]
        non_compliant_tests = [r for r in test_results if r.success and not r.meets_production_requirements()]
        failed_tests = [r for r in test_results if not r.success]
        
        categories = ['Compliant', 'Non-Compliant', 'Failed']
        counts = [len(compliant_tests), len(non_compliant_tests), len(failed_tests)]
        colors = ['green', 'orange', 'red']
        
        wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%')
        ax.set_title('Production Requirements Compliance - ML4T Performance Testing')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_dashboard(self, 
                         test_results: List[PerformanceResult],
                         summary_report: Dict[str, Any],
                         output_file: Path):
        """Create comprehensive performance dashboard."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Summary metrics (top row)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Performance charts (middle and bottom rows)
        ax4 = fig.add_subplot(gs[1, :2])  # Memory usage
        ax5 = fig.add_subplot(gs[1, 2])   # Compliance pie
        ax6 = fig.add_subplot(gs[2, :])   # Throughput and latency
        
        # Summary statistics
        successful_tests = [r for r in test_results if r.success]
        
        if 'test_summary' in summary_report:
            summary = summary_report['test_summary']
            
            # Test success rate
            ax1.pie([summary.get('passed_tests', 0), summary.get('total_tests', 1) - summary.get('passed_tests', 0)],
                   labels=['Passed', 'Failed'], colors=['green', 'red'], autopct='%1.1f%%')
            ax1.set_title('Test Success Rate')
            
        # Memory usage summary
        if successful_tests:
            memory_values = [r.peak_memory_mb for r in successful_tests]
            ax2.hist(memory_values, bins=10, color='skyblue', alpha=0.7)
            ax2.axvline(x=16384, color='red', linestyle='--', label='16GB Limit')
            ax2.set_xlabel('Memory (MB)')
            ax2.set_ylabel('Tests')
            ax2.set_title('Memory Usage Distribution')
            ax2.legend()
            
        # Execution time summary
        if successful_tests:
            time_values = [r.execution_time_seconds for r in successful_tests]
            ax3.hist(time_values, bins=10, color='lightgreen', alpha=0.7)
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Tests')
            ax3.set_title('Execution Time Distribution')
            
        # Detailed performance metrics
        if successful_tests:
            # Memory usage by test
            test_names = [r.benchmark_name[:20] for r in successful_tests]  # Truncate names
            memory_usage = [r.peak_memory_mb for r in successful_tests]
            
            bars = ax4.bar(range(len(test_names)), memory_usage)
            ax4.axhline(y=16384, color='red', linestyle='--', alpha=0.7)
            ax4.set_xlabel('Tests')
            ax4.set_ylabel('Peak Memory (MB)')
            ax4.set_title('Memory Usage by Test')
            ax4.set_xticks(range(len(test_names)))
            ax4.set_xticklabels(test_names, rotation=45, ha='right', fontsize=8)
            
        # Compliance summary
        compliant = len([r for r in successful_tests if r.meets_production_requirements()])
        non_compliant = len(successful_tests) - compliant
        
        ax5.pie([compliant, non_compliant], labels=['Compliant', 'Non-Compliant'],
               colors=['green', 'orange'], autopct='%1.1f%%')
        ax5.set_title('Requirements Compliance')
        
        # Combined throughput and latency
        throughput_tests = [r for r in successful_tests if r.throughput_ops_per_sec]
        latency_tests = [r for r in successful_tests if r.latency_p95_ms]
        
        if throughput_tests or latency_tests:
            ax6_twin = ax6.twinx()
            
            if throughput_tests:
                x_pos = range(len(throughput_tests))
                throughput_values = [r.throughput_ops_per_sec for r in throughput_tests]
                bars1 = ax6.bar([x - 0.2 for x in x_pos], throughput_values, width=0.4, 
                              color='blue', alpha=0.7, label='Throughput')
                ax6.set_ylabel('Throughput (ops/sec)', color='blue')
                
            if latency_tests:
                x_pos = range(len(latency_tests))
                latency_values = [r.latency_p95_ms for r in latency_tests]
                bars2 = ax6_twin.bar([x + 0.2 for x in x_pos], latency_values, width=0.4,
                                   color='red', alpha=0.7, label='Latency P95')
                ax6_twin.set_ylabel('Latency P95 (ms)', color='red')
                
            ax6.set_xlabel('Tests')
            ax6.set_title('Throughput vs Latency Performance')
            
            # Combine legends
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
        plt.suptitle('ML4T Performance Testing Dashboard - Issue #30 Stream B', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_csv_summary(self, test_results: List[PerformanceResult], output_file: Path):
        """Generate CSV summary of test results."""
        try:
            data = []
            for result in test_results:
                data.append({
                    'test_name': result.benchmark_name,
                    'success': result.success,
                    'production_ready': result.meets_production_requirements(),
                    'execution_time_seconds': result.execution_time_seconds,
                    'peak_memory_mb': result.peak_memory_mb,
                    'cpu_usage_percent': result.cpu_usage_percent,
                    'throughput_ops_per_sec': result.throughput_ops_per_sec,
                    'latency_p50_ms': result.latency_p50_ms,
                    'latency_p95_ms': result.latency_p95_ms,
                    'latency_p99_ms': result.latency_p99_ms,
                    'information_coefficient': result.information_coefficient,
                    'sharpe_ratio': result.sharpe_ratio,
                    'error_message': result.error_message,
                    'warning_count': len(result.warnings)
                })
                
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            
        except Exception as e:
            self.logger.error(f"Failed to generate CSV summary: {e}")
            
    def generate_production_readiness_certificate(self, 
                                                 test_results: List[PerformanceResult],
                                                 summary_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate production readiness certificate."""
        # Analyze results for certification
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if r.success])
        production_ready_tests = len([r for r in test_results if r.success and r.meets_production_requirements()])
        
        # Key requirements check
        requirements_met = {
            'all_tests_passed': successful_tests == total_tests,
            'all_requirements_met': production_ready_tests == total_tests,
            'memory_compliant': all(r.peak_memory_mb <= 16384 for r in test_results if r.success),
            'latency_compliant': all(r.latency_p95_ms <= 100 for r in test_results 
                                   if r.success and r.latency_p95_ms),
            'throughput_achieved': any(r.throughput_ops_per_sec >= 1500 for r in test_results 
                                     if r.success and r.throughput_ops_per_sec),
            'ic_achieved': any(r.information_coefficient >= 0.05 for r in test_results 
                             if r.success and r.information_coefficient)
        }
        
        overall_ready = all(requirements_met.values())
        
        certificate = {
            'certificate_id': f"ML4T-PERF-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'issue_number': '#30',
            'stream': 'B - Performance & Load Testing',
            'certification_date': datetime.now().isoformat(),
            'production_ready': overall_ready,
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'production_ready_tests': production_ready_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'readiness_rate': production_ready_tests / total_tests if total_tests > 0 else 0
            },
            'requirements_validation': requirements_met,
            'certification_status': 'CERTIFIED' if overall_ready else 'NOT_CERTIFIED',
            'next_steps': self._generate_next_steps(overall_ready, requirements_met, test_results)
        }
        
        # Save certificate
        cert_file = self.output_dir / f"production_readiness_certificate_{certificate['certificate_id']}.json"
        with open(cert_file, 'w') as f:
            json.dump(certificate, f, indent=2, default=str)
            
        return certificate
        
    def _generate_next_steps(self, 
                           overall_ready: bool, 
                           requirements_met: Dict[str, bool],
                           test_results: List[PerformanceResult]) -> List[str]:
        """Generate next steps based on certification results."""
        if overall_ready:
            return [
                "✅ All performance requirements met - System ready for production deployment",
                "📋 Proceed with final integration testing (Stream A & C)",
                "🚀 Prepare for production deployment with Taiwan market validation"
            ]
        else:
            next_steps = []
            
            if not requirements_met.get('all_tests_passed'):
                failed_tests = [r.benchmark_name for r in test_results if not r.success]
                next_steps.append(f"❌ Fix failed tests: {', '.join(failed_tests)}")
                
            if not requirements_met.get('memory_compliant'):
                next_steps.append("🧠 Optimize memory usage - implement chunking or streaming processing")
                
            if not requirements_met.get('latency_compliant'):
                next_steps.append("⚡ Optimize latency - implement caching and parallel processing")
                
            if not requirements_met.get('throughput_achieved'):
                next_steps.append("🚀 Optimize throughput - implement batch processing and load balancing")
                
            if not requirements_met.get('ic_achieved'):
                next_steps.append("📊 Improve model performance - enhance feature engineering and model tuning")
                
            return next_steps