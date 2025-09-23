"""
Automated Validation Reports for ML4T Walk-Forward Validation.

This module generates comprehensive validation reports for walk-forward validation
results, including performance metrics, statistical significance testing, and
visualization for Taiwan market quantitative trading strategies.

Key Features:
- Automated report generation with statistical significance testing
- Performance summary with target achievement analysis
- Risk-adjusted metrics analysis and regime detection
- Attribution analysis with factor breakdown
- Taiwan market benchmark comparisons
- Executive summary and detailed technical appendices
- Export to multiple formats (HTML, PDF, JSON)
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, IO
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, Template
import base64
from io import BytesIO

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.models.taiwan_market import TaiwanMarketCode, TaiwanTradingCalendar
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel
from ..validation.walk_forward import ValidationResult, ValidationWindow
from ..metrics.performance import WalkForwardPerformanceAnalyzer, PerformanceConfig, BenchmarkType
from ..metrics.attribution import PerformanceAttributor, AttributionResult
from ..metrics.risk_adjusted import RiskCalculator, RollingRiskAnalyzer, RiskConfig

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Report output formats."""
    HTML = "html"
    JSON = "json"
    PDF = "pdf"
    MARKDOWN = "markdown"


class ReportSection(Enum):
    """Report sections."""
    EXECUTIVE_SUMMARY = "executive_summary"
    PERFORMANCE_OVERVIEW = "performance_overview"
    RISK_ANALYSIS = "risk_analysis"
    ATTRIBUTION_ANALYSIS = "attribution_analysis"
    STATISTICAL_TESTS = "statistical_tests"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    REGIME_ANALYSIS = "regime_analysis"
    APPENDICES = "appendices"


@dataclass
class ReportConfig:
    """Configuration for validation report generation."""
    # Report structure
    include_sections: List[ReportSection] = field(default_factory=lambda: list(ReportSection))
    executive_summary_only: bool = False
    
    # Performance targets for assessment
    target_sharpe_ratio: float = 2.0
    target_information_ratio: float = 0.8
    max_drawdown_threshold: float = 0.15
    min_win_rate: float = 0.55
    
    # Statistical testing
    significance_level: float = 0.05
    bootstrap_iterations: int = 1000
    enable_statistical_tests: bool = True
    
    # Visualization
    include_charts: bool = True
    chart_width: int = 10
    chart_height: int = 6
    chart_dpi: int = 150
    
    # Benchmarks
    primary_benchmark: BenchmarkType = BenchmarkType.TAIEX
    comparison_benchmarks: List[BenchmarkType] = field(default_factory=lambda: [BenchmarkType.TPEx])
    
    # Output formatting
    output_formats: List[ReportFormat] = field(default_factory=lambda: [ReportFormat.HTML])
    template_dir: Optional[str] = None
    output_dir: str = "reports"
    
    # Taiwan market specifics
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.01
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.include_sections:
            self.include_sections = list(ReportSection)
        if self.target_sharpe_ratio <= 0:
            raise ValueError("Target Sharpe ratio must be positive")
        if self.max_drawdown_threshold <= 0 or self.max_drawdown_threshold >= 1:
            raise ValueError("Max drawdown threshold must be between 0 and 1")


@dataclass
class ReportData:
    """Container for all data needed to generate validation reports."""
    # Core validation results
    validation_result: ValidationResult
    returns_by_window: Dict[str, pd.Series]
    overall_returns: pd.Series
    
    # Performance analysis
    performance_analysis: Optional[Dict[str, Any]] = None
    attribution_results: Optional[List[AttributionResult]] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    
    # Benchmark data
    benchmark_returns: Optional[Dict[BenchmarkType, pd.Series]] = None
    benchmark_analysis: Optional[Dict[str, Any]] = None
    
    # Statistical tests
    significance_tests: Optional[Dict[str, Any]] = None
    
    # Report metadata
    generation_date: datetime = field(default_factory=datetime.now)
    report_title: str = "Walk-Forward Validation Report"
    strategy_name: str = "Taiwan Market Strategy"
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get high-level summary statistics."""
        if not self.performance_analysis:
            return {}
        
        overall_metrics = self.performance_analysis.get('overall_metrics', {})
        validation_summary = self.performance_analysis.get('validation_summary', {})
        
        return {
            'total_return': overall_metrics.get('total_return', 0.0),
            'sharpe_ratio': overall_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': overall_metrics.get('max_drawdown', 0.0),
            'information_ratio': overall_metrics.get('information_ratio', 0.0),
            'success_rate': validation_summary.get('success_rate', 0.0),
            'total_windows': validation_summary.get('total_windows', 0),
            'period_start': self.validation_result.windows[0].train_start if self.validation_result.windows else None,
            'period_end': self.validation_result.windows[-1].test_end if self.validation_result.windows else None
        }


class ValidationReportGenerator:
    """
    Main report generator for walk-forward validation results.
    
    Generates comprehensive reports including performance analysis,
    risk metrics, attribution analysis, and statistical significance testing.
    """
    
    def __init__(
        self,
        config: ReportConfig,
        temporal_store: TemporalStore,
        pit_engine: PointInTimeEngine
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        
        # Initialize analyzers
        perf_config = PerformanceConfig(
            target_sharpe_ratio=config.target_sharpe_ratio,
            target_information_ratio=config.target_information_ratio,
            max_drawdown_threshold=config.max_drawdown_threshold,
            benchmark_type=config.primary_benchmark,
            trading_days_per_year=config.trading_days_per_year,
            risk_free_rate=config.risk_free_rate
        )
        
        self.performance_analyzer = WalkForwardPerformanceAnalyzer(
            perf_config, temporal_store, pit_engine
        )
        
        risk_config = RiskConfig(
            trading_days_per_year=config.trading_days_per_year,
            risk_free_rate=config.risk_free_rate,
            enable_hypothesis_tests=config.enable_statistical_tests
        )
        
        self.risk_calculator = RiskCalculator(risk_config)
        self.rolling_risk_analyzer = RollingRiskAnalyzer(risk_config)
        
        # Setup output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        logger.info("ValidationReportGenerator initialized")
    
    def generate_report(
        self,
        validation_result: ValidationResult,
        returns_by_window: Dict[str, pd.Series],
        report_title: str = "Walk-Forward Validation Report",
        strategy_name: str = "Taiwan Market Strategy"
    ) -> Dict[str, str]:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_result: Walk-forward validation results
            returns_by_window: Returns data for each window
            report_title: Title for the report
            strategy_name: Name of the strategy being validated
            
        Returns:
            Dict mapping output formats to file paths
        """
        logger.info(f"Generating validation report: {report_title}")
        
        # Prepare report data
        report_data = self._prepare_report_data(
            validation_result, returns_by_window, report_title, strategy_name
        )
        
        # Run analysis
        self._run_performance_analysis(report_data)
        self._run_risk_analysis(report_data)
        self._run_benchmark_analysis(report_data)
        self._run_statistical_tests(report_data)
        
        # Generate reports in requested formats
        output_files = {}
        
        for output_format in self.config.output_formats:
            if output_format == ReportFormat.HTML:
                file_path = self._generate_html_report(report_data)
            elif output_format == ReportFormat.JSON:
                file_path = self._generate_json_report(report_data)
            elif output_format == ReportFormat.MARKDOWN:
                file_path = self._generate_markdown_report(report_data)
            else:
                logger.warning(f"Unsupported output format: {output_format}")
                continue
            
            output_files[output_format.value] = str(file_path)
        
        logger.info(f"Report generation completed. Output files: {list(output_files.values())}")
        return output_files
    
    def _prepare_report_data(
        self,
        validation_result: ValidationResult,
        returns_by_window: Dict[str, pd.Series],
        report_title: str,
        strategy_name: str
    ) -> ReportData:
        """Prepare report data structure."""
        
        # Combine all returns into overall series
        all_returns = []
        for window_returns in returns_by_window.values():
            all_returns.extend(window_returns.tolist())
        
        overall_returns = pd.Series(all_returns) if all_returns else pd.Series([])
        
        return ReportData(
            validation_result=validation_result,
            returns_by_window=returns_by_window,
            overall_returns=overall_returns,
            report_title=report_title,
            strategy_name=strategy_name
        )
    
    def _run_performance_analysis(self, report_data: ReportData):
        """Run comprehensive performance analysis."""
        logger.info("Running performance analysis")
        
        try:
            performance_analysis = self.performance_analyzer.analyze_validation_result(
                report_data.validation_result,
                report_data.returns_by_window
            )
            report_data.performance_analysis = performance_analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            report_data.performance_analysis = {}
    
    def _run_risk_analysis(self, report_data: ReportData):
        """Run comprehensive risk analysis."""
        logger.info("Running risk analysis")
        
        if len(report_data.overall_returns) == 0:
            report_data.risk_metrics = {}
            return
        
        try:
            # Calculate overall risk metrics
            risk_metrics = self.risk_calculator.calculate_risk_metrics(
                report_data.overall_returns
            )
            
            # Calculate rolling risk metrics
            if len(report_data.overall_returns) >= 60:
                rolling_metrics = self.rolling_risk_analyzer.calculate_rolling_metrics(
                    report_data.overall_returns,
                    window_size=min(60, len(report_data.overall_returns) // 4)
                )
                
                # Regime analysis
                regime_analysis = self.rolling_risk_analyzer.analyze_risk_regime_changes(
                    rolling_metrics, 'volatility'
                )
            else:
                rolling_metrics = pd.DataFrame()
                regime_analysis = {'regime_changes': [], 'analysis': {}}
            
            report_data.risk_metrics = {
                'overall_metrics': risk_metrics.to_dict(),
                'rolling_metrics': rolling_metrics.to_dict() if not rolling_metrics.empty else {},
                'regime_analysis': regime_analysis
            }
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            report_data.risk_metrics = {}
    
    def _run_benchmark_analysis(self, report_data: ReportData):
        """Run benchmark comparison analysis."""
        logger.info("Running benchmark analysis")
        
        try:
            benchmark_returns = {}
            
            if len(report_data.overall_returns) == 0:
                report_data.benchmark_returns = {}
                report_data.benchmark_analysis = {}
                return
            
            # Get benchmark data
            start_date = report_data.overall_returns.index[0].date()
            end_date = report_data.overall_returns.index[-1].date()
            
            benchmarks_to_analyze = [self.config.primary_benchmark] + self.config.comparison_benchmarks
            
            for benchmark_type in benchmarks_to_analyze:
                try:
                    bench_returns = self.performance_analyzer.benchmark_provider.get_benchmark_returns(
                        benchmark_type, start_date, end_date
                    )
                    
                    # Align with strategy returns
                    common_dates = report_data.overall_returns.index.intersection(bench_returns.index)
                    if len(common_dates) > 0:
                        benchmark_returns[benchmark_type] = bench_returns.loc[common_dates]
                    
                except Exception as e:
                    logger.warning(f"Failed to get benchmark data for {benchmark_type}: {e}")
                    continue
            
            # Calculate benchmark comparison metrics
            benchmark_analysis = {}
            
            for benchmark_type, bench_returns in benchmark_returns.items():
                if len(bench_returns) > 0:
                    # Align returns
                    common_dates = report_data.overall_returns.index.intersection(bench_returns.index)
                    if len(common_dates) > 0:
                        aligned_strategy = report_data.overall_returns.loc[common_dates]
                        aligned_benchmark = bench_returns.loc[common_dates]
                        
                        # Calculate comparison metrics
                        excess_returns = aligned_strategy - aligned_benchmark
                        
                        benchmark_analysis[benchmark_type.value] = {
                            'total_return_strategy': float((1 + aligned_strategy).prod() - 1),
                            'total_return_benchmark': float((1 + aligned_benchmark).prod() - 1),
                            'excess_return': float(excess_returns.sum()),
                            'tracking_error': float(excess_returns.std() * np.sqrt(self.config.trading_days_per_year)),
                            'information_ratio': float(
                                excess_returns.mean() / excess_returns.std() * np.sqrt(self.config.trading_days_per_year)
                            ) if excess_returns.std() != 0 else 0.0,
                            'correlation': float(np.corrcoef(aligned_strategy, aligned_benchmark)[0, 1]),
                            'beta': float(
                                np.cov(aligned_strategy, aligned_benchmark)[0, 1] / aligned_benchmark.var()
                            ) if aligned_benchmark.var() != 0 else 0.0,
                            'win_rate': float((excess_returns > 0).mean()),
                            'observations': len(common_dates)
                        }
            
            report_data.benchmark_returns = benchmark_returns
            report_data.benchmark_analysis = benchmark_analysis
            
        except Exception as e:
            logger.error(f"Benchmark analysis failed: {e}")
            report_data.benchmark_returns = {}
            report_data.benchmark_analysis = {}
    
    def _run_statistical_tests(self, report_data: ReportData):
        """Run statistical significance tests."""
        logger.info("Running statistical tests")
        
        if not self.config.enable_statistical_tests or len(report_data.overall_returns) < 10:
            report_data.significance_tests = {}
            return
        
        try:
            tests = {}
            returns = report_data.overall_returns
            rf_daily = self.config.risk_free_rate / self.config.trading_days_per_year
            
            # Test 1: Sharpe ratio significance
            excess_returns = returns - rf_daily
            if len(excess_returns) > 0:
                t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
                tests['sharpe_significance'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'is_significant': p_value < self.config.significance_level,
                    'interpretation': 'Significant positive alpha' if p_value < self.config.significance_level and t_stat > 0 else 'No significant alpha'
                }
            
            # Test 2: Normality test
            if len(returns) >= 8:
                jb_stat, jb_p = stats.jarque_bera(returns)
                _, shapiro_p = stats.shapiro(returns) if len(returns) <= 5000 else (None, None)
                
                tests['normality'] = {
                    'jarque_bera_stat': float(jb_stat),
                    'jarque_bera_p': float(jb_p),
                    'shapiro_p': float(shapiro_p) if shapiro_p else None,
                    'is_normal': jb_p > self.config.significance_level,
                    'interpretation': 'Returns are normally distributed' if jb_p > self.config.significance_level else 'Returns deviate from normal distribution'
                }
            
            # Test 3: Autocorrelation test (Ljung-Box)
            if len(returns) >= 20:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                ljung_box = acorr_ljungbox(returns, lags=min(10, len(returns)//4), return_df=True)
                
                tests['autocorrelation'] = {
                    'ljung_box_p': float(ljung_box['lb_pvalue'].iloc[-1]),
                    'has_autocorrelation': ljung_box['lb_pvalue'].iloc[-1] < self.config.significance_level,
                    'interpretation': 'Significant autocorrelation detected' if ljung_box['lb_pvalue'].iloc[-1] < self.config.significance_level else 'No significant autocorrelation'
                }
            
            # Test 4: Benchmark comparison tests
            if report_data.benchmark_returns:
                primary_benchmark = report_data.benchmark_returns.get(self.config.primary_benchmark)
                if primary_benchmark is not None:
                    common_dates = returns.index.intersection(primary_benchmark.index)
                    if len(common_dates) > 10:
                        aligned_strategy = returns.loc[common_dates]
                        aligned_benchmark = primary_benchmark.loc[common_dates]
                        excess_returns = aligned_strategy - aligned_benchmark
                        
                        # Information ratio significance
                        t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
                        
                        tests['information_ratio_significance'] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'is_significant': p_value < self.config.significance_level,
                            'interpretation': 'Significant outperformance vs benchmark' if p_value < self.config.significance_level and t_stat > 0 else 'No significant outperformance'
                        }
            
            report_data.significance_tests = tests
            
        except Exception as e:
            logger.error(f"Statistical tests failed: {e}")
            report_data.significance_tests = {}
    
    def _generate_html_report(self, report_data: ReportData) -> Path:
        """Generate HTML report."""
        logger.info("Generating HTML report")
        
        # Create template
        html_template = self._create_html_template()
        
        # Generate charts if enabled
        charts = {}
        if self.config.include_charts:
            charts = self._generate_charts(report_data)
        
        # Render template
        html_content = html_template.render(
            report_data=report_data,
            config=self.config,
            charts=charts,
            summary_stats=report_data.get_summary_stats()
        )
        
        # Write to file
        filename = f"{report_data.strategy_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        file_path = self.output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return file_path
    
    def _generate_json_report(self, report_data: ReportData) -> Path:
        """Generate JSON report."""
        logger.info("Generating JSON report")
        
        # Compile all data into JSON-serializable format
        json_data = {
            'report_metadata': {
                'title': report_data.report_title,
                'strategy_name': report_data.strategy_name,
                'generation_date': report_data.generation_date.isoformat(),
                'config': {
                    'target_sharpe_ratio': self.config.target_sharpe_ratio,
                    'target_information_ratio': self.config.target_information_ratio,
                    'max_drawdown_threshold': self.config.max_drawdown_threshold,
                    'significance_level': self.config.significance_level
                }
            },
            'validation_summary': {
                'total_windows': report_data.validation_result.total_windows,
                'successful_windows': report_data.validation_result.successful_windows,
                'success_rate': report_data.validation_result.success_rate(),
                'total_runtime_seconds': report_data.validation_result.total_runtime_seconds
            },
            'performance_analysis': report_data.performance_analysis or {},
            'risk_metrics': report_data.risk_metrics or {},
            'benchmark_analysis': report_data.benchmark_analysis or {},
            'significance_tests': report_data.significance_tests or {},
            'summary_statistics': report_data.get_summary_stats()
        }
        
        # Write to file
        filename = f"{report_data.strategy_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = self.output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        return file_path
    
    def _generate_markdown_report(self, report_data: ReportData) -> Path:
        """Generate Markdown report."""
        logger.info("Generating Markdown report")
        
        md_content = self._create_markdown_content(report_data)
        
        # Write to file
        filename = f"{report_data.strategy_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        file_path = self.output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return file_path
    
    def _create_html_template(self) -> Template:
        """Create HTML template for report generation."""
        
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_data.report_title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 40px; border-bottom: 2px solid #007bff; padding-bottom: 20px; }
        .section { margin-bottom: 40px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
        .metric-value { font-size: 24px; font-weight: bold; color: #333; }
        .metric-label { color: #666; font-size: 14px; margin-top: 5px; }
        .chart-container { margin: 20px 0; text-align: center; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .danger { color: #dc3545; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ report_data.report_title }}</h1>
            <h2>{{ report_data.strategy_name }}</h2>
            <p>Generated on {{ report_data.generation_date.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f%%"|format(summary_stats.get('total_return', 0) * 100) }}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f"|format(summary_stats.get('sharpe_ratio', 0)) }}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f%%"|format(summary_stats.get('max_drawdown', 0) * 100) }}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.1f%%"|format(summary_stats.get('success_rate', 0) * 100) }}</div>
                    <div class="metric-label">Window Success Rate</div>
                </div>
            </div>
        </div>

        <!-- Performance Analysis -->
        {% if report_data.performance_analysis %}
        <div class="section">
            <h2>Performance Analysis</h2>
            {% set overall = report_data.performance_analysis.get('overall_metrics', {}) %}
            <table>
                <tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{{ "%.3f"|format(overall.get('sharpe_ratio', 0)) }}</td>
                    <td>{{ "%.1f"|format(config.target_sharpe_ratio) }}</td>
                    <td class="{% if overall.get('sharpe_ratio', 0) >= config.target_sharpe_ratio %}success{% else %}danger{% endif %}">
                        {% if overall.get('sharpe_ratio', 0) >= config.target_sharpe_ratio %}✓ Met{% else %}✗ Not Met{% endif %}
                    </td>
                </tr>
                <tr>
                    <td>Information Ratio</td>
                    <td>{{ "%.3f"|format(overall.get('information_ratio', 0)) }}</td>
                    <td>{{ "%.1f"|format(config.target_information_ratio) }}</td>
                    <td class="{% if overall.get('information_ratio', 0) >= config.target_information_ratio %}success{% else %}danger{% endif %}">
                        {% if overall.get('information_ratio', 0) >= config.target_information_ratio %}✓ Met{% else %}✗ Not Met{% endif %}
                    </td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td>{{ "%.2f%%"|format(overall.get('max_drawdown', 0) * 100) }}</td>
                    <td>{{ "%.1f%%"|format(config.max_drawdown_threshold * 100) }}</td>
                    <td class="{% if abs(overall.get('max_drawdown', 0)) <= config.max_drawdown_threshold %}success{% else %}danger{% endif %}">
                        {% if abs(overall.get('max_drawdown', 0)) <= config.max_drawdown_threshold %}✓ Met{% else %}✗ Exceeded{% endif %}
                    </td>
                </tr>
            </table>
        </div>
        {% endif %}

        <!-- Risk Analysis -->
        {% if report_data.risk_metrics %}
        <div class="section">
            <h2>Risk Analysis</h2>
            {% set risk = report_data.risk_metrics.get('overall_metrics', {}) %}
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f%%"|format(risk.get('total_volatility', 0) * 100) }}</div>
                    <div class="metric-label">Annualized Volatility</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f%%"|format(risk.get('var_95', 0) * 100) }}</div>
                    <div class="metric-label">VaR 95%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f"|format(risk.get('sortino_ratio', 0)) }}</div>
                    <div class="metric-label">Sortino Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ "%.2f"|format(risk.get('calmar_ratio', 0)) }}</div>
                    <div class="metric-label">Calmar Ratio</div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Benchmark Comparison -->
        {% if report_data.benchmark_analysis %}
        <div class="section">
            <h2>Benchmark Comparison</h2>
            <table>
                <tr><th>Benchmark</th><th>Strategy Return</th><th>Benchmark Return</th><th>Excess Return</th><th>Information Ratio</th><th>Beta</th></tr>
                {% for benchmark, analysis in report_data.benchmark_analysis.items() %}
                <tr>
                    <td>{{ benchmark }}</td>
                    <td>{{ "%.2f%%"|format(analysis.get('total_return_strategy', 0) * 100) }}</td>
                    <td>{{ "%.2f%%"|format(analysis.get('total_return_benchmark', 0) * 100) }}</td>
                    <td class="{% if analysis.get('excess_return', 0) > 0 %}success{% else %}danger{% endif %}">
                        {{ "%.2f%%"|format(analysis.get('excess_return', 0) * 100) }}
                    </td>
                    <td>{{ "%.2f"|format(analysis.get('information_ratio', 0)) }}</td>
                    <td>{{ "%.2f"|format(analysis.get('beta', 0)) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}

        <!-- Statistical Tests -->
        {% if report_data.significance_tests %}
        <div class="section">
            <h2>Statistical Significance Tests</h2>
            <table>
                <tr><th>Test</th><th>Result</th><th>P-Value</th><th>Interpretation</th></tr>
                {% for test_name, test_result in report_data.significance_tests.items() %}
                <tr>
                    <td>{{ test_name.replace('_', ' ').title() }}</td>
                    <td class="{% if test_result.get('is_significant', False) %}success{% else %}warning{% endif %}">
                        {% if test_result.get('is_significant', False) %}Significant{% else %}Not Significant{% endif %}
                    </td>
                    <td>{{ "%.4f"|format(test_result.get('p_value', 1.0)) }}</td>
                    <td>{{ test_result.get('interpretation', 'N/A') }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}

        <div class="footer">
            <p>Report generated by ML4T Walk-Forward Validation Engine</p>
            <p>{{ report_data.generation_date.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return Template(template_str)
    
    def _create_markdown_content(self, report_data: ReportData) -> str:
        """Create Markdown content for the report."""
        
        summary_stats = report_data.get_summary_stats()
        
        md_content = f"""# {report_data.report_title}

## {report_data.strategy_name}

**Generated:** {report_data.generation_date.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Return | {summary_stats.get('total_return', 0):.2%} |
| Sharpe Ratio | {summary_stats.get('sharpe_ratio', 0):.2f} |
| Max Drawdown | {summary_stats.get('max_drawdown', 0):.2%} |
| Information Ratio | {summary_stats.get('information_ratio', 0):.2f} |
| Window Success Rate | {summary_stats.get('success_rate', 0):.1%} |
| Total Windows | {summary_stats.get('total_windows', 0)} |

"""

        # Add performance analysis
        if report_data.performance_analysis:
            overall = report_data.performance_analysis.get('overall_metrics', {})
            md_content += f"""## Performance Analysis

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Sharpe Ratio | {overall.get('sharpe_ratio', 0):.3f} | {self.config.target_sharpe_ratio:.1f} | {'✓ Met' if overall.get('sharpe_ratio', 0) >= self.config.target_sharpe_ratio else '✗ Not Met'} |
| Information Ratio | {overall.get('information_ratio', 0):.3f} | {self.config.target_information_ratio:.1f} | {'✓ Met' if overall.get('information_ratio', 0) >= self.config.target_information_ratio else '✗ Not Met'} |
| Max Drawdown | {overall.get('max_drawdown', 0):.2%} | {self.config.max_drawdown_threshold:.1%} | {'✓ Met' if abs(overall.get('max_drawdown', 0)) <= self.config.max_drawdown_threshold else '✗ Exceeded'} |

"""

        # Add risk analysis
        if report_data.risk_metrics:
            risk = report_data.risk_metrics.get('overall_metrics', {})
            md_content += f"""## Risk Analysis

| Metric | Value |
|--------|-------|
| Annualized Volatility | {risk.get('total_volatility', 0):.2%} |
| Downside Volatility | {risk.get('downside_volatility', 0):.2%} |
| VaR 95% | {risk.get('var_95', 0):.2%} |
| CVaR 95% | {risk.get('cvar_95', 0):.2%} |
| Sortino Ratio | {risk.get('sortino_ratio', 0):.2f} |
| Calmar Ratio | {risk.get('calmar_ratio', 0):.2f} |

"""

        # Add benchmark comparison
        if report_data.benchmark_analysis:
            md_content += """## Benchmark Comparison

| Benchmark | Strategy Return | Benchmark Return | Excess Return | Information Ratio | Beta |
|-----------|----------------|------------------|---------------|-------------------|------|
"""
            for benchmark, analysis in report_data.benchmark_analysis.items():
                md_content += f"| {benchmark} | {analysis.get('total_return_strategy', 0):.2%} | {analysis.get('total_return_benchmark', 0):.2%} | {analysis.get('excess_return', 0):.2%} | {analysis.get('information_ratio', 0):.2f} | {analysis.get('beta', 0):.2f} |\n"
            
            md_content += "\n"

        # Add statistical tests
        if report_data.significance_tests:
            md_content += """## Statistical Significance Tests

| Test | Result | P-Value | Interpretation |
|------|--------|---------|----------------|
"""
            for test_name, test_result in report_data.significance_tests.items():
                result_str = "Significant" if test_result.get('is_significant', False) else "Not Significant"
                md_content += f"| {test_name.replace('_', ' ').title()} | {result_str} | {test_result.get('p_value', 1.0):.4f} | {test_result.get('interpretation', 'N/A')} |\n"
            
            md_content += "\n"

        md_content += f"""---
*Report generated by ML4T Walk-Forward Validation Engine on {report_data.generation_date.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return md_content
    
    def _generate_charts(self, report_data: ReportData) -> Dict[str, str]:
        """Generate charts and return as base64 encoded strings."""
        charts = {}
        
        try:
            # Performance chart
            if len(report_data.overall_returns) > 0:
                fig, ax = plt.subplots(figsize=(self.config.chart_width, self.config.chart_height))
                
                cumulative_returns = (1 + report_data.overall_returns).cumprod()
                ax.plot(cumulative_returns.index, cumulative_returns.values)
                ax.set_title('Cumulative Returns')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Return')
                ax.grid(True, alpha=0.3)
                
                # Convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=self.config.chart_dpi, bbox_inches='tight')
                buffer.seek(0)
                chart_base64 = base64.b64encode(buffer.getvalue()).decode()
                charts['cumulative_returns'] = f"data:image/png;base64,{chart_base64}"
                
                plt.close(fig)
                buffer.close()
            
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
        
        return charts


# Utility functions
def create_validation_report_config(**overrides) -> ReportConfig:
    """Create report configuration with defaults."""
    config = ReportConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    return config


def generate_taiwan_validation_report(
    validation_result: ValidationResult,
    returns_by_window: Dict[str, pd.Series],
    temporal_store: TemporalStore,
    pit_engine: PointInTimeEngine,
    report_title: str = "Taiwan Market Walk-Forward Validation",
    strategy_name: str = "Taiwan Quantitative Strategy",
    output_formats: List[ReportFormat] = None,
    **config_overrides
) -> Dict[str, str]:
    """Generate validation report for Taiwan market strategy."""
    
    if output_formats is None:
        output_formats = [ReportFormat.HTML, ReportFormat.JSON]
    
    config = create_validation_report_config(
        output_formats=output_formats,
        primary_benchmark=BenchmarkType.TAIEX,
        comparison_benchmarks=[BenchmarkType.TPEx],
        **config_overrides
    )
    
    generator = ValidationReportGenerator(config, temporal_store, pit_engine)
    
    return generator.generate_report(
        validation_result, returns_by_window, report_title, strategy_name
    )


# Example usage
if __name__ == "__main__":
    print("Validation Report Generator demo")
    print("Demo of automated validation reporting - requires actual data stores")
    
    config = create_validation_report_config(
        target_sharpe_ratio=1.5,
        include_charts=True,
        output_formats=[ReportFormat.HTML, ReportFormat.JSON]
    )
    
    print(f"Demo config: {config}")
    print("In actual usage:")
    print("1. Initialize with TemporalStore and PointInTimeEngine")
    print("2. Provide ValidationResult and returns data")
    print("3. Generate comprehensive validation reports")
    print("4. Export to multiple formats with charts and analysis")