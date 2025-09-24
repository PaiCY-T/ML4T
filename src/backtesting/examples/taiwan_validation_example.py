"""
Taiwan Market Walk-Forward Validation Example.

This example demonstrates the complete walk-forward validation workflow with
performance attribution and metrics analysis for Taiwan market strategies.

Usage:
    python taiwan_validation_example.py

Features demonstrated:
- Walk-forward validation setup
- Performance metrics calculation
- Attribution analysis
- Risk-adjusted metrics
- Automated report generation
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd

# Import validation framework components
from ..validation.walk_forward import (
    WalkForwardSplitter, WalkForwardValidator, WalkForwardConfig,
    create_default_config
)
from ..validation.time_series_cv import PurgedKFold, create_taiwan_purged_kfold
from ..validation.taiwan_specific import TaiwanMarketValidator, create_standard_taiwan_validator

# Import metrics and attribution
from ..metrics import (
    create_taiwan_performance_analyzer,
    create_taiwan_attribution_engine,
    create_taiwan_risk_calculator,
    BenchmarkType
)

# Import reporting
from ..reporting import generate_taiwan_validation_report, ReportFormat

# Mock imports for demonstration (replace with actual imports in production)
from ...data.core.temporal import TemporalStore, DataType
from ...data.pipeline.pit_engine import PointInTimeEngine

logger = logging.getLogger(__name__)


class TaiwanValidationExample:
    """
    Complete example of Taiwan market walk-forward validation.
    
    Demonstrates the integration of all validation components:
    - Walk-forward validation with 156-week training / 26-week testing
    - Taiwan market calendar and settlement handling
    - Performance metrics and attribution analysis
    - Risk-adjusted metrics calculation
    - Automated report generation
    """
    
    def __init__(self):
        self.setup_logging()
        logger.info("Taiwan Validation Example initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def create_mock_data_infrastructure(self):
        """Create mock data infrastructure for demonstration."""
        logger.info("Creating mock data infrastructure")
        
        # In production, these would be actual TemporalStore and PointInTimeEngine
        # instances connected to your data sources
        
        class MockTemporalStore:
            def __init__(self):
                self.data = {}
            
            def get(self, symbol, data_type, date_range):
                # Return mock data
                return []
        
        class MockPointInTimeEngine:
            def __init__(self, temporal_store):
                self.temporal_store = temporal_store
            
            def query(self, pit_query):
                # Return mock query results
                return {}
            
            def check_data_availability(self, pit_query):
                return True
        
        temporal_store = MockTemporalStore()
        pit_engine = MockPointInTimeEngine(temporal_store)
        
        return temporal_store, pit_engine
    
    def generate_sample_strategy_returns(
        self,
        start_date: date,
        end_date: date,
        annual_return: float = 0.12,
        annual_volatility: float = 0.15
    ) -> pd.Series:
        """Generate sample strategy returns for demonstration."""
        
        # Calculate number of trading days
        total_days = (end_date - start_date).days
        trading_days = int(total_days * 252 / 365)  # Approximate trading days
        
        # Daily parameters
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # Generate returns with some autocorrelation (more realistic)
        np.random.seed(42)  # For reproducible results
        
        returns = []
        prev_return = 0
        
        for i in range(trading_days):
            # Add some momentum/mean reversion
            momentum = 0.1 * prev_return  # Small momentum effect
            noise = np.random.normal(0, daily_volatility)
            
            daily_ret = daily_return + momentum + noise
            returns.append(daily_ret)
            prev_return = daily_ret
        
        # Create date index
        dates = pd.date_range(start=start_date, periods=trading_days, freq='B')
        
        return pd.Series(returns, index=dates)
    
    def run_walk_forward_validation(
        self,
        temporal_store,
        pit_engine,
        symbols: List[str],
        start_date: date,
        end_date: date
    ):
        """Run complete walk-forward validation."""
        logger.info("Starting walk-forward validation")
        
        # 1. Setup walk-forward configuration
        wf_config = create_default_config(
            train_weeks=156,    # 3 years training
            test_weeks=26,      # 6 months testing
            purge_weeks=2,      # 2 week purge period
            rebalance_weeks=4,  # Monthly rebalancing
            use_taiwan_calendar=True,
            settlement_lag_days=2
        )
        
        # 2. Create walk-forward splitter
        splitter = WalkForwardSplitter(wf_config, temporal_store, pit_engine)
        
        # 3. Create validator
        validator = WalkForwardValidator(splitter, symbols)
        
        # 4. Run validation
        validation_result = validator.run_validation(
            start_date, end_date, validate_windows=True
        )
        
        logger.info(f"Validation completed: {validation_result.successful_windows}/{validation_result.total_windows} windows successful")
        
        return validation_result
    
    def generate_window_returns(
        self,
        validation_result,
        base_annual_return: float = 0.12,
        base_annual_volatility: float = 0.15
    ) -> Dict[str, pd.Series]:
        """Generate sample returns for each validation window."""
        logger.info("Generating sample returns for validation windows")
        
        returns_by_window = {}
        
        for window in validation_result.windows:
            if window.window_id:
                # Generate returns for this window's test period
                window_returns = self.generate_sample_strategy_returns(
                    window.test_start,
                    window.test_end,
                    annual_return=base_annual_return + np.random.normal(0, 0.02),  # Add some variation
                    annual_volatility=base_annual_volatility + np.random.normal(0, 0.01)
                )
                returns_by_window[window.window_id] = window_returns
        
        logger.info(f"Generated returns for {len(returns_by_window)} windows")
        return returns_by_window
    
    def run_performance_analysis(
        self,
        validation_result,
        returns_by_window: Dict[str, pd.Series],
        temporal_store,
        pit_engine
    ):
        """Run comprehensive performance analysis."""
        logger.info("Running performance analysis")
        
        # Create performance analyzer
        performance_analyzer = create_taiwan_performance_analyzer(
            temporal_store,
            pit_engine,
            benchmark_type=BenchmarkType.TAIEX,
            target_sharpe_ratio=2.0,
            target_information_ratio=0.8,
            max_drawdown_threshold=0.15
        )
        
        # Run analysis
        performance_analysis = performance_analyzer.analyze_validation_result(
            validation_result,
            returns_by_window
        )
        
        logger.info("Performance analysis completed")
        return performance_analysis
    
    def run_attribution_analysis(
        self,
        returns_by_window: Dict[str, pd.Series],
        temporal_store,
        pit_engine
    ):
        """Run performance attribution analysis."""
        logger.info("Running attribution analysis")
        
        # Create attribution engine
        attributor = create_taiwan_attribution_engine(temporal_store, pit_engine)
        
        # For demonstration, create mock portfolio and benchmark weights
        sample_symbols = ['2330.TW', '2317.TW', '2454.TW', '2412.TW', '6505.TW']
        
        attribution_results = []
        
        for window_id, window_returns in returns_by_window.items():
            if len(window_returns) > 0:
                try:
                    # Create mock weights (equal weight portfolio)
                    n_symbols = len(sample_symbols)
                    equal_weight = 1.0 / n_symbols
                    
                    portfolio_weights = {
                        symbol: pd.Series(equal_weight, index=window_returns.index)
                        for symbol in sample_symbols
                    }
                    
                    # Benchmark weights (mock TAIEX weights)
                    benchmark_weights = {
                        '2330.TW': pd.Series(0.25, index=window_returns.index),  # TSMC
                        '2317.TW': pd.Series(0.15, index=window_returns.index),  # Foxconn
                        '2454.TW': pd.Series(0.10, index=window_returns.index),  # MediaTek
                        '2412.TW': pd.Series(0.08, index=window_returns.index),  # Chunghwa Telecom
                        '6505.TW': pd.Series(0.05, index=window_returns.index),  # Formosa Petrochemical
                    }
                    
                    # Run attribution for this window
                    attribution = attributor.attribute_performance(
                        portfolio_returns=window_returns,
                        portfolio_weights=portfolio_weights,
                        benchmark_weights=benchmark_weights,
                        start_date=window_returns.index[0].date(),
                        end_date=window_returns.index[-1].date()
                    )
                    
                    attribution_results.append(attribution)
                    
                except Exception as e:
                    logger.warning(f"Attribution failed for window {window_id}: {e}")
                    continue
        
        logger.info(f"Attribution analysis completed for {len(attribution_results)} windows")
        return attribution_results
    
    def run_risk_analysis(
        self,
        returns_by_window: Dict[str, pd.Series]
    ):
        """Run comprehensive risk analysis."""
        logger.info("Running risk analysis")
        
        # Create risk calculator
        risk_calculator = create_taiwan_risk_calculator(
            var_method="cornish_fisher",
            confidence_levels=[0.95, 0.99],
            enable_hypothesis_tests=True
        )
        
        # Combine all returns for overall risk analysis
        all_returns = []
        for window_returns in returns_by_window.values():
            all_returns.extend(window_returns.tolist())
        
        if all_returns:
            overall_returns = pd.Series(all_returns)
            
            # Calculate risk metrics
            risk_metrics = risk_calculator.calculate_risk_metrics(overall_returns)
            
            logger.info("Risk analysis completed")
            return risk_metrics
        else:
            logger.warning("No returns available for risk analysis")
            return None
    
    def generate_validation_report(
        self,
        validation_result,
        returns_by_window: Dict[str, pd.Series],
        temporal_store,
        pit_engine,
        strategy_name: str = "Taiwan Market Demo Strategy"
    ):
        """Generate comprehensive validation report."""
        logger.info("Generating validation report")
        
        report_files = generate_taiwan_validation_report(
            validation_result=validation_result,
            returns_by_window=returns_by_window,
            temporal_store=temporal_store,
            pit_engine=pit_engine,
            report_title="Taiwan Market Walk-Forward Validation Report",
            strategy_name=strategy_name,
            output_formats=[ReportFormat.HTML, ReportFormat.JSON, ReportFormat.MARKDOWN],
            target_sharpe_ratio=2.0,
            target_information_ratio=0.8,
            max_drawdown_threshold=0.15,
            include_charts=True
        )
        
        logger.info(f"Validation report generated: {list(report_files.values())}")
        return report_files
    
    def run_complete_example(self):
        """Run the complete validation example."""
        logger.info("=== Starting Taiwan Market Walk-Forward Validation Example ===")
        
        try:
            # 1. Setup data infrastructure
            temporal_store, pit_engine = self.create_mock_data_infrastructure()
            
            # 2. Define validation parameters
            symbols = ['2330.TW', '2317.TW', '2454.TW', '2412.TW', '6505.TW']  # Top Taiwan stocks
            start_date = date(2020, 1, 1)
            end_date = date(2023, 12, 31)
            
            # 3. Run walk-forward validation
            validation_result = self.run_walk_forward_validation(
                temporal_store, pit_engine, symbols, start_date, end_date
            )
            
            # 4. Generate sample returns
            returns_by_window = self.generate_window_returns(validation_result)
            
            # 5. Run performance analysis
            performance_analysis = self.run_performance_analysis(
                validation_result, returns_by_window, temporal_store, pit_engine
            )
            
            # 6. Run attribution analysis
            attribution_results = self.run_attribution_analysis(
                returns_by_window, temporal_store, pit_engine
            )
            
            # 7. Run risk analysis
            risk_metrics = self.run_risk_analysis(returns_by_window)
            
            # 8. Generate comprehensive report
            report_files = self.generate_validation_report(
                validation_result, returns_by_window, temporal_store, pit_engine
            )
            
            # 9. Print summary
            self.print_summary(validation_result, performance_analysis, risk_metrics, report_files)
            
            logger.info("=== Taiwan Market Walk-Forward Validation Example Completed ===")
            
        except Exception as e:
            logger.error(f"Example failed: {e}")
            raise
    
    def print_summary(self, validation_result, performance_analysis, risk_metrics, report_files):
        """Print a summary of the validation results."""
        
        print("\n" + "="*60)
        print("TAIWAN MARKET WALK-FORWARD VALIDATION SUMMARY")
        print("="*60)
        
        # Validation summary
        print(f"\n📊 Validation Overview:")
        print(f"   Total Windows: {validation_result.total_windows}")
        print(f"   Successful Windows: {validation_result.successful_windows}")
        print(f"   Success Rate: {validation_result.success_rate():.1%}")
        print(f"   Runtime: {validation_result.total_runtime_seconds:.1f} seconds")
        
        # Performance summary
        if performance_analysis and 'overall_metrics' in performance_analysis:
            metrics = performance_analysis['overall_metrics']
            print(f"\n📈 Performance Metrics:")
            print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Information Ratio: {metrics.get('information_ratio', 0):.2f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        # Risk summary
        if risk_metrics:
            print(f"\n⚡ Risk Metrics:")
            print(f"   Volatility: {risk_metrics.total_volatility:.2%}")
            print(f"   VaR 95%: {risk_metrics.var_95:.2%}")
            print(f"   Sortino Ratio: {risk_metrics.sortino_ratio:.2f}")
            print(f"   Calmar Ratio: {risk_metrics.calmar_ratio:.2f}")
        
        # Report files
        print(f"\n📄 Generated Reports:")
        for format_name, file_path in report_files.items():
            print(f"   {format_name.upper()}: {file_path}")
        
        print("\n" + "="*60)
        print("Example completed successfully!")
        print("="*60 + "\n")


def main():
    """Main function to run the Taiwan validation example."""
    
    # Create and run example
    example = TaiwanValidationExample()
    example.run_complete_example()


if __name__ == "__main__":
    main()