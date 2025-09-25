#!/usr/bin/env python3
"""
ML4T-Alpha Integration Demonstration.

This script demonstrates the complete ML4T-Alpha integration functionality,
including data interface, format conversion, streaming, and optimization.

Usage:
    python demo_ml4t_alpha_integration.py [--symbols SYMBOL1,SYMBOL2] [--days N]
"""

import asyncio
import argparse
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import time
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ML4T integration components
try:
    from src.integration.ml4t_data_interface import create_ml4t_interface, ML4TDataConfig
    from src.integration.format_converters import (
        FinLabToML4TConverter, ConversionConfig, DataFormat,
        OpenFEDataAdapter, BacktestDataFormatter
    )
    from src.integration.streaming_engine import (
        create_streaming_engine, StreamingMode, StreamingMessage
    )
    from src.integration.backtest_optimizer import (
        create_backtest_optimizer, OptimizationLevel
    )
    from src.data.ingestion.finlab_auth import AuthConfig
    from src.data.ingestion.finlab_connector import FinLabConfig
    from src.data.core.temporal import DataType
except ImportError as e:
    logger.error(f"Failed to import ML4T integration components: {e}")
    logger.info("Please ensure the ML4T integration modules are properly installed")
    exit(1)


class ML4TIntegrationDemo:
    """Demonstration of ML4T-Alpha integration capabilities."""

    def __init__(self, symbols: List[str], lookback_days: int = 30):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.end_date = date.today()
        self.start_date = self.end_date - timedelta(days=lookback_days)

        # Initialize components
        self.ml4t_interface = None
        self.optimizer = None
        self.streaming_engine = None

        # Results storage
        self.results = {
            'data_loading': {},
            'format_conversion': {},
            'streaming': {},
            'optimization': {},
            'export': {},
            'performance': {}
        }

        logger.info(f"Initialized ML4T integration demo for {len(symbols)} symbols")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")

    def setup_interface(self) -> None:
        """Set up ML4T data interface."""
        logger.info("Setting up ML4T data interface...")

        try:
            # Create interface with environment-based authentication
            self.ml4t_interface = create_ml4t_interface(
                enable_pit=True,
                enable_cache=True,
                max_workers=4
            )

            logger.info("ML4T data interface created successfully")

        except Exception as e:
            logger.error(f"Failed to create ML4T interface: {e}")
            # Create mock interface for demo purposes
            self.ml4t_interface = self._create_mock_interface()
            logger.info("Created mock interface for demonstration")

    def _create_mock_interface(self):
        """Create mock interface for demonstration when real connection fails."""
        from unittest.mock import Mock

        mock_interface = Mock()
        mock_interface._connected = False

        # Mock connection methods
        mock_interface.connect = Mock()
        mock_interface.disconnect = Mock()

        # Mock data methods
        def mock_get_symbols():
            return ["2330.TW", "2454.TW", "2317.TW", "2382.TW", "2412.TW"]

        def mock_get_price_data(symbols, start_date, end_date, fields=None):
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            # Filter to business days only (simplified)
            dates = dates[dates.weekday < 5]

            data = {}
            for field in (fields or ['open', 'high', 'low', 'close', 'volume', 'adj_close']):
                for symbol in symbols:
                    if field == 'volume':
                        values = np.random.randint(100000, 10000000, len(dates))
                    else:
                        # Generate realistic price data
                        base_price = 100 + hash(symbol) % 200  # Symbol-specific base price
                        returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
                        prices = base_price * np.exp(np.cumsum(returns))

                        if field in ['high']:
                            values = prices * (1 + np.random.uniform(0, 0.03, len(dates)))
                        elif field in ['low']:
                            values = prices * (1 - np.random.uniform(0, 0.03, len(dates)))
                        elif field in ['open']:
                            values = prices * (1 + np.random.normal(0, 0.01, len(dates)))
                        elif field in ['adj_close']:
                            values = prices * (1 - np.random.uniform(0, 0.001, len(dates)))  # Slight adjustment
                        else:  # close
                            values = prices

                    data[(field, symbol)] = values

            df = pd.DataFrame(data, index=dates)
            df.columns = pd.MultiIndex.from_tuples(df.columns)
            return df

        def mock_get_openfe_compatible_data(symbols, start_date, end_date, include_fundamentals=True):
            price_data = mock_get_price_data(symbols, start_date, end_date)
            result = {'price_data': price_data}

            if include_fundamentals:
                # Mock fundamental data
                fund_dates = pd.date_range(start=start_date, end=end_date, freq='Q')
                fund_data = {}
                for field in ['revenue', 'net_income', 'total_assets', 'eps']:
                    for symbol in symbols:
                        values = np.random.uniform(1e6, 1e9, len(fund_dates))
                        fund_data[(field, symbol)] = values

                fund_df = pd.DataFrame(fund_data, index=fund_dates)
                fund_df.columns = pd.MultiIndex.from_tuples(fund_df.columns)
                result['fundamental_data'] = fund_df

            return result

        def mock_export_backtest_data(symbols, start_date, end_date, output_path=None, format='hdf5'):
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"demo_backtest_data_{timestamp}.{format}")

            # Create dummy file for demo
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("Demo export file")

            return output_path

        def mock_validate_backtest_readiness(symbols, start_date, end_date):
            return {
                'ready': True,
                'issues': [],
                'warnings': [],
                'data_quality_score': 92.5,
                'symbol_coverage': {symbol: 0.95 for symbol in symbols},
                'recommendations': ['Data quality is good for backtesting']
            }

        def mock_get_performance_stats():
            return {
                'query_count': 10,
                'total_query_time': 2.5,
                'avg_query_time': 0.25,
                'cache_hits': 7,
                'cache_hit_rate': 0.7,
                'connected': True
            }

        # Assign mock methods
        mock_interface.get_symbols = mock_get_symbols
        mock_interface.get_price_data = mock_get_price_data
        mock_interface.get_openfe_compatible_data = mock_get_openfe_compatible_data
        mock_interface.export_backtest_data = mock_export_backtest_data
        mock_interface.validate_backtest_readiness = mock_validate_backtest_readiness
        mock_interface.get_performance_stats = mock_get_performance_stats

        return mock_interface

    def demo_data_loading(self) -> None:
        """Demonstrate data loading capabilities."""
        logger.info("=" * 60)
        logger.info("DEMO: Data Loading")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Connect to data source
            self.ml4t_interface.connect()

            # Get available symbols
            available_symbols = self.ml4t_interface.get_symbols()
            logger.info(f"Available symbols: {len(available_symbols)} (showing first 10: {available_symbols[:10]})")

            # Filter to requested symbols
            valid_symbols = [s for s in self.symbols if s in available_symbols]
            if not valid_symbols:
                valid_symbols = available_symbols[:len(self.symbols)]
                logger.warning(f"Using available symbols instead: {valid_symbols}")

            # Load price data
            logger.info(f"Loading price data for {len(valid_symbols)} symbols...")
            price_data = self.ml4t_interface.get_price_data(
                valid_symbols, self.start_date, self.end_date
            )

            # Store results
            self.results['data_loading'] = {
                'symbols_count': len(valid_symbols),
                'date_range_days': (self.end_date - self.start_date).days,
                'data_shape': price_data.shape,
                'data_columns': list(price_data.columns) if hasattr(price_data.columns, '__iter__') else 'MultiIndex',
                'missing_values': price_data.isnull().sum().sum() if hasattr(price_data, 'isnull') else 0,
                'load_time_seconds': time.time() - start_time
            }

            logger.info(f"✓ Loaded data shape: {price_data.shape}")
            logger.info(f"✓ Loading time: {self.results['data_loading']['load_time_seconds']:.2f} seconds")
            logger.info(f"✓ Missing values: {self.results['data_loading']['missing_values']}")

            # Preview data
            logger.info("\nData preview:")
            if hasattr(price_data, 'head'):
                print(price_data.head())
            else:
                logger.info("Price data loaded successfully")

        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            self.results['data_loading']['error'] = str(e)

    def demo_format_conversion(self) -> None:
        """Demonstrate format conversion capabilities."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO: Format Conversion")
        logger.info("=" * 60)

        try:
            # Get sample data
            valid_symbols = self.symbols[:2]  # Use subset for conversion demo
            price_data = self.ml4t_interface.get_price_data(
                valid_symbols, self.start_date, self.end_date
            )

            # Create conversion configuration
            conversion_config = ConversionConfig(
                target_format=DataFormat.ML4T_ALPHA,
                forward_fill_missing=True,
                handle_corporate_actions=True
            )

            # Test FinLab to ML4T conversion
            logger.info("Converting to ML4T-Alpha format...")
            converter = FinLabToML4TConverter(conversion_config)

            # Create sample temporal values for conversion
            from src.data.core.temporal import TemporalValue
            temporal_values = []

            if hasattr(price_data, 'index') and hasattr(price_data, 'columns'):
                for date_idx in price_data.index[:10]:  # Limit for demo
                    for symbol in valid_symbols:
                        try:
                            if isinstance(price_data.columns, pd.MultiIndex):
                                close_value = price_data.loc[date_idx, ('close', symbol)]
                            else:
                                close_value = price_data.loc[date_idx, f'close_{symbol}']

                            if pd.notna(close_value):
                                tv = TemporalValue(
                                    value=float(close_value),
                                    as_of_date=date_idx.date() if hasattr(date_idx, 'date') else date_idx,
                                    value_date=date_idx.date() if hasattr(date_idx, 'date') else date_idx,
                                    data_type=DataType.PRICE,
                                    symbol=symbol,
                                    metadata={"field": "close", "source": "demo"}
                                )
                                temporal_values.append(tv)
                        except Exception:
                            continue

            if temporal_values:
                ml4t_data = converter.convert_price_data(temporal_values, valid_symbols)

                logger.info(f"✓ Converted {len(temporal_values)} temporal values")
                logger.info(f"✓ Output shape: {ml4t_data.shape if hasattr(ml4t_data, 'shape') else 'N/A'}")

                # Test OpenFE adapter
                logger.info("Preparing data for openFE...")
                openfe_config = ConversionConfig(target_format=DataFormat.OPENFE)
                openfe_adapter = OpenFEDataAdapter(openfe_config)

                openfe_data = self.ml4t_interface.get_openfe_compatible_data(
                    valid_symbols, self.start_date, self.end_date
                )

                if 'price_data' in openfe_data:
                    features = openfe_adapter.prepare_feature_matrix(openfe_data['price_data'])
                    logger.info(f"✓ OpenFE feature matrix shape: {features.shape if hasattr(features, 'shape') else 'N/A'}")

                # Test Backtrader format
                logger.info("Converting to Backtrader format...")
                bt_formatter = BacktestDataFormatter(conversion_config)

                if hasattr(price_data, 'shape') and len(valid_symbols) > 0:
                    bt_data = bt_formatter.format_for_backtrader(price_data, valid_symbols[0])
                    logger.info(f"✓ Backtrader format columns: {list(bt_data.columns) if hasattr(bt_data, 'columns') else 'N/A'}")

                # Store results
                self.results['format_conversion'] = {
                    'temporal_values_converted': len(temporal_values),
                    'ml4t_format_shape': str(ml4t_data.shape) if hasattr(ml4t_data, 'shape') else 'N/A',
                    'openfe_features_count': features.shape[1] if hasattr(features, 'shape') else 0,
                    'conversion_stats': converter.get_conversion_stats()
                }

            else:
                logger.warning("No temporal values created for conversion demo")
                self.results['format_conversion'] = {'error': 'No temporal values for conversion'}

        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            self.results['format_conversion']['error'] = str(e)

    async def demo_streaming(self) -> None:
        """Demonstrate streaming capabilities."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO: Real-time Streaming")
        logger.info("=" * 60)

        try:
            # Create streaming engine
            self.streaming_engine = create_streaming_engine(
                self.ml4t_interface,
                symbols=self.symbols[:2],  # Limit for demo
                mode=StreamingMode.SIMULATION,
                update_interval_ms=500,  # 0.5 second intervals
            )

            # Set up message handler
            messages_received = []

            def message_handler(message: StreamingMessage):
                messages_received.append(message)
                logger.info(f"📊 Stream: {message.symbol} @ {message.timestamp}: {message.data}")

            self.streaming_engine.add_data_callback(message_handler)

            # Start streaming
            logger.info("Starting streaming engine...")
            await self.streaming_engine.start()

            # Let it run for a few seconds
            logger.info("Streaming live data (5 seconds)...")
            await asyncio.sleep(5)

            # Stop streaming
            logger.info("Stopping streaming engine...")
            await self.streaming_engine.stop()

            # Get streaming DataFrame
            if messages_received:
                streaming_df = self.streaming_engine.get_streaming_dataframe(
                    self.symbols[0], lookback_minutes=10
                )

                logger.info(f"✓ Received {len(messages_received)} streaming messages")
                logger.info(f"✓ Streaming DataFrame shape: {streaming_df.shape if hasattr(streaming_df, 'shape') else 'N/A'}")

                # Get performance stats
                stats = self.streaming_engine.get_performance_stats()
                logger.info(f"✓ Messages per second: {stats.get('messages_per_second', 0):.2f}")

                self.results['streaming'] = {
                    'messages_received': len(messages_received),
                    'streaming_duration_seconds': 5,
                    'performance_stats': stats
                }
            else:
                logger.warning("No streaming messages received")
                self.results['streaming'] = {'error': 'No streaming messages'}

        except Exception as e:
            logger.error(f"Streaming demo failed: {e}")
            self.results['streaming']['error'] = str(e)

    def demo_optimization(self) -> None:
        """Demonstrate optimization capabilities."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO: Performance Optimization")
        logger.info("=" * 60)

        try:
            # Create optimizer
            self.optimizer = create_backtest_optimizer(
                self.ml4t_interface,
                optimization_level=OptimizationLevel.BALANCED,
                cache_size_mb=512,
                enable_parallel=True
            )

            # Test data loading optimization
            logger.info("Testing optimized data loading...")
            start_time = time.time()

            optimized_data = self.optimizer.optimize_data_loading(
                self.symbols, self.start_date, self.end_date
            )

            first_load_time = time.time() - start_time
            logger.info(f"✓ First load time: {first_load_time:.2f} seconds")

            # Test cached loading
            start_time = time.time()
            cached_data = self.optimizer.optimize_data_loading(
                self.symbols, self.start_date, self.end_date
            )
            cached_load_time = time.time() - start_time
            logger.info(f"✓ Cached load time: {cached_load_time:.2f} seconds")
            logger.info(f"✓ Speed improvement: {first_load_time / max(cached_load_time, 0.001):.1f}x")

            # Test backtest optimization
            logger.info("Testing backtest execution optimization...")

            def sample_backtest(data, lookback_period=20):
                """Sample backtest strategy."""
                if hasattr(data, 'columns') and hasattr(data, 'index'):
                    # Simple moving average strategy
                    if isinstance(data.columns, pd.MultiIndex):
                        # Find a close price column
                        close_cols = [col for col in data.columns if col[0] == 'close']
                        if close_cols:
                            prices = data[close_cols[0]]
                            sma = prices.rolling(lookback_period).mean()
                            signals = (prices > sma).astype(int)
                            returns = prices.pct_change() * signals.shift(1)

                            return {
                                'total_return': (1 + returns.fillna(0)).prod() - 1,
                                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                                'win_rate': (returns > 0).mean()
                            }

                # Fallback for demo
                return {
                    'total_return': np.random.uniform(-0.1, 0.3),
                    'sharpe_ratio': np.random.uniform(0, 2),
                    'win_rate': np.random.uniform(0.4, 0.6)
                }

            backtest_result = self.optimizer.optimize_backtest_execution(
                sample_backtest, optimized_data
            )

            logger.info(f"✓ Backtest results: {backtest_result}")

            # Get optimization stats
            opt_stats = self.optimizer.get_performance_stats()
            logger.info(f"✓ Cache hit rate: {opt_stats.get('cache_hit_rate', 0):.1%}")
            logger.info(f"✓ Memory optimizations: {opt_stats.get('memory_optimizations', 0)}")

            self.results['optimization'] = {
                'first_load_time': first_load_time,
                'cached_load_time': cached_load_time,
                'speed_improvement': first_load_time / max(cached_load_time, 0.001),
                'backtest_result': backtest_result,
                'optimization_stats': opt_stats
            }

        except Exception as e:
            logger.error(f"Optimization demo failed: {e}")
            self.results['optimization']['error'] = str(e)

    def demo_export_and_validation(self) -> None:
        """Demonstrate export and validation capabilities."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO: Export and Validation")
        logger.info("=" * 60)

        try:
            # Export data in multiple formats
            export_dir = Path("demo_exports")
            export_dir.mkdir(exist_ok=True)

            formats = ['hdf5', 'parquet', 'csv']
            export_paths = {}

            for fmt in formats:
                logger.info(f"Exporting data to {fmt.upper()} format...")
                export_path = self.ml4t_interface.export_backtest_data(
                    self.symbols[:2],  # Limit for demo
                    self.start_date,
                    self.end_date,
                    output_path=export_dir / f"demo_data.{fmt}",
                    format=fmt
                )
                export_paths[fmt] = export_path
                logger.info(f"✓ Exported to: {export_path}")

            # Validate backtest readiness
            logger.info("Validating backtest readiness...")
            validation_report = self.ml4t_interface.validate_backtest_readiness(
                self.symbols, self.start_date, self.end_date
            )

            logger.info(f"✓ Ready for backtesting: {validation_report.get('ready', False)}")
            logger.info(f"✓ Data quality score: {validation_report.get('data_quality_score', 0):.1f}")

            if validation_report.get('issues'):
                logger.warning(f"Issues found: {validation_report['issues']}")

            if validation_report.get('recommendations'):
                logger.info(f"Recommendations: {validation_report['recommendations']}")

            self.results['export'] = {
                'formats_exported': list(export_paths.keys()),
                'export_paths': {k: str(v) for k, v in export_paths.items()},
                'validation_report': validation_report
            }

        except Exception as e:
            logger.error(f"Export and validation failed: {e}")
            self.results['export']['error'] = str(e)

    def demo_performance_summary(self) -> None:
        """Generate performance summary."""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO: Performance Summary")
        logger.info("=" * 60)

        try:
            # Collect performance stats from all components
            interface_stats = self.ml4t_interface.get_performance_stats()

            if self.optimizer:
                optimizer_stats = self.optimizer.get_performance_stats()
            else:
                optimizer_stats = {}

            if self.streaming_engine:
                streaming_stats = self.streaming_engine.get_performance_stats()
            else:
                streaming_stats = {}

            # Calculate overall metrics
            total_operations = sum([
                self.results.get('data_loading', {}).get('symbols_count', 0),
                len(self.results.get('format_conversion', {}).get('conversion_stats', {})),
                self.results.get('streaming', {}).get('messages_received', 0),
                optimizer_stats.get('optimization_count', 0)
            ])

            performance_summary = {
                'total_operations': total_operations,
                'interface_performance': interface_stats,
                'optimizer_performance': optimizer_stats,
                'streaming_performance': streaming_stats,
                'demo_results': self.results
            }

            self.results['performance'] = performance_summary

            # Display summary
            logger.info(f"📊 Total operations performed: {total_operations}")
            logger.info(f"📊 Interface queries: {interface_stats.get('query_count', 0)}")
            logger.info(f"📊 Cache hit rate: {interface_stats.get('cache_hit_rate', 0):.1%}")
            logger.info(f"📊 Average query time: {interface_stats.get('avg_query_time', 0):.3f}s")

            if optimizer_stats:
                logger.info(f"📊 Optimizations performed: {optimizer_stats.get('optimization_count', 0)}")
                logger.info(f"📊 Memory optimizations: {optimizer_stats.get('memory_optimizations', 0)}")

            if streaming_stats:
                logger.info(f"📊 Streaming messages: {streaming_stats.get('messages_processed', 0)}")
                logger.info(f"📊 Messages per second: {streaming_stats.get('messages_per_second', 0):.2f}")

        except Exception as e:
            logger.error(f"Performance summary failed: {e}")
            self.results['performance']['error'] = str(e)

    async def run_complete_demo(self) -> dict:
        """Run complete demonstration of all features."""
        logger.info("🚀 Starting ML4T-Alpha Integration Demonstration")
        logger.info("=" * 80)

        demo_start_time = time.time()

        try:
            # Setup
            self.setup_interface()

            # Run demonstrations
            self.demo_data_loading()
            self.demo_format_conversion()
            await self.demo_streaming()
            self.demo_optimization()
            self.demo_export_and_validation()
            self.demo_performance_summary()

            # Calculate total demo time
            total_demo_time = time.time() - demo_start_time

            logger.info("\n" + "=" * 80)
            logger.info("🎉 ML4T-Alpha Integration Demonstration Complete!")
            logger.info(f"⏱️  Total demonstration time: {total_demo_time:.2f} seconds")
            logger.info("=" * 80)

            # Final results
            self.results['demo_metadata'] = {
                'symbols_tested': self.symbols,
                'lookback_days': self.lookback_days,
                'total_time_seconds': total_demo_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            self.results['demo_metadata'] = {
                'symbols_tested': self.symbols,
                'error': str(e),
                'success': False
            }

        finally:
            # Cleanup
            try:
                if self.streaming_engine and hasattr(self.streaming_engine, 'stop'):
                    await self.streaming_engine.stop()
                if self.ml4t_interface and hasattr(self.ml4t_interface, 'disconnect'):
                    self.ml4t_interface.disconnect()
            except:
                pass

        return self.results


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description='ML4T-Alpha Integration Demonstration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--symbols',
        type=str,
        default='2330.TW,2454.TW,2317.TW',
        help='Comma-separated list of symbols to test'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days of historical data to use'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]

    # Create and run demo
    demo = ML4TIntegrationDemo(symbols=symbols, lookback_days=args.days)

    # Run async demo
    try:
        results = asyncio.run(demo.run_complete_demo())

        # Save results
        import json
        results_file = f"ml4t_integration_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert any non-serializable objects
        def serialize_results(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return obj

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=serialize_results)

        logger.info(f"\n📄 Demo results saved to: {results_file}")

        # Print success summary
        success = results.get('demo_metadata', {}).get('success', False)
        if success:
            logger.info("✅ Demonstration completed successfully!")
        else:
            logger.error("❌ Demonstration encountered errors.")

    except KeyboardInterrupt:
        logger.info("\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()