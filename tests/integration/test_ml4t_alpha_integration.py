"""
Integration Tests for ML4T-Alpha Integration.

This module provides comprehensive integration tests for the ML4T-Alpha
data interface, format converters, streaming engine, and backtest optimizer.
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock
import time

from src.integration.ml4t_data_interface import (
    ML4TDataInterface, ML4TDataConfig, create_ml4t_interface
)
from src.integration.format_converters import (
    FinLabToML4TConverter, ConversionConfig, DataFormat,
    OpenFEDataAdapter, BacktestDataFormatter
)
from src.integration.streaming_engine import (
    ML4TStreamingEngine, StreamingConfig, StreamingMode,
    create_streaming_engine, StreamingMessage
)
from src.integration.backtest_optimizer import (
    BacktestOptimizer, OptimizationConfig, OptimizationLevel,
    create_backtest_optimizer
)
from src.data.ingestion.finlab_auth import AuthConfig
from src.data.ingestion.finlab_connector import FinLabConfig
from src.data.core.temporal import TemporalValue, DataType, InMemoryTemporalStore
from src.data.models.taiwan_market import TaiwanMarketData


class TestML4TDataInterface:
    """Test cases for ML4T data interface."""

    @pytest.fixture
    def mock_auth_config(self):
        """Create mock auth configuration."""
        return AuthConfig(
            finlab_token="test_token",
            api_key="test_api_key"
        )

    @pytest.fixture
    def mock_finlab_config(self, mock_auth_config):
        """Create mock FinLab configuration."""
        return FinLabConfig(auth_config=mock_auth_config)

    @pytest.fixture
    def test_config(self, mock_finlab_config):
        """Create test ML4T data configuration."""
        return ML4TDataConfig(
            finlab_config=mock_finlab_config,
            enable_cache=True,
            enable_pit=True,
            max_workers=2
        )

    @pytest.fixture
    def mock_temporal_store(self):
        """Create mock temporal store with sample data."""
        store = InMemoryTemporalStore()

        # Add sample price data
        symbols = ["2330.TW", "2454.TW", "2317.TW"]
        start_date = date(2024, 1, 1)

        for i in range(30):  # 30 days of data
            current_date = start_date + timedelta(days=i)

            for symbol in symbols:
                # Create sample price data
                base_price = 100 + np.random.randn() * 10
                tv = TemporalValue(
                    value=base_price,
                    as_of_date=current_date,
                    value_date=current_date,
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    metadata={"field": "close", "source": "test"}
                )
                store.store(tv)

        return store

    @pytest.fixture
    def ml4t_interface(self, test_config, mock_temporal_store):
        """Create ML4T interface with mocked dependencies."""
        with patch('src.integration.ml4t_data_interface.FinLabConnector') as mock_connector:
            # Configure mock connector
            mock_connector_instance = Mock()
            mock_connector_instance._connected = False
            mock_connector_instance.connect = Mock()
            mock_connector_instance.disconnect = Mock()
            mock_connector_instance.get_available_symbols = Mock(return_value=["2330.TW", "2454.TW"])

            # Mock price data method
            def mock_get_price_data(symbols, start_date, end_date, fields=None):
                # Return sample temporal values
                values = []
                for symbol in symbols:
                    for day in range((end_date - start_date).days + 1):
                        current_date = start_date + timedelta(days=day)
                        base_price = 100 + np.random.randn() * 10

                        for field in (fields or ['close', 'volume']):
                            value = base_price if field == 'close' else 1000000
                            tv = TemporalValue(
                                value=value,
                                as_of_date=current_date,
                                value_date=current_date,
                                data_type=DataType.PRICE,
                                symbol=symbol,
                                metadata={"field": field, "source": "test"}
                            )
                            values.append(tv)
                return values

            mock_connector_instance.get_price_data = Mock(side_effect=mock_get_price_data)
            mock_connector_instance.get_fundamental_data = Mock(return_value=[])
            mock_connector_instance.get_data_quality_metrics = Mock(return_value={"overall_quality_score": 95.0})
            mock_connector.return_value = mock_connector_instance

            interface = ML4TDataInterface(
                test_config,
                finlab_connector=mock_connector_instance,
                temporal_store=mock_temporal_store
            )
            return interface

    def test_interface_initialization(self, ml4t_interface):
        """Test ML4T interface initialization."""
        assert ml4t_interface is not None
        assert not ml4t_interface._connected
        assert ml4t_interface.config.enable_cache
        assert ml4t_interface.config.enable_pit

    def test_connection_management(self, ml4t_interface):
        """Test connection management."""
        # Test connection
        ml4t_interface.connect()
        assert ml4t_interface._connected

        # Test disconnection
        ml4t_interface.disconnect()
        assert not ml4t_interface._connected

    def test_get_symbols(self, ml4t_interface):
        """Test symbol retrieval."""
        symbols = ml4t_interface.get_symbols()
        assert isinstance(symbols, list)
        assert "2330.TW" in symbols
        assert "2454.TW" in symbols

    def test_get_price_data(self, ml4t_interface):
        """Test price data retrieval."""
        symbols = ["2330.TW", "2454.TW"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)

        price_data = ml4t_interface.get_price_data(
            symbols, start_date, end_date
        )

        assert isinstance(price_data, pd.DataFrame)
        assert not price_data.empty

        # Check data structure
        if isinstance(price_data.columns, pd.MultiIndex):
            # Multi-level columns format
            assert len(price_data.columns.get_level_values(1).unique()) >= 2  # Multiple symbols
        else:
            # Simple columns format
            assert len(price_data.columns) > 0

    def test_get_openfe_compatible_data(self, ml4t_interface):
        """Test openFE compatible data format."""
        symbols = ["2330.TW"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)

        data = ml4t_interface.get_openfe_compatible_data(
            symbols, start_date, end_date
        )

        assert isinstance(data, dict)
        assert "price_data" in data
        assert isinstance(data["price_data"], pd.DataFrame)

    def test_export_backtest_data(self, ml4t_interface):
        """Test backtest data export."""
        symbols = ["2330.TW"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.h5"

            exported_path = ml4t_interface.export_backtest_data(
                symbols, start_date, end_date, output_path, format="hdf5"
            )

            assert exported_path.exists()
            assert exported_path.suffix == ".h5"

    def test_validate_backtest_readiness(self, ml4t_interface):
        """Test backtest readiness validation."""
        symbols = ["2330.TW"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        report = ml4t_interface.validate_backtest_readiness(
            symbols, start_date, end_date
        )

        assert isinstance(report, dict)
        assert "ready" in report
        assert "data_quality_score" in report
        assert "symbol_coverage" in report

    def test_performance_stats(self, ml4t_interface):
        """Test performance statistics retrieval."""
        # Perform some operations to generate stats
        ml4t_interface.connect()
        symbols = ml4t_interface.get_symbols()

        stats = ml4t_interface.get_performance_stats()

        assert isinstance(stats, dict)
        assert "query_count" in stats
        assert "connected" in stats
        assert stats["connected"] == ml4t_interface._connected


class TestFormatConverters:
    """Test cases for format converters."""

    @pytest.fixture
    def conversion_config(self):
        """Create test conversion configuration."""
        return ConversionConfig(
            target_format=DataFormat.ML4T_ALPHA,
            forward_fill_missing=True,
            min_price_threshold=0.01
        )

    @pytest.fixture
    def sample_temporal_values(self):
        """Create sample temporal values for testing."""
        values = []
        symbols = ["2330.TW", "2454.TW"]
        start_date = date(2024, 1, 1)

        for i in range(5):
            current_date = start_date + timedelta(days=i)
            for symbol in symbols:
                for field in ['open', 'high', 'low', 'close', 'volume']:
                    base_value = 100 + np.random.randn() * 10 if field != 'volume' else 1000000
                    tv = TemporalValue(
                        value=base_value,
                        as_of_date=current_date,
                        value_date=current_date,
                        data_type=DataType.PRICE if field != 'volume' else DataType.VOLUME,
                        symbol=symbol,
                        metadata={"field": field, "source": "test"}
                    )
                    values.append(tv)

        return values

    def test_finlab_to_ml4t_converter(self, conversion_config, sample_temporal_values):
        """Test FinLab to ML4T format conversion."""
        converter = FinLabToML4TConverter(conversion_config)

        result = converter.convert_price_data(sample_temporal_values)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # Check conversion stats
        stats = converter.get_conversion_stats()
        assert stats["conversion_count"] > 0

    def test_openfe_data_adapter(self, conversion_config):
        """Test openFE data adapter."""
        adapter = OpenFEDataAdapter(conversion_config)

        # Create sample price data
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        price_data = pd.DataFrame({
            ('close', '2330.TW'): np.random.randn(len(dates)) * 10 + 100,
            ('close', '2454.TW'): np.random.randn(len(dates)) * 10 + 100,
        }, index=dates)
        price_data.columns = pd.MultiIndex.from_tuples(price_data.columns)

        features = adapter.prepare_feature_matrix(price_data)

        assert isinstance(features, pd.DataFrame)
        assert not features.empty

    def test_backtest_data_formatter(self, conversion_config):
        """Test backtest data formatter."""
        formatter = BacktestDataFormatter(conversion_config)

        # Create sample data
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        price_data = pd.DataFrame({
            ('open', '2330.TW'): np.random.randn(len(dates)) * 10 + 100,
            ('high', '2330.TW'): np.random.randn(len(dates)) * 10 + 105,
            ('low', '2330.TW'): np.random.randn(len(dates)) * 10 + 95,
            ('close', '2330.TW'): np.random.randn(len(dates)) * 10 + 100,
            ('volume', '2330.TW'): np.random.randint(1000, 10000, len(dates)),
        }, index=dates)
        price_data.columns = pd.MultiIndex.from_tuples(price_data.columns)

        # Test Backtrader format
        backtrader_data = formatter.format_for_backtrader(price_data, "2330.TW")

        assert isinstance(backtrader_data, pd.DataFrame)
        assert all(col in backtrader_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

        # Test QuantLib format
        quantlib_data = formatter.format_for_quantlib(price_data, "2330.TW")

        assert isinstance(quantlib_data, dict)
        assert "dates" in quantlib_data
        assert "prices" in quantlib_data
        assert "symbol" in quantlib_data

    def test_zipline_bundle_creation(self, conversion_config, sample_temporal_values):
        """Test Zipline bundle creation."""
        converter = FinLabToML4TConverter(conversion_config)
        price_data = converter.convert_price_data(sample_temporal_values)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_files = converter.convert_to_zipline_bundle(
                price_data, output_dir=Path(tmpdir)
            )

            assert isinstance(bundle_files, dict)
            if "prices" in bundle_files:
                assert bundle_files["prices"].exists()
            if "metadata" in bundle_files:
                assert bundle_files["metadata"].exists()


class TestStreamingEngine:
    """Test cases for streaming engine."""

    @pytest.fixture
    def streaming_config(self):
        """Create test streaming configuration."""
        return StreamingConfig(
            mode=StreamingMode.SIMULATION,
            symbols=["2330.TW", "2454.TW"],
            update_interval_ms=100,  # Fast for testing
            buffer_size=100,
            enable_data_validation=True
        )

    @pytest.fixture
    def mock_ml4t_interface(self):
        """Create mock ML4T interface for streaming tests."""
        mock_interface = Mock()
        mock_interface._connected = False
        mock_interface.connect = Mock()
        mock_interface.disconnect = Mock()

        # Mock price data for streaming simulation
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        price_data = pd.DataFrame({
            ('close', '2330.TW'): np.random.randn(len(dates)) * 10 + 100,
            ('close', '2454.TW'): np.random.randn(len(dates)) * 10 + 100,
        }, index=dates)
        price_data.columns = pd.MultiIndex.from_tuples(price_data.columns)

        mock_interface.get_price_data = Mock(return_value=price_data)

        return mock_interface

    @pytest.fixture
    def streaming_engine(self, streaming_config, mock_ml4t_interface):
        """Create test streaming engine."""
        return ML4TStreamingEngine(streaming_config, mock_ml4t_interface)

    @pytest.mark.asyncio
    async def test_streaming_engine_lifecycle(self, streaming_engine):
        """Test streaming engine start/stop lifecycle."""
        # Test start
        await streaming_engine.start()
        assert streaming_engine.status.value in ["running", "starting"]

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Test stop
        await streaming_engine.stop()
        assert streaming_engine.status.value == "stopped"

    @pytest.mark.asyncio
    async def test_streaming_data_flow(self, streaming_engine):
        """Test streaming data flow."""
        received_messages = []

        def data_callback(message: StreamingMessage):
            received_messages.append(message)

        streaming_engine.add_data_callback(data_callback)

        await streaming_engine.start()
        await asyncio.sleep(0.3)  # Let some messages flow
        await streaming_engine.stop()

        # Check that messages were received
        assert len(received_messages) > 0
        assert all(isinstance(msg, StreamingMessage) for msg in received_messages)

    @pytest.mark.asyncio
    async def test_streaming_buffer(self, streaming_engine):
        """Test streaming buffer functionality."""
        await streaming_engine.start()
        await asyncio.sleep(0.3)
        await streaming_engine.stop()

        # Test buffer access
        latest_data = streaming_engine.get_latest_data(count=5)
        assert isinstance(latest_data, list)

        if latest_data:
            symbol_data = streaming_engine.get_latest_data(symbol="2330.TW", count=3)
            assert isinstance(symbol_data, list)

        # Test DataFrame conversion
        df = streaming_engine.get_streaming_dataframe("2330.TW", lookback_minutes=60)
        assert isinstance(df, pd.DataFrame)

    def test_streaming_engine_factory(self, mock_ml4t_interface):
        """Test streaming engine factory function."""
        engine = create_streaming_engine(
            mock_ml4t_interface,
            symbols=["2330.TW"],
            mode=StreamingMode.SIMULATION
        )

        assert isinstance(engine, ML4TStreamingEngine)
        assert engine.config.symbols == ["2330.TW"]
        assert engine.config.mode == StreamingMode.SIMULATION


class TestBacktestOptimizer:
    """Test cases for backtest optimizer."""

    @pytest.fixture
    def optimization_config(self):
        """Create test optimization configuration."""
        return OptimizationConfig(
            level=OptimizationLevel.BALANCED,
            cache_size_mb=100,  # Small for testing
            enable_parallel=True,
            max_workers=2
        )

    @pytest.fixture
    def mock_ml4t_interface(self):
        """Create mock ML4T interface for optimizer tests."""
        mock_interface = Mock()

        # Mock price data method
        def mock_get_price_data(symbols, start_date, end_date, fields=None):
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
            data = {}

            for field in (fields or ['open', 'high', 'low', 'close', 'volume']):
                for symbol in symbols:
                    if field == 'volume':
                        values = np.random.randint(1000, 10000, len(dates))
                    else:
                        values = np.random.randn(len(dates)) * 10 + 100
                    data[(field, symbol)] = values

            df = pd.DataFrame(data, index=dates)
            df.columns = pd.MultiIndex.from_tuples(df.columns)
            return df

        mock_interface.get_price_data = Mock(side_effect=mock_get_price_data)
        return mock_interface

    @pytest.fixture
    def backtest_optimizer(self, optimization_config, mock_ml4t_interface):
        """Create test backtest optimizer."""
        return BacktestOptimizer(optimization_config, mock_ml4t_interface)

    def test_optimizer_initialization(self, backtest_optimizer):
        """Test optimizer initialization."""
        assert backtest_optimizer is not None
        assert backtest_optimizer.config.level == OptimizationLevel.BALANCED
        assert backtest_optimizer.cache is not None

    def test_optimize_data_loading(self, backtest_optimizer):
        """Test optimized data loading."""
        symbols = ["2330.TW", "2454.TW"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)

        data = backtest_optimizer.optimize_data_loading(
            symbols, start_date, end_date
        )

        assert isinstance(data, pd.DataFrame)
        assert not data.empty

        # Test caching - second call should be faster
        start_time = time.time()
        cached_data = backtest_optimizer.optimize_data_loading(
            symbols, start_date, end_date
        )
        cache_time = time.time() - start_time

        assert isinstance(cached_data, pd.DataFrame)
        assert cache_time < 0.1  # Should be very fast from cache

    def test_backtest_execution_optimization(self, backtest_optimizer):
        """Test backtest execution optimization."""
        # Create sample data
        dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
        data = pd.DataFrame({
            'close': np.random.randn(len(dates)) * 10 + 100,
        }, index=dates)

        # Simple backtest function
        def simple_backtest(data, **kwargs):
            returns = data['close'].pct_change().dropna()
            return {
                'total_return': (1 + returns).prod() - 1,
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
                'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min()
            }

        result = backtest_optimizer.optimize_backtest_execution(
            simple_backtest, data
        )

        assert isinstance(result, dict)
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result

    def test_batch_optimize_backtests(self, backtest_optimizer):
        """Test batch backtest optimization."""
        def simple_backtest(data, multiplier=1.0):
            returns = data['close'].pct_change().dropna() * multiplier
            return returns.mean()

        # Create multiple backtest configurations
        configs = []
        for i in range(3):
            dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
            data = pd.DataFrame({
                'close': np.random.randn(len(dates)) * 10 + 100,
            }, index=dates)

            configs.append({
                'function': simple_backtest,
                'data': data,
                'kwargs': {'multiplier': 1.0 + i * 0.1}
            })

        results = backtest_optimizer.batch_optimize_backtests(configs, parallel=False)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, (int, float)) for r in results)

    def test_cache_functionality(self, backtest_optimizer):
        """Test cache functionality."""
        cache = backtest_optimizer.cache

        # Test cache operations
        test_key = "test_key"
        test_data = pd.DataFrame({'test': [1, 2, 3]})

        # Put and get
        cache.put(test_key, test_data)
        retrieved = cache.get(test_key)

        assert retrieved is not None
        pd.testing.assert_frame_equal(test_data, retrieved)

        # Test cache stats
        stats = cache.get_stats()
        assert isinstance(stats, dict)
        assert stats['cache_hits'] > 0

    def test_performance_stats(self, backtest_optimizer):
        """Test performance statistics."""
        # Perform some operations
        symbols = ["2330.TW"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        backtest_optimizer.optimize_data_loading(symbols, start_date, end_date)

        stats = backtest_optimizer.get_performance_stats()

        assert isinstance(stats, dict)
        assert 'optimization_count' in stats
        assert 'total_optimization_time' in stats
        assert 'optimization_level' in stats

    def test_optimizer_factory(self, mock_ml4t_interface):
        """Test optimizer factory function."""
        optimizer = create_backtest_optimizer(
            mock_ml4t_interface,
            optimization_level=OptimizationLevel.SPEED,
            cache_size_mb=200
        )

        assert isinstance(optimizer, BacktestOptimizer)
        assert optimizer.config.level == OptimizationLevel.SPEED
        assert optimizer.config.cache_size_mb == 200


class TestIntegrationWorkflow:
    """Test complete integration workflow."""

    @pytest.fixture
    def integration_setup(self):
        """Set up complete integration test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock configuration
            auth_config = Mock()
            finlab_config = Mock()
            finlab_config.auth_config = auth_config

            ml4t_config = ML4TDataConfig(
                finlab_config=finlab_config,
                data_cache_dir=Path(tmpdir)
            )

            # Mock ML4T interface
            with patch('src.integration.ml4t_data_interface.FinLabConnector'):
                interface = create_ml4t_interface()
                interface._connected = True

                # Mock data methods
                dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="D")
                sample_data = pd.DataFrame({
                    ('close', '2330.TW'): np.random.randn(len(dates)) * 10 + 100,
                    ('close', '2454.TW'): np.random.randn(len(dates)) * 10 + 100,
                }, index=dates)
                sample_data.columns = pd.MultiIndex.from_tuples(sample_data.columns)

                interface.get_price_data = Mock(return_value=sample_data)
                interface.get_symbols = Mock(return_value=["2330.TW", "2454.TW"])

                yield {
                    'interface': interface,
                    'tmpdir': tmpdir,
                    'sample_data': sample_data
                }

    def test_complete_integration_workflow(self, integration_setup):
        """Test complete ML4T-Alpha integration workflow."""
        interface = integration_setup['interface']
        tmpdir = integration_setup['tmpdir']

        symbols = ["2330.TW", "2454.TW"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)

        # 1. Test data loading
        price_data = interface.get_price_data(symbols, start_date, end_date)
        assert not price_data.empty

        # 2. Test format conversion
        conversion_config = ConversionConfig(target_format=DataFormat.ML4T_ALPHA)

        # Create temporal values from sample data
        temporal_values = []
        for date_idx in price_data.index:
            for symbol in symbols:
                tv = TemporalValue(
                    value=price_data.loc[date_idx, ('close', symbol)],
                    as_of_date=date_idx.date(),
                    value_date=date_idx.date(),
                    data_type=DataType.PRICE,
                    symbol=symbol,
                    metadata={"field": "close", "source": "test"}
                )
                temporal_values.append(tv)

        converter = FinLabToML4TConverter(conversion_config)
        converted_data = converter.convert_price_data(temporal_values)
        assert not converted_data.empty

        # 3. Test optimization
        optimizer = create_backtest_optimizer(interface)
        optimized_data = optimizer.optimize_data_loading(symbols, start_date, end_date)
        assert not optimized_data.empty

        # 4. Test export
        export_path = interface.export_backtest_data(
            symbols, start_date, end_date,
            output_path=Path(tmpdir) / "integrated_test.h5"
        )
        assert export_path.exists()

        # 5. Test validation
        validation_report = interface.validate_backtest_readiness(
            symbols, start_date, end_date
        )
        assert isinstance(validation_report, dict)
        assert "ready" in validation_report

        # Verify all components worked together
        assert converter.conversion_count > 0
        assert optimizer.optimization_count > 0

    @pytest.mark.asyncio
    async def test_streaming_integration_workflow(self, integration_setup):
        """Test streaming integration workflow."""
        interface = integration_setup['interface']

        # Create streaming engine
        streaming_config = StreamingConfig(
            mode=StreamingMode.SIMULATION,
            symbols=["2330.TW"],
            update_interval_ms=50,  # Fast for testing
            buffer_size=50
        )

        engine = ML4TStreamingEngine(streaming_config, interface)

        # Test streaming workflow
        messages_received = []

        def message_handler(message: StreamingMessage):
            messages_received.append(message)

        engine.add_data_callback(message_handler)

        # Start streaming
        await engine.start()
        await asyncio.sleep(0.2)  # Let messages flow
        await engine.stop()

        # Verify streaming worked
        assert len(messages_received) > 0
        assert all(isinstance(msg, StreamingMessage) for msg in messages_received)

        # Test performance stats
        stats = engine.get_performance_stats()
        assert stats['messages_processed'] > 0


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_data_loading_performance(self, integration_setup):
        """Benchmark data loading performance."""
        interface = integration_setup['interface']

        symbols = ["2330.TW", "2454.TW"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 30)

        # Measure loading time
        start_time = time.time()
        data = interface.get_price_data(symbols, start_date, end_date)
        load_time = time.time() - start_time

        assert not data.empty
        assert load_time < 1.0  # Should complete within 1 second

    def test_optimization_performance(self, integration_setup):
        """Benchmark optimization performance."""
        interface = integration_setup['interface']
        optimizer = create_backtest_optimizer(interface)

        symbols = ["2330.TW", "2454.TW", "2317.TW"]
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 30)

        # Measure optimization time
        start_time = time.time()
        optimized_data = optimizer.optimize_data_loading(symbols, start_date, end_date)
        optimization_time = time.time() - start_time

        assert not optimized_data.empty
        assert optimization_time < 2.0  # Should complete within 2 seconds

        # Test cache performance
        start_time = time.time()
        cached_data = optimizer.optimize_data_loading(symbols, start_date, end_date)
        cache_time = time.time() - start_time

        assert cache_time < 0.1  # Cached access should be very fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])