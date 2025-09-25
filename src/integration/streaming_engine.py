"""
Real-time Streaming Engine for ML4T-Alpha Integration.

This module provides real-time data streaming capabilities for ML4T-Alpha
backtesting framework, enabling live data feeds and streaming analysis.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

from ..data.core.temporal import TemporalValue, DataType
from ..data.models.taiwan_market import TaiwanTradingCalendar, is_taiwan_trading_day
from .ml4t_data_interface import ML4TDataInterface

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming operation modes."""
    LIVE = "live"              # Live market data
    SIMULATION = "simulation"  # Historical data replay
    HYBRID = "hybrid"          # Live + historical combination


class StreamingStatus(Enum):
    """Streaming engine status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StreamingConfig:
    """Configuration for streaming engine."""
    # Streaming mode and timing
    mode: StreamingMode = StreamingMode.SIMULATION
    update_interval_ms: int = 1000  # 1 second default
    buffer_size: int = 1000
    max_latency_ms: int = 100

    # Data sources and symbols
    symbols: List[str] = field(default_factory=list)
    data_types: List[DataType] = field(default_factory=lambda: [DataType.PRICE, DataType.VOLUME])

    # Quality and reliability
    enable_heartbeat: bool = True
    heartbeat_interval_s: int = 30
    max_reconnect_attempts: int = 5
    reconnect_delay_s: int = 5

    # Performance optimization
    enable_compression: bool = True
    batch_size: int = 100
    async_processing: bool = True
    max_workers: int = 4

    # Data validation
    enable_data_validation: bool = True
    max_price_change_pct: float = 0.2  # 20% price change limit
    stale_data_threshold_s: int = 300  # 5 minutes


@dataclass
class StreamingMessage:
    """Streaming data message."""
    symbol: str
    timestamp: datetime
    data_type: DataType
    data: Dict[str, Any]
    sequence_number: int = 0
    source: str = "finlab"

    def to_temporal_value(self) -> TemporalValue:
        """Convert to TemporalValue for processing."""
        return TemporalValue(
            value=self.data.get('value'),
            as_of_date=self.timestamp.date(),
            value_date=self.timestamp.date(),
            data_type=self.data_type,
            symbol=self.symbol,
            metadata={
                'source': self.source,
                'sequence_number': self.sequence_number,
                'raw_data': self.data
            }
        )


class StreamingBuffer:
    """High-performance circular buffer for streaming data."""

    def __init__(self, size: int = 1000):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = threading.Lock()
        self._total_messages = 0

    def put(self, message: StreamingMessage) -> None:
        """Add message to buffer."""
        with self.lock:
            self.buffer.append(message)
            self._total_messages += 1

    def get_latest(self, count: int = 1) -> List[StreamingMessage]:
        """Get latest messages."""
        with self.lock:
            if count >= len(self.buffer):
                return list(self.buffer)
            else:
                return list(self.buffer)[-count:]

    def get_by_symbol(self, symbol: str, count: int = 10) -> List[StreamingMessage]:
        """Get latest messages for specific symbol."""
        with self.lock:
            symbol_messages = [msg for msg in self.buffer if msg.symbol == symbol]
            return symbol_messages[-count:] if count < len(symbol_messages) else symbol_messages

    def get_by_timerange(self,
                        start_time: datetime,
                        end_time: datetime) -> List[StreamingMessage]:
        """Get messages within time range."""
        with self.lock:
            return [msg for msg in self.buffer
                   if start_time <= msg.timestamp <= end_time]

    def clear(self) -> None:
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                'current_size': len(self.buffer),
                'max_size': self.size,
                'total_messages': self._total_messages,
                'utilization': len(self.buffer) / self.size
            }


class StreamingDataProvider:
    """Base class for streaming data providers."""

    async def connect(self) -> None:
        """Connect to data source."""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Disconnect from data source."""
        raise NotImplementedError

    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> None:
        """Subscribe to data streams."""
        raise NotImplementedError

    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> None:
        """Unsubscribe from data streams."""
        raise NotImplementedError

    async def get_data_stream(self) -> AsyncGenerator[StreamingMessage, None]:
        """Get async data stream."""
        raise NotImplementedError


class FinLabStreamingProvider(StreamingDataProvider):
    """FinLab-based streaming data provider."""

    def __init__(self,
                 ml4t_interface: ML4TDataInterface,
                 config: StreamingConfig):
        self.ml4t_interface = ml4t_interface
        self.config = config
        self._connected = False
        self._subscriptions: Dict[str, Set[DataType]] = defaultdict(set)
        self._sequence_number = 0

    async def connect(self) -> None:
        """Connect to FinLab data source."""
        try:
            if not self.ml4t_interface._connected:
                self.ml4t_interface.connect()
            self._connected = True
            logger.info("FinLab streaming provider connected")
        except Exception as e:
            logger.error(f"Failed to connect FinLab streaming provider: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from FinLab data source."""
        if self._connected:
            self.ml4t_interface.disconnect()
            self._connected = False
            logger.info("FinLab streaming provider disconnected")

    async def subscribe(self, symbols: List[str], data_types: List[DataType]) -> None:
        """Subscribe to symbol data streams."""
        for symbol in symbols:
            self._subscriptions[symbol].update(data_types)
        logger.info(f"Subscribed to {len(symbols)} symbols with {len(data_types)} data types")

    async def unsubscribe(self, symbols: List[str], data_types: List[DataType]) -> None:
        """Unsubscribe from symbol data streams."""
        for symbol in symbols:
            self._subscriptions[symbol] -= set(data_types)
            if not self._subscriptions[symbol]:
                del self._subscriptions[symbol]

    async def get_data_stream(self) -> AsyncGenerator[StreamingMessage, None]:
        """Generate streaming data messages."""
        if not self._connected:
            await self.connect()

        if self.config.mode == StreamingMode.SIMULATION:
            async for message in self._simulate_data_stream():
                yield message
        elif self.config.mode == StreamingMode.LIVE:
            async for message in self._live_data_stream():
                yield message
        else:  # HYBRID
            async for message in self._hybrid_data_stream():
                yield message

    async def _simulate_data_stream(self) -> AsyncGenerator[StreamingMessage, None]:
        """Simulate streaming data from historical data."""
        # Get historical data for simulation
        end_date = date.today()
        start_date = end_date - timedelta(days=30)  # 30 days of data

        symbols = list(self._subscriptions.keys())
        if not symbols:
            return

        # Get price data for simulation
        price_data = self.ml4t_interface.get_price_data(
            symbols, start_date, end_date
        )

        if price_data.empty:
            logger.warning("No historical data available for simulation")
            return

        # Convert to streaming messages
        for date_idx in price_data.index:
            current_time = datetime.combine(date_idx.date(), datetime.min.time())

            for symbol in symbols:
                data_types = self._subscriptions[symbol]

                if DataType.PRICE in data_types:
                    # Extract price data for this symbol and date
                    if isinstance(price_data.columns, pd.MultiIndex):
                        try:
                            symbol_data = price_data.xs(symbol, axis=1, level=1).loc[date_idx]
                            price_message = StreamingMessage(
                                symbol=symbol,
                                timestamp=current_time,
                                data_type=DataType.PRICE,
                                data={
                                    'open': symbol_data.get('open', np.nan),
                                    'high': symbol_data.get('high', np.nan),
                                    'low': symbol_data.get('low', np.nan),
                                    'close': symbol_data.get('close', np.nan),
                                    'volume': symbol_data.get('volume', 0)
                                },
                                sequence_number=self._sequence_number,
                                source="finlab_simulation"
                            )
                            self._sequence_number += 1
                            yield price_message
                        except Exception as e:
                            logger.debug(f"Failed to extract data for {symbol} on {date_idx}: {e}")

            # Simulate real-time delay
            await asyncio.sleep(self.config.update_interval_ms / 1000.0)

    async def _live_data_stream(self) -> AsyncGenerator[StreamingMessage, None]:
        """Generate live data stream (placeholder for real implementation)."""
        logger.warning("Live data streaming not implemented - falling back to simulation")
        async for message in self._simulate_data_stream():
            yield message

    async def _hybrid_data_stream(self) -> AsyncGenerator[StreamingMessage, None]:
        """Generate hybrid live/historical data stream."""
        logger.warning("Hybrid data streaming not implemented - falling back to simulation")
        async for message in self._simulate_data_stream():
            yield message


class ML4TStreamingEngine:
    """
    High-performance streaming engine for ML4T-Alpha integration.
    """

    def __init__(self,
                 config: StreamingConfig,
                 ml4t_interface: ML4TDataInterface,
                 data_provider: Optional[StreamingDataProvider] = None):
        self.config = config
        self.ml4t_interface = ml4t_interface

        # Initialize data provider
        if data_provider is None:
            data_provider = FinLabStreamingProvider(ml4t_interface, config)
        self.data_provider = data_provider

        # Streaming infrastructure
        self.buffer = StreamingBuffer(config.buffer_size)
        self.status = StreamingStatus.STOPPED
        self._streaming_task: Optional[asyncio.Task] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Callbacks and handlers
        self._data_callbacks: List[Callable[[StreamingMessage], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # Performance metrics
        self.messages_processed = 0
        self.start_time: Optional[datetime] = None
        self.last_message_time: Optional[datetime] = None
        self.error_count = 0

        # Quality control
        self._last_prices: Dict[str, float] = {}

        logger.info("ML4T streaming engine initialized")

    def add_data_callback(self, callback: Callable[[StreamingMessage], None]) -> None:
        """Add callback for incoming data messages."""
        self._data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add callback for error handling."""
        self._error_callbacks.append(callback)

    async def start(self) -> None:
        """Start streaming engine."""
        if self.status == StreamingStatus.RUNNING:
            logger.warning("Streaming engine already running")
            return

        try:
            self.status = StreamingStatus.STARTING
            logger.info("Starting ML4T streaming engine")

            # Connect data provider
            await self.data_provider.connect()

            # Subscribe to configured symbols and data types
            if self.config.symbols:
                await self.data_provider.subscribe(
                    self.config.symbols,
                    self.config.data_types
                )

            # Start streaming task
            self._streaming_task = asyncio.create_task(self._streaming_loop())
            self.status = StreamingStatus.RUNNING
            self.start_time = datetime.utcnow()

            logger.info("ML4T streaming engine started successfully")

        except Exception as e:
            self.status = StreamingStatus.ERROR
            self.error_count += 1
            logger.error(f"Failed to start streaming engine: {e}")
            await self._handle_error(e)
            raise

    async def stop(self) -> None:
        """Stop streaming engine."""
        logger.info("Stopping ML4T streaming engine")

        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass

        await self.data_provider.disconnect()
        self.status = StreamingStatus.STOPPED

        logger.info("ML4T streaming engine stopped")

    async def pause(self) -> None:
        """Pause streaming engine."""
        if self.status == StreamingStatus.RUNNING:
            self.status = StreamingStatus.PAUSED
            logger.info("ML4T streaming engine paused")

    async def resume(self) -> None:
        """Resume streaming engine."""
        if self.status == StreamingStatus.PAUSED:
            self.status = StreamingStatus.RUNNING
            logger.info("ML4T streaming engine resumed")

    async def _streaming_loop(self) -> None:
        """Main streaming loop."""
        try:
            async for message in self.data_provider.get_data_stream():
                if self.status == StreamingStatus.PAUSED:
                    await asyncio.sleep(0.1)
                    continue

                if self.status != StreamingStatus.RUNNING:
                    break

                # Process message
                await self._process_message(message)

        except asyncio.CancelledError:
            logger.info("Streaming loop cancelled")
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
            self.status = StreamingStatus.ERROR
            await self._handle_error(e)

    async def _process_message(self, message: StreamingMessage) -> None:
        """Process incoming streaming message."""
        try:
            # Data quality validation
            if self.config.enable_data_validation:
                if not self._validate_message(message):
                    logger.debug(f"Message validation failed for {message.symbol}")
                    return

            # Add to buffer
            self.buffer.put(message)

            # Update metrics
            self.messages_processed += 1
            self.last_message_time = message.timestamp

            # Execute callbacks
            if self.config.async_processing:
                # Async callback execution
                tasks = [
                    asyncio.create_task(self._run_callback(callback, message))
                    for callback in self._data_callbacks
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Sync callback execution
                for callback in self._data_callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._handle_error(e)

    async def _run_callback(self, callback: Callable, message: StreamingMessage) -> None:
        """Run callback asynchronously."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                # Run sync callback in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, message)
        except Exception as e:
            logger.warning(f"Async callback error: {e}")

    def _validate_message(self, message: StreamingMessage) -> bool:
        """Validate streaming message quality."""
        # Check data freshness
        age_seconds = (datetime.utcnow() - message.timestamp).total_seconds()
        if age_seconds > self.config.stale_data_threshold_s:
            return False

        # Price change validation
        if message.data_type == DataType.PRICE and 'close' in message.data:
            current_price = message.data['close']
            if current_price and not pd.isna(current_price):
                last_price = self._last_prices.get(message.symbol)
                if last_price:
                    price_change_pct = abs(current_price - last_price) / last_price
                    if price_change_pct > self.config.max_price_change_pct:
                        logger.warning(f"Large price change detected for {message.symbol}: {price_change_pct:.2%}")
                        # Still accept but log the warning

                self._last_prices[message.symbol] = current_price

        return True

    async def _handle_error(self, error: Exception) -> None:
        """Handle streaming errors."""
        self.error_count += 1

        # Execute error callbacks
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def get_latest_data(self,
                       symbol: Optional[str] = None,
                       data_type: Optional[DataType] = None,
                       count: int = 1) -> List[StreamingMessage]:
        """Get latest streaming data."""
        if symbol:
            messages = self.buffer.get_by_symbol(symbol, count)
        else:
            messages = self.buffer.get_latest(count)

        if data_type:
            messages = [msg for msg in messages if msg.data_type == data_type]

        return messages

    def get_streaming_dataframe(self,
                              symbol: str,
                              lookback_minutes: int = 60) -> pd.DataFrame:
        """Get recent streaming data as DataFrame."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=lookback_minutes)

        messages = self.buffer.get_by_timerange(start_time, end_time)
        symbol_messages = [msg for msg in messages if msg.symbol == symbol]

        if not symbol_messages:
            return pd.DataFrame()

        # Convert to DataFrame
        data_rows = []
        for msg in symbol_messages:
            row = {
                'timestamp': msg.timestamp,
                'symbol': msg.symbol,
                'data_type': msg.data_type.value,
                'sequence_number': msg.sequence_number
            }
            row.update(msg.data)
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        return df

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get streaming engine performance statistics."""
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()

        messages_per_second = self.messages_processed / max(uptime_seconds, 1)

        stats = {
            'status': self.status.value,
            'messages_processed': self.messages_processed,
            'uptime_seconds': uptime_seconds,
            'messages_per_second': messages_per_second,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.messages_processed, 1),
            'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None
        }

        # Add buffer stats
        stats.update({f'buffer_{k}': v for k, v in self.buffer.get_stats().items()})

        return stats

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


def create_streaming_engine(
    ml4t_interface: ML4TDataInterface,
    symbols: List[str],
    mode: StreamingMode = StreamingMode.SIMULATION,
    update_interval_ms: int = 1000,
    **kwargs
) -> ML4TStreamingEngine:
    """
    Factory function to create ML4T streaming engine.

    Args:
        ml4t_interface: ML4T data interface
        symbols: Symbols to stream
        mode: Streaming mode
        update_interval_ms: Update interval in milliseconds
        **kwargs: Additional configuration parameters

    Returns:
        Configured streaming engine
    """
    config = StreamingConfig(
        mode=mode,
        symbols=symbols,
        update_interval_ms=update_interval_ms,
        **kwargs
    )

    return ML4TStreamingEngine(config, ml4t_interface)