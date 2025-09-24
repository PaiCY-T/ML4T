"""
High-frequency tick data handler for Taiwan market microstructure calculations.

This module provides efficient processing of tick-by-tick market data for 
Taiwan equity markets, supporting:
- Tick data cleaning and validation
- Taiwan session filtering (09:00-13:30 TST)
- VWAP and microstructure metric calculations
- Memory-efficient streaming processing
- Variable tick size handling
"""

from datetime import datetime, time, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Iterator, Generator
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

# Import dependencies - will be mocked for testing if not available
try:
    from ..data.core.temporal import TemporalValue, DataType
    from ..data.models.taiwan_market import TradingStatus
except ImportError:
    TemporalValue = object
    DataType = object
    TradingStatus = object

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Single tick data point."""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    trade_type: Optional[str] = None  # 'buy', 'sell', 'unknown'
    
    @property
    def midpoint(self) -> Optional[float]:
        """Calculate bid-ask midpoint."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None and self.ask > 0:
            return (self.ask - self.bid) / self.ask
        return None


@dataclass
class IntradayMetrics:
    """Intraday microstructure metrics."""
    symbol: str
    date: date
    
    # Price metrics
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None  
    close_price: Optional[float] = None
    vwap: Optional[float] = None
    
    # Volume metrics
    total_volume: int = 0
    trade_count: int = 0
    avg_trade_size: Optional[float] = None
    
    # Microstructure metrics
    avg_spread: Optional[float] = None
    median_spread: Optional[float] = None
    spread_volatility: Optional[float] = None
    
    # Time-based metrics
    opening_volume: Optional[int] = None    # First 30 minutes
    closing_volume: Optional[int] = None    # Last 30 minutes
    session_volume: Optional[int] = None    # Core session volume
    
    # Quality metrics
    tick_count: int = 0
    valid_spread_count: int = 0
    data_completeness: Optional[float] = None


class TaiwanSessionFilter:
    """Filter for Taiwan trading session (09:00-13:30 TST)."""
    
    def __init__(self):
        self.session_start = time(9, 0)    # 09:00
        self.session_end = time(13, 30)    # 13:30
        self.opening_period = 30           # First 30 minutes
        self.closing_period = 30           # Last 30 minutes
    
    def is_trading_time(self, timestamp: datetime) -> bool:
        """Check if timestamp is within Taiwan trading hours."""
        t = timestamp.time()
        return self.session_start <= t <= self.session_end
    
    def is_opening_period(self, timestamp: datetime) -> bool:
        """Check if timestamp is in opening period (09:00-09:30)."""
        t = timestamp.time()
        return time(9, 0) <= t <= time(9, 30)
    
    def is_closing_period(self, timestamp: datetime) -> bool:
        """Check if timestamp is in closing period (13:00-13:30)."""
        t = timestamp.time()
        return time(13, 0) <= t <= time(13, 30)
    
    def get_session_period(self, timestamp: datetime) -> str:
        """Get session period classification."""
        if not self.is_trading_time(timestamp):
            return 'after_hours'
        elif self.is_opening_period(timestamp):
            return 'opening'
        elif self.is_closing_period(timestamp):
            return 'closing'
        else:
            return 'core'


class TickDataCleaner:
    """Clean and validate tick data for Taiwan market."""
    
    def __init__(self):
        self.session_filter = TaiwanSessionFilter()
        self.price_limit = 0.10  # 10% daily price limit
        
    def clean_tick_data(self, raw_ticks: List[Dict]) -> List[TickData]:
        """Clean and validate raw tick data."""
        cleaned_ticks = []
        
        for tick_dict in raw_ticks:
            try:
                tick = self._parse_tick(tick_dict)
                
                if tick and self._validate_tick(tick):
                    cleaned_ticks.append(tick)
                    
            except Exception as e:
                logger.warning(f"Error processing tick: {e}")
                continue
        
        return sorted(cleaned_ticks, key=lambda x: x.timestamp)
    
    def _parse_tick(self, tick_dict: Dict) -> Optional[TickData]:
        """Parse raw tick dictionary into TickData object."""
        try:
            return TickData(
                timestamp=self._parse_timestamp(tick_dict.get('timestamp')),
                symbol=str(tick_dict.get('symbol', '')),
                price=float(tick_dict.get('price', 0)),
                volume=int(tick_dict.get('volume', 0)),
                bid=self._safe_float(tick_dict.get('bid')),
                ask=self._safe_float(tick_dict.get('ask')),
                bid_size=self._safe_int(tick_dict.get('bid_size')),
                ask_size=self._safe_int(tick_dict.get('ask_size')),
                trade_type=tick_dict.get('trade_type')
            )
        except Exception as e:
            logger.warning(f"Error parsing tick: {e}")
            return None
    
    def _validate_tick(self, tick: TickData) -> bool:
        """Validate tick data quality."""
        # Basic validation
        if not tick.symbol or tick.price <= 0 or tick.volume < 0:
            return False
        
        # Taiwan trading session validation
        if not self.session_filter.is_trading_time(tick.timestamp):
            return False
        
        # Price reasonableness check
        if tick.price > 10000 or tick.price < 0.01:  # Extreme price check
            return False
        
        # Spread validation
        if tick.bid is not None and tick.ask is not None:
            if tick.bid <= 0 or tick.ask <= 0 or tick.bid >= tick.ask:
                return False
            
            spread = tick.spread
            if spread and spread > 0.20:  # 20% spread is unrealistic
                return False
        
        return True
    
    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse various timestamp formats."""
        if isinstance(ts, datetime):
            return ts
        elif isinstance(ts, str):
            # Try common formats
            for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y%m%d %H:%M:%S']:
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
        elif isinstance(ts, (int, float)):
            # Unix timestamp
            return datetime.fromtimestamp(ts)
        
        raise ValueError(f"Unable to parse timestamp: {ts}")
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert to float."""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value: Any) -> Optional[int]:
        """Safely convert to integer."""
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None


class IntradayMetricsCalculator:
    """Calculate intraday microstructure metrics from tick data."""
    
    def __init__(self):
        self.session_filter = TaiwanSessionFilter()
    
    def calculate_metrics(self, ticks: List[TickData], target_date: date) -> Dict[str, IntradayMetrics]:
        """Calculate intraday metrics by symbol."""
        # Group ticks by symbol
        symbol_ticks = defaultdict(list)
        
        for tick in ticks:
            if tick.timestamp.date() == target_date:
                symbol_ticks[tick.symbol].append(tick)
        
        # Calculate metrics for each symbol
        metrics = {}
        for symbol, symbol_tick_list in symbol_ticks.items():
            if len(symbol_tick_list) > 0:
                metrics[symbol] = self._calculate_symbol_metrics(
                    symbol, target_date, symbol_tick_list
                )
        
        return metrics
    
    def _calculate_symbol_metrics(self, symbol: str, target_date: date, 
                                 ticks: List[TickData]) -> IntradayMetrics:
        """Calculate metrics for a single symbol."""
        # Sort ticks by timestamp
        ticks = sorted(ticks, key=lambda x: x.timestamp)
        
        metrics = IntradayMetrics(symbol=symbol, date=target_date)
        metrics.tick_count = len(ticks)
        
        if not ticks:
            return metrics
        
        # Price metrics
        prices = [tick.price for tick in ticks]
        volumes = [tick.volume for tick in ticks]
        
        metrics.open_price = prices[0]
        metrics.high_price = max(prices)
        metrics.low_price = min(prices)
        metrics.close_price = prices[-1]
        
        # VWAP calculation
        price_volume_sum = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        metrics.total_volume = total_volume
        
        if total_volume > 0:
            metrics.vwap = price_volume_sum / total_volume
        
        # Volume metrics
        trade_volumes = [v for v in volumes if v > 0]
        if trade_volumes:
            metrics.trade_count = len(trade_volumes)
            metrics.avg_trade_size = sum(trade_volumes) / len(trade_volumes)
        
        # Spread metrics
        spreads = [tick.spread for tick in ticks if tick.spread is not None]
        if spreads:
            metrics.avg_spread = np.mean(spreads)
            metrics.median_spread = np.median(spreads)
            metrics.spread_volatility = np.std(spreads)
            metrics.valid_spread_count = len(spreads)
        
        # Time-based volume analysis
        opening_ticks = [t for t in ticks if self.session_filter.is_opening_period(t.timestamp)]
        closing_ticks = [t for t in ticks if self.session_filter.is_closing_period(t.timestamp)]
        
        metrics.opening_volume = sum(t.volume for t in opening_ticks)
        metrics.closing_volume = sum(t.volume for t in closing_ticks)
        metrics.session_volume = total_volume
        
        # Data quality
        expected_ticks = 270 * 60  # 4.5 hours * 60 minutes (assuming 1 tick per minute)
        metrics.data_completeness = min(1.0, len(ticks) / expected_ticks) if expected_ticks > 0 else 0
        
        return metrics


class TickDataAggregator:
    """Aggregate tick data into various time buckets."""
    
    def __init__(self):
        self.session_filter = TaiwanSessionFilter()
    
    def aggregate_to_minutes(self, ticks: List[TickData], 
                           interval_minutes: int = 1) -> pd.DataFrame:
        """Aggregate ticks to minute-level OHLCV data."""
        if not ticks:
            return pd.DataFrame()
        
        # Convert to DataFrame for easier manipulation
        tick_df = pd.DataFrame([{
            'timestamp': t.timestamp,
            'symbol': t.symbol,
            'price': t.price,
            'volume': t.volume,
            'bid': t.bid,
            'ask': t.ask
        } for t in ticks])
        
        # Create time buckets
        tick_df['minute_bucket'] = tick_df['timestamp'].dt.floor(f'{interval_minutes}min')
        
        # Aggregate by symbol and time bucket
        agg_data = []
        
        for (symbol, minute_bucket), group in tick_df.groupby(['symbol', 'minute_bucket']):
            if len(group) == 0:
                continue
            
            # OHLCV calculation
            prices = group['price'].values
            volumes = group['volume'].values
            
            agg_row = {
                'timestamp': minute_bucket,
                'symbol': symbol,
                'open': prices[0],
                'high': np.max(prices),
                'low': np.min(prices),
                'close': prices[-1],
                'volume': np.sum(volumes),
                'trade_count': len(group),
                'vwap': np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else prices[-1]
            }
            
            # Add spread information if available
            spreads = group['bid'].combine(group['ask'], 
                                         lambda b, a: (a - b) / a if pd.notna(b) and pd.notna(a) and a > 0 else np.nan)
            valid_spreads = spreads.dropna()
            
            if len(valid_spreads) > 0:
                agg_row.update({
                    'avg_spread': valid_spreads.mean(),
                    'median_spread': valid_spreads.median()
                })
            
            agg_data.append(agg_row)
        
        return pd.DataFrame(agg_data).sort_values(['symbol', 'timestamp'])
    
    def calculate_session_metrics(self, ticks: List[TickData]) -> Dict[str, Dict[str, float]]:
        """Calculate session-level metrics by trading period."""
        period_metrics = defaultdict(lambda: defaultdict(list))
        
        # Group ticks by symbol and session period
        for tick in ticks:
            period = self.session_filter.get_session_period(tick.timestamp)
            
            period_metrics[tick.symbol]['period'].append(period)
            period_metrics[tick.symbol]['price'].append(tick.price)
            period_metrics[tick.symbol]['volume'].append(tick.volume)
            
            if tick.spread is not None:
                period_metrics[tick.symbol]['spread'].append(tick.spread)
        
        # Calculate aggregated metrics
        results = {}
        
        for symbol, data in period_metrics.items():
            symbol_results = {}
            
            # Group by period
            df = pd.DataFrame(data)
            
            for period, period_group in df.groupby('period'):
                period_results = {
                    'volume': period_group['volume'].sum(),
                    'trade_count': len(period_group),
                    'avg_price': period_group['price'].mean(),
                    'price_volatility': period_group['price'].std(),
                }
                
                if 'spread' in period_group.columns:
                    spreads = period_group['spread'].dropna()
                    if len(spreads) > 0:
                        period_results.update({
                            'avg_spread': spreads.mean(),
                            'spread_volatility': spreads.std()
                        })
                
                symbol_results[period] = period_results
            
            results[symbol] = symbol_results
        
        return results


class TickDataHandler:
    """Main handler for Taiwan tick data processing."""
    
    def __init__(self, memory_limit_mb: int = 1000):
        """
        Initialize tick data handler.
        
        Args:
            memory_limit_mb: Memory limit for processing in MB
        """
        self.cleaner = TickDataCleaner()
        self.metrics_calculator = IntradayMetricsCalculator()
        self.aggregator = TickDataAggregator()
        self.session_filter = TaiwanSessionFilter()
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
    
    def process_daily_ticks(self, raw_tick_data: List[Dict], 
                          target_date: date) -> Dict[str, Any]:
        """Process a full day of tick data."""
        
        results = {
            'date': target_date,
            'processed_ticks': 0,
            'valid_ticks': 0,
            'symbols_processed': 0,
            'intraday_metrics': {},
            'minute_data': pd.DataFrame(),
            'session_metrics': {},
            'processing_stats': {}
        }
        
        try:
            start_time = datetime.now()
            
            # Clean tick data
            logger.info(f"Processing {len(raw_tick_data)} raw ticks for {target_date}")
            cleaned_ticks = self.cleaner.clean_tick_data(raw_tick_data)
            
            results['processed_ticks'] = len(raw_tick_data)
            results['valid_ticks'] = len(cleaned_ticks)
            
            if not cleaned_ticks:
                logger.warning(f"No valid ticks found for {target_date}")
                return results
            
            # Filter for target date
            date_ticks = [t for t in cleaned_ticks if t.timestamp.date() == target_date]
            
            if not date_ticks:
                logger.warning(f"No ticks found for target date {target_date}")
                return results
            
            # Calculate intraday metrics
            logger.info(f"Calculating intraday metrics for {len(date_ticks)} valid ticks")
            results['intraday_metrics'] = self.metrics_calculator.calculate_metrics(
                date_ticks, target_date
            )
            
            # Aggregate to minute data
            logger.info("Aggregating to minute-level data")
            results['minute_data'] = self.aggregator.aggregate_to_minutes(date_ticks, interval_minutes=1)
            
            # Calculate session metrics
            logger.info("Calculating session-level metrics")
            results['session_metrics'] = self.aggregator.calculate_session_metrics(date_ticks)
            
            # Processing statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            symbols_processed = len(set(t.symbol for t in date_ticks))
            
            results['symbols_processed'] = symbols_processed
            results['processing_stats'] = {
                'processing_time_seconds': processing_time,
                'ticks_per_second': len(date_ticks) / processing_time if processing_time > 0 else 0,
                'data_quality_ratio': len(date_ticks) / len(raw_tick_data) if len(raw_tick_data) > 0 else 0
            }
            
            logger.info(
                f"Processed {symbols_processed} symbols, {len(date_ticks)} valid ticks "
                f"in {processing_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error processing tick data for {target_date}: {e}")
            results['processing_stats']['error'] = str(e)
        
        return results
    
    def stream_process_ticks(self, tick_stream: Iterator[Dict]) -> Generator[TickData, None, None]:
        """Stream process ticks with memory management."""
        
        buffer = []
        buffer_size = 0
        
        for raw_tick in tick_stream:
            try:
                # Estimate memory usage (rough approximation)
                tick_size = len(str(raw_tick)) * 2  # Rough estimate
                
                if buffer_size + tick_size > self.memory_limit_bytes:
                    # Process buffer
                    if buffer:
                        cleaned_ticks = self.cleaner.clean_tick_data(buffer)
                        for tick in cleaned_ticks:
                            yield tick
                    
                    # Reset buffer
                    buffer = []
                    buffer_size = 0
                
                buffer.append(raw_tick)
                buffer_size += tick_size
                
            except Exception as e:
                logger.warning(f"Error in stream processing: {e}")
                continue
        
        # Process remaining buffer
        if buffer:
            cleaned_ticks = self.cleaner.clean_tick_data(buffer)
            for tick in cleaned_ticks:
                yield tick
    
    def get_microstructure_features(self, ticks: List[TickData], 
                                  symbol: str) -> Dict[str, float]:
        """Extract microstructure features for factor calculations."""
        
        symbol_ticks = [t for t in ticks if t.symbol == symbol]
        
        if not symbol_ticks:
            return {}
        
        features = {}
        
        # Price impact measures
        price_changes = np.diff([t.price for t in symbol_ticks])
        volume_weights = np.array([t.volume for t in symbol_ticks[1:]])
        
        if len(price_changes) > 0 and len(volume_weights) > 0:
            # Volume-weighted price impact
            nonzero_volume = volume_weights > 0
            if np.sum(nonzero_volume) > 0:
                weighted_impact = np.average(
                    np.abs(price_changes)[nonzero_volume], 
                    weights=volume_weights[nonzero_volume]
                )
                features['volume_weighted_impact'] = weighted_impact
        
        # Spread statistics
        spreads = [t.spread for t in symbol_ticks if t.spread is not None]
        if spreads:
            features.update({
                'avg_spread': np.mean(spreads),
                'spread_volatility': np.std(spreads),
                'median_spread': np.median(spreads)
            })
        
        # Volume clustering
        volumes = [t.volume for t in symbol_ticks]
        if volumes:
            volume_array = np.array(volumes)
            features.update({
                'volume_cv': np.std(volume_array) / np.mean(volume_array) if np.mean(volume_array) > 0 else 0,
                'volume_skewness': self._safe_skewness(volume_array)
            })
        
        # Tick frequency features  
        if len(symbol_ticks) >= 2:
            timestamps = [t.timestamp for t in symbol_ticks]
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                        for i in range(1, len(timestamps))]
            
            features.update({
                'avg_tick_interval': np.mean(intervals),
                'tick_regularity': 1.0 / (np.std(intervals) + 1e-6) if intervals else 0
            })
        
        return features
    
    def _safe_skewness(self, data: np.ndarray) -> float:
        """Safely calculate skewness."""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            # Fallback calculation
            if len(data) < 3:
                return 0.0
            
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val == 0:
                return 0.0
            
            return np.mean(((data - mean_val) / std_val) ** 3)