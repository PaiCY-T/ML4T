"""
Backtest Performance Optimizer for ML4T-Alpha Integration.

This module provides performance optimization for ML4T-Alpha backtesting workflows,
focusing on data access optimization, memory management, and execution speed.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import pickle
import gc
import psutil
import time

from ..data.core.temporal import DataType
from ..data.pipeline.pit_engine import PITQuery, QueryMode, BiasCheckLevel
from .ml4t_data_interface import ML4TDataInterface

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for different performance needs."""
    BASIC = "basic"           # Basic optimizations
    AGGRESSIVE = "aggressive" # Aggressive optimizations
    MEMORY = "memory"         # Memory-focused optimizations
    SPEED = "speed"          # Speed-focused optimizations
    BALANCED = "balanced"     # Balanced optimization


class CacheStrategy(Enum):
    """Data caching strategies."""
    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"


@dataclass
class OptimizationConfig:
    """Configuration for backtest optimization."""
    # Optimization level
    level: OptimizationLevel = OptimizationLevel.BALANCED

    # Caching configuration
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID
    cache_size_mb: int = 1024  # 1GB default
    cache_directory: Optional[Path] = None

    # Parallel processing
    enable_parallel: bool = True
    max_workers: int = multiprocessing.cpu_count()
    chunk_size: int = 1000

    # Memory management
    enable_memory_optimization: bool = True
    memory_limit_mb: int = 4096  # 4GB default
    gc_frequency: int = 100  # Garbage collect every N operations

    # Data access optimization
    bulk_query_threshold: int = 50
    prefetch_window_days: int = 7
    enable_data_compression: bool = True

    # Performance tuning
    enable_vectorization: bool = True
    use_numba: bool = False  # JIT compilation
    optimize_pandas: bool = True

    def __post_init__(self):
        """Initialize derived configurations."""
        if self.cache_directory is None:
            self.cache_directory = Path.cwd() / "cache" / "backtest_optimizer"
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        # Adjust settings based on optimization level
        if self.level == OptimizationLevel.SPEED:
            self.enable_parallel = True
            self.cache_strategy = CacheStrategy.MEMORY
            self.enable_vectorization = True
            self.use_numba = True
        elif self.level == OptimizationLevel.MEMORY:
            self.cache_strategy = CacheStrategy.DISK
            self.enable_memory_optimization = True
            self.gc_frequency = 50
        elif self.level == OptimizationLevel.AGGRESSIVE:
            self.enable_parallel = True
            self.use_numba = True
            self.enable_vectorization = True
            self.cache_strategy = CacheStrategy.HYBRID


class DataCache:
    """High-performance data cache for backtest optimization."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}
        self.last_access: Dict[str, datetime] = {}

        # Memory tracking
        self.current_memory_mb = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _make_key(self, *args) -> str:
        """Generate cache key from arguments."""
        return str(hash(tuple(str(arg) for arg in args)))

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.memory_cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.last_access[key] = datetime.utcnow()
            self.cache_hits += 1
            return self.memory_cache[key]

        # Try disk cache if enabled
        if self.config.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
            disk_path = self.config.cache_directory / f"{key}.pkl"
            if disk_path.exists():
                try:
                    with open(disk_path, 'rb') as f:
                        data = pickle.load(f)

                    # Move to memory cache if there's space
                    if self.config.cache_strategy == CacheStrategy.HYBRID:
                        self._put_memory(key, data)

                    self.cache_hits += 1
                    return data
                except Exception as e:
                    logger.debug(f"Failed to load from disk cache: {e}")

        self.cache_misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if self.config.cache_strategy == CacheStrategy.NONE:
            return

        # Estimate memory size
        memory_size = self._estimate_size(value)

        if self.config.cache_strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
            if self.current_memory_mb + memory_size <= self.config.cache_size_mb:
                self._put_memory(key, value)
            elif self.config.cache_strategy == CacheStrategy.HYBRID:
                # Spill to disk
                self._put_disk(key, value)
        elif self.config.cache_strategy == CacheStrategy.DISK:
            self._put_disk(key, value)

    def _put_memory(self, key: str, value: Any) -> None:
        """Put item in memory cache."""
        memory_size = self._estimate_size(value)

        # Evict if necessary
        while (self.current_memory_mb + memory_size > self.config.cache_size_mb and
               self.memory_cache):
            self._evict_lru()

        self.memory_cache[key] = value
        self.access_count[key] = 1
        self.last_access[key] = datetime.utcnow()
        self.current_memory_mb += memory_size

    def _put_disk(self, key: str, value: Any) -> None:
        """Put item in disk cache."""
        disk_path = self.config.cache_directory / f"{key}.pkl"
        try:
            with open(disk_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")

    def _evict_lru(self) -> None:
        """Evict least recently used item from memory cache."""
        if not self.memory_cache:
            return

        # Find LRU item
        lru_key = min(self.last_access.keys(), key=lambda k: self.last_access[k])

        # Remove from memory
        value = self.memory_cache.pop(lru_key)
        self.access_count.pop(lru_key)
        self.last_access.pop(lru_key)

        # Update memory usage
        self.current_memory_mb -= self._estimate_size(value)

        # Optionally save to disk
        if self.config.cache_strategy == CacheStrategy.HYBRID:
            self._put_disk(lru_key, value)

    def _estimate_size(self, obj: Any) -> float:
        """Estimate memory size in MB."""
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj[:10]) * len(obj) / 10
            else:
                # Rough estimate
                return len(str(obj)) / (1024 * 1024)
        except:
            return 0.1  # Default 100KB estimate

    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        self.access_count.clear()
        self.last_access.clear()
        self.current_memory_mb = 0

        # Clear disk cache
        if self.config.cache_directory.exists():
            for cache_file in self.config.cache_directory.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)

        return {
            'memory_items': len(self.memory_cache),
            'memory_size_mb': self.current_memory_mb,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'max_size_mb': self.config.cache_size_mb
        }


class BacktestOptimizer:
    """
    High-performance optimizer for ML4T-Alpha backtesting workflows.
    """

    def __init__(self,
                 config: OptimizationConfig,
                 ml4t_interface: ML4TDataInterface):
        self.config = config
        self.ml4t_interface = ml4t_interface
        self.cache = DataCache(config)

        # Performance metrics
        self.optimization_count = 0
        self.total_optimization_time = 0.0
        self.memory_optimizations = 0

        # System monitoring
        self.process = psutil.Process()

        logger.info(f"Backtest optimizer initialized with {config.level.value} optimization level")

    def optimize_data_loading(self,
                            symbols: List[str],
                            start_date: date,
                            end_date: date,
                            data_types: Optional[List[DataType]] = None) -> pd.DataFrame:
        """
        Optimized data loading for backtesting.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            data_types: Data types to load

        Returns:
            Optimized DataFrame for backtesting
        """
        start_time = time.time()

        if data_types is None:
            data_types = [DataType.PRICE, DataType.VOLUME]

        # Generate cache key
        cache_key = self.cache._make_key("data_loading", symbols, start_date, end_date, data_types)

        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Using cached data for {len(symbols)} symbols")
            return cached_data

        # Optimize loading strategy based on data size
        if len(symbols) >= self.config.bulk_query_threshold:
            data = self._bulk_load_data(symbols, start_date, end_date, data_types)
        else:
            data = self._sequential_load_data(symbols, start_date, end_date, data_types)

        # Apply optimizations
        if self.config.enable_memory_optimization:
            data = self._optimize_dataframe_memory(data)

        if self.config.optimize_pandas:
            data = self._optimize_pandas_operations(data)

        # Cache the result
        self.cache.put(cache_key, data)

        # Update metrics
        self.optimization_count += 1
        self.total_optimization_time += time.time() - start_time

        # Garbage collection if needed
        if self.optimization_count % self.config.gc_frequency == 0:
            self._perform_memory_cleanup()

        return data

    def _bulk_load_data(self,
                       symbols: List[str],
                       start_date: date,
                       end_date: date,
                       data_types: List[DataType]) -> pd.DataFrame:
        """Load data using bulk operations."""
        logger.debug(f"Bulk loading data for {len(symbols)} symbols")

        if self.config.enable_parallel and len(symbols) > 10:
            return self._parallel_load_data(symbols, start_date, end_date, data_types)
        else:
            # Single bulk load
            return self.ml4t_interface.get_price_data(symbols, start_date, end_date)

    def _sequential_load_data(self,
                            symbols: List[str],
                            start_date: date,
                            end_date: date,
                            data_types: List[DataType]) -> pd.DataFrame:
        """Load data sequentially for small symbol sets."""
        logger.debug(f"Sequential loading data for {len(symbols)} symbols")

        all_data = []
        for symbol in symbols:
            symbol_data = self.ml4t_interface.get_price_data([symbol], start_date, end_date)
            if not symbol_data.empty:
                all_data.append(symbol_data)

        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()

    def _parallel_load_data(self,
                          symbols: List[str],
                          start_date: date,
                          end_date: date,
                          data_types: List[DataType]) -> pd.DataFrame:
        """Load data using parallel processing."""
        logger.debug(f"Parallel loading data for {len(symbols)} symbols")

        def load_symbol_chunk(symbol_chunk: List[str]) -> pd.DataFrame:
            return self.ml4t_interface.get_price_data(symbol_chunk, start_date, end_date)

        # Split symbols into chunks
        chunk_size = max(1, len(symbols) // self.config.max_workers)
        symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]

        all_data = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(load_symbol_chunk, chunk) for chunk in symbol_chunks]

            for future in as_completed(futures):
                try:
                    chunk_data = future.result()
                    if not chunk_data.empty:
                        all_data.append(chunk_data)
                except Exception as e:
                    logger.warning(f"Failed to load data chunk: {e}")

        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if df.empty:
            return df

        original_memory = df.memory_usage(deep=True).sum()

        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dtype == 'float64':
                # Check if we can downcast to float32
                if df[col].max() < np.finfo(np.float32).max and df[col].min() > np.finfo(np.float32).min:
                    df[col] = df[col].astype(np.float32)
            elif df[col].dtype == 'int64':
                # Check if we can downcast to smaller int types
                if df[col].max() < np.iinfo(np.int32).max and df[col].min() > np.iinfo(np.int32).min:
                    df[col] = df[col].astype(np.int32)

        # Optimize categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')

        optimized_memory = df.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory

        if memory_reduction > 0.1:  # More than 10% reduction
            self.memory_optimizations += 1
            logger.debug(f"Memory optimization: {memory_reduction:.1%} reduction")

        return df

    def _optimize_pandas_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pandas-specific optimizations."""
        if df.empty:
            return df

        # Sort index for better performance
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # Remove completely null columns
        df = df.dropna(axis=1, how='all')

        return df

    def optimize_backtest_execution(self,
                                  backtest_function: Callable,
                                  data: pd.DataFrame,
                                  **kwargs) -> Any:
        """
        Optimize backtest execution with performance enhancements.

        Args:
            backtest_function: Backtest function to optimize
            data: Input data
            **kwargs: Additional arguments for backtest function

        Returns:
            Optimized backtest results
        """
        start_time = time.time()

        # Pre-process data for optimal performance
        optimized_data = self._preprocess_for_backtest(data)

        # Execute backtest with optimizations
        if self.config.use_numba:
            result = self._execute_with_numba(backtest_function, optimized_data, **kwargs)
        elif self.config.enable_vectorization:
            result = self._execute_vectorized(backtest_function, optimized_data, **kwargs)
        else:
            result = backtest_function(optimized_data, **kwargs)

        # Post-process results
        if self.config.enable_memory_optimization:
            if hasattr(result, 'memory_usage'):
                result = self._optimize_dataframe_memory(result)

        execution_time = time.time() - start_time
        self.total_optimization_time += execution_time

        logger.debug(f"Backtest execution optimized: {execution_time:.2f}s")

        return result

    def _preprocess_for_backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for optimal backtest performance."""
        # Remove any remaining NaN values in critical columns
        if 'close' in data.columns:
            data = data.dropna(subset=['close'], how='all')

        # Ensure consistent data types
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].dtype == 'object':
                data[col] = pd.to_numeric(data[col], errors='coerce')

        return data

    def _execute_with_numba(self,
                          backtest_function: Callable,
                          data: pd.DataFrame,
                          **kwargs) -> Any:
        """Execute backtest with Numba JIT compilation."""
        try:
            import numba
            from numba import jit

            # This is a placeholder - actual implementation would need
            # to wrap the backtest function with Numba decorators
            logger.warning("Numba optimization placeholder - not fully implemented")
            return backtest_function(data, **kwargs)

        except ImportError:
            logger.warning("Numba not available, falling back to standard execution")
            return backtest_function(data, **kwargs)

    def _execute_vectorized(self,
                          backtest_function: Callable,
                          data: pd.DataFrame,
                          **kwargs) -> Any:
        """Execute backtest with vectorization optimizations."""
        # Enable pandas optimizations
        with pd.option_context('mode.chained_assignment', None):
            return backtest_function(data, **kwargs)

    def batch_optimize_backtests(self,
                               backtest_configs: List[Dict[str, Any]],
                               parallel: bool = None) -> List[Any]:
        """
        Optimize execution of multiple backtests.

        Args:
            backtest_configs: List of backtest configurations
            parallel: Use parallel execution (None = auto-decide)

        Returns:
            List of backtest results
        """
        if parallel is None:
            parallel = self.config.enable_parallel and len(backtest_configs) > 1

        if parallel and len(backtest_configs) > 1:
            return self._parallel_backtests(backtest_configs)
        else:
            return self._sequential_backtests(backtest_configs)

    def _parallel_backtests(self, configs: List[Dict[str, Any]]) -> List[Any]:
        """Execute backtests in parallel."""
        results = []

        def run_single_backtest(config: Dict[str, Any]) -> Any:
            try:
                return self._run_backtest_config(config)
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                return None

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(run_single_backtest, config) for config in configs]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    def _sequential_backtests(self, configs: List[Dict[str, Any]]) -> List[Any]:
        """Execute backtests sequentially."""
        return [self._run_backtest_config(config) for config in configs]

    def _run_backtest_config(self, config: Dict[str, Any]) -> Any:
        """Run a single backtest configuration."""
        # Extract configuration parameters
        backtest_function = config['function']
        data = config.get('data')
        kwargs = config.get('kwargs', {})

        # Load data if not provided
        if data is None:
            symbols = config['symbols']
            start_date = config['start_date']
            end_date = config['end_date']
            data = self.optimize_data_loading(symbols, start_date, end_date)

        # Execute optimized backtest
        return self.optimize_backtest_execution(backtest_function, data, **kwargs)

    def _perform_memory_cleanup(self) -> None:
        """Perform memory cleanup and garbage collection."""
        # Force garbage collection
        collected = gc.collect()

        # Monitor memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        logger.debug(f"Memory cleanup: {collected} objects collected, {memory_mb:.1f}MB used")

        # Clear caches if memory usage is high
        if memory_mb > self.config.memory_limit_mb:
            logger.warning("High memory usage detected, clearing caches")
            self.cache.clear()

    def warm_up_optimizer(self,
                        sample_symbols: List[str],
                        lookback_days: int = 30) -> None:
        """Warm up the optimizer with sample data."""
        logger.info("Warming up backtest optimizer")

        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        # Load sample data to warm caches
        try:
            self.optimize_data_loading(sample_symbols[:5], start_date, end_date)
            logger.info("Optimizer warm-up completed")
        except Exception as e:
            logger.warning(f"Optimizer warm-up failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get optimizer performance statistics."""
        avg_optimization_time = (
            self.total_optimization_time / max(self.optimization_count, 1)
        )

        # System resource usage
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()

        stats = {
            'optimization_count': self.optimization_count,
            'total_optimization_time': self.total_optimization_time,
            'avg_optimization_time': avg_optimization_time,
            'memory_optimizations': self.memory_optimizations,
            'system_memory_mb': memory_info.rss / (1024 * 1024),
            'system_cpu_percent': cpu_percent,
            'optimization_level': self.config.level.value
        }

        # Add cache stats
        cache_stats = self.cache.get_stats()
        stats.update({f'cache_{k}': v for k, v in cache_stats.items()})

        return stats

    def clear_all_caches(self) -> None:
        """Clear all optimizer caches."""
        self.cache.clear()
        logger.info("All optimizer caches cleared")


def create_backtest_optimizer(
    ml4t_interface: ML4TDataInterface,
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
    cache_size_mb: int = 1024,
    enable_parallel: bool = True,
    **kwargs
) -> BacktestOptimizer:
    """
    Factory function to create backtest optimizer.

    Args:
        ml4t_interface: ML4T data interface
        optimization_level: Optimization level
        cache_size_mb: Cache size in MB
        enable_parallel: Enable parallel processing
        **kwargs: Additional configuration parameters

    Returns:
        Configured BacktestOptimizer instance
    """
    config = OptimizationConfig(
        level=optimization_level,
        cache_size_mb=cache_size_mb,
        enable_parallel=enable_parallel,
        **kwargs
    )

    return BacktestOptimizer(config, ml4t_interface)