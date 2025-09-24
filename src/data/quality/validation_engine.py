"""
Data Quality Validation Engine - Core Framework.

This module provides the core extensible validation framework with plugin architecture
for the ML4T data quality system. Designed for high-performance validation with
<10ms latency requirements.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union, 
    Protocol, TypeVar, Generic
)

from ..core.temporal import TemporalValue, DataType, TemporalStore
from ..models.taiwan_market import TaiwanMarketData, TaiwanSettlement
from .validators import QualityIssue, SeverityLevel, QualityCheckType

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ValidationResult(Enum):
    """Validation result status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


class ValidationPriority(Enum):
    """Validation priority levels for performance optimization."""
    CRITICAL = 1    # Must run first, blocks on failure
    HIGH = 2        # Important validations
    MEDIUM = 3      # Standard validations  
    LOW = 4         # Nice-to-have validations
    BACKGROUND = 5  # Can run asynchronously


@dataclass
class ValidationContext:
    """Context information for validation execution."""
    symbol: str
    data_date: date
    as_of_date: date
    data_type: DataType
    metadata: Dict[str, Any] = field(default_factory=dict)
    historical_data: Optional[List[TemporalValue]] = None
    market_session: Optional[str] = None
    trading_calendar: Optional[Dict[date, Any]] = None
    
    def get_metadata(self, key: str, default=None) -> Any:
        """Get metadata value with default."""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value


@dataclass
class ValidationOutput:
    """Result of a validation operation."""
    validator_name: str
    validation_id: str
    result: ValidationResult
    issues: List[QualityIssue] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """Check if validation passed."""
        return self.result == ValidationResult.PASS
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any(issue.severity == SeverityLevel.CRITICAL for issue in self.issues)


class ValidationPlugin(ABC):
    """Abstract base class for validation plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get plugin name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Get plugin version."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> ValidationPriority:
        """Get validation priority."""
        pass
    
    @property
    @abstractmethod
    def supported_data_types(self) -> Set[DataType]:
        """Get supported data types."""
        pass
    
    @abstractmethod
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Perform validation on a temporal value."""
        pass
    
    @abstractmethod
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if this plugin can validate the given value."""
        pass
    
    def initialize(self, config: Dict[str, Any] = None) -> None:
        """Initialize plugin with configuration."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass


class ValidationRegistry:
    """Registry for validation plugins with dependency management."""
    
    def __init__(self):
        self._plugins: Dict[str, ValidationPlugin] = {}
        self._plugins_by_priority: Dict[ValidationPriority, List[ValidationPlugin]] = defaultdict(list)
        self._plugins_by_data_type: Dict[DataType, List[ValidationPlugin]] = defaultdict(list)
        self._plugin_dependencies: Dict[str, Set[str]] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        
    def register_plugin(self, plugin: ValidationPlugin, 
                       dependencies: Set[str] = None,
                       config: Dict[str, Any] = None) -> None:
        """Register a validation plugin."""
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin {plugin.name} already registered")
        
        # Initialize plugin
        plugin.initialize(config or {})
        
        # Store plugin
        self._plugins[plugin.name] = plugin
        self._plugins_by_priority[plugin.priority].append(plugin)
        self._plugin_dependencies[plugin.name] = dependencies or set()
        self._plugin_configs[plugin.name] = config or {}
        
        # Index by supported data types
        for data_type in plugin.supported_data_types:
            self._plugins_by_data_type[data_type].append(plugin)
        
        logger.info(f"Registered validation plugin: {plugin.name} v{plugin.version} "
                   f"(priority: {plugin.priority.name})")
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a validation plugin."""
        if plugin_name not in self._plugins:
            return
        
        plugin = self._plugins[plugin_name]
        
        # Cleanup plugin
        plugin.cleanup()
        
        # Remove from registries
        del self._plugins[plugin_name]
        self._plugins_by_priority[plugin.priority].remove(plugin)
        del self._plugin_dependencies[plugin_name]
        del self._plugin_configs[plugin_name]
        
        # Remove from data type index
        for data_type in plugin.supported_data_types:
            if plugin in self._plugins_by_data_type[data_type]:
                self._plugins_by_data_type[data_type].remove(plugin)
        
        logger.info(f"Unregistered validation plugin: {plugin_name}")
    
    def get_plugins_for_data_type(self, data_type: DataType) -> List[ValidationPlugin]:
        """Get all plugins that support a specific data type."""
        return self._plugins_by_data_type.get(data_type, [])
    
    def get_plugins_by_priority(self, priority: ValidationPriority) -> List[ValidationPlugin]:
        """Get all plugins with a specific priority."""
        return self._plugins_by_priority.get(priority, [])
    
    def get_plugin(self, name: str) -> Optional[ValidationPlugin]:
        """Get a specific plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())
    
    def check_dependencies(self, plugin_name: str) -> bool:
        """Check if all dependencies for a plugin are satisfied."""
        deps = self._plugin_dependencies.get(plugin_name, set())
        return all(dep in self._plugins for dep in deps)


class ValidationEngine:
    """High-performance validation engine with plugin architecture for <10ms real-time validation."""
    
    def __init__(self, 
                 registry: ValidationRegistry,
                 temporal_store: TemporalStore,
                 max_workers: int = 8,  # Increased for better parallelism
                 timeout_ms: int = 5000,  # Reduced to 5s for faster fail-fast
                 enable_async: bool = True,
                 enable_fast_path: bool = True,  # New: Enable fast-path optimizations
                 max_cache_size: int = 50000):   # New: Larger cache for better hit rates
        self.registry = registry
        self.temporal_store = temporal_store
        self.max_workers = max_workers
        self.timeout_ms = timeout_ms
        self.enable_async = enable_async
        self.enable_fast_path = enable_fast_path
        self.max_cache_size = max_cache_size
        
        # Performance tracking
        self.validation_count = 0
        self.total_execution_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.fast_path_hits = 0  # New: Track fast-path usage
        
        # Result cache for performance optimization
        self._result_cache: Dict[str, Tuple[ValidationOutput, datetime]] = {}
        self._cache_ttl_seconds = 180  # Reduced to 3 minutes for fresher data
        
        # Execution metrics by plugin
        self._plugin_metrics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "avg_time": 0.0, "fast_path_count": 0}
        )
        
        # Pre-computed validation paths for common patterns
        self._fast_path_cache: Dict[Tuple[DataType, str], List[ValidationPlugin]] = {}
        
        # Performance thresholds for alerting
        self.latency_threshold_ms = 10.0  # Alert if validation takes >10ms
        self.critical_latency_threshold_ms = 50.0  # Critical alert threshold
        
        logger.info(f"Validation engine initialized with {len(registry.list_plugins())} plugins, "
                   f"fast-path: {enable_fast_path}, max_cache: {max_cache_size}")
    
    def _make_cache_key(self, value: TemporalValue, context: ValidationContext, 
                       plugin_name: str) -> str:
        """Create cache key for validation result."""
        return f"{plugin_name}:{value.symbol}:{value.data_type.value}:{value.value_date}:{value.as_of_date}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[ValidationOutput]:
        """Get cached validation result."""
        if cache_key in self._result_cache:
            result, timestamp = self._result_cache[cache_key]
            
            # Check TTL
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl_seconds:
                self.cache_hits += 1
                return result
            else:
                # Expired
                del self._result_cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    def _cache_result(self, cache_key: str, result: ValidationOutput) -> None:
        """Cache validation result."""
        self._result_cache[cache_key] = (result, datetime.utcnow())
        
        # Simple LRU eviction if cache gets too large
        if len(self._result_cache) > self.max_cache_size:
            # Remove oldest 20% of entries for better performance
            sorted_items = sorted(
                self._result_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            to_remove = len(sorted_items) // 5  # Remove 20%
            for key, _ in sorted_items[:to_remove]:
                del self._result_cache[key]
            logger.debug(f"Evicted {to_remove} cache entries, current size: {len(self._result_cache)}")
    
    def _get_fast_path_plugins(self, data_type: DataType, symbol: str) -> Optional[List[ValidationPlugin]]:
        """Get pre-computed fast validation path for common patterns."""
        if not self.enable_fast_path:
            return None
        
        # Create cache key for this pattern
        cache_key = (data_type, symbol[:4] if symbol else "")  # Use first 4 chars for pattern matching
        
        if cache_key in self._fast_path_cache:
            self.fast_path_hits += 1
            return self._fast_path_cache[cache_key]
        
        # Compute fast path for this pattern
        all_plugins = self.registry.get_plugins_for_data_type(data_type)
        
        # Sort by priority and pre-filter for common patterns
        fast_plugins = []
        for plugin in all_plugins:
            if plugin.priority in [ValidationPriority.CRITICAL, ValidationPriority.HIGH]:
                fast_plugins.append(plugin)
        
        # Cache the result
        self._fast_path_cache[cache_key] = fast_plugins
        
        return fast_plugins
    
    def _check_performance_thresholds(self, execution_time_ms: float, 
                                    symbol: str, data_type: DataType) -> None:
        """Check if validation exceeded performance thresholds."""
        if execution_time_ms > self.critical_latency_threshold_ms:
            logger.error(f"CRITICAL: Validation latency {execution_time_ms:.2f}ms exceeded "
                        f"threshold {self.critical_latency_threshold_ms}ms for {symbol} {data_type.value}")
        elif execution_time_ms > self.latency_threshold_ms:
            logger.warning(f"Validation latency {execution_time_ms:.2f}ms exceeded "
                          f"threshold {self.latency_threshold_ms}ms for {symbol} {data_type.value}")
    
    async def validate_value(self, value: TemporalValue, 
                           context: Optional[ValidationContext] = None,
                           plugin_names: Optional[List[str]] = None,
                           use_fast_path: bool = True) -> List[ValidationOutput]:
        """Validate a single temporal value with <10ms target latency."""
        start_time = time.perf_counter()
        
        # Create default context if not provided
        if context is None:
            context = ValidationContext(
                symbol=value.symbol or "",
                data_date=value.value_date,
                as_of_date=value.as_of_date,
                data_type=value.data_type
            )
        
        # Fast path optimization: Try pre-computed validation paths first
        applicable_plugins = None
        if use_fast_path and not plugin_names:
            applicable_plugins = self._get_fast_path_plugins(value.data_type, value.symbol or "")
        
        # Get applicable plugins if fast path didn't work
        if applicable_plugins is None:
            if plugin_names:
                plugins = [self.registry.get_plugin(name) for name in plugin_names]
                plugins = [p for p in plugins if p is not None]
            else:
                plugins = self.registry.get_plugins_for_data_type(value.data_type)
            
            # Filter plugins that can validate this value
            applicable_plugins = [
                plugin for plugin in plugins 
                if plugin.can_validate(value, context)
            ]
            
            # Sort by priority
            applicable_plugins.sort(key=lambda p: p.priority.value)
        
        # Execute validations with optimized strategy
        results = []
        
        # For real-time validation, prefer parallel execution when beneficial
        if (self.enable_async and len(applicable_plugins) > 2 and 
            not any(p.priority == ValidationPriority.CRITICAL for p in applicable_plugins)):
            # Parallel execution for non-critical validations
            results = await self._validate_parallel(value, context, applicable_plugins)
        else:
            # Sequential execution for critical validations or small sets
            results = await self._validate_sequential(value, context, applicable_plugins)
        
        # Update metrics and check performance
        execution_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self.validation_count += 1
        self.total_execution_time += execution_time
        
        # Check performance thresholds
        self._check_performance_thresholds(execution_time, value.symbol or "", value.data_type)
        
        logger.debug(f"Validated {value.symbol} {value.data_type.value} using "
                    f"{len(applicable_plugins)} plugins in {execution_time:.2f}ms "
                    f"(fast_path: {use_fast_path and applicable_plugins is not None})")
        
        return results
    
    async def _validate_sequential(self, value: TemporalValue, 
                                 context: ValidationContext,
                                 plugins: List[ValidationPlugin]) -> List[ValidationOutput]:
        """Execute validations sequentially."""
        results = []
        
        for plugin in plugins:
            # Check cache first
            cache_key = self._make_cache_key(value, context, plugin.name)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                results.append(cached_result)
                continue
            
            # Execute validation
            start_time = time.perf_counter()
            
            try:
                # Apply timeout
                result = await asyncio.wait_for(
                    plugin.validate(value, context),
                    timeout=self.timeout_ms / 1000.0
                )
                
                # Update execution time
                execution_time = (time.perf_counter() - start_time) * 1000
                result.execution_time_ms = execution_time
                
                # Update plugin metrics
                metrics = self._plugin_metrics[plugin.name]
                metrics["count"] += 1
                metrics["total_time"] += execution_time
                metrics["avg_time"] = metrics["total_time"] / metrics["count"]
                
                # Cache result
                self._cache_result(cache_key, result)
                
                results.append(result)
                
                # Stop on critical failure if configured
                if result.has_critical_issues and plugin.priority == ValidationPriority.CRITICAL:
                    logger.warning(f"Critical validation failure in {plugin.name}, stopping validation chain")
                    break
                    
            except asyncio.TimeoutError:
                logger.error(f"Validation timeout for plugin {plugin.name}")
                
                timeout_result = ValidationOutput(
                    validator_name=plugin.name,
                    validation_id=f"{plugin.name}_{int(time.time())}",
                    result=ValidationResult.FAIL,
                    issues=[QualityIssue(
                        check_type=QualityCheckType.VALIDITY,
                        severity=SeverityLevel.ERROR,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Validation timeout in {plugin.name}"
                    )],
                    execution_time_ms=self.timeout_ms
                )
                results.append(timeout_result)
                
            except Exception as e:
                logger.error(f"Validation error in plugin {plugin.name}: {e}")
                
                error_result = ValidationOutput(
                    validator_name=plugin.name,
                    validation_id=f"{plugin.name}_{int(time.time())}",
                    result=ValidationResult.FAIL,
                    issues=[QualityIssue(
                        check_type=QualityCheckType.VALIDITY,
                        severity=SeverityLevel.CRITICAL,
                        symbol=value.symbol or "",
                        data_type=value.data_type,
                        data_date=value.value_date,
                        issue_date=datetime.utcnow(),
                        description=f"Validation error in {plugin.name}: {str(e)}"
                    )],
                    execution_time_ms=(time.perf_counter() - start_time) * 1000
                )
                results.append(error_result)
        
        return results
    
    async def _validate_parallel(self, value: TemporalValue,
                                context: ValidationContext,
                                plugins: List[ValidationPlugin]) -> List[ValidationOutput]:
        """Execute validations in parallel for better performance."""
        # Separate critical plugins (must run first)
        critical_plugins = [p for p in plugins if p.priority == ValidationPriority.CRITICAL]
        other_plugins = [p for p in plugins if p.priority != ValidationPriority.CRITICAL]
        
        results = []
        
        # Run critical plugins first (sequential)
        if critical_plugins:
            critical_results = await self._validate_sequential(value, context, critical_plugins)
            results.extend(critical_results)
            
            # Check for critical failures
            if any(r.has_critical_issues for r in critical_results):
                logger.warning("Critical validation failure, skipping remaining validations")
                return results
        
        # Run other plugins in parallel
        if other_plugins:
            async def run_plugin(plugin: ValidationPlugin) -> ValidationOutput:
                cache_key = self._make_cache_key(value, context, plugin.name)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    return cached_result
                
                start_time = time.perf_counter()
                
                try:
                    result = await asyncio.wait_for(
                        plugin.validate(value, context),
                        timeout=self.timeout_ms / 1000.0
                    )
                    
                    execution_time = (time.perf_counter() - start_time) * 1000
                    result.execution_time_ms = execution_time
                    
                    # Update plugin metrics
                    metrics = self._plugin_metrics[plugin.name]
                    metrics["count"] += 1
                    metrics["total_time"] += execution_time
                    metrics["avg_time"] = metrics["total_time"] / metrics["count"]
                    
                    self._cache_result(cache_key, result)
                    return result
                    
                except Exception as e:
                    logger.error(f"Parallel validation error in {plugin.name}: {e}")
                    return ValidationOutput(
                        validator_name=plugin.name,
                        validation_id=f"{plugin.name}_{int(time.time())}",
                        result=ValidationResult.FAIL,
                        issues=[QualityIssue(
                            check_type=QualityCheckType.VALIDITY,
                            severity=SeverityLevel.ERROR,
                            symbol=value.symbol or "",
                            data_type=value.data_type,
                            data_date=value.value_date,
                            issue_date=datetime.utcnow(),
                            description=f"Parallel validation error: {str(e)}"
                        )],
                        execution_time_ms=(time.perf_counter() - start_time) * 1000
                    )
            
            # Execute in parallel
            tasks = [run_plugin(plugin) for plugin in other_plugins]
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in parallel_results:
                if isinstance(result, ValidationOutput):
                    results.append(result)
                else:
                    logger.error(f"Parallel validation task failed: {result}")
        
        return results
    
    async def validate_batch(self, values: List[TemporalValue],
                           contexts: Optional[List[ValidationContext]] = None) -> Dict[TemporalValue, List[ValidationOutput]]:
        """Validate a batch of temporal values efficiently."""
        if not values:
            return {}
        
        start_time = time.perf_counter()
        
        # Create contexts if not provided
        if contexts is None:
            contexts = [
                ValidationContext(
                    symbol=value.symbol or "",
                    data_date=value.value_date,
                    as_of_date=value.as_of_date,
                    data_type=value.data_type
                )
                for value in values
            ]
        
        # Validate each value
        results = {}
        
        # Use ThreadPoolExecutor for true parallelism across multiple values
        if len(values) > 1 and self.max_workers > 1:
            async def validate_single(value: TemporalValue, context: ValidationContext):
                return value, await self.validate_value(value, context)
            
            tasks = [validate_single(value, context) for value, context in zip(values, contexts)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, tuple):
                    value, validation_outputs = result
                    results[value] = validation_outputs
                else:
                    logger.error(f"Batch validation task failed: {result}")
        else:
            # Sequential batch processing
            for value, context in zip(values, contexts):
                validation_outputs = await self.validate_value(value, context)
                results[value] = validation_outputs
        
        execution_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Batch validated {len(values)} values in {execution_time:.2f}ms "
                   f"({execution_time/len(values):.2f}ms avg per value)")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get validation engine performance statistics."""
        avg_execution_time = (
            self.total_execution_time / max(self.validation_count, 1)
        )
        
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        fast_path_rate = self.fast_path_hits / max(self.validation_count, 1)
        
        # Calculate SLA compliance
        latency_sla_compliance = (avg_execution_time <= self.latency_threshold_ms)
        
        return {
            "validation_count": self.validation_count,
            "avg_execution_time_ms": round(avg_execution_time, 3),
            "total_execution_time_ms": round(self.total_execution_time, 3),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "fast_path_hit_rate": round(fast_path_rate, 3),
            "cache_size": len(self._result_cache),
            "fast_path_cache_size": len(self._fast_path_cache),
            "registered_plugins": len(self.registry.list_plugins()),
            "latency_threshold_ms": self.latency_threshold_ms,
            "latency_sla_compliance": latency_sla_compliance,
            "performance_grade": "A" if latency_sla_compliance else "B" if avg_execution_time <= 20 else "C",
            "plugin_metrics": dict(self._plugin_metrics)
        }
    
    def clear_cache(self) -> None:
        """Clear validation result cache."""
        self._result_cache.clear()
        logger.info("Validation cache cleared")
    
    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set cache TTL in seconds."""
        self._cache_ttl_seconds = ttl_seconds
        logger.info(f"Cache TTL set to {ttl_seconds} seconds")


# Utility functions for creating validation contexts

def create_validation_context(temporal_value: TemporalValue,
                            historical_data: Optional[List[TemporalValue]] = None,
                            market_metadata: Optional[Dict[str, Any]] = None) -> ValidationContext:
    """Create a validation context for a temporal value."""
    context = ValidationContext(
        symbol=temporal_value.symbol or "",
        data_date=temporal_value.value_date,
        as_of_date=temporal_value.as_of_date,
        data_type=temporal_value.data_type,
        historical_data=historical_data
    )
    
    if market_metadata:
        context.metadata.update(market_metadata)
    
    return context


def create_batch_contexts(temporal_values: List[TemporalValue],
                        historical_data_map: Optional[Dict[str, List[TemporalValue]]] = None) -> List[ValidationContext]:
    """Create validation contexts for a batch of temporal values."""
    contexts = []
    
    for value in temporal_values:
        historical_data = None
        if historical_data_map and value.symbol:
            historical_data = historical_data_map.get(value.symbol)
        
        context = create_validation_context(value, historical_data)
        contexts.append(context)
    
    return contexts