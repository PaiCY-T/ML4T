"""
Point-in-Time Integration for Data Quality Validation Framework.

This module integrates the data quality validation system with the point-in-time
data management system, ensuring validation occurs without look-ahead bias and
maintains <10ms latency for real-time processing.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from ..core.temporal import TemporalValue, TemporalStore, DataType
from ..pipeline.pit_engine import PointInTimeEngine, PITQuery, PITResult, QueryMode, BiasCheckLevel
from ..models.taiwan_market import TaiwanMarketData, TaiwanTradingCalendar

from .validation_engine import (
    ValidationEngine, ValidationRegistry, ValidationContext, ValidationOutput,
    ValidationResult, ValidationPriority, create_validation_context
)
from .taiwan_validators import create_taiwan_validators
from .monitor import QualityMonitor, QualityMetrics, create_taiwan_market_monitor
from .alerting import AlertingSystem
from .metrics import QualityMetrics as QualityMetricsCalculator
from .validators import QualityIssue, SeverityLevel, QualityCheckType

logger = logging.getLogger(__name__)


class PITValidationMode(Enum):
    """Point-in-time validation modes."""
    STRICT = "strict"           # Full bias checking and validation
    PERFORMANCE = "performance" # Optimized for speed, minimal bias checking
    BATCH = "batch"            # Bulk validation optimization
    REALTIME = "realtime"      # Real-time streaming validation


@dataclass
class PITValidationConfig:
    """Configuration for point-in-time validation."""
    mode: PITValidationMode = PITValidationMode.STRICT
    max_latency_ms: float = 10.0
    bias_check_level: BiasCheckLevel = BiasCheckLevel.STRICT
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    taiwan_market_rules: bool = True
    parallel_validation: bool = True
    max_workers: int = 4
    quality_score_threshold: float = 95.0
    
    # Taiwan market specific settings
    price_limit_tolerance: float = 0.01  # 1% buffer
    volume_spike_threshold: float = 5.0  # 5x average
    fundamental_lag_strict: bool = True  # Strict 60-day lag enforcement


@dataclass
class PITValidationResult:
    """Result of point-in-time validation."""
    query: PITQuery
    validation_results: Dict[str, Dict[DataType, List[ValidationOutput]]]
    quality_metrics: Dict[str, QualityMetrics]
    execution_time_ms: float
    bias_violations: List[str] = field(default_factory=list)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score across all validated data."""
        if not self.quality_metrics:
            return 0.0
        
        scores = [m.quality_score for m in self.quality_metrics.values()]
        return sum(scores) / len(scores)
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues were found."""
        for symbol_results in self.validation_results.values():
            for datatype_results in symbol_results.values():
                for result in datatype_results:
                    if result.has_critical_issues:
                        return True
        return False
    
    def get_symbol_quality(self, symbol: str) -> Optional[float]:
        """Get quality score for specific symbol."""
        metrics = self.quality_metrics.get(symbol)
        return metrics.quality_score if metrics else None


class PITValidationOrchestrator:
    """Orchestrates point-in-time validation with bias checking and performance optimization."""
    
    def __init__(self,
                 pit_engine: PointInTimeEngine,
                 temporal_store: TemporalStore,
                 config: Optional[PITValidationConfig] = None):
        self.pit_engine = pit_engine
        self.temporal_store = temporal_store
        self.config = config or PITValidationConfig()
        
        # Initialize validation components
        self._setup_validation_engine()
        self._setup_monitoring()
        
        # Performance tracking
        self.validation_count = 0
        self.total_execution_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Cache for validation contexts
        self._context_cache: Dict[str, ValidationContext] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        logger.info(f"PIT validation orchestrator initialized with mode: {self.config.mode.value}")
    
    def _setup_validation_engine(self) -> None:
        """Initialize validation engine with Taiwan market validators."""
        # Create validation registry
        self.registry = ValidationRegistry()
        
        # Register Taiwan market validators
        if self.config.taiwan_market_rules:
            taiwan_validators = create_taiwan_validators()
            for validator in taiwan_validators:
                self.registry.register_plugin(validator)
        
        # Create validation engine
        self.validation_engine = ValidationEngine(
            registry=self.registry,
            temporal_store=self.temporal_store,
            max_workers=self.config.max_workers,
            timeout_ms=int(self.config.max_latency_ms * 1000),
            enable_async=self.config.parallel_validation
        )
        
        # Set cache TTL
        if self.config.enable_caching:
            self.validation_engine.set_cache_ttl(self.config.cache_ttl_seconds)
    
    def _setup_monitoring(self) -> None:
        """Initialize quality monitoring."""
        self.monitor = create_taiwan_market_monitor(self.validation_engine)
        self.monitor.start_monitoring()
        
        # Setup alerting if available
        try:
            from .alerting import create_taiwan_market_alerting
            self.alerting = create_taiwan_market_alerting()
        except ImportError:
            logger.warning("Alerting system not available")
            self.alerting = None
    
    async def validate_point_in_time(self, 
                                   symbols: List[str],
                                   as_of_date: date,
                                   data_types: List[DataType],
                                   include_historical: bool = True) -> PITValidationResult:
        """Validate data at a specific point in time with bias checking."""
        start_time = time.perf_counter()
        
        # Create PIT query
        pit_query = PITQuery(
            symbols=symbols,
            as_of_date=as_of_date,
            data_types=data_types,
            mode=self._get_pit_query_mode(),
            bias_check=self.config.bias_check_level,
            include_metadata=True
        )
        
        try:
            # Execute point-in-time query
            pit_result = self.pit_engine.execute_query(pit_query)
            
            # Validate retrieved data
            validation_results = await self._validate_pit_data(pit_result, include_historical)
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(validation_results)
            
            # Update performance stats
            execution_time = (time.perf_counter() - start_time) * 1000
            self.validation_count += 1
            self.total_execution_time += execution_time
            
            # Create result
            result = PITValidationResult(
                query=pit_query,
                validation_results=validation_results,
                quality_metrics=quality_metrics,
                execution_time_ms=execution_time,
                bias_violations=pit_result.bias_violations,
                performance_stats={
                    'pit_execution_time_ms': pit_result.execution_time_ms,
                    'validation_execution_time_ms': execution_time - pit_result.execution_time_ms,
                    'cache_hit_rate': pit_result.cache_hit_rate,
                    'total_values_validated': sum(
                        len(dt_results) for symbol_results in validation_results.values()
                        for dt_results in symbol_results.values()
                    )
                }
            )
            
            # Check quality thresholds
            if result.overall_quality_score < self.config.quality_score_threshold:
                logger.warning(f"Quality score {result.overall_quality_score:.1f} below threshold "
                             f"{self.config.quality_score_threshold}")
                
                if self.alerting:
                    await self._trigger_quality_alert(result)
            
            # Log performance
            logger.debug(f"PIT validation completed for {len(symbols)} symbols, "
                        f"{len(data_types)} data types in {execution_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"PIT validation failed: {e}")
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Return error result
            return PITValidationResult(
                query=pit_query,
                validation_results={},
                quality_metrics={},
                execution_time_ms=execution_time,
                bias_violations=[f"Validation error: {str(e)}"]
            )
    
    async def validate_streaming_data(self,
                                    value: TemporalValue,
                                    validate_against_pit: bool = True) -> ValidationOutput:
        """Validate streaming data with optional point-in-time consistency checking."""
        start_time = time.perf_counter()
        
        try:
            # Create validation context
            context = await self._create_validation_context(value, validate_against_pit)
            
            # Perform validation
            validation_results = await self.validation_engine.validate_value(value, context)
            
            # Monitor validation
            if self.monitor:
                quality_metrics = await self.monitor.monitor_validation(value, context)
                
                # Check latency constraint
                latency_ms = (time.perf_counter() - start_time) * 1000
                if latency_ms > self.config.max_latency_ms:
                    logger.warning(f"Validation latency {latency_ms:.2f}ms exceeds limit "
                                 f"{self.config.max_latency_ms}ms")
            
            # Return primary validation result
            return validation_results[0] if validation_results else ValidationOutput(
                validator_name="pit_orchestrator",
                validation_id=f"pit_{int(time.time())}",
                result=ValidationResult.SKIP,
                metadata={"reason": "No applicable validators"}
            )
            
        except Exception as e:
            logger.error(f"Streaming validation failed: {e}")
            return ValidationOutput(
                validator_name="pit_orchestrator",
                validation_id=f"pit_error_{int(time.time())}",
                result=ValidationResult.FAIL,
                issues=[QualityIssue(
                    check_type=QualityCheckType.VALIDITY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Streaming validation error: {str(e)}"
                )],
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def validate_bulk_data(self,
                               values: List[TemporalValue],
                               batch_size: int = 100) -> Dict[TemporalValue, ValidationOutput]:
        """Validate bulk data with batching for optimal performance."""
        results = {}
        
        # Process in batches for memory efficiency
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            
            # Create contexts for batch
            contexts = []
            for value in batch:
                context = await self._create_validation_context(value, validate_against_pit=False)
                contexts.append(context)
            
            # Validate batch
            batch_results = await self.validation_engine.validate_batch(batch, contexts)
            
            # Extract primary results
            for value, validation_outputs in batch_results.items():
                results[value] = validation_outputs[0] if validation_outputs else ValidationOutput(
                    validator_name="pit_orchestrator",
                    validation_id=f"bulk_{int(time.time())}",
                    result=ValidationResult.SKIP
                )
        
        logger.info(f"Bulk validation completed for {len(values)} values")
        return results
    
    async def _validate_pit_data(self, 
                               pit_result: PITResult,
                               include_historical: bool) -> Dict[str, Dict[DataType, List[ValidationOutput]]]:
        """Validate data retrieved from point-in-time query."""
        validation_results = defaultdict(lambda: defaultdict(list))
        
        # Process each symbol and data type
        for symbol, symbol_data in pit_result.data.items():
            for data_type, temporal_value in symbol_data.items():
                # Create validation context
                context = await self._create_validation_context(
                    temporal_value, 
                    validate_against_pit=False,
                    include_historical=include_historical
                )
                
                # Validate the temporal value
                outputs = await self.validation_engine.validate_value(temporal_value, context)
                validation_results[symbol][data_type] = outputs
        
        return dict(validation_results)
    
    async def _create_validation_context(self,
                                       value: TemporalValue,
                                       validate_against_pit: bool = True,
                                       include_historical: bool = True) -> ValidationContext:
        """Create validation context with historical data and market metadata."""
        # Check cache first
        cache_key = f"{value.symbol}:{value.data_type.value}:{value.value_date}:{value.as_of_date}"
        
        if self.config.enable_caching and cache_key in self._context_cache:
            cached_time = self._cache_timestamps.get(cache_key)
            if cached_time and (datetime.utcnow() - cached_time).seconds < self.config.cache_ttl_seconds:
                self.cache_hits += 1
                return self._context_cache[cache_key]
        
        self.cache_misses += 1
        
        # Create new context
        context = ValidationContext(
            symbol=value.symbol or "",
            data_date=value.value_date,
            as_of_date=value.as_of_date,
            data_type=value.data_type
        )
        
        # Add historical data if requested
        if include_historical and value.symbol:
            historical_data = await self._get_historical_data(
                value.symbol, value.value_date, value.data_type
            )
            context.historical_data = historical_data
        
        # Add Taiwan market metadata
        if self.config.taiwan_market_rules:
            taiwan_metadata = await self._get_taiwan_market_metadata(value)
            context.metadata.update(taiwan_metadata)
        
        # Cache the context
        if self.config.enable_caching:
            self._context_cache[cache_key] = context
            self._cache_timestamps[cache_key] = datetime.utcnow()
        
        return context
    
    async def _get_historical_data(self,
                                 symbol: str,
                                 current_date: date,
                                 data_type: DataType,
                                 lookback_days: int = 30) -> List[TemporalValue]:
        """Get historical data for validation context."""
        try:
            start_date = current_date - timedelta(days=lookback_days)
            historical_values = self.temporal_store.get_range(
                symbol, start_date, current_date, data_type
            )
            return historical_values[:50]  # Limit for performance
        except Exception as e:
            logger.warning(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    async def _get_taiwan_market_metadata(self, value: TemporalValue) -> Dict[str, Any]:
        """Get Taiwan market specific metadata for validation."""
        metadata = {}
        
        try:
            # Get trading calendar info
            if value.value_date:
                # Check if trading day
                is_trading_day = value.value_date.weekday() < 5  # Simplified
                metadata['is_trading_day'] = is_trading_day
                
                # Add market session info
                metadata['market_session'] = 'regular'  # Could be enhanced
            
            # Get stock type information (would come from market data service)
            if value.symbol:
                # This would typically query a reference data service
                metadata['stock_type'] = 'STOCK'  # Default
                
                # Check if ETF, warrant, etc.
                if 'ETF' in value.symbol:
                    metadata['stock_type'] = 'ETF'
                elif 'W' in value.symbol:
                    metadata['stock_type'] = 'WARRANT'
            
            # Add settlement information
            if value.data_type in [DataType.PRICE, DataType.VOLUME]:
                settlement_date = value.value_date + timedelta(days=2)  # T+2
                metadata['settlement_info'] = {
                    'settlement_date': settlement_date.isoformat(),
                    'settlement_rule': 'T+2'
                }
            
        except Exception as e:
            logger.warning(f"Failed to get Taiwan market metadata: {e}")
        
        return metadata
    
    async def _calculate_quality_metrics(self,
                                       validation_results: Dict[str, Dict[DataType, List[ValidationOutput]]]) -> Dict[str, QualityMetrics]:
        """Calculate quality metrics from validation results."""
        quality_metrics = {}
        
        for symbol, symbol_results in validation_results.items():
            # Aggregate results for symbol
            all_outputs = []
            for datatype_outputs in symbol_results.values():
                all_outputs.extend(datatype_outputs)
            
            if not all_outputs:
                continue
            
            # Calculate metrics
            validation_count = len(all_outputs)
            passed_count = sum(1 for output in all_outputs if output.result == ValidationResult.PASS)
            warning_count = sum(1 for output in all_outputs if output.result == ValidationResult.WARNING)
            
            error_count = sum(
                len([issue for issue in output.issues if issue.severity == SeverityLevel.ERROR])
                for output in all_outputs
            )
            critical_count = sum(
                len([issue for issue in output.issues if issue.severity == SeverityLevel.CRITICAL])
                for output in all_outputs
            )
            
            avg_latency = sum(output.execution_time_ms for output in all_outputs) / validation_count
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(all_outputs)
            
            # Create metrics object
            from .monitor import QualityMetrics
            metrics = QualityMetrics(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                data_type=list(symbol_results.keys())[0],  # Primary data type
                quality_score=quality_score,
                validation_count=validation_count,
                passed_validations=passed_count,
                warning_count=warning_count,
                error_count=error_count,
                critical_count=critical_count,
                validation_latency_ms=avg_latency
            )
            
            quality_metrics[symbol] = metrics
        
        return quality_metrics
    
    def _calculate_quality_score(self, validation_outputs: List[ValidationOutput]) -> float:
        """Calculate quality score from validation outputs."""
        if not validation_outputs:
            return 100.0
        
        base_score = 100.0
        
        for output in validation_outputs:
            if output.result == ValidationResult.FAIL:
                base_score -= 50.0
            elif output.result == ValidationResult.WARNING:
                base_score -= 10.0
            
            # Penalty for issues
            for issue in output.issues:
                if issue.severity == SeverityLevel.CRITICAL:
                    base_score -= 30.0
                elif issue.severity == SeverityLevel.ERROR:
                    base_score -= 15.0
                elif issue.severity == SeverityLevel.WARNING:
                    base_score -= 5.0
        
        return max(0.0, base_score)
    
    def _get_pit_query_mode(self) -> QueryMode:
        """Map validation mode to PIT query mode."""
        mode_mapping = {
            PITValidationMode.STRICT: QueryMode.STRICT,
            PITValidationMode.PERFORMANCE: QueryMode.FAST,
            PITValidationMode.BATCH: QueryMode.BULK,
            PITValidationMode.REALTIME: QueryMode.REALTIME
        }
        return mode_mapping.get(self.config.mode, QueryMode.STRICT)
    
    async def _trigger_quality_alert(self, result: PITValidationResult) -> None:
        """Trigger quality alert for low scores."""
        if not self.alerting:
            return
        
        try:
            alert_message = (
                f"Quality score {result.overall_quality_score:.1f} below threshold "
                f"{self.config.quality_score_threshold} for query: "
                f"{len(result.query.symbols)} symbols, {len(result.query.data_types)} data types"
            )
            
            # Would trigger actual alert through alerting system
            logger.warning(f"Quality alert: {alert_message}")
            
        except Exception as e:
            logger.error(f"Failed to trigger quality alert: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics."""
        avg_execution_time = (
            self.total_execution_time / max(self.validation_count, 1)
        )
        
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        
        stats = {
            'validation_count': self.validation_count,
            'avg_execution_time_ms': avg_execution_time,
            'total_execution_time_ms': self.total_execution_time,
            'cache_hit_rate': cache_hit_rate,
            'config': {
                'mode': self.config.mode.value,
                'max_latency_ms': self.config.max_latency_ms,
                'bias_check_level': self.config.bias_check_level.value,
                'taiwan_market_rules': self.config.taiwan_market_rules
            }
        }
        
        # Add engine stats
        if hasattr(self.validation_engine, 'get_performance_stats'):
            stats['validation_engine'] = self.validation_engine.get_performance_stats()
        
        # Add PIT engine stats
        if hasattr(self.pit_engine, 'get_performance_stats'):
            stats['pit_engine'] = self.pit_engine.get_performance_stats()
        
        # Add monitor stats
        if self.monitor:
            stats['monitor'] = self.monitor.get_current_metrics()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._context_cache.clear()
        self._cache_timestamps.clear()
        
        if hasattr(self.validation_engine, 'clear_cache'):
            self.validation_engine.clear_cache()
        
        if hasattr(self.pit_engine, 'clear_cache'):
            self.pit_engine.clear_cache()
        
        logger.info("All caches cleared")
    
    def shutdown(self) -> None:
        """Shutdown orchestrator and cleanup resources."""
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Cleanup validation engine
        for plugin_name in self.registry.list_plugins():
            self.registry.unregister_plugin(plugin_name)
        
        logger.info("PIT validation orchestrator shutdown")


# Utility functions for easy setup

def create_pit_validation_orchestrator(
    temporal_store: TemporalStore,
    trading_calendar: Dict[date, TaiwanTradingCalendar],
    config: Optional[PITValidationConfig] = None
) -> PITValidationOrchestrator:
    """Create a fully configured PIT validation orchestrator."""
    
    # Create PIT engine
    pit_engine = PointInTimeEngine(
        store=temporal_store,
        trading_calendar=trading_calendar,
        enable_cache=True,
        max_workers=4
    )
    
    # Create orchestrator with Taiwan market configuration
    if config is None:
        config = PITValidationConfig(
            mode=PITValidationMode.STRICT,
            max_latency_ms=10.0,
            bias_check_level=BiasCheckLevel.STRICT,
            taiwan_market_rules=True,
            parallel_validation=True
        )
    
    orchestrator = PITValidationOrchestrator(
        pit_engine=pit_engine,
        temporal_store=temporal_store,
        config=config
    )
    
    logger.info("PIT validation orchestrator created for Taiwan market")
    return orchestrator


async def validate_taiwan_market_data(
    orchestrator: PITValidationOrchestrator,
    symbols: List[str],
    as_of_date: date,
    data_types: Optional[List[DataType]] = None
) -> PITValidationResult:
    """Convenience function for Taiwan market data validation."""
    
    if data_types is None:
        data_types = [DataType.PRICE, DataType.VOLUME, DataType.MARKET_DATA]
    
    return await orchestrator.validate_point_in_time(
        symbols=symbols,
        as_of_date=as_of_date,
        data_types=data_types,
        include_historical=True
    )