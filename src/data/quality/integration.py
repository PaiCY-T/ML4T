"""
Integration module for data quality validation with Issue #21 components.

This module integrates the validation framework with the temporal storage system,
Taiwan market models, and point-in-time engine from Issue #21.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field

from ..core.temporal import TemporalStore, TemporalValue, DataType, TemporalDataManager
from ..models.taiwan_market import (
    TaiwanMarketData, TaiwanSettlement, TaiwanTradingCalendar,
    create_taiwan_trading_calendar, TaiwanMarketDataValidator
)
from ..pipeline.pit_engine import PointInTimeEngine, PITQuery, QueryMode, BiasCheckLevel

from .validation_engine import (
    ValidationEngine, ValidationRegistry, ValidationContext, ValidationOutput,
    create_validation_context, create_batch_contexts
)
from .validators import create_enhanced_validators
from .taiwan_validators import create_taiwan_validators
from .rules_engine import RulesEngine, RulesBasedValidator, create_rules_engine_with_taiwan_rules

logger = logging.getLogger(__name__)


@dataclass
class QualityValidationConfig:
    """Configuration for quality validation integration."""
    enable_taiwan_validators: bool = True
    enable_rules_engine: bool = True
    enable_enhanced_validators: bool = True
    max_validation_time_ms: int = 10  # Target <10ms validation
    cache_size: int = 10000
    enable_async_validation: bool = True
    max_workers: int = 4
    
    # Taiwan market specific settings
    price_limit_pct: float = 0.10
    volume_spike_threshold: float = 5.0
    fundamental_lag_days: int = 60
    
    # Performance settings
    enable_performance_monitoring: bool = True
    performance_alert_threshold_ms: float = 5.0


class IntegratedQualityValidator:
    """Integrated quality validator that works with Issue #21 components."""
    
    def __init__(self,
                 temporal_store: TemporalStore,
                 pit_engine: PointInTimeEngine,
                 config: Optional[QualityValidationConfig] = None):
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        self.config = config or QualityValidationConfig()
        
        # Initialize validation components
        self.registry = ValidationRegistry()
        self.validation_engine = ValidationEngine(
            registry=self.registry,
            temporal_store=temporal_store,
            max_workers=self.config.max_workers,
            timeout_ms=self.config.max_validation_time_ms * 1000,
            enable_async=self.config.enable_async_validation
        )
        
        # Initialize rules engine
        self.rules_engine = None
        if self.config.enable_rules_engine:
            self.rules_engine = create_rules_engine_with_taiwan_rules()
        
        # Initialize Taiwan market components
        self.temporal_data_manager = TemporalDataManager(temporal_store)
        self.taiwan_market_validator = TaiwanMarketDataValidator()
        self.trading_calendar = create_taiwan_trading_calendar(datetime.now().year)
        
        # Performance tracking
        self.validation_stats = {
            "total_validations": 0,
            "total_time_ms": 0.0,
            "avg_time_ms": 0.0,
            "performance_violations": 0,
            "cache_hit_rate": 0.0
        }
        
        # Register validators
        self._register_validators()
        
        logger.info("Integrated quality validator initialized")
    
    def _register_validators(self):
        """Register all validation plugins."""
        plugins_registered = 0
        
        # Register enhanced validators
        if self.config.enable_enhanced_validators:
            enhanced_validators = create_enhanced_validators()
            for validator in enhanced_validators:
                self.registry.register_plugin(validator)
                plugins_registered += 1
        
        # Register Taiwan market validators
        if self.config.enable_taiwan_validators:
            taiwan_validators = create_taiwan_validators()
            for validator in taiwan_validators:
                self.registry.register_plugin(validator)
                plugins_registered += 1
        
        # Register rules engine validator
        if self.config.enable_rules_engine and self.rules_engine:
            rules_validator = RulesBasedValidator(self.rules_engine)
            self.registry.register_plugin(rules_validator)
            plugins_registered += 1
        
        logger.info(f"Registered {plugins_registered} validation plugins")
    
    async def validate_temporal_value(self, 
                                    value: TemporalValue,
                                    include_historical_context: bool = True) -> List[ValidationOutput]:
        """Validate a temporal value with full integration."""
        start_time = datetime.utcnow()
        
        try:
            # Create validation context with historical data
            context = await self._create_validation_context(
                value, include_historical_context
            )
            
            # Perform validation
            results = await self.validation_engine.validate_value(value, context)
            
            # Update performance statistics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_performance_stats(execution_time)
            
            # Check performance threshold
            if (self.config.enable_performance_monitoring and 
                execution_time > self.config.performance_alert_threshold_ms):
                
                logger.warning(f"Validation performance warning: {execution_time:.2f}ms > "
                             f"{self.config.performance_alert_threshold_ms}ms for "
                             f"{value.symbol} {value.data_type.value}")
                self.validation_stats["performance_violations"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in integrated validation: {e}")
            raise
    
    async def validate_batch(self, 
                           values: List[TemporalValue],
                           include_historical_context: bool = True) -> Dict[TemporalValue, List[ValidationOutput]]:
        """Validate a batch of temporal values efficiently."""
        start_time = datetime.utcnow()
        
        try:
            # Create contexts for all values
            contexts = await self._create_batch_validation_contexts(
                values, include_historical_context
            )
            
            # Perform batch validation
            results = await self.validation_engine.validate_batch(values, contexts)
            
            # Update performance statistics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            avg_time_per_value = execution_time / len(values) if values else 0
            
            logger.info(f"Batch validated {len(values)} values in {execution_time:.2f}ms "
                       f"({avg_time_per_value:.2f}ms avg per value)")
            
            self._update_performance_stats(execution_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch validation: {e}")
            raise
    
    async def validate_taiwan_market_data(self, 
                                        market_data: TaiwanMarketData,
                                        include_settlement_validation: bool = True) -> List[ValidationOutput]:
        """Validate Taiwan market data with specialized handling."""
        
        # Convert to temporal values
        temporal_values = market_data.to_temporal_values()
        
        results = []
        for value in temporal_values:
            # Create context with Taiwan-specific metadata
            context = await self._create_taiwan_market_context(
                value, market_data, include_settlement_validation
            )
            
            # Validate using Taiwan-specific validators
            taiwan_results = await self.validation_engine.validate_value(
                value, context, 
                plugin_names=[
                    "taiwan_price_limit_validator",
                    "taiwan_volume_validator", 
                    "taiwan_trading_hours_validator",
                    "taiwan_settlement_validator"
                ]
            )
            
            results.extend(taiwan_results)
        
        return results
    
    async def validate_with_pit_integration(self,
                                          symbols: List[str],
                                          as_of_date: date,
                                          data_types: List[DataType]) -> Dict[str, List[ValidationOutput]]:
        """Validate data using point-in-time engine integration."""
        
        # Query data using PIT engine
        pit_query = PITQuery(
            symbols=symbols,
            as_of_date=as_of_date,
            data_types=data_types,
            mode=QueryMode.FAST,
            bias_check=BiasCheckLevel.STRICT
        )
        
        pit_result = self.pit_engine.execute_query(pit_query)
        
        # Validate each temporal value
        validation_results = {}
        
        for symbol, symbol_data in pit_result.data.items():
            validation_results[symbol] = []
            
            for data_type, temporal_value in symbol_data.items():
                # Create context with PIT metadata
                context = ValidationContext(
                    symbol=symbol,
                    data_date=temporal_value.value_date,
                    as_of_date=as_of_date,
                    data_type=data_type,
                    metadata={
                        "pit_query_time_ms": pit_result.execution_time_ms,
                        "pit_cache_hit_rate": pit_result.cache_hit_rate,
                        "bias_violations": pit_result.bias_violations
                    }
                )
                
                # Validate
                results = await self.validation_engine.validate_value(temporal_value, context)
                validation_results[symbol].extend(results)
        
        return validation_results
    
    async def _create_validation_context(self, 
                                       value: TemporalValue,
                                       include_historical: bool = True) -> ValidationContext:
        """Create validation context with historical data."""
        
        context = ValidationContext(
            symbol=value.symbol or "",
            data_date=value.value_date,
            as_of_date=value.as_of_date,
            data_type=value.data_type,
            trading_calendar=self.trading_calendar
        )
        
        # Add historical data if requested
        if include_historical and value.symbol:
            historical_data = await self._get_historical_data(
                value.symbol, value.value_date, value.data_type
            )
            context.historical_data = historical_data
        
        # Add Taiwan market metadata
        if value.symbol:
            taiwan_metadata = await self._get_taiwan_market_metadata(value.symbol, value.value_date)
            context.metadata.update(taiwan_metadata)
        
        return context
    
    async def _create_batch_validation_contexts(self,
                                              values: List[TemporalValue],
                                              include_historical: bool = True) -> List[ValidationContext]:
        """Create validation contexts for batch processing."""
        
        contexts = []
        
        # Group by symbol for efficient historical data retrieval
        symbol_groups = {}
        for i, value in enumerate(values):
            if value.symbol not in symbol_groups:
                symbol_groups[value.symbol] = []
            symbol_groups[value.symbol].append((i, value))
        
        # Pre-fetch historical data for all symbols
        historical_data_cache = {}
        if include_historical:
            for symbol in symbol_groups.keys():
                if symbol:
                    historical_data_cache[symbol] = await self._get_historical_data_range(
                        symbol, 
                        min(v.value_date for _, v in symbol_groups[symbol]) - timedelta(days=30),
                        max(v.value_date for _, v in symbol_groups[symbol])
                    )
        
        # Create contexts
        for value in values:
            context = ValidationContext(
                symbol=value.symbol or "",
                data_date=value.value_date,
                as_of_date=value.as_of_date,
                data_type=value.data_type,
                trading_calendar=self.trading_calendar
            )
            
            # Add historical data from cache
            if include_historical and value.symbol in historical_data_cache:
                relevant_history = [
                    h for h in historical_data_cache[value.symbol]
                    if (h.data_type == value.data_type and 
                        h.value_date < value.value_date)
                ]
                context.historical_data = relevant_history[-20:]  # Last 20 values
            
            contexts.append(context)
        
        return contexts
    
    async def _create_taiwan_market_context(self,
                                          value: TemporalValue,
                                          market_data: TaiwanMarketData,
                                          include_settlement: bool = True) -> ValidationContext:
        """Create Taiwan market-specific validation context."""
        
        context = ValidationContext(
            symbol=value.symbol or "",
            data_date=value.value_date,
            as_of_date=value.as_of_date,
            data_type=value.data_type,
            trading_calendar=self.trading_calendar
        )
        
        # Add Taiwan market metadata
        context.metadata.update({
            "trading_status": market_data.trading_status.value,
            "price_limit_up": float(market_data.price_limit_up) if market_data.price_limit_up else None,
            "price_limit_down": float(market_data.price_limit_down) if market_data.price_limit_down else None,
            "turnover": float(market_data.turnover) if market_data.turnover else None,
            "foreign_ownership_pct": float(market_data.foreign_ownership_pct) if market_data.foreign_ownership_pct else None
        })
        
        # Add settlement information
        if include_settlement:
            settlement_info = TaiwanSettlement.calculate_t2_settlement(
                value.value_date, self.trading_calendar
            )
            context.metadata["settlement_info"] = {
                "settlement_date": settlement_info.settlement_date.isoformat(),
                "settlement_lag_days": settlement_info.settlement_lag_days,
                "is_regular_settlement": settlement_info.is_regular_settlement
            }
        
        return context
    
    async def _get_historical_data(self, 
                                 symbol: str, 
                                 current_date: date,
                                 data_type: DataType,
                                 lookback_days: int = 30) -> List[TemporalValue]:
        """Get historical data for validation context."""
        
        start_date = current_date - timedelta(days=lookback_days)
        
        try:
            historical_values = self.temporal_store.get_range(
                symbol, start_date, current_date, data_type
            )
            return historical_values
            
        except Exception as e:
            logger.debug(f"Could not retrieve historical data for {symbol}: {e}")
            return []
    
    async def _get_historical_data_range(self,
                                       symbol: str,
                                       start_date: date,
                                       end_date: date) -> List[TemporalValue]:
        """Get historical data for a date range."""
        
        try:
            # Get all data types for the symbol in the range
            all_values = []
            for data_type in DataType:
                values = self.temporal_store.get_range(symbol, start_date, end_date, data_type)
                all_values.extend(values)
            
            return all_values
            
        except Exception as e:
            logger.debug(f"Could not retrieve historical range data for {symbol}: {e}")
            return []
    
    async def _get_taiwan_market_metadata(self, 
                                        symbol: str, 
                                        trade_date: date) -> Dict[str, Any]:
        """Get Taiwan market-specific metadata."""
        
        metadata = {}
        
        # Check if it's a trading day
        if trade_date in self.trading_calendar:
            calendar_entry = self.trading_calendar[trade_date]
            metadata["is_trading_day"] = calendar_entry.is_trading_day
            metadata["market_session"] = calendar_entry.market_session.value
            metadata["trading_hours"] = calendar_entry.trading_hours
        
        # Add settlement calculation
        settlement_date = self.temporal_data_manager.get_taiwan_settlement_date(trade_date)
        metadata["settlement_date"] = settlement_date.isoformat()
        metadata["settlement_lag_days"] = (settlement_date - trade_date).days
        
        return metadata
    
    def _update_performance_stats(self, execution_time_ms: float):
        """Update performance statistics."""
        self.validation_stats["total_validations"] += 1
        self.validation_stats["total_time_ms"] += execution_time_ms
        self.validation_stats["avg_time_ms"] = (
            self.validation_stats["total_time_ms"] / self.validation_stats["total_validations"]
        )
        
        # Update cache hit rate from validation engine
        engine_stats = self.validation_engine.get_performance_stats()
        self.validation_stats["cache_hit_rate"] = engine_stats.get("cache_hit_rate", 0.0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        engine_stats = self.validation_engine.get_performance_stats()
        pit_stats = self.pit_engine.get_performance_stats()
        
        return {
            "validation_engine": engine_stats,
            "pit_engine": pit_stats,
            "integrated_validator": self.validation_stats,
            "performance_summary": {
                "avg_validation_time_ms": self.validation_stats["avg_time_ms"],
                "performance_violations": self.validation_stats["performance_violations"],
                "target_time_ms": self.config.max_validation_time_ms,
                "performance_rating": "GOOD" if self.validation_stats["avg_time_ms"] < self.config.max_validation_time_ms else "NEEDS_IMPROVEMENT"
            }
        }
    
    def configure_alerting(self, alert_callbacks: List[callable]):
        """Configure alerting for quality issues."""
        # This would integrate with the monitoring system from Stream B
        # For now, just log that alerting is configured
        logger.info(f"Configured {len(alert_callbacks)} alert callbacks")
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run health check on all integrated components."""
        
        health_status = {
            "overall": "healthy",
            "components": {},
            "issues": []
        }
        
        try:
            # Check validation engine
            engine_stats = self.validation_engine.get_performance_stats()
            health_status["components"]["validation_engine"] = {
                "status": "healthy",
                "registered_plugins": engine_stats["registered_plugins"],
                "avg_execution_time_ms": engine_stats["avg_execution_time_ms"]
            }
            
            # Check PIT engine
            pit_stats = self.pit_engine.get_performance_stats()
            health_status["components"]["pit_engine"] = {
                "status": "healthy",
                "avg_execution_time_ms": pit_stats["avg_execution_time_ms"],
                "bias_violation_rate": pit_stats["bias_violation_rate"]
            }
            
            # Check rules engine
            if self.rules_engine:
                rules_stats = self.rules_engine.get_statistics()
                health_status["components"]["rules_engine"] = {
                    "status": "healthy",
                    "total_rules": rules_stats["total_rules"],
                    "enabled_rules": rules_stats["enabled_rules"]
                }
            
            # Performance checks
            if self.validation_stats["avg_time_ms"] > self.config.max_validation_time_ms:
                health_status["issues"].append(
                    f"Average validation time {self.validation_stats['avg_time_ms']:.2f}ms "
                    f"exceeds target {self.config.max_validation_time_ms}ms"
                )
                health_status["overall"] = "degraded"
            
        except Exception as e:
            health_status["overall"] = "unhealthy"
            health_status["issues"].append(f"Health check error: {str(e)}")
        
        return health_status


# Factory function for creating integrated validator

def create_integrated_validator(temporal_store: TemporalStore,
                               pit_engine: PointInTimeEngine,
                               config: Optional[QualityValidationConfig] = None) -> IntegratedQualityValidator:
    """Create an integrated quality validator with Issue #21 components."""
    
    return IntegratedQualityValidator(
        temporal_store=temporal_store,
        pit_engine=pit_engine,
        config=config
    )


# Example usage and testing

async def example_usage():
    """Example of how to use the integrated quality validator."""
    
    # This would be provided by Issue #21 components
    from ..core.temporal import InMemoryTemporalStore
    from ..pipeline.pit_engine import PointInTimeEngine
    
    # Create components
    temporal_store = InMemoryTemporalStore()
    pit_engine = PointInTimeEngine(temporal_store)
    
    # Create integrated validator
    config = QualityValidationConfig(
        max_validation_time_ms=5,  # Very strict performance target
        enable_performance_monitoring=True
    )
    
    validator = create_integrated_validator(temporal_store, pit_engine, config)
    
    # Example validation
    test_value = TemporalValue(
        value={"close_price": 100.0, "volume": 1000000},
        as_of_date=date.today(),
        value_date=date.today(),
        data_type=DataType.MARKET_DATA,
        symbol="2330"  # Taiwan Semiconductor
    )
    
    # Validate single value
    results = await validator.validate_temporal_value(test_value)
    
    print(f"Validation completed with {len(results)} results")
    for result in results:
        if result.issues:
            print(f"Issues found in {result.validator_name}: {len(result.issues)}")
    
    # Get performance summary
    performance = validator.get_performance_summary()
    print(f"Performance summary: {performance}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())