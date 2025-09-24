"""
Integration between validation engine and PostgreSQL temporal store.

This module provides integration between the data quality validation system
and the point-in-time temporal data store from Task #21.
"""

import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass

from ..core.temporal import (
    TemporalValue, DataType, TemporalStore, PostgreSQLTemporalStore,
    TemporalDataManager, calculate_data_lag
)
from ..models.taiwan_market import TaiwanMarketData, TaiwanSettlement
from .validation_engine import (
    ValidationEngine, ValidationContext, ValidationOutput, ValidationResult
)
from .taiwan_validators import create_optimized_taiwan_engine
from .rules_engine import create_enhanced_taiwan_rules_engine, RulesEngine

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for temporal validation integration."""
    enable_cache: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    max_historical_days: int = 60  # For fundamental lag validation
    enable_fast_path: bool = True
    enable_parallel_validation: bool = True
    latency_threshold_ms: float = 10.0  # Target <10ms latency


class TemporalValidationIntegrator:
    """Integrates validation engine with PostgreSQL temporal store."""
    
    def __init__(self, 
                 temporal_store: TemporalStore,
                 config: Optional[ValidationConfig] = None):
        self.temporal_store = temporal_store
        self.config = config or ValidationConfig()
        
        # Create optimized validation engine for Taiwan market
        self.validation_engine = create_optimized_taiwan_engine(temporal_store)
        
        # Create enhanced rules engine for fundamental lag validation
        self.rules_engine = create_enhanced_taiwan_rules_engine()
        
        # Create temporal data manager for settlement and lag calculations
        self.temporal_manager = TemporalDataManager(temporal_store)
        
        # Performance metrics
        self.validation_stats = {
            "total_validations": 0,
            "avg_latency_ms": 0.0,
            "cache_hit_rate": 0.0,
            "temporal_queries": 0,
            "lag_validations": 0
        }
        
        logger.info("Temporal validation integrator initialized with PostgreSQL store")
    
    async def validate_temporal_value(self, value: TemporalValue,
                                    include_historical: bool = True,
                                    validate_lag: bool = True) -> List[ValidationOutput]:
        """Validate a temporal value with full temporal context."""
        start_time = datetime.utcnow()
        
        try:
            # Create enhanced validation context with temporal data
            context = await self._create_temporal_context(value, include_historical)
            
            # Add lag validation metadata for fundamental data
            if value.data_type == DataType.FUNDAMENTAL and validate_lag:
                await self._add_lag_validation_metadata(value, context)
            
            # Run validation with fast-path optimization
            results = await self.validation_engine.validate_value(
                value, context, use_fast_path=self.config.enable_fast_path
            )
            
            # Apply rules engine for additional validation
            if self.rules_engine:
                rule_issues = self.rules_engine.evaluate_rules(value, context)
                
                if rule_issues:
                    # Convert rule issues to validation output
                    rules_output = self._create_rules_validation_output(rule_issues)
                    results.append(rules_output)
            
            # Update performance metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_performance_stats(execution_time)
            
            # Check latency SLA
            if execution_time > self.config.latency_threshold_ms:
                logger.warning(f"Validation latency {execution_time:.2f}ms exceeded "
                             f"threshold {self.config.latency_threshold_ms}ms for "
                             f"{value.symbol} {value.data_type.value}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in temporal validation: {e}")
            raise
    
    async def validate_batch_temporal(self, values: List[TemporalValue],
                                    parallel: bool = None) -> Dict[TemporalValue, List[ValidationOutput]]:
        """Validate a batch of temporal values with optimized performance."""
        if parallel is None:
            parallel = self.config.enable_parallel_validation
        
        start_time = datetime.utcnow()
        
        # Group values by symbol for efficient historical data retrieval
        symbol_groups = self._group_values_by_symbol(values)
        
        # Pre-fetch historical data for all symbols
        historical_data_cache = await self._prefetch_historical_data(symbol_groups)
        
        # Create contexts with cached historical data
        contexts = []
        for value in values:
            context = await self._create_temporal_context(
                value, include_historical=True, 
                historical_cache=historical_data_cache
            )
            
            # Add lag validation metadata if needed
            if value.data_type == DataType.FUNDAMENTAL:
                await self._add_lag_validation_metadata(value, context)
            
            contexts.append(context)
        
        # Run batch validation
        results = await self.validation_engine.validate_batch(values, contexts)
        
        # Apply rules engine to all values
        if self.rules_engine:
            for value, context in zip(values, contexts):
                rule_issues = self.rules_engine.evaluate_rules(value, context)
                if rule_issues:
                    rules_output = self._create_rules_validation_output(rule_issues)
                    if value in results:
                        results[value].append(rules_output)
                    else:
                        results[value] = [rules_output]
        
        # Update performance metrics
        batch_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        avg_per_value = batch_time / len(values)
        
        logger.info(f"Batch validated {len(values)} values in {batch_time:.2f}ms "
                   f"({avg_per_value:.2f}ms avg per value)")
        
        return results
    
    async def validate_taiwan_market_data(self, market_data: TaiwanMarketData,
                                        settlement: TaiwanSettlement) -> List[ValidationOutput]:
        """Validate Taiwan market data with settlement context."""
        # Convert to temporal values
        temporal_values = market_data.to_temporal_values()
        
        all_results = []
        
        for value in temporal_values:
            # Create context with Taiwan market metadata
            context = await self._create_temporal_context(value)
            
            # Add Taiwan-specific metadata
            context.set_metadata("settlement_date", settlement.settlement_date)
            context.set_metadata("settlement_lag_days", settlement.settlement_lag_days)
            context.set_metadata("is_regular_settlement", settlement.is_regular_settlement)
            context.set_metadata("market", "TWSE")
            
            # Add price change metadata for price limit validation
            if value.data_type in [DataType.PRICE, DataType.MARKET_DATA]:
                price_change_pct = await self._calculate_price_change(value, context)
                if price_change_pct is not None:
                    context.set_metadata("price_change_pct", price_change_pct)
            
            # Validate with full context
            results = await self.validate_temporal_value(value)
            all_results.extend(results)
        
        return all_results
    
    async def _create_temporal_context(self, value: TemporalValue,
                                     include_historical: bool = True,
                                     historical_cache: Optional[Dict[str, List[TemporalValue]]] = None) -> ValidationContext:
        """Create validation context with temporal information."""
        context = ValidationContext(
            symbol=value.symbol or "",
            data_date=value.value_date,
            as_of_date=value.as_of_date,
            data_type=value.data_type
        )
        
        # Add historical data if requested
        if include_historical and value.symbol:
            if historical_cache and value.symbol in historical_cache:
                # Use cached historical data
                context.historical_data = historical_cache[value.symbol]
                self.validation_stats["cache_hit_rate"] += 1
            else:
                # Fetch historical data from temporal store
                lookback_date = value.value_date - timedelta(days=self.config.max_historical_days)
                
                try:
                    historical_data = self.temporal_store.get_range(
                        value.symbol, lookback_date, value.value_date, value.data_type
                    )
                    context.historical_data = historical_data
                    self.validation_stats["temporal_queries"] += 1
                except Exception as e:
                    logger.warning(f"Failed to fetch historical data for {value.symbol}: {e}")
                    context.historical_data = []
        
        return context
    
    async def _add_lag_validation_metadata(self, value: TemporalValue, context: ValidationContext) -> None:
        """Add lag validation metadata for fundamental data."""
        if value.data_type != DataType.FUNDAMENTAL:
            return
        
        # Calculate data lag
        lag_days = (value.as_of_date - value.value_date).days
        context.set_metadata("lag_days", lag_days)
        
        # Add expected lag based on data type
        expected_lag = calculate_data_lag(value.data_type)
        context.set_metadata("expected_lag_days", expected_lag.days)
        
        # Determine if this violates Taiwan regulations
        if isinstance(value.value, dict):
            fiscal_quarter = value.value.get("fiscal_quarter")
            if fiscal_quarter == 4:  # Annual report
                max_allowed_lag = 90
            else:  # Quarterly report
                max_allowed_lag = 60
            
            context.set_metadata("max_allowed_lag_days", max_allowed_lag)
            context.set_metadata("violates_taiwan_regulation", lag_days > max_allowed_lag)
        
        self.validation_stats["lag_validations"] += 1
    
    async def _calculate_price_change(self, value: TemporalValue, 
                                    context: ValidationContext) -> Optional[float]:
        """Calculate price change percentage for price limit validation."""
        current_price = None
        
        if value.data_type == DataType.PRICE:
            current_price = float(value.value)
        elif value.data_type == DataType.MARKET_DATA and isinstance(value.value, dict):
            current_price = value.value.get("close_price")
            if current_price:
                current_price = float(current_price)
        
        if not current_price or not value.symbol:
            return None
        
        # Get previous trading day's close
        try:
            previous_date = value.value_date - timedelta(days=1)
            # Find actual previous trading day (skip weekends/holidays)
            for i in range(7):  # Max 7 days lookback
                prev_value = self.temporal_store.get_point_in_time(
                    value.symbol, previous_date, DataType.PRICE
                )
                if prev_value:
                    previous_price = float(prev_value.value)
                    price_change_pct = abs(current_price - previous_price) / previous_price
                    return price_change_pct
                previous_date -= timedelta(days=1)
        except Exception as e:
            logger.debug(f"Could not calculate price change for {value.symbol}: {e}")
        
        return None
    
    def _group_values_by_symbol(self, values: List[TemporalValue]) -> Dict[str, List[TemporalValue]]:
        """Group temporal values by symbol for efficient batch processing."""
        groups = {}
        for value in values:
            if value.symbol:
                if value.symbol not in groups:
                    groups[value.symbol] = []
                groups[value.symbol].append(value)
        return groups
    
    async def _prefetch_historical_data(self, 
                                      symbol_groups: Dict[str, List[TemporalValue]]) -> Dict[str, List[TemporalValue]]:
        """Pre-fetch historical data for all symbols in batch."""
        historical_cache = {}
        
        for symbol, values in symbol_groups.items():
            # Determine date range needed
            min_date = min(v.value_date for v in values) - timedelta(days=self.config.max_historical_days)
            max_date = max(v.value_date for v in values)
            
            # Fetch data for all data types needed
            data_types = set(v.data_type for v in values)
            
            all_historical = []
            for data_type in data_types:
                try:
                    historical_data = self.temporal_store.get_range(
                        symbol, min_date, max_date, data_type
                    )
                    all_historical.extend(historical_data)
                except Exception as e:
                    logger.warning(f"Failed to prefetch {data_type} data for {symbol}: {e}")
            
            historical_cache[symbol] = all_historical
        
        return historical_cache
    
    def _create_rules_validation_output(self, rule_issues) -> ValidationOutput:
        """Create validation output from rules engine issues."""
        from .validators import SeverityLevel
        
        # Determine overall result based on highest severity
        has_critical = any(issue.severity == SeverityLevel.CRITICAL for issue in rule_issues)
        has_error = any(issue.severity == SeverityLevel.ERROR for issue in rule_issues)
        
        if has_critical or has_error:
            result = ValidationResult.FAIL
        else:
            result = ValidationResult.WARNING
        
        return ValidationOutput(
            validator_name="taiwan_rules_engine",
            validation_id=f"rules_{int(datetime.utcnow().timestamp())}",
            result=result,
            issues=rule_issues
        )
    
    def _update_performance_stats(self, execution_time_ms: float) -> None:
        """Update performance statistics."""
        self.validation_stats["total_validations"] += 1
        
        # Update running average
        count = self.validation_stats["total_validations"]
        current_avg = self.validation_stats["avg_latency_ms"]
        new_avg = (current_avg * (count - 1) + execution_time_ms) / count
        self.validation_stats["avg_latency_ms"] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        engine_stats = self.validation_engine.get_performance_stats()
        
        # Combine with temporal integration stats
        combined_stats = {
            **engine_stats,
            "temporal_integration": self.validation_stats,
            "sla_compliance": {
                "latency_threshold_ms": self.config.latency_threshold_ms,
                "meets_sla": engine_stats["avg_execution_time_ms"] <= self.config.latency_threshold_ms,
                "performance_grade": "A" if engine_stats["avg_execution_time_ms"] <= 10 else "B"
            }
        }
        
        return combined_stats


# Factory functions for easy setup

def create_postgres_validation_system(connection_params: Dict[str, Any],
                                     config: Optional[ValidationConfig] = None) -> TemporalValidationIntegrator:
    """Create a complete validation system with PostgreSQL temporal store."""
    # Create PostgreSQL temporal store
    temporal_store = PostgreSQLTemporalStore(connection_params)
    
    # Create integrator
    integrator = TemporalValidationIntegrator(temporal_store, config)
    
    logger.info("Created PostgreSQL validation system with Taiwan market validators")
    return integrator


async def validate_taiwan_data_with_postgres(connection_params: Dict[str, Any],
                                           data: TaiwanMarketData,
                                           settlement: TaiwanSettlement) -> List[ValidationOutput]:
    """Quick function to validate Taiwan market data using PostgreSQL store."""
    integrator = create_postgres_validation_system(connection_params)
    
    try:
        results = await integrator.validate_taiwan_market_data(data, settlement)
        return results
    finally:
        # Clean up connections
        if hasattr(integrator.temporal_store, 'close'):
            integrator.temporal_store.close()