"""
Incremental Data Updater with Temporal Consistency.

This module provides incremental data update capabilities for the point-in-time system,
ensuring temporal consistency and no look-ahead bias during data ingestion.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import time
import hashlib

from ..core.temporal import (
    TemporalValue, TemporalStore, DataType, TemporalDataManager,
    calculate_data_lag, is_taiwan_trading_day, get_previous_trading_day
)
from ..models.taiwan_market import (
    TaiwanMarketData, TaiwanSettlement, TaiwanTradingCalendar,
    TaiwanMarketDataValidator, create_taiwan_trading_calendar
)
from ..ingestion.finlab_connector import FinLabConnector, FinLabConfig
from .finlab_dataset_config import FinLabDatasetConfig, FinLabField, DatasetUpdateStrategy
from .data_validation import DataValidator, ValidationReport
from .monitoring import pipeline_monitor, monitor_performance, AlertLevel

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        return delay


@dataclass
class ErrorContext:
    """Context information for error recovery."""
    error_type: str
    error_message: str
    symbol: Optional[str] = None
    data_type: Optional[DataType] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorRecoveryManager:
    """Manages error recovery strategies and patterns."""

    def __init__(self):
        self.error_history: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.recovery_strategies = {
            "ConnectionError": self._handle_connection_error,
            "AuthenticationError": self._handle_auth_error,
            "DataValidationError": self._handle_validation_error,
            "RateLimitError": self._handle_rate_limit_error,
            "TimeoutError": self._handle_timeout_error
        }

    def should_retry(self, error_context: ErrorContext, max_retries: int = 3) -> bool:
        """Determine if an error should be retried."""
        if not error_context.recoverable:
            return False

        if error_context.retry_count >= max_retries:
            return False

        # Check error patterns for this symbol/type combination
        key = f"{error_context.symbol}_{error_context.data_type}"
        recent_errors = [
            e for e in self.error_history[key]
            if (datetime.utcnow() - e.timestamp).total_seconds() < 3600  # Last hour
        ]

        # Don't retry if too many recent errors for same symbol/type
        if len(recent_errors) > 5:
            return False

        return True

    def record_error(self, error_context: ErrorContext) -> None:
        """Record error for pattern analysis."""
        key = f"{error_context.symbol}_{error_context.data_type}"
        self.error_history[key].append(error_context)

        # Keep only recent errors (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.error_history[key] = [
            e for e in self.error_history[key] if e.timestamp >= cutoff
        ]

    def get_recovery_strategy(self, error_type: str) -> Optional[Callable]:
        """Get recovery strategy for error type."""
        return self.recovery_strategies.get(error_type)

    def _handle_connection_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Handle connection errors."""
        return {
            "action": "retry",
            "delay": 5.0 * (error_context.retry_count + 1),
            "max_retries": 3
        }

    def _handle_auth_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Handle authentication errors."""
        return {
            "action": "refresh_auth",
            "delay": 2.0,
            "max_retries": 2
        }

    def _handle_validation_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Handle validation errors."""
        return {
            "action": "skip_and_log",
            "delay": 0.0,
            "max_retries": 0  # Don't retry validation errors
        }

    def _handle_rate_limit_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Handle rate limiting errors."""
        return {
            "action": "retry",
            "delay": 60.0 * (error_context.retry_count + 1),  # Longer delays for rate limits
            "max_retries": 5
        }

    def _handle_timeout_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Handle timeout errors."""
        return {
            "action": "retry",
            "delay": 10.0 * (error_context.retry_count + 1),
            "max_retries": 2
        }


class UpdateMode(Enum):
    """Data update modes."""
    INCREMENTAL = "incremental"  # Only new/changed data
    FULL_REFRESH = "full_refresh"  # Complete data refresh
    SELECTIVE = "selective"       # Selected symbols only
    BACKFILL = "backfill"        # Historical data backfill


class UpdatePriority(Enum):
    """Update priority levels."""
    CRITICAL = "critical"    # Real-time market data
    HIGH = "high"           # End-of-day prices
    MEDIUM = "medium"       # Technical indicators
    LOW = "low"             # Fundamental data


class DataChangeType(Enum):
    """Types of data changes."""
    NEW = "new"              # New data point
    UPDATED = "updated"      # Modified existing data
    CORRECTED = "corrected"  # Error correction
    DELETED = "deleted"      # Data removal


@dataclass
class UpdateRequest:
    """Data update request specification."""
    symbols: List[str]
    data_types: List[DataType]
    start_date: date
    end_date: date
    mode: UpdateMode = UpdateMode.INCREMENTAL
    priority: UpdatePriority = UpdatePriority.MEDIUM
    force_update: bool = False
    validate_consistency: bool = True
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        """Validate update request."""
        if self.end_date < self.start_date:
            raise ValueError("End date must be after start date")
        if not self.symbols:
            raise ValueError("At least one symbol required")


@dataclass
class UpdateResult:
    """Result of data update operation."""
    request: UpdateRequest
    success: bool
    processed_count: int = 0
    new_count: int = 0
    updated_count: int = 0
    error_count: int = 0
    execution_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_quality_issues: List[str] = field(default_factory=list)
    
    @property
    def total_changes(self) -> int:
        """Total number of data changes."""
        return self.new_count + self.updated_count


@dataclass
class DataCheckpoint:
    """Checkpoint for incremental updates."""
    symbol: str
    data_type: DataType
    last_update_date: date
    last_update_timestamp: datetime
    hash_value: Optional[str] = None
    version: int = 1


class UpdateQueue:
    """Priority queue for update requests."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queues = {
            UpdatePriority.CRITICAL: deque(),
            UpdatePriority.HIGH: deque(),
            UpdatePriority.MEDIUM: deque(),
            UpdatePriority.LOW: deque()
        }
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
    
    def put(self, request: UpdateRequest) -> bool:
        """Add update request to queue."""
        with self._lock:
            queue = self._queues[request.priority]
            
            if len(queue) >= self.max_size // 4:  # Per-priority limit
                logger.warning(f"Update queue full for priority {request.priority}")
                return False
            
            queue.append(request)
            self._not_empty.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[UpdateRequest]:
        """Get next update request by priority."""
        with self._not_empty:
            # Check queues in priority order
            for priority in [UpdatePriority.CRITICAL, UpdatePriority.HIGH, 
                           UpdatePriority.MEDIUM, UpdatePriority.LOW]:
                queue = self._queues[priority]
                if queue:
                    return queue.popleft()
            
            # Wait for new requests
            if timeout is None:
                self._not_empty.wait()
            else:
                self._not_empty.wait(timeout)
            
            # Try again after waiting
            for priority in [UpdatePriority.CRITICAL, UpdatePriority.HIGH,
                           UpdatePriority.MEDIUM, UpdatePriority.LOW]:
                queue = self._queues[priority]
                if queue:
                    return queue.popleft()
            
            return None
    
    def size(self) -> int:
        """Get total queue size."""
        with self._lock:
            return sum(len(q) for q in self._queues.values())
    
    def clear(self) -> None:
        """Clear all queues."""
        with self._lock:
            for queue in self._queues.values():
                queue.clear()


class TemporalConsistencyChecker:
    """Validates temporal consistency during updates."""
    
    def __init__(self, trading_calendar: Dict[date, TaiwanTradingCalendar]):
        self.trading_calendar = trading_calendar
        self.validator = TaiwanMarketDataValidator()
    
    def check_temporal_order(self, 
                           new_value: TemporalValue,
                           existing_values: List[TemporalValue]) -> List[str]:
        """Check if new value maintains temporal order."""
        issues = []
        
        # Check as_of_date ordering
        later_values = [v for v in existing_values 
                       if v.as_of_date > new_value.as_of_date and 
                          v.value_date <= new_value.value_date]
        
        if later_values:
            issues.append(
                f"New value as_of_date {new_value.as_of_date} violates temporal order "
                f"with {len(later_values)} existing values"
            )
        
        return issues
    
    def check_settlement_consistency(self, 
                                   value: TemporalValue,
                                   trade_date: date) -> List[str]:
        """Check T+2 settlement consistency."""
        issues = []
        
        if value.data_type in [DataType.PRICE, DataType.VOLUME]:
            # Price/volume data should be available same day or next day
            max_lag = timedelta(days=1)
            actual_lag = value.as_of_date - value.value_date
            
            if actual_lag > max_lag:
                issues.append(
                    f"Price/volume data lag {actual_lag.days} days exceeds maximum {max_lag.days} days"
                )
        
        elif value.data_type == DataType.FUNDAMENTAL:
            # Fundamental data has regulatory reporting lags
            max_lag = timedelta(days=60)  # Taiwan regulation
            actual_lag = value.as_of_date - value.value_date
            
            if actual_lag > max_lag:
                issues.append(
                    f"Fundamental data lag {actual_lag.days} days exceeds regulatory maximum {max_lag.days} days"
                )
        
        return issues
    
    def check_data_availability_window(self, 
                                     value: TemporalValue,
                                     current_time: datetime) -> List[str]:
        """Check if data is available within expected time window."""
        issues = []
        
        # Convert dates to datetime for comparison
        value_datetime = datetime.combine(value.value_date, datetime.min.time())
        as_of_datetime = datetime.combine(value.as_of_date, datetime.min.time())
        
        # Check if data is from the future
        if as_of_datetime > current_time:
            issues.append(f"Data as_of_date {value.as_of_date} is in the future")
        
        if value_datetime > current_time:
            issues.append(f"Data value_date {value.value_date} is in the future")
        
        return issues


class IncrementalUpdater:
    """Enhanced incremental data updater with comprehensive FinLab support.

    Features:
    - Comprehensive FinLab dataset support with 278+ fields
    - Robust error handling and automatic recovery
    - Real-time performance monitoring and alerting
    - Advanced data validation and quality checks
    - Optimized batch processing with configurable strategies
    - Temporal consistency validation with settlement rules
    - Configurable retry logic with exponential backoff
    - Integration with authentication and validation frameworks
    """

    def __init__(self,
                 temporal_store: TemporalStore,
                 finlab_connector: FinLabConnector,
                 trading_calendar: Optional[Dict[date, TaiwanTradingCalendar]] = None,
                 max_workers: int = 4,
                 enable_queue: bool = True,
                 config_file: Optional[Path] = None,
                 enable_monitoring: bool = True,
                 enable_validation: bool = True):

        self.temporal_store = temporal_store
        self.finlab_connector = finlab_connector
        self.trading_calendar = trading_calendar or create_taiwan_trading_calendar(2024)
        self.max_workers = max_workers

        # Enhanced configuration management
        self.dataset_config = FinLabDatasetConfig()
        self.update_strategy = DatasetUpdateStrategy(self.dataset_config)
        self.field_configs = self.dataset_config.get_all_fields()

        # Data validation framework
        self.data_validator = DataValidator() if enable_validation else None
        self.enable_validation = enable_validation

        # Monitoring and alerting
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitor = pipeline_monitor
            self.monitor.update_component_status("incremental_updater", "initializing", "Starting up...")

        # Update tracking with enhanced checkpointing
        self.checkpoints: Dict[Tuple[str, DataType], DataCheckpoint] = {}
        self.update_queue = UpdateQueue() if enable_queue else None
        self.checkpoint_file = config_file.parent / "checkpoints.json" if config_file else Path("checkpoints.json")

        # Consistency checking
        self.consistency_checker = TemporalConsistencyChecker(self.trading_calendar)

        # Enhanced error handling and recovery
        self.error_recovery = ErrorRecoveryManager()
        self.retry_config = RetryConfig()

        # Performance metrics (enhanced)
        self.update_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        self.validation_reports: List[ValidationReport] = []
        self.performance_history = deque(maxlen=1000)

        # Background processing
        self._stop_event = threading.Event()
        self._worker_thread = None

        # Load checkpoints and initialize
        self._load_checkpoints()

        if self.enable_monitoring:
            self.monitor.update_component_status("incremental_updater", "healthy", "Initialized successfully")

        logger.info(f"Enhanced incremental updater initialized with {len(self.field_configs)} field configurations")
    
    def start_background_processing(self) -> None:
        """Start background update processing thread."""
        if self.update_queue and not self._worker_thread:
            self._worker_thread = threading.Thread(
                target=self._background_worker,
                daemon=True
            )
            self._worker_thread.start()
            logger.info("Background update processing started")
    
    def stop_background_processing(self) -> None:
        """Stop background update processing."""
        if self._worker_thread:
            self._stop_event.set()
            self._worker_thread.join(timeout=30)
            self._worker_thread = None
            logger.info("Background update processing stopped")
    
    def _background_worker(self) -> None:
        """Background worker thread for processing updates."""
        while not self._stop_event.is_set():
            try:
                request = self.update_queue.get(timeout=1.0)
                if request:
                    result = self.execute_update(request)
                    if request.callback:
                        request.callback(result)
                        
            except Exception as e:
                logger.error(f"Background worker error: {e}")
                time.sleep(1)
    
    def schedule_update(self, request: UpdateRequest) -> bool:
        """Schedule an update request for background processing."""
        if self.update_queue:
            return self.update_queue.put(request)
        else:
            # Execute immediately if no queue
            result = self.execute_update(request)
            if request.callback:
                request.callback(result)
            return result.success
    
    def execute_update(self, request: UpdateRequest) -> UpdateResult:
        """Execute data update request."""
        start_time = time.time()
        
        logger.info(f"Executing update: {len(request.symbols)} symbols, "
                   f"{len(request.data_types)} data types, "
                   f"{request.start_date} to {request.end_date}")
        
        result = UpdateResult(
            request=request,
            success=False
        )
        
        try:
            if request.mode == UpdateMode.INCREMENTAL:
                self._execute_incremental_update(request, result)
            elif request.mode == UpdateMode.FULL_REFRESH:
                self._execute_full_refresh(request, result)
            elif request.mode == UpdateMode.BACKFILL:
                self._execute_backfill(request, result)
            else:
                self._execute_selective_update(request, result)
            
            result.success = result.error_count == 0
            
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            result.errors.append(str(e))
            result.success = False
        
        finally:
            result.execution_time_seconds = time.time() - start_time
            self.update_count += 1
            self.total_processing_time += result.execution_time_seconds
            
            if not result.success:
                self.error_count += 1
        
        logger.info(f"Update completed: {result.total_changes} changes, "
                   f"{result.error_count} errors, "
                   f"{result.execution_time_seconds:.2f}s")
        
        return result
    
    def _execute_incremental_update(self,
                                   request: UpdateRequest,
                                   result: UpdateResult) -> None:
        """Execute incremental update with comprehensive error handling and monitoring."""

        if self.enable_monitoring:
            self.monitor.performance_tracker.start_timer("incremental_update")

        try:
            # Enhanced processing strategy selection
            if len(request.symbols) > 50:  # Large batch processing
                self._execute_large_batch_update(request, result)
            elif len(request.symbols) > 20:  # Medium batch processing
                self._execute_medium_batch_update(request, result)
            else:
                # Parallel processing for smaller requests
                self._parallel_process_symbols_enhanced(request.symbols, request, result)

            # Save checkpoints after successful update
            self._save_checkpoints()

        except Exception as e:
            logger.error(f"Incremental update execution failed: {e}")
            result.errors.append(f"Update execution error: {str(e)}")

            if self.enable_monitoring:
                self.monitor.alert_manager.raise_alert(
                    AlertLevel.ERROR,
                    "Incremental update execution failed",
                    str(e),
                    "incremental_updater",
                    {"symbols_count": len(request.symbols), "error": str(e)}
                )

        finally:
            if self.enable_monitoring:
                duration = self.monitor.performance_tracker.end_timer("incremental_update")
                self.monitor.track_update_operation(
                    "batch" if len(request.symbols) > 1 else request.symbols[0],
                    result.processed_count,
                    duration,
                    result.error_count
                )
    
    def _parallel_process_symbols_enhanced(self,
                                          symbols: List[str],
                                          request: UpdateRequest,
                                          result: UpdateResult) -> None:
        """Process symbols in parallel with enhanced error handling."""

        # Execute updates in parallel with enhanced monitoring
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._update_single_symbol_with_validation, symbol, request): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    new_cnt, updated_cnt, errors = future.result(timeout=300)  # 5-minute timeout
                    result.new_count += new_cnt
                    result.updated_count += updated_cnt
                    result.errors.extend(errors)
                    result.processed_count += 1
                except Exception as e:
                    logger.error(f"Parallel processing failed for symbol {symbol}: {e}")
                    result.errors.append(f"Parallel processing error for {symbol}: {e}")
                    result.error_count += 1
    
    def _execute_full_refresh(self, 
                             request: UpdateRequest,
                             result: UpdateResult) -> None:
        """Execute full data refresh."""
        # Clear existing checkpoints
        for symbol in request.symbols:
            for data_type in request.data_types:
                checkpoint_key = (symbol, data_type)
                if checkpoint_key in self.checkpoints:
                    del self.checkpoints[checkpoint_key]
        
        # Use incremental logic with force_update
        request.force_update = True
        self._execute_incremental_update(request, result)
    
    def _execute_backfill(self,
                         request: UpdateRequest,
                         result: UpdateResult) -> None:
        """Execute historical data backfill."""
        # Backfill is similar to full refresh but for historical dates
        self._execute_incremental_update(request, result)
    
    def _execute_selective_update(self,
                                 request: UpdateRequest,
                                 result: UpdateResult) -> None:
        """Execute selective update for specific symbols/types."""
        self._execute_incremental_update(request, result)
    
    def _validate_value_consistency(self, value: TemporalValue) -> List[str]:
        """Validate temporal consistency of a value."""
        issues = []
        
        # Check temporal order
        existing_values = self.temporal_store.get_range(
            value.symbol,
            value.value_date - timedelta(days=7),
            value.value_date + timedelta(days=7),
            value.data_type
        )
        
        temporal_issues = self.consistency_checker.check_temporal_order(
            value, existing_values
        )
        issues.extend(temporal_issues)
        
        # Check settlement consistency
        settlement_issues = self.consistency_checker.check_settlement_consistency(
            value, value.value_date
        )
        issues.extend(settlement_issues)
        
        # Check data availability window
        availability_issues = self.consistency_checker.check_data_availability_window(
            value, datetime.utcnow()
        )
        issues.extend(availability_issues)
        
        return issues
    
    def _values_differ(self, existing: TemporalValue, new: TemporalValue) -> bool:
        """Check if two temporal values are different."""
        # Compare values (accounting for decimal precision)
        if isinstance(existing.value, (int, float)) and isinstance(new.value, (int, float)):
            return abs(float(existing.value) - float(new.value)) > 1e-6
        else:
            return existing.value != new.value
    
    def _batch_process_symbols(self, 
                              symbols: List[str],
                              request: UpdateRequest,
                              result: UpdateResult) -> None:
        """Process symbols in optimized batches for better performance."""
        batch_size = min(10, len(symbols))  # Optimal batch size for database operations
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            # Process batch in parallel
            def process_batch(batch_symbols: List[str]) -> Tuple[int, int, List[str]]:
                batch_new = 0
                batch_updated = 0
                batch_errors = []
                
                for symbol in batch_symbols:
                    try:
                        symbol_new, symbol_updated, symbol_errors = self._update_single_symbol(
                            symbol, request
                        )
                        batch_new += symbol_new
                        batch_updated += symbol_updated
                        batch_errors.extend(symbol_errors)
                    except Exception as e:
                        batch_errors.append(f"Batch processing error for {symbol}: {e}")
                
                return batch_new, batch_updated, batch_errors
            
            # Execute batch
            batch_new, batch_updated, batch_errors = process_batch(batch_symbols)
            
            result.new_count += batch_new
            result.updated_count += batch_updated
            result.errors.extend(batch_errors)
            result.processed_count += len(batch_symbols)
    
    def _update_single_symbol(self, 
                             symbol: str, 
                             request: UpdateRequest) -> Tuple[int, int, List[str]]:
        """Update a single symbol with enhanced error handling and validation."""
        new_count = 0
        updated_count = 0
        errors = []
        
        try:
            for data_type in request.data_types:
                # Get checkpoint for this symbol/data_type
                checkpoint_key = (symbol, data_type)
                checkpoint = self.checkpoints.get(checkpoint_key)
                
                # Determine date range for incremental update
                if checkpoint and not request.force_update:
                    update_start = checkpoint.last_update_date + timedelta(days=1)
                else:
                    update_start = request.start_date
                
                # Skip if no new data needed
                if update_start > request.end_date:
                    continue
                
                # Get new data from FinLab with enhanced error handling
                try:
                    new_values = self._fetch_data_with_retry(
                        symbol, data_type, update_start, request.end_date
                    )
                except Exception as e:
                    errors.append(f"Data fetch failed for {symbol} {data_type}: {e}")
                    continue
                
                # Process and validate values
                processed_values = self._process_and_validate_values(
                    new_values, request, errors
                )
                
                # Store values efficiently
                store_new, store_updated = self._store_values_efficiently(
                    processed_values, symbol, data_type
                )
                
                new_count += store_new
                updated_count += store_updated
                
                # Update checkpoint
                if processed_values:
                    latest_date = max(v.value_date for v in processed_values)
                    self._update_checkpoint(symbol, data_type, latest_date)
        
        except Exception as e:
            errors.append(f"Symbol update failed for {symbol}: {e}")
        
        return new_count, updated_count, errors
    
    def _fetch_data_with_retry(self, 
                              symbol: str,
                              data_type: DataType,
                              start_date: date,
                              end_date: date,
                              max_retries: int = 3) -> List[TemporalValue]:
        """Fetch data with retry logic for resilience."""
        for attempt in range(max_retries):
            try:
                if data_type == DataType.PRICE:
                    return self.finlab_connector.get_price_data(symbol, start_date, end_date)
                elif data_type == DataType.FUNDAMENTAL:
                    return self.finlab_connector.get_fundamental_data(symbol, start_date, end_date)
                elif data_type == DataType.CORPORATE_ACTION:
                    return self.finlab_connector.get_corporate_actions(symbol, start_date, end_date)
                else:
                    return []
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Data fetch attempt {attempt + 1} failed for {symbol}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return []
    
    def _process_and_validate_values(self, 
                                   values: List[TemporalValue],
                                   request: UpdateRequest,
                                   errors: List[str]) -> List[TemporalValue]:
        """Process and validate temporal values with enhanced checks."""
        processed_values = []
        
        for value in values:
            # Enhanced temporal consistency validation
            if request.validate_consistency:
                consistency_issues = self._enhanced_consistency_check(value)
                if consistency_issues:
                    errors.extend(consistency_issues)
                    continue
            
            # Data quality validation
            quality_issues = self._validate_data_quality(value)
            if quality_issues:
                errors.extend(quality_issues)
                # Still process but log issues
            
            processed_values.append(value)
        
        return processed_values
    
    def _enhanced_consistency_check(self, value: TemporalValue) -> List[str]:
        """Enhanced temporal consistency checking."""
        issues = []
        
        # Basic temporal validation
        basic_issues = self._validate_value_consistency(value)
        issues.extend(basic_issues)
        
        # Additional enhanced checks
        
        # Check for reasonable value ranges
        if value.data_type == DataType.PRICE and isinstance(value.value, (int, float)):
            price = float(value.value)
            if price <= 0:
                issues.append(f"Invalid price value: {price} <= 0")
            elif price > 10000:  # Taiwan stock price upper bound check
                issues.append(f"Unusually high price value: {price}")
        
        # Check metadata consistency
        if value.metadata:
            if "source" not in value.metadata:
                issues.append("Missing source information in metadata")
            
            # Check field consistency for FinLab data
            if value.metadata.get("source") == "finlab":
                if "field" not in value.metadata:
                    issues.append("Missing field information for FinLab data")
        
        # Check settlement timing for Taiwan market
        if value.data_type in [DataType.PRICE, DataType.VOLUME]:
            # Price data should be available same day or with minimal lag
            lag = (value.as_of_date - value.value_date).days
            if lag > 1:
                issues.append(f"Price data lag {lag} days exceeds expected maximum")
        
        return issues
    
    def _validate_data_quality(self, value: TemporalValue) -> List[str]:
        """Validate data quality for individual temporal values."""
        issues = []
        
        # Check for null or invalid values
        if value.value is None:
            issues.append("Null value detected")
            return issues
        
        # Type-specific validation
        if value.data_type == DataType.PRICE:
            if isinstance(value.value, (int, float)):
                price = float(value.value)
                if price < 0:
                    issues.append(f"Negative price detected: {price}")
                elif price == 0:
                    issues.append("Zero price detected - may indicate data issue")
        
        elif value.data_type == DataType.VOLUME:
            if isinstance(value.value, (int, float)):
                volume = int(value.value)
                if volume < 0:
                    issues.append(f"Negative volume detected: {volume}")
        
        elif value.data_type == DataType.FUNDAMENTAL:
            # Validate fundamental data structure
            if isinstance(value.value, dict):
                required_fields = ["revenue", "net_income", "total_assets"]
                missing_fields = [f for f in required_fields if f not in value.value]
                if missing_fields:
                    issues.append(f"Missing fundamental fields: {missing_fields}")
        
        return issues
    
    def _store_values_efficiently(self, 
                                 values: List[TemporalValue],
                                 symbol: str,
                                 data_type: DataType) -> Tuple[int, int]:
        """Store values efficiently with bulk operations when possible."""
        if not values:
            return 0, 0
        
        new_count = 0
        updated_count = 0
        
        # Check if temporal store supports bulk operations
        if hasattr(self.temporal_store, 'bulk_store'):
            # Use bulk storage for new values
            new_values = []
            updated_values = []
            
            for value in values:
                existing = self.temporal_store.get_point_in_time(
                    symbol, value.as_of_date, data_type
                )
                
                if existing:
                    if self._values_differ(existing, value):
                        updated_values.append(value)
                else:
                    new_values.append(value)
            
            # Bulk store new values
            if new_values:
                self.temporal_store.bulk_store(new_values)
                new_count = len(new_values)
            
            # Store updated values individually (may need special handling)
            for value in updated_values:
                self.temporal_store.store(value)
                updated_count += 1
        
        else:
            # Fallback to individual storage
            for value in values:
                existing = self.temporal_store.get_point_in_time(
                    symbol, value.as_of_date, data_type
                )
                
                if existing:
                    if self._values_differ(existing, value):
                        self.temporal_store.store(value)
                        updated_count += 1
                else:
                    self.temporal_store.store(value)
                    new_count += 1
        
        return new_count, updated_count
    
    def _update_checkpoint(self, 
                          symbol: str,
                          data_type: DataType,
                          last_date: date) -> None:
        """Update checkpoint for symbol/data_type."""
        checkpoint_key = (symbol, data_type)
        
        checkpoint = DataCheckpoint(
            symbol=symbol,
            data_type=data_type,
            last_update_date=last_date,
            last_update_timestamp=datetime.utcnow(),
            version=1
        )
        
        self.checkpoints[checkpoint_key] = checkpoint
    
    def get_update_status(self, symbol: str, data_type: DataType) -> Optional[DataCheckpoint]:
        """Get update status for symbol/data_type."""
        return self.checkpoints.get((symbol, data_type))
    
    def reset_checkpoint(self, symbol: str, data_type: DataType) -> None:
        """Reset checkpoint for symbol/data_type."""
        checkpoint_key = (symbol, data_type)
        if checkpoint_key in self.checkpoints:
            del self.checkpoints[checkpoint_key]
            logger.info(f"Reset checkpoint for {symbol} {data_type}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get updater performance statistics."""
        avg_processing_time = (
            self.total_processing_time / max(self.update_count, 1)
        )
        
        stats = {
            "update_count": self.update_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.update_count, 1),
            "avg_processing_time_seconds": avg_processing_time,
            "total_processing_time_seconds": self.total_processing_time,
            "checkpoint_count": len(self.checkpoints),
            "queue_size": self.update_queue.size() if self.update_queue else 0
        }
        
        return stats

    def _fetch_data_with_enhanced_retry(self,
                                       symbol: str,
                                       data_type: DataType,
                                       start_date: date,
                                       end_date: date) -> List[TemporalValue]:
        """Fetch data with enhanced retry logic and recovery."""
        for attempt in range(self.retry_config.max_attempts):
            try:
                if data_type == DataType.PRICE:
                    return self.finlab_connector.get_price_data(symbol, start_date, end_date)
                elif data_type == DataType.FUNDAMENTAL:
                    return self.finlab_connector.get_fundamental_data(symbol, start_date, end_date)
                elif data_type == DataType.CORPORATE_ACTION:
                    return self.finlab_connector.get_corporate_actions(symbol, start_date, end_date)
                else:
                    return []
            except Exception as e:
                if attempt == self.retry_config.max_attempts - 1:
                    raise e

                # Log retry attempt
                logger.warning(f"Enhanced data fetch attempt {attempt + 1} failed for {symbol}: {e}")

                # Apply enhanced delay strategy
                delay = self.retry_config.get_delay(attempt)
                time.sleep(delay)

        return []

    def _process_and_validate_with_context(self,
                                         values: List[TemporalValue],
                                         field_configs: Dict[str, FinLabField],
                                         request: UpdateRequest,
                                         errors: List[str]) -> List[TemporalValue]:
        """Process and validate temporal values with enhanced context."""
        processed_values = []

        for value in values:
            field_name = value.metadata.get("field", "unknown")
            field_config = field_configs.get(field_name)

            if not field_config:
                errors.append(f"Unknown field configuration for {field_name}")
                continue

            # Enhanced temporal consistency validation
            if request.validate_consistency:
                consistency_issues = self._enhanced_consistency_check_with_context(value, field_config)
                if consistency_issues:
                    errors.extend(consistency_issues)
                    # Still process but with warnings

            # Data quality validation with field-specific rules
            quality_issues = self._validate_data_quality_with_field_config(value, field_config)
            if quality_issues:
                # Critical issues block processing, others just log
                critical_issues = [issue for issue in quality_issues if "critical" in issue.lower()]
                if critical_issues:
                    errors.extend(critical_issues)
                    continue
                else:
                    errors.extend(quality_issues)  # Log non-critical issues

            processed_values.append(value)

        return processed_values

    def _enhanced_consistency_check_with_context(self,
                                               value: TemporalValue,
                                               field_config: FinLabField) -> List[str]:
        """Enhanced temporal consistency checking with field context."""
        issues = []

        # Check expected data lag against field configuration
        expected_lag = timedelta(days=field_config.lag_days)
        actual_lag = value.as_of_date - value.value_date

        if actual_lag < timedelta(days=0):
            issues.append(f"CRITICAL: as_of_date {value.as_of_date} is before value_date {value.value_date}")

        # Allow some flexibility but flag significant deviations
        max_expected_lag = expected_lag + timedelta(days=3)  # 3-day buffer
        if actual_lag > max_expected_lag:
            issues.append(f"Data lag {actual_lag.days} days exceeds expected {field_config.lag_days} days for {field_config.name}")

        # Check for reasonable value ranges based on field type
        if field_config.temporal_type == DataType.PRICE and isinstance(value.value, (int, float)):
            price = float(value.value)
            if price <= 0:
                issues.append(f"CRITICAL: Invalid price value: {price} <= 0")
            elif price > 10000:  # Taiwan stock price upper bound
                issues.append(f"Unusually high price value: {price} for Taiwan market")

        return issues

    def _validate_data_quality_with_field_config(self,
                                                value: TemporalValue,
                                                field_config: FinLabField) -> List[str]:
        """Validate data quality with field-specific configuration."""
        issues = []

        # Check for null values
        if value.value is None:
            if field_config.required:
                issues.append(f"CRITICAL: Null value for required field {field_config.name}")
            else:
                issues.append(f"Null value for optional field {field_config.name}")
            return issues

        # Type validation
        expected_type = field_config.data_type
        if expected_type == "float" and not isinstance(value.value, (int, float, Decimal)):
            issues.append(f"CRITICAL: Expected float for {field_config.name}, got {type(value.value).__name__}")
        elif expected_type == "int" and not isinstance(value.value, int):
            issues.append(f"CRITICAL: Expected int for {field_config.name}, got {type(value.value).__name__}")

        # Field-specific validation rules
        if field_config.validation_rules:
            for rule_name, rule_value in field_config.validation_rules.items():
                if rule_name == "min_value" and isinstance(value.value, (int, float)):
                    if float(value.value) < rule_value:
                        issues.append(f"Value {value.value} below minimum {rule_value} for {field_config.name}")
                elif rule_name == "max_value" and isinstance(value.value, (int, float)):
                    if float(value.value) > rule_value:
                        issues.append(f"Value {value.value} above maximum {rule_value} for {field_config.name}")

        return issues

    def _store_values_with_monitoring(self,
                                    values: List[TemporalValue],
                                    symbol: str,
                                    data_type: DataType) -> Tuple[int, int]:
        """Store values with performance monitoring."""
        if not values:
            return 0, 0

        if self.enable_monitoring:
            self.monitor.performance_tracker.start_timer(f"store_values_{data_type.value}")

        try:
            new_count, updated_count = self._store_values_efficiently(values, symbol, data_type)

            if self.enable_monitoring:
                self.monitor.performance_tracker.record_counter(
                    "values_stored",
                    len(values),
                    {"symbol": symbol, "data_type": data_type.value}
                )

            return new_count, updated_count

        finally:
            if self.enable_monitoring:
                self.monitor.performance_tracker.end_timer(
                    f"store_values_{data_type.value}",
                    {"symbol": symbol, "count": len(values)}
                )

    def _update_checkpoint_with_validation(self,
                                         symbol: str,
                                         data_type: DataType,
                                         last_date: date) -> None:
        """Update checkpoint with validation and monitoring."""
        try:
            checkpoint_key = (symbol, data_type)

            # Generate hash for data integrity validation
            hash_data = f"{symbol}_{data_type.value}_{last_date.isoformat()}"
            hash_value = hashlib.md5(hash_data.encode()).hexdigest()

            checkpoint = DataCheckpoint(
                symbol=symbol,
                data_type=data_type,
                last_update_date=last_date,
                last_update_timestamp=datetime.utcnow(),
                hash_value=hash_value,
                version=2  # Enhanced checkpoint version
            )

            self.checkpoints[checkpoint_key] = checkpoint

            logger.debug(f"Updated checkpoint for {symbol} {data_type} to {last_date}")

        except Exception as e:
            logger.error(f"Failed to update checkpoint for {symbol} {data_type}: {e}")

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status including all enhancements."""
        base_stats = self.get_performance_stats()

        # Add enhanced metrics
        enhanced_stats = {
            "dataset_config": {
                "total_fields": len(self.field_configs),
                "field_types": {
                    "etl": len([f for f in self.field_configs.values() if f.dataset_type.value == "etl"]),
                    "financial_statement": len([f for f in self.field_configs.values() if f.dataset_type.value == "financial_statement"]),
                    "fundamental_features": len([f for f in self.field_configs.values() if f.dataset_type.value == "fundamental_features"])
                }
            },
            "validation": {
                "enabled": self.enable_validation,
                "reports_generated": len(self.validation_reports),
                "last_report_date": self.validation_reports[-1].validation_date.isoformat() if self.validation_reports else None
            },
            "monitoring": {
                "enabled": self.enable_monitoring,
                "performance_history_size": len(self.performance_history)
            },
            "error_recovery": {
                "retry_config": {
                    "max_attempts": self.retry_config.max_attempts,
                    "base_delay": self.retry_config.base_delay,
                    "max_delay": self.retry_config.max_delay
                },
                "error_patterns": len(self.error_recovery.error_history)
            },
            "checkpoints": {
                "file_path": str(self.checkpoint_file),
                "persistent_storage": self.checkpoint_file.exists()
            }
        }

        return {**base_stats, **enhanced_stats}
    
    def create_daily_update_request(self,
                                   symbols: List[str],
                                   target_date: Optional[date] = None) -> UpdateRequest:
        """Create standard daily update request."""
        if target_date is None:
            target_date = date.today()
        
        # For daily updates, check previous trading day
        if not is_taiwan_trading_day(target_date):
            target_date = get_previous_trading_day(target_date)
        
        return UpdateRequest(
            symbols=symbols,
            data_types=[DataType.PRICE, DataType.VOLUME, DataType.MARKET_DATA],
            start_date=target_date,
            end_date=target_date,
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.HIGH,
            validate_consistency=True
        )
    
    def create_fundamental_update_request(self,
                                         symbols: List[str],
                                         lookback_days: int = 90) -> UpdateRequest:
        """Create fundamental data update request."""
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        return UpdateRequest(
            symbols=symbols,
            data_types=[DataType.FUNDAMENTAL, DataType.CORPORATE_ACTION],
            start_date=start_date,
            end_date=end_date,
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.MEDIUM,
            validate_consistency=True
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.start_background_processing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_background_processing()


# Utility functions for update management

def create_market_open_update_schedule() -> List[UpdateRequest]:
    """Create update schedule for Taiwan market open."""
    # Taiwan market opens at 9:00 AM
    symbols = ["2330", "2317", "2454", "2881", "6505"]  # Top 5 Taiwan stocks
    
    return [
        UpdateRequest(
            symbols=symbols,
            data_types=[DataType.PRICE, DataType.VOLUME],
            start_date=date.today(),
            end_date=date.today(),
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.CRITICAL
        )
    ]


def create_eod_update_schedule() -> List[UpdateRequest]:
    """Create end-of-day update schedule."""
    # Get all active symbols (simplified)
    symbols = ["2330", "2317", "2454", "2881", "6505", "2412", "2382", "2308"]
    
    return [
        UpdateRequest(
            symbols=symbols,
            data_types=[DataType.PRICE, DataType.VOLUME, DataType.MARKET_DATA],
            start_date=date.today(),
            end_date=date.today(),
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.HIGH
        ),
        UpdateRequest(
            symbols=symbols,
            data_types=[DataType.FUNDAMENTAL],
            start_date=date.today() - timedelta(days=7),
            end_date=date.today(),
            mode=UpdateMode.INCREMENTAL,
            priority=UpdatePriority.LOW
        )
    ]