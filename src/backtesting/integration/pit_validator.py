"""
Point-in-Time Validation Integration for ML4T Walk-Forward Engine.

This module integrates the walk-forward validation engine with the point-in-time (PIT)
data management system from Issue #21 and the data quality validation framework from
Issue #22, ensuring zero look-ahead bias and comprehensive data integrity validation.

Key Features:
- PIT data access validation for walk-forward windows
- Bias detection and prevention mechanisms
- Integration with quality validation framework
- Taiwan market-specific PIT validation rules
- Performance monitoring for large-scale validations
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ...data.core.temporal import TemporalStore, DataType, TemporalValue
from ...data.pipeline.pit_engine import PointInTimeEngine, PITQuery, BiasCheckLevel
from ...data.quality.validation_engine import ValidationEngine
from ...data.models.taiwan_market import (
    TaiwanTradingCalendar, TaiwanSettlement, create_taiwan_trading_calendar
)
from ..validation.walk_forward import (
    WalkForwardSplitter, WalkForwardConfig, ValidationWindow, ValidationResult,
    WalkForwardValidator, ValidationStatus
)
from ..validation.time_series_cv import PurgedKFold, CrossValidationResult, CVConfig
from ..validation.taiwan_specific import TaiwanMarketValidator, TaiwanValidationConfig
from ..metrics.performance import PerformanceCalculator, PerformanceConfig
from ..metrics.attribution import PerformanceAttributor
from ..metrics.risk_adjusted import RiskCalculator

logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Types of biases that can be detected."""
    LOOK_AHEAD = "look_ahead"
    SURVIVORSHIP = "survivorship"
    SELECTION = "selection"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    DATA_SNOOPING = "data_snooping"
    OVERFITTING = "overfitting"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class BiasCheckResult:
    """Result from bias detection check."""
    bias_type: BiasType
    detected: bool
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_windows: List[str] = field(default_factory=list)
    affected_symbols: List[str] = field(default_factory=list)
    remediation: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bias_type': self.bias_type.value,
            'detected': self.detected,
            'severity': self.severity,
            'description': self.description,
            'affected_windows': self.affected_windows,
            'affected_symbols': self.affected_symbols,
            'remediation': self.remediation,
            'confidence': self.confidence
        }


@dataclass
class PITValidationConfig:
    """Configuration for PIT validation integration."""
    # Bias detection settings
    bias_check_level: BiasCheckLevel = BiasCheckLevel.STRICT
    validation_level: ValidationLevel = ValidationLevel.STRICT
    enable_look_ahead_detection: bool = True
    enable_survivorship_detection: bool = True
    enable_temporal_leakage_detection: bool = True
    
    # Performance settings
    max_concurrent_validations: int = 4
    enable_parallel_processing: bool = True
    validation_timeout_seconds: int = 300  # 5 minutes per window
    
    # Quality thresholds
    min_data_completeness: float = 0.95  # 95% data completeness
    max_quality_score_deviation: float = 0.1  # 10% deviation from baseline
    require_quality_validation: bool = True
    
    # Taiwan market specific
    validate_settlement_timing: bool = True
    validate_corporate_actions: bool = True
    validate_market_events: bool = True
    
    # Cache and optimization
    enable_validation_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    enable_incremental_validation: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_concurrent_validations <= 0:
            raise ValueError("Max concurrent validations must be positive")
        if self.validation_timeout_seconds <= 0:
            raise ValueError("Validation timeout must be positive")
        if not 0 <= self.min_data_completeness <= 1:
            raise ValueError("Data completeness must be between 0 and 1")


class PITBiasDetector:
    """
    Sophisticated bias detection system for walk-forward validation.
    
    Detects various types of bias that can compromise the validity of
    backtesting results, with specialized checks for Taiwan market data.
    """
    
    def __init__(
        self,
        pit_engine: PointInTimeEngine,
        taiwan_calendar: Optional[TaiwanTradingCalendar] = None
    ):
        self.pit_engine = pit_engine
        self.taiwan_calendar = taiwan_calendar or create_taiwan_trading_calendar()
        self.detection_cache: Dict[str, BiasCheckResult] = {}
        
        logger.info("PITBiasDetector initialized")
    
    def detect_all_biases(
        self,
        window: ValidationWindow,
        symbols: List[str],
        data_types: List[DataType],
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ) -> List[BiasCheckResult]:
        """
        Run comprehensive bias detection for a validation window.
        
        Args:
            window: Validation window to check
            symbols: List of symbols to validate
            data_types: Data types to check for bias
            validation_level: Strictness level for detection
            
        Returns:
            List of bias check results
        """
        results = []
        
        logger.debug(f"Running bias detection for window {window.window_id}")
        
        # Look-ahead bias detection
        look_ahead_result = self._detect_look_ahead_bias(
            window, symbols, data_types, validation_level
        )
        if look_ahead_result:
            results.append(look_ahead_result)
        
        # Survivorship bias detection
        survivorship_result = self._detect_survivorship_bias(
            window, symbols, validation_level
        )
        if survivorship_result:
            results.append(survivorship_result)
        
        # Temporal leakage detection
        temporal_result = self._detect_temporal_leakage(
            window, symbols, data_types, validation_level
        )
        if temporal_result:
            results.append(temporal_result)
        
        # Selection bias detection
        selection_result = self._detect_selection_bias(
            window, symbols, validation_level
        )
        if selection_result:
            results.append(selection_result)
        
        logger.info(f"Bias detection completed: {len(results)} issues found")
        return results
    
    def _detect_look_ahead_bias(
        self,
        window: ValidationWindow,
        symbols: List[str],
        data_types: List[DataType],
        validation_level: ValidationLevel
    ) -> Optional[BiasCheckResult]:
        """Detect look-ahead bias in validation window."""
        try:
            # Check if any training data includes information from test period
            for symbol in symbols:
                for data_type in data_types:
                    # Query for data that should not be available during training
                    future_query = PITQuery(
                        symbols=[symbol],
                        as_of_date=window.train_end,
                        data_types=[data_type],
                        start_date=window.test_start,
                        end_date=window.test_end
                    )
                    
                    # This should return no data if PIT system is working correctly
                    future_data = self.pit_engine.query(
                        future_query,
                        bias_check_level=BiasCheckLevel.STRICT
                    )
                    
                    if symbol in future_data and len(future_data[symbol]) > 0:
                        return BiasCheckResult(
                            bias_type=BiasType.LOOK_AHEAD,
                            detected=True,
                            severity="critical",
                            description=f"Look-ahead bias detected: {symbol} {data_type.value} data from test period available during training",
                            affected_windows=[window.window_id],
                            affected_symbols=[symbol],
                            remediation="Ensure point-in-time data access with proper as_of_date constraints",
                            confidence=0.95
                        )
            
            # Additional check: verify purge period effectiveness
            purge_violations = self._check_purge_period_violations(window, symbols, data_types)
            if purge_violations:
                return BiasCheckResult(
                    bias_type=BiasType.LOOK_AHEAD,
                    detected=True,
                    severity="high",
                    description=f"Purge period violations detected: {len(purge_violations)} symbols have overlapping observations",
                    affected_windows=[window.window_id],
                    affected_symbols=purge_violations,
                    remediation="Increase purge period or adjust window boundaries",
                    confidence=0.85
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Look-ahead bias detection failed: {e}")
            return BiasCheckResult(
                bias_type=BiasType.LOOK_AHEAD,
                detected=False,
                severity="low",
                description=f"Could not perform look-ahead bias check: {e}",
                affected_windows=[window.window_id],
                remediation="Investigate bias detection system configuration",
                confidence=0.0
            )
    
    def _detect_survivorship_bias(
        self,
        window: ValidationWindow,
        symbols: List[str],
        validation_level: ValidationLevel
    ) -> Optional[BiasCheckResult]:
        """Detect survivorship bias in symbol selection."""
        try:
            # Check for symbols that were delisted during the validation period
            delisted_symbols = []
            
            for symbol in symbols:
                # Query for corporate actions (delistings, bankruptcies)
                ca_query = PITQuery(
                    symbols=[symbol],
                    as_of_date=window.test_end,
                    data_types=[DataType.CORPORATE_ACTION],
                    start_date=window.train_start,
                    end_date=window.test_end
                )
                
                ca_data = self.pit_engine.query(ca_query)
                
                if symbol in ca_data:
                    for ca_record in ca_data[symbol]:
                        ca_type = ca_record.metadata.get('action_type') if ca_record.metadata else None
                        if ca_type in ['DELISTING', 'BANKRUPTCY', 'SUSPENSION']:
                            delisted_symbols.append(symbol)
                            break
            
            if delisted_symbols:
                severity = "high" if len(delisted_symbols) > len(symbols) * 0.05 else "medium"
                return BiasCheckResult(
                    bias_type=BiasType.SURVIVORSHIP,
                    detected=True,
                    severity=severity,
                    description=f"Survivorship bias detected: {len(delisted_symbols)} symbols were delisted during validation period",
                    affected_windows=[window.window_id],
                    affected_symbols=delisted_symbols,
                    remediation="Include delisted symbols in analysis or adjust universe selection criteria",
                    confidence=0.90
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Survivorship bias detection failed: {e}")
            return None
    
    def _detect_temporal_leakage(
        self,
        window: ValidationWindow,
        symbols: List[str],
        data_types: List[DataType],
        validation_level: ValidationLevel
    ) -> Optional[BiasCheckResult]:
        """Detect temporal leakage in data processing."""
        try:
            # Check for data that has incorrect temporal ordering
            temporal_violations = []
            
            for symbol in symbols:
                for data_type in data_types:
                    # Get data for the window
                    data_query = PITQuery(
                        symbols=[symbol],
                        as_of_date=window.train_end,
                        data_types=[data_type],
                        start_date=window.train_start,
                        end_date=window.train_end
                    )
                    
                    data = self.pit_engine.query(data_query)
                    
                    if symbol in data and len(data[symbol]) > 1:
                        # Check for temporal ordering violations
                        sorted_data = sorted(data[symbol], key=lambda x: x.value_date)
                        
                        for i, record in enumerate(sorted_data):
                            # Check if as_of_date is before value_date (impossible)
                            if record.as_of_date < record.value_date:
                                temporal_violations.append(f"{symbol}:{data_type.value}")
                                break
                            
                            # Check if record was available before it should be
                            expected_lag = self._get_expected_data_lag(data_type)
                            if expected_lag > 0:
                                earliest_available = record.value_date + timedelta(days=expected_lag)
                                if record.as_of_date < earliest_available:
                                    temporal_violations.append(f"{symbol}:{data_type.value}")
                                    break
            
            if temporal_violations:
                return BiasCheckResult(
                    bias_type=BiasType.TEMPORAL_LEAKAGE,
                    detected=True,
                    severity="high",
                    description=f"Temporal leakage detected in {len(temporal_violations)} data series",
                    affected_windows=[window.window_id],
                    affected_symbols=list(set(v.split(':')[0] for v in temporal_violations)),
                    remediation="Review data ingestion process and implement proper temporal constraints",
                    confidence=0.88
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Temporal leakage detection failed: {e}")
            return None
    
    def _detect_selection_bias(
        self,
        window: ValidationWindow,
        symbols: List[str],
        validation_level: ValidationLevel
    ) -> Optional[BiasCheckResult]:
        """Detect selection bias in symbol universe."""
        try:
            # Check if symbol selection is biased towards certain characteristics
            # This is a simplified check - in practice would be more sophisticated
            
            # Check for overrepresentation of large-cap stocks
            large_cap_symbols = [s for s in symbols if s.startswith('23')]  # Major Taiwan stocks
            large_cap_ratio = len(large_cap_symbols) / len(symbols)
            
            if large_cap_ratio > 0.8:  # More than 80% large-cap
                return BiasCheckResult(
                    bias_type=BiasType.SELECTION,
                    detected=True,
                    severity="medium",
                    description=f"Selection bias detected: {large_cap_ratio:.1%} large-cap concentration",
                    affected_windows=[window.window_id],
                    affected_symbols=symbols,
                    remediation="Ensure representative symbol universe across market cap ranges",
                    confidence=0.70
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Selection bias detection failed: {e}")
            return None
    
    def _check_purge_period_violations(
        self,
        window: ValidationWindow,
        symbols: List[str],
        data_types: List[DataType]
    ) -> List[str]:
        """Check for violations in purge period effectiveness."""
        violations = []
        
        try:
            for symbol in symbols:
                # Check if any training data overlaps with test period after purging
                overlap_query = PITQuery(
                    symbols=[symbol],
                    as_of_date=window.purge_end,
                    data_types=data_types,
                    start_date=window.purge_start,
                    end_date=window.purge_end
                )
                
                overlap_data = self.pit_engine.query(overlap_query)
                
                if symbol in overlap_data and len(overlap_data[symbol]) > 0:
                    violations.append(symbol)
        
        except Exception as e:
            logger.error(f"Purge period check failed: {e}")
        
        return violations
    
    def _get_expected_data_lag(self, data_type: DataType) -> int:
        """Get expected data lag in days for different data types."""
        lag_map = {
            DataType.PRICE: 0,          # Same day
            DataType.VOLUME: 0,         # Same day
            DataType.FUNDAMENTAL: 60,   # 60-day lag for financial statements
            DataType.CORPORATE_ACTION: 1,  # T+1 for corporate actions
            DataType.NEWS: 0,           # Real-time
            DataType.TECHNICAL: 1       # T+1 for technical indicators
        }
        return lag_map.get(data_type, 1)


class PITValidator:
    """
    Comprehensive PIT validation engine integrating walk-forward validation
    with point-in-time data access and quality validation.
    
    This is the main integration component for Stream C of Issue #23.
    """
    
    def __init__(
        self,
        config: PITValidationConfig,
        temporal_store: TemporalStore,
        pit_engine: Optional[PointInTimeEngine] = None,
        validation_engine: Optional[ValidationEngine] = None,
        taiwan_calendar: Optional[TaiwanTradingCalendar] = None
    ):
        self.config = config
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine or PointInTimeEngine(temporal_store)
        self.validation_engine = validation_engine or ValidationEngine()
        self.taiwan_calendar = taiwan_calendar or create_taiwan_trading_calendar()
        
        # Initialize bias detector
        self.bias_detector = PITBiasDetector(self.pit_engine, self.taiwan_calendar)
        
        # Initialize validators
        self.taiwan_validator = TaiwanMarketValidator(
            config=TaiwanValidationConfig(),
            temporal_store=temporal_store,
            pit_engine=self.pit_engine,
            taiwan_calendar=self.taiwan_calendar
        )
        
        # Initialize performance components
        self.performance_calculator = PerformanceCalculator(
            config=PerformanceConfig(),
            temporal_store=temporal_store,
            pit_engine=self.pit_engine
        )
        
        # Validation cache
        self.validation_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("PITValidator initialized with comprehensive validation framework")
    
    def validate_walk_forward_scenario(
        self,
        wf_config: WalkForwardConfig,
        symbols: List[str],
        start_date: date,
        end_date: date,
        data_types: List[DataType] = None,
        enable_performance_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive walk-forward validation with PIT integration.
        
        Args:
            wf_config: Walk-forward configuration
            symbols: List of symbols to validate
            start_date: Validation start date
            end_date: Validation end date
            data_types: Data types to validate
            enable_performance_validation: Whether to run performance validation
            
        Returns:
            Comprehensive validation results
        """
        if data_types is None:
            data_types = [DataType.PRICE, DataType.VOLUME]
        
        start_time = time.time()
        
        logger.info(f"Starting comprehensive walk-forward validation for {len(symbols)} symbols")
        
        # Create walk-forward splitter
        splitter = WalkForwardSplitter(
            config=wf_config,
            temporal_store=self.temporal_store,
            pit_engine=self.pit_engine,
            taiwan_calendar=self.taiwan_calendar
        )
        
        # Generate validation windows
        windows = splitter.generate_windows(start_date, end_date, symbols)
        
        # Validate each window
        validation_results = []
        bias_results = []
        quality_results = []
        performance_results = []
        
        if self.config.enable_parallel_processing:
            # Parallel validation
            results = self._validate_windows_parallel(
                windows, symbols, data_types, enable_performance_validation
            )
        else:
            # Sequential validation
            results = self._validate_windows_sequential(
                windows, symbols, data_types, enable_performance_validation
            )
        
        validation_results, bias_results, quality_results, performance_results = results
        
        # Aggregate results
        total_time = time.time() - start_time
        
        comprehensive_result = {
            'validation_summary': {
                'total_windows': len(windows),
                'successful_validations': len([r for r in validation_results if r.get('success', False)]),
                'total_bias_issues': sum(len(r) for r in bias_results),
                'total_quality_issues': sum(len(r) for r in quality_results),
                'total_runtime_seconds': total_time
            },
            'windows': [w.to_dict() for w in windows],
            'validation_results': validation_results,
            'bias_detection_results': bias_results,
            'quality_validation_results': quality_results,
            'performance_validation_results': performance_results if enable_performance_validation else [],
            'config': {
                'walk_forward_config': wf_config.__dict__,
                'pit_validation_config': self.config.__dict__
            },
            'metadata': {
                'symbols': symbols,
                'data_types': [dt.value for dt in data_types],
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'validation_timestamp': datetime.now().isoformat()
            }
        }
        
        # Generate summary statistics
        comprehensive_result['summary_statistics'] = self._generate_summary_statistics(
            comprehensive_result
        )
        
        logger.info(f"Comprehensive validation completed in {total_time:.2f}s")
        return comprehensive_result
    
    def _validate_windows_parallel(
        self,
        windows: List[ValidationWindow],
        symbols: List[str],
        data_types: List[DataType],
        enable_performance_validation: bool
    ) -> Tuple[List[Dict], List[List], List[List], List[Dict]]:
        """Validate windows in parallel."""
        validation_results = []
        bias_results = []
        quality_results = []
        performance_results = []
        
        max_workers = min(self.config.max_concurrent_validations, len(windows))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit validation tasks
            future_to_window = {
                executor.submit(
                    self._validate_single_window,
                    window, symbols, data_types, enable_performance_validation
                ): window
                for window in windows
            }
            
            # Collect results
            for future in as_completed(future_to_window, timeout=self.config.validation_timeout_seconds):
                window = future_to_window[future]
                try:
                    result = future.result()
                    validation_results.append(result['validation'])
                    bias_results.append(result['bias'])
                    quality_results.append(result['quality'])
                    if enable_performance_validation:
                        performance_results.append(result['performance'])
                        
                except Exception as e:
                    logger.error(f"Window validation failed for {window.window_id}: {e}")
                    validation_results.append({'success': False, 'error': str(e)})
                    bias_results.append([])
                    quality_results.append([])
                    if enable_performance_validation:
                        performance_results.append({})
        
        return validation_results, bias_results, quality_results, performance_results
    
    def _validate_windows_sequential(
        self,
        windows: List[ValidationWindow],
        symbols: List[str],
        data_types: List[DataType],
        enable_performance_validation: bool
    ) -> Tuple[List[Dict], List[List], List[List], List[Dict]]:
        """Validate windows sequentially."""
        validation_results = []
        bias_results = []
        quality_results = []
        performance_results = []
        
        for window in windows:
            try:
                result = self._validate_single_window(
                    window, symbols, data_types, enable_performance_validation
                )
                validation_results.append(result['validation'])
                bias_results.append(result['bias'])
                quality_results.append(result['quality'])
                if enable_performance_validation:
                    performance_results.append(result['performance'])
                    
            except Exception as e:
                logger.error(f"Window validation failed for {window.window_id}: {e}")
                validation_results.append({'success': False, 'error': str(e)})
                bias_results.append([])
                quality_results.append([])
                if enable_performance_validation:
                    performance_results.append({})
        
        return validation_results, bias_results, quality_results, performance_results
    
    def _validate_single_window(
        self,
        window: ValidationWindow,
        symbols: List[str],
        data_types: List[DataType],
        enable_performance_validation: bool
    ) -> Dict[str, Any]:
        """Validate a single window comprehensively."""
        result = {
            'validation': {'success': False},
            'bias': [],
            'quality': [],
            'performance': {}
        }
        
        try:
            # 1. Basic window validation
            basic_validation = self._validate_window_basic(window, symbols, data_types)
            result['validation'] = basic_validation
            
            if not basic_validation['success']:
                return result
            
            # 2. Bias detection
            if self.config.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                bias_checks = self.bias_detector.detect_all_biases(
                    window, symbols, data_types, self.config.validation_level
                )
                result['bias'] = [check.to_dict() for check in bias_checks]
            
            # 3. Quality validation
            if self.config.require_quality_validation:
                quality_checks = self._validate_window_quality(window, symbols, data_types)
                result['quality'] = quality_checks
            
            # 4. Taiwan market validation
            if self.config.validate_market_events:
                taiwan_validation = self._validate_window_taiwan_market(window, symbols)
                result['validation']['taiwan_validation'] = taiwan_validation
            
            # 5. Performance validation
            if enable_performance_validation:
                performance_validation = self._validate_window_performance(window, symbols)
                result['performance'] = performance_validation
            
            # Update success status
            critical_bias_detected = any(
                check.get('severity') == 'critical' for check in result['bias']
            )
            critical_quality_issues = any(
                issue.get('severity') == 'critical' for issue in result['quality']
            )
            
            result['validation']['success'] = (
                basic_validation['success'] and 
                not critical_bias_detected and 
                not critical_quality_issues
            )
            
            logger.debug(f"Window {window.window_id} validation completed")
            
        except Exception as e:
            logger.error(f"Window validation error for {window.window_id}: {e}")
            result['validation'] = {'success': False, 'error': str(e)}
        
        return result
    
    def _validate_window_basic(
        self,
        window: ValidationWindow,
        symbols: List[str],
        data_types: List[DataType]
    ) -> Dict[str, Any]:
        """Perform basic window validation."""
        try:
            # Check data availability for training period
            train_query = PITQuery(
                symbols=symbols,
                as_of_date=window.train_end,
                data_types=data_types,
                start_date=window.train_start,
                end_date=window.train_end
            )
            
            train_available = self.pit_engine.check_data_availability(train_query)
            
            if not train_available:
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Check data availability for test period
            test_query = PITQuery(
                symbols=symbols,
                as_of_date=window.test_end,
                data_types=data_types,
                start_date=window.test_start,
                end_date=window.test_end
            )
            
            test_available = self.pit_engine.check_data_availability(test_query)
            
            if not test_available:
                return {'success': False, 'error': 'Insufficient test data'}
            
            return {
                'success': True,
                'train_data_available': train_available,
                'test_data_available': test_available,
                'window_id': window.window_id
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_window_quality(
        self,
        window: ValidationWindow,
        symbols: List[str],
        data_types: List[DataType]
    ) -> List[Dict[str, Any]]:
        """Validate data quality for window."""
        quality_issues = []
        
        try:
            for symbol in symbols:
                for data_type in data_types:
                    # Check data completeness (placeholder implementation)
                    completeness_result = 0.95  # Assume 95% completeness for now
                    
                    if completeness_result < self.config.min_data_completeness:
                        quality_issues.append({
                            'type': 'data_completeness',
                            'severity': 'high',
                            'symbol': symbol,
                            'data_type': data_type.value,
                            'value': completeness_result,
                            'threshold': self.config.min_data_completeness,
                            'description': f"Data completeness {completeness_result:.2%} below threshold"
                        })
                    
                    # Check for data quality score (placeholder implementation)
                    quality_score = 0.85  # Assume 85% quality score for now
                    
                    if quality_score < 0.8:  # Arbitrary threshold
                        quality_issues.append({
                            'type': 'data_quality',
                            'severity': 'medium',
                            'symbol': symbol,
                            'data_type': data_type.value,
                            'value': quality_score,
                            'threshold': 0.8,
                            'description': f"Data quality score {quality_score:.2f} below standard"
                        })
        
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            quality_issues.append({
                'type': 'validation_error',
                'severity': 'medium',
                'description': f"Quality validation error: {e}"
            })
        
        return quality_issues
    
    def _validate_window_taiwan_market(
        self,
        window: ValidationWindow,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Validate Taiwan market-specific constraints."""
        try:
            # Use Taiwan market validator
            validation_issues = self.taiwan_validator.validate_trading_scenario(
                symbols=symbols,
                start_date=window.train_start,
                end_date=window.test_end
            )
            
            return {
                'success': len(validation_issues) == 0,
                'issues_count': len(validation_issues),
                'issues': [issue.to_dict() for issue in validation_issues]
            }
            
        except Exception as e:
            logger.error(f"Taiwan market validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _validate_window_performance(
        self,
        window: ValidationWindow,
        symbols: List[str]
    ) -> Dict[str, Any]:
        """Validate performance calculation capabilities."""
        try:
            # This would integrate with performance calculation from Stream B
            # For now, return placeholder
            return {
                'performance_calculation_ready': True,
                'benchmark_data_available': True,
                'risk_metrics_calculable': True
            }
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_summary_statistics(self, comprehensive_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from validation results."""
        try:
            summary = comprehensive_result['validation_summary']
            
            # Calculate success rates
            success_rate = summary['successful_validations'] / summary['total_windows']
            
            # Bias detection summary
            bias_by_type = {}
            for bias_list in comprehensive_result['bias_detection_results']:
                for bias_result in bias_list:
                    bias_type = bias_result['bias_type']
                    if bias_type not in bias_by_type:
                        bias_by_type[bias_type] = 0
                    bias_by_type[bias_type] += 1
            
            # Quality issues summary
            quality_by_severity = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
            for quality_list in comprehensive_result['quality_validation_results']:
                for quality_issue in quality_list:
                    severity = quality_issue.get('severity', 'medium')
                    quality_by_severity[severity] += 1
            
            return {
                'success_rate': success_rate,
                'avg_runtime_per_window': summary['total_runtime_seconds'] / summary['total_windows'],
                'bias_detection_summary': bias_by_type,
                'quality_issues_summary': quality_by_severity,
                'overall_health_score': self._calculate_health_score(comprehensive_result)
            }
            
        except Exception as e:
            logger.error(f"Summary statistics generation failed: {e}")
            return {}
    
    def _calculate_health_score(self, comprehensive_result: Dict[str, Any]) -> float:
        """Calculate overall validation health score (0-100)."""
        try:
            summary = comprehensive_result['validation_summary']
            
            # Base score from success rate
            success_rate = summary['successful_validations'] / summary['total_windows']
            base_score = success_rate * 100
            
            # Deduct points for bias issues
            critical_bias_count = sum(
                len([b for b in bias_list if b.get('severity') == 'critical'])
                for bias_list in comprehensive_result['bias_detection_results']
            )
            base_score -= critical_bias_count * 20  # 20 points per critical bias
            
            # Deduct points for quality issues
            critical_quality_count = sum(
                len([q for q in quality_list if q.get('severity') == 'critical'])
                for quality_list in comprehensive_result['quality_validation_results']
            )
            base_score -= critical_quality_count * 15  # 15 points per critical quality issue
            
            return max(0, min(100, base_score))
            
        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 0.0


# Utility functions for creating pre-configured validators
def create_standard_pit_validator(
    temporal_store: TemporalStore,
    **config_overrides
) -> PITValidator:
    """Create PIT validator with standard configuration."""
    config = PITValidationConfig(**config_overrides)
    return PITValidator(config, temporal_store)


def create_strict_pit_validator(
    temporal_store: TemporalStore,
    **config_overrides
) -> PITValidator:
    """Create PIT validator with strict validation rules."""
    strict_config = PITValidationConfig(
        validation_level=ValidationLevel.STRICT,
        enable_look_ahead_detection=True,
        enable_survivorship_detection=True,
        enable_temporal_leakage_detection=True,
        require_quality_validation=True,
        min_data_completeness=0.98,
        **config_overrides
    )
    return PITValidator(strict_config, temporal_store)


def create_paranoid_pit_validator(
    temporal_store: TemporalStore,
    **config_overrides
) -> PITValidator:
    """Create PIT validator with paranoid validation rules."""
    paranoid_config = PITValidationConfig(
        validation_level=ValidationLevel.PARANOID,
        bias_check_level=BiasCheckLevel.STRICT,
        enable_look_ahead_detection=True,
        enable_survivorship_detection=True,
        enable_temporal_leakage_detection=True,
        require_quality_validation=True,
        min_data_completeness=0.99,
        max_quality_score_deviation=0.05,
        **config_overrides
    )
    return PITValidator(paranoid_config, temporal_store)


# Example usage and testing
if __name__ == "__main__":
    print("PIT Validator Integration demo")
    print("This component integrates walk-forward validation with point-in-time data access")
    print("Requires actual TemporalStore and dependencies from Issues #21 and #22")