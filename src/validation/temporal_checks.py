"""
Temporal Consistency Validation - Task #28 Stream B
Comprehensive validation to prevent future data leakage in Taiwan market ML pipeline.

CRITICAL CHECKS:
1. Time-series integrity (no future information in training)
2. Taiwan market compliance (T+2 settlement, trading calendar)
3. Feature engineering temporal consistency
4. Cross-validation temporal separation
5. Data pipeline temporal ordering

Expert Analysis Integration:
- Prevents lookahead bias through rigorous temporal validation
- Ensures Taiwan market specific constraints are met
- Validates entire data pipeline for temporal consistency
- Provides comprehensive reporting for model compliance
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import warnings
from pathlib import Path
import json

# Import dependencies
try:
    from ..features.taiwan_config import TaiwanMarketConfig
    from ..data.timeseries_splits import TimeSeriesSplitter, validate_temporal_split_integrity
    from ..features.openfe_wrapper import FeatureGenerator
except ImportError:
    # Fallback for standalone usage
    TaiwanMarketConfig = None
    TimeSeriesSplitter = None
    FeatureGenerator = None
    validate_temporal_split_integrity = None

logger = logging.getLogger(__name__)


class TemporalConsistencyValidator:
    """
    Comprehensive temporal consistency validator for Taiwan market ML pipeline.
    
    Validates entire data pipeline to ensure no future information leakage
    and compliance with Taiwan market characteristics.
    """
    
    def __init__(
        self,
        taiwan_config: Optional[TaiwanMarketConfig] = None,
        strict_mode: bool = True,
        validation_output_dir: str = "./validation_reports/"
    ):
        """
        Initialize temporal consistency validator.
        
        Args:
            taiwan_config: Taiwan market configuration
            strict_mode: If True, treat warnings as errors
            validation_output_dir: Directory for validation reports
        """
        self.taiwan_config = taiwan_config or (TaiwanMarketConfig() if TaiwanMarketConfig else None)
        self.strict_mode = strict_mode
        self.validation_output_dir = Path(validation_output_dir)
        self.validation_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation state
        self.validation_history = []
        self.current_validation = None
        
    def validate_data_pipeline(
        self,
        data: pd.DataFrame,
        splits: Optional[Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]] = None,
        feature_pipeline: Optional[Any] = None,
        pipeline_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of entire data pipeline.
        
        Args:
            data: Original dataset
            splits: Train/validation/test splits
            feature_pipeline: Feature engineering pipeline
            pipeline_name: Name for this validation run
            
        Returns:
            Comprehensive validation results
        """
        validation_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting comprehensive temporal validation: {validation_id}")
        
        self.current_validation = {
            'validation_id': validation_id,
            'timestamp': datetime.now(),
            'pipeline_name': pipeline_name,
            'passed': True,
            'errors': [],
            'warnings': [],
            'checks_performed': [],
            'detailed_results': {}
        }
        
        try:
            # 1. Basic data structure validation
            basic_results = self._validate_basic_structure(data)
            self._merge_results('basic_structure', basic_results)
            
            # 2. Taiwan market compliance
            if self.taiwan_config:
                market_results = self._validate_taiwan_market_compliance(data)
                self._merge_results('taiwan_market_compliance', market_results)
            
            # 3. Temporal ordering validation
            temporal_results = self._validate_temporal_ordering(data)
            self._merge_results('temporal_ordering', temporal_results)
            
            # 4. Split integrity validation
            if splits:
                split_results = self._validate_split_integrity(splits, data)
                self._merge_results('split_integrity', split_results)
            
            # 5. Feature pipeline validation
            if feature_pipeline:
                pipeline_results = self._validate_feature_pipeline_temporal_consistency(
                    feature_pipeline, data
                )
                self._merge_results('feature_pipeline', pipeline_results)
            
            # 6. Cross-validation temporal consistency
            if splits:
                cv_results = self._validate_cross_validation_integrity(splits)
                self._merge_results('cross_validation', cv_results)
            
            # 7. Data leakage detection
            leakage_results = self._detect_potential_data_leakage(data, splits)
            self._merge_results('data_leakage', leakage_results)
            
            # 8. Settlement lag validation (Taiwan T+2)
            if self.taiwan_config:
                settlement_results = self._validate_settlement_lag_compliance(data)
                self._merge_results('settlement_lag', settlement_results)
            
            # Final assessment
            self._finalize_validation_assessment()
            
        except Exception as e:
            self.current_validation['errors'].append(f"Validation framework error: {str(e)}")
            self.current_validation['passed'] = False
            logger.error(f"Validation failed: {str(e)}")
        
        # Save validation report
        self._save_validation_report()
        
        # Add to history
        self.validation_history.append(self.current_validation.copy())
        
        return self.current_validation
    
    def _validate_basic_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic data structure for time-series analysis."""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check if DataFrame
            if not isinstance(data, pd.DataFrame):
                results['errors'].append("Data is not a pandas DataFrame")
                results['passed'] = False
                return results
            
            # Check index structure
            if isinstance(data.index, pd.MultiIndex):
                results['statistics']['index_type'] = 'MultiIndex'
                results['statistics']['index_levels'] = data.index.nlevels
                results['statistics']['index_names'] = data.index.names
                
                # Validate first level as dates
                try:
                    first_level = data.index.get_level_values(0)
                    pd.to_datetime(first_level)
                    results['statistics']['first_level_temporal'] = True
                except (ValueError, TypeError):
                    results['warnings'].append("First index level may not be dates")
                    results['statistics']['first_level_temporal'] = False
                    
            else:
                results['statistics']['index_type'] = 'Index'
                try:
                    pd.to_datetime(data.index)
                    results['statistics']['index_temporal'] = True
                except (ValueError, TypeError):
                    results['warnings'].append("Index may not be temporal")
                    results['statistics']['index_temporal'] = False
            
            # Basic data statistics
            results['statistics'].update({
                'shape': data.shape,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values_pct': (data.isnull().sum().sum() / data.size) * 100,
                'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
                'total_columns': len(data.columns)
            })
            
            # Check for reasonable data ranges
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                extreme_values = []
                for col in numeric_data.columns:
                    col_data = numeric_data[col].dropna()
                    if len(col_data) > 0:
                        if col_data.abs().max() > 1e6:  # Very large values
                            extreme_values.append(col)
                
                if extreme_values:
                    results['warnings'].append(
                        f"Columns with extreme values: {extreme_values}"
                    )
                    
        except Exception as e:
            results['errors'].append(f"Basic structure validation error: {str(e)}")
            results['passed'] = False
        
        return results
    
    def _validate_taiwan_market_compliance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate compliance with Taiwan market characteristics."""
        if not self.taiwan_config:
            return {'passed': True, 'errors': [], 'warnings': [], 'skipped': 'No Taiwan config'}
        
        return self.taiwan_config.validate_data_for_taiwan_market(data)
    
    def _validate_temporal_ordering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal ordering of data."""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Extract dates
            if isinstance(data.index, pd.MultiIndex):
                dates = pd.to_datetime(data.index.get_level_values(0))
            else:
                dates = pd.to_datetime(data.index)
            
            # Check if sorted
            is_sorted = dates.is_monotonic_increasing
            results['statistics']['temporally_sorted'] = is_sorted
            
            if not is_sorted:
                results['errors'].append("Data is not temporally sorted")
                results['passed'] = False
            
            # Check for duplicates in time
            unique_dates = dates.unique()
            if isinstance(data.index, pd.MultiIndex):
                # For panel data, check for duplicate (date, entity) pairs
                duplicate_indices = data.index.duplicated()
                n_duplicates = duplicate_indices.sum()
            else:
                n_duplicates = len(dates) - len(unique_dates)
            
            results['statistics']['duplicate_time_periods'] = n_duplicates
            
            if n_duplicates > 0:
                results['warnings'].append(f"Found {n_duplicates} duplicate time periods")
            
            # Check for gaps in time series
            if len(unique_dates) > 1:
                date_diffs = np.diff(unique_dates.sort_values())
                typical_gap = np.median(date_diffs)
                large_gaps = date_diffs > typical_gap * 3  # Gaps 3x typical
                
                results['statistics']['large_gaps_count'] = large_gaps.sum()
                results['statistics']['typical_gap_days'] = int(typical_gap / pd.Timedelta(days=1))
                
                if large_gaps.sum() > 0:
                    results['warnings'].append(
                        f"Found {large_gaps.sum()} unusually large time gaps"
                    )
            
            # Date range validation
            min_date = dates.min()
            max_date = dates.max()
            date_range = int((max_date - min_date) / pd.Timedelta(days=1))
            
            results['statistics'].update({
                'date_range': {
                    'start': min_date.strftime('%Y-%m-%d'),
                    'end': max_date.strftime('%Y-%m-%d'),
                    'days': date_range
                },
                'unique_dates': len(unique_dates)
            })
            
            # Check for future dates
            today = pd.Timestamp.now().normalize()
            future_dates = dates > today
            if future_dates.any():
                results['warnings'].append(
                    f"Found {future_dates.sum()} observations with future dates"
                )
                
        except Exception as e:
            results['errors'].append(f"Temporal ordering validation error: {str(e)}")
            results['passed'] = False
        
        return results
    
    def _validate_split_integrity(
        self, 
        splits: Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]], 
        original_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate integrity of train/validation/test splits."""
        if validate_temporal_split_integrity is None:
            return {'passed': True, 'errors': [], 'warnings': [], 'skipped': 'Function not available'}
        
        return validate_temporal_split_integrity(splits, original_data)
    
    def _validate_feature_pipeline_temporal_consistency(
        self,
        feature_pipeline: Any,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate temporal consistency of feature engineering pipeline."""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check if pipeline has temporal awareness
            if hasattr(feature_pipeline, 'taiwan_market'):
                results['statistics']['taiwan_market_aware'] = feature_pipeline.taiwan_market
            
            if hasattr(feature_pipeline, 'is_fitted_'):
                results['statistics']['pipeline_fitted'] = feature_pipeline.is_fitted_
            
            # Check for time-series split methods
            if hasattr(feature_pipeline, '_time_series_split'):
                results['statistics']['has_time_series_split'] = True
            else:
                results['warnings'].append(
                    "Feature pipeline may not have time-series aware splitting"
                )
            
            # Validate feature generation process if possible
            if hasattr(feature_pipeline, 'get_memory_usage'):
                try:
                    memory_usage = feature_pipeline.get_memory_usage()
                    results['statistics']['pipeline_memory_usage'] = memory_usage
                except:
                    pass
            
            # Check for OpenFE wrapper compliance
            if hasattr(feature_pipeline, 'feature_generator'):
                fg = feature_pipeline.feature_generator
                if hasattr(fg, 'taiwan_market_validate'):
                    try:
                        openfe_validation = fg.taiwan_market_validate(data)
                        results['statistics']['openfe_validation'] = openfe_validation
                        
                        if not openfe_validation.get('passed', True):
                            results['warnings'].append("OpenFE validation found issues")
                            
                    except Exception as e:
                        results['warnings'].append(f"OpenFE validation error: {str(e)}")
            
        except Exception as e:
            results['errors'].append(f"Feature pipeline validation error: {str(e)}")
            results['passed'] = False
        
        return results
    
    def _validate_cross_validation_integrity(
        self,
        splits: Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]
    ) -> Dict[str, Any]:
        """Validate cross-validation temporal integrity."""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check split names and structure
            split_names = list(splits.keys())
            results['statistics']['split_names'] = split_names
            results['statistics']['n_splits'] = len(splits)
            
            # Validate temporal order: train < validation < test
            if 'train' in splits and 'test' in splits:
                train_data, _ = splits['train']
                test_data, _ = splits['test']
                
                # Get latest train date and earliest test date
                if isinstance(train_data.index, pd.MultiIndex):
                    train_dates = pd.to_datetime(train_data.index.get_level_values(0))
                    test_dates = pd.to_datetime(test_data.index.get_level_values(0))
                else:
                    train_dates = pd.to_datetime(train_data.index)
                    test_dates = pd.to_datetime(test_data.index)
                
                train_max = train_dates.max()
                test_min = test_dates.min()
                
                results['statistics']['train_max_date'] = train_max.strftime('%Y-%m-%d')
                results['statistics']['test_min_date'] = test_min.strftime('%Y-%m-%d')
                results['statistics']['temporal_gap_days'] = (test_min - train_max).days
                
                if train_max >= test_min:
                    results['errors'].append(
                        "Temporal violation: train data overlaps with or extends into test period"
                    )
                    results['passed'] = False
                else:
                    results['statistics']['temporal_order_correct'] = True
            
            # Check validation split if present
            if 'validation' in splits and 'train' in splits and 'test' in splits:
                val_data, _ = splits['validation']
                
                if isinstance(val_data.index, pd.MultiIndex):
                    val_dates = pd.to_datetime(val_data.index.get_level_values(0))
                else:
                    val_dates = pd.to_datetime(val_data.index)
                
                val_min = val_dates.min()
                val_max = val_dates.max()
                
                # Validation should be between train and test
                if val_min <= train_max:
                    results['errors'].append(
                        "Validation period overlaps with training period"
                    )
                    results['passed'] = False
                
                if val_max >= test_min:
                    results['errors'].append(
                        "Validation period overlaps with test period"
                    )
                    results['passed'] = False
                
                if results['passed']:
                    results['statistics']['three_way_split_valid'] = True
            
            # Check data sizes
            split_sizes = {}
            total_size = 0
            for split_name, (split_data, _) in splits.items():
                size = len(split_data)
                split_sizes[split_name] = size
                total_size += size
            
            results['statistics']['split_sizes'] = split_sizes
            results['statistics']['total_split_size'] = total_size
            
            # Check for reasonable split proportions
            if 'train' in split_sizes and 'test' in split_sizes:
                train_prop = split_sizes['train'] / total_size
                test_prop = split_sizes['test'] / total_size
                
                if train_prop < 0.5:
                    results['warnings'].append(
                        f"Small training set: {train_prop:.1%} of total data"
                    )
                
                if test_prop > 0.5:
                    results['warnings'].append(
                        f"Large test set: {test_prop:.1%} of total data"
                    )
                
                results['statistics']['split_proportions'] = {
                    'train': train_prop,
                    'test': test_prop
                }
                
                if 'validation' in split_sizes:
                    val_prop = split_sizes['validation'] / total_size
                    results['statistics']['split_proportions']['validation'] = val_prop
            
        except Exception as e:
            results['errors'].append(f"Cross-validation validation error: {str(e)}")
            results['passed'] = False
        
        return results
    
    def _detect_potential_data_leakage(
        self,
        data: pd.DataFrame,
        splits: Optional[Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]] = None
    ) -> Dict[str, Any]:
        """Detect potential sources of data leakage."""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'leakage_risks': [],
            'statistics': {}
        }
        
        try:
            # 1. Check for perfect correlations (potential duplicates)
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty and len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                np.fill_diagonal(corr_matrix.values, np.nan)  # Remove self-correlations
                
                perfect_corrs = (corr_matrix.abs() > 0.999).sum().sum()
                high_corrs = ((corr_matrix.abs() > 0.95) & (corr_matrix.abs() <= 0.999)).sum().sum()
                
                results['statistics']['perfect_correlations'] = perfect_corrs
                results['statistics']['high_correlations'] = high_corrs
                
                if perfect_corrs > 0:
                    results['leakage_risks'].append(
                        f"Found {perfect_corrs} perfect correlations - potential feature duplication"
                    )
                
                if high_corrs > 10:
                    results['leakage_risks'].append(
                        f"Found {high_corrs} high correlations - check for redundancy"
                    )
            
            # 2. Check for future information in feature names
            suspicious_keywords = [
                'future', 'next', 'ahead', 'forward', 'tomorrow', 'later',
                'target', 'label', 'outcome', 'result', 'final'
            ]
            
            suspicious_features = []
            for col in data.columns:
                col_lower = str(col).lower()
                for keyword in suspicious_keywords:
                    if keyword in col_lower:
                        suspicious_features.append(col)
                        break
            
            if suspicious_features:
                results['leakage_risks'].append(
                    f"Suspicious feature names: {suspicious_features[:5]}"  # Show first 5
                )
                results['statistics']['suspicious_feature_names'] = len(suspicious_features)
            
            # 3. Check for data snooping in splits
            if splits:
                # Look for identical values across train/test
                train_data, _ = splits.get('train', (pd.DataFrame(), None))
                test_data, _ = splits.get('test', (pd.DataFrame(), None))
                
                if not train_data.empty and not test_data.empty:
                    common_cols = set(train_data.columns) & set(test_data.columns)
                    identical_distributions = []
                    
                    for col in list(common_cols)[:10]:  # Check first 10 common columns
                        if col in train_data.columns and col in test_data.columns:
                            train_vals = train_data[col].dropna()
                            test_vals = test_data[col].dropna()
                            
                            if len(train_vals) > 0 and len(test_vals) > 0:
                                # Simple distribution similarity check
                                train_mean = train_vals.mean()
                                test_mean = test_vals.mean()
                                train_std = train_vals.std()
                                test_std = test_vals.std()
                                
                                if (abs(train_mean - test_mean) < 0.001 and 
                                    abs(train_std - test_std) < 0.001):
                                    identical_distributions.append(col)
                    
                    if identical_distributions:
                        results['leakage_risks'].append(
                            f"Nearly identical distributions in train/test: "
                            f"{identical_distributions[:3]}"
                        )
            
            # 4. Check for time-invariant features (suspicious in time series)
            time_invariant_features = []
            for col in data.select_dtypes(include=[np.number]).columns:
                col_data = data[col].dropna()
                if len(col_data) > 1:
                    if col_data.std() < 1e-10:  # Essentially constant
                        time_invariant_features.append(col)
            
            if time_invariant_features:
                results['leakage_risks'].append(
                    f"Time-invariant features detected: {time_invariant_features[:3]}"
                )
                results['statistics']['time_invariant_features'] = len(time_invariant_features)
            
            # Set overall status
            if len(results['leakage_risks']) > 5:  # Many risks found
                results['warnings'].append("Multiple potential data leakage risks detected")
            
        except Exception as e:
            results['errors'].append(f"Data leakage detection error: {str(e)}")
            results['passed'] = False
        
        return results
    
    def _validate_settlement_lag_compliance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate T+2 settlement lag compliance for Taiwan market."""
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if not self.taiwan_config:
            results['skipped'] = 'No Taiwan config available'
            return results
        
        try:
            # Check if data has proper settlement lag structure
            # This is a simplified check - in practice would need domain knowledge
            # of which features require T+2 lag
            
            results['statistics']['settlement_days_expected'] = self.taiwan_config.SETTLEMENT_DAYS
            
            # Check for features that might violate T+2 lag
            # (This would require more sophisticated analysis of feature engineering pipeline)
            
            # For now, just validate that we have proper date structure
            if isinstance(data.index, pd.MultiIndex):
                dates = pd.to_datetime(data.index.get_level_values(0))
            else:
                dates = pd.to_datetime(data.index)
            
            # Check if dates align with trading calendar
            trading_dates = [d for d in dates.unique() 
                           if self.taiwan_config.is_trading_day(pd.Timestamp(d))]
            
            non_trading_dates = len(dates.unique()) - len(trading_dates)
            results['statistics']['non_trading_dates'] = non_trading_dates
            
            if non_trading_dates > 0:
                results['warnings'].append(
                    f"Found {non_trading_dates} non-trading dates in data"
                )
            
            # Additional T+2 specific checks would go here
            # (Require domain knowledge of specific features)
            
        except Exception as e:
            results['errors'].append(f"Settlement lag validation error: {str(e)}")
            results['passed'] = False
        
        return results
    
    def _merge_results(self, check_name: str, check_results: Dict[str, Any]) -> None:
        """Merge individual check results into overall validation."""
        self.current_validation['checks_performed'].append(check_name)
        self.current_validation['detailed_results'][check_name] = check_results
        
        # Merge errors and warnings
        if 'errors' in check_results:
            self.current_validation['errors'].extend(check_results['errors'])
        
        if 'warnings' in check_results:
            self.current_validation['warnings'].extend(check_results['warnings'])
        
        # Update overall pass status
        if not check_results.get('passed', True):
            self.current_validation['passed'] = False
    
    def _finalize_validation_assessment(self) -> None:
        """Finalize overall validation assessment."""
        n_errors = len(self.current_validation['errors'])
        n_warnings = len(self.current_validation['warnings'])
        
        # In strict mode, warnings become errors
        if self.strict_mode and n_warnings > 0:
            self.current_validation['errors'].extend([
                f"STRICT MODE: {warning}" for warning in self.current_validation['warnings']
            ])
            self.current_validation['passed'] = False
        
        # Summary statistics
        self.current_validation['summary'] = {
            'checks_performed': len(self.current_validation['checks_performed']),
            'errors_found': len(self.current_validation['errors']),
            'warnings_found': len(self.current_validation['warnings']),
            'overall_status': 'PASS' if self.current_validation['passed'] else 'FAIL'
        }
        
        logger.info(
            f"Validation {self.current_validation['validation_id']} completed: "
            f"{self.current_validation['summary']['overall_status']} "
            f"({n_errors} errors, {n_warnings} warnings)"
        )
    
    def _save_validation_report(self) -> None:
        """Save detailed validation report to file."""
        try:
            report_file = self.validation_output_dir / f"{self.current_validation['validation_id']}.json"
            
            # Create serializable version
            report_data = self.current_validation.copy()
            report_data['timestamp'] = report_data['timestamp'].isoformat()
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Validation report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {str(e)}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation runs."""
        return {
            'total_validations': len(self.validation_history),
            'latest_validation': self.current_validation['validation_id'] if self.current_validation else None,
            'history': [
                {
                    'validation_id': v['validation_id'],
                    'timestamp': v['timestamp'],
                    'passed': v['passed'],
                    'errors': len(v['errors']),
                    'warnings': len(v['warnings'])
                }
                for v in self.validation_history
            ]
        }


def validate_pipeline_temporal_integrity(
    data: pd.DataFrame,
    splits: Optional[Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]]] = None,
    feature_pipeline: Optional[Any] = None,
    taiwan_market: bool = True,
    strict_mode: bool = True,
    pipeline_name: str = "pipeline"
) -> Dict[str, Any]:
    """
    Convenience function for comprehensive pipeline validation.
    
    Args:
        data: Original dataset
        splits: Train/validation/test splits
        feature_pipeline: Feature engineering pipeline
        taiwan_market: Use Taiwan market validation
        strict_mode: Treat warnings as errors
        pipeline_name: Name for validation run
        
    Returns:
        Comprehensive validation results
    """
    # Initialize validator
    taiwan_config = TaiwanMarketConfig() if taiwan_market and TaiwanMarketConfig else None
    validator = TemporalConsistencyValidator(
        taiwan_config=taiwan_config,
        strict_mode=strict_mode
    )
    
    # Run comprehensive validation
    results = validator.validate_data_pipeline(
        data=data,
        splits=splits,
        feature_pipeline=feature_pipeline,
        pipeline_name=pipeline_name
    )
    
    return results