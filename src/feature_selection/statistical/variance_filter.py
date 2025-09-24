"""
Variance-Based Feature Filtering

This module implements sophisticated variance thresholding to remove low-information
features while preserving features that capture market dynamics and price movements
in Taiwan stock market data.

Key Features:
- Adaptive variance thresholding based on feature distribution
- Quasi-constant feature detection and removal
- Taiwan market-aware variance validation
- Memory-efficient processing for large feature sets
- Progressive thresholding with information preservation tracking
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, RobustScaler
import psutil

logger = logging.getLogger(__name__)

class VarianceFilter:
    """
    Advanced variance-based feature filtering for Taiwan stock market data.
    
    Implements multiple variance filtering strategies to remove low-information
    features while preserving features that capture meaningful market dynamics.
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.01,
        quasi_constant_threshold: float = 0.001,
        adaptive_threshold: bool = True,
        standardize_before_filter: bool = True,
        scaler_method: str = 'robust',
        memory_limit_gb: float = 8.0,
        preserve_rate: float = 0.85
    ):
        """
        Initialize variance filter.
        
        Args:
            variance_threshold: Minimum variance threshold for feature retention
            quasi_constant_threshold: Threshold for quasi-constant feature detection
            adaptive_threshold: Whether to use adaptive thresholding
            standardize_before_filter: Whether to standardize features before filtering
            scaler_method: Scaling method ('standard' or 'robust')
            memory_limit_gb: Memory limit for processing
            preserve_rate: Target feature preservation rate for adaptive thresholding
        """
        self.variance_threshold = variance_threshold
        self.quasi_constant_threshold = quasi_constant_threshold
        self.adaptive_threshold = adaptive_threshold
        self.standardize_before_filter = standardize_before_filter
        self.scaler_method = scaler_method
        self.memory_limit_gb = memory_limit_gb
        self.preserve_rate = preserve_rate
        
        # Processing components
        self.scaler_ = None
        self.variance_selector_ = None
        
        # Results storage
        self.selected_features_ = []
        self.feature_variances_ = {}
        self.eliminated_features_ = {}
        self.adaptive_threshold_ = variance_threshold
        self.processing_stats_ = {}
        self.memory_stats_ = {}
        
    def _monitor_memory(self, stage: str) -> Dict[str, float]:
        """Monitor memory usage during processing."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1024 / 1024 / 1024
        
        self.memory_stats_[stage] = {
            'memory_gb': memory_gb,
            'timestamp': datetime.now(),
            'warning': memory_gb > self.memory_limit_gb
        }
        
        if memory_gb > self.memory_limit_gb:
            logger.warning(f"Memory usage ({memory_gb:.2f}GB) exceeds limit at {stage}")
            
        return self.memory_stats_[stage]
        
    def _validate_input_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data for variance filtering."""
        logger.info(f"Validating input data: {X.shape}")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
            
        # Select only numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < len(X.columns):
            logger.warning(f"Filtered {len(X.columns) - len(numeric_columns)} non-numeric columns")
            X = X[numeric_columns]
            
        if X.empty:
            raise ValueError("No numeric columns found in input data")
            
        # Handle missing values
        initial_shape = X.shape
        X_clean = X.dropna(axis=1, how='all')  # Remove columns with all NaN
        
        if X_clean.shape[1] < initial_shape[1]:
            logger.warning(f"Removed {initial_shape[1] - X_clean.shape[1]} all-NaN columns")
            
        # Fill remaining NaN values with median (robust to outliers)
        for col in X_clean.columns:
            if X_clean[col].isna().any():
                median_value = X_clean[col].median()
                X_clean[col] = X_clean[col].fillna(median_value)
                
        logger.info(f"Data validation completed: {initial_shape} → {X_clean.shape}")
        return X_clean
        
    def _calculate_feature_variances(
        self, 
        X: pd.DataFrame,
        standardized: bool = False
    ) -> Dict[str, float]:
        """
        Calculate variance for each feature with optional standardization.
        
        Args:
            X: Input features
            standardized: Whether to calculate variance on standardized features
            
        Returns:
            Dictionary mapping feature names to variance values
        """
        logger.info("Calculating feature variances")
        self._monitor_memory("variance_calc_start")
        
        if standardized and self.standardize_before_filter:
            # Apply scaling before variance calculation
            if self.scaler_method == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
                
            logger.info(f"Applying {self.scaler_method} scaling before variance calculation")
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            self.scaler_ = scaler
        else:
            X_scaled = X.copy()
            
        # Calculate variances
        variances = {}
        for col in X_scaled.columns:
            try:
                var_value = X_scaled[col].var(ddof=1)  # Sample variance
                if pd.isna(var_value):
                    var_value = 0.0
                variances[col] = var_value
            except Exception as e:
                logger.warning(f"Variance calculation failed for {col}: {str(e)}")
                variances[col] = 0.0
                
        self._monitor_memory("variance_calc_end")
        logger.info(f"Calculated variances for {len(variances)} features")
        
        return variances
        
    def _detect_quasi_constant_features(
        self, 
        X: pd.DataFrame
    ) -> List[str]:
        """
        Detect quasi-constant features that have very little variation.
        
        Quasi-constant features are those where the vast majority of values
        are the same, providing little information for modeling.
        """
        logger.info("Detecting quasi-constant features")
        
        quasi_constant_features = []
        
        for col in X.columns:
            try:
                # Calculate the frequency of the most common value
                value_counts = X[col].value_counts()
                if len(value_counts) == 0:
                    quasi_constant_features.append(col)
                    continue
                    
                most_common_freq = value_counts.iloc[0]
                total_count = len(X[col].dropna())
                
                if total_count == 0:
                    quasi_constant_features.append(col)
                    continue
                    
                # Calculate proportion of most common value
                most_common_prop = most_common_freq / total_count
                
                # Check if feature is quasi-constant
                if most_common_prop >= (1 - self.quasi_constant_threshold):
                    quasi_constant_features.append(col)
                    logger.debug(f"Quasi-constant: {col} ({most_common_prop:.4f} same values)")
                    
            except Exception as e:
                logger.warning(f"Quasi-constant detection failed for {col}: {str(e)}")
                # Conservative approach: mark as quasi-constant if error
                quasi_constant_features.append(col)
                
        logger.info(f"Detected {len(quasi_constant_features)} quasi-constant features")
        return quasi_constant_features
        
    def _calculate_adaptive_threshold(
        self, 
        variances: Dict[str, float]
    ) -> float:
        """
        Calculate adaptive variance threshold based on feature distribution.
        
        Uses percentile-based thresholding to preserve the desired rate of features
        while filtering out the lowest-variance features.
        """
        if not self.adaptive_threshold:
            return self.variance_threshold
            
        logger.info("Calculating adaptive variance threshold")
        
        variance_values = [v for v in variances.values() if v > 0 and not pd.isna(v)]
        
        if len(variance_values) == 0:
            logger.warning("No valid variance values found, using default threshold")
            return self.variance_threshold
            
        # Calculate percentile threshold to preserve desired rate
        preserve_percentile = (1 - self.preserve_rate) * 100
        adaptive_threshold = np.percentile(variance_values, preserve_percentile)
        
        # Ensure minimum threshold
        adaptive_threshold = max(adaptive_threshold, self.variance_threshold)
        
        logger.info(f"Adaptive threshold: {adaptive_threshold:.6f} "
                   f"(preserves ~{self.preserve_rate:.1%} of features)")
        
        return adaptive_threshold
        
    def _apply_variance_filtering(
        self, 
        X: pd.DataFrame,
        variances: Dict[str, float],
        threshold: float
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Apply variance filtering with the specified threshold.
        
        Args:
            X: Input features
            variances: Calculated variance values
            threshold: Variance threshold to apply
            
        Returns:
            (selected_features, elimination_reasons) tuple
        """
        logger.info(f"Applying variance filtering with threshold {threshold:.6f}")
        
        selected_features = []
        elimination_reasons = {}
        
        for feature, variance in variances.items():
            if variance >= threshold:
                selected_features.append(feature)
            else:
                elimination_reasons[feature] = f"low_variance_{variance:.6f}_below_{threshold:.6f}"
                
        logger.info(f"Variance filtering: {len(variances)} → {len(selected_features)} features "
                   f"(removed {len(elimination_reasons)})")
        
        return selected_features, elimination_reasons
        
    def filter_low_variance_features(
        self, 
        X: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Perform comprehensive variance-based feature filtering.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Dictionary containing filtering results and statistics
        """
        logger.info(f"Starting variance filtering for {X.shape[1]} features")
        start_time = datetime.now()
        self._monitor_memory("filtering_start")
        
        # Step 1: Validate and clean input data
        X_clean = self._validate_input_data(X)
        
        # Step 2: Detect quasi-constant features
        logger.info("=== Step 1: Quasi-Constant Feature Detection ===")
        quasi_constant_features = self._detect_quasi_constant_features(X_clean)
        
        # Remove quasi-constant features
        non_quasi_constant = [col for col in X_clean.columns if col not in quasi_constant_features]
        X_filtered = X_clean[non_quasi_constant]
        
        # Record quasi-constant eliminations
        for feature in quasi_constant_features:
            self.eliminated_features_[feature] = f"quasi_constant_threshold_{self.quasi_constant_threshold}"
        
        logger.info(f"Removed {len(quasi_constant_features)} quasi-constant features")
        
        if X_filtered.empty or len(X_filtered.columns) == 0:
            logger.warning("All features eliminated as quasi-constant")
            self.selected_features_ = []
            return self._generate_results(X, start_time)
        
        # Step 3: Calculate feature variances
        logger.info("=== Step 2: Feature Variance Calculation ===")
        self.feature_variances_ = self._calculate_feature_variances(
            X_filtered, 
            standardized=self.standardize_before_filter
        )
        
        # Step 4: Calculate adaptive threshold
        logger.info("=== Step 3: Adaptive Threshold Calculation ===")
        self.adaptive_threshold_ = self._calculate_adaptive_threshold(self.feature_variances_)
        
        # Step 5: Apply variance filtering
        logger.info("=== Step 4: Variance Filtering ===")
        variance_selected, variance_eliminations = self._apply_variance_filtering(
            X_filtered, 
            self.feature_variances_, 
            self.adaptive_threshold_
        )
        
        # Combine elimination reasons
        self.eliminated_features_.update(variance_eliminations)
        self.selected_features_ = variance_selected
        
        # Generate final results
        results = self._generate_results(X, start_time)
        
        self._monitor_memory("filtering_end")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Variance filtering completed in {elapsed_time:.1f}s")
        logger.info(f"Feature reduction: {X.shape[1]} → {len(self.selected_features_)} "
                   f"({len(self.selected_features_)/X.shape[1]:.1%} retention)")
        
        return results
        
    def _generate_results(self, original_X: pd.DataFrame, start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive results dictionary."""
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate variance statistics
        if self.feature_variances_:
            variance_values = list(self.feature_variances_.values())
            variance_stats = {
                'mean_variance': np.mean(variance_values),
                'median_variance': np.median(variance_values),
                'std_variance': np.std(variance_values),
                'min_variance': np.min(variance_values),
                'max_variance': np.max(variance_values)
            }
        else:
            variance_stats = {}
            
        # Processing statistics
        self.processing_stats_ = {
            'input_features': original_X.shape[1],
            'selected_features': len(self.selected_features_),
            'eliminated_features': len(self.eliminated_features_),
            'retention_rate': len(self.selected_features_) / original_X.shape[1] if original_X.shape[1] > 0 else 0,
            'processing_time_seconds': elapsed_time,
            'adaptive_threshold_used': self.adaptive_threshold_,
            'memory_peak_gb': max(stats['memory_gb'] for stats in self.memory_stats_.values()) if self.memory_stats_ else 0,
            'variance_statistics': variance_stats
        }
        
        return {
            'selected_features': self.selected_features_.copy(),
            'eliminated_features': self.eliminated_features_.copy(),
            'feature_variances': self.feature_variances_.copy(),
            'processing_stats': self.processing_stats_.copy(),
            'memory_stats': self.memory_stats_.copy(),
            'parameters': {
                'variance_threshold': self.variance_threshold,
                'quasi_constant_threshold': self.quasi_constant_threshold,
                'adaptive_threshold': self.adaptive_threshold,
                'adaptive_threshold_used': self.adaptive_threshold_,
                'standardize_before_filter': self.standardize_before_filter,
                'preserve_rate': self.preserve_rate
            }
        }
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data by selecting only the filtered features.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Filtered feature matrix containing only selected features
        """
        if not self.selected_features_:
            raise ValueError("Variance filter not fitted - call filter_low_variance_features first")
            
        # Select available features from the fitted selection
        available_features = [f for f in self.selected_features_ if f in X.columns]
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features in transform data")
            
        if not available_features:
            logger.warning("No selected features found in transform data")
            return pd.DataFrame(index=X.index)
            
        return X[available_features].copy()
        
    def get_variance_summary(self) -> Dict[str, Any]:
        """Get summary of variance filtering results."""
        if not self.feature_variances_:
            raise ValueError("Variance filtering not completed")
            
        # Calculate quantile statistics
        variance_values = [v for v in self.feature_variances_.values() if v > 0]
        
        if variance_values:
            quantile_stats = {
                'q25': np.percentile(variance_values, 25),
                'q50': np.percentile(variance_values, 50),
                'q75': np.percentile(variance_values, 75),
                'q90': np.percentile(variance_values, 90),
                'q95': np.percentile(variance_values, 95),
                'q99': np.percentile(variance_values, 99)
            }
        else:
            quantile_stats = {}
            
        # Features by variance category
        low_var_features = [f for f, v in self.feature_variances_.items() 
                           if v < self.variance_threshold]
        medium_var_features = [f for f, v in self.feature_variances_.items() 
                              if self.variance_threshold <= v < np.median(variance_values) if variance_values else False]
        high_var_features = [f for f, v in self.feature_variances_.items() 
                            if v >= np.median(variance_values) if variance_values else False]
        
        return {
            'variance_quantiles': quantile_stats,
            'threshold_used': self.adaptive_threshold_,
            'feature_categories': {
                'low_variance': {'count': len(low_var_features), 'features': low_var_features[:10]},
                'medium_variance': {'count': len(medium_var_features), 'features': medium_var_features[:10]}, 
                'high_variance': {'count': len(high_var_features), 'features': high_var_features[:10]}
            },
            'elimination_summary': {
                reason: len([f for f, r in self.eliminated_features_.items() if reason in r])
                for reason in ['quasi_constant', 'low_variance']
            }
        }
        
    def save_results(self, output_path: str) -> None:
        """Save variance filtering results to JSON file."""
        import json
        from pathlib import Path
        
        if not self.selected_features_:
            raise ValueError("Variance filtering not completed")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results = {
            'selected_features': self.selected_features_,
            'eliminated_features': self.eliminated_features_,
            'processing_stats': self.processing_stats_,
            'variance_summary': self.get_variance_summary(),
            'parameters': {
                'variance_threshold': self.variance_threshold,
                'quasi_constant_threshold': self.quasi_constant_threshold,
                'adaptive_threshold': self.adaptive_threshold,
                'adaptive_threshold_used': self.adaptive_threshold_,
                'preserve_rate': self.preserve_rate
            }
        }
        
        # Convert datetime objects
        for stage, stats in self.memory_stats_.items():
            if 'timestamp' in stats:
                stats['timestamp'] = stats['timestamp'].isoformat()
        
        results['memory_stats'] = self.memory_stats_
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Variance filtering results saved to {output_path}")