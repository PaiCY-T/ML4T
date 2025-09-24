"""
Mutual Information-Based Feature Selection

This module implements mutual information analysis for ranking features based
on their information content with respect to target returns in Taiwan stock
market data.

Key Features:
- Mutual information calculation for both continuous and discrete targets
- Non-linear relationship detection between features and returns
- Robust estimation with missing data handling
- Memory-efficient processing for 500+ features
- Taiwan market-specific validation and filtering
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import psutil

logger = logging.getLogger(__name__)

class MutualInfoSelector:
    """
    Mutual information-based feature selection for Taiwan stock market data.
    
    Uses mutual information to identify features that have the highest
    information content with respect to target returns, capturing both
    linear and non-linear relationships.
    """
    
    def __init__(
        self,
        task_type: str = 'regression',
        n_neighbors: int = 3,
        random_state: int = 42,
        n_jobs: int = -1,
        memory_limit_gb: float = 8.0,
        discretize_target: bool = False,
        n_bins: int = 10,
        min_mi_score: float = 0.01
    ):
        """
        Initialize mutual information selector.
        
        Args:
            task_type: 'regression' or 'classification'
            n_neighbors: Number of neighbors for MI estimation
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            memory_limit_gb: Memory limit for processing
            discretize_target: Whether to discretize continuous targets
            n_bins: Number of bins for target discretization
            min_mi_score: Minimum MI score to consider feature relevant
        """
        self.task_type = task_type
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.memory_limit_gb = memory_limit_gb
        self.discretize_target = discretize_target
        self.n_bins = n_bins
        self.min_mi_score = min_mi_score
        
        # Processing components
        self.scaler_ = None
        self.discretizer_ = None
        
        # Results storage
        self.mi_scores_ = {}
        self.selected_features_ = []
        self.eliminated_features_ = {}
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
        
    def _validate_input_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and clean input data."""
        logger.info(f"Validating input data: X{X.shape}, y{len(y)}")
        
        if X.empty:
            raise ValueError("Feature matrix X is empty")
        if len(y) == 0:
            raise ValueError("Target vector y is empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
            
        # Select only numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < len(X.columns):
            logger.warning(f"Filtered {len(X.columns) - len(numeric_columns)} non-numeric columns")
            X = X[numeric_columns]
            
        if X.empty:
            raise ValueError("No numeric columns found in feature matrix")
            
        # Handle missing values
        # Remove samples with missing target values
        valid_target_mask = ~y.isna()
        X_clean = X.loc[valid_target_mask]
        y_clean = y.loc[valid_target_mask]
        
        if len(y_clean) == 0:
            raise ValueError("No valid target values found")
            
        logger.info(f"Removed {len(y) - len(y_clean)} samples with missing targets")
        
        # Handle missing features - use median imputation
        for col in X_clean.columns:
            if X_clean[col].isna().any():
                median_value = X_clean[col].median()
                if pd.isna(median_value):
                    # If median is NaN, use 0
                    median_value = 0.0
                X_clean[col] = X_clean[col].fillna(median_value)
                
        logger.info(f"Data validation completed: X{X_clean.shape}, y{len(y_clean)}")
        return X_clean, y_clean
        
    def _prepare_target_variable(self, y: pd.Series) -> pd.Series:
        """
        Prepare target variable for mutual information calculation.
        
        For regression tasks with continuous targets, optionally discretizes
        the target to capture non-linear relationships more effectively.
        """
        logger.info("Preparing target variable for MI calculation")
        
        if self.task_type == 'classification':
            # For classification, ensure integer labels
            y_prepared = y.astype(int)
            logger.info(f"Classification target: {len(y_prepared.unique())} classes")
            
        elif self.discretize_target and self.task_type == 'regression':
            # Discretize continuous target for better MI estimation
            logger.info(f"Discretizing continuous target into {self.n_bins} bins")
            
            self.discretizer_ = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy='quantile',
                random_state=self.random_state
            )
            
            # Reshape for sklearn
            y_reshaped = y.values.reshape(-1, 1)
            y_discretized = self.discretizer_.fit_transform(y_reshaped)
            y_prepared = pd.Series(y_discretized.flatten(), index=y.index, dtype=int)
            
            logger.info(f"Target discretized: {len(y_prepared.unique())} bins")
            
        else:
            # Keep continuous target as-is
            y_prepared = y.copy()
            logger.info("Using continuous target for MI calculation")
            
        return y_prepared
        
    def _calculate_mutual_information_chunked(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        chunk_size: int = 50
    ) -> Dict[str, float]:
        """
        Calculate mutual information scores in chunks to manage memory.
        
        For large feature sets, processes MI calculation in chunks to
        avoid memory overflow while maintaining accuracy.
        """
        logger.info(f"Calculating MI scores for {X.shape[1]} features")
        self._monitor_memory("mi_calc_start")
        
        mi_scores = {}
        n_features = X.shape[1]
        feature_names = X.columns.tolist()
        
        # Determine chunk size based on memory constraints
        if n_features <= chunk_size:
            # Calculate all at once for smaller datasets
            logger.info("Calculating MI scores for all features at once")
            
            try:
                if self.task_type == 'classification' or self.discretize_target:
                    scores = mutual_info_classif(
                        X.values, 
                        y.values,
                        random_state=self.random_state,
                        n_neighbors=self.n_neighbors
                    )
                else:
                    scores = mutual_info_regression(
                        X.values,
                        y.values, 
                        random_state=self.random_state,
                        n_neighbors=self.n_neighbors
                    )
                    
                mi_scores = dict(zip(feature_names, scores))
                
            except Exception as e:
                logger.error(f"MI calculation failed: {str(e)}")
                # Fallback: assign zero MI to all features
                mi_scores = {feature: 0.0 for feature in feature_names}
                
        else:
            # Process in chunks
            logger.info(f"Processing MI calculation in chunks of {chunk_size}")
            n_chunks = int(np.ceil(n_features / chunk_size))
            
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_features)
                chunk_features = feature_names[start_idx:end_idx]
                
                logger.info(f"Processing chunk {i+1}/{n_chunks}: features {start_idx}-{end_idx-1}")
                
                try:
                    X_chunk = X[chunk_features]
                    
                    if self.task_type == 'classification' or self.discretize_target:
                        chunk_scores = mutual_info_classif(
                            X_chunk.values,
                            y.values,
                            random_state=self.random_state,
                            n_neighbors=self.n_neighbors
                        )
                    else:
                        chunk_scores = mutual_info_regression(
                            X_chunk.values,
                            y.values,
                            random_state=self.random_state, 
                            n_neighbors=self.n_neighbors
                        )
                        
                    # Store chunk results
                    for feature, score in zip(chunk_features, chunk_scores):
                        mi_scores[feature] = score
                        
                except Exception as e:
                    logger.warning(f"MI calculation failed for chunk {i+1}: {str(e)}")
                    # Assign zero MI to failed chunk
                    for feature in chunk_features:
                        mi_scores[feature] = 0.0
                        
                # Monitor progress
                progress = (i + 1) / n_chunks * 100
                self._monitor_memory(f"mi_chunk_{progress:.0f}pct")
                
        self._monitor_memory("mi_calc_end")
        
        # Log MI score statistics
        valid_scores = [score for score in mi_scores.values() if score > 0]
        if valid_scores:
            logger.info(f"MI score statistics - Mean: {np.mean(valid_scores):.4f}, "
                       f"Max: {np.max(valid_scores):.4f}, "
                       f"Positive scores: {len(valid_scores)}/{len(mi_scores)}")
        else:
            logger.warning("No positive MI scores calculated")
            
        return mi_scores
        
    def _rank_features_by_mutual_information(
        self, 
        mi_scores: Dict[str, float],
        top_k: Optional[int] = None
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Rank and select features based on mutual information scores.
        
        Args:
            mi_scores: Dictionary of feature MI scores
            top_k: Number of top features to select (None for all above threshold)
            
        Returns:
            (selected_features, elimination_reasons) tuple
        """
        logger.info("Ranking features by mutual information scores")
        
        # Filter out features with MI scores below minimum threshold
        valid_features = {
            feature: score for feature, score in mi_scores.items()
            if score >= self.min_mi_score
        }
        
        if not valid_features:
            logger.warning(f"No features have MI score >= {self.min_mi_score}")
            return [], {feature: f"mi_score_{score:.6f}_below_threshold_{self.min_mi_score}" 
                       for feature, score in mi_scores.items()}
        
        # Sort features by MI score (descending)
        sorted_features = sorted(valid_features.items(), key=lambda x: x[1], reverse=True)
        
        # Select top-k features if specified
        if top_k is not None:
            selected_items = sorted_features[:top_k]
            eliminated_items = sorted_features[top_k:]
        else:
            selected_items = sorted_features
            eliminated_items = []
            
        # Prepare results
        selected_features = [feature for feature, score in selected_items]
        
        elimination_reasons = {}
        
        # Add eliminations for features below threshold
        for feature, score in mi_scores.items():
            if feature not in selected_features:
                if score < self.min_mi_score:
                    elimination_reasons[feature] = f"mi_score_{score:.6f}_below_threshold_{self.min_mi_score}"
                else:
                    # Feature eliminated by top-k selection
                    rank = next(i for i, (f, s) in enumerate(sorted_features, 1) if f == feature)
                    elimination_reasons[feature] = f"mi_rank_{rank}_below_top_{top_k}_score_{score:.6f}"
        
        logger.info(f"Selected {len(selected_features)} features based on MI scores")
        if top_k:
            logger.info(f"Applied top-{top_k} selection from {len(valid_features)} valid features")
            
        return selected_features, elimination_reasons
        
    def select_features_by_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform mutual information-based feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            top_k: Number of top features to select (None for threshold-based)
            
        Returns:
            Dictionary containing selection results and statistics
        """
        logger.info(f"Starting MI-based feature selection for {X.shape[1]} features")
        start_time = datetime.now()
        self._monitor_memory("selection_start")
        
        # Step 1: Validate and clean input data
        X_clean, y_clean = self._validate_input_data(X, y)
        
        # Step 2: Prepare target variable
        logger.info("=== Step 1: Target Variable Preparation ===")
        y_prepared = self._prepare_target_variable(y_clean)
        
        # Step 3: Calculate mutual information scores
        logger.info("=== Step 2: Mutual Information Calculation ===")
        self.mi_scores_ = self._calculate_mutual_information_chunked(X_clean, y_prepared)
        
        # Step 4: Rank and select features
        logger.info("=== Step 3: Feature Ranking and Selection ===")
        self.selected_features_, self.eliminated_features_ = self._rank_features_by_mutual_information(
            self.mi_scores_, top_k
        )
        
        # Generate results
        results = self._generate_results(X, y, start_time, top_k)
        
        self._monitor_memory("selection_end")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"MI-based feature selection completed in {elapsed_time:.1f}s")
        logger.info(f"Feature reduction: {X.shape[1]} → {len(self.selected_features_)} "
                   f"({len(self.selected_features_)/X.shape[1]:.1%} retention)")
        
        return results
        
    def _generate_results(
        self, 
        original_X: pd.DataFrame, 
        original_y: pd.Series,
        start_time: datetime,
        top_k: Optional[int]
    ) -> Dict[str, Any]:
        """Generate comprehensive results dictionary."""
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate MI score statistics
        valid_scores = [score for score in self.mi_scores_.values() if score > 0]
        if valid_scores:
            mi_stats = {
                'mean_mi_score': np.mean(list(self.mi_scores_.values())),
                'median_mi_score': np.median(list(self.mi_scores_.values())),
                'std_mi_score': np.std(list(self.mi_scores_.values())),
                'min_mi_score': np.min(list(self.mi_scores_.values())),
                'max_mi_score': np.max(list(self.mi_scores_.values())),
                'positive_scores_count': len(valid_scores),
                'zero_scores_count': len([s for s in self.mi_scores_.values() if s == 0])
            }
        else:
            mi_stats = {key: 0.0 for key in ['mean_mi_score', 'median_mi_score', 'std_mi_score', 'min_mi_score', 'max_mi_score']}
            mi_stats['positive_scores_count'] = 0
            mi_stats['zero_scores_count'] = len(self.mi_scores_)
        
        # Processing statistics
        self.processing_stats_ = {
            'input_features': original_X.shape[1],
            'input_samples': len(original_y),
            'selected_features': len(self.selected_features_),
            'eliminated_features': len(self.eliminated_features_),
            'retention_rate': len(self.selected_features_) / original_X.shape[1] if original_X.shape[1] > 0 else 0,
            'processing_time_seconds': elapsed_time,
            'top_k_used': top_k,
            'min_mi_threshold': self.min_mi_score,
            'memory_peak_gb': max(stats['memory_gb'] for stats in self.memory_stats_.values()) if self.memory_stats_ else 0,
            'mi_statistics': mi_stats
        }
        
        return {
            'selected_features': self.selected_features_.copy(),
            'eliminated_features': self.eliminated_features_.copy(),
            'mi_scores': self.mi_scores_.copy(),
            'processing_stats': self.processing_stats_.copy(),
            'memory_stats': self.memory_stats_.copy(),
            'parameters': {
                'task_type': self.task_type,
                'n_neighbors': self.n_neighbors,
                'discretize_target': self.discretize_target,
                'n_bins': self.n_bins,
                'min_mi_score': self.min_mi_score,
                'top_k': top_k
            }
        }
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data by selecting only the MI-selected features.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Transformed feature matrix with selected features only
        """
        if not self.selected_features_:
            raise ValueError("MI selector not fitted - call select_features_by_mutual_information first")
            
        # Select available features from the fitted selection
        available_features = [f for f in self.selected_features_ if f in X.columns]
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features in transform data")
            
        if not available_features:
            logger.warning("No selected features found in transform data")
            return pd.DataFrame(index=X.index)
            
        return X[available_features].copy()
        
    def get_feature_ranking(self, top_n: int = 50) -> List[Tuple[str, float]]:
        """
        Get top-N features ranked by MI score.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, mi_score) tuples sorted by score (descending)
        """
        if not self.mi_scores_:
            raise ValueError("MI calculation not completed")
            
        sorted_features = sorted(self.mi_scores_.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]
        
    def get_mi_score_distribution(self) -> Dict[str, Any]:
        """Get distribution analysis of MI scores."""
        if not self.mi_scores_:
            raise ValueError("MI calculation not completed")
            
        scores = list(self.mi_scores_.values())
        positive_scores = [s for s in scores if s > 0]
        
        # Calculate percentiles
        if positive_scores:
            percentiles = {
                'p10': np.percentile(positive_scores, 10),
                'p25': np.percentile(positive_scores, 25),
                'p50': np.percentile(positive_scores, 50),
                'p75': np.percentile(positive_scores, 75),
                'p90': np.percentile(positive_scores, 90),
                'p95': np.percentile(positive_scores, 95),
                'p99': np.percentile(positive_scores, 99)
            }
        else:
            percentiles = {key: 0.0 for key in ['p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']}
        
        # Categorize features by MI score
        high_mi_features = [f for f, s in self.mi_scores_.items() if s >= np.percentile(positive_scores, 75) if positive_scores else False]
        medium_mi_features = [f for f, s in self.mi_scores_.items() if np.percentile(positive_scores, 25) <= s < np.percentile(positive_scores, 75) if positive_scores else False] 
        low_mi_features = [f for f, s in self.mi_scores_.items() if 0 < s < np.percentile(positive_scores, 25) if positive_scores else False]
        zero_mi_features = [f for f, s in self.mi_scores_.items() if s == 0]
        
        return {
            'score_percentiles': percentiles,
            'feature_categories': {
                'high_mi': {'count': len(high_mi_features), 'features': high_mi_features[:10]},
                'medium_mi': {'count': len(medium_mi_features), 'features': medium_mi_features[:10]},
                'low_mi': {'count': len(low_mi_features), 'features': low_mi_features[:10]},
                'zero_mi': {'count': len(zero_mi_features), 'features': zero_mi_features[:10]}
            },
            'total_features': len(self.mi_scores_),
            'positive_mi_features': len(positive_scores),
            'above_threshold_features': len([s for s in scores if s >= self.min_mi_score])
        }
        
    def save_results(self, output_path: str) -> None:
        """Save MI selection results to JSON file."""
        import json
        from pathlib import Path
        
        if not self.selected_features_:
            raise ValueError("MI selection not completed")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results = {
            'selected_features': self.selected_features_,
            'eliminated_features': self.eliminated_features_,
            'mi_scores': self.mi_scores_,
            'processing_stats': self.processing_stats_,
            'mi_distribution': self.get_mi_score_distribution(),
            'top_features': self.get_feature_ranking(20),
            'parameters': {
                'task_type': self.task_type,
                'n_neighbors': self.n_neighbors,
                'discretize_target': self.discretize_target,
                'n_bins': self.n_bins,
                'min_mi_score': self.min_mi_score
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
            
        logger.info(f"MI selection results saved to {output_path}")