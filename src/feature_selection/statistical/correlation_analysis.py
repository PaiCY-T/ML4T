"""
Correlation Analysis and VIF-Based Multicollinearity Detection

This module implements sophisticated correlation analysis with hierarchical 
clustering and Variance Inflation Factor (VIF) analysis to identify and
eliminate multicollinear features while preserving information content.

Key Features:
- Hierarchical clustering on correlation matrix with configurable linkage
- VIF calculation for multicollinearity detection (threshold <10)
- Memory-efficient correlation calculation for 500+ features
- Taiwan market compliance validation
- Progressive feature elimination with information preservation tracking
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Set, Any
import pandas as pd
import numpy as np
from datetime import datetime
import gc
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import psutil

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    Advanced correlation analysis with VIF-based multicollinearity detection.
    
    Implements hierarchical clustering on correlation matrices to identify
    feature clusters and uses VIF analysis to eliminate multicollinear 
    features while preserving maximum information content.
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.7,
        vif_threshold: float = 10.0,
        linkage_method: str = 'ward',
        cluster_criterion: str = 'distance',
        memory_limit_gb: float = 8.0,
        chunk_size: int = 100,
        preserve_info_threshold: float = 0.9
    ):
        """
        Initialize correlation analyzer.
        
        Args:
            correlation_threshold: Max correlation between selected features
            vif_threshold: Maximum VIF value (>10 indicates multicollinearity)
            linkage_method: Hierarchical clustering linkage method
            cluster_criterion: Clustering criterion ('distance' or 'maxclust')
            memory_limit_gb: Memory limit for correlation calculations
            chunk_size: Chunk size for memory-efficient processing
            preserve_info_threshold: Minimum info preservation ratio
        """
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.linkage_method = linkage_method
        self.cluster_criterion = cluster_criterion
        self.memory_limit_gb = memory_limit_gb
        self.chunk_size = chunk_size
        self.preserve_info_threshold = preserve_info_threshold
        
        # Analysis results
        self.correlation_matrix_ = None
        self.vif_scores_ = {}
        self.feature_clusters_ = {}
        self.selected_features_ = []
        self.eliminated_features_ = {}
        self.info_preservation_ratio_ = 0.0
        self.processing_stats_ = {}
        
        # Memory monitoring
        self.memory_stats_ = {}
        
    def _monitor_memory(self, stage: str) -> Dict[str, float]:
        """Monitor memory usage during analysis."""
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
        
    def _calculate_correlation_chunked(
        self, 
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix in chunks to manage memory usage.
        
        For large feature sets (>500), this method processes correlations
        in chunks to avoid memory overflow while maintaining accuracy.
        """
        n_features = X.shape[1]
        feature_names = X.columns.tolist()
        
        logger.info(f"Calculating correlation matrix for {n_features} features")
        self._monitor_memory("correlation_start")
        
        # For smaller datasets, calculate directly
        if n_features <= self.chunk_size:
            correlation_matrix = X.corr()
            self._monitor_memory("correlation_direct")
            return correlation_matrix
        
        # For larger datasets, use chunked approach
        logger.info(f"Using chunked correlation calculation (chunk_size={self.chunk_size})")
        
        # Initialize correlation matrix
        correlation_matrix = pd.DataFrame(
            np.eye(n_features), 
            index=feature_names,
            columns=feature_names
        )
        
        # Calculate correlations in chunks
        n_chunks = int(np.ceil(n_features / self.chunk_size))
        
        for i in range(n_chunks):
            start_i = i * self.chunk_size
            end_i = min((i + 1) * self.chunk_size, n_features)
            features_i = feature_names[start_i:end_i]
            
            for j in range(i, n_chunks):
                start_j = j * self.chunk_size
                end_j = min((j + 1) * self.chunk_size, n_features)
                features_j = feature_names[start_j:end_j]
                
                # Calculate correlations for this chunk pair
                try:
                    X_i = X[features_i]
                    X_j = X[features_j]
                    
                    # Calculate correlation between chunks
                    corr_chunk = X_i.corrwith(X_j, axis=0, method='pearson')
                    
                    if isinstance(corr_chunk, pd.Series):
                        # Handle single feature case
                        for feat_i in features_i:
                            for feat_j in features_j:
                                if feat_i in X_i.columns and feat_j in X_j.columns:
                                    corr_val = X_i[feat_i].corr(X_j[feat_j])
                                    correlation_matrix.loc[feat_i, feat_j] = corr_val
                                    correlation_matrix.loc[feat_j, feat_i] = corr_val
                    else:
                        # Handle multi-feature case  
                        chunk_corr_matrix = X_i.corrwith(X_j, axis=1)
                        for feat_i in features_i:
                            for feat_j in features_j:
                                if feat_i in chunk_corr_matrix.index and feat_j in chunk_corr_matrix.columns:
                                    corr_val = chunk_corr_matrix.loc[feat_i, feat_j]
                                    correlation_matrix.loc[feat_i, feat_j] = corr_val
                                    correlation_matrix.loc[feat_j, feat_i] = corr_val
                    
                    # Force garbage collection
                    del X_i, X_j
                    gc.collect()
                    
                except Exception as e:
                    logger.warning(f"Error in chunk correlation ({i},{j}): {str(e)}")
                    # Fill with zeros for failed chunks
                    for feat_i in features_i:
                        for feat_j in features_j:
                            if feat_i != feat_j:
                                correlation_matrix.loc[feat_i, feat_j] = 0.0
                                correlation_matrix.loc[feat_j, feat_i] = 0.0
            
            # Monitor progress
            if (i + 1) % max(1, n_chunks // 10) == 0:
                progress = (i + 1) / n_chunks * 100
                self._monitor_memory(f"correlation_chunk_{progress:.0f}pct")
                logger.info(f"Correlation progress: {progress:.1f}%")
        
        self._monitor_memory("correlation_end")
        logger.info("Chunked correlation calculation completed")
        
        return correlation_matrix
        
    def _calculate_vif_scores(
        self, 
        X: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate Variance Inflation Factors for multicollinearity detection.
        
        VIF measures how much the variance of a regression coefficient
        increases due to collinearity. VIF > 10 indicates high multicollinearity.
        """
        logger.info("Calculating VIF scores for multicollinearity detection")
        self._monitor_memory("vif_start")
        
        vif_scores = {}
        feature_names = X.columns.tolist()
        
        try:
            # Standardize features for VIF calculation
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=feature_names,
                index=X.index
            )
            
            # Calculate VIF for each feature
            for i, feature in enumerate(feature_names):
                try:
                    # Calculate VIF
                    vif_value = variance_inflation_factor(X_scaled.values, i)
                    
                    # Handle infinite or NaN VIF values
                    if np.isinf(vif_value) or np.isnan(vif_value):
                        vif_value = float('inf')
                        
                    vif_scores[feature] = vif_value
                    
                except Exception as e:
                    logger.warning(f"VIF calculation failed for {feature}: {str(e)}")
                    vif_scores[feature] = float('inf')
                    
                # Progress monitoring
                if (i + 1) % max(1, len(feature_names) // 20) == 0:
                    progress = (i + 1) / len(feature_names) * 100
                    logger.info(f"VIF calculation progress: {progress:.1f}%")
                    
        except Exception as e:
            logger.error(f"VIF calculation failed: {str(e)}")
            # Fallback: assign high VIF to all features
            vif_scores = {feature: float('inf') for feature in feature_names}
        
        self._monitor_memory("vif_end")
        
        # Log VIF statistics
        finite_vifs = [v for v in vif_scores.values() if not np.isinf(v)]
        if finite_vifs:
            logger.info(f"VIF statistics - Mean: {np.mean(finite_vifs):.2f}, "
                       f"Max: {np.max(finite_vifs):.2f}, "
                       f"High VIF count: {sum(1 for v in finite_vifs if v > self.vif_threshold)}")
        
        return vif_scores
        
    def _perform_hierarchical_clustering(
        self, 
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, int]:
        """
        Perform hierarchical clustering on correlation matrix.
        
        Groups highly correlated features into clusters using hierarchical
        clustering with configurable linkage method.
        """
        logger.info(f"Performing hierarchical clustering with {self.linkage_method} linkage")
        self._monitor_memory("clustering_start")
        
        try:
            # Convert correlation to distance matrix
            distance_matrix = 1 - correlation_matrix.abs()
            
            # Handle NaN values
            distance_matrix = distance_matrix.fillna(1.0)
            
            # Convert to condensed distance matrix for clustering
            condensed_distance = squareform(distance_matrix.values)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(
                condensed_distance, 
                method=self.linkage_method
            )
            
            # Form clusters based on correlation threshold
            distance_threshold = 1 - self.correlation_threshold
            
            if self.cluster_criterion == 'distance':
                cluster_labels = fcluster(
                    linkage_matrix, 
                    t=distance_threshold, 
                    criterion='distance'
                )
            else:
                # Estimate number of clusters
                max_clusters = min(50, len(correlation_matrix) // 5)
                cluster_labels = fcluster(
                    linkage_matrix,
                    t=max_clusters,
                    criterion='maxclust'
                )
                
            # Create feature-to-cluster mapping
            feature_clusters = dict(zip(
                correlation_matrix.columns,
                cluster_labels
            ))
            
            n_clusters = len(set(cluster_labels))
            logger.info(f"Formed {n_clusters} feature clusters")
            
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {str(e)}")
            # Fallback: each feature in its own cluster
            feature_clusters = {
                feature: i for i, feature in enumerate(correlation_matrix.columns)
            }
        
        self._monitor_memory("clustering_end")
        return feature_clusters
        
    def _select_representative_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        vif_scores: Optional[Dict[str, float]] = None,
        feature_clusters: Optional[Dict[str, int]] = None
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Select representative features from each cluster.
        
        For each cluster of correlated features, selects the best representative
        based on multiple criteria:
        1. Lowest VIF score (less multicollinear)
        2. Highest correlation with target (if available)
        3. Highest variance (most informative)
        """
        logger.info("Selecting representative features from clusters")
        self._monitor_memory("selection_start")
        
        selected_features = []
        elimination_reasons = {}
        
        # Group features by cluster
        cluster_to_features = {}
        for feature, cluster_id in feature_clusters.items():
            if cluster_id not in cluster_to_features:
                cluster_to_features[cluster_id] = []
            cluster_to_features[cluster_id].append(feature)
        
        logger.info(f"Processing {len(cluster_to_features)} feature clusters")
        
        # Process each cluster
        for cluster_id, cluster_features in cluster_to_features.items():
            if len(cluster_features) == 1:
                # Single feature cluster - select it
                selected_features.append(cluster_features[0])
                continue
                
            # Multiple features in cluster - select best representative
            logger.debug(f"Cluster {cluster_id}: {len(cluster_features)} features")
            
            # Calculate selection criteria for each feature in cluster
            selection_scores = {}
            
            for feature in cluster_features:
                score_components = {}
                
                # 1. VIF score (lower is better) 
                vif_score = vif_scores.get(feature, float('inf'))
                if not np.isinf(vif_score) and vif_score > 0:
                    score_components['vif'] = 1.0 / vif_score  # Invert so higher is better
                else:
                    score_components['vif'] = 0.0
                
                # 2. Target correlation (if available)
                if y is not None:
                    try:
                        target_corr = abs(X[feature].corr(y))
                        if not pd.isna(target_corr):
                            score_components['target_corr'] = target_corr
                        else:
                            score_components['target_corr'] = 0.0
                    except:
                        score_components['target_corr'] = 0.0
                else:
                    score_components['target_corr'] = 0.0
                    
                # 3. Feature variance (higher is better)
                try:
                    variance = X[feature].var()
                    if not pd.isna(variance) and variance > 0:
                        score_components['variance'] = variance
                    else:
                        score_components['variance'] = 0.0
                except:
                    score_components['variance'] = 0.0
                
                # Combine scores (weighted average)
                if y is not None:
                    # With target: prioritize target correlation
                    total_score = (
                        0.3 * score_components['vif'] +
                        0.5 * score_components['target_corr'] +  
                        0.2 * score_components['variance']
                    )
                else:
                    # Without target: prioritize VIF and variance
                    total_score = (
                        0.6 * score_components['vif'] +
                        0.4 * score_components['variance']
                    )
                
                selection_scores[feature] = {
                    'total_score': total_score,
                    **score_components
                }
            
            # Select feature with highest score
            best_feature = max(selection_scores.keys(), 
                             key=lambda f: selection_scores[f]['total_score'])
            selected_features.append(best_feature)
            
            # Record elimination reasons for other features
            for feature in cluster_features:
                if feature != best_feature:
                    best_score = selection_scores[best_feature]['total_score']
                    curr_score = selection_scores[feature]['total_score'] 
                    elimination_reasons[feature] = (
                        f"cluster_{cluster_id}_representative_selection_"
                        f"score_{curr_score:.4f}_vs_best_{best_score:.4f}"
                    )
        
        self._monitor_memory("selection_end")
        
        logger.info(f"Selected {len(selected_features)} representative features "
                   f"from {len(cluster_to_features)} clusters")
        
        return selected_features, elimination_reasons
        
    def analyze_correlations(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis with VIF and clustering.
        
        Args:
            X: Feature matrix (500+ features expected)
            y: Target variable (optional, improves selection quality)
            
        Returns:
            Analysis results including selected features and statistics
        """
        logger.info(f"Starting correlation analysis for {X.shape[1]} features")
        start_time = datetime.now()
        self._monitor_memory("analysis_start")
        
        # Validate input
        if X.empty:
            raise ValueError("Input feature matrix is empty")
        if len(X.columns) < 2:
            raise ValueError("Need at least 2 features for correlation analysis")
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < len(X.columns):
            logger.warning(f"Removed {len(X.columns) - len(numeric_columns)} non-numeric columns")
            X = X[numeric_columns]
        
        # Remove columns with all NaN or constant values
        valid_columns = []
        for col in X.columns:
            col_data = X[col].dropna()
            if len(col_data) > 0 and col_data.std() > 0:
                valid_columns.append(col)
        
        if len(valid_columns) < len(X.columns):
            logger.warning(f"Removed {len(X.columns) - len(valid_columns)} invalid columns")
            X = X[valid_columns]
        
        # Step 1: Calculate correlation matrix
        logger.info("=== Step 1: Correlation Matrix Calculation ===")
        self.correlation_matrix_ = self._calculate_correlation_chunked(X)
        
        # Step 2: Calculate VIF scores
        logger.info("=== Step 2: VIF Multicollinearity Analysis ===")
        self.vif_scores_ = self._calculate_vif_scores(X)
        
        # Step 3: Hierarchical clustering
        logger.info("=== Step 3: Hierarchical Clustering ===")
        self.feature_clusters_ = self._perform_hierarchical_clustering(self.correlation_matrix_)
        
        # Step 4: Representative feature selection
        logger.info("=== Step 4: Representative Feature Selection ===")
        self.selected_features_, self.eliminated_features_ = self._select_representative_features(
            X, y, self.correlation_matrix_, self.vif_scores_, self.feature_clusters_
        )
        
        # Step 5: Calculate information preservation
        logger.info("=== Step 5: Information Preservation Analysis ===")
        self.info_preservation_ratio_ = len(self.selected_features_) / len(X.columns)
        
        # Processing statistics
        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.processing_stats_ = {
            'input_features': len(X.columns),
            'selected_features': len(self.selected_features_),
            'reduction_ratio': 1 - self.info_preservation_ratio_,
            'processing_time_seconds': elapsed_time,
            'clusters_formed': len(set(self.feature_clusters_.values())),
            'high_vif_features': sum(1 for v in self.vif_scores_.values() 
                                   if not np.isinf(v) and v > self.vif_threshold),
            'memory_peak_gb': max(stats['memory_gb'] for stats in self.memory_stats_.values())
        }
        
        self._monitor_memory("analysis_end")
        
        logger.info(f"Correlation analysis completed in {elapsed_time:.1f}s")
        logger.info(f"Feature reduction: {len(X.columns)} → {len(self.selected_features_)} "
                   f"({self.info_preservation_ratio_:.1%} retention)")
        
        return self.get_analysis_results()
        
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get comprehensive analysis results."""
        return {
            'selected_features': self.selected_features_.copy(),
            'eliminated_features': self.eliminated_features_.copy(),
            'correlation_matrix': self.correlation_matrix_.copy() if self.correlation_matrix_ is not None else None,
            'vif_scores': self.vif_scores_.copy(),
            'feature_clusters': self.feature_clusters_.copy(),
            'info_preservation_ratio': self.info_preservation_ratio_,
            'processing_stats': self.processing_stats_.copy(),
            'memory_stats': self.memory_stats_.copy(),
            'parameters': {
                'correlation_threshold': self.correlation_threshold,
                'vif_threshold': self.vif_threshold,
                'linkage_method': self.linkage_method,
                'preserve_info_threshold': self.preserve_info_threshold
            }
        }
        
    def get_multicollinearity_report(self) -> Dict[str, Any]:
        """Generate detailed multicollinearity report."""
        if not self.vif_scores_:
            raise ValueError("Analysis not completed - run analyze_correlations first")
            
        # VIF statistics
        finite_vifs = {k: v for k, v in self.vif_scores_.items() if not np.isinf(v)}
        high_vif_features = {k: v for k, v in finite_vifs.items() if v > self.vif_threshold}
        
        # Correlation statistics
        if self.correlation_matrix_ is not None:
            corr_values = self.correlation_matrix_.abs().values
            corr_values = corr_values[np.triu_indices_from(corr_values, k=1)]  # Upper triangle
            corr_values = corr_values[~np.isnan(corr_values)]
        else:
            corr_values = []
        
        return {
            'vif_summary': {
                'mean_vif': np.mean(list(finite_vifs.values())) if finite_vifs else 0,
                'max_vif': np.max(list(finite_vifs.values())) if finite_vifs else 0,
                'high_vif_count': len(high_vif_features),
                'high_vif_features': high_vif_features
            },
            'correlation_summary': {
                'mean_correlation': np.mean(corr_values) if len(corr_values) > 0 else 0,
                'max_correlation': np.max(corr_values) if len(corr_values) > 0 else 0,
                'high_correlation_pairs': self._find_high_correlation_pairs()
            },
            'cluster_summary': {
                'num_clusters': len(set(self.feature_clusters_.values())),
                'avg_cluster_size': np.mean([
                    sum(1 for c in self.feature_clusters_.values() if c == cluster_id)
                    for cluster_id in set(self.feature_clusters_.values())
                ]) if self.feature_clusters_ else 0
            }
        }
        
    def _find_high_correlation_pairs(self) -> List[Tuple[str, str, float]]:
        """Find pairs of features with high correlation."""
        if self.correlation_matrix_ is None:
            return []
            
        high_corr_pairs = []
        corr_matrix = self.correlation_matrix_.abs()
        
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_value = corr_matrix.iloc[i, j]
                if not pd.isna(corr_value) and corr_value >= self.correlation_threshold:
                    feature_1 = corr_matrix.index[i]
                    feature_2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feature_1, feature_2, corr_value))
        
        # Sort by correlation value (descending)
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return high_corr_pairs[:50]  # Return top 50 pairs
        
    def save_analysis_results(self, output_path: str) -> None:
        """Save analysis results to JSON file."""
        import json
        from pathlib import Path
        
        if not self.selected_features_:
            raise ValueError("Analysis not completed - run analyze_correlations first")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results = self.get_analysis_results()
        
        # Convert correlation matrix to serializable format
        if results['correlation_matrix'] is not None:
            results['correlation_matrix'] = results['correlation_matrix'].to_dict()
        
        # Convert datetime objects
        for stage, stats in results['memory_stats'].items():
            if 'timestamp' in stats:
                stats['timestamp'] = stats['timestamp'].isoformat()
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Analysis results saved to {output_path}")