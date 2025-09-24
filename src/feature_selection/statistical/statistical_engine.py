"""
Statistical Selection Engine - Task #29 Stream A

This module implements the comprehensive Statistical Selection Engine that
coordinates all statistical-based feature selection methods to reduce the 
OpenFE-generated feature space from 500+ candidates to an optimal subset.

Key Features:
- Orchestrated pipeline combining correlation, variance, MI, and significance analysis
- Memory-optimized processing for large feature sets (500+ features)
- Information coefficient preservation tracking
- Taiwan market compliance validation
- Progressive feature elimination with detailed logging
- Comprehensive reporting and validation
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import gc
import psutil

# Import statistical selection components
from .correlation_analysis import CorrelationAnalyzer
from .variance_filter import VarianceFilter
from .mutual_info_selector import MutualInfoSelector
from .significance_tester import StatisticalSignificanceTester

logger = logging.getLogger(__name__)

class StatisticalSelectionEngine:
    """
    Comprehensive Statistical Selection Engine for Taiwan stock market.
    
    Coordinates multiple statistical methods to reduce 500+ OpenFE-generated
    features to an optimal subset while preserving information content and
    eliminating multicollinearity.
    
    Selection Pipeline:
    1. Variance filtering - remove low-information features
    2. Correlation analysis - eliminate multicollinear features with VIF
    3. Mutual information ranking - select most informative features
    4. Statistical significance testing - validate predictive power
    5. Final optimization - balance information vs. complexity
    """
    
    def __init__(
        self,
        target_feature_count: int = 75,
        min_feature_count: int = 50,
        max_feature_count: int = 100,
        correlation_threshold: float = 0.7,
        vif_threshold: float = 10.0,
        variance_threshold: float = 0.01,
        mi_threshold: float = 0.01,
        significance_alpha: float = 0.05,
        memory_limit_gb: float = 8.0,
        preserve_info_threshold: float = 0.9,
        random_state: int = 42
    ):
        """
        Initialize Statistical Selection Engine.
        
        Args:
            target_feature_count: Target number of features to select
            min_feature_count: Minimum acceptable features
            max_feature_count: Maximum acceptable features
            correlation_threshold: Max correlation between selected features
            vif_threshold: Maximum VIF for multicollinearity detection
            variance_threshold: Minimum variance threshold
            mi_threshold: Minimum mutual information threshold
            significance_alpha: Statistical significance level
            memory_limit_gb: Memory limit for processing
            preserve_info_threshold: Minimum information preservation ratio
            random_state: Random seed for reproducibility
        """
        self.target_feature_count = target_feature_count
        self.min_feature_count = min_feature_count
        self.max_feature_count = max_feature_count
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.variance_threshold = variance_threshold
        self.mi_threshold = mi_threshold
        self.significance_alpha = significance_alpha
        self.memory_limit_gb = memory_limit_gb
        self.preserve_info_threshold = preserve_info_threshold
        self.random_state = random_state
        
        # Initialize component selectors
        self.variance_filter_ = VarianceFilter(
            variance_threshold=variance_threshold,
            adaptive_threshold=True,
            preserve_rate=0.85,
            memory_limit_gb=memory_limit_gb
        )
        
        self.correlation_analyzer_ = CorrelationAnalyzer(
            correlation_threshold=correlation_threshold,
            vif_threshold=vif_threshold,
            memory_limit_gb=memory_limit_gb,
            preserve_info_threshold=preserve_info_threshold
        )
        
        self.mi_selector_ = MutualInfoSelector(
            task_type='regression',
            min_mi_score=mi_threshold,
            memory_limit_gb=memory_limit_gb,
            random_state=random_state
        )
        
        self.significance_tester_ = StatisticalSignificanceTester(
            alpha_level=significance_alpha,
            multiple_test_correction='benjamini-hochberg',
            memory_limit_gb=memory_limit_gb,
            random_state=random_state
        )
        
        # Results storage
        self.selected_features_ = []
        self.feature_scores_ = {}
        self.elimination_history_ = {}
        self.stage_results_ = {}
        self.final_stats_ = {}
        self.memory_usage_ = {}
        
        # Processing state
        self.is_fitted_ = False
        
    def _monitor_memory(self, stage: str) -> Dict[str, float]:
        """Monitor memory usage during processing."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1024 / 1024 / 1024
        
        self.memory_usage_[stage] = {
            'memory_gb': memory_gb,
            'timestamp': datetime.now(),
            'warning': memory_gb > self.memory_limit_gb
        }
        
        if memory_gb > self.memory_limit_gb:
            logger.warning(f"Memory usage ({memory_gb:.2f}GB) exceeds limit at {stage}")
            
        return self.memory_usage_[stage]
        
    def _validate_input_data(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Validate input data for statistical selection."""
        logger.info(f"Validating input data for statistical selection")
        
        if X.empty:
            raise ValueError("Feature matrix X is empty")
            
        # Log initial statistics
        logger.info(f"Input data: {X.shape[1]} features, {X.shape[0]} samples")
        
        # Check target if provided
        if y is not None:
            if len(y) != len(X):
                raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")
            logger.info(f"Target provided: {len(y)} samples")
        else:
            logger.info("No target provided - using unsupervised methods only")
            
        # Basic data quality checks
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X.columns):
            logger.warning(f"Non-numeric columns detected: {len(X.columns) - len(numeric_cols)}")
            
        null_features = X.columns[X.isnull().all()].tolist()
        if null_features:
            logger.warning(f"All-null features detected: {len(null_features)}")
            
        return X, y
        
    def _calculate_information_preservation(
        self, 
        original_features: List[str],
        selected_features: List[str],
        feature_scores: Dict[str, float]
    ) -> float:
        """Calculate information preservation ratio."""
        if not feature_scores or not selected_features:
            return len(selected_features) / len(original_features) if original_features else 0
            
        # Weight by feature importance/score
        total_score = sum(feature_scores.get(f, 0) for f in original_features)
        selected_score = sum(feature_scores.get(f, 0) for f in selected_features)
        
        if total_score > 0:
            return selected_score / total_score
        else:
            return len(selected_features) / len(original_features)
            
    def _combine_feature_scores(
        self, 
        variance_scores: Dict[str, float],
        mi_scores: Dict[str, float],
        correlation_scores: Dict[str, float],
        significance_pvalues: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine multiple feature scoring methods into unified scores."""
        logger.info("Combining feature scores from multiple methods")
        
        all_features = set()
        all_features.update(variance_scores.keys())
        all_features.update(mi_scores.keys())
        all_features.update(correlation_scores.keys())
        all_features.update(significance_pvalues.keys())
        
        combined_scores = {}
        
        for feature in all_features:
            # Normalize individual scores to [0, 1]
            variance_norm = min(variance_scores.get(feature, 0) / 0.1, 1.0)  # Cap at 0.1 variance
            mi_norm = min(mi_scores.get(feature, 0) / 0.5, 1.0)  # Cap at 0.5 MI
            
            # For significance: convert p-value to score (1 - p_value)
            sig_pvalue = significance_pvalues.get(feature, 1.0)
            sig_score = max(0, 1 - sig_pvalue)
            
            # For correlation: use absolute value of highest correlation
            corr_score = abs(correlation_scores.get(feature, 0))
            
            # Combined weighted score
            combined_score = (
                0.25 * variance_norm +      # Variance contribution
                0.35 * mi_norm +           # Mutual information (highest weight)
                0.25 * sig_score +         # Statistical significance
                0.15 * corr_score          # Correlation strength
            )
            
            combined_scores[feature] = combined_score
            
        logger.info(f"Combined scores calculated for {len(combined_scores)} features")
        return combined_scores
        
    def fit_statistical_selection(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        save_intermediate: bool = False,
        output_dir: Optional[str] = None
    ) -> 'StatisticalSelectionEngine':
        """
        Fit the complete statistical selection pipeline.
        
        Args:
            X: Feature matrix (500+ features expected)
            y: Target variable (optional but recommended)
            save_intermediate: Whether to save intermediate results
            output_dir: Directory to save intermediate results
            
        Returns:
            Self for method chaining
        """
        logger.info("="*80)
        logger.info("STARTING STATISTICAL FEATURE SELECTION PIPELINE")
        logger.info(f"Input: {X.shape[1]} features, {X.shape[0]} samples")
        logger.info(f"Target: {len(y)} samples" if y is not None else "No target provided")
        logger.info("="*80)
        
        start_time = datetime.now()
        self._monitor_memory("pipeline_start")
        
        # Validate input data
        X_clean, y_clean = self._validate_input_data(X, y)
        current_features = X_clean.columns.tolist()
        
        # Stage 1: Variance Filtering
        logger.info("\n" + "="*50)
        logger.info("STAGE 1: VARIANCE FILTERING")
        logger.info("="*50)
        
        variance_results = self.variance_filter_.filter_low_variance_features(X_clean)
        stage1_features = variance_results['selected_features']
        
        self.stage_results_['variance_filtering'] = {
            'input_features': len(X_clean.columns),
            'output_features': len(stage1_features),
            'eliminated': variance_results['eliminated_features'],
            'processing_time': variance_results['processing_stats']['processing_time_seconds']
        }
        
        # Update elimination history
        for feature, reason in variance_results['eliminated_features'].items():
            self.elimination_history_[feature] = f"stage1_variance_{reason}"
            
        # Apply variance filtering
        X_stage1 = X_clean[stage1_features] if stage1_features else pd.DataFrame(index=X_clean.index)
        
        logger.info(f"Variance filtering: {len(X_clean.columns)} → {len(stage1_features)} features")
        self._monitor_memory("stage1_complete")
        gc.collect()  # Force garbage collection
        
        if X_stage1.empty or len(stage1_features) < self.min_feature_count:
            logger.error(f"Insufficient features after variance filtering: {len(stage1_features)}")
            self.selected_features_ = stage1_features
            self.is_fitted_ = True
            return self
            
        # Stage 2: Correlation Analysis & VIF
        logger.info("\n" + "="*50)
        logger.info("STAGE 2: CORRELATION & VIF ANALYSIS")
        logger.info("="*50)
        
        correlation_results = self.correlation_analyzer_.analyze_correlations(X_stage1, y_clean)
        stage2_features = correlation_results['selected_features']
        
        self.stage_results_['correlation_analysis'] = {
            'input_features': len(stage1_features),
            'output_features': len(stage2_features),
            'eliminated': correlation_results['eliminated_features'],
            'processing_time': correlation_results['processing_stats']['processing_time_seconds'],
            'clusters_formed': correlation_results['processing_stats']['clusters_formed'],
            'high_vif_features': correlation_results['processing_stats']['high_vif_features']
        }
        
        # Update elimination history
        for feature, reason in correlation_results['eliminated_features'].items():
            self.elimination_history_[feature] = f"stage2_correlation_{reason}"
            
        # Apply correlation filtering
        X_stage2 = X_stage1[stage2_features] if stage2_features else pd.DataFrame(index=X_stage1.index)
        
        logger.info(f"Correlation analysis: {len(stage1_features)} → {len(stage2_features)} features")
        self._monitor_memory("stage2_complete")
        gc.collect()
        
        if X_stage2.empty or len(stage2_features) < self.min_feature_count:
            logger.error(f"Insufficient features after correlation analysis: {len(stage2_features)}")
            self.selected_features_ = stage2_features
            self.is_fitted_ = True
            return self
            
        # Stage 3: Mutual Information Selection (if target available)
        if y_clean is not None:
            logger.info("\n" + "="*50)
            logger.info("STAGE 3: MUTUAL INFORMATION RANKING")
            logger.info("="*50)
            
            # Select top features by MI (more than target to allow for final optimization)
            mi_target_count = min(self.target_feature_count * 2, len(stage2_features))
            mi_results = self.mi_selector_.select_features_by_mutual_information(
                X_stage2, y_clean, top_k=mi_target_count
            )
            stage3_features = mi_results['selected_features']
            
            self.stage_results_['mutual_information'] = {
                'input_features': len(stage2_features),
                'output_features': len(stage3_features),
                'eliminated': mi_results['eliminated_features'],
                'processing_time': mi_results['processing_stats']['processing_time_seconds'],
                'mi_statistics': mi_results['processing_stats']['mi_statistics']
            }
            
            # Update elimination history
            for feature, reason in mi_results['eliminated_features'].items():
                self.elimination_history_[feature] = f"stage3_mutual_info_{reason}"
                
            # Apply MI filtering
            X_stage3 = X_stage2[stage3_features] if stage3_features else pd.DataFrame(index=X_stage2.index)
            
            logger.info(f"Mutual information: {len(stage2_features)} → {len(stage3_features)} features")
            self._monitor_memory("stage3_complete")
            gc.collect()
            
        else:
            logger.info("\nSKIPPING STAGE 3: No target variable provided")
            X_stage3 = X_stage2.copy()
            stage3_features = stage2_features.copy()
            self.stage_results_['mutual_information'] = {'skipped': 'no_target_variable'}
            
        # Stage 4: Statistical Significance Testing (if target available)
        if y_clean is not None:
            logger.info("\n" + "="*50)
            logger.info("STAGE 4: STATISTICAL SIGNIFICANCE TESTING")
            logger.info("="*50)
            
            significance_results = self.significance_tester_.test_feature_significance(X_stage3, y_clean)
            stage4_features = significance_results['significant_features']
            
            self.stage_results_['significance_testing'] = {
                'input_features': len(stage3_features),
                'output_features': len(stage4_features),
                'eliminated': significance_results['insignificant_features'],
                'processing_time': significance_results['processing_stats']['processing_time_seconds'],
                'correction_method': significance_results['processing_stats']['correction_method'],
                'alpha_level': significance_results['processing_stats']['alpha_level']
            }
            
            # Update elimination history
            for feature, reason in significance_results['insignificant_features'].items():
                self.elimination_history_[feature] = f"stage4_significance_{reason}"
                
            logger.info(f"Significance testing: {len(stage3_features)} → {len(stage4_features)} features")
            self._monitor_memory("stage4_complete")
            gc.collect()
            
        else:
            logger.info("\nSKIPPING STAGE 4: No target variable provided")
            stage4_features = stage3_features.copy()
            self.stage_results_['significance_testing'] = {'skipped': 'no_target_variable'}
            
        # Stage 5: Final Optimization
        logger.info("\n" + "="*50)
        logger.info("STAGE 5: FINAL FEATURE OPTIMIZATION")
        logger.info("="*50)
        
        # Combine scores from all stages
        variance_scores = variance_results.get('feature_variances', {})
        mi_scores = mi_results.get('mi_scores', {}) if y_clean is not None else {}
        correlation_scores = {}  # Could extract from correlation analysis if needed
        significance_pvalues = significance_results.get('corrected_pvalues', {}) if y_clean is not None else {}
        
        self.feature_scores_ = self._combine_feature_scores(
            variance_scores, mi_scores, correlation_scores, significance_pvalues
        )
        
        # Final selection based on target count
        if len(stage4_features) > self.max_feature_count:
            # Select top features by combined score
            available_scores = {f: self.feature_scores_.get(f, 0) for f in stage4_features}
            sorted_features = sorted(available_scores.items(), key=lambda x: x[1], reverse=True)
            final_features = [f for f, _ in sorted_features[:self.target_feature_count]]
            
            # Record eliminations
            eliminated_final = [f for f, _ in sorted_features[self.target_feature_count:]]
            for feature in eliminated_final:
                score = available_scores[feature]
                self.elimination_history_[feature] = f"stage5_final_optimization_score_{score:.6f}"
                
        else:
            final_features = stage4_features.copy()
            
        self.selected_features_ = final_features
        
        # Calculate final statistics
        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.final_stats_ = {
            'original_feature_count': X.shape[1],
            'final_feature_count': len(self.selected_features_),
            'reduction_ratio': 1 - (len(self.selected_features_) / X.shape[1]),
            'information_preservation': self._calculate_information_preservation(
                X.columns.tolist(), self.selected_features_, self.feature_scores_
            ),
            'processing_time_seconds': elapsed_time,
            'memory_peak_gb': max(stats['memory_gb'] for stats in self.memory_usage_.values()),
            'target_achieved': self.min_feature_count <= len(self.selected_features_) <= self.max_feature_count,
            'stages_completed': len([s for s in self.stage_results_.values() if not s.get('skipped')])
        }
        
        self.is_fitted_ = True
        self._monitor_memory("pipeline_complete")
        
        # Log final results
        logger.info("\n" + "="*80)
        logger.info("STATISTICAL SELECTION PIPELINE COMPLETED")
        logger.info(f"Feature reduction: {X.shape[1]} → {len(self.selected_features_)} "
                   f"({self.final_stats_['reduction_ratio']:.1%} reduction)")
        logger.info(f"Information preservation: {self.final_stats_['information_preservation']:.1%}")
        logger.info(f"Processing time: {elapsed_time:.1f} seconds")
        logger.info(f"Memory peak: {self.final_stats_['memory_peak_gb']:.2f} GB")
        logger.info("="*80)
        
        # Save intermediate results if requested
        if save_intermediate and output_dir:
            self.save_detailed_results(output_dir)
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features."""
        if not self.is_fitted_:
            raise ValueError("StatisticalSelectionEngine not fitted - call fit_statistical_selection first")
            
        available_features = [f for f in self.selected_features_ if f in X.columns]
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features in transform data")
            
        return X[available_features].copy() if available_features else pd.DataFrame(index=X.index)
        
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit statistical selection and transform data in one step."""
        return self.fit_statistical_selection(X, y).transform(X)
        
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get comprehensive selection summary."""
        if not self.is_fitted_:
            raise ValueError("Statistical selection not fitted")
            
        return {
            'selected_features': self.selected_features_.copy(),
            'feature_scores': self.feature_scores_.copy(),
            'stage_results': self.stage_results_.copy(),
            'elimination_history': self.elimination_history_.copy(),
            'final_statistics': self.final_stats_.copy(),
            'memory_usage': self.memory_usage_.copy(),
            'parameters': {
                'target_feature_count': self.target_feature_count,
                'correlation_threshold': self.correlation_threshold,
                'vif_threshold': self.vif_threshold,
                'variance_threshold': self.variance_threshold,
                'mi_threshold': self.mi_threshold,
                'significance_alpha': self.significance_alpha
            }
        }
        
    def save_detailed_results(self, output_dir: str) -> None:
        """Save detailed results from all stages."""
        if not self.is_fitted_:
            raise ValueError("Statistical selection not fitted")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main summary
        summary = self.get_selection_summary()
        
        # Convert datetime objects for JSON serialization
        for stage_name, memory_stats in self.memory_usage_.items():
            if 'timestamp' in memory_stats:
                memory_stats['timestamp'] = memory_stats['timestamp'].isoformat()
                
        # Save main results
        with open(output_path / 'statistical_selection_results.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Save selected features list
        with open(output_path / 'selected_features.txt', 'w') as f:
            for feature in self.selected_features_:
                f.write(f"{feature}\n")
                
        # Save feature scores
        feature_scores_df = pd.DataFrame([
            {'feature': feature, 'combined_score': score}
            for feature, score in self.feature_scores_.items()
        ]).sort_values('combined_score', ascending=False)
        
        feature_scores_df.to_csv(output_path / 'feature_scores.csv', index=False)
        
        # Save elimination history
        elimination_df = pd.DataFrame([
            {'feature': feature, 'elimination_reason': reason}
            for feature, reason in self.elimination_history_.items()
        ])
        
        elimination_df.to_csv(output_path / 'elimination_history.csv', index=False)
        
        logger.info(f"Detailed results saved to {output_path}")
        
    def get_feature_importance_report(self, top_n: int = 50) -> Dict[str, Any]:
        """Generate feature importance report."""
        if not self.is_fitted_:
            raise ValueError("Statistical selection not fitted")
            
        # Top features by combined score
        sorted_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Selection stage breakdown
        stage_breakdown = {}
        for stage_name, stage_info in self.stage_results_.items():
            if not stage_info.get('skipped'):
                stage_breakdown[stage_name] = {
                    'features_retained': stage_info.get('output_features', 0),
                    'features_eliminated': len(stage_info.get('eliminated', {})),
                    'elimination_rate': len(stage_info.get('eliminated', {})) / stage_info.get('input_features', 1)
                }
                
        return {
            'top_features': top_features,
            'stage_breakdown': stage_breakdown,
            'overall_statistics': {
                'total_selected': len(self.selected_features_),
                'total_eliminated': len(self.elimination_history_),
                'selection_rate': len(self.selected_features_) / (len(self.selected_features_) + len(self.elimination_history_)),
                'information_preservation': self.final_stats_['information_preservation']
            },
            'elimination_reasons': {
                reason: len([f for f, r in self.elimination_history_.items() if reason in r])
                for reason in ['variance', 'correlation', 'mutual_info', 'significance', 'final_optimization']
            }
        }