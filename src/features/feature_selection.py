"""
Feature Selection Algorithms - Task #28 Stream C

Automated feature selection for Taiwan stock market using correlation filtering,
importance ranking, and statistical validation to reduce feature space from
1000+ generated features to 200-500 high-quality features.

Key Features:
- Correlation-based filtering to remove redundant features  
- Statistical importance ranking using multiple methods
- Taiwan market compliance validation
- Memory-efficient processing for large feature sets
- Integration with OpenFE feature generation pipeline
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from pathlib import Path
import gc

# Statistical and ML imports
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, mutual_info_regression, 
    mutual_info_classif, VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import psutil

# Import project modules
from .taiwan_config import TaiwanMarketConfig
from .quality_metrics import FeatureQualityMetrics

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Comprehensive feature selection for Taiwan stock market.
    
    Reduces feature space from 1000+ generated features to 200-500 high-quality
    features through multiple selection strategies:
    
    1. Variance filtering - remove low-variance features
    2. Correlation filtering - remove highly correlated features  
    3. Statistical tests - univariate feature selection
    4. Model-based selection - tree/linear model importance
    5. Taiwan compliance - market-specific constraints
    """
    
    def __init__(
        self,
        target_feature_count: int = 350,
        min_feature_count: int = 200,
        max_feature_count: int = 500,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01,
        taiwan_config: Optional[TaiwanMarketConfig] = None,
        memory_limit_gb: float = 8.0,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize feature selector with Taiwan market parameters.
        
        Args:
            target_feature_count: Target number of features to select
            min_feature_count: Minimum acceptable number of features
            max_feature_count: Maximum acceptable number of features  
            correlation_threshold: Correlation threshold for filtering
            variance_threshold: Minimum variance threshold
            taiwan_config: Taiwan market configuration
            memory_limit_gb: Memory limit in GB
            n_jobs: Number of parallel jobs
            random_state: Random seed for reproducibility
        """
        self.target_feature_count = target_feature_count
        self.min_feature_count = min_feature_count
        self.max_feature_count = max_feature_count
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.taiwan_config = taiwan_config or TaiwanMarketConfig()
        self.memory_limit_gb = memory_limit_gb
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Initialize quality metrics
        self.quality_metrics = FeatureQualityMetrics(taiwan_config=taiwan_config)
        
        # Selection state
        self.is_fitted_ = False
        self.selected_features_ = []
        self.feature_scores_ = {}
        self.selection_stages_ = {}
        self.memory_usage_ = {}
        self.elimination_reasons_ = {}
        
        # Selection methods
        self.selectors_ = {}
        
    def _check_memory_usage(self, stage: str) -> Dict[str, float]:
        """Monitor memory usage during feature selection."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = {
            'stage': stage,
            'rss_gb': memory_info.rss / 1024 / 1024 / 1024,
            'vms_gb': memory_info.vms / 1024 / 1024 / 1024,
            'percent': psutil.virtual_memory().percent
        }
        
        if memory_usage['rss_gb'] > self.memory_limit_gb:
            logger.warning(
                f"Memory usage ({memory_usage['rss_gb']:.2f}GB) exceeds limit "
                f"({self.memory_limit_gb}GB) at stage: {stage}"
            )
            
        self.memory_usage_[stage] = memory_usage
        return memory_usage
        
    def _validate_input_data(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """Validate input data for feature selection."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
        if X.empty:
            raise ValueError("X cannot be empty")
            
        # Check for all-NaN columns
        null_cols = X.columns[X.isnull().all()].tolist()
        if null_cols:
            logger.warning(f"Found {len(null_cols)} all-NaN columns: {null_cols[:5]}")
            
        # Check data types
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X.columns):
            logger.warning("Non-numeric columns detected - will be filtered out")
            
        # Check target if provided
        if y is not None:
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise TypeError("y must be pandas Series or numpy array")
            if len(y) != len(X):
                raise ValueError("X and y must have same number of samples")
                
    def remove_low_variance_features(
        self, 
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with variance below threshold.
        
        Args:
            X: Input features
            threshold: Variance threshold (uses instance default if None)
            
        Returns:
            (filtered_X, removed_features) tuple
        """
        threshold = threshold or self.variance_threshold
        logger.info(f"Removing low variance features (threshold={threshold})")
        
        # Calculate variance for numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        # Use sklearn's VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_numeric)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_cols = numeric_cols[selected_mask]
        removed_cols = numeric_cols[~selected_mask]
        
        # Store elimination reasons
        for col in removed_cols:
            self.elimination_reasons_[col] = f"low_variance_{X_numeric[col].var():.6f}"
        
        X_filtered = X[selected_cols].copy()
        
        logger.info(
            f"Variance filtering: {len(X.columns)} → {len(selected_cols)} features "
            f"(removed {len(removed_cols)})"
        )
        
        return X_filtered, removed_cols.tolist()
        
    def remove_highly_correlated_features(
        self, 
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features using hierarchical approach.
        
        For correlated pairs above threshold, keeps the feature with higher
        average absolute correlation with target (if available) or higher variance.
        
        Args:
            X: Input features  
            threshold: Correlation threshold (uses instance default if None)
            
        Returns:
            (filtered_X, removed_features) tuple
        """
        threshold = threshold or self.correlation_threshold
        logger.info(f"Removing highly correlated features (threshold={threshold})")
        self._check_memory_usage("correlation_start")
        
        # Calculate correlation matrix efficiently
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs (upper triangle)
        high_corr_pairs = []
        n_features = len(corr_matrix.columns)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr_value = corr_matrix.iloc[i, j]
                if not pd.isna(corr_value) and corr_value >= threshold:
                    feature_1 = corr_matrix.columns[i]
                    feature_2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feature_1, feature_2, corr_value))
        
        logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs")
        
        # Determine which features to remove
        features_to_remove = set()
        
        for feature_1, feature_2, corr_value in high_corr_pairs:
            # Skip if already marked for removal
            if feature_1 in features_to_remove or feature_2 in features_to_remove:
                continue
                
            # Choose feature to remove based on variance (higher variance wins)
            var_1 = X[feature_1].var()
            var_2 = X[feature_2].var()
            
            if var_1 >= var_2:
                remove_feature = feature_2
                keep_feature = feature_1
            else:
                remove_feature = feature_1
                keep_feature = feature_2
                
            features_to_remove.add(remove_feature)
            self.elimination_reasons_[remove_feature] = f"high_correlation_with_{keep_feature}_{corr_value:.3f}"
        
        # Create filtered dataset
        remaining_features = [col for col in X.columns if col not in features_to_remove]
        X_filtered = X[remaining_features].copy()
        
        self._check_memory_usage("correlation_end")
        
        logger.info(
            f"Correlation filtering: {len(X.columns)} → {len(remaining_features)} features "
            f"(removed {len(features_to_remove)})"
        )
        
        return X_filtered, list(features_to_remove)
        
    def select_features_univariate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        k: int = 100,
        task_type: str = 'regression'
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """
        Select top k features using univariate statistical tests.
        
        Args:
            X: Input features
            y: Target variable
            k: Number of features to select
            task_type: 'regression' or 'classification'
            
        Returns:
            (selected_X, selected_features, scores) tuple
        """
        logger.info(f"Univariate feature selection: top {k} features for {task_type}")
        
        # Choose appropriate scoring function
        if task_type == 'regression':
            score_func = f_regression
        else:
            score_func = f_classif
            
        # Fit selector
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        
        try:
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()
            
            # Get feature scores
            scores = dict(zip(X.columns[selected_mask], selector.scores_[selected_mask]))
            
            # Store elimination reasons for removed features
            removed_features = X.columns[~selected_mask]
            for feature in removed_features:
                idx = X.columns.get_loc(feature)
                score = selector.scores_[idx] if idx < len(selector.scores_) else 0
                self.elimination_reasons_[feature] = f"univariate_score_{score:.3f}_below_top_{k}"
            
            X_result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        except Exception as e:
            logger.error(f"Univariate selection failed: {str(e)}")
            # Fallback: select by variance
            variances = X.var().sort_values(ascending=False)
            selected_features = variances.head(k).index.tolist()
            X_result = X[selected_features].copy()
            scores = variances.head(k).to_dict()
            
        logger.info(f"Univariate selection completed: {len(selected_features)} features")
        return X_result, selected_features, scores
        
    def select_features_model_based(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest',
        task_type: str = 'regression',
        max_features: int = 200
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
        """
        Select features using model-based importance.
        
        Args:
            X: Input features
            y: Target variable
            model_type: 'random_forest', 'lasso', or 'elastic_net'
            task_type: 'regression' or 'classification'
            max_features: Maximum number of features to select
            
        Returns:
            (selected_X, selected_features, importances) tuple
        """
        logger.info(f"Model-based selection using {model_type} for {task_type}")
        self._check_memory_usage("model_selection_start")
        
        try:
            # Choose appropriate model
            if model_type == 'random_forest':
                if task_type == 'regression':
                    model = RandomForestRegressor(
                        n_estimators=100, 
                        random_state=self.random_state,
                        n_jobs=self.n_jobs
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=self.random_state, 
                        n_jobs=self.n_jobs
                    )
                    
            elif model_type == 'lasso':
                model = LassoCV(random_state=self.random_state, n_jobs=self.n_jobs)
                
            elif model_type == 'elastic_net':
                model = ElasticNetCV(random_state=self.random_state, n_jobs=self.n_jobs)
                
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            # Fit model
            model.fit(X, y)
            
            # Get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = np.mean(importances, axis=0)
            else:
                raise ValueError(f"Model {model_type} does not provide feature importance")
                
            # Create importance dictionary
            importance_dict = dict(zip(X.columns, importances))
            
            # Select top features by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, _ in sorted_features[:max_features]]
            
            # Store elimination reasons
            eliminated_features = [feat for feat, _ in sorted_features[max_features:]]
            for feature in eliminated_features:
                importance = importance_dict[feature]
                self.elimination_reasons_[feature] = f"{model_type}_importance_{importance:.6f}_below_top_{max_features}"
            
            X_selected = X[selected_features].copy()
            selected_importances = {feat: importance_dict[feat] for feat in selected_features}
            
        except Exception as e:
            logger.error(f"Model-based selection failed: {str(e)}")
            # Fallback: select by variance
            variances = X.var().sort_values(ascending=False)
            selected_features = variances.head(max_features).index.tolist()
            X_selected = X[selected_features].copy()
            selected_importances = variances.head(max_features).to_dict()
            
        self._check_memory_usage("model_selection_end")
        
        logger.info(f"Model-based selection completed: {len(selected_features)} features")
        return X_selected, selected_features, selected_importances
        
    def apply_taiwan_market_filters(
        self, 
        X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply Taiwan market specific feature filters.
        
        Removes features that violate Taiwan market constraints:
        - Features that don't respect T+2 settlement
        - Features using non-trading hour data
        - Features that violate price limit constraints
        
        Args:
            X: Input features
            
        Returns:
            (filtered_X, removed_features) tuple
        """
        logger.info("Applying Taiwan market compliance filters")
        
        removed_features = []
        taiwan_compliant_features = []
        
        # Check each feature for Taiwan compliance
        for feature_name in X.columns:
            # Check if feature name suggests it might violate Taiwan rules
            violations = []
            
            # Check for intraday features during non-trading hours
            if any(term in feature_name.lower() for term in 
                   ['overnight', 'after_hours', 'pre_market', 'extended']):
                violations.append("non_trading_hours")
                
            # Check for features that might not respect T+2 settlement
            if any(term in feature_name.lower() for term in 
                   ['instant', 'realtime', 'immediate', 'same_day']):
                violations.append("t2_settlement_violation")
                
            # Check for features using prohibited data
            if any(term in feature_name.lower() for term in 
                   ['short_interest', 'margin_call', 'insider']):
                violations.append("prohibited_data")
            
            if violations:
                removed_features.append(feature_name)
                self.elimination_reasons_[feature_name] = f"taiwan_compliance_{'+'.join(violations)}"
            else:
                taiwan_compliant_features.append(feature_name)
        
        # Additional Taiwan-specific validation using quality metrics
        for feature_name in taiwan_compliant_features.copy():
            feature_data = X[feature_name]
            
            # Use quality metrics for validation
            quality_result = self.quality_metrics.validate_feature_quality(
                feature_data, 
                feature_name=feature_name
            )
            
            if not quality_result['taiwan_compliant']:
                taiwan_compliant_features.remove(feature_name)
                removed_features.append(feature_name)
                self.elimination_reasons_[feature_name] = "taiwan_quality_validation_failed"
        
        X_filtered = X[taiwan_compliant_features].copy()
        
        logger.info(
            f"Taiwan compliance filtering: {len(X.columns)} → {len(taiwan_compliant_features)} features "
            f"(removed {len(removed_features)})"
        )
        
        return X_filtered, removed_features
        
    def fit(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        task_type: str = 'regression'
    ) -> 'FeatureSelector':
        """
        Fit the feature selector using multi-stage selection process.
        
        Selection stages:
        1. Variance filtering
        2. Correlation filtering  
        3. Taiwan compliance filtering
        4. Univariate selection (if target provided)
        5. Model-based selection (if target provided)
        6. Final selection to target count
        
        Args:
            X: Input features
            y: Target variable (optional)
            task_type: 'regression' or 'classification'
            
        Returns:
            Self for method chaining
        """
        logger.info(
            f"Starting feature selection: {X.shape[1]} features → "
            f"{self.target_feature_count} target features"
        )
        
        start_time = datetime.now()
        self._check_memory_usage("selection_start")
        
        # Validate input
        self._validate_input_data(X, y)
        
        current_X = X.copy()
        current_features = X.columns.tolist()
        
        # Stage 1: Variance filtering
        logger.info("=== Stage 1: Variance Filtering ===")
        current_X, removed = self.remove_low_variance_features(current_X)
        self.selection_stages_['variance_filtering'] = {
            'removed': removed,
            'remaining': len(current_X.columns)
        }
        
        # Stage 2: Correlation filtering  
        logger.info("=== Stage 2: Correlation Filtering ===")
        current_X, removed = self.remove_highly_correlated_features(current_X)
        self.selection_stages_['correlation_filtering'] = {
            'removed': removed,
            'remaining': len(current_X.columns)
        }
        
        # Stage 3: Taiwan compliance filtering
        logger.info("=== Stage 3: Taiwan Market Compliance ===")
        current_X, removed = self.apply_taiwan_market_filters(current_X)
        self.selection_stages_['taiwan_compliance'] = {
            'removed': removed,
            'remaining': len(current_X.columns)
        }
        
        # If we have target, apply supervised selection
        if y is not None:
            # Stage 4: Univariate selection
            logger.info("=== Stage 4: Univariate Selection ===")
            univariate_k = min(self.target_feature_count * 2, len(current_X.columns))
            current_X, selected_features, scores = self.select_features_univariate(
                current_X, y, k=univariate_k, task_type=task_type
            )
            self.feature_scores_['univariate'] = scores
            self.selection_stages_['univariate'] = {
                'selected': selected_features,
                'remaining': len(current_X.columns)
            }
            
            # Stage 5: Model-based selection  
            logger.info("=== Stage 5: Model-based Selection ===")
            model_features = min(self.target_feature_count, len(current_X.columns))
            current_X, selected_features, importances = self.select_features_model_based(
                current_X, y, max_features=model_features, task_type=task_type
            )
            self.feature_scores_['model_based'] = importances
            self.selection_stages_['model_based'] = {
                'selected': selected_features,
                'remaining': len(current_X.columns)
            }
        
        # Stage 6: Final selection to target count
        logger.info("=== Stage 6: Final Selection ===")
        if len(current_X.columns) > self.max_feature_count:
            # Use variance as tiebreaker for final selection
            variances = current_X.var().sort_values(ascending=False)
            final_features = variances.head(self.target_feature_count).index.tolist()
            current_X = current_X[final_features]
            
            # Mark eliminated features
            eliminated = [f for f in variances.index if f not in final_features]
            for feature in eliminated:
                if feature not in self.elimination_reasons_:
                    self.elimination_reasons_[feature] = f"final_variance_cutoff_{variances[feature]:.6f}"
        
        # Store final results
        self.selected_features_ = current_X.columns.tolist()
        self.is_fitted_ = True
        
        # Calculate processing time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Final validation
        final_count = len(self.selected_features_)
        if final_count < self.min_feature_count:
            logger.warning(
                f"Final feature count ({final_count}) below minimum ({self.min_feature_count})"
            )
        elif final_count > self.max_feature_count:
            logger.warning(
                f"Final feature count ({final_count}) above maximum ({self.max_feature_count})"
            )
            
        self._check_memory_usage("selection_end")
        
        logger.info(
            f"Feature selection completed in {elapsed_time:.1f}s: "
            f"{X.shape[1]} → {final_count} features "
            f"({final_count/X.shape[1]*100:.1f}% retention)"
        )
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted feature selector."""
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted before transform")
            
        # Select only the chosen features
        available_features = [f for f in self.selected_features_ if f in X.columns]
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features in transform data")
            
        return X[available_features].copy()
        
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        task_type: str = 'regression'
    ) -> pd.DataFrame:
        """Fit selector and transform data in one step."""
        return self.fit(X, y, task_type).transform(X)
        
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted first")
        return self.selected_features_.copy()
        
    def get_feature_scores(self) -> Dict[str, Dict[str, float]]:
        """Get feature scores from different selection methods."""
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted first")
        return self.feature_scores_.copy()
        
    def get_elimination_reasons(self) -> Dict[str, str]:
        """Get reasons why features were eliminated."""
        return self.elimination_reasons_.copy()
        
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get comprehensive selection summary."""
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted first")
            
        return {
            'total_input_features': sum(stage.get('remaining', 0) for stage in self.selection_stages_.values()) // len(self.selection_stages_) if self.selection_stages_ else 0,
            'final_selected_features': len(self.selected_features_),
            'selection_stages': self.selection_stages_.copy(),
            'feature_scores': self.feature_scores_.copy(),
            'elimination_reasons': self.elimination_reasons_.copy(),
            'memory_usage': self.memory_usage_.copy(),
            'target_achieved': self.min_feature_count <= len(self.selected_features_) <= self.max_feature_count
        }
        
    def save_selection_results(self, output_path: str) -> None:
        """Save selection results to file."""
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted first")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create results dictionary
        results = {
            'selected_features': self.selected_features_,
            'selection_summary': self.get_selection_summary(),
            'taiwan_config': {
                'settlement_days': self.taiwan_config.SETTLEMENT_DAYS,
                'price_limit_percent': self.taiwan_config.PRICE_LIMIT_PERCENT,
                'trading_hours': f"{self.taiwan_config.TRADING_START}-{self.taiwan_config.TRADING_END}"
            }
        }
        
        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Selection results saved to {output_path}")


def create_feature_selector(
    target_features: int = 350,
    taiwan_config: Optional[TaiwanMarketConfig] = None,
    **kwargs
) -> FeatureSelector:
    """
    Factory function to create a properly configured FeatureSelector.
    
    Args:
        target_features: Target number of features to select
        taiwan_config: Taiwan market configuration
        **kwargs: Additional configuration options
        
    Returns:
        Configured FeatureSelector instance
    """
    # Default configuration for Taiwan market
    config = {
        'target_feature_count': target_features,
        'min_feature_count': max(200, int(target_features * 0.6)),
        'max_feature_count': min(500, int(target_features * 1.4)),
        'correlation_threshold': 0.95,
        'variance_threshold': 0.01,
        'memory_limit_gb': 8.0,
        'n_jobs': -1,
        'random_state': 42
    }
    
    # Apply overrides
    config.update(kwargs)
    
    # Create selector
    selector = FeatureSelector(
        taiwan_config=taiwan_config or TaiwanMarketConfig(),
        **config
    )
    
    logger.info(f"Created FeatureSelector with target {target_features} features")
    return selector