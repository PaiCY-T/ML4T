"""
Recursive Feature Elimination with Cross-Validation

Implementation of RFE with time-series cross-validation for stable feature selection
in financial time series data with Taiwan market considerations.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


@dataclass
class RFEConfig:
    """Configuration for Recursive Feature Elimination."""
    
    # RFE parameters
    n_features_to_select: Optional[int] = None  # If None, uses optimal from CV
    step: Union[int, float] = 0.1  # Remove 10% of features per step
    min_features: int = 10  # Minimum features to keep
    max_features: Optional[int] = None  # Maximum features to start with
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_scoring: str = 'ic'  # 'ic', 'mse', 'mae', 'r2'
    cv_tolerance: float = 0.001  # Tolerance for selecting optimal features
    
    # LightGBM parameters for RFE
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # Training parameters
    n_estimators: int = 100
    early_stopping_rounds: int = 10
    
    # Taiwan market parameters
    target_horizon: int = 5
    min_ic_threshold: float = 0.05
    stability_threshold: float = 0.7
    
    # Performance optimization
    parallel_jobs: int = 1  # Number of parallel jobs for CV
    memory_efficient: bool = True


class RecursiveFeatureEliminator:
    """
    Recursive Feature Elimination with time-series cross-validation.
    
    Eliminates features recursively based on model performance while maintaining
    stability across different time periods and market regimes.
    """
    
    def __init__(self, config: Optional[RFEConfig] = None):
        """Initialize the RFE selector.
        
        Args:
            config: Configuration for RFE
        """
        self.config = config or RFEConfig()
        self.selected_features_: List[str] = []
        self.feature_ranking_: Optional[pd.Series] = None
        self.cv_scores_: Optional[Dict[str, List[float]]] = None
        self.elimination_history_: List[Dict[str, Any]] = []
        self.optimal_n_features_: Optional[int] = None
        
        logger.info("Recursive Feature Eliminator initialized")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None
    ) -> 'RecursiveFeatureEliminator':
        """
        Fit the RFE selector to find optimal feature subset.
        
        Args:
            X: Feature matrix with MultiIndex (date, symbol)
            y: Target variable (forward returns)
            sample_weight: Optional sample weights
            feature_groups: Optional feature groupings for balanced elimination
            
        Returns:
            self
        """
        logger.info(f"Starting RFE with {X.shape[1]} initial features")
        
        # Prepare data
        X_clean, y_clean = self._prepare_data(X, y)
        
        # Initialize feature set
        current_features = list(X_clean.columns)
        
        # Determine elimination parameters
        if isinstance(self.config.step, float):
            # Percentage-based elimination
            step_size = max(1, int(len(current_features) * self.config.step))
        else:
            # Fixed number elimination
            step_size = self.config.step
        
        # RFE elimination loop
        step_num = 0
        best_score = -np.inf
        best_features = current_features.copy()
        
        while len(current_features) > self.config.min_features:
            step_num += 1
            logger.info(f"RFE Step {step_num}: {len(current_features)} features")
            
            # Cross-validate current feature set
            cv_scores = self._cross_validate_features(
                X_clean[current_features], y_clean, sample_weight
            )
            
            # Calculate mean CV score
            mean_score = np.mean(cv_scores['scores'])
            std_score = np.std(cv_scores['scores'])
            
            # Record step
            step_info = {
                'step': step_num,
                'n_features': len(current_features),
                'features': current_features.copy(),
                'cv_score': mean_score,
                'cv_std': std_score,
                'feature_importances': cv_scores.get('importances', {})
            }
            self.elimination_history_.append(step_info)
            
            # Check if this is the best performing set
            if mean_score > best_score + self.config.cv_tolerance:
                best_score = mean_score
                best_features = current_features.copy()
                logger.info(f"New best score: {best_score:.6f} with {len(best_features)} features")
            
            # Stop if we've reached the target number of features
            if (self.config.n_features_to_select is not None and 
                len(current_features) <= self.config.n_features_to_select):
                break
            
            # Eliminate least important features
            if len(current_features) > self.config.min_features:
                features_to_remove = self._select_features_to_eliminate(
                    current_features, 
                    cv_scores.get('importances', {}),
                    step_size,
                    feature_groups
                )
                
                current_features = [f for f in current_features if f not in features_to_remove]
                
                logger.info(f"Eliminated {len(features_to_remove)} features: {features_to_remove[:3]}...")
        
        # Select optimal feature set
        if self.config.n_features_to_select is not None:
            # Use specified number of features
            self.selected_features_ = self._get_top_n_features(
                best_features, self.config.n_features_to_select
            )
            self.optimal_n_features_ = self.config.n_features_to_select
        else:
            # Use best performing feature set
            self.selected_features_ = best_features
            self.optimal_n_features_ = len(best_features)
        
        # Generate final feature ranking
        self.feature_ranking_ = self._generate_feature_ranking(X_clean.columns)
        
        # Store final CV results
        self.cv_scores_ = self._cross_validate_features(
            X_clean[self.selected_features_], y_clean, sample_weight
        )
        
        logger.info(f"RFE completed. Selected {len(self.selected_features_)} features")
        logger.info(f"Final CV score: {np.mean(self.cv_scores_['scores']):.6f} ± {np.std(self.cv_scores_['scores']):.6f}")
        
        return self
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and clean data for RFE."""
        # Remove NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Limit features if specified
        if self.config.max_features is not None:
            if len(X_clean.columns) > self.config.max_features:
                # Select top features by variance as initial filter
                feature_var = X_clean.var().sort_values(ascending=False)
                selected_cols = feature_var.head(self.config.max_features).index
                X_clean = X_clean[selected_cols]
                logger.info(f"Limited to {self.config.max_features} features by variance")
        
        logger.info(f"Data prepared: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        
        return X_clean, y_clean
    
    def _cross_validate_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Cross-validate current feature set."""
        
        cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        cv_scores = []
        cv_importances = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Sample weights for fold
            train_weight = sample_weight[train_idx] if sample_weight is not None else None
            val_weight = sample_weight[val_idx] if sample_weight is not None else None
            
            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                **self.config.lgb_params
            )
            
            model.fit(
                X_train, y_train,
                sample_weight=train_weight,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[val_weight] if val_weight is not None else None,
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds),
                    lgb.log_evaluation(0)  # Silent
                ]
            )
            
            # Predict and score
            y_pred = model.predict(X_val)
            fold_score = self._calculate_score(y_val, y_pred, self.config.cv_scoring)
            cv_scores.append(fold_score)
            
            # Store feature importance
            fold_importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            )
            cv_importances.append(fold_importance)
        
        # Aggregate importance across folds
        mean_importance = pd.concat(cv_importances, axis=1).mean(axis=1)
        
        return {
            'scores': cv_scores,
            'importances': mean_importance.to_dict()
        }
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray, scoring: str) -> float:
        """Calculate evaluation score based on scoring method."""
        
        if scoring == 'ic':
            # Information Coefficient (Spearman correlation)
            return pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        elif scoring == 'mse':
            return -mean_squared_error(y_true, y_pred)  # Negative for maximization
        elif scoring == 'mae':
            return -np.mean(np.abs(y_true - y_pred))  # Negative for maximization
        elif scoring == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown scoring method: {scoring}")
    
    def _select_features_to_eliminate(
        self,
        current_features: List[str],
        importances: Dict[str, float],
        step_size: int,
        feature_groups: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """Select features to eliminate based on importance and grouping."""
        
        # Sort features by importance (ascending for elimination)
        feature_importance_pairs = [(f, importances.get(f, 0)) for f in current_features]
        feature_importance_pairs.sort(key=lambda x: x[1])
        
        # Select features to eliminate
        if feature_groups is None:
            # Simple elimination by lowest importance
            features_to_remove = [f for f, _ in feature_importance_pairs[:step_size]]
        else:
            # Group-balanced elimination
            features_to_remove = self._balanced_elimination(
                feature_importance_pairs, step_size, feature_groups
            )
        
        return features_to_remove
    
    def _balanced_elimination(
        self,
        feature_importance_pairs: List[Tuple[str, float]],
        step_size: int,
        feature_groups: Dict[str, List[str]]
    ) -> List[str]:
        """Eliminate features while maintaining group balance."""
        
        # Group features by their categories
        grouped_features = {}
        ungrouped_features = []
        
        for feature, importance in feature_importance_pairs:
            assigned = False
            for group_name, group_features in feature_groups.items():
                if feature in group_features:
                    if group_name not in grouped_features:
                        grouped_features[group_name] = []
                    grouped_features[group_name].append((feature, importance))
                    assigned = True
                    break
            
            if not assigned:
                ungrouped_features.append((feature, importance))
        
        # Eliminate proportionally from each group
        features_to_remove = []
        remaining_to_remove = step_size
        
        # Calculate elimination per group
        total_grouped_features = sum(len(group) for group in grouped_features.values())
        total_features = total_grouped_features + len(ungrouped_features)
        
        for group_name, group_features in grouped_features.items():
            group_size = len(group_features)
            group_elimination = max(1, int(step_size * group_size / total_features))
            group_elimination = min(group_elimination, remaining_to_remove, group_size - 1)
            
            # Sort group features by importance and eliminate worst
            group_features.sort(key=lambda x: x[1])
            for i in range(group_elimination):
                features_to_remove.append(group_features[i][0])
                remaining_to_remove -= 1
                if remaining_to_remove <= 0:
                    break
        
        # Eliminate from ungrouped features if needed
        if remaining_to_remove > 0 and ungrouped_features:
            ungrouped_features.sort(key=lambda x: x[1])
            for i in range(min(remaining_to_remove, len(ungrouped_features))):
                features_to_remove.append(ungrouped_features[i][0])
        
        return features_to_remove
    
    def _get_top_n_features(self, features: List[str], n: int) -> List[str]:
        """Get top N features from the elimination history."""
        
        if not self.elimination_history_:
            return features[:n]
        
        # Find step with feature count closest to n
        best_step = None
        min_diff = float('inf')
        
        for step in self.elimination_history_:
            diff = abs(step['n_features'] - n)
            if diff < min_diff:
                min_diff = diff
                best_step = step
        
        if best_step and len(best_step['features']) >= n:
            # Sort by importance and select top N
            importances = best_step.get('feature_importances', {})
            feature_importance_pairs = [(f, importances.get(f, 0)) for f in best_step['features']]
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return [f for f, _ in feature_importance_pairs[:n]]
        
        return features[:n]
    
    def _generate_feature_ranking(self, all_features: pd.Index) -> pd.Series:
        """Generate final feature ranking based on elimination order."""
        
        ranking = pd.Series(index=all_features, dtype=int, name='elimination_rank')
        
        # Features never eliminated get rank 1
        for feature in self.selected_features_:
            ranking[feature] = 1
        
        # Assign ranks based on elimination step
        current_rank = len(self.selected_features_) + 1
        
        for step in reversed(self.elimination_history_):
            step_features = set(step['features'])
            next_step_features = set()
            
            # Find features in next step
            step_idx = step['step']
            if step_idx < len(self.elimination_history_):
                next_step_features = set(self.elimination_history_[step_idx]['features'])
            else:
                next_step_features = set(self.selected_features_)
            
            # Features eliminated in this step
            eliminated_features = step_features - next_step_features
            
            for feature in eliminated_features:
                if feature in ranking.index:
                    ranking[feature] = current_rank
            
            current_rank += len(eliminated_features)
        
        return ranking.sort_values()
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix to selected features."""
        if not self.selected_features_:
            raise ValueError("No features selected. Call fit() first.")
        
        # Check feature availability
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        return X[self.selected_features_].copy()
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **fit_params
    ) -> pd.DataFrame:
        """Fit RFE and transform the data."""
        return self.fit(X, y, **fit_params).transform(X)
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features_.copy()
    
    def get_feature_ranking(self) -> pd.Series:
        """Get feature ranking (1 = best, higher = eliminated earlier)."""
        if self.feature_ranking_ is None:
            raise ValueError("Features have not been ranked yet. Call fit() first.")
        return self.feature_ranking_.copy()
    
    def plot_rfe_curve(self, save_path: Optional[str] = None) -> None:
        """Plot RFE elimination curve."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.elimination_history_:
                raise ValueError("No RFE history available")
            
            # Extract data for plotting
            n_features = [step['n_features'] for step in self.elimination_history_]
            cv_scores = [step['cv_score'] for step in self.elimination_history_]
            cv_stds = [step['cv_std'] for step in self.elimination_history_]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot CV score vs number of features
            ax.errorbar(n_features, cv_scores, yerr=cv_stds, 
                       marker='o', capsize=5, capthick=2)
            
            # Highlight optimal number of features
            if self.optimal_n_features_:
                ax.axvline(x=self.optimal_n_features_, color='red', linestyle='--', 
                          label=f'Optimal: {self.optimal_n_features_} features')
            
            ax.set_xlabel('Number of Features')
            ax.set_ylabel(f'CV Score ({self.config.cv_scoring.upper()})')
            ax.set_title('Recursive Feature Elimination Curve')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"RFE curve saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error creating RFE plot: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the RFE process."""
        if not self.selected_features_:
            return {"status": "RFE not performed"}
        
        # Calculate statistics from elimination history
        if self.elimination_history_:
            initial_features = self.elimination_history_[0]['n_features']
            best_score = max(step['cv_score'] for step in self.elimination_history_)
            elimination_steps = len(self.elimination_history_)
        else:
            initial_features = len(self.selected_features_)
            best_score = np.mean(self.cv_scores_['scores']) if self.cv_scores_ else None
            elimination_steps = 0
        
        summary = {
            "initial_features": initial_features,
            "selected_features": len(self.selected_features_),
            "reduction_ratio": len(self.selected_features_) / initial_features,
            "optimal_n_features": self.optimal_n_features_,
            "elimination_steps": elimination_steps,
            "final_cv_score": np.mean(self.cv_scores_['scores']) if self.cv_scores_ else None,
            "final_cv_std": np.std(self.cv_scores_['scores']) if self.cv_scores_ else None,
            "best_score": best_score,
            "selected_feature_list": self.selected_features_,
            "config": {
                "cv_folds": self.config.cv_folds,
                "cv_scoring": self.config.cv_scoring,
                "step": self.config.step,
                "min_features": self.config.min_features
            }
        }
        
        return summary