"""
Forward/Backward Feature Selection with Performance Validation

Implementation of sequential forward and backward selection with comprehensive
performance validation for financial time series data.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from itertools import combinations

logger = logging.getLogger(__name__)


@dataclass
class ForwardBackwardConfig:
    """Configuration for Forward/Backward Selection."""
    
    # Selection strategy
    strategy: str = 'forward'  # 'forward', 'backward', 'bidirectional'
    max_features: Optional[int] = None  # Maximum features to select
    min_features: int = 5  # Minimum features to maintain
    
    # Performance criteria
    scoring: str = 'ic'  # 'ic', 'ic_ir', 'sharpe', 'mse'
    tolerance: float = 0.001  # Improvement threshold
    patience: int = 3  # Steps without improvement before stopping
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_test_size: float = 0.2  # For validation split
    
    # LightGBM parameters
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
    
    # Performance validation parameters
    ic_threshold: float = 0.05  # Minimum IC for feature acceptance
    stability_threshold: float = 0.6  # Cross-fold stability requirement
    
    # Efficiency parameters
    max_candidates_per_step: int = 20  # Limit candidate evaluation per step
    parallel_evaluation: bool = True
    early_stopping_features: bool = True  # Stop if no improvement


class ForwardBackwardSelector:
    """
    Sequential Forward and Backward Feature Selection with Performance Validation.
    
    Supports three strategies:
    1. Forward: Start empty, add features sequentially
    2. Backward: Start full, remove features sequentially  
    3. Bidirectional: Combine forward and backward approaches
    """
    
    def __init__(self, config: Optional[ForwardBackwardConfig] = None):
        """Initialize the selector.
        
        Args:
            config: Configuration for selection process
        """
        self.config = config or ForwardBackwardConfig()
        self.selected_features_: List[str] = []
        self.selection_history_: List[Dict[str, Any]] = []
        self.feature_scores_: Optional[pd.DataFrame] = None
        self.cv_results_: Optional[Dict[str, Any]] = None
        self.best_score_: float = -np.inf
        
        logger.info(f"Forward/Backward Selector initialized with strategy: {self.config.strategy}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        initial_features: Optional[List[str]] = None
    ) -> 'ForwardBackwardSelector':
        """
        Fit the selector to find optimal feature subset.
        
        Args:
            X: Feature matrix with MultiIndex (date, symbol)
            y: Target variable (forward returns)
            sample_weight: Optional sample weights
            feature_groups: Optional feature groupings for priority
            initial_features: Optional initial feature set (for backward/bidirectional)
            
        Returns:
            self
        """
        logger.info(f"Starting {self.config.strategy} selection with {X.shape[1]} features")
        
        # Prepare data
        X_clean, y_clean = self._prepare_data(X, y)
        
        # Execute selection strategy
        if self.config.strategy == 'forward':
            self._forward_selection(X_clean, y_clean, sample_weight, feature_groups)
        elif self.config.strategy == 'backward':
            initial_set = initial_features or list(X_clean.columns)
            self._backward_elimination(X_clean, y_clean, sample_weight, initial_set)
        elif self.config.strategy == 'bidirectional':
            self._bidirectional_selection(X_clean, y_clean, sample_weight, feature_groups)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        # Final validation
        self._final_validation(X_clean, y_clean, sample_weight)
        
        logger.info(f"Selection completed. Selected {len(self.selected_features_)} features")
        logger.info(f"Best score: {self.best_score_:.6f}")
        
        return self
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and clean data for selection."""
        # Remove NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        logger.info(f"Data prepared: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        
        return X_clean, y_clean
    
    def _forward_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        feature_groups: Optional[Dict[str, List[str]]]
    ) -> None:
        """Execute forward feature selection."""
        
        selected_features = []
        available_features = list(X.columns)
        best_score = -np.inf
        patience_counter = 0
        
        step = 0
        while (len(available_features) > 0 and 
               patience_counter < self.config.patience and
               (self.config.max_features is None or len(selected_features) < self.config.max_features)):
            
            step += 1
            logger.info(f"Forward step {step}: {len(selected_features)} selected, {len(available_features)} available")
            
            # Evaluate candidate features
            best_candidate, best_candidate_score = self._evaluate_forward_candidates(
                X, y, selected_features, available_features, sample_weight
            )
            
            # Check for improvement
            if best_candidate_score > best_score + self.config.tolerance:
                # Accept the best candidate
                selected_features.append(best_candidate)
                available_features.remove(best_candidate)
                best_score = best_candidate_score
                patience_counter = 0
                
                logger.info(f"Added feature: {best_candidate}, score: {best_candidate_score:.6f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement, patience: {patience_counter}/{self.config.patience}")
            
            # Record step
            self.selection_history_.append({
                'step': step,
                'action': 'add' if best_candidate_score > best_score else 'no_change',
                'feature': best_candidate if best_candidate_score > best_score else None,
                'selected_features': selected_features.copy(),
                'score': best_candidate_score,
                'best_score': best_score
            })
        
        self.selected_features_ = selected_features
        self.best_score_ = best_score
    
    def _evaluate_forward_candidates(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        available_features: List[str],
        sample_weight: Optional[np.ndarray]
    ) -> Tuple[str, float]:
        """Evaluate candidate features for forward selection."""
        
        # Limit candidates for efficiency
        if len(available_features) > self.config.max_candidates_per_step:
            # Select top candidates by individual performance
            candidate_scores = []
            for feature in available_features:
                individual_score = self._evaluate_individual_feature(X[[feature]], y, sample_weight)
                candidate_scores.append((feature, individual_score))
            
            # Sort and select top candidates
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            candidates = [f for f, _ in candidate_scores[:self.config.max_candidates_per_step]]
        else:
            candidates = available_features
        
        best_candidate = None
        best_score = -np.inf
        
        # Evaluate each candidate
        for candidate in candidates:
            test_features = selected_features + [candidate]
            
            # Cross-validate feature set
            cv_score = self._cross_validate_feature_set(X[test_features], y, sample_weight)
            
            if cv_score > best_score:
                best_score = cv_score
                best_candidate = candidate
        
        return best_candidate, best_score
    
    def _backward_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        initial_features: List[str]
    ) -> None:
        """Execute backward feature elimination."""
        
        selected_features = initial_features.copy()
        
        # Initial score
        current_score = self._cross_validate_feature_set(X[selected_features], y, sample_weight)
        best_score = current_score
        patience_counter = 0
        
        step = 0
        while (len(selected_features) > self.config.min_features and 
               patience_counter < self.config.patience):
            
            step += 1
            logger.info(f"Backward step {step}: {len(selected_features)} features remaining")
            
            # Evaluate feature removals
            worst_feature, best_removal_score = self._evaluate_backward_candidates(
                X, y, selected_features, sample_weight
            )
            
            # Check for improvement
            if best_removal_score > current_score + self.config.tolerance:
                # Remove the worst feature
                selected_features.remove(worst_feature)
                current_score = best_removal_score
                if current_score > best_score:
                    best_score = current_score
                patience_counter = 0
                
                logger.info(f"Removed feature: {worst_feature}, score: {best_removal_score:.6f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement, patience: {patience_counter}/{self.config.patience}")
            
            # Record step
            self.selection_history_.append({
                'step': step,
                'action': 'remove' if best_removal_score > current_score else 'no_change',
                'feature': worst_feature if best_removal_score > current_score else None,
                'selected_features': selected_features.copy(),
                'score': best_removal_score,
                'best_score': best_score
            })
        
        self.selected_features_ = selected_features
        self.best_score_ = best_score
    
    def _evaluate_backward_candidates(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        sample_weight: Optional[np.ndarray]
    ) -> Tuple[str, float]:
        """Evaluate candidate features for removal in backward elimination."""
        
        best_removal = None
        best_score = -np.inf
        
        # Try removing each feature
        for feature_to_remove in selected_features:
            test_features = [f for f in selected_features if f != feature_to_remove]
            
            if len(test_features) >= self.config.min_features:
                # Cross-validate reduced feature set
                cv_score = self._cross_validate_feature_set(X[test_features], y, sample_weight)
                
                if cv_score > best_score:
                    best_score = cv_score
                    best_removal = feature_to_remove
        
        return best_removal, best_score
    
    def _bidirectional_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        feature_groups: Optional[Dict[str, List[str]]]
    ) -> None:
        """Execute bidirectional feature selection."""
        
        # Start with forward selection to get initial set
        logger.info("Phase 1: Forward selection")
        self._forward_selection(X, y, sample_weight, feature_groups)
        forward_features = self.selected_features_.copy()
        forward_score = self.best_score_
        
        # Reset for backward phase
        logger.info("Phase 2: Backward elimination")
        self.selection_history_ = []  # Reset history for second phase
        
        # Use forward result as starting point for backward
        self._backward_elimination(X, y, sample_weight, forward_features)
        backward_features = self.selected_features_.copy()
        backward_score = self.best_score_
        
        # Choose better result
        if forward_score > backward_score:
            self.selected_features_ = forward_features
            self.best_score_ = forward_score
            logger.info("Selected forward result")
        else:
            logger.info("Selected backward result")
        
        # Optional: Additional refinement phase
        if len(self.selected_features_) > self.config.min_features * 2:
            logger.info("Phase 3: Refinement")
            self._refinement_phase(X, y, sample_weight)
    
    def _refinement_phase(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> None:
        """Additional refinement phase for bidirectional selection."""
        
        current_features = self.selected_features_.copy()
        available_features = [f for f in X.columns if f not in current_features]
        
        improved = True
        refinement_steps = 0
        max_refinement_steps = 5
        
        while improved and refinement_steps < max_refinement_steps:
            refinement_steps += 1
            improved = False
            
            # Try adding one feature and removing one feature
            for add_candidate in available_features[:self.config.max_candidates_per_step]:
                for remove_candidate in current_features:
                    test_features = [f for f in current_features if f != remove_candidate] + [add_candidate]
                    
                    test_score = self._cross_validate_feature_set(X[test_features], y, sample_weight)
                    
                    if test_score > self.best_score_ + self.config.tolerance:
                        # Accept swap
                        current_features = test_features
                        available_features = [f for f in X.columns if f not in current_features]
                        self.best_score_ = test_score
                        improved = True
                        
                        logger.info(f"Refinement: swapped {remove_candidate} -> {add_candidate}")
                        break
                
                if improved:
                    break
        
        self.selected_features_ = current_features
    
    def _cross_validate_feature_set(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> float:
        """Cross-validate a feature set and return score."""
        
        cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X):
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
            
            try:
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
                fold_score = self._calculate_score(y_val, y_pred)
                cv_scores.append(fold_score)
                
            except Exception as e:
                logger.warning(f"CV fold failed: {e}")
                cv_scores.append(-np.inf)
        
        # Return mean score with penalty for instability
        mean_score = np.mean(cv_scores)
        score_stability = 1 - (np.std(cv_scores) / (abs(mean_score) + 1e-8))
        
        # Apply stability penalty
        final_score = mean_score * max(score_stability, self.config.stability_threshold)
        
        return final_score
    
    def _evaluate_individual_feature(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> float:
        """Evaluate individual feature performance."""
        try:
            return self._cross_validate_feature_set(X, y, sample_weight)
        except:
            return -np.inf
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate evaluation score based on scoring method."""
        
        if self.config.scoring == 'ic':
            # Information Coefficient (Spearman correlation)
            return pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        elif self.config.scoring == 'ic_ir':
            # Information Coefficient / Information Ratio
            ic = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
            # Calculate rolling IC for IR (simplified)
            rolling_ic = []
            window_size = min(252, len(y_true) // 5)  # ~1 year or 1/5 of data
            for i in range(window_size, len(y_true)):
                window_ic = pd.Series(y_true[i-window_size:i]).corr(
                    pd.Series(y_pred[i-window_size:i]), method='spearman')
                rolling_ic.append(window_ic)
            
            if len(rolling_ic) > 0:
                ir = ic / (np.std(rolling_ic) + 1e-8)
                return ir
            else:
                return ic
        elif self.config.scoring == 'sharpe':
            # Simplified Sharpe-like metric
            returns = y_pred * y_true  # Simplified returns calculation
            if np.std(returns) > 0:
                return np.mean(returns) / np.std(returns)
            else:
                return 0.0
        elif self.config.scoring == 'mse':
            return -np.mean((y_true - y_pred) ** 2)  # Negative for maximization
        else:
            raise ValueError(f"Unknown scoring method: {self.config.scoring}")
    
    def _final_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> None:
        """Perform final validation of selected features."""
        
        if not self.selected_features_:
            logger.warning("No features selected")
            return
        
        logger.info("Performing final validation")
        
        # Comprehensive cross-validation
        cv_results = self._comprehensive_cv_validation(X[self.selected_features_], y, sample_weight)
        
        # Feature importance analysis
        importance_analysis = self._analyze_feature_importance(X[self.selected_features_], y, sample_weight)
        
        # Store results
        self.cv_results_ = cv_results
        self.feature_scores_ = importance_analysis
        
        logger.info(f"Final validation - Mean IC: {cv_results['mean_score']:.6f} ± {cv_results['std_score']:.6f}")
    
    def _comprehensive_cv_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Comprehensive cross-validation with multiple metrics."""
        
        cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        cv_results = {
            'scores': [],
            'ic_scores': [],
            'mse_scores': [],
            'r2_scores': [],
            'feature_importances': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Sample weights
            train_weight = sample_weight[train_idx] if sample_weight is not None else None
            
            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                **self.config.lgb_params
            )
            
            model.fit(X_train, y_train, sample_weight=train_weight)
            
            # Predictions
            y_pred = model.predict(X_val)
            
            # Calculate multiple metrics
            ic_score = pd.Series(y_val).corr(pd.Series(y_pred), method='spearman')
            mse_score = np.mean((y_val - y_pred) ** 2)
            
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            
            # Store scores
            cv_results['scores'].append(self._calculate_score(y_val, y_pred))
            cv_results['ic_scores'].append(ic_score)
            cv_results['mse_scores'].append(mse_score)
            cv_results['r2_scores'].append(r2_score)
            
            # Feature importance
            fold_importance = pd.Series(model.feature_importances_, index=X.columns)
            cv_results['feature_importances'].append(fold_importance)
        
        # Aggregate results
        cv_results['mean_score'] = np.mean(cv_results['scores'])
        cv_results['std_score'] = np.std(cv_results['scores'])
        cv_results['mean_ic'] = np.mean(cv_results['ic_scores'])
        cv_results['mean_mse'] = np.mean(cv_results['mse_scores'])
        cv_results['mean_r2'] = np.mean(cv_results['r2_scores'])
        
        return cv_results
    
    def _analyze_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> pd.DataFrame:
        """Analyze final feature importance scores."""
        
        # Train final model
        model = lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators * 2,  # More estimators for final model
            **self.config.lgb_params
        )
        
        model.fit(X, y, sample_weight=sample_weight)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_,
            'selection_order': [self.selected_features_.index(f) + 1 for f in X.columns]
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix to selected features."""
        if not self.selected_features_:
            raise ValueError("No features selected. Call fit() first.")
        
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
        """Fit selector and transform the data."""
        return self.fit(X, y, **fit_params).transform(X)
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features_.copy()
    
    def get_feature_scores(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.feature_scores_ is None:
            raise ValueError("Features have not been scored yet. Call fit() first.")
        return self.feature_scores_.copy()
    
    def plot_selection_history(self, save_path: Optional[str] = None) -> None:
        """Plot selection history."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.selection_history_:
                raise ValueError("No selection history available")
            
            # Extract data
            steps = [step['step'] for step in self.selection_history_]
            scores = [step['score'] for step in self.selection_history_]
            n_features = [len(step['selected_features']) for step in self.selection_history_]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot score history
            ax1.plot(steps, scores, marker='o', linewidth=2, markersize=6)
            ax1.set_xlabel('Selection Step')
            ax1.set_ylabel(f'CV Score ({self.config.scoring.upper()})')
            ax1.set_title(f'{self.config.strategy.title()} Selection Score History')
            ax1.grid(True, alpha=0.3)
            
            # Plot number of features
            ax2.plot(steps, n_features, marker='s', linewidth=2, markersize=6, color='orange')
            ax2.set_xlabel('Selection Step')
            ax2.set_ylabel('Number of Selected Features')
            ax2.set_title('Feature Count History')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Selection history plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error creating selection plot: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the selection process."""
        if not self.selected_features_:
            return {"status": "Selection not performed"}
        
        summary = {
            "strategy": self.config.strategy,
            "selected_features": len(self.selected_features_),
            "selection_steps": len(self.selection_history_),
            "final_score": self.best_score_,
            "feature_list": self.selected_features_,
            "cv_performance": {
                "mean_score": self.cv_results_['mean_score'] if self.cv_results_ else None,
                "std_score": self.cv_results_['std_score'] if self.cv_results_ else None,
                "mean_ic": self.cv_results_['mean_ic'] if self.cv_results_ else None,
            } if self.cv_results_ else None,
            "config": {
                "scoring": self.config.scoring,
                "cv_folds": self.config.cv_folds,
                "tolerance": self.config.tolerance,
                "patience": self.config.patience,
                "max_features": self.config.max_features,
                "min_features": self.config.min_features
            }
        }
        
        return summary