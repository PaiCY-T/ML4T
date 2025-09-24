"""
LightGBM-Based Feature Importance Ranking

Implementation of tree-based importance ranking using LightGBM feature importance
with stability analysis and cross-validation for robust feature selection.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import shap

logger = logging.getLogger(__name__)


@dataclass
class ImportanceRankerConfig:
    """Configuration for LightGBM importance ranking."""
    
    # LightGBM parameters for feature importance
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
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_shuffle: bool = False  # False for time-series data
    random_state: int = 42
    
    # Feature importance parameters
    importance_types: List[str] = field(default_factory=lambda: ['gain', 'split'])
    min_importance_threshold: float = 0.001
    stability_threshold: float = 0.7  # Stability across CV folds
    
    # SHAP analysis parameters
    use_shap: bool = True
    shap_sample_size: int = 1000  # Sample size for SHAP analysis
    
    # Taiwan market specific parameters
    target_horizon: int = 5  # Days for forward returns
    min_ic_threshold: float = 0.05
    
    # Performance parameters
    n_estimators: int = 100  # Faster training for importance ranking
    early_stopping_rounds: int = 10


class LightGBMImportanceRanker:
    """
    LightGBM-based feature importance ranking with cross-validation stability.
    
    Features:
    - Multiple importance types (gain, split, permutation)
    - Cross-validation stability analysis
    - SHAP value integration for interpretability
    - Taiwan market regime awareness
    - Memory-efficient processing for large feature sets
    """
    
    def __init__(self, config: Optional[ImportanceRankerConfig] = None):
        """Initialize the importance ranker.
        
        Args:
            config: Configuration for importance ranking
        """
        self.config = config or ImportanceRankerConfig()
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.cv_results_: Optional[Dict[str, Any]] = None
        self.shap_values_: Optional[Dict[str, Any]] = None
        self.stability_scores_: Optional[pd.Series] = None
        
        logger.info(f"LightGBM Importance Ranker initialized")
    
    def rank_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        market_regime_labels: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Rank features using LightGBM importance with cross-validation.
        
        Args:
            X: Feature matrix with MultiIndex (date, symbol)
            y: Target variable (forward returns)
            sample_weight: Optional sample weights
            market_regime_labels: Optional market regime labels for regime-aware analysis
            
        Returns:
            DataFrame with feature rankings and importance scores
        """
        logger.info(f"Starting feature importance ranking for {X.shape[1]} features")
        
        # Prepare data
        X_clean, y_clean = self._prepare_data(X, y)
        
        # Cross-validation feature importance
        cv_importance = self._cross_validate_importance(
            X_clean, y_clean, sample_weight
        )
        
        # Calculate stability scores
        stability_scores = self._calculate_stability_scores(cv_importance)
        
        # SHAP analysis (if enabled)
        shap_importance = None
        if self.config.use_shap:
            shap_importance = self._calculate_shap_importance(
                X_clean, y_clean, sample_weight
            )
        
        # Regime-specific analysis (if provided)
        regime_importance = None
        if market_regime_labels is not None:
            regime_importance = self._analyze_regime_importance(
                X_clean, y_clean, market_regime_labels
            )
        
        # Combine importance scores
        feature_rankings = self._combine_importance_scores(
            cv_importance, stability_scores, shap_importance, regime_importance
        )
        
        # Store results
        self.feature_importance_ = feature_rankings
        self.stability_scores_ = stability_scores
        
        logger.info(f"Feature ranking completed. Top feature: {feature_rankings.iloc[0]['feature']}")
        
        return feature_rankings
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and clean data for importance analysis."""
        # Remove NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        logger.info(f"Data prepared: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        
        return X_clean, y_clean
    
    def _cross_validate_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, List[pd.DataFrame]]:
        """Perform cross-validation to get stable importance scores."""
        
        # Use TimeSeriesSplit for temporal data
        cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        cv_importance = {imp_type: [] for imp_type in self.config.importance_types}
        cv_scores = []
        
        logger.info(f"Running {self.config.cv_folds}-fold cross-validation")
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            logger.debug(f"Processing fold {fold + 1}/{self.config.cv_folds}")
            
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
            
            # Calculate fold score (IC)
            y_pred = model.predict(X_val)
            fold_ic = pd.Series(y_val).corr(pd.Series(y_pred), method='spearman')
            cv_scores.append(fold_ic)
            
            # Extract importance scores
            for imp_type in self.config.importance_types:
                if imp_type == 'gain':
                    importance = model.feature_importances_
                elif imp_type == 'split':
                    importance = model.booster_.feature_importance(importance_type='split')
                else:
                    importance = model.feature_importances_  # Default to gain
                
                fold_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importance,
                    'fold': fold
                })
                
                cv_importance[imp_type].append(fold_importance)
        
        # Store CV results
        self.cv_results_ = {
            'cv_scores': cv_scores,
            'mean_ic': np.mean(cv_scores),
            'std_ic': np.std(cv_scores),
            'importance_by_fold': cv_importance
        }
        
        logger.info(f"CV completed. Mean IC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return cv_importance
    
    def _calculate_stability_scores(
        self,
        cv_importance: Dict[str, List[pd.DataFrame]]
    ) -> pd.Series:
        """Calculate stability scores across CV folds."""
        
        # Focus on gain importance for stability calculation
        gain_importance = cv_importance.get('gain', cv_importance[list(cv_importance.keys())[0]])
        
        # Create importance matrix (features x folds)
        importance_matrix = []
        feature_names = gain_importance[0]['feature'].values
        
        for fold_data in gain_importance:
            importance_matrix.append(fold_data['importance'].values)
        
        importance_matrix = np.array(importance_matrix).T  # Features x Folds
        
        # Calculate stability as coefficient of variation (lower is more stable)
        mean_importance = np.mean(importance_matrix, axis=1)
        std_importance = np.std(importance_matrix, axis=1)
        
        # Avoid division by zero
        cv_score = np.where(mean_importance > 0, std_importance / mean_importance, np.inf)
        
        # Convert to stability score (higher is more stable)
        stability_scores = np.exp(-cv_score)
        
        stability_df = pd.Series(
            stability_scores,
            index=feature_names,
            name='stability_score'
        )
        
        logger.info(f"Stability analysis completed. Mean stability: {stability_scores.mean():.3f}")
        
        return stability_df
    
    def _calculate_shap_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> Optional[pd.Series]:
        """Calculate SHAP-based feature importance."""
        try:
            # Sample data for SHAP analysis (memory efficiency)
            sample_size = min(self.config.shap_sample_size, len(X))
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            sample_weight_sample = sample_weight[sample_idx] if sample_weight is not None else None
            
            # Train model on sample
            model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                **self.config.lgb_params
            )
            
            model.fit(X_sample, y_sample, sample_weight=sample_weight_sample)
            
            # Calculate SHAP values
            explainer = shap.Explainer(model)
            shap_values = explainer(X_sample)
            
            # Calculate mean absolute SHAP importance
            shap_importance = pd.Series(
                np.mean(np.abs(shap_values.values), axis=0),
                index=X.columns,
                name='shap_importance'
            )
            
            # Store SHAP values
            self.shap_values_ = {
                'values': shap_values,
                'base_values': explainer.expected_value,
                'feature_names': X.columns.tolist()
            }
            
            logger.info("SHAP importance calculated successfully")
            
            return shap_importance
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            return None
    
    def _analyze_regime_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_labels: pd.Series
    ) -> Dict[str, pd.Series]:
        """Analyze feature importance across market regimes."""
        
        regime_importance = {}
        unique_regimes = regime_labels.unique()
        
        logger.info(f"Analyzing importance across {len(unique_regimes)} market regimes")
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) < 50:  # Skip regimes with insufficient data
                continue
            
            # Train model on regime data
            model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                **self.config.lgb_params
            )
            
            model.fit(X_regime, y_regime)
            
            # Store regime importance
            regime_importance[f'regime_{regime}'] = pd.Series(
                model.feature_importances_,
                index=X.columns,
                name=f'importance_regime_{regime}'
            )
        
        return regime_importance
    
    def _combine_importance_scores(
        self,
        cv_importance: Dict[str, List[pd.DataFrame]],
        stability_scores: pd.Series,
        shap_importance: Optional[pd.Series] = None,
        regime_importance: Optional[Dict[str, pd.Series]] = None
    ) -> pd.DataFrame:
        """Combine different importance measures into final ranking."""
        
        # Calculate mean importance across CV folds
        mean_importance = {}
        for imp_type, fold_data in cv_importance.items():
            all_folds = pd.concat(fold_data, ignore_index=True)
            mean_imp = all_folds.groupby('feature')['importance'].mean()
            mean_importance[f'importance_{imp_type}'] = mean_imp
        
        # Create combined DataFrame
        feature_rankings = pd.DataFrame(index=stability_scores.index)
        feature_rankings['feature'] = stability_scores.index
        
        # Add importance scores
        for imp_name, imp_series in mean_importance.items():
            feature_rankings[imp_name] = imp_series
        
        # Add stability scores
        feature_rankings['stability_score'] = stability_scores
        
        # Add SHAP importance (if available)
        if shap_importance is not None:
            feature_rankings['shap_importance'] = shap_importance
        
        # Add regime importance (if available)
        if regime_importance is not None:
            for regime_name, regime_imp in regime_importance.items():
                feature_rankings[f'importance_{regime_name}'] = regime_imp
        
        # Calculate composite score
        # Weight: 40% gain importance, 25% stability, 20% SHAP, 15% split importance
        weights = {
            'importance_gain': 0.4,
            'stability_score': 0.25,
            'shap_importance': 0.2 if shap_importance is not None else 0.0,
            'importance_split': 0.15 if 'importance_split' in feature_rankings.columns else 0.0
        }
        
        # Normalize importance scores to [0, 1]
        for col in feature_rankings.columns:
            if col.startswith('importance_') or col in ['shap_importance']:
                if feature_rankings[col].max() > 0:
                    feature_rankings[f'{col}_normalized'] = (
                        feature_rankings[col] / feature_rankings[col].max()
                    )
        
        # Calculate composite score
        composite_score = np.zeros(len(feature_rankings))
        total_weight = 0.0
        
        for score_type, weight in weights.items():
            if weight > 0:
                normalized_col = f'{score_type}_normalized'
                if normalized_col in feature_rankings.columns:
                    composite_score += weight * feature_rankings[normalized_col].fillna(0)
                    total_weight += weight
        
        # Normalize composite score
        if total_weight > 0:
            composite_score = composite_score / total_weight
        
        feature_rankings['composite_score'] = composite_score
        
        # Sort by composite score
        feature_rankings = feature_rankings.sort_values('composite_score', ascending=False)
        feature_rankings = feature_rankings.reset_index(drop=True)
        
        # Add ranking
        feature_rankings['rank'] = range(1, len(feature_rankings) + 1)
        
        # Filter by minimum importance threshold
        significant_features = feature_rankings[
            feature_rankings['importance_gain'] >= self.config.min_importance_threshold
        ]
        
        logger.info(f"Combined importance ranking completed. {len(significant_features)} significant features identified")
        
        return significant_features
    
    def get_top_features(
        self,
        n_features: int,
        min_stability: Optional[float] = None
    ) -> List[str]:
        """Get top N features based on ranking."""
        if self.feature_importance_ is None:
            raise ValueError("Features have not been ranked yet. Call rank_features() first.")
        
        # Apply stability filter if specified
        if min_stability is not None:
            filtered_features = self.feature_importance_[
                self.feature_importance_['stability_score'] >= min_stability
            ]
        else:
            filtered_features = self.feature_importance_
        
        # Get top N features
        top_features = filtered_features.head(n_features)['feature'].tolist()
        
        logger.info(f"Retrieved top {len(top_features)} features with stability >= {min_stability}")
        
        return top_features
    
    def plot_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """Plot feature importance rankings."""
        try:
            import matplotlib.pyplot as plt
            
            if self.feature_importance_ is None:
                raise ValueError("Features have not been ranked yet")
            
            # Get top N features
            top_features = self.feature_importance_.head(top_n)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Gain importance
            ax1.barh(range(len(top_features)), top_features['importance_gain'])
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features['feature'])
            ax1.set_xlabel('Gain Importance')
            ax1.set_title('LightGBM Gain Importance')
            ax1.invert_yaxis()
            
            # Stability scores
            ax2.barh(range(len(top_features)), top_features['stability_score'])
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features['feature'])
            ax2.set_xlabel('Stability Score')
            ax2.set_title('Feature Stability Across CV Folds')
            ax2.invert_yaxis()
            
            # Composite score
            ax3.barh(range(len(top_features)), top_features['composite_score'])
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_xlabel('Composite Score')
            ax3.set_title('Final Composite Ranking')
            ax3.invert_yaxis()
            
            # SHAP importance (if available)
            if 'shap_importance' in top_features.columns:
                ax4.barh(range(len(top_features)), top_features['shap_importance'])
                ax4.set_xlabel('SHAP Importance')
                ax4.set_title('SHAP Feature Importance')
            else:
                # Split importance as fallback
                ax4.barh(range(len(top_features)), top_features.get('importance_split', 0))
                ax4.set_xlabel('Split Importance')
                ax4.set_title('Split Importance')
            
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features['feature'])
            ax4.invert_yaxis()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Importance plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error creating importance plot: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the importance analysis."""
        if self.feature_importance_ is None:
            return {"status": "No ranking performed"}
        
        summary = {
            "total_features": len(self.feature_importance_),
            "significant_features": len(self.feature_importance_[
                self.feature_importance_['importance_gain'] >= self.config.min_importance_threshold
            ]),
            "mean_stability": self.stability_scores_.mean() if self.stability_scores_ is not None else None,
            "top_10_features": self.feature_importance_.head(10)['feature'].tolist(),
            "cv_performance": self.cv_results_['mean_ic'] if self.cv_results_ else None,
            "config": {
                "cv_folds": self.config.cv_folds,
                "min_importance_threshold": self.config.min_importance_threshold,
                "stability_threshold": self.config.stability_threshold,
                "use_shap": self.config.use_shap
            }
        }
        
        return summary