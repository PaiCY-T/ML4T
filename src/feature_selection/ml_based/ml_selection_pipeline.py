"""
ML Feature Selection Pipeline

Integrated pipeline combining all ML-based feature selection methods
with LightGBM pipeline integration and Taiwan market optimizations.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# Import our ML-based selection components
from .importance_ranking import LightGBMImportanceRanker, ImportanceRankerConfig
from .recursive_elimination import RecursiveFeatureEliminator, RFEConfig
from .forward_backward_selection import ForwardBackwardSelector, ForwardBackwardConfig
from .stability_analysis import StabilityAnalyzer, StabilityConfig

# Import LightGBM model for integration
try:
    from ...models.lightgbm_alpha import LightGBMAlphaModel, ModelConfig
except ImportError:
    logger.warning("Could not import LightGBMAlphaModel. Some integration features may be limited.")
    LightGBMAlphaModel = None
    ModelConfig = None

logger = logging.getLogger(__name__)


@dataclass
class MLSelectionConfig:
    """Configuration for ML-based feature selection pipeline."""
    
    # Pipeline strategy
    selection_strategy: str = 'comprehensive'  # 'comprehensive', 'fast', 'stability_focused'
    target_features: int = 100  # Target number of features to select
    min_features: int = 20  # Minimum features to maintain
    max_features: int = 200  # Maximum features for intermediate steps
    
    # Method weighting for ensemble selection
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        'importance_ranking': 0.3,
        'rfe': 0.25,
        'forward_backward': 0.25,
        'stability_analysis': 0.2
    })
    
    # Performance criteria
    ic_threshold: float = 0.05  # Minimum IC for feature acceptance
    stability_threshold: float = 0.7  # Minimum stability score
    validation_folds: int = 5
    
    # Taiwan market specific parameters
    taiwan_market_weight: float = 0.15  # Additional weight for Taiwan compliance
    sector_balance: bool = True  # Maintain sector balance in selection
    regime_awareness: bool = True  # Consider market regimes
    
    # Component configurations
    importance_config: ImportanceRankerConfig = field(default_factory=ImportanceRankerConfig)
    rfe_config: RFEConfig = field(default_factory=RFEConfig)
    fb_config: ForwardBackwardConfig = field(default_factory=ForwardBackwardConfig)
    stability_config: StabilityConfig = field(default_factory=StabilityConfig)
    
    # Integration parameters
    integrate_with_lgb: bool = True  # Integrate with LightGBM pipeline
    save_intermediate_results: bool = True
    output_dir: Optional[str] = None  # Directory for saving results
    
    # Performance optimization
    parallel_execution: bool = True
    memory_efficient: bool = True
    early_stopping: bool = True


class MLFeatureSelectionPipeline:
    """
    Comprehensive ML-based Feature Selection Pipeline.
    
    Integrates multiple selection methods:
    1. LightGBM importance ranking
    2. Recursive feature elimination
    3. Forward/backward selection
    4. Stability analysis
    
    Provides ensemble selection with Taiwan market optimizations.
    """
    
    def __init__(self, config: Optional[MLSelectionConfig] = None):
        """Initialize ML feature selection pipeline.
        
        Args:
            config: Configuration for the pipeline
        """
        self.config = config or MLSelectionConfig()
        
        # Initialize components
        self.importance_ranker = LightGBMImportanceRanker(self.config.importance_config)
        self.rfe_selector = RecursiveFeatureEliminator(self.config.rfe_config)
        self.fb_selector = ForwardBackwardSelector(self.config.fb_config)
        self.stability_analyzer = StabilityAnalyzer(self.config.stability_config)
        
        # Results storage
        self.selected_features_: List[str] = []
        self.feature_scores_: Optional[pd.DataFrame] = None
        self.method_results_: Dict[str, Any] = {}
        self.final_validation_: Optional[Dict[str, Any]] = None
        self.lgb_integration_: Optional[Dict[str, Any]] = None
        
        # Create output directory if specified
        if self.config.output_dir:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("ML Feature Selection Pipeline initialized")
    
    def fit_select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        dates: Optional[pd.Series] = None
    ) -> List[str]:
        """
        Comprehensive feature selection using multiple ML methods.
        
        Args:
            X: Feature matrix with MultiIndex (date, symbol)
            y: Target variable (forward returns)
            sample_weight: Optional sample weights
            market_data: Market data for regime analysis
            feature_groups: Feature groupings for balanced selection
            dates: Date series for time analysis
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Starting comprehensive ML feature selection on {X.shape[1]} features")
        
        # Prepare data
        X_clean, y_clean = self._prepare_data(X, y)
        
        # Execute selection strategy
        if self.config.selection_strategy == 'comprehensive':
            selected_features = self._comprehensive_selection(
                X_clean, y_clean, sample_weight, market_data, feature_groups, dates
            )
        elif self.config.selection_strategy == 'fast':
            selected_features = self._fast_selection(
                X_clean, y_clean, sample_weight, feature_groups
            )
        elif self.config.selection_strategy == 'stability_focused':
            selected_features = self._stability_focused_selection(
                X_clean, y_clean, sample_weight, market_data, dates
            )
        else:
            raise ValueError(f"Unknown selection strategy: {self.config.selection_strategy}")
        
        # Final validation and LightGBM integration
        self._final_validation(X_clean[selected_features], y_clean, sample_weight)
        
        if self.config.integrate_with_lgb and LightGBMAlphaModel is not None:
            self._integrate_with_lightgbm(X_clean[selected_features], y_clean)
        
        # Save results
        if self.config.save_intermediate_results and self.config.output_dir:
            self._save_results()
        
        self.selected_features_ = selected_features
        
        logger.info(f"ML feature selection completed. Selected {len(selected_features)} features")
        logger.info(f"Top 5 features: {selected_features[:5]}")
        
        return selected_features
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and clean data for feature selection."""
        
        # Remove NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        # Handle infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Memory optimization
        if self.config.memory_efficient:
            # Convert to float32 for memory efficiency
            float_cols = X_clean.select_dtypes(include=[np.float64]).columns
            X_clean[float_cols] = X_clean[float_cols].astype(np.float32)
        
        logger.info(f"Data prepared: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        
        return X_clean, y_clean
    
    def _comprehensive_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        market_data: Optional[pd.DataFrame],
        feature_groups: Optional[Dict[str, List[str]]],
        dates: Optional[pd.Series]
    ) -> List[str]:
        """Comprehensive selection using all methods with ensemble combination."""
        
        logger.info("Executing comprehensive selection strategy")
        
        # Step 1: Importance-based ranking
        logger.info("Step 1: LightGBM importance ranking")
        importance_results = self.importance_ranker.rank_features(
            X, y, sample_weight, market_data.iloc[:, 0] if market_data is not None else None
        )
        
        # Get top features from importance ranking
        top_importance_features = self.importance_ranker.get_top_features(
            n_features=min(self.config.max_features, len(X.columns)),
            min_stability=0.5
        )
        
        self.method_results_['importance'] = {
            'rankings': importance_results,
            'selected_features': top_importance_features,
            'method_score': importance_results['composite_score'].mean()
        }
        
        # Step 2: Recursive Feature Elimination (on top importance features)
        logger.info("Step 2: Recursive Feature Elimination")
        X_importance_filtered = X[top_importance_features]
        
        # Configure RFE for target number of features
        self.config.rfe_config.n_features_to_select = min(
            self.config.target_features, 
            len(top_importance_features) // 2
        )
        
        self.rfe_selector.fit(X_importance_filtered, y, sample_weight, feature_groups)
        rfe_features = self.rfe_selector.get_selected_features()
        
        self.method_results_['rfe'] = {
            'selected_features': rfe_features,
            'elimination_history': self.rfe_selector.elimination_history_,
            'method_score': self.rfe_selector.cv_scores_['mean_score'] if self.rfe_selector.cv_scores_ else 0.0
        }
        
        # Step 3: Forward/Backward Selection (on RFE results)
        logger.info("Step 3: Forward/Backward Selection")
        
        # Use bidirectional strategy for comprehensive approach
        self.config.fb_config.strategy = 'bidirectional'
        self.config.fb_config.max_features = min(
            self.config.target_features,
            len(rfe_features) + 20  # Allow some expansion
        )
        
        X_rfe_filtered = X[rfe_features]
        self.fb_selector.fit(X_rfe_filtered, y, sample_weight, feature_groups, rfe_features)
        fb_features = self.fb_selector.get_selected_features()
        
        self.method_results_['forward_backward'] = {
            'selected_features': fb_features,
            'selection_history': self.fb_selector.selection_history_,
            'method_score': self.fb_selector.best_score_
        }
        
        # Step 4: Stability Analysis (on all candidates)
        logger.info("Step 4: Stability Analysis")
        
        # Combine all candidate features from previous steps
        candidate_features = list(set(top_importance_features + rfe_features + fb_features))
        
        X_candidates = X[candidate_features]
        stability_results = self.stability_analyzer.analyze_stability(
            X_candidates, y, dates, market_data, candidate_features
        )
        
        stable_features = self.stability_analyzer.get_stable_features(
            n_features=self.config.target_features,
            stability_threshold=self.config.stability_threshold
        )
        
        self.method_results_['stability'] = {
            'stability_scores': stability_results,
            'selected_features': stable_features,
            'method_score': stability_results['composite_stability'].mean()
        }
        
        # Step 5: Ensemble Feature Selection
        logger.info("Step 5: Ensemble combination")
        final_features = self._ensemble_selection(candidate_features)
        
        return final_features
    
    def _fast_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        feature_groups: Optional[Dict[str, List[str]]]
    ) -> List[str]:
        """Fast selection using importance ranking + stability check."""
        
        logger.info("Executing fast selection strategy")
        
        # Use importance ranking as primary method
        importance_results = self.importance_ranker.rank_features(X, y, sample_weight)
        
        # Get top features
        top_features = self.importance_ranker.get_top_features(
            n_features=self.config.target_features * 2,  # Select more initially
            min_stability=self.config.stability_threshold
        )
        
        # Quick stability check on top features
        X_top = X[top_features]
        stability_results = self.stability_analyzer.analyze_stability(X_top, y)
        
        # Select final features based on combined score
        combined_scores = []
        for feature in top_features:
            importance_score = importance_results.loc[
                importance_results['feature'] == feature, 'composite_score'
            ].iloc[0] if len(importance_results[importance_results['feature'] == feature]) > 0 else 0
            
            stability_score = stability_results.loc[feature, 'composite_stability'] if feature in stability_results.index else 0
            
            combined_score = (0.7 * importance_score + 0.3 * stability_score)
            combined_scores.append((feature, combined_score))
        
        # Sort and select top features
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        final_features = [f for f, _ in combined_scores[:self.config.target_features]]
        
        return final_features
    
    def _stability_focused_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray],
        market_data: Optional[pd.DataFrame],
        dates: Optional[pd.Series]
    ) -> List[str]:
        """Stability-focused selection prioritizing feature stability."""
        
        logger.info("Executing stability-focused selection strategy")
        
        # Primary stability analysis
        stability_results = self.stability_analyzer.analyze_stability(
            X, y, dates, market_data, list(X.columns)
        )
        
        # Get most stable features
        stable_features = self.stability_analyzer.get_stable_features(
            n_features=self.config.target_features * 2,  # Select more initially
            stability_threshold=self.config.stability_threshold * 0.8  # Lower threshold
        )
        
        # Validation with importance ranking on stable features
        X_stable = X[stable_features]
        importance_results = self.importance_ranker.rank_features(X_stable, y, sample_weight)
        
        # Select final features based on combined stability and importance
        final_features = []
        for _, row in importance_results.head(self.config.target_features).iterrows():
            final_features.append(row['feature'])
        
        return final_features
    
    def _ensemble_selection(self, candidate_features: List[str]) -> List[str]:
        """Ensemble selection combining results from all methods."""
        
        logger.info("Performing ensemble feature selection")
        
        # Calculate ensemble scores for each candidate feature
        feature_ensemble_scores = {}
        
        for feature in candidate_features:
            ensemble_score = 0.0
            total_weight = 0.0
            
            # Importance ranking contribution
            if 'importance' in self.method_results_:
                importance_results = self.method_results_['importance']['rankings']
                if feature in importance_results['feature'].values:
                    feature_idx = importance_results[importance_results['feature'] == feature].index[0]
                    importance_score = importance_results.loc[feature_idx, 'composite_score']
                    ensemble_score += self.config.method_weights['importance_ranking'] * importance_score
                    total_weight += self.config.method_weights['importance_ranking']
            
            # RFE contribution
            if 'rfe' in self.method_results_:
                rfe_features = self.method_results_['rfe']['selected_features']
                if feature in rfe_features:
                    # Higher score for features selected by RFE
                    rfe_score = 1.0 - (rfe_features.index(feature) / len(rfe_features))
                    ensemble_score += self.config.method_weights['rfe'] * rfe_score
                    total_weight += self.config.method_weights['rfe']
            
            # Forward/Backward selection contribution
            if 'forward_backward' in self.method_results_:
                fb_features = self.method_results_['forward_backward']['selected_features']
                if feature in fb_features:
                    fb_score = 1.0 - (fb_features.index(feature) / len(fb_features))
                    ensemble_score += self.config.method_weights['forward_backward'] * fb_score
                    total_weight += self.config.method_weights['forward_backward']
            
            # Stability analysis contribution
            if 'stability' in self.method_results_:
                stability_results = self.method_results_['stability']['stability_scores']
                if feature in stability_results.index:
                    stability_score = stability_results.loc[feature, 'composite_stability']
                    ensemble_score += self.config.method_weights['stability_analysis'] * stability_score
                    total_weight += self.config.method_weights['stability_analysis']
            
            # Normalize ensemble score
            if total_weight > 0:
                feature_ensemble_scores[feature] = ensemble_score / total_weight
            else:
                feature_ensemble_scores[feature] = 0.0
        
        # Sort features by ensemble score
        sorted_features = sorted(
            feature_ensemble_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply Taiwan market considerations
        if self.config.sector_balance:
            final_features = self._apply_sector_balance(sorted_features)
        else:
            final_features = [f for f, _ in sorted_features[:self.config.target_features]]
        
        # Store ensemble scores
        self.feature_scores_ = pd.DataFrame([
            {'feature': feature, 'ensemble_score': score}
            for feature, score in sorted_features
        ])
        
        return final_features
    
    def _apply_sector_balance(self, sorted_features: List[Tuple[str, float]]) -> List[str]:
        """Apply sector balance constraints for Taiwan market."""
        
        # Simple heuristic for sector balance
        # In a real implementation, this would use feature metadata
        
        selected_features = []
        feature_categories = {
            'technical': ['rsi', 'macd', 'bb', 'ma', 'volume', 'momentum', 'volatility'],
            'fundamental': ['pe', 'pb', 'roe', 'eps', 'revenue', 'debt', 'margin'],
            'market': ['beta', 'alpha', 'correlation', 'sector', 'market_cap'],
            'macro': ['interest', 'inflation', 'gdp', 'currency', 'vix']
        }
        
        category_counts = {cat: 0 for cat in feature_categories.keys()}
        category_limits = {
            'technical': int(self.config.target_features * 0.4),
            'fundamental': int(self.config.target_features * 0.3),
            'market': int(self.config.target_features * 0.2),
            'macro': int(self.config.target_features * 0.1)
        }
        
        # First pass: select features respecting category limits
        for feature, score in sorted_features:
            if len(selected_features) >= self.config.target_features:
                break
            
            # Determine feature category
            feature_category = 'technical'  # Default category
            for cat, keywords in feature_categories.items():
                if any(keyword in feature.lower() for keyword in keywords):
                    feature_category = cat
                    break
            
            # Check if we can add this feature
            if category_counts[feature_category] < category_limits[feature_category]:
                selected_features.append(feature)
                category_counts[feature_category] += 1
        
        # Second pass: fill remaining slots with best available features
        while len(selected_features) < self.config.target_features:
            for feature, score in sorted_features:
                if feature not in selected_features:
                    selected_features.append(feature)
                    break
            else:
                break  # No more features available
        
        logger.info(f"Sector balance applied: {category_counts}")
        
        return selected_features
    
    def _final_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray]
    ) -> None:
        """Perform final validation of selected features."""
        
        logger.info("Performing final validation")
        
        # Cross-validation performance
        cv = TimeSeriesSplit(n_splits=self.config.validation_folds)
        cv_scores = []
        cv_ics = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_weight = sample_weight[train_idx] if sample_weight is not None else None
            
            # Train LightGBM model
            model = lgb.LGBMRegressor(**self.config.importance_config.lgb_params)
            model.fit(X_train, y_train, sample_weight=train_weight)
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            
            # Calculate IC
            ic = pd.Series(y_val).corr(pd.Series(y_pred), method='spearman')
            cv_ics.append(ic if not np.isnan(ic) else 0.0)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            cv_scores.append(-rmse)  # Negative for consistency
        
        # Store final validation results
        self.final_validation_ = {
            'mean_ic': np.mean(cv_ics),
            'std_ic': np.std(cv_ics),
            'mean_rmse': -np.mean(cv_scores),  # Convert back to positive
            'std_rmse': np.std(cv_scores),
            'cv_folds': self.config.validation_folds,
            'selected_feature_count': len(X.columns),
            'ic_threshold_met': np.mean(cv_ics) >= self.config.ic_threshold
        }
        
        logger.info(f"Final validation - IC: {np.mean(cv_ics):.4f} ± {np.std(cv_ics):.4f}")
        logger.info(f"Final validation - RMSE: {-np.mean(cv_scores):.6f}")
    
    def _integrate_with_lightgbm(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """Integrate with LightGBM pipeline for production readiness."""
        
        logger.info("Integrating with LightGBM pipeline")
        
        try:
            # Create LightGBM model with optimized configuration
            lgb_config = ModelConfig()
            lgb_model = LightGBMAlphaModel(lgb_config)
            
            # Prepare training data
            X_train, y_train = lgb_model.prepare_training_data(X, y)
            
            # Train model
            training_stats = lgb_model.train(X_train, y_train)
            
            # Get feature importance from trained model
            feature_importance = lgb_model.get_feature_importance()
            
            # Store integration results
            self.lgb_integration_ = {
                'training_stats': training_stats,
                'feature_importance': feature_importance.to_dict('records'),
                'model_config': lgb_config,
                'integration_successful': True
            }
            
            logger.info("LightGBM integration completed successfully")
            
        except Exception as e:
            logger.error(f"LightGBM integration failed: {e}")
            self.lgb_integration_ = {
                'integration_successful': False,
                'error': str(e)
            }
    
    def _save_results(self) -> None:
        """Save intermediate and final results."""
        
        output_dir = Path(self.config.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save selected features
        features_file = output_dir / f"selected_features_{timestamp}.json"
        with open(features_file, 'w') as f:
            json.dump({
                'selected_features': self.selected_features_,
                'config': {
                    'strategy': self.config.selection_strategy,
                    'target_features': self.config.target_features,
                    'ic_threshold': self.config.ic_threshold,
                    'stability_threshold': self.config.stability_threshold
                },
                'timestamp': timestamp
            }, f, indent=2)
        
        # Save feature scores
        if self.feature_scores_ is not None:
            scores_file = output_dir / f"feature_scores_{timestamp}.csv"
            self.feature_scores_.to_csv(scores_file, index=False)
        
        # Save method results
        results_file = output_dir / f"method_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.method_results_, f)
        
        # Save final validation
        if self.final_validation_ is not None:
            validation_file = output_dir / f"validation_results_{timestamp}.json"
            with open(validation_file, 'w') as f:
                json.dump(self.final_validation_, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix to selected features."""
        if not self.selected_features_:
            raise ValueError("No features selected. Call fit_select() first.")
        
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data: {missing_features}")
        
        return X[self.selected_features_].copy()
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self.selected_features_.copy()
    
    def get_feature_scores(self) -> pd.DataFrame:
        """Get feature ensemble scores."""
        if self.feature_scores_ is None:
            raise ValueError("Feature scores not available. Call fit_select() first.")
        return self.feature_scores_.copy()
    
    def get_method_results(self) -> Dict[str, Any]:
        """Get detailed results from each selection method."""
        return self.method_results_.copy()
    
    def get_validation_results(self) -> Dict[str, Any]:
        """Get final validation results."""
        if self.final_validation_ is None:
            raise ValueError("Validation not performed yet. Call fit_select() first.")
        return self.final_validation_.copy()
    
    def plot_selection_summary(self, save_path: Optional[str] = None) -> None:
        """Plot comprehensive selection summary."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.selected_features_:
                raise ValueError("No selection results available")
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Plot 1: Method comparison (feature counts)
            if self.method_results_:
                method_names = []
                feature_counts = []
                for method, results in self.method_results_.items():
                    method_names.append(method.replace('_', ' ').title())
                    feature_counts.append(len(results.get('selected_features', [])))
                
                axes[0, 0].bar(method_names, feature_counts, color='skyblue')
                axes[0, 0].set_title('Features Selected by Each Method')
                axes[0, 0].set_ylabel('Number of Features')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Feature ensemble scores
            if self.feature_scores_ is not None:
                top_features = self.feature_scores_.head(20)
                axes[0, 1].barh(range(len(top_features)), top_features['ensemble_score'])
                axes[0, 1].set_yticks(range(len(top_features)))
                axes[0, 1].set_yticklabels(top_features['feature'], fontsize=8)
                axes[0, 1].set_xlabel('Ensemble Score')
                axes[0, 1].set_title('Top 20 Features by Ensemble Score')
                axes[0, 1].invert_yaxis()
            
            # Plot 3: Validation performance
            if self.final_validation_:
                metrics = ['IC', 'RMSE']
                values = [self.final_validation_['mean_ic'], self.final_validation_['mean_rmse']]
                errors = [self.final_validation_['std_ic'], self.final_validation_['std_rmse']]
                
                axes[0, 2].bar(metrics, values, yerr=errors, capsize=5, color=['green', 'red'], alpha=0.7)
                axes[0, 2].set_title('Final Validation Performance')
                axes[0, 2].set_ylabel('Score')
            
            # Plot 4: Selection process flow
            axes[1, 0].text(0.5, 0.5, f'Selection Strategy: {self.config.selection_strategy.title()}\n'
                                      f'Target Features: {self.config.target_features}\n'
                                      f'Selected Features: {len(self.selected_features_)}\n'
                                      f'IC Threshold: {self.config.ic_threshold}\n'
                                      f'Stability Threshold: {self.config.stability_threshold}',
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            axes[1, 0].set_title('Selection Configuration')
            axes[1, 0].axis('off')
            
            # Plot 5: Method scores comparison
            if self.method_results_:
                method_names = []
                method_scores = []
                for method, results in self.method_results_.items():
                    method_names.append(method.replace('_', ' ').title())
                    method_scores.append(results.get('method_score', 0.0))
                
                axes[1, 1].bar(method_names, method_scores, color='orange', alpha=0.7)
                axes[1, 1].set_title('Method Performance Scores')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Plot 6: Feature reduction overview
            if hasattr(self, 'method_results_') and self.method_results_:
                # Estimate initial features (this would be passed in real implementation)
                initial_features = 500  # Placeholder
                
                reduction_stages = ['Initial', 'Importance', 'RFE', 'F/B Selection', 'Final']
                feature_counts_reduction = [
                    initial_features,
                    len(self.method_results_.get('importance', {}).get('selected_features', [])),
                    len(self.method_results_.get('rfe', {}).get('selected_features', [])),
                    len(self.method_results_.get('forward_backward', {}).get('selected_features', [])),
                    len(self.selected_features_)
                ]
                
                axes[1, 2].plot(reduction_stages, feature_counts_reduction, marker='o', linewidth=2)
                axes[1, 2].set_title('Feature Reduction Pipeline')
                axes[1, 2].set_ylabel('Number of Features')
                axes[1, 2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Selection summary plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Error creating selection summary plot: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        
        summary = {
            "pipeline_config": {
                "strategy": self.config.selection_strategy,
                "target_features": self.config.target_features,
                "ic_threshold": self.config.ic_threshold,
                "stability_threshold": self.config.stability_threshold
            },
            "selection_results": {
                "selected_features": len(self.selected_features_),
                "feature_list": self.selected_features_[:10],  # Top 10
                "method_weights": self.config.method_weights
            },
            "validation_performance": self.final_validation_ if self.final_validation_ else None,
            "method_contributions": {}
        }
        
        # Add method-specific contributions
        for method, results in self.method_results_.items():
            summary["method_contributions"][method] = {
                "features_selected": len(results.get('selected_features', [])),
                "method_score": results.get('method_score', 0.0)
            }
        
        # Add integration status
        if self.lgb_integration_:
            summary["lightgbm_integration"] = {
                "successful": self.lgb_integration_.get('integration_successful', False),
                "training_performance": self.lgb_integration_.get('training_stats', {}).get('best_score', None) if self.lgb_integration_.get('training_stats') else None
            }
        
        return summary