"""
Comprehensive Training Pipeline for LightGBM Alpha Model

This module integrates hyperparameter optimization, time-series cross-validation,
and performance tracking into a unified training pipeline for Taiwan market models.
"""

import logging
import pickle
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator

from .lightgbm_alpha import LightGBMAlphaModel, ModelConfig
from ..optimization.hyperopt import HyperparameterOptimizer, OptimizationConfig, create_taiwan_optimization_config
from ..validation.timeseries_cv import TimeSeriesCrossValidator, create_taiwan_cv_config, TimeSeriesCVConfig
from ..metrics.model_performance import ModelPerformanceTracker, calculate_model_metrics
from ..backtesting.validation.walk_forward import WalkForwardSplitter, WalkForwardConfig, create_default_config
from ..data.core.temporal import TemporalStore
from ..data.pipeline.pit_engine import PointInTimeEngine

logger = logging.getLogger(__name__)


@dataclass
class TrainingPipelineConfig:
    """Configuration for the complete training pipeline."""
    
    # Model configuration
    model_config: Optional[ModelConfig] = None
    
    # Hyperparameter optimization configuration
    optimization_config: Optional[OptimizationConfig] = None
    enable_hyperopt: bool = True
    hyperopt_trials: int = 50
    hyperopt_timeout_hours: int = 4
    
    # Cross-validation configuration  
    cv_config: Optional[TimeSeriesCVConfig] = None
    enable_cv: bool = True
    cv_folds: int = 5
    
    # Walk-forward validation configuration
    walkforward_config: Optional[WalkForwardConfig] = None
    enable_walkforward: bool = True
    
    # Performance tracking configuration
    enable_performance_tracking: bool = True
    track_feature_importance: bool = True
    
    # Taiwan market specific configuration
    target_horizons: List[int] = field(default_factory=lambda: [5, 10, 20])  # Days
    performance_targets: Dict[str, float] = field(default_factory=lambda: {
        'ic_threshold': 0.05,
        'sharpe_threshold': 2.0,
        'max_drawdown_threshold': 0.15,
        'hit_rate_threshold': 0.52
    })
    
    # Training configuration
    retrain_frequency_days: int = 30  # Monthly retraining
    validation_split: float = 0.2
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 50
    
    # Output configuration
    save_models: bool = True
    save_results: bool = True
    output_dir: str = "models/training_results"
    model_version: Optional[str] = None


@dataclass
class TrainingResult:
    """Results from complete training pipeline."""
    
    # Training metadata
    pipeline_id: str
    timestamp: datetime
    config: TrainingPipelineConfig
    training_duration_seconds: float
    
    # Model results
    best_model: Optional[LightGBMAlphaModel] = None
    model_path: Optional[str] = None
    
    # Optimization results
    optimization_result: Optional[Dict[str, Any]] = None
    best_hyperparameters: Optional[Dict[str, Any]] = None
    
    # Cross-validation results
    cv_results: Optional[Dict[str, Any]] = None
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    
    # Walk-forward validation results
    walkforward_results: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    performance_snapshot: Optional[Dict[str, Any]] = None
    performance_summary: Dict[str, float] = field(default_factory=dict)
    
    # Feature analysis
    feature_importance: Optional[pd.DataFrame] = None
    feature_stability: Optional[Dict[str, float]] = None
    
    # Success indicators
    meets_performance_targets: bool = False
    training_successful: bool = False
    validation_passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'pipeline_id': self.pipeline_id,
            'timestamp': self.timestamp.isoformat(),
            'training_duration_seconds': self.training_duration_seconds,
            'best_hyperparameters': self.best_hyperparameters,
            'cv_scores': self.cv_scores,
            'performance_summary': self.performance_summary,
            'meets_performance_targets': self.meets_performance_targets,
            'training_successful': self.training_successful,
            'validation_passed': self.validation_passed,
            'model_path': self.model_path
        }


class TaiwanMarketTrainingPipeline:
    """
    Comprehensive training pipeline for Taiwan market alpha models.
    
    This class orchestrates the complete training process including:
    - Data preparation and validation
    - Hyperparameter optimization
    - Time-series cross-validation
    - Walk-forward validation
    - Performance tracking and monitoring
    - Model serialization and deployment preparation
    """
    
    def __init__(
        self,
        config: Optional[TrainingPipelineConfig] = None,
        temporal_store: Optional[TemporalStore] = None,
        pit_engine: Optional[PointInTimeEngine] = None
    ):
        """
        Initialize the training pipeline.
        
        Args:
            config: Training pipeline configuration
            temporal_store: Temporal data store for walk-forward validation
            pit_engine: Point-in-time engine for bias prevention
        """
        self.config = config or TrainingPipelineConfig()
        self.temporal_store = temporal_store
        self.pit_engine = pit_engine
        
        # Initialize components
        self._initialize_configurations()
        self._initialize_components()
        
        # Create output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TaiwanMarketTrainingPipeline initialized")
    
    def _initialize_configurations(self) -> None:
        """Initialize default configurations if not provided."""
        
        # Model configuration
        if self.config.model_config is None:
            self.config.model_config = ModelConfig()
            self.config.model_config.target_horizons = self.config.target_horizons
        
        # Optimization configuration
        if self.config.optimization_config is None and self.config.enable_hyperopt:
            self.config.optimization_config = create_taiwan_optimization_config(
                n_trials=self.config.hyperopt_trials,
                timeout_seconds=self.config.hyperopt_timeout_hours * 3600
            )
        
        # Cross-validation configuration
        if self.config.cv_config is None and self.config.enable_cv:
            self.config.cv_config = create_taiwan_cv_config(
                n_splits=self.config.cv_folds
            )
        
        # Walk-forward configuration
        if self.config.walkforward_config is None and self.config.enable_walkforward:
            self.config.walkforward_config = create_default_config(
                train_weeks=104,  # 2 years
                test_weeks=26,    # 6 months
                rebalance_weeks=4  # Monthly
            )
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components."""
        
        # Hyperparameter optimizer
        if self.config.enable_hyperopt:
            self.hyperopt_optimizer = HyperparameterOptimizer(
                self.config.optimization_config
            )
        else:
            self.hyperopt_optimizer = None
        
        # Cross-validator
        if self.config.enable_cv:
            self.cv_validator = TimeSeriesCrossValidator(
                config=self.config.cv_config,
                temporal_store=self.temporal_store,
                pit_engine=self.pit_engine
            )
        else:
            self.cv_validator = None
        
        # Walk-forward splitter
        if self.config.enable_walkforward and self.temporal_store:
            self.walkforward_splitter = WalkForwardSplitter(
                config=self.config.walkforward_config,
                temporal_store=self.temporal_store,
                pit_engine=self.pit_engine
            )
        else:
            self.walkforward_splitter = None
        
        # Performance tracker
        if self.config.enable_performance_tracking:
            model_id = f"lightgbm_alpha_{self.config.model_version or 'default'}"
            self.performance_tracker = ModelPerformanceTracker(model_id)
        else:
            self.performance_tracker = None
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        symbol_universe: Optional[List[str]] = None
    ) -> TrainingResult:
        """
        Execute the complete training pipeline.
        
        Args:
            X: Feature matrix with MultiIndex (date, symbol)
            y: Target variable (forward returns)
            benchmark_returns: Benchmark returns for comparison
            symbol_universe: List of symbols in the universe
            
        Returns:
            Complete training results
        """
        start_time = datetime.now()
        pipeline_id = f"pipeline_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting training pipeline {pipeline_id}")
        logger.info(f"Training data: {len(X)} samples, {len(X.columns)} features")
        
        # Initialize result object
        result = TrainingResult(
            pipeline_id=pipeline_id,
            timestamp=start_time,
            config=self.config,
            training_duration_seconds=0.0
        )
        
        try:
            # Step 1: Data preparation and validation
            logger.info("Step 1: Data preparation and validation")
            X_clean, y_clean = self._prepare_and_validate_data(X, y)
            
            if len(X_clean) == 0:
                raise ValueError("No valid data after cleaning")
            
            # Step 2: Hyperparameter optimization (if enabled)
            best_params = None
            if self.config.enable_hyperopt and self.hyperopt_optimizer:
                logger.info("Step 2: Hyperparameter optimization")
                opt_result = self.hyperopt_optimizer.optimize(
                    X_clean, y_clean,
                    walk_forward_splitter=self.walkforward_splitter,
                    base_model_config=self.config.model_config
                )
                
                result.optimization_result = opt_result.to_dict()
                result.best_hyperparameters = opt_result.best_params
                best_params = opt_result.best_params
                
                logger.info(f"Optimization completed: best {self.config.optimization_config.primary_metric} = {opt_result.best_value:.6f}")
            
            # Step 3: Model training with best parameters
            logger.info("Step 3: Model training")
            model = self._train_final_model(X_clean, y_clean, best_params)
            result.best_model = model
            
            # Step 4: Cross-validation (if enabled)
            if self.config.enable_cv and self.cv_validator:
                logger.info("Step 4: Cross-validation")
                cv_results = self.cv_validator.cross_validate_model(
                    model.model, X_clean, y_clean
                )
                
                result.cv_results = cv_results
                result.cv_scores = cv_results['scores']
                
                logger.info(f"CV completed: {cv_results['summary']['success_rate']:.1%} success rate")
            
            # Step 5: Walk-forward validation (if enabled)
            if self.config.enable_walkforward and self.walkforward_splitter:
                logger.info("Step 5: Walk-forward validation")
                
                # This would be implemented with actual walk-forward testing
                # For now, we'll skip this step or implement a simplified version
                logger.info("Walk-forward validation: Implementation pending")
            
            # Step 6: Performance evaluation
            logger.info("Step 6: Performance evaluation")
            predictions = model.predict(X_clean)
            
            if self.performance_tracker:
                snapshot = self.performance_tracker.calculate_performance(
                    pd.Series(predictions, index=y_clean.index),
                    y_clean,
                    benchmark_returns
                )
                result.performance_snapshot = snapshot.to_dict()
            
            # Calculate quick performance metrics
            result.performance_summary = calculate_model_metrics(
                pd.Series(predictions, index=y_clean.index),
                y_clean,
                benchmark_returns
            )
            
            # Step 7: Feature importance analysis
            if self.config.track_feature_importance:
                logger.info("Step 7: Feature importance analysis")
                result.feature_importance = model.get_feature_importance()
                result.feature_stability = self._analyze_feature_stability(model)
            
            # Step 8: Validation checks
            logger.info("Step 8: Final validation")
            result.validation_passed = self._validate_model_performance(result)
            result.meets_performance_targets = self._check_performance_targets(result)
            result.training_successful = True
            
            # Step 9: Model saving (if enabled)
            if self.config.save_models:
                logger.info("Step 9: Model saving")
                model_filename = f"{pipeline_id}_model.pkl"
                model_path = Path(self.config.output_dir) / model_filename
                model.save_model(model_path)
                result.model_path = str(model_path)
            
            logger.info(f"Training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            result.training_successful = False
            raise
        
        finally:
            # Calculate total training time
            result.training_duration_seconds = (datetime.now() - start_time).total_seconds()
            
            # Save results if enabled
            if self.config.save_results:
                self._save_training_results(result)
        
        return result
    
    def _prepare_and_validate_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and validate training data."""
        
        logger.info("Preparing and validating training data")
        
        # Ensure alignment
        common_idx = X.index.intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        
        logger.info(f"Data aligned: {len(common_idx)} common samples")
        
        # Remove NaN values
        valid_mask = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())
        X_clean = X_aligned[valid_mask]
        y_clean = y_aligned[valid_mask]
        
        logger.info(f"Data cleaned: {len(X_clean)} valid samples")
        
        # Validate data quality
        self._validate_data_quality(X_clean, y_clean)
        
        return X_clean, y_clean
    
    def _validate_data_quality(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate data quality and raise warnings."""
        
        # Check minimum sample size
        if len(X) < 1000:
            logger.warning(f"Low sample size: {len(X)} < 1000")
        
        # Check feature completeness
        missing_rate = X.isnull().sum() / len(X)
        high_missing_features = missing_rate[missing_rate > 0.1].index.tolist()
        if high_missing_features:
            logger.warning(f"Features with >10% missing: {high_missing_features}")
        
        # Check target distribution
        target_std = y.std()
        if target_std < 0.001:
            logger.warning(f"Low target variance: std = {target_std:.6f}")
        
        # Check for outliers in target
        target_abs = y.abs()
        outlier_threshold = target_abs.quantile(0.99)
        outlier_rate = (target_abs > outlier_threshold).mean()
        if outlier_rate > 0.05:
            logger.warning(f"High outlier rate in target: {outlier_rate:.1%}")
    
    def _train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        best_params: Optional[Dict[str, Any]] = None
    ) -> LightGBMAlphaModel:
        """Train the final model with optimized parameters."""
        
        # Create model configuration
        model_config = ModelConfig()
        model_config.target_horizons = self.config.target_horizons
        
        # Apply optimized parameters if available
        if best_params:
            model_config.base_params.update(best_params)
        
        # Initialize and train model
        model = LightGBMAlphaModel(model_config)
        
        # Prepare training data
        X_train, y_train = model.prepare_training_data(
            features=X, 
            returns=y.to_frame('returns'), 
            target_horizon=self.config.target_horizons[0]
        )
        
        # Train with validation split
        if self.config.validation_split > 0:
            split_idx = int(len(X_train) * (1 - self.config.validation_split))
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train.iloc[:split_idx]
            
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Train the model
        training_stats = model.train(
            X_train, y_train,
            validation_data=validation_data,
            verbose=True
        )
        
        logger.info(f"Model trained: {training_stats['training_time_seconds']:.1f}s")
        
        return model
    
    def _analyze_feature_stability(self, model: LightGBMAlphaModel) -> Dict[str, float]:
        """Analyze feature importance stability."""
        
        if model.feature_importance_ is None:
            return {}
        
        # Calculate feature stability metrics
        top_features = model.feature_importance_.head(20)
        
        stability_metrics = {
            'top_feature_importance': top_features.iloc[0]['importance'] / top_features['importance'].sum(),
            'top_10_concentration': top_features.head(10)['importance'].sum() / top_features['importance'].sum(),
            'importance_gini': self._calculate_gini_coefficient(top_features['importance'].values)
        }
        
        return stability_metrics
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for feature importance concentration."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return gini
    
    def _validate_model_performance(self, result: TrainingResult) -> bool:
        """Validate that model performance meets minimum standards."""
        
        if not result.performance_summary:
            return False
        
        # Check critical metrics
        critical_checks = [
            result.performance_summary.get('ic', 0) > 0.02,  # Minimum IC
            result.performance_summary.get('hit_rate', 0) > 0.48,  # Minimum hit rate
            result.performance_summary.get('sharpe_ratio', 0) > 0.5,  # Minimum Sharpe
        ]
        
        return all(critical_checks)
    
    def _check_performance_targets(self, result: TrainingResult) -> bool:
        """Check if performance meets Taiwan market targets."""
        
        if not result.performance_summary:
            return False
        
        targets = self.config.performance_targets
        
        target_checks = [
            result.performance_summary.get('ic', 0) >= targets['ic_threshold'],
            result.performance_summary.get('sharpe_ratio', 0) >= targets['sharpe_threshold'],
            result.performance_summary.get('max_drawdown', 1) <= targets['max_drawdown_threshold'],
            result.performance_summary.get('hit_rate', 0) >= targets['hit_rate_threshold']
        ]
        
        return all(target_checks)
    
    def _save_training_results(self, result: TrainingResult) -> None:
        """Save training results to disk."""
        
        results_filename = f"{result.pipeline_id}_results.json"
        results_path = Path(self.config.output_dir) / results_filename
        
        with open(results_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Training results saved to {results_path}")
    
    def load_trained_model(self, model_path: str) -> LightGBMAlphaModel:
        """Load a previously trained model."""
        model = LightGBMAlphaModel()
        model.load_model(model_path)
        return model
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training pipeline configuration."""
        return {
            'pipeline_components': {
                'hyperopt_enabled': self.config.enable_hyperopt,
                'cv_enabled': self.config.enable_cv,
                'walkforward_enabled': self.config.enable_walkforward,
                'performance_tracking_enabled': self.config.enable_performance_tracking
            },
            'model_configuration': {
                'target_horizons': self.config.target_horizons,
                'validation_split': self.config.validation_split,
                'early_stopping': self.config.enable_early_stopping
            },
            'performance_targets': self.config.performance_targets,
            'output_directory': self.config.output_dir
        }


# Utility functions

def create_taiwan_training_config(**kwargs) -> TrainingPipelineConfig:
    """Create training pipeline configuration optimized for Taiwan market."""
    config = TrainingPipelineConfig()
    
    # Taiwan market specific defaults
    config.target_horizons = [5, 10, 20]  # 1 week, 2 weeks, 1 month
    config.performance_targets = {
        'ic_threshold': 0.06,  # Higher target for Taiwan
        'sharpe_threshold': 2.0,  # Aggressive target
        'max_drawdown_threshold': 0.15,  # Strict risk control
        'hit_rate_threshold': 0.52  # Above random
    }
    config.hyperopt_trials = 100  # Thorough optimization
    config.cv_folds = 5  # Standard CV
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    return config


def run_quick_training(
    X: pd.DataFrame,
    y: pd.Series,
    enable_optimization: bool = True,
    n_trials: int = 20
) -> TrainingResult:
    """Quick training pipeline for development/testing."""
    
    config = create_taiwan_training_config(
        enable_hyperopt=enable_optimization,
        hyperopt_trials=n_trials,
        hyperopt_timeout_hours=1,
        cv_folds=3,
        enable_walkforward=False  # Skip for quick training
    )
    
    pipeline = TaiwanMarketTrainingPipeline(config)
    return pipeline.train_model(X, y)


if __name__ == "__main__":
    # Demo training pipeline
    print("Taiwan Market Training Pipeline for LightGBM Alpha Model")
    print("Integrates hyperparameter optimization, CV, and performance tracking")
    
    # Example configuration
    config = create_taiwan_training_config()
    print(f"Default training configuration created")
    print(f"Target horizons: {config.target_horizons} days")
    print(f"Performance targets: {config.performance_targets}")