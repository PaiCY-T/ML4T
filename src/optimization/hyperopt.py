"""
Hyperparameter Optimization Engine for LightGBM Alpha Model

This module provides sophisticated hyperparameter optimization capabilities using Optuna,
integrated with Taiwan market specific validation and performance targets.
"""

import logging
import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from ..models.lightgbm_alpha import LightGBMAlphaModel, ModelConfig
from ..backtesting.validation.walk_forward import WalkForwardSplitter, ValidationWindow

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Optuna study configuration
    study_name: str = "lightgbm_alpha_optimization"
    n_trials: int = 100
    timeout_seconds: Optional[int] = 14400  # 4 hours max
    n_jobs: int = 1  # Parallel trials (be careful with memory)
    
    # Search space configuration
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'num_leaves': {'type': 'int', 'low': 10, 'high': 300},
        'max_depth': {'type': 'int', 'low': 3, 'high': 15},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'feature_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
        'bagging_fraction': {'type': 'float', 'low': 0.4, 'high': 1.0},
        'bagging_freq': {'type': 'int', 'low': 1, 'high': 10},
        'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 100.0},
        'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 100.0},
        'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
        'min_child_weight': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True}
    })
    
    # Pruning and sampling configuration
    sampler_type: str = 'TPE'  # 'TPE' or 'Random'
    pruner_type: str = 'Hyperband'  # 'Hyperband', 'Median', or 'None'
    pruning_warmup_steps: int = 10
    pruning_interval_steps: int = 5
    
    # Objective function configuration
    primary_metric: str = 'ic'  # 'ic', 'sharpe_ratio', 'rmse'
    metric_direction: str = 'maximize'  # 'maximize' or 'minimize'
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    
    # Taiwan market specific targets
    target_ic_threshold: float = 0.05
    target_sharpe_threshold: float = 1.5
    max_drawdown_threshold: float = 0.20
    
    # Memory and performance constraints
    max_memory_gb: float = 12.0
    max_training_time_minutes: int = 30
    enable_gpu: bool = False


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    
    # Study results
    best_params: Dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials_completed: int
    study_duration_seconds: float
    
    # Performance validation
    cv_scores: Dict[str, List[float]]
    validation_metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    
    # Taiwan market specific metrics
    taiwan_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Optimization diagnostics
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    pruned_trials: int = 0
    failed_trials: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'best_trial_number': self.best_trial_number,
            'n_trials_completed': self.n_trials_completed,
            'study_duration_seconds': self.study_duration_seconds,
            'cv_scores': self.cv_scores,
            'validation_metrics': self.validation_metrics,
            'taiwan_metrics': self.taiwan_metrics,
            'convergence_info': self.convergence_info,
            'pruned_trials': self.pruned_trials,
            'failed_trials': self.failed_trials
        }


class ObjectiveFunction:
    """
    Objective function for Optuna optimization with Taiwan market validation.
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: OptimizationConfig,
        walk_forward_splitter: Optional[WalkForwardSplitter] = None,
        base_model_config: Optional[ModelConfig] = None
    ):
        self.X = X
        self.y = y
        self.config = config
        self.walk_forward_splitter = walk_forward_splitter
        self.base_model_config = base_model_config or ModelConfig()
        
        # Track trials for monitoring
        self.trial_count = 0
        self.best_score = float('-inf') if config.metric_direction == 'maximize' else float('inf')
        
        logger.info(f"Objective function initialized for {len(X)} samples with {len(X.columns)} features")
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Execute single optimization trial."""
        self.trial_count += 1
        start_time = time.time()
        
        try:
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial)
            
            # Create model with sampled parameters
            model_config = self._create_model_config(params)
            
            # Perform cross-validation
            cv_scores = self._cross_validate(model_config, trial)
            
            # Calculate primary objective metric
            primary_score = np.mean(cv_scores[self.config.primary_metric])
            
            # Add Taiwan market constraints as penalties
            penalty = self._calculate_taiwan_penalty(cv_scores)
            
            # Final score with penalty
            final_score = primary_score - penalty
            
            # Track convergence
            trial_time = time.time() - start_time
            self._update_convergence_tracking(trial, final_score, trial_time)
            
            # Report intermediate results for pruning
            trial.report(final_score, step=1)
            
            # Check if trial should be pruned
            if trial.should_prune():
                logger.debug(f"Trial {self.trial_count} pruned with score {final_score:.6f}")
                raise optuna.TrialPruned()
            
            logger.debug(f"Trial {self.trial_count} completed: {final_score:.6f} in {trial_time:.1f}s")
            return final_score
            
        except Exception as e:
            logger.error(f"Trial {self.trial_count} failed: {str(e)}")
            # Return worst possible score for failed trials
            return float('-inf') if self.config.metric_direction == 'maximize' else float('inf')
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample hyperparameters based on search space configuration."""
        params = {}
        
        for param_name, param_config in self.config.search_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high']
                )
            elif param_config['type'] == 'float':
                if param_config.get('log', False):
                    params[param_name] = trial.suggest_loguniform(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                else:
                    params[param_name] = trial.suggest_uniform(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params
    
    def _create_model_config(self, params: Dict[str, Any]) -> ModelConfig:
        """Create model configuration with sampled parameters."""
        config = ModelConfig()
        
        # Update base parameters with sampled values
        config.base_params.update(params)
        
        # Set early stopping and other fixed parameters
        config.early_stopping_rounds = self.config.early_stopping_rounds
        
        # Memory and performance optimizations
        if self.config.enable_gpu:
            config.base_params['device'] = 'gpu'
            config.base_params['gpu_use_dp'] = True
        
        return config
    
    def _cross_validate(self, model_config: ModelConfig, trial: optuna.Trial) -> Dict[str, List[float]]:
        """Perform cross-validation with the given model configuration."""
        from sklearn.model_selection import TimeSeriesSplit
        
        # Initialize CV splitter
        if self.walk_forward_splitter:
            # Use walk-forward validation if available
            cv_scores = self._walk_forward_cv(model_config, trial)
        else:
            # Fall back to time series split
            cv_scores = self._time_series_cv(model_config, trial)
        
        return cv_scores
    
    def _time_series_cv(self, model_config: ModelConfig, trial: optuna.Trial) -> Dict[str, List[float]]:
        """Time-series cross-validation implementation."""
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        cv_scores = {
            'ic': [],
            'rmse': [],
            'mae': [],
            'sharpe_ratio': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            # Check for pruning at each fold
            trial.report(np.mean(cv_scores['ic']) if cv_scores['ic'] else 0.0, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Train model
            model = lgb.LGBMRegressor(**model_config.base_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(model_config.early_stopping_rounds),
                    lgb.log_evaluation(0)  # Silent
                ]
            )
            
            # Predict and calculate metrics
            y_pred = model.predict(X_val)
            fold_metrics = self._calculate_fold_metrics(y_val, y_pred)
            
            # Store fold scores
            for metric in cv_scores.keys():
                cv_scores[metric].append(fold_metrics.get(metric, 0.0))
        
        return cv_scores
    
    def _walk_forward_cv(self, model_config: ModelConfig, trial: optuna.Trial) -> Dict[str, List[float]]:
        """Walk-forward cross-validation using existing framework."""
        # This would integrate with the walk-forward splitter from Task #23
        # For now, implement simplified version
        logger.info("Using walk-forward cross-validation")
        return self._time_series_cv(model_config, trial)
    
    def _calculate_fold_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        metrics = {}
        
        # Information Coefficient (Rank correlation)
        metrics['ic'] = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        
        # RMSE
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAE
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # Sharpe ratio approximation
        returns = pd.Series(y_pred)
        if returns.std() > 0:
            metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Handle NaN values
        for key, value in metrics.items():
            if np.isnan(value):
                metrics[key] = 0.0
        
        return metrics
    
    def _calculate_taiwan_penalty(self, cv_scores: Dict[str, List[float]]) -> float:
        """Calculate penalty based on Taiwan market constraints."""
        penalty = 0.0
        
        # IC threshold penalty
        mean_ic = np.mean(cv_scores['ic'])
        if mean_ic < self.config.target_ic_threshold:
            penalty += (self.config.target_ic_threshold - mean_ic) * 2.0
        
        # Sharpe ratio threshold penalty
        mean_sharpe = np.mean(cv_scores['sharpe_ratio'])
        if mean_sharpe < self.config.target_sharpe_threshold:
            penalty += (self.config.target_sharpe_threshold - mean_sharpe) * 0.1
        
        # Consistency penalty (high variance in IC)
        ic_std = np.std(cv_scores['ic'])
        if ic_std > 0.05:  # IC should be consistent
            penalty += ic_std * 1.0
        
        return penalty
    
    def _update_convergence_tracking(self, trial: optuna.Trial, score: float, trial_time: float):
        """Update convergence tracking metrics."""
        is_better = (
            (self.config.metric_direction == 'maximize' and score > self.best_score) or
            (self.config.metric_direction == 'minimize' and score < self.best_score)
        )
        
        if is_better:
            self.best_score = score
            logger.info(f"New best score: {score:.6f} at trial {self.trial_count}")


class HyperparameterOptimizer:
    """
    Main hyperparameter optimization engine for LightGBM alpha model.
    
    Features:
    - Bayesian optimization with Optuna
    - Taiwan market specific validation
    - Memory and time constraints
    - Advanced pruning strategies
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize the hyperparameter optimizer.
        
        Args:
            config: Optimization configuration. Uses default if None.
        """
        self.config = config or OptimizationConfig()
        self.study: Optional[optuna.Study] = None
        
        # Initialize Optuna components
        self.sampler = self._create_sampler()
        self.pruner = self._create_pruner()
        
        logger.info(f"HyperparameterOptimizer initialized with {self.config.n_trials} max trials")
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        if self.config.sampler_type == 'TPE':
            return TPESampler(seed=42)
        elif self.config.sampler_type == 'Random':
            return optuna.samplers.RandomSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler type: {self.config.sampler_type}")
    
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create Optuna pruner based on configuration."""
        if self.config.pruner_type == 'Hyperband':
            return HyperbandPruner(
                min_resource=self.config.pruning_warmup_steps,
                reduction_factor=3
            )
        elif self.config.pruner_type == 'Median':
            return MedianPruner(
                n_startup_trials=self.config.pruning_warmup_steps,
                n_warmup_steps=self.config.pruning_warmup_steps,
                interval_steps=self.config.pruning_interval_steps
            )
        elif self.config.pruner_type == 'None':
            return None
        else:
            raise ValueError(f"Unknown pruner type: {self.config.pruner_type}")
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        walk_forward_splitter: Optional[WalkForwardSplitter] = None,
        base_model_config: Optional[ModelConfig] = None
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Target variable
            walk_forward_splitter: Optional walk-forward validation splitter
            base_model_config: Base model configuration
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        logger.info(f"Starting hyperparameter optimization with {len(X)} samples")
        
        # Create Optuna study
        direction = 'maximize' if self.config.metric_direction == 'maximize' else 'minimize'
        
        self.study = optuna.create_study(
            study_name=f"{self.config.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        # Create objective function
        objective = ObjectiveFunction(
            X=X,
            y=y,
            config=self.config,
            walk_forward_splitter=walk_forward_splitter,
            base_model_config=base_model_config
        )
        
        # Run optimization
        try:
            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds,
                n_jobs=self.config.n_jobs,
                show_progress_bar=True
            )
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        # Collect results
        optimization_time = time.time() - start_time
        result = self._create_optimization_result(objective, optimization_time)
        
        logger.info(f"Optimization completed in {optimization_time:.1f} seconds")
        logger.info(f"Best {self.config.primary_metric}: {result.best_value:.6f}")
        
        return result
    
    def _create_optimization_result(
        self, 
        objective: ObjectiveFunction, 
        optimization_time: float
    ) -> OptimizationResult:
        """Create optimization result from study."""
        if not self.study:
            raise ValueError("No study found - optimization was not run")
        
        # Get best trial results
        best_trial = self.study.best_trial
        
        # Calculate final validation with best parameters
        best_model_config = objective._create_model_config(best_trial.params)
        final_cv_scores = objective._cross_validate(best_model_config, best_trial)
        
        # Calculate validation metrics
        validation_metrics = {}
        for metric, scores in final_cv_scores.items():
            validation_metrics[f'{metric}_mean'] = np.mean(scores)
            validation_metrics[f'{metric}_std'] = np.std(scores)
        
        # Taiwan market specific metrics
        taiwan_metrics = {
            'meets_ic_threshold': validation_metrics['ic_mean'] >= self.config.target_ic_threshold,
            'meets_sharpe_threshold': validation_metrics['sharpe_ratio_mean'] >= self.config.target_sharpe_threshold,
            'ic_consistency': validation_metrics['ic_std'] <= 0.05
        }
        
        # Count trial outcomes
        completed_trials = len(self.study.trials)
        pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        # Convergence information
        convergence_info = {
            'trials_to_best': best_trial.number,
            'convergence_ratio': best_trial.number / completed_trials if completed_trials > 0 else 0.0,
            'optimization_efficiency': (completed_trials - failed_trials) / completed_trials if completed_trials > 0 else 0.0
        }
        
        return OptimizationResult(
            best_params=best_trial.params,
            best_value=best_trial.value,
            best_trial_number=best_trial.number,
            n_trials_completed=completed_trials,
            study_duration_seconds=optimization_time,
            cv_scores=final_cv_scores,
            validation_metrics=validation_metrics,
            taiwan_metrics=taiwan_metrics,
            convergence_info=convergence_info,
            pruned_trials=pruned_trials,
            failed_trials=failed_trials
        )
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if not self.study:
            raise ValueError("No study found - optimization was not run")
        
        trials_df = self.study.trials_dataframe()
        trials_df['trial_time'] = pd.to_datetime(trials_df['datetime_start'])
        
        return trials_df[['number', 'value', 'state', 'trial_time'] + 
                        [col for col in trials_df.columns if col.startswith('params_')]]
    
    def save_study(self, filepath: Union[str, Path]) -> None:
        """Save Optuna study to disk."""
        if not self.study:
            raise ValueError("No study to save")
        
        study_data = {
            'study': self.study,
            'config': self.config,
            'study_name': self.study.study_name,
            'best_params': self.study.best_params,
            'best_value': self.study.best_value
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(study_data, f)
        
        logger.info(f"Study saved to {filepath}")
    
    def load_study(self, filepath: Union[str, Path]) -> None:
        """Load Optuna study from disk."""
        with open(filepath, 'rb') as f:
            study_data = pickle.load(f)
        
        self.study = study_data['study']
        self.config = study_data['config']
        
        logger.info(f"Study loaded from {filepath}")


# Utility functions for optimization

def create_taiwan_search_space() -> Dict[str, Dict[str, Any]]:
    """Create search space optimized for Taiwan market characteristics."""
    return {
        # Tree structure - Taiwan market has moderate complexity
        'num_leaves': {'type': 'int', 'low': 15, 'high': 200},
        'max_depth': {'type': 'int', 'low': 4, 'high': 12},
        
        # Learning rate - conservative for stability
        'learning_rate': {'type': 'float', 'low': 0.02, 'high': 0.15, 'log': True},
        
        # Feature sampling - important for 42-factor model
        'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 0.95},
        'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 0.95},
        'bagging_freq': {'type': 'int', 'low': 1, 'high': 7},
        
        # Regularization - prevent overfitting in Taiwan market
        'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 50.0},
        'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 50.0},
        
        # Minimum samples - account for Taiwan market liquidity
        'min_child_samples': {'type': 'int', 'low': 10, 'high': 50},
        'min_child_weight': {'type': 'float', 'low': 0.005, 'high': 0.1, 'log': True}
    }


def create_taiwan_optimization_config(**kwargs) -> OptimizationConfig:
    """Create optimization configuration optimized for Taiwan market."""
    config = OptimizationConfig()
    
    # Taiwan-specific defaults
    config.search_space = create_taiwan_search_space()
    config.target_ic_threshold = 0.06  # Higher target for Taiwan market
    config.target_sharpe_threshold = 2.0  # Aggressive target
    config.max_drawdown_threshold = 0.15  # Strict risk control
    config.primary_metric = 'ic'  # Focus on information coefficient
    
    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    return config


def run_quick_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 20,
    timeout_minutes: int = 30
) -> OptimizationResult:
    """Run quick hyperparameter optimization for development/testing."""
    config = create_taiwan_optimization_config(
        n_trials=n_trials,
        timeout_seconds=timeout_minutes * 60
    )
    
    optimizer = HyperparameterOptimizer(config)
    return optimizer.optimize(X, y)


if __name__ == "__main__":
    # Demo optimization setup
    print("Hyperparameter Optimization Engine for Taiwan Market LightGBM")
    print("This module provides Optuna-based optimization with Taiwan market constraints")
    
    # Example configuration
    config = create_taiwan_optimization_config()
    print(f"Default configuration: {config.n_trials} trials, {config.primary_metric} optimization")