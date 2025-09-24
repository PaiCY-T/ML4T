"""
Training Pipeline for LightGBM Alpha Model

Comprehensive training pipeline with memory optimization, cross-validation,
and Taiwan market integration for 2000-stock universe.
"""

import gc
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .lightgbm_alpha import LightGBMAlphaModel, ModelConfig
from .feature_pipeline import FeaturePipeline, FeatureConfig
from .taiwan_market import TaiwanMarketModel, MarketAdaptations

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline."""
    
    # Data parameters
    train_start_date: str = '2020-01-01'
    train_end_date: str = '2023-12-31'
    validation_start_date: str = '2024-01-01'
    validation_end_date: str = '2024-06-30'
    
    # Target parameters
    target_horizon_days: int = 5
    rebalance_frequency_days: int = 5
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_gap_days: int = 2  # Gap between train/val to prevent leakage (T+2 settlement)
    expanding_window: bool = True
    
    # Memory optimization
    chunk_size: int = 500  # Process stocks in chunks
    memory_limit_gb: float = 12.0
    cleanup_intermediate: bool = True
    use_dask: bool = False  # Option to use Dask for larger datasets
    
    # Model ensemble
    ensemble_size: int = 1  # Number of models to train
    bootstrap_samples: bool = False
    
    # Performance targets
    min_ic_threshold: float = 0.05
    target_sharpe: float = 2.0
    max_training_hours: float = 6.0


class TrainingPipeline:
    """
    Memory-optimized training pipeline for Taiwan market alpha model.
    
    Handles the complete workflow from data loading to model training
    with specific optimizations for 2000-stock universe.
    """
    
    def __init__(
        self, 
        training_config: Optional[TrainingConfig] = None,
        model_config: Optional[ModelConfig] = None,
        feature_config: Optional[FeatureConfig] = None
    ):
        """Initialize training pipeline."""
        self.training_config = training_config or TrainingConfig()
        self.model_config = model_config or ModelConfig()
        self.feature_config = feature_config or FeatureConfig()
        
        # Initialize components
        self.feature_pipeline = FeaturePipeline(self.feature_config)
        self.market_adaptations = MarketAdaptations()
        
        # Training state
        self.trained_models: List[TaiwanMarketModel] = []
        self.training_history: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, Any] = {}
        
        logger.info("TrainingPipeline initialized with memory optimization")
    
    def run_complete_training(
        self,
        data_sources: Dict[str, Union[str, pd.DataFrame, Callable]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline with memory optimization.
        
        Args:
            data_sources: Dictionary with data sources/loaders
                - 'price_data': Price/return data
                - 'volume_data': Volume data
                - 'fundamental_data': Optional fundamental data
                - 'market_data': Additional market data
            output_dir: Directory to save models and results
            
        Returns:
            Training summary with performance metrics
        """
        start_time = datetime.now()
        logger.info("Starting complete training pipeline")
        
        output_dir = Path(output_dir) if output_dir else Path('./models/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Load and validate data
            logger.info("Step 1: Loading and validating data")
            train_data, val_data = self._load_and_validate_data(data_sources)
            
            # Step 2: Memory optimization check
            memory_usage = self._estimate_memory_usage(train_data)
            if memory_usage > self.training_config.memory_limit_gb:
                logger.warning(f"Data may exceed memory limit ({memory_usage:.1f}GB > {self.training_config.memory_limit_gb}GB)")
                train_data, val_data = self._optimize_data_memory(train_data, val_data)
            
            # Step 3: Feature engineering
            logger.info("Step 3: Feature engineering and processing")
            X_train, y_train = self._prepare_training_features(train_data)
            X_val, y_val = self._prepare_validation_features(val_data)
            
            # Step 4: Cross-validation
            logger.info("Step 4: Running cross-validation")
            cv_results = self._run_cross_validation(X_train, y_train)
            
            # Step 5: Train final model(s)
            logger.info("Step 5: Training final model ensemble")
            final_models = self._train_model_ensemble(X_train, y_train, X_val, y_val)
            
            # Step 6: Final validation
            logger.info("Step 6: Final validation and performance analysis")
            validation_results = self._final_validation(final_models, X_val, y_val)
            
            # Step 7: Save results
            logger.info("Step 7: Saving models and results")
            self._save_training_results(final_models, output_dir)
            
            training_time = (datetime.now() - start_time).total_seconds() / 3600
            
            # Compile final results
            final_results = {
                'training_time_hours': training_time,
                'models_trained': len(final_models),
                'cv_results': cv_results,
                'validation_results': validation_results,
                'data_stats': self._get_data_statistics(train_data, val_data),
                'memory_usage_gb': memory_usage,
                'output_directory': str(output_dir)
            }
            
            logger.info(f"Training pipeline completed in {training_time:.2f} hours")
            logger.info(f"Best validation IC: {validation_results.get('best_ic', 'N/A'):.4f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
        finally:
            # Cleanup memory
            if self.training_config.cleanup_intermediate:
                gc.collect()
    
    def _load_and_validate_data(
        self, 
        data_sources: Dict[str, Any]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Load and validate data with memory optimization."""
        logger.info("Loading data with memory optimization")
        
        # Load training data
        train_data = {}
        val_data = {}
        
        for data_type, source in data_sources.items():
            if callable(source):
                # Data loader function
                full_data = source()
            elif isinstance(source, str):
                # File path
                full_data = pd.read_parquet(source) if source.endswith('.parquet') else pd.read_csv(source)
            else:
                # DataFrame
                full_data = source.copy()
            
            # Split into train/validation
            train_mask = (
                (full_data.index.get_level_values('date') >= self.training_config.train_start_date) &
                (full_data.index.get_level_values('date') <= self.training_config.train_end_date)
            )
            val_mask = (
                (full_data.index.get_level_values('date') >= self.training_config.validation_start_date) &
                (full_data.index.get_level_values('date') <= self.training_config.validation_end_date)
            )
            
            train_data[data_type] = full_data[train_mask]
            val_data[data_type] = full_data[val_mask]
            
            logger.info(f"Loaded {data_type}: {len(train_data[data_type])} train, {len(val_data[data_type])} val samples")
        
        # Validate data quality
        self._validate_data_quality(train_data)
        
        return train_data, val_data
    
    def _validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> None:
        """Validate data quality and completeness."""
        validation_issues = []
        
        for data_type, df in data.items():
            # Check for excessive missing values
            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            if missing_pct > 0.3:
                validation_issues.append(f"{data_type} has {missing_pct:.1%} missing values")
            
            # Check for data gaps
            if 'date' in df.index.names:
                date_range = pd.date_range(
                    df.index.get_level_values('date').min(),
                    df.index.get_level_values('date').max(),
                    freq='D'
                )
                actual_dates = df.index.get_level_values('date').unique()
                missing_dates = len(date_range) - len(actual_dates)
                if missing_dates > len(date_range) * 0.1:  # More than 10% missing
                    validation_issues.append(f"{data_type} missing {missing_dates} dates")
        
        if validation_issues:
            logger.warning(f"Data quality issues found: {validation_issues}")
        else:
            logger.info("Data quality validation passed")
    
    def _estimate_memory_usage(self, data: Dict[str, pd.DataFrame]) -> float:
        """Estimate total memory usage in GB."""
        total_bytes = 0
        for df in data.values():
            total_bytes += df.memory_usage(deep=True).sum()
        
        return total_bytes / (1024**3)  # Convert to GB
    
    def _optimize_data_memory(
        self, 
        train_data: Dict[str, pd.DataFrame],
        val_data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Optimize data memory usage."""
        logger.info("Optimizing data memory usage")
        
        optimized_train = {}
        optimized_val = {}
        
        for data_type in train_data.keys():
            # Convert to optimal dtypes
            optimized_train[data_type] = self._optimize_dataframe_memory(train_data[data_type])
            optimized_val[data_type] = self._optimize_dataframe_memory(val_data[data_type])
        
        return optimized_train, optimized_val
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize single DataFrame memory usage."""
        optimized_df = df.copy()
        
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            # Convert float64 to float32 if precision allows
            if optimized_df[col].max() < np.finfo(np.float32).max and \
               optimized_df[col].min() > np.finfo(np.float32).min:
                optimized_df[col] = optimized_df[col].astype(np.float32)
        
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            # Convert int64 to smaller int types if possible
            col_max = optimized_df[col].max()
            col_min = optimized_df[col].min()
            
            if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
        
        return optimized_df
    
    def _prepare_training_features(self, train_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training features with memory optimization."""
        logger.info("Preparing training features")
        
        # Calculate returns for target
        price_data = train_data['price_data']
        returns = self._calculate_forward_returns(price_data, self.training_config.target_horizon_days)
        
        # Fit feature pipeline
        X_train = self.feature_pipeline.fit_transform(
            train_data.get('price_data', pd.DataFrame()),
            train_data.get('volume_data', pd.DataFrame()),
            returns,
            train_data.get('fundamental_data'),
            train_data.get('market_data')
        )
        
        # Align features and targets
        common_idx = X_train.index.intersection(returns.index)
        X_train = X_train.loc[common_idx]
        y_train = returns.loc[common_idx]
        
        logger.info(f"Training features prepared: {X_train.shape}")
        
        return X_train, y_train
    
    def _prepare_validation_features(self, val_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare validation features using fitted pipeline."""
        logger.info("Preparing validation features")
        
        # Calculate returns for target
        price_data = val_data['price_data']
        returns = self._calculate_forward_returns(price_data, self.training_config.target_horizon_days)
        
        # Transform features using fitted pipeline
        X_val = self.feature_pipeline.transform(
            val_data.get('price_data', pd.DataFrame()),
            val_data.get('volume_data', pd.DataFrame()),
            val_data.get('fundamental_data'),
            val_data.get('market_data')
        )
        
        # Align features and targets
        common_idx = X_val.index.intersection(returns.index)
        X_val = X_val.loc[common_idx]
        y_val = returns.loc[common_idx]
        
        logger.info(f"Validation features prepared: {X_val.shape}")
        
        return X_val, y_val
    
    def _calculate_forward_returns(self, price_data: pd.DataFrame, horizon: int) -> pd.Series:
        """Calculate forward returns with Taiwan market adjustments."""
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")
        
        returns_list = []
        
        # Calculate returns by symbol to handle MultiIndex properly
        for symbol in price_data.index.get_level_values('symbol').unique():
            symbol_prices = price_data.xs(symbol, level='symbol')
            
            # Calculate forward returns
            forward_returns = (
                symbol_prices['close'].shift(-horizon) / symbol_prices['close'] - 1
            )
            
            # Add symbol back to index
            forward_returns.index = pd.MultiIndex.from_product(
                [forward_returns.index, [symbol]], 
                names=['date', 'symbol']
            )
            
            returns_list.append(forward_returns)
        
        return pd.concat(returns_list).sort_index()
    
    def _run_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Run time-series cross-validation with memory optimization."""
        logger.info(f"Running {self.training_config.cv_folds}-fold cross-validation")
        
        # Create time-series splits
        dates = X.index.get_level_values('date').unique().sort_values()
        cv_results = {
            'fold_scores': [],
            'fold_metrics': [],
            'average_metrics': {}
        }
        
        # Calculate fold size
        n_dates = len(dates)
        if self.training_config.expanding_window:
            # Expanding window CV
            initial_train_size = max(n_dates // (self.training_config.cv_folds + 1), 100)
            test_size = n_dates // self.training_config.cv_folds
        else:
            # Rolling window CV
            fold_size = n_dates // self.training_config.cv_folds
        
        for fold in range(self.training_config.cv_folds):
            logger.info(f"Processing CV fold {fold + 1}/{self.training_config.cv_folds}")
            
            if self.training_config.expanding_window:
                train_end_idx = initial_train_size + fold * test_size
                test_start_idx = train_end_idx + self.training_config.cv_gap_days
                test_end_idx = min(test_start_idx + test_size, n_dates)
                
                train_dates = dates[:train_end_idx]
                test_dates = dates[test_start_idx:test_end_idx]
            else:
                test_start_idx = fold * fold_size
                test_end_idx = min(test_start_idx + fold_size, n_dates)
                train_start_idx = max(0, test_start_idx - fold_size * 2)  # Use 2x data for training
                
                train_dates = dates[train_start_idx:test_start_idx - self.training_config.cv_gap_days]
                test_dates = dates[test_start_idx:test_end_idx]
            
            if len(test_dates) == 0:
                logger.warning(f"No test data for fold {fold + 1}, skipping")
                continue
            
            # Create fold data
            train_mask = X.index.get_level_values('date').isin(train_dates)
            test_mask = X.index.get_level_values('date').isin(test_dates)
            
            X_fold_train = X[train_mask]
            y_fold_train = y[train_mask] 
            X_fold_test = X[test_mask]
            y_fold_test = y[test_mask]
            
            # Train fold model
            fold_model = LightGBMAlphaModel(self.model_config)
            fold_model.train(X_fold_train, y_fold_train, verbose=False)
            
            # Evaluate fold
            predictions = fold_model.predict(X_fold_test)
            fold_metrics = self._calculate_fold_metrics(y_fold_test, predictions)
            
            cv_results['fold_scores'].append(fold_metrics['ic'])
            cv_results['fold_metrics'].append(fold_metrics)
            
            logger.info(f"Fold {fold + 1} IC: {fold_metrics['ic']:.4f}")
            
            # Memory cleanup
            del fold_model, X_fold_train, y_fold_train, X_fold_test, y_fold_test
            gc.collect()
        
        # Calculate average metrics
        if cv_results['fold_metrics']:
            cv_results['average_metrics'] = {
                'mean_ic': np.mean([m['ic'] for m in cv_results['fold_metrics']]),
                'std_ic': np.std([m['ic'] for m in cv_results['fold_metrics']]),
                'mean_rmse': np.mean([m['rmse'] for m in cv_results['fold_metrics']]),
                'mean_sharpe': np.mean([m.get('sharpe', 0) for m in cv_results['fold_metrics']])
            }
            
            logger.info(f"CV Results - IC: {cv_results['average_metrics']['mean_ic']:.4f} ± {cv_results['average_metrics']['std_ic']:.4f}")
        
        return cv_results
    
    def _calculate_fold_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single CV fold."""
        metrics = {}
        
        # Information Coefficient (Spearman correlation)
        ic = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        metrics['ic'] = ic if not pd.isna(ic) else 0.0
        
        # RMSE
        metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Pearson correlation
        pearson_corr = pd.Series(y_true).corr(pd.Series(y_pred))
        metrics['correlation'] = pearson_corr if not pd.isna(pearson_corr) else 0.0
        
        # Hit rate (directional accuracy)
        if len(y_true) > 1:
            y_true_sign = np.sign(y_true)
            y_pred_sign = np.sign(y_pred)
            metrics['hit_rate'] = np.mean(y_true_sign == y_pred_sign)
        else:
            metrics['hit_rate'] = 0.5
        
        return metrics
    
    def _train_model_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> List[TaiwanMarketModel]:
        """Train ensemble of models."""
        logger.info(f"Training ensemble of {self.training_config.ensemble_size} models")
        
        models = []
        
        for i in range(self.training_config.ensemble_size):
            logger.info(f"Training model {i + 1}/{self.training_config.ensemble_size}")
            
            # Create model with slight variations for ensemble diversity
            model_config = self.model_config
            if self.training_config.bootstrap_samples and i > 0:
                # Add some randomness for ensemble diversity
                model_config.base_params['random_state'] = 42 + i
                model_config.base_params['feature_fraction'] = max(0.5, 0.8 - i * 0.05)
            
            # Create and train model
            base_model = LightGBMAlphaModel(model_config)
            training_stats = base_model.train(
                X_train, y_train,
                validation_data=(X_val, y_val),
                verbose=False
            )
            
            # Wrap with Taiwan market adaptations
            taiwan_model = TaiwanMarketModel(base_model, self.market_adaptations)
            models.append(taiwan_model)
            
            # Store training stats
            self.training_history.append({
                'model_id': i,
                'training_stats': training_stats,
                'timestamp': datetime.now()
            })
            
            logger.info(f"Model {i + 1} trained - Validation RMSE: {training_stats.get('best_score', 'N/A')}")
        
        self.trained_models = models
        return models
    
    def _final_validation(
        self,
        models: List[TaiwanMarketModel],
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Perform final validation of trained models."""
        logger.info("Running final validation")
        
        validation_results = {
            'individual_model_performance': [],
            'ensemble_performance': {},
            'feature_importance_analysis': {},
            'taiwan_market_analysis': {}
        }
        
        # Evaluate individual models
        all_predictions = []
        for i, model in enumerate(models):
            predictions = model.base_model.predict(X_val)
            all_predictions.append(predictions)
            
            # Calculate individual metrics
            individual_metrics = self._calculate_comprehensive_metrics(y_val, predictions)
            individual_metrics['model_id'] = i
            validation_results['individual_model_performance'].append(individual_metrics)
        
        # Evaluate ensemble
        if len(models) > 1:
            ensemble_pred = np.mean(all_predictions, axis=0)
            ensemble_metrics = self._calculate_comprehensive_metrics(y_val, ensemble_pred)
            validation_results['ensemble_performance'] = ensemble_metrics
        
        # Feature importance analysis
        if len(models) > 0:
            validation_results['feature_importance_analysis'] = (
                self._analyze_feature_importance(models)
            )
        
        # Best model selection
        best_ic = max([m['ic'] for m in validation_results['individual_model_performance']])
        validation_results['best_ic'] = best_ic
        validation_results['meets_ic_threshold'] = best_ic >= self.training_config.min_ic_threshold
        
        logger.info(f"Final validation completed - Best IC: {best_ic:.4f}")
        
        return validation_results
    
    def _calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['ic'] = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        metrics['correlation'] = pd.Series(y_true).corr(pd.Series(y_pred))
        metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # Handle NaN values
        for key in ['ic', 'correlation']:
            if pd.isna(metrics[key]):
                metrics[key] = 0.0
        
        # Quantile performance
        pred_quantiles = pd.qcut(y_pred, 5, labels=False)
        quantile_returns = []
        for q in range(5):
            mask = pred_quantiles == q
            if np.sum(mask) > 0:
                quantile_returns.append(y_true.iloc[mask].mean())
            else:
                quantile_returns.append(0)
        
        # Long-short performance (Q5 - Q1)
        metrics['long_short_return'] = quantile_returns[4] - quantile_returns[0]
        metrics['hit_rate'] = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        # Approximate Sharpe ratio
        if np.std(quantile_returns) > 0:
            metrics['sharpe_ratio'] = np.mean(quantile_returns) / np.std(quantile_returns)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        return metrics
    
    def _analyze_feature_importance(self, models: List[TaiwanMarketModel]) -> Dict[str, Any]:
        """Analyze feature importance across models."""
        importance_analysis = {
            'top_features': [],
            'stability_metrics': {},
            'category_analysis': {}
        }
        
        # Collect feature importance from all models
        all_importance = []
        for model in models:
            if model.base_model.feature_importance_ is not None:
                importance_df = model.base_model.feature_importance_.copy()
                importance_df['model_id'] = len(all_importance)
                all_importance.append(importance_df)
        
        if not all_importance:
            return importance_analysis
        
        # Combine importance across models
        combined_importance = pd.concat(all_importance)
        
        # Calculate average importance per feature
        avg_importance = combined_importance.groupby('feature')['importance'].agg([
            'mean', 'std', 'count'
        ]).sort_values('mean', ascending=False)
        
        # Top features
        importance_analysis['top_features'] = avg_importance.head(20).to_dict('index')
        
        # Stability metrics (coefficient of variation)
        avg_importance['cv'] = avg_importance['std'] / avg_importance['mean']
        importance_analysis['stability_metrics'] = {
            'mean_cv': avg_importance['cv'].mean(),
            'stable_features': len(avg_importance[avg_importance['cv'] < 0.5]),
            'total_features': len(avg_importance)
        }
        
        return importance_analysis
    
    def _save_training_results(
        self, 
        models: List[TaiwanMarketModel], 
        output_dir: Path
    ) -> None:
        """Save trained models and results."""
        # Save models
        model_dir = output_dir / 'models'
        model_dir.mkdir(exist_ok=True)
        
        for i, model in enumerate(models):
            model_path = model_dir / f'lightgbm_alpha_model_{i}.pkl'
            model.base_model.save_model(model_path)
        
        # Save feature pipeline
        pipeline_path = output_dir / 'feature_pipeline.pkl'
        joblib.dump(self.feature_pipeline, pipeline_path)
        
        # Save training history
        history_path = output_dir / 'training_history.pkl'
        joblib.dump(self.training_history, history_path)
        
        # Save validation results
        results_path = output_dir / 'validation_results.pkl'
        joblib.dump(self.validation_results, results_path)
        
        logger.info(f"Training results saved to {output_dir}")
    
    def _get_data_statistics(
        self, 
        train_data: Dict[str, pd.DataFrame],
        val_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        stats = {}
        
        for data_type in train_data.keys():
            train_df = train_data[data_type]
            val_df = val_data[data_type]
            
            stats[data_type] = {
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'train_date_range': [
                    train_df.index.get_level_values('date').min(),
                    train_df.index.get_level_values('date').max()
                ],
                'val_date_range': [
                    val_df.index.get_level_values('date').min(),
                    val_df.index.get_level_values('date').max()
                ],
                'unique_symbols': train_df.index.get_level_values('symbol').nunique(),
                'columns': list(train_df.columns)
            }
        
        return stats