"""
Example: LightGBM Alpha Model Training for Taiwan Market

Demonstrates the complete training pipeline with memory optimization
for 2000-stock Taiwan market universe.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import LightGBMAlphaModel, FeaturePipeline, TaiwanMarketModel, MarketAdaptations
from models.training_pipeline import TrainingPipeline, TrainingConfig
from models.feature_pipeline import FeatureConfig
from models.lightgbm_alpha import ModelConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_taiwan_data(n_stocks=100, n_days=252*2) -> dict:
    """Create mock Taiwan market data for demonstration."""
    logger.info(f"Creating mock data: {n_stocks} stocks, {n_days} days")
    
    # Create date range (2 years)
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')  # Business days
    
    # Create stock symbols (Taiwan format: 4-digit codes)
    symbols = [f"{2300 + i:04d}.TW" for i in range(n_stocks)]  # Start from 2300 (similar to TSMC: 2330)
    
    # Create MultiIndex
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    # Generate price data with some structure
    np.random.seed(42)
    n_samples = len(index)
    
    # Price data (OHLCV)
    price_data = pd.DataFrame({
        'open': 50 + np.random.randn(n_samples) * 5 + np.cumsum(np.random.randn(n_samples) * 0.1),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.lognormal(15, 1, n_samples),  # Log-normal volume
        'returns': np.random.randn(n_samples) * 0.02  # 2% volatility
    }, index=index)
    
    # Ensure OHLC consistency
    price_data['high'] = price_data['open'] + np.abs(np.random.randn(n_samples) * 2)
    price_data['low'] = price_data['open'] - np.abs(np.random.randn(n_samples) * 2)
    price_data['close'] = price_data['open'] + np.random.randn(n_samples) * 1.5
    
    # Ensure high >= low and OHLC are consistent
    price_data['high'] = np.maximum(price_data[['open', 'close']].max(axis=1), price_data['high'])
    price_data['low'] = np.minimum(price_data[['open', 'close']].min(axis=1), price_data['low'])
    
    # Volume data
    volume_data = pd.DataFrame({
        'volume': price_data['volume'],
        'turnover': price_data['volume'] * price_data['close'],
        'vwap': price_data['close'] * (1 + np.random.randn(n_samples) * 0.001)  # Close to close price
    }, index=index)
    
    # Fundamental data (simplified)
    fundamental_data = pd.DataFrame({
        'market_cap': price_data['close'] * 100_000_000 + np.random.randn(n_samples) * 10_000_000,
        'pe_ratio': 15 + np.random.randn(n_samples) * 5,
        'pb_ratio': 2 + np.random.randn(n_samples) * 1,
        'roe': 0.15 + np.random.randn(n_samples) * 0.05,
        'debt_equity': 0.5 + np.random.randn(n_samples) * 0.2
    }, index=index)
    
    # Market-specific data
    market_data = pd.DataFrame({
        'foreign_ownership': 0.3 + np.random.randn(n_samples) * 0.1,
        'margin_trading_ratio': 0.1 + np.random.randn(n_samples) * 0.05,
        'sector': np.random.choice(['technology', 'financials', 'materials'], n_samples)
    }, index=index)
    
    return {
        'price_data': price_data,
        'volume_data': volume_data,
        'fundamental_data': fundamental_data,
        'market_data': market_data
    }


def main():
    """Main example function."""
    logger.info("Starting LightGBM Alpha Model Training Example")
    
    # 1. Create configurations
    logger.info("Setting up configurations")
    
    training_config = TrainingConfig(
        train_start_date='2022-01-01',
        train_end_date='2023-06-30',
        validation_start_date='2023-07-01', 
        validation_end_date='2023-12-31',
        target_horizon_days=5,
        cv_folds=3,  # Reduced for demo
        chunk_size=50,  # Small chunks for demo
        memory_limit_gb=4.0,  # Conservative for demo
        ensemble_size=1  # Single model for demo
    )
    
    model_config = ModelConfig(
        # Optimized for demo/small dataset
        base_params={
            'objective': 'regression',
            'boosting_type': 'gbdt', 
            'metric': 'rmse',
            'num_leaves': 20,  # Reduced for small dataset
            'max_depth': 5,
            'learning_rate': 0.1,  # Faster learning for demo
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        },
        n_estimators=100,  # Reduced for demo
        early_stopping_rounds=20
    )
    
    feature_config = FeatureConfig(
        max_features=20,  # Limited for demo
        scaler_type='robust',
        winsorize_quantiles=(0.02, 0.98),
        use_float32=True
    )
    
    # 2. Create mock data
    logger.info("Creating mock Taiwan market data")
    mock_data = create_mock_taiwan_data(n_stocks=50, n_days=500)  # Small demo dataset
    
    # 3. Initialize training pipeline
    logger.info("Initializing training pipeline")
    pipeline = TrainingPipeline(
        training_config=training_config,
        model_config=model_config,
        feature_config=feature_config
    )
    
    # 4. Run training
    logger.info("Running complete training pipeline")
    try:
        results = pipeline.run_complete_training(
            data_sources=mock_data,
            output_dir='./models/demo_output'
        )
        
        # 5. Display results
        logger.info("\n" + "="*50)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("="*50)
        logger.info(f"Training time: {results['training_time_hours']:.2f} hours")
        logger.info(f"Models trained: {results['models_trained']}")
        logger.info(f"Memory usage: {results['memory_usage_gb']:.2f} GB")
        
        # CV Results
        cv_results = results['cv_results']
        if cv_results['fold_metrics']:
            avg_ic = cv_results['average_metrics']['mean_ic']
            std_ic = cv_results['average_metrics']['std_ic']
            logger.info(f"Cross-validation IC: {avg_ic:.4f} ± {std_ic:.4f}")
        
        # Validation Results
        val_results = results['validation_results']
        best_ic = val_results.get('best_ic', 0)
        logger.info(f"Best validation IC: {best_ic:.4f}")
        logger.info(f"Meets IC threshold: {val_results.get('meets_ic_threshold', False)}")
        
        # Feature importance (top 10)
        feature_analysis = val_results.get('feature_importance_analysis', {})
        top_features = feature_analysis.get('top_features', {})
        if top_features:
            logger.info("\nTop 10 Features:")
            for i, (feature, stats) in enumerate(list(top_features.items())[:10]):
                logger.info(f"  {i+1:2d}. {feature}: {stats['mean']:.3f}")
        
        # Data statistics
        data_stats = results['data_stats']
        price_stats = data_stats.get('price_data', {})
        logger.info(f"\nData: {price_stats.get('train_samples', 0)} train, {price_stats.get('val_samples', 0)} val samples")
        logger.info(f"Symbols: {price_stats.get('unique_symbols', 0)}")
        
        logger.info(f"\nResults saved to: {results['output_directory']}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    # 6. Demonstrate prediction
    logger.info("\n" + "="*50)
    logger.info("PREDICTION DEMONSTRATION")
    logger.info("="*50)
    
    if pipeline.trained_models:
        model = pipeline.trained_models[0]  # Use first model
        
        # Create small prediction dataset
        pred_data = create_mock_taiwan_data(n_stocks=10, n_days=20)
        
        # Prepare features for prediction
        X_pred = pipeline.feature_pipeline.transform(
            pred_data['price_data'],
            pred_data['volume_data'],
            pred_data.get('fundamental_data'),
            pred_data.get('market_data')
        )
        
        # Generate predictions
        if len(X_pred) > 0:
            predictions = model.base_model.predict(X_pred)
            
            logger.info(f"Generated predictions for {len(predictions)} samples")
            logger.info(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
            logger.info(f"Prediction mean: {predictions.mean():.4f}")
            logger.info(f"Prediction std: {predictions.std():.4f}")
        else:
            logger.warning("No valid prediction data")
    
    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    main()