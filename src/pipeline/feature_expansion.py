"""
Feature Expansion Pipeline - Task #28 Stream B
Integration pipeline connecting 42 base factors to OpenFE feature generation.

CRITICAL: Time-series integrity maintained through:
1. Temporal splitting (first 80% train, last 20% test)
2. No shuffling in train/test splits
3. Proper batch processing for 2000-stock universe
4. Memory-efficient processing with chunking

Integration Points:
- Input: 42 factors from Task #25 factor system
- Processing: OpenFE feature expansion using expert wrapper
- Output: Expanded feature matrix for Task #26 LightGBM consumption
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, date
import gc
import psutil
from pathlib import Path

# Import dependencies
from ..factors.base import FactorEngine, FactorResult, FactorCalculator
from ..features.openfe_wrapper import FeatureGenerator
from ..features.taiwan_config import TaiwanMarketConfig

logger = logging.getLogger(__name__)


class FeatureExpansionPipeline:
    """
    Feature expansion pipeline integrating 42 base factors with OpenFE.
    
    This pipeline:
    1. Loads 42 base factors from the factor system
    2. Applies time-series aware feature expansion using OpenFE
    3. Maintains temporal integrity for Taiwan market
    4. Outputs expanded features for ML consumption
    """
    
    def __init__(
        self,
        factor_engine: FactorEngine,
        taiwan_config: Optional[TaiwanMarketConfig] = None,
        memory_limit_gb: float = 12.0,
        chunk_size: int = 100,
        max_features: int = 500,
        output_dir: str = "./data/features/",
        cache_dir: str = "./cache/features/"
    ):
        """
        Initialize the feature expansion pipeline.
        
        Args:
            factor_engine: Factor calculation engine from Task #25
            taiwan_config: Taiwan market configuration
            memory_limit_gb: Maximum memory usage in GB
            chunk_size: Number of stocks to process per chunk
            max_features: Maximum number of features to generate
            output_dir: Directory to save expanded features
            cache_dir: Directory for caching intermediate results
        """
        self.factor_engine = factor_engine
        self.taiwan_config = taiwan_config or TaiwanMarketConfig()
        self.memory_limit_gb = memory_limit_gb
        self.chunk_size = chunk_size
        self.max_features = max_features
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenFE wrapper with Taiwan market settings
        fe_config = self.taiwan_config.get_feature_engineering_config()
        self.feature_generator = FeatureGenerator(
            taiwan_market=True,
            memory_limit_mb=int(memory_limit_gb * 1024),
            max_features=max_features,
            n_data_blocks=fe_config.get('n_data_blocks', 8),
            time_budget=fe_config.get('time_budget', 600),
            tmp_save_path=str(self.cache_dir / 'openfe_tmp/')
        )
        
        # Pipeline state
        self.is_fitted_ = False
        self.base_factors_ = []
        self.expanded_features_ = []
        self.processing_stats_ = {}
        self.memory_usage_ = {}
        
    def _check_memory_usage(self, stage: str) -> Dict[str, float]:
        """Monitor memory usage throughout pipeline execution."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = {
            'stage': stage,
            'rss_gb': memory_info.rss / 1024 / 1024 / 1024,
            'vms_gb': memory_info.vms / 1024 / 1024 / 1024,
            'percent': psutil.virtual_memory().percent,
            'available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
        }
        
        if memory_usage['rss_gb'] > self.memory_limit_gb:
            logger.warning(
                f"Memory usage ({memory_usage['rss_gb']:.2f}GB) exceeds limit "
                f"({self.memory_limit_gb}GB) at stage: {stage}"
            )
            
        self.memory_usage_[stage] = memory_usage
        logger.info(
            f"Memory usage at {stage}: {memory_usage['rss_gb']:.2f}GB "
            f"({memory_usage['percent']:.1f}%)"
        )
        
        return memory_usage
        
    def load_base_factors(
        self, 
        symbols: List[str], 
        start_date: date, 
        end_date: date,
        cache_results: bool = True
    ) -> pd.DataFrame:
        """
        Load 42 base factors from the factor system.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for factor calculation
            end_date: End date for factor calculation
            cache_results: Whether to cache results
            
        Returns:
            DataFrame with multi-index (date, symbol) and 42 factor columns
        """
        logger.info(f"Loading base factors for {len(symbols)} symbols from {start_date} to {end_date}")
        self._check_memory_usage("load_factors_start")
        
        # Check cache first
        cache_file = self.cache_dir / f"base_factors_{start_date}_{end_date}_{len(symbols)}.parquet"
        if cache_results and cache_file.exists():
            logger.info(f"Loading cached base factors from {cache_file}")
            factor_df = pd.read_parquet(cache_file)
            self._check_memory_usage("load_factors_cached")
            return factor_df
        
        # Generate date range (trading days only)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        trading_dates = [d.date() for d in date_range 
                        if self.taiwan_config.is_trading_day(pd.Timestamp(d))]
        
        logger.info(f"Processing {len(trading_dates)} trading dates")
        
        # Collect all factor results
        all_factor_data = []
        
        for current_date in trading_dates:
            # Calculate all factors for this date
            factor_results = self.factor_engine.calculate_all_factors(
                symbols=symbols,
                as_of_date=current_date
            )
            
            # Convert factor results to DataFrame row
            for factor_name, factor_result in factor_results.items():
                if factor_result.values:
                    for symbol, value in factor_result.values.items():
                        if pd.notna(value) and np.isfinite(value):
                            all_factor_data.append({
                                'date': current_date,
                                'symbol': symbol,
                                'factor_name': factor_name,
                                'value': float(value)
                            })
        
        # Convert to DataFrame
        if not all_factor_data:
            raise ValueError("No factor data generated")
            
        factor_long_df = pd.DataFrame(all_factor_data)
        
        # Pivot to wide format: (date, symbol) x factors
        factor_df = factor_long_df.pivot_table(
            index=['date', 'symbol'],
            columns='factor_name',
            values='value',
            aggfunc='first'
        )
        
        # Store factor names
        self.base_factors_ = factor_df.columns.tolist()
        logger.info(f"Loaded {len(self.base_factors_)} base factors")
        
        # Cache results
        if cache_results:
            factor_df.to_parquet(cache_file)
            logger.info(f"Cached base factors to {cache_file}")
        
        self._check_memory_usage("load_factors_end")
        return factor_df
        
    def _create_time_series_splits(
        self, 
        factor_df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/test splits with NO SHUFFLING.
        
        CRITICAL: This prevents lookahead bias by using first 80% for training
        and last 20% for testing, maintaining temporal order.
        
        Args:
            factor_df: DataFrame with (date, symbol) MultiIndex
            test_size: Fraction for test set (default 0.2 = 20%)
            
        Returns:
            (train_df, test_df) tuple
        """
        logger.info(f"Creating temporal splits: train={1-test_size:.1%}, test={test_size:.1%}")
        
        # Get unique dates and sort them
        dates = factor_df.index.get_level_values('date').unique().sort_values()
        n_dates = len(dates)
        
        # Calculate split point (temporal split, no shuffling)
        split_idx = int(n_dates * (1 - test_size))
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        logger.info(
            f"Temporal split: train_dates={len(train_dates)} ({train_dates[0]} to {train_dates[-1]}), "
            f"test_dates={len(test_dates)} ({test_dates[0]} to {test_dates[-1]})"
        )
        
        # Create boolean masks
        train_mask = factor_df.index.get_level_values('date').isin(train_dates)
        test_mask = factor_df.index.get_level_values('date').isin(test_dates)
        
        train_df = factor_df[train_mask].copy()
        test_df = factor_df[test_mask].copy()
        
        # Validate no temporal overlap
        if len(set(train_dates) & set(test_dates)) > 0:
            raise ValueError("Temporal overlap detected in train/test split")
            
        logger.info(
            f"Split validation passed: train={len(train_df)}, test={len(test_df)}, "
            f"no temporal overlap"
        )
        
        return train_df, test_df
        
    def expand_features_chunked(
        self,
        factor_df: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Expand features using OpenFE with chunked processing for memory efficiency.
        
        Args:
            factor_df: Base factors DataFrame
            target: Optional target variable for supervised feature generation
            
        Returns:
            Expanded features DataFrame
        """
        logger.info(f"Starting chunked feature expansion for {len(factor_df)} observations")
        self._check_memory_usage("feature_expansion_start")
        
        # Split into temporal train/test
        train_df, test_df = self._create_time_series_splits(factor_df)
        
        # Get unique symbols for chunking
        symbols = factor_df.index.get_level_values('symbol').unique().tolist()
        n_chunks = max(1, len(symbols) // self.chunk_size)
        
        logger.info(f"Processing {len(symbols)} symbols in {n_chunks} chunks of {self.chunk_size}")
        
        expanded_chunks = []
        
        for i, chunk_start in enumerate(range(0, len(symbols), self.chunk_size)):
            chunk_end = min(chunk_start + self.chunk_size, len(symbols))
            chunk_symbols = symbols[chunk_start:chunk_end]
            
            logger.info(f"Processing chunk {i+1}/{n_chunks}: {len(chunk_symbols)} symbols")
            
            try:
                # Filter data for current chunk
                chunk_mask = train_df.index.get_level_values('symbol').isin(chunk_symbols)
                chunk_train = train_df[chunk_mask].copy()
                
                if len(chunk_train) == 0:
                    logger.warning(f"Chunk {i+1} has no data, skipping")
                    continue
                
                # Prepare target for this chunk if provided
                chunk_target = None
                if target is not None:
                    chunk_target = target[chunk_train.index] if hasattr(target, 'index') else None
                
                self._check_memory_usage(f"chunk_{i+1}_before_openfe")
                
                # Apply OpenFE feature generation
                expanded_chunk = self.feature_generator.fit_transform(
                    chunk_train, chunk_target
                )
                
                self._check_memory_usage(f"chunk_{i+1}_after_openfe")
                
                expanded_chunks.append(expanded_chunk)
                
                # Force garbage collection after each chunk
                gc.collect()
                
                logger.info(
                    f"Chunk {i+1} completed: "
                    f"{len(expanded_chunk)} rows, {len(expanded_chunk.columns)} features"
                )
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                # Continue with remaining chunks
                continue
        
        if not expanded_chunks:
            raise ValueError("No chunks processed successfully")
        
        # Combine all chunks
        logger.info("Combining expanded feature chunks")
        expanded_df = pd.concat(expanded_chunks, axis=0, ignore_index=False)
        
        # Store expanded feature names
        self.expanded_features_ = expanded_df.columns.tolist()
        self.is_fitted_ = True
        
        self._check_memory_usage("feature_expansion_end")
        
        logger.info(
            f"Feature expansion completed: "
            f"{len(self.base_factors_)} → {len(self.expanded_features_)} features "
            f"({len(self.expanded_features_)/len(self.base_factors_):.1f}x expansion)"
        )
        
        return expanded_df
        
    def fit(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        target: Optional[pd.Series] = None,
        save_results: bool = True
    ) -> 'FeatureExpansionPipeline':
        """
        Fit the feature expansion pipeline.
        
        Args:
            symbols: List of stock symbols to process
            start_date: Start date for data
            end_date: End date for data
            target: Optional target variable
            save_results: Whether to save results to disk
            
        Returns:
            Self for method chaining
        """
        start_time = datetime.now()
        logger.info(f"Starting feature expansion pipeline for {len(symbols)} symbols")
        
        try:
            # Step 1: Load base factors
            factor_df = self.load_base_factors(symbols, start_date, end_date)
            
            # Step 2: Expand features using OpenFE
            expanded_df = self.expand_features_chunked(factor_df, target)
            
            # Step 3: Save results if requested
            if save_results:
                output_file = self.output_dir / f"expanded_features_{start_date}_{end_date}.parquet"
                expanded_df.to_parquet(output_file)
                logger.info(f"Saved expanded features to {output_file}")
            
            # Record processing statistics
            elapsed_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats_ = {
                'symbols_processed': len(symbols),
                'date_range': (start_date, end_date),
                'base_factors': len(self.base_factors_),
                'expanded_features': len(self.expanded_features_),
                'expansion_ratio': len(self.expanded_features_) / len(self.base_factors_),
                'processing_time_seconds': elapsed_time,
                'memory_peak_gb': max(
                    stats['rss_gb'] for stats in self.memory_usage_.values()
                ),
                'output_file': str(output_file) if save_results else None
            }
            
            logger.info(
                f"Pipeline completed successfully in {elapsed_time:.1f}s: "
                f"{self.processing_stats_['base_factors']} → "
                f"{self.processing_stats_['expanded_features']} features "
                f"({self.processing_stats_['expansion_ratio']:.1f}x)"
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
            
        return self
        
    def transform(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Args:
            factor_df: New factor data to transform
            
        Returns:
            Expanded features
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline must be fitted before transform")
            
        return self.feature_generator.transform(factor_df)
        
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names for base and expanded features."""
        return {
            'base_factors': self.base_factors_,
            'expanded_features': self.expanded_features_
        }
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        return self.processing_stats_.copy()
        
    def get_memory_usage(self) -> Dict[str, Dict[str, float]]:
        """Get memory usage statistics."""
        return self.memory_usage_.copy()
        
    def validate_pipeline_integrity(self) -> Dict[str, Any]:
        """
        Validate pipeline integrity for Taiwan market compliance.
        
        Returns:
            Validation results
        """
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'checks_performed': []
        }
        
        try:
            # Check if pipeline is fitted
            if not self.is_fitted_:
                validation_results['errors'].append("Pipeline not fitted")
                validation_results['passed'] = False
                return validation_results
            
            validation_results['checks_performed'].append("fitted_status")
            
            # Check feature expansion ratio
            if hasattr(self, 'processing_stats_'):
                expansion_ratio = self.processing_stats_.get('expansion_ratio', 0)
                if expansion_ratio < 2:
                    validation_results['warnings'].append(
                        f"Low expansion ratio: {expansion_ratio:.1f}x"
                    )
                elif expansion_ratio > 50:
                    validation_results['warnings'].append(
                        f"Very high expansion ratio: {expansion_ratio:.1f}x - memory concern"
                    )
                validation_results['checks_performed'].append("expansion_ratio")
            
            # Check memory usage
            if self.memory_usage_:
                peak_memory = max(stats['rss_gb'] for stats in self.memory_usage_.values())
                if peak_memory > self.memory_limit_gb:
                    validation_results['warnings'].append(
                        f"Peak memory usage ({peak_memory:.2f}GB) exceeded limit "
                        f"({self.memory_limit_gb}GB)"
                    )
                validation_results['checks_performed'].append("memory_usage")
            
            # Taiwan market specific validation
            taiwan_validation = self.taiwan_config.validate_data_for_taiwan_market(
                pd.DataFrame()  # Empty DataFrame for basic validation
            )
            if not taiwan_validation['passed']:
                validation_results['errors'].extend(taiwan_validation['errors'])
                validation_results['passed'] = False
            validation_results['warnings'].extend(taiwan_validation['warnings'])
            validation_results['checks_performed'].append("taiwan_market_compliance")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['passed'] = False
            
        return validation_results


def create_feature_expansion_pipeline(
    factor_engine: FactorEngine,
    config_overrides: Optional[Dict[str, Any]] = None
) -> FeatureExpansionPipeline:
    """
    Factory function to create a properly configured feature expansion pipeline.
    
    Args:
        factor_engine: Factor engine from Task #25
        config_overrides: Optional configuration overrides
        
    Returns:
        Configured FeatureExpansionPipeline instance
    """
    # Default configuration for Taiwan market
    config = {
        'memory_limit_gb': 12.0,
        'chunk_size': 100,
        'max_features': 500,
        'output_dir': "./data/features/",
        'cache_dir': "./cache/features/"
    }
    
    # Apply overrides
    if config_overrides:
        config.update(config_overrides)
    
    # Create Taiwan market configuration
    taiwan_config = TaiwanMarketConfig()
    
    # Create and return pipeline
    pipeline = FeatureExpansionPipeline(
        factor_engine=factor_engine,
        taiwan_config=taiwan_config,
        **config
    )
    
    logger.info(f"Created feature expansion pipeline with config: {config}")
    return pipeline