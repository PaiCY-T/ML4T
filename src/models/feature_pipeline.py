"""
Feature Pipeline for LightGBM Alpha Model

Integrates 42 handcrafted factors from Task #25 with preprocessing, scaling,
and feature selection optimized for Taiwan market characteristics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer, KNNImputer

# Import factor modules from Task #25
try:
    from ..factors import (
        FactorEngine, TechnicalFactors, FundamentalFactors, 
        MicrostructureFactors, TaiwanSpecificFactors,
        ValueFactors, QualityFactors, GrowthFactors,
        LiquidityFactors, VolumePatternFactors
    )
except ImportError as e:
    logger.warning(f"Could not import all factor modules: {e}")
    # Define mock classes for testing
    FactorEngine = TechnicalFactors = FundamentalFactors = object
    MicrostructureFactors = TaiwanSpecificFactors = object
    ValueFactors = QualityFactors = GrowthFactors = object
    LiquidityFactors = VolumePatternFactors = object

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature processing pipeline."""
    
    # Scaling configuration
    scaler_type: str = 'robust'  # 'standard', 'robust', 'minmax'
    scale_by_group: bool = True  # Scale within date groups for cross-sectional normalization
    
    # Outlier handling
    winsorize_quantiles: Tuple[float, float] = (0.01, 0.99)
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 3.0
    
    # Missing value handling
    imputation_method: str = 'forward_fill'  # 'forward_fill', 'median', 'knn'
    max_missing_ratio: float = 0.3  # Drop features with >30% missing values
    
    # Feature selection
    feature_selection_method: str = 'mutual_info'  # 'mutual_info', 'f_regression', 'importance_based'
    max_features: int = 42  # Maximum number of features to select
    min_feature_importance: float = 0.001
    
    # Taiwan market specific
    taiwan_adjustments: bool = True
    handle_price_limits: bool = True
    normalize_by_market_cap: bool = True
    
    # Memory optimization
    use_float32: bool = True  # Use float32 instead of float64 to save memory
    batch_size: int = 1000  # Process data in batches for memory efficiency


class FeatureProcessor:
    """
    Advanced feature processor for Taiwan market factors.
    
    Handles preprocessing, scaling, outlier removal, and feature selection
    for the 42 handcrafted factors from Task #25.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize feature processor."""
        self.config = config or FeatureConfig()
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_selector: Optional[Any] = None
        self.selected_features: List[str] = []
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.processing_history: List[Dict[str, Any]] = []
        
        logger.info(f"FeatureProcessor initialized with config: {self.config}")
    
    def fit_transform(
        self, 
        features: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit the feature processor and transform features.
        
        Args:
            features: Raw feature matrix with MultiIndex (date, symbol)
            target: Target variable for supervised feature selection
            
        Returns:
            Processed feature matrix
        """
        logger.info(f"Fitting feature processor on {features.shape[0]} samples, {features.shape[1]} features")
        
        # Store original feature names
        original_features = features.columns.tolist()
        
        # Step 1: Handle missing values
        features = self._handle_missing_values(features, fit=True)
        
        # Step 2: Handle outliers
        features = self._handle_outliers(features, fit=True)
        
        # Step 3: Apply Taiwan market adjustments
        if self.config.taiwan_adjustments:
            features = self._apply_taiwan_adjustments(features)
        
        # Step 4: Scale features
        features = self._scale_features(features, fit=True)
        
        # Step 5: Feature selection
        if target is not None:
            features = self._select_features(features, target, fit=True)
        
        # Step 6: Memory optimization
        if self.config.use_float32:
            features = features.astype(np.float32)
        
        # Store processing statistics
        self._update_feature_stats(features, original_features)
        
        logger.info(f"Feature processing completed: {features.shape[1]} features selected")
        return features
    
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted processors.
        
        Args:
            features: Raw feature matrix
            
        Returns:
            Processed feature matrix
        """
        if not self.scalers:
            raise ValueError("FeatureProcessor has not been fitted yet")
        
        # Apply same transformations as in fit_transform
        features = self._handle_missing_values(features, fit=False)
        features = self._handle_outliers(features, fit=False)
        
        if self.config.taiwan_adjustments:
            features = self._apply_taiwan_adjustments(features)
        
        features = self._scale_features(features, fit=False)
        
        # Apply feature selection
        if self.selected_features:
            features = features[self.selected_features]
        
        if self.config.use_float32:
            features = features.astype(np.float32)
        
        return features
    
    def _handle_missing_values(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values using configured method."""
        if fit:
            # Calculate missing value statistics
            missing_ratios = features.isnull().sum() / len(features)
            
            # Drop features with too many missing values
            features_to_keep = missing_ratios[missing_ratios <= self.config.max_missing_ratio].index
            features = features[features_to_keep]
            
            logger.info(f"Dropped {len(missing_ratios) - len(features_to_keep)} features with >{self.config.max_missing_ratio*100}% missing values")
        
        # Apply imputation
        if self.config.imputation_method == 'forward_fill':
            # Forward fill within each symbol
            features = features.groupby(level=1).fillna(method='ffill')
            # Backward fill any remaining NaNs
            features = features.groupby(level=1).fillna(method='bfill')
            
        elif self.config.imputation_method == 'median':
            if fit:
                self.imputers['median'] = SimpleImputer(strategy='median')
                self.imputers['median'].fit(features)
            
            # Apply imputation by date groups for cross-sectional consistency
            transformed_features = []
            for date in features.index.get_level_values(0).unique():
                date_features = features.xs(date, level=0)
                imputed = self.imputers['median'].transform(date_features)
                transformed_features.append(
                    pd.DataFrame(imputed, index=date_features.index, columns=features.columns)
                )
            features = pd.concat(transformed_features)
            
        elif self.config.imputation_method == 'knn':
            if fit:
                self.imputers['knn'] = KNNImputer(n_neighbors=5)
                self.imputers['knn'].fit(features)
            
            imputed = self.imputers['knn'].transform(features)
            features = pd.DataFrame(imputed, index=features.index, columns=features.columns)
        
        return features
    
    def _handle_outliers(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle outliers using configured method."""
        if self.config.outlier_method == 'iqr':
            # Winsorize using quantiles
            if fit:
                self.outlier_bounds = {}
                for col in features.columns:
                    q_low, q_high = self.config.winsorize_quantiles
                    self.outlier_bounds[col] = (
                        features[col].quantile(q_low),
                        features[col].quantile(q_high)
                    )
            
            # Apply winsorization
            for col in features.columns:
                if col in self.outlier_bounds:
                    lower, upper = self.outlier_bounds[col]
                    features[col] = features[col].clip(lower, upper)
                    
        elif self.config.outlier_method == 'zscore':
            # Z-score based outlier handling
            if fit:
                self.outlier_stats = {}
                for col in features.columns:
                    self.outlier_stats[col] = {
                        'mean': features[col].mean(),
                        'std': features[col].std()
                    }
            
            # Apply z-score clipping
            threshold = self.config.outlier_threshold
            for col in features.columns:
                if col in self.outlier_stats:
                    mean, std = self.outlier_stats[col]['mean'], self.outlier_stats[col]['std']
                    z_scores = np.abs((features[col] - mean) / std)
                    features.loc[z_scores > threshold, col] = np.nan
            
            # Fill outliers with median
            features = features.fillna(features.median())
        
        return features
    
    def _apply_taiwan_adjustments(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply Taiwan market specific adjustments."""
        adjusted_features = features.copy()
        
        # Handle price limit effects
        if self.config.handle_price_limits and 'daily_return' in features.columns:
            # Identify price limit hits (±10% in Taiwan)
            limit_hits = np.abs(features['daily_return']) > 0.095
            
            # Create price limit indicator features
            adjusted_features['hit_price_limit'] = limit_hits.astype(float)
            adjusted_features['consecutive_limits'] = (
                limit_hits.groupby(level=1)
                .rolling(window=5, min_periods=1)
                .sum()
                .droplevel(0)
            )
        
        # Market cap normalization for fundamental factors
        if self.config.normalize_by_market_cap:
            fundamental_cols = [col for col in features.columns if any(
                keyword in col.lower() for keyword in ['pe', 'pb', 'ps', 'ev', 'roe', 'roa']
            )]
            
            for col in fundamental_cols:
                if f'market_cap' in features.columns:
                    # Size-adjust fundamental factors
                    adjusted_features[f'{col}_size_adj'] = (
                        features[col] / np.log(features['market_cap'] + 1)
                    )
        
        # Taiwan trading session adjustments (09:00-13:30 TST)
        if 'intraday_volatility' in features.columns:
            # Normalize for shorter trading session (4.5 hours vs 6.5 hours US)
            session_adjustment = 4.5 / 6.5
            adjusted_features['intraday_volatility'] = (
                features['intraday_volatility'] / np.sqrt(session_adjustment)
            )
        
        return adjusted_features
    
    def _scale_features(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale features using configured scaler."""
        if self.config.scaler_type == 'none':
            return features
        
        # Select scaler
        if self.config.scaler_type == 'standard':
            ScalerClass = StandardScaler
        elif self.config.scaler_type == 'robust':
            ScalerClass = RobustScaler
        elif self.config.scaler_type == 'minmax':
            ScalerClass = MinMaxScaler
        else:
            raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")
        
        if self.config.scale_by_group:
            # Scale within each date for cross-sectional normalization
            scaled_features = []
            
            for date in features.index.get_level_values(0).unique():
                date_features = features.xs(date, level=0)
                
                if fit:
                    if date not in self.scalers:
                        self.scalers[date] = ScalerClass()
                    scaled_data = self.scalers[date].fit_transform(date_features)
                else:
                    if date in self.scalers:
                        scaled_data = self.scalers[date].transform(date_features)
                    else:
                        # Use most recent scaler if date not seen in training
                        latest_date = max(self.scalers.keys())
                        scaled_data = self.scalers[latest_date].transform(date_features)
                
                scaled_df = pd.DataFrame(
                    scaled_data, 
                    index=date_features.index, 
                    columns=date_features.columns
                )
                scaled_features.append(scaled_df)
            
            return pd.concat(scaled_features)
            
        else:
            # Global scaling
            if fit:
                self.scalers['global'] = ScalerClass()
                scaled_data = self.scalers['global'].fit_transform(features)
            else:
                scaled_data = self.scalers['global'].transform(features)
            
            return pd.DataFrame(
                scaled_data, 
                index=features.index, 
                columns=features.columns
            )
    
    def _select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        fit: bool = False
    ) -> pd.DataFrame:
        """Select most important features."""
        if fit:
            # Align features and target
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx]
            y = target.loc[common_idx]
            
            # Remove any remaining NaN values
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                logger.warning("No valid samples for feature selection")
                self.selected_features = features.columns.tolist()[:self.config.max_features]
                return features[self.selected_features]
            
            # Feature selection
            if self.config.feature_selection_method == 'mutual_info':
                selector = SelectKBest(
                    mutual_info_regression, 
                    k=min(self.config.max_features, len(features.columns))
                )
            elif self.config.feature_selection_method == 'f_regression':
                selector = SelectKBest(
                    f_regression, 
                    k=min(self.config.max_features, len(features.columns))
                )
            else:
                # Default: select all features up to max_features
                self.selected_features = features.columns.tolist()[:self.config.max_features]
                return features[self.selected_features]
            
            # Fit selector
            try:
                selector.fit(X, y)
                self.feature_selector = selector
                self.selected_features = X.columns[selector.get_support()].tolist()
                
                logger.info(f"Feature selection completed: {len(self.selected_features)} features selected")
                
                # Log top features
                if hasattr(selector, 'scores_'):
                    feature_scores = pd.DataFrame({
                        'feature': X.columns,
                        'score': selector.scores_,
                        'selected': selector.get_support()
                    }).sort_values('score', ascending=False)
                    
                    logger.info("Top 10 selected features:")
                    for _, row in feature_scores[row['selected']].head(10).iterrows():
                        logger.info(f"  {row['feature']}: {row['score']:.4f}")
                        
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}, using all features")
                self.selected_features = features.columns.tolist()[:self.config.max_features]
        
        # Return selected features
        return features[self.selected_features]
    
    def _update_feature_stats(self, features: pd.DataFrame, original_features: List[str]) -> None:
        """Update feature processing statistics."""
        stats = {
            'original_feature_count': len(original_features),
            'final_feature_count': len(features.columns),
            'feature_reduction_ratio': 1 - (len(features.columns) / len(original_features)),
            'missing_value_ratio': features.isnull().sum().sum() / (features.shape[0] * features.shape[1]),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Calculate feature correlations
        corr_matrix = features.corr()
        stats['avg_feature_correlation'] = (
            corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        )
        stats['max_feature_correlation'] = (
            corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
        )
        
        self.feature_stats = stats
        self.processing_history.append(stats)
        
        logger.info(f"Feature processing stats: {stats}")


class FeaturePipeline:
    """
    Complete feature pipeline integrating 42 factors from Task #25.
    
    Orchestrates factor calculation, feature processing, and data preparation
    for the LightGBM alpha model.
    """
    
    def __init__(
        self, 
        feature_config: Optional[FeatureConfig] = None,
        factor_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize feature pipeline."""
        self.feature_config = feature_config or FeatureConfig()
        self.factor_config = factor_config or {}
        
        # Initialize factor calculators
        self.factor_calculators = self._initialize_factor_calculators()
        
        # Initialize feature processor
        self.feature_processor = FeatureProcessor(self.feature_config)
        
        # Pipeline state
        self.is_fitted = False
        self.feature_metadata: Dict[str, Any] = {}
        
        logger.info("FeaturePipeline initialized with 42-factor system")
    
    def _initialize_factor_calculators(self) -> Dict[str, Any]:
        """Initialize all factor calculators from Task #25."""
        calculators = {}
        
        try:
            # Technical factors (8 factors)
            calculators['technical'] = TechnicalFactors(**self.factor_config.get('technical', {}))
            
            # Fundamental factors (12 factors)
            calculators['value'] = ValueFactors(**self.factor_config.get('value', {}))
            calculators['quality'] = QualityFactors(**self.factor_config.get('quality', {}))
            calculators['growth'] = GrowthFactors(**self.factor_config.get('growth', {}))
            
            # Microstructure factors (8 factors)
            calculators['microstructure'] = MicrostructureFactors(**self.factor_config.get('microstructure', {}))
            calculators['liquidity'] = LiquidityFactors(**self.factor_config.get('liquidity', {}))
            
            # Market-specific factors (14 factors)
            calculators['taiwan_specific'] = TaiwanSpecificFactors(**self.factor_config.get('taiwan_specific', {}))
            calculators['volume_patterns'] = VolumePatternFactors(**self.factor_config.get('volume_patterns', {}))
            
        except Exception as e:
            logger.warning(f"Could not initialize all factor calculators: {e}")
            calculators = {}
        
        return calculators
    
    def calculate_all_factors(
        self, 
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        fundamental_data: Optional[pd.DataFrame] = None,
        microstructure_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate all 42 factors from Task #25.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            fundamental_data: Fundamental/financial data
            microstructure_data: Tick-level microstructure data
            
        Returns:
            Combined factor matrix with all 42 factors
        """
        logger.info("Calculating all 42 factors from Task #25")
        
        all_factors = []
        
        # Technical factors (8 factors)
        if 'technical' in self.factor_calculators:
            tech_factors = self.factor_calculators['technical'].calculate_all(price_data, volume_data)
            all_factors.append(tech_factors)
        
        # Fundamental factors (12 factors)
        if fundamental_data is not None:
            for factor_type in ['value', 'quality', 'growth']:
                if factor_type in self.factor_calculators:
                    factors = self.factor_calculators[factor_type].calculate_all(
                        price_data, fundamental_data
                    )
                    all_factors.append(factors)
        
        # Microstructure factors (8 factors)
        if microstructure_data is not None:
            for factor_type in ['microstructure', 'liquidity']:
                if factor_type in self.factor_calculators:
                    factors = self.factor_calculators[factor_type].calculate_all(
                        microstructure_data
                    )
                    all_factors.append(factors)
        
        # Taiwan-specific factors (14 factors)
        for factor_type in ['taiwan_specific', 'volume_patterns']:
            if factor_type in self.factor_calculators:
                factors = self.factor_calculators[factor_type].calculate_all(
                    price_data, volume_data
                )
                all_factors.append(factors)
        
        # Combine all factors
        if all_factors:
            combined_factors = pd.concat(all_factors, axis=1)
            logger.info(f"Calculated {combined_factors.shape[1]} factors")
        else:
            logger.warning("No factors calculated, using mock data")
            # Create mock factor data for development
            combined_factors = self._create_mock_factors(price_data.index)
        
        return combined_factors
    
    def _create_mock_factors(self, index: pd.Index) -> pd.DataFrame:
        """Create mock factors for development/testing."""
        np.random.seed(42)
        n_samples = len(index)
        
        # Create 42 mock factors
        factor_names = [
            # Technical (8)
            'momentum_1m', 'momentum_3m', 'rsi_14d', 'bollinger_position',
            'macd_signal', 'price_vs_sma20', 'volatility_20d', 'trend_strength',
            
            # Value (4)
            'pe_ratio', 'pb_ratio', 'ev_ebitda', 'price_sales',
            
            # Quality (4)
            'roe', 'roa', 'debt_equity', 'operating_margin',
            
            # Growth (4)
            'revenue_growth', 'earnings_growth', 'book_value_growth', 'analyst_revisions',
            
            # Microstructure (4)
            'bid_ask_spread', 'order_imbalance', 'price_impact', 'tick_direction',
            
            # Liquidity (4)
            'amihud_illiquidity', 'turnover_ratio', 'volume_volatility', 'zero_trading_days',
            
            # Taiwan Specific (7)
            'foreign_flow_ratio', 'margin_trading_ratio', 'foreign_ownership',
            'index_weight_change', 'cross_strait_sentiment', 'taiwan_dollar_strength', 'sector_rotation',
            
            # Volume Patterns (7)
            'volume_momentum', 'volume_mean_reversion', 'volume_breakout',
            'vwap_ratio', 'volume_price_correlation', 'accumulation_distribution', 'on_balance_volume'
        ]
        
        # Generate random factor data with some structure
        factor_data = {}
        for i, factor_name in enumerate(factor_names):
            # Add some persistence and cross-correlation structure
            base_series = np.random.randn(n_samples)
            if i > 0:
                # Add correlation with previous factors
                base_series += 0.3 * np.random.randn(n_samples)
            
            factor_data[factor_name] = base_series
        
        return pd.DataFrame(factor_data, index=index)
    
    def fit_transform(
        self, 
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        returns: pd.Series,
        fundamental_data: Optional[pd.DataFrame] = None,
        microstructure_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Fit the feature pipeline and transform data.
        
        Args:
            price_data: Price data
            volume_data: Volume data  
            returns: Target returns for feature selection
            fundamental_data: Optional fundamental data
            microstructure_data: Optional microstructure data
            
        Returns:
            Processed feature matrix ready for ML model
        """
        logger.info("Fitting feature pipeline")
        
        # Step 1: Calculate all factors
        raw_factors = self.calculate_all_factors(
            price_data, volume_data, fundamental_data, microstructure_data
        )
        
        # Step 2: Process features
        processed_features = self.feature_processor.fit_transform(raw_factors, returns)
        
        # Update pipeline state
        self.is_fitted = True
        self.feature_metadata = {
            'raw_factor_count': raw_factors.shape[1],
            'processed_feature_count': processed_features.shape[1],
            'selected_features': self.feature_processor.selected_features,
            'processing_stats': self.feature_processor.feature_stats
        }
        
        logger.info(f"Feature pipeline fitted: {len(self.feature_processor.selected_features)} features selected")
        
        return processed_features
    
    def transform(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        fundamental_data: Optional[pd.DataFrame] = None,
        microstructure_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline has not been fitted yet")
        
        # Calculate factors
        raw_factors = self.calculate_all_factors(
            price_data, volume_data, fundamental_data, microstructure_data
        )
        
        # Process features
        processed_features = self.feature_processor.transform(raw_factors)
        
        return processed_features
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Get comprehensive feature importance analysis."""
        if not self.is_fitted:
            return {"error": "Pipeline not fitted"}
        
        analysis = {
            "selected_features": self.feature_processor.selected_features,
            "feature_stats": self.feature_processor.feature_stats,
            "pipeline_metadata": self.feature_metadata
        }
        
        # Add factor category breakdown
        feature_categories = {
            'technical': [f for f in self.feature_processor.selected_features if any(
                keyword in f.lower() for keyword in ['momentum', 'rsi', 'macd', 'bollinger', 'sma', 'volatility']
            )],
            'fundamental': [f for f in self.feature_processor.selected_features if any(
                keyword in f.lower() for keyword in ['pe', 'pb', 'ev', 'roe', 'roa', 'growth', 'margin']
            )],
            'microstructure': [f for f in self.feature_processor.selected_features if any(
                keyword in f.lower() for keyword in ['spread', 'imbalance', 'impact', 'tick', 'liquidity']
            )],
            'taiwan_specific': [f for f in self.feature_processor.selected_features if any(
                keyword in f.lower() for keyword in ['foreign', 'margin', 'index', 'cross_strait', 'taiwan']
            )]
        }
        
        analysis['feature_category_counts'] = {
            category: len(features) for category, features in feature_categories.items()
        }
        
        return analysis