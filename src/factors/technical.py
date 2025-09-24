"""
Technical factors orchestrator for Taiwan market ML pipeline.

This module provides the main TechnicalFactors class that coordinates
all technical factor calculations including momentum, mean reversion,
and volatility factors.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Set
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .base import FactorEngine, FactorResult, FactorCategory, FactorMetadata
from .taiwan_adjustments import TaiwanMarketAdjustments
from .momentum import (
    PriceMomentumCalculator, 
    RSIMomentumCalculator, 
    MACDSignalCalculator
)
from .mean_reversion import (
    MovingAverageReversionCalculator,
    BollingerBandPositionCalculator,
    ZScoreReversionCalculator,
    ShortTermReversalCalculator
)
from .volatility import (
    RealizedVolatilityCalculator,
    GARCHVolatilityCalculator,
    TaiwanVIXCalculator,
    VolatilityRiskPremiumCalculator
)

from ..data.pipeline.pit_engine import PITQueryEngine

logger = logging.getLogger(__name__)


class TechnicalFactors:
    """
    Main orchestrator for technical factor calculations.
    
    This class manages all 18 technical factors across momentum,
    mean reversion, and volatility categories.
    """
    
    def __init__(self, pit_engine: PITQueryEngine, 
                 taiwan_adjustments: Optional[TaiwanMarketAdjustments] = None):
        """
        Initialize technical factors calculator.
        
        Args:
            pit_engine: Point-in-time query engine
            taiwan_adjustments: Taiwan market adjustments (will create if None)
        """
        self.pit_engine = pit_engine
        self.taiwan_adj = taiwan_adjustments or TaiwanMarketAdjustments()
        self.logger = logging.getLogger(__name__)
        
        # Initialize factor engine
        self.factor_engine = FactorEngine(pit_engine)
        
        # Register all technical factor calculators
        self._register_all_calculators()
        
        # Performance tracking
        self.calculation_stats = {}
    
    def _register_all_calculators(self):
        """Register all technical factor calculators."""
        
        # Momentum factors (6 factors)
        momentum_calculators = [
            PriceMomentumCalculator(self.pit_engine, self.taiwan_adj),
            RSIMomentumCalculator(self.pit_engine, self.taiwan_adj),
            MACDSignalCalculator(self.pit_engine, self.taiwan_adj),
        ]
        
        # Mean reversion factors (6 factors) 
        mean_reversion_calculators = [
            MovingAverageReversionCalculator(self.pit_engine, self.taiwan_adj),
            BollingerBandPositionCalculator(self.pit_engine, self.taiwan_adj),
            ZScoreReversionCalculator(self.pit_engine, self.taiwan_adj),
            ShortTermReversalCalculator(self.pit_engine, self.taiwan_adj),
        ]
        
        # Volatility factors (6 factors)
        volatility_calculators = [
            RealizedVolatilityCalculator(self.pit_engine, self.taiwan_adj),
            GARCHVolatilityCalculator(self.pit_engine, self.taiwan_adj),
            TaiwanVIXCalculator(self.pit_engine, self.taiwan_adj),
            VolatilityRiskPremiumCalculator(self.pit_engine, self.taiwan_adj),
        ]
        
        # Register all calculators
        all_calculators = momentum_calculators + mean_reversion_calculators + volatility_calculators
        
        for calculator in all_calculators:
            self.factor_engine.register_calculator(calculator)
            
        self.logger.info(f"Registered {len(all_calculators)} technical factor calculators")
    
    def calculate_all_factors(self, symbols: List[str], as_of_date: date,
                            parallel: bool = True, 
                            max_workers: Optional[int] = None) -> Dict[str, FactorResult]:
        """
        Calculate all technical factors for given symbols and date.
        
        Args:
            symbols: List of stock symbols
            as_of_date: Calculation date
            parallel: Whether to calculate factors in parallel
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary mapping factor names to results
        """
        start_time = time.time()
        
        self.logger.info(
            f"Starting technical factor calculation for {len(symbols)} symbols "
            f"as of {as_of_date}"
        )
        
        # Validate requirements
        validation_results = self.factor_engine.validate_factor_requirements(as_of_date)
        failed_factors = [name for name, valid in validation_results.items() if not valid]
        
        if failed_factors:
            self.logger.warning(f"Validation failed for factors: {failed_factors}")
        
        # Calculate factors
        if parallel and len(symbols) > 10:  # Use parallel for larger universes
            results = self._calculate_parallel(symbols, as_of_date, max_workers)
        else:
            results = self.factor_engine.calculate_all_factors(
                symbols, as_of_date, categories=[FactorCategory.TECHNICAL]
            )
        
        # Update statistics
        elapsed_time = time.time() - start_time
        self._update_calculation_stats(symbols, as_of_date, results, elapsed_time)
        
        self.logger.info(
            f"Technical factor calculation completed in {elapsed_time:.2f}s. "
            f"Generated {len(results)} factors with average coverage "
            f"{np.mean([r.coverage for r in results.values() if r.coverage]):.1%}"
        )
        
        return results
    
    def _calculate_parallel(self, symbols: List[str], as_of_date: date,
                          max_workers: Optional[int] = None) -> Dict[str, FactorResult]:
        """Calculate factors in parallel for better performance."""
        
        if max_workers is None:
            max_workers = min(4, len(self.factor_engine.calculators))
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all factor calculations
            future_to_factor = {}
            
            for factor_name, calculator in self.factor_engine.calculators.items():
                future = executor.submit(calculator.calculate, symbols, as_of_date)
                future_to_factor[future] = factor_name
            
            # Collect results
            for future in as_completed(future_to_factor):
                factor_name = future_to_factor[future]
                
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per factor
                    result.calculation_time = datetime.now()
                    results[factor_name] = result
                    
                    self.logger.debug(
                        f"Completed {factor_name}: {result.coverage:.1%} coverage"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error calculating factor {factor_name}: {e}")
                    continue
        
        return results
    
    def calculate_factor_subset(self, factor_names: List[str], symbols: List[str], 
                              as_of_date: date) -> Dict[str, FactorResult]:
        """
        Calculate specific subset of factors.
        
        Args:
            factor_names: List of factor names to calculate
            symbols: List of symbols
            as_of_date: Calculation date
            
        Returns:
            Dictionary mapping factor names to results
        """
        results = {}
        
        for factor_name in factor_names:
            if factor_name not in self.factor_engine.calculators:
                self.logger.warning(f"Factor {factor_name} not found")
                continue
            
            try:
                result = self.factor_engine.calculate_factor(
                    factor_name, symbols, as_of_date
                )
                results[factor_name] = result
                
            except Exception as e:
                self.logger.error(f"Error calculating {factor_name}: {e}")
                continue
        
        return results
    
    def get_factor_categories(self) -> Dict[str, List[str]]:
        """Get factors organized by category."""
        
        categories = {
            'momentum': [],
            'mean_reversion': [],  
            'volatility': []
        }
        
        for factor_name, calculator in self.factor_engine.calculators.items():
            metadata = calculator.metadata
            
            if 'momentum' in factor_name or 'rsi' in factor_name or 'macd' in factor_name:
                categories['momentum'].append(factor_name)
            elif 'reversion' in factor_name or 'zscore' in factor_name or 'bb_' in factor_name or 'reversal' in factor_name:
                categories['mean_reversion'].append(factor_name)
            elif 'vol' in factor_name or 'garch' in factor_name or 'vix' in factor_name:
                categories['volatility'].append(factor_name)
        
        return categories
    
    def get_factor_metadata(self) -> Dict[str, FactorMetadata]:
        """Get metadata for all technical factors."""
        return self.factor_engine.get_factor_metadata()
    
    def validate_factor_universe_coverage(self, symbols: List[str], 
                                        as_of_date: date, 
                                        min_coverage: float = 0.8) -> Dict[str, bool]:
        """
        Validate that factors have sufficient universe coverage.
        
        Args:
            symbols: Universe of symbols
            as_of_date: Date to validate
            min_coverage: Minimum required coverage
            
        Returns:
            Dictionary mapping factor names to validation results
        """
        
        # Sample a subset of symbols for validation
        sample_size = min(50, len(symbols))  
        sample_symbols = symbols[:sample_size]
        
        validation_results = {}
        
        for factor_name in self.factor_engine.calculators.keys():
            try:
                result = self.factor_engine.calculate_factor(
                    factor_name, sample_symbols, as_of_date
                )
                
                coverage = result.coverage or 0.0
                validation_results[factor_name] = coverage >= min_coverage
                
                if coverage < min_coverage:
                    self.logger.warning(
                        f"Factor {factor_name} has low coverage: {coverage:.1%} < {min_coverage:.1%}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Validation failed for {factor_name}: {e}")
                validation_results[factor_name] = False
        
        return validation_results
    
    def calculate_factor_correlations(self, symbols: List[str], 
                                    as_of_date: date,
                                    lookback_days: int = 60) -> pd.DataFrame:
        """
        Calculate correlation matrix between factors.
        
        Args:
            symbols: List of symbols
            as_of_date: End date
            lookback_days: Number of days to look back
            
        Returns:
            Factor correlation matrix
        """
        
        # Calculate factors for multiple dates
        dates = pd.bdate_range(
            end=as_of_date, 
            periods=min(lookback_days, 60)  # Limit to avoid excessive computation
        )
        
        factor_data = {}
        
        for date in dates[-30:]:  # Use last 30 business days
            try:
                results = self.calculate_all_factors(symbols, date.date(), parallel=False)
                
                for factor_name, result in results.items():
                    if factor_name not in factor_data:
                        factor_data[factor_name] = []
                    
                    # Use cross-sectional rank correlation
                    if result.percentile_ranks:
                        avg_percentile = np.mean(list(result.percentile_ranks.values()))
                        factor_data[factor_name].append(avg_percentile)
                    else:
                        factor_data[factor_name].append(np.nan)
                        
            except Exception as e:
                self.logger.debug(f"Error calculating factors for {date}: {e}")
                continue
        
        # Create correlation matrix
        if factor_data:
            factor_df = pd.DataFrame(factor_data)
            correlation_matrix = factor_df.corr()
            return correlation_matrix
        else:
            return pd.DataFrame()
    
    def _update_calculation_stats(self, symbols: List[str], as_of_date: date,
                                results: Dict[str, FactorResult], 
                                elapsed_time: float):
        """Update calculation statistics."""
        
        stats = {
            'date': as_of_date,
            'universe_size': len(symbols),
            'factors_calculated': len(results),
            'elapsed_time_seconds': elapsed_time,
            'average_coverage': np.mean([r.coverage for r in results.values() if r.coverage]),
            'factors_with_full_coverage': sum(1 for r in results.values() if r.coverage and r.coverage > 0.95),
            'timestamp': datetime.now()
        }
        
        self.calculation_stats[as_of_date] = stats
    
    def get_calculation_stats(self) -> Dict[date, Dict[str, Any]]:
        """Get calculation performance statistics."""
        return self.calculation_stats
    
    def export_factor_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Export factor definitions for documentation."""
        
        definitions = {}
        
        for factor_name, calculator in self.factor_engine.calculators.items():
            metadata = calculator.metadata
            
            definitions[factor_name] = {
                'name': metadata.name,
                'category': metadata.category.value,
                'frequency': metadata.frequency.value,
                'description': metadata.description,
                'lookback_days': metadata.lookback_days,
                'min_history_days': metadata.min_history_days,
                'data_requirements': [dt.value for dt in metadata.data_requirements],
                'taiwan_specific': metadata.taiwan_specific,
                'expected_ic': metadata.expected_ic,
                'expected_turnover': metadata.expected_turnover
            }
        
        return definitions
    
    def cleanup_resources(self):
        """Cleanup resources and close connections."""
        # Close any open connections or cleanup resources
        if hasattr(self.pit_engine, 'cleanup'):
            self.pit_engine.cleanup()
        
        self.logger.info("Technical factors resources cleaned up")


# Convenience functions for direct access
def calculate_technical_factors(pit_engine: PITQueryEngine, 
                              symbols: List[str], 
                              as_of_date: date,
                              taiwan_adjustments: Optional[TaiwanMarketAdjustments] = None,
                              **kwargs) -> Dict[str, FactorResult]:
    """
    Convenience function to calculate all technical factors.
    
    Args:
        pit_engine: Point-in-time query engine
        symbols: List of symbols
        as_of_date: Calculation date
        taiwan_adjustments: Optional Taiwan market adjustments
        **kwargs: Additional arguments for calculation
        
    Returns:
        Dictionary of factor results
    """
    
    tech_factors = TechnicalFactors(pit_engine, taiwan_adjustments)
    return tech_factors.calculate_all_factors(symbols, as_of_date, **kwargs)


def get_technical_factor_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all technical factors without initializing engines.
    
    Returns:
        Dictionary of factor metadata
    """
    
    # This would require a lightweight way to get metadata
    # For now, return static metadata
    metadata = {
        'price_momentum': {
            'description': 'Multi-period price momentum (1M, 3M, 6M, 12M returns)',
            'category': 'momentum',
            'expected_ic': 0.05,
            'taiwan_specific': True
        },
        'rsi_momentum': {
            'description': 'RSI-based momentum with trend strength',
            'category': 'momentum', 
            'expected_ic': 0.03,
            'taiwan_specific': True
        },
        'macd_signal': {
            'description': 'MACD histogram and signal line strength',
            'category': 'momentum',
            'expected_ic': 0.025,
            'taiwan_specific': True
        },
        'ma_reversion': {
            'description': 'Price relative to moving averages (20D, 50D, 200D)',
            'category': 'mean_reversion',
            'expected_ic': 0.035,
            'taiwan_specific': True
        },
        'bb_position': {
            'description': 'Position within Bollinger Bands with reversion signals',
            'category': 'mean_reversion',
            'expected_ic': 0.04,
            'taiwan_specific': True
        },
        'zscore_reversion': {
            'description': 'Z-score reversion relative to historical price mean',
            'category': 'mean_reversion',
            'expected_ic': 0.03,
            'taiwan_specific': True
        },
        'short_term_reversal': {
            'description': 'Short-term price reversal patterns (1-5 days)',
            'category': 'mean_reversion',
            'expected_ic': 0.02,
            'taiwan_specific': True
        },
        'realized_volatility': {
            'description': 'Multi-period realized volatility (5D, 20D, 60D)',
            'category': 'volatility',
            'expected_ic': 0.04,
            'taiwan_specific': True
        },
        'garch_volatility': {
            'description': 'GARCH(1,1) volatility forecasting',
            'category': 'volatility',
            'expected_ic': 0.035,
            'taiwan_specific': True
        },
        'taiwan_vix': {
            'description': 'Taiwan market volatility index proxy',
            'category': 'volatility',
            'expected_ic': 0.05,
            'taiwan_specific': True
        },
        'vol_risk_premium': {
            'description': 'Volatility risk premium and term structure',
            'category': 'volatility',
            'expected_ic': 0.025,
            'taiwan_specific': True
        }
    }
    
    return metadata