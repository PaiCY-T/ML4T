"""
Sector Neutrality and Style Exposure Analysis for ML4T Taiwan Equity Alpha.

This module provides comprehensive analysis of sector neutrality, style exposures,
and factor analysis for portfolio construction with Taiwan market-specific sectors.

Key Features:
- Taiwan sector classification (GICS/TSE sectors)
- Sector neutrality validation
- Style exposure analysis (value, growth, momentum)
- Factor loading analysis
- Concentration risk assessment

Author: ML4T Team
Date: 2025-09-24
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from decimal import Decimal

logger = logging.getLogger(__name__)


class TaiwanSector(Enum):
    """Taiwan market sector classifications (TSE sectors)."""
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    UTILITIES = "utilities"
    HEALTHCARE = "healthcare"
    REAL_ESTATE = "real_estate"
    TELECOMMUNICATIONS = "telecommunications"
    ENERGY = "energy"


class StyleFactor(Enum):
    """Style factors for Taiwan equity analysis."""
    VALUE = "value"
    GROWTH = "growth"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"
    SIZE = "size"
    LIQUIDITY = "liquidity"


class ExposureLevel(Enum):
    """Exposure level classifications."""
    VERY_LOW = "very_low"      # < -2 std dev
    LOW = "low"                # -2 to -1 std dev
    NEUTRAL = "neutral"        # -1 to +1 std dev
    HIGH = "high"              # +1 to +2 std dev
    VERY_HIGH = "very_high"    # > +2 std dev


@dataclass
class SectorExposure:
    """Sector exposure analysis result."""
    sector: TaiwanSector
    portfolio_weight: float
    benchmark_weight: float
    active_weight: float
    relative_exposure: float  # Active weight / benchmark weight
    exposure_level: ExposureLevel
    number_of_stocks: int
    concentration_ratio: float  # HHI within sector
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'sector': self.sector.value,
            'portfolio_weight': self.portfolio_weight,
            'benchmark_weight': self.benchmark_weight,
            'active_weight': self.active_weight,
            'relative_exposure': self.relative_exposure,
            'exposure_level': self.exposure_level.value,
            'number_of_stocks': self.number_of_stocks,
            'concentration_ratio': self.concentration_ratio
        }


@dataclass
class StyleExposure:
    """Style factor exposure analysis result."""
    factor: StyleFactor
    portfolio_loading: float
    benchmark_loading: float
    active_loading: float
    t_statistic: float
    significance_level: float
    exposure_level: ExposureLevel
    factor_volatility: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'factor': self.factor.value,
            'portfolio_loading': self.portfolio_loading,
            'benchmark_loading': self.benchmark_loading,
            'active_loading': self.active_loading,
            't_statistic': self.t_statistic,
            'significance_level': self.significance_level,
            'exposure_level': self.exposure_level.value,
            'factor_volatility': self.factor_volatility
        }


@dataclass
class NeutralityResult:
    """Overall neutrality analysis result."""
    is_sector_neutral: bool
    sector_exposures: List[SectorExposure]
    style_exposures: List[StyleExposure]
    total_active_risk: float
    sector_contribution_to_risk: float
    style_contribution_to_risk: float
    concentration_hhi: float
    max_sector_deviation: float
    max_style_deviation: float
    neutrality_score: float  # 0-100, higher is more neutral
    
    def get_violations(self, config: 'SectorConfig') -> List[str]:
        """Get list of neutrality violations."""
        violations = []
        
        for exposure in self.sector_exposures:
            if abs(exposure.active_weight) > config.max_sector_deviation:
                violations.append(
                    f"Sector {exposure.sector.value} deviation {exposure.active_weight:.3f} "
                    f"exceeds limit {config.max_sector_deviation:.3f}"
                )
        
        for exposure in self.style_exposures:
            if abs(exposure.active_loading) > config.max_style_deviation:
                violations.append(
                    f"Style {exposure.factor.value} loading {exposure.active_loading:.3f} "
                    f"exceeds limit {config.max_style_deviation:.3f}"
                )
        
        if self.concentration_hhi > config.max_concentration_hhi:
            violations.append(
                f"Portfolio concentration HHI {self.concentration_hhi:.4f} "
                f"exceeds limit {config.max_concentration_hhi:.4f}"
            )
        
        return violations


@dataclass
class SectorConfig:
    """Configuration for sector neutrality analysis."""
    
    # Sector neutrality thresholds
    max_sector_deviation: float = 0.05  # 5% max active weight
    max_style_deviation: float = 0.5    # 0.5 max factor loading
    max_concentration_hhi: float = 0.10 # 10% max HHI
    
    # Analysis parameters
    benchmark_index: str = "TAIEX"
    lookback_period_days: int = 252  # 1 year
    factor_model: str = "taiwan_factor_model"
    
    # Taiwan market specific
    include_tpex_stocks: bool = True
    min_market_cap_ntd: float = 1e9  # 1B NTD minimum
    min_daily_volume_ntd: float = 10e6  # 10M NTD daily volume
    
    # Style factor definitions
    value_metrics: List[str] = field(default_factory=lambda: [
        'price_to_book', 'price_to_earnings', 'enterprise_value_to_ebitda'
    ])
    growth_metrics: List[str] = field(default_factory=lambda: [
        'revenue_growth_1y', 'earnings_growth_1y', 'roe_growth_1y'
    ])
    momentum_metrics: List[str] = field(default_factory=lambda: [
        'price_momentum_3m', 'price_momentum_12m', 'earnings_revision'
    ])
    quality_metrics: List[str] = field(default_factory=lambda: [
        'roe', 'roa', 'debt_to_equity', 'current_ratio'
    ])


class SectorNeutralityAnalyzer:
    """Analyzes sector neutrality and style exposures for Taiwan equity portfolios."""
    
    def __init__(self, config: Optional[SectorConfig] = None):
        """Initialize the sector neutrality analyzer.
        
        Args:
            config: Configuration for sector analysis
        """
        self.config = config or SectorConfig()
        self.sector_mappings = self._load_sector_mappings()
        self.factor_model = self._initialize_factor_model()
        self.benchmark_weights = self._load_benchmark_weights()
        
    def analyze_neutrality(
        self,
        portfolio_weights: Dict[str, float],
        date: date,
        sector_data: Optional[pd.DataFrame] = None,
        style_data: Optional[pd.DataFrame] = None
    ) -> NeutralityResult:
        """Analyze sector neutrality and style exposures.
        
        Args:
            portfolio_weights: Dictionary of symbol -> weight
            date: Analysis date
            sector_data: Optional sector classification data
            style_data: Optional style factor data
            
        Returns:
            Comprehensive neutrality analysis result
        """
        try:
            # Get sector and benchmark data
            if sector_data is None:
                sector_data = self._get_sector_data(list(portfolio_weights.keys()), date)
            
            if style_data is None:
                style_data = self._get_style_data(list(portfolio_weights.keys()), date)
            
            # Analyze sector exposures
            sector_exposures = self._analyze_sector_exposures(
                portfolio_weights, sector_data, date
            )
            
            # Analyze style exposures
            style_exposures = self._analyze_style_exposures(
                portfolio_weights, style_data, date
            )
            
            # Calculate risk decomposition
            risk_decomposition = self._calculate_risk_decomposition(
                portfolio_weights, sector_exposures, style_exposures, date
            )
            
            # Calculate concentration metrics
            concentration_hhi = self._calculate_concentration_hhi(portfolio_weights)
            
            # Determine overall neutrality
            is_neutral = self._determine_neutrality(sector_exposures, style_exposures)
            
            # Calculate neutrality score
            neutrality_score = self._calculate_neutrality_score(
                sector_exposures, style_exposures, concentration_hhi
            )
            
            return NeutralityResult(
                is_sector_neutral=is_neutral,
                sector_exposures=sector_exposures,
                style_exposures=style_exposures,
                total_active_risk=risk_decomposition['total_risk'],
                sector_contribution_to_risk=risk_decomposition['sector_risk'],
                style_contribution_to_risk=risk_decomposition['style_risk'],
                concentration_hhi=concentration_hhi,
                max_sector_deviation=max(abs(exp.active_weight) for exp in sector_exposures),
                max_style_deviation=max(abs(exp.active_loading) for exp in style_exposures),
                neutrality_score=neutrality_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing neutrality: {e}")
            raise
    
    def _analyze_sector_exposures(
        self,
        portfolio_weights: Dict[str, float],
        sector_data: pd.DataFrame,
        date: date
    ) -> List[SectorExposure]:
        """Analyze sector exposures against benchmark."""
        exposures = []
        
        # Calculate portfolio sector weights
        portfolio_sector_weights = self._calculate_portfolio_sector_weights(
            portfolio_weights, sector_data
        )
        
        # Get benchmark sector weights for date
        benchmark_sector_weights = self._get_benchmark_sector_weights(date)
        
        for sector in TaiwanSector:
            portfolio_weight = portfolio_sector_weights.get(sector, 0.0)
            benchmark_weight = benchmark_sector_weights.get(sector, 0.0)
            active_weight = portfolio_weight - benchmark_weight
            
            # Calculate relative exposure
            relative_exposure = (
                active_weight / benchmark_weight if benchmark_weight > 0 else 0.0
            )
            
            # Determine exposure level
            exposure_level = self._classify_exposure_level(relative_exposure)
            
            # Calculate sector statistics
            sector_stocks = sector_data[sector_data['sector'] == sector.value]
            num_stocks = len(sector_stocks)
            
            # Calculate concentration within sector
            if num_stocks > 0:
                sector_portfolio_weights = {
                    symbol: portfolio_weights.get(symbol, 0.0)
                    for symbol in sector_stocks['symbol']
                }
                concentration = self._calculate_concentration_hhi(sector_portfolio_weights)
            else:
                concentration = 0.0
            
            exposures.append(SectorExposure(
                sector=sector,
                portfolio_weight=portfolio_weight,
                benchmark_weight=benchmark_weight,
                active_weight=active_weight,
                relative_exposure=relative_exposure,
                exposure_level=exposure_level,
                number_of_stocks=num_stocks,
                concentration_ratio=concentration
            ))
        
        return exposures
    
    def _analyze_style_exposures(
        self,
        portfolio_weights: Dict[str, float],
        style_data: pd.DataFrame,
        date: date
    ) -> List[StyleExposure]:
        """Analyze style factor exposures."""
        exposures = []
        
        # Calculate portfolio factor loadings
        portfolio_loadings = self._calculate_portfolio_factor_loadings(
            portfolio_weights, style_data
        )
        
        # Get benchmark factor loadings
        benchmark_loadings = self._get_benchmark_factor_loadings(date)
        
        for factor in StyleFactor:
            portfolio_loading = portfolio_loadings.get(factor, 0.0)
            benchmark_loading = benchmark_loadings.get(factor, 0.0)
            active_loading = portfolio_loading - benchmark_loading
            
            # Calculate t-statistic for significance
            factor_vol = self._get_factor_volatility(factor, date)
            t_stat = active_loading / (factor_vol / np.sqrt(252)) if factor_vol > 0 else 0.0
            significance = 2 * (1 - self._t_distribution_cdf(abs(t_stat)))
            
            # Determine exposure level
            exposure_level = self._classify_exposure_level(
                active_loading / factor_vol if factor_vol > 0 else 0
            )
            
            exposures.append(StyleExposure(
                factor=factor,
                portfolio_loading=portfolio_loading,
                benchmark_loading=benchmark_loading,
                active_loading=active_loading,
                t_statistic=t_stat,
                significance_level=significance,
                exposure_level=exposure_level,
                factor_volatility=factor_vol
            ))
        
        return exposures
    
    def _calculate_portfolio_sector_weights(
        self,
        portfolio_weights: Dict[str, float],
        sector_data: pd.DataFrame
    ) -> Dict[TaiwanSector, float]:
        """Calculate portfolio weights by sector."""
        sector_weights = {sector: 0.0 for sector in TaiwanSector}
        
        for symbol, weight in portfolio_weights.items():
            sector_info = sector_data[sector_data['symbol'] == symbol]
            if not sector_info.empty:
                sector_name = sector_info.iloc[0]['sector']
                try:
                    sector = TaiwanSector(sector_name)
                    sector_weights[sector] += weight
                except ValueError:
                    logger.warning(f"Unknown sector '{sector_name}' for symbol {symbol}")
        
        return sector_weights
    
    def _calculate_portfolio_factor_loadings(
        self,
        portfolio_weights: Dict[str, float],
        style_data: pd.DataFrame
    ) -> Dict[StyleFactor, float]:
        """Calculate portfolio factor loadings."""
        factor_loadings = {factor: 0.0 for factor in StyleFactor}
        
        for symbol, weight in portfolio_weights.items():
            stock_data = style_data[style_data['symbol'] == symbol]
            if not stock_data.empty:
                for factor in StyleFactor:
                    factor_col = f"{factor.value}_loading"
                    if factor_col in stock_data.columns:
                        loading = stock_data.iloc[0][factor_col]
                        if pd.notna(loading):
                            factor_loadings[factor] += weight * loading
        
        return factor_loadings
    
    def _calculate_risk_decomposition(
        self,
        portfolio_weights: Dict[str, float],
        sector_exposures: List[SectorExposure],
        style_exposures: List[StyleExposure],
        date: date
    ) -> Dict[str, float]:
        """Calculate risk decomposition into sector and style components."""
        # Simplified risk decomposition
        sector_risk = sum(exp.active_weight ** 2 for exp in sector_exposures)
        style_risk = sum(exp.active_loading ** 2 for exp in style_exposures)
        
        # Total active risk (simplified)
        total_risk = np.sqrt(sector_risk + style_risk)
        
        return {
            'total_risk': total_risk,
            'sector_risk': sector_risk / total_risk if total_risk > 0 else 0,
            'style_risk': style_risk / total_risk if total_risk > 0 else 0
        }
    
    def _calculate_concentration_hhi(
        self, weights: Dict[str, float]
    ) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration."""
        return sum(weight ** 2 for weight in weights.values())
    
    def _determine_neutrality(
        self,
        sector_exposures: List[SectorExposure],
        style_exposures: List[StyleExposure]
    ) -> bool:
        """Determine if portfolio is neutral within configured limits."""
        sector_neutral = all(
            abs(exp.active_weight) <= self.config.max_sector_deviation
            for exp in sector_exposures
        )
        
        style_neutral = all(
            abs(exp.active_loading) <= self.config.max_style_deviation
            for exp in style_exposures
        )
        
        return sector_neutral and style_neutral
    
    def _calculate_neutrality_score(
        self,
        sector_exposures: List[SectorExposure],
        style_exposures: List[StyleExposure],
        concentration_hhi: float
    ) -> float:
        """Calculate overall neutrality score (0-100)."""
        # Sector neutrality component (40%)
        max_sector_dev = max(abs(exp.active_weight) for exp in sector_exposures)
        sector_score = max(0, 100 * (1 - max_sector_dev / self.config.max_sector_deviation))
        
        # Style neutrality component (40%)
        max_style_dev = max(abs(exp.active_loading) for exp in style_exposures)
        style_score = max(0, 100 * (1 - max_style_dev / self.config.max_style_deviation))
        
        # Concentration component (20%)
        concentration_score = max(0, 100 * (1 - concentration_hhi / self.config.max_concentration_hhi))
        
        return 0.4 * sector_score + 0.4 * style_score + 0.2 * concentration_score
    
    def _classify_exposure_level(self, exposure_ratio: float) -> ExposureLevel:
        """Classify exposure level based on standard deviations."""
        if exposure_ratio < -2:
            return ExposureLevel.VERY_LOW
        elif exposure_ratio < -1:
            return ExposureLevel.LOW
        elif exposure_ratio <= 1:
            return ExposureLevel.NEUTRAL
        elif exposure_ratio <= 2:
            return ExposureLevel.HIGH
        else:
            return ExposureLevel.VERY_HIGH
    
    def _load_sector_mappings(self) -> Dict[str, TaiwanSector]:
        """Load sector mappings for Taiwan stocks."""
        # This would typically load from a database or file
        # For now, return empty dict - to be populated with real data
        return {}
    
    def _initialize_factor_model(self) -> Any:
        """Initialize the Taiwan factor model."""
        # This would initialize the actual factor model
        # For now, return None - to be implemented with real model
        return None
    
    def _load_benchmark_weights(self) -> Dict[str, Dict[TaiwanSector, float]]:
        """Load historical benchmark sector weights."""
        # This would load from database
        # For now, return empty dict - to be populated with real data
        return {}
    
    def _get_sector_data(self, symbols: List[str], date: date) -> pd.DataFrame:
        """Get sector classification data for symbols."""
        # This would query the database for sector data
        # For now, return empty DataFrame - to be implemented with real data
        return pd.DataFrame()
    
    def _get_style_data(self, symbols: List[str], date: date) -> pd.DataFrame:
        """Get style factor data for symbols."""
        # This would query the database for style factor data
        # For now, return empty DataFrame - to be implemented with real data
        return pd.DataFrame()
    
    def _get_benchmark_sector_weights(self, date: date) -> Dict[TaiwanSector, float]:
        """Get benchmark sector weights for date."""
        # This would query historical benchmark weights
        # For now, return equal weights - to be implemented with real data
        return {sector: 1.0 / len(TaiwanSector) for sector in TaiwanSector}
    
    def _get_benchmark_factor_loadings(self, date: date) -> Dict[StyleFactor, float]:
        """Get benchmark factor loadings for date."""
        # This would query historical benchmark factor loadings
        # For now, return zero loadings - to be implemented with real data
        return {factor: 0.0 for factor in StyleFactor}
    
    def _get_factor_volatility(self, factor: StyleFactor, date: date) -> float:
        """Get factor volatility for risk calculations."""
        # This would query historical factor volatility
        # For now, return fixed volatility - to be implemented with real data
        return 0.15  # 15% annualized volatility
    
    def _t_distribution_cdf(self, x: float, df: int = 252) -> float:
        """Cumulative distribution function for t-distribution."""
        # Simplified approximation using normal distribution
        from scipy.stats import t
        try:
            return t.cdf(x, df)
        except ImportError:
            # Fallback to normal approximation
            from math import erf, sqrt
            return 0.5 * (1 + erf(x / sqrt(2)))


def create_standard_sector_analyzer(
    benchmark: str = "TAIEX",
    max_sector_dev: float = 0.05
) -> SectorNeutralityAnalyzer:
    """Create a standard sector neutrality analyzer for Taiwan markets.
    
    Args:
        benchmark: Benchmark index name
        max_sector_dev: Maximum sector deviation (default 5%)
        
    Returns:
        Configured sector neutrality analyzer
    """
    config = SectorConfig(
        benchmark_index=benchmark,
        max_sector_deviation=max_sector_dev,
        max_style_deviation=0.5,
        max_concentration_hhi=0.10
    )
    
    return SectorNeutralityAnalyzer(config)


def create_strict_sector_analyzer(
    benchmark: str = "TAIEX"
) -> SectorNeutralityAnalyzer:
    """Create a strict sector neutrality analyzer with tight limits.
    
    Args:
        benchmark: Benchmark index name
        
    Returns:
        Strictly configured sector neutrality analyzer
    """
    config = SectorConfig(
        benchmark_index=benchmark,
        max_sector_deviation=0.02,  # 2% max deviation
        max_style_deviation=0.3,    # Tighter style limits
        max_concentration_hhi=0.05  # Lower concentration limit
    )
    
    return SectorNeutralityAnalyzer(config)