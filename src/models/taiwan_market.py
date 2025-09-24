"""
Taiwan Market Adaptations for LightGBM Alpha Model

Market-specific optimizations and adaptations for Taiwan Stock Exchange (TWSE)
including trading rules, market structure, and regulatory constraints.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import warnings
from enum import Enum

import numpy as np
import pandas as pd
from decimal import Decimal

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Taiwan market trading sessions."""
    PRE_MARKET = "pre_market"     # 08:30-09:00
    OPENING = "opening"           # 09:00-09:05 (price discovery)
    CONTINUOUS = "continuous"     # 09:05-13:25 (continuous trading)
    CLOSING = "closing"           # 13:25-13:30 (closing auction)
    POST_MARKET = "post_market"   # 13:30-14:00 (after hours)


class PriceLimitType(Enum):
    """Taiwan price limit types."""
    NORMAL = "normal"         # ±10% for most stocks
    ETF = "etf"              # ±10% for ETFs
    WARRANT = "warrant"       # No limits for warrants
    NEWLY_LISTED = "newly_listed"  # ±30% for first 5 trading days


@dataclass
class TaiwanMarketRules:
    """Taiwan Stock Exchange trading rules and constraints."""
    
    # Trading hours (Taiwan Standard Time)
    market_open: time = time(9, 0)      # 09:00 TST
    market_close: time = time(13, 30)   # 13:30 TST
    session_length_hours: float = 4.5
    
    # Price limits
    normal_price_limit: float = 0.10    # ±10% for regular stocks
    etf_price_limit: float = 0.10       # ±10% for ETFs
    newly_listed_limit: float = 0.30    # ±30% for first 5 days
    
    # Settlement
    settlement_cycle: int = 2           # T+2 settlement
    
    # Tick size rules (simplified)
    tick_sizes: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'under_10': {'tick': 0.01},
        '10_to_50': {'tick': 0.05},
        '50_to_100': {'tick': 0.1},
        '100_to_500': {'tick': 0.5},
        '500_to_1000': {'tick': 1.0},
        'over_1000': {'tick': 5.0}
    })
    
    # Order types and constraints
    min_order_size: int = 1000          # 1 lot = 1000 shares
    max_single_order_ratio: float = 0.03  # Max 3% of daily volume in single order
    
    # Market structure
    total_listed_companies: int = 2000   # Approximate universe size
    average_daily_turnover_ntd: float = 200_000_000_000  # ~200B NTD daily
    
    # Foreign investment limits
    foreign_ownership_limit: float = 0.50  # 50% aggregate limit
    single_foreign_limit: float = 0.10     # 10% per single foreign entity


@dataclass  
class MarketAdaptations:
    """Taiwan market specific adaptations for ML models."""
    
    def __init__(self, rules: Optional[TaiwanMarketRules] = None):
        """Initialize with Taiwan market rules."""
        self.rules = rules or TaiwanMarketRules()
        self.holiday_calendar = self._initialize_holiday_calendar()
        self.sector_weights = self._initialize_sector_weights()
        
    def _initialize_holiday_calendar(self) -> Set[str]:
        """Initialize Taiwan market holiday calendar."""
        # Key Taiwan holidays (simplified)
        holidays_2024 = {
            '2024-01-01',  # New Year
            '2024-02-08', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12',  # Chinese New Year
            '2024-02-28',  # Peace Memorial Day
            '2024-04-04', '2024-04-05',  # Tomb Sweeping Day
            '2024-05-01',  # Labor Day
            '2024-06-10',  # Dragon Boat Festival
            '2024-09-17',  # Mid-Autumn Festival
            '2024-10-10',  # National Day
        }
        return holidays_2024
    
    def _initialize_sector_weights(self) -> Dict[str, float]:
        """Initialize Taiwan market sector weights (approximate)."""
        return {
            'technology': 0.55,      # TSMC, MediaTek, etc.
            'financials': 0.15,      # Banks, insurance
            'materials': 0.08,       # Chemicals, steel
            'industrials': 0.07,     # Manufacturing
            'consumer_discretionary': 0.05,
            'healthcare': 0.04,
            'utilities': 0.03,
            'energy': 0.02,
            'others': 0.01
        }
    
    def is_trading_day(self, date: pd.Timestamp) -> bool:
        """Check if given date is a trading day."""
        # Check weekends
        if date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check holidays
        date_str = date.strftime('%Y-%m-%d')
        return date_str not in self.holiday_calendar
    
    def get_trading_session(self, timestamp: pd.Timestamp) -> TradingSession:
        """Determine trading session for given timestamp."""
        time_of_day = timestamp.time()
        
        if time_of_day < time(9, 0):
            return TradingSession.PRE_MARKET
        elif time_of_day < time(9, 5):
            return TradingSession.OPENING
        elif time_of_day < time(13, 25):
            return TradingSession.CONTINUOUS
        elif time_of_day < time(13, 30):
            return TradingSession.CLOSING
        else:
            return TradingSession.POST_MARKET
    
    def calculate_price_limits(
        self, 
        reference_price: float, 
        limit_type: PriceLimitType = PriceLimitType.NORMAL
    ) -> Tuple[float, float]:
        """Calculate price limits based on reference price and type."""
        if limit_type == PriceLimitType.WARRANT:
            return 0.0, float('inf')  # No limits for warrants
        
        if limit_type == PriceLimitType.NEWLY_LISTED:
            limit_pct = self.rules.newly_listed_limit
        else:  # NORMAL or ETF
            limit_pct = self.rules.normal_price_limit
        
        lower_limit = reference_price * (1 - limit_pct)
        upper_limit = reference_price * (1 + limit_pct)
        
        return lower_limit, upper_limit
    
    def get_tick_size(self, price: float) -> float:
        """Get appropriate tick size based on price level."""
        if price < 10:
            return self.rules.tick_sizes['under_10']['tick']
        elif price < 50:
            return self.rules.tick_sizes['10_to_50']['tick']  
        elif price < 100:
            return self.rules.tick_sizes['50_to_100']['tick']
        elif price < 500:
            return self.rules.tick_sizes['100_to_500']['tick']
        elif price < 1000:
            return self.rules.tick_sizes['500_to_1000']['tick']
        else:
            return self.rules.tick_sizes['over_1000']['tick']
    
    def adjust_for_corporate_actions(
        self, 
        price_data: pd.DataFrame,
        corporate_actions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Adjust price data for Taiwan-specific corporate actions."""
        adjusted_data = price_data.copy()
        
        if corporate_actions is not None:
            # Handle dividend adjustments (Taiwan typically pays annual dividends)
            dividend_dates = corporate_actions[
                corporate_actions['action_type'] == 'dividend'
            ]
            
            for _, action in dividend_dates.iterrows():
                symbol = action['symbol']
                ex_date = action['ex_date']
                dividend_amount = action['amount']
                
                # Adjust prices before ex-date
                mask = (adjusted_data.index.get_level_values('symbol') == symbol) & \
                       (adjusted_data.index.get_level_values('date') < ex_date)
                
                adjustment_factor = 1 - (dividend_amount / action['reference_price'])
                adjusted_data.loc[mask, ['open', 'high', 'low', 'close']] *= adjustment_factor
        
        return adjusted_data
    
    def calculate_foreign_flow_impact(
        self, 
        foreign_flow_data: pd.DataFrame,
        market_cap_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate foreign flow impact metrics."""
        impact_metrics = pd.DataFrame(index=foreign_flow_data.index)
        
        # Net foreign flow as percentage of market cap
        impact_metrics['foreign_flow_pct_mcap'] = (
            foreign_flow_data['net_foreign_flow'] / market_cap_data['market_cap']
        )
        
        # Foreign ownership percentage
        if 'foreign_ownership' in foreign_flow_data.columns:
            impact_metrics['foreign_ownership_pct'] = foreign_flow_data['foreign_ownership']
            
            # Distance from foreign ownership limit
            impact_metrics['foreign_limit_distance'] = (
                self.rules.foreign_ownership_limit - impact_metrics['foreign_ownership_pct']
            )
        
        # Rolling foreign flow momentum
        impact_metrics['foreign_flow_momentum_5d'] = (
            foreign_flow_data['net_foreign_flow'].rolling(5).sum()
        )
        impact_metrics['foreign_flow_momentum_20d'] = (
            foreign_flow_data['net_foreign_flow'].rolling(20).sum()
        )
        
        return impact_metrics
    
    def adjust_volume_for_session_length(self, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Adjust volume metrics for Taiwan's 4.5-hour trading session."""
        adjusted_volume = volume_data.copy()
        
        # Standard trading day is often assumed to be 6.5 hours (US market)
        # Taiwan market is 4.5 hours, so normalize volume metrics
        session_adjustment = 6.5 / 4.5
        
        if 'normalized_volume' in adjusted_volume.columns:
            adjusted_volume['normalized_volume'] *= session_adjustment
        
        if 'volume_volatility' in adjusted_volume.columns:
            adjusted_volume['volume_volatility'] *= np.sqrt(session_adjustment)
        
        return adjusted_volume
    
    def get_market_regime_indicators(
        self, 
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Taiwan market regime indicators."""
        regime_indicators = pd.DataFrame(index=price_data.index)
        
        # Technology sector dominance (TSMC effect)
        if 'sector' in price_data.columns:
            tech_mask = price_data['sector'] == 'technology'
            tech_performance = price_data[tech_mask]['returns'].groupby('date').mean()
            market_performance = price_data['returns'].groupby('date').mean()
            
            regime_indicators['tech_vs_market'] = (
                tech_performance / market_performance - 1
            )
        
        # Foreign flow regime
        regime_indicators['high_foreign_flow'] = (
            volume_data.get('foreign_volume_ratio', 0) > 0.4
        ).astype(int)
        
        # Price limit regime (many stocks hitting limits)
        if 'hit_price_limit' in price_data.columns:
            daily_limit_hits = price_data.groupby('date')['hit_price_limit'].sum()
            regime_indicators['high_limit_regime'] = (
                daily_limit_hits > daily_limit_hits.quantile(0.8)
            ).astype(int)
        
        return regime_indicators


class TaiwanMarketModel:
    """
    Taiwan market optimized ML model wrapper.
    
    Provides Taiwan-specific enhancements to the base LightGBM model including:
    - Market structure awareness
    - Trading rule compliance
    - Performance attribution by market factors
    """
    
    def __init__(self, base_model, adaptations: Optional[MarketAdaptations] = None):
        """Initialize Taiwan market model wrapper."""
        self.base_model = base_model
        self.adaptations = adaptations or MarketAdaptations()
        self.market_factors: Optional[pd.DataFrame] = None
        self.performance_attribution: Dict[str, Any] = {}
        
        logger.info("TaiwanMarketModel initialized")
    
    def prepare_taiwan_features(
        self, 
        features: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Prepare Taiwan-specific features."""
        taiwan_features = features.copy()
        
        # Add market structure features
        if 'price_data' in market_data:
            price_data = market_data['price_data']
            
            # Price limit proximity
            taiwan_features['price_limit_proximity'] = self._calculate_price_limit_proximity(price_data)
            
            # Session-based features
            taiwan_features = self._add_session_features(taiwan_features, price_data)
        
        # Add foreign flow features
        if 'foreign_flow' in market_data:
            foreign_features = self.adaptations.calculate_foreign_flow_impact(
                market_data['foreign_flow'], 
                market_data.get('market_cap', pd.DataFrame())
            )
            taiwan_features = taiwan_features.join(foreign_features, how='left')
        
        # Add market regime features
        if all(k in market_data for k in ['price_data', 'volume_data']):
            regime_features = self.adaptations.get_market_regime_indicators(
                market_data['price_data'], 
                market_data['volume_data']
            )
            taiwan_features = taiwan_features.join(regime_features, how='left')
        
        return taiwan_features
    
    def _calculate_price_limit_proximity(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate proximity to price limits."""
        if 'reference_price' not in price_data.columns:
            return pd.Series(0, index=price_data.index)
        
        proximity = pd.Series(index=price_data.index, dtype=float)
        
        for idx, row in price_data.iterrows():
            lower_limit, upper_limit = self.adaptations.calculate_price_limits(
                row['reference_price']
            )
            
            # Distance from current price to nearest limit
            current_price = row.get('close', row['reference_price'])
            distance_to_upper = (upper_limit - current_price) / current_price
            distance_to_lower = (current_price - lower_limit) / current_price
            
            # Use minimum distance (closest to either limit)
            proximity[idx] = min(distance_to_upper, distance_to_lower)
        
        return proximity
    
    def _add_session_features(
        self, 
        features: pd.DataFrame, 
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add Taiwan trading session features."""
        if 'timestamp' not in price_data.columns:
            return features
        
        session_features = features.copy()
        
        # Session classification
        price_data['trading_session'] = price_data['timestamp'].apply(
            self.adaptations.get_trading_session
        )
        
        # Session-specific volatility
        session_vol = price_data.groupby('trading_session').apply(
            lambda x: (x['high'] - x['low']) / x['open']
        ).mean()
        
        # Add session volatility as feature
        for session in TradingSession:
            session_features[f'vol_{session.value}'] = session_vol.get(session, 0)
        
        return session_features
    
    def train_with_taiwan_adaptations(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, Any]:
        """Train model with Taiwan market adaptations."""
        logger.info("Training with Taiwan market adaptations")
        
        # Prepare Taiwan-specific features
        X_taiwan = self.prepare_taiwan_features(X, market_data)
        
        # Filter trading days only
        trading_days = X_taiwan.index.get_level_values('date').map(
            self.adaptations.is_trading_day
        )
        X_taiwan = X_taiwan[trading_days]
        y = y[trading_days]
        
        # Train base model
        training_stats = self.base_model.train(X_taiwan, y, **kwargs)
        
        # Add Taiwan-specific analysis
        taiwan_analysis = self._analyze_taiwan_factors(X_taiwan, y)
        training_stats.update(taiwan_analysis)
        
        return training_stats
    
    def _analyze_taiwan_factors(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Any]:
        """Analyze Taiwan market factors impact."""
        analysis = {}
        
        # Sector performance analysis
        if 'sector' in X.columns:
            sector_returns = y.groupby(X['sector']).mean()
            analysis['sector_performance'] = sector_returns.to_dict()
        
        # Foreign flow impact analysis
        foreign_cols = [col for col in X.columns if 'foreign' in col.lower()]
        if foreign_cols:
            foreign_impact = X[foreign_cols].corrwith(y)
            analysis['foreign_flow_correlation'] = foreign_impact.to_dict()
        
        # Price limit impact analysis
        price_limit_cols = [col for col in X.columns if 'limit' in col.lower()]
        if price_limit_cols:
            limit_impact = X[price_limit_cols].corrwith(y)
            analysis['price_limit_correlation'] = limit_impact.to_dict()
        
        return analysis
    
    def predict_with_market_context(
        self,
        X: pd.DataFrame,
        market_data: Dict[str, pd.DataFrame],
        include_attribution: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Generate predictions with market context analysis."""
        # Prepare features
        X_taiwan = self.prepare_taiwan_features(X, market_data)
        
        # Generate predictions
        predictions = self.base_model.predict(X_taiwan)
        
        # Performance attribution
        attribution = None
        if include_attribution:
            attribution = self._calculate_performance_attribution(
                X_taiwan, predictions, market_data
            )
        
        return predictions, attribution
    
    def _calculate_performance_attribution(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate performance attribution to Taiwan market factors."""
        attribution = {}
        
        # Sector attribution
        if 'sector' in features.columns:
            sector_pred = pd.DataFrame({
                'sector': features['sector'],
                'prediction': predictions
            }).groupby('sector')['prediction'].mean()
            attribution['sector_attribution'] = sector_pred.to_dict()
        
        # Market regime attribution
        regime_cols = [col for col in features.columns if 'regime' in col.lower()]
        if regime_cols:
            regime_attribution = {}
            for col in regime_cols:
                high_regime = features[col] > 0.5
                regime_attribution[col] = {
                    'high_regime_pred': predictions[high_regime].mean(),
                    'low_regime_pred': predictions[~high_regime].mean()
                }
            attribution['regime_attribution'] = regime_attribution
        
        return attribution
    
    def get_taiwan_market_summary(self) -> Dict[str, Any]:
        """Get Taiwan market specific model summary."""
        summary = self.base_model.get_model_summary()
        
        # Add Taiwan-specific information
        summary['taiwan_adaptations'] = {
            'trading_hours': f"{self.adaptations.rules.market_open} - {self.adaptations.rules.market_close}",
            'session_length': f"{self.adaptations.rules.session_length_hours} hours",
            'settlement_cycle': f"T+{self.adaptations.rules.settlement_cycle}",
            'price_limits': f"±{self.adaptations.rules.normal_price_limit*100}%",
            'universe_size': self.adaptations.rules.total_listed_companies
        }
        
        if self.performance_attribution:
            summary['performance_attribution'] = self.performance_attribution
        
        return summary
    
    def validate_trading_rules(
        self, 
        signals: pd.DataFrame,
        position_sizes: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Validate signals against Taiwan trading rules."""
        validation_results = {
            'valid_signals': 0,
            'rule_violations': [],
            'warnings': []
        }
        
        # Check trading day constraints
        trading_day_violations = 0
        for date in signals.index.get_level_values('date').unique():
            if not self.adaptations.is_trading_day(pd.Timestamp(date)):
                trading_day_violations += len(signals.xs(date, level='date'))
        
        if trading_day_violations > 0:
            validation_results['rule_violations'].append(
                f"Signals on {trading_day_violations} non-trading day observations"
            )
        
        # Check position size constraints (if provided)
        if position_sizes is not None:
            # Check minimum lot size (1000 shares)
            min_lot_violations = (position_sizes % 1000 != 0).sum().sum()
            if min_lot_violations > 0:
                validation_results['rule_violations'].append(
                    f"{min_lot_violations} positions not in multiples of 1000 shares"
                )
        
        # Calculate valid signals
        validation_results['valid_signals'] = (
            len(signals) - trading_day_violations - min_lot_violations
        )
        
        # Overall validation status
        validation_results['is_valid'] = len(validation_results['rule_violations']) == 0
        
        return validation_results