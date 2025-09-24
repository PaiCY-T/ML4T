"""
Taiwan-specific market microstructure factors.

This module implements 4 unique Taiwan market factors:
1. Foreign Institutional Flow Impact - Foreign investment flow effects
2. Margin Trading Ratios - Margin buying/selling indicators
3. Index Inclusion Effects - Taiwan index rebalancing impacts  
4. Cross-Strait Sentiment - Taiwan-China relation sentiment proxy

These factors capture unique characteristics of the Taiwan equity market:
- Foreign ownership limits (50% individual stock caps)
- Margin trading regulations and patterns
- TAIEX/TWSE 50/TPEx index rebalancing effects
- Cross-strait political and economic relationship impacts
"""

from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .base import FactorResult, FactorMetadata, FactorCategory, FactorFrequency
from .microstructure import (
    MicrostructureFactorCalculator, ForeignFlowData, MarginTradingData,
    IndexCompositionData, CrossStraitSentiment
)

# Import dependencies - will be mocked for testing if not available
try:
    from ..data.pipeline.pit_engine import PITQueryEngine
    from ..data.core.temporal import DataType
except ImportError:
    PITQueryEngine = object
    DataType = object

logger = logging.getLogger(__name__)


@dataclass
class TaiwanMarketStructure:
    """Taiwan market structure parameters."""
    foreign_ownership_cap: float = 0.50  # 50% individual stock cap
    margin_initial_rate: float = 0.60     # 60% initial margin requirement
    margin_maintenance_rate: float = 0.30  # 30% maintenance margin
    
    # Major Taiwan indices
    major_indices: List[str] = field(default_factory=lambda: [
        'TAIEX', 'TWSE50', 'TPEx', 'TWSE_MID100', 'TWSE_SMALL'
    ])
    
    # Quarterly rebalancing months
    rebalancing_months: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    
    # Taiwan market holidays (approximate - would use proper calendar)
    typical_holidays: List[str] = field(default_factory=lambda: [
        'Lunar_New_Year', 'Tomb_Sweeping_Day', 'Dragon_Boat_Festival', 
        'Mid_Autumn_Festival', 'National_Day'
    ])


class ForeignFlowImpactCalculator(MicrostructureFactorCalculator):
    """
    Foreign Institutional Flow Impact Calculator.
    
    Analyzes the impact of foreign institutional investment flows:
    - Net foreign buying/selling pressure
    - Foreign ownership ratio trends
    - Flow momentum and persistence
    - Impact on price discovery and liquidity
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="foreign_flow_impact",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Foreign institutional flow impact and ownership dynamics",
            lookback_days=126,  # ~6 months for flow analysis
            data_requirements=[DataType.FOREIGN_FLOWS, DataType.OHLCV],
            taiwan_specific=True,
            expected_ic=0.05,  # Foreign flows typically have strong predictive power
            expected_turnover=0.20
        )
        super().__init__(pit_engine, metadata)
        self.market_structure = TaiwanMarketStructure()
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate foreign flow impact factors."""
        
        # Get foreign flow data
        flow_data = self._get_historical_data(
            symbols, as_of_date, DataType.FOREIGN_FLOWS, self.metadata.lookback_days
        )
        
        # Get price data for flow impact analysis
        price_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_flows = flow_data[flow_data['symbol'] == symbol].copy()
                symbol_prices = price_data[price_data['symbol'] == symbol].copy()
                
                if symbol_flows.empty or symbol_prices.empty:
                    continue
                
                # Sort by date
                symbol_flows = symbol_flows.sort_values('date')
                symbol_prices = symbol_prices.sort_values('date')
                
                # Calculate flow metrics
                flow_metrics = self._calculate_flow_metrics(symbol_flows)
                
                # Calculate ownership dynamics
                ownership_dynamics = self._calculate_ownership_dynamics(symbol_flows)
                
                # Calculate flow-price relationship
                flow_price_impact = self._calculate_flow_price_impact(symbol_flows, symbol_prices)
                
                # Calculate flow persistence and momentum
                flow_momentum = self._calculate_flow_momentum(symbol_flows)
                
                # Combine all foreign flow factors
                factor_value = (
                    0.3 * flow_metrics +
                    0.25 * ownership_dynamics + 
                    0.25 * flow_price_impact +
                    0.2 * flow_momentum
                )
                
                factor_values[symbol] = factor_value
                
            except Exception as e:
                logger.warning(f"Error calculating foreign flow impact for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_flow_metrics(self, flows: pd.DataFrame) -> float:
        """Calculate basic foreign flow metrics."""
        if len(flows) < 10:
            return 0.0
        
        recent_flows = flows.tail(20)
        
        # Net flow ratio (recent vs historical)
        recent_net_flow = recent_flows['foreign_net_value'].sum()
        historical_net_flow = flows['foreign_net_value'].sum()
        
        # Flow concentration (how concentrated are the flows)
        flow_std = recent_flows['foreign_net_value'].std()
        flow_mean = abs(recent_flows['foreign_net_value'].mean())
        
        flow_concentration = flow_std / flow_mean if flow_mean > 0 else 0
        
        # Recent vs historical intensity
        flow_intensity = np.tanh(recent_net_flow / (abs(historical_net_flow) + 1e6))
        
        return (flow_intensity + np.tanh(flow_concentration - 1)) / 2
    
    def _calculate_ownership_dynamics(self, flows: pd.DataFrame) -> float:
        """Calculate foreign ownership dynamics."""
        if len(flows) < 10:
            return 0.0
        
        recent_flows = flows.tail(20)
        
        # Ownership trend
        if 'foreign_ownership_pct' in recent_flows.columns:
            ownership_start = recent_flows['foreign_ownership_pct'].iloc[0]
            ownership_end = recent_flows['foreign_ownership_pct'].iloc[-1]
            
            ownership_trend = (ownership_end - ownership_start) / (ownership_start + 1e-6)
            
            # Distance from ownership cap
            cap_distance = self.market_structure.foreign_ownership_cap - ownership_end
            cap_pressure = np.tanh(5 * (1 - cap_distance / self.market_structure.foreign_ownership_cap))
            
            return (np.tanh(ownership_trend * 10) + cap_pressure) / 2
        
        return 0.0
    
    def _calculate_flow_price_impact(self, flows: pd.DataFrame, prices: pd.DataFrame) -> float:
        """Calculate how foreign flows impact prices."""
        if len(flows) < 20 or len(prices) < 20:
            return 0.0
        
        # Merge flows and prices
        merged = pd.merge(flows[['date', 'foreign_net_value']], 
                         prices[['date', 'close']], on='date', how='inner')
        
        if len(merged) < 15:
            return 0.0
        
        merged = merged.sort_values('date')
        merged['price_change'] = merged['close'].pct_change()
        
        # Calculate correlation between flows and next-day returns
        if len(merged) >= 20:
            flow_values = merged['foreign_net_value'].iloc[:-1].values
            next_day_returns = merged['price_change'].iloc[1:].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(flow_values) | np.isnan(next_day_returns))
            if np.sum(valid_mask) >= 10:
                correlation = np.corrcoef(flow_values[valid_mask], next_day_returns[valid_mask])[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _calculate_flow_momentum(self, flows: pd.DataFrame) -> float:
        """Calculate foreign flow momentum and persistence."""
        if len(flows) < 20:
            return 0.0
        
        recent_flows = flows.tail(20)['foreign_net_value'].values
        
        # Calculate momentum (positive flows followed by positive flows)
        flow_signs = np.sign(recent_flows)
        momentum_score = 0.0
        
        for i in range(1, len(flow_signs)):
            if flow_signs[i] == flow_signs[i-1] and flow_signs[i] != 0:
                momentum_score += abs(recent_flows[i]) / (abs(recent_flows).sum() + 1e-6)
        
        return np.tanh(momentum_score * 2)


class MarginTradingRatioCalculator(MicrostructureFactorCalculator):
    """
    Margin Trading Ratio Calculator.
    
    Analyzes margin trading patterns:
    - Margin buy vs sell ratios
    - Margin balance trends
    - Leverage patterns and risk indicators
    - Taiwan-specific margin regulations impact
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="margin_trading_ratio",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Margin trading ratios and leverage pattern analysis",
            lookback_days=63,  # ~3 months for margin analysis
            data_requirements=[DataType.MARGIN_TRADING, DataType.OHLCV],
            taiwan_specific=True,
            expected_ic=0.03,
            expected_turnover=0.25
        )
        super().__init__(pit_engine, metadata)
        self.market_structure = TaiwanMarketStructure()
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate margin trading ratio factors."""
        
        # Get margin trading data
        margin_data = self._get_historical_data(
            symbols, as_of_date, DataType.MARGIN_TRADING, self.metadata.lookback_days
        )
        
        # Get price data for context
        price_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_margin = margin_data[margin_data['symbol'] == symbol].copy()
                symbol_prices = price_data[price_data['symbol'] == symbol].copy()
                
                if symbol_margin.empty:
                    continue
                
                # Sort by date
                symbol_margin = symbol_margin.sort_values('date')
                
                # Calculate margin metrics
                margin_ratio_trend = self._calculate_margin_ratio_trend(symbol_margin)
                leverage_pattern = self._calculate_leverage_pattern(symbol_margin)
                margin_sentiment = self._calculate_margin_sentiment(symbol_margin)
                risk_indicator = self._calculate_margin_risk_indicator(symbol_margin, symbol_prices)
                
                # Combine margin trading factors
                factor_value = (
                    0.3 * margin_ratio_trend +
                    0.25 * leverage_pattern +
                    0.25 * margin_sentiment +
                    0.2 * risk_indicator
                )
                
                factor_values[symbol] = factor_value
                
            except Exception as e:
                logger.warning(f"Error calculating margin trading ratio for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_margin_ratio_trend(self, margin_data: pd.DataFrame) -> float:
        """Calculate margin buy/sell ratio trends."""
        if len(margin_data) < 10:
            return 0.0
        
        recent_data = margin_data.tail(10)
        
        # Calculate buy/sell ratios
        recent_data['margin_buy_sell_ratio'] = (
            recent_data['margin_buy_volume'] / (recent_data['margin_sell_volume'] + 1)
        )
        
        # Trend in margin ratio
        ratio_trend = (
            recent_data['margin_buy_sell_ratio'].iloc[-3:].mean() /
            recent_data['margin_buy_sell_ratio'].iloc[:3].mean()
        ) - 1
        
        return np.tanh(ratio_trend)
    
    def _calculate_leverage_pattern(self, margin_data: pd.DataFrame) -> float:
        """Calculate leverage usage patterns."""
        if len(margin_data) < 10:
            return 0.0
        
        # Margin balance as proxy for leverage
        recent_balance = margin_data.tail(5)['margin_balance'].mean()
        historical_balance = margin_data['margin_balance'].mean()
        
        leverage_change = (recent_balance / historical_balance - 1) if historical_balance > 0 else 0
        
        return np.tanh(leverage_change)
    
    def _calculate_margin_sentiment(self, margin_data: pd.DataFrame) -> float:
        """Calculate margin trading sentiment indicator."""
        if len(margin_data) < 15:
            return 0.0
        
        recent_data = margin_data.tail(15)
        
        # More buying than selling indicates bullish sentiment
        total_margin_buy = recent_data['margin_buy_volume'].sum()
        total_margin_sell = recent_data['margin_sell_volume'].sum()
        
        if total_margin_sell > 0:
            sentiment_ratio = total_margin_buy / total_margin_sell - 1
            return np.tanh(sentiment_ratio)
        
        return 0.0
    
    def _calculate_margin_risk_indicator(self, margin_data: pd.DataFrame, 
                                       price_data: pd.DataFrame) -> float:
        """Calculate margin-related risk indicators."""
        if len(margin_data) < 10 or price_data.empty:
            return 0.0
        
        recent_margin = margin_data.tail(10)
        
        # High margin balance during volatile periods indicates higher risk
        if 'margin_ratio_pct' in recent_margin.columns:
            avg_margin_ratio = recent_margin['margin_ratio_pct'].mean()
            
            # Compare to Taiwan margin requirements
            excess_leverage = avg_margin_ratio - self.market_structure.margin_initial_rate
            risk_score = np.tanh(excess_leverage * 5)  # Amplify excess leverage impact
            
            return risk_score
        
        return 0.0


class IndexInclusionEffectCalculator(MicrostructureFactorCalculator):
    """
    Index Inclusion Effect Calculator.
    
    Analyzes Taiwan index inclusion/exclusion effects:
    - TAIEX index rebalancing impacts
    - TWSE 50 inclusion probability
    - Passive flow anticipation
    - Index weight change effects
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="index_inclusion_effect",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Taiwan index inclusion/exclusion effects and rebalancing impacts",
            lookback_days=252,  # 1 year for index pattern analysis
            data_requirements=[DataType.INDEX_COMPOSITION, DataType.OHLCV, DataType.MARKET_CAP],
            taiwan_specific=True,
            expected_ic=0.04,
            expected_turnover=0.15
        )
        super().__init__(pit_engine, metadata)
        self.market_structure = TaiwanMarketStructure()
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate index inclusion effect factors."""
        
        # Get index composition data
        index_data = self._get_historical_data(
            symbols, as_of_date, DataType.INDEX_COMPOSITION, self.metadata.lookback_days
        )
        
        # Get market cap data for inclusion probability
        mcap_data = self._get_historical_data(
            symbols, as_of_date, DataType.MARKET_CAP, 126
        )
        
        # Get price data for impact analysis
        price_data = self._get_historical_data(
            symbols, as_of_date, DataType.OHLCV, 63
        )
        
        factor_values = {}
        
        for symbol in symbols:
            try:
                symbol_index = index_data[index_data['symbol'] == symbol].copy()
                symbol_mcap = mcap_data[mcap_data['symbol'] == symbol].copy()
                symbol_prices = price_data[price_data['symbol'] == symbol].copy()
                
                # Calculate inclusion probability
                inclusion_probability = self._calculate_inclusion_probability(
                    symbol, symbol_index, symbol_mcap, as_of_date
                )
                
                # Calculate rebalancing effect
                rebalancing_effect = self._calculate_rebalancing_effect(
                    symbol_index, as_of_date
                )
                
                # Calculate weight change impact
                weight_change_impact = self._calculate_weight_change_impact(
                    symbol_index, symbol_prices
                )
                
                # Calculate passive flow anticipation
                passive_flow_anticipation = self._calculate_passive_flow_anticipation(
                    inclusion_probability, rebalancing_effect, as_of_date
                )
                
                # Combine index effect factors
                factor_value = (
                    0.3 * inclusion_probability +
                    0.25 * rebalancing_effect +
                    0.25 * weight_change_impact +
                    0.2 * passive_flow_anticipation
                )
                
                factor_values[symbol] = factor_value
                
            except Exception as e:
                logger.warning(f"Error calculating index inclusion effect for {symbol}: {e}")
                continue
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_inclusion_probability(self, symbol: str, index_data: pd.DataFrame,
                                       mcap_data: pd.DataFrame, as_of_date: date) -> float:
        """Calculate probability of index inclusion."""
        if mcap_data.empty:
            return 0.0
        
        # Check current index status
        current_status = self._get_current_index_status(symbol, index_data, as_of_date)
        
        # Market cap ranking proxy for inclusion probability
        latest_mcap = mcap_data['market_cap'].iloc[-1] if not mcap_data.empty else 0
        
        # Simplified probability model based on market cap
        # In practice, would use more sophisticated model with liquidity, float, etc.
        if latest_mcap > 0:
            # Proxy: larger market cap = higher inclusion probability
            mcap_score = np.log10(latest_mcap / 1e9)  # Log of market cap in billions
            probability = np.tanh(mcap_score - 2)  # Adjust threshold as needed
            
            # If already included, inclusion probability is 1, focus on exclusion risk
            if current_status['is_included']:
                return -probability  # Negative indicates exclusion risk
            else:
                return probability
        
        return 0.0
    
    def _calculate_rebalancing_effect(self, index_data: pd.DataFrame, as_of_date: date) -> float:
        """Calculate rebalancing timing effects."""
        current_month = as_of_date.month
        current_day = as_of_date.day
        
        # Taiwan indices typically rebalance quarterly
        if current_month in self.market_structure.rebalancing_months:
            # Stronger effect closer to month-end
            days_to_rebalancing = 31 - current_day
            rebalancing_proximity = np.exp(-days_to_rebalancing / 10)  # Exponential decay
            
            return rebalancing_proximity
        
        return 0.0
    
    def _calculate_weight_change_impact(self, index_data: pd.DataFrame, 
                                      price_data: pd.DataFrame) -> float:
        """Calculate impact of index weight changes."""
        if index_data.empty or len(index_data) < 5:
            return 0.0
        
        recent_index = index_data.tail(10)
        
        # Look for weight changes in major indices
        weight_changes = []
        
        for col in ['taiex_weight', 'twse50_weight']:
            if col in recent_index.columns:
                weights = recent_index[col].dropna()
                if len(weights) >= 2:
                    weight_change = weights.iloc[-1] - weights.iloc[0]
                    weight_changes.append(weight_change)
        
        if weight_changes:
            avg_weight_change = np.mean(weight_changes)
            return np.tanh(avg_weight_change * 100)  # Scale weight changes
        
        return 0.0
    
    def _calculate_passive_flow_anticipation(self, inclusion_prob: float, 
                                           rebalancing_effect: float, as_of_date: date) -> float:
        """Calculate anticipated passive flow effects."""
        # Combine inclusion probability with rebalancing timing
        base_anticipation = inclusion_prob * rebalancing_effect
        
        # Taiwan market has significant passive ETF flows
        # Scale by market structure factors
        taiwan_passive_multiplier = 1.2  # Taiwan has growing passive market
        
        return base_anticipation * taiwan_passive_multiplier
    
    def _get_current_index_status(self, symbol: str, index_data: pd.DataFrame, 
                                 as_of_date: date) -> Dict[str, Any]:
        """Get current index membership status."""
        if index_data.empty:
            return {'is_included': False, 'indices': []}
        
        latest_data = index_data.iloc[-1]
        indices = []
        
        if pd.notna(latest_data.get('taiex_weight', 0)) and latest_data.get('taiex_weight', 0) > 0:
            indices.append('TAIEX')
            
        if pd.notna(latest_data.get('twse50_weight', 0)) and latest_data.get('twse50_weight', 0) > 0:
            indices.append('TWSE50')
        
        return {
            'is_included': len(indices) > 0,
            'indices': indices
        }


class CrossStraitSentimentCalculator(MicrostructureFactorCalculator):
    """
    Cross-Strait Sentiment Calculator.
    
    Analyzes Taiwan-China relationship sentiment impacts:
    - News sentiment analysis
    - Policy announcement effects  
    - Trade relationship indicators
    - Market risk perception changes
    """
    
    def __init__(self, pit_engine: PITQueryEngine):
        metadata = FactorMetadata(
            name="cross_strait_sentiment",
            category=FactorCategory.MICROSTRUCTURE,
            frequency=FactorFrequency.DAILY,
            description="Cross-strait relationship sentiment impact on Taiwan market",
            lookback_days=63,  # ~3 months for sentiment analysis
            data_requirements=[DataType.NEWS_SENTIMENT, DataType.OHLCV],
            taiwan_specific=True,
            expected_ic=0.02,  # Sentiment typically has modest but persistent impact
            expected_turnover=0.10
        )
        super().__init__(pit_engine, metadata)
    
    def calculate(self, symbols: List[str], as_of_date: date,
                 universe_data: Optional[Dict[str, Any]] = None) -> FactorResult:
        """Calculate cross-strait sentiment factors."""
        
        # Get sentiment data (news, policy announcements, etc.)
        sentiment_data = self._get_historical_data(
            ['TAIWAN_SENTIMENT'], as_of_date, DataType.NEWS_SENTIMENT, self.metadata.lookback_days
        )
        
        # Get market data for sentiment impact analysis
        market_data = self._get_historical_data(
            ['TAIEX'], as_of_date, DataType.OHLCV, self.metadata.lookback_days
        )
        
        factor_values = {}
        
        try:
            # Calculate overall cross-strait sentiment metrics
            sentiment_score = self._calculate_sentiment_score(sentiment_data, as_of_date)
            sentiment_trend = self._calculate_sentiment_trend(sentiment_data)
            sentiment_volatility = self._calculate_sentiment_volatility(sentiment_data)
            market_impact = self._calculate_market_impact(sentiment_data, market_data)
            
            # Base cross-strait factor (applies to all Taiwan stocks)
            base_factor = (
                0.4 * sentiment_score +
                0.3 * sentiment_trend +
                0.2 * market_impact +
                0.1 * sentiment_volatility
            )
            
            # Apply sector-specific adjustments
            for symbol in symbols:
                try:
                    sector_adjustment = self._get_sector_adjustment(symbol)
                    stock_specific_factor = base_factor * (1 + sector_adjustment)
                    
                    factor_values[symbol] = stock_specific_factor
                    
                except Exception as e:
                    # Use base factor if sector adjustment fails
                    factor_values[symbol] = base_factor
        
        except Exception as e:
            logger.warning(f"Error calculating cross-strait sentiment: {e}")
            # Return zero factors if sentiment calculation fails
            for symbol in symbols:
                factor_values[symbol] = 0.0
        
        return FactorResult(
            factor_name=self.metadata.name,
            date=as_of_date,
            values=factor_values,
            metadata=self.metadata
        )
    
    def _calculate_sentiment_score(self, sentiment_data: pd.DataFrame, as_of_date: date) -> float:
        """Calculate current cross-strait sentiment score."""
        if sentiment_data.empty:
            return 0.0
        
        # Get recent sentiment (last 5 days)
        recent_sentiment = sentiment_data.tail(5)
        
        if 'sentiment_score' in recent_sentiment.columns:
            avg_sentiment = recent_sentiment['sentiment_score'].mean()
            return np.tanh(avg_sentiment)  # Bound between -1 and 1
        
        # Fallback: analyze keyword patterns
        return self._analyze_keyword_sentiment(recent_sentiment)
    
    def _calculate_sentiment_trend(self, sentiment_data: pd.DataFrame) -> float:
        """Calculate sentiment trend direction."""
        if len(sentiment_data) < 10:
            return 0.0
        
        if 'sentiment_score' in sentiment_data.columns:
            recent_avg = sentiment_data.tail(5)['sentiment_score'].mean()
            historical_avg = sentiment_data.head(10)['sentiment_score'].mean()
            
            trend = (recent_avg - historical_avg) / (abs(historical_avg) + 0.1)
            return np.tanh(trend)
        
        return 0.0
    
    def _calculate_sentiment_volatility(self, sentiment_data: pd.DataFrame) -> float:
        """Calculate sentiment volatility (uncertainty indicator)."""
        if len(sentiment_data) < 10:
            return 0.0
        
        if 'sentiment_score' in sentiment_data.columns:
            sentiment_vol = sentiment_data['sentiment_score'].std()
            
            # High volatility in sentiment indicates uncertainty (negative factor)
            return -np.tanh(sentiment_vol * 2)
        
        return 0.0
    
    def _calculate_market_impact(self, sentiment_data: pd.DataFrame, 
                               market_data: pd.DataFrame) -> float:
        """Calculate how sentiment correlates with market movements."""
        if sentiment_data.empty or market_data.empty or len(sentiment_data) < 20:
            return 0.0
        
        # Merge sentiment and market data
        try:
            merged = pd.merge(
                sentiment_data[['date', 'sentiment_score']], 
                market_data[['date', 'close']], 
                on='date', 
                how='inner'
            )
            
            if len(merged) < 15:
                return 0.0
            
            merged = merged.sort_values('date')
            merged['market_return'] = merged['close'].pct_change()
            
            # Calculate correlation between sentiment and next-day returns
            sentiment_values = merged['sentiment_score'].iloc[:-1].values
            next_returns = merged['market_return'].iloc[1:].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(sentiment_values) | np.isnan(next_returns))
            
            if np.sum(valid_mask) >= 10:
                correlation = np.corrcoef(sentiment_values[valid_mask], next_returns[valid_mask])[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
        
        except Exception as e:
            logger.warning(f"Error calculating sentiment-market correlation: {e}")
        
        return 0.0
    
    def _analyze_keyword_sentiment(self, sentiment_data: pd.DataFrame) -> float:
        """Analyze sentiment from keywords when sentiment scores not available."""
        if sentiment_data.empty:
            return 0.0
        
        # Positive keywords related to cross-strait relations
        positive_keywords = ['cooperation', 'agreement', 'trade', 'investment', 'dialogue']
        negative_keywords = ['tension', 'conflict', 'sanctions', 'military', 'dispute']
        
        total_positive = 0
        total_negative = 0
        
        if 'keyword_mentions' in sentiment_data.columns:
            for _, row in sentiment_data.iterrows():
                if pd.notna(row.get('keyword_mentions')):
                    # Simplified keyword analysis
                    keyword_dict = row.get('keyword_mentions', {})
                    
                    for keyword in positive_keywords:
                        total_positive += keyword_dict.get(keyword, 0)
                    
                    for keyword in negative_keywords:
                        total_negative += keyword_dict.get(keyword, 0)
        
        # Calculate sentiment score
        total_mentions = total_positive + total_negative
        if total_mentions > 0:
            sentiment_ratio = (total_positive - total_negative) / total_mentions
            return np.tanh(sentiment_ratio)
        
        return 0.0
    
    def _get_sector_adjustment(self, symbol: str) -> float:
        """Get sector-specific adjustment for cross-strait sentiment."""
        # Different sectors have different sensitivity to cross-strait relations
        # This is a simplified mapping - in practice would use proper sector classification
        
        sector_sensitivities = {
            # High sensitivity sectors
            'TECH': 0.3,      # Technology companies (export-dependent)
            'FINANCE': 0.2,   # Financial services  
            'SHIPPING': 0.25, # Shipping and logistics
            
            # Medium sensitivity
            'MANUFACTURING': 0.15,  # Manufacturing
            'RETAIL': 0.1,          # Retail and consumer
            
            # Lower sensitivity  
            'UTILITIES': 0.05,      # Utilities
            'TELECOM': 0.05,        # Telecommunications
        }
        
        # Simplified sector detection from symbol
        # In practice, would use proper sector classification data
        if any(tech_indicator in symbol.upper() for tech_indicator in ['TSM', 'UMC', 'ASE']):
            return sector_sensitivities['TECH']
        elif any(fin_indicator in symbol.upper() for fin_indicator in ['FUBON', 'CATHAY', 'SINOPAC']):
            return sector_sensitivities['FINANCE']
        else:
            return 0.1  # Default medium sensitivity


class TaiwanSpecificFactors:
    """Container for all Taiwan-specific factor calculators."""
    
    def __init__(self, pit_engine: PITQueryEngine):
        self.pit_engine = pit_engine
        self.calculators = {
            'foreign_flow_impact': ForeignFlowImpactCalculator(pit_engine),
            'margin_trading_ratio': MarginTradingRatioCalculator(pit_engine),
            'index_inclusion_effect': IndexInclusionEffectCalculator(pit_engine),
            'cross_strait_sentiment': CrossStraitSentimentCalculator(pit_engine)
        }
    
    def get_all_calculators(self) -> Dict[str, MicrostructureFactorCalculator]:
        """Get all Taiwan-specific factor calculators."""
        return self.calculators.copy()
    
    def calculate_all_factors(self, symbols: List[str], as_of_date: date) -> Dict[str, FactorResult]:
        """Calculate all Taiwan-specific factors."""
        results = {}
        
        for name, calculator in self.calculators.items():
            try:
                result = calculator.calculate(symbols, as_of_date)
                results[name] = result
                logger.info(f"Calculated {name}: {result.coverage:.1%} coverage")
            except Exception as e:
                logger.error(f"Error calculating Taiwan-specific factor {name}: {e}")
                continue
        
        return results