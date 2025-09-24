"""
Foreign institutional flow analysis for Taiwan market.

This module provides comprehensive analysis of foreign institutional investment
flows in Taiwan equity markets:
- Daily foreign buy/sell flow analysis
- Foreign ownership tracking and limits
- Flow impact on price discovery  
- Institutional flow pattern recognition
- Taiwan-specific foreign investment regulations compliance
"""

from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from .microstructure import ForeignFlowData
from .base import FactorResult, FactorMetadata

# Import dependencies - will be mocked for testing if not available  
try:
    from ..data.core.temporal import DataType
except ImportError:
    DataType = object

logger = logging.getLogger(__name__)


class FlowDirection(Enum):
    """Foreign flow direction classification."""
    NET_BUY = "net_buy"
    NET_SELL = "net_sell"  
    BALANCED = "balanced"


class FlowIntensity(Enum):
    """Foreign flow intensity levels."""
    VERY_HIGH = "very_high"      # >3 standard deviations
    HIGH = "high"                # >2 standard deviations  
    MODERATE = "moderate"        # >1 standard deviation
    LOW = "low"                  # <1 standard deviation


@dataclass
class FlowAnalysisResult:
    """Result of foreign flow analysis."""
    symbol: str
    analysis_date: date
    
    # Current flow metrics
    net_flow_value: float
    flow_direction: FlowDirection
    flow_intensity: FlowIntensity
    
    # Historical context
    flow_percentile: float          # Percentile vs historical flows
    momentum_score: float           # Flow persistence measure
    reversal_probability: float     # Likelihood of flow reversal
    
    # Ownership metrics  
    foreign_ownership_pct: float
    ownership_trend: float          # Change in ownership
    distance_to_cap: float          # Distance to 50% cap
    
    # Impact metrics
    price_impact_correlation: float # Correlation with price changes
    liquidity_impact: float         # Impact on trading liquidity
    market_timing_score: float      # Quality of foreign timing
    
    # Risk metrics
    concentration_risk: float       # Flow concentration in few days
    volatility_timing: float        # Flows during volatile periods


@dataclass
class ForeignInstitutionProfile:
    """Profile of foreign institutional activity."""
    total_institutions: int
    active_institutions: int       # Institutions trading in period
    flow_concentration_ratio: float # Share of flows from top institutions  
    avg_trade_size: float
    trading_frequency: float
    
    # Behavioral patterns
    momentum_followers: int         # Count following trends
    contrarian_traders: int        # Count going against trends  
    flow_correlation: float        # Correlation between institutions


class ForeignFlowAnalyzer:
    """Comprehensive foreign flow analysis engine."""
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize foreign flow analyzer.
        
        Args:
            lookback_days: Days of historical data for analysis
        """
        self.lookback_days = lookback_days
        self.foreign_cap = 0.50  # Taiwan 50% foreign ownership cap
        self.flow_history = defaultdict(deque)  # symbol -> flow history
        
    def analyze_flows(self, flow_data: List[ForeignFlowData], 
                     analysis_date: date) -> Dict[str, FlowAnalysisResult]:
        """
        Analyze foreign flows for all symbols.
        
        Args:
            flow_data: List of foreign flow data points
            analysis_date: Date for analysis
            
        Returns:
            Dictionary mapping symbols to analysis results
        """
        # Group flows by symbol
        symbol_flows = defaultdict(list)
        for flow in flow_data:
            symbol_flows[flow.symbol].append(flow)
        
        results = {}
        
        for symbol, flows in symbol_flows.items():
            try:
                # Sort flows by date
                flows.sort(key=lambda x: x.date)
                
                # Analyze individual symbol
                result = self._analyze_symbol_flows(symbol, flows, analysis_date)
                results[symbol] = result
                
            except Exception as e:
                logger.warning(f"Error analyzing flows for {symbol}: {e}")
                continue
        
        return results
    
    def _analyze_symbol_flows(self, symbol: str, flows: List[ForeignFlowData],
                            analysis_date: date) -> FlowAnalysisResult:
        """Analyze flows for a single symbol."""
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([{
            'date': f.date,
            'net_value': f.foreign_net_value,
            'buy_value': f.foreign_buy_value,
            'sell_value': f.foreign_sell_value,
            'ownership_pct': f.foreign_ownership_pct
        } for f in flows])
        
        df = df.sort_values('date')
        
        # Get current flow metrics
        current_flow = df[df['date'] == analysis_date]
        if current_flow.empty:
            # Use most recent available data
            current_flow = df.tail(1)
        
        net_flow_value = current_flow['net_value'].iloc[0]
        current_ownership = current_flow['ownership_pct'].iloc[0]
        
        # Classify flow direction and intensity
        flow_direction = self._classify_flow_direction(net_flow_value)
        flow_intensity = self._classify_flow_intensity(net_flow_value, df['net_value'])
        
        # Calculate historical context metrics
        flow_percentile = self._calculate_flow_percentile(net_flow_value, df['net_value'])
        momentum_score = self._calculate_momentum_score(df)
        reversal_probability = self._calculate_reversal_probability(df)
        
        # Calculate ownership metrics
        ownership_trend = self._calculate_ownership_trend(df)
        distance_to_cap = self.foreign_cap - current_ownership
        
        # Calculate impact metrics (would require price data in practice)
        price_impact_correlation = self._estimate_price_impact_correlation(df)
        liquidity_impact = self._estimate_liquidity_impact(df)
        market_timing_score = self._calculate_market_timing_score(df)
        
        # Calculate risk metrics
        concentration_risk = self._calculate_concentration_risk(df)
        volatility_timing = self._estimate_volatility_timing(df)
        
        return FlowAnalysisResult(
            symbol=symbol,
            analysis_date=analysis_date,
            net_flow_value=net_flow_value,
            flow_direction=flow_direction,
            flow_intensity=flow_intensity,
            flow_percentile=flow_percentile,
            momentum_score=momentum_score,
            reversal_probability=reversal_probability,
            foreign_ownership_pct=current_ownership,
            ownership_trend=ownership_trend,
            distance_to_cap=distance_to_cap,
            price_impact_correlation=price_impact_correlation,
            liquidity_impact=liquidity_impact,
            market_timing_score=market_timing_score,
            concentration_risk=concentration_risk,
            volatility_timing=volatility_timing
        )
    
    def _classify_flow_direction(self, net_flow: float) -> FlowDirection:
        """Classify flow direction."""
        threshold = 1e6  # 1 million TWD threshold
        
        if net_flow > threshold:
            return FlowDirection.NET_BUY
        elif net_flow < -threshold:
            return FlowDirection.NET_SELL
        else:
            return FlowDirection.BALANCED
    
    def _classify_flow_intensity(self, current_flow: float, 
                               historical_flows: pd.Series) -> FlowIntensity:
        """Classify flow intensity based on historical distribution."""
        if len(historical_flows) < 10:
            return FlowIntensity.LOW
        
        flow_std = historical_flows.std()
        flow_mean = historical_flows.mean()
        
        if flow_std == 0:
            return FlowIntensity.LOW
        
        z_score = abs((current_flow - flow_mean) / flow_std)
        
        if z_score > 3:
            return FlowIntensity.VERY_HIGH
        elif z_score > 2:
            return FlowIntensity.HIGH
        elif z_score > 1:
            return FlowIntensity.MODERATE
        else:
            return FlowIntensity.LOW
    
    def _calculate_flow_percentile(self, current_flow: float, 
                                 historical_flows: pd.Series) -> float:
        """Calculate percentile of current flow vs historical."""
        if len(historical_flows) < 10:
            return 0.5
        
        return (historical_flows < current_flow).sum() / len(historical_flows)
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate flow momentum (persistence) score."""
        if len(df) < 10:
            return 0.0
        
        recent_flows = df.tail(10)['net_value']
        
        # Count consecutive days with same flow direction
        flow_signs = np.sign(recent_flows)
        momentum_score = 0.0
        
        current_streak = 1
        for i in range(1, len(flow_signs)):
            if flow_signs.iloc[i] == flow_signs.iloc[i-1] and flow_signs.iloc[i] != 0:
                current_streak += 1
            else:
                current_streak = 1
        
        # Normalize momentum score
        momentum_score = min(current_streak / 10.0, 1.0)
        
        return momentum_score
    
    def _calculate_reversal_probability(self, df: pd.DataFrame) -> float:
        """Calculate probability of flow reversal."""
        if len(df) < 20:
            return 0.5
        
        # Look at historical patterns after similar flow sequences
        recent_pattern = df.tail(5)['net_value'].values
        historical_data = df.head(-5)
        
        # Simple pattern matching (in practice would use more sophisticated methods)
        reversal_count = 0
        pattern_count = 0
        
        for i in range(len(historical_data) - 10):
            hist_pattern = historical_data.iloc[i:i+5]['net_value'].values
            
            # Check if patterns are similar (correlation > 0.5)
            if len(hist_pattern) == len(recent_pattern):
                correlation = np.corrcoef(hist_pattern, recent_pattern)[0, 1]
                
                if not np.isnan(correlation) and correlation > 0.5:
                    pattern_count += 1
                    
                    # Check if next period had flow reversal
                    if i + 6 < len(historical_data):
                        next_flow = historical_data.iloc[i + 5]['net_value']
                        recent_flow = recent_pattern[-1]
                        
                        # Consider it a reversal if signs are opposite
                        if np.sign(next_flow) != np.sign(recent_flow) and recent_flow != 0:
                            reversal_count += 1
        
        if pattern_count > 0:
            return reversal_count / pattern_count
        else:
            return 0.5  # Default neutral probability
    
    def _calculate_ownership_trend(self, df: pd.DataFrame) -> float:
        """Calculate foreign ownership trend."""
        if len(df) < 5:
            return 0.0
        
        recent_ownership = df.tail(10)['ownership_pct']
        
        if len(recent_ownership) >= 2:
            # Simple linear trend
            x = np.arange(len(recent_ownership))
            y = recent_ownership.values
            
            if len(x) > 1:
                slope, _ = np.polyfit(x, y, 1)
                return slope
        
        return 0.0
    
    def _estimate_price_impact_correlation(self, df: pd.DataFrame) -> float:
        """Estimate correlation between flows and price impact."""
        # Placeholder - in practice would correlate with actual price data
        # For now, use flow volatility as proxy
        
        if len(df) < 10:
            return 0.0
        
        flow_volatility = df['net_value'].std()
        flow_magnitude = df['net_value'].abs().mean()
        
        if flow_magnitude > 0:
            # Higher volatility relative to magnitude suggests price impact
            impact_proxy = flow_volatility / flow_magnitude
            return np.tanh(impact_proxy)  # Bound between -1 and 1
        
        return 0.0
    
    def _estimate_liquidity_impact(self, df: pd.DataFrame) -> float:
        """Estimate impact on market liquidity."""
        if len(df) < 10:
            return 0.0
        
        # Large flows typically reduce liquidity
        recent_flows = df.tail(5)['net_value'].abs()
        historical_flows = df['net_value'].abs()
        
        if len(historical_flows) > 0:
            avg_historical_flow = historical_flows.mean()
            avg_recent_flow = recent_flows.mean()
            
            if avg_historical_flow > 0:
                liquidity_impact = (avg_recent_flow / avg_historical_flow - 1) * 0.5
                return np.tanh(liquidity_impact)
        
        return 0.0
    
    def _calculate_market_timing_score(self, df: pd.DataFrame) -> float:
        """Calculate quality of foreign institutional timing."""
        # Placeholder for market timing analysis
        # In practice would compare flow timing with subsequent returns
        
        if len(df) < 20:
            return 0.0
        
        # Simple proxy: consistency of flow direction
        flows = df['net_value']
        flow_consistency = abs(flows.sum()) / flows.abs().sum() if flows.abs().sum() > 0 else 0
        
        return flow_consistency
    
    def _calculate_concentration_risk(self, df: pd.DataFrame) -> float:
        """Calculate flow concentration risk."""
        if len(df) < 10:
            return 0.0
        
        recent_flows = df.tail(5)['net_value'].abs()
        total_recent_flow = recent_flows.sum()
        
        if total_recent_flow == 0:
            return 0.0
        
        # Calculate what fraction of recent flows came from largest flow days
        sorted_flows = recent_flows.sort_values(ascending=False)
        top_flow = sorted_flows.iloc[0] if len(sorted_flows) > 0 else 0
        
        concentration = top_flow / total_recent_flow if total_recent_flow > 0 else 0
        return concentration
    
    def _estimate_volatility_timing(self, df: pd.DataFrame) -> float:
        """Estimate foreign flow timing relative to market volatility."""
        # Placeholder - in practice would use actual volatility data
        
        if len(df) < 10:
            return 0.0
        
        # Use flow variability as proxy for volatility timing
        flow_cv = df['net_value'].std() / (abs(df['net_value'].mean()) + 1e-6)
        
        # Higher coefficient of variation suggests poor volatility timing
        return -np.tanh(flow_cv - 1)  # Negative score for high variability


class ForeignFlowRegimeDetector:
    """Detect regime changes in foreign institutional flows."""
    
    def __init__(self, window_size: int = 20, sensitivity: float = 2.0):
        """
        Initialize regime detector.
        
        Args:
            window_size: Rolling window for regime detection
            sensitivity: Sensitivity threshold (standard deviations)
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
    
    def detect_regime_changes(self, flows: pd.Series, 
                            dates: pd.Series) -> List[Tuple[date, str, float]]:
        """
        Detect regime changes in foreign flows.
        
        Returns:
            List of (date, regime_type, confidence) tuples
        """
        if len(flows) < self.window_size * 2:
            return []
        
        regime_changes = []
        
        for i in range(self.window_size, len(flows) - self.window_size):
            # Compare recent window with historical window
            recent_flows = flows.iloc[i-self.window_size:i]
            historical_flows = flows.iloc[:i-self.window_size]
            
            if len(historical_flows) < self.window_size:
                continue
            
            # Calculate statistical differences
            recent_mean = recent_flows.mean()
            recent_std = recent_flows.std()
            historical_mean = historical_flows.mean()
            historical_std = historical_flows.std()
            
            # Detect mean shift
            if historical_std > 0:
                mean_shift_zscore = abs(recent_mean - historical_mean) / historical_std
                
                if mean_shift_zscore > self.sensitivity:
                    regime_type = "mean_shift_positive" if recent_mean > historical_mean else "mean_shift_negative"
                    confidence = min(mean_shift_zscore / self.sensitivity, 3.0) / 3.0
                    
                    regime_changes.append((dates.iloc[i], regime_type, confidence))
            
            # Detect volatility shift  
            if historical_std > 0 and recent_std > 0:
                vol_ratio = recent_std / historical_std
                
                if vol_ratio > (1 + self.sensitivity * 0.5) or vol_ratio < (1 - self.sensitivity * 0.5):
                    regime_type = "volatility_increase" if vol_ratio > 1 else "volatility_decrease"
                    confidence = min(abs(vol_ratio - 1) * 2, 1.0)
                    
                    regime_changes.append((dates.iloc[i], regime_type, confidence))
        
        return regime_changes


class ForeignFlowForecaster:
    """Simple forecasting model for foreign institutional flows."""
    
    def __init__(self, model_type: str = "momentum"):
        """
        Initialize flow forecaster.
        
        Args:
            model_type: Type of forecasting model ("momentum", "mean_reversion", "mixed")
        """
        self.model_type = model_type
    
    def forecast_flows(self, historical_flows: pd.Series, 
                      horizon_days: int = 5) -> Dict[str, float]:
        """
        Forecast future flows based on historical patterns.
        
        Args:
            historical_flows: Historical flow time series
            horizon_days: Forecast horizon in days
            
        Returns:
            Dictionary with forecast statistics
        """
        if len(historical_flows) < 20:
            return {"forecast": 0.0, "confidence": 0.0, "direction": "neutral"}
        
        forecast_result = {}
        
        if self.model_type == "momentum":
            forecast_result = self._momentum_forecast(historical_flows, horizon_days)
        elif self.model_type == "mean_reversion":
            forecast_result = self._mean_reversion_forecast(historical_flows, horizon_days)
        else:  # mixed model
            momentum_forecast = self._momentum_forecast(historical_flows, horizon_days)
            reversion_forecast = self._mean_reversion_forecast(historical_flows, horizon_days)
            
            # Weighted combination
            forecast_result = {
                "forecast": 0.6 * momentum_forecast["forecast"] + 0.4 * reversion_forecast["forecast"],
                "confidence": (momentum_forecast["confidence"] + reversion_forecast["confidence"]) / 2,
                "direction": momentum_forecast["direction"]
            }
        
        return forecast_result
    
    def _momentum_forecast(self, flows: pd.Series, horizon: int) -> Dict[str, float]:
        """Momentum-based forecast."""
        recent_flows = flows.tail(10)
        
        # Simple momentum: recent trend continues
        if len(recent_flows) >= 2:
            trend = recent_flows.iloc[-1] - recent_flows.iloc[0]
            forecast = trend * (horizon / len(recent_flows))
            
            # Confidence based on trend consistency
            flow_changes = recent_flows.diff().dropna()
            consistent_direction = (flow_changes > 0).sum() if trend > 0 else (flow_changes < 0).sum()
            confidence = consistent_direction / len(flow_changes) if len(flow_changes) > 0 else 0
            
            direction = "buy" if forecast > 0 else "sell" if forecast < 0 else "neutral"
            
            return {"forecast": forecast, "confidence": confidence, "direction": direction}
        
        return {"forecast": 0.0, "confidence": 0.0, "direction": "neutral"}
    
    def _mean_reversion_forecast(self, flows: pd.Series, horizon: int) -> Dict[str, float]:
        """Mean reversion forecast."""
        long_term_mean = flows.mean()
        recent_level = flows.tail(5).mean()
        
        # Forecast reversion to mean
        reversion_speed = 0.1  # 10% reversion per period
        forecast = (long_term_mean - recent_level) * reversion_speed * horizon
        
        # Confidence based on deviation from mean
        deviation = abs(recent_level - long_term_mean)
        historical_std = flows.std()
        
        confidence = min(deviation / (historical_std + 1e-6), 2.0) / 2.0 if historical_std > 0 else 0
        direction = "buy" if forecast > 0 else "sell" if forecast < 0 else "neutral"
        
        return {"forecast": forecast, "confidence": confidence, "direction": direction}