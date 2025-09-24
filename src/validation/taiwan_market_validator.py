"""
Taiwan Market-Specific Validation Extensions

Specialized validation components for Taiwan equity market with:
- T+2 settlement impact analysis
- Price limit compliance validation  
- Market structure specific metrics
- Regulatory compliance checks
- Foreign ownership impact analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

from .statistical_validator import ValidationConfig, ValidationResults

logger = logging.getLogger(__name__)


@dataclass
class TaiwanMarketConfig:
    """Taiwan market-specific configuration parameters."""
    
    # Market structure
    trading_hours: Dict[str, float] = field(default_factory=lambda: {'start': 9.0, 'end': 13.5, 'duration': 4.5})
    settlement_cycle: int = 2  # T+2
    price_limits: Dict[str, float] = field(default_factory=lambda: {'daily': 0.10, 'warning': 0.095})
    
    # Market segments
    main_board_min_market_cap: float = 5e8  # TWD 500M
    otc_market_threshold: float = 1e8  # TWD 100M
    
    # Foreign ownership limits
    foreign_ownership_limit: float = 0.50  # 50% general limit
    foreign_ownership_warning: float = 0.45  # 45% warning threshold
    
    # Regulatory parameters
    insider_trading_lookback: int = 30  # Days
    margin_trading_threshold: float = 0.30  # 30% margin requirement
    
    # Market timing
    market_holidays: List[str] = field(default_factory=list)  # Will be loaded dynamically
    quarterly_earnings_months: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    
    # Sector classifications (TWSE industry codes)
    technology_sectors: List[str] = field(default_factory=lambda: ['24', '26', '27', '31'])
    financial_sectors: List[str] = field(default_factory=lambda: ['17', '18'])
    traditional_sectors: List[str] = field(default_factory=lambda: ['08', '09', '10', '11', '14'])


class TaiwanSettlementValidator:
    """Validator for T+2 settlement impact on model performance."""
    
    def __init__(self, config: TaiwanMarketConfig):
        self.config = config
        
    def analyze_settlement_impact(
        self,
        predictions: pd.Series,
        returns_t0: pd.Series,
        returns_t1: pd.Series,
        returns_t2: pd.Series,
        trading_volumes: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze impact of T+2 settlement on model predictions.
        
        Args:
            predictions: Model predictions at T
            returns_t0: Returns at T (same day)
            returns_t1: Returns at T+1
            returns_t2: Returns at T+2 (settlement day)
            trading_volumes: Trading volume data
            
        Returns:
            Settlement impact analysis results
        """
        results = {}
        
        # Align data
        aligned_data = pd.DataFrame({
            'pred': predictions,
            'ret_t0': returns_t0,
            'ret_t1': returns_t1,
            'ret_t2': returns_t2,
            'volume': trading_volumes
        }).dropna()
        
        if len(aligned_data) < 30:
            return {'error': 'insufficient_data'}
        
        # Calculate IC at different horizons
        ic_t0, _ = stats.spearmanr(aligned_data['pred'], aligned_data['ret_t0'])
        ic_t1, _ = stats.spearmanr(aligned_data['pred'], aligned_data['ret_t1'])
        ic_t2, _ = stats.spearmanr(aligned_data['pred'], aligned_data['ret_t2'])
        
        # Settlement decay pattern
        ic_decay = {
            't0': ic_t0,
            't1': ic_t1,
            't2': ic_t2,
            'decay_rate_t1': (ic_t0 - ic_t1) / abs(ic_t0) if abs(ic_t0) > 1e-6 else 0,
            'decay_rate_t2': (ic_t0 - ic_t2) / abs(ic_t0) if abs(ic_t0) > 1e-6 else 0
        }
        
        # Cumulative returns analysis
        aligned_data['cum_ret_t2'] = aligned_data['ret_t0'] + aligned_data['ret_t1'] + aligned_data['ret_t2']
        ic_cumulative, _ = stats.spearmanr(aligned_data['pred'], aligned_data['cum_ret_t2'])
        
        # Volume impact on settlement
        high_volume_mask = aligned_data['volume'] > aligned_data['volume'].quantile(0.8)
        low_volume_mask = aligned_data['volume'] < aligned_data['volume'].quantile(0.2)
        
        ic_high_vol, _ = stats.spearmanr(
            aligned_data.loc[high_volume_mask, 'pred'],
            aligned_data.loc[high_volume_mask, 'ret_t2']
        ) if high_volume_mask.sum() > 10 else (0, 1)
        
        ic_low_vol, _ = stats.spearmanr(
            aligned_data.loc[low_volume_mask, 'pred'],
            aligned_data.loc[low_volume_mask, 'ret_t2']
        ) if low_volume_mask.sum() > 10 else (0, 1)
        
        results['settlement_analysis'] = {
            'ic_decay_pattern': ic_decay,
            'ic_cumulative_t2': ic_cumulative,
            'volume_impact': {
                'ic_high_volume': ic_high_vol,
                'ic_low_volume': ic_low_vol,
                'volume_effect': ic_high_vol - ic_low_vol
            },
            'settlement_efficiency': abs(ic_t2 / ic_t0) if abs(ic_t0) > 1e-6 else 0
        }
        
        # Trading cost implications
        results['trading_cost_impact'] = self._estimate_trading_costs(aligned_data)
        
        return results
    
    def _estimate_trading_costs(self, data: pd.DataFrame) -> Dict[str, float]:
        """Estimate trading costs impact on T+2 settlement."""
        # Simplified trading cost model for Taiwan market
        
        # Volume-based impact cost (Taiwan market structure)
        avg_daily_volume = data['volume'].mean()
        volume_impact_bps = min(50, 1000 / (avg_daily_volume / 1e6))  # Max 50 bps
        
        # Market impact during settlement window
        settlement_impact_bps = 5.0  # Average 5 bps for T+2 settlement
        
        # Spread cost (Taiwan market typical bid-ask spreads)
        spread_cost_bps = 15.0  # Average 15 bps spread
        
        total_cost_bps = volume_impact_bps + settlement_impact_bps + spread_cost_bps
        
        return {
            'volume_impact_bps': volume_impact_bps,
            'settlement_impact_bps': settlement_impact_bps,
            'spread_cost_bps': spread_cost_bps,
            'total_cost_bps': total_cost_bps,
            'cost_impact_on_ic': min(0.2, total_cost_bps / 1000)  # Max 20% IC degradation
        }


class PriceLimitValidator:
    """Validator for Taiwan daily price limit impact."""
    
    def __init__(self, config: TaiwanMarketConfig):
        self.config = config
        
    def analyze_price_limit_impact(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        prices: pd.DataFrame,  # OHLC data
        limit_hit_indicators: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Analyze impact of ±10% daily price limits on model performance.
        
        Args:
            predictions: Model predictions
            returns: Daily returns
            prices: OHLC price data
            limit_hit_indicators: Optional pre-computed limit hit indicators
            
        Returns:
            Price limit impact analysis
        """
        # Calculate limit hit indicators if not provided
        if limit_hit_indicators is None:
            limit_hit_indicators = self._identify_limit_hits(prices, returns)
        
        # Align data
        aligned_data = pd.DataFrame({
            'pred': predictions,
            'ret': returns,
            'limit_up': limit_hit_indicators == 1,
            'limit_down': limit_hit_indicators == -1,
            'near_limit_up': (returns > self.config.price_limits['warning']),
            'near_limit_down': (returns < -self.config.price_limits['warning'])
        }).dropna()
        
        if len(aligned_data) < 50:
            return {'error': 'insufficient_data'}
        
        results = {}
        
        # Overall impact statistics
        limit_up_pct = aligned_data['limit_up'].mean()
        limit_down_pct = aligned_data['limit_down'].mean()
        near_limit_pct = (aligned_data['near_limit_up'] | aligned_data['near_limit_down']).mean()
        
        results['limit_statistics'] = {
            'limit_up_frequency': limit_up_pct,
            'limit_down_frequency': limit_down_pct,
            'near_limit_frequency': near_limit_pct,
            'total_limit_events': limit_up_pct + limit_down_pct
        }
        
        # Model performance on limit days
        limit_days = aligned_data['limit_up'] | aligned_data['limit_down']
        normal_days = ~limit_days
        
        if limit_days.sum() > 5 and normal_days.sum() > 5:
            ic_limit_days, _ = stats.spearmanr(
                aligned_data.loc[limit_days, 'pred'],
                aligned_data.loc[limit_days, 'ret']
            )
            ic_normal_days, _ = stats.spearmanr(
                aligned_data.loc[normal_days, 'pred'],
                aligned_data.loc[normal_days, 'ret']
            )
            
            results['performance_impact'] = {
                'ic_limit_days': ic_limit_days,
                'ic_normal_days': ic_normal_days,
                'ic_degradation': ic_normal_days - ic_limit_days,
                'relative_impact': (ic_normal_days - ic_limit_days) / abs(ic_normal_days) if abs(ic_normal_days) > 1e-6 else 0
            }
        
        # Prediction accuracy near limits
        results['prediction_accuracy'] = self._analyze_limit_prediction_accuracy(aligned_data)
        
        # Limit clustering analysis
        results['clustering_analysis'] = self._analyze_limit_clustering(aligned_data)
        
        return results
    
    def _identify_limit_hits(self, prices: pd.DataFrame, returns: pd.Series) -> pd.Series:
        """Identify days when stocks hit price limits."""
        limit_indicators = pd.Series(0, index=returns.index)
        
        # Upper limit: return ≥ 9.5% (close to 10% limit)
        limit_indicators[returns >= self.config.price_limits['warning']] = 1
        
        # Lower limit: return ≤ -9.5%
        limit_indicators[returns <= -self.config.price_limits['warning']] = -1
        
        return limit_indicators
    
    def _analyze_limit_prediction_accuracy(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze prediction accuracy for limit events."""
        results = {}
        
        # Accuracy for predicting limit up events
        limit_up_predictions = data['pred'] > data['pred'].quantile(0.95)  # Top 5% predictions
        limit_up_actual = data['limit_up']
        
        if limit_up_actual.sum() > 0:
            precision_up = (limit_up_predictions & limit_up_actual).sum() / limit_up_predictions.sum()
            recall_up = (limit_up_predictions & limit_up_actual).sum() / limit_up_actual.sum()
        else:
            precision_up = recall_up = 0.0
        
        # Accuracy for predicting limit down events
        limit_down_predictions = data['pred'] < data['pred'].quantile(0.05)  # Bottom 5% predictions
        limit_down_actual = data['limit_down']
        
        if limit_down_actual.sum() > 0:
            precision_down = (limit_down_predictions & limit_down_actual).sum() / limit_down_predictions.sum()
            recall_down = (limit_down_predictions & limit_down_actual).sum() / limit_down_actual.sum()
        else:
            precision_down = recall_down = 0.0
        
        results['limit_up_precision'] = precision_up
        results['limit_up_recall'] = recall_up
        results['limit_down_precision'] = precision_down
        results['limit_down_recall'] = recall_down
        
        # F1 scores
        results['f1_limit_up'] = 2 * precision_up * recall_up / (precision_up + recall_up) if (precision_up + recall_up) > 0 else 0
        results['f1_limit_down'] = 2 * precision_down * recall_down / (precision_down + recall_down) if (precision_down + recall_down) > 0 else 0
        
        return results
    
    def _analyze_limit_clustering(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze clustering of limit events."""
        limit_events = data['limit_up'] | data['limit_down']
        
        if limit_events.sum() < 3:
            return {'clustering_coefficient': 0.0}
        
        # Calculate clustering coefficient (simplified)
        limit_dates = limit_events[limit_events].index
        
        # Find consecutive limit events
        consecutive_events = 0
        total_events = len(limit_dates)
        
        for i in range(1, len(limit_dates)):
            if (limit_dates[i] - limit_dates[i-1]).days <= 3:  # Within 3 days
                consecutive_events += 1
        
        clustering_coefficient = consecutive_events / total_events if total_events > 0 else 0.0
        
        return {
            'clustering_coefficient': clustering_coefficient,
            'consecutive_events': consecutive_events,
            'total_events': total_events,
            'avg_gap_between_events': np.mean([(limit_dates[i] - limit_dates[i-1]).days 
                                             for i in range(1, len(limit_dates))]) if len(limit_dates) > 1 else 0
        }


class MarketStructureValidator:
    """Validator for Taiwan market structure specifics."""
    
    def __init__(self, config: TaiwanMarketConfig):
        self.config = config
        
    def analyze_sector_performance(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        sector_mapping: pd.Series,
        market_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze model performance across Taiwan market sectors.
        
        Args:
            predictions: Model predictions
            returns: Actual returns
            sector_mapping: Stock to sector mapping
            market_data: Additional market data
            
        Returns:
            Sector-wise performance analysis
        """
        # Align data with sector information
        aligned_data = pd.DataFrame({
            'pred': predictions,
            'ret': returns,
            'sector': sector_mapping
        }).dropna()
        
        sector_performance = {}
        
        for sector in aligned_data['sector'].unique():
            sector_data = aligned_data[aligned_data['sector'] == sector]
            
            if len(sector_data) < 10:
                continue
            
            # Basic performance metrics
            ic, ic_p_value = stats.spearmanr(sector_data['pred'], sector_data['ret'])
            hit_rate = np.mean(np.sign(sector_data['pred']) == np.sign(sector_data['ret']))
            
            # Sector-specific metrics
            sector_vol = sector_data['ret'].std() * np.sqrt(252)
            sector_return = sector_data['ret'].mean() * 252
            sharpe = sector_return / sector_vol if sector_vol > 0 else 0
            
            # Technology sector specific analysis
            if sector in self.config.technology_sectors:
                tech_metrics = self._analyze_technology_sector_specifics(sector_data, market_data)
            else:
                tech_metrics = {}
            
            sector_performance[sector] = {
                'ic': ic,
                'ic_p_value': ic_p_value,
                'ic_significant': ic_p_value < 0.05,
                'hit_rate': hit_rate,
                'annual_return': sector_return,
                'annual_volatility': sector_vol,
                'sharpe_ratio': sharpe,
                'n_stocks': len(sector_data),
                'sector_weight': len(sector_data) / len(aligned_data),
                **tech_metrics
            }
        
        return sector_performance
    
    def _analyze_technology_sector_specifics(
        self,
        sector_data: pd.DataFrame,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze Taiwan technology sector specific patterns."""
        
        # Technology sector typically has higher beta to global tech indices
        # and is more sensitive to US market movements
        
        results = {}
        
        if 'us_tech_returns' in market_data.columns:
            # Correlation with US tech sector
            aligned_global = pd.merge(sector_data, market_data[['us_tech_returns']], 
                                    left_index=True, right_index=True, how='inner')
            
            if len(aligned_global) > 20:
                global_corr, _ = stats.pearsonr(aligned_global['ret'], aligned_global['us_tech_returns'])
                results['us_tech_correlation'] = global_corr
                
                # Beta to US tech
                beta = np.cov(aligned_global['ret'], aligned_global['us_tech_returns'])[0, 1] / np.var(aligned_global['us_tech_returns'])
                results['us_tech_beta'] = beta
        
        # Technology sector volatility clustering
        returns_sq = sector_data['ret'] ** 2
        vol_clustering = returns_sq.autocorr(lag=1) if len(returns_sq) > 1 else 0
        results['volatility_clustering'] = vol_clustering
        
        return results
    
    def analyze_market_timing(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        timestamps: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze model performance across Taiwan trading session timing.
        
        Args:
            predictions: Intraday predictions
            returns: Intraday returns
            timestamps: Timestamp information
            
        Returns:
            Market timing analysis
        """
        # Extract hour from timestamps
        hours = pd.to_datetime(timestamps).dt.hour + pd.to_datetime(timestamps).dt.minute / 60.0
        
        aligned_data = pd.DataFrame({
            'pred': predictions,
            'ret': returns,
            'hour': hours
        }).dropna()
        
        results = {}
        
        # Define Taiwan market sessions
        morning_session = (aligned_data['hour'] >= 9.0) & (aligned_data['hour'] < 12.0)
        afternoon_session = (aligned_data['hour'] >= 13.0) & (aligned_data['hour'] <= 13.5)
        
        # Performance by session
        if morning_session.sum() > 10:
            morning_ic, _ = stats.spearmanr(
                aligned_data.loc[morning_session, 'pred'],
                aligned_data.loc[morning_session, 'ret']
            )
            results['morning_session_ic'] = morning_ic
        
        if afternoon_session.sum() > 10:
            afternoon_ic, _ = stats.spearmanr(
                aligned_data.loc[afternoon_session, 'pred'],
                aligned_data.loc[afternoon_session, 'ret']
            )
            results['afternoon_session_ic'] = afternoon_ic
        
        # Hourly performance pattern
        hourly_performance = {}
        for hour in np.arange(9.0, 14.0, 0.5):  # 30-minute intervals
            hour_mask = (aligned_data['hour'] >= hour) & (aligned_data['hour'] < hour + 0.5)
            
            if hour_mask.sum() > 5:
                hour_ic, _ = stats.spearmanr(
                    aligned_data.loc[hour_mask, 'pred'],
                    aligned_data.loc[hour_mask, 'ret']
                )
                hourly_performance[f'{hour:.1f}'] = hour_ic
        
        results['hourly_ic_pattern'] = hourly_performance
        
        return results


class TaiwanMarketValidator:
    """Comprehensive Taiwan market-specific validator."""
    
    def __init__(self, config: Optional[TaiwanMarketConfig] = None):
        self.config = config or TaiwanMarketConfig()
        self.settlement_validator = TaiwanSettlementValidator(self.config)
        self.limit_validator = PriceLimitValidator(self.config)
        self.structure_validator = MarketStructureValidator(self.config)
        
        logger.info("Taiwan Market Validator initialized")
    
    def comprehensive_taiwan_validation(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        market_data: pd.DataFrame,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive Taiwan market-specific validation.
        
        Args:
            predictions: Model predictions
            returns: Actual returns
            market_data: Comprehensive market data
            additional_data: Additional data for specific validations
            
        Returns:
            Comprehensive Taiwan market validation results
        """
        results = {
            'timestamp': datetime.now(),
            'validation_type': 'taiwan_market_comprehensive',
            'market_config': {
                'trading_hours': self.config.trading_hours,
                'settlement_cycle': self.config.settlement_cycle,
                'price_limits': self.config.price_limits
            }
        }
        
        # Settlement impact analysis
        if all(col in market_data.columns for col in ['ret_t0', 'ret_t1', 'ret_t2', 'volume']):
            results['settlement_analysis'] = self.settlement_validator.analyze_settlement_impact(
                predictions,
                market_data['ret_t0'],
                market_data['ret_t1'], 
                market_data['ret_t2'],
                market_data['volume']
            )
        
        # Price limit impact analysis
        if 'prices' in market_data.columns or 'ohlc' in additional_data:
            price_data = market_data.get('prices', additional_data.get('ohlc'))
            results['price_limit_analysis'] = self.limit_validator.analyze_price_limit_impact(
                predictions, returns, price_data
            )
        
        # Market structure analysis
        if 'sector' in market_data.columns:
            results['sector_analysis'] = self.structure_validator.analyze_sector_performance(
                predictions, returns, market_data['sector'], market_data
            )
        
        # Market timing analysis
        if 'timestamp' in market_data.columns:
            results['timing_analysis'] = self.structure_validator.analyze_market_timing(
                predictions, returns, market_data['timestamp']
            )
        
        # Regulatory compliance checks
        results['compliance_check'] = self._perform_regulatory_compliance_check(
            predictions, returns, market_data, additional_data
        )
        
        # Generate Taiwan-specific recommendations
        results['taiwan_recommendations'] = self._generate_taiwan_recommendations(results)
        
        return results
    
    def _perform_regulatory_compliance_check(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        market_data: pd.DataFrame,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform regulatory compliance checks for Taiwan market."""
        compliance_results = {}
        
        # Position concentration check
        if 'position_sizes' in additional_data:
            position_sizes = additional_data['position_sizes']
            max_position = position_sizes.max()
            compliance_results['max_position_concentration'] = max_position
            compliance_results['position_limit_compliant'] = max_position <= 0.10  # 10% limit
        
        # Foreign ownership impact
        if 'foreign_ownership' in market_data.columns:
            high_foreign_ownership = market_data['foreign_ownership'] > self.config.foreign_ownership_warning
            
            if high_foreign_ownership.sum() > 0:
                aligned_data = pd.DataFrame({
                    'pred': predictions,
                    'ret': returns,
                    'high_foreign': high_foreign_ownership
                }).dropna()
                
                high_foreign_ic, _ = stats.spearmanr(
                    aligned_data.loc[aligned_data['high_foreign'], 'pred'],
                    aligned_data.loc[aligned_data['high_foreign'], 'ret']
                ) if aligned_data['high_foreign'].sum() > 10 else (0, 1)
                
                compliance_results['foreign_ownership_impact'] = {
                    'high_foreign_ownership_ic': high_foreign_ic,
                    'stocks_near_limit': high_foreign_ownership.sum(),
                    'compliance_risk': 'medium' if high_foreign_ownership.sum() > len(predictions) * 0.1 else 'low'
                }
        
        # Margin trading compliance
        if 'margin_ratios' in additional_data:
            margin_ratios = additional_data['margin_ratios']
            high_margin_stocks = margin_ratios > self.config.margin_trading_threshold
            compliance_results['margin_trading_exposure'] = {
                'high_margin_stocks': high_margin_stocks.sum(),
                'avg_margin_ratio': margin_ratios.mean(),
                'max_margin_ratio': margin_ratios.max()
            }
        
        return compliance_results
    
    def _generate_taiwan_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate Taiwan market-specific recommendations."""
        recommendations = []
        
        # Settlement analysis recommendations
        if 'settlement_analysis' in validation_results:
            settlement = validation_results['settlement_analysis']
            if 'settlement_efficiency' in settlement and settlement['settlement_efficiency'] < 0.5:
                recommendations.append(
                    "Consider adjusting prediction horizon to account for T+2 settlement decay - current efficiency is low"
                )
            
            if 'trading_cost_impact' in settlement:
                cost_impact = settlement['trading_cost_impact'].get('cost_impact_on_ic', 0)
                if cost_impact > 0.1:
                    recommendations.append(
                        f"High trading costs ({cost_impact:.1%} IC impact) - consider optimizing position sizing and turnover"
                    )
        
        # Price limit recommendations
        if 'price_limit_analysis' in validation_results:
            limit_analysis = validation_results['price_limit_analysis']
            if 'limit_statistics' in limit_analysis:
                total_limit_events = limit_analysis['limit_statistics'].get('total_limit_events', 0)
                if total_limit_events > 0.05:  # More than 5% limit events
                    recommendations.append(
                        f"High frequency of price limit events ({total_limit_events:.1%}) - consider incorporating limit probability in model"
                    )
        
        # Sector analysis recommendations  
        if 'sector_analysis' in validation_results:
            sector_perf = validation_results['sector_analysis']
            weak_sectors = [sector for sector, metrics in sector_perf.items() 
                          if metrics.get('ic', 0) < 0.02 and metrics.get('sector_weight', 0) > 0.1]
            
            if weak_sectors:
                recommendations.append(
                    f"Weak performance in major sectors {weak_sectors} - consider sector-specific feature engineering"
                )
        
        # Compliance recommendations
        if 'compliance_check' in validation_results:
            compliance = validation_results['compliance_check']
            if compliance.get('position_limit_compliant', True) is False:
                recommendations.append(
                    "Position concentration exceeds regulatory limits - review position sizing algorithm"
                )
            
            if 'foreign_ownership_impact' in compliance:
                foreign_impact = compliance['foreign_ownership_impact']
                if foreign_impact.get('compliance_risk') == 'medium':
                    recommendations.append(
                        "High exposure to stocks with elevated foreign ownership - monitor for regulatory changes"
                    )
        
        return recommendations