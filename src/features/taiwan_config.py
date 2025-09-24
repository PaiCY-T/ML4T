"""
Taiwan Market Configuration for OpenFE Integration

Market-specific parameters for Taiwan Stock Exchange (TSE) and 
Taipei Exchange (TPEx) to ensure compliance with local market structure.

Key Taiwan Market Characteristics:
- T+2 settlement cycle
- 10% daily price limits (±10% from previous close)
- Trading hours: 09:00-13:30 (4.5 hours)
- No short selling restrictions for institutional investors
- Currency: TWD (New Taiwan Dollar)
"""

import logging
from datetime import time, datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TaiwanMarketConfig:
    """Taiwan Stock Exchange market configuration and parameters."""
    
    # Market Structure Constants
    SETTLEMENT_DAYS = 2  # T+2 settlement
    CURRENCY = 'TWD'
    TIMEZONE = 'Asia/Taipei'
    
    # Trading Hours (TSE/TPEx)
    TRADING_START = time(9, 0)    # 09:00
    TRADING_END = time(13, 30)    # 13:30
    LUNCH_BREAK = False           # No lunch break in Taiwan
    
    # Price Limits
    PRICE_LIMIT_PERCENT = 0.10    # ±10% daily price limits
    PRICE_TICK_SIZE = 0.01        # Minimum price movement (1 cent)
    
    # Market Calendar
    TRADING_DAYS_PER_YEAR = 252
    TRADING_HOURS_PER_DAY = 4.5   # 09:00-13:30
    
    # Stock Universe
    TSE_STOCKS_APPROX = 1000      # Taiwan Stock Exchange
    TPEX_STOCKS_APPROX = 800      # Taipei Exchange
    TOTAL_STOCKS_APPROX = 1800    # Combined universe
    
    # Feature Engineering Constraints
    MIN_HISTORY_DAYS = 252        # Minimum 1 year for factor calculation
    MAX_LOOKBACK_DAYS = 504       # Maximum 2 years lookback
    
    def __init__(self):
        """Initialize Taiwan market configuration."""
        self.market_holidays = self._load_taiwan_holidays()
        self.trading_calendar = self._generate_trading_calendar()
        
    def _load_taiwan_holidays(self) -> List[str]:
        """
        Load Taiwan market holidays.
        
        Key holidays:
        - Lunar New Year (3-5 days)
        - Dragon Boat Festival
        - Mid-Autumn Festival
        - National Day (October 10)
        - Various government holidays
        """
        # Basic holidays (would normally load from external source)
        return [
            # 2024 holidays (example)
            '2024-01-01',  # New Year's Day
            '2024-02-08',  # Lunar New Year Eve
            '2024-02-09',  # Lunar New Year 1st day
            '2024-02-12',  # Lunar New Year 2nd day
            '2024-02-13',  # Lunar New Year 3rd day
            '2024-02-14',  # Lunar New Year 4th day
            '2024-02-28',  # Peace Memorial Day
            '2024-04-04',  # Children's Day
            '2024-04-05',  # Tomb Sweeping Day
            '2024-05-01',  # Labor Day
            '2024-06-10',  # Dragon Boat Festival
            '2024-09-17',  # Mid-Autumn Festival
            '2024-10-10',  # National Day
            # Add more holidays as needed
        ]
        
    def _generate_trading_calendar(self, 
                                  start_date: str = '2020-01-01',
                                  end_date: str = '2025-12-31') -> pd.DatetimeIndex:
        """Generate Taiwan market trading calendar."""
        # Create business days
        all_days = pd.bdate_range(start=start_date, end=end_date, freq='B')
        
        # Remove Taiwan holidays
        holiday_dates = pd.to_datetime(self.market_holidays)
        trading_days = all_days.difference(holiday_dates)
        
        return trading_days
        
    def is_trading_day(self, date: pd.Timestamp) -> bool:
        """Check if given date is a trading day."""
        return date in self.trading_calendar
        
    def get_trading_hours(self) -> Dict[str, time]:
        """Get trading hours."""
        return {
            'start': self.TRADING_START,
            'end': self.TRADING_END,
            'lunch_start': None,  # No lunch break
            'lunch_end': None
        }
        
    def apply_price_limits(self, 
                          current_price: float, 
                          previous_close: float) -> Tuple[float, float]:
        """
        Calculate price limits based on previous close.
        
        Args:
            current_price: Current price to check
            previous_close: Previous day's closing price
            
        Returns:
            (upper_limit, lower_limit) tuple
        """
        limit_amount = previous_close * self.PRICE_LIMIT_PERCENT
        upper_limit = previous_close + limit_amount
        lower_limit = previous_close - limit_amount
        
        return upper_limit, lower_limit
        
    def is_price_within_limits(self, 
                              price: float, 
                              previous_close: float) -> bool:
        """Check if price is within daily limits."""
        upper_limit, lower_limit = self.apply_price_limits(price, previous_close)
        return lower_limit <= price <= upper_limit
        
    def get_settlement_date(self, trade_date: pd.Timestamp) -> pd.Timestamp:
        """
        Get settlement date for T+2 system.
        
        Args:
            trade_date: Trading date
            
        Returns:
            Settlement date (T+2 business days)
        """
        settlement_date = trade_date
        days_added = 0
        
        while days_added < self.SETTLEMENT_DAYS:
            settlement_date += timedelta(days=1)
            if self.is_trading_day(settlement_date):
                days_added += 1
                
        return settlement_date
        
    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """
        Get configuration for feature engineering specific to Taiwan market.
        
        Returns:
            Configuration dictionary for OpenFE wrapper
        """
        return {
            # Time-series parameters
            'settlement_lag': self.SETTLEMENT_DAYS,
            'min_history_days': self.MIN_HISTORY_DAYS,
            'max_lookback_days': self.MAX_LOOKBACK_DAYS,
            'trading_hours_per_day': self.TRADING_HOURS_PER_DAY,
            
            # Market structure
            'price_limit_percent': self.PRICE_LIMIT_PERCENT,
            'tick_size': self.PRICE_TICK_SIZE,
            'currency': self.CURRENCY,
            'timezone': self.TIMEZONE,
            
            # Data processing
            'trading_calendar': self.trading_calendar,
            'market_holidays': self.market_holidays,
            
            # OpenFE specific
            'openfe_task': 'classification',  # or 'regression'
            'n_data_blocks': 8,  # For memory efficiency
            'time_budget': 600,   # 10 minutes
            'max_features': 500,  # Limit for memory
            
            # Memory management for Taiwan universe
            'expected_stocks': self.TOTAL_STOCKS_APPROX,
            'expected_memory_gb': 12,  # Estimated memory for full universe
            'chunk_size': 100,         # Process in chunks of 100 stocks
        }
        
    def validate_data_for_taiwan_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data compliance with Taiwan market characteristics.
        
        Args:
            data: Market data to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Check date range
            if isinstance(data.index, pd.MultiIndex):
                dates = pd.to_datetime(data.index.get_level_values(0))
            else:
                dates = pd.to_datetime(data.index)
                
            date_range = (dates.min(), dates.max())
            validation_results['statistics']['date_range'] = date_range
            
            # Check for trading days alignment
            non_trading_dates = []
            for date in dates.unique():
                if not self.is_trading_day(pd.Timestamp(date)):
                    non_trading_dates.append(date)
                    
            if non_trading_dates:
                validation_results['warnings'].append(
                    f"Found {len(non_trading_dates)} non-trading dates in data"
                )
                
            # Check for reasonable price ranges (basic sanity check)
            if 'close' in data.columns:
                close_prices = data['close'].dropna()
                if len(close_prices) > 0:
                    price_stats = {
                        'min': close_prices.min(),
                        'max': close_prices.max(),
                        'mean': close_prices.mean(),
                        'std': close_prices.std()
                    }
                    validation_results['statistics']['price_stats'] = price_stats
                    
                    # Check for unreasonable prices
                    if price_stats['min'] <= 0:
                        validation_results['errors'].append("Found non-positive prices")
                        validation_results['passed'] = False
                        
                    if price_stats['max'] > 10000:  # Extremely high price (>10,000 TWD)
                        validation_results['warnings'].append(
                            f"Found very high prices (max: {price_stats['max']:.2f})"
                        )
                        
            # Check currency alignment
            if 'currency' in data.columns:
                currencies = data['currency'].unique()
                if self.CURRENCY not in currencies:
                    validation_results['warnings'].append(
                        f"Expected currency {self.CURRENCY} not found. Found: {currencies}"
                    )
                    
            # Memory estimation
            memory_usage_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            validation_results['statistics']['memory_usage_mb'] = memory_usage_mb
            
            if memory_usage_mb > 1000:  # > 1GB
                validation_results['warnings'].append(
                    f"High memory usage: {memory_usage_mb:.1f}MB"
                )
                
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['passed'] = False
            
        return validation_results
        
    def get_sector_mapping(self) -> Dict[str, List[str]]:
        """
        Get Taiwan market sector mapping.
        
        Returns:
            Dictionary mapping sector names to stock codes
        """
        # Simplified sector mapping (would normally load from external source)
        return {
            'Technology': ['2330', '2454', '3008', '2382', '2317'],  # TSMC, MediaTek, etc.
            'Financial': ['2882', '2883', '2880', '2881', '2892'],   # Banks
            'Manufacturing': ['2002', '1301', '1303', '2105', '1216'], # Traditional manufacturing
            'Textiles': ['1102', '1229', '1210', '1218', '1432'],    # Textile companies
            'Food': ['1101', '1102', '1216', '1217', '1227'],        # Food & beverage
            'Petrochemical': ['1301', '1303', '1326', '6505', '4904'], # Petrochemicals
            'Steel': ['2002', '2006', '2015', '2027', '2049'],       # Steel industry
            'Auto': ['2201', '2207', '2227', '2231', '9103'],        # Automotive
            'Construction': ['2501', '2504', '2505', '2511', '2515'], # Construction
            'Retail': ['2912', '2915', '9904', '9905', '9917']       # Retail & services
        }
        
    def apply_taiwan_feature_constraints(self, 
                                       features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Taiwan-specific constraints to generated features.
        
        Args:
            features: Generated features from OpenFE
            
        Returns:
            Constrained features compliant with Taiwan market
        """
        constrained_features = features.copy()
        
        # Apply T+2 settlement lag to relevant features
        settlement_sensitive_cols = [col for col in features.columns 
                                   if any(term in col.lower() for term in 
                                         ['volume', 'turnover', 'trade', 'flow'])]
        
        for col in settlement_sensitive_cols:
            # Shift by T+2 to account for settlement
            if isinstance(features.index, pd.MultiIndex):
                # Panel data - shift within each stock group
                constrained_features[col] = (
                    constrained_features.groupby(level=1)[col]
                    .shift(self.SETTLEMENT_DAYS)
                )
            else:
                # Time series - simple shift
                constrained_features[col] = (
                    constrained_features[col].shift(self.SETTLEMENT_DAYS)
                )
                
        # Remove features during non-trading hours (if timestamp data available)
        # This would be implemented if intraday data is used
        
        return constrained_features.dropna()
        
    def __repr__(self) -> str:
        """String representation of Taiwan market config."""
        return (f"TaiwanMarketConfig("
                f"settlement_days={self.SETTLEMENT_DAYS}, "
                f"trading_hours={self.TRADING_START}-{self.TRADING_END}, "
                f"price_limit={self.PRICE_LIMIT_PERCENT*100}%)")


# Global instance for easy import
taiwan_config = TaiwanMarketConfig()


def get_taiwan_openfe_config() -> Dict[str, Any]:
    """Convenience function to get Taiwan OpenFE configuration."""
    return taiwan_config.get_feature_engineering_config()


def validate_taiwan_market_data(data: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function to validate data for Taiwan market."""
    return taiwan_config.validate_data_for_taiwan_market(data)