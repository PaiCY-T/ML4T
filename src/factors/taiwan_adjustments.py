"""
Taiwan market specific adjustments for factor calculations.

This module implements Taiwan Stock Exchange (TWSE) specific adjustments
including daily price limits, settlement cycles, and market timing constraints.
"""

from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import logging
from decimal import Decimal

from ..data.models.taiwan_market import (
    TaiwanMarketCode, TradingStatus, CorporateActionType,
    TaiwanTradingCalendar
)

logger = logging.getLogger(__name__)


class TaiwanMarketAdjustments:
    """Taiwan market specific adjustments for factor calculations."""
    
    # Taiwan market constants
    DAILY_PRICE_LIMIT = 0.10  # ±10% daily price limit
    SETTLEMENT_DAYS = 2  # T+2 settlement
    MARKET_OPEN_TIME = time(9, 0)  # 09:00 TST
    MARKET_CLOSE_TIME = time(13, 30)  # 13:30 TST
    TRADING_DAYS_PER_YEAR = 245  # Approximate trading days in Taiwan
    
    def __init__(self, trading_calendar: Optional[TaiwanTradingCalendar] = None):
        self.trading_calendar = trading_calendar
        self.logger = logging.getLogger(__name__)
    
    def adjust_for_price_limits(self, prices: pd.DataFrame, 
                               returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Adjust price and return data for daily price limits.
        
        Taiwan stocks have ±10% daily price limits. Returns exceeding this
        indicate special situations (ex-dividend, corporate actions, etc.)
        
        Args:
            prices: DataFrame with price data
            returns: DataFrame with return data
            
        Returns:
            Tuple of (adjusted_prices, adjusted_returns)
        """
        adjusted_returns = returns.copy()
        
        # Mark returns exceeding price limits
        limit_exceeded = (np.abs(adjusted_returns) > self.DAILY_PRICE_LIMIT + 0.001)
        
        if limit_exceeded.any().any():
            self.logger.info(f"Found {limit_exceeded.sum().sum()} price limit violations")
            
            # For factor calculations, we might want to:
            # 1. Cap returns at price limits
            # 2. Flag these dates for special treatment
            # 3. Exclude these observations
            
            # Option 1: Cap at price limits (conservative approach)
            adjusted_returns = adjusted_returns.clip(
                lower=-self.DAILY_PRICE_LIMIT,
                upper=self.DAILY_PRICE_LIMIT
            )
        
        # Recalculate adjusted prices
        adjusted_prices = prices.copy()
        for symbol in prices.columns:
            if symbol in adjusted_returns.columns:
                # Reconstruct price series from adjusted returns
                price_series = adjusted_prices[symbol].iloc[0] * (1 + adjusted_returns[symbol]).cumprod()
                adjusted_prices[symbol] = price_series
        
        return adjusted_prices, adjusted_returns
    
    def adjust_for_settlement_lag(self, factor_date: date, 
                                 data_cutoff_time: Optional[time] = None) -> date:
        """
        Adjust factor calculation date for T+2 settlement.
        
        In Taiwan, stock transactions settle T+2. For factor calculations,
        we need to ensure we're using data that was actually available
        for trading decisions.
        
        Args:
            factor_date: Target factor calculation date
            data_cutoff_time: Time cutoff for data availability
            
        Returns:
            Adjusted date for data access
        """
        # For T+2 settlement, data available for trading decisions
        # should be at least 2 trading days old
        adjusted_date = factor_date - timedelta(days=self.SETTLEMENT_DAYS)
        
        # If we have trading calendar, adjust for non-trading days
        if self.trading_calendar:
            while not self.trading_calendar.is_trading_day(adjusted_date):
                adjusted_date -= timedelta(days=1)
        
        return adjusted_date
    
    def adjust_for_market_hours(self, timestamp: datetime) -> datetime:
        """
        Adjust timestamp for Taiwan market hours.
        
        Taiwan market trades 09:00-13:30 TST with no lunch break.
        """
        market_date = timestamp.date()
        
        if timestamp.time() < self.MARKET_OPEN_TIME:
            # Before market open, use previous close
            previous_close = datetime.combine(market_date - timedelta(days=1), 
                                            self.MARKET_CLOSE_TIME)
            return previous_close
        elif timestamp.time() > self.MARKET_CLOSE_TIME:
            # After market close, use current close
            return datetime.combine(market_date, self.MARKET_CLOSE_TIME)
        else:
            # During market hours
            return timestamp
    
    def handle_corporate_actions(self, prices: pd.DataFrame,
                                actions: Optional[Dict] = None) -> pd.DataFrame:
        """
        Adjust prices for corporate actions.
        
        Args:
            prices: Price data
            actions: Corporate action data
            
        Returns:
            Adjusted price DataFrame
        """
        if actions is None:
            return prices
        
        adjusted_prices = prices.copy()
        
        for symbol, symbol_actions in actions.items():
            if symbol not in adjusted_prices.columns:
                continue
            
            symbol_prices = adjusted_prices[symbol].copy()
            
            for action in symbol_actions:
                action_date = action.get('date')
                action_type = action.get('type')
                
                if action_type == CorporateActionType.DIVIDEND_CASH:
                    dividend = action.get('amount', 0)
                    # Adjust prices before ex-dividend date
                    mask = symbol_prices.index < action_date
                    if mask.any():
                        symbol_prices[mask] = symbol_prices[mask] - dividend
                
                elif action_type == CorporateActionType.STOCK_SPLIT:
                    split_ratio = action.get('ratio', 1.0)
                    # Adjust prices before split date
                    mask = symbol_prices.index < action_date
                    if mask.any():
                        symbol_prices[mask] = symbol_prices[mask] / split_ratio
                
                # Add other corporate action types as needed
            
            adjusted_prices[symbol] = symbol_prices
        
        return adjusted_prices
    
    def calculate_taiwan_trading_days(self, start_date: date, 
                                    end_date: date) -> int:
        """
        Calculate number of trading days between dates in Taiwan.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of trading days
        """
        if self.trading_calendar:
            return self.trading_calendar.get_trading_days_count(start_date, end_date)
        
        # Fallback: approximate calculation
        total_days = (end_date - start_date).days
        # Assume 5/7 of days are trading days, adjust for holidays
        approx_trading_days = int(total_days * (self.TRADING_DAYS_PER_YEAR / 365))
        return approx_trading_days
    
    def annualize_factor(self, factor_value: float, 
                        frequency_days: int) -> float:
        """
        Annualize factor values based on Taiwan trading calendar.
        
        Args:
            factor_value: Original factor value
            frequency_days: Frequency in days
            
        Returns:
            Annualized factor value
        """
        if frequency_days <= 0:
            return factor_value
        
        scaling_factor = self.TRADING_DAYS_PER_YEAR / frequency_days
        return factor_value * scaling_factor
    
    def handle_lunar_new_year_seasonality(self, dates: pd.DatetimeIndex,
                                        factors: pd.DataFrame) -> pd.DataFrame:
        """
        Handle Lunar New Year seasonality effects.
        
        Lunar New Year is a significant market event in Taiwan with
        typical seasonal patterns before and after the holiday.
        
        Args:
            dates: Date index
            factors: Factor values
            
        Returns:
            Seasonality-adjusted factors
        """
        adjusted_factors = factors.copy()
        
        for year in dates.year.unique():
            # Lunar New Year typically falls between Jan 21 - Feb 20
            # This is a simplified approach - would need actual lunar calendar
            lny_period_start = pd.Timestamp(f"{year}-01-15")
            lny_period_end = pd.Timestamp(f"{year}-02-25")
            
            mask = (dates >= lny_period_start) & (dates <= lny_period_end)
            
            if mask.any():
                # Apply seasonal adjustment (example: reduce momentum factors)
                # This would be calibrated based on historical analysis
                seasonal_adjustment = 0.9  # 10% reduction during LNY period
                
                # Apply to relevant factor columns
                for col in adjusted_factors.columns:
                    if 'momentum' in col.lower():
                        adjusted_factors.loc[mask, col] *= seasonal_adjustment
        
        return adjusted_factors
    
    def validate_taiwan_market_data(self, data: pd.DataFrame, 
                                  symbols: List[str]) -> Dict[str, List[str]]:
        """
        Validate data for Taiwan market compliance.
        
        Args:
            data: Data to validate
            symbols: List of symbols
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid_symbols': [],
            'invalid_symbols': [],
            'warnings': [],
            'errors': []
        }
        
        for symbol in symbols:
            if symbol not in data.columns:
                validation_results['invalid_symbols'].append(symbol)
                validation_results['errors'].append(f"Missing data for {symbol}")
                continue
            
            symbol_data = data[symbol].dropna()
            
            if len(symbol_data) == 0:
                validation_results['invalid_symbols'].append(symbol)
                validation_results['errors'].append(f"No valid data for {symbol}")
                continue
            
            # Check for reasonable price ranges (Taiwan stocks typically > 1 TWD)
            if (symbol_data <= 0).any():
                validation_results['warnings'].append(f"Non-positive prices for {symbol}")
            
            # Check for extreme returns (beyond daily limits)
            returns = symbol_data.pct_change().dropna()
            extreme_returns = returns[np.abs(returns) > self.DAILY_PRICE_LIMIT * 2]
            
            if len(extreme_returns) > 0:
                validation_results['warnings'].append(
                    f"Extreme returns detected for {symbol}: {len(extreme_returns)} instances"
                )
            
            validation_results['valid_symbols'].append(symbol)
        
        return validation_results
    
    def get_taiwan_market_metadata(self) -> Dict[str, any]:
        """Get Taiwan market metadata for factor calculations."""
        return {
            'price_limit': self.DAILY_PRICE_LIMIT,
            'settlement_days': self.SETTLEMENT_DAYS,
            'market_open': self.MARKET_OPEN_TIME,
            'market_close': self.MARKET_CLOSE_TIME,
            'trading_days_per_year': self.TRADING_DAYS_PER_YEAR,
            'timezone': 'Asia/Taipei',
            'currency': 'TWD',
            'market_codes': ['TWSE', 'TPEx']
        }