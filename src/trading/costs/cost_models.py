"""
Core transaction cost models for Taiwan market.

This module implements linear and non-linear transaction cost models specifically
calibrated for the Taiwan Stock Exchange (TSE) and Taipei Exchange (TPEx).

Key Features:
- Linear and non-linear cost modeling
- Taiwan regulatory cost calculation
- Market impact integration
- Performance attribution
- Cost optimization support
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from decimal import Decimal
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CostComponent(Enum):
    """Types of transaction cost components."""
    COMMISSION = "commission"
    TRANSACTION_TAX = "transaction_tax"  
    EXCHANGE_FEE = "exchange_fee"
    SETTLEMENT_FEE = "settlement_fee"
    CUSTODY_FEE = "custody_fee"
    MARKET_IMPACT = "market_impact"
    BID_ASK_SPREAD = "bid_ask_spread"
    TIMING_COST = "timing_cost"
    SLIPPAGE = "slippage"


class TradeDirection(Enum):
    """Trade direction for cost calculation."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeCostBreakdown:
    """Detailed breakdown of transaction costs."""
    # Basic trade info
    symbol: str
    trade_date: date
    direction: TradeDirection
    quantity: float
    price: float
    trade_value: float
    
    # Regulatory costs (in TWD)
    commission: float = 0.0
    transaction_tax: float = 0.0
    exchange_fee: float = 0.0
    settlement_fee: float = 0.0
    custody_fee: float = 0.0
    
    # Market microstructure costs (in TWD)
    market_impact: float = 0.0
    bid_ask_spread_cost: float = 0.0
    timing_cost: float = 0.0
    slippage: float = 0.0
    
    # Summary metrics
    total_regulatory_cost: float = field(init=False)
    total_market_cost: float = field(init=False)
    total_cost: float = field(init=False)
    cost_bps: float = field(init=False)
    
    def __post_init__(self):
        """Calculate summary metrics."""
        self.total_regulatory_cost = (
            self.commission + self.transaction_tax + 
            self.exchange_fee + self.settlement_fee + self.custody_fee
        )
        
        self.total_market_cost = (
            self.market_impact + self.bid_ask_spread_cost + 
            self.timing_cost + self.slippage
        )
        
        self.total_cost = self.total_regulatory_cost + self.total_market_cost
        
        # Cost in basis points
        self.cost_bps = (self.total_cost / self.trade_value) * 10000 if self.trade_value > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            'symbol': self.symbol,
            'trade_date': self.trade_date.isoformat(),
            'direction': self.direction.value,
            'quantity': self.quantity,
            'price': self.price,
            'trade_value': self.trade_value,
            'commission': self.commission,
            'transaction_tax': self.transaction_tax,
            'exchange_fee': self.exchange_fee,
            'settlement_fee': self.settlement_fee,
            'custody_fee': self.custody_fee,
            'market_impact': self.market_impact,
            'bid_ask_spread_cost': self.bid_ask_spread_cost,
            'timing_cost': self.timing_cost,
            'slippage': self.slippage,
            'total_regulatory_cost': self.total_regulatory_cost,
            'total_market_cost': self.total_market_cost,
            'total_cost': self.total_cost,
            'cost_bps': self.cost_bps
        }


@dataclass
class TradeInfo:
    """Information required for cost calculation."""
    symbol: str
    trade_date: date
    direction: TradeDirection
    quantity: float
    price: float
    
    # Market context
    previous_close: Optional[float] = None
    daily_volume: Optional[float] = None
    volatility: Optional[float] = None
    bid_ask_spread: Optional[float] = None
    
    # Execution context
    execution_delay_seconds: Optional[float] = None
    order_size_vs_avg: Optional[float] = None  # Order size vs average daily volume
    market_session: Optional[str] = None  # 'morning', 'afternoon'
    
    # Custom parameters
    commission_rate: Optional[float] = None
    use_institutional_rates: bool = True
    
    @property
    def trade_value(self) -> float:
        """Calculate trade value in TWD."""
        return abs(self.quantity) * self.price


class BaseCostModel(ABC):
    """
    Abstract base class for transaction cost models.
    
    Defines the interface that all cost models must implement.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def calculate_cost(self, trade: TradeInfo) -> TradeCostBreakdown:
        """Calculate transaction costs for a trade."""
        pass
    
    @abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters for inspection and validation."""
        pass
    
    def validate_trade(self, trade: TradeInfo) -> List[str]:
        """Validate trade information."""
        issues = []
        
        if trade.quantity == 0:
            issues.append("Trade quantity cannot be zero")
        
        if trade.price <= 0:
            issues.append("Trade price must be positive")
        
        if trade.trade_date > date.today():
            issues.append("Trade date cannot be in the future")
        
        return issues


class TaiwanCostCalculator:
    """
    Taiwan market regulatory cost calculator.
    
    Implements all Taiwan Stock Exchange and Taipei Exchange
    regulatory costs including taxes, fees, and commissions.
    """
    
    def __init__(self):
        # Taiwan regulatory rates (as of 2024)
        self.transaction_tax_rate = 0.003  # 0.3% on stock sales
        self.max_commission_rate = 0.001425  # 0.1425% maximum
        self.exchange_fee_rate = 0.00025  # 0.025% (0.025%)
        self.settlement_fee = 1.0  # NT$1 per transaction
        self.custody_fee_annual_rate = 0.0002  # 0.02% annually
        
        # Typical institutional rates
        self.institutional_commission_rate = 0.0008  # 0.08%
        self.retail_commission_rate = 0.001425  # Full rate
        
        logger.info("TaiwanCostCalculator initialized with 2024 rates")
    
    def calculate_commission(
        self, 
        trade_value: float, 
        commission_rate: Optional[float] = None,
        is_institutional: bool = True
    ) -> float:
        """
        Calculate brokerage commission.
        
        Args:
            trade_value: Trade value in TWD
            commission_rate: Custom commission rate (optional)
            is_institutional: Whether to use institutional rates
            
        Returns:
            Commission in TWD
        """
        if commission_rate is None:
            rate = (self.institutional_commission_rate if is_institutional 
                   else self.retail_commission_rate)
        else:
            rate = min(commission_rate, self.max_commission_rate)
        
        commission = trade_value * rate
        
        # Minimum commission (typically NT$20)
        min_commission = 20.0
        return max(commission, min_commission)
    
    def calculate_transaction_tax(
        self, 
        trade_value: float, 
        direction: TradeDirection
    ) -> float:
        """
        Calculate securities transaction tax.
        
        Taiwan charges transaction tax only on sales.
        
        Args:
            trade_value: Trade value in TWD
            direction: Buy or sell
            
        Returns:
            Transaction tax in TWD
        """
        if direction == TradeDirection.SELL:
            return trade_value * self.transaction_tax_rate
        else:
            return 0.0
    
    def calculate_exchange_fee(self, trade_value: float) -> float:
        """
        Calculate Taiwan Stock Exchange fee.
        
        Args:
            trade_value: Trade value in TWD
            
        Returns:
            Exchange fee in TWD
        """
        fee = trade_value * self.exchange_fee_rate
        
        # Minimum fee
        min_fee = 1.0
        return max(fee, min_fee)
    
    def calculate_settlement_fee(self) -> float:
        """
        Calculate settlement fee.
        
        Returns:
            Settlement fee in TWD (flat fee per transaction)
        """
        return self.settlement_fee
    
    def calculate_custody_fee(
        self, 
        position_value: float, 
        holding_days: int = 1
    ) -> float:
        """
        Calculate custody fee for position holding.
        
        Args:
            position_value: Position value in TWD
            holding_days: Number of days held
            
        Returns:
            Custody fee in TWD
        """
        daily_rate = self.custody_fee_annual_rate / 365
        return position_value * daily_rate * holding_days
    
    def calculate_all_regulatory_costs(
        self, 
        trade: TradeInfo
    ) -> Dict[str, float]:
        """
        Calculate all regulatory costs for a trade.
        
        Args:
            trade: Trade information
            
        Returns:
            Dictionary with all regulatory cost components
        """
        trade_value = trade.trade_value
        
        costs = {
            'commission': self.calculate_commission(
                trade_value, 
                trade.commission_rate,
                trade.use_institutional_rates
            ),
            'transaction_tax': self.calculate_transaction_tax(
                trade_value, 
                trade.direction
            ),
            'exchange_fee': self.calculate_exchange_fee(trade_value),
            'settlement_fee': self.calculate_settlement_fee(),
            'custody_fee': self.calculate_custody_fee(trade_value, 1)  # 1 day default
        }
        
        return costs


class LinearCostModel(BaseCostModel):
    """
    Linear transaction cost model.
    
    Simple model where costs scale linearly with trade size.
    Suitable for small orders and liquid stocks.
    """
    
    def __init__(
        self, 
        base_cost_bps: float = 10.0,
        size_penalty_bps: float = 2.0,
        volatility_multiplier: float = 1.5
    ):
        super().__init__("LinearCostModel")
        self.base_cost_bps = base_cost_bps
        self.size_penalty_bps = size_penalty_bps
        self.volatility_multiplier = volatility_multiplier
        self.taiwan_calculator = TaiwanCostCalculator()
    
    def calculate_cost(self, trade: TradeInfo) -> TradeCostBreakdown:
        """Calculate linear transaction costs."""
        # Validate trade
        issues = self.validate_trade(trade)
        if issues:
            raise ValueError(f"Trade validation failed: {', '.join(issues)}")
        
        # Calculate regulatory costs
        regulatory_costs = self.taiwan_calculator.calculate_all_regulatory_costs(trade)
        
        # Calculate market impact (linear model)
        market_impact = self._calculate_linear_market_impact(trade)
        
        # Calculate bid-ask spread cost
        spread_cost = self._calculate_spread_cost(trade)
        
        # Calculate timing cost (if execution delay)
        timing_cost = self._calculate_timing_cost(trade)
        
        return TradeCostBreakdown(
            symbol=trade.symbol,
            trade_date=trade.trade_date,
            direction=trade.direction,
            quantity=trade.quantity,
            price=trade.price,
            trade_value=trade.trade_value,
            commission=regulatory_costs['commission'],
            transaction_tax=regulatory_costs['transaction_tax'],
            exchange_fee=regulatory_costs['exchange_fee'],
            settlement_fee=regulatory_costs['settlement_fee'],
            custody_fee=regulatory_costs['custody_fee'],
            market_impact=market_impact,
            bid_ask_spread_cost=spread_cost,
            timing_cost=timing_cost,
            slippage=0.0  # No slippage in base linear model
        )
    
    def _calculate_linear_market_impact(self, trade: TradeInfo) -> float:
        """Calculate linear market impact."""
        # Base impact
        base_impact_bps = self.base_cost_bps
        
        # Size penalty
        if trade.order_size_vs_avg is not None:
            size_impact_bps = self.size_penalty_bps * trade.order_size_vs_avg
        else:
            size_impact_bps = 0.0
        
        # Volatility adjustment
        if trade.volatility is not None:
            vol_adjustment = self.volatility_multiplier * trade.volatility
        else:
            vol_adjustment = 1.0
        
        total_impact_bps = (base_impact_bps + size_impact_bps) * vol_adjustment
        
        return (total_impact_bps / 10000) * trade.trade_value
    
    def _calculate_spread_cost(self, trade: TradeInfo) -> float:
        """Calculate bid-ask spread cost."""
        if trade.bid_ask_spread is not None:
            # Pay half the spread on average
            spread_cost = (trade.bid_ask_spread / 2) * abs(trade.quantity)
            return spread_cost
        else:
            # Estimate based on price and volatility
            estimated_spread_bps = 5.0  # Default 5 bps
            if trade.volatility is not None:
                estimated_spread_bps *= (1 + trade.volatility)
            
            return (estimated_spread_bps / 10000) * trade.trade_value
    
    def _calculate_timing_cost(self, trade: TradeInfo) -> float:
        """Calculate timing cost for delayed execution."""
        if trade.execution_delay_seconds is None or trade.execution_delay_seconds <= 0:
            return 0.0
        
        # Estimate timing cost based on delay and volatility
        delay_minutes = trade.execution_delay_seconds / 60
        
        if trade.volatility is not None:
            # Convert annual volatility to per-minute volatility
            annual_to_minute = np.sqrt(1 / (252 * 390))  # 252 trading days, 390 minutes
            minute_vol = trade.volatility * annual_to_minute
            
            # Timing cost as function of delay and volatility
            timing_cost_bps = delay_minutes * minute_vol * 10000 * 0.1  # 10% of volatility impact
            
            return (timing_cost_bps / 10000) * trade.trade_value
        
        return 0.0
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_type': 'linear',
            'base_cost_bps': self.base_cost_bps,
            'size_penalty_bps': self.size_penalty_bps,
            'volatility_multiplier': self.volatility_multiplier
        }


class NonLinearCostModel(BaseCostModel):
    """
    Non-linear transaction cost model.
    
    Advanced model with non-linear market impact based on order size
    relative to average daily volume. Includes temporary and permanent
    impact components.
    """
    
    def __init__(
        self,
        temp_impact_factor: float = 0.5,
        perm_impact_factor: float = 0.3,
        size_penalty_exp: float = 0.6,
        volatility_multiplier: float = 2.0,
        liquidity_penalty: float = 1.2
    ):
        super().__init__("NonLinearCostModel")
        self.temp_impact_factor = temp_impact_factor
        self.perm_impact_factor = perm_impact_factor
        self.size_penalty_exp = size_penalty_exp
        self.volatility_multiplier = volatility_multiplier
        self.liquidity_penalty = liquidity_penalty
        self.taiwan_calculator = TaiwanCostCalculator()
    
    def calculate_cost(self, trade: TradeInfo) -> TradeCostBreakdown:
        """Calculate non-linear transaction costs."""
        # Validate trade
        issues = self.validate_trade(trade)
        if issues:
            raise ValueError(f"Trade validation failed: {', '.join(issues)}")
        
        # Calculate regulatory costs
        regulatory_costs = self.taiwan_calculator.calculate_all_regulatory_costs(trade)
        
        # Calculate non-linear market impact
        market_impact = self._calculate_nonlinear_market_impact(trade)
        
        # Calculate spread cost with liquidity adjustment
        spread_cost = self._calculate_liquidity_adjusted_spread_cost(trade)
        
        # Calculate timing cost
        timing_cost = self._calculate_timing_cost(trade)
        
        # Calculate slippage
        slippage = self._calculate_slippage(trade)
        
        return TradeCostBreakdown(
            symbol=trade.symbol,
            trade_date=trade.trade_date,
            direction=trade.direction,
            quantity=trade.quantity,
            price=trade.price,
            trade_value=trade.trade_value,
            commission=regulatory_costs['commission'],
            transaction_tax=regulatory_costs['transaction_tax'],
            exchange_fee=regulatory_costs['exchange_fee'],
            settlement_fee=regulatory_costs['settlement_fee'],
            custody_fee=regulatory_costs['custody_fee'],
            market_impact=market_impact,
            bid_ask_spread_cost=spread_cost,
            timing_cost=timing_cost,
            slippage=slippage
        )
    
    def _calculate_nonlinear_market_impact(self, trade: TradeInfo) -> float:
        """Calculate non-linear market impact with temporary and permanent components."""
        if trade.order_size_vs_avg is None:
            # Default to linear model if no volume data
            return self._calculate_fallback_impact(trade)
        
        participation_rate = trade.order_size_vs_avg
        
        # Temporary impact (mean-reverting)
        temp_impact_bps = (
            self.temp_impact_factor * 
            (participation_rate ** self.size_penalty_exp) * 
            10000  # Convert to bps
        )
        
        # Permanent impact (information-based)
        perm_impact_bps = (
            self.perm_impact_factor * 
            (participation_rate ** 0.5) * 
            10000  # Convert to bps
        )
        
        # Volatility adjustment
        if trade.volatility is not None:
            vol_multiplier = 1 + (self.volatility_multiplier - 1) * trade.volatility
        else:
            vol_multiplier = 1.0
        
        total_impact_bps = (temp_impact_bps + perm_impact_bps) * vol_multiplier
        
        return (total_impact_bps / 10000) * trade.trade_value
    
    def _calculate_liquidity_adjusted_spread_cost(self, trade: TradeInfo) -> float:
        """Calculate spread cost with liquidity adjustments."""
        if trade.bid_ask_spread is not None:
            base_spread_cost = (trade.bid_ask_spread / 2) * abs(trade.quantity)
        else:
            # Estimate spread
            base_spread_bps = 8.0  # Higher default for non-linear model
            if trade.volatility is not None:
                base_spread_bps *= (1 + trade.volatility)
            base_spread_cost = (base_spread_bps / 10000) * trade.trade_value
        
        # Liquidity penalty for large orders
        if trade.order_size_vs_avg is not None and trade.order_size_vs_avg > 0.05:  # > 5% of ADV
            liquidity_multiplier = 1 + (trade.order_size_vs_avg - 0.05) * self.liquidity_penalty
            base_spread_cost *= liquidity_multiplier
        
        return base_spread_cost
    
    def _calculate_timing_cost(self, trade: TradeInfo) -> float:
        """Calculate timing cost with non-linear delay penalty."""
        if trade.execution_delay_seconds is None or trade.execution_delay_seconds <= 0:
            return 0.0
        
        delay_minutes = trade.execution_delay_seconds / 60
        
        if trade.volatility is not None:
            # Non-linear delay penalty
            annual_to_minute = np.sqrt(1 / (252 * 390))
            minute_vol = trade.volatility * annual_to_minute
            
            # Quadratic delay penalty for longer delays
            if delay_minutes <= 5:
                timing_cost_bps = delay_minutes * minute_vol * 10000 * 0.2
            else:
                timing_cost_bps = (5 * minute_vol * 10000 * 0.2 + 
                                  (delay_minutes - 5) ** 1.5 * minute_vol * 10000 * 0.1)
            
            return (timing_cost_bps / 10000) * trade.trade_value
        
        return 0.0
    
    def _calculate_slippage(self, trade: TradeInfo) -> float:
        """Calculate execution slippage."""
        if trade.previous_close is None:
            return 0.0
        
        # Calculate slippage as difference from reference price
        price_diff_pct = abs(trade.price - trade.previous_close) / trade.previous_close
        
        # Expected slippage should be minimal for normal execution
        expected_slippage_bps = 2.0  # 2 bps base
        
        # Adjust for volatility and order size
        if trade.volatility is not None:
            expected_slippage_bps *= (1 + trade.volatility)
        
        if trade.order_size_vs_avg is not None:
            expected_slippage_bps *= (1 + trade.order_size_vs_avg)
        
        # If actual price movement exceeds expected, attribute excess to slippage
        expected_move_pct = expected_slippage_bps / 10000
        if price_diff_pct > expected_move_pct:
            excess_slippage_bps = (price_diff_pct - expected_move_pct) * 10000
            return (excess_slippage_bps / 10000) * trade.trade_value
        
        return 0.0
    
    def _calculate_fallback_impact(self, trade: TradeInfo) -> float:
        """Fallback linear impact when no volume data available."""
        base_impact_bps = 8.0  # Higher than linear model default
        
        if trade.volatility is not None:
            vol_adjustment = 1 + trade.volatility * (self.volatility_multiplier - 1)
        else:
            vol_adjustment = 1.0
        
        total_impact_bps = base_impact_bps * vol_adjustment
        return (total_impact_bps / 10000) * trade.trade_value
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_type': 'nonlinear',
            'temp_impact_factor': self.temp_impact_factor,
            'perm_impact_factor': self.perm_impact_factor,
            'size_penalty_exp': self.size_penalty_exp,
            'volatility_multiplier': self.volatility_multiplier,
            'liquidity_penalty': self.liquidity_penalty
        }


class CostModelFactory:
    """Factory for creating transaction cost models."""
    
    @staticmethod
    def create_linear_model(**kwargs) -> LinearCostModel:
        """Create linear cost model with Taiwan defaults."""
        defaults = {
            'base_cost_bps': 8.0,      # Lower for Taiwan liquid stocks
            'size_penalty_bps': 1.5,   # Moderate size penalty
            'volatility_multiplier': 1.3  # Conservative volatility impact
        }
        defaults.update(kwargs)
        return LinearCostModel(**defaults)
    
    @staticmethod
    def create_nonlinear_model(**kwargs) -> NonLinearCostModel:
        """Create non-linear cost model with Taiwan calibration."""
        defaults = {
            'temp_impact_factor': 0.4,    # Taiwan market characteristics
            'perm_impact_factor': 0.25,   # Lower permanent impact
            'size_penalty_exp': 0.65,     # Moderate non-linearity
            'volatility_multiplier': 1.8,  # Higher vol sensitivity
            'liquidity_penalty': 1.5      # Liquidity penalty
        }
        defaults.update(kwargs)
        return NonLinearCostModel(**defaults)
    
    @staticmethod
    def create_conservative_model(**kwargs) -> NonLinearCostModel:
        """Create conservative cost model for risk management."""
        defaults = {
            'temp_impact_factor': 0.6,    # Higher temporary impact
            'perm_impact_factor': 0.4,    # Higher permanent impact
            'size_penalty_exp': 0.7,      # Higher non-linearity
            'volatility_multiplier': 2.2,  # Higher volatility sensitivity
            'liquidity_penalty': 2.0      # Higher liquidity penalty
        }
        defaults.update(kwargs)
        return NonLinearCostModel(**defaults)


# Example usage and testing
if __name__ == "__main__":
    print("Taiwan Transaction Cost Models Demo")
    
    # Sample trade
    sample_trade = TradeInfo(
        symbol="2330.TW",  # TSMC
        trade_date=date.today(),
        direction=TradeDirection.BUY,
        quantity=1000,
        price=500.0,
        daily_volume=50000,
        volatility=0.25,
        bid_ask_spread=0.1,
        order_size_vs_avg=0.02  # 2% of ADV
    )
    
    # Test linear model
    linear_model = CostModelFactory.create_linear_model()
    linear_cost = linear_model.calculate_cost(sample_trade)
    print(f"\nLinear Model Cost: {linear_cost.cost_bps:.2f} bps")
    print(f"Total Cost: NT${linear_cost.total_cost:.2f}")
    
    # Test non-linear model
    nonlinear_model = CostModelFactory.create_nonlinear_model()
    nonlinear_cost = nonlinear_model.calculate_cost(sample_trade)
    print(f"\nNon-Linear Model Cost: {nonlinear_cost.cost_bps:.2f} bps")
    print(f"Total Cost: NT${nonlinear_cost.total_cost:.2f}")
    
    # Compare models
    print(f"\nCost Difference: {nonlinear_cost.cost_bps - linear_cost.cost_bps:.2f} bps")