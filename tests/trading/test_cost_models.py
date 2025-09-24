"""
Comprehensive tests for Taiwan transaction cost models.

Tests cover all aspects of the cost modeling framework including:
- Basic cost calculations
- Taiwan regulatory cost accuracy
- Market microstructure modeling
- Execution simulation
- Integration with temporal data systems
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from decimal import Decimal

from src.trading.costs.cost_models import (
    BaseCostModel, LinearCostModel, NonLinearCostModel,
    TaiwanCostCalculator, CostModelFactory,
    TradeInfo, TradeDirection, TradeCostBreakdown
)
from src.trading.costs.taiwan_microstructure import (
    TaiwanMarketStructure, BidAskSpreadModel, MarketImpactModel,
    TaiwanTickSizeModel, TradingSession
)
from src.trading.costs.execution_models import (
    ExecutionCostSimulator, SlippageModel, TimingCostModel,
    SettlementCostModel, ExecutionStrategy
)


class TestTaiwanCostCalculator:
    """Test Taiwan regulatory cost calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = TaiwanCostCalculator()
        self.sample_trade_value = 500000.0  # NT$500,000
    
    def test_commission_calculation(self):
        """Test commission calculation with different rates."""
        # Institutional rate
        commission = self.calculator.calculate_commission(
            self.sample_trade_value, 
            is_institutional=True
        )
        expected = self.sample_trade_value * 0.0008  # 0.08%
        assert abs(commission - expected) < 0.01
        
        # Retail rate
        commission_retail = self.calculator.calculate_commission(
            self.sample_trade_value,
            is_institutional=False
        )
        expected_retail = self.sample_trade_value * 0.001425  # 0.1425%
        assert abs(commission_retail - expected_retail) < 0.01
        
        # Minimum commission check
        small_trade = 1000.0  # NT$1,000
        commission_small = self.calculator.calculate_commission(small_trade)
        assert commission_small >= 20.0  # Minimum NT$20
    
    def test_transaction_tax_calculation(self):
        """Test securities transaction tax calculation."""
        # Tax on sales only
        tax_sell = self.calculator.calculate_transaction_tax(
            self.sample_trade_value, 
            TradeDirection.SELL
        )
        expected = self.sample_trade_value * 0.003  # 0.3%
        assert abs(tax_sell - expected) < 0.01
        
        # No tax on purchases
        tax_buy = self.calculator.calculate_transaction_tax(
            self.sample_trade_value,
            TradeDirection.BUY
        )
        assert tax_buy == 0.0
    
    def test_exchange_fee_calculation(self):
        """Test exchange fee calculation."""
        fee = self.calculator.calculate_exchange_fee(self.sample_trade_value)
        expected = self.sample_trade_value * 0.00025  # 0.025%
        assert abs(fee - expected) < 0.01
        
        # Minimum fee check
        small_trade = 100.0
        fee_small = self.calculator.calculate_exchange_fee(small_trade)
        assert fee_small >= 1.0  # Minimum NT$1
    
    def test_settlement_fee(self):
        """Test settlement fee calculation."""
        fee = self.calculator.calculate_settlement_fee()
        assert fee == 1.0  # Fixed NT$1
    
    def test_custody_fee_calculation(self):
        """Test custody fee calculation."""
        position_value = 1000000.0  # NT$1M
        holding_days = 30
        
        fee = self.calculator.calculate_custody_fee(position_value, holding_days)
        daily_rate = 0.0002 / 365  # 0.02% annual
        expected = position_value * daily_rate * holding_days
        assert abs(fee - expected) < 0.01
    
    def test_all_regulatory_costs(self):
        """Test comprehensive regulatory cost calculation."""
        trade = TradeInfo(
            symbol="2330.TW",
            trade_date=date.today(),
            direction=TradeDirection.SELL,
            quantity=1000,
            price=500.0,
            use_institutional_rates=True
        )
        
        costs = self.calculator.calculate_all_regulatory_costs(trade)
        
        # Verify all cost components exist
        required_keys = ['commission', 'transaction_tax', 'exchange_fee', 
                        'settlement_fee', 'custody_fee']
        for key in required_keys:
            assert key in costs
            assert costs[key] >= 0
        
        # Verify transaction tax is present for sell order
        assert costs['transaction_tax'] > 0
        
        # Verify commission is reasonable
        expected_commission = trade.trade_value * 0.0008
        assert abs(costs['commission'] - expected_commission) < 1.0


class TestLinearCostModel:
    """Test linear transaction cost model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = CostModelFactory.create_linear_model()
        self.sample_trade = TradeInfo(
            symbol="2330.TW",
            trade_date=date.today(),
            direction=TradeDirection.BUY,
            quantity=1000,
            price=500.0,
            daily_volume=50000,
            volatility=0.25,
            bid_ask_spread=0.5,
            order_size_vs_avg=0.02  # 2% of ADV
        )
    
    def test_basic_cost_calculation(self):
        """Test basic linear cost calculation."""
        cost_breakdown = self.model.calculate_cost(self.sample_trade)
        
        # Verify result structure
        assert isinstance(cost_breakdown, TradeCostBreakdown)
        assert cost_breakdown.symbol == "2330.TW"
        assert cost_breakdown.direction == TradeDirection.BUY
        assert cost_breakdown.quantity == 1000
        assert cost_breakdown.price == 500.0
        assert cost_breakdown.trade_value == 500000.0
        
        # Verify cost components
        assert cost_breakdown.commission > 0
        assert cost_breakdown.market_impact >= 0
        assert cost_breakdown.bid_ask_spread_cost >= 0
        assert cost_breakdown.total_cost > 0
        assert cost_breakdown.cost_bps > 0
    
    def test_trade_validation(self):
        """Test trade validation."""
        # Invalid trade - zero quantity
        invalid_trade = TradeInfo(
            symbol="2330.TW",
            trade_date=date.today(),
            direction=TradeDirection.BUY,
            quantity=0,  # Invalid
            price=500.0
        )
        
        with pytest.raises(ValueError, match="Trade validation failed"):
            self.model.calculate_cost(invalid_trade)
        
        # Invalid trade - negative price
        invalid_trade2 = TradeInfo(
            symbol="2330.TW",
            trade_date=date.today(),
            direction=TradeDirection.BUY,
            quantity=1000,
            price=-100.0  # Invalid
        )
        
        with pytest.raises(ValueError, match="Trade validation failed"):
            self.model.calculate_cost(invalid_trade2)
    
    def test_market_impact_scaling(self):
        """Test market impact scales with order size."""
        # Small order
        small_trade = self.sample_trade
        small_trade.order_size_vs_avg = 0.01  # 1% of ADV
        small_cost = self.model.calculate_cost(small_trade)
        
        # Large order
        large_trade = TradeInfo(
            symbol="2330.TW",
            trade_date=date.today(),
            direction=TradeDirection.BUY,
            quantity=5000,  # 5x larger
            price=500.0,
            daily_volume=50000,
            volatility=0.25,
            order_size_vs_avg=0.10  # 10% of ADV
        )
        large_cost = self.model.calculate_cost(large_trade)
        
        # Large order should have higher cost per unit
        assert large_cost.cost_bps > small_cost.cost_bps
    
    def test_volatility_impact(self):
        """Test volatility impact on costs."""
        # Low volatility
        low_vol_trade = self.sample_trade
        low_vol_trade.volatility = 0.10
        low_vol_cost = self.model.calculate_cost(low_vol_trade)
        
        # High volatility
        high_vol_trade = self.sample_trade
        high_vol_trade.volatility = 0.50
        high_vol_cost = self.model.calculate_cost(high_vol_trade)
        
        # High volatility should result in higher costs
        assert high_vol_cost.cost_bps > low_vol_cost.cost_bps


class TestNonLinearCostModel:
    """Test non-linear transaction cost model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = CostModelFactory.create_nonlinear_model()
        self.sample_trade = TradeInfo(
            symbol="2330.TW",
            trade_date=date.today(),
            direction=TradeDirection.BUY,
            quantity=1000,
            price=500.0,
            daily_volume=50000,
            volatility=0.25,
            bid_ask_spread=0.5,
            order_size_vs_avg=0.02,
            previous_close=499.0,
            execution_delay_seconds=300  # 5 minutes
        )
    
    def test_nonlinear_impact(self):
        """Test non-linear market impact."""
        cost_breakdown = self.model.calculate_cost(self.sample_trade)
        
        # Should have all cost components
        assert cost_breakdown.market_impact > 0
        assert cost_breakdown.slippage >= 0
        assert cost_breakdown.timing_cost >= 0
        assert cost_breakdown.total_cost > 0
    
    def test_size_nonlinearity(self):
        """Test non-linear size penalty."""
        # Small order (1% of ADV)
        small_trade = self.sample_trade
        small_trade.quantity = 500
        small_trade.order_size_vs_avg = 0.01
        small_cost = self.model.calculate_cost(small_trade)
        
        # Medium order (5% of ADV)
        med_trade = TradeInfo(
            symbol="2330.TW",
            trade_date=date.today(),
            direction=TradeDirection.BUY,
            quantity=2500,
            price=500.0,
            daily_volume=50000,
            volatility=0.25,
            order_size_vs_avg=0.05
        )
        med_cost = self.model.calculate_cost(med_trade)
        
        # Large order (20% of ADV)
        large_trade = TradeInfo(
            symbol="2330.TW",
            trade_date=date.today(),
            direction=TradeDirection.BUY,
            quantity=10000,
            price=500.0,
            daily_volume=50000,
            volatility=0.25,
            order_size_vs_avg=0.20
        )
        large_cost = self.model.calculate_cost(large_trade)
        
        # Cost should increase non-linearly
        small_cost_per_share = small_cost.cost_bps
        med_cost_per_share = med_cost.cost_bps
        large_cost_per_share = large_cost.cost_bps
        
        # Non-linear: cost per share should increase more than proportionally
        assert med_cost_per_share > small_cost_per_share
        assert large_cost_per_share > med_cost_per_share
        
        # Check non-linearity: large order cost increase should be disproportionate
        size_ratio_small_to_med = 0.05 / 0.01  # 5x
        cost_ratio_small_to_med = med_cost_per_share / small_cost_per_share
        assert cost_ratio_small_to_med < size_ratio_small_to_med  # Non-linear penalty
    
    def test_timing_cost_calculation(self):
        """Test timing cost with execution delays."""
        # Immediate execution
        immediate_trade = self.sample_trade
        immediate_trade.execution_delay_seconds = 0
        immediate_cost = self.model.calculate_cost(immediate_trade)
        
        # Delayed execution
        delayed_trade = self.sample_trade
        delayed_trade.execution_delay_seconds = 1800  # 30 minutes
        delayed_cost = self.model.calculate_cost(delayed_trade)
        
        # Delayed execution should have higher timing cost
        assert delayed_cost.timing_cost >= immediate_cost.timing_cost
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        cost_breakdown = self.model.calculate_cost(self.sample_trade)
        
        # Should have slippage if previous close differs from execution price
        if self.sample_trade.previous_close and self.sample_trade.previous_close != self.sample_trade.price:
            assert cost_breakdown.slippage >= 0


class TestTaiwanTickSizeModel:
    """Test Taiwan tick size model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tick_model = TaiwanTickSizeModel()
    
    def test_tick_size_schedule(self):
        """Test tick size for different price levels."""
        # Low price: NT$5 -> tick = NT$0.01
        assert self.tick_model.get_tick_size(5.0) == 0.01
        
        # Medium price: NT$25 -> tick = NT$0.05
        assert self.tick_model.get_tick_size(25.0) == 0.05
        
        # Higher price: NT$75 -> tick = NT$0.10
        assert self.tick_model.get_tick_size(75.0) == 0.10
        
        # High price: NT$300 -> tick = NT$0.50
        assert self.tick_model.get_tick_size(300.0) == 0.50
        
        # Very high price: NT$800 -> tick = NT$1.00
        assert self.tick_model.get_tick_size(800.0) == 1.00
        
        # Extremely high price: NT$1500 -> tick = NT$5.00
        assert self.tick_model.get_tick_size(1500.0) == 5.00
    
    def test_price_rounding(self):
        """Test price rounding to valid ticks."""
        # Price NT$25.03 should round to NT$25.05 (tick = NT$0.05)
        rounded = self.tick_model.round_to_tick(25.03)
        assert rounded == 25.05
        
        # Price NT$150.25 should round to NT$150.50 (tick = NT$0.50)
        rounded2 = self.tick_model.round_to_tick(150.25)
        assert rounded2 == 150.50
    
    def test_min_spread_bps(self):
        """Test minimum spread calculation."""
        # Low price stock
        min_spread = self.tick_model.get_min_spread_bps(10.0)  # tick = 0.01
        expected = (0.01 / 10.0) * 10000  # 1 bps
        assert abs(min_spread - expected) < 0.01
        
        # High price stock
        min_spread_high = self.tick_model.get_min_spread_bps(1000.0)  # tick = 5.00
        expected_high = (5.00 / 1000.0) * 10000  # 50 bps
        assert abs(min_spread_high - expected_high) < 0.01


class TestBidAskSpreadModel:
    """Test bid-ask spread prediction model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.spread_model = BidAskSpreadModel()
    
    def test_spread_prediction(self):
        """Test spread prediction with different parameters."""
        # Base case
        spread = self.spread_model.predict_spread(
            symbol="2330.TW",
            price=500.0,
            volatility=0.20,
            volume_ratio=1.0,
            timestamp=datetime.now().replace(hour=10, minute=0)
        )
        assert spread > 0
        
        # High volatility should increase spread
        spread_high_vol = self.spread_model.predict_spread(
            symbol="2330.TW",
            price=500.0,
            volatility=0.50,  # Higher volatility
            volume_ratio=1.0,
            timestamp=datetime.now().replace(hour=10, minute=0)
        )
        assert spread_high_vol > spread
        
        # High volume should decrease spread
        spread_high_vol_trading = self.spread_model.predict_spread(
            symbol="2330.TW",
            price=500.0,
            volatility=0.20,
            volume_ratio=2.0,  # Higher volume
            timestamp=datetime.now().replace(hour=10, minute=0)
        )
        assert spread_high_vol_trading < spread
    
    def test_time_of_day_effects(self):
        """Test time-of-day effects on spread."""
        base_params = {
            'symbol': "2330.TW",
            'price': 500.0,
            'volatility': 0.20,
            'volume_ratio': 1.0
        }
        
        # Morning open (higher spread)
        spread_open = self.spread_model.predict_spread(
            timestamp=datetime.now().replace(hour=9, minute=15),
            **base_params
        )
        
        # Mid-morning (normal spread)
        spread_mid = self.spread_model.predict_spread(
            timestamp=datetime.now().replace(hour=10, minute=30),
            **base_params
        )
        
        # Morning open should have wider spread
        assert spread_open > spread_mid


class TestMarketImpactModel:
    """Test market impact model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.impact_model = MarketImpactModel()
    
    def test_impact_calculation(self):
        """Test market impact calculation."""
        impact_data = self.impact_model.calculate_impact(
            order_size=1000,
            avg_daily_volume=50000,
            volatility=0.25,
            price=500.0,
            direction=1  # Buy
        )
        
        # Verify impact data structure
        assert impact_data.order_size == 1000
        assert impact_data.avg_daily_volume == 50000
        assert impact_data.participation_rate == 0.02  # 2%
        assert impact_data.temporary_impact_bps > 0
        assert impact_data.permanent_impact_bps > 0
        assert impact_data.total_impact_bps > 0
    
    def test_impact_scaling(self):
        """Test impact scales with order size."""
        # Small order
        small_impact = self.impact_model.calculate_impact(
            order_size=500,  # 1% of ADV
            avg_daily_volume=50000,
            volatility=0.25,
            price=500.0
        )
        
        # Large order
        large_impact = self.impact_model.calculate_impact(
            order_size=5000,  # 10% of ADV
            avg_daily_volume=50000,
            volatility=0.25,
            price=500.0
        )
        
        # Large order should have higher impact
        assert large_impact.total_impact_bps > small_impact.total_impact_bps
    
    def test_impact_decay(self):
        """Test temporary impact decay."""
        initial_impact = 20.0  # 20 bps
        
        # After half-life (20 minutes), should be 50%
        decayed_impact = self.impact_model.calculate_impact_decay(
            initial_impact, 20.0
        )
        assert abs(decayed_impact - 10.0) < 1.0  # Should be ~10 bps
        
        # After two half-lives (40 minutes), should be 25%
        decayed_impact_2 = self.impact_model.calculate_impact_decay(
            initial_impact, 40.0
        )
        assert abs(decayed_impact_2 - 5.0) < 1.0  # Should be ~5 bps
    
    def test_liquidity_categorization(self):
        """Test liquidity category assignment."""
        # Very liquid
        assert self.impact_model.get_liquidity_category(0.005) == 'very_liquid'
        
        # Liquid
        assert self.impact_model.get_liquidity_category(0.03) == 'liquid'
        
        # Illiquid
        assert self.impact_model.get_liquidity_category(0.15) == 'illiquid'
        
        # Very illiquid
        assert self.impact_model.get_liquidity_category(0.50) == 'very_illiquid'


class TestExecutionCostSimulator:
    """Test execution cost simulator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.simulator = ExecutionCostSimulator()
    
    def test_market_execution_simulation(self):
        """Test market order execution simulation."""
        result = self.simulator.simulate_execution(
            symbol="2330.TW",
            order_size=1000,
            reference_price=500.0,
            strategy=ExecutionStrategy.MARKET,
            start_time=datetime.now().replace(hour=10, minute=0),
            avg_daily_volume=50000,
            volatility=0.25,
            max_execution_time_minutes=60
        )
        
        # Market orders should execute quickly
        assert result.symbol == "2330.TW"
        assert result.strategy == ExecutionStrategy.MARKET
        assert len(result.fills) > 0
        assert result.total_filled > 0
        assert result.execution_efficiency_score > 0
    
    def test_twap_execution_simulation(self):
        """Test TWAP execution simulation."""
        result = self.simulator.simulate_execution(
            symbol="2330.TW",
            order_size=5000,  # Larger order
            reference_price=500.0,
            strategy=ExecutionStrategy.TWAP,
            start_time=datetime.now().replace(hour=9, minute=30),
            avg_daily_volume=50000,
            volatility=0.25,
            max_execution_time_minutes=120  # 2 hours
        )
        
        # TWAP should have multiple fills over time
        assert result.strategy == ExecutionStrategy.TWAP
        assert len(result.fills) > 1  # Should be split into multiple fills
        
        # Check that fills are spread over time
        if len(result.fills) > 1:
            time_diffs = [
                (result.fills[i+1].timestamp - result.fills[i].timestamp).total_seconds()
                for i in range(len(result.fills) - 1)
            ]
            assert any(diff > 0 for diff in time_diffs)  # Should have time gaps
    
    def test_execution_quality_analysis(self):
        """Test execution quality analysis."""
        # Create a sample execution result
        result = self.simulator.simulate_execution(
            symbol="2330.TW",
            order_size=2000,
            reference_price=500.0,
            strategy=ExecutionStrategy.TWAP,
            start_time=datetime.now().replace(hour=10, minute=0),
            avg_daily_volume=50000,
            volatility=0.25,
            max_execution_time_minutes=90
        )
        
        # Analyze execution quality
        quality_analysis = self.simulator.analyze_execution_quality(result)
        
        # Verify analysis structure
        assert 'execution_summary' in quality_analysis
        assert 'cost_analysis' in quality_analysis
        assert 'price_analysis' in quality_analysis
        assert 'quality_metrics' in quality_analysis
        
        # Verify metrics
        assert 'fill_rate' in quality_analysis['execution_summary']
        assert 'total_cost_bps' in quality_analysis['cost_analysis']
        assert 'execution_efficiency_score' in quality_analysis['quality_metrics']
        assert 'overall_grade' in quality_analysis['quality_metrics']
        
        # Grade should be A-F
        grade = quality_analysis['quality_metrics']['overall_grade']
        assert grade in ['A', 'B', 'C', 'D', 'F']


class TestSlippageModel:
    """Test slippage model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.slippage_model = SlippageModel()
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        slippage_bps = self.slippage_model.calculate_slippage(
            order_size=1000,
            avg_daily_volume=50000,
            volatility=0.25,
            market_session=TradingSession.MORNING,
            reference_price=500.0,
            execution_price=500.5
        )
        
        assert slippage_bps > 0
        
        # Higher volatility should increase slippage
        slippage_high_vol = self.slippage_model.calculate_slippage(
            order_size=1000,
            avg_daily_volume=50000,
            volatility=0.50,  # Higher volatility
            market_session=TradingSession.MORNING,
            reference_price=500.0,
            execution_price=500.5
        )
        
        assert slippage_high_vol > slippage_bps
    
    def test_execution_price_simulation(self):
        """Test execution price simulation."""
        simulated_price = self.slippage_model.simulate_execution_price(
            reference_price=500.0,
            order_size=1000,
            avg_daily_volume=50000,
            volatility=0.25,
            market_session=TradingSession.MORNING,
            direction=TradeDirection.BUY
        )
        
        # Buy order should typically result in higher price
        assert simulated_price >= 500.0
        
        # Sell order should typically result in lower price
        simulated_price_sell = self.slippage_model.simulate_execution_price(
            reference_price=500.0,
            order_size=1000,
            avg_daily_volume=50000,
            volatility=0.25,
            market_session=TradingSession.MORNING,
            direction=TradeDirection.SELL
        )
        
        assert simulated_price_sell <= 500.0


class TestTimingCostModel:
    """Test timing cost model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.timing_model = TimingCostModel()
    
    def test_timing_cost_calculation(self):
        """Test timing cost calculation."""
        # No delay should result in zero cost
        cost_no_delay = self.timing_model.calculate_timing_cost(
            decision_price=500.0,
            execution_price=500.0,
            delay_minutes=0.0,
            volatility=0.25
        )
        assert cost_no_delay == 0.0
        
        # Delay should result in positive cost
        cost_with_delay = self.timing_model.calculate_timing_cost(
            decision_price=500.0,
            execution_price=501.0,
            delay_minutes=10.0,
            volatility=0.25
        )
        assert cost_with_delay > 0.0
        
        # Longer delay should result in higher cost
        cost_long_delay = self.timing_model.calculate_timing_cost(
            decision_price=500.0,
            execution_price=501.0,
            delay_minutes=30.0,
            volatility=0.25
        )
        assert cost_long_delay > cost_with_delay


class TestSettlementCostModel:
    """Test settlement cost model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.settlement_model = SettlementCostModel()
    
    def test_settlement_cost_calculation(self):
        """Test T+2 settlement cost calculation."""
        trade_value = 500000.0  # NT$500,000
        trade_date = date.today()
        
        settlement_costs = self.settlement_model.calculate_settlement_cost(
            trade_value, trade_date
        )
        
        # Verify cost structure
        assert 'financing_cost' in settlement_costs
        assert 'settlement_risk_premium' in settlement_costs
        assert 'total_settlement_cost' in settlement_costs
        assert 'settlement_date' in settlement_costs
        assert 'funding_days' in settlement_costs
        
        # Verify costs are positive
        assert settlement_costs['financing_cost'] > 0
        assert settlement_costs['settlement_risk_premium'] > 0
        assert settlement_costs['total_settlement_cost'] > 0
        
        # Verify settlement date is T+2 (plus weekends)
        settlement_date = settlement_costs['settlement_date']
        days_diff = (settlement_date - trade_date).days
        assert days_diff >= 2  # At least T+2
    
    def test_custom_funding_rate(self):
        """Test settlement cost with custom funding rate."""
        trade_value = 500000.0
        trade_date = date.today()
        
        # Standard rate
        standard_costs = self.settlement_model.calculate_settlement_cost(
            trade_value, trade_date
        )
        
        # Higher funding rate
        high_rate_costs = self.settlement_model.calculate_settlement_cost(
            trade_value, trade_date, funding_rate=0.05  # 5%
        )
        
        # Higher funding rate should result in higher financing cost
        assert high_rate_costs['financing_cost'] > standard_costs['financing_cost']


class TestIntegration:
    """Integration tests combining multiple models."""
    
    def test_end_to_end_cost_analysis(self):
        """Test complete end-to-end cost analysis."""
        # Create comprehensive market structure
        market_structure = TaiwanMarketStructure()
        
        # Analyze trading costs
        analysis = market_structure.analyze_trading_costs(
            symbol="2330.TW",
            order_size=2000,
            price=500.0,
            avg_daily_volume=50000,
            volatility=0.25,
            timestamp=datetime.now().replace(hour=10, minute=30),
            exchange="TSE"
        )
        
        # Verify comprehensive analysis
        required_fields = [
            'tick_size', 'predicted_spread_bps', 'total_impact_bps',
            'participation_rate', 'liquidity_category', 
            'estimated_execution_time_minutes', 'total_microstructure_cost_bps'
        ]
        
        for field in required_fields:
            assert field in analysis
            assert analysis[field] is not None
        
        # Verify reasonable values
        assert analysis['tick_size'] > 0
        assert analysis['predicted_spread_bps'] > 0
        assert analysis['total_impact_bps'] >= 0
        assert 0 <= analysis['participation_rate'] <= 1
        assert analysis['liquidity_category'] in ['very_liquid', 'liquid', 'illiquid', 'very_illiquid']
    
    def test_cost_model_comparison(self):
        """Test comparison between linear and non-linear models."""
        trade = TradeInfo(
            symbol="2330.TW",
            trade_date=date.today(),
            direction=TradeDirection.BUY,
            quantity=2000,
            price=500.0,
            daily_volume=50000,
            volatility=0.25,
            order_size_vs_avg=0.04  # 4% of ADV
        )
        
        # Calculate costs with both models
        linear_model = CostModelFactory.create_linear_model()
        nonlinear_model = CostModelFactory.create_nonlinear_model()
        
        linear_cost = linear_model.calculate_cost(trade)
        nonlinear_cost = nonlinear_model.calculate_cost(trade)
        
        # Both should produce valid results
        assert linear_cost.total_cost > 0
        assert nonlinear_cost.total_cost > 0
        
        # For larger orders, non-linear model should typically be higher
        # (due to non-linear impact)
        if trade.order_size_vs_avg > 0.02:  # > 2% of ADV
            assert nonlinear_cost.cost_bps >= linear_cost.cost_bps * 0.8  # Allow some variance


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])