"""
Real Taiwan Market Scenario Integration Tests.

This module tests the complete Point-in-Time Data Management System
with realistic Taiwan market scenarios, ensuring proper handling of:
- Real market data patterns
- Corporate action timing
- Settlement procedures
- Holiday handling
- Market microstructure specifics
"""

import pytest
import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from data.core.temporal import (
    TemporalValue, InMemoryTemporalStore, DataType, TemporalDataManager
)
from data.models.taiwan_market import (
    TaiwanMarketData, TaiwanSettlement, TaiwanCorporateAction,
    TaiwanFundamental, TradingStatus, CorporateActionType,
    TaiwanMarketCode, create_taiwan_trading_calendar,
    TaiwanMarketDataValidator
)
from data.pipeline.pit_engine import (
    PointInTimeEngine, PITQuery, QueryMode, BiasCheckLevel
)
from data.pipeline.incremental_updater import (
    IncrementalUpdater, UpdateRequest, UpdateMode, UpdatePriority
)
from data.api.pit_data_service import PITDataService


@pytest.fixture
def taiwan_market_calendar():
    """Create Taiwan trading calendar for 2024."""
    return create_taiwan_trading_calendar(2024)


@pytest.fixture
def temporal_store():
    """Create temporal store for testing."""
    return InMemoryTemporalStore()


@pytest.fixture
def pit_engine(temporal_store, taiwan_market_calendar):
    """Create PIT engine with Taiwan market calendar."""
    return PointInTimeEngine(temporal_store, taiwan_market_calendar)


@pytest.fixture
def market_validator():
    """Create Taiwan market data validator."""
    return TaiwanMarketDataValidator()


class TestTSMCDividendScenario:
    """Test realistic TSMC dividend scenario."""
    
    def test_tsmc_q1_2024_dividend_complete_cycle(self, temporal_store, pit_engine):
        """Test complete TSMC Q1 2024 dividend cycle."""
        
        # TSMC Q1 2024 dividend scenario (realistic dates)
        announcement_date = date(2024, 4, 18)  # Earnings call
        ex_date = date(2024, 7, 18)  # Ex-dividend date
        record_date = date(2024, 7, 19)  # Record date
        payable_date = date(2024, 10, 17)  # Payment date
        dividend_amount = Decimal("2.75")  # TWD 2.75 per share
        
        # Store price data before announcement
        pre_announcement_price = TaiwanMarketData(
            symbol="2330",
            data_date=date(2024, 4, 17),
            as_of_date=date(2024, 4, 17),
            open_price=Decimal("740.00"),
            high_price=Decimal("748.00"),
            low_price=Decimal("735.00"),
            close_price=Decimal("745.00"),
            volume=28500000,
            turnover=Decimal("21217500000"),
            trading_status=TradingStatus.NORMAL,
            market_cap=Decimal("19325000000000")  # ~19.3T TWD
        )
        
        for temporal_value in pre_announcement_price.to_temporal_values():
            temporal_store.store(temporal_value)
        
        # Store dividend announcement
        dividend_action = TaiwanCorporateAction(
            symbol="2330",
            action_type=CorporateActionType.DIVIDEND_CASH,
            announcement_date=announcement_date,
            ex_date=ex_date,
            record_date=record_date,
            payable_date=payable_date,
            amount=dividend_amount,
            description="Q1 2024 cash dividend - TWD 2.75 per share"
        )
        
        temporal_store.store(dividend_action.to_temporal_value())
        
        # Store price data on ex-date (should reflect dividend adjustment)
        ex_date_price = TaiwanMarketData(
            symbol="2330",
            data_date=ex_date,
            as_of_date=ex_date,
            open_price=Decimal("742.25"),  # Opening ~2.75 lower (dividend adjustment)
            high_price=Decimal("750.00"),
            low_price=Decimal("740.00"),
            close_price=Decimal("747.50"),
            volume=35200000,  # Higher volume on ex-date
            turnover=Decimal("26292000000"),
            trading_status=TradingStatus.NORMAL
        )
        
        for temporal_value in ex_date_price.to_temporal_values():
            temporal_store.store(temporal_value)
        
        # Test queries at different stages of dividend cycle
        
        # 1. Before announcement - no dividend data available
        pre_announcement_query = PITQuery(
            symbols=["2330"],
            as_of_date=date(2024, 4, 15),
            data_types=[DataType.CORPORATE_ACTION, DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        pre_result = pit_engine.execute_query(pre_announcement_query)
        assert len(pre_result.bias_violations) == 0
        
        # Should not see dividend action
        corp_actions = pre_result.data.get("2330", {}).get(DataType.CORPORATE_ACTION)
        assert corp_actions is None, "Dividend visible before announcement"
        
        # 2. After announcement but before ex-date - dividend visible but not effective
        post_announcement_query = PITQuery(
            symbols=["2330"],
            as_of_date=date(2024, 5, 15),
            data_types=[DataType.CORPORATE_ACTION, DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        post_result = pit_engine.execute_query(post_announcement_query)
        assert len(post_result.bias_violations) == 0
        
        # Should see dividend action
        corp_actions = post_result.data.get("2330", {}).get(DataType.CORPORATE_ACTION)
        assert corp_actions is not None, "Dividend not visible after announcement"
        assert corp_actions.value["amount"] == 2.75
        
        # 3. On ex-date - dividend should affect pricing
        ex_date_query = PITQuery(
            symbols=["2330"],
            as_of_date=ex_date,
            data_types=[DataType.CORPORATE_ACTION, DataType.PRICE],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        ex_result = pit_engine.execute_query(ex_date_query)
        assert len(ex_result.bias_violations) == 0
        
        # Should see both dividend and ex-date price
        corp_actions = ex_result.data.get("2330", {}).get(DataType.CORPORATE_ACTION)
        price_data = ex_result.data.get("2330", {}).get(DataType.PRICE)
        
        assert corp_actions is not None
        assert price_data is not None
        
        # Verify dividend affects pricing as expected
        assert corp_actions.value["amount"] == 2.75
        assert price_data.value == Decimal("747.50")  # Ex-date closing price


class TestTaiwanEarningsScenario:
    """Test realistic Taiwan earnings scenario."""
    
    def test_taiwan_q4_2023_earnings_cycle(self, temporal_store, pit_engine):
        """Test Q4 2023 earnings release cycle for major Taiwan stocks."""
        
        # Realistic Taiwan Q4 2023 earnings scenario
        quarter_end = date(2023, 12, 31)
        
        # Different companies have different reporting timelines
        earnings_data = [
            {
                "symbol": "2330",  # TSMC
                "name": "Taiwan Semiconductor",
                "announcement_date": date(2024, 1, 18),  # 18 days after quarter
                "revenue": Decimal("625851000000"),  # 625.85B TWD
                "net_income": Decimal("295900000000"),  # 295.9B TWD  
                "eps": Decimal("11.41"),  # TWD 11.41
                "roe": Decimal("26.4")
            },
            {
                "symbol": "2317",  # Hon Hai
                "name": "Hon Hai Precision",
                "announcement_date": date(2024, 2, 14),  # 45 days after quarter
                "revenue": Decimal("1850000000000"),  # 1.85T TWD
                "net_income": Decimal("35200000000"),  # 35.2B TWD
                "eps": Decimal("2.51"),  # TWD 2.51
                "roe": Decimal("8.9")
            },
            {
                "symbol": "2454",  # MediaTek
                "name": "MediaTek Inc",
                "announcement_date": date(2024, 1, 31),  # 31 days after quarter
                "revenue": Decimal("186220000000"),  # 186.22B TWD
                "net_income": Decimal("25800000000"),  # 25.8B TWD
                "eps": Decimal("16.25"),  # TWD 16.25
                "roe": Decimal("18.7")
            }
        ]
        
        # Store earnings data for each company
        for earning in earnings_data:
            fundamental = TaiwanFundamental(
                symbol=earning["symbol"],
                report_date=quarter_end,
                announcement_date=earning["announcement_date"],
                fiscal_year=2023,
                fiscal_quarter=4,
                revenue=earning["revenue"],
                net_income=earning["net_income"],
                eps=earning["eps"],
                roe=earning["roe"]
            )
            
            temporal_store.store(fundamental.to_temporal_value())
        
        # Test earnings visibility at different dates
        
        # 1. Before any earnings announcements
        pre_earnings_query = PITQuery(
            symbols=["2330", "2317", "2454"],
            as_of_date=date(2024, 1, 10),
            data_types=[DataType.FUNDAMENTAL],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        pre_result = pit_engine.execute_query(pre_earnings_query)
        assert len(pre_result.bias_violations) == 0
        
        # Should not see any Q4 2023 earnings yet
        for symbol in ["2330", "2317", "2454"]:
            fundamentals = pre_result.data.get(symbol, {}).get(DataType.FUNDAMENTAL)
            if fundamentals:
                # If there's data, it should be from an earlier quarter
                assert fundamentals.metadata.get("fiscal_year", 0) < 2023 or \
                       fundamentals.metadata.get("fiscal_quarter", 0) < 4
        
        # 2. After TSMC earnings but before others
        post_tsmc_query = PITQuery(
            symbols=["2330", "2317", "2454"],
            as_of_date=date(2024, 1, 25),
            data_types=[DataType.FUNDAMENTAL],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        post_tsmc_result = pit_engine.execute_query(post_tsmc_query)
        assert len(post_tsmc_result.bias_violations) == 0
        
        # Should see TSMC Q4 2023 earnings
        tsmc_fundamentals = post_tsmc_result.data.get("2330", {}).get(DataType.FUNDAMENTAL)
        assert tsmc_fundamentals is not None, "TSMC Q4 earnings not visible after announcement"
        assert tsmc_fundamentals.value["eps"] == 11.41
        assert tsmc_fundamentals.metadata["fiscal_year"] == 2023
        assert tsmc_fundamentals.metadata["fiscal_quarter"] == 4
        
        # Should not see Hon Hai Q4 2023 earnings yet
        hon_hai_fundamentals = post_tsmc_result.data.get("2317", {}).get(DataType.FUNDAMENTAL)
        if hon_hai_fundamentals:
            assert not (hon_hai_fundamentals.metadata.get("fiscal_year") == 2023 and 
                       hon_hai_fundamentals.metadata.get("fiscal_quarter") == 4)
        
        # 3. After all earnings announcements
        post_all_query = PITQuery(
            symbols=["2330", "2317", "2454"],
            as_of_date=date(2024, 2, 20),
            data_types=[DataType.FUNDAMENTAL],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        post_all_result = pit_engine.execute_query(post_all_query)
        assert len(post_all_result.bias_violations) == 0
        
        # Should see all Q4 2023 earnings
        for symbol in ["2330", "2317", "2454"]:
            fundamentals = post_all_result.data.get(symbol, {}).get(DataType.FUNDAMENTAL)
            assert fundamentals is not None, f"Q4 earnings not visible for {symbol}"
            assert fundamentals.metadata["fiscal_year"] == 2023
            assert fundamentals.metadata["fiscal_quarter"] == 4
        
        # Verify specific values
        assert post_all_result.data["2330"][DataType.FUNDAMENTAL].value["eps"] == 11.41
        assert post_all_result.data["2317"][DataType.FUNDAMENTAL].value["eps"] == 2.51
        assert post_all_result.data["2454"][DataType.FUNDAMENTAL].value["eps"] == 16.25


class TestTaiwanMarketCrashScenario:
    """Test Taiwan market crash scenario (like 2008 or COVID-19)."""
    
    def test_covid19_market_crash_march_2020(self, temporal_store, pit_engine, market_validator):
        """Test COVID-19 market crash scenario in Taiwan market."""
        
        # Simulate COVID-19 market crash in Taiwan (March 2020)
        crash_date = date(2020, 3, 19)  # Taiwan market crash day
        
        # Pre-crash market data (normal trading)
        pre_crash_data = [
            TaiwanMarketData(
                symbol="0050",  # Taiwan 50 ETF
                data_date=date(2020, 3, 18),
                as_of_date=date(2020, 3, 18),
                open_price=Decimal("88.50"),
                high_price=Decimal("89.20"),
                low_price=Decimal("87.80"),
                close_price=Decimal("88.90"),
                volume=12500000,
                turnover=Decimal("1112500000"),
                trading_status=TradingStatus.NORMAL
            ),
            TaiwanMarketData(
                symbol="2330",  # TSMC
                data_date=date(2020, 3, 18),
                as_of_date=date(2020, 3, 18),
                open_price=Decimal("295.00"),
                high_price=Decimal("299.50"),
                low_price=Decimal("293.00"),
                close_price=Decimal("297.50"),
                volume=35200000,
                turnover=Decimal("10472000000"),
                trading_status=TradingStatus.NORMAL
            )
        ]
        
        # Crash day data (extreme movements)
        crash_data = [
            TaiwanMarketData(
                symbol="0050",  # Taiwan 50 ETF
                data_date=crash_date,
                as_of_date=crash_date,
                open_price=Decimal("80.01"),  # Gap down opening (daily limit)
                high_price=Decimal("81.50"),
                low_price=Decimal("80.01"),  # Hit daily limit
                close_price=Decimal("80.01"),  # Closed at daily limit down
                volume=85000000,  # Massive volume (7x normal)
                turnover=Decimal("6800000000"),
                trading_status=TradingStatus.LIMIT_DOWN,
                price_limit_up=Decimal("97.79"),  # +10% limit
                price_limit_down=Decimal("80.01")  # -10% limit (hit)
            ),
            TaiwanMarketData(
                symbol="2330",  # TSMC
                data_date=crash_date,
                as_of_date=crash_date,
                open_price=Decimal("267.75"),  # -10% gap down
                high_price=Decimal("275.00"),
                low_price=Decimal("267.75"),
                close_price=Decimal("267.75"),  # Closed at daily limit
                volume=125000000,  # 3.5x normal volume
                turnover=Decimal("33437500000"),
                trading_status=TradingStatus.LIMIT_DOWN,
                price_limit_up=Decimal("327.25"),
                price_limit_down=Decimal("267.75")
            )
        ]
        
        # Store all market data
        for market_data in pre_crash_data + crash_data:
            for temporal_value in market_data.to_temporal_values():
                temporal_store.store(temporal_value)
        
        # Test market crash day queries
        crash_query = PITQuery(
            symbols=["0050", "2330"],
            as_of_date=crash_date,
            data_types=[DataType.PRICE, DataType.VOLUME],
            mode=QueryMode.STRICT,
            bias_check=BiasCheckLevel.STRICT
        )
        
        crash_result = pit_engine.execute_query(crash_query)
        assert len(crash_result.bias_violations) == 0
        
        # Verify crash data is properly handled
        etf_price = crash_result.data["0050"][DataType.PRICE]
        tsmc_price = crash_result.data["2330"][DataType.PRICE]
        
        # Both should show limit down prices
        assert etf_price.value == Decimal("80.01")
        assert tsmc_price.value == Decimal("267.75")
        
        # Verify volume is elevated
        etf_volume = crash_result.data["0050"][DataType.VOLUME]
        tsmc_volume = crash_result.data["2330"][DataType.VOLUME]
        
        assert etf_volume.value == 85000000  # Massive volume
        assert tsmc_volume.value == 125000000
        
        # Test data validation on crash day
        for market_data in crash_data:
            # Previous day close for validation
            prev_close = Decimal("88.90") if market_data.symbol == "0050" else Decimal("297.50")
            
            issues = market_validator.validate_price_data(market_data, prev_close)
            
            # Should detect daily limit hit
            daily_limit_issues = [i for i in issues if "daily limit" in i.lower()]
            assert len(daily_limit_issues) > 0, f"Daily limit not detected for {market_data.symbol}"
        
        # Test subsequent recovery day
        recovery_date = date(2020, 3, 20)
        
        recovery_data = TaiwanMarketData(
            symbol="0050",
            data_date=recovery_date,
            as_of_date=recovery_date,
            open_price=Decimal("81.50"),  # Small gap up
            high_price=Decimal("85.00"),
            low_price=Decimal("80.50"),
            close_price=Decimal("84.20"),  # +5.2% recovery
            volume=45000000,  # Still elevated but lower
            turnover=Decimal("3780000000"),
            trading_status=TradingStatus.NORMAL
        )
        
        for temporal_value in recovery_data.to_temporal_values():
            temporal_store.store(temporal_value)
        
        # Verify recovery data
        recovery_query = PITQuery(
            symbols=["0050"],
            as_of_date=recovery_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT
        )
        
        recovery_result = pit_engine.execute_query(recovery_query)
        recovery_price = recovery_result.data["0050"][DataType.PRICE]
        assert recovery_price.value == Decimal("84.20")


class TestTaiwanHolidayScenario:
    """Test Taiwan market holiday scenarios."""
    
    def test_chinese_new_year_holiday_period(self, temporal_store, pit_engine, taiwan_market_calendar):
        """Test Chinese New Year holiday period data handling."""
        
        # 2024 Chinese New Year period
        cny_eve = date(2024, 2, 9)   # Friday - Half day trading
        cny_start = date(2024, 2, 10)  # Saturday - Holiday start
        cny_week = [date(2024, 2, 10) + timedelta(days=i) for i in range(7)]
        trading_resumes = date(2024, 2, 17)  # Saturday (but first trading day is Monday 2/19)
        
        # Store data before holiday
        pre_holiday_data = TaiwanMarketData(
            symbol="2330",
            data_date=cny_eve,
            as_of_date=cny_eve,
            open_price=Decimal("600.00"),
            high_price=Decimal("605.00"),
            low_price=Decimal("598.00"),
            close_price=Decimal("602.00"),
            volume=18500000,  # Lower volume (half day)
            turnover=Decimal("11137000000"),
            trading_status=TradingStatus.NORMAL
        )
        
        for temporal_value in pre_holiday_data.to_temporal_values():
            temporal_store.store(temporal_value)
        
        # Store data after holiday (Monday 2/19)
        post_holiday_date = date(2024, 2, 19)  # Monday after CNY
        post_holiday_data = TaiwanMarketData(
            symbol="2330",
            data_date=post_holiday_date,
            as_of_date=post_holiday_date,
            open_price=Decimal("595.00"),  # Gap down after holiday
            high_price=Decimal("608.00"),
            low_price=Decimal("592.00"),
            close_price=Decimal("605.50"),
            volume=42000000,  # Higher volume on resumption
            turnover=Decimal("25410000000"),
            trading_status=TradingStatus.NORMAL
        )
        
        for temporal_value in post_holiday_data.to_temporal_values():
            temporal_store.store(temporal_value)
        
        # Test queries during holiday period
        for holiday_date in cny_week:
            holiday_query = PITQuery(
                symbols=["2330"],
                as_of_date=holiday_date,
                data_types=[DataType.PRICE],
                mode=QueryMode.STRICT,
                bias_check=BiasCheckLevel.STRICT
            )
            
            holiday_result = pit_engine.execute_query(holiday_query)
            assert len(holiday_result.bias_violations) == 0
            
            # Should get last trading day's data (not future data)
            if "2330" in holiday_result.data and DataType.PRICE in holiday_result.data["2330"]:
                price_data = holiday_result.data["2330"][DataType.PRICE]
                assert price_data.as_of_date <= holiday_date
                assert price_data.value == Decimal("602.00")  # Should be pre-holiday price
        
        # Test query after holiday resumption
        post_holiday_query = PITQuery(
            symbols=["2330"],
            as_of_date=post_holiday_date,
            data_types=[DataType.PRICE],
            mode=QueryMode.STRICT
        )
        
        post_result = pit_engine.execute_query(post_holiday_query)
        price_data = post_result.data["2330"][DataType.PRICE]
        assert price_data.value == Decimal("605.50")  # Should be post-holiday price


class TestTaiwanSettlementScenario:
    """Test Taiwan T+2 settlement scenarios."""
    
    def test_complex_settlement_scenarios(self, temporal_store, pit_engine, taiwan_market_calendar):
        """Test complex T+2 settlement scenarios including weekends and holidays."""
        
        # Test various settlement scenarios
        settlement_scenarios = [
            {
                "name": "Normal weekday settlement",
                "trade_date": date(2024, 1, 15),  # Monday
                "expected_settlement": date(2024, 1, 17),  # Wednesday
                "scenario": "normal"
            },
            {
                "name": "Weekend skip settlement", 
                "trade_date": date(2024, 1, 18),  # Thursday
                "expected_settlement": date(2024, 1, 22),  # Monday (skip weekend)
                "scenario": "weekend_skip"
            },
            {
                "name": "Holiday skip settlement",
                "trade_date": date(2024, 2, 7),   # Wednesday before CNY
                "expected_settlement": date(2024, 2, 19),  # Monday after CNY holiday
                "scenario": "holiday_skip"
            },
            {
                "name": "Month-end settlement",
                "trade_date": date(2024, 1, 30),  # Tuesday
                "expected_settlement": date(2024, 2, 1),   # Thursday (cross month)
                "scenario": "month_cross"
            }
        ]
        
        for scenario in settlement_scenarios:
            trade_date = scenario["trade_date"]
            expected_settlement = scenario["expected_settlement"]
            
            # Create Taiwan settlement
            settlement = TaiwanSettlement.calculate_t2_settlement(trade_date, taiwan_market_calendar)
            
            # Verify settlement date calculation
            assert settlement.settlement_date >= expected_settlement, \
                f"Settlement calculation wrong for {scenario['name']}: " \
                f"got {settlement.settlement_date}, expected >= {expected_settlement}"
            
            # Verify T+2 minimum requirement
            assert settlement.settlement_lag_days >= 2, \
                f"Settlement lag too short for {scenario['name']}: {settlement.settlement_lag_days}"
            
            # Store trade data
            trade_value = TemporalValue(
                value=Decimal("100.00"),
                as_of_date=trade_date,
                value_date=trade_date,
                data_type=DataType.PRICE,
                symbol="2330",
                metadata={
                    "scenario": scenario["scenario"],
                    "settlement_date": settlement.settlement_date.isoformat(),
                    "trade_execution": True
                }
            )
            temporal_store.store(trade_value)
        
        # Test queries respecting settlement timing
        for scenario in settlement_scenarios:
            trade_date = scenario["trade_date"]
            
            # Query should work on trade date
            trade_query = PITQuery(
                symbols=["2330"],
                as_of_date=trade_date,
                data_types=[DataType.PRICE],
                mode=QueryMode.STRICT,
                bias_check=BiasCheckLevel.STRICT
            )
            
            trade_result = pit_engine.execute_query(trade_query)
            assert len(trade_result.bias_violations) == 0
            
            # Should find the trade data
            price_data = trade_result.data.get("2330", {}).get(DataType.PRICE)
            if price_data and price_data.metadata.get("scenario") == scenario["scenario"]:
                assert price_data.metadata["trade_execution"] is True


class TestTaiwanComprehensiveIntegration:
    """Comprehensive integration test combining all Taiwan market features."""
    
    def test_comprehensive_taiwan_market_simulation(self, temporal_store, pit_engine, 
                                                   taiwan_market_calendar, market_validator):
        """Test comprehensive Taiwan market simulation with all features."""
        
        # Create a comprehensive Taiwan market simulation
        symbols = ["2330", "2317", "0050"]  # TSMC, Hon Hai, Taiwan 50 ETF
        start_date = date(2024, 1, 1)
        end_date = date(2024, 3, 31)  # Q1 2024
        
        total_data_points = 0
        total_violations = 0
        
        # Generate comprehensive market data
        current_date = start_date
        while current_date <= end_date:
            # Check if trading day using calendar
            if current_date in taiwan_market_calendar and taiwan_market_calendar[current_date].is_trading_day:
                
                for symbol in symbols:
                    # Generate realistic price data
                    base_prices = {"2330": 600, "2317": 90, "0050": 120}
                    base_price = base_prices[symbol]
                    
                    # Add some realistic volatility
                    days_from_start = (current_date - start_date).days
                    volatility = 0.02 * (1 + 0.1 * (days_from_start % 30) / 30)  # Variable volatility
                    
                    price_change = base_price * volatility * ((days_from_start % 7) - 3) / 3
                    current_price = Decimal(str(base_price + price_change))
                    
                    # Create market data
                    market_data = TaiwanMarketData(
                        symbol=symbol,
                        data_date=current_date,
                        as_of_date=current_date,
                        open_price=current_price - Decimal("1"),
                        high_price=current_price + Decimal("2"),
                        low_price=current_price - Decimal("2"),
                        close_price=current_price,
                        volume=1000000 + (days_from_start * 10000),
                        turnover=current_price * Decimal(str(1000000)),
                        trading_status=TradingStatus.NORMAL
                    )
                    
                    # Validate data quality
                    prev_close = current_price + Decimal("0.5")  # Simulate previous close
                    validation_issues = market_validator.validate_price_data(market_data, prev_close)
                    total_violations += len(validation_issues)
                    
                    # Store data
                    for temporal_value in market_data.to_temporal_values():
                        temporal_store.store(temporal_value)
                        total_data_points += 1
            
            current_date += timedelta(days=1)
        
        # Add some corporate actions
        corporate_actions = [
            {
                "symbol": "2330",
                "type": CorporateActionType.DIVIDEND_CASH,
                "announcement": date(2024, 1, 18),
                "ex_date": date(2024, 4, 18),  # Future ex-date
                "amount": Decimal("2.75")
            },
            {
                "symbol": "0050", 
                "type": CorporateActionType.DIVIDEND_CASH,
                "announcement": date(2024, 2, 15),
                "ex_date": date(2024, 5, 15),
                "amount": Decimal("1.50")
            }
        ]
        
        for action in corporate_actions:
            corp_action = TaiwanCorporateAction(
                symbol=action["symbol"],
                action_type=action["type"],
                announcement_date=action["announcement"],
                ex_date=action["ex_date"],
                amount=action["amount"]
            )
            temporal_store.store(corp_action.to_temporal_value())
        
        # Add quarterly fundamentals
        q4_2023_fundamentals = [
            {
                "symbol": "2330",
                "announcement": date(2024, 1, 18),
                "quarter_end": date(2023, 12, 31),
                "eps": Decimal("11.41"),
                "revenue": Decimal("625851000000")
            },
            {
                "symbol": "2317", 
                "announcement": date(2024, 2, 14),
                "quarter_end": date(2023, 12, 31),
                "eps": Decimal("2.51"),
                "revenue": Decimal("1850000000000")
            }
        ]
        
        for fund in q4_2023_fundamentals:
            fundamental = TaiwanFundamental(
                symbol=fund["symbol"],
                report_date=fund["quarter_end"],
                announcement_date=fund["announcement"],
                fiscal_year=2023,
                fiscal_quarter=4,
                eps=fund["eps"],
                revenue=fund["revenue"]
            )
            temporal_store.store(fundamental.to_temporal_value())
        
        # Run comprehensive backtesting queries
        backtest_results = []
        query_count = 0
        bias_violations_total = 0
        
        test_dates = [start_date + timedelta(days=i*7) for i in range(13)]  # Weekly queries
        
        for test_date in test_dates:
            if test_date <= end_date:
                # Comprehensive query
                query = PITQuery(
                    symbols=symbols,
                    as_of_date=test_date,
                    data_types=[DataType.PRICE, DataType.VOLUME, DataType.FUNDAMENTAL, DataType.CORPORATE_ACTION],
                    mode=QueryMode.STRICT,
                    bias_check=BiasCheckLevel.STRICT
                )
                
                result = pit_engine.execute_query(query)
                query_count += 1
                bias_violations_total += len(result.bias_violations)
                
                # Validate temporal consistency
                for symbol in result.data:
                    for data_type, temporal_value in result.data[symbol].items():
                        assert temporal_value.as_of_date <= test_date, \
                            f"Future as_of_date: {temporal_value.as_of_date} > {test_date}"
                        assert temporal_value.value_date <= test_date, \
                            f"Future value_date: {temporal_value.value_date} > {test_date}"
                
                backtest_results.append({
                    "date": test_date,
                    "symbols_found": len(result.data),
                    "data_types_found": sum(len(dt) for dt in result.data.values()),
                    "bias_violations": len(result.bias_violations)
                })
        
        # Verify comprehensive simulation results
        print(f"\nComprehensive Taiwan Market Simulation Results:")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Total data points stored: {total_data_points:,}")
        print(f"  Data validation violations: {total_violations}")
        print(f"  Queries executed: {query_count}")
        print(f"  Total bias violations: {bias_violations_total}")
        print(f"  Average symbols per query: {sum(r['symbols_found'] for r in backtest_results) / len(backtest_results):.1f}")
        
        # Comprehensive test assertions
        assert total_data_points > 1000, f"Too few data points: {total_data_points}"
        assert query_count == len(test_dates), f"Query count mismatch: {query_count} vs {len(test_dates)}"
        assert bias_violations_total == 0, f"Found {bias_violations_total} bias violations"
        
        # Verify data coverage
        final_query = PITQuery(
            symbols=symbols,
            as_of_date=end_date,
            data_types=[DataType.PRICE, DataType.FUNDAMENTAL, DataType.CORPORATE_ACTION],
            mode=QueryMode.STRICT
        )
        
        final_result = pit_engine.execute_query(final_query)
        
        # Should have price data for all symbols
        for symbol in symbols:
            assert symbol in final_result.data, f"Missing data for symbol {symbol}"
            assert DataType.PRICE in final_result.data[symbol], f"Missing price data for {symbol}"
        
        # Should have some fundamental data
        fundamental_symbols = [s for s in symbols if DataType.FUNDAMENTAL in final_result.data.get(s, {})]
        assert len(fundamental_symbols) >= 2, f"Too few symbols with fundamental data: {len(fundamental_symbols)}"
        
        # Should have some corporate actions
        corp_action_symbols = [s for s in symbols if DataType.CORPORATE_ACTION in final_result.data.get(s, {})]
        assert len(corp_action_symbols) >= 2, f"Too few symbols with corporate actions: {len(corp_action_symbols)}"


if __name__ == "__main__":
    # Run Taiwan market integration tests
    pytest.main([__file__, "-v", "--tb=short"])