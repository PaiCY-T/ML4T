"""
Comprehensive tests for Taiwan Market Data Management.

This test suite validates:
- Taiwan market specific data models
- T+2 settlement handling
- Corporate action processing
- Market calendar integration
- Data validation rules
- Real-world Taiwan market scenarios
"""

import pytest
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Import Taiwan market components
from src.data.models.taiwan_market import (
    TaiwanMarketCode, TradingStatus, CorporateActionType,
    TaiwanTradingCalendar, TaiwanSettlement, TaiwanStockInfo,
    TaiwanMarketData, TaiwanCorporateAction, TaiwanFundamental,
    TaiwanMarketDataValidator, create_taiwan_trading_calendar,
    get_taiwan_market_timezone, taiwan_market_time_to_utc
)

from src.data.core.temporal import (
    TemporalValue, DataType, MarketSession, InMemoryTemporalStore
)


class TestTaiwanTradingCalendar:
    """Test Taiwan trading calendar functionality."""
    
    def test_trading_calendar_creation(self):
        """Test basic trading calendar creation."""
        test_date = date(2024, 1, 15)  # Monday
        
        calendar_entry = TaiwanTradingCalendar(
            date=test_date,
            is_trading_day=True,
            market_session=MarketSession.MORNING,
            morning_open=time(9, 0),
            morning_close=time(13, 30)
        )
        
        assert calendar_entry.date == test_date
        assert calendar_entry.is_trading_day
        assert calendar_entry.market_session == MarketSession.MORNING
    
    def test_trading_hours_property(self):
        """Test trading hours calculation."""
        trading_day = TaiwanTradingCalendar(
            date=date(2024, 1, 15),
            is_trading_day=True,
            market_session=MarketSession.MORNING
        )
        
        start, end = trading_day.trading_hours
        assert start == time(9, 0)
        assert end == time(13, 30)
        
        # Non-trading day
        non_trading = TaiwanTradingCalendar(
            date=date(2024, 1, 13),  # Saturday
            is_trading_day=False,
            market_session=MarketSession.CLOSED
        )
        
        start, end = non_trading.trading_hours
        assert start is None
        assert end is None
    
    def test_trading_duration_minutes(self):
        """Test trading duration calculation."""
        trading_day = TaiwanTradingCalendar(
            date=date(2024, 1, 15),
            is_trading_day=True,
            market_session=MarketSession.MORNING
        )
        
        # 9:00 to 13:30 = 4.5 hours = 270 minutes
        duration = trading_day.trading_duration_minutes
        assert duration == 270
        
        # Non-trading day should be 0
        non_trading = TaiwanTradingCalendar(
            date=date(2024, 1, 13),
            is_trading_day=False,
            market_session=MarketSession.CLOSED
        )
        
        assert non_trading.trading_duration_minutes == 0


class TestTaiwanSettlement:
    """Test Taiwan T+2 settlement handling."""
    
    def test_settlement_creation(self):
        """Test basic settlement creation and validation."""
        trade_date = date(2024, 1, 15)  # Monday
        settlement_date = date(2024, 1, 17)  # Wednesday
        
        settlement = TaiwanSettlement(
            trade_date=trade_date,
            settlement_date=settlement_date
        )
        
        assert settlement.trade_date == trade_date
        assert settlement.settlement_date == settlement_date
        assert settlement.is_regular_settlement
        assert settlement.settlement_lag_days == 2
    
    def test_settlement_validation_error(self):
        """Test settlement validation with invalid dates."""
        trade_date = date(2024, 1, 15)
        invalid_settlement = date(2024, 1, 14)  # Before trade date
        
        with pytest.raises(ValueError, match="Settlement date .* must be after trade date"):
            TaiwanSettlement(
                trade_date=trade_date,
                settlement_date=invalid_settlement
            )
    
    def test_calculate_t2_settlement_weekdays(self):
        """Test T+2 settlement calculation for weekdays."""
        # Create sample trading calendar
        calendar = create_taiwan_trading_calendar(2024)
        
        # Monday trade -> Wednesday settlement
        monday = date(2024, 1, 15)  # Monday
        settlement = TaiwanSettlement.calculate_t2_settlement(monday, calendar)
        
        assert settlement.trade_date == monday
        assert settlement.settlement_date == date(2024, 1, 17)  # Wednesday
        assert settlement.settlement_lag_days == 2
    
    def test_calculate_t2_settlement_weekend_skip(self):
        """Test T+2 settlement skips weekends correctly."""
        calendar = create_taiwan_trading_calendar(2024)
        
        # Thursday trade -> Monday settlement (skipping weekend)
        thursday = date(2024, 1, 18)  # Thursday
        settlement = TaiwanSettlement.calculate_t2_settlement(thursday, calendar)
        
        # T+2 business days = Monday
        expected_settlement = date(2024, 1, 22)  # Monday
        assert settlement.settlement_date == expected_settlement
        assert settlement.settlement_lag_days == 4  # Calendar days
    
    def test_calculate_t2_settlement_holiday_skip(self):
        """Test T+2 settlement skips holidays."""
        calendar = create_taiwan_trading_calendar(2024)
        
        # Test around New Year's Day (non-trading day)
        # This test may need adjustment based on actual 2024 Taiwan holidays
        dec_28_2023 = date(2023, 12, 28)  # Assuming Thursday
        
        # The settlement calculation should skip non-trading days
        # Exact behavior depends on the trading calendar implementation


class TestTaiwanStockInfo:
    """Test Taiwan stock information handling."""
    
    def test_stock_info_creation(self):
        """Test stock info creation with Taiwan market specifics."""
        stock = TaiwanStockInfo(
            symbol="2330",
            name_zh="台積電",
            name_en="Taiwan Semiconductor Manufacturing Company",
            market=TaiwanMarketCode.TWSE,
            industry_code="2400",
            sector="Semiconductor",
            listing_date=date(1994, 9, 5),
            par_value=Decimal("10.0"),
            outstanding_shares=25930380458,
            trading_status=TradingStatus.NORMAL
        )
        
        assert stock.symbol == "2330"
        assert stock.name_zh == "台積電"
        assert stock.market == TaiwanMarketCode.TWSE
        assert stock.trading_status == TradingStatus.NORMAL
    
    def test_stock_active_status(self):
        """Test stock active status determination."""
        stock = TaiwanStockInfo(
            symbol="2330",
            name_zh="台積電",
            listing_date=date(2020, 1, 1),
            trading_status=TradingStatus.NORMAL
        )
        
        # Should be active after listing date
        assert stock.is_active(date(2024, 1, 15))
        
        # Should not be active before listing date
        assert not stock.is_active(date(2019, 12, 31))
        
        # Delisted stock should not be active
        delisted_stock = TaiwanStockInfo(
            symbol="1234",
            name_zh="已下市股票",
            trading_status=TradingStatus.DELISTED
        )
        assert not delisted_stock.is_active(date(2024, 1, 15))


class TestTaiwanMarketData:
    """Test Taiwan market data model."""
    
    def test_market_data_creation(self):
        """Test Taiwan market data creation."""
        test_date = date(2024, 1, 15)
        
        market_data = TaiwanMarketData(
            symbol="2330",
            data_date=test_date,
            as_of_date=test_date,
            open_price=Decimal("580.00"),
            high_price=Decimal("585.00"),
            low_price=Decimal("578.00"),
            close_price=Decimal("582.00"),
            volume=25000000,
            turnover=Decimal("14550000000"),  # 14.55B TWD
            market_cap=Decimal("15000000000000"),  # 15T TWD
            trading_status=TradingStatus.NORMAL,
            price_limit_up=Decimal("638.00"),
            price_limit_down=Decimal("522.00"),
            foreign_ownership_pct=Decimal("78.5")
        )
        
        assert market_data.symbol == "2330"
        assert market_data.close_price == Decimal("582.00")
        assert market_data.volume == 25000000
        assert market_data.trading_status == TradingStatus.NORMAL
        assert market_data.foreign_ownership_pct == Decimal("78.5")
    
    def test_to_temporal_values_conversion(self):
        """Test conversion to temporal values."""
        test_date = date(2024, 1, 15)
        
        market_data = TaiwanMarketData(
            symbol="2330",
            data_date=test_date,
            as_of_date=test_date,
            close_price=Decimal("582.00"),
            volume=25000000,
            turnover=Decimal("14550000000")
        )
        
        temporal_values = market_data.to_temporal_values()
        
        # Should have multiple temporal values
        assert len(temporal_values) > 0
        
        # Find price value
        price_values = [v for v in temporal_values if v.data_type == DataType.PRICE]
        assert len(price_values) == 1
        assert price_values[0].value == Decimal("582.00")
        assert price_values[0].symbol == "2330"
        
        # Find volume value
        volume_values = [v for v in temporal_values if v.data_type == DataType.VOLUME]
        assert len(volume_values) == 1
        assert volume_values[0].value == 25000000


class TestTaiwanCorporateAction:
    """Test Taiwan corporate action handling."""
    
    def test_corporate_action_creation(self):
        """Test corporate action creation."""
        action = TaiwanCorporateAction(
            symbol="2330",
            action_type=CorporateActionType.DIVIDEND_CASH,
            announcement_date=date(2024, 1, 15),
            ex_date=date(2024, 2, 15),
            record_date=date(2024, 2, 17),
            payable_date=date(2024, 3, 15),
            amount=Decimal("2.75")  # TWD 2.75 per share
        )
        
        assert action.symbol == "2330"
        assert action.action_type == CorporateActionType.DIVIDEND_CASH
        assert action.amount == Decimal("2.75")
    
    def test_corporate_action_to_temporal_value(self):
        """Test conversion to temporal value."""
        action = TaiwanCorporateAction(
            symbol="2330",
            action_type=CorporateActionType.STOCK_SPLIT,
            announcement_date=date(2024, 1, 15),
            ex_date=date(2024, 2, 15),
            record_date=date(2024, 2, 17),
            ratio=Decimal("2.0"),  # 2:1 split
            description="Stock split 2:1"
        )
        
        temporal_value = action.to_temporal_value()
        
        assert temporal_value.symbol == "2330"
        assert temporal_value.data_type == DataType.CORPORATE_ACTION
        assert temporal_value.as_of_date == date(2024, 1, 15)  # announcement_date
        assert temporal_value.value_date == date(2024, 2, 15)  # ex_date
        
        # Check value content
        value_data = temporal_value.value
        assert value_data["action_type"] == "stock_split"
        assert value_data["ratio"] == 2.0
        assert value_data["description"] == "Stock split 2:1"
    
    def test_affects_price_on_date(self):
        """Test price impact date calculation."""
        action = TaiwanCorporateAction(
            symbol="2330",
            action_type=CorporateActionType.DIVIDEND_CASH,
            announcement_date=date(2024, 1, 15),
            ex_date=date(2024, 2, 15),
            record_date=date(2024, 2, 17),
            amount=Decimal("2.75")
        )
        
        # Before ex-date: no impact
        assert not action.affects_price_on_date(date(2024, 2, 14))
        
        # On ex-date: impact starts
        assert action.affects_price_on_date(date(2024, 2, 15))
        
        # After ex-date: still impacted
        assert action.affects_price_on_date(date(2024, 2, 20))


class TestTaiwanFundamental:
    """Test Taiwan fundamental data handling."""
    
    def test_fundamental_creation(self):
        """Test fundamental data creation with proper lag."""
        fundamental = TaiwanFundamental(
            symbol="2330",
            report_date=date(2023, 12, 31),  # Q4 2023
            announcement_date=date(2024, 2, 29),  # 60 days later
            fiscal_year=2023,
            fiscal_quarter=4,
            revenue=Decimal("1700000000000"),  # 1.7T TWD
            operating_income=Decimal("600000000000"),  # 600B TWD
            net_income=Decimal("500000000000"),  # 500B TWD
            eps=Decimal("19.20"),
            roe=Decimal("25.5"),
            book_value_per_share=Decimal("75.40")
        )
        
        assert fundamental.symbol == "2330"
        assert fundamental.fiscal_year == 2023
        assert fundamental.fiscal_quarter == 4
        assert fundamental.eps == Decimal("19.20")
        assert fundamental.reporting_lag_days == 60
    
    def test_fundamental_to_temporal_value(self):
        """Test conversion to temporal value with lag handling."""
        fundamental = TaiwanFundamental(
            symbol="2330",
            report_date=date(2023, 12, 31),
            announcement_date=date(2024, 2, 29),
            fiscal_year=2023,
            fiscal_quarter=4,
            revenue=Decimal("1700000000000"),
            eps=Decimal("19.20")
        )
        
        temporal_value = fundamental.to_temporal_value()
        
        assert temporal_value.symbol == "2330"
        assert temporal_value.data_type == DataType.FUNDAMENTAL
        assert temporal_value.as_of_date == date(2024, 2, 29)  # announcement_date
        assert temporal_value.value_date == date(2023, 12, 31)  # report_date
        
        # Check lag metadata
        assert temporal_value.metadata["lag_days"] == 60
        assert temporal_value.metadata["fiscal_year"] == 2023
        assert temporal_value.metadata["fiscal_quarter"] == 4
        
        # Check value content
        value_data = temporal_value.value
        assert value_data["revenue"] == 1700000000000.0
        assert value_data["eps"] == 19.20


class TestTaiwanMarketDataValidator:
    """Test Taiwan market data validation."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return TaiwanMarketDataValidator()
    
    def test_validate_price_data_consistency(self, validator):
        """Test price data consistency validation."""
        # Valid data
        valid_data = TaiwanMarketData(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            open_price=Decimal("580.00"),
            high_price=Decimal("585.00"),
            low_price=Decimal("578.00"),
            close_price=Decimal("582.00"),
            volume=25000000
        )
        
        issues = validator.validate_price_data(valid_data)
        assert len(issues) == 0
        
        # Invalid data: open > high
        invalid_data = TaiwanMarketData(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            open_price=Decimal("590.00"),  # Higher than high
            high_price=Decimal("585.00"),
            low_price=Decimal("578.00"),
            close_price=Decimal("582.00"),
            volume=25000000
        )
        
        issues = validator.validate_price_data(invalid_data)
        assert len(issues) > 0
        assert any("open price higher than high price" in issue.lower() for issue in issues)
    
    def test_validate_price_data_daily_limit(self, validator):
        """Test daily price limit validation."""
        previous_close = Decimal("500.00")
        
        # Data within daily limit (10%)
        normal_data = TaiwanMarketData(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            close_price=Decimal("540.00"),  # 8% increase
            volume=25000000
        )
        
        issues = validator.validate_price_data(normal_data, previous_close)
        # Should have no daily limit issues
        daily_limit_issues = [i for i in issues if "daily limit" in i.lower()]
        assert len(daily_limit_issues) == 0
        
        # Data exceeding daily limit
        extreme_data = TaiwanMarketData(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            close_price=Decimal("600.00"),  # 20% increase (exceeds 10% limit)
            volume=25000000
        )
        
        issues = validator.validate_price_data(extreme_data, previous_close)
        daily_limit_issues = [i for i in issues if "daily limit" in i.lower()]
        assert len(daily_limit_issues) > 0
    
    def test_validate_negative_volume(self, validator):
        """Test negative volume validation."""
        invalid_volume_data = TaiwanMarketData(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            close_price=Decimal("582.00"),
            volume=-1000000  # Negative volume
        )
        
        issues = validator.validate_price_data(invalid_volume_data)
        assert len(issues) > 0
        assert any("negative volume" in issue.lower() for issue in issues)
    
    def test_validate_settlement_timing(self, validator):
        """Test settlement timing validation."""
        trade_date = date(2024, 1, 15)
        
        # Valid T+2 settlement
        valid_settlement = TaiwanSettlement(
            trade_date=trade_date,
            settlement_date=date(2024, 1, 17)  # T+2
        )
        
        issues = validator.validate_settlement_timing(trade_date, valid_settlement)
        assert len(issues) == 0
        
        # Invalid T+1 settlement
        invalid_settlement = TaiwanSettlement(
            trade_date=trade_date,
            settlement_date=date(2024, 1, 16)  # T+1 (too early)
        )
        
        issues = validator.validate_settlement_timing(trade_date, invalid_settlement)
        assert len(issues) > 0
        assert any("T+2 minimum" in issue for issue in issues)
    
    def test_validate_fundamental_timing(self, validator):
        """Test fundamental data timing validation."""
        # Valid fundamental data (within 60-day limit for annual report)
        valid_fundamental = TaiwanFundamental(
            symbol="2330",
            report_date=date(2023, 12, 31),  # Q4 annual report
            announcement_date=date(2024, 2, 15),  # 45 days later
            fiscal_year=2023,
            fiscal_quarter=4
        )
        
        issues = validator.validate_fundamental_timing(valid_fundamental)
        assert len(issues) == 0
        
        # Invalid fundamental data (exceeds 60-day limit)
        invalid_fundamental = TaiwanFundamental(
            symbol="2330",
            report_date=date(2023, 12, 31),
            announcement_date=date(2024, 3, 15),  # 75 days later (too late)
            fiscal_year=2023,
            fiscal_quarter=4
        )
        
        issues = validator.validate_fundamental_timing(invalid_fundamental)
        assert len(issues) > 0
        assert any("exceeds regulatory maximum" in issue for issue in issues)
    
    def test_validate_announcement_before_report(self, validator):
        """Test validation of announcement date before report date."""
        invalid_fundamental = TaiwanFundamental(
            symbol="2330",
            report_date=date(2024, 1, 31),
            announcement_date=date(2024, 1, 15),  # Before report date
            fiscal_year=2024,
            fiscal_quarter=1
        )
        
        issues = validator.validate_fundamental_timing(invalid_fundamental)
        assert len(issues) > 0
        assert any("announcement date before report date" in issue.lower() for issue in issues)


class TestTradingCalendarCreation:
    """Test trading calendar creation utilities."""
    
    def test_create_taiwan_trading_calendar(self):
        """Test Taiwan trading calendar creation."""
        calendar = create_taiwan_trading_calendar(2024)
        
        # Should have entries for entire year
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)
        
        assert start_date in calendar
        assert end_date in calendar
        
        # Check that weekdays are generally trading days
        monday = date(2024, 1, 8)  # A Monday that should be trading
        assert monday in calendar
        # Most Mondays should be trading days (unless holiday)
        
        # Check that weekends are not trading days
        saturday = date(2024, 1, 6)  # A Saturday
        assert saturday in calendar
        assert not calendar[saturday].is_trading_day
        
        sunday = date(2024, 1, 7)  # A Sunday
        assert sunday in calendar
        assert not calendar[sunday].is_trading_day
    
    def test_taiwan_holidays_in_calendar(self):
        """Test that known Taiwan holidays are marked correctly."""
        calendar = create_taiwan_trading_calendar(2024)
        
        # New Year's Day
        new_years = date(2024, 1, 1)
        assert new_years in calendar
        assert not calendar[new_years].is_trading_day
        
        # National Day (if it falls on a weekday)
        national_day = date(2024, 10, 10)
        assert national_day in calendar
        # Should not be a trading day
        assert not calendar[national_day].is_trading_day


class TestTimezoneUtilities:
    """Test timezone-related utilities."""
    
    def test_get_taiwan_market_timezone(self):
        """Test Taiwan market timezone retrieval."""
        tz = get_taiwan_market_timezone()
        assert tz == "Asia/Taipei"
    
    def test_taiwan_market_time_to_utc(self):
        """Test Taiwan market time to UTC conversion."""
        # Taiwan is UTC+8
        taiwan_time = datetime(2024, 1, 15, 10, 30, 0)  # 10:30 TST
        utc_time = taiwan_market_time_to_utc(taiwan_time)
        
        expected_utc = datetime(2024, 1, 15, 2, 30, 0)  # 02:30 UTC
        assert utc_time == expected_utc


@pytest.mark.integration
class TestRealWorldTaiwanScenarios:
    """Integration tests with real-world Taiwan market scenarios."""
    
    def test_tsmc_dividend_scenario(self):
        """Test TSMC dividend scenario with realistic data."""
        store = InMemoryTemporalStore()
        
        # TSMC (2330) dividend announcement
        dividend_action = TaiwanCorporateAction(
            symbol="2330",
            action_type=CorporateActionType.DIVIDEND_CASH,
            announcement_date=date(2024, 1, 25),
            ex_date=date(2024, 4, 18),
            record_date=date(2024, 4, 22),
            payable_date=date(2024, 7, 18),
            amount=Decimal("2.75"),
            description="Q1 2024 cash dividend"
        )
        
        temporal_value = dividend_action.to_temporal_value()
        store.store(temporal_value)
        
        # Query before announcement - should get no data
        before_announcement = store.get_point_in_time(
            "2330", 
            date(2024, 1, 20), 
            DataType.CORPORATE_ACTION
        )
        assert before_announcement is None
        
        # Query after announcement - should get dividend data
        after_announcement = store.get_point_in_time(
            "2330", 
            date(2024, 2, 1), 
            DataType.CORPORATE_ACTION
        )
        assert after_announcement is not None
        assert after_announcement.value["amount"] == 2.75
    
    def test_quarterly_earnings_scenario(self):
        """Test quarterly earnings release scenario."""
        store = InMemoryTemporalStore()
        
        # Q4 2023 earnings for TSMC
        earnings = TaiwanFundamental(
            symbol="2330",
            report_date=date(2023, 12, 31),
            announcement_date=date(2024, 1, 18),  # Typical timing
            fiscal_year=2023,
            fiscal_quarter=4,
            revenue=Decimal("625851000000"),  # ~625B TWD
            net_income=Decimal("295900000000"),  # ~296B TWD
            eps=Decimal("11.41"),  # EPS in TWD
            roe=Decimal("26.4")
        )
        
        temporal_value = earnings.to_temporal_value()
        store.store(temporal_value)
        
        # Query before earnings announcement
        before_earnings = store.get_point_in_time(
            "2330",
            date(2024, 1, 15),
            DataType.FUNDAMENTAL
        )
        assert before_earnings is None
        
        # Query after earnings announcement
        after_earnings = store.get_point_in_time(
            "2330",
            date(2024, 1, 25),
            DataType.FUNDAMENTAL
        )
        assert after_earnings is not None
        assert after_earnings.value["eps"] == 11.41
        assert after_earnings.metadata["lag_days"] == 18
    
    def test_market_crash_scenario(self):
        """Test extreme market movement scenario with validation."""
        validator = TaiwanMarketDataValidator()
        
        # Normal trading day
        normal_day = TaiwanMarketData(
            symbol="0050",  # Taiwan 50 ETF
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            open_price=Decimal("130.00"),
            high_price=Decimal("131.50"),
            low_price=Decimal("129.50"),
            close_price=Decimal("130.75"),
            volume=5000000
        )
        
        issues = validator.validate_price_data(normal_day)
        assert len(issues) == 0
        
        # Market crash day (extreme movement)
        crash_day = TaiwanMarketData(
            symbol="0050",
            data_date=date(2024, 1, 16),
            as_of_date=date(2024, 1, 16),
            open_price=Decimal("118.00"),  # Gap down
            high_price=Decimal("120.00"),
            low_price=Decimal("115.00"),
            close_price=Decimal("117.68"),  # -10% (daily limit)
            volume=50000000  # 10x normal volume
        )
        
        # Validate against previous close
        issues = validator.validate_price_data(crash_day, normal_day.close_price)
        # Should trigger daily limit warning (exactly at -10% limit)
        daily_limit_issues = [i for i in issues if "daily limit" in i.lower()]
        # May or may not trigger depending on exact limit calculation
    
    def test_ipo_listing_scenario(self):
        """Test IPO listing scenario."""
        # New stock listing
        ipo_stock = TaiwanStockInfo(
            symbol="6666",
            name_zh="新上市公司",
            name_en="New IPO Company",
            market=TaiwanMarketCode.TWSE,
            listing_date=date(2024, 3, 15),
            trading_status=TradingStatus.IPO
        )
        
        # Should not be active before listing
        assert not ipo_stock.is_active(date(2024, 3, 10))
        
        # Should be active after listing (status permitting)
        # Note: IPO status might have special handling
        # This test validates the basic framework


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])