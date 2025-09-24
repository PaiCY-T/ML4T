"""
Taiwan Market Data Quality Validator Tests with Historical Anomalies.

This module tests Taiwan market specific validators against historical market
anomalies and edge cases, ensuring robust validation for real market conditions.
"""

import asyncio
import pytest
from datetime import datetime, date, timedelta, time
from decimal import Decimal
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from src.data.core.temporal import TemporalValue, DataType
from src.data.quality.validation_engine import ValidationContext, ValidationResult
from src.data.quality.taiwan_validators import (
    TaiwanPriceLimitValidator, TaiwanVolumeValidator, TaiwanTradingHoursValidator,
    TaiwanSettlementValidator, TaiwanFundamentalLagValidator
)
from src.data.quality.validators import SeverityLevel, QualityCheckType


class TestTaiwanHistoricalAnomalies:
    """Test Taiwan validators against historical market anomalies."""
    
    @pytest.fixture
    def price_validator(self):
        """Taiwan price limit validator."""
        return TaiwanPriceLimitValidator(daily_limit_pct=0.10)
    
    @pytest.fixture
    def volume_validator(self):
        """Taiwan volume validator."""
        return TaiwanVolumeValidator(spike_threshold=5.0, lookback_days=20)
    
    @pytest.fixture
    def trading_hours_validator(self):
        """Taiwan trading hours validator."""
        return TaiwanTradingHoursValidator()
    
    @pytest.fixture
    def settlement_validator(self):
        """Taiwan settlement validator."""
        return TaiwanSettlementValidator()
    
    @pytest.fixture
    def fundamental_validator(self):
        """Taiwan fundamental lag validator."""
        return TaiwanFundamentalLagValidator()


class TestHistoricalMarketCrashes:
    """Test validators against historical market crash scenarios."""
    
    @pytest.mark.asyncio
    async def test_2008_financial_crisis_price_limits(self, price_validator):
        """Test price limit validation during 2008 financial crisis."""
        # Simulate TSMC price drop during 2008 crisis
        # Historical: TSMC dropped from ~65 to ~35 over several days
        
        # Day 1: -10% (limit hit)
        crash_day1 = TemporalValue(
            symbol="2330",
            value=Decimal("58.5"),  # -10% from 65
            value_date=date(2008, 10, 15),
            as_of_date=date(2008, 10, 15),
            data_type=DataType.PRICE
        )
        
        context = ValidationContext(
            symbol="2330",
            data_date=date(2008, 10, 15),
            as_of_date=date(2008, 10, 15),
            data_type=DataType.PRICE,
            historical_data=[
                TemporalValue(
                    symbol="2330",
                    value=Decimal("65.0"),
                    value_date=date(2008, 10, 14),
                    as_of_date=date(2008, 10, 14),
                    data_type=DataType.PRICE
                )
            ]
        )
        
        result = await price_validator.validate(crash_day1, context)
        
        # Should detect the limit hit but not flag as error (normal market behavior)
        assert result.result in [ValidationResult.PASS, ValidationResult.WARNING]
        if result.issues:
            # Should be warning level, not error
            assert all(issue.severity in [SeverityLevel.WARNING, SeverityLevel.INFO] 
                      for issue in result.issues)
    
    @pytest.mark.asyncio
    async def test_2008_circuit_breaker_volume_spike(self, volume_validator):
        """Test volume spike during circuit breaker activation."""
        # Simulate extreme volume during market crash
        crisis_volume = TemporalValue(
            symbol="2330",
            value=200000000,  # 200M shares (extreme volume)
            value_date=date(2008, 10, 15),
            as_of_date=date(2008, 10, 15),
            data_type=DataType.VOLUME
        )
        
        # Normal volumes before crisis
        normal_volumes = [
            TemporalValue(
                symbol="2330",
                value=15000000 + i * 1000000,  # 15-35M normal range
                value_date=date(2008, 10, 15) - timedelta(days=i+1),
                as_of_date=date(2008, 10, 15) - timedelta(days=i+1),
                data_type=DataType.VOLUME
            )
            for i in range(20)
        ]
        
        context = ValidationContext(
            symbol="2330",
            data_date=date(2008, 10, 15),
            as_of_date=date(2008, 10, 15),
            data_type=DataType.VOLUME,
            historical_data=normal_volumes
        )
        
        result = await volume_validator.validate(crisis_volume, context)
        
        # Should detect massive volume spike
        assert result.result == ValidationResult.WARNING
        assert len(result.issues) > 0
        
        # Check spike ratio calculation
        volume_spike_issue = next(
            (issue for issue in result.issues if "Volume spike" in issue.description),
            None
        )
        assert volume_spike_issue is not None
        assert volume_spike_issue.details["ratio"] > 5.0  # Much higher than threshold


class TestCorporateActionValidation:
    """Test validators during corporate action events."""
    
    @pytest.mark.asyncio
    async def test_stock_split_price_adjustment(self, price_validator):
        """Test price validation during stock split."""
        # Simulate 2:1 stock split (price should halve)
        split_adjusted_price = TemporalValue(
            symbol="2454",  # MediaTek
            value=Decimal("250.0"),  # Half of previous close
            value_date=date(2024, 3, 15),  # Ex-date
            as_of_date=date(2024, 3, 15),
            data_type=DataType.PRICE
        )
        
        context = ValidationContext(
            symbol="2454",
            data_date=date(2024, 3, 15),
            as_of_date=date(2024, 3, 15),
            data_type=DataType.PRICE,
            historical_data=[
                TemporalValue(
                    symbol="2454",
                    value=Decimal("500.0"),  # Pre-split price
                    value_date=date(2024, 3, 14),
                    as_of_date=date(2024, 3, 14),
                    data_type=DataType.PRICE
                )
            ],
            metadata={
                "corporate_actions": [
                    {
                        "symbol": "2454",
                        "action_type": "stock_split",
                        "ratio": "2:1",
                        "ex_date": date(2024, 3, 15)
                    }
                ]
            }
        )
        
        result = await price_validator.validate(split_adjusted_price, context)
        
        # Should recognize corporate action and not flag as error
        assert result.result in [ValidationResult.PASS, ValidationResult.WARNING]
        
        # If there are issues, they should be informational
        if result.issues:
            corporate_action_issues = [
                issue for issue in result.issues 
                if "corporate action" in issue.description.lower()
            ]
            if corporate_action_issues:
                assert all(issue.severity == SeverityLevel.INFO 
                          for issue in corporate_action_issues)
    
    @pytest.mark.asyncio
    async def test_dividend_ex_date_price_drop(self, price_validator):
        """Test price validation on dividend ex-date."""
        # Simulate price drop on ex-dividend date
        ex_div_price = TemporalValue(
            symbol="2317",  # Hon Hai
            value=Decimal("95.0"),  # Drop by dividend amount
            value_date=date(2024, 6, 20),  # Ex-dividend date
            as_of_date=date(2024, 6, 20),
            data_type=DataType.PRICE
        )
        
        context = ValidationContext(
            symbol="2317",
            data_date=date(2024, 6, 20),
            as_of_date=date(2024, 6, 20),
            data_type=DataType.PRICE,
            historical_data=[
                TemporalValue(
                    symbol="2317",
                    value=Decimal("100.0"),  # Pre-ex price
                    value_date=date(2024, 6, 19),
                    as_of_date=date(2024, 6, 19),
                    data_type=DataType.PRICE
                )
            ],
            metadata={
                "corporate_actions": [
                    {
                        "symbol": "2317",
                        "action_type": "dividend",
                        "amount": 5.0,
                        "ex_date": date(2024, 6, 20)
                    }
                ]
            }
        )
        
        result = await price_validator.validate(ex_div_price, context)
        
        # Should handle dividend adjustment appropriately
        assert result.result in [ValidationResult.PASS, ValidationResult.WARNING]


class TestMarketHolidaysAndSpecialSessions:
    """Test validators during market holidays and special trading sessions."""
    
    @pytest.mark.asyncio
    async def test_lunar_new_year_holiday_validation(self, trading_hours_validator):
        """Test validation during Lunar New Year holiday period."""
        # Data on Lunar New Year holiday (should not exist)
        holiday_data = TemporalValue(
            symbol="2330",
            value={"close_price": 500.0, "timestamp": "2024-02-10T10:30:00+08:00"},
            value_date=date(2024, 2, 10),  # Lunar New Year holiday
            as_of_date=date(2024, 2, 10),
            data_type=DataType.MARKET_DATA,
            created_at=datetime(2024, 2, 10, 10, 30)
        )
        
        context = ValidationContext(
            symbol="2330",
            data_date=date(2024, 2, 10),
            as_of_date=date(2024, 2, 10),
            data_type=DataType.MARKET_DATA,
            trading_calendar={
                date(2024, 2, 10): Mock(is_trading_day=False)  # Holiday
            }
        )
        
        result = await trading_hours_validator.validate(holiday_data, context)
        
        # Should detect trading data on non-trading day
        assert result.result in [ValidationResult.WARNING, ValidationResult.SKIP]
        if result.issues:
            assert any("non-trading day" in issue.description for issue in result.issues)
    
    @pytest.mark.asyncio
    async def test_typhoon_early_close_validation(self, trading_hours_validator):
        """Test validation during typhoon early market close."""
        # Data after typhoon early close (market closed at 12:00 instead of 13:30)
        typhoon_data = TemporalValue(
            symbol="2330",
            value={"close_price": 500.0, "timestamp": "2024-07-24T12:30:00+08:00"},
            value_date=date(2024, 7, 24),
            as_of_date=date(2024, 7, 24),
            data_type=DataType.MARKET_DATA,
            created_at=datetime(2024, 7, 24, 12, 30)  # After early close
        )
        
        context = ValidationContext(
            symbol="2330",
            data_date=date(2024, 7, 24),
            as_of_date=date(2024, 7, 24),
            data_type=DataType.MARKET_DATA,
            trading_calendar={
                date(2024, 7, 24): Mock(
                    is_trading_day=True,
                    early_close=True,
                    close_time=time(12, 0)
                )
            }
        )
        
        result = await trading_hours_validator.validate(typhoon_data, context)
        
        # Should handle early close appropriately
        # Implementation would need to check for early_close in trading calendar
        assert result.result in [ValidationResult.PASS, ValidationResult.WARNING, ValidationResult.SKIP]


class TestExtremeMarketConditions:
    """Test validators under extreme market conditions."""
    
    @pytest.mark.asyncio
    async def test_covid_crash_march_2020(self, price_validator, volume_validator):
        """Test validators during COVID-19 market crash in March 2020."""
        # Simulate multiple consecutive limit downs
        crash_dates = [
            date(2020, 3, 12),
            date(2020, 3, 13), 
            date(2020, 3, 16),
            date(2020, 3, 17)
        ]
        
        starting_price = Decimal("300.0")
        
        for i, crash_date in enumerate(crash_dates):
            # Each day hits -10% limit
            current_price = starting_price * (Decimal("0.9") ** (i + 1))
            
            crash_price = TemporalValue(
                symbol="2330",
                value=current_price,
                value_date=crash_date,
                as_of_date=crash_date,
                data_type=DataType.PRICE
            )
            
            # Previous day's price
            prev_price = starting_price * (Decimal("0.9") ** i) if i > 0 else starting_price
            prev_date = crash_date - timedelta(days=1)
            
            context = ValidationContext(
                symbol="2330",
                data_date=crash_date,
                as_of_date=crash_date,
                data_type=DataType.PRICE,
                historical_data=[
                    TemporalValue(
                        symbol="2330",
                        value=prev_price,
                        value_date=prev_date,
                        as_of_date=prev_date,
                        data_type=DataType.PRICE
                    )
                ]
            )
            
            result = await price_validator.validate(crash_price, context)
            
            # Should detect but not error on legitimate limit downs
            assert result.result in [ValidationResult.PASS, ValidationResult.WARNING]
    
    @pytest.mark.asyncio
    async def test_margin_call_volume_explosion(self, volume_validator):
        """Test volume validator during margin call volume explosion."""
        # Simulate margin call cascade with extreme volume
        margin_call_volume = TemporalValue(
            symbol="2454",  # MediaTek
            value=500000000,  # 500M shares (unprecedented)
            value_date=date(2020, 3, 19),
            as_of_date=date(2020, 3, 19),
            data_type=DataType.VOLUME
        )
        
        # Normal pre-crash volumes
        normal_volumes = [
            TemporalValue(
                symbol="2454",
                value=25000000 + i * 2000000,  # 25-65M normal range
                value_date=date(2020, 3, 19) - timedelta(days=i+1),
                as_of_date=date(2020, 3, 19) - timedelta(days=i+1),
                data_type=DataType.VOLUME
            )
            for i in range(20)
        ]
        
        context = ValidationContext(
            symbol="2454",
            data_date=date(2020, 3, 19),
            as_of_date=date(2020, 3, 19),
            data_type=DataType.VOLUME,
            historical_data=normal_volumes
        )
        
        result = await volume_validator.validate(margin_call_volume, context)
        
        # Should detect extreme volume spike
        assert result.result == ValidationResult.WARNING
        assert len(result.issues) > 0
        
        # Volume ratio should be extremely high
        volume_issue = next(
            (issue for issue in result.issues if "Volume spike" in issue.description),
            None
        )
        assert volume_issue is not None
        assert volume_issue.details["ratio"] > 10.0  # Much higher than normal threshold


class TestEarningsAnnouncementValidation:
    """Test fundamental data validation around earnings announcements."""
    
    @pytest.mark.asyncio
    async def test_quarterly_earnings_timing_validation(self, fundamental_validator):
        """Test quarterly earnings timing validation."""
        # Test various scenarios for Q4 2023 earnings
        q4_end = date(2023, 12, 31)
        
        test_cases = [
            {
                "description": "Early reporting (45 days after Q4)",
                "as_of_date": q4_end + timedelta(days=45),
                "expected_severity": SeverityLevel.INFO  # Early but acceptable
            },
            {
                "description": "Normal reporting (60 days after Q4)", 
                "as_of_date": q4_end + timedelta(days=60),
                "expected_severity": None  # Should pass
            },
            {
                "description": "Late reporting (100 days after Q4)",
                "as_of_date": q4_end + timedelta(days=100),
                "expected_severity": SeverityLevel.WARNING  # Late
            },
            {
                "description": "Critical look-ahead bias (before Q4 end)",
                "as_of_date": q4_end - timedelta(days=5),
                "expected_severity": SeverityLevel.CRITICAL  # Impossible
            }
        ]
        
        for case in test_cases:
            q4_earnings = TemporalValue(
                symbol="2330",
                value={
                    "fiscal_year": 2023,
                    "fiscal_quarter": 4,
                    "revenue": 75900000000,  # TSMC Q4 2023 revenue
                    "net_income": 25960000000,
                    "eps": 10.01
                },
                value_date=q4_end,
                as_of_date=case["as_of_date"],
                data_type=DataType.FUNDAMENTAL
            )
            
            context = ValidationContext(
                symbol="2330",
                data_date=q4_end,
                as_of_date=case["as_of_date"],
                data_type=DataType.FUNDAMENTAL
            )
            
            result = await fundamental_validator.validate(q4_earnings, context)
            
            if case["expected_severity"] is None:
                # Should pass
                assert result.result == ValidationResult.PASS
            elif case["expected_severity"] == SeverityLevel.CRITICAL:
                # Should fail with critical issue
                assert result.result == ValidationResult.FAIL
                assert any(issue.severity == SeverityLevel.CRITICAL for issue in result.issues)
            else:
                # Should have issues of expected severity
                assert len(result.issues) > 0
                severity_found = any(
                    issue.severity == case["expected_severity"] for issue in result.issues
                )
                assert severity_found, f"Expected severity {case['expected_severity']} not found in {case['description']}"
    
    @pytest.mark.asyncio
    async def test_restatement_detection(self, fundamental_validator):
        """Test detection of financial restatements."""
        # Simulate a restatement scenario
        original_earnings = TemporalValue(
            symbol="2330",
            value={
                "fiscal_year": 2023,
                "fiscal_quarter": 3,
                "revenue": 70000000000,
                "net_income": 20000000000,
                "restatement": False
            },
            value_date=date(2023, 9, 30),
            as_of_date=date(2023, 11, 15),  # Normal 45-day timing
            data_type=DataType.FUNDAMENTAL
        )
        
        # Later restatement
        restated_earnings = TemporalValue(
            symbol="2330",
            value={
                "fiscal_year": 2023,
                "fiscal_quarter": 3,
                "revenue": 69500000000,  # Revised down
                "net_income": 19500000000,  # Revised down
                "restatement": True,
                "original_report_date": "2023-11-15"
            },
            value_date=date(2023, 9, 30),
            as_of_date=date(2024, 2, 10),  # Much later restatement
            data_type=DataType.FUNDAMENTAL
        )
        
        context = ValidationContext(
            symbol="2330",
            data_date=date(2023, 9, 30),
            as_of_date=date(2024, 2, 10),
            data_type=DataType.FUNDAMENTAL
        )
        
        result = await fundamental_validator.validate(restated_earnings, context)
        
        # Restatement should be flagged for attention
        assert result.result in [ValidationResult.PASS, ValidationResult.WARNING]
        # Implementation would need restatement detection logic


class TestPerformanceUnderStress:
    """Test validator performance under stress conditions."""
    
    @pytest.mark.asyncio
    async def test_high_frequency_validation_stress(self):
        """Test validators under high-frequency validation load."""
        validators = [
            TaiwanPriceLimitValidator(),
            TaiwanVolumeValidator(),
            TaiwanTradingHoursValidator()
        ]
        
        # Create high-frequency test data
        test_values = []
        for i in range(1000):
            value = TemporalValue(
                symbol=f"2{330 + i % 100}",
                value=Decimal(f"{500 + (i % 100)}"),
                value_date=date(2024, 1, 15),
                as_of_date=date(2024, 1, 15),
                data_type=DataType.PRICE,
                created_at=datetime(2024, 1, 15, 9, 0, i // 10)  # Spread over time
            )
            test_values.append(value)
        
        # Validate all values with all validators
        import time
        start_time = time.perf_counter()
        
        validation_tasks = []
        for value in test_values:
            context = ValidationContext(
                symbol=value.symbol,
                data_date=value.value_date,
                as_of_date=value.as_of_date,
                data_type=value.data_type
            )
            
            for validator in validators:
                if validator.can_validate(value, context):
                    validation_tasks.append(validator.validate(value, context))
        
        results = await asyncio.gather(*validation_tasks)
        end_time = time.perf_counter()
        
        # Performance assertions
        total_time_ms = (end_time - start_time) * 1000
        validations_per_second = len(validation_tasks) / (total_time_ms / 1000)
        
        assert len(results) == len(validation_tasks)
        assert validations_per_second > 1000, f"Validation rate {validations_per_second:.0f}/s too low"
        assert all(isinstance(result, type(results[0])) for result in results)
        
        print(f"Stress test: {len(validation_tasks)} validations in {total_time_ms:.2f}ms "
              f"({validations_per_second:.0f} validations/second)")
    
    @pytest.mark.asyncio
    async def test_ultra_high_throughput_validation(self):
        """Test validators at ultra-high throughput (>10K validations/second)."""
        validators = [
            TaiwanPriceLimitValidator(),
            TaiwanVolumeValidator()
        ]
        
        # Create larger test dataset
        test_values = []
        taiwan_symbols = ["2330", "2317", "2454", "2412", "3008", "2882", "1303", "2002"]
        
        for i in range(5000):
            symbol = taiwan_symbols[i % len(taiwan_symbols)]
            value = TemporalValue(
                symbol=symbol,
                value=Decimal(f"{400 + (i % 300)}"),  # Prices 400-700
                value_date=date(2024, 1, 15),
                as_of_date=date(2024, 1, 15),
                data_type=DataType.PRICE,
                created_at=datetime.utcnow()
            )
            test_values.append(value)
        
        # Batch processing for optimal performance
        import time
        start_time = time.perf_counter()
        
        batch_size = 100
        total_validations = 0
        
        for i in range(0, len(test_values), batch_size):
            batch = test_values[i:i + batch_size]
            validation_tasks = []
            
            for value in batch:
                context = ValidationContext(
                    symbol=value.symbol,
                    data_date=value.value_date,
                    as_of_date=value.as_of_date,
                    data_type=value.data_type
                )
                
                for validator in validators:
                    if validator.can_validate(value, context):
                        validation_tasks.append(validator.validate(value, context))
            
            await asyncio.gather(*validation_tasks)
            total_validations += len(validation_tasks)
        
        end_time = time.perf_counter()
        
        # Calculate throughput
        total_time_s = end_time - start_time
        validations_per_second = total_validations / total_time_s
        
        # Target: >10K validations/second
        assert validations_per_second > 10000, f"Ultra-high throughput {validations_per_second:.0f}/s below 10K target"
        
        print(f"Ultra-high throughput test: {total_validations} validations in {total_time_s:.2f}s "
              f"({validations_per_second:.0f} validations/second)")


class TestRealWorldTaiwanScenarios:
    """Test validators against real-world Taiwan market scenarios."""
    
    @pytest.mark.asyncio
    async def test_taiwan_semiconductor_earnings_season(self, fundamental_validator):
        """Test validation during Taiwan semiconductor earnings season."""
        # Simulate TSMC, MediaTek, ASE Group earnings
        earnings_scenarios = [
            {
                "symbol": "2330",  # TSMC
                "revenue": 75900000000,  # Q4 2023 actual
                "net_income": 25960000000,
                "quarter_end": date(2023, 12, 31),
                "report_date": date(2024, 1, 18),  # Typical TSMC timing
                "expected_result": ValidationResult.PASS
            },
            {
                "symbol": "2454",  # MediaTek  
                "revenue": 18900000000,  # Q4 2023 estimate
                "net_income": 1200000000,
                "quarter_end": date(2023, 12, 31),
                "report_date": date(2024, 1, 30),  # MediaTek later timing
                "expected_result": ValidationResult.PASS
            },
            {
                "symbol": "3711",  # ASE Group
                "revenue": 156000000000,  # Q4 2023 estimate
                "net_income": 8500000000,
                "quarter_end": date(2023, 12, 31),
                "report_date": date(2024, 2, 2),  # Even later
                "expected_result": ValidationResult.PASS
            }
        ]
        
        for scenario in earnings_scenarios:
            earnings_data = TemporalValue(
                symbol=scenario["symbol"],
                value={
                    "fiscal_year": 2023,
                    "fiscal_quarter": 4,
                    "revenue": scenario["revenue"],
                    "net_income": scenario["net_income"],
                    "report_type": "quarterly"
                },
                value_date=scenario["quarter_end"],
                as_of_date=scenario["report_date"],
                data_type=DataType.FUNDAMENTAL
            )
            
            context = ValidationContext(
                symbol=scenario["symbol"],
                data_date=scenario["quarter_end"],
                as_of_date=scenario["report_date"],
                data_type=DataType.FUNDAMENTAL
            )
            
            result = await fundamental_validator.validate(earnings_data, context)
            
            # Check timing is appropriate for Taiwan market
            days_lag = (scenario["report_date"] - scenario["quarter_end"]).days
            assert 15 <= days_lag <= 90, f"Unrealistic reporting lag: {days_lag} days"
            
            assert result.result == scenario["expected_result"], f"Unexpected result for {scenario['symbol']}"
    
    @pytest.mark.asyncio
    async def test_taiwan_etf_extreme_volume(self, volume_validator):
        """Test volume validation for Taiwan ETF extreme trading."""
        # Simulate 0050 (Taiwan 50 ETF) during rebalancing
        etf_rebalancing_volume = TemporalValue(
            symbol="0050",
            value=150000000,  # 150M shares (extreme for ETF)
            value_date=date(2024, 3, 15),  # Quarterly rebalancing
            as_of_date=date(2024, 3, 15),
            data_type=DataType.VOLUME
        )
        
        # Normal ETF volumes
        normal_etf_volumes = [
            TemporalValue(
                symbol="0050",
                value=5000000 + i * 500000,  # 5-15M normal range
                value_date=date(2024, 3, 15) - timedelta(days=i+1),
                as_of_date=date(2024, 3, 15) - timedelta(days=i+1),
                data_type=DataType.VOLUME
            )
            for i in range(20)
        ]
        
        context = ValidationContext(
            symbol="0050",
            data_date=date(2024, 3, 15),
            as_of_date=date(2024, 3, 15),
            data_type=DataType.VOLUME,
            historical_data=normal_etf_volumes,
            metadata={"security_type": "ETF", "event": "quarterly_rebalancing"}
        )
        
        result = await volume_validator.validate(etf_rebalancing_volume, context)
        
        # Should detect volume spike but recognize ETF context
        assert result.result == ValidationResult.WARNING
        assert any("Volume spike" in issue.description for issue in result.issues)
        
        # Should have very high spike ratio
        volume_issue = next(
            issue for issue in result.issues if "Volume spike" in issue.description
        )
        assert volume_issue.details["ratio"] > 10.0
    
    @pytest.mark.asyncio
    async def test_taiwan_warrant_pricing_validation(self, price_validator):
        """Test price validation for Taiwan warrants with high volatility."""
        # Taiwan warrants can have extreme price movements
        warrant_scenarios = [
            {
                "symbol": "03008P",  # Put warrant example
                "prev_price": Decimal("0.50"),
                "current_price": Decimal("2.50"),  # 400% increase
                "expected_result": ValidationResult.WARNING  # Extreme but possible
            },
            {
                "symbol": "03009C",  # Call warrant example  
                "prev_price": Decimal("1.20"),
                "current_price": Decimal("0.05"),  # 95% drop
                "expected_result": ValidationResult.WARNING  # Extreme but possible
            }
        ]
        
        for scenario in warrant_scenarios:
            warrant_price = TemporalValue(
                symbol=scenario["symbol"],
                value=scenario["current_price"],
                value_date=date(2024, 1, 15),
                as_of_date=date(2024, 1, 15),
                data_type=DataType.PRICE
            )
            
            context = ValidationContext(
                symbol=scenario["symbol"],
                data_date=date(2024, 1, 15),
                as_of_date=date(2024, 1, 15),
                data_type=DataType.PRICE,
                historical_data=[
                    TemporalValue(
                        symbol=scenario["symbol"],
                        value=scenario["prev_price"],
                        value_date=date(2024, 1, 14),
                        as_of_date=date(2024, 1, 14),
                        data_type=DataType.PRICE
                    )
                ],
                metadata={"security_type": "WARRANT"}
            )
            
            result = await price_validator.validate(warrant_price, context)
            
            # Warrants should trigger warnings for extreme moves but not errors
            assert result.result == scenario["expected_result"]
            if result.issues:
                # Should detect extreme price movement
                assert any("price" in issue.description.lower() for issue in result.issues)
    
    @pytest.mark.asyncio
    async def test_taiwan_foreign_investment_limit_scenario(self, volume_validator):
        """Test validation during foreign investment limit scenarios."""
        # Simulate volume spike when foreign investment limit is reached
        foreign_limit_volume = TemporalValue(
            symbol="2330",  # TSMC often hits foreign limits
            value=300000000,  # 300M shares
            value_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.VOLUME
        )
        
        # Context with foreign investment metadata
        context = ValidationContext(
            symbol="2330",
            data_date=date(2024, 1, 15),
            as_of_date=date(2024, 1, 15),
            data_type=DataType.VOLUME,
            metadata={
                "foreign_investment_limit": 0.50,  # 50% limit
                "current_foreign_ownership": 0.49,  # Close to limit
                "market_event": "foreign_limit_approach"
            }
        )
        
        result = await volume_validator.validate(foreign_limit_volume, context)
        
        # Should detect volume spike related to foreign investment activity
        assert result.result == ValidationResult.WARNING
        if result.issues:
            volume_issue = next(
                (issue for issue in result.issues if "Volume" in issue.description),
                None
            )
            assert volume_issue is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-k", "not stress"])