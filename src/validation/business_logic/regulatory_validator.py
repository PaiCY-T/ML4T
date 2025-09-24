"""
Taiwan Regulatory Compliance Validator for ML4T.

This module validates trading activities against Taiwan securities regulations,
including T+2 settlement, position limits, price limits, and foreign ownership constraints.

Key Features:
- T+2 settlement cycle validation
- Daily price limit compliance (±10%)
- Foreign ownership limit checks
- Position size and concentration limits
- Margin requirement validation
- Trading halt impact assessment

Author: ML4T Team
Date: 2025-09-24
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from decimal import Decimal

logger = logging.getLogger(__name__)


class ComplianceSeverity(Enum):
    """Compliance issue severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceCategory(Enum):
    """Categories of compliance checks."""
    SETTLEMENT = "settlement"
    POSITION_LIMITS = "position_limits"
    PRICE_LIMITS = "price_limits"
    FOREIGN_OWNERSHIP = "foreign_ownership"
    MARGIN_REQUIREMENTS = "margin_requirements"
    TRADING_RESTRICTIONS = "trading_restrictions"
    MARKET_STRUCTURE = "market_structure"


@dataclass
class ComplianceIssue:
    """A regulatory compliance issue."""
    severity: ComplianceSeverity
    category: ComplianceCategory
    description: str
    rule_reference: str
    symbol: Optional[str] = None
    date: Optional[date] = None
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    breach_percentage: Optional[float] = None
    remediation: Optional[str] = None
    regulatory_citation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'severity': self.severity.value,
            'category': self.category.value,
            'description': self.description,
            'rule_reference': self.rule_reference,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'breach_percentage': self.breach_percentage,
            'remediation': self.remediation,
            'regulatory_citation': self.regulatory_citation
        }


@dataclass
class TaiwanRegulatoryConfig:
    """Configuration for Taiwan regulatory compliance validation."""
    
    # T+2 Settlement Rules
    enforce_t_plus_2: bool = True
    settlement_grace_days: int = 0
    validate_settlement_funds: bool = True
    
    # Position Limits (Article 43, Securities and Exchange Act)
    max_single_stock_weight: float = 0.10  # 10% max position
    max_sector_concentration: float = 0.30  # 30% max sector exposure
    foreign_ownership_limit: float = 0.50   # 50% foreign ownership limit
    
    # Price Limits
    daily_price_limit: float = 0.10  # ±10% daily price limit
    validate_price_continuity: bool = True
    
    # Volume and Liquidity
    max_daily_volume_pct: float = 0.10  # Max 10% of average daily volume
    min_market_cap_threshold: float = 1e9  # Min NT$1B market cap
    
    # Margin Requirements (Article 60)
    margin_ratio_stocks: float = 0.60  # 60% margin ratio for stocks
    margin_ratio_etfs: float = 0.90    # 90% margin ratio for ETFs
    validate_margin_compliance: bool = True
    
    # Trading Restrictions
    check_trading_halts: bool = True
    check_suspension_list: bool = True
    validate_circuit_breakers: bool = True
    
    # Market Structure
    validate_market_hours: bool = True
    enforce_tick_size_rules: bool = True
    check_lot_size_compliance: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.max_single_stock_weight <= 1:
            raise ValueError("Single stock weight must be between 0 and 1")
        if not 0 < self.foreign_ownership_limit <= 1:
            raise ValueError("Foreign ownership limit must be between 0 and 1")
        if self.daily_price_limit <= 0:
            raise ValueError("Price limit must be positive")


class RegulatoryValidator:
    """
    Taiwan Securities Regulatory Compliance Validator.
    
    Validates trading activities against Taiwan securities regulations
    including FSC rules, TWSE/TPEx regulations, and market structure requirements.
    """
    
    def __init__(
        self,
        config: TaiwanRegulatoryConfig,
        market_data_provider: Optional[Any] = None,
        position_tracker: Optional[Any] = None
    ):
        self.config = config
        self.market_data_provider = market_data_provider
        self.position_tracker = position_tracker
        
        # Load regulatory parameters
        self._load_regulatory_parameters()
        
        logger.info("RegulatoryValidator initialized with Taiwan market rules")
    
    def validate_portfolio_compliance(
        self,
        portfolio: Dict[str, float],
        portfolio_date: date,
        market_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[ComplianceIssue]:
        """
        Validate portfolio compliance with Taiwan regulations.
        
        Args:
            portfolio: Dictionary of symbol -> weight/position
            portfolio_date: Date of the portfolio
            market_data: Market data for compliance checks
            
        Returns:
            List of compliance issues
        """
        issues = []
        
        logger.info(f"Validating portfolio compliance for {len(portfolio)} positions on {portfolio_date}")
        
        # 1. Position concentration limits
        issues.extend(self._check_position_concentration(portfolio, portfolio_date))
        
        # 2. Sector concentration limits
        issues.extend(self._check_sector_concentration(portfolio, portfolio_date))
        
        # 3. Foreign ownership compliance
        if self.config.foreign_ownership_limit < 1.0:
            issues.extend(self._check_foreign_ownership_limits(portfolio, portfolio_date))
        
        # 4. Market cap and liquidity requirements
        if market_data:
            issues.extend(self._check_market_cap_requirements(portfolio, market_data, portfolio_date))
            issues.extend(self._check_liquidity_requirements(portfolio, market_data, portfolio_date))
        
        # 5. Margin requirements
        if self.config.validate_margin_compliance:
            issues.extend(self._check_margin_requirements(portfolio, portfolio_date))
        
        logger.info(f"Portfolio compliance validation completed: {len(issues)} issues found")
        return issues
    
    def validate_trade_compliance(
        self,
        symbol: str,
        quantity: float,
        price: float,
        trade_date: date,
        trade_type: str = "buy"
    ) -> List[ComplianceIssue]:
        """
        Validate individual trade compliance.
        
        Args:
            symbol: Stock symbol
            quantity: Trade quantity
            price: Trade price
            trade_date: Trade execution date
            trade_type: Type of trade (buy/sell)
            
        Returns:
            List of compliance issues
        """
        issues = []
        
        # 1. T+2 Settlement validation
        if self.config.enforce_t_plus_2:
            settlement_issues = self._validate_t_plus_2_settlement(
                symbol, quantity, trade_date
            )
            issues.extend(settlement_issues)
        
        # 2. Price limit validation
        price_issues = self._validate_price_limits(symbol, price, trade_date)
        issues.extend(price_issues)
        
        # 3. Volume limit validation
        volume_issues = self._validate_volume_limits(symbol, quantity, trade_date)
        issues.extend(volume_issues)
        
        # 4. Trading restriction checks
        if self.config.check_trading_halts:
            restriction_issues = self._check_trading_restrictions(symbol, trade_date)
            issues.extend(restriction_issues)
        
        # 5. Market structure compliance
        structure_issues = self._validate_market_structure_compliance(
            symbol, quantity, price, trade_date
        )
        issues.extend(structure_issues)
        
        return issues
    
    def validate_settlement_compliance(
        self,
        trades: List[Dict[str, Any]],
        settlement_date: date
    ) -> List[ComplianceIssue]:
        """
        Validate T+2 settlement compliance for a batch of trades.
        
        Args:
            trades: List of trade dictionaries
            settlement_date: Expected settlement date
            
        Returns:
            List of settlement compliance issues
        """
        issues = []
        
        for trade in trades:
            trade_date = trade.get('trade_date')
            symbol = trade.get('symbol')
            quantity = trade.get('quantity', 0)
            
            if not trade_date or not symbol:
                issues.append(ComplianceIssue(
                    severity=ComplianceSeverity.HIGH,
                    category=ComplianceCategory.SETTLEMENT,
                    description="Incomplete trade information for settlement validation",
                    rule_reference="SETTLEMENT_001",
                    symbol=symbol,
                    date=settlement_date,
                    remediation="Ensure all trades have complete information"
                ))
                continue
            
            # Calculate expected settlement date
            expected_settlement = self._calculate_settlement_date(trade_date)
            
            # Check settlement timing
            if settlement_date != expected_settlement:
                days_diff = (settlement_date - expected_settlement).days
                severity = ComplianceSeverity.HIGH if abs(days_diff) > 1 else ComplianceSeverity.MEDIUM
                
                issues.append(ComplianceIssue(
                    severity=severity,
                    category=ComplianceCategory.SETTLEMENT,
                    description=f"Settlement date mismatch: expected T+2 ({expected_settlement}), got {settlement_date}",
                    rule_reference="SETTLEMENT_002",
                    symbol=symbol,
                    date=trade_date,
                    current_value=days_diff,
                    limit_value=0,
                    regulatory_citation="Securities and Exchange Act Article 43",
                    remediation=f"Adjust settlement to T+2: {expected_settlement}"
                ))
        
        return issues
    
    def _load_regulatory_parameters(self):
        """Load Taiwan-specific regulatory parameters."""
        
        # Taiwan Stock Exchange parameters
        self.twse_params = {
            'price_limit_pct': 0.10,
            'tick_sizes': {
                (0, 10): 0.01,
                (10, 50): 0.05,
                (50, 100): 0.10,
                (100, 500): 0.50,
                (500, 1000): 1.00,
                (1000, float('inf')): 5.00
            },
            'lot_size': 1000,  # Standard lot size
            'trading_hours': {
                'open': '09:00',
                'close': '13:30',
                'lunch_start': '12:00',
                'lunch_end': '13:00'
            }
        }
        
        # Taipei Exchange parameters
        self.tpex_params = {
            'price_limit_pct': 0.10,
            'tick_sizes': {
                (0, 5): 0.01,
                (5, 10): 0.02,
                (10, 50): 0.05,
                (50, 100): 0.10,
                (100, 500): 0.50,
                (500, float('inf')): 1.00
            },
            'lot_size': 1000
        }
        
        # Sector classification for concentration limits
        self.sector_limits = {
            'TECHNOLOGY': 0.40,      # Higher limit for tech-heavy Taiwan market
            'FINANCIALS': 0.30,
            'INDUSTRIALS': 0.25,
            'CONSUMER_GOODS': 0.25,
            'MATERIALS': 0.20,
            'ENERGY': 0.15,
            'UTILITIES': 0.15,
            'HEALTHCARE': 0.20,
            'TELECOMMUNICATIONS': 0.15,
            'REAL_ESTATE': 0.15
        }
    
    def _check_position_concentration(
        self,
        portfolio: Dict[str, float],
        portfolio_date: date
    ) -> List[ComplianceIssue]:
        """Check individual position concentration limits."""
        issues = []
        
        for symbol, weight in portfolio.items():
            abs_weight = abs(weight)
            
            if abs_weight > self.config.max_single_stock_weight:
                breach_pct = (abs_weight - self.config.max_single_stock_weight) / self.config.max_single_stock_weight
                
                severity = ComplianceSeverity.CRITICAL if breach_pct > 0.5 else ComplianceSeverity.HIGH
                
                issues.append(ComplianceIssue(
                    severity=severity,
                    category=ComplianceCategory.POSITION_LIMITS,
                    description=f"Position weight {abs_weight:.2%} exceeds limit {self.config.max_single_stock_weight:.2%}",
                    rule_reference="POSITION_001",
                    symbol=symbol,
                    date=portfolio_date,
                    current_value=abs_weight,
                    limit_value=self.config.max_single_stock_weight,
                    breach_percentage=breach_pct,
                    regulatory_citation="Securities Investment Trust and Consulting Act Article 35",
                    remediation=f"Reduce position to {self.config.max_single_stock_weight:.2%} or below"
                ))
        
        return issues
    
    def _check_sector_concentration(
        self,
        portfolio: Dict[str, float],
        portfolio_date: date
    ) -> List[ComplianceIssue]:
        """Check sector concentration limits."""
        issues = []
        
        # This would typically use sector classification data
        # For demonstration, implement basic sector checking
        sector_exposures = self._calculate_sector_exposures(portfolio)
        
        for sector, exposure in sector_exposures.items():
            sector_limit = self.sector_limits.get(sector, self.config.max_sector_concentration)
            
            if exposure > sector_limit:
                breach_pct = (exposure - sector_limit) / sector_limit
                
                severity = ComplianceSeverity.HIGH if breach_pct > 0.3 else ComplianceSeverity.MEDIUM
                
                issues.append(ComplianceIssue(
                    severity=severity,
                    category=ComplianceCategory.POSITION_LIMITS,
                    description=f"Sector {sector} exposure {exposure:.2%} exceeds limit {sector_limit:.2%}",
                    rule_reference="SECTOR_001",
                    date=portfolio_date,
                    current_value=exposure,
                    limit_value=sector_limit,
                    breach_percentage=breach_pct,
                    regulatory_citation="Securities Investment Trust and Consulting Act Article 35",
                    remediation=f"Reduce {sector} exposure to {sector_limit:.2%} or below"
                ))
        
        return issues
    
    def _check_foreign_ownership_limits(
        self,
        portfolio: Dict[str, float],
        portfolio_date: date
    ) -> List[ComplianceIssue]:
        """Check foreign ownership compliance."""
        issues = []
        
        # This would query actual foreign ownership data
        for symbol, weight in portfolio.items():
            # Simulate foreign ownership check
            if self._is_foreign_ownership_constrained(symbol):
                current_foreign_ownership = self._get_foreign_ownership_pct(symbol, portfolio_date)
                
                if current_foreign_ownership and current_foreign_ownership > self.config.foreign_ownership_limit:
                    issues.append(ComplianceIssue(
                        severity=ComplianceSeverity.HIGH,
                        category=ComplianceCategory.FOREIGN_OWNERSHIP,
                        description=f"Foreign ownership {current_foreign_ownership:.2%} exceeds limit {self.config.foreign_ownership_limit:.2%}",
                        rule_reference="FOREIGN_001",
                        symbol=symbol,
                        date=portfolio_date,
                        current_value=current_foreign_ownership,
                        limit_value=self.config.foreign_ownership_limit,
                        regulatory_citation="Securities and Exchange Act Article 43-1",
                        remediation="Monitor foreign ownership limits before trading"
                    ))
        
        return issues
    
    def _check_market_cap_requirements(
        self,
        portfolio: Dict[str, float],
        market_data: Dict[str, Dict[str, Any]],
        portfolio_date: date
    ) -> List[ComplianceIssue]:
        """Check market capitalization requirements."""
        issues = []
        
        for symbol, weight in portfolio.items():
            if abs(weight) < 0.001:  # Skip tiny positions
                continue
                
            symbol_data = market_data.get(symbol, {})
            market_cap = symbol_data.get('market_cap')
            
            if market_cap and market_cap < self.config.min_market_cap_threshold:
                issues.append(ComplianceIssue(
                    severity=ComplianceSeverity.MEDIUM,
                    category=ComplianceCategory.MARKET_STRUCTURE,
                    description=f"Market cap NT${market_cap:,.0f} below threshold NT${self.config.min_market_cap_threshold:,.0f}",
                    rule_reference="MARKET_CAP_001",
                    symbol=symbol,
                    date=portfolio_date,
                    current_value=market_cap,
                    limit_value=self.config.min_market_cap_threshold,
                    remediation="Consider liquidity and market impact for small cap stocks"
                ))
        
        return issues
    
    def _check_liquidity_requirements(
        self,
        portfolio: Dict[str, float],
        market_data: Dict[str, Dict[str, Any]],
        portfolio_date: date
    ) -> List[ComplianceIssue]:
        """Check liquidity requirements."""
        issues = []
        
        for symbol, weight in portfolio.items():
            if abs(weight) < 0.001:  # Skip tiny positions
                continue
                
            symbol_data = market_data.get(symbol, {})
            avg_daily_volume = symbol_data.get('avg_daily_volume')
            
            if avg_daily_volume:
                # Estimate position size impact
                estimated_shares = abs(weight) * 1000000  # Assume $1M portfolio
                volume_pct = estimated_shares / avg_daily_volume
                
                if volume_pct > self.config.max_daily_volume_pct:
                    issues.append(ComplianceIssue(
                        severity=ComplianceSeverity.MEDIUM,
                        category=ComplianceCategory.MARKET_STRUCTURE,
                        description=f"Position represents {volume_pct:.2%} of daily volume, exceeds {self.config.max_daily_volume_pct:.2%}",
                        rule_reference="LIQUIDITY_001",
                        symbol=symbol,
                        date=portfolio_date,
                        current_value=volume_pct,
                        limit_value=self.config.max_daily_volume_pct,
                        remediation="Consider market impact and execution timing"
                    ))
        
        return issues
    
    def _check_margin_requirements(
        self,
        portfolio: Dict[str, float],
        portfolio_date: date
    ) -> List[ComplianceIssue]:
        """Check margin requirement compliance."""
        issues = []
        
        # This would integrate with actual margin calculations
        for symbol, weight in portfolio.items():
            if weight > 0:  # Long positions may require margin
                margin_ratio = self._get_margin_ratio(symbol)
                
                if weight > margin_ratio:
                    issues.append(ComplianceIssue(
                        severity=ComplianceSeverity.MEDIUM,
                        category=ComplianceCategory.MARGIN_REQUIREMENTS,
                        description=f"Position weight {weight:.2%} exceeds margin ratio {margin_ratio:.2%}",
                        rule_reference="MARGIN_001",
                        symbol=symbol,
                        date=portfolio_date,
                        current_value=weight,
                        limit_value=margin_ratio,
                        regulatory_citation="Securities and Exchange Act Article 60",
                        remediation="Ensure sufficient margin coverage"
                    ))
        
        return issues
    
    def _validate_t_plus_2_settlement(
        self,
        symbol: str,
        quantity: float,
        trade_date: date
    ) -> List[ComplianceIssue]:
        """Validate T+2 settlement timing."""
        issues = []
        
        settlement_date = self._calculate_settlement_date(trade_date)
        
        # Check if settlement date is reasonable
        if (settlement_date - trade_date).days > 2 + self.config.settlement_grace_days:
            issues.append(ComplianceIssue(
                severity=ComplianceSeverity.MEDIUM,
                category=ComplianceCategory.SETTLEMENT,
                description=f"Settlement date {settlement_date} exceeds T+2+{self.config.settlement_grace_days}",
                rule_reference="SETTLEMENT_003",
                symbol=symbol,
                date=trade_date,
                remediation="Verify trading calendar and settlement procedures"
            ))
        
        return issues
    
    def _validate_price_limits(
        self,
        symbol: str,
        price: float,
        trade_date: date
    ) -> List[ComplianceIssue]:
        """Validate price limit compliance."""
        issues = []
        
        # This would check actual price limits vs previous close
        # For demonstration, implement basic structure
        
        return issues
    
    def _validate_volume_limits(
        self,
        symbol: str,
        quantity: float,
        trade_date: date
    ) -> List[ComplianceIssue]:
        """Validate volume limits."""
        issues = []
        
        # Implementation would check against average daily volume
        return issues
    
    def _check_trading_restrictions(
        self,
        symbol: str,
        trade_date: date
    ) -> List[ComplianceIssue]:
        """Check for trading halts and restrictions."""
        issues = []
        
        # This would check actual trading halt/suspension data
        return issues
    
    def _validate_market_structure_compliance(
        self,
        symbol: str,
        quantity: float,
        price: float,
        trade_date: date
    ) -> List[ComplianceIssue]:
        """Validate market structure compliance."""
        issues = []
        
        # Check tick size compliance
        if self.config.enforce_tick_size_rules:
            tick_issues = self._check_tick_size_compliance(symbol, price)
            issues.extend(tick_issues)
        
        # Check lot size compliance
        if self.config.check_lot_size_compliance:
            lot_issues = self._check_lot_size_compliance(symbol, quantity)
            issues.extend(lot_issues)
        
        return issues
    
    def _calculate_settlement_date(self, trade_date: date) -> date:
        """Calculate T+2 settlement date accounting for holidays."""
        settlement_date = trade_date + timedelta(days=2)
        
        # This would use actual Taiwan trading calendar
        # For now, skip weekends
        while settlement_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            settlement_date += timedelta(days=1)
        
        return settlement_date
    
    def _calculate_sector_exposures(self, portfolio: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector exposures from portfolio."""
        # This would use actual sector classification data
        # For demonstration, return mock data
        return {
            'TECHNOLOGY': sum(w for s, w in portfolio.items() if s.startswith('23')),  # Tech stocks often start with 23xx
            'FINANCIALS': sum(w for s, w in portfolio.items() if s.startswith('28')),   # Financial stocks often start with 28xx
        }
    
    def _is_foreign_ownership_constrained(self, symbol: str) -> bool:
        """Check if symbol has foreign ownership constraints."""
        # This would check actual regulatory database
        return True  # Assume all stocks have constraints for demo
    
    def _get_foreign_ownership_pct(self, symbol: str, date: date) -> Optional[float]:
        """Get current foreign ownership percentage."""
        # This would query actual foreign ownership data
        return 0.40  # Demo value
    
    def _get_margin_ratio(self, symbol: str) -> float:
        """Get margin ratio for symbol."""
        # This would check if symbol is stock, ETF, etc.
        return self.config.margin_ratio_stocks
    
    def _check_tick_size_compliance(self, symbol: str, price: float) -> List[ComplianceIssue]:
        """Check tick size compliance."""
        issues = []
        
        # Get appropriate tick size table
        tick_sizes = self.twse_params['tick_sizes']  # Assume TWSE for demo
        
        tick_size = None
        for (min_price, max_price), tick in tick_sizes.items():
            if min_price <= price < max_price:
                tick_size = tick
                break
        
        if tick_size and (price % tick_size) != 0:
            issues.append(ComplianceIssue(
                severity=ComplianceSeverity.LOW,
                category=ComplianceCategory.MARKET_STRUCTURE,
                description=f"Price {price} not aligned with tick size {tick_size}",
                rule_reference="TICK_001",
                symbol=symbol,
                current_value=price,
                limit_value=tick_size,
                remediation=f"Round price to nearest tick size: {tick_size}"
            ))
        
        return issues
    
    def _check_lot_size_compliance(self, symbol: str, quantity: float) -> List[ComplianceIssue]:
        """Check lot size compliance."""
        issues = []
        
        lot_size = self.twse_params['lot_size']
        
        if quantity % lot_size != 0:
            issues.append(ComplianceIssue(
                severity=ComplianceSeverity.LOW,
                category=ComplianceCategory.MARKET_STRUCTURE,
                description=f"Quantity {quantity} not in lot size multiples of {lot_size}",
                rule_reference="LOT_001",
                symbol=symbol,
                current_value=quantity,
                limit_value=lot_size,
                remediation=f"Adjust quantity to lot size multiples: {lot_size}"
            ))
        
        return issues


def create_standard_regulatory_validator(**config_overrides) -> RegulatoryValidator:
    """Create regulatory validator with standard Taiwan market configuration."""
    config = TaiwanRegulatoryConfig(**config_overrides)
    return RegulatoryValidator(config)


def create_strict_regulatory_validator(**config_overrides) -> RegulatoryValidator:
    """Create regulatory validator with strict compliance requirements."""
    strict_config = TaiwanRegulatoryConfig(
        max_single_stock_weight=0.08,  # Stricter than standard 10%
        max_sector_concentration=0.25,  # Stricter than standard 30%
        max_daily_volume_pct=0.05,     # Stricter volume limits
        validate_margin_compliance=True,
        check_trading_halts=True,
        validate_price_continuity=True,
        **config_overrides
    )
    return RegulatoryValidator(strict_config)


# Example usage
if __name__ == "__main__":
    print("Taiwan Regulatory Validator demo")
    
    # Create validator
    validator = create_standard_regulatory_validator()
    
    # Demo portfolio
    demo_portfolio = {
        '2330.TW': 0.08,  # TSMC
        '2317.TW': 0.05,  # Hon Hai
        '2454.TW': 0.04,  # MediaTek
    }
    
    # Validate portfolio
    issues = validator.validate_portfolio_compliance(
        demo_portfolio,
        date.today()
    )
    
    print(f"Found {len(issues)} compliance issues:")
    for issue in issues:
        print(f"- {issue.severity.value}: {issue.description}")