"""
Taiwan Market Compliance Validator for Feature Selection.

Validates that selected features comply with Taiwan Securities Exchange (TSE)
regulations, T+2 settlement constraints, and local market microstructure.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
import re
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance level enumeration."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


@dataclass
class ComplianceRule:
    """Represents a single compliance rule."""
    rule_id: str
    name: str
    description: str
    severity: ComplianceLevel
    validation_function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ComplianceResult:
    """Result of compliance validation for a feature."""
    feature_name: str
    rule_id: str
    status: ComplianceLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaiwanMarketConfig:
    """Configuration for Taiwan market compliance validation."""
    
    # Settlement and trading constraints
    settlement_cycle: int = 2  # T+2 settlement
    trading_hours_start: str = "09:00"
    trading_hours_end: str = "13:30"
    max_daily_price_limit: float = 0.10  # 10% daily price limit
    
    # Market holidays (simplified - in practice would be comprehensive)
    market_holidays: List[str] = field(default_factory=lambda: [
        "2024-01-01",  # New Year
        "2024-02-10",  # Lunar New Year
        "2024-04-04",  # Children's Day
        "2024-05-01",  # Labor Day
        "2024-10-10",  # National Day
    ])
    
    # Sector classifications (TSE sector codes)
    tse_sectors: List[str] = field(default_factory=lambda: [
        "01",  # Cement
        "02",  # Food
        "03",  # Plastic
        "04",  # Textile
        "05",  # Electric Machinery
        "06",  # Electrical and Cable
        "08",  # Glass and Ceramics
        "09",  # Paper and Pulp
        "10",  # Iron and Steel
        "11",  # Rubber
        "12",  # Automobile
        "14",  # Building Materials
        "15",  # Shipping
        "16",  # Tourism
        "17",  # Financial
        "18",  # Trading
        "21",  # Chemical
        "22",  # Biotechnology
        "23",  # Oil, Gas and Electricity
        "24",  # Semiconductor
        "25",  # Computer
        "26",  # Optoelectronic
        "27",  # Communication
        "28",  # Electronic Components
        "29",  # Electronic Channel
        "30",  # Information Service
        "31",  # Other Electronic
        "50",  # ETF
    ])
    
    # Regulatory constraints
    foreign_ownership_limit: float = 0.30  # 30% foreign ownership limit for some stocks
    margin_trading_requirement: float = 0.50  # 50% margin requirement
    tick_size_rules: Dict[float, float] = field(default_factory=lambda: {
        10.0: 0.01,    # Stocks < $10 TWD: 1 cent tick
        50.0: 0.05,    # Stocks $10-50 TWD: 5 cent tick  
        100.0: 0.10,   # Stocks $50-100 TWD: 10 cent tick
        500.0: 0.50,   # Stocks $100-500 TWD: 50 cent tick
        1000.0: 1.00,  # Stocks $500-1000 TWD: $1 tick
        float('inf'): 5.00  # Stocks >$1000 TWD: $5 tick
    })
    
    # Validation thresholds
    min_trading_volume: float = 1000.0  # Minimum daily volume
    min_market_cap: float = 1e9  # Minimum market cap (TWD)
    max_beta: float = 3.0  # Maximum beta threshold
    
    # Feature naming conventions
    prohibited_keywords: List[str] = field(default_factory=lambda: [
        "insider",
        "manipulation", 
        "front_run",
        "illegal",
        "unauthorized"
    ])
    
    required_prefixes: Dict[str, List[str]] = field(default_factory=lambda: {
        "price": ["price_", "close_", "open_", "high_", "low_"],
        "volume": ["volume_", "turnover_", "shares_"],
        "fundamental": ["eps_", "roe_", "pe_", "pb_", "debt_"],
        "technical": ["rsi_", "macd_", "bb_", "ma_", "momentum_"]
    })


class TaiwanMarketComplianceValidator:
    """
    Validates features for compliance with Taiwan Securities Exchange regulations.
    
    Key validation areas:
    1. Settlement cycle constraints (T+2)
    2. Trading hour restrictions
    3. Price limit considerations
    4. Market microstructure compliance
    5. Regulatory naming conventions
    6. Data availability and quality
    """
    
    def __init__(self, config: Optional[TaiwanMarketConfig] = None):
        """Initialize Taiwan market compliance validator.
        
        Args:
            config: Taiwan market configuration
        """
        self.config = config or TaiwanMarketConfig()
        self.compliance_rules = self._initialize_compliance_rules()
        self.validation_results: List[ComplianceResult] = []
        
        logger.info("Taiwan Market Compliance Validator initialized")
        logger.info(f"Loaded {len(self.compliance_rules)} compliance rules")
    
    def _initialize_compliance_rules(self) -> List[ComplianceRule]:
        """Initialize comprehensive compliance rules."""
        
        rules = [
            ComplianceRule(
                rule_id="TSE-001",
                name="Settlement Cycle Compliance",
                description="Features must respect T+2 settlement cycle",
                severity=ComplianceLevel.CRITICAL,
                validation_function="_validate_settlement_cycle",
                parameters={"settlement_days": self.config.settlement_cycle}
            ),
            ComplianceRule(
                rule_id="TSE-002", 
                name="Trading Hours Compliance",
                description="Features must align with TSE trading hours (09:00-13:30)",
                severity=ComplianceLevel.WARNING,
                validation_function="_validate_trading_hours",
                parameters={
                    "start_time": self.config.trading_hours_start,
                    "end_time": self.config.trading_hours_end
                }
            ),
            ComplianceRule(
                rule_id="TSE-003",
                name="Price Limit Compliance", 
                description="Features must account for 10% daily price limits",
                severity=ComplianceLevel.WARNING,
                validation_function="_validate_price_limits",
                parameters={"max_limit": self.config.max_daily_price_limit}
            ),
            ComplianceRule(
                rule_id="TSE-004",
                name="Look-ahead Bias Prevention",
                description="Features must not contain future information",
                severity=ComplianceLevel.CRITICAL,
                validation_function="_validate_no_lookahead",
                parameters={}
            ),
            ComplianceRule(
                rule_id="TSE-005",
                name="Market Holiday Compliance",
                description="Features must handle market holidays correctly",
                severity=ComplianceLevel.WARNING,
                validation_function="_validate_market_holidays",
                parameters={"holidays": self.config.market_holidays}
            ),
            ComplianceRule(
                rule_id="TSE-006",
                name="Sector Classification Compliance",
                description="Sector-based features must use valid TSE sector codes",
                severity=ComplianceLevel.WARNING,
                validation_function="_validate_sector_codes", 
                parameters={"valid_sectors": self.config.tse_sectors}
            ),
            ComplianceRule(
                rule_id="TSE-007",
                name="Volume Threshold Compliance",
                description="Volume-based features must meet minimum trading volume",
                severity=ComplianceLevel.WARNING,
                validation_function="_validate_volume_threshold",
                parameters={"min_volume": self.config.min_trading_volume}
            ),
            ComplianceRule(
                rule_id="TSE-008",
                name="Market Cap Compliance", 
                description="Features should focus on stocks meeting minimum market cap",
                severity=ComplianceLevel.WARNING,
                validation_function="_validate_market_cap",
                parameters={"min_market_cap": self.config.min_market_cap}
            ),
            ComplianceRule(
                rule_id="TSE-009",
                name="Feature Naming Compliance",
                description="Feature names must follow TSE-compliant naming conventions",
                severity=ComplianceLevel.WARNING,
                validation_function="_validate_feature_naming",
                parameters={
                    "prohibited": self.config.prohibited_keywords,
                    "required_prefixes": self.config.required_prefixes
                }
            ),
            ComplianceRule(
                rule_id="TSE-010",
                name="Beta Threshold Compliance",
                description="Beta-based features should be within reasonable bounds",
                severity=ComplianceLevel.WARNING,
                validation_function="_validate_beta_threshold",
                parameters={"max_beta": self.config.max_beta}
            )
        ]
        
        return rules
    
    def validate_features(
        self,
        features: List[str],
        feature_data: Optional[pd.DataFrame] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[ComplianceResult]]:
        """
        Validate features for Taiwan market compliance.
        
        Args:
            features: List of feature names to validate
            feature_data: Optional feature data for value-based validation
            market_data: Optional market data for context validation
            feature_metadata: Optional metadata about features
            
        Returns:
            Dictionary mapping feature names to compliance results
        """
        logger.info(f"Validating {len(features)} features for Taiwan market compliance")
        
        validation_results = {}
        self.validation_results = []
        
        for feature in features:
            feature_results = []
            
            for rule in self.compliance_rules:
                if not rule.enabled:
                    continue
                    
                try:
                    # Execute validation function
                    validation_func = getattr(self, rule.validation_function)
                    result = validation_func(
                        feature, 
                        feature_data,
                        market_data,
                        feature_metadata,
                        rule.parameters
                    )
                    
                    # Create compliance result
                    compliance_result = ComplianceResult(
                        feature_name=feature,
                        rule_id=rule.rule_id,
                        status=result['status'],
                        message=result['message'],
                        details=result.get('details', {})
                    )
                    
                    feature_results.append(compliance_result)
                    self.validation_results.append(compliance_result)
                    
                except Exception as e:
                    logger.error(f"Error validating feature {feature} with rule {rule.rule_id}: {e}")
                    
                    error_result = ComplianceResult(
                        feature_name=feature,
                        rule_id=rule.rule_id,
                        status=ComplianceLevel.CRITICAL,
                        message=f"Validation error: {str(e)}",
                        details={"error": str(e)}
                    )
                    
                    feature_results.append(error_result)
                    self.validation_results.append(error_result)
            
            validation_results[feature] = feature_results
        
        # Log summary
        self._log_validation_summary(validation_results)
        
        return validation_results
    
    def _validate_settlement_cycle(
        self, 
        feature: str, 
        feature_data: Optional[pd.DataFrame],
        market_data: Optional[pd.DataFrame],
        feature_metadata: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate settlement cycle compliance (T+2)."""
        
        # Check if feature name suggests forward-looking calculation
        forward_indicators = ['_forward', '_future', '_next', '_lead', '_ahead']
        has_forward_indicator = any(indicator in feature.lower() for indicator in forward_indicators)
        
        # Check for proper lag indicators
        lag_indicators = ['_lag', '_prev', '_past', '_delayed', '_t-']
        has_lag_indicator = any(indicator in feature.lower() for indicator in lag_indicators)
        
        # Extract lag period if specified in feature name
        lag_match = re.search(r'(?:lag|delay|t-)(\d+)', feature.lower())
        specified_lag = int(lag_match.group(1)) if lag_match else 0
        
        if has_forward_indicator:
            return {
                'status': ComplianceLevel.CRITICAL,
                'message': f"Feature {feature} appears to use forward-looking data, violating T+{params['settlement_days']} settlement",
                'details': {'settlement_days': params['settlement_days'], 'violation_type': 'forward_looking'}
            }
        
        if specified_lag > 0 and specified_lag < params['settlement_days']:
            return {
                'status': ComplianceLevel.WARNING,
                'message': f"Feature {feature} uses {specified_lag}-day lag, less than T+{params['settlement_days']} settlement cycle",
                'details': {'specified_lag': specified_lag, 'required_lag': params['settlement_days']}
            }
        
        # If feature data available, check for data alignment
        if feature_data is not None and feature in feature_data.columns:
            # Check if feature values are properly lagged
            # This is a simplified check - in practice would be more sophisticated
            feature_series = feature_data[feature]
            if feature_series.isna().sum() / len(feature_series) > 0.5:
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Feature {feature} has high missing data rate, may indicate settlement alignment issues",
                    'details': {'missing_rate': feature_series.isna().sum() / len(feature_series)}
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} appears to comply with T+{params['settlement_days']} settlement cycle",
            'details': {'has_lag_indicator': has_lag_indicator}
        }
    
    def _validate_trading_hours(
        self,
        feature: str,
        feature_data: Optional[pd.DataFrame],
        market_data: Optional[pd.DataFrame], 
        feature_metadata: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate trading hours compliance."""
        
        # Check if feature is time-sensitive
        time_sensitive_keywords = ['intraday', 'hourly', 'minute', 'real_time', 'live', 'tick']
        is_time_sensitive = any(keyword in feature.lower() for keyword in time_sensitive_keywords)
        
        if is_time_sensitive:
            # Check if feature name includes trading hour constraints
            trading_hour_keywords = ['9_to_13', 'trading_hours', 'market_hours']
            has_hour_constraint = any(keyword in feature.lower() for keyword in trading_hour_keywords)
            
            if not has_hour_constraint:
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Time-sensitive feature {feature} should specify trading hour constraints ({params['start_time']}-{params['end_time']})",
                    'details': {
                        'is_time_sensitive': True,
                        'trading_hours': f"{params['start_time']}-{params['end_time']}"
                    }
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} complies with trading hours requirements",
            'details': {'is_time_sensitive': is_time_sensitive}
        }
    
    def _validate_price_limits(
        self,
        feature: str,
        feature_data: Optional[pd.DataFrame],
        market_data: Optional[pd.DataFrame],
        feature_metadata: Optional[Dict[str, Any]], 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate price limit compliance."""
        
        # Check if feature is price-related
        price_keywords = ['price', 'return', 'momentum', 'volatility', 'change']
        is_price_related = any(keyword in feature.lower() for keyword in price_keywords)
        
        if is_price_related:
            # Check if feature accounts for price limits
            limit_keywords = ['limit_adj', 'capped', 'bounded', 'limit_aware']
            has_limit_adjustment = any(keyword in feature.lower() for keyword in limit_keywords)
            
            # If feature data available, check for limit hit patterns
            if feature_data is not None and feature in feature_data.columns:
                feature_series = feature_data[feature]
                
                # Check for values that might indicate price limit hits
                if is_price_related and 'return' in feature.lower():
                    extreme_values = (abs(feature_series) >= params['max_limit'] * 0.95).sum()
                    if extreme_values > len(feature_series) * 0.01:  # More than 1% extreme values
                        return {
                            'status': ComplianceLevel.WARNING,
                            'message': f"Feature {feature} has many values near ±{params['max_limit']*100}% price limits",
                            'details': {
                                'extreme_value_count': extreme_values,
                                'extreme_value_rate': extreme_values / len(feature_series),
                                'price_limit': params['max_limit']
                            }
                        }
            
            if not has_limit_adjustment:
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Price-related feature {feature} should consider ±{params['max_limit']*100}% daily price limits",
                    'details': {'is_price_related': True, 'max_limit': params['max_limit']}
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} complies with price limit requirements",
            'details': {'is_price_related': is_price_related}
        }
    
    def _validate_no_lookahead(
        self,
        feature: str,
        feature_data: Optional[pd.DataFrame],
        market_data: Optional[pd.DataFrame],
        feature_metadata: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate no look-ahead bias."""
        
        # Suspicious keywords that might indicate look-ahead bias
        lookahead_keywords = [
            'future', 'forward', 'next', 'lead', 'ahead', 'tomorrow',
            'target', 'label', 'actual', 'realized', 'final'
        ]
        
        has_lookahead_keyword = any(keyword in feature.lower() for keyword in lookahead_keywords)
        
        if has_lookahead_keyword:
            # Check for legitimate forward-looking features (e.g., lagged targets)
            legitimate_patterns = ['_target_lag', '_forward_return_lag', '_next_period_lag']
            is_legitimate = any(pattern in feature.lower() for pattern in legitimate_patterns)
            
            if not is_legitimate:
                return {
                    'status': ComplianceLevel.CRITICAL,
                    'message': f"Feature {feature} may contain look-ahead bias with keyword: {[k for k in lookahead_keywords if k in feature.lower()]}",
                    'details': {'suspicious_keywords': [k for k in lookahead_keywords if k in feature.lower()]}
                }
        
        # Check feature metadata for calculation method
        if feature_metadata and feature in feature_metadata:
            calc_method = feature_metadata[feature].get('calculation_method', '')
            if 'forward' in calc_method.lower() or 'future' in calc_method.lower():
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Feature {feature} metadata indicates forward-looking calculation",
                    'details': {'calculation_method': calc_method}
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} appears free of look-ahead bias",
            'details': {'has_lookahead_keyword': has_lookahead_keyword}
        }
    
    def _validate_market_holidays(
        self,
        feature: str,
        feature_data: Optional[pd.DataFrame],
        market_data: Optional[pd.DataFrame],
        feature_metadata: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate market holiday handling."""
        
        # Check if feature should be sensitive to market holidays
        holiday_sensitive_keywords = ['daily', 'trading_day', 'business_day', 'market_day']
        is_holiday_sensitive = any(keyword in feature.lower() for keyword in holiday_sensitive_keywords)
        
        if is_holiday_sensitive:
            # Check if feature name indicates holiday adjustment
            holiday_adjustment_keywords = ['holiday_adj', 'business_day', 'trading_day', 'market_day']
            has_holiday_adjustment = any(keyword in feature.lower() for keyword in holiday_adjustment_keywords)
            
            if not has_holiday_adjustment:
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Holiday-sensitive feature {feature} should specify market holiday handling",
                    'details': {
                        'is_holiday_sensitive': True,
                        'holiday_count': len(params['holidays'])
                    }
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} appropriately handles market holidays",
            'details': {'is_holiday_sensitive': is_holiday_sensitive}
        }
    
    def _validate_sector_codes(
        self,
        feature: str,
        feature_data: Optional[pd.DataFrame],
        market_data: Optional[pd.DataFrame],
        feature_metadata: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate TSE sector code compliance."""
        
        # Check if feature is sector-related
        sector_keywords = ['sector', 'industry', 'gics', 'classification']
        is_sector_related = any(keyword in feature.lower() for keyword in sector_keywords)
        
        if is_sector_related:
            # Check for TSE sector code references
            tse_pattern = r'tse[_-]?(\d{2})'
            tse_match = re.search(tse_pattern, feature.lower())
            
            if tse_match:
                sector_code = tse_match.group(1)
                if sector_code not in params['valid_sectors']:
                    return {
                        'status': ComplianceLevel.WARNING,
                        'message': f"Feature {feature} references invalid TSE sector code: {sector_code}",
                        'details': {
                            'invalid_sector_code': sector_code,
                            'valid_sectors': params['valid_sectors']
                        }
                    }
            elif is_sector_related:
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Sector-related feature {feature} should use TSE sector codes",
                    'details': {'valid_sectors': params['valid_sectors'][:5]}  # Show first 5
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} complies with sector classification requirements",
            'details': {'is_sector_related': is_sector_related}
        }
    
    def _validate_volume_threshold(
        self,
        feature: str,
        feature_data: Optional[pd.DataFrame], 
        market_data: Optional[pd.DataFrame],
        feature_metadata: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate minimum volume threshold compliance."""
        
        # Check if feature is volume-related
        volume_keywords = ['volume', 'turnover', 'shares', 'liquidity', 'trade_count']
        is_volume_related = any(keyword in feature.lower() for keyword in volume_keywords)
        
        if is_volume_related:
            # Check if feature data is available for validation
            if feature_data is not None and feature in feature_data.columns:
                feature_series = feature_data[feature]
                
                # For volume-based features, check if values suggest minimum volume filtering
                if 'volume' in feature.lower():
                    min_volume_in_data = feature_series.min() if not feature_series.empty else 0
                    if min_volume_in_data < params['min_volume']:
                        return {
                            'status': ComplianceLevel.WARNING,
                            'message': f"Volume feature {feature} includes values below minimum threshold {params['min_volume']}",
                            'details': {
                                'min_volume_in_data': min_volume_in_data,
                                'min_threshold': params['min_volume']
                            }
                        }
            
            # Check if feature name indicates volume filtering
            volume_filter_keywords = ['min_vol', 'liquid', 'active', 'filtered']
            has_volume_filter = any(keyword in feature.lower() for keyword in volume_filter_keywords)
            
            if not has_volume_filter:
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Volume-related feature {feature} should consider minimum volume threshold {params['min_volume']}",
                    'details': {'min_volume': params['min_volume']}
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} complies with volume requirements", 
            'details': {'is_volume_related': is_volume_related}
        }
    
    def _validate_market_cap(
        self,
        feature: str,
        feature_data: Optional[pd.DataFrame],
        market_data: Optional[pd.DataFrame],
        feature_metadata: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate market capitalization compliance."""
        
        # Check if feature should consider market cap constraints
        market_cap_keywords = ['market_cap', 'cap_weighted', 'size', 'large_cap', 'mid_cap', 'small_cap']
        is_cap_sensitive = any(keyword in feature.lower() for keyword in market_cap_keywords)
        
        # Features that typically benefit from market cap filtering
        filter_beneficial_keywords = ['fundamental', 'eps', 'roe', 'revenue', 'pe', 'pb']
        would_benefit_from_filtering = any(keyword in feature.lower() for keyword in filter_beneficial_keywords)
        
        if is_cap_sensitive or would_benefit_from_filtering:
            # Check if feature indicates market cap filtering
            cap_filter_keywords = ['large_cap', 'min_cap', 'cap_filter', 'size_filter']
            has_cap_filter = any(keyword in feature.lower() for keyword in cap_filter_keywords)
            
            if not has_cap_filter and would_benefit_from_filtering:
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Feature {feature} would benefit from minimum market cap filtering ({params['min_market_cap']:,.0f} TWD)",
                    'details': {'min_market_cap': params['min_market_cap']}
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} appropriately considers market cap constraints",
            'details': {
                'is_cap_sensitive': is_cap_sensitive,
                'would_benefit_from_filtering': would_benefit_from_filtering
            }
        }
    
    def _validate_feature_naming(
        self,
        feature: str,
        feature_data: Optional[pd.DataFrame],
        market_data: Optional[pd.DataFrame],
        feature_metadata: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate feature naming conventions."""
        
        # Check for prohibited keywords
        prohibited_found = [word for word in params['prohibited'] if word in feature.lower()]
        if prohibited_found:
            return {
                'status': ComplianceLevel.CRITICAL,
                'message': f"Feature {feature} contains prohibited keywords: {prohibited_found}",
                'details': {'prohibited_keywords_found': prohibited_found}
            }
        
        # Check for required prefixes based on feature category
        has_valid_prefix = False
        detected_category = None
        
        for category, prefixes in params['required_prefixes'].items():
            if any(prefix in feature.lower() for prefix in prefixes):
                has_valid_prefix = True
                detected_category = category
                break
        
        # For features without clear prefixes, provide guidance
        guidance_keywords = {
            'price': ['close', 'open', 'high', 'low', 'price'],
            'volume': ['volume', 'turnover', 'shares'],
            'fundamental': ['eps', 'roe', 'pe', 'pb', 'debt', 'revenue'],
            'technical': ['rsi', 'macd', 'ma', 'momentum', 'volatility']
        }
        
        if not has_valid_prefix:
            suggested_category = None
            for category, keywords in guidance_keywords.items():
                if any(keyword in feature.lower() for keyword in keywords):
                    suggested_category = category
                    break
            
            if suggested_category:
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Feature {feature} should use standard prefix for {suggested_category} features",
                    'details': {
                        'suggested_category': suggested_category,
                        'suggested_prefixes': params['required_prefixes'][suggested_category]
                    }
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} follows naming conventions",
            'details': {
                'has_valid_prefix': has_valid_prefix,
                'detected_category': detected_category
            }
        }
    
    def _validate_beta_threshold(
        self,
        feature: str,
        feature_data: Optional[pd.DataFrame],
        market_data: Optional[pd.DataFrame],
        feature_metadata: Optional[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate beta threshold compliance."""
        
        # Check if feature is beta-related
        beta_keywords = ['beta', 'market_sensitivity', 'systematic_risk']
        is_beta_related = any(keyword in feature.lower() for keyword in beta_keywords)
        
        if is_beta_related:
            # Check if feature data is available
            if feature_data is not None and feature in feature_data.columns:
                feature_series = feature_data[feature]
                
                # Check for extreme beta values
                if 'beta' in feature.lower():
                    max_beta = feature_series.max() if not feature_series.empty else 0
                    min_beta = feature_series.min() if not feature_series.empty else 0
                    
                    if abs(max_beta) > params['max_beta'] or abs(min_beta) > params['max_beta']:
                        return {
                            'status': ComplianceLevel.WARNING,
                            'message': f"Beta feature {feature} has extreme values (max: {max_beta:.2f}, min: {min_beta:.2f})",
                            'details': {
                                'max_beta_observed': max_beta,
                                'min_beta_observed': min_beta,
                                'max_beta_threshold': params['max_beta']
                            }
                        }
            
            # Check if feature name indicates beta capping
            capping_keywords = ['capped', 'bounded', 'winsorized', 'trimmed']
            has_capping = any(keyword in feature.lower() for keyword in capping_keywords)
            
            if not has_capping:
                return {
                    'status': ComplianceLevel.WARNING,
                    'message': f"Beta feature {feature} should consider capping extreme values (±{params['max_beta']})",
                    'details': {'max_beta_threshold': params['max_beta']}
                }
        
        return {
            'status': ComplianceLevel.COMPLIANT,
            'message': f"Feature {feature} complies with beta requirements",
            'details': {'is_beta_related': is_beta_related}
        }
    
    def _log_validation_summary(self, validation_results: Dict[str, List[ComplianceResult]]) -> None:
        """Log validation summary statistics."""
        
        total_features = len(validation_results)
        total_validations = sum(len(results) for results in validation_results.values())
        
        # Count by status
        status_counts = {status: 0 for status in ComplianceLevel}
        for results in validation_results.values():
            for result in results:
                status_counts[result.status] += 1
        
        # Calculate compliance rate
        compliant_count = status_counts[ComplianceLevel.COMPLIANT] + status_counts[ComplianceLevel.WARNING]
        compliance_rate = compliant_count / total_validations if total_validations > 0 else 0
        
        logger.info(f"Taiwan Market Compliance Validation Summary:")
        logger.info(f"  Total Features: {total_features}")
        logger.info(f"  Total Validations: {total_validations}")
        logger.info(f"  Compliance Rate: {compliance_rate:.1%}")
        logger.info(f"  Status Distribution:")
        for status, count in status_counts.items():
            logger.info(f"    {status.value.title()}: {count} ({count/total_validations:.1%})")
        
        # Log critical violations
        critical_violations = [
            result for results in validation_results.values() 
            for result in results 
            if result.status == ComplianceLevel.CRITICAL
        ]
        
        if critical_violations:
            logger.warning(f"Found {len(critical_violations)} critical compliance violations:")
            for violation in critical_violations[:5]:  # Log first 5
                logger.warning(f"  {violation.feature_name}: {violation.message}")
    
    def get_compliant_features(
        self,
        validation_results: Dict[str, List[ComplianceResult]],
        min_compliance_level: ComplianceLevel = ComplianceLevel.WARNING
    ) -> List[str]:
        """Get list of features meeting minimum compliance level."""
        
        compliant_features = []
        
        for feature, results in validation_results.items():
            # Check if feature has any violations above the threshold
            has_violation = any(
                result.status in [ComplianceLevel.CRITICAL, ComplianceLevel.VIOLATION]
                for result in results
            )
            
            # For WARNING threshold, only exclude CRITICAL and VIOLATION
            if min_compliance_level == ComplianceLevel.WARNING:
                if not has_violation:
                    compliant_features.append(feature)
            
            # For COMPLIANT threshold, exclude WARNING and above
            elif min_compliance_level == ComplianceLevel.COMPLIANT:
                has_warning_or_above = any(
                    result.status in [ComplianceLevel.WARNING, ComplianceLevel.VIOLATION, ComplianceLevel.CRITICAL]
                    for result in results
                )
                if not has_warning_or_above:
                    compliant_features.append(feature)
        
        logger.info(f"Found {len(compliant_features)} features meeting {min_compliance_level.value} compliance level")
        
        return compliant_features
    
    def generate_compliance_report(
        self,
        validation_results: Dict[str, List[ComplianceResult]],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate comprehensive compliance report."""
        
        report_data = []
        
        for feature, results in validation_results.items():
            for result in results:
                report_data.append({
                    'Feature': result.feature_name,
                    'Rule_ID': result.rule_id,
                    'Status': result.status.value,
                    'Message': result.message,
                    'Details': str(result.details),
                    'Timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        report_df = pd.DataFrame(report_data)
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            logger.info(f"Compliance report saved to {output_path}")
        
        return report_df
    
    def get_compliance_score(
        self,
        validation_results: Dict[str, List[ComplianceResult]]
    ) -> Dict[str, float]:
        """Calculate compliance scores for features."""
        
        feature_scores = {}
        
        # Scoring weights
        score_weights = {
            ComplianceLevel.COMPLIANT: 1.0,
            ComplianceLevel.WARNING: 0.7,
            ComplianceLevel.VIOLATION: 0.3,
            ComplianceLevel.CRITICAL: 0.0
        }
        
        for feature, results in validation_results.items():
            if not results:
                feature_scores[feature] = 0.0
                continue
            
            # Calculate weighted average score
            total_score = sum(score_weights[result.status] for result in results)
            feature_scores[feature] = total_score / len(results)
        
        return feature_scores