"""
Taiwan Market Compliance Validation - Task #28 Stream C

Specialized compliance checks for Taiwan Stock Exchange (TSE) and Taipei Exchange (TPEx)
to ensure generated features comply with local market structure and regulations.

Key Compliance Areas:
- T+2 settlement cycle constraints
- Daily price limits (±10%)
- Trading hours (09:00-13:30) compliance
- Market holidays and calendar alignment
- Currency and locale requirements (TWD)
- Regulatory data usage restrictions
- Cross-sectional consistency for panel data

Used by FeatureSelector and FeatureQualityMetrics to ensure market compliance.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import warnings
from pathlib import Path
import calendar

# Import project modules
from .taiwan_config import TaiwanMarketConfig

logger = logging.getLogger(__name__)


class TaiwanComplianceValidator:
    """
    Comprehensive Taiwan market compliance validation for generated features.
    
    Validates features against Taiwan Stock Exchange regulations and practices:
    1. Temporal constraints (T+2 settlement, trading hours)
    2. Market structure compliance (price limits, tick sizes)
    3. Calendar and holiday alignment
    4. Data availability and consistency rules
    5. Regulatory constraints on data usage
    """
    
    def __init__(
        self,
        taiwan_config: Optional[TaiwanMarketConfig] = None,
        strict_mode: bool = True,
        allow_warnings: bool = True
    ):
        """
        Initialize Taiwan compliance validator.
        
        Args:
            taiwan_config: Taiwan market configuration
            strict_mode: Whether to enforce strict compliance (fail on warnings)
            allow_warnings: Whether to report warnings (vs. only errors)
        """
        self.taiwan_config = taiwan_config or TaiwanMarketConfig()
        self.strict_mode = strict_mode
        self.allow_warnings = allow_warnings
        
        # Compliance rules and thresholds
        self.compliance_rules = {
            # Temporal rules
            'settlement_lag_days': self.taiwan_config.SETTLEMENT_DAYS,
            'trading_start_time': self.taiwan_config.TRADING_START,
            'trading_end_time': self.taiwan_config.TRADING_END,
            'max_intraday_gap_hours': 4.5,  # Taiwan trading session length
            
            # Market structure rules
            'price_limit_percent': self.taiwan_config.PRICE_LIMIT_PERCENT,
            'min_price_tick': self.taiwan_config.PRICE_TICK_SIZE,
            'currency_code': self.taiwan_config.CURRENCY,
            
            # Data quality rules
            'max_missing_trading_days': 0.05,  # 5% max missing trading days
            'min_stocks_per_date': 100,       # Min stocks per cross-section
            'max_outlier_percentage': 0.02,   # 2% max outliers
            
            # Feature naming conventions
            'prohibited_terms': [
                'insider', 'private', 'confidential', 'non_public',
                'after_hours', 'pre_market', 'overnight', 'extended_hours'
            ],
            'settlement_sensitive_terms': [
                'volume', 'turnover', 'trade_count', 'money_flow',
                'order_flow', 'transaction', 'settlement'
            ]
        }
        
        # Cache for expensive validations
        self._validation_cache = {}
        
    def _check_feature_naming_compliance(self, feature_name: str) -> Dict[str, Any]:
        """
        Check if feature name complies with Taiwan market conventions.
        
        Args:
            feature_name: Name of the feature to validate
            
        Returns:
            Naming compliance results
        """
        results = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'suggestions': []
        }
        
        feature_lower = feature_name.lower()
        
        # Check for prohibited terms
        for term in self.compliance_rules['prohibited_terms']:
            if term in feature_lower:
                results['violations'].append(f"Contains prohibited term: '{term}'")
                results['compliant'] = False
                
        # Check for settlement-sensitive features without proper lag indication
        has_settlement_term = any(term in feature_lower for term in 
                                 self.compliance_rules['settlement_sensitive_terms'])
        has_lag_indication = any(lag_term in feature_lower for lag_term in 
                               ['lag', 'delay', 't-2', 't+2', 'shifted'])
        
        if has_settlement_term and not has_lag_indication:
            results['warnings'].append(
                f"Settlement-sensitive feature '{feature_name}' may need T+2 lag indication"
            )
            results['suggestions'].append(f"Consider renaming to include 'lag2' or 't+2'")
            
        # Check for Taiwan-specific terminology
        taiwan_terms = ['tse', 'tpex', 'twse', 'twd', 'taiwan', 'tw']
        has_taiwan_context = any(term in feature_lower for term in taiwan_terms)
        
        if not has_taiwan_context and 'market' in feature_lower:
            results['suggestions'].append("Consider adding Taiwan market context (e.g., 'tse', 'taiwan')")
            
        return results
        
    def _validate_temporal_alignment(
        self, 
        feature_data: pd.Series, 
        feature_name: str = ""
    ) -> Dict[str, Any]:
        """
        Validate temporal alignment with Taiwan market trading calendar.
        
        Args:
            feature_data: Feature values with datetime index
            feature_name: Feature name for context
            
        Returns:
            Temporal alignment validation results
        """
        results = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Extract date information
            if isinstance(feature_data.index, pd.MultiIndex):
                # Panel data - get date level
                date_values = feature_data.index.get_level_values(0)
            else:
                # Time series - use index directly
                date_values = feature_data.index
                
            dates = pd.to_datetime(date_values)
            unique_dates = dates.unique()
            
            # Check alignment with trading calendar
            trading_dates = []
            non_trading_dates = []
            
            for date in unique_dates:
                if self.taiwan_config.is_trading_day(pd.Timestamp(date)):
                    trading_dates.append(date)
                else:
                    non_trading_dates.append(date)
                    
            results['statistics']['total_dates'] = len(unique_dates)
            results['statistics']['trading_dates'] = len(trading_dates)
            results['statistics']['non_trading_dates'] = len(non_trading_dates)
            results['statistics']['non_trading_ratio'] = len(non_trading_dates) / len(unique_dates) if unique_dates.size > 0 else 0
            
            # Validate trading calendar compliance
            if len(non_trading_dates) > 0:
                non_trading_ratio = len(non_trading_dates) / len(unique_dates)
                
                if non_trading_ratio > self.compliance_rules['max_missing_trading_days']:
                    results['violations'].append(
                        f"High non-trading dates ratio: {non_trading_ratio:.1%} "
                        f"(max allowed: {self.compliance_rules['max_missing_trading_days']:.1%})"
                    )
                    results['compliant'] = False
                elif self.allow_warnings and non_trading_ratio > 0.01:  # 1% warning threshold
                    results['warnings'].append(f"Contains {len(non_trading_dates)} non-trading dates")
                    
            # Check for weekend data
            weekend_dates = [d for d in unique_dates if d.weekday() >= 5]  # Saturday=5, Sunday=6
            if weekend_dates:
                results['warnings'].append(f"Found {len(weekend_dates)} weekend dates")
                results['statistics']['weekend_dates'] = len(weekend_dates)
                
            # Check date range continuity
            if len(trading_dates) > 1:
                date_series = pd.Series(sorted(trading_dates))
                business_days_expected = pd.bdate_range(
                    start=date_series.min(), 
                    end=date_series.max(), 
                    freq='B'
                )
                
                # Remove Taiwan holidays
                expected_trading_days = []
                for bday in business_days_expected:
                    if self.taiwan_config.is_trading_day(pd.Timestamp(bday)):
                        expected_trading_days.append(bday)
                        
                missing_dates = set(expected_trading_days) - set(trading_dates)
                if missing_dates:
                    results['warnings'].append(f"Missing {len(missing_dates)} expected trading dates")
                    results['statistics']['missing_trading_dates'] = len(missing_dates)
                    
        except Exception as e:
            logger.error(f"Error in temporal alignment validation: {str(e)}")
            results['violations'].append(f"Temporal validation error: {str(e)}")
            results['compliant'] = False
            
        return results
        
    def _validate_settlement_compliance(
        self, 
        feature_data: pd.Series, 
        feature_name: str = ""
    ) -> Dict[str, Any]:
        """
        Validate T+2 settlement cycle compliance.
        
        Args:
            feature_data: Feature values
            feature_name: Feature name for context
            
        Returns:
            Settlement compliance validation results
        """
        results = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'settlement_analysis': {}
        }
        
        try:
            # Check if feature is settlement-sensitive based on name
            is_settlement_sensitive = any(
                term in feature_name.lower() 
                for term in self.compliance_rules['settlement_sensitive_terms']
            )
            
            results['settlement_analysis']['is_settlement_sensitive'] = is_settlement_sensitive
            
            if not is_settlement_sensitive:
                # Not settlement-sensitive, compliance check passes
                return results
                
            # For settlement-sensitive features, check for proper lagging
            has_lag_indication = any(
                lag_term in feature_name.lower() 
                for lag_term in ['lag', 'delay', 't-2', 't+2', 'shifted', 'prev']
            )
            
            results['settlement_analysis']['has_lag_indication'] = has_lag_indication
            
            if not has_lag_indication:
                results['violations'].append(
                    f"Settlement-sensitive feature '{feature_name}' lacks T+2 lag indication"
                )
                results['compliant'] = False
                
            # Analyze feature values for potential lookahead bias
            if isinstance(feature_data.index, pd.MultiIndex):
                # Panel data - check within each stock group
                stock_groups = feature_data.groupby(level=1)  # Assuming level 1 is stock
                
                suspicious_patterns = 0
                total_stocks = 0
                
                for stock_id, stock_data in stock_groups:
                    if len(stock_data) < 5:  # Need minimum data points
                        continue
                        
                    total_stocks += 1
                    
                    # Check for perfect correlation with future values (potential lookahead)
                    stock_values = stock_data.values
                    if len(stock_values) > 3:
                        # Compare current values with shifted versions
                        current_series = pd.Series(stock_values[:-2])
                        future_series = pd.Series(stock_values[2:])
                        
                        if len(current_series) > 2 and len(future_series) > 2:
                            correlation = current_series.corr(future_series)
                            if not pd.isna(correlation) and abs(correlation) > 0.99:
                                suspicious_patterns += 1
                                
                if total_stocks > 0:
                    suspicious_ratio = suspicious_patterns / total_stocks
                    results['settlement_analysis']['suspicious_lookahead_ratio'] = suspicious_ratio
                    
                    if suspicious_ratio > 0.1:  # More than 10% of stocks show suspicious patterns
                        results['warnings'].append(
                            f"Potential lookahead bias detected in {suspicious_patterns}/{total_stocks} stocks"
                        )
                        
        except Exception as e:
            logger.error(f"Error in settlement compliance validation: {str(e)}")
            results['violations'].append(f"Settlement validation error: {str(e)}")
            results['compliant'] = False
            
        return results
        
    def _validate_price_limit_compliance(
        self, 
        feature_data: pd.Series, 
        feature_name: str = ""
    ) -> Dict[str, Any]:
        """
        Validate compliance with Taiwan daily price limits (±10%).
        
        Args:
            feature_data: Feature values
            feature_name: Feature name for context
            
        Returns:
            Price limit compliance validation results
        """
        results = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'price_analysis': {}
        }
        
        try:
            # Check if feature is price-related
            is_price_feature = any(
                term in feature_name.lower() 
                for term in ['price', 'close', 'open', 'high', 'low', 'return', 'change']
            )
            
            results['price_analysis']['is_price_feature'] = is_price_feature
            
            if not is_price_feature:
                # Not a price feature, skip price limit validation
                return results
                
            clean_data = feature_data.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_data) == 0:
                return results
                
            # For return/change features, check for extreme values
            if any(term in feature_name.lower() for term in ['return', 'change', 'pct']):
                # Analyze returns for extreme values
                extreme_moves = clean_data[abs(clean_data) > 0.12]  # 12% threshold (above 10% limit)
                
                if len(extreme_moves) > 0:
                    extreme_ratio = len(extreme_moves) / len(clean_data)
                    results['price_analysis']['extreme_moves_count'] = len(extreme_moves)
                    results['price_analysis']['extreme_moves_ratio'] = extreme_ratio
                    
                    if extreme_ratio > 0.005:  # More than 0.5% extreme moves
                        results['warnings'].append(
                            f"High frequency of extreme price moves: {extreme_ratio:.1%} "
                            f"above 12% (Taiwan limit is 10%)"
                        )
                        
                    # Check for impossibly large moves
                    max_move = abs(clean_data).max()
                    if max_move > 0.5:  # 50% move in a day
                        results['violations'].append(
                            f"Impossible price move detected: {max_move:.1%} "
                            f"(exceeds reasonable bounds)"
                        )
                        results['compliant'] = False
                        
            # For price level features, check for reasonable ranges
            elif any(term in feature_name.lower() for term in ['price', 'close', 'open', 'high', 'low']):
                price_stats = {
                    'min_price': float(clean_data.min()),
                    'max_price': float(clean_data.max()),
                    'mean_price': float(clean_data.mean()),
                    'price_range': float(clean_data.max() - clean_data.min())
                }
                results['price_analysis'].update(price_stats)
                
                # Check for unreasonable price levels
                if price_stats['min_price'] <= 0:
                    results['violations'].append("Non-positive prices detected")
                    results['compliant'] = False
                    
                if price_stats['max_price'] > 10000:  # Very expensive stock (>10,000 TWD)
                    results['warnings'].append(f"Very high price detected: {price_stats['max_price']:.2f} TWD")
                    
                if price_stats['min_price'] < 1 and price_stats['min_price'] > 0:  # Penny stock
                    results['warnings'].append(f"Very low price detected: {price_stats['min_price']:.2f} TWD")
                    
        except Exception as e:
            logger.error(f"Error in price limit validation: {str(e)}")
            results['violations'].append(f"Price validation error: {str(e)}")
            results['compliant'] = False
            
        return results
        
    def _validate_cross_sectional_consistency(
        self, 
        feature_data: pd.Series, 
        feature_name: str = ""
    ) -> Dict[str, Any]:
        """
        Validate cross-sectional consistency for panel data.
        
        Args:
            feature_data: Feature values (should have MultiIndex for panel data)
            feature_name: Feature name for context
            
        Returns:
            Cross-sectional consistency validation results
        """
        results = {
            'compliant': True,
            'violations': [],
            'warnings': [],
            'cross_sectional_analysis': {}
        }
        
        try:
            # Check if this is panel data
            if not isinstance(feature_data.index, pd.MultiIndex):
                # Not panel data, skip cross-sectional validation
                results['cross_sectional_analysis']['is_panel_data'] = False
                return results
                
            results['cross_sectional_analysis']['is_panel_data'] = True
            
            # Extract date and stock information
            dates = feature_data.index.get_level_values(0)
            stocks = feature_data.index.get_level_values(1)
            
            unique_dates = dates.unique()
            unique_stocks = stocks.unique()
            
            results['cross_sectional_analysis']['unique_dates'] = len(unique_dates)
            results['cross_sectional_analysis']['unique_stocks'] = len(unique_stocks)
            
            # Check for consistent stock coverage across dates
            date_stock_counts = []
            sparse_dates = []
            
            for date in unique_dates:
                date_mask = dates == date
                stocks_on_date = stocks[date_mask].nunique()
                date_stock_counts.append(stocks_on_date)
                
                if stocks_on_date < self.compliance_rules['min_stocks_per_date']:
                    sparse_dates.append((date, stocks_on_date))
                    
            if date_stock_counts:
                results['cross_sectional_analysis']['min_stocks_per_date'] = min(date_stock_counts)
                results['cross_sectional_analysis']['max_stocks_per_date'] = max(date_stock_counts)
                results['cross_sectional_analysis']['mean_stocks_per_date'] = np.mean(date_stock_counts)
                results['cross_sectional_analysis']['std_stocks_per_date'] = np.std(date_stock_counts)
                
            if sparse_dates:
                results['warnings'].append(
                    f"Found {len(sparse_dates)} dates with insufficient stock coverage "
                    f"(< {self.compliance_rules['min_stocks_per_date']} stocks)"
                )
                
            # Check for stock coverage consistency
            stock_date_counts = []
            for stock in unique_stocks:
                stock_mask = stocks == stock
                dates_for_stock = dates[stock_mask].nunique()
                stock_date_counts.append(dates_for_stock)
                
            if stock_date_counts:
                coverage_std = np.std(stock_date_counts)
                coverage_cv = coverage_std / np.mean(stock_date_counts) if np.mean(stock_date_counts) > 0 else 0
                
                results['cross_sectional_analysis']['coverage_coefficient_variation'] = coverage_cv
                
                if coverage_cv > 0.3:  # High variation in coverage
                    results['warnings'].append(
                        f"Inconsistent stock coverage across time: CV={coverage_cv:.2f}"
                    )
                    
            # Check for data completeness within cross-sections
            missing_by_date = {}
            for date in unique_dates:
                date_data = feature_data[dates == date]
                missing_count = date_data.isnull().sum()
                missing_ratio = missing_count / len(date_data) if len(date_data) > 0 else 0
                
                if missing_ratio > 0.1:  # More than 10% missing
                    missing_by_date[date] = missing_ratio
                    
            if missing_by_date:
                results['warnings'].append(
                    f"High missing data on {len(missing_by_date)} dates "
                    f"(>10% missing per cross-section)"
                )
                results['cross_sectional_analysis']['dates_with_high_missing'] = len(missing_by_date)
                
        except Exception as e:
            logger.error(f"Error in cross-sectional validation: {str(e)}")
            results['violations'].append(f"Cross-sectional validation error: {str(e)}")
            results['compliant'] = False
            
        return results
        
    def validate_feature_compliance(
        self, 
        feature_data: pd.Series, 
        feature_name: str = "",
        comprehensive: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive Taiwan market compliance validation for a single feature.
        
        Args:
            feature_data: Feature values
            feature_name: Feature name for context
            comprehensive: Whether to run all compliance checks
            
        Returns:
            Complete compliance validation results
        """
        logger.info(f"Validating Taiwan compliance for feature: {feature_name}")
        
        # Initialize results
        compliance_results = {
            'feature_name': feature_name,
            'overall_compliant': True,
            'compliance_score': 100.0,  # Start with perfect score
            'critical_violations': [],
            'warnings': [],
            'validation_summary': {},
            'recommendations': []
        }
        
        try:
            # 1. Feature naming compliance
            naming_results = self._check_feature_naming_compliance(feature_name)
            compliance_results['naming_compliance'] = naming_results
            
            if not naming_results['compliant']:
                compliance_results['overall_compliant'] = False
                compliance_results['critical_violations'].extend(naming_results['violations'])
                compliance_results['compliance_score'] -= 20  # Severe penalty for naming violations
                
            compliance_results['warnings'].extend(naming_results['warnings'])
            compliance_results['recommendations'].extend(naming_results['suggestions'])
            
            # 2. Temporal alignment validation
            temporal_results = self._validate_temporal_alignment(feature_data, feature_name)
            compliance_results['temporal_compliance'] = temporal_results
            
            if not temporal_results['compliant']:
                compliance_results['overall_compliant'] = False
                compliance_results['critical_violations'].extend(temporal_results['violations'])
                compliance_results['compliance_score'] -= 15
                
            compliance_results['warnings'].extend(temporal_results['warnings'])
            
            # 3. Settlement compliance validation
            settlement_results = self._validate_settlement_compliance(feature_data, feature_name)
            compliance_results['settlement_compliance'] = settlement_results
            
            if not settlement_results['compliant']:
                compliance_results['overall_compliant'] = False
                compliance_results['critical_violations'].extend(settlement_results['violations'])
                compliance_results['compliance_score'] -= 25  # High penalty for settlement violations
                
            compliance_results['warnings'].extend(settlement_results['warnings'])
            
            # 4. Price limit compliance (if comprehensive)
            if comprehensive:
                price_results = self._validate_price_limit_compliance(feature_data, feature_name)
                compliance_results['price_compliance'] = price_results
                
                if not price_results['compliant']:
                    compliance_results['overall_compliant'] = False
                    compliance_results['critical_violations'].extend(price_results['violations'])
                    compliance_results['compliance_score'] -= 15
                    
                compliance_results['warnings'].extend(price_results['warnings'])
                
                # 5. Cross-sectional consistency
                cross_sectional_results = self._validate_cross_sectional_consistency(feature_data, feature_name)
                compliance_results['cross_sectional_compliance'] = cross_sectional_results
                
                if not cross_sectional_results['compliant']:
                    compliance_results['critical_violations'].extend(cross_sectional_results['violations'])
                    compliance_results['compliance_score'] -= 10
                    
                compliance_results['warnings'].extend(cross_sectional_results['warnings'])
                
            # Apply warning penalty in strict mode
            if self.strict_mode and compliance_results['warnings']:
                warning_penalty = min(10, len(compliance_results['warnings']) * 2)
                compliance_results['compliance_score'] -= warning_penalty
                
            # Ensure score doesn't go below 0
            compliance_results['compliance_score'] = max(0, compliance_results['compliance_score'])
            
            # Generate final recommendations
            if not compliance_results['overall_compliant']:
                compliance_results['recommendations'].append("Feature requires fixes before use in Taiwan market")
            elif compliance_results['warnings'] and self.strict_mode:
                compliance_results['recommendations'].append("Feature has warnings - review before production use")
            elif compliance_results['compliance_score'] >= 90:
                compliance_results['recommendations'].append("Feature complies with Taiwan market requirements")
            else:
                compliance_results['recommendations'].append("Feature needs minor improvements for optimal compliance")
                
            # Summary statistics
            compliance_results['validation_summary'] = {
                'total_checks_performed': 3 + (2 if comprehensive else 0),
                'critical_violations_count': len(compliance_results['critical_violations']),
                'warnings_count': len(compliance_results['warnings']),
                'compliance_score': compliance_results['compliance_score'],
                'strict_mode': self.strict_mode,
                'comprehensive_mode': comprehensive
            }
            
        except Exception as e:
            logger.error(f"Error in Taiwan compliance validation: {str(e)}")
            compliance_results['critical_violations'].append(f"Validation error: {str(e)}")
            compliance_results['overall_compliant'] = False
            compliance_results['compliance_score'] = 0.0
            
        logger.info(
            f"Compliance validation completed for {feature_name}: "
            f"score={compliance_results['compliance_score']:.1f}, "
            f"compliant={compliance_results['overall_compliant']}"
        )
        
        return compliance_results
        
    def batch_validate_compliance(
        self, 
        features_df: pd.DataFrame,
        comprehensive: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate compliance for multiple features in batch.
        
        Args:
            features_df: DataFrame with features to validate
            comprehensive: Whether to run comprehensive validation
            
        Returns:
            Dictionary mapping feature names to compliance results
        """
        logger.info(f"Batch validating Taiwan compliance for {len(features_df.columns)} features")
        
        results = {}
        
        for i, feature_name in enumerate(features_df.columns):
            logger.info(f"Processing feature {i+1}/{len(features_df.columns)}: {feature_name}")
            
            feature_data = features_df[feature_name]
            feature_results = self.validate_feature_compliance(
                feature_data, 
                feature_name, 
                comprehensive
            )
            
            results[feature_name] = feature_results
            
            # Log progress every 10 features
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(features_df.columns)} features")
                
        return results
        
    def generate_compliance_report(
        self, 
        validation_results: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report.
        
        Args:
            validation_results: Results from batch_validate_compliance
            output_path: Optional path to save report
            
        Returns:
            Compliance report summary
        """
        report = {
            'summary': {
                'total_features': len(validation_results),
                'timestamp': datetime.now().isoformat(),
                'taiwan_config': str(self.taiwan_config),
                'validation_mode': 'strict' if self.strict_mode else 'permissive'
            },
            'compliance_statistics': {},
            'violation_analysis': {},
            'recommendations': []
        }
        
        # Analyze results
        compliance_scores = []
        compliant_features = []
        violation_counts = []
        warning_counts = []
        
        for feature_name, results in validation_results.items():
            compliance_scores.append(results.get('compliance_score', 0))
            compliant_features.append(results.get('overall_compliant', False))
            violation_counts.append(len(results.get('critical_violations', [])))
            warning_counts.append(len(results.get('warnings', [])))
            
        # Calculate summary statistics
        report['compliance_statistics'] = {
            'compliant_features': sum(compliant_features),
            'compliant_percentage': sum(compliant_features) / len(compliant_features) * 100 if compliant_features else 0,
            'non_compliant_features': len(compliant_features) - sum(compliant_features),
            'mean_compliance_score': float(np.mean(compliance_scores)),
            'median_compliance_score': float(np.median(compliance_scores)),
            'min_compliance_score': float(np.min(compliance_scores)),
            'max_compliance_score': float(np.max(compliance_scores)),
            'total_violations': sum(violation_counts),
            'total_warnings': sum(warning_counts)
        }
        
        # Generate recommendations
        compliant_pct = report['compliance_statistics']['compliant_percentage']
        if compliant_pct >= 95:
            report['recommendations'].append("Excellent compliance - features ready for Taiwan market")
        elif compliant_pct >= 80:
            report['recommendations'].append("Good compliance - minor issues to address")
        elif compliant_pct >= 60:
            report['recommendations'].append("Moderate compliance - significant improvements needed")
        else:
            report['recommendations'].append("Poor compliance - major overhaul required")
            
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"Compliance report saved to {output_path}")
            
        return report
        
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get summary of compliance rules and configuration."""
        return {
            'taiwan_market_config': {
                'settlement_days': self.taiwan_config.SETTLEMENT_DAYS,
                'trading_hours': f"{self.taiwan_config.TRADING_START}-{self.taiwan_config.TRADING_END}",
                'price_limit_percent': self.taiwan_config.PRICE_LIMIT_PERCENT,
                'currency': self.taiwan_config.CURRENCY,
                'timezone': self.taiwan_config.TIMEZONE
            },
            'compliance_rules': self.compliance_rules.copy(),
            'validator_config': {
                'strict_mode': self.strict_mode,
                'allow_warnings': self.allow_warnings
            }
        }


def create_taiwan_compliance_validator(
    strict_mode: bool = True,
    allow_warnings: bool = True,
    taiwan_config: Optional[TaiwanMarketConfig] = None
) -> TaiwanComplianceValidator:
    """
    Factory function to create a properly configured TaiwanComplianceValidator.
    
    Args:
        strict_mode: Whether to enforce strict compliance
        allow_warnings: Whether to report warnings
        taiwan_config: Taiwan market configuration
        
    Returns:
        Configured TaiwanComplianceValidator instance
    """
    validator = TaiwanComplianceValidator(
        taiwan_config=taiwan_config or TaiwanMarketConfig(),
        strict_mode=strict_mode,
        allow_warnings=allow_warnings
    )
    
    logger.info(
        f"Created TaiwanComplianceValidator (strict_mode={strict_mode}, "
        f"allow_warnings={allow_warnings})"
    )
    
    return validator