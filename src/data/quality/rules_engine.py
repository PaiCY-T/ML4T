"""
Configurable Validation Rules Engine.

This module provides a flexible rules engine for data quality validation,
allowing dynamic configuration of validation rules through YAML/JSON files
and programmatic rule definition.
"""

import asyncio
import logging
import yaml
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable

from ..core.temporal import TemporalValue, DataType
from .validation_engine import (
    ValidationPlugin, ValidationContext, ValidationOutput, ValidationResult,
    ValidationPriority
)
from .validators import QualityIssue, SeverityLevel, QualityCheckType

logger = logging.getLogger(__name__)


class RuleOperator(Enum):
    """Operators for rule conditions."""
    EQ = "eq"          # Equal
    NE = "ne"          # Not equal
    GT = "gt"          # Greater than
    GE = "ge"          # Greater than or equal
    LT = "lt"          # Less than
    LE = "le"          # Less than or equal
    IN = "in"          # In list
    NOT_IN = "not_in"  # Not in list
    BETWEEN = "between" # Between two values
    CONTAINS = "contains" # String contains
    REGEX = "regex"    # Regex match
    IS_NULL = "is_null" # Is null/None
    IS_NOT_NULL = "is_not_null" # Is not null/None


class RuleAction(Enum):
    """Actions to take when rule is triggered."""
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SKIP = "skip"


@dataclass
class RuleCondition:
    """A single rule condition."""
    field_path: str  # Path to field in data (e.g., "value.close_price", "metadata.volume")
    operator: RuleOperator
    value: Any  # Expected value or threshold
    description: Optional[str] = None
    
    def evaluate(self, data: Any) -> bool:
        """Evaluate condition against provided data."""
        try:
            # Extract field value using path
            field_value = self._extract_field_value(data, self.field_path)
            
            # Apply operator
            return self._apply_operator(field_value, self.operator, self.value)
            
        except Exception as e:
            logger.debug(f"Rule condition evaluation failed: {e}")
            return False
    
    def _extract_field_value(self, data: Any, path: str) -> Any:
        """Extract field value using dot notation path."""
        parts = path.split('.')
        current = data
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                current = current[idx] if 0 <= idx < len(current) else None
            else:
                return None
        
        return current
    
    def _apply_operator(self, field_value: Any, operator: RuleOperator, expected: Any) -> bool:
        """Apply operator to field value and expected value."""
        if operator == RuleOperator.IS_NULL:
            return field_value is None
        
        if operator == RuleOperator.IS_NOT_NULL:
            return field_value is not None
        
        if field_value is None:
            return False
        
        try:
            if operator == RuleOperator.EQ:
                return field_value == expected
            elif operator == RuleOperator.NE:
                return field_value != expected
            elif operator == RuleOperator.GT:
                return float(field_value) > float(expected)
            elif operator == RuleOperator.GE:
                return float(field_value) >= float(expected)
            elif operator == RuleOperator.LT:
                return float(field_value) < float(expected)
            elif operator == RuleOperator.LE:
                return float(field_value) <= float(expected)
            elif operator == RuleOperator.IN:
                return field_value in expected
            elif operator == RuleOperator.NOT_IN:
                return field_value not in expected
            elif operator == RuleOperator.BETWEEN:
                if len(expected) != 2:
                    return False
                min_val, max_val = expected
                return float(min_val) <= float(field_value) <= float(max_val)
            elif operator == RuleOperator.CONTAINS:
                return str(expected) in str(field_value)
            elif operator == RuleOperator.REGEX:
                import re
                return bool(re.search(str(expected), str(field_value)))
            else:
                return False
                
        except (ValueError, TypeError):
            return False


@dataclass
class ValidationRule:
    """A configurable validation rule."""
    name: str
    description: str
    data_types: List[DataType]
    conditions: List[RuleCondition]
    action: RuleAction
    priority: ValidationPriority = ValidationPriority.MEDIUM
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Logical operators
    condition_logic: str = "AND"  # "AND" or "OR"
    
    def evaluate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Evaluate all conditions for this rule."""
        if not self.enabled:
            return False
        
        if value.data_type not in self.data_types:
            return False
        
        # Create evaluation context with value and context data
        eval_data = {
            "value": value.value,
            "symbol": value.symbol,
            "data_date": value.value_date,
            "as_of_date": value.as_of_date,
            "data_type": value.data_type.value,
            "metadata": value.metadata or {},
            "context": context.metadata
        }
        
        # Evaluate conditions
        condition_results = [
            condition.evaluate(eval_data) for condition in self.conditions
        ]
        
        if not condition_results:
            return False
        
        # Apply logic
        if self.condition_logic.upper() == "OR":
            return any(condition_results)
        else:  # Default to AND
            return all(condition_results)
    
    def create_issue(self, value: TemporalValue, context: ValidationContext) -> QualityIssue:
        """Create a quality issue when rule is triggered."""
        severity_map = {
            RuleAction.WARNING: SeverityLevel.WARNING,
            RuleAction.ERROR: SeverityLevel.ERROR,
            RuleAction.CRITICAL: SeverityLevel.CRITICAL
        }
        
        severity = severity_map.get(self.action, SeverityLevel.INFO)
        
        return QualityIssue(
            check_type=QualityCheckType.BUSINESS_RULES,
            severity=severity,
            symbol=value.symbol or "",
            data_type=value.data_type,
            data_date=value.value_date,
            issue_date=datetime.utcnow(),
            description=f"Rule '{self.name}' triggered: {self.description}",
            details={
                "rule_name": self.name,
                "rule_tags": self.tags,
                "conditions_met": len(self.conditions),
                "action": self.action.value,
                "metadata": self.metadata
            },
            suggested_action=self.metadata.get("suggested_action", "Review rule conditions")
        )


class RulesEngine:
    """Configurable rules engine for data validation."""
    
    def __init__(self):
        self._rules: Dict[str, ValidationRule] = {}
        self._rules_by_data_type: Dict[DataType, List[ValidationRule]] = {}
        self._rules_by_tag: Dict[str, List[ValidationRule]] = {}
        
        # Performance metrics
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.rules_triggered = 0
        
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule to the engine."""
        if rule.name in self._rules:
            raise ValueError(f"Rule '{rule.name}' already exists")
        
        self._rules[rule.name] = rule
        
        # Index by data types
        for data_type in rule.data_types:
            if data_type not in self._rules_by_data_type:
                self._rules_by_data_type[data_type] = []
            self._rules_by_data_type[data_type].append(rule)
        
        # Index by tags
        for tag in rule.tags:
            if tag not in self._rules_by_tag:
                self._rules_by_tag[tag] = []
            self._rules_by_tag[tag].append(rule)
        
        logger.info(f"Added validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule."""
        if rule_name not in self._rules:
            return False
        
        rule = self._rules[rule_name]
        
        # Remove from indices
        for data_type in rule.data_types:
            if data_type in self._rules_by_data_type:
                self._rules_by_data_type[data_type].remove(rule)
        
        for tag in rule.tags:
            if tag in self._rules_by_tag:
                self._rules_by_tag[tag].remove(rule)
        
        del self._rules[rule_name]
        logger.info(f"Removed validation rule: {rule_name}")
        return True
    
    def get_rule(self, rule_name: str) -> Optional[ValidationRule]:
        """Get a specific rule."""
        return self._rules.get(rule_name)
    
    def list_rules(self, data_type: Optional[DataType] = None, 
                  tag: Optional[str] = None) -> List[ValidationRule]:
        """List rules, optionally filtered by data type or tag."""
        if data_type:
            return self._rules_by_data_type.get(data_type, [])
        elif tag:
            return self._rules_by_tag.get(tag, [])
        else:
            return list(self._rules.values())
    
    def evaluate_rules(self, value: TemporalValue, 
                      context: ValidationContext) -> List[QualityIssue]:
        """Evaluate all applicable rules for a temporal value."""
        start_time = datetime.utcnow()
        issues = []
        
        # Get applicable rules
        applicable_rules = self._rules_by_data_type.get(value.data_type, [])
        
        for rule in applicable_rules:
            try:
                if rule.evaluate(value, context):
                    if rule.action in [RuleAction.WARNING, RuleAction.ERROR, RuleAction.CRITICAL]:
                        issue = rule.create_issue(value, context)
                        issues.append(issue)
                        self.rules_triggered += 1
                        
                        logger.debug(f"Rule '{rule.name}' triggered for {value.symbol} "
                                   f"{value.data_type.value}")
                
            except Exception as e:
                logger.error(f"Error evaluating rule '{rule.name}': {e}")
                
                # Create error issue
                error_issue = QualityIssue(
                    check_type=QualityCheckType.VALIDITY,
                    severity=SeverityLevel.ERROR,
                    symbol=value.symbol or "",
                    data_type=value.data_type,
                    data_date=value.value_date,
                    issue_date=datetime.utcnow(),
                    description=f"Rule evaluation error: {rule.name}",
                    details={"error": str(e), "rule_name": rule.name}
                )
                issues.append(error_issue)
        
        # Update metrics
        self.evaluation_count += 1
        evaluation_time = (datetime.utcnow() - start_time).total_seconds()
        self.total_evaluation_time += evaluation_time
        
        return issues
    
    def load_rules_from_yaml(self, file_path: Union[str, Path]) -> int:
        """Load rules from a YAML file."""
        try:
            with open(file_path, 'r') as f:
                rules_config = yaml.safe_load(f)
            
            return self._load_rules_from_config(rules_config)
            
        except Exception as e:
            logger.error(f"Failed to load rules from {file_path}: {e}")
            raise
    
    def load_rules_from_json(self, file_path: Union[str, Path]) -> int:
        """Load rules from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                rules_config = json.load(f)
            
            return self._load_rules_from_config(rules_config)
            
        except Exception as e:
            logger.error(f"Failed to load rules from {file_path}: {e}")
            raise
    
    def _load_rules_from_config(self, config: Dict[str, Any]) -> int:
        """Load rules from configuration dictionary."""
        rules_loaded = 0
        
        rules_list = config.get('rules', [])
        
        for rule_config in rules_list:
            try:
                rule = self._create_rule_from_config(rule_config)
                self.add_rule(rule)
                rules_loaded += 1
                
            except Exception as e:
                logger.error(f"Failed to create rule from config: {e}")
                logger.debug(f"Rule config: {rule_config}")
        
        logger.info(f"Loaded {rules_loaded} rules from configuration")
        return rules_loaded
    
    def _create_rule_from_config(self, config: Dict[str, Any]) -> ValidationRule:
        """Create a ValidationRule from configuration."""
        # Parse data types
        data_types = []
        for dt_str in config.get('data_types', []):
            try:
                data_types.append(DataType(dt_str))
            except ValueError:
                logger.warning(f"Unknown data type: {dt_str}")
        
        # Parse conditions
        conditions = []
        for cond_config in config.get('conditions', []):
            operator = RuleOperator(cond_config['operator'])
            condition = RuleCondition(
                field_path=cond_config['field_path'],
                operator=operator,
                value=cond_config['value'],
                description=cond_config.get('description')
            )
            conditions.append(condition)
        
        # Parse action and priority
        action = RuleAction(config['action'])
        priority = ValidationPriority.MEDIUM
        
        if 'priority' in config:
            priority_str = config['priority'].upper()
            if hasattr(ValidationPriority, priority_str):
                priority = getattr(ValidationPriority, priority_str)
        
        return ValidationRule(
            name=config['name'],
            description=config['description'],
            data_types=data_types,
            conditions=conditions,
            action=action,
            priority=priority,
            enabled=config.get('enabled', True),
            tags=config.get('tags', []),
            metadata=config.get('metadata', {}),
            condition_logic=config.get('condition_logic', 'AND')
        )
    
    def export_rules_to_yaml(self, file_path: Union[str, Path]) -> None:
        """Export all rules to a YAML file."""
        config = {
            'rules': [self._rule_to_config(rule) for rule in self._rules.values()]
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Exported {len(self._rules)} rules to {file_path}")
    
    def _rule_to_config(self, rule: ValidationRule) -> Dict[str, Any]:
        """Convert a ValidationRule to configuration dictionary."""
        return {
            'name': rule.name,
            'description': rule.description,
            'data_types': [dt.value for dt in rule.data_types],
            'conditions': [
                {
                    'field_path': cond.field_path,
                    'operator': cond.operator.value,
                    'value': cond.value,
                    'description': cond.description
                }
                for cond in rule.conditions
            ],
            'action': rule.action.value,
            'priority': rule.priority.name.lower(),
            'enabled': rule.enabled,
            'tags': rule.tags,
            'metadata': rule.metadata,
            'condition_logic': rule.condition_logic
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rules engine statistics."""
        avg_evaluation_time = (
            self.total_evaluation_time / max(self.evaluation_count, 1)
        )
        
        rules_by_action = {}
        rules_by_priority = {}
        
        for rule in self._rules.values():
            action = rule.action.value
            rules_by_action[action] = rules_by_action.get(action, 0) + 1
            
            priority = rule.priority.name
            rules_by_priority[priority] = rules_by_priority.get(priority, 0) + 1
        
        return {
            'total_rules': len(self._rules),
            'enabled_rules': sum(1 for r in self._rules.values() if r.enabled),
            'evaluation_count': self.evaluation_count,
            'rules_triggered': self.rules_triggered,
            'avg_evaluation_time_seconds': avg_evaluation_time,
            'rules_by_action': rules_by_action,
            'rules_by_priority': rules_by_priority,
            'data_types_covered': len(self._rules_by_data_type)
        }


class RulesBasedValidator(ValidationPlugin):
    """Validation plugin that uses the rules engine."""
    
    def __init__(self, rules_engine: RulesEngine):
        self.rules_engine = rules_engine
    
    @property
    def name(self) -> str:
        return "rules_based_validator"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def priority(self) -> ValidationPriority:
        return ValidationPriority.MEDIUM
    
    @property
    def supported_data_types(self) -> Set[DataType]:
        # Support all data types that have rules
        return set(self.rules_engine._rules_by_data_type.keys())
    
    def can_validate(self, value: TemporalValue, context: ValidationContext) -> bool:
        """Check if there are applicable rules for this value."""
        return len(self.rules_engine.list_rules(value.data_type)) > 0
    
    async def validate(self, value: TemporalValue, context: ValidationContext) -> ValidationOutput:
        """Run rules engine validation."""
        start_time = datetime.utcnow()
        
        # Evaluate rules
        issues = self.rules_engine.evaluate_rules(value, context)
        
        # Determine result
        result = ValidationResult.PASS
        if any(issue.severity == SeverityLevel.CRITICAL for issue in issues):
            result = ValidationResult.FAIL
        elif any(issue.severity in [SeverityLevel.ERROR, SeverityLevel.WARNING] for issue in issues):
            result = ValidationResult.WARNING
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ValidationOutput(
            validator_name=self.name,
            validation_id=f"{self.name}_{int(start_time.timestamp())}",
            result=result,
            issues=issues,
            execution_time_ms=execution_time,
            metadata={
                'rules_evaluated': len(self.rules_engine.list_rules(value.data_type)),
                'rules_triggered': len(issues)
            }
        )


# Predefined rule sets for Taiwan market

def create_taiwan_market_rules() -> List[ValidationRule]:
    """Create predefined rules for Taiwan market validation."""
    rules = []
    
    # Price validation rules
    rules.append(ValidationRule(
        name="price_positive",
        description="Stock price must be positive",
        data_types=[DataType.PRICE],
        conditions=[
            RuleCondition("value", RuleOperator.GT, 0)
        ],
        action=RuleAction.ERROR,
        priority=ValidationPriority.CRITICAL,
        tags=["price", "basic"]
    ))
    
    rules.append(ValidationRule(
        name="price_reasonable_range",
        description="Stock price should be in reasonable range",
        data_types=[DataType.PRICE],
        conditions=[
            RuleCondition("value", RuleOperator.BETWEEN, [0.01, 10000])
        ],
        action=RuleAction.WARNING,
        tags=["price", "range"]
    ))
    
    # Volume validation rules
    rules.append(ValidationRule(
        name="volume_non_negative",
        description="Trading volume must be non-negative",
        data_types=[DataType.VOLUME],
        conditions=[
            RuleCondition("value", RuleOperator.GE, 0)
        ],
        action=RuleAction.ERROR,
        priority=ValidationPriority.CRITICAL,
        tags=["volume", "basic"]
    ))
    
    # Market data completeness rules
    rules.append(ValidationRule(
        name="market_data_required_fields",
        description="Market data must contain required fields",
        data_types=[DataType.MARKET_DATA],
        conditions=[
            RuleCondition("value.close_price", RuleOperator.IS_NOT_NULL, None),
            RuleCondition("value.volume", RuleOperator.IS_NOT_NULL, None)
        ],
        action=RuleAction.ERROR,
        condition_logic="AND",
        tags=["completeness", "market_data"]
    ))
    
    # Fundamental data timing rules
    rules.append(ValidationRule(
        name="fundamental_not_future",
        description="Fundamental data cannot be from the future",
        data_types=[DataType.FUNDAMENTAL],
        conditions=[
            RuleCondition("data_date", RuleOperator.LE, date.today().isoformat())
        ],
        action=RuleAction.CRITICAL,
        priority=ValidationPriority.CRITICAL,
        tags=["temporal", "fundamental"]
    ))
    
    return rules


def create_rules_engine_with_taiwan_rules() -> RulesEngine:
    """Create a rules engine with Taiwan market rules pre-loaded."""
    engine = RulesEngine()
    
    taiwan_rules = create_taiwan_market_rules()
    for rule in taiwan_rules:
        engine.add_rule(rule)
    
    logger.info(f"Created rules engine with {len(taiwan_rules)} Taiwan market rules")
    return engine


# Example rule configuration files

TAIWAN_RULES_YAML = """
rules:
  - name: "price_limit_check"
    description: "Check daily price movement against 10% limit"
    data_types: ["price"]
    conditions:
      - field_path: "context.previous_close"
        operator: "is_not_null"
        value: null
        description: "Previous close price must be available"
      - field_path: "context.price_change_pct"
        operator: "le"
        value: 0.10
        description: "Price change must not exceed 10%"
    action: "error"
    priority: "critical"
    enabled: true
    tags: ["taiwan", "price_limit"]
    metadata:
      suggested_action: "Check for trading halts or corporate actions"

  - name: "volume_spike_detection"
    description: "Detect unusual volume spikes"
    data_types: ["volume"]
    conditions:
      - field_path: "context.volume_ratio"
        operator: "gt"
        value: 5.0
        description: "Volume ratio vs average > 5x"
    action: "warning"
    priority: "medium"
    enabled: true
    tags: ["taiwan", "volume", "anomaly"]
    metadata:
      suggested_action: "Check for news or events"

  - name: "trading_hours_check"
    description: "Validate data is within trading hours"
    data_types: ["price", "volume", "market_data"]
    conditions:
      - field_path: "context.trading_hours"
        operator: "eq"
        value: true
        description: "Must be within trading hours"
    action: "warning"
    priority: "medium"
    enabled: true
    tags: ["taiwan", "timing"]
"""

# Enhanced Taiwan market validation rules for Stream A requirements

def create_taiwan_fundamental_lag_rules() -> List[ValidationRule]:
    """Create Taiwan-specific fundamental data lag validation rules for 60-day requirement."""
    rules = []
    
    # Quarterly report lag rule (60 days max)
    rules.append(ValidationRule(
        name="taiwan_quarterly_fundamental_lag",
        description="Taiwan quarterly fundamental data must be available within 60 days",
        data_types={DataType.FUNDAMENTAL},
        conditions=[
            RuleCondition(
                field_path="value.fiscal_quarter",
                operator=RuleOperator.IN,
                value=[1, 2, 3],  # Q1, Q2, Q3
                description="Applies to quarterly reports (not annual)"
            ),
            RuleCondition(
                field_path="metadata.lag_days",
                operator=RuleOperator.GT,
                value=60,
                description="Lag exceeds 60 days"
            )
        ],
        action=RuleAction.CRITICAL,
        enabled=True,
        tags={"taiwan", "fundamental", "lag", "regulatory"},
        metadata={
            "regulation": "Taiwan Securities and Exchange Act",
            "max_lag_days": 60,
            "suggested_action": "Verify quarterly report filing timeline with Taiwan regulators"
        }
    ))
    
    # Annual report lag rule (90 days max)
    rules.append(ValidationRule(
        name="taiwan_annual_fundamental_lag",
        description="Taiwan annual fundamental data must be available within 90 days",
        data_types={DataType.FUNDAMENTAL},
        conditions=[
            RuleCondition(
                field_path="value.fiscal_quarter",
                operator=RuleOperator.EQ,
                value=4,  # Q4 = Annual report
                description="Applies to annual reports"
            ),
            RuleCondition(
                field_path="metadata.lag_days",
                operator=RuleOperator.GT,
                value=90,
                description="Lag exceeds 90 days"
            )
        ],
        action=RuleAction.CRITICAL,
        enabled=True,
        tags={"taiwan", "fundamental", "lag", "regulatory", "annual"},
        metadata={
            "regulation": "Taiwan Securities and Exchange Act",
            "max_lag_days": 90,
            "suggested_action": "Verify annual report filing timeline with Taiwan regulators"
        }
    ))
    
    # Future fundamental data rule (look-ahead bias prevention)
    rules.append(ValidationRule(
        name="taiwan_fundamental_future_data",
        description="Fundamental data cannot have negative lag (future data)",
        data_types={DataType.FUNDAMENTAL},
        conditions=[
            RuleCondition(
                field_path="metadata.lag_days",
                operator=RuleOperator.LT,
                value=0,
                description="Negative lag indicates future data"
            )
        ],
        action=RuleAction.CRITICAL,
        enabled=True,
        tags={"taiwan", "fundamental", "temporal", "look_ahead_bias"},
        metadata={
            "suggested_action": "Correct announcement date - cannot be before report date"
        }
    ))
    
    return rules


def create_enhanced_taiwan_rules_engine() -> RulesEngine:
    """Create an enhanced rules engine with comprehensive Taiwan market validation rules."""
    engine = RulesEngine()
    
    # Add Stream A enhanced rules
    taiwan_rules = create_taiwan_fundamental_lag_rules()
    
    for rule in taiwan_rules:
        engine.add_rule(rule)
    
    logger.info(f"Created enhanced Taiwan rules engine with {len(taiwan_rules)} new fundamental lag rules")
    return engine


if __name__ == "__main__":
    # Example usage
    engine = create_rules_engine_with_taiwan_rules()
    print(f"Rules engine created with {len(engine.list_rules())} rules")
    
    # Create enhanced engine for Stream A requirements
    enhanced_engine = create_enhanced_taiwan_rules_engine()
    print(f"Enhanced engine created with {len(enhanced_engine.list_rules())} fundamental lag rules")
    
    # Example of loading from YAML
    # engine.load_rules_from_yaml("taiwan_rules.yaml")