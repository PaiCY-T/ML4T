"""
Backtesting Integration Module for ML4T.

This module provides integration components that connect the walk-forward validation
engine with point-in-time data systems and quality validation frameworks.

Key Components:
- PITValidator: Main integration class for PIT validation
- PITBiasDetector: Sophisticated bias detection system
- BiasCheckResult: Bias detection result container
- PITValidationConfig: Configuration for PIT validation

Example Usage:
    from src.backtesting.integration import (
        PITValidator, create_strict_pit_validator
    )
    
    # Create validator with strict settings
    validator = create_strict_pit_validator(temporal_store)
    
    # Run comprehensive validation
    results = validator.validate_walk_forward_scenario(
        wf_config=walk_forward_config,
        symbols=['2330.TW', '2317.TW'],
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31)
    )
"""

from .pit_validator import (
    PITValidator,
    PITBiasDetector,
    BiasCheckResult,
    PITValidationConfig,
    BiasType,
    ValidationLevel,
    create_standard_pit_validator,
    create_strict_pit_validator,
    create_paranoid_pit_validator
)

__all__ = [
    'PITValidator',
    'PITBiasDetector', 
    'BiasCheckResult',
    'PITValidationConfig',
    'BiasType',
    'ValidationLevel',
    'create_standard_pit_validator',
    'create_strict_pit_validator',
    'create_paranoid_pit_validator'
]