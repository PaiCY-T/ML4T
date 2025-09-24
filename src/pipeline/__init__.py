"""
Pipeline Integration Module - Task #28 Stream B
Feature expansion pipeline connecting 42 base factors to OpenFE.

This module provides:
- FeatureExpansionPipeline: Main integration pipeline
- Integration with Task #25 factor system
- Memory-efficient batch processing
- Taiwan market compliance
- Temporal consistency validation

Key Components:
- feature_expansion: Main pipeline implementation
"""

from .feature_expansion import (
    FeatureExpansionPipeline,
    create_feature_expansion_pipeline
)

__all__ = [
    'FeatureExpansionPipeline',
    'create_feature_expansion_pipeline'
]