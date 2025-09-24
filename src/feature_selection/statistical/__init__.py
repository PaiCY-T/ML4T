"""
Statistical Feature Selection Module - Task #29 Stream A

This module implements statistical-based feature selection methods for reducing
the OpenFE-generated feature space from 500+ candidates to an optimal subset
of 50-100 features while preserving information content and eliminating 
multicollinearity.

Key Components:
- Correlation matrix analysis with VIF multicollinearity detection
- Variance thresholding to remove low-information features  
- Mutual information ranking for feature-target relationships
- Statistical significance testing for feature predictive power
- Memory-optimized processing for large feature sets
"""

from .correlation_analysis import CorrelationAnalyzer
from .variance_filter import VarianceFilter  
from .mutual_info_selector import MutualInfoSelector
from .significance_tester import StatisticalSignificanceTester
from .statistical_engine import StatisticalSelectionEngine

__all__ = [
    'CorrelationAnalyzer',
    'VarianceFilter', 
    'MutualInfoSelector',
    'StatisticalSignificanceTester',
    'StatisticalSelectionEngine'
]

# Version info
__version__ = '1.0.0'
__author__ = 'ML4T Team'
__description__ = 'Statistical Feature Selection Engine for Taiwan Stock Market'