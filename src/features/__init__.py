"""
ML4T Features Module

OpenFE integration for automated feature engineering with Taiwan market specifics.
Expert-validated time-series handling to prevent lookahead bias.
"""

__version__ = "1.0.0"

from .openfe_wrapper import FeatureGenerator
from .taiwan_config import TaiwanMarketConfig

__all__ = ['FeatureGenerator', 'TaiwanMarketConfig']