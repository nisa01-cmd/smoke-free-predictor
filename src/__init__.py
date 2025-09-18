"""
Smoke-Free Predictor Package
===========================

A comprehensive machine learning package for predicting smoke-free behavior outcomes.

Modules:
- predictor: Main prediction interface and utilities
- model: Machine learning model implementations
- data_processor: Data loading, cleaning, and preprocessing
- config: Configuration management
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# src/__init__.py

from .predictor import SmokeFreePredictor
from .predictor.model import SmokeFreeModel
from .predictor.data_processor import DataProcessor
from .predictor.config import Config


__all__ = [
    'SmokeFreePredictor',
    'SmokeFreeModel', 
    'DataProcessor',
    'Config'
]