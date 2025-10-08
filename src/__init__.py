"""
Nifty Prediction XGBoost Package

This package contains modules for building and training machine learning models
to predict Nifty index movements based on historical OHLCV data.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator

__all__ = [
    'DataLoader',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator'
]
