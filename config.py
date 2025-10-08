"""
Configuration file for Nifty Prediction Model

This file contains all hyperparameters, paths, and settings for the ML pipeline.
Modify these parameters to customize model behavior without changing code.
"""

import os
from pathlib import Path

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Base directory - root of the project
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model directory - where trained models are saved
MODELS_DIR = BASE_DIR / 'models'

# Results directory - evaluation metrics, plots, reports
RESULTS_DIR = BASE_DIR / 'results'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Input data file name (place in data/raw/ directory)
INPUT_DATA_FILE = 'nifty_1min_data.csv'

# Date column configurations - adjust based on your CSV format
DATE_COLUMNS = ['Date', 'Time']  # Separate date and time columns
# Alternative: DATETIME_COLUMN = 'Datetime'  # Single datetime column

# Expected OHLCV columns in your data
OHLCV_COLUMNS = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}

# ============================================================================
# TRAIN-TEST SPLIT CONFIGURATION
# ============================================================================

# Number of months for training (from the start of data)
TRAIN_MONTHS = 7

# Number of months for testing (after training period)
TEST_MONTHS = 5

# Whether to use time-series cross-validation during training
# This creates multiple train-validation splits respecting temporal order
USE_TIME_SERIES_CV = True

# Number of splits for time-series cross-validation
N_SPLITS_CV = 5

# ============================================================================
# TARGET VARIABLE CONFIGURATION
# ============================================================================

# Target type: 'regression' or 'classification'
# - regression: Predict actual price or returns
# - classification: Predict direction (up/down)
TARGET_TYPE = 'regression'

# What to predict:
# - 'next_close': Next day's closing price
# - 'next_return': Next day's return percentage
# - 'next_direction': Up (1) or Down (0)
PREDICTION_TARGET = 'next_close'

# For classification: threshold for determining up/down
# E.g., 0.0 means any positive return is 'up'
CLASSIFICATION_THRESHOLD = 0.0

# Look-ahead period: How many periods ahead to predict
# 1 = next period, 375 = next day (assuming 375 minutes in trading day)
LOOKAHEAD_PERIODS = 375  # Predicting next day (1 day = ~375 1-min candles)

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Technical Indicator Periods
INDICATOR_PERIODS = {
    # Moving averages
    'sma_short': 20,      # Short-term simple moving average
    'sma_medium': 50,     # Medium-term simple moving average
    'sma_long': 200,      # Long-term simple moving average
    'ema_short': 12,      # Short-term exponential moving average
    'ema_long': 26,       # Long-term exponential moving average
    
    # Momentum indicators
    'rsi': 14,            # Relative Strength Index period
    'macd_fast': 12,      # MACD fast period
    'macd_slow': 26,      # MACD slow period
    'macd_signal': 9,     # MACD signal period
    
    # Volatility indicators
    'bb_period': 20,      # Bollinger Bands period
    'bb_std': 2,          # Bollinger Bands standard deviation multiplier
    'atr': 14,            # Average True Range period
    
    # Volume indicators
    'volume_sma': 20,     # Volume moving average period
}

# Price action lookback periods for returns and volatility
RETURN_PERIODS = [1, 5, 15, 30, 60]  # Minutes

# Volatility calculation windows
VOLATILITY_WINDOWS = [10, 30, 60]  # Minutes

# ============================================================================
# XGBOOST MODEL HYPERPARAMETERS
# ============================================================================

XGBOOST_PARAMS = {
    # Learning task parameters
    'objective': 'reg:squarederror',  # For regression; use 'binary:logistic' for classification
    'eval_metric': 'rmse',             # Evaluation metric; use 'logloss' for classification
    
    # Booster parameters
    'booster': 'gbtree',               # Tree-based model
    'tree_method': 'hist',             # Histogram-based algorithm (faster)
    
    # Learning control
    'learning_rate': 0.01,             # Step size shrinkage (lower = more robust, slower)
    'n_estimators': 1000,              # Number of boosting rounds
    'max_depth': 6,                    # Maximum tree depth (deeper = more complex)
    
    # Regularization
    'min_child_weight': 3,             # Minimum sum of instance weight in a child
    'gamma': 0.1,                      # Minimum loss reduction for split
    'subsample': 0.8,                  # Fraction of samples for each tree
    'colsample_bytree': 0.8,           # Fraction of features for each tree
    'colsample_bylevel': 0.8,          # Fraction of features for each level
    'reg_alpha': 0.1,                  # L1 regularization
    'reg_lambda': 1.0,                 # L2 regularization
    
    # Other parameters
    'random_state': 42,                # For reproducibility
    'n_jobs': -1,                      # Use all CPU cores
    'verbosity': 1,                    # Print messages
    
    # Early stopping
    'early_stopping_rounds': 50,       # Stop if no improvement for N rounds
}

# ============================================================================
# FEATURE SCALING CONFIGURATION
# ============================================================================

# Whether to scale features
SCALE_FEATURES = True

# Scaling method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
SCALING_METHOD = 'standard'

# Features to exclude from scaling (e.g., binary or categorical features)
EXCLUDE_FROM_SCALING = ['hour', 'day_of_week', 'minute_of_day']

# ============================================================================
# MODEL EVALUATION CONFIGURATION
# ============================================================================

# Metrics to calculate for regression
REGRESSION_METRICS = ['rmse', 'mae', 'mape', 'r2']

# Metrics to calculate for classification
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Whether to generate plots
GENERATE_PLOTS = True

# Plot configurations
PLOT_CONFIG = {
    'figsize': (14, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
    'save_format': 'png',
}

# ============================================================================
# FEATURE IMPORTANCE CONFIGURATION
# ============================================================================

# Number of top features to display in importance plots
TOP_N_FEATURES = 20

# Feature importance type: 'weight', 'gain', or 'cover'
# - weight: Number of times feature is used
# - gain: Average gain when feature is used
# - cover: Average coverage when feature is used
FEATURE_IMPORTANCE_TYPE = 'gain'

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# Whether to save logs to file
SAVE_LOGS = True

# Log file path
LOG_FILE = BASE_DIR / 'model_training.log'

# ============================================================================
# EXTERNAL FEATURES CONFIGURATION (For Future Extensions)
# ============================================================================

# Paths to external feature files (when available)
EXTERNAL_FEATURES = {
    'news_sentiment': None,        # Path to news sentiment CSV
    'fii_cash': None,              # Path to FII cash data CSV
    'fii_options': None,           # Path to FII options data CSV
    'dii_cash': None,              # Path to DII cash data CSV
    'dii_options': None,           # Path to DII options data CSV
}

# How to handle missing values in external features
EXTERNAL_FEATURES_FILL_METHOD = 'ffill'  # Forward fill

# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42
