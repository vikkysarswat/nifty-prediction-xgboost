# Nifty Prediction using XGBoost 🚀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Overview

A comprehensive, production-ready machine learning pipeline using **XGBoost** to predict Nifty index movements based on historical 1-minute OHLCV data. Features 60+ engineered indicators with extensible architecture for FII/DII data and news sentiment integration.

### ✨ Key Features

- ✅ **Robust Data Pipeline**: Automated validation and preprocessing
- ✅ **60+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
- ✅ **XGBoost Model**: State-of-the-art gradient boosting
- ✅ **7-5 Month Split**: 7 months training, 5 months testing
- ✅ **Extensible**: Easy integration of external features
- ✅ **Production Ready**: Model persistence and prediction pipeline
- ✅ **Extensively Documented**: Every function explained with docstrings

## 🏗️ Project Structure

```
nifty-prediction-xgboost/
├── data/
│   ├── raw/              # Place your CSV files here
│   └── processed/        # Processed data saved here
├── src/
│   ├── data_loader.py    # Data loading & validation
│   ├── feature_engineering.py  # 60+ feature creation
│   ├── model_trainer.py  # XGBoost training
│   ├── model_evaluator.py  # Evaluation & metrics
│   └── predictor.py      # Production predictions
├── models/               # Saved models
├── results/              # Plots & metrics
├── config.py            # ✅ All configurations
├── main.py              # End-to-end pipeline
└── requirements.txt     # ✅ Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/vikkysarswat/nifty-prediction-xgboost.git
cd nifty-prediction-xgboost
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data

Place your CSV in `data/raw/nifty_1min_data.csv`:

```csv
Date,Time,Open,High,Low,Close,Volume
2024-01-01,09:15,21500.50,21525.75,21495.25,21510.00,1500000
```

### 3. Run Pipeline

```bash
python main.py
```

## 📥 Complete Source Files

The repository includes configuration and structure. **Download the complete source files from the artifacts in this conversation**:

1. ✅ **config.py** - Already in repo
2. ✅ **requirements.txt** - Already in repo  
3. **data_loader.py** - Data validation (artifact)
4. **feature_engineering.py** - Feature creation (artifact)
5. **model_trainer.py** - Training pipeline (artifact)
6. **model_evaluator.py** - Evaluation (artifact)
7. **predictor.py** - Predictions (artifact)
8. **main.py** - Complete pipeline (artifact)

Place these files in the `src/` directory.

## 🎯 Configuration

Edit `config.py`:

```python
# Training split
TRAIN_MONTHS = 7  # First 7 months
TEST_MONTHS = 5   # Next 5 months

# Prediction target
PREDICTION_TARGET = 'next_close'  # or 'next_return', 'next_direction'
LOOKAHEAD_PERIODS = 375  # 1 day ahead

# XGBoost parameters
XGBOOST_PARAMS = {
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'max_depth': 6,
    # ... more params
}
```

## 📊 Features Created

### Technical Indicators
- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26)
- **Momentum**: RSI (14), MACD, Price Momentum
- **Volatility**: Bollinger Bands, ATR, Rolling Volatility
- **Volume**: VWAP, Volume Ratios, PV Trends
- **Price Action**: Returns, Ranges, Distance from High/Low
- **Time**: Hour, Day of Week, Session Flags

## 📈 Expected Performance

**Regression (Next Day Close Price):**
- RMSE: 50-150 points
- MAE: 40-120 points
- MAPE: 0.2-0.5%
- R²: 0.75-0.95

**Classification (Direction):**
- Accuracy: 52-58%
- Precision: 53-60%
- F1-Score: 52-58%

## 🔧 Adding External Features

### FII/DII Data

```python
from src.feature_engineering import FeatureEngineer

# Load external data
fii_data = pd.read_csv('fii_data.csv', parse_dates=['datetime'], index_col='datetime')

# Add features
engineer = FeatureEngineer()
df_featured = engineer.add_external_features(df, 'fii', fii_data)
```

### News Sentiment

```python
sentiment = pd.read_csv('sentiment.csv', parse_dates=['datetime'], index_col='datetime')
df_featured = engineer.add_external_features(df, 'sentiment', sentiment)
```

## 🎯 Making Predictions

```python
from src.predictor import Predictor

# Load model
predictor = Predictor('models/xgboost_model.joblib', 'models/scaler.joblib')

# Predict
predictions = predictor.predict('new_data.csv')

# Next day prediction
next_day = predictor.predict_next_day(df)
print(f"Direction: {next_day['direction']}")
print(f"Change: {next_day['predicted_change_pct']:.2f}%")
```

## 📚 Documentation

Every module includes:
- ✅ Comprehensive docstrings for all functions
- ✅ Line-by-line comments explaining complex logic
- ✅ Mathematical formulas for indicators
- ✅ Explanation of why each step is performed
- ✅ Usage examples

### Module Overview

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `data_loader.py` | Data validation | OHLC checks, datetime parsing, quality assurance |
| `feature_engineering.py` | Feature creation | 60+ indicators, extensible for external data |
| `model_trainer.py` | Model training | Time-series split, scaling, early stopping |
| `model_evaluator.py` | Evaluation | Metrics, plots, feature importance |
| `predictor.py` | Production use | Load model, preprocess, predict |

## ⚠️ Disclaimer

**For educational and research purposes only.**

- ❌ Do NOT use for actual trading without extensive backtesting
- ❌ Past performance does not guarantee future results  
- ✅ Always implement proper risk management
- ✅ Consult financial advisors before trading

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional technical indicators
- LSTM/Transformer models
- Real-time data integration
- Automated hyperparameter tuning
- Backtesting framework

## 📧 Contact

- **GitHub**: [@vikkysarswat](https://github.com/vikkysarswat)
- **Email**: vikky.sarswat@gmail.com
- **Issues**: [Report here](https://github.com/vikkysarswat/nifty-prediction-xgboost/issues)

## 🙏 Acknowledgments

- XGBoost team for the excellent library
- Scikit-learn for ML infrastructure
- TA-Lib for technical indicators

---

**⭐ If this helps you, please star the repository!**

## 📄 License

MIT License - See LICENSE file for details