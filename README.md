# CISC 4900 - NFL Game Spread Predictor

## Overview
A machine learning project that predicts NFL game outcomes and point spreads. The project explores multiple supervised learning approaches and feature engineering strategies to build an accurate prediction model.

## Project Pipeline
1. **Data Collection** - Retrieve NFL game data using the `nfl_data_py` library
2. **Preprocessing** - Clean and transform raw data for analysis
3. **Feature Engineering** - Construct multiple feature sets for model comparison
4. **Model Training** - Train and evaluate models including Random Forest, XGBoost, and others
5. **Evaluation & Analysis** - Hyperparameter tuning, SHAP analysis, and model comparison against a baseline

## Project Structure
```
├── data/
│   ├── raw/                   # Original untouched data
│   └── processed/             # Cleaned and transformed data
├── notebooks/                 # Jupyter notebooks for exploration
├── src/
│   ├── data_retrieval.py      # Data collection scripts
│   ├── preprocessing.py       # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature set creation
│   ├── models.py              # Model training and tuning
│   └── evaluation.py          # Metrics, SHAP, and comparison
├── results/                   # Model outputs and visualizations
├── requirements.txt           # Python dependencies
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```
