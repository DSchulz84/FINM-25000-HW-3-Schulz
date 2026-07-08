#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from config import TICKER
from data_features import get_historical_data, engineering_features
from ml_pipeline import train_ml_pipeline
from backtesting_trades import run_backtest, execute_paper_trade_signal

if __name__ == "__main__":
    # 1. Fetch data
    raw_data = get_historical_data(TICKER, years=5)

    # 2. Extract Technical indicators
    processed_data = engineering_features(raw_data)

    # 3. Scale, PCA, & Fit ML Model
    ml_data, scaler, pca, model, test_split_idx = train_ml_pipeline(processed_data)

    # 4. Out-Of-Sample validation testing
    run_backtest(ml_data, test_split_idx)

    # 5. Live Paper check & potential trade allocation
    execute_paper_trade_signal(scaler, pca, model)

