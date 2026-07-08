#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

# API Credentials
ALPACA_API_KEY = "Put personal key here"
ALPACA_SECRET_KEY = "Put personal secret key here"

# Global User Input
TICKER = input("Enter ticker to analyze/trade (e.g., AAPL, MSFT, SPY): ").upper().strip()

# Initialize Clients
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Shared constants
FEATURE_COLS = [
    'SMA_20', 'EMA_20', 'RSI_14', 'WILLR_14', 'BBU_20', 
    'BBL_20', 'ATR_14', 'log_return', 'rolling_mean_5', 'rolling_std_5'
]

