#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import numpy as np
import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config import data_client

def get_historical_data(ticker, years=5):
    print(f"\n--- Task 1: Fetching {years} years of historical data for {ticker} ---")

    end_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=20)
    start_date = end_date - datetime.timedelta(days=years * 365)

    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )

    df_bars = data_client.get_stock_bars(request_params).df
    df_bars = df_bars.xs(ticker, level=0) 
    df_bars.index = pd.to_datetime(df_bars.index).tz_localize(None) 
    return df_bars

def engineering_features(df):
    print("--- Task 2: Engineering Technical Indicators ---")
    df = df.copy()

    # Trend
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Momentum
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['WILLR_14'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low + 1e-10))

    # Volatility
    rolling_std = df['close'].rolling(window=20).std()
    df['BBU_20'] = df['SMA_20'] + (rolling_std * 2)
    df['BBL_20'] = df['SMA_20'] - (rolling_std * 2)

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()

    # Custom Metrics
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['rolling_mean_5'] = df['close'].rolling(5).mean()
    df['rolling_std_5'] = df['close'].rolling(5).std()

    # Target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.dropna(inplace=True)
    return df

