# FINM-25000-HW-3-Schulz
Built live paper trading system in python to trade on Alpaca. Uses random forest machine learning model and PCA (on several technical indicators) to invest intelligently in a stock of the users choosing. Provides pop-up of charts representing backtest results of the strategy being applied to historical data, as well as performance metrics table. 

## Features

* **Streamlit backtester:** Connects to Alpaca via API, pulls historical stock data, and backtests the random forest trading strategy.
* **Graphical Representation:** Creates charts graphing backtesting data and performance metrics.
* **Real-time paper trading:** Allows user to select a stock and invest real-time in Alpaca's paper trading mode.
  
```bash
pip install alpaca-py pandas numpy scikit-learn backtesting matplotlib
