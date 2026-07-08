#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import warnings
import webbrowser
from backtesting import Backtest, Strategy, set_bokeh_output
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import TICKER, FEATURE_COLS, data_client, trading_client
from data_features import engineering_features

class MLSingleStockStrategy(Strategy):
    def init(self):
        self.signal = self.I(lambda: self.data.signal)

    def next(self):
        if self.signal[-1] == 1 and not self.position:
            self.buy()
        elif self.signal[-1] == 0 and self.position:
            self.position.close()

def run_backtest(df, split_idx):
    print("--- Task 5 & 6: Backtesting Out-Of-Sample Data & Metrics ---")
    set_bokeh_output(notebook=False)

    df_test = df.iloc[split_idx:].copy()
    df_test.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)

    bt = Backtest(df_test, MLSingleStockStrategy, cash=100000, commission=.000)
    stats = bt.run()

    total_return = stats.get('Return [%]', 0.0)
    bh_return = stats.get('Buy & Hold Return [%]', 0.0)
    cagr = stats.get('Return (Ann.) [%]', 0.0)
    vol = stats.get('Volatility (Ann.) [%]', 0.0)
    sharpe = stats.get('Sharpe Ratio', 0.0)
    sortino = stats.get('Sortino Ratio', 0.0)
    max_dd = stats.get('Max Drawdown [%]', stats.get('Max. Drawdown [%]', 0.0))
    win_rate = stats.get('Win Rate [%]', stats.get('Win. Rate [%]', 0.0))

    print("\n================ SYSTEM PERFORMANCE METRICS ================")
    print(f"Total Return (ML Strategy): {total_return:.2f}%")
    print(f"Buy & Hold Return:          {bh_return:.2f}%")
    print(f"CAGR:                       {cagr:.2f}%")
    print(f"Volatility (Ann.):          {vol:.2f}%")
    print(f"Sharpe Ratio:               {sharpe:.2f}")
    print(f"Sortino Ratio:              {sortino:.2f}")
    print(f"Max Drawdown:               {max_dd:.2f}%")
    print(f"Win Rate:                   {win_rate:.2f}%")
    print("=============================================================\n")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*Superimposed OHLC plot.*")
        bt.plot(filename='backtest_results.html', open_browser=False)

    print("Backtest dashboard layout successfully rendered to 'backtest_results.html'.")
    webbrowser.open('backtest_results.html')

def execute_paper_trade_signal(scaler, pca, model):
    print("--- Task 7: Querying Real-Time Signal for Alpaca Paper Trading ---")

    account = trading_client.get_account()
    if account.status != 'ACTIVE':
        print("Error: Alpaca account is not active.")
        return

    print(f"Connected to Paper Trading Account: {account.account_number}")

    end_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=20)
    start_date = end_date - datetime.timedelta(days=50)

    request_params = StockBarsRequest(symbol_or_symbols=TICKER, timeframe=TimeFrame.Day, start=start_date, end=end_date)
    recent_df = data_client.get_stock_bars(request_params).df.xs(TICKER, level=0)
    recent_df.index = pd.to_datetime(recent_df.index).tz_localize(None)

    live_df = engineering_features(recent_df)
    last_row = live_df.iloc[[-1]]

    X_live = last_row[FEATURE_COLS]
    X_live_scaled = scaler.transform(X_live)
    X_live_pca = pca.transform(X_live_scaled)

    prob = model.predict_proba(X_live_pca)[0][1]
    signal = "LONG" if prob > 0.6 else "FLAT"

    print(f"\n[LOG] Date Analyzed: {last_row.index[0].date()}")
    print(f"[LOG] Target Signal Probability: {prob:.4f} -> Decision: {signal}")

    positions = trading_client.get_all_positions()
    has_position = any(p.symbol == TICKER for p in positions)

    if signal == "LONG" and not has_position:
        print(f"[ORDER] Sending Market BUY order for {TICKER}...")
        order = trading_client.submit_order(order_data=MarketOrderRequest(
            symbol=TICKER, qty=10, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
        ))
        print(f"[SUCCESS] Order Executed. ID: {order.id}")
    elif signal == "FLAT" and has_position:
        print(f"[ORDER] Signal turned FLAT. Liquidating {TICKER} positions...")
        trading_client.close_position(TICKER)
        print(f"[SUCCESS] Liquidation message sent.")
    else:
        print("[LOG] Portfolio aligns with current ML signal. No trades required.")

