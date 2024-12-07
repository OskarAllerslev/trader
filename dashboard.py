
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date
from alpaca.trading.client import TradingClient
import os
import requests
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

# Alpaca API credentials (use your actual keys)
API_KEY = 'PKTRHQWHETKU0MRD2119'
API_SECRET = 'OYbFiGWVC9KEQw5KalyalLVl8b4xvMxZghhpvXpd'

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

symbol_spy = "SPY"
symbol_trade = "SPXL"
ENTRY_THRESHOLD = 0.9  # The threshold used in the regime switching strategy

# Page config
st.set_page_config(page_title="Trading Dashboard", layout="wide")


# Style adjustments (simple)
st.markdown("""
<style>
    .main { 
        background-color: #f8f9fa; 
    }
    .stMarkdown h1, h2, h3, h4 {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        color: #222;
    }
    .metric-label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def fetch_account_info():
    account = trading_client.get_account()
    balance = float(account.cash)
    buying_power = float(account.buying_power)
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    portfolio_value = float(account.portfolio_value)
    return balance, buying_power, equity, last_equity, portfolio_value

def fetch_portfolio_history_from_api():
    url = "https://paper-api.alpaca.markets/v2/account/portfolio/history?intraday_reporting=market_hours&pnl_reset=per_day"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.warning(f"Unable to fetch portfolio history. Status code: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    # data should have keys: timestamp, equity, profit_loss, profit_loss_pct, base_value
    if "timestamp" not in data or "equity" not in data:
        st.warning("Unexpected portfolio history format.")
        return pd.DataFrame()

    df = pd.DataFrame({
        "timestamp": data["timestamp"],
        "equity": data["equity"],
        "profit_loss": data["profit_loss"],
        "profit_loss_pct": data["profit_loss_pct"]
    })
    # Convert timestamp (Unix) to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    df.set_index("timestamp", inplace=True)
    return df

def fetch_last_p0_data():
    if os.path.exists("last_p0_data.csv"):
        df = pd.read_csv("last_p0_data.csv")
        if not df.empty:
            row = df.iloc[-1]
            timestamp_str = row["timestamp"]
            last_p0_val = float(row["last_p0"])
            return timestamp_str, last_p0_val
    return None, None

from alpaca.data.requests import StockLatestTradeRequest

def fetch_current_positions():
    positions = trading_client.get_all_positions()
    data = []
    for pos in positions:
        if pos.symbol == symbol_trade:
            qty = float(pos.qty)
            entry_price = float(pos.avg_entry_price)

            # Fetch latest trade from Alpaca (most recent price) using the correct request object
            try:
                latest_trade_req = StockLatestTradeRequest(symbol_or_symbols=[symbol_trade])
                latest_trade = data_client.get_stock_latest_trade(latest_trade_req)
                # latest_trade is a dict keyed by symbol, so access the data by symbol
                current_price = latest_trade[symbol_trade].price
            except Exception as e:
                st.warning(f"Could not fetch current price from Alpaca for {symbol_trade}: {e}")
                current_price = entry_price

            unrealized_pnl = (current_price - entry_price) * qty
            data.append([pos.symbol, qty, entry_price, current_price, unrealized_pnl])

    if data:
        position_df = pd.DataFrame(data, columns=["Symbol", "Quantity", "Entry Price", "Current Price", "Unrealized PnL"])
        position_df["Quantity"] = pd.to_numeric(position_df["Quantity"], errors="coerce")
        position_df["Entry Price"] = pd.to_numeric(position_df["Entry Price"], errors="coerce")
        position_df["Current Price"] = pd.to_numeric(position_df["Current Price"], errors="coerce")
        position_df["Unrealized PnL"] = pd.to_numeric(position_df["Unrealized PnL"], errors="coerce")
    else:
        position_df = pd.DataFrame(columns=["Symbol", "Quantity", "Entry Price", "Current Price", "Unrealized PnL"])
    return position_df


def fetch_historical_data(symbol, start_dt, end_dt):
    data = yf.download(symbol, start=start_dt, end=end_dt, interval="1d")
    if 'Adj Close' in data.columns:
        data['Log Return'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        data.dropna(inplace=True)
    return data

st.markdown("<h1 style='text-align: center;'>Trading Dashboard</h1>", unsafe_allow_html=True)
tabs = st.tabs(["Account Overview", "Regime Switching trader", "Beta Neutral trader"])

