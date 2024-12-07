
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

# ---------------------- ACCOUNT OVERVIEW ----------------------
with tabs[0]:
    st.markdown("## Account Overview")
    balance, buying_power, equity, last_equity, portfolio_value = fetch_account_info()

    st.write("General account information and portfolio overview:")
    colA, colB, colC, colD, colE = st.columns(5)
    colA.metric(label="**Account Balance (USD)**", value=f"${balance:,.2f}")
    colB.metric(label="**Buying Power (USD)**", value=f"${buying_power:,.2f}")
    colC.metric(label="**Equity (USD)**", value=f"${equity:,.2f}")
    delta_equity = equity - last_equity
    colD.metric(label="**Last Equity (USD)**", value=f"${last_equity:,.2f}", delta=f"${delta_equity:,.2f}")
    colE.metric(label="**Portfolio Value (USD)**", value=f"${portfolio_value:,.2f}")

    st.markdown("---")
    st.markdown("**Historical Account Performance**")

    ph_df = fetch_portfolio_history_from_api()
    if not ph_df.empty:
        # Calculate cumulative profit_loss and cumulative profit_loss_pct
        ph_df['cumulative_profit_loss'] = ph_df['profit_loss'].cumsum()
        ph_df['cumulative_profit_loss_pct'] = ph_df['profit_loss_pct'].cumsum()

        # Create DataFrame for profit_loss and cumulative_profit_loss
        pl_data = ph_df[['profit_loss', 'cumulative_profit_loss']]

        # Use columns to display three charts side-by-side
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Equity (USD)**")
            st.line_chart(ph_df['equity'], use_container_width=True)

        with col2:
            st.markdown("**Profit/Loss (USD)**")
            st.line_chart(pl_data, use_container_width=True)

        with col3:
            st.markdown("**Cumulative Profit/Loss Percentage**")
            st.line_chart(ph_df['cumulative_profit_loss_pct'], use_container_width=True)

    else:
        st.write("No historical equity data available.")




# ---------------------- REGIME SWITCHING TRADER ----------------------
with tabs[1]:
    st.markdown("## Regime Switching trader")
    st.write(f"**Entry/Exit Threshold:** {ENTRY_THRESHOLD:.2f}")
    st.write("This threshold indicates the probability above which the strategy enters the market, and below which it exits.")

    timestamp_str, last_p0_val = fetch_last_p0_data()
    if last_p0_val is not None:
        st.markdown("**Markov Model Probability P(X(t+1))**")
        col1, col2 = st.columns(2)
        col1.metric(label="P(X(t+1)) of Positive Regime", value=f"{last_p0_val:.2%}")
        col2.markdown(f"**Data Timestamp:** {timestamp_str}")
    else:
        st.write("No probability data available yet. Ensure msv10.py is running and updating `last_p0_data.csv`.")

    st.markdown("---")
    st.markdown("**Current Position**")
    position_df = fetch_current_positions()
    if position_df.empty:
        st.write("No current positions for this strategy.")
    else:
        st.dataframe(position_df.style.format({
            "Quantity": "{:,.0f}",
            "Entry Price": "{:,.2f}",
            "Current Price": "{:,.2f}",
            "Unrealized PnL": "{:,.2f}"
        }), use_container_width=True)

    st.markdown("---")
    st.markdown("**Historical Data & Log Returns**")

    today = date.today()
    default_start = today.replace(year=today.year - 1)  # default to 1 year ago
    chosen_start = st.date_input("Start Date:", default_start)
    chosen_end = st.date_input("End Date:", today)

    if chosen_start > chosen_end:
        st.error("Start date must be before the end date.")
    else:
        historical_data = fetch_historical_data(symbol_spy, chosen_start.strftime("%Y-%m-%d"), chosen_end.strftime("%Y-%m-%d"))

        if not historical_data.empty and 'Adj Close' in historical_data.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Historical Prices (Adj Close)**")
                st.line_chart(historical_data['Adj Close'], use_container_width=True)
            with col2:
                st.markdown("**Log Returns**")
                st.line_chart(historical_data['Log Return'], use_container_width=True)
        else:
            st.write("No historical data available for the given date range.")

    if st.button("Refresh Data"):
        st.experimental_rerun()


# ---------------------- BETA NEUTRAL TRADER ----------------------
with tabs[2]:
    st.markdown("## Beta Neutral trader")
    st.write("This is a placeholder for your beta neutral strategy. You can add relevant metrics, plots, and logic here.")
