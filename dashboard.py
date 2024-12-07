
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
