import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from asyncio import Lock
import os
import asyncpg
import yaml
from statsmodels import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

config_path = r"C:\Users\Oskar\OneDrive\strategytrader\trader\config\config.yaml.txt"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)


# Alpaca credentials (replace with your keys)
API_KEY = config['alpaca']['api_key']
API_SECRET = config['alpaca']['api_secret']
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Strategy Parameters
symbol = "SPY"
quantile = 0.1
price_drop_threshold = 0.25
profit_target = 0.3
max_drop = 0.8
lookback_days = 600
drop_lookahead = 50
cash_allocation = config['strategies']['tail_reaper']['allocation_percentage']  # Use 90% of available cash

current_position = None
trade_entry_price = None
trade_entry_day = None

async def fetch_recent_data(symbol, days):
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            limit=days * 390  # Approx. 390 minutes per trading day
        )
        bars = data_client.get_stock_bars(request_params).df
        return bars[bars['symbol'] == symbol]
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_threshold(log_returns):
    try:
        params = stats.t.fit(log_returns)
        return stats.t.ppf(quantile, *params)
    except Exception as e:
        logging.error(f"Error calculating threshold: {e}")
        return None

async def enter_trade(latest_price):
    global current_position, trade_entry_price, trade_entry_day
    account = trading_client.get_account()
    cash = float(account.cash) * cash_allocation
    quantity = int(cash // latest_price)
    if quantity > 0:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order_data=order)
        current_position = "long"
        trade_entry_price = latest_price
        trade_entry_day = datetime.now()
        logging.info(f"Entered position: {quantity} shares at ${latest_price:.2f}")

async def exit_trade(latest_price):
    global current_position
    positions = trading_client.get_all_positions()
    for position in positions:
        if position.symbol == symbol:
            quantity = int(position.qty)
            order = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order_data=order)
            current_position = None
            logging.info(f"Exited position: {quantity} shares at ${latest_price:.2f}")
            return

async def trading_logic():
    global current_position, trade_entry_price, trade_entry_day
    while True:
        try:
            # Fetch latest data
            data = await fetch_recent_data(symbol, days=1)
            if data.empty:
                await asyncio.sleep(60)
                continue

            spy_prices = data['close'].values
            log_returns = np.log(spy_prices[1:] / spy_prices[:-1])
            threshold = calculate_threshold(log_returns)
            if threshold is None:
                await asyncio.sleep(60)
                continue

            recent_high = max(spy_prices[-lookback_days:])
            current_price = spy_prices[-1]
            price_drop = (recent_high - current_price) / recent_high

            # Entry Condition
            if current_position is None and price_drop >= price_drop_threshold:
                await enter_trade(current_price)

            # Exit Conditions
            if current_position == "long":
                target_price = trade_entry_price * (1 + profit_target)
                if current_price >= target_price:
                    await exit_trade(current_price)
                    continue

                days_since_entry = (datetime.now() - trade_entry_day).days
                drop_threshold_price = trade_entry_price * (1 - max_drop)
                if days_since_entry <= drop_lookahead and current_price <= drop_threshold_price:
                    await exit_trade(current_price)

            await asyncio.sleep(60)  # Check every minute

        except Exception as e:
            logging.error(f"Error in trading logic: {e}")

async def main():
    await trading_logic()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Strategy stopped by user.")
