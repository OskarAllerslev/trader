# Import necessary libraries
import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
import pytz
from asyncio import Lock
import os
import csv

# Import Alpaca Market Data API modules
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Alpaca API credentials
API_KEY = 'PKTRHQWHETKU0MRD2119'
API_SECRET = 'OYbFiGWVC9KEQw5KalyalLVl8b4xvMxZghhpvXpd'

# Initialize Alpaca trading client
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Global variables
symbol_spy = "SPY"          # Symbol to fit the model on
symbol_trade = "SPXL"       # Symbol to trade
log_returns_spy = []
fitted_model = None
positive_regime = 0         # Set positive_regime to 0 as per your request
last_p0 = None              # Stores the last smoothed probability of being in the positive regime
last_p0_timestamp = None    # Stores the timestamp when last_p0 was updated
current_position = None     # Tracks whether we are "long" or "None"
#ENTRY_THRESHOLD = 0.98878852547  # Optimized probability threshold
ENTRY_THRESHOLD = 0.9      # Adjust the threshold as needed
TRADE_PERCENTAGE = 0.95     # Percentage of account balance to trade
eastern = pytz.timezone('US/Eastern')

def is_market_open():
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception as e:
        logging.error(f"Error checking market status: {e}")
        return False

async def initialize_historical_data():
    global fitted_model, log_returns_spy, positive_regime, last_p0, last_p0_timestamp

    logging.info("Fetching historical SPY data for initialization.")

    try:
        # Fetch historical data from yfinance
        start_date = '1990-01-01'
        end_date = datetime.now(pytz.UTC).strftime('%Y-%m-%d')  # Include today's date
        data_yf = yf.download(
            symbol_spy,
            start=start_date,
            end=end_date,
            interval='1d',
            group_by='ticker',
            auto_adjust=False
        )
        data_yf.dropna(inplace=True)

        # If the columns are MultiIndex, flatten them
        if isinstance(data_yf.columns, pd.MultiIndex):
            data_yf.columns = data_yf.columns.get_level_values(-1)
            logging.info("Flattened data_yf.columns")

        # Adjust the index to represent the trading day
        data_yf.index = data_yf.index - pd.Timedelta(days=1)

        # Ensure the index is in UTC
        if data_yf.index.tz is None:
            data_yf.index = data_yf.index.tz_localize('UTC')
        else:
            data_yf.index = data_yf.index.tz_convert('UTC')

        # Log date range of yfinance data
        logging.info(f"YFinance Data Date Range: {data_yf.index.min()} to {data_yf.index.max()}")

        # Fetch recent data from Alpaca
        recent_start_date = datetime.now(pytz.UTC) - timedelta(days=5)
        recent_end_date = datetime.now(pytz.UTC)  

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol_spy,
            timeframe=TimeFrame.Day,
            start=recent_start_date,
            end=recent_end_date,
            feed='iex'
        )

        bars = data_client.get_stock_bars(request_params).df

        if bars.empty:
            logging.warning("No recent data fetched from Alpaca API.")
            data_alpaca = pd.DataFrame()
        else:
            bars = bars.reset_index()
            logging.info(f"bars.columns after reset_index: {bars.columns}")

            # Filter for the desired symbol
            data_alpaca = bars[bars['symbol'] == symbol_spy].copy()

            # Set 'timestamp' as the index
            data_alpaca.set_index('timestamp', inplace=True)

            # Drop 'symbol' column
            data_alpaca.drop(columns=['symbol'], inplace=True)

            # Ensure the index is in UTC
            if data_alpaca.index.tz is None:
                data_alpaca.index = data_alpaca.index.tz_localize('UTC')
            else:
                data_alpaca.index = data_alpaca.index.tz_convert('UTC')

            # Log date range of Alpaca data
            logging.info(f"Alpaca Data Date Range: {data_alpaca.index.min()} to {data_alpaca.index.max()}")

            # Keep only 'close' column and rename to 'Adj Close'
            data_alpaca = data_alpaca[['close']]
            data_alpaca.rename(columns={'close': 'Adj Close'}, inplace=True)
            data_alpaca.sort_index(inplace=True)

            # Exclude overlapping dates using date components
            data_alpaca_dates = data_alpaca.index.date
            data_yf_dates = data_yf.index.date
            data_alpaca = data_alpaca[~np.isin(data_alpaca_dates, data_yf_dates)]

        # Combine data_yf and data_alpaca
        data_combined = pd.concat([data_yf, data_alpaca])
        data_combined.sort_index(inplace=True)
     
         # Remove duplicates
        data_combined = data_combined[~data_combined.index.duplicated(keep='first')]

        # Log date range of combined data
        logging.info(f"Combined Data Date Range: {data_combined.index.min()} to {data_combined.index.max()}")

        # Verify 'Adj Close' values
        if data_combined['Adj Close'].isnull().any():
            null_rows = data_combined[data_combined['Adj Close'].isnull()]
            logging.error(f"Null values found in 'Adj Close' at:\n{null_rows.index}")
            return
     
        # Calculate log returns
        data_combined['Log Return'] = np.log(
            data_combined['Adj Close'] / data_combined['Adj Close'].shift(1)
        )
        #data_combined.dropna(inplace=True)
        # Drop only the first row of the DataFrame
        data_combined = data_combined.iloc[1:]

        # Adjust index to represent market close time in US/Eastern
        data_combined.index = data_combined.index.tz_convert(eastern)
        data_combined.index = data_combined.index.map(lambda dt: dt.replace(hour=16, minute=0, second=0, microsecond=0))

        logging.info(f"Data with log returns (last 5 rows):\n{data_combined[['Adj Close', 'Log Return']].tail()}")

        # Check if data_combined is empty after dropping NaNs
        if data_combined.empty:
            logging.error("Combined data is empty after dropping NaNs. Cannot fit the model.")
            return

        log_returns_spy = data_combined['Log Return'].values
        # Check if log_returns_spy is empty
        if len(log_returns_spy) == 0:
            logging.error("Log returns array is empty. Cannot fit the model.")
            return

        fitted_model = fit_markov_model(log_returns_spy)

        # Positive regime remains as Regime 0
        positive_regime = 0
        logging.info(f"Positive regime identified: Regime {positive_regime}")

        # Extract smoothed probabilities
        smoothed_probs = fitted_model.smoothed_marginal_probabilities[:, positive_regime]

        # Print last five probabilities with corresponding dates
        last_five_dates = data_combined.index[-5:]
        last_five_probs = smoothed_probs[-5:]
        for date, prob in zip(last_five_dates, last_five_probs):
            logging.info(f"Smoothed probability on {date.strftime('%Y-%m-%d %H:%M:%S %Z')}: {prob:.6f}")

        # Update last_p0 and last_p0_timestamp
 
        last_p0 = smoothed_probs[-2]
        last_p0_timestamp = last_five_dates[-2]
        logging.info(
            f"Initial probability of positive regime: {last_p0:.6f} at {last_p0_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        raise e

def fit_markov_model(log_returns):
    logging.info("Fitting Markov Switching Model on SPY data.")
    model = MarkovRegression(
        log_returns,
        k_regimes=2,
        trend='c',
        switching_variance=True
    )
    fitted_model = model.fit(em_iter=1000, cov_type='robust')
    logging.info("Model fitted successfully.")
    return fitted_model

async def check_and_cancel_conflicting_orders(direction):
    """
    Check and cancel any open orders that conflict with the given direction.
    :param direction: "BUY" or "SELL" indicating the desired action.
    """
    try:
        open_orders_request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol_trade],
            nested=False
        )
        open_orders = trading_client.get_orders(filter=open_orders_request)

        for order in open_orders:
            if order.side != direction.upper():
                trading_client.cancel_order_by_id(order.id)  # Cancel the specific order
                logging.info(f"Cancelled conflicting order: {order.id} ({order.side})")
    except Exception as e:
        logging.error(f"Error checking and cancelling conflicting orders: {e}")

async def enter_position():
    global current_position

    if not is_market_open():
        logging.info("Market is closed. Cannot enter position.")
        return

    try:
        # Check and cancel conflicting orders
        await check_and_cancel_conflicting_orders("BUY")

        # Fetch account information
        account = trading_client.get_account()
        current_buying_power = float(account.buying_power)  # Use buying power appropriate for overnight positions

        # Get latest price of SPXL
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol_trade,
            timeframe=TimeFrame.Minute,
            limit=1,
            feed="iex"
        )
        bars = data_client.get_stock_bars(request_params).df

        if not bars.empty:
            latest_price = bars['close'].iloc[-1]
        else:
            logging.error(f"Could not fetch latest price for {symbol_trade}.")
            return

        # Calculate position size
        position_value = current_buying_power * TRADE_PERCENTAGE

        # Calculate number of shares to buy
        max_shares = int(position_value / latest_price)

        if max_shares <= 0:
            logging.info("Not enough buying power to enter position.")
            return

        # Check if already fully invested
        current_positions = trading_client.get_all_positions()
        for position in current_positions:
            if position.symbol == symbol_trade:
                current_quantity = int(float(position.qty))
                if current_quantity * latest_price >= position_value * 0.95:
                    logging.info("Already fully invested. No new trades executed.")
                    current_position = "long"
                    return

        # Place a buy order
        order_data = MarketOrderRequest(
            symbol=symbol_trade,
            qty=max_shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order_data=order_data)
        logging.info(f"Entered long position: Bought {max_shares} shares of {symbol_trade}.")
        current_position = "long"

    except Exception as e:
        logging.error(f"Error entering position: {e}")

async def exit_position():
    global current_position

    if not is_market_open():
        logging.info("Market is closed. Cannot exit position.")
        return

    try:
        # Check and cancel conflicting orders
        await check_and_cancel_conflicting_orders("SELL")

        # Fetch open positions
        current_positions = trading_client.get_all_positions()
        for position in current_positions:
            if position.symbol == symbol_trade:
                current_quantity = int(float(position.qty))

                if current_quantity > 0:
                    # Place a sell order to close the position
                    order_data = MarketOrderRequest(
                        symbol=symbol_trade,
                        qty=current_quantity,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    trading_client.submit_order(order_data=order_data)
                    logging.info(f"Closed position: Sold {current_quantity} shares of {symbol_trade}.")
                    current_position = None
                return
        logging.info("No open position to close.")
        current_position = None
    except Exception as e:
        logging.error(f"Error closing position: {e}")

async def refit_markov_model():
    global fitted_model, log_returns_spy, positive_regime, last_p0, last_p0_timestamp

    while True:
        # Wait until the next hour to update
        now = datetime.now(eastern)
        next_hour = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
        sleep_duration = (next_hour - now).total_seconds()
        logging.info(f"Sleeping for {sleep_duration / 60:.2f} minutes before refitting the model.")
        await asyncio.sleep(sleep_duration)

        try:
            logging.info("Refitting Markov Switching Model with latest SPY data.")

            # Fetch historical data from yfinance
            start_date = '1990-01-01'
            end_date = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
            data_yf = yf.download(symbol_spy, start=start_date, end=end_date, interval='1d')
            data_yf.dropna(inplace=True)

            # Ensure the index is in UTC
            if data_yf.index.tz is None:
                data_yf.index = data_yf.index.tz_localize('UTC')
            else:
                data_yf.index = data_yf.index.tz_convert('UTC')

            # Log date range of yfinance data
            logging.info(f"YFinance Data Date Range: {data_yf.index.min()} to {data_yf.index.max()}")

            # Fetch recent data from Alpaca (same logic as in initialize_historical_data)
            recent_start_date = datetime.now(pytz.UTC) - timedelta(days=5)
            recent_end_date = datetime.now(pytz.UTC) 

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol_spy,
                timeframe=TimeFrame.Day,
                start=recent_start_date,
                end=recent_end_date,
                feed='iex'
            )

            bars = data_client.get_stock_bars(request_params).df

            if bars.empty:
                logging.warning("No recent data fetched from Alpaca API.")
                data_alpaca = pd.DataFrame()
            else:
                bars = bars.reset_index()
                logging.info(f"bars.columns after reset_index: {bars.columns}")

                # Filter for the desired symbol
                data_alpaca = bars[bars['symbol'] == symbol_spy].copy()

                # Set 'timestamp' as the index
                data_alpaca.set_index('timestamp', inplace=True)

                # Drop 'symbol' column
                data_alpaca.drop(columns=['symbol'], inplace=True)

                # Ensure the index is in UTC
                if data_alpaca.index.tz is None:
                    data_alpaca.index = data_alpaca.index.tz_localize('UTC')
                else:
                    data_alpaca.index = data_alpaca.index.tz_convert('UTC')

                # Log date range of Alpaca data
                logging.info(f"Alpaca Data Date Range: {data_alpaca.index.min()} to {data_alpaca.index.max()}")

                # Keep only 'close' column and rename to 'Adj Close'
                data_alpaca = data_alpaca[['close']]
                data_alpaca.rename(columns={'close': 'Adj Close'}, inplace=True)
                data_alpaca.sort_index(inplace=True)

                # Exclude overlapping dates using date components
                data_alpaca_dates = data_alpaca.index.date
                data_yf_dates = data_yf.index.date
                data_alpaca = data_alpaca[~np.isin(data_alpaca_dates, data_yf_dates)]

            # Combine data_yf and data_alpaca
            data_combined = pd.concat([data_yf, data_alpaca])
            data_combined.sort_index(inplace=True)

            # Remove duplicates
            data_combined = data_combined[~data_combined.index.duplicated(keep='first')]

            # Log date range of combined data
            logging.info(f"Combined Data Date Range: {data_combined.index.min()} to {data_combined.index.max()}")

            # Verify 'Adj Close' values
            if data_combined['Adj Close'].isnull().any():
                null_rows = data_combined[data_combined['Adj Close'].isnull()]
                logging.error(f"Null values found in 'Adj Close' at:\n{null_rows.index}")
                continue

            # Calculate log returns
            data_combined['Log Return'] = np.log(
                data_combined['Adj Close'] / data_combined['Adj Close'].shift(1)
            )
            data_combined = data_combined.iloc[1:]

            # Adjust index to represent market close time in US/Eastern
            data_combined.index = data_combined.index.tz_convert(eastern)
            data_combined.index = data_combined.index.normalize() + pd.Timedelta(hours=16)

            # Check if data_combined is empty after dropping NaNs
            if data_combined.empty:
                logging.error("Combined data is empty after dropping NaNs. Cannot fit the model.")
                continue

            log_returns_spy = data_combined['Log Return'].values

            # Check if log_returns_spy is empty
            if len(log_returns_spy) == 0:
                logging.error("Log returns array is empty. Cannot fit the model.")
                continue

            # Refit the model
            fitted_model = fit_markov_model(log_returns_spy)

            # Positive regime remains as Regime 0
            positive_regime = 0

            # Extract smoothed probabilities
            smoothed_probs = fitted_model.smoothed_marginal_probabilities[:, positive_regime]

            # Print last five probabilities with corresponding dates
            last_five_dates = data_combined.index[-5:]
            last_five_probs = smoothed_probs[-5:]
            for date, prob in zip(last_five_dates, last_five_probs):
                logging.info(f"Smoothed probability on {date.strftime('%Y-%m-%d %H:%M:%S %Z')}: {prob:.6f}")

            # Update last_p0 and last_p0_timestamp
            last_p0 = smoothed_probs[-2]
            last_p0_timestamp = last_five_dates[-2]

            logging.info(f"Updated probability of positive regime: {last_p0:.6f} at {last_p0_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        except Exception as e:
            logging.error(f"Error refitting Markov model: {e}")

# Define a global lock
update_lock = Lock()




async def trading_logic():
    global last_p0, last_p0_timestamp

    # Ensure the data directory exists
    csv_directory = "data"
    os.makedirs(csv_directory, exist_ok=True)
    csv_path = os.path.join(csv_directory, "last_p0_data.csv")

    while True:
        await asyncio.sleep(5)  # Run every 5 seconds

        try:
            # Lock and safely retrieve the current probability and timestamp
            async with update_lock:
                current_last_p0 = last_p0
                current_last_p0_timestamp = last_p0_timestamp

            # Append the probability to the CSV file for historical tracking
            if current_last_p0 is not None and current_last_p0_timestamp is not None:
                file_exists = os.path.isfile(csv_path)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["timestamp", "last_p0"])  # Add headers if file is new
                    writer.writerow([current_last_p0_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z'), current_last_p0])

            # Fetch current position
            current_positions = trading_client.get_all_positions()
            position_info = next((p for p in current_positions if p.symbol == symbol_trade), None)

            if position_info:
                position_qty = float(position_info.qty)
                position_side = "long" if position_qty > 0 else "none"
            else:
                position_qty = 0
                position_side = "none"

            # Logging details
            p0_time_str = (current_last_p0_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
                           if current_last_p0_timestamp else "N/A")
            logging.info(f"Position: {position_side}, Qty: {position_qty}, "
                         f"P(X(t+1)=0): {current_last_p0:.6f} at {p0_time_str}, threshold: {ENTRY_THRESHOLD}")

            # Trading logic
            if current_last_p0 > ENTRY_THRESHOLD:
                if position_side != "long":
                    logging.info("Condition met to enter position.")
                    await enter_position()
            else:
                if position_side == "long":
                    logging.info("Condition met to exit position.")
                    await exit_position()

        except Exception as e:
            logging.error(f"Error in trading logic: {e}")


async def main():
    await initialize_historical_data()

    tasks = [
        asyncio.create_task(trading_logic()),
        asyncio.create_task(refit_markov_model()),
    ]

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
