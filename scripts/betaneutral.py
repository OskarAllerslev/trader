import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize
import yaml
from numpy.linalg import eigvals
from sklearn.decomposition import PCA

# Load configuration from config.yaml
config_path = r"C:\Users\Oskar\OneDrive\strategytrader\trader\config\config.yaml.txt"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Retrieve parameters from config
beta_neutral_allocation = config['strategies']['beta_neutral']['allocation_percentage']
min_return_threshold = config['strategies']['beta_neutral'].get('min_return', 0.0002)
sp500_csv_path = config.get('data', {}).get('sp500_csv', "data/SP500.csv")

def optimize_portfolio():
    # Load S&P 500 tickers
    market_tickers = pd.read_csv(sp500_csv_path)
    tickers = market_tickers['Symbol'].tolist()

    # Add SPY to the tickers list
    tickers.append("SPY")

    # Define date range for historical data
    start_date = (datetime.now() - timedelta(days=252*2 )).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Download historical data
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", auto_adjust=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return



    close = data.xs(key = 'Close', axis = 1, level = 1)

    # Remove delisted or missing tickers
    valid_tickers = close.columns.dropna()
    close = close[valid_tickers]

    # Calculate log returns
    log_returns = np.log(close / close.shift(1))
    log_returns = log_returns.dropna(axis=1, how="all")[1:]  # Remove NaNs and empty columns

    if 'SPY' not in log_returns:
        print("SPY data missing. Cannot compute beta-neutral portfolio.")
        return

    # Separate SPY returns and stock returns
    spy_returns = log_returns['SPY']
    stock_returns = log_returns.drop(columns=['SPY'])

    # Calculate betas for each stock
    beta_dict = {}
    for stock in stock_returns.columns:
        aligned_stock, aligned_spy = stock_returns[stock].align(spy_returns, join="inner")
        if aligned_stock.empty or aligned_spy.empty:
            print(f"Skipping {stock} due to insufficient data.")
            continue

        # Calculate beta
        cov = np.cov(aligned_stock, aligned_spy)[0, 1]
        var = np.var(aligned_spy)
        if var == 0:
            print(f"Skipping {stock} due to zero variance in SPY.")
            continue

        beta = cov / var
        beta_dict[stock] = beta

    # Convert beta dictionary to DataFrame
    beta_values = pd.DataFrame.from_dict(beta_dict, orient='index', columns=['Beta'])
    beta_values = beta_values.dropna()
    



    # Filter stock returns to match beta values
    filtered_stocks = beta_values.index
    stock_returns = stock_returns[filtered_stocks]
    expected_returns = stock_returns.mean().values

    cov_matrix = stock_returns.cov()

        # Scale expected returns
    expected_returns /= np.max(np.abs(expected_returns))

    # Scale covariance matrix
    cov_matrix /= np.max(np.abs(cov_matrix))

  
    


    def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate = 0):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return -risk_free_rate)/portfolio_std
        return - sharpe_ratio
    
    def portfolio_beta(weights, betas):
        return np.dot(weights,betas)
    

    print("Covariance Matrix:")
    print(cov_matrix)
    print("Covariance Matrix Eigenvalues:", eigvals(cov_matrix))
    print("Expected Returns:", expected_returns)
    print("Beta Values:", beta_values['Beta'].values)

    # Define optimization parameters
    num_stocks = len(filtered_stocks)
    initial_weights = np.ones(num_stocks) / num_stocks

    constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Ensure weights sum to 1
    ]


    def penalty_function(weights, betas):
        return np.abs(portfolio_beta(weights,betas))

    bounds = [(0.025,1) for _ in range(num_stocks)]

    result = minimize(
        lambda w: negative_sharpe_ratio(w, expected_returns, cov_matrix),
        initial_weights,
        constraints=constraints,
        bounds=bounds,
        method='trust-constr',
    )

        # Check results
    if result.success:
        optimized_weights = result.x
        print("Optimization successful!")
        print("Optimized Portfolio Weights:", optimized_weights)
        print("Portfolio Beta:", portfolio_beta(optimized_weights, beta_values['Beta'].values))
        portfolio_return = np.dot(optimized_weights, expected_returns)
        portfolio_variance = np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights))
        portfolio_std_dev = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return) / portfolio_std_dev
        print("Portfolio Return:", portfolio_return)
        print("Portfolio Volatility:", portfolio_std_dev)
        print("Sharpe Ratio:", sharpe_ratio)
        return optimized_weights
    else:
        print("Optimization failed:", result.message)
        return None  # Explicitly return None if optimization fails

    return optimized_weights

if __name__ == "__main__":
    weights = optimize_portfolio()
    if weights is not None:
        print(weights)
        print("Sum of Weights:", sum(weights['Weight']))
