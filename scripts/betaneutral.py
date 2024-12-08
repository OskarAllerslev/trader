
def optimize_portfolio():
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    from scipy.optimize import minimize
    market_tickers = pd.read_csv("data/SP500.csv")
    tickers = market_tickers['Symbol'].tolist()

    tickers.append("SPY")


    start_date = datetime.now() - timedelta(days = 3)

    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, end = end_date)
    
    #subset = ["SPY", "APC", "ADI", "ANDV", "ANSS", "ANTM", "AON", "APA", "AIV", "AAPL", "AMAT", "APTV", "ADM", "ARNC", "AJG", "AIZ", "T", "ADSK", "ADP", "AZO", "AVB", "AVY", "BHGE", "BLL", "BAC", "BAX", "BBT", "BDX", "BRK.B", "BBY", "BIIB", "BLK", "HRB", "BA", "BWA", "BXP", "BSX", "BHF", "BMY", "AVGO", "BF.B", "CHRW", "CA", "COG"]
    #data = yf.download(subset, start=start_date, end = end_date)



    adj_close = data['Adj Close']

    log_returns = np.log(adj_close / adj_close.shift(1))
    log_returns = log_returns.dropna(axis=1, how="all")[1:]


    spy_returns = log_returns['SPY']
    stock_returns = log_returns.drop(columns=['SPY'])



    beta_dict = {}

    for stock in stock_returns.columns:
        aligned_stock, aligned_spy = stock_returns[stock].align(spy_returns, join="inner")
        
        if aligned_stock.empty or aligned_spy.empty:
            print(f"Skipping {stock} due to empty aligned data.")
            continue  
        
        cov = np.cov(aligned_stock, aligned_spy)[0, 1]
        var = np.var(aligned_spy)
        beta = cov / var
        beta_dict[stock] = beta


    beta_values = pd.DataFrame.from_dict(beta_dict, orient='index', columns=['Beta'])
    beta_values = beta_values.dropna()


    filtered_stocks = beta_values.index


    cov_matrix = stock_returns[filtered_stocks].cov().values
    expected_returns = stock_returns[filtered_stocks].mean().values




    num_stocks = len(filtered_stocks)
    initial_weights = np.ones(num_stocks) / num_stocks



    def sharpe_ratio(weights):
        port_return = np.dot(weights, expected_returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return -port_return / np.sqrt(port_variance)  # Negative for minimization



    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix,weights))

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda w: np.dot(w, beta_values['Beta'].values)},  # Beta neutral
        {'type': 'ineq', 'fun': lambda w: np.dot(w, expected_returns) - 0.0002}  # Min expected return
    ]

    # Bounds to disallow short selling and enforce minimum weight per stock
    bounds = [(0.01, 1) for _ in range(num_stocks)]



    result = minimize(portfolio_variance, initial_weights, constraints=constraints, bounds=bounds)

    # Results
    optimized_weights = result.x 



    # Calculate portfolio metrics
    weights_df = pd.DataFrame({'Stock': beta_values.index, 'Weight': optimized_weights})
    optimized_return = np.dot(optimized_weights, expected_returns)
    optimized_variance = portfolio_variance(optimized_weights)
    annualized_return = (1 + optimized_return) ** 252 - 1
    annualized_variance = optimized_variance * 252
    annualized_volatility = np.sqrt(annualized_variance)
    beta_portfolio = np.dot(optimized_weights, beta_values)



    
    print("Expected Return:", optimized_return)
    print("Portfolio Volatility:", np.sqrt(optimized_variance))
    print("Annualized Return:", annualized_return)
    print("Annualized Volatility:", annualized_volatility)
    print("Portfolio Beta",beta_portfolio)
    
    return weights_df

if __name__ == "__main__":
    weights = optimize_portfolio()
    print(weights)
    print(sum(weights['Weight']))