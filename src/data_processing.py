import pandas as pd
import os


def load_stock_data(filepath="data/raw/stock_prices.csv"):

    prices = pd.read_csv(filepath, index_col=0, parse_dates=True)

    return prices

def calculate_daily_return(prices):
    
    # Calculate changes between days
    returns = prices.pct_change(fill_method=None)

    # Remove first row
    returns = returns.iloc[1:]

    return returns

def save_processed_data(data, filepath="data/processed/stock_returns.csv"):

    # Make directory
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save processed data
    data.to_csv(filepath)

def process_stock_data(filepath="data/raw/stock_prices.csv"):
    
    # Load the data
    prices = load_stock_data(filepath)

    # Calculate Returns
    returns = calculate_daily_return(prices)

    # Save Processed Data
    save_processed_data(returns)

    return returns
