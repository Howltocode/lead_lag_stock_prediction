"""
Stage 2: turn prices into daily returns.

Daily return = (today's price - yesterday's price) / yesterday's price
Example: price goes 100 -> 102, return = 0.02 (= +2%).
"""

import pandas as pd

from src import config


def load_stock_data(filepath=config.RAW_PRICES_PATH):
    """Read the saved price CSV (dates as the index)."""
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


def calculate_daily_return(prices):
    """Percent change from the previous trading day, for every ticker."""
    returns = prices.pct_change(fill_method=None)

    # The first row has no "yesterday", so it is empty. Remove it.
    return returns.iloc[1:]


def save_processed_data(data, filepath):
    """Save a DataFrame to CSV, creating the folder if needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filepath)


def process_stock_data(prices_path=config.RAW_PRICES_PATH, output_path=config.RETURNS_PATH):
    """Load prices -> calculate returns -> save returns. Returns the returns table."""
    prices = load_stock_data(prices_path)
    returns = calculate_daily_return(prices)
    save_processed_data(returns, output_path)
    return returns
