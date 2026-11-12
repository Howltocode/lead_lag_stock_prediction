"""
Stage 1: download daily stock prices from Yahoo Finance.
"""

import pandas as pd
import yfinance as yf

from src import config


def download_stock_prices(tickers, start=config.START_DATE, end=config.END_DATE,
                          filepath=config.RAW_PRICES_PATH, force_download=False):
    """
    Get daily closing prices (adjusted for splits and dividends).

    To avoid downloading the same data every run, we first check whether the
    CSV file already exists and already has every ticker we need. We only
    download again if a ticker is missing or if force_download=True.

    Returns a DataFrame: one row per trading day, one column per ticker.
    """
    if filepath.exists() and not force_download:
        saved = pd.read_csv(filepath, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in saved.columns]

        if not missing:
            print(f"Using saved prices from {filepath}")
            return saved[tickers]

        print(f"Saved prices are missing {missing}, downloading again...")

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjust prices for splits and dividends
        progress=False,
    )

    prices = data["Close"]

    # With a single ticker yfinance returns a Series; turn it into a table.
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Drop days where no ticker has a price (e.g. market holidays).
    # Done BEFORE saving so the saved file is the cleaned version.
    prices = prices.dropna(how="all")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(filepath)
    print(f"Saved prices to {filepath}")

    return prices[tickers]
