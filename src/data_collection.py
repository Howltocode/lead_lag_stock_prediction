import pandas as pd
import yfinance as yf
import os


def download_stock_prices(tickers, start="2018-01-01", end=None, save=True):
    """Download daily adjusted closing prices for the requested tickers."""
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    prices = data["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    # Save the stock data for processing
    if save:
        os.makedirs("data/raw", exist_ok=True)

        prices.to_csv("data/raw/stock_prices.csv")

    return prices.dropna(how="all")
