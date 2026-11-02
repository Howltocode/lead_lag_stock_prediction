import pandas as pd
import yfinance as yf


def download_stock_prices(tickers, start="2018-01-01", end=None):
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

    return prices.dropna(how="all")
