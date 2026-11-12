"""
Project settings in one place.

If you want to change the stock list, the date range, or where files are
saved, change it here instead of editing the other files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------

# The stocks we study for lead-lag relationships.
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "AMD", "META", "AVGO", "TSM", "JPM"]

# The "market" we compare every stock against when computing residual returns.
# SPY is an ETF that tracks the S&P 500, so it is a common stand-in for
# "how the whole market moved today".
BENCHMARK = "SPY"

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

START_DATE = "2018-01-01"

# A fixed end date keeps results reproducible: everyone on the team gets the
# exact same data. (yfinance treats the end date as exclusive, so this
# includes trading days up to 2026-11-04.)
# Set it to None if you want the latest data instead.
END_DATE = "2026-11-05"

# ---------------------------------------------------------------------------
# Residual-return settings
# ---------------------------------------------------------------------------

# How many past trading days to use when estimating each stock's beta.
# About 60 trading days = about 3 months.
BETA_WINDOW = 60

# ---------------------------------------------------------------------------
# Lead-lag settings
# ---------------------------------------------------------------------------

# Significance level for the correlation tests (before the multiple-testing
# correction described in lead_lag.py).
ALPHA = 0.05

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

# Paths are built from the project folder, so the code works no matter which
# folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PRICES_PATH = PROJECT_ROOT / "data" / "raw" / "stock_prices.csv"
RETURNS_PATH = PROJECT_ROOT / "data" / "processed" / "stock_returns.csv"
RESIDUALS_PATH = PROJECT_ROOT / "data" / "processed" / "residual_returns.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
