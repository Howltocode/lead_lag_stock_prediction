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
TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",

    "META",
    "AVGO",
    "AMD",
    "ORCL",
    "CRM",
    "ADBE",
    "CSCO",
    "QCOM",
    "INTC",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "V",
    "WFC",
    "JNJ",
    "UNH",
    "PFE",
    "ABBV",
    "MRK",
    "LLY",
    "XOM",
    "CVX",
    "COP",
    "CAT",
    "DE",
    "GE",
    "HON",
    "BA",
    "UPS",
    "WMT",
    "COST",
    "HD",
    "MCD",
    "SBUX",
    "KO",
    "PEP",
    "NKE",
    "NFLX",
    "DIS",
    "TMO",
    "AMGN",
    "IBM",
    "PLTR",
]

# GICS sector of each stock. Only used to check the clustering results
# (the clustering itself never sees these labels).
SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech",
    "AMD": "Tech", "ORCL": "Tech", "CRM": "Tech", "ADBE": "Tech",
    "CSCO": "Tech", "QCOM": "Tech", "INTC": "Tech", "IBM": "Tech",
    "PLTR": "Tech",
    "GOOGL": "Communication", "META": "Communication",
    "NFLX": "Communication", "DIS": "Communication",
    "AMZN": "Consumer Disc.", "HD": "Consumer Disc.", "MCD": "Consumer Disc.",
    "SBUX": "Consumer Disc.", "NKE": "Consumer Disc.",
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "V": "Financials", "WFC": "Financials",
    "JNJ": "Health Care", "UNH": "Health Care", "PFE": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "LLY": "Health Care",
    "TMO": "Health Care", "AMGN": "Health Care",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "CAT": "Industrials", "DE": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "BA": "Industrials", "UPS": "Industrials",
}

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
# Model evaluation settings (walk-forward validation, see validation.py)
# ---------------------------------------------------------------------------

# The most recent share of the data kept as a final holdout test.
# It is used exactly once, after all model choices are made.
HOLDOUT_FRACTION = 0.20

# Walk-forward folds inside the development part (in trading days).
# ~252 trading days = 1 year.
INITIAL_TRAIN_DAYS = 756   # first training window: ~3 years
VALIDATION_DAYS = 252      # each validation block: ~1 year

# Decision tree depths tried in the train-vs-validation curve.
TREE_DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16]

# Depth used for the decision tree and random forest in the comparison.
# Set this from the depth curve: the depth with the best validation accuracy.
TREE_MAX_DEPTH = 1

# K-means clustering of stocks (clustering.py). Silhouette is highest at
# k = 3 (broad style groups) and has a second peak at k = 8 (sector level).
N_CLUSTERS = 8
N_CLUSTERS_COARSE = 3

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
