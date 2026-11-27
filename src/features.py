"""
Stage 4b: multi-leader feature dataset for one lagging stock.

The original pair models (modeling.py) use ONE feature: a single leader's
return from the previous day. That is too little to compare different
model types, so here we build a richer dataset for one lagger:

    features (all known at the close of day t-1):
      - previous-day raw return of every leader stock
      - previous-day market (SPY) return
      - previous-day residual return of the lagger itself

    target:
      - 1 if the lagger's residual return on day t is positive, else 0

Every feature is shifted by one day, so the row for day t only uses
information available before day t starts (no look-ahead).
"""

import pandas as pd

from src import config


def build_lagger_dataset(returns, residuals, lagger, leaders=None,
                         benchmark=config.BENCHMARK, include_market=True,
                         include_own_lag=True):
    """
    Build (X, y) for predicting the direction of `lagger`'s residual return.

    returns:   daily returns, must include the leader columns and the benchmark
    residuals: residual returns, must include the lagger column
    leaders:   list of leader tickers (default: every ticker except the lagger)

    Leaders without data for the whole period (e.g. a stock that IPO'd later)
    are dropped so we don't throw away years of rows. The dropped tickers are
    returned so the caller can report them.

    Returns (X, y, dropped_leaders). X and y share the same date index.
    """
    if leaders is None:
        leaders = [t for t in config.TICKERS if t != lagger]

    target = residuals[lagger]
    start, end = target.first_valid_index(), target.last_valid_index()

    # Drop leaders that are missing data inside the lagger's date range.
    window = returns.loc[start:end, leaders].shift(1).iloc[1:]
    dropped = [t for t in leaders if window[t].isna().any()]
    kept = [t for t in leaders if t not in dropped]

    features = returns[kept].shift(1).add_suffix("_prev")
    if include_market:
        features[f"{benchmark}_prev"] = returns[benchmark].shift(1)
    if include_own_lag:
        features[f"{lagger}_resid_prev"] = residuals[lagger].shift(1)

    data = features.join(target.rename("target_return"), how="inner").dropna()

    X = data.drop(columns="target_return")
    y = (data["target_return"] > 0).astype(int).rename("target")
    return X, y, dropped
