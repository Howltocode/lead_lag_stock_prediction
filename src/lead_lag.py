"""
Stage 3: look for lead-lag relationships.

THE QUESTION
------------
Does stock A's return YESTERDAY tell us anything about stock B's residual
return TODAY? If yes, A "leads" B and B "lags" A.

WHAT THIS FILE PRODUCES
-----------------------
1. A lagged correlation matrix
     rows    = leader (its return on day t-1)
     columns = lagger (its residual return on day t)
   Each cell is a correlation between -1 and 1. Near 0 = no relationship.

2. A p-value for every cell (how likely a correlation this big is by pure luck).

3. A "multiple-testing" correction. With 10 stocks we test 10 x 10 = 100
   pairs. Even with zero real relationships, about 5 of 100 pairs would look
   "significant" at the usual 0.05 level just by chance. The Bonferroni
   correction fixes this by requiring p < 0.05 / (number of tests).

4. A stability check: correlation of a pair year by year. A real
   relationship should show up in most years, not just one lucky year.

NOTE ON THE DIAGONAL
--------------------
Cells where leader == lagger (e.g. AAPL -> AAPL) measure whether a stock's
own move yesterday predicts its own residual today. That is not lead-lag
between two stocks, but it is a useful baseline, so we keep it in the matrix
and label it in the pairs table.
"""

import math

import matplotlib
matplotlib.use("Agg")  # save plots to files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config


def align_leaders_and_laggers(leader_returns, lagger_residuals, lag=1):
    """
    Line up yesterday's leader returns with today's lagger residuals.

    After shift(lag), the row for day t holds the leader return from day t-lag.
    We then keep only days where both tables have data.
    """
    leaders_shifted = leader_returns.shift(lag)
    common_days = leaders_shifted.index.intersection(lagger_residuals.index)
    return leaders_shifted.loc[common_days], lagger_residuals.loc[common_days]


def lagged_correlation_matrix(leader_returns, lagger_residuals, lag=1):
    """
    Correlation between every leader (day t-lag) and every lagger (day t).

    Returns (correlations, sample_sizes), both DataFrames with
    rows = leaders, columns = laggers.
    """
    leaders, laggers = align_leaders_and_laggers(leader_returns, lagger_residuals, lag)

    correlations = pd.DataFrame(index=leaders.columns, columns=laggers.columns, dtype=float)
    sample_sizes = pd.DataFrame(index=leaders.columns, columns=laggers.columns, dtype=int)

    for leader in leaders.columns:
        for lagger in laggers.columns:
            # Only use days where both values exist.
            pair = pd.concat([leaders[leader], laggers[lagger]], axis=1).dropna()
            correlations.loc[leader, lagger] = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            sample_sizes.loc[leader, lagger] = len(pair)

    return correlations, sample_sizes


def correlation_p_value(r, n):
    """
    Two-sided p-value for a correlation r measured on n days.

    We turn r into a test statistic:  z = r * sqrt((n - 2) / (1 - r^2))
    With our sample sizes (~2000 days) this statistic is very close to a
    standard normal distribution, so we use the normal distribution to get
    the p-value. (This avoids adding scipy just for one formula.)
    """
    if n < 3 or abs(r) >= 1:
        return float("nan")
    z = r * math.sqrt((n - 2) / (1 - r ** 2))
    return math.erfc(abs(z) / math.sqrt(2))


def p_value_matrix(correlations, sample_sizes):
    """Apply correlation_p_value to every cell of the correlation matrix."""
    p_values = correlations.copy()
    for leader in correlations.index:
        for lagger in correlations.columns:
            p_values.loc[leader, lagger] = correlation_p_value(
                correlations.loc[leader, lagger], sample_sizes.loc[leader, lagger]
            )
    return p_values


def build_pairs_table(correlations, sample_sizes, p_values, alpha=config.ALPHA):
    """
    Flatten the matrices into one row per (leader, lagger) pair, sorted by
    the strength of the relationship (biggest absolute correlation first).

    Columns:
      correlation         - the lag-1 correlation
      n_days              - number of days used
      p_value             - chance of a correlation this large by luck
      significant         - p_value < alpha              (no correction)
      significant_bonf    - p_value < alpha / num_tests  (Bonferroni corrected)
      same_stock          - True for the diagonal (stock vs itself)
    """
    num_tests = correlations.size
    bonferroni_alpha = alpha / num_tests

    rows = []
    for leader in correlations.index:
        for lagger in correlations.columns:
            p = p_values.loc[leader, lagger]
            rows.append({
                "leader": leader,
                "lagger": lagger,
                "correlation": correlations.loc[leader, lagger],
                "n_days": int(sample_sizes.loc[leader, lagger]),
                "p_value": p,
                "significant": p < alpha,
                "significant_bonf": p < bonferroni_alpha,
                "same_stock": leader == lagger,
            })

    pairs = pd.DataFrame(rows)
    pairs["abs_correlation"] = pairs["correlation"].abs()
    pairs = pairs.sort_values("abs_correlation", ascending=False).drop(columns="abs_correlation")
    return pairs.reset_index(drop=True)


def yearly_correlations(leader_returns, lagger_residuals, pairs, lag=1):
    """
    Stability check: the lag-1 correlation of each given pair, computed
    separately for each calendar year.

    pairs: list of (leader, lagger) tuples
    Returns a DataFrame: rows = years, columns = "LEADER -> LAGGER".
    """
    leaders, laggers = align_leaders_and_laggers(leader_returns, lagger_residuals, lag)

    result = {}
    for leader, lagger in pairs:
        pair = pd.concat([leaders[leader], laggers[lagger]], axis=1).dropna()
        pair.columns = ["leader", "lagger"]
        by_year = pair.groupby(pair.index.year)
        result[f"{leader} -> {lagger}"] = by_year.apply(lambda d: d["leader"].corr(d["lagger"]))

    yearly = pd.DataFrame(result)
    yearly.index.name = "year"
    return yearly


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(correlations, filepath):
    """Heatmap of the lagged correlation matrix. Red = positive, blue = negative."""
    limit = np.nanmax(np.abs(correlations.values))  # same scale on both sides of 0

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlations, annot=True, fmt=".3f", cmap="RdBu_r",
                vmin=-limit, vmax=limit, center=0, ax=ax)
    ax.set_xlabel("Lagger (residual return on day t)")
    ax.set_ylabel("Leader (return on day t-1)")
    ax.set_title("Lag-1 correlation: leader yesterday vs lagger residual today")
    fig.tight_layout()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_yearly_correlations(yearly, filepath):
    """Line chart of each pair's correlation by year. A stable pair stays on one side of 0."""
    fig, ax = plt.subplots(figsize=(9, 5))
    yearly.plot(ax=ax, marker="o")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Lag-1 correlation")
    ax.set_title("Is the lead-lag relationship stable over time?")
    ax.legend(fontsize=8)
    fig.tight_layout()

    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def run_lead_lag_analysis(leader_returns, lagger_residuals, lag=1, top_n=5):
    """
    Run the full Stage 3 analysis and save the outputs to results/.

    leader_returns:   raw daily returns of the stocks (used as leaders)
    lagger_residuals: residual returns of the stocks (used as laggers)
    top_n:            how many of the strongest cross-stock pairs to check year by year

    Returns the pairs table (one row per leader/lagger pair).
    """
    correlations, sample_sizes = lagged_correlation_matrix(leader_returns, lagger_residuals, lag)
    p_values = p_value_matrix(correlations, sample_sizes)
    pairs = build_pairs_table(correlations, sample_sizes, p_values)

    # Strongest pairs between DIFFERENT stocks (the actual lead-lag candidates).
    top_pairs = pairs[~pairs["same_stock"]].head(top_n)
    yearly = yearly_correlations(
        leader_returns, lagger_residuals,
        list(zip(top_pairs["leader"], top_pairs["lagger"])), lag,
    )

    # Save everything.
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(config.RESULTS_DIR / "lead_lag_correlations.csv")
    pairs.to_csv(config.RESULTS_DIR / "lead_lag_pairs.csv", index=False)
    yearly.to_csv(config.RESULTS_DIR / "lead_lag_yearly.csv")
    plot_correlation_heatmap(correlations, config.FIGURES_DIR / "lead_lag_heatmap.png")
    plot_yearly_correlations(yearly, config.FIGURES_DIR / "lead_lag_yearly.png")

    return pairs
