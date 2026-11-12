"""
Stage 2b: residual returns (each stock's move with the market's move taken out).

WHY WE NEED THIS
----------------
Most stocks go up on days the whole market goes up. If we predict raw
returns, we would mostly be predicting "the market", not anything special
about the stock. The residual return keeps only the part of the move that
is specific to that stock.

HOW IT WORKS
------------
1. Beta = how strongly a stock follows the market.
   beta = 1.5 means: when the market moves +1%, this stock tends to move +1.5%.

       beta = Cov(stock return, market return) / Var(market return)

2. Residual = actual return - what the market alone would explain.

       residual(t) = stock_return(t) - beta * market_return(t)

   Example: market +1%, beta 1.5, stock actually moved +2%.
            expected from market = 1.5%, residual = 2% - 1.5% = +0.5%.

AVOIDING LOOK-AHEAD (IMPORTANT)
-------------------------------
Beta changes over time, so we estimate it on a rolling window (the last
BETA_WINDOW days). The beta used for day t must be computed ONLY from days
before t. Otherwise day t's own return leaks into its own residual, which
would make our later results look better than they really are.
We do that with .shift(1) below.
"""

from src import config


def compute_rolling_beta(stock_returns, market_returns, window=config.BETA_WINDOW):
    """
    Rolling beta of every stock vs the market.

    stock_returns:  DataFrame, one column per stock
    market_returns: Series, the market's daily returns (e.g. SPY)

    Returns a DataFrame shaped like stock_returns. The value on day t is the
    beta estimated from the `window` days ending on day t-1.
    """
    # Rolling covariance of each stock with the market, and rolling variance of the market.
    covariance = stock_returns.rolling(window).cov(market_returns)
    market_variance = market_returns.rolling(window).var()

    beta = covariance.div(market_variance, axis=0)

    # Shift by one day so day t only uses information up to day t-1.
    return beta.shift(1)


def compute_residual_returns(returns, tickers=config.TICKERS, benchmark=config.BENCHMARK,
                             window=config.BETA_WINDOW):
    """
    Residual returns for every stock in `tickers`.

    returns: DataFrame of daily returns that includes the benchmark column (SPY).

    Returns (residuals, betas). The first `window` days are dropped because
    there is not enough history yet to estimate beta.
    """
    stock_returns = returns[tickers]
    market_returns = returns[benchmark]

    betas = compute_rolling_beta(stock_returns, market_returns, window)

    # residual = actual return - beta * market return
    expected_from_market = betas.mul(market_returns, axis=0)
    residuals = stock_returns - expected_from_market

    # Drop the early rows where beta could not be estimated yet.
    residuals = residuals.dropna(how="all")
    betas = betas.loc[residuals.index]

    return residuals, betas
