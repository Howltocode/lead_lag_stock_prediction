# Lead Lag Stock Prediction

Team Flippers: Liam Liangsu, Gavin Lee, Howl Liu, and Josh Guzman.

This project investigates whether previous-day movements in leading stocks can help predict the next-day residual-return direction of lagging stocks.

## Project stages

1. Collect historical stock-price data ✅
2. Clean the data and calculate returns ✅ (including residual returns)
3. Investigate lead-lag correlations ✅
4. Create machine-learning features and labels
5. Train and evaluate prediction models

## Setup and run

```bash
pip install -r requirements.txt
python main.py
```

`main.py` runs every finished stage in order. Prices are only downloaded if
`data/raw/stock_prices.csv` is missing or doesn't have every ticker yet.

## Project structure

```
main.py                  Runs the whole pipeline
src/
  config.py              Settings: tickers, dates, beta window, file paths
  data_collection.py     Stage 1: download prices from Yahoo Finance
  data_processing.py     Stage 2: prices -> daily returns
  residuals.py           Stage 2: daily returns -> residual returns
  lead_lag.py            Stage 3: lead-lag correlation analysis + plots
data/
  raw/stock_prices.csv           Daily closing prices (stocks + SPY)
  processed/stock_returns.csv    Daily returns
  processed/residual_returns.csv Residual returns (market effect removed)
results/                 Created when you run main.py (tables + figures)
```

To change the stock list, dates, or other settings, edit `src/config.py`.

## Key ideas

**Residual return.** Most stocks move with the market. To focus on what is
specific to each stock, we remove the market's part of the move:

```
residual(t) = stock_return(t) - beta * SPY_return(t)
```

Beta is estimated from the previous 60 trading days only (no future data),
so the residual on day t never uses information from day t or later.

**Lead-lag correlation.** For every pair of stocks (A, B) we measure the
correlation between A's return on day t-1 and B's residual return on day t.
Because we test 100 pairs, we also report a Bonferroni-corrected
significance flag to avoid counting lucky pairs as real.

## Outputs of Stage 3 (in `results/`)

| File | What it shows |
|------|---------------|
| `lead_lag_correlations.csv` | 10x10 matrix: rows = leader (day t-1), columns = lagger (day t) |
| `lead_lag_pairs.csv` | One row per pair, sorted by strength, with p-values and significance flags |
| `lead_lag_yearly.csv` | Correlation of the strongest pairs, year by year |
| `figures/lead_lag_heatmap.png` | Heatmap of the correlation matrix |
| `figures/lead_lag_yearly.png` | Stability of the strongest pairs over time |
