"""
Runs the project pipeline from start to finish:

  1. Download prices        -> data/raw/stock_prices.csv
  2. Calculate returns      -> data/processed/stock_returns.csv
  2b. Residual returns      -> data/processed/residual_returns.csv
  3. Lead-lag analysis      -> results/ (CSV tables + figures)

Run from the project folder:  python main.py
"""

from src import config
from src.data_collection import download_stock_prices
from src.data_processing import process_stock_data, save_processed_data
from src.residuals import compute_residual_returns
from src.lead_lag import run_lead_lag_analysis
from src.modeling import train_all_pair_models



def main():
    # 1. Raw prices (stocks + the market benchmark, SPY)
    prices = download_stock_prices(config.TICKERS + [config.BENCHMARK])
    print("\nPrice data shape:", prices.shape)

    # 2. Daily returns
    returns = process_stock_data()
   
    print("Return data shape:", returns.shape)

    print("First date:", returns.index.min())
    print("Last date:", returns.index.max())

    # 2b. Residual returns (market effect removed)
    residuals, betas = compute_residual_returns(returns)
    save_processed_data(residuals, config.RESIDUALS_PATH)
    print("Residual data shape:", residuals.shape)
    print("\nAverage beta of each stock:")
    print(betas.mean().round(2).to_string())

    pair_models, pair_results = train_all_pair_models(
        returns,
        residuals,
        config.TICKERS,
    )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pair_results_path = config.RESULTS_DIR / "pair_model_results.csv"
    pair_results.to_csv(pair_results_path, index=False)
    print("\nTrained stock-pair models:", len(pair_models))
    print("Mean test accuracy:", round(pair_results["model_accuracy"].mean(), 4))
    print("Mean test ROC-AUC:", round(pair_results["model_auc"].mean(), 4))
    different_stock_results = pair_results[~pair_results["same_stock"]]
    print("Different-stock pairs beating majority baseline:",
          int((different_stock_results["model_accuracy"] >
               different_stock_results["baseline_accuracy"]).sum()))
    print("\nBest logistic regression stock pairs:")
    print(pair_results[~pair_results["same_stock"]][[
        "leader", "lagger", "model_accuracy", "model_auc",
        "baseline_accuracy",
    ]].head(10).round(4).to_string(index=False))

    # 3. Lead-lag analysis
    #    Leaders use raw returns ("how the stock moved yesterday").
    #    Laggers use residual returns ("the stock-specific move today").
    pairs = run_lead_lag_analysis(returns[config.TICKERS], residuals)

    print("\nStrongest lead-lag pairs (different stocks only):")
    print(pairs[~pairs["same_stock"]].head(10).round(4).to_string(index=False))

    num_sig = pairs["significant"].sum()
    num_sig_bonf = pairs["significant_bonf"].sum()
    print(f"\nSignificant pairs: {num_sig} of {len(pairs)} without correction, "
          f"{num_sig_bonf} after Bonferroni correction.")
    print(f"Tables and figures saved in {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
