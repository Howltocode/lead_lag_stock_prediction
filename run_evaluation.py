"""
Model evaluation: walk-forward validation + holdout test on lead-lag pairs.

Uses the CSV files that main.py already saved (no download), so run
`python main.py` once first. Then:

    python run_evaluation.py

Steps (details in src/model_comparison.py):
  1. One holdout cutoff date: the most recent 20% of days
  2. Pair selection on the development period only (Bonferroni)
  3. Decision tree depth curve (walk-forward, development period only)
  4. Five models on every selected pair: walk-forward, then holdout once
  5. Tables in results/evaluation/, figures in results/figures/evaluation/
"""

import pandas as pd

from src import config
from src.evaluation_plots import (plot_confusion_matrices, plot_roc_curves,
                                  plot_tree_depth_curve)
from src.model_comparison import (compare_models, get_models, select_pairs,
                                  summarize, tree_depth_curve)
from src.validation import describe_folds, holdout_cutoff_date, split_by_date, walk_forward_folds

TABLES_DIR = config.RESULTS_DIR / "evaluation"
FIGURES_DIR = config.FIGURES_DIR / "evaluation"


def main():
    returns = pd.read_csv(config.RETURNS_PATH, index_col=0, parse_dates=True)
    residuals = pd.read_csv(config.RESIDUALS_PATH, index_col=0, parse_dates=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Holdout cutoff (same date for every pair)
    cutoff = holdout_cutoff_date(residuals.index)
    development, holdout = split_by_date(residuals.index, cutoff)
    print(f"Holdout starts {cutoff.date()}. Walk-forward folds (full-history pair):")
    print(describe_folds(residuals.index, walk_forward_folds(len(development)), holdout))

    # 2. Pair selection without the holdout period
    selected, all_dev_pairs = select_pairs(returns, residuals, cutoff)
    selected.to_csv(TABLES_DIR / "selected_pairs.csv", index=False)
    different = all_dev_pairs[~all_dev_pairs["same_stock"]]
    print(f"\nDevelopment-period pair test: {int(different['significant'].sum())} of "
          f"{len(different)} pairs have p < {config.ALPHA} "
          f"(~{round(len(different) * config.ALPHA)} expected by chance); "
          f"{len(selected)} pass Bonferroni.")

    # 3. Tree depth curve -> pick TREE_MAX_DEPTH in config.py
    curve = tree_depth_curve(returns, residuals, selected, cutoff)
    curve.to_csv(TABLES_DIR / "tree_depth_curve.csv", index=False)
    plot_tree_depth_curve(curve, FIGURES_DIR / "tree_depth_curve.png")
    best_depth = int(curve.loc[curve["validation_accuracy"].idxmax(), "depth"])
    print("\nDecision tree depth curve (mean over pairs and folds):")
    print(curve.round(4).to_string(index=False))
    print(f"Best validation depth: {best_depth} "
          f"(config.TREE_MAX_DEPTH = {config.TREE_MAX_DEPTH})")

    # 4. Model comparison
    models = get_models()
    fold_results, holdout_results, holdout_predictions = compare_models(
        returns, residuals, selected, cutoff, models)
    fold_results.to_csv(TABLES_DIR / "fold_results.csv", index=False)
    holdout_results.to_csv(TABLES_DIR / "holdout_results.csv", index=False)

    summary = summarize(fold_results, holdout_results)
    summary.to_csv(TABLES_DIR / "model_summary.csv", index=False)
    print("\nModel comparison (mean over selected pairs):")
    print(summary.round(4).to_string(index=False))

    # 5. Figures
    names = list(models.keys())
    plot_confusion_matrices(holdout_predictions, names,
                            FIGURES_DIR / "confusion_matrices.png")
    plot_roc_curves(holdout_predictions, names, FIGURES_DIR / "roc_curves.png")
    print(f"\nTables saved in {TABLES_DIR}\nFigures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
