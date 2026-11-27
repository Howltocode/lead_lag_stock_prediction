"""
Walk-forward evaluation of multi-leader models.

Uses the CSV files that main.py already saved (no download), so run
`python main.py` once first. Then:

    python run_evaluation.py

For each lagger in config.EVAL_LAGGERS:
  1. Build the multi-leader dataset (features.py)
  2. Split into development + holdout (validation.py)
  3. Walk-forward validate on the development part (evaluation.py)
  4. Score once on the holdout

Outputs go to results/evaluation/.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluation import holdout_evaluate, walk_forward_evaluate
from src.features import build_lagger_dataset
from src.validation import describe_folds, split_holdout, walk_forward_folds

OUTPUT_DIR = config.RESULTS_DIR / "evaluation"


def make_model():
    # Scaling matters once there are many features on different scales.
    # The scaler sits inside the pipeline, so it is fit on each training
    # block only (fitting it on all the data would leak future statistics).
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def main():
    returns = pd.read_csv(config.RETURNS_PATH, index_col=0, parse_dates=True)
    residuals = pd.read_csv(config.RESIDUALS_PATH, index_col=0, parse_dates=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for lagger in config.EVAL_LAGGERS:
        X, y, dropped = build_lagger_dataset(returns, residuals, lagger)

        development, holdout = split_holdout(len(X))
        folds = walk_forward_folds(len(development))

        print(f"\n=== {lagger}: {X.shape[1]} features, {len(X)} days, "
              f"up-days {y.mean():.1%} ===")
        if dropped:
            print(f"  dropped leaders without full history: {dropped}")
        print(describe_folds(X.index, folds, holdout))

        model = make_model()
        fold_results, _ = walk_forward_evaluate(model, X, y, folds)
        _, holdout_metrics, _ = holdout_evaluate(model, X, y, development, holdout)

        fold_results.insert(0, "lagger", lagger)
        fold_results.to_csv(OUTPUT_DIR / f"walk_forward_{lagger}.csv", index=False)
        print(fold_results[["fold", "accuracy", "baseline_accuracy", "auc"]]
              .round(4).to_string(index=False))

        summary_rows.append({
            "lagger": lagger,
            "cv_accuracy_mean": fold_results["accuracy"].mean(),
            "cv_baseline_mean": fold_results["baseline_accuracy"].mean(),
            "cv_auc_mean": fold_results["auc"].mean(),
            "cv_auc_std": fold_results["auc"].std(),
            "holdout_accuracy": holdout_metrics["accuracy"],
            "holdout_baseline": holdout_metrics["baseline_accuracy"],
            "holdout_auc": holdout_metrics["auc"],
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print("\n=== Summary (logistic regression) ===")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
