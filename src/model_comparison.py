"""
Stage 5d: compare model types on the selected lead-lag pairs.

STEPS
-----
1. Pick one holdout cutoff date (the most recent 20% of days).
2. Select pairs using ONLY the development period (before the cutoff):
   rerun the Stage 3 correlation test and keep the different-stock pairs
   that pass the Bonferroni correction. Selecting pairs with the full data
   would let the holdout period influence which pairs we test.
3. For every selected pair, build the same dataset as modeling.py
   (leader's previous-day return -> lagger's residual direction today).
4. For every model: walk-forward validation on the development period,
   then one final score on the holdout period.

MODELS (zyBook chapter in brackets)
-----------------------------------
  logistic regression  [2 Classification]  the original project model
  SVM (RBF kernel)     [7 SVM]              non-linear decision boundary
  decision tree        [8 Decision Trees]   threshold rules, e.g. "big move"
  random forest        [9 Ensembles]        many trees averaged
  neural network (MLP) [10 Deep Learning]   small one-hidden-layer network

The decision tree depth is chosen with walk-forward validation
(see tree_depth_curve), which also shows over/underfitting [5, 6].
"""

import warnings

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.evaluation import fit_and_score, holdout_evaluate, walk_forward_evaluate
from src.lead_lag import build_pairs_table, lagged_correlation_matrix, p_value_matrix
from src.modeling import create_pair_dataset
from src.validation import split_by_date, walk_forward_folds

FEATURES = ["leader_return_previous_day"]
SEED = 42


def get_models(tree_depth=config.TREE_MAX_DEPTH):
    """
    The models we compare. Scaling is inside each pipeline, so it is fit on
    the training block only. (Trees don't need scaling.)
    min_samples_leaf keeps the trees from memorizing single days.
    """
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        # SVC has no probabilities of its own; the calibration wrapper turns
        # its decision scores into probabilities (needed for ROC-AUC).
        "SVM (RBF)": make_pipeline(
            StandardScaler(), CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, random_state=SEED), ensemble=False)),
        "Decision tree": DecisionTreeClassifier(
            max_depth=tree_depth, min_samples_leaf=50, random_state=SEED),
        "Random forest": RandomForestClassifier(
            n_estimators=200, max_depth=tree_depth, min_samples_leaf=50,
            random_state=SEED, n_jobs=-1),
        "Neural network (MLP)": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3,
                                            max_iter=500, random_state=SEED)),
    }


def select_pairs(returns, residuals, cutoff, tickers=config.TICKERS):
    """
    Stage 3 test on the development period only (days before `cutoff`).
    Returns the Bonferroni-significant pairs between different stocks.
    """
    dev_returns = returns.loc[returns.index < cutoff, tickers]
    dev_residuals = residuals.loc[residuals.index < cutoff]

    correlations, sample_sizes = lagged_correlation_matrix(dev_returns, dev_residuals)
    pairs = build_pairs_table(correlations, sample_sizes,
                              p_value_matrix(correlations, sample_sizes))

    selected = pairs[~pairs["same_stock"] & pairs["significant_bonf"]]
    return selected.drop(columns="same_stock").reset_index(drop=True), pairs


def pair_split(returns, residuals, leader, lagger, cutoff):
    """Pair dataset plus its development/holdout rows and walk-forward folds."""
    data = create_pair_dataset(returns, residuals, leader, lagger)
    X, y = data[FEATURES], data["target"]
    development, holdout = split_by_date(X.index, cutoff)
    folds = walk_forward_folds(len(development))
    return X, y, development, holdout, folds


def compare_models(returns, residuals, selected_pairs, cutoff, models):
    """
    Walk-forward + holdout results for every (pair, model).

    Returns:
      fold_results:        one row per pair x model x fold
      holdout_results:     one row per pair x model
      holdout_predictions: every holdout prediction (for confusion matrix / ROC)
    """
    fold_rows, holdout_rows, holdout_preds = [], [], []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        for leader, lagger in zip(selected_pairs["leader"], selected_pairs["lagger"]):
            X, y, development, holdout, folds = pair_split(
                returns, residuals, leader, lagger, cutoff)

            for name, model in models.items():
                tag = {"leader": leader, "lagger": lagger, "model": name}

                folds_df, _ = walk_forward_evaluate(model, X, y, folds)
                fold_rows.append(folds_df.assign(**tag))

                _, metrics, preds = holdout_evaluate(model, X, y, development, holdout)
                holdout_rows.append({**tag, **metrics})
                holdout_preds.append(preds.assign(**tag))

    return (pd.concat(fold_rows, ignore_index=True),
            pd.DataFrame(holdout_rows),
            pd.concat(holdout_preds))


def summarize(fold_results, holdout_results):
    """One row per model: averages over pairs (and folds), plus win counts."""
    cv = fold_results.groupby("model").agg(
        cv_train_accuracy=("train_accuracy", "mean"),
        cv_accuracy=("accuracy", "mean"),
        cv_baseline=("baseline_accuracy", "mean"),
        cv_auc=("auc", "mean"),
    )
    holdout = holdout_results.groupby("model").agg(
        holdout_accuracy=("accuracy", "mean"),
        holdout_baseline=("baseline_accuracy", "mean"),
        holdout_auc=("auc", "mean"),
        pairs=("accuracy", "size"),
        pairs_beating_baseline=("accuracy_minus_baseline", lambda d: int((d > 0).sum())),
    )
    order = list(get_models().keys())
    return cv.join(holdout).reindex(order).reset_index()


def tree_depth_curve(returns, residuals, selected_pairs, cutoff,
                     depths=config.TREE_DEPTHS):
    """
    Train vs walk-forward validation accuracy of a decision tree at each
    depth, averaged over pairs and folds. Used to pick the tree depth.
    (No min_samples_leaf here, so deep trees can show overfitting clearly.)
    Never touches the holdout period.
    """
    rows = []
    for leader, lagger in zip(selected_pairs["leader"], selected_pairs["lagger"]):
        X, y, _, _, folds = pair_split(returns, residuals, leader, lagger, cutoff)
        for depth in depths:
            tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
            for fold in folds:
                _, metrics, _ = fit_and_score(tree, X, y, fold.train, fold.validation)
                rows.append({
                    "depth": depth,
                    "train_accuracy": metrics["train_accuracy"],
                    "validation_accuracy": metrics["accuracy"],
                    "baseline_accuracy": metrics["baseline_accuracy"],
                })

    return pd.DataFrame(rows).groupby("depth").mean().reset_index()
