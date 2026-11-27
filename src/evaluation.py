"""
Stage 5c: run a model through walk-forward validation and the holdout test.

Any scikit-learn classifier (or Pipeline) can be passed in. For every fold
we fit a fresh copy on the training block and score it on the validation
block. We always report a majority-class baseline next to the model,
because with ~50/50 targets an accuracy of 0.52 can still be no better
than always guessing the most common class.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score


def score_predictions(y_true, predictions, probabilities, y_train):
    """Accuracy, ROC-AUC, and the majority-class baseline for one block."""
    majority_class = int(y_train.mode().iloc[0])
    baseline = accuracy_score(y_true, np.full(len(y_true), majority_class))
    auc = (roc_auc_score(y_true, probabilities)
           if y_true.nunique() == 2 else float("nan"))

    accuracy = accuracy_score(y_true, predictions)
    return {
        "accuracy": accuracy,
        "auc": auc,
        "baseline_accuracy": baseline,
        "accuracy_minus_baseline": accuracy - baseline,
        "predicted_up_share": float(np.mean(predictions)),
    }


def fit_and_score(model, X, y, train_rows, eval_rows):
    """Fit a fresh copy of `model` on train_rows and score it on eval_rows."""
    fitted = clone(model)
    X_train, y_train = X.iloc[train_rows], y.iloc[train_rows]
    X_eval, y_eval = X.iloc[eval_rows], y.iloc[eval_rows]

    fitted.fit(X_train, y_train)
    predictions = fitted.predict(X_eval)
    probabilities = fitted.predict_proba(X_eval)[:, 1]

    metrics = score_predictions(y_eval, predictions, probabilities, y_train)
    # Accuracy on the data the model was fit on. A big gap between this and
    # the evaluation accuracy is the sign of overfitting.
    metrics["train_accuracy"] = accuracy_score(y_train, fitted.predict(X_train))
    metrics["train_days"] = len(train_rows)
    metrics["eval_days"] = len(eval_rows)

    eval_predictions = pd.DataFrame({
        "y_true": y_eval.values,
        "y_pred": predictions,
        "probability_up": probabilities,
    }, index=y_eval.index)

    return fitted, metrics, eval_predictions


def walk_forward_evaluate(model, X, y, folds):
    """
    Score `model` on every walk-forward fold.

    Returns (fold_results, validation_predictions):
      fold_results:           one row of metrics per fold
      validation_predictions: out-of-sample predictions from all validation
                              blocks stacked together (for confusion matrices
                              and ROC curves later)
    """
    rows, predictions = [], []
    for fold in folds:
        _, metrics, preds = fit_and_score(model, X, y, fold.train, fold.validation)
        rows.append({"fold": fold.number, **metrics})
        predictions.append(preds.assign(fold=fold.number))

    return pd.DataFrame(rows), pd.concat(predictions)


def holdout_evaluate(model, X, y, development_rows, holdout_rows):
    """
    Final test: fit on the whole development part, score once on the holdout.
    Call this only after model choices are finished.
    """
    return fit_and_score(model, X, y, development_rows, holdout_rows)
