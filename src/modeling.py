import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def safe_roc_auc(y_true, probabilities):
    if y_true.nunique() < 2:
        return float("nan")
    return roc_auc_score(y_true, probabilities)


def create_pair_dataset(returns, residuals, leader, lagger):
    data = pd.DataFrame({
        "leader_return_previous_day": returns[leader].shift(1),
        "lagger_residual_today": residuals[lagger],
    })

    data = data.dropna()

    data["target"] = (
        data["lagger_residual_today"] > 0
    ).astype(int)

    return data

def train_pair_model(pair_data):
    if len(pair_data) < 5:
        raise ValueError("pair_data must contain at least 5 rows")

    X = pair_data[["leader_return_previous_day"]]
    y = pair_data["target"]

    n = len(pair_data)

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_validation = X.iloc[train_end:validation_end]
    y_validation = y.iloc[train_end:validation_end]

    X_test = X.iloc[validation_end:]
    y_test = y.iloc[validation_end:]

    if y_train.nunique() < 2:
        raise ValueError("training split must contain both target classes")
    if y_validation.empty or y_test.empty:
        raise ValueError("validation and test splits must contain at least one row")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)[:, 1]

    validation_accuracy = accuracy_score(y_validation, validation_predictions)
    validation_auc = safe_roc_auc(y_validation, validation_probabilities)
    model_accuracy = accuracy_score(y_test, test_predictions)
    model_auc = safe_roc_auc(y_test, test_probabilities)

    majority_class = y_train.mode().iloc[0]
    baseline_predictions = np.repeat(majority_class, len(y_test))
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    always_up_predictions = np.ones(len(y_test), dtype=int)
    always_up_accuracy = accuracy_score(y_test, always_up_predictions)

    repeat_sequence = pd.concat([
        y_validation.iloc[-1:],
        y_test,
    ]).shift(1).iloc[1:].astype(int)
    repeat_previous_accuracy = accuracy_score(y_test, repeat_sequence)

    metrics = {
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
        "testing_rows": len(X_test),
        "validation_accuracy": validation_accuracy,
        "validation_auc": validation_auc,
        "model_accuracy": model_accuracy,
        "model_auc": model_auc,
        "baseline_accuracy": baseline_accuracy,
        "baseline_always_up_accuracy": always_up_accuracy,
        "baseline_repeat_previous_accuracy": repeat_previous_accuracy,
    }

    return model, metrics


def train_all_pair_models(returns, residuals, tickers):
    rows = []
    models = {}

    for leader in tickers:
        for lagger in tickers:
            pair_data = create_pair_dataset(
                returns,
                residuals,
                leader=leader,
                lagger=lagger,
            )

            if len(pair_data) < 20 or pair_data["target"].nunique() < 2:
                continue

            train_end = int(len(pair_data) * 0.60)
            validation_end = int(len(pair_data) * 0.80)
            y_train = pair_data["target"].iloc[:train_end]
            y_validation = pair_data["target"].iloc[train_end:validation_end]
            y_test = pair_data["target"].iloc[validation_end:]
            if (
                y_train.nunique() < 2
                or y_validation.empty
                or y_test.empty
            ):
                continue

            model, metrics = train_pair_model(pair_data)
            rows.append({
                "leader": leader,
                "lagger": lagger,
                "same_stock": leader == lagger,
                **metrics,
            })
            models[(leader, lagger)] = model

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(
            ["model_accuracy", "model_auc"],
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return models, results
