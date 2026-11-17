import pandas as pd


def create_pair_dataset(returns, residuals, leader, lagger):
    data = pd.DataFrame({
        "leader_return_today": returns[leader],
        "lagger_residual_next_day": residuals[lagger].shift(-1),
    })

    data = data.dropna()

    data["target"] = (
        data["lagger_residual_next_day"] > 0
    ).astype(int)

    return data

