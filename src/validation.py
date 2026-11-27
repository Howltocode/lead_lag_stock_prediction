"""
Stage 5b: walk-forward validation for time-series data.

WHY NOT A NORMAL K-FOLD?
------------------------
Regular k-fold shuffles days into folds, so a model could be trained on
2024 and validated on 2021. With time series that lets the future leak
into training and makes results look better than they would be in real
use. Walk-forward validation always trains on the past and validates on
the period right after it.

LAYOUT
------
The data is split into two parts by date:

  |<----------- development ----------->|<-- holdout test -->|

1. Development part: used for walk-forward validation (model selection,
   hyperparameter tuning). With an expanding window:

     fold 1:  [train      ][val ]
     fold 2:  [train            ][val ]
     fold 3:  [train                  ][val ]
     ...

   Each fold trains on everything before its validation block.

2. Holdout test part: the most recent data. It is touched only ONCE, at
   the very end, after every choice has been made. This is the honest
   estimate of how the final model would do on unseen data.
"""

from dataclasses import dataclass

import numpy as np

from src import config


@dataclass
class Fold:
    """Row positions (not dates) for one walk-forward fold."""
    number: int
    train: np.ndarray
    validation: np.ndarray


def split_holdout(n_rows, holdout_fraction=config.HOLDOUT_FRACTION):
    """
    Split row positions 0..n_rows-1 into development and holdout parts.
    The holdout is the LAST `holdout_fraction` of the rows (most recent days).
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    holdout_start = int(n_rows * (1 - holdout_fraction))
    return np.arange(holdout_start), np.arange(holdout_start, n_rows)


def walk_forward_folds(n_rows, initial_train=config.INITIAL_TRAIN_DAYS,
                       validation_size=config.VALIDATION_DAYS,
                       expanding=True, gap=0):
    """
    Walk-forward folds over row positions 0..n_rows-1 (the development part).

    initial_train:   rows in the first training window
    validation_size: rows in each validation block
    expanding:       True  -> training window grows every fold
                     False -> training window slides (fixed size)
    gap:             rows skipped between train and validation (0 is fine
                     here because every feature is already lagged by a day)

    The last fold is kept if it has at least half a validation block.
    """
    if initial_train + gap >= n_rows:
        raise ValueError("not enough rows for even one fold; lower initial_train")

    folds = []
    train_end = initial_train
    number = 1
    while True:
        val_start = train_end + gap
        val_end = min(val_start + validation_size, n_rows)
        if val_end - val_start < max(1, validation_size // 2):
            break

        train_start = 0 if expanding else train_end - initial_train
        folds.append(Fold(
            number=number,
            train=np.arange(train_start, train_end),
            validation=np.arange(val_start, val_end),
        ))

        if val_end == n_rows:
            break
        train_end += validation_size
        number += 1

    return folds


def describe_folds(index, folds, holdout):
    """Readable summary (with dates) of the folds and the holdout period."""
    lines = []
    for f in folds:
        lines.append(
            f"  fold {f.number}: train {index[f.train[0]].date()} -> "
            f"{index[f.train[-1]].date()} ({len(f.train)} days) | "
            f"val {index[f.validation[0]].date()} -> "
            f"{index[f.validation[-1]].date()} ({len(f.validation)} days)"
        )
    lines.append(
        f"  holdout: {index[holdout[0]].date()} -> {index[holdout[-1]].date()} "
        f"({len(holdout)} days)"
    )
    return "\n".join(lines)
