"""
Figures for the model evaluation stage.

  tree_depth_curve.png   train vs validation accuracy by tree depth
  confusion_matrices.png holdout confusion matrix for every model
  roc_curves.png         holdout ROC curve for every model
"""

import matplotlib
matplotlib.use("Agg")  # save plots to files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Fixed colors per model (same model = same color in every figure).
MODEL_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"]
TRAIN_COLOR, VALIDATION_COLOR = "#257de0", "#f47230"
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"

# One-hue light -> dark ramp for counts in the confusion matrices.
BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f7fd", "#84b9f3", "#1b64be"])


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tree_depth_curve(curve, filepath):
    """Overfitting picture: train accuracy rises with depth, validation does not."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(curve))

    ax.plot(x, curve["train_accuracy"], color=TRAIN_COLOR, linewidth=2,
            marker="o", markersize=7, label="Train accuracy")
    ax.plot(x, curve["validation_accuracy"], color=VALIDATION_COLOR, linewidth=2,
            marker="o", markersize=7, label="Validation accuracy (walk-forward)")
    ax.plot(x, curve["baseline_accuracy"], color=MUTED, linewidth=1.5,
            linestyle="--", label="Majority-class baseline")

    best = int(curve["validation_accuracy"].idxmax())
    ax.annotate(f"best validation: depth {curve['depth'][best]}",
                (x[best], curve["validation_accuracy"][best]),
                xytext=(0, -22), textcoords="offset points", ha="center",
                color=TEXT, fontsize=9)

    low = min(curve["validation_accuracy"].min(), curve["baseline_accuracy"].min())
    ax.set_ylim(bottom=low - 0.03)  # room for the "best validation" label
    ax.set_xticks(x, curve["depth"].astype(str))
    ax.set_xlabel("Decision tree max depth", color=TEXT)
    ax.set_ylabel("Accuracy", color=TEXT)
    ax.set_title("Deeper trees fit the training data, not the future",
                 color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    _save(fig, filepath)


def plot_confusion_matrices(holdout_predictions, model_names, filepath):
    """Holdout confusion matrix per model, all selected pairs pooled together."""
    fig, axes = plt.subplots(1, len(model_names), figsize=(3.2 * len(model_names), 3.6))

    for ax, name in zip(np.atleast_1d(axes), model_names):
        preds = holdout_predictions[holdout_predictions["model"] == name]
        matrix = confusion_matrix(preds["y_true"], preds["y_pred"], labels=[0, 1])
        row_share = matrix / matrix.sum(axis=1, keepdims=True)

        ax.imshow(matrix, cmap=BLUES, vmin=0)
        for i in range(2):
            for j in range(2):
                dark = matrix[i, j] > matrix.max() * 0.6
                ax.text(j, i, f"{matrix[i, j]}\n({row_share[i, j]:.0%})",
                        ha="center", va="center", fontsize=9,
                        color="white" if dark else TEXT)

        accuracy = np.trace(matrix) / matrix.sum()
        ax.set_title(f"{name}\naccuracy {accuracy:.3f}", fontsize=9, color=TEXT)
        ax.set_xticks([0, 1], ["Down", "Up"])
        ax.set_yticks([0, 1], ["Down", "Up"])
        ax.set_xlabel("Predicted", fontsize=9, color=MUTED)
        ax.set_ylabel("Actual", fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Holdout confusion matrices (all selected pairs pooled; "
                 "row % = share of actual class)", color=TEXT, fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout()
    _save(fig, filepath)


def plot_roc_curves(holdout_predictions, model_names, filepath):
    """Holdout ROC curve per model. The diagonal is random guessing (AUC 0.5)."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.5,
            label="Random guess (AUC 0.500)")

    for name, color in zip(model_names, MODEL_COLORS):
        preds = holdout_predictions[holdout_predictions["model"] == name]
        fpr, tpr, _ = roc_curve(preds["y_true"], preds["probability_up"])
        auc = roc_auc_score(preds["y_true"], preds["probability_up"])
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC {auc:.3f})")

    ax.set_xlabel("False positive rate", color=TEXT)
    ax.set_ylabel("True positive rate", color=TEXT)
    ax.set_title("Holdout ROC curves (all selected pairs pooled)",
                 color=TEXT, loc="left")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    _save(fig, filepath)
