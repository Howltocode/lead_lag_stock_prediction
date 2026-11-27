"""
Stage 6: cluster the stocks by how their residual returns move together.

IDEA
----
Each stock is described by its correlation with every other stock's
residual return (a 49-number vector). K-means groups stocks with similar
vectors, i.e. stocks that tend to move together after the market move is
removed. The clustering never sees the sector labels; we compare with
them afterwards to check whether the clusters are meaningful.

Why residual returns and not raw returns? Raw returns are dominated by the
market: on most days nearly everything moves together, so every stock looks
alike. Removing the market part (Stage 2b) lets industry groups show up.

Only the development period (before the holdout cutoff) is used, the same
as the rest of the evaluation stage.

Choosing k: inertia (elbow) and silhouette score for k = 2..12.
"""

import matplotlib
matplotlib.use("Agg")  # save plots to files without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src import config

SEED = 42
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
# Diverging: blue (negative) <-> gray (zero) <-> red (positive)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#1b64be", "#84b9f3", "#efefef", "#f08a8b", "#c42f30"])


def residual_correlations(residuals, cutoff):
    """Correlation matrix of residual returns over the development period."""
    return residuals.loc[residuals.index < cutoff].corr()


def fit_kmeans(correlations, k):
    """K-means on the rows of the correlation matrix. Returns cluster labels."""
    model = KMeans(n_clusters=k, n_init=20, random_state=SEED)
    labels = model.fit_predict(correlations.values)
    return pd.Series(labels, index=correlations.index, name=f"cluster_k{k}"), model


def choose_k(correlations, k_values=range(2, 13)):
    """Inertia (for the elbow plot) and silhouette score for each k."""
    rows = []
    for k in k_values:
        labels, model = fit_kmeans(correlations, k)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(correlations.values, labels),
        })
    return pd.DataFrame(rows)


def renumber_by_sector(labels, sectors):
    """
    K-means cluster numbers are arbitrary. Renumber them so clusters are
    ordered by their most common sector, which makes tables easier to read.
    """
    majority = labels.groupby(labels).apply(
        lambda c: pd.Series([sectors[t] for t in c.index]).mode().iloc[0])
    order = majority.sort_values(kind="stable").index
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return labels.map(mapping)


def cluster_table(labels_by_k, sectors):
    """One row per stock: sector and cluster for each k."""
    table = pd.DataFrame(labels_by_k)
    table.insert(0, "sector", [sectors[t] for t in table.index])
    table.index.name = "ticker"
    return table.sort_values(list(table.columns[1:]) + ["sector"])


def sector_purity(labels, sectors):
    """Share of stocks whose cluster's most common sector is their own sector."""
    sector = pd.Series({t: sectors[t] for t in labels.index})
    majority = sector.groupby(labels).transform(lambda s: s.mode().iloc[0])
    return float((sector == majority).mean())


def lead_lag_by_cluster(dev_pairs, labels):
    """
    Are lead-lag relationships stronger inside a cluster than across clusters?
    dev_pairs: the development-period pairs table from select_pairs().
    """
    pairs = dev_pairs[~dev_pairs["same_stock"]].copy()
    pairs["same_cluster"] = (pairs["leader"].map(labels).values
                             == pairs["lagger"].map(labels).values)
    pairs["abs_correlation"] = pairs["correlation"].abs()
    return pairs.groupby("same_cluster").agg(
        pairs=("abs_correlation", "size"),
        mean_abs_correlation=("abs_correlation", "mean"),
        bonferroni_pairs=("significant_bonf", "sum"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_choose_k(scores, chosen_k, filepath):
    """Elbow (inertia) and silhouette side by side: two charts, one axis each."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

    left.plot(scores["k"], scores["inertia"], color="#257de0", linewidth=2,
              marker="o", markersize=7)
    left.set_title("Elbow: inertia (lower = tighter clusters)", color=TEXT,
                   loc="left", fontsize=10)
    left.set_ylabel("Inertia", color=TEXT)

    right.plot(scores["k"], scores["silhouette"], color="#257de0", linewidth=2,
               marker="o", markersize=7)
    right.set_title("Silhouette score (higher = better separated)", color=TEXT,
                    loc="left", fontsize=10)
    right.set_ylabel("Silhouette", color=TEXT)

    for ax in (left, right):
        ax.axvline(chosen_k, color=MUTED, linestyle="--", linewidth=1.2)
        ax.set_xlabel("Number of clusters k", color=TEXT)
        ax.set_xticks(scores["k"])
        _style(ax)
    right.annotate(f"k = {chosen_k} (used)", (chosen_k, scores["silhouette"].max()),
                   xytext=(6, 0), textcoords="offset points", color=TEXT, fontsize=9)

    fig.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


CLUSTER_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac",
                  "#008a17", "#6a4ce2", "#ec4647"]


def _label_points(ax, fig, names, points):
    """
    Put a ticker label next to every dot, trying a few positions around the
    dot and taking the first one that does not overlap a label already placed.
    """
    candidates = [(5, 4), (5, -11), (-6, 4), (-6, -11), (5, 13), (-6, 13),
                  (5, -20), (-6, -20)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = []
    for name, (x, y) in zip(names, points):
        for dx, dy in candidates:
            label = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points",
                                ha="left" if dx > 0 else "right", fontsize=8, color=TEXT)
            box = label.get_window_extent(renderer).expanded(1.05, 1.1)
            if not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
            label.remove()
        else:  # every position was taken: keep the default one
            ax.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points",
                        fontsize=8, color=TEXT)


def plot_cluster_scatter(correlations, labels, filepath, cluster_names=None):
    """
    Every stock as a dot, colored by K-means cluster.

    Each stock is really a 49-number vector, so we use PCA to squash it to
    2 numbers just for drawing. PCA does not change the clusters; it only
    picks the 2 directions that keep the most spread between the stocks.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=SEED)
    points = pca.fit_transform(correlations.values)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for number, color in zip(sorted(labels.unique()), CLUSTER_COLORS):
        mask = (labels == number).values
        name = (cluster_names or {}).get(number, f"Cluster {number}")
        ax.scatter(points[mask, 0], points[mask, 1], s=70, color=color,
                   edgecolor="white", linewidth=1.5, label=name, zorder=3)

    _label_points(ax, fig, correlations.index, points)

    ax.set_xlabel(f"PCA component 1 ({explained[0]:.0%} of variance)", color=TEXT)
    ax.set_ylabel(f"PCA component 2 ({explained[1]:.0%} of variance)", color=TEXT)
    ax.set_title(f"Stocks clustered by residual-return correlation "
                 f"(K-means, k = {labels.nunique()})", color=TEXT, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="best")
    _style(ax)

    fig.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clustered_heatmap(correlations, labels, sectors, filepath, fine_labels=None):
    """
    Residual-return correlation matrix with stocks ordered by cluster.
    Stocks that move together form blocks along the diagonal.
    Lines mark the main clusters; inside each one, stocks are ordered by the
    finer clustering (if given) so smaller blocks are visible too.
    """
    fine = fine_labels if fine_labels is not None else labels
    order = sorted(labels.index, key=lambda t: (labels[t], fine[t], sectors[t], t))
    matrix = correlations.loc[order, order].values.copy()
    np.fill_diagonal(matrix, np.nan)  # self-correlation (1.0) would dominate the scale
    limit = np.nanmax(np.abs(matrix))

    fig, ax = plt.subplots(figsize=(11, 10))
    image = ax.imshow(matrix, cmap=DIVERGING, vmin=-limit, vmax=limit)

    # Cluster boundaries
    sizes = labels.loc[order].groupby(labels.loc[order], sort=False).size().values
    edges = np.cumsum(sizes)[:-1] - 0.5
    for edge in edges:
        ax.axhline(edge, color=TEXT, linewidth=1)
        ax.axvline(edge, color=TEXT, linewidth=1)

    names = [f"{t} ({sectors[t][:4]})" for t in order]
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7, color=TEXT)
    ax.set_yticks(range(len(order)), names, fontsize=7, color=TEXT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(image, ax=ax, shrink=0.7)
    bar.set_label("Residual return correlation", color=TEXT)
    ax.set_title(f"Residual-return correlations, stocks ordered by K-means cluster "
                 f"(k = {labels.nunique()})", color=TEXT, loc="left")

    fig.tight_layout()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def run_clustering(residuals, cutoff, dev_pairs, tables_dir, figures_dir,
                   k=config.N_CLUSTERS, fine_k=config.N_CLUSTERS_FINE,
                   sectors=config.SECTORS):
    """Stage 6: choose k, cluster, compare with sectors, save tables and figures."""
    correlations = residual_correlations(residuals, cutoff)

    scores = choose_k(correlations)
    scores.to_csv(tables_dir / "cluster_k_scores.csv", index=False)
    plot_choose_k(scores, k, figures_dir / "cluster_choose_k.png")

    labels = {}
    for kk in (k, fine_k):
        raw, _ = fit_kmeans(correlations, kk)
        labels[kk] = renumber_by_sector(raw, sectors)

    table = cluster_table({f"cluster_k{kk}": lab for kk, lab in labels.items()}, sectors)
    table.to_csv(tables_dir / "clusters.csv")

    plot_cluster_scatter(correlations, labels[k], figures_dir / "cluster_scatter.png",
                         cluster_names=config.CLUSTER_NAMES if k == 3 else None)
    plot_clustered_heatmap(correlations, labels[k], sectors,
                           figures_dir / "cluster_heatmap.png", fine_labels=labels[fine_k])

    by_cluster = lead_lag_by_cluster(dev_pairs, labels[k])
    by_cluster.to_csv(tables_dir / "lead_lag_by_cluster.csv", index=False)

    purity = {kk: sector_purity(lab, sectors) for kk, lab in labels.items()}
    return scores, table, by_cluster, purity
