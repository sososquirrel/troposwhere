"""
sensitivity_test_plot_functions.py — plotting helpers for cluster-count sensitivity figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


# Qualitative palette that scales to any number of states
_PALETTE_10 = [
    '#bcbcdd', '#ff9895', '#ffe066', '#1e78b3', '#9be7c4',
    '#5ec962', '#e02222', '#f4a261', '#2ec4b6', '#e76f51',
]


def _make_cmap(num_states):
    colors = _PALETTE_10[:num_states]
    cmap   = ListedColormap(colors)
    norm   = Normalize(vmin=0, vmax=num_states - 1)
    return cmap, norm, colors


def plot_pca_clusters(
    emb_pca: np.ndarray,
    states: np.ndarray,
    num_states: int,
    pca=None,
    intervals: list = None,
    smooth_win: int = 7,
    ax=None,
    scatter_alpha: float = 0.35,
    scatter_s: float = 8,
) -> None:
    """
    Scatter plot of 2-D PCA embeddings coloured by cluster assignment.

    Parameters
    ----------
    emb_pca     : (N, 2) PCA-projected embeddings
    states      : (N,)   integer cluster labels
    num_states  : total number of clusters (for colour palette)
    pca         : fitted sklearn PCA object (used to label axes with explained variance)
    intervals   : list of (start, end) index pairs to overlay as trajectory lines
    smooth_win  : box-car window for trajectory smoothing
    ax          : matplotlib Axes (created if None)
    """
    if ax is None:
        _, ax = plt.subplots()

    cmap, norm, colors = _make_cmap(num_states)

    ax.scatter(
        emb_pca[:, 0], emb_pca[:, 1],
        c=states, cmap=cmap, norm=norm,
        s=scatter_s, alpha=scatter_alpha,
        linewidth=0,
    )

    if intervals:
        for start, end in intervals:
            seg = emb_pca[start:end]
            if smooth_win > 1 and len(seg) >= smooth_win:
                pad = smooth_win // 2
                seg_pad = np.pad(seg, ((pad, pad), (0, 0)), mode='edge')
                kernel  = np.ones(smooth_win) / smooth_win
                xs = np.convolve(seg_pad[:, 0], kernel, mode='valid')
                ys = np.convolve(seg_pad[:, 1], kernel, mode='valid')
                seg = np.column_stack([xs, ys])
            ax.plot(seg[:, 0], seg[:, 1], color='black', lw=1.2, alpha=0.7, zorder=5)

    xlab = "PCA 1"
    ylab = "PCA 2"
    if pca is not None and hasattr(pca, 'explained_variance_ratio_'):
        ev = pca.explained_variance_ratio_
        xlab = f"PCA 1 ({ev[0]*100:.1f}%)"
        ylab = f"PCA 2 ({ev[1]*100:.1f}%)"

    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
