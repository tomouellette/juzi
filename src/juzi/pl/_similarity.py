# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from anndata import AnnData
from typing import Tuple


def similarity(
    adata: AnnData,
    thresholds: np.ndarray | None = None,
    figsize: Tuple[float, float] = (7.0, 3.0),
    color: str = "#2b5566",
    fontsize: int = 8,
    bins: int = 50,
    show_gmm: bool = True,
    ax_retention: plt.Axes | None = None,
    ax_hist: plt.Axes | None = None,
) -> Tuple[plt.Axes, plt.Axes]:
    """Plot similarity filtering diagnostics.

    Two panels are shown side by side:

        Left  — retention curve: number of factors retained at each
                min_similarity threshold, computed from the current
                similarity matrix.

        Right — histogram of maximum similarity per factor. When
                show_gmm=True, a two-component Gaussian mixture model
                is fitted to the distribution and overlaid. The crossover
                point between the lower- and higher-similarity components
                is marked as a suggested threshold.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_similarity and juzi_similarity_idx in .uns,
        produced by juzi.gp.similarity.
    thresholds : np.ndarray | None
        Array of similarity thresholds for the retention curve. If None,
        uses np.linspace(0, 1, 100).
    figsize : Tuple[float, float]
        Figure size in inches.
    color : str
        Line and bar colour.
    fontsize : int
        Font size for all text elements.
    bins : int
        Number of histogram bins for the max similarity distribution.
    show_gmm : bool
        If True, fit a two-component Gaussian mixture model to the max
        similarity distribution and overlay the component densities and
        crossover threshold on the histogram.
    ax_retention : plt.Axes | None
        Axes for the retention curve. Both ax_retention and ax_hist must
        be provided together or both None.
    ax_hist : plt.Axes | None
        Axes for the histogram.

    Returns
    -------
    Tuple[plt.Axes, plt.Axes]
        (ax_retention, ax_hist) — the two matplotlib Axes objects.
    """
    for field in ["juzi_similarity", "juzi_similarity_idx", "juzi_names"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. " "Run juzi.gp.similarity first."
            )

    if (ax_retention is None) != (ax_hist is None):
        raise ValueError("Provide both ax_retention and ax_hist, or neither.")

    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)

    # Compute max similarity per factor in global factor space

    sim = np.array(adata.uns["juzi_similarity"], dtype=float)
    sim_idx = np.array(adata.uns["juzi_similarity_idx"], dtype=int)
    n_total = len(adata.uns["juzi_names"])

    max_per_factor = np.zeros(n_total, dtype=float)
    if sim.shape[0] > 0:
        max_per_factor[sim_idx] = sim.max(axis=1)

    # Base eligibility mask:
    # factors that actually entered similarity space
    in_similarity = np.zeros(n_total, dtype=bool)
    in_similarity[sim_idx] = True

    # Use current similarity keep mask if present, otherwise all factors
    # in similarity space are considered eligible
    keep_similarity = adata.uns.get(
        "juzi_keep_similarity",
        in_similarity.copy(),
    )

    # Retention curve:
    # preserve any upstream exclusions outside similarity filtering
    keep_prune = adata.uns.get("juzi_keep_prune", np.ones(n_total, dtype=bool))
    eligible_mask = in_similarity & keep_prune

    n_retained = np.array(
        [(eligible_mask & (max_per_factor >= t)).sum() for t in thresholds],
        dtype=int,
    )

    max_sim_values = max_per_factor[sim_idx]

    # Figure setup

    created_fig = ax_retention is None
    if created_fig:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
        ax_retention = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])

    # Left panel — retention curve

    ax_retention.plot(
        thresholds,
        n_retained,
        color=color,
        linewidth=1.0,
        solid_capstyle="round",
    )

    # Mark currently active filtering threshold if available
    current_min_similarity = None
    if "juzi_similarity_meta" in adata.uns:
        current_min_similarity = adata.uns["juzi_similarity_meta"].get("min_similarity")

    if current_min_similarity is not None:
        ax_retention.axvline(
            current_min_similarity,
            color="black",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
        )

    ax_retention.set_xlabel("Min similarity threshold", fontsize=fontsize)
    ax_retention.set_ylabel("Factors retained", fontsize=fontsize)
    ax_retention.tick_params(axis="both", length=2, labelsize=fontsize)

    for spine in ["top", "right"]:
        ax_retention.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_retention.spines[spine].set_linewidth(0.5)

    # Right panel — max similarity distribution

    counts, bin_edges, _ = ax_hist.hist(
        max_sim_values,
        bins=bins,
        color=color,
        edgecolor="none",
        alpha=0.85,
    )

    # GMM overlay

    gmm_threshold = None
    if show_gmm and len(max_sim_values) >= 4:
        gmm_threshold = _fit_gmm(
            values=max_sim_values,
            ax=ax_hist,
            bin_edges=bin_edges,
            color=color,
            fontsize=fontsize,
        )

        if gmm_threshold is not None:
            ax_retention.axvline(
                gmm_threshold,
                color="darkred",
                linewidth=0.8,
                linestyle=":",
                alpha=0.7,
            )

    ax_hist.set_xlabel("Max similarity per factor", fontsize=fontsize)
    ax_hist.set_ylabel("Count", fontsize=fontsize)
    ax_hist.tick_params(axis="both", length=2, labelsize=fontsize)

    for spine in ["top", "right"]:
        ax_hist.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_hist.spines[spine].set_linewidth(0.5)

    # Small annotation block

    n_in_sim = int(in_similarity.sum())
    n_kept = int(keep_similarity.sum())

    ax_hist.text(
        0.98,
        0.98,
        f"n in similarity = {n_in_sim}\n" f"n kept = {n_kept}",
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize - 1,
        color="black",
    )

    return ax_retention, ax_hist


def _fit_gmm(
    values: np.ndarray,
    ax: plt.Axes,
    bin_edges: np.ndarray,
    color: str,
    fontsize: int,
) -> float | None:
    """Fit a two-component GMM to max similarity values and return the crossover."""
    try:
        from sklearn.mixture import GaussianMixture
        from scipy.stats import norm
    except ImportError:
        warnings.warn(
            "sklearn and scipy are required for GMM fitting. "
            "Install them or set show_gmm=False.",
            UserWarning,
            stacklevel=3,
        )
        return None

    X = values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=5)
    gmm.fit(X)

    means = gmm.means_.flatten()
    noise_c = int(np.argmin(means))
    sig_c = int(np.argmax(means))

    if np.abs(means[noise_c] - means[sig_c]) < 0.05:
        warnings.warn(
            f"GMM components have similar means ({means[noise_c]:.3f} and "
            f"{means[sig_c]:.3f}) — distribution may not be bimodal. "
            "The suggested threshold may be unreliable.",
            UserWarning,
            stacklevel=3,
        )

    bin_width = bin_edges[1] - bin_edges[0]
    n = len(values)
    x_grid = np.linspace(0, 1, 500)

    for c_idx, (linestyle, label) in enumerate(
        zip(["--", "-"], ["Lower-similarity", "Higher-similarity"])
    ):
        c = noise_c if c_idx == 0 else sig_c
        mu = gmm.means_[c, 0]
        sigma = np.sqrt(gmm.covariances_[c, 0, 0])
        w = gmm.weights_[c]
        density_scaled = w * norm.pdf(x_grid, mu, sigma) * n * bin_width

        ax.plot(
            x_grid,
            density_scaled,
            color=["black", "darkorange"][c_idx],
            linewidth=0.8,
            linestyle=linestyle,
            alpha=0.8,
            label=label,
        )

    grid = x_grid.reshape(-1, 1)
    proba = gmm.predict_proba(grid)
    cross_idx = np.argmax(proba[:, sig_c] >= proba[:, noise_c])
    threshold = float(x_grid[cross_idx])

    ax.axvline(
        threshold,
        color="darkred",
        linewidth=0.8,
        linestyle=":",
        alpha=0.8,
        label=f"GMM threshold = {threshold:.3f}",
    )

    ax.legend(
        fontsize=fontsize - 1,
        frameon=False,
        loc="upper left",
        handlelength=1.5,
        labelspacing=0.3,
    )

    return threshold
