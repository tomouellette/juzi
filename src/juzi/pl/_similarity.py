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
    """Plot min_similarity threshold versus factors retained and the
    maximum similarity distribution per factor.

    Two panels are shown side by side:
        Left  — retention curve: number of factors retained at each
                min_similarity threshold.
        Right — histogram of maximum similarity per factor. When
                show_gmm=True, a two-component Gaussian mixture model
                is fitted to the distribution and overlaid. The crossover
                point between the noise and signal components is marked
                as a suggested threshold. A bimodal distribution indicates
                a natural gap — the valley is a principled threshold choice.

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
        crossover threshold on the histogram. A warning is raised if the
        two components have similar means, indicating the distribution
        may not be bimodal and the suggested threshold may be unreliable.
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

    # Compute max similarity per factor

    sim = adata.uns["juzi_similarity"]
    sim_idx = adata.uns["juzi_similarity_idx"]
    n_total = len(adata.uns["juzi_names"])

    max_per_factor = np.zeros(n_total)
    max_per_factor[sim_idx] = sim.max(axis=1)

    base_mask = adata.uns.get(
        "juzi_keep_similarity",
        np.zeros(n_total, dtype=bool),
    )

    n_retained = np.array(
        [(base_mask & (max_per_factor >= t)).sum() for t in thresholds]
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

    if show_gmm and len(max_sim_values) >= 4:
        gmm_threshold = _fit_gmm(
            values=max_sim_values,
            ax=ax_hist,
            bin_edges=bin_edges,
            color=color,
            fontsize=fontsize,
        )

        # Mark suggested threshold on retention curve too
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

    n_in_sim = len(sim_idx)
    n_base = int(base_mask.sum())

    return ax_retention, ax_hist


def _fit_gmm(
    values: np.ndarray,
    ax: plt.Axes,
    bin_edges: np.ndarray,
    color: str,
    fontsize: int,
) -> float | None:
    """Fit a two-component GMM to max similarity values, overlay on ax,
    and return the crossover threshold between noise and signal components.

    Parameters
    ----------
    values : np.ndarray
        Max similarity values per factor.
    ax : plt.Axes
        Axes to draw the GMM overlay on.
    bin_edges : np.ndarray
        Bin edges from the histogram — used to scale density to counts.
    color : str
        Base colour for component curves.
    fontsize : int
        Font size for the legend.

    Returns
    -------
    float | None
        The crossover threshold or None if the GMM fit is unreliable.
    """
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

    # Warn if components are not well separated
    if np.abs(means[noise_c] - means[sig_c]) < 0.05:
        warnings.warn(
            f"GMM components have similar means ({means[noise_c]:.3f} and "
            f"{means[sig_c]:.3f}) — distribution may not be bimodal. "
            "The suggested threshold may be unreliable. "
            "Inspect the histogram before applying.",
            UserWarning,
            stacklevel=3,
        )

    # Scale density to histogram counts
    bin_width = bin_edges[1] - bin_edges[0]
    n = len(values)
    x_grid = np.linspace(0, 1, 500)

    for c_idx, (linestyle, label) in enumerate(zip(["--", "-"], ["Noise", "Signal"])):
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

    # Find crossover where P(signal | x) >= P(noise | x)
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
