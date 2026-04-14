# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from typing import Tuple


def similarity(
    adata: AnnData,
    thresholds: np.ndarray | None = None,
    figsize: Tuple[float, float] = (4.0, 3.0),
    color: str = "#2b5566",
    fontsize: int = 8,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot min_similarity threshold versus number of factors retained.

    Use this plot to select a min_similarity threshold before running
    juzi.gp.select_similarity. The curve shows how many factors survive
    at each threshold.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_similarity in .uns, produced by
        juzi.gp.similarity.
    thresholds : np.ndarray | None
        Array of similarity thresholds to evaluate. If None, uses
        np.linspace(0, 1, 100).
    figsize : Tuple[float, float]
        Figure size in inches.
    color : str
        Line and marker colour.
    fontsize : int
        Font size for all text elements.
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """
    if "juzi_similarity" not in adata.uns:
        raise KeyError(
            "'juzi_similarity' not found in .uns. " "Run juzi.gp.similarity first."
        )

    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)

    # Compute retention curve

    sim = adata.uns["juzi_similarity"]
    max_per_row = sim.max(axis=1)

    # Start from drop_zeros mask if available
    base_mask = adata.uns.get(
        "juzi_keep_similarity",
        np.ones(sim.shape[0], dtype=bool),
    )

    n_retained = np.array([(base_mask & (max_per_row >= t)).sum() for t in thresholds])

    # Figure setup

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot

    ax.plot(
        thresholds,
        n_retained,
        color=color,
        linewidth=1.0,
        solid_capstyle="round",
    )

    ax.set_xlabel("Min similarity threshold", fontsize=fontsize)
    ax.set_ylabel("Factors retained", fontsize=fontsize)
    ax.tick_params(axis="both", length=2, labelsize=fontsize)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.5)

    if created_fig:
        fig.tight_layout()

    return ax
