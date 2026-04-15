# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from typing import Tuple


def threshold(
    adata: AnnData,
    figsize: Tuple[float, float] = (4.0, 3.0),
    color: str = "#2b5566",
    optimal_color: str = "#c94040",
    fontsize: int = 8,
    show_optimal: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the cluster threshold sweep results from juzi.gp.select_threshold.

    Displays the contrast metric versus threshold curve and marks the
    optimal threshold selected by juzi.gp.select_threshold.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_threshold_sweep in .uns, produced by
        juzi.gp.select_threshold.
    figsize : Tuple[float, float]
        Figure size in inches.
    color : str
        Line colour for the metric curve.
    optimal_color : str
        Colour for the vertical line marking the optimal threshold.
    fontsize : int
        Font size for all text elements.
    show_optimal : bool
        If True, draw a vertical line at the optimal threshold.
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """
    if "juzi_threshold_sweep" not in adata.uns:
        raise KeyError(
            "'juzi_threshold_sweep' not found in .uns. "
            "Run juzi.gp.select_threshold first."
        )

    # Extract sweep results

    sweep = adata.uns["juzi_threshold_sweep"]
    thresholds = sweep["thresholds"]
    metric = sweep["metric"]
    metric_name = sweep["metric_name"]
    optimal = sweep["optimal"]

    # Figure setup

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot metric curve

    # Only plot valid (non-nan) values
    valid = ~np.isnan(metric)
    ax.plot(
        thresholds[valid],
        metric[valid],
        color=color,
        linewidth=1.0,
        solid_capstyle="round",
    )

    # Mark optimal threshold

    if show_optimal:
        ax.axvline(
            optimal,
            color=optimal_color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.8,
            label=f"Optimal = {optimal:.3f}",
        )
        ax.legend(
            fontsize=fontsize - 1,
            frameon=False,
            loc="upper right",
        )

    # Axis labels

    metric_labels = {
        "ratio": "Inner / outer similarity",
        "delta": "Inner − outer similarity",
        "silhouette": "Mean silhouette score",
    }

    ax.set_xlabel("Clustering threshold", fontsize=fontsize)
    ax.set_ylabel(metric_labels.get(metric_name, metric_name), fontsize=fontsize)
    ax.tick_params(axis="both", length=2, labelsize=fontsize)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.5)

    if created_fig:
        fig.tight_layout()

    return ax
