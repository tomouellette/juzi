# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from typing import Literal, Tuple


def programs_gene_overlap(
    adata: AnnData,
    metric: Literal["jaccard", "overlap_coefficient"] = "jaccard",
    figsize: Tuple[float, float] = (5.5, 5.0),
    ax: plt.Axes | None = None,
    cmap: str | None = None,
    fontsize: int = 8,
    rasterized: bool = False,
    show_values: bool = False,
    value_fmt: str = ".2f",
    cbar_pad: float = 0.03,
    cbar_aspect: float = 12.0,
    cbar_shrink: float = 0.8,
    cbar_ticks: list[float] = [0.0, 0.5, 1.0],
    cbar_tick_length: float = 0.0,
    cbar_label: str | None = None,
    cbar_legend_pos: str = "top",
    cbar_labelpad: float = 8.0,
) -> plt.Axes:
    """Plot pairwise overlap between canonical program gene sets.

    Displays a program-by-program heatmap computed from
    `adata.uns["juzi_cluster_genes"]`, which is the canonical program
    definition for both centroid and progressive clustering.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_genes and juzi_cluster_labels in .uns,
        produced by juzi.gp.programs_cluster.
    metric : {"jaccard", "overlap_coefficient"}
        Overlap metric to compute:
            - "jaccard" : |A ∩ B| / |A ∪ B|
            - "overlap_coefficient" : |A ∩ B| / min(|A|, |B|)
    figsize : Tuple[float, float]
        Figure size in inches.
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.
    cmap : str | None
        Colormap for overlap values. If None, a default white-to-teal
        colormap is used.
    fontsize : int
        Font size for all text elements.
    rasterized : bool
        If True, rasterize the heatmap for performance.
    show_values : bool
        If True, annotate each cell with the overlap value.
    value_fmt : str
        Format string for cell annotations, e.g. ".2f".
    cbar_pad : float
        Padding between heatmap and colorbar.
    cbar_aspect : float
        Aspect ratio of the colorbar.
    cbar_shrink : float
        Fraction by which to shrink the colorbar.
    cbar_ticks : list[float]
        Tick positions on the colorbar.
    cbar_tick_length : float
        Length of colorbar ticks.
    cbar_label : str | None
        Label for the colorbar. If None, inferred from metric.
    cbar_legend_pos : str
        Position of the colorbar label ("top" or "bottom").
    cbar_labelpad : float
        Padding between colorbar and its label.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """
    for field in ["juzi_cluster_genes", "juzi_cluster_labels"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. "
                "Run juzi.gp.programs_cluster before plotting gene overlap."
            )

    if metric not in ("jaccard", "overlap_coefficient"):
        raise ValueError("metric must be 'jaccard' or 'overlap_coefficient'.")

    cluster_genes = adata.uns["juzi_cluster_genes"]
    labels = np.array(adata.uns["juzi_cluster_labels"])
    unique_C = np.unique(labels)
    n_programs = len(unique_C)

    if n_programs == 0:
        raise ValueError("No programs found in juzi_cluster_labels.")

    prog_labels = [f"C{int(c)}" for c in unique_C]
    gene_sets = [set(cluster_genes.get(int(c), [])) for c in unique_C]

    overlap = np.zeros((n_programs, n_programs), dtype=float)

    for i in range(n_programs):
        A = gene_sets[i]
        for j in range(n_programs):
            B = gene_sets[j]

            if len(A) == 0 and len(B) == 0:
                val = 0.0
            else:
                inter = len(A & B)

                if metric == "jaccard":
                    union = len(A | B)
                    val = inter / union if union > 0 else 0.0
                else:
                    denom = min(len(A), len(B))
                    val = inter / denom if denom > 0 else 0.0

            overlap[i, j] = val

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "white_to_teal",
            [
                (1.0, 1.0, 1.0),
                (0.75, 0.67, 0.75),
                (0.21, 0.33, 0.33),
            ],
        )

    im = ax.imshow(
        overlap,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
        rasterized=rasterized,
    )

    # cell borders
    for x in np.arange(-0.5, n_programs, 1):
        ax.vlines(
            x,
            -0.5,
            n_programs - 0.5,
            color="white",
            linewidth=0.75,
        )
    ax.vlines(
        n_programs - 0.5,
        -0.5,
        n_programs - 0.5,
        color="white",
        linewidth=0.75,
    )

    for y in np.arange(-0.5, n_programs, 1):
        ax.hlines(
            y,
            -0.5,
            n_programs - 0.5,
            color="white",
            linewidth=0.75,
        )
    ax.hlines(
        n_programs - 0.5,
        -0.5,
        n_programs - 0.5,
        color="white",
        linewidth=0.75,
    )

    if show_values:
        for i in range(n_programs):
            for j in range(n_programs):
                v = overlap[i, j]
                text_color = "black" if v < 0.6 else "white"
                ax.text(
                    j,
                    i,
                    format(v, value_fmt),
                    ha="center",
                    va="center",
                    fontsize=fontsize - 1,
                    color=text_color,
                )

    ax.set_xticks(np.arange(n_programs))
    ax.set_xticklabels(prog_labels, fontsize=fontsize)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0)

    ax.set_yticks(np.arange(n_programs))
    ax.set_yticklabels(prog_labels, fontsize=fontsize)
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel("Programs", fontsize=fontsize, labelpad=8)
    ax.set_ylabel("Programs", fontsize=fontsize)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if cbar_label is None:
        cbar_label = "Jaccard overlap" if metric == "jaccard" else "Overlap coefficient"

    cbar = plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        ticks=cbar_ticks,
        pad=cbar_pad,
        aspect=cbar_aspect,
        shrink=cbar_shrink,
    )
    cbar.ax.tick_params(labelsize=fontsize - 1, length=cbar_tick_length)
    cbar.ax.set_xlabel(cbar_label, fontsize=fontsize, labelpad=cbar_labelpad)
    cbar.ax.xaxis.set_label_position(cbar_legend_pos)

    if created_fig:
        fig.tight_layout()

    return ax
