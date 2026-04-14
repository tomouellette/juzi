# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple, List


def similarity(
    adata: AnnData,
    figsize: Tuple[float, float] = (6., 6.),
    ax: plt.Axes | None = None,
    cmap: str | None = None,
    palette: Dict[int, str] | None = None,
    label_buffer: float = 0.075,
    cbar_pad: float = 0.075,
    cbar_aspect: float = 10.,
    cbar_shrink: float = 0.2,
    cbar_ticks: List[float] = [0, 0.5, 1],
    cbar_tick_length: float = 0.,
    cbar_label: str = "Similarity",
    cbar_legend_pos: str = "top",
    cbar_labelpad: float = 10.,
    box_color: str = "black",
    box_style: str = "dotted",
    box_linewidth: float = 1.5,
    rasterized: bool = False,
    add_cluster_colors: bool = True,
    add_cluster_labels: bool = True,
    fontsize: int = 10,
    vmin: float = 0.,
    vmax: float = 1.,
) -> plt.Axes:
    """Plot the factor similarity matrix with consensus program annotations.

    Displays the reordered factor × factor Jaccard similarity matrix
    produced by juzi.gp.cluster, with cluster boundaries, colour bars,
    and program labels annotated.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_similarity and juzi_cluster_labels
        in .uns, produced by juzi.gp.cluster.
    figsize : Tuple[float, float]
        Figure size in inches as (width, height).
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.
    cmap : str | None
        Colormap for the similarity matrix. If None, a default white-to-teal
        colormap is used.
    palette : Dict[int, str] | None
        Dictionary mapping cluster label integers to hex or named colours.
        If None, colours are generated automatically via glasbey.
    label_buffer : float
        Fractional buffer between cluster boundary and label position.
    cbar_pad : float
        Padding between the heatmap and the colorbar.
    cbar_aspect : float
        Aspect ratio of the colorbar.
    cbar_shrink : float
        Fraction by which to shrink the colorbar.
    cbar_ticks : List[float]
        Tick positions on the colorbar.
    cbar_tick_length : float
        Length of colorbar ticks.
    cbar_label : str
        Label for the colorbar.
    cbar_legend_pos : str
        Position of the colorbar label ("top" or "bottom").
    cbar_labelpad : float
        Padding between colorbar and its label.
    box_color : str
        Color of the cluster boundary rectangles.
    box_style : str
        Linestyle of the cluster boundary rectangles.
    box_linewidth : float
        Line width of the cluster boundary rectangles.
    rasterized : bool
        If True, rasterize the heatmap for performance with large matrices.
    add_cluster_colors : bool
        If True, add a colored strip below the x-axis for each cluster.
    add_cluster_labels : bool
        If True, annotate cluster labels beside the heatmap.
    fontsize : int
        Font size for all text elements.
    vmin : float
        Minimum value for colormap scaling.
    vmax : float
        Maximum value for colormap scaling.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """
    for field in ["juzi_cluster_similarity", "juzi_cluster_labels"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. "
                "Run juzi.gp.cluster before plotting."
            )

    S = adata.uns["juzi_cluster_similarity"]
    C = adata.uns["juzi_cluster_labels"]
    unique_C = np.unique(C)

    # Figure setup

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    # Colormap

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "white_to_teal",
            [
                (1.0, 1.0, 1.0),
                (0.75, 0.67, 0.75),
                (0.21, 0.33, 0.33),
            ]
        )

    # Cluster colors

    if palette is None:
        colors = glasbey.create_palette(
            len(unique_C),
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_C)}

    # Main heatmap

    ax.imshow(S, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=rasterized)

    # Colorbar

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        ),
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

    # Cluster boundaries and annotations

    edges  = np.where(np.diff(C) != 0)[0] + 1
    bounds = list(edges) + [len(C)]
    buffer = edges[0] * label_buffer if len(edges) > 0 else len(C) * label_buffer

    start = 0
    for i, end in enumerate(bounds):
        size       = end - start
        cluster_id = int(C[start])
        color      = palette[cluster_id]

        # Cluster boundary rectangle
        ax.add_patch(plt.Rectangle(
            (start - 0.5, start - 0.5),
            size, size,
            fill=False,
            edgecolor=box_color,
            linewidth=box_linewidth,
            linestyle=box_style,
        ))

        # Cluster color strip below x-axis
        if add_cluster_colors:
            ax.add_patch(plt.Rectangle(
                (start, -2.5),
                size, -5,
                facecolor=color,
                edgecolor="none",
                linewidth=0.,
                clip_on=False,
                transform=ax.transData,
            ))

        # Cluster label
        if add_cluster_labels:
            mid    = (start + end) / 2
            x_pos  = end + buffer
            ha     = "left"
            if end > len(C) / 2:
                x_pos = start - buffer
                ha    = "right"

            ax.text(
                x_pos, mid,
                f"C{cluster_id}",
                fontsize=fontsize,
                ha=ha, va="center",
                color="black",
            )

        start = end

    # Axes styling

    n_factors  = S.shape[0]
    n_programs = len(unique_C)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        f"NMF factors (n = {n_factors}, programs = {n_programs})",
        fontsize=fontsize,
        labelpad=5,
    )
    ax.set_ylabel("NMF factors", fontsize=fontsize)

    for spine in ax.spines.values():
        spine.set_alpha(0.25)

    if created_fig:
        fig.tight_layout()

    return ax
