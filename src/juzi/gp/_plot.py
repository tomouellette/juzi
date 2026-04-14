# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple, List


def plot_programs(
    adata: AnnData,
    figsize: Tuple[float, float] = (6., 6.),
    ax: plt.Axes | None = None,
    cmap: str | None = None,
    palette: Dict[int, str] = None,
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
    """Plot programs identified from clustering multi-sample similarity matrix.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the necessary program or similarity data.
    figsize : Tuple[float, float]
        Figure size in inches as (width, height), by default (5.0, 5.0).
    ax : plt.Axes | None
        Axes object to use for plotting. If None, a new figure and axes are created.
    cmap : str | None
        Colormap to be used for the plot. If None, a default colormap is used.
    palette : Dict[int, str] | None
        Dictionary mapping cluster labels to their corresponding colors.
    label_buffer : float
        Fractional space between the plot and its labels.
    cbar_pad : float
        Padding between the plot and the color bar.
    cbar_aspect : float
        Aspect ratio of the color bar.
    cbar_shrink : float
        Fraction by which to shrink the color bar.
    cbar_ticks : list of float
        Tick positions for the color bar.
    cbar_tick_length : float
        Length of the ticks on the color bar.
    cbar_label : str
        Label for the color bar.
    cbar_legend_pos : str
        Position of the color bar legend.
    cbar_labelpad : str
        Padding below colorbar title text.
    box_color : str
        Color of the bounding box drawn around the plot.
    box_style : str
        Style of the bounding box (e.g., "dotted", "solid").
    box_linewidth : float
        Line width of the bounding box, by default 1.5.
    rasterized : bool
        Whether to rasterize the plot objects (for performance with large datasets),
        by default False.
    add_cluster_colors : bool
        Whether to add cluster-specific colors to the plot.
    add_cluster_labels : bool
        Whether to add labels for clusters in the plot.
    fontsize : int
        Font size for text annotations and labels.
    vmin : float
        Minimum value for the colormap scaling.
    vmax : float
        Maximum value for the colormap scaling.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object with the program plot.
    """
    if np.any([
        "juzi_cluster_similarity" not in adata.uns,
        "juzi_cluster_labels" not in adata.uns,
    ]):
        raise KeyError(
            "Please run juzi.cs.cluster before plotting programs. " +
            "If it was already run, make sure 'juzi_cluster_similarity' " +
            "and 'juzi_cluster_labels' are present in .uns"
        )

    S = adata.uns["juzi_cluster_similarity"]
    C = adata.uns["juzi_cluster_labels"]

    tight_layout = True
    if ax is None:
        tight_layout = False
        fig, ax = plt.subplots(figsize=figsize)

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "white_to_dark_pink",
            [
                (1.0, 1.0, 1.0),
                (0.75, 0.67, 0.75),
                (0.21, 0.33, 0.33)
            ])

    if palette is None:
        colors = glasbey.create_palette(
            len(np.unique(C)), chroma_bounds=(5, 40), lightness_bounds=(0, 100))
    else:
        colors = [palette[k] for k in np.unique(C)]

    ax.imshow(S, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=rasterized)

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax)
        ),
        ax=ax,
        orientation="vertical",
        ticks=cbar_ticks,
        pad=cbar_pad,
        aspect=cbar_aspect,
        shrink=cbar_shrink,
    )

    cbar.ax.tick_params(labelsize=fontsize-1, length=cbar_tick_length)
    cbar.ax.set_xlabel(cbar_label, fontsize=fontsize, labelpad=cbar_labelpad)
    cbar.ax.xaxis.set_label_position(cbar_legend_pos)

    edges = np.where(np.diff(C) != 0)[0] + 1
    start = 0
    for end in list(edges) + [len(C)]:
        size = end - start
        rect = plt.Rectangle(
            (start - 0.5, start-0.5),
            size,
            size,
            fill=False,
            edgecolor=box_color,
            linewidth=box_linewidth,
            linestyle=box_style
        )
        ax.add_patch(rect)

        cluster_id = C[start]
        color = cluster_id
        width = end - start

        if add_cluster_colors:
            ax.add_patch(plt.Rectangle(
                (start, -2.5),
                width,
                -5,
                facecolor=colors[color],
                transform=ax.transData,
                edgecolor='black',
                clip_on=False,
                linewidth=0.
            ))

        start = end

    ax.set_yticks([])
    ax.set_xticks([])
    for i in ["top", "right", "left", "bottom"]:
        ax.spines[i].set_alpha(0.25)

    start, buffer = 0, edges[0] * label_buffer
    for c in np.unique(C):
        name = f"C{c}"
        end = (list(edges) + [len(C)])[c]
        y = np.mean([start, end])

        x_pos, ha = end + buffer, "left"
        if end > len(C) / 2:
            x_pos = start - buffer
            ha = "right"

        ax.text(
            x_pos,
            y,
            name,
            fontsize=fontsize,
            ha=ha,
            va="center",
            color="black",
        )

        start = end

    n_programs = S.shape[0]
    ax.set_xlabel(
        f"NMF programs (n = {n_programs})",
        fontsize=fontsize,
        labelpad=5
    )
    ax.set_ylabel(
        "NMF programs",
        fontsize=fontsize
    )

    if tight_layout:
        fig.tight_layout()

    return ax
