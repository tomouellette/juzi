# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple, List


def programs_heatmap(
    adata: AnnData,
    figsize: Tuple[float, float] = (6.0, 6.0),
    ax: plt.Axes | None = None,
    cmap: str | None = None,
    palette: Dict[int, str] | None = None,
    label_buffer: float = 0.075,
    cbar_pad: float = 0.075,
    cbar_aspect: float = 10.0,
    cbar_shrink: float = 0.2,
    cbar_ticks: List[float] | None = None,
    cbar_tick_length: float = 0.0,
    cbar_label: str = "Similarity (Jaccard)",
    cbar_legend_pos: str = "top",
    cbar_labelpad: float = 10.0,
    box_color: str = "red",
    box_style: str = "dashed",
    box_linewidth: float = 0.5,
    rasterized: bool = False,
    add_cluster_colors: bool = True,
    add_cluster_labels: bool = True,
    fontsize: int = 10,
    vmin: float = 2.0,
    vmax: float = 25.0,
    transform_jaccard: bool = True,
) -> plt.Axes:
    """Plot the factor similarity matrix with consensus program annotations.

    Displays the reordered factor × factor similarity matrix produced by
    juzi.gp.programs_cluster, with cluster boundaries, a colour bar, and
    optional program labels. Works identically for centroid and progressive
    clustering — both strategies write juzi_cluster_similarity and
    juzi_cluster_labels in the same display-ready format.

    By default the raw Jaccard values are transformed to
    100 * J / (100 - J) before plotting, matching the Gavish et al.
    R implementation. This amplifies mid-range similarities and makes
    cluster structure visually clearer. The default vmin / vmax
    of [2, 25] are calibrated for this transformed space. Set
    transform_jaccard=False and vmin=0, vmax=1 to plot raw values.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_similarity and
        juzi_cluster_labels in .uns.
    figsize : Tuple[float, float]
        Figure size in inches as (width, height).
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.
    cmap : str | None
        Colormap for the similarity matrix. If None, a default white-to-teal
        colormap is used.
    palette : Dict[int, str] | None
        Dict mapping cluster label integers to hex or named colours.
        If None, colours are generated automatically via glasbey.
    label_buffer : float
        Fractional buffer between the cluster boundary and label position,
        expressed as a fraction of the first cluster's size.
    cbar_pad : float
        Padding between the heatmap and the colorbar.
    cbar_aspect : float
        Aspect ratio of the colorbar.
    cbar_shrink : float
        Fraction by which to shrink the colorbar.
    cbar_ticks : List[float] | None
        Tick positions on the colorbar. Defaults to [2, 13, 25] when
        transform_jaccard=True and [0, 0.5, 1] otherwise.
    cbar_tick_length : float
        Length of colorbar ticks.
    cbar_label : str
        Label for the colorbar.
    cbar_legend_pos : str
        Position of the colorbar label ("top" or "bottom").
    cbar_labelpad : float
        Padding between colorbar and its label.
    box_color : str
        Colour of the cluster boundary rectangles.
    box_style : str
        Linestyle of the cluster boundary rectangles.
    box_linewidth : float
        Line width of the cluster boundary rectangles.
    rasterized : bool
        If True, rasterize the heatmap for performance with large matrices.
    add_cluster_colors : bool
        If True, add a coloured strip below the x-axis for each cluster.
    add_cluster_labels : bool
        If True, annotate cluster labels beside the heatmap.
    fontsize : int
        Font size for all text elements.
    vmin : float
        Minimum value for colormap scaling. Default 2.0 matches the
        Gavish et al. paper's transformed-space lower bound.
    vmax : float
        Maximum value for colormap scaling. Default 25.0 matches the
        Gavish et al. paper's transformed-space upper bound.
    transform_jaccard : bool
        If True (default), apply 100 * J / (100 - J) to the similarity
        matrix before plotting, matching the R paper exactly. If False,
        plot raw Jaccard values (set vmin=0, vmax=1 accordingly).

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """
    for field in ["juzi_cluster_similarity", "juzi_cluster_labels"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. "
                "Run juzi.gp.programs_cluster before plotting."
            )

    S = np.array(adata.uns["juzi_cluster_similarity"], dtype=float)
    C = np.array(adata.uns["juzi_cluster_labels"])
    unique_C = np.unique(C)

    # Apply Gavish et al. Jaccard transform: 100*J / (100 - J)
    # This is equivalent to the intersection size divided by the non-
    # overlapping portion, amplifying mid-range similarities.
    if transform_jaccard:
        # Clip to [0, 1) to avoid division by zero at J=1
        S_plot = 100.0 * S / np.where(S < 1.0, 100.0 - 100.0 * S, 1.0)
    else:
        S_plot = S

    # Default colorbar ticks depend on whether transform is applied
    if cbar_ticks is None:
        cbar_ticks = [2, 13, 25] if transform_jaccard else [0, 0.5, 1]

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
            ],
        )

    # Cluster colours
    if palette is None:
        colors = glasbey.create_palette(
            len(unique_C),
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_C)}

    # Main heatmap
    im = ax.imshow(S_plot, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=rasterized)

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
    # C is already in contiguous-block display order after juzi.gp.programs_cluster
    edges = np.where(np.diff(C) != 0)[0] + 1
    bounds = np.concatenate([edges, [len(C)]]).tolist()
    buffer = edges[0] * label_buffer if len(edges) > 0 else len(C) * label_buffer

    start = 0
    for end in bounds:
        size = end - start
        cluster_id = int(C[start])
        color = palette[cluster_id]

        # Cluster boundary rectangle — matches R's geom_rect with dashed red borders
        ax.add_patch(
            mpatches.Rectangle(
                (start - 0.5, start - 0.5),
                size,
                size,
                fill=False,
                edgecolor=box_color,
                linewidth=box_linewidth,
                linestyle=box_style,
            )
        )

        # Cluster colour strip below the x-axis.
        # Use a blended transform: x in data coordinates, y in axes fraction.
        # This avoids the fragile negative-data-coordinate trick and survives
        # any axis limits set after the patches are added.
        if add_cluster_colors:
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.add_patch(
                mpatches.Rectangle(
                    (start - 0.5, -0.04),  # x in data coords, y in axes fraction
                    size,  # width in data coords
                    0.02,  # height in axes fraction
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0.0,
                    clip_on=False,
                    transform=trans,
                )
            )

        # Cluster label — left side for clusters in the left half,
        # right side for clusters in the right half, matching visual convention
        if add_cluster_labels:
            mid = (start + end) / 2
            if end <= len(C) / 2:
                x_pos = end + buffer
                ha = "left"
            else:
                x_pos = start - buffer
                ha = "right"

            ax.text(
                x_pos,
                mid,
                f"C{cluster_id}",
                fontsize=fontsize,
                ha=ha,
                va="center",
                color="black",
            )

        start = end

    # Axes styling
    n_factors = S.shape[0]
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
