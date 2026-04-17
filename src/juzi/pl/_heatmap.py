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
    cbar_label: str = "Similarity",
    cbar_legend_pos: str = "top",
    cbar_labelpad: float = 10.0,
    box_color: str = "red",
    box_style: str = "dashed",
    box_linewidth: float = 0.5,
    rasterized: bool = False,
    add_cluster_colors: bool = True,
    add_cluster_labels: bool = True,
    fontsize: int = 10,
    vmin: float | None = None,
    vmax: float | None = None,
    transform_jaccard: bool = False,
) -> plt.Axes:
    """Plot the factor similarity matrix with consensus program annotations.

    Displays the reordered factor x factor similarity matrix produced by
    juzi.gp.programs_cluster, with cluster boundaries, a colour bar, and
    optional program labels. Works identically for centroid and progressive
    clustering.

    The diagonal of juzi_cluster_similarity is zero (self-similarity pairs
    are never computed during the similarity step) and is set to 1.0 before
    plotting so the block-diagonal structure is visually correct.

    vmin and vmax default to None, which triggers automatic scaling from
    the 1st and 99th percentiles of the off-diagonal positive values,
    ensuring the colormap always spans the actual data range.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_similarity and juzi_cluster_labels
        in .uns.
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
        Tick positions on the colorbar. If None, three evenly spaced ticks
        between vmin and vmax are used.
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
        If True, add a coloured strip above the top of the matrix for each cluster.
    add_cluster_labels : bool
        If True, annotate cluster labels beside the heatmap.
    fontsize : int
        Font size for all text elements.
    vmin : float | None
        Minimum value for colormap scaling. If None, set to the 1st
        percentile of positive off-diagonal values.
    vmax : float | None
        Maximum value for colormap scaling. If None, set to the 99th
        percentile of positive off-diagonal values.
    transform_jaccard : bool
        If True, apply J / (1 - J) to raw Jaccard values before plotting
        (the raw-Jaccard equivalent of the Gavish et al. R transform).
        Default False — raw Jaccard is plotted directly.

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

    # The diagonal is 0 in juzi_cluster_similarity because self-pairs are
    # never computed during similarity(). Set to 1.0 so the block-diagonal
    # structure is visually correct — each factor is maximally similar to itself.
    np.fill_diagonal(S, 1.0)

    # Optionally apply J / (1 - J) transform.
    if transform_jaccard:
        S_plot = np.where(S < 1.0, S / (1.0 - S), 1.0)
    else:
        S_plot = S.copy()

    # Auto-scale vmin/vmax from the off-diagonal positive values.
    # Exclude the diagonal (now 1.0) to avoid skewing vmax upward.
    # Exclude exact zeros — most off-diagonal entries are zero for unrelated
    # factor pairs and including them would collapse vmin to 0.
    if vmin is None or vmax is None:
        diag_mask = ~np.eye(S_plot.shape[0], dtype=bool)
        off_diag = S_plot[diag_mask]
        positive = off_diag[off_diag > 0]

        if len(positive) == 0:
            _vmin, _vmax = 0.0, 1.0
        else:
            _vmin = float(np.percentile(positive, 1))
            _vmax = float(np.percentile(positive, 99))
            if _vmax - _vmin < 1e-6:
                _vmin = 0.0
                _vmax = float(positive.max())

        vmin = vmin if vmin is not None else _vmin
        vmax = vmax if vmax is not None else _vmax

    if cbar_ticks is None:
        mid = (vmin + vmax) / 2.0
        cbar_ticks = [round(vmin, 3), round(mid, 3), round(vmax, 3)]

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
    ax.imshow(S_plot, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=rasterized)

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
    edges = np.where(np.diff(C) != 0)[0] + 1
    bounds = np.concatenate([edges, [len(C)]]).tolist()
    buffer = edges[0] * label_buffer if len(edges) > 0 else len(C) * label_buffer

    start = 0
    for end in bounds:
        size = end - start
        cluster_id = int(C[start])
        color = palette[cluster_id]

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

        if add_cluster_colors:
            # Blended transform: x in data coords, y in axes fraction.
            # y=1.01 places the strip just above the top spine.
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.add_patch(
                mpatches.Rectangle(
                    (start - 0.5, 1.01),
                    size,
                    0.02,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0.0,
                    clip_on=False,
                    transform=trans,
                )
            )

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
