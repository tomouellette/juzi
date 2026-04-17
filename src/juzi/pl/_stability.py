# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple, List


def programs_stability(
    adata: AnnData,
    figsize: Tuple[float, float] = (6.0, 3.0),
    ax: plt.Axes | None = None,
    cmap: str | None = None,
    palette: Dict[int, str] | None = None,
    cbar_pad: float = 0.03,
    cbar_aspect: float = 12.0,
    cbar_shrink: float = 0.8,
    cbar_ticks: List[float] = [0, 0.5, 1.0],
    cbar_tick_length: float = 0.0,
    cbar_label: str = "Stability",
    cbar_legend_pos: str = "top",
    cbar_labelpad: float = 8.0,
    add_program_colors: bool = True,
    add_program_labels: bool = True,
    label_rotation: float = 0.0,
    show_mean: bool = True,
    mean_marker: str = "o",
    mean_marker_size: float = 16.0,
    mean_marker_edgecolor: str = "black",
    mean_marker_linewidth: float = 0.5,
    grid_color: str = "white",
    grid_linewidth: float = 0.75,
    rasterized: bool = False,
    fontsize: int = 10,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Axes:
    """Plot leave-one-donor-out program stability as a program-by-donor heatmap.

    Displays the per-program per-donor Jaccard similarities stored in
    adata.uns["juzi_stability"]["matrix"], where each entry reflects how
    similar the leave-one-donor-out recomputed program gene set is to the
    original canonical program gene set.

    Rows correspond to programs and columns correspond to held-out donors.
    Optionally overlays the mean stability score per program as a marker
    to the right of the heatmap.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_stability in .uns, produced by
        juzi.gp.programs_stability.
    figsize : Tuple[float, float]
        Figure size in inches as (width, height).
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.
    cmap : str | None
        Colormap for the stability matrix. If None, a default white-to-teal
        colormap is used.
    palette : Dict[int, str] | None
        Dictionary mapping program label integers to colours.
        If None, colours are generated automatically via glasbey.
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
    add_program_colors : bool
        If True, add a coloured strip to the left of each program row.
    add_program_labels : bool
        If True, show program labels on the y-axis.
    label_rotation : float
        Rotation angle for donor labels on the x-axis.
    show_mean : bool
        If True, overlay the mean stability per program as a marker to the
        right of the heatmap.
    mean_marker : str
        Marker style for mean stability overlay.
    mean_marker_size : float
        Marker size for mean stability overlay.
    mean_marker_edgecolor : str
        Edge color for mean stability markers.
    mean_marker_linewidth : float
        Line width for mean stability marker edges.
    grid_color : str
        Color of cell-separating grid lines.
    grid_linewidth : float
        Width of cell-separating grid lines.
    rasterized : bool
        If True, rasterize the heatmap for performance with large matrices.
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
    if "juzi_stability" not in adata.uns:
        raise KeyError(
            "'juzi_stability' not found in .uns. "
            "Run juzi.gp.programs_stability before plotting."
        )

    stab = adata.uns["juzi_stability"]
    matrix = np.array(stab["matrix"], dtype=float)
    programs = list(stab["programs"])
    donors = list(stab["donors"])
    score = np.array(stab["score"], dtype=float)

    if matrix.ndim != 2:
        raise ValueError("juzi_stability['matrix'] must be a 2D array.")

    n_programs, n_donors = matrix.shape

    if len(programs) != n_programs:
        raise ValueError(
            "Length of juzi_stability['programs'] does not match matrix rows."
        )

    if len(donors) != n_donors:
        raise ValueError(
            "Length of juzi_stability['donors'] does not match matrix columns."
        )

    if len(score) != n_programs:
        raise ValueError(
            "Length of juzi_stability['score'] does not match matrix rows."
        )

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

    # Program colors

    program_ids = []
    for p in programs:
        try:
            program_ids.append(int(str(p).replace("C", "")))
        except Exception:
            program_ids.append(len(program_ids))

    unique_ids = list(dict.fromkeys(program_ids))

    if palette is None:
        colors = glasbey.create_palette(
            len(unique_ids),
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_ids)}

    row_colors = [palette[int(pid)] for pid in program_ids]

    # Heatmap

    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        rasterized=rasterized,
    )

    # Grid lines

    for x in np.arange(-0.5, n_donors, 1):
        ax.vlines(x, -0.5, n_programs - 0.5, color=grid_color, linewidth=grid_linewidth)
    ax.vlines(
        n_donors - 0.5,
        -0.5,
        n_programs - 0.5,
        color=grid_color,
        linewidth=grid_linewidth,
    )

    for y in np.arange(-0.5, n_programs, 1):
        ax.hlines(y, -0.5, n_donors - 0.5, color=grid_color, linewidth=grid_linewidth)
    ax.hlines(
        n_programs - 0.5,
        -0.5,
        n_donors - 0.5,
        color=grid_color,
        linewidth=grid_linewidth,
    )

    # Program color strip

    if add_program_colors:
        for i, color in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle(
                    (-1.0, i - 0.5),
                    0.35,
                    1.0,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0.0,
                    clip_on=False,
                    transform=ax.transData,
                )
            )

    # Mean stability overlay

    if show_mean:
        x_mean = n_donors + 0.35
        ax.scatter(
            np.full(n_programs, x_mean),
            np.arange(n_programs),
            c=score,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=mean_marker_size,
            marker=mean_marker,
            edgecolors=mean_marker_edgecolor,
            linewidths=mean_marker_linewidth,
            clip_on=False,
            zorder=3,
        )

        ax.text(
            x_mean,
            -1.1,
            "Mean",
            ha="center",
            va="bottom",
            fontsize=fontsize - 1,
            rotation=90,
        )

    # Colorbar

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

    # Axis labels and ticks

    ax.set_xticks(np.arange(n_donors))
    ax.set_xticklabels(
        donors, rotation=label_rotation, ha="right" if label_rotation else "center"
    )
    ax.tick_params(axis="x", labelsize=fontsize - 1, length=0)

    ax.set_yticks(np.arange(n_programs))
    if add_program_labels:
        ax.set_yticklabels(programs)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", labelsize=fontsize - 1, length=0)

    ax.set_xlabel(
        f"Held-out donors (n = {n_donors})",
        fontsize=fontsize,
        labelpad=5,
    )
    ax.set_ylabel(
        f"Programs (n = {n_programs})",
        fontsize=fontsize,
    )

    # Limits to leave space for overlays

    x_right = n_donors - 0.5
    if show_mean:
        x_right = n_donors + 0.8
    ax.set_xlim(-0.5, x_right)
    ax.set_ylim(n_programs - 0.5, -0.5)

    # Styling

    for spine in ax.spines.values():
        spine.set_alpha(0.25)

    if created_fig:
        fig.tight_layout()

    return ax
