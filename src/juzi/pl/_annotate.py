# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Tuple


def programs_annotate(
    adata: AnnData,
    top_n: int = 10,
    padj_thresh: float = 0.05,
    figsize: Tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    cmap: str | None = None,
    palette: Dict[int, str] | None = None,
    fontsize: int = 8,
    grid_linewidth: float = 1.0,
    rasterized: bool = False,
    cbar_pad: float = 0.03,
    cbar_aspect: float = 12.0,
    cbar_shrink: float = 0.8,
    cbar_ticks: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    cbar_tick_length: float = 0.0,
    cbar_label: str = "Jaccard",
    cbar_legend_pos: str = "top",
    cbar_labelpad: float = 8.0,
    add_program_colors: bool = True,
) -> plt.Axes:
    """Plot program annotation results as a heatmap.

    Displays the top reference gene sets per program as a heatmap where
    cell fill encodes Jaccard similarity and cell border color encodes
    statistical significance:

        - black border : padj < padj_thresh
        - white border : padj >= padj_thresh or missing

    Rows are gene sets and columns are programs. Gene sets are selected
    from the top_n lowest-padj hits per program, then ordered by their
    best Jaccard score across all programs.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_annotation in .uns, produced by
        juzi.gp.programs_annotate.
    top_n : int
        Maximum number of gene sets to show per program, selected by
        lowest adjusted p-value.
    padj_thresh : float
        Adjusted p-value threshold used for significance border encoding.
    figsize : Tuple[float, float] | None
        Figure size in inches. If None, inferred from number of programs
        and gene sets.
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.
    cmap : str | None
        Colormap for Jaccard similarity. If None, a default white-to-teal
        colormap is used.
    palette : Dict[int, str] | None
        Dictionary mapping program label integers to colours. If None,
        colours are generated automatically via glasbey.
    fontsize : int
        Font size for all text elements.
    grid_linewidth : float
        Line width of cell borders.
    rasterized : bool
        If True, rasterize the heatmap for performance with large matrices.
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
    cbar_label : str
        Label for the colorbar.
    cbar_legend_pos : str
        Position of the colorbar label ("top" or "bottom").
    cbar_labelpad : float
        Padding between colorbar and label.
    add_program_colors : bool
        If True, add a colored strip above the program columns.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """
    # Validate

    if "juzi_annotation" not in adata.uns:
        raise KeyError(
            "'juzi_annotation' not found in .uns. "
            "Run juzi.gp.programs_annotate before plotting."
        )

    if "juzi_cluster_labels" not in adata.uns:
        raise KeyError(
            "'juzi_cluster_labels' not found in .uns. "
            "Run juzi.gp.programs_cluster before plotting."
        )

    if not 0.0 <= padj_thresh <= 1.0:
        raise ValueError("padj_thresh must be in [0, 1].")

    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    # Setup

    df = adata.uns["juzi_annotation"].copy()
    labels = np.array(adata.uns["juzi_cluster_labels"])
    unique_C = np.unique(labels)
    n_programs = len(unique_C)

    if len(df) == 0:
        raise ValueError(
            "juzi_annotation is empty. Run juzi.gp.programs_annotate first."
        )

    # Palette

    if palette is None:
        colors = glasbey.create_palette(
            n_programs,
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_C)}

    # Keep only top_n gene sets per program by padj
    top_per_program = (
        df.sort_values(["program", "padj", "jaccard"], ascending=[True, True, False])
        .groupby("program", sort=False)
        .head(top_n)
        .copy()
    )

    if len(top_per_program) == 0:
        raise ValueError(
            "No annotation rows available to plot. " "Check juzi_annotation contents."
        )

    # Order gene sets by best Jaccard across displayed rows
    gs_order = (
        top_per_program.groupby("gene_set")["jaccard"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )

    prog_order = [f"C{int(c)}" for c in unique_C]

    # Pivot matrices

    jaccard_mat = (
        top_per_program.pivot(index="gene_set", columns="program", values="jaccard")
        .reindex(index=gs_order, columns=prog_order)
        .fillna(0.0)
    )

    padj_mat = (
        top_per_program.pivot(index="gene_set", columns="program", values="padj")
        .reindex(index=gs_order, columns=prog_order)
        .fillna(1.0)
    )

    n_gs = len(gs_order)
    n_prog = len(prog_order)

    # Figure setup

    created_fig = ax is None
    if created_fig:
        if figsize is None:
            figsize = (
                max(2.5, 0.55 * n_prog + 1.6),
                max(2.0, 0.28 * n_gs + 1.0),
            )
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

    # Main heatmap

    im = ax.imshow(
        jaccard_mat.values,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        rasterized=rasterized,
    )

    # Cell borders: black for significant, white otherwise

    for i in range(n_gs):
        for j in range(n_prog):
            is_sig = padj_mat.iloc[i, j] < padj_thresh
            edgecolor = "black" if is_sig else "white"

            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=edgecolor,
                    linewidth=grid_linewidth,
                )
            )

    # Program color strip on top

    if add_program_colors:
        for j, prog in enumerate(prog_order):
            c_int = int(prog.replace("C", ""))
            color = palette.get(c_int, "#888888")
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, -1.15),
                    1.0,
                    0.35,
                    facecolor=color,
                    edgecolor="none",
                    clip_on=False,
                    transform=ax.transData,
                )
            )

    # Axes

    ax.set_xlim(-0.5, n_prog - 0.5)
    ax.set_ylim(n_gs - 0.5, -0.5)

    ax.set_xticks(np.arange(n_prog))
    ax.set_xticklabels(prog_order, fontsize=fontsize, rotation=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0)

    ax.set_yticks(np.arange(n_gs))
    ax.set_yticklabels(gs_order, fontsize=fontsize)
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel(
        f"Programs (n = {n_prog})",
        fontsize=fontsize,
        labelpad=8,
    )
    ax.set_ylabel(
        f"Gene sets (n = {n_gs})",
        fontsize=fontsize,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

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

    if created_fig:
        fig.tight_layout()

    return ax
