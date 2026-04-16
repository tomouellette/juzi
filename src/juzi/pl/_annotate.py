# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from anndata import AnnData
from matplotlib.colors import Normalize
from typing import Dict, List, Tuple


def programs_annotate(
    adata: AnnData,
    top_n: int = 10,
    padj_thresh: float = 0.05,
    figsize: Tuple[float, float] | None = None,
    palette: Dict[int, str] | None = None,
    cmap: str = "Reds",
    fontsize: int = 8,
    dot_scale: float = 100.0,
    show_colorbar: bool = True,
    show_legend: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot program annotation results as a dot plot.

    Displays the top reference gene sets per consensus program as a
    dot plot where dot size encodes Jaccard similarity and dot colour
    encodes significance (-log10 adjusted p-value). Only significant
    gene set associations (padj < padj_thresh) are shown.

    Programs are shown as columns and gene sets as rows, with gene sets
    ordered by their best Jaccard score across all programs.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_annotation in .uns, produced by
        juzi.gp.annotate.
    top_n : int
        Maximum number of gene sets to show per program, selected by
        lowest adjusted p-value.
    padj_thresh : float
        Adjusted p-value threshold. Only gene set associations below
        this threshold are displayed.
    figsize : Tuple[float, float] | None
        Figure size in inches. If None, inferred from number of programs
        and gene sets.
    palette : Dict[int, str] | None
        Dictionary mapping cluster label integers to colours for the
        program column headers. If None, generated via glasbey to match
        other juzi plots.
    cmap : str
        Colormap for dot colour encoding -log10(padj).
    fontsize : int
        Font size for all text elements.
    dot_scale : float
        Scaling factor for dot size. Larger values produce bigger dots.
    show_colorbar : bool
        If True, add a colorbar for the -log10(padj) colour encoding.
    show_legend : bool
        If True, add a dot size legend for Jaccard similarity.
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """
    #  Validate

    if "juzi_annotation" not in adata.uns:
        raise KeyError(
            "'juzi_annotation' not found in .uns. "
            "Run juzi.gp.annotate before plotting."
        )

    if "juzi_cluster_labels" not in adata.uns:
        raise KeyError(
            "'juzi_cluster_labels' not found in .uns. "
            "Run juzi.gp.cluster before plotting."
        )

    if not 0.0 <= padj_thresh <= 1.0:
        raise ValueError("padj_thresh must be in [0, 1].")

    if top_n < 1:
        raise ValueError("top_n must be >= 1.")

    # Setup

    df = adata.uns["juzi_annotation"].copy()
    labels = adata.uns["juzi_cluster_labels"]
    unique_C = np.unique(labels)
    n_programs = len(unique_C)

    # Filter to significant associations
    df = df[df["padj"] < padj_thresh].copy()

    if len(df) == 0:
        raise ValueError(
            f"No significant gene set associations found at "
            f"padj < {padj_thresh}. Lower padj_thresh or check "
            "your gene sets overlap with juzi_G_genes."
        )

    # Palette

    if palette is None:
        colors = glasbey.create_palette(
            n_programs,
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_C)}

    # Select top N gene sets per program

    top_per_program = df.sort_values("padj").groupby("program").head(top_n)

    # Union of gene sets to display ordered by best Jaccard across programs
    gs_order = (
        top_per_program.groupby("gene_set")["jaccard"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )

    prog_order = [f"C{int(c)}" for c in unique_C]

    # Pivot to matrix for plotting

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

    log_padj_mat = -np.log10(padj_mat.clip(lower=1e-300))

    n_gs = len(gs_order)
    n_prog = len(prog_order)

    # Figure setup

    created_fig = ax is None
    if created_fig:
        if figsize is None:
            figsize = (
                max(2.0, 0.6 * n_prog + 1.5),
                max(2.0, 0.25 * n_gs + 1.0),
            )
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Dot plot

    vmax = (
        log_padj_mat.values[log_padj_mat.values > 0].max()
        if (log_padj_mat.values > 0).any()
        else 1.0
    )
    norm = Normalize(vmin=0, vmax=vmax)
    cmap_obj = matplotlib.colormaps.get_cmap(cmap)

    for i, gs in enumerate(gs_order):
        for j, prog in enumerate(prog_order):
            jacc = jaccard_mat.loc[gs, prog]
            lpadj = log_padj_mat.loc[gs, prog]

            if jacc == 0.0:
                continue

            ax.scatter(
                j,
                i,
                s=jacc * dot_scale,
                c=[cmap_obj(norm(lpadj))],
                edgecolors="none",
                zorder=3,
            )

    # Grid

    for j in range(n_prog):
        ax.axvline(j, color="lightgrey", linewidth=0.5, zorder=0)
    for i in range(n_gs):
        ax.axhline(i, color="lightgrey", linewidth=0.5, zorder=0)

    # Program color strip on top

    for j, prog in enumerate(prog_order):
        c_int = int(prog.replace("C", ""))
        color = palette.get(c_int, "#888888")
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, n_gs - 0.5),
                1.0,
                0.6,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
                transform=ax.transData,
            )
        )

    # Axes

    ax.set_xlim(-0.5, n_prog - 0.5)
    ax.set_ylim(-0.5, n_gs - 0.5)

    ax.set_xticks(range(n_prog))
    ax.set_xticklabels(prog_order, fontsize=fontsize, rotation=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", length=0)

    ax.set_yticks(range(n_gs))
    ax.set_yticklabels(gs_order, fontsize=fontsize)
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar

    if show_colorbar and created_fig:
        cbar_ax = fig.add_axes([1.02, 0.5, 0.02, 0.3])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm),
            cax=cbar_ax,
            orientation="vertical",
        )
        cbar.ax.tick_params(labelsize=fontsize - 1, length=2)
        cbar.outline.set_visible(False)
        cbar.set_label("-log₁₀(FDR)", fontsize=fontsize, labelpad=4)

    # Dot size legend

    if show_legend and created_fig:
        legend_jaccards = [0.25, 0.50, 0.75, 1.00]
        legend_handles = [
            plt.scatter(
                [],
                [],
                s=j * dot_scale,
                c="grey",
                edgecolors="none",
                label=f"{j:.2f}",
            )
            for j in legend_jaccards
        ]
        ax.legend(
            handles=legend_handles,
            title="Jaccard",
            title_fontsize=fontsize,
            fontsize=fontsize - 1,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.08, 0.45),
            handletextpad=0.5,
            labelspacing=0.8,
        )

    return ax
