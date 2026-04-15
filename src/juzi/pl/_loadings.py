# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from typing import Dict, List, Tuple


def loadings(
    adata: AnnData,
    n_top_genes: int = 10,
    use_specificity: bool = True,
    figsize: Tuple[float, float] | None = None,
    palette: Dict[int, str] | None = None,
    fontsize: int = 8,
    bar_height: float = 0.7,
    ncols: int = 4,
    show_values: bool = False,
) -> plt.Figure:
    """Plot top gene loadings per consensus program as horizontal bar charts.

    For each consensus program identified by juzi.gp.cluster, displays the
    top n_top_genes genes as a horizontal bar chart. Genes are selected by
    the same ranking used in juzi.ut.program_genes and juzi.gp.score —
    either specificity (loading / total loading across programs) or raw
    loading magnitude. Bar length shows normalised loading magnitude so
    bars are always in [0, 1].

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_G and juzi_cluster_labels in .uns,
        produced by juzi.gp.cluster.
    n_top_genes : int
        Number of top genes to display per program.
    use_specificity : bool
        If True, rank genes by specificity score (loading in this program
        divided by total loading across all programs). Matches the ranking
        used in juzi.ut.program_genes and juzi.gp.score. If False, rank
        by raw loading magnitude.
    figsize : Tuple[float, float] | None
        Figure size in inches. If None, inferred from number of programs
        and n_top_genes.
    palette : Dict[int, str] | None
        Dictionary mapping cluster label integers to colours. If None,
        generated automatically via glasbey.
    fontsize : int
        Font size for all text elements.
    bar_height : float
        Height of each bar as a fraction of available row space.
    ncols : int
        Number of columns in the figure grid.
    show_values : bool
        If True, annotate each bar with its normalised loading value.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing all program panels.
    """
    for field in ["juzi_cluster_G", "juzi_cluster_labels"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. "
                "Run juzi.gp.cluster before plotting loadings."
            )

    if "juzi_G_genes" not in adata.uns:
        raise KeyError(
            "'juzi_G_genes' not found in .uns. "
            "Run juzi.gp.nmf before plotting loadings."
        )

    if n_top_genes < 1:
        raise ValueError("n_top_genes must be >= 1.")

    # Setup

    G = adata.uns["juzi_cluster_G"]  # (n_programs × n_genes)
    gene_names = np.array(adata.uns["juzi_G_genes"])
    labels = adata.uns["juzi_cluster_labels"]
    unique_C = np.unique(labels)
    n_programs = len(unique_C)

    if n_top_genes > G.shape[1]:
        raise ValueError(
            f"n_top_genes={n_top_genes} exceeds number of genes ({G.shape[1]})."
        )

    # Palette

    if palette is None:
        colors = glasbey.create_palette(
            n_programs,
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_C)}

    # Gene ranking

    if use_specificity:
        total = G.sum(axis=0, keepdims=True) + 1e-8
        G_rank = G / total
    else:
        G_rank = G

    # Display normalisation

    G_max = G.max(axis=1, keepdims=True)
    G_max[G_max == 0] = 1
    G_norm = G / G_max  # (n_programs × n_genes)

    # Figure layout

    n_cols = min(n_programs, ncols)
    n_rows = int(np.ceil(n_programs / n_cols))

    if figsize is None:
        panel_w = 2.2
        panel_h = 0.25 * n_top_genes + 0.5
        figsize = (panel_w * n_cols, panel_h * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.array(axes).flatten() if n_programs > 1 else [axes]

    # Plot each program

    for i, (c, ax) in enumerate(zip(unique_C, axes_flat)):
        color = palette[int(c)]

        # Select top genes by rank (specificity or raw loading)
        top_idx = np.argsort(G_rank[i])[-n_top_genes:][::-1]
        top_genes = gene_names[top_idx]

        # Display bar length as normalised loading magnitude
        top_vals = G_norm[i][top_idx]

        # Plot in ascending order so highest-ranked gene is at top
        y_pos = np.arange(n_top_genes)
        ax.barh(
            y_pos,
            top_vals[::-1],
            height=bar_height,
            color=color,
            edgecolor="none",
        )

        if show_values:
            for y, v in zip(y_pos, top_vals[::-1]):
                ax.text(
                    v + 0.01,
                    y,
                    f"{v:.2f}",
                    fontsize=fontsize - 1,
                    va="center",
                    ha="left",
                    color="black",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            top_genes[::-1],
            fontsize=fontsize,
            style="italic",
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Normalised loading", fontsize=fontsize)
        ax.tick_params(axis="x", length=2, labelsize=fontsize)
        ax.tick_params(axis="y", length=0)

        ax.set_title(
            f"C{int(c)}",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
            pad=4,
        )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.5)

    # Hide unused axes

    for ax in axes_flat[n_programs:]:
        ax.set_visible(False)

    fig.tight_layout()

    return fig
