# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from typing import Dict, Tuple

from juzi.gp._nmf import _combined_score


def programs_loadings(
    adata: AnnData,
    n_top_genes: int = 10,
    use_combined: bool = True,
    figsize: Tuple[float, float] | None = None,
    palette: Dict[int, str] | None = None,
    fontsize: int = 8,
    bar_height: float = 0.7,
    ncols: int = 4,
    show_values: bool = False,
) -> plt.Figure:
    """Plot top gene loadings per consensus program as horizontal bar charts.

    Genes are selected and displayed using the combined loading × specificity
    score when use_combined=True, matching the ranking used in
    juzi.ut.program_genes and juzi.gp.score.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_G and juzi_cluster_labels in .uns.
    n_top_genes : int
        Number of top genes to display per program.
    use_combined : bool
        If True, rank and display genes by combined loading × specificity
        score. Bar length reflects normalised combined score. If False,
        rank and display by raw loading magnitude.
    figsize : Tuple[float, float] | None
        Figure size in inches. If None, inferred automatically.
    palette : Dict[int, str] | None
        Cluster label to colour mapping. If None, generated via glasbey.
    fontsize : int
        Font size for all text elements.
    bar_height : float
        Height of each bar as a fraction of available row space.
    ncols : int
        Number of columns in the figure grid.
    show_values : bool
        If True, annotate each bar with its score value.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """
    for field in ["juzi_cluster_G", "juzi_cluster_labels"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. "
                "Run juzi.gp.cluster before plotting loadings."
            )

    if "juzi_G_genes" not in adata.uns:
        raise KeyError(
            "'juzi_G_genes' not found in .uns. Run juzi.gp.nmf first."
        )

    if n_top_genes < 1:
        raise ValueError("n_top_genes must be >= 1.")

    # Setup

    G          = adata.uns["juzi_cluster_G"]
    gene_names = np.array(adata.uns["juzi_G_genes"])
    labels     = adata.uns["juzi_cluster_labels"]
    unique_C   = np.unique(labels)
    n_programs = len(unique_C)

    if n_top_genes > G.shape[1]:
        raise ValueError(
            f"n_top_genes={n_top_genes} exceeds number of genes ({G.shape[1]})."
        )

    # Palette

    if palette is None:
        colors  = glasbey.create_palette(
            n_programs,
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_C)}

    # Gene ranking and display scores
    # Ranking selects which genes to show.
    # Display normalises within program to [0, 1] for bar length.
    # Both use the same quantity so bars directly reflect ranking.

    if use_combined:
        G_rank    = _combined_score(G)
        xlabel    = "Normalised specificity × loading"
    else:
        G_rank    = G
        xlabel    = "Normalised loading"

    G_rank_max            = G_rank.max(axis=1, keepdims=True)
    G_rank_max[G_rank_max == 0] = 1
    G_display             = G_rank / G_rank_max # (n_programs × n_genes) in [0, 1]

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

        top_idx   = np.argsort(G_rank[i])[-n_top_genes:][::-1]
        top_genes = gene_names[top_idx]
        top_vals  = G_display[i][top_idx]

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
                    v + 0.01, y,
                    f"{v:.2f}",
                    fontsize=fontsize - 1,
                    va="center",
                    ha="left",
                    color="black",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_genes[::-1], fontsize=fontsize, style="italic")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(xlabel, fontsize=fontsize)
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

    for ax in axes_flat[n_programs:]:
        ax.set_visible(False)

    fig.tight_layout()

    return fig
