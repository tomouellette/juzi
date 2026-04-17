# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from typing import Dict, Tuple


def programs_loadings(
    adata: AnnData,
    figsize: Tuple[float, float] | None = None,
    palette: Dict[int, str] | None = None,
    fontsize: int = 8,
    bar_height: float = 0.7,
    ncols: int = 4,
    show_values: bool = False,
) -> plt.Figure:
    """Plot canonical program genes with centroid-based loading magnitudes.

    Genes displayed are the canonical gene sets from juzi_cluster_genes,
    computed at cluster time by juzi.gp.programs_cluster. For centroid mode
    these are typically top genes by combined score; for progressive mode
    these are the MP genes.

    Bar lengths reflect the corresponding values in juzi_cluster_G,
    normalised to [0, 1] within each program for display.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_genes, juzi_cluster_G,
        juzi_cluster_labels, and juzi_G_genes in .uns, produced by
        juzi.gp.programs_cluster.
    figsize : Tuple[float, float] | None
        Figure size in inches. If None, inferred automatically.
    palette : Dict[int, str] | None
        Program label to colour mapping. If None, generated via glasbey.
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
        The matplotlib Figure object.
    """
    # Validate

    for field in [
        "juzi_cluster_genes",
        "juzi_cluster_G",
        "juzi_cluster_labels",
        "juzi_G_genes",
    ]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. "
                "Run juzi.gp.programs_cluster before plotting loadings."
            )

    # Setup

    cluster_genes = adata.uns["juzi_cluster_genes"]
    G = np.asarray(adata.uns["juzi_cluster_G"])  # (n_programs x n_genes)
    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)
    labels = np.array(adata.uns["juzi_cluster_labels"])
    unique_C = np.unique(labels)
    n_programs = len(unique_C)

    if G.shape[0] != n_programs:
        raise ValueError(
            "juzi_cluster_G row count does not match the number of unique "
            "cluster labels."
        )

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Infer max genes to display from canonical program gene sets
    n_top_genes = max(len(cluster_genes.get(int(c), [])) for c in unique_C)

    if n_top_genes == 0:
        raise ValueError(
            "juzi_cluster_genes is empty. Re-run juzi.gp.programs_cluster."
        )

    # Palette

    if palette is None:
        colors = glasbey.create_palette(
            n_programs,
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_C)}

    # Figure layout

    n_cols = min(n_programs, ncols)
    n_rows = int(np.ceil(n_programs / n_cols))

    if figsize is None:
        panel_w = 2.2
        panel_h = 0.25 * n_top_genes + 0.5
        figsize = (panel_w * n_cols, panel_h * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()

    # Plot each program

    for i, (c, ax) in enumerate(zip(unique_C, axes_flat)):
        color = palette[int(c)]
        genes = list(cluster_genes.get(int(c), []))

        if len(genes) == 0:
            ax.text(
                0.5,
                0.5,
                f"C{int(c)}\n(no genes)",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        present_genes = [g for g in genes if g in gene_to_idx]

        if len(present_genes) == 0:
            ax.text(
                0.5,
                0.5,
                f"C{int(c)}\n(no mapped genes)",
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        gene_idx = np.array([gene_to_idx[g] for g in present_genes], dtype=int)

        # Use canonical gene order from cluster time
        raw_vals = G[i, gene_idx]

        # Normalise to [0, 1] within program for display
        max_val = raw_vals.max()
        if max_val == 0:
            max_val = 1.0
        disp_vals = raw_vals / max_val

        n_genes = len(present_genes)
        y_pos = np.arange(n_genes)

        # Reverse so top-ranked gene appears at top
        ax.barh(
            y_pos,
            disp_vals[::-1],
            height=bar_height,
            color=color,
            edgecolor="none",
        )

        if show_values:
            for y, v in zip(y_pos, disp_vals[::-1]):
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
            present_genes[::-1],
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

    for ax in axes_flat[n_programs:]:
        ax.set_visible(False)

    fig.tight_layout()

    return fig
