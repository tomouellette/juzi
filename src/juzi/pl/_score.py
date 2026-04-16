# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from anndata import AnnData
from typing import Tuple


def score_embedding(
    adata: AnnData,
    basis: str = "X_umap",
    figsize: Tuple[float, float] | None = None,
    cmap: str = "RdBu_r",
    fontsize: int = 8,
    dot_size: float = 2.0,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float = 0.0,
    ncols: int = 4,
    rasterized: bool = True,
    show_colorbar: bool = True,
    colorbar_label: str = "Program score",
) -> plt.Figure:
    """Plot per-cell program scores on a 2D embedding.

    For each consensus program, plots a scatter of cells on the provided
    embedding coloured by their program score from
    obsm["juzi_program_scores"]. Programs are arranged as a grid of
    small multiples, one panel per program.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_program_scores in .obsm, produced by
        juzi.gp.score, and a 2D embedding in .obsm[basis].
    basis : str
        Key in adata.obsm containing the 2D embedding coordinates.
        Common values: "X_umap", "X_tsne", "X_pca".
    figsize : Tuple[float, float] | None
        Figure size in inches. If None, inferred from number of programs.
    cmap : str
        Colormap for program scores. Diverging colormaps (e.g. "RdBu_r",
        "coolwarm") work well since scores can be negative after control
        subtraction.
    fontsize : int
        Font size for all text elements.
    dot_size : float
        Size of cell scatter points.
    vmin : float | None
        Minimum value for colormap scaling. If None, uses the 1st
        percentile of scores across all programs.
    vmax : float | None
        Maximum value for colormap scaling. If None, uses the 99th
        percentile of scores across all programs.
    vcenter : float
        Centre value for diverging colormap normalisation. Default 0.
    ncols : int
        Number of columns in the panel grid.
    rasterized : bool
        If True, rasterize scatter points for performance with many cells.
    show_colorbar : bool
        If True, add a shared colorbar to the right of the figure.
    colorbar_label : str
        Label for the shared colorbar.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing all program panels.
    """
    # Validate

    if "juzi_program_scores" not in adata.obsm:
        raise KeyError(
            "'juzi_program_scores' not found in .obsm. "
            "Run juzi.gp.score before plotting."
        )

    if basis not in adata.obsm:
        raise KeyError(
            f"'{basis}' not found in adata.obsm. "
            "Check your basis argument or compute an embedding first."
        )

    if "juzi_cluster_labels" not in adata.uns:
        raise KeyError(
            "'juzi_cluster_labels' not found in .uns. "
            "Run juzi.gp.cluster before plotting."
        )

    embedding = adata.obsm[basis]
    if embedding.shape[1] < 2:
        raise ValueError(f"Embedding '{basis}' must have at least 2 dimensions.")

    # Setup

    program_scores = adata.obsm["juzi_program_scores"]  # (n_cells × n_programs)
    n_programs = program_scores.shape[1]
    labels = adata.uns["juzi_cluster_labels"]
    unique_C = np.unique(labels)

    coords_x = embedding[:, 0]
    coords_y = embedding[:, 1]

    # Colormap normalisation

    all_scores = program_scores.ravel()
    all_scores = all_scores[np.isfinite(all_scores)]

    if vmin is None:
        vmin = float(np.percentile(all_scores, 1))
    if vmax is None:
        vmax = float(np.percentile(all_scores, 99))

    # Use TwoSlopeNorm if vcenter is within [vmin, vmax]
    if vmin < vcenter < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Figure layout

    n_cols = min(n_programs, ncols)
    n_rows = int(np.ceil(n_programs / n_cols))

    if figsize is None:
        panel_size = 2.0
        cbar_w = 0.3 if show_colorbar else 0.0
        figsize = (panel_size * n_cols + cbar_w, panel_size * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
    )

    axes_flat = np.array(axes).flatten() if n_programs > 1 else [axes]

    # Plot each program

    sc = None
    for i, (c, ax) in enumerate(zip(unique_C, axes_flat)):
        prog_scores = program_scores[:, i]

        sc = ax.scatter(
            coords_x,
            coords_y,
            c=prog_scores,
            cmap=cmap,
            norm=norm,
            s=dot_size,
            edgecolors="none",
            rasterized=rasterized,
        )

        ax.set_title(
            f"C{int(c)}",
            fontsize=fontsize,
            pad=3,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        # Axis label showing basis name
        basis_label = basis.replace("X_", "").upper()
        ax.set_xlabel(f"{basis_label} 1", fontsize=fontsize - 1)
        ax.set_ylabel(f"{basis_label} 2", fontsize=fontsize - 1)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.3)

    # Hide unused axes

    for ax in axes_flat[n_programs:]:
        ax.set_visible(False)

    # Shared colorbar

    if show_colorbar and sc is not None:
        cbar_ax = fig.add_axes(
            [
                1.01,
                0.15,
                0.02,
                0.7,
            ]
        )
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=fontsize - 1, length=2)
        cbar.outline.set_visible(False)
        cbar.set_label(colorbar_label, fontsize=fontsize, labelpad=4)

    return fig
