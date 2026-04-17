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
        juzi.gp.score_cells, and a 2D embedding in .obsm[basis].
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
            "Run juzi.gp.score_cells before plotting."
        )

    if basis not in adata.obsm:
        raise KeyError(
            f"'{basis}' not found in adata.obsm. "
            "Check your basis argument or compute an embedding first."
        )

    embedding = np.asarray(adata.obsm[basis])
    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(f"Embedding '{basis}' must have at least 2 dimensions.")

    # Setup

    program_scores = np.asarray(adata.obsm["juzi_program_scores"], dtype=float)
    if program_scores.ndim != 2:
        raise ValueError("'juzi_program_scores' must be a 2D array.")

    n_programs = program_scores.shape[1]

    # Program labels: prefer cluster labels if available, otherwise use P#
    if "juzi_cluster_labels" in adata.uns:
        unique_C = np.unique(np.asarray(adata.uns["juzi_cluster_labels"]))
        if len(unique_C) == n_programs:
            prog_labels = [f"C{int(c)}" for c in unique_C]
        else:
            prog_labels = [f"P{i}" for i in range(n_programs)]
    else:
        prog_labels = [f"P{i}" for i in range(n_programs)]

    coords_x = embedding[:, 0]
    coords_y = embedding[:, 1]

    # Colormap normalisation

    all_scores = program_scores.ravel()
    all_scores = all_scores[np.isfinite(all_scores)]

    if len(all_scores) == 0:
        raise ValueError("juzi_program_scores contains no finite values.")

    if vmin is None:
        vmin = float(np.percentile(all_scores, 1))
    if vmax is None:
        vmax = float(np.percentile(all_scores, 99))

    if vmin == vmax:
        vmax = vmin + 1e-6

    if vmin < vcenter < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Figure layout

    n_cols = min(n_programs, ncols)
    n_rows = int(np.ceil(n_programs / n_cols))

    if figsize is None:
        panel_size = 2.0
        extra_w = 0.45 if show_colorbar else 0.0
        figsize = (panel_size * n_cols + extra_w, panel_size * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()

    # Plot each program

    sc = None
    basis_label = basis.replace("X_", "").upper()

    for i, ax in enumerate(axes_flat[:n_programs]):
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
            prog_labels[i],
            fontsize=fontsize,
            pad=3,
        )

        ax.set_xticks([])
        ax.set_yticks([])

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
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=fontsize - 1, length=2)
        cbar.outline.set_visible(False)
        cbar.set_label(colorbar_label, fontsize=fontsize, labelpad=4)
    else:
        fig.tight_layout()

    return fig
