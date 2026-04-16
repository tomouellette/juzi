# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from typing import Dict, Tuple


def programs_jackknife(
    adata: AnnData,
    figsize: Tuple[float, float] | None = None,
    palette: Dict[int, str] | None = None,
    fontsize: int = 8,
    cmap: str = "Reds_r",
    show_donors: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot donor jackknife stability scores per consensus program.

    Displays a heatmap of shape (K x N) where rows are consensus programs
    and columns are donors. Cell colour encodes the maximum Jaccard
    similarity between the reference program and any jackknife program
    when that donor is held out. The mean stability score per program
    is annotated on the right.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_jackknife in .uns, produced by
        juzi.gp.programs_jackknife.
    figsize : Tuple[float, float] | None
        Figure size in inches. If None, inferred from number of programs
        and donors.
    palette : Dict[int, str] | None
        Dictionary mapping cluster label integers to colours for the
        program row labels. If None, generated via glasbey.
    fontsize : int
        Font size for all text elements.
    cmap : str
        Colormap for the stability heatmap. Default "Reds_r" so low
        stability (unstable) appears dark red and high stability appears
        light.
    show_donors : bool
        If True, show donor names on the x-axis. If False, hide donor
        labels — useful when N is large.
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """
    if "juzi_jackknife" not in adata.uns:
        raise KeyError(
            "'juzi_jackknife' not found in .uns. "
            "Run juzi.gp.programs_jackknife first."
        )

    if "juzi_cluster_labels" not in adata.uns:
        raise KeyError(
            "'juzi_cluster_labels' not found in .uns. "
            "Run juzi.gp.programs_cluster first."
        )

    # Extract results

    jack = adata.uns["juzi_jackknife"]
    S_mat = jack["stability_matrix"] # (K × N)
    stability  = jack["stability"] # (K,)
    donors     = jack["donors"] # (N,)
    labels     = adata.uns["juzi_cluster_labels"]
    unique_C   = np.unique(labels)
    K          = len(unique_C)
    N          = len(donors)

    prog_labels = [f"C{int(c)}" for c in unique_C]

    # Palette

    if palette is None:
        colors  = glasbey.create_palette(
            K,
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_C)}

    # Figure setup

    created_fig = ax is None
    if created_fig:
        if figsize is None:
            figsize = (
                max(4.0, 0.25 * N + 2.0),
                max(2.0, 0.4  * K + 1.0),
            )
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # Heatmap

    im = ax.imshow(
        S_mat,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    # Program colour strip on left

    for k, c in enumerate(unique_C):
        color = palette[int(c)]
        ax.add_patch(plt.Rectangle(
            (-1.5, k - 0.5),
            0.8, 1.0,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
            transform=ax.transData,
        ))

    # Mean stability annotation on right

    for k in range(K):
        ax.text(
            N + 0.3,
            k,
            f"{stability[k]:.2f}",
            fontsize=fontsize - 1,
            va="center",
            ha="left",
            color="black",
        )

    ax.text(
        N + 0.3,
        -0.8,
        "Mean",
        fontsize=fontsize - 1,
        va="center",
        ha="left",
        color="grey",
    )

    # Axes

    ax.set_yticks(range(K))
    ax.set_yticklabels(prog_labels, fontsize=fontsize)
    ax.tick_params(axis="y", length=0)

    if show_donors:
        ax.set_xticks(range(N))
        ax.set_xticklabels(
            donors,
            fontsize=fontsize - 1,
            rotation=90,
            ha="center",
        )
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"Donors (n = {N})", fontsize=fontsize)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar

    if created_fig:
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            shrink=0.4,
            pad=0.12,
        )
        cbar.ax.tick_params(labelsize=fontsize - 1, length=2)
        cbar.outline.set_visible(False)
        cbar.set_label("Jaccard stability", fontsize=fontsize, labelpad=4)

    return ax
