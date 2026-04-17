# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from typing import Tuple


def programs_samples(
    adata: AnnData,
    metric: str = "n_factors",
    figsize: Tuple[float, float] | None = None,
    cmap: str = "Blues",
    fontsize: int = 8,
    rasterized: bool = False,
    show_colorbar: bool = True,
    cbar_label: str | None = None,
) -> plt.Axes:
    """Plot donor contributions to each program.

    Displays a program-by-donor heatmap showing how much each donor
    contributes to each consensus program.

    Parameters
    ----------
    adata : AnnData
        Requires:
            - juzi_cluster_labels
            - juzi_cluster_names
            - optionally juzi_aggregate_scores (for metric="mean_score")
    metric : str
        One of:
            - "n_factors" : number of member factors per donor (default)
            - "fraction"  : fraction of program factors from each donor
            - "mean_score": mean program score per donor (requires aggregate)
    figsize : Tuple[float, float] | None
        Figure size in inches.
    cmap : str
        Colormap.
    fontsize : int
        Font size.
    rasterized : bool
        Rasterize heatmap.
    show_colorbar : bool
        Whether to show colorbar.
    cbar_label : str | None
        Label for colorbar.

    Returns
    -------
    plt.Axes
    """
    # Validate

    for field in ["juzi_cluster_labels", "juzi_cluster_names"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. " "Run juzi.gp.programs_cluster first."
            )

    labels = np.array(adata.uns["juzi_cluster_labels"])
    names = np.array(adata.uns["juzi_cluster_names"])

    unique_C = np.unique(labels)
    donors = np.unique(names)

    n_programs = len(unique_C)
    n_donors = len(donors)

    # Compute matrix

    M = np.zeros((n_programs, n_donors), dtype=float)

    if metric in ("n_factors", "fraction"):
        for i, c in enumerate(unique_C):
            mask = labels == c
            donors_c = names[mask]

            for j, d in enumerate(donors):
                count = np.sum(donors_c == d)
                M[i, j] = count

            if metric == "fraction":
                total = M[i].sum()
                if total > 0:
                    M[i] /= total

    elif metric == "mean_score":
        if "juzi_aggregate_scores" not in adata.uns:
            raise KeyError(
                "'juzi_aggregate_scores' not found. "
                "Run juzi.gp.score_aggregate first."
            )

        agg = adata.uns["juzi_aggregate_scores"]

        for i, c in enumerate(unique_C):
            col = f"P{int(c)}"
            if col not in agg.columns:
                raise KeyError(f"{col} not found in aggregate scores.")

            for j, d in enumerate(donors):
                if d in agg.index:
                    M[i, j] = agg.loc[d, col]

    else:
        raise ValueError(
            "metric must be one of {'n_factors', 'fraction', 'mean_score'}."
        )

    # Figure

    if figsize is None:
        figsize = (0.6 * n_donors + 2, 0.4 * n_programs + 1.5)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        M,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        rasterized=rasterized,
    )

    # Axes

    ax.set_xticks(np.arange(n_donors))
    ax.set_xticklabels(donors, rotation=90, fontsize=fontsize)

    ax.set_yticks(np.arange(n_programs))
    ax.set_yticklabels([f"C{int(c)}" for c in unique_C], fontsize=fontsize)

    ax.set_xlabel("Donors", fontsize=fontsize)
    ax.set_ylabel("Programs", fontsize=fontsize)

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar

    if show_colorbar:
        if cbar_label is None:
            cbar_label = metric.replace("_", " ")

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=fontsize - 1, length=0)
        cbar.set_label(cbar_label, fontsize=fontsize)

    fig.tight_layout()

    return ax
