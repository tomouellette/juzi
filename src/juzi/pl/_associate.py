# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import glasbey
import numpy as np
import matplotlib.pyplot as plt

from anndata import AnnData
from typing import Dict, List, Tuple


def associate(
    adata: AnnData,
    figsize: Tuple[float, float] | None = None,
    palette: Dict[int, str] | None = None,
    fontsize: int = 8,
    dot_size: float = 20.0,
    linewidth: float = 1.0,
    padj_thresh: float = 0.05,
    show_threshold: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot covariate association results per consensus program.

    Displays a coefficient plot with one point per program showing the
    beta coefficient and standard error from juzi.gp.associate. Programs
    are ordered by beta value, coloured by program identity, and styled
    by significance.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_association in .uns, produced by
        juzi.gp.associate.
    figsize : Tuple[float, float] | None
        Figure size in inches. If None, inferred from number of programs.
    palette : Dict[int, str] | None
        Dictionary mapping program label integers to colours. Should match
        the palette used in juzi.pl.similarity and juzi.pl.loadings.
        If None, generated automatically via glasbey.
    fontsize : int
        Font size for all text elements.
    dot_size : float
        Size of the point markers.
    linewidth : float
        Line width of the error bars.
    padj_thresh : float
        Adjusted p-value threshold for significance. Programs below this
        threshold are plotted as filled circles, others as open circles.
    show_threshold : bool
        If True, draw a vertical dashed line at beta = 0.
    ax : plt.Axes | None
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """
    if "juzi_association" not in adata.uns:
        raise KeyError(
            "'juzi_association' not found in .uns. "
            "Run juzi.gp.associate before plotting."
        )

    if "juzi_cluster_labels" not in adata.uns:
        raise KeyError(
            "'juzi_cluster_labels' not found in .uns. "
            "Run juzi.gp.cluster before plotting."
        )

    if not 0.0 <= padj_thresh <= 1.0:
        raise ValueError("padj_thresh must be in [0, 1].")

    # Setup

    df = adata.uns["juzi_association"].copy()
    labels = adata.uns["juzi_cluster_labels"]
    unique_C = np.unique(labels)
    n_programs = len(unique_C)

    # Palette

    if palette is None:
        colors = glasbey.create_palette(
            n_programs,
            chroma_bounds=(5, 40),
            lightness_bounds=(0, 100),
        )
        palette = {int(c): colors[i] for i, c in enumerate(unique_C)}

    # Map program string (P0, P1, ...) to cluster integer label
    prog_to_label = {f"P{int(c)}": int(c) for c in unique_C}

    # Sort by beta

    df = df.sort_values("beta", ascending=True).reset_index(drop=True)

    # Figure setup

    created_fig = ax is None
    if created_fig:
        if figsize is None:
            figsize = (2.5, 0.35 * n_programs + 0.75)
        fig, ax = plt.subplots(figsize=figsize)

    # Plot

    for i, row in df.iterrows():
        prog = row["program"]
        beta = row["beta"]
        se = row["se"]
        padj = row["padj"]
        label_int = prog_to_label.get(prog, 0)
        color = palette.get(label_int, "#888888")
        sig = padj < padj_thresh

        # Error bar
        ax.plot(
            [beta - se, beta + se],
            [i, i],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
        )

        # Point — filled if significant, open if not
        ax.scatter(
            beta,
            i,
            s=dot_size,
            color=color if sig else "white",
            edgecolors=color,
            linewidths=linewidth,
            zorder=3,
        )

    # Zero line

    if show_threshold:
        ax.axvline(
            0,
            color="black",
            linewidth=0.5,
            linestyle="--",
            alpha=0.5,
            zorder=0,
        )

    # y-axis — program labels

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        df["program"].tolist(),
        fontsize=fontsize,
    )
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, labelsize=fontsize)

    # x-axis label

    covariate = df["covariate"].iloc[0] if len(df) > 0 else "covariate"
    ax.set_xlabel(f"β ({covariate})", fontsize=fontsize)

    # Spine styling

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.margins(y=0.05)

    # Legend

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=4,
            label=f"FDR < {padj_thresh}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=4,
            label=f"FDR ≥ {padj_thresh}",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        fontsize=fontsize - 1,
        frameon=False,
        loc="lower right",
        handletextpad=0.3,
        labelspacing=0.3,
    )

    if created_fig:
        fig.tight_layout()

    return ax
