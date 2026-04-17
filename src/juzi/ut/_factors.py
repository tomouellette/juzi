# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import pandas as pd

from anndata import AnnData
from typing import Dict


def factor_loadings(
    adata: AnnData,
    kept_only: bool = True,
) -> pd.DataFrame:
    """Return the gene × factor loading matrix as a labelled DataFrame.

    Parameters
    ----------
    adata : AnnData
        AnnData object produced by juzi.gp.nmf_fit.
    kept_only : bool
        If True, return only factors where juzi_keep is True.

    Returns
    -------
    pd.DataFrame
        (n_genes × n_factors) loading matrix with gene names as index
        and global factor labels as columns.
    """
    for field, store in [
        ("juzi_G", "varm"),
        ("juzi_names", "uns"),
        ("juzi_G_genes", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. Run juzi.gp.nmf_fit first."
            )

    G = np.asarray(adata.varm["juzi_G"])  # (n_genes × n_factors)
    names = np.array(adata.uns["juzi_names"], dtype=object)
    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)

    if G.shape[1] != len(names):
        raise ValueError("juzi_G column count does not match juzi_names length.")

    if kept_only:
        if "juzi_keep" not in adata.uns:
            raise KeyError(
                "'juzi_keep' not found in .uns. " "Run at least juzi.gp.nmf_fit first."
            )
        keep = np.array(adata.uns["juzi_keep"], dtype=bool)
        if len(keep) != G.shape[1]:
            raise ValueError(
                "juzi_keep length does not match number of factors in juzi_G."
            )
        G = G[:, keep]
        names = names[keep]

    # Build global factor labels: sample_F#
    sample_counts: Dict[str, int] = {}
    col_labels = []
    for name in names:
        idx = sample_counts.get(str(name), 0)
        col_labels.append(f"{name}_F{idx}")
        sample_counts[str(name)] = idx + 1

    return pd.DataFrame(G, index=gene_names, columns=col_labels)


def factor_scores(
    adata: AnnData,
) -> pd.DataFrame:
    """Return the cell × per-sample-factor score matrix as a labelled DataFrame.

    Requires keep_scores=True in juzi.gp.nmf_fit.

    Notes
    -----
    `juzi_scores` is stored in per-sample factor space, not global factor
    space. Each cell is scored only against the factors fit within its own
    sample, so columns are shared local factor coordinates (F0, F1, ...)
    rather than global factor identities.

    Parameters
    ----------
    adata : AnnData
        AnnData object produced by juzi.gp.nmf_fit with keep_scores=True.

    Returns
    -------
    pd.DataFrame
        (n_cells × sum(k)) score matrix with cell barcodes as index and
        local factor labels as columns.
    """
    if "juzi_scores" not in adata.obsm:
        raise KeyError(
            "'juzi_scores' not found in .obsm. "
            "Run juzi.gp.nmf_fit with keep_scores=True."
        )

    scores = np.asarray(adata.obsm["juzi_scores"])

    if scores.ndim != 2:
        raise ValueError("'juzi_scores' must be a 2D array.")

    n_local_factors = scores.shape[1]
    col_labels = [f"F{i}" for i in range(n_local_factors)]

    return pd.DataFrame(
        scores,
        index=adata.obs_names,
        columns=col_labels,
    )
