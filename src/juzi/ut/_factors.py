# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import pandas as pd

from anndata import AnnData
from typing import Dict, List

from juzi.gp._nmf import _combined_score


def factor_loadings(
    adata: AnnData,
    kept_only: bool = True,
) -> pd.DataFrame:
    """Extract the gene × factor loading matrix as a labelled DataFrame.

    Parameters
    ----------
    adata : AnnData
        AnnData object produced by juzi.gp.nmf.
    kept_only : bool
        If True, return only factors where juzi_keep is True.

    Returns
    -------
    pd.DataFrame
        (n_genes × n_factors) loading DataFrame.
    """
    for field, store in [
        ("juzi_G", "varm"),
        ("juzi_G_genes", "uns"),
        ("juzi_names", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(f"'{field}' not found in .{store}. Run juzi.gp.nmf first.")

    G = adata.varm["juzi_G"]
    gene_names = adata.uns["juzi_G_genes"]
    names = np.array(adata.uns["juzi_names"])

    donor_counts = {}
    col_labels = []
    for name in names:
        idx = donor_counts.get(name, 0)
        col_labels.append(f"{name}_{idx}")
        donor_counts[name] = idx + 1

    if kept_only and "juzi_keep" in adata.uns:
        keep = adata.uns["juzi_keep"]
        G = G[:, keep]
        col_labels = [col_labels[i] for i in range(len(col_labels)) if keep[i]]

    return pd.DataFrame(G, index=gene_names, columns=col_labels)


def factor_scores(
    adata: AnnData,
    kept_only: bool = True,
) -> pd.DataFrame:
    """Extract the cell × factor score matrix as a labelled DataFrame.

    Parameters
    ----------
    adata : AnnData
        AnnData object produced by juzi.gp.nmf with keep_scores=True.
    kept_only : bool
        If True, return only factors where juzi_keep is True.

    Returns
    -------
    pd.DataFrame
        (n_cells × n_factors) score DataFrame.
    """
    if "juzi_scores" not in adata.obsm:
        raise KeyError(
            "'juzi_scores' not found in .obsm. "
            "Run juzi.gp.nmf with keep_scores=True first."
        )

    if "juzi_names" not in adata.uns:
        raise KeyError("'juzi_names' not found in .uns. Run juzi.gp.nmf first.")

    scores = adata.obsm["juzi_scores"]
    names = np.array(adata.uns["juzi_names"])

    donor_counts = {}
    col_labels = []
    for name in names:
        idx = donor_counts.get(name, 0)
        col_labels.append(f"{name}_{idx}")
        donor_counts[name] = idx + 1

    if kept_only and "juzi_keep" in adata.uns:
        keep = adata.uns["juzi_keep"]
        scores = scores[:, keep]
        col_labels = [col_labels[i] for i in range(len(col_labels)) if keep[i]]

    return pd.DataFrame(scores, index=adata.obs_names, columns=col_labels)
