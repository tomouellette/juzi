# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import pandas as pd

from anndata import AnnData
from typing import Dict, List


def programs_genes(
    adata: AnnData,
) -> Dict[str, List[str]]:
    """Extract canonical gene set per consensus program.

    Returns the gene sets stored in `juzi_cluster_genes`, which is the
    canonical program definition in the refactored API. For centroid mode
    these are typically the top genes selected at cluster time; for
    progressive mode these are the MP genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_genes in .uns, produced by
        juzi.gp.programs_cluster.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping program label (e.g. "C0") to gene names.
    """
    for field in ["juzi_cluster_genes", "juzi_cluster_labels"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. " "Run juzi.gp.programs_cluster first."
            )

    cluster_genes = adata.uns["juzi_cluster_genes"]
    unique_C = np.unique(np.array(adata.uns["juzi_cluster_labels"]))

    return {f"C{int(c)}": list(cluster_genes.get(int(c), [])) for c in unique_C}


def programs_compare(
    adata_a: AnnData,
    adata_b: AnnData,
) -> pd.DataFrame:
    """Compare consensus programs between two datasets via Jaccard similarity.

    Uses the canonical gene sets from `juzi_cluster_genes` in each dataset.

    Parameters
    ----------
    adata_a : AnnData
        First AnnData object produced by juzi.gp.programs_cluster.
    adata_b : AnnData
        Second AnnData object produced by juzi.gp.programs_cluster.

    Returns
    -------
    pd.DataFrame
        (n_programs_a × n_programs_b) Jaccard similarity DataFrame.
    """
    genes_a = programs_genes(adata_a)
    genes_b = programs_genes(adata_b)

    programs_a = list(genes_a.keys())
    programs_b = list(genes_b.keys())

    sim = np.zeros((len(programs_a), len(programs_b)), dtype=float)

    for i, pa in enumerate(programs_a):
        set_a = set(genes_a[pa])
        for j, pb in enumerate(programs_b):
            set_b = set(genes_b[pb])
            union = set_a | set_b
            sim[i, j] = len(set_a & set_b) / len(union) if union else 0.0

    return pd.DataFrame(sim, index=programs_a, columns=programs_b)


def programs_donors(
    adata: AnnData,
) -> pd.DataFrame:
    """Compute per-donor factor contribution to each consensus program.

    Parameters
    ----------
    adata : AnnData
        AnnData object produced by juzi.gp.programs_cluster.

    Returns
    -------
    pd.DataFrame
        (n_donors × n_programs) factor count DataFrame. Each cell
        contains the number of factors from that donor that were
        assigned to that program.
    """
    for field in ["juzi_cluster_labels", "juzi_cluster_names"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. " "Run juzi.gp.programs_cluster first."
            )

    labels = np.array(adata.uns["juzi_cluster_labels"])
    kept_names = np.array(adata.uns["juzi_cluster_names"], dtype=object)
    unique_C = np.unique(labels)

    if len(labels) != len(kept_names):
        raise ValueError(
            "juzi_cluster_labels and juzi_cluster_names must have the same length."
        )

    all_donors = np.unique(kept_names)

    result = pd.DataFrame(
        0,
        index=all_donors,
        columns=[f"C{int(c)}" for c in unique_C],
        dtype=int,
    )

    for c in unique_C:
        program_mask = labels == c
        program_names = kept_names[program_mask]
        donor_counts = pd.Series(program_names).value_counts()

        for donor, count in donor_counts.items():
            result.loc[donor, f"C{int(c)}"] = int(count)

    result.index.name = "donor"

    return result
