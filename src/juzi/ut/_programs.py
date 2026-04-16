# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import pandas as pd

from anndata import AnnData
from typing import Dict, List

from juzi.gp._nmf import _combined_score


def program_genes(
    adata: AnnData,
    n_top_genes: int = 50,
    use_combined: bool = True,
) -> Dict[str, List[str]]:
    """Extract top genes per consensus program.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_G in .uns.
    n_top_genes : int
        Number of top genes per program.
    use_combined : bool
        If True, rank by combined loading × specificity score.
        If False, rank by raw loading magnitude.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping program label to top gene names.
    """
    for field in ["juzi_cluster_G", "juzi_cluster_labels", "juzi_G_genes"]:
        if field not in adata.uns:
            raise KeyError(f"'{field}' not found in .uns. Run juzi.gp.cluster first.")

    if n_top_genes < 1:
        raise ValueError("n_top_genes must be >= 1.")

    G = adata.uns["juzi_cluster_G"]
    gene_names = np.array(adata.uns["juzi_G_genes"])
    labels = adata.uns["juzi_cluster_labels"]
    unique_C = np.unique(labels)

    if n_top_genes > G.shape[1]:
        raise ValueError(
            f"n_top_genes={n_top_genes} exceeds number of genes ({G.shape[1]})."
        )

    G_rank = _combined_score(G) if use_combined else G

    result = {}
    for i, c in enumerate(unique_C):
        top_idx = np.argsort(G_rank[i])[-n_top_genes:][::-1]
        result[f"C{int(c)}"] = gene_names[top_idx].tolist()

    return result


def program_compare(
    adata_a: AnnData,
    adata_b: AnnData,
    n_top_genes: int = 50,
    use_combined: bool = True,
) -> pd.DataFrame:
    """Compare consensus programs between two datasets via Jaccard similarity.

    Parameters
    ----------
    adata_a : AnnData
        First AnnData object.
    adata_b : AnnData
        Second AnnData object.
    n_top_genes : int
        Number of top genes per program for Jaccard computation.
    use_combined : bool
        If True, rank by combined loading × specificity score.
        If False, rank by raw loading magnitude.

    Returns
    -------
    pd.DataFrame
        (n_programs_a × n_programs_b) Jaccard similarity DataFrame.
    """
    genes_a = program_genes(adata_a, n_top_genes=n_top_genes, use_combined=use_combined)
    genes_b = program_genes(adata_b, n_top_genes=n_top_genes, use_combined=use_combined)
    programs_a = list(genes_a.keys())
    programs_b = list(genes_b.keys())

    sim = np.zeros((len(programs_a), len(programs_b)))
    for i, pa in enumerate(programs_a):
        set_a = set(genes_a[pa])
        for j, pb in enumerate(programs_b):
            set_b = set(genes_b[pb])
            union = set_a | set_b
            sim[i, j] = len(set_a & set_b) / len(union) if union else 0.0

    return pd.DataFrame(sim, index=programs_a, columns=programs_b)


def program_donors(adata: AnnData) -> pd.DataFrame:
    """Compute per-donor factor contribution to each consensus program.

    Parameters
    ----------
    adata : AnnData
        AnnData object produced by juzi.gp.cluster.

    Returns
    -------
    pd.DataFrame
        (n_donors × n_programs) factor count DataFrame.
    """
    for field in [
        "juzi_cluster_labels",
        "juzi_cluster_names",
        "juzi_cluster_samples",
    ]:
        if field not in adata.uns:
            raise KeyError(f"'{field}' not found in .uns. Run juzi.gp.cluster first.")

    labels = adata.uns["juzi_cluster_labels"]
    kept_names = np.array(adata.uns["juzi_cluster_names"])
    cluster_samples = adata.uns["juzi_cluster_samples"]
    unique_C = np.unique(labels)

    all_donors = sorted({d for donors in cluster_samples.values() for d in donors})

    result = pd.DataFrame(
        0,
        index=all_donors,
        columns=[f"C{int(c)}" for c in unique_C],
    )

    for c in unique_C:
        program_mask = labels == c
        program_names = kept_names[program_mask]
        for donor in program_names:
            if donor in result.index:
                result.loc[donor, f"C{int(c)}"] += 1

    return result
