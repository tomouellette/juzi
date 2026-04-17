# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np

from anndata import AnnData
from typing import Dict, List

from ._nmf import _combined_score


def programs_stability(
    adata: AnnData,
    top_k: int = 50,
    min_program_genes: int = 5,
    copy: bool = False,
) -> AnnData | None:
    """Assess program stability by leave-one-donor-out gene-set recovery.

    For each consensus program, computes how stable its canonical gene set
    remains when each contributing donor is removed in turn.

    For a given program and held-out donor:

    1. Collect all retained cluster-member factors for that program that do
       not come from the held-out donor.
    2. Recompute a leave-one-donor-out program gene set from those remaining
       member factors using the same rule used during clustering:

       - Centroid mode: top genes by combined score of the mean factor
         loading across remaining members.
       - Progressive mode: top genes by frequency across member top-top_k
         sets, with ties at the boundary broken by the maximum NMF loading
         score of the tied gene across remaining members.

    3. Compute Jaccard similarity between the recomputed gene set and the
       original canonical program gene set.

    The final stability score for each program is the mean Jaccard similarity
    across all valid donor holdouts (donors that contribute at least one
    factor to the program).

    Parameters
    ----------
    adata : AnnData
        AnnData object produced by juzi.gp.programs_cluster.
        Requires juzi_cluster_genes, juzi_cluster_labels,
        juzi_cluster_names, juzi_cluster_order, juzi_G_genes in .uns
        and juzi_G in .varm.
    top_k : int
        Maximum number of genes in each recomputed leave-one-donor-out
        program gene set. Should match the top_k / n_top_genes used during
        clustering. If smaller than the stored canonical gene list, the
        canonical set is truncated and a warning is emitted.
    min_program_genes : int
        Minimum number of genes required in the original canonical program
        gene set to evaluate stability. Programs with fewer genes will still
        be scored but a warning is emitted.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with juzi_stability and juzi_stability_meta in .uns.
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_cluster_genes", "uns"),
        ("juzi_cluster_labels", "uns"),
        ("juzi_cluster_names", "uns"),
        ("juzi_cluster_order", "uns"),
        ("juzi_G_genes", "uns"),
        ("juzi_G", "varm"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. "
                "Run juzi.gp.programs_cluster first."
            )

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    if min_program_genes < 1:
        raise ValueError("min_program_genes must be >= 1.")

    # Setup

    cluster_genes = adata.uns["juzi_cluster_genes"]
    cluster_labels = np.array(adata.uns["juzi_cluster_labels"])
    cluster_names = np.array(adata.uns["juzi_cluster_names"], dtype=object)
    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)
    full_G = adata.varm["juzi_G"].T  # (n_total_factors, n_genes)

    # juzi_cluster_order records the global factor index at each cluster-space
    # position, in the same display order as juzi_cluster_labels / names.
    # Indexing full_G with it gives G_cluster rows that are 1-to-1 aligned
    # with cluster_labels and cluster_names — no ordering mismatch.
    cluster_order = np.array(adata.uns["juzi_cluster_order"], dtype=int)
    G_cluster = full_G[cluster_order]  # (n_cluster_factors, n_genes)

    strategy = adata.uns.get("juzi_cluster_meta", {}).get("strategy", "centroid")
    if strategy not in ("centroid", "progressive"):
        warnings.warn(
            "Unknown clustering strategy; falling back to centroid-style "
            "gene reconstruction for stability analysis.",
            UserWarning,
            stacklevel=2,
        )
        strategy = "centroid"

    unique_clusters = np.unique(cluster_labels)
    program_labels = [f"C{int(c)}" for c in unique_clusters]

    donors = sorted(np.unique(cluster_names).tolist())
    n_programs = len(unique_clusters)
    n_donors = len(donors)

    stability_matrix = np.full((n_programs, n_donors), np.nan, dtype=np.float32)
    stability_score = np.zeros(n_programs, dtype=np.float32)

    gene_to_idx = {g: i for i, g in enumerate(gene_names.tolist())}

    for p_idx, c in enumerate(unique_clusters):
        prog_key = int(c)
        canonical_genes = cluster_genes.get(prog_key, [])
        canonical_genes = [g for g in canonical_genes if g in gene_to_idx]

        if len(canonical_genes) == 0:
            warnings.warn(
                f"Program C{prog_key} has no canonical genes; stability will be 0.",
                UserWarning,
                stacklevel=2,
            )
            stability_score[p_idx] = 0.0
            continue

        if len(canonical_genes) < min_program_genes:
            warnings.warn(
                f"Program C{prog_key} has fewer than min_program_genes="
                f"{min_program_genes}; stability may be uninformative.",
                UserWarning,
                stacklevel=2,
            )

        if len(canonical_genes) > top_k:
            warnings.warn(
                f"Program C{prog_key} has {len(canonical_genes)} canonical genes "
                f"but top_k={top_k}; canonical set truncated to {top_k} genes "
                "for stability computation.",
                UserWarning,
                stacklevel=2,
            )

        canonical_set = set(canonical_genes[:top_k])

        # member_mask selects cluster-space rows for this program.
        # G_cluster is indexed via cluster_order so rows align exactly with
        # cluster_labels and cluster_names.
        member_mask = cluster_labels == c
        member_names = cluster_names[member_mask]
        member_G = G_cluster[member_mask]

        for d_idx, donor in enumerate(donors):
            loo_mask = member_names != donor

            if loo_mask.sum() == 0:
                continue

            G_loo = member_G[loo_mask]

            loo_genes = _recompute_program_genes(
                G=G_loo,
                gene_names=gene_names,
                gene_to_idx=gene_to_idx,
                top_k=top_k,
                strategy=strategy,
            )

            loo_set = set(loo_genes)
            union = canonical_set | loo_set
            jacc = len(canonical_set & loo_set) / len(union) if union else 0.0
            stability_matrix[p_idx, d_idx] = float(jacc)

        valid = ~np.isnan(stability_matrix[p_idx])
        stability_score[p_idx] = (
            float(np.nanmean(stability_matrix[p_idx])) if valid.any() else 0.0
        )

    n_valid_donors = (~np.isnan(stability_matrix)).sum(axis=1).astype(int)

    adata.uns["juzi_stability"] = {
        "score": stability_score,
        "matrix": stability_matrix,
        "programs": program_labels,
        "donors": donors,
        "top_k": top_k,
        "strategy": strategy,
        "n_valid_donors": n_valid_donors,
    }

    adata.uns["juzi_stability_meta"] = {
        "top_k": int(top_k),
        "min_program_genes": int(min_program_genes),
        "strategy": strategy,
        "n_programs": int(n_programs),
        "n_donors": int(n_donors),
    }

    return adata if copy else None


def _recompute_program_genes(
    G: np.ndarray,
    gene_names: np.ndarray,
    gene_to_idx: Dict[str, int],
    top_k: int,
    strategy: str,
) -> List[str]:
    """Recompute a leave-one-donor-out program gene set from factor loadings."""
    if G.shape[0] == 0:
        return []

    top_k = min(top_k, G.shape[1])

    if strategy == "centroid":
        centroid = G.mean(axis=0, keepdims=True)
        rank = _combined_score(centroid)[0]
        top_idx = np.argsort(rank)[-top_k:][::-1]
        return gene_names[top_idx].tolist()

    if strategy == "progressive":
        return _recompute_progressive(
            G=G,
            gene_names=gene_names,
            gene_to_idx=gene_to_idx,
            top_k=top_k,
        )

    # Fallback
    centroid = G.mean(axis=0, keepdims=True)
    rank = _combined_score(centroid)[0]
    top_idx = np.argsort(rank)[-top_k:][::-1]
    return gene_names[top_idx].tolist()


def _recompute_progressive(
    G: np.ndarray,
    gene_names: np.ndarray,
    gene_to_idx: Dict[str, int],
    top_k: int,
) -> List[str]:
    """Reconstruct a progressive-mode MP gene set from a subset of member factors.

    Mirrors _mp_from_history in _cluster.py exactly: frequency table over
    all member top-k gene sets, with ties at the top_k boundary broken by
    maximum NMF loading score across remaining members.
    """
    gene_freq: Dict[str, int] = {}
    gene_max_score: Dict[str, float] = {}

    for i in range(G.shape[0]):
        top_idx = np.argsort(G[i])[-top_k:]
        genes_i = gene_names[top_idx].tolist()

        for g in genes_i:
            gene_freq[g] = gene_freq.get(g, 0) + 1
            s = float(G[i, gene_to_idx[g]])
            if s > gene_max_score.get(g, 0.0):
                gene_max_score[g] = s

    if not gene_freq:
        return []

    sorted_genes = sorted(gene_freq.keys(), key=lambda g: gene_freq[g], reverse=True)

    if len(sorted_genes) <= top_k:
        return sorted(
            sorted_genes,
            key=lambda g: (gene_freq[g], gene_max_score.get(g, 0.0)),
            reverse=True,
        )

    boundary_freq = gene_freq[sorted_genes[top_k - 1]]
    above = [g for g in sorted_genes if gene_freq[g] > boundary_freq]
    at_border = [g for g in sorted_genes if gene_freq[g] == boundary_freq]
    n_needed = top_k - len(above)

    if len(at_border) <= n_needed:
        return above + at_border

    at_border_sorted = sorted(
        at_border,
        key=lambda g: gene_max_score.get(g, 0.0),
        reverse=True,
    )

    return above + at_border_sorted[:n_needed]
