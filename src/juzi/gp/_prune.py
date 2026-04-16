# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np

from anndata import AnnData
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from typing import List
from tqdm import tqdm

from ._nmf import _recompute_keep, _combined_score


def nmf_prune(
    adata: AnnData,
    top_k: int = 50,
    min_similarity: float = 0.7,
    min_k: int = 1,
    matching: str = "greedy",
    use_combined: bool = True,
    n_jobs: int = 1,
    prefer: str | None = None,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Prune non-recurrent intra-sample factors based on top-gene Jaccard similarity.

    For each sample, NMF was fit at multiple resolutions k. A factor is
    considered recurrent if it shares sufficient top-gene overlap with at
    least one factor from each of min_k other resolutions. Non-recurrent
    factors are masked out before inter-sample similarity is computed.

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.gp.nmf_fit. Must contain juzi_G in .varm
        and juzi_k, juzi_names in .uns.
    top_k : int
        Number of top-loaded genes used to compute Jaccard similarity
        between factors. Must be less than or equal to the number of genes.
    min_similarity : float
        Minimum Jaccard similarity for two factors to be considered
        recurrent matches. Must be in [0, 1].
    min_k : int
        Minimum number of other k resolutions a factor must match to be
        retained. Must be less than or equal to len(juzi_k).
    matching : str
        Strategy for matching factors across resolutions. "greedy" takes
        the best available match for each factor independently. "hungarian"
        finds the globally optimal one-to-one assignment between resolutions
        via the Hungarian algorithm.
    use_combined : bool
        If True, rank genes by combined loading × specificity score
        (G * G / G.sum(axis=0)) before selecting top_k genes for Jaccard
        computation. Downweights genes that load broadly across all factors
        within a donor, making within-donor matching more discriminative.
        If False, rank by raw loading magnitude.
    n_jobs : int
        Number of parallel workers for pruning across samples.
    prefer : str | None
        Joblib parallelisation backend preference.
    silent : bool
        If True, suppress progress bar.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields updated:
            .uns["juzi_keep_prune"] : boolean mask of recurrent factors
            .uns["juzi_keep"]       : intersection of all three stage masks
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_G",     "varm"),
        ("juzi_k",     "uns"),
        ("juzi_names", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. Run juzi.gp.nmf_fit first."
            )

    if top_k > adata.n_vars:
        raise ValueError(
            f"top_k={top_k} exceeds number of genes ({adata.n_vars}). "
            "Lower top_k."
        )

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1].")

    if matching not in ("greedy", "hungarian"):
        raise ValueError("matching must be 'greedy' or 'hungarian'.")

    n_resolutions = len(adata.uns["juzi_k"])
    if min_k > n_resolutions:
        raise ValueError(
            f"min_k={min_k} exceeds number of k resolutions ({n_resolutions}). "
            "Lower min_k or add more values to k."
        )

    # Split juzi_G into per-sample factor blocks

    names    = np.array(adata.uns["juzi_names"])
    G        = adata.varm["juzi_G"].T # (n_total_factors × n_genes)
    k_list   = adata.uns["juzi_k"]
    n_comps  = int(np.sum(k_list))
    n_unique = len(np.unique(names))

    if len(names) != n_unique * n_comps:
        raise ValueError(
            f"juzi_names length ({len(names)}) does not match "
            f"n_samples ({n_unique}) × sum(k) ({n_comps}). "
            "juzi_G may be corrupted — re-run juzi.gp.nmf_fit."
        )

    split_points = np.arange(n_comps, n_unique * n_comps, n_comps)
    per_sample_G = np.split(G, split_points)

    # Prune per sample in parallel

    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_prune)(
            factors=sample_G,
            k=k_list,
            top_k=top_k,
            min_similarity=min_similarity,
            min_k=min_k,
            matching=matching,
            use_combined=use_combined,
        )
        for sample_G in tqdm(
            per_sample_G,
            desc="[juzi] Pruning",
            disable=silent,
        )
    )

    # Build global boolean keep mask

    mask               = np.zeros(len(names), dtype=bool)
    cumulative_offsets = np.arange(n_unique) * n_comps

    for sample_idx, local_keep_idx in enumerate(results):
        if len(local_keep_idx) > 0:
            global_idx       = cumulative_offsets[sample_idx] + local_keep_idx
            mask[global_idx] = True

    # Update stage mask and recompute juzi_keep

    adata.uns["juzi_keep_prune"] = mask
    _recompute_keep(adata)

    return adata if copy else None


def _jaccard(set1: set, set2: set) -> float:
    union = set1 | set2
    if len(union) == 0:
        return 0.0
    return len(set1 & set2) / len(union)


def _similarity_matrix(
    top_genes_a: list[set],
    top_genes_b: list[set],
) -> np.ndarray:
    sim = np.zeros((len(top_genes_a), len(top_genes_b)))
    for i, genes_i in enumerate(top_genes_a):
        for j, genes_j in enumerate(top_genes_b):
            sim[i, j] = _jaccard(genes_i, genes_j)
    return sim


def _match_greedy(
    top_genes_a: list[set],
    top_genes_b: list[set],
    min_similarity: float,
) -> set[int]:
    sim        = _similarity_matrix(top_genes_a, top_genes_b)
    best_per_a = sim.max(axis=1)
    return set(np.where(best_per_a >= min_similarity)[0].tolist())


def _match_hungarian(
    top_genes_a: list[set],
    top_genes_b: list[set],
    min_similarity: float,
) -> set[int]:
    sim              = _similarity_matrix(top_genes_a, top_genes_b)
    row_idx, col_idx = linear_sum_assignment(-sim)
    return {r for r, c in zip(row_idx, col_idx) if sim[r, c] >= min_similarity}


def _prune(
    factors: np.ndarray,
    k: List[int],
    top_k: int,
    min_similarity: float,
    min_k: int,
    matching: str,
    use_combined: bool,
) -> np.ndarray:
    """Identify recurrent factors within a single sample across k resolutions."""
    split_points  = np.cumsum(k)[:-1]
    factors_by_k  = np.split(factors, split_points)

    top_genes_by_k = []
    for resolution in factors_by_k:
        # resolution shape: (k_i × n_genes)
        scored = _combined_score(resolution) if use_combined else resolution
        top_genes_by_k.append([
            set(np.argsort(scored[i])[-top_k:])
            for i in range(len(resolution))
        ])

    match_fn           = _match_hungarian if matching == "hungarian" else _match_greedy
    keep_local_idx     = []
    cumulative_offsets = np.concatenate([[0], np.cumsum(k)])

    for k_idx, resolution_genes in enumerate(top_genes_by_k):
        n_matching = np.zeros(len(resolution_genes), dtype=int)

        for other_k_idx, other_resolution_genes in enumerate(top_genes_by_k):
            if other_k_idx == k_idx:
                continue
            matched_indices = match_fn(
                resolution_genes,
                other_resolution_genes,
                min_similarity,
            )
            for i in matched_indices:
                n_matching[i] += 1

        for i, n in enumerate(n_matching):
            if n >= min_k - 1:
                keep_local_idx.append(cumulative_offsets[k_idx] + i)

    return np.array(keep_local_idx, dtype=int)
