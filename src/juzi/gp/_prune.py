# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
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
    min_similarity: float = 0.1,
    dedup_similarity: float = 0.7,
    min_k: int = 1,
    matching: str = "hungarian",
    deduplicate: bool = True,
    use_combined: bool = False,
    n_jobs: int = 1,
    prefer: str | None = None,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Prune non-recurrent and redundant intra-sample factors.

    Applies two sequential filters per sample:

    1. Recurrence filter — a factor is kept if it shares sufficient
       top-gene overlap (Jaccard >= min_similarity) with at least one
       factor from each of min_k other resolutions.

    2. Deduplication filter (when deduplicate=True) — among factors that
       passed recurrence, any two factors with Jaccard >= dedup_similarity
       are considered redundant. The most central factor per redundant
       group is retained. dedup_similarity should be set higher than
       min_similarity since deduplication tests for near-identical
       programs rather than merely related ones.

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.gp.nmf_fit.
    top_k : int
        Number of top-loaded genes for Jaccard computation.
    min_similarity : float
        Minimum Jaccard for recurrence detection across resolutions.
        Must be in [0, 1].
    dedup_similarity : float
        Minimum Jaccard for within-sample deduplication. Should be
        higher than min_similarity — two factors sharing this fraction
        of top genes are considered the same program. Must be in [0, 1].
        Ignored when deduplicate=False.
    min_k : int
        Minimum other resolutions a factor must match.
    matching : str
        "hungarian" (default) or "greedy".
    deduplicate : bool
        If True, apply within-sample deduplication after recurrence.
    use_combined : bool
        If True, rank genes by weighted log-ratio score.
    n_jobs : int
        Parallel workers.
    prefer : str | None
        Joblib backend preference.
    silent : bool
        Suppress progress bar.
    copy : bool
        Return modified copy if True.

    Returns
    -------
    AnnData | None
        .uns["juzi_keep_prune"] and .uns["juzi_keep"] updated.
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
            f"top_k={top_k} exceeds number of genes ({adata.n_vars})."
        )

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1].")

    if not 0.0 <= dedup_similarity <= 1.0:
        raise ValueError("dedup_similarity must be in [0, 1].")

    if dedup_similarity < min_similarity:
        warnings.warn(
            f"dedup_similarity={dedup_similarity} is lower than "
            f"min_similarity={min_similarity}. This will aggressively "
            "deduplicate factors that are only weakly similar. "
            "Consider setting dedup_similarity >= min_similarity.",
            UserWarning,
            stacklevel=2,
        )

    if matching not in ("greedy", "hungarian"):
        raise ValueError("matching must be 'greedy' or 'hungarian'.")

    n_resolutions = len(adata.uns["juzi_k"])
    if min_k > n_resolutions:
        raise ValueError(
            f"min_k={min_k} exceeds number of k resolutions ({n_resolutions})."
        )

    # Split into per-sample blocks

    names    = np.array(adata.uns["juzi_names"])
    G        = adata.varm["juzi_G"].T
    k_list   = adata.uns["juzi_k"]
    n_comps  = int(np.sum(k_list))
    n_unique = len(np.unique(names))

    if len(names) != n_unique * n_comps:
        raise ValueError(
            f"juzi_names length ({len(names)}) does not match "
            f"n_samples ({n_unique}) x sum(k) ({n_comps}). "
            "Re-run juzi.gp.nmf_fit."
        )

    split_points = np.arange(n_comps, n_unique * n_comps, n_comps)
    per_sample_G = np.split(G, split_points)

    # Parallel pruning

    jobs = [
        delayed(_prune_sample)(
            factors=sample_G,
            k=k_list,
            top_k=top_k,
            min_similarity=min_similarity,
            dedup_similarity=dedup_similarity,
            min_k=min_k,
            matching=matching,
            deduplicate=deduplicate,
            use_combined=use_combined,
        )
        for sample_G in per_sample_G
    ]

    results = list(
        tqdm(
            Parallel(n_jobs=n_jobs, prefer=prefer, return_as="generator")(jobs),
            total=len(per_sample_G),
            desc="[juzi] Pruning",
            disable=silent,
        )
    )

    # Build global mask

    mask               = np.zeros(len(names), dtype=bool)
    cumulative_offsets = np.arange(n_unique) * n_comps

    for sample_idx, local_keep_idx in enumerate(results):
        if len(local_keep_idx) > 0:
            global_idx       = cumulative_offsets[sample_idx] + local_keep_idx
            mask[global_idx] = True

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
    sim = _similarity_matrix(top_genes_a, top_genes_b)
    best_per_a = sim.max(axis=1)
    return set(np.where(best_per_a >= min_similarity)[0].tolist())


def _match_hungarian(
    top_genes_a: list[set],
    top_genes_b: list[set],
    min_similarity: float,
) -> set[int]:
    sim = _similarity_matrix(top_genes_a, top_genes_b)
    row_idx, col_idx = linear_sum_assignment(-sim)
    return {r for r, c in zip(row_idx, col_idx) if sim[r, c] >= min_similarity}


def _prune_sample(
    factors: np.ndarray,
    k: List[int],
    top_k: int,
    min_similarity: float,
    dedup_similarity: float,
    min_k: int,
    matching: str,
    deduplicate: bool,
    use_combined: bool,
) -> np.ndarray:
    """Prune one sample — recurrence filter then optional deduplication."""

    split_points  = np.cumsum(k)[:-1]
    factors_by_k  = np.split(factors, split_points)

    top_genes_by_k = []
    for resolution in factors_by_k:
        scored = _combined_score(resolution) if use_combined else resolution
        top_genes_by_k.append([
            set(np.argsort(scored[i])[-top_k:])
            for i in range(len(resolution))
        ])

    # Recurrence filter

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

    if len(keep_local_idx) == 0:
        return np.array([], dtype=int)

    keep_local_idx = np.array(keep_local_idx, dtype=int)

    if not deduplicate or len(keep_local_idx) <= 1:
        return keep_local_idx

    # Deduplication filter
    # Uses dedup_similarity — separate and typically higher than min_similarity

    kept_top_genes = []
    for idx in keep_local_idx:
        k_idx_for = np.searchsorted(cumulative_offsets[1:], idx, side="right")
        pos_in_k  = idx - cumulative_offsets[k_idx_for]
        kept_top_genes.append(top_genes_by_k[k_idx_for][pos_in_k])

    n_kept  = len(keep_local_idx)
    sim_mat = np.zeros((n_kept, n_kept))
    for i in range(n_kept):
        for j in range(i + 1, n_kept):
            s = _jaccard(kept_top_genes[i], kept_top_genes[j])
            sim_mat[i, j] = s
            sim_mat[j, i] = s

    # Union-find grouping at dedup_similarity threshold
    parent = list(range(n_kept))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n_kept):
        for j in range(i + 1, n_kept):
            if sim_mat[i, j] >= dedup_similarity:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n_kept):
        root = find(i)
        groups.setdefault(root, []).append(i)

    dedup_keep = []
    for group_members in groups.values():
        if len(group_members) == 1:
            dedup_keep.append(group_members[0])
        else:
            mean_sims = np.array([
                sim_mat[i, group_members].mean()
                for i in group_members
            ])
            best = group_members[int(np.argmax(mean_sims))]
            dedup_keep.append(best)

    return keep_local_idx[sorted(dedup_keep)]
