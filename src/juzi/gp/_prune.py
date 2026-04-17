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
    min_other_resolutions: int = 1,
    matching: str = "hungarian",
    deduplicate: bool = True,
    use_combined: bool = False,
    n_jobs: int = 1,
    prefer: str | None = None,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Prune non-recurrent and redundant intra-sample factors.

    Applies two sequential filters independently within each sample:

    1. Recurrence filter — a factor is kept if it shares sufficient
       top-gene overlap (Jaccard >= min_similarity) with at least one
       factor from each of min_other_resolutions other NMF resolutions.

    2. Deduplication filter (when deduplicate=True) — among factors that
       passed recurrence, any two factors with Jaccard >= dedup_similarity
       are considered redundant. The most central factor per redundant
       group is retained. dedup_similarity should typically be higher than
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
    min_other_resolutions : int
        Minimum number of other NMF resolutions a factor must match to
        survive the recurrence filter. Must be >= 0.
    matching : str
        Matching strategy across resolutions. One of:
            - "hungarian" : one-to-one optimal matching
            - "greedy"    : keep any factor with a best match above threshold
    deduplicate : bool
        If True, apply within-sample deduplication after recurrence.
    use_combined : bool
        If True, rank genes by weighted log-ratio score.
    n_jobs : int
        Parallel workers.
    prefer : str | None
        Joblib backend preference.
    silent : bool
        If True, suppress progress bar.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields updated:
            .uns["juzi_keep_prune"]   : boolean mask over global factors
            .uns["juzi_keep"]         : intersection of all stage masks
            .uns["juzi_prune"]        : prune parameter metadata
            .uns["juzi_prune_matches"]: number of matched other resolutions
                                        per global factor
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_G", "varm"),
        ("juzi_k", "uns"),
        ("juzi_names", "uns"),
        ("juzi_G_genes", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. Run juzi.gp.nmf_fit first."
            )

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    if top_k > adata.n_vars:
        raise ValueError(f"top_k={top_k} exceeds number of genes ({adata.n_vars}).")

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1].")

    if not 0.0 <= dedup_similarity <= 1.0:
        raise ValueError("dedup_similarity must be in [0, 1].")

    if min_other_resolutions < 0:
        raise ValueError("min_other_resolutions must be >= 0.")

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
    if min_other_resolutions > max(0, n_resolutions - 1):
        raise ValueError(
            f"min_other_resolutions={min_other_resolutions} exceeds the number "
            f"of other available resolutions ({max(0, n_resolutions - 1)})."
        )

    # Setup

    names = np.array(adata.uns["juzi_names"])
    G = adata.varm["juzi_G"].T
    k_list = list(adata.uns["juzi_k"])
    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)

    n_comps = int(np.sum(k_list))
    n_unique = len(np.unique(names))
    n_total = G.shape[0]

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
            gene_names=gene_names,
            top_k=top_k,
            min_similarity=min_similarity,
            dedup_similarity=dedup_similarity,
            min_other_resolutions=min_other_resolutions,
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

    # Build global outputs

    mask = np.zeros(n_total, dtype=bool)
    match_counts = np.zeros(n_total, dtype=int)
    cumulative_offsets = np.arange(n_unique) * n_comps

    for sample_idx, (local_keep_idx, local_match_counts) in enumerate(results):
        base = cumulative_offsets[sample_idx]

        if len(local_match_counts) != n_comps:
            raise ValueError(
                "Internal error: local_match_counts length does not match sum(k)."
            )

        match_counts[base : base + n_comps] = local_match_counts

        if len(local_keep_idx) > 0:
            global_idx = base + local_keep_idx
            mask[global_idx] = True

    adata.uns["juzi_keep_prune"] = mask
    adata.uns["juzi_prune_matches"] = match_counts
    adata.uns["juzi_prune"] = {
        "top_k": top_k,
        "min_similarity": min_similarity,
        "dedup_similarity": dedup_similarity,
        "min_other_resolutions": min_other_resolutions,
        "matching": matching,
        "deduplicate": deduplicate,
        "use_combined": use_combined,
    }

    _recompute_keep(adata)

    return adata if copy else None


def _jaccard(set1: set[str], set2: set[str]) -> float:
    """Compute Jaccard similarity between two gene sets."""
    union = set1 | set2
    if len(union) == 0:
        return 0.0
    return len(set1 & set2) / len(union)


def _similarity_matrix(
    top_genes_a: list[set[str]],
    top_genes_b: list[set[str]],
) -> np.ndarray:
    """Pairwise Jaccard similarity matrix between two factor gene-set lists."""
    sim = np.zeros((len(top_genes_a), len(top_genes_b)), dtype=float)
    for i, genes_i in enumerate(top_genes_a):
        for j, genes_j in enumerate(top_genes_b):
            sim[i, j] = _jaccard(genes_i, genes_j)
    return sim


def _match_greedy(
    top_genes_a: list[set[str]],
    top_genes_b: list[set[str]],
    min_similarity: float,
) -> set[int]:
    """Return indices in A whose best match in B exceeds min_similarity."""
    sim = _similarity_matrix(top_genes_a, top_genes_b)
    best_per_a = sim.max(axis=1)
    return set(np.where(best_per_a >= min_similarity)[0].tolist())


def _match_hungarian(
    top_genes_a: list[set[str]],
    top_genes_b: list[set[str]],
    min_similarity: float,
) -> set[int]:
    """Return indices in A retained by one-to-one optimal matching to B."""
    sim = _similarity_matrix(top_genes_a, top_genes_b)
    row_idx, col_idx = linear_sum_assignment(-sim)
    return {r for r, c in zip(row_idx, col_idx) if sim[r, c] >= min_similarity}


def _prune_sample(
    factors: np.ndarray,
    k: List[int],
    gene_names: np.ndarray,
    top_k: int,
    min_similarity: float,
    dedup_similarity: float,
    min_other_resolutions: int,
    matching: str,
    deduplicate: bool,
    use_combined: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Prune one sample: recurrence filter then optional deduplication.

    Parameters
    ----------
    factors : np.ndarray
        Sample-specific factor loading matrix of shape (sum(k), n_genes).
    k : List[int]
        NMF ranks used for this sample.
    gene_names : np.ndarray
        Gene names aligned to factor loading columns.
    top_k : int
        Number of top genes per factor.
    min_similarity : float
        Jaccard threshold for recurrence.
    dedup_similarity : float
        Jaccard threshold for deduplication.
    min_other_resolutions : int
        Required number of other resolutions matched.
    matching : str
        "hungarian" or "greedy".
    deduplicate : bool
        Whether to deduplicate recurrent factors.
    use_combined : bool
        Whether to rank genes by _combined_score.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        keep_local_idx, match_counts
        - keep_local_idx : retained factor indices within this sample block
        - match_counts   : recurrence counts for all local factors
    """
    split_points = np.cumsum(k)[:-1]
    factors_by_k = np.split(factors, split_points)

    # Build top-gene sets per factor using gene names, not integer indices
    top_genes_by_k: list[list[set[str]]] = []
    for resolution in factors_by_k:
        scored = _combined_score(resolution) if use_combined else resolution
        top_genes_by_k.append(
            [
                set(gene_names[np.argsort(scored[i])[-top_k:]].tolist())
                for i in range(len(resolution))
            ]
        )

    # Recurrence filter

    match_fn = _match_hungarian if matching == "hungarian" else _match_greedy
    cumulative_offsets = np.concatenate([[0], np.cumsum(k)])
    n_local = int(np.sum(k))
    match_counts = np.zeros(n_local, dtype=int)

    for k_idx, resolution_genes in enumerate(top_genes_by_k):
        counts_this_resolution = np.zeros(len(resolution_genes), dtype=int)

        for other_k_idx, other_resolution_genes in enumerate(top_genes_by_k):
            if other_k_idx == k_idx:
                continue

            matched_indices = match_fn(
                resolution_genes,
                other_resolution_genes,
                min_similarity,
            )
            for i in matched_indices:
                counts_this_resolution[i] += 1

        start = cumulative_offsets[k_idx]
        end = cumulative_offsets[k_idx + 1]
        match_counts[start:end] = counts_this_resolution

    keep_local_idx = np.where(match_counts >= min_other_resolutions)[0]

    if len(keep_local_idx) == 0:
        return np.array([], dtype=int), match_counts

    if not deduplicate or len(keep_local_idx) <= 1:
        return keep_local_idx.astype(int), match_counts

    # Deduplication filter
    # Uses dedup_similarity — separate and typically higher than min_similarity

    kept_top_genes: list[set[str]] = []
    for idx in keep_local_idx:
        k_idx_for = np.searchsorted(cumulative_offsets[1:], idx, side="right")
        pos_in_k = idx - cumulative_offsets[k_idx_for]
        kept_top_genes.append(top_genes_by_k[k_idx_for][pos_in_k])

    n_kept = len(keep_local_idx)
    sim_mat = np.zeros((n_kept, n_kept), dtype=float)
    for i in range(n_kept):
        for j in range(i + 1, n_kept):
            s = _jaccard(kept_top_genes[i], kept_top_genes[j])
            sim_mat[i, j] = s
            sim_mat[j, i] = s

    # Union-find grouping at dedup_similarity threshold

    parent = list(range(n_kept))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
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
            mean_sims = np.array(
                [sim_mat[i, group_members].mean() for i in group_members]
            )
            best = group_members[int(np.argmax(mean_sims))]
            dedup_keep.append(best)

    return keep_local_idx[np.sort(dedup_keep)].astype(int), match_counts
