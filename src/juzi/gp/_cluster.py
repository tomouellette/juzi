# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np
import scipy as sp

from anndata import AnnData
from typing import Tuple, List
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from ._nmf import _recompute_keep


def programs_cluster(
    adata: AnnData,
    threshold: float = 0.1,
    min_cluster: int = 2,
    method: str = "average",
    reorder: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Cluster the factor similarity matrix into consensus gene programs.

    Factors are iteratively merged using hierarchical clustering until the
    maximum inter-cluster similarity falls below threshold. Clusters with
    fewer than min_cluster unique contributing samples are removed and the
    procedure repeats until all remaining clusters meet the minimum sample
    requirement.

    When similarity_compute was run with intra_sample=False, within-sample
    pairs are excluded from the inter-cluster mean similarity computation.
    This prevents the zero entries from artificially suppressing merges.

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.gp.nmf_fit, juzi.gp.nmf_prune, and
        juzi.gp.similarity_compute.
    threshold : float
        Merge clusters until maximum inter-cluster similarity falls below
        this value. Must be in [0, 1].
    min_cluster : int
        Minimum number of unique samples per cluster. Must be >= 1.
    method : str
        Linkage method. One of "average", "complete", "ward".
    reorder : bool
        If True, sort clusters by size and factors within each cluster
        by internal similarity.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields populated:
            .uns["juzi_keep_cluster"]       : boolean mask length n_total
            .uns["juzi_keep"]               : intersection of all three masks
            .uns["juzi_cluster_similarity"] : reordered factor similarity matrix
            .uns["juzi_cluster_labels"]     : cluster label per retained factor
            .uns["juzi_cluster_names"]      : sample name per retained factor
            .uns["juzi_cluster_G"]          : centroid gene loading per cluster
            .uns["juzi_cluster_samples"]    : unique contributing samples per cluster
            .uns["juzi_cluster_stats"]      : silhouette, inner/outer similarity
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_similarity",     "uns"),
        ("juzi_similarity_idx", "uns"),
        ("juzi_names",          "uns"),
        ("juzi_G",              "varm"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. "
                "Run juzi.gp.nmf_fit, juzi.gp.similarity_compute first."
            )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    if min_cluster < 1:
        raise ValueError("min_cluster must be >= 1.")

    if method not in ("average", "complete", "ward"):
        raise ValueError("method must be 'average', 'complete', or 'ward'.")

    if method == "ward":
        warnings.warn(
            "method='ward' requires Euclidean distances. 1 - Jaccard similarity "
            "is not strictly Euclidean — Ward linkage is an approximation in "
            "this setting. Consider method='average'.",
            UserWarning,
            stacklevel=2,
        )

    # Detect intra-sample exclusion
    # If similarity_compute was run with intra_sample=False, within-sample
    # pairs are zero in the matrix. Exclude them from merge criterion.

    exclude_intra = not adata.uns.get("juzi_similarity_intra", True)

    # Initialise juzi_keep_cluster

    n_total = adata.varm["juzi_G"].shape[1]
    adata.uns["juzi_keep_cluster"] = np.ones(n_total, dtype=bool)
    _recompute_keep(adata)

    # Setup

    sim_idx    = adata.uns["juzi_similarity_idx"]
    full_G     = adata.varm["juzi_G"].T
    full_names = np.array(adata.uns["juzi_names"])

    global_keep  = adata.uns["juzi_keep"]
    cluster_mask = global_keep[sim_idx].copy()

    # Iterative clustering

    while True:
        S     = adata.uns["juzi_similarity"][np.ix_(cluster_mask, cluster_mask)]
        names = full_names[sim_idx[cluster_mask]]
        n     = S.shape[0]

        if n == 0:
            raise ValueError(
                "No factors remain after applying juzi_keep mask. "
                "Lower min_similarity or min_cluster thresholds."
            )

        Z          = sp.cluster.hierarchy.linkage(1.0 - S, method=method)
        leaf_order = sp.cluster.hierarchy.leaves_list(Z)
        clusters   = np.empty(n, dtype=int)
        for new_label, position in enumerate(leaf_order):
            clusters[position] = new_label

        while True:
            max_pair = _find_max_similar(
                S, clusters, threshold,
                names=names if exclude_intra else None,
            )
            if max_pair is None:
                break
            i, j = max_pair
            clusters[clusters == clusters[j]] = clusters[i]

        unique_clusters = np.unique(clusters)
        sample_counts   = np.array([
            len(np.unique(names[clusters == c]))
            for c in unique_clusters
        ])
        passes = sample_counts >= min_cluster

        if passes.all():
            break

        keep_clusters = unique_clusters[passes]
        factor_passes = np.isin(clusters, keep_clusters)
        local_indices = np.where(cluster_mask)[0]
        cluster_mask[local_indices[~factor_passes]] = False

    # Map cluster_mask back to global space

    keep_cluster_global = np.zeros(n_total, dtype=bool)
    keep_cluster_global[sim_idx[cluster_mask]] = True
    adata.uns["juzi_keep_cluster"] = keep_cluster_global
    _recompute_keep(adata)

    G_masked = full_G[sim_idx[cluster_mask]]

    # Reorder

    if reorder:
        reorder_idx = _reorder_clusters(S, clusters, method=method)
        S           = S[np.ix_(reorder_idx, reorder_idx)]
        names       = names[reorder_idx]
        clusters    = clusters[reorder_idx]
        G_masked    = G_masked[reorder_idx]

    # Remap cluster labels

    _, first_occurrence = np.unique(clusters, return_index=True)
    ordered_old         = clusters[np.sort(first_occurrence)]

    remapped = np.empty_like(clusters)
    for new_label, old_label in enumerate(ordered_old):
        remapped[clusters == old_label] = new_label
    clusters = remapped

    # Centroid gene loadings

    unique_clusters = np.unique(clusters)
    cluster_G       = np.array([
        G_masked[clusters == c].mean(axis=0)
        for c in unique_clusters
    ])

    # Per-cluster sample lists

    cluster_samples = {
        int(c): np.unique(names[clusters == c]).tolist()
        for c in unique_clusters
    }

    # Cluster quality statistics

    inner_mask = clusters[:, None] == clusters[None, :]
    outer_mask = ~inner_mask

    inner_sim  = float(S[inner_mask].mean()) if inner_mask.any() else 0.0
    outer_sim  = float(S[outer_mask].mean()) if outer_mask.any() else 0.0

    sil_score   = None
    nc          = len(unique_clusters)
    dist_matrix = 1.0 - S
    np.fill_diagonal(dist_matrix, 0.0)

    if nc > 1 and nc < S.shape[0] - 1:
        sil_score = float(silhouette_score(
            dist_matrix, clusters, metric="precomputed"
        ))

    # Store results

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"]     = clusters
    adata.uns["juzi_cluster_names"]      = names.tolist()
    adata.uns["juzi_cluster_G"]          = cluster_G
    adata.uns["juzi_cluster_samples"]    = cluster_samples
    adata.uns["juzi_cluster_stats"]      = {
        "silhouette_score": sil_score,
        "inner_similarity":  inner_sim,
        "outer_similarity":  outer_sim,
    }

    return adata if copy else None


def _find_max_similar(
    S: np.ndarray,
    clusters: np.ndarray,
    threshold: float,
    names: np.ndarray | None = None,
) -> Tuple[int, int] | None:
    """Find the pair of clusters with the highest mean inter-cluster similarity.

    When names is provided, within-sample pairs are excluded from the mean
    so that zero entries from intra_sample=False do not suppress merges.
    """
    c_unique, first_indices = np.unique(clusters, return_index=True)

    X = (clusters[:, None] == c_unique[None, :]).astype(float)

    if names is not None:
        cross_mask = (names[:, None] != names[None, :]).astype(float)
        sum_sims   = X.T @ (S * cross_mask) @ X
        sum_valid  = X.T @ cross_mask @ X
        mean_sims = np.zeros_like(sum_sims)
        np.divide(sum_sims, sum_valid, out=mean_sims, where=sum_valid > 0)
    else:
        c_counts  = np.array([np.sum(clusters == c) for c in c_unique])
        sum_sims  = X.T @ S @ X
        mean_sims = sum_sims / np.outer(c_counts, c_counts)

    np.fill_diagonal(mean_sims, -np.inf)

    best_idx       = int(np.argmax(mean_sims))
    max_similarity = float(mean_sims.flat[best_idx])

    if max_similarity <= threshold:
        return None

    group_i, group_j = np.unravel_index(best_idx, mean_sims.shape)
    return (int(first_indices[group_i]), int(first_indices[group_j]))


def _reorder_clusters(
    S: np.ndarray,
    clusters: np.ndarray,
    method: str = "average",
) -> np.ndarray:
    """Reorder factors so clusters are sorted by size and internally
    by similarity using the same linkage method as the main clustering."""
    c_unique = np.unique(clusters)
    c_sizes  = {c: np.sum(clusters == c) for c in c_unique}
    c_sorted = sorted(c_unique, key=lambda c: c_sizes[c], reverse=True)

    reorder_idx = []
    for c in c_sorted:
        idx = np.where(clusters == c)[0]
        if len(idx) > 1:
            Si        = S[np.ix_(idx, idx)]
            D         = sp.spatial.distance.squareform(1.0 - Si, checks=False)
            Z         = sp.cluster.hierarchy.linkage(D, method=method)
            sub_order = sp.cluster.hierarchy.leaves_list(Z)
            reorder_idx.extend(idx[sub_order].tolist())
        else:
            reorder_idx.extend(idx.tolist())

    return np.array(reorder_idx, dtype=int)


def _cluster_at_threshold(
    S_full: np.ndarray,
    names: np.ndarray,
    cluster_mask: np.ndarray,
    threshold: float,
    min_cluster: int,
    method: str = "average",
    exclude_intra: bool = False,
) -> np.ndarray | None:
    """Run iterative clustering at a single threshold value.

    Modifies cluster_mask in place — caller should pass a copy.

    Parameters
    ----------
    S_full : np.ndarray
        Full (n_kept × n_kept) similarity matrix.
    names : np.ndarray
        Sample name per factor in S_full row order.
    cluster_mask : np.ndarray
        Local boolean mask of active factors. Modified in place.
    threshold : float
        Similarity threshold for merging.
    min_cluster : int
        Minimum unique samples per cluster.
    method : str
        Linkage method.
    exclude_intra : bool
        If True, exclude within-sample pairs from inter-cluster mean
        similarity computation. Should be True when similarity_compute
        was run with intra_sample=False.

    Returns
    -------
    np.ndarray | None
        Cluster label per active factor after removal, or None.
    """
    max_iter = 100
    for _ in range(max_iter):
        S            = S_full[np.ix_(cluster_mask, cluster_mask)]
        names_active = names[cluster_mask]
        n            = S.shape[0]

        if n == 0:
            return None

        Z          = sp.cluster.hierarchy.linkage(1.0 - S, method=method)
        leaf_order = sp.cluster.hierarchy.leaves_list(Z)
        clusters   = np.empty(n, dtype=int)
        for new_label, position in enumerate(leaf_order):
            clusters[position] = new_label

        for _ in range(n * n):
            max_pair = _find_max_similar(
                S, clusters, threshold,
                names=names_active if exclude_intra else None,
            )
            if max_pair is None:
                break
            i, j = max_pair
            clusters[clusters == clusters[j]] = clusters[i]

        unique_clusters = np.unique(clusters)
        sample_counts   = np.array([
            len(np.unique(names_active[clusters == c]))
            for c in unique_clusters
        ])
        passes = sample_counts >= min_cluster

        if passes.all():
            return clusters

        keep_clusters = unique_clusters[passes]
        factor_passes = np.isin(clusters, keep_clusters)
        local_indices = np.where(cluster_mask)[0]
        cluster_mask[local_indices[~factor_passes]] = False

        if not cluster_mask.any():
            return None

    return None


def programs_threshold(
    adata: AnnData,
    thresholds: np.ndarray | None = None,
    min_cluster: int = 2,
    method: str = "average",
    metric: str = "ratio",
    copy: bool = False,
    silent: bool = False,
) -> float:
    """Select the optimal clustering threshold by maximising cluster contrast.

    Sweeps across threshold values, fits clustering at each, and computes
    a contrast metric between inner and outer cluster similarity. All local
    maxima are stored alongside the global optimum.

    When similarity_compute was run with intra_sample=False, within-sample
    pairs are excluded from the inter-cluster mean similarity computation
    at each threshold evaluation.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_similarity and juzi_similarity_idx in .uns.
    thresholds : np.ndarray | None
        Grid of threshold values. If None, uses np.linspace(0.0, 1.0, 50).
    min_cluster : int
        Minimum unique samples per cluster.
    method : str
        Linkage method. One of "average", "complete", "ward".
    metric : str
        Contrast metric. One of "ratio", "delta", "silhouette".
    copy : bool
        If True, return a modified copy. If False, modify in place.
    silent : bool
        If True, suppress progress bar.

    Returns
    -------
    float
        Optimal threshold. Stores results in .uns["juzi_threshold_sweep"].
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field in ["juzi_similarity", "juzi_similarity_idx", "juzi_names"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. "
                "Run juzi.gp.similarity_compute first."
            )

    if metric not in ("ratio", "delta", "silhouette"):
        raise ValueError("metric must be 'ratio', 'delta', or 'silhouette'.")

    if method not in ("average", "complete", "ward"):
        raise ValueError("method must be 'average', 'complete', or 'ward'.")

    if min_cluster < 1:
        raise ValueError("min_cluster must be >= 1.")

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 50)

    # Setup

    exclude_intra     = not adata.uns.get("juzi_similarity_intra", True)
    sim_idx           = adata.uns["juzi_similarity_idx"]
    full_names        = np.array(adata.uns["juzi_names"])
    global_keep       = adata.uns["juzi_keep"] \
                        if "juzi_keep" in adata.uns \
                        else np.ones(len(full_names), dtype=bool)

    base_cluster_mask = global_keep[sim_idx].copy()
    S_full            = adata.uns["juzi_similarity"]
    names             = full_names[sim_idx]

    # Sweep

    metric_values = np.full(len(thresholds), np.nan)

    for t_idx, threshold in enumerate(
        tqdm(thresholds, desc="[juzi] Selecting threshold", disable=silent)
    ):
        local_mask = base_cluster_mask.copy()

        try:
            clusters = _cluster_at_threshold(
                S_full=S_full,
                names=names,
                cluster_mask=local_mask,
                threshold=threshold,
                min_cluster=min_cluster,
                method=method,
                exclude_intra=exclude_intra,
            )
        except ValueError:
            continue

        if clusters is None:
            continue

        S  = S_full[np.ix_(local_mask, local_mask)]
        nc = len(np.unique(clusters))

        if nc < 2 or nc >= S.shape[0]:
            continue

        inner_mask = clusters[:, None] == clusters[None, :]
        outer_mask = ~inner_mask

        inner_sim = float(S[inner_mask].mean()) if inner_mask.any() else np.nan
        outer_sim = float(S[outer_mask].mean()) if outer_mask.any() else np.nan

        if np.isnan(inner_sim) or np.isnan(outer_sim):
            continue

        if metric == "ratio":
            if outer_sim > 0:
                metric_values[t_idx] = inner_sim / outer_sim
        elif metric == "delta":
            metric_values[t_idx] = inner_sim - outer_sim
        elif metric == "silhouette":
            dist_matrix = 1.0 - S
            np.fill_diagonal(dist_matrix, 0.0)
            try:
                metric_values[t_idx] = silhouette_score(
                    dist_matrix, clusters, metric="precomputed"
                )
            except Exception:
                continue

    # Find local maxima

    valid = ~np.isnan(metric_values)
    if not valid.any():
        raise ValueError(
            "No valid partitions found across the threshold sweep. "
            "Lower min_cluster or widen the threshold range."
        )

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        finite_metric = np.where(np.isnan(metric_values), -np.inf, metric_values)
        sign_changes  = np.diff(np.sign(np.diff(finite_metric)))

    local_max_idx = (sign_changes < 0).nonzero()[0] + 1
    local_max_idx = local_max_idx[~np.isnan(metric_values[local_max_idx])]

    local_max_thresholds = thresholds[local_max_idx]
    local_max_values     = metric_values[local_max_idx]

    if len(local_max_idx) > 0:
        best    = int(np.argmax(local_max_values))
        optimal = float(local_max_thresholds[best])
    else:
        valid_positions  = np.where(valid)[0]
        best_valid_pos   = int(np.argmax(metric_values[valid_positions]))
        optimal_full_idx = valid_positions[best_valid_pos]
        optimal          = float(thresholds[optimal_full_idx])
        local_max_thresholds = np.array([optimal])
        local_max_values     = np.array([metric_values[optimal_full_idx]])

    # Store

    adata.uns["juzi_threshold_sweep"] = {
        "thresholds":          thresholds,
        "metric":              metric_values,
        "metric_name":         metric,
        "optimal":             optimal,
        "local_maxima":        local_max_thresholds,
        "local_maxima_values": local_max_values,
        "min_cluster":         min_cluster,
        "method":              method,
    }

    return optimal


def programs_merge(
    adata: AnnData,
    clusters: List[int] | List[List[int]],
    copy: bool = False,
) -> AnnData | None:
    """Manually merge consensus programs post-hoc.

    Merges one or more sets of cluster labels into single programs.
    The merged program centroid is recomputed as the mean loading across
    all member factors. The similarity matrix and cluster labels are
    reordered so merged factors are contiguous and clusters are sorted
    by size. juzi_keep masks are not modified since no factors are removed.

    Parameters
    ----------
    adata : AnnData
        AnnData object produced by juzi.gp.programs_cluster.
    clusters : List[int] | List[List[int]]
        Clusters to merge. Two formats:
            - Flat list: [0, 2] merges C0 and C2.
            - List of lists: [[0, 2], [1, 3]] merges C0+C2 and C1+C3.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with cluster fields updated and juzi_jackknife dropped.
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field in [
        "juzi_cluster_labels",
        "juzi_cluster_G",
        "juzi_cluster_names",
        "juzi_cluster_samples",
        "juzi_cluster_similarity",
        "juzi_keep_cluster",
    ]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. "
                "Run juzi.gp.programs_cluster before merging."
            )

    labels   = adata.uns["juzi_cluster_labels"].copy()
    unique_C = set(np.unique(labels).tolist())

    if len(clusters) == 0:
        raise ValueError("clusters must be a non-empty list.")

    if isinstance(clusters[0], (int, np.integer)):
        merge_groups = [list(clusters)]
    else:
        merge_groups = [list(g) for g in clusters]

    all_mentioned = []
    for group in merge_groups:
        if len(group) < 2:
            raise ValueError(
                f"Each merge group must contain at least 2 labels. Got: {group}"
            )
        for c in group:
            if int(c) not in unique_C:
                raise ValueError(
                    f"Cluster label {c} not found. "
                    f"Available: {sorted(unique_C)}"
                )
            all_mentioned.append(int(c))

    if len(all_mentioned) != len(set(all_mentioned)):
        raise ValueError("Each cluster label may only appear in one merge group.")

    # Apply merges

    for group in merge_groups:
        group  = [int(c) for c in group]
        target = min(group)
        for c in group:
            if c != target:
                labels[labels == c] = target

    # Reorder

    S        = adata.uns["juzi_cluster_similarity"].copy()
    names    = np.array(adata.uns["juzi_cluster_names"])
    G_masked = adata.varm["juzi_G"].T[adata.uns["juzi_keep_cluster"]]
    method   = adata.uns.get("juzi_threshold_sweep", {}).get("method", "average")

    reorder_idx = _reorder_clusters(S, labels, method=method)
    S           = S[np.ix_(reorder_idx, reorder_idx)]
    names       = names[reorder_idx]
    labels      = labels[reorder_idx]
    G_masked    = G_masked[reorder_idx]

    # Remap labels

    _, first_occurrence = np.unique(labels, return_index=True)
    ordered_old         = labels[np.sort(first_occurrence)]

    remapped = np.empty_like(labels)
    for new_label, old_label in enumerate(ordered_old):
        remapped[labels == old_label] = new_label
    labels = remapped

    # Recompute centroids and stats

    unique_clusters = np.unique(labels)
    cluster_G       = np.array([
        G_masked[labels == c].mean(axis=0)
        for c in unique_clusters
    ])
    cluster_samples = {
        int(c): np.unique(names[labels == c]).tolist()
        for c in unique_clusters
    }

    inner_mask = labels[:, None] == labels[None, :]
    outer_mask = ~inner_mask
    inner_sim  = float(S[inner_mask].mean()) if inner_mask.any() else 0.0
    outer_sim  = float(S[outer_mask].mean()) if outer_mask.any() else 0.0

    sil_score_val = None
    nc            = len(unique_clusters)
    dist_matrix   = 1.0 - S
    np.fill_diagonal(dist_matrix, 0.0)

    if nc > 1 and nc < S.shape[0] - 1:
        try:
            sil_score_val = float(silhouette_score(
                dist_matrix, labels, metric="precomputed"
            ))
        except Exception:
            pass

    # Drop jackknife

    if "juzi_jackknife" in adata.uns:
        del adata.uns["juzi_jackknife"]

    # Store

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"]     = labels
    adata.uns["juzi_cluster_names"]      = names.tolist()
    adata.uns["juzi_cluster_G"]          = cluster_G
    adata.uns["juzi_cluster_samples"]    = cluster_samples
    adata.uns["juzi_cluster_stats"]      = {
        "silhouette_score": sil_score_val,
        "inner_similarity":  inner_sim,
        "outer_similarity":  outer_sim,
    }

    return adata if copy else None


def programs_remove(
    adata: AnnData,
    clusters: List[int],
    copy: bool = False,
) -> AnnData | None:
    """Remove consensus programs post-hoc.

    Removes one or more programs from the clustering result and updates
    all downstream fields. Unlike programs_merge, programs_remove modifies
    juzi_keep_cluster — factors belonging to removed programs are set to
    False, and juzi_keep is recomputed. This reflects the fact that factors
    are being permanently excluded from the consensus, not reassigned.

    Remaining cluster labels are remapped to contiguous 0-based integers
    after removal. The similarity matrix and labels are reordered by size.

    Parameters
    ----------
    adata : AnnData
        AnnData object produced by juzi.gp.programs_cluster.
    clusters : List[int]
        List of cluster label integers to remove. Each must be a valid
        label in juzi_cluster_labels.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields updated:
            .uns["juzi_keep_cluster"]       : updated boolean mask
            .uns["juzi_keep"]               : recomputed intersection
            .uns["juzi_cluster_similarity"] : updated similarity matrix
            .uns["juzi_cluster_labels"]     : updated and reordered labels
            .uns["juzi_cluster_names"]      : updated donor names per factor
            .uns["juzi_cluster_G"]          : updated centroid loadings
            .uns["juzi_cluster_samples"]    : updated contributing donors
            .uns["juzi_cluster_stats"]      : updated inner/outer/silhouette
            .uns["juzi_jackknife"]          : dropped if present — must be
                                              rerun after removal
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_cluster_labels",     "uns"),
        ("juzi_cluster_G",          "uns"),
        ("juzi_cluster_names",      "uns"),
        ("juzi_cluster_samples",    "uns"),
        ("juzi_cluster_similarity", "uns"),
        ("juzi_keep_cluster",       "uns"),
        ("juzi_keep",               "uns"),
        ("juzi_similarity_idx",     "uns"),
        ("juzi_G",                  "varm"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. "
                "Run juzi.gp.programs_cluster before removing programs."
            )

    if len(clusters) == 0:
        raise ValueError("clusters must be a non-empty list.")

    labels   = adata.uns["juzi_cluster_labels"].copy()
    unique_C = set(np.unique(labels).tolist())

    # Validate cluster labels
    for c in clusters:
        if int(c) not in unique_C:
            raise ValueError(
                f"Cluster label {c} not found in juzi_cluster_labels. "
                f"Available labels: {sorted(unique_C)}"
            )

    if len(set(int(c) for c in clusters)) != len(clusters):
        raise ValueError("clusters contains duplicate labels.")

    if len(set(int(c) for c in clusters)) == len(unique_C):
        raise ValueError(
            "Cannot remove all programs. At least one program must remain."
        )

    remove_set = {int(c) for c in clusters}

    # Map cluster-space positions to global factor indices
    # juzi_cluster_labels[i] and juzi_cluster_names[i] are aligned —
    # position i in cluster space corresponds to a specific global factor.
    # sim_idx gives global indices of factors that entered similarity.
    # juzi_keep_cluster within sim_idx identifies which entered clustering.
    # sim_cluster_global[i] is therefore the global index of cluster position i.

    sim_idx            = adata.uns["juzi_similarity_idx"]
    keep_cluster_global = adata.uns["juzi_keep_cluster"].copy()
    in_cluster         = keep_cluster_global[sim_idx] # (n_kept,) bool
    sim_cluster_global = sim_idx[in_cluster] # global idx per cluster position

    # Identify cluster-space positions belonging to removed programs
    remove_positions = np.where(np.isin(labels, list(remove_set)))[0]

    # Set corresponding global factors to False
    for pos in remove_positions:
        keep_cluster_global[sim_cluster_global[pos]] = False

    adata.uns["juzi_keep_cluster"] = keep_cluster_global

    # Recompute juzi_keep

    adata.uns["juzi_keep"] = (
        adata.uns["juzi_keep_prune"]      &
        adata.uns["juzi_keep_similarity"] &
        adata.uns["juzi_keep_cluster"]
    )

    # Subset cluster fields to remaining factors

    factor_keep = ~np.isin(labels, list(remove_set))
    names       = np.array(adata.uns["juzi_cluster_names"])[factor_keep]
    labels      = labels[factor_keep]
    S           = adata.uns["juzi_cluster_similarity"][np.ix_(factor_keep, factor_keep)]

    # Reconstruct G_masked from the updated juzi_keep_cluster
    G_masked    = adata.varm["juzi_G"].T[adata.uns["juzi_keep_cluster"]]

    # Reorder by size

    method      = adata.uns.get("juzi_threshold_sweep", {}).get("method", "average")
    reorder_idx = _reorder_clusters(S, labels, method=method)

    S        = S[np.ix_(reorder_idx, reorder_idx)]
    names    = names[reorder_idx]
    labels   = labels[reorder_idx]
    G_masked = G_masked[reorder_idx]

    # Remap labels to contiguous 0-based integers

    _, first_occurrence = np.unique(labels, return_index=True)
    ordered_old         = labels[np.sort(first_occurrence)]

    remapped = np.empty_like(labels)
    for new_label, old_label in enumerate(ordered_old):
        remapped[labels == old_label] = new_label
    labels = remapped

    # Recompute centroid gene loadings

    unique_clusters = np.unique(labels)
    cluster_G       = np.array([
        G_masked[labels == c].mean(axis=0)
        for c in unique_clusters
    ])

    # Recompute per-cluster sample lists

    cluster_samples = {
        int(c): np.unique(names[labels == c]).tolist()
        for c in unique_clusters
    }

    # Recompute cluster quality statistics

    inner_mask = labels[:, None] == labels[None, :]
    outer_mask = ~inner_mask

    inner_sim  = float(S[inner_mask].mean()) if inner_mask.any() else 0.0
    outer_sim  = float(S[outer_mask].mean()) if outer_mask.any() else 0.0

    sil_score_val = None
    nc            = len(unique_clusters)
    dist_matrix   = 1.0 - S
    np.fill_diagonal(dist_matrix, 0.0)

    if nc > 1 and nc < S.shape[0] - 1:
        try:
            sil_score_val = float(silhouette_score(
                dist_matrix, labels, metric="precomputed"
            ))
        except Exception:
            pass

    # Drop jackknife must be rerun after removal

    if "juzi_jackknife" in adata.uns:
        del adata.uns["juzi_jackknife"]

    # Store results

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"]     = labels
    adata.uns["juzi_cluster_names"]      = names.tolist()
    adata.uns["juzi_cluster_G"]          = cluster_G
    adata.uns["juzi_cluster_samples"]    = cluster_samples
    adata.uns["juzi_cluster_stats"]      = {
        "silhouette_score": sil_score_val,
        "inner_similarity":  inner_sim,
        "outer_similarity":  outer_sim,
    }

    return adata if copy else None
