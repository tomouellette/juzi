# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np
import scipy as sp

from anndata import AnnData
from typing import Tuple
from scipy.cluster.hierarchy import ClusterWarning
from sklearn.metrics import silhouette_score
from typing import List

from ._nmf import _recompute_keep


def cluster(
    adata: AnnData,
    threshold: float = 0.1,
    min_cluster: int = 2,
    reorder: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Cluster the factor similarity matrix into consensus gene programs.

    Factors are iteratively merged using average-linkage hierarchical
    clustering until the maximum inter-cluster similarity falls below
    threshold. Clusters with fewer than min_cluster unique contributing
    samples are removed and the procedure repeats until all remaining
    clusters meet the minimum sample requirement.

    Factors removed by min_cluster are tracked in juzi_keep_cluster
    (length n_total). juzi_keep is recomputed as the intersection of
    juzi_keep_prune, juzi_keep_similarity, and juzi_keep_cluster.
    Re-running cluster only resets juzi_keep_cluster — upstream masks
    are never modified.

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.gp.nmf, juzi.gp.prune, and
        juzi.gp.similarity.
    threshold : float
        Merge clusters until maximum inter-cluster similarity falls below
        this value. Higher values produce more clusters. Must be in [0, 1].
    min_cluster : int
        Minimum number of unique samples that must contribute factors to
        a cluster for it to be retained. Must be >= 1.
    reorder : bool
        If True, clusters are sorted by size (largest first) and factors
        within each cluster are sorted by internal similarity.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields populated:
            .uns["juzi_keep_cluster"]       : boolean mask length n_total
            .uns["juzi_keep"]               : intersection of all three stage masks
            .uns["juzi_cluster_similarity"] : reordered factor similarity matrix
            .uns["juzi_cluster_labels"]     : cluster label per retained factor
            .uns["juzi_cluster_names"]      : donor name per retained factor
            .uns["juzi_cluster_G"]          : centroid gene loading per cluster
            .uns["juzi_cluster_samples"]    : unique sample names per cluster
            .uns["juzi_cluster_stats"]      : silhouette, inner/outer similarity
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_similarity", "uns"),
        ("juzi_similarity_idx", "uns"),
        ("juzi_names", "uns"),
        ("juzi_G", "varm"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. "
                "Run juzi.gp.nmf, juzi.gp.similarity first."
            )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    if min_cluster < 1:
        raise ValueError("min_cluster must be >= 1.")

    # Initialise juzi_keep_cluster
    # Reset to all True (n_total length) at the start of each run.

    n_total = adata.varm["juzi_G"].shape[1]
    adata.uns["juzi_keep_cluster"] = np.ones(n_total, dtype=bool)
    _recompute_keep(adata)

    # Setup
    # juzi_similarity is (n_kept × n_kept) in the pruned+similarity space.
    # sim_idx maps local row/col indices → global factor indices.
    # juzi_keep gives the current set of globally retained factors.
    # cluster_mask is a LOCAL boolean mask over the n_kept similarity rows.

    sim_idx = adata.uns["juzi_similarity_idx"]  # (n_kept,) global indices
    full_G = adata.varm["juzi_G"].T  # (n_total × n_genes)
    full_names = np.array(adata.uns["juzi_names"])  # (n_total,)

    # juzi_keep within the similarity space — local mask for rows/cols
    # A factor passes if it is True in juzi_keep AND is in sim_idx
    global_keep = adata.uns["juzi_keep"]
    cluster_mask = global_keep[sim_idx]  # (n_kept,) local mask

    # Iterative clustering

    while True:
        S = adata.uns["juzi_similarity"][np.ix_(cluster_mask, cluster_mask)]
        names = full_names[sim_idx[cluster_mask]]
        n = S.shape[0]

        if n == 0:
            raise ValueError(
                "No factors remain after applying juzi_keep mask. "
                "Lower min_similarity or min_cluster thresholds."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ClusterWarning)
            Z = sp.cluster.hierarchy.linkage(
                1.0 - S, method="average", optimal_ordering=True
            )

        leaf_order = sp.cluster.hierarchy.leaves_list(Z)
        clusters = np.empty(n, dtype=int)
        for new_label, position in enumerate(leaf_order):
            clusters[position] = new_label

        while True:
            max_pair = _find_max_similar(S, clusters, threshold)
            if max_pair is None:
                break
            i, j = max_pair
            clusters[clusters == clusters[j]] = clusters[i]

        unique_clusters = np.unique(clusters)
        sample_counts = np.array(
            [len(np.unique(names[clusters == c])) for c in unique_clusters]
        )

        passes = sample_counts >= min_cluster

        if passes.all():
            break

        # Remove factors from under-represented clusters
        # cluster_mask is local (n_kept) — update it in local space
        keep_clusters = unique_clusters[passes]
        factor_passes = np.isin(clusters, keep_clusters)
        local_indices = np.where(cluster_mask)[0]  # local indices that were active
        cluster_mask[local_indices[~factor_passes]] = False

    # Map cluster_mask back to global space for juzi_keep_cluster

    keep_cluster_global = np.zeros(n_total, dtype=bool)
    keep_cluster_global[sim_idx[cluster_mask]] = True
    adata.uns["juzi_keep_cluster"] = keep_cluster_global
    _recompute_keep(adata)

    # Extract G and names for retained factors

    G_masked = full_G[sim_idx[cluster_mask]]  # (n_active × n_genes)

    # Reorder

    if reorder:
        reorder_idx = _reorder_clusters(S, clusters)
        S = S[np.ix_(reorder_idx, reorder_idx)]
        names = names[reorder_idx]
        clusters = clusters[reorder_idx]
        G_masked = G_masked[reorder_idx]

    # Remap cluster labels

    _, first_occurrence = np.unique(clusters, return_index=True)
    ordered_old = clusters[np.sort(first_occurrence)]

    remapped = np.empty_like(clusters)
    for new_label, old_label in enumerate(ordered_old):
        remapped[clusters == old_label] = new_label
    clusters = remapped

    # Centroid gene loadings

    unique_clusters = np.unique(clusters)
    cluster_G = np.array(
        [G_masked[clusters == c].mean(axis=0) for c in unique_clusters]
    )

    # Per-cluster sample lists

    cluster_samples = {
        int(c): np.unique(names[clusters == c]).tolist() for c in unique_clusters
    }

    # Cluster quality statistics

    inner_mask = clusters[:, None] == clusters[None, :]
    outer_mask = ~inner_mask

    inner_sim = float(S[inner_mask].mean()) if inner_mask.any() else 0.0
    outer_sim = float(S[outer_mask].mean()) if outer_mask.any() else 0.0

    sil_score = None
    nc = len(unique_clusters)
    dist_matrix = 1.0 - S
    np.fill_diagonal(dist_matrix, 0.0)

    if nc > 1 and nc < S.shape[0] - 1:
        sil_score = float(
            silhouette_score(
                dist_matrix,
                clusters,
                metric="precomputed",
            )
        )

    # Store results

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"] = clusters
    adata.uns["juzi_cluster_names"] = names.tolist()
    adata.uns["juzi_cluster_G"] = cluster_G
    adata.uns["juzi_cluster_samples"] = cluster_samples
    adata.uns["juzi_cluster_stats"] = {
        "silhouette_score": sil_score,
        "inner_similarity": inner_sim,
        "outer_similarity": outer_sim,
    }

    return adata if copy else None


def _find_max_similar(
    S: np.ndarray,
    clusters: np.ndarray,
    threshold: float,
) -> Tuple[int, int] | None:
    c_unique, first_indices = np.unique(clusters, return_index=True)
    c_counts = np.array([np.sum(clusters == c) for c in c_unique])

    X = (clusters[:, None] == c_unique[None, :]).astype(float)
    sum_sims = X.T @ S @ X
    mean_sims = sum_sims / np.outer(c_counts, c_counts)
    np.fill_diagonal(mean_sims, -np.inf)

    best_idx = int(np.argmax(mean_sims))
    max_similarity = float(mean_sims.flat[best_idx])

    if max_similarity <= threshold:
        return None

    group_i, group_j = np.unravel_index(best_idx, mean_sims.shape)
    return (int(first_indices[group_i]), int(first_indices[group_j]))


def _reorder_clusters(
    S: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    c_unique = np.unique(clusters)
    c_sizes = {c: np.sum(clusters == c) for c in c_unique}
    c_sorted = sorted(c_unique, key=lambda c: c_sizes[c], reverse=True)

    reorder_idx = []
    for c in c_sorted:
        idx = np.where(clusters == c)[0]
        if len(idx) > 1:
            Si = S[np.ix_(idx, idx)]
            D = sp.spatial.distance.squareform(1.0 - Si, checks=False)
            Z = sp.cluster.hierarchy.linkage(D, method="average")
            sub_order = sp.cluster.hierarchy.leaves_list(Z)
            reorder_idx.extend(idx[sub_order].tolist())
        else:
            reorder_idx.extend(idx.tolist())

    return np.array(reorder_idx, dtype=int)


# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np
import scipy as sp

from anndata import AnnData
from scipy.cluster.hierarchy import ClusterWarning
from typing import List


def select_threshold(
    adata: AnnData,
    thresholds: np.ndarray | None = None,
    min_cluster: int = 2,
    metric: str = "ratio",
    copy: bool = False,
) -> float:
    """Select the optimal clustering threshold by maximising cluster contrast.

    Sweeps across a grid of threshold values, fits clustering at each,
    and computes a contrast metric between inner-cluster and outer-cluster
    similarity. The optimal threshold is the one that maximises the metric
    and is stored alongside the full sweep results for plotting via
    juzi.pl.threshold.

    Degenerate partitions (single cluster or every factor its own cluster)
    are excluded from optimisation since inner or outer similarity is
    undefined.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_similarity and juzi_similarity_idx in .uns,
        produced by juzi.gp.similarity and juzi.gp.select_similarity.
    thresholds : np.ndarray | None
        Grid of threshold values to evaluate. If None, uses
        np.linspace(0.0, 1.0, 50).
    min_cluster : int
        Minimum number of unique samples per cluster. Must match the
        value used in juzi.gp.cluster. Clusters below this threshold
        are removed before computing the metric at each threshold value.
    metric : str
        Contrast metric to maximise. One of:
            "ratio"      — inner_sim / outer_sim.
            "delta"      — inner_sim - outer_sim.
            "silhouette" — mean silhouette score across all factors.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    float
        The optimal threshold value that maximises the contrast metric.
        Stores the following in .uns:
            .uns["juzi_threshold_sweep"] : dict with keys:
                "thresholds"  : array of evaluated thresholds
                "metric"      : metric values per threshold
                "metric_name" : name of the metric used
                "optimal"     : optimal threshold value
                "min_cluster" : min_cluster value used
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field in ["juzi_similarity", "juzi_similarity_idx", "juzi_names"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. " "Run juzi.gp.similarity first."
            )

    if metric not in ("ratio", "delta", "silhouette"):
        raise ValueError("metric must be 'ratio', 'delta', or 'silhouette'.")

    if min_cluster < 1:
        raise ValueError("min_cluster must be >= 1.")

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 50)

    # Setup

    sim_idx = adata.uns["juzi_similarity_idx"]
    full_names = np.array(adata.uns["juzi_names"])
    global_keep = (
        adata.uns["juzi_keep"]
        if "juzi_keep" in adata.uns
        else np.ones(len(full_names), dtype=bool)
    )

    # Base local mask within similarity space
    base_cluster_mask = global_keep[sim_idx].copy()

    S_full = adata.uns["juzi_similarity"]  # (n_kept × n_kept)
    names = full_names[sim_idx]

    # Sweep thresholds

    metric_values = np.full(len(thresholds), np.nan)

    for t_idx, threshold in enumerate(thresholds):
        # Fresh local copy of cluster_mask for each threshold
        # _cluster_at_threshold modifies cluster_mask in place as it removes
        # under-represented clusters — we need the final state to compute S
        local_mask = base_cluster_mask.copy()

        try:
            clusters = _cluster_at_threshold(
                S_full=S_full,
                names=names,
                cluster_mask=local_mask,
                threshold=threshold,
                min_cluster=min_cluster,
            )
        except ValueError:
            continue

        if clusters is None:
            continue

        # Recompute S from final local_mask after min_cluster removal
        # clusters has length local_mask.sum() — must match S dimensions
        S = S_full[np.ix_(local_mask, local_mask)]
        nc = len(np.unique(clusters))

        # Skip degenerate partitions
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
            from sklearn.metrics import silhouette_score

            dist_matrix = 1.0 - S
            np.fill_diagonal(dist_matrix, 0.0)
            try:
                metric_values[t_idx] = silhouette_score(
                    dist_matrix, clusters, metric="precomputed"
                )
            except Exception:
                continue

    # Find optimal threshold

    valid = ~np.isnan(metric_values)
    if not valid.any():
        raise ValueError(
            "No valid partitions found across the threshold sweep. "
            "Lower min_cluster or widen the threshold range."
        )

    optimal_idx = int(np.argmax(metric_values[valid]))
    optimal = float(thresholds[valid][optimal_idx])

    # Store results

    adata.uns["juzi_threshold_sweep"] = {
        "thresholds": thresholds,
        "metric": metric_values,
        "metric_name": metric,
        "optimal": optimal,
        "min_cluster": min_cluster,
    }

    return optimal


def _cluster_at_threshold(
    S_full: np.ndarray,
    names: np.ndarray,
    cluster_mask: np.ndarray,
    threshold: float,
    min_cluster: int,
) -> np.ndarray | None:
    """Run iterative clustering at a single threshold value.

    Modifies cluster_mask in place — the caller should pass a copy and
    read the final state of cluster_mask after this function returns to
    correctly compute S for the surviving factors.

    Parameters
    ----------
    S_full : np.ndarray
        Full (n_kept × n_kept) similarity matrix.
    names : np.ndarray
        Donor name per factor in S_full row order.
    cluster_mask : np.ndarray
        Local boolean mask of active factors (length n_kept). Modified
        in place as under-represented clusters are removed.
    threshold : float
        Similarity threshold for merging.
    min_cluster : int
        Minimum unique donors per cluster.

    Returns
    -------
    np.ndarray | None
        Cluster label per active factor (length cluster_mask.sum() after
        removal) or None if no valid partition found.
    """
    max_iter = 100
    for _ in range(max_iter):
        S = S_full[np.ix_(cluster_mask, cluster_mask)]
        names_active = names[cluster_mask]
        n = S.shape[0]

        if n == 0:
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ClusterWarning)
            Z = sp.cluster.hierarchy.linkage(
                1.0 - S, method="average", optimal_ordering=True
            )

        leaf_order = sp.cluster.hierarchy.leaves_list(Z)
        clusters = np.empty(n, dtype=int)
        for new_label, position in enumerate(leaf_order):
            clusters[position] = new_label

        # Merge until threshold is met
        for _ in range(n * n):
            c_unique, first_indices = np.unique(clusters, return_index=True)
            c_counts = np.array([np.sum(clusters == c) for c in c_unique])
            X = (clusters[:, None] == c_unique[None, :]).astype(float)
            sum_sims = X.T @ S @ X
            mean_sims = sum_sims / np.outer(c_counts, c_counts)
            np.fill_diagonal(mean_sims, -np.inf)
            best_idx = int(np.argmax(mean_sims))
            if float(mean_sims.flat[best_idx]) <= threshold:
                break
            gi, gj = np.unravel_index(best_idx, mean_sims.shape)
            clusters[clusters == c_unique[gj]] = c_unique[gi]

        # Check min_cluster
        unique_clusters = np.unique(clusters)
        sample_counts = np.array(
            [len(np.unique(names_active[clusters == c])) for c in unique_clusters]
        )
        passes = sample_counts >= min_cluster

        if passes.all():
            return clusters

        # Remove under-represented clusters — modifies cluster_mask in place
        keep_clusters = unique_clusters[passes]
        factor_passes = np.isin(clusters, keep_clusters)
        local_indices = np.where(cluster_mask)[0]
        cluster_mask[local_indices[~factor_passes]] = False

        if not cluster_mask.any():
            return None

    return None


def merge_clusters(
    adata: AnnData,
    clusters: List[int] | List[List[int]],
    copy: bool = False,
) -> AnnData | None:
    """Manually merge consensus programs post-hoc.

    Merges one or more sets of cluster labels into single programs.
    The merged program centroid is recomputed as the mean loading across
    all member factors. Cluster labels are remapped to contiguous 0-based
    integers after merging. juzi_keep masks are not modified since no
    factors are removed.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_labels and juzi_cluster_G in .uns,
        produced by juzi.gp.cluster.
    clusters : List[int] | List[List[int]]
        Clusters to merge. Two formats are accepted:
            - A flat list of integers merges all listed clusters into one:
              [0, 2] merges C0 and C2.
            - A list of lists performs multiple independent merges:
              [[0, 2], [1, 3]] merges C0+C2 and C1+C3 simultaneously.
        In both cases the merged cluster takes the lowest label among
        the input clusters and labels are remapped contiguously after.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields updated:
            .uns["juzi_cluster_labels"]  : updated cluster labels per factor
            .uns["juzi_cluster_G"]       : updated centroid loadings
            .uns["juzi_cluster_samples"] : updated contributing donors per cluster
            .uns["juzi_cluster_stats"]   : updated inner/outer/silhouette stats
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field in [
        "juzi_cluster_labels",
        "juzi_cluster_G",
        "juzi_cluster_names",
        "juzi_cluster_samples",
        "juzi_cluster_similarity",
    ]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. " "Run juzi.gp.cluster before merging."
            )

    labels = adata.uns["juzi_cluster_labels"].copy()
    unique_C = set(np.unique(labels).tolist())

    # Normalise input to list of lists

    if len(clusters) == 0:
        raise ValueError("clusters must be a non-empty list.")

    if isinstance(clusters[0], (int, np.integer)):
        merge_groups = [list(clusters)]
    else:
        merge_groups = [list(g) for g in clusters]

    # Validate cluster labels

    all_mentioned = []
    for group in merge_groups:
        if len(group) < 2:
            raise ValueError(
                f"Each merge group must contain at least 2 cluster labels. "
                f"Got: {group}"
            )
        for c in group:
            if int(c) not in unique_C:
                raise ValueError(
                    f"Cluster label {c} not found in juzi_cluster_labels. "
                    f"Available labels: {sorted(unique_C)}"
                )
            all_mentioned.append(int(c))

    if len(all_mentioned) != len(set(all_mentioned)):
        raise ValueError("Each cluster label may only appear in one merge group.")

    # Apply merges
    # For each merge group, assign all labels to the lowest label in the group

    for group in merge_groups:
        group = [int(c) for c in group]
        target = min(group)
        for c in group:
            if c != target:
                labels[labels == c] = target

    # Remap to contiguous 0-based integers

    _, first_occurrence = np.unique(labels, return_index=True)
    ordered_old = labels[np.sort(first_occurrence)]

    remapped = np.empty_like(labels)
    for new_label, old_label in enumerate(ordered_old):
        remapped[labels == old_label] = new_label
    labels = remapped

    # Recompute centroid gene loadings

    G_masked = adata.varm["juzi_G"].T[adata.uns["juzi_keep_cluster"]]
    names = np.array(adata.uns["juzi_cluster_names"])
    unique_clusters = np.unique(labels)

    # G_masked and labels are both in the reordered clustered space
    cluster_G = np.array([G_masked[labels == c].mean(axis=0) for c in unique_clusters])

    # Recompute per-cluster sample lists

    cluster_samples = {
        int(c): np.unique(names[labels == c]).tolist() for c in unique_clusters
    }

    # Recompute cluster quality statistics

    S = adata.uns["juzi_cluster_similarity"]
    inner_mask = labels[:, None] == labels[None, :]
    outer_mask = ~inner_mask

    inner_sim = float(S[inner_mask].mean()) if inner_mask.any() else 0.0
    outer_sim = float(S[outer_mask].mean()) if outer_mask.any() else 0.0

    sil_score = None
    nc = len(unique_clusters)
    dist_matrix = 1.0 - S
    np.fill_diagonal(dist_matrix, 0.0)

    if nc > 1 and nc < S.shape[0] - 1:
        from sklearn.metrics import silhouette_score

        try:
            sil_score = float(
                silhouette_score(
                    dist_matrix,
                    labels,
                    metric="precomputed",
                )
            )
        except Exception:
            pass

    # Store results

    adata.uns["juzi_cluster_labels"] = labels
    adata.uns["juzi_cluster_G"] = cluster_G
    adata.uns["juzi_cluster_samples"] = cluster_samples
    adata.uns["juzi_cluster_stats"] = {
        "silhouette_score": sil_score,
        "inner_similarity": inner_sim,
        "outer_similarity": outer_sim,
    }

    return adata if copy else None
