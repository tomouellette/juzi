# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np
import scipy as sp

from anndata import AnnData
from typing import Tuple
from scipy.cluster.hierarchy import ClusterWarning
from sklearn.metrics import silhouette_score


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
        a cluster for it to be retained. Clusters below this threshold
        are removed and clustering is repeated on the remaining factors.
    reorder : bool
        If True, clusters are sorted by size (largest first) and factors
        within each cluster are sorted by internal similarity.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields populated:
            .uns["juzi_cluster_similarity"] : reordered factor similarity matrix
            .uns["juzi_cluster_labels"]     : cluster label per factor
            .uns["juzi_cluster_G"]          : centroid gene loading per cluster
            .uns["juzi_cluster_samples"]    : sample names contributing per cluster
            .uns["juzi_cluster_stats"]      : silhouette, inner/outer similarity
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_similarity", "uns"),
        ("juzi_names",      "uns"),
        ("juzi_G",          "varm"),
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

    # Initialise mask

    n_factors  = adata.uns["juzi_similarity"].shape[0]
    mask       = (
        np.array(adata.uns["juzi_keep"], dtype=bool)
        if "juzi_keep" in adata.uns
        else np.ones(n_factors, dtype=bool)
    )

    full_G     = adata.varm["juzi_G"].T # (n_factors × n_genes)
    full_names = np.array(adata.uns["juzi_names"])

    # Iterative clustering

    while True:
        S     = adata.uns["juzi_similarity"][np.ix_(mask, mask)]
        names = full_names[mask]
        n     = S.shape[0]

        if n == 0:
            raise ValueError(
                "No factors remain after applying juzi_keep mask. "
                "Lower min_similarity or min_cluster thresholds."
            )

        # Initial leaf ordering via hierarchical clustering
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ClusterWarning)
            Z = sp.cluster.hierarchy.linkage(
                1.0 - S, method="average", optimal_ordering=True
            )

        leaf_order = sp.cluster.hierarchy.leaves_list(Z)
        clusters   = np.empty(n, dtype=int)
        for new_label, position in enumerate(leaf_order):
            clusters[position] = new_label

        # Iterative merging until threshold is met
        while True:
            max_pair = _find_max_similar(S, clusters, threshold)
            if max_pair is None:
                break
            i, j = max_pair
            clusters[clusters == clusters[j]] = clusters[i]

        # Check min_cluster requirement per cluster
        unique_clusters = np.unique(clusters)
        sample_counts   = np.array([
            len(np.unique(names[clusters == c]))
            for c in unique_clusters
        ])

        passes = sample_counts >= min_cluster

        if passes.all():
            break

        # Remove factors belonging to under-represented clusters
        keep_clusters = unique_clusters[passes]
        factor_passes = np.isin(clusters, keep_clusters)
        mask_indices  = np.where(mask)[0]
        mask[mask_indices[~factor_passes]] = False
        adata.uns["juzi_keep"] = mask

    # Reorder
    # Extract G_masked here so reorder_idx can be applied consistently
    # to G, S, names, and clusters in one place

    G_masked = full_G[mask]

    if reorder:
        reorder_idx = _reorder_clusters(S, clusters)
        S           = S[np.ix_(reorder_idx, reorder_idx)]
        names       = names[reorder_idx]
        clusters    = clusters[reorder_idx]
        G_masked    = G_masked[reorder_idx]

    # Remap cluster labels
    # Use first-appearance order so label 0 = first cluster in array =
    # largest cluster when reorder=True. Avoid in-place collision by
    # writing to a separate array.

    _, first_occurrence = np.unique(clusters, return_index=True)
    ordered_old         = clusters[np.sort(first_occurrence)]

    remapped = np.empty_like(clusters)
    for new_label, old_label in enumerate(ordered_old):
        remapped[clusters == old_label] = new_label
    clusters = remapped

    # Centroid gene loadings
    # G_masked and clusters are now aligned — both reordered or both not

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
            dist_matrix,
            clusters,
            metric="precomputed",
        ))

    # Store results

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"]     = clusters
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
) -> Tuple[int, int] | None:
    """Find the pair of clusters with the highest mean inter-cluster similarity.

    Parameters
    ----------
    S : np.ndarray
        Symmetric similarity matrix, shape (n_factors × n_factors).
    clusters : np.ndarray
        Cluster label per factor.
    threshold : float
        Only return a pair if their mean similarity exceeds this value.

    Returns
    -------
    Tuple[int, int] | None
        Indices of one factor from each of the two most similar clusters,
        or None if no pair exceeds threshold.
    """
    c_unique, first_indices = np.unique(clusters, return_index=True)
    c_counts                = np.array([np.sum(clusters == c) for c in c_unique])

    X         = (clusters[:, None] == c_unique[None, :]).astype(float)
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
) -> np.ndarray:
    """Reorder factors so clusters are sorted by size and internally by similarity.

    Parameters
    ----------
    S : np.ndarray
        Symmetric similarity matrix, shape (n_factors × n_factors).
    clusters : np.ndarray
        Cluster label per factor.

    Returns
    -------
    np.ndarray
        Reordered factor indices.
    """
    c_unique = np.unique(clusters)
    c_sizes  = {c: np.sum(clusters == c) for c in c_unique}
    c_sorted = sorted(c_unique, key=lambda c: c_sizes[c], reverse=True)

    reorder_idx = []
    for c in c_sorted:
        idx = np.where(clusters == c)[0]
        if len(idx) > 1:
            Si        = S[np.ix_(idx, idx)]
            D         = sp.spatial.distance.squareform(1.0 - Si, checks=False)
            Z         = sp.cluster.hierarchy.linkage(D, method="average")
            sub_order = sp.cluster.hierarchy.leaves_list(Z)
            reorder_idx.extend(idx[sub_order].tolist())
        else:
            reorder_idx.extend(idx.tolist())

    return np.array(reorder_idx, dtype=int)
