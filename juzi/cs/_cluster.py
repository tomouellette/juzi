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
    silent: bool = False,
    copy: bool = False
) -> AnnData | None:
    """Cluster the factor similarity matrix by iterative merging.

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.cs.nmf.
    threshold : float
        Merge elements/clusters until the maximum similarity between clusters
        reduces to the specified similarity threshold. A value closer to 1 will
        create more clusters. A value closer to 0 will create fewer clusters.
    min_cluster : int
        Minimum number of unique labels/samples contributing. Note that the
        labels/samples were given in the .nmf key argument.
    reorder : bool
        If True, cluster labels will be sorted by cluser size (largest first).
    silent : bool
        If True, disable progress bar.
    copy : bool
        If True, a copy of the anndata is returned.
    """
    if "juzi_similarity" not in adata.uns or "juzi_names" not in adata.uns:
        raise KeyError("Please run juzi.cs.similarity before clustering.")

    if threshold < 0. or threshold > 1.:
        raise ValueError("'threshold' must be in [0, 1]")

    mask = np.ones(adata.uns["juzi_similarity"].shape[0], dtype=bool)
    if "juzi_keep" in adata.uns:
        mask = np.array(adata.uns["juzi_keep"], dtype=bool)

    while True:
        S = adata.uns["juzi_similarity"][:, mask][mask, :]
        labels = np.array(adata.uns["juzi_names"])[mask]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ClusterWarning)
            Z = sp.cluster.hierarchy.linkage(
                1.-S, method="average", optimal_ordering=True)

        leaf_order = sp.cluster.hierarchy.leaves_list(Z)
        clusters = np.empty(S.shape[0], dtype=int)
        for new_label, idx in enumerate(leaf_order):
            clusters[idx] = new_label

        while True:
            max_pair = _find_max_similar(S, clusters, threshold)
            if max_pair is None:
                break

            i, j = max_pair
            clusters[clusters == clusters[j]] = clusters[i]

        counts = []
        assignments = np.unique(clusters)
        for c in assignments:
            c_mask = np.array(clusters == c, dtype=bool)
            c_name = labels[c_mask]
            counts.append(len(np.unique(c_name)))

        counts = np.array(counts)
        c_mask = np.isin(clusters, assignments[counts >= min_cluster])
        mask[np.where(mask)[0]] = c_mask
        adata.uns["juzi_keep"] = mask

        if np.all(counts >= min_cluster):
            break

    if reorder:
        idx = _reorder_clusters(S, clusters)
        S = S[np.ix_(idx, idx)]

    clusters_ = np.array([clusters[i] for i in idx])
    for i, j in enumerate(np.unique(clusters_)):
        clusters[clusters_ == j] = i

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"] = clusters

    adata.uns["juzi_cluster_names"] = np.array(
        [labels[c_mask][i] for i in idx])

    adata.uns["juzi_cluster_G"] = np.array(
        [adata.varm["juzi_G"].T[mask][i] for i in idx])

    adata.uns["juzi_cluster_stats"] = {
        "silhouette_score": None,
        "outer_similarity": np.mean(S[clusters[:, None] != clusters[None, :]]),
        "inner_similarity": np.mean(S[clusters[:, None] == clusters[None, :]]),
    }

    nc = len(np.unique(clusters))
    if nc > 1 and nc < S.shape[0] - 1:
        adata.uns["juzi_cluster_stats"]["silhouette_score"] = silhouette_score(
            S, clusters)

    return adata if copy else None


def _find_max_similar(
    S: np.ndarray,
    clusters: np.ndarray,
    threshold: float,
) -> Tuple[int, int] | None:
    c_unique, first_indices = np.unique(clusters, return_index=True)
    c_counts = np.array([np.sum(clusters == ul) for ul in c_unique])

    X = (clusters[:, None] == c_unique[None, :]).astype(float)

    sum_sims = X.T @ S @ X
    mean_sims = sum_sims / np.outer(c_counts, c_counts)

    np.fill_diagonal(mean_sims, -np.inf)

    best_idx = np.argmax(mean_sims)
    max_similarity = mean_sims.flat[best_idx]

    if max_similarity > threshold:
        group_i, group_j = np.unravel_index(best_idx, mean_sims.shape)
        best_pair = (first_indices[group_i], first_indices[group_j])
    else:
        best_pair = None

    if best_pair is None:
        return None

    i, j = best_pair

    return i, j


def _reorder_clusters(
    S: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    c_unique = np.unique(clusters)
    c_size = {c: np.sum(np.array(clusters) == c) for c in c_unique}
    c_sort = sorted(c_unique, key=lambda x: c_size[x], reverse=True)

    reorder_idx = []
    for c in c_sort:
        idx = np.where(np.array(clusters) == c)[0]
        if len(idx) > 1:
            Si = S[np.ix_(idx, idx)]
            D = sp.spatial.distance.squareform(1 - Si, checks=False)
            Z = sp.cluster.hierarchy.linkage(D, method="average")
            sub_order = sp.cluster.hierarchy.leaves_list(Z)
            reorder_idx.extend(np.array(idx)[sub_order].tolist())
        else:
            reorder_idx.extend(idx.tolist())

    return np.array(reorder_idx)
