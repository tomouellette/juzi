# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np
import scipy as sp

from anndata import AnnData
from typing import Dict, List, Set, Tuple
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from ._nmf import _recompute_keep, _combined_score


def programs_cluster(
    adata: AnnData,
    strategy: str = "centroid",
    min_cluster: int = 2,
    reorder: bool = True,
    n_top_genes: int = 50,
    threshold: float = 0.1,
    method: str = "average",
    top_k: int = 50,
    min_overlap: int = 10,
    min_founder_overlaps: int = 6,
    copy: bool = False,
) -> AnnData | None:
    """Cluster NMF factors into consensus programs.

    Two clustering strategies are supported:

    - strategy="centroid":
        Iterative hierarchical clustering on the similarity matrix with
        cluster merging controlled by `threshold`.

    - strategy="progressive":
        Progressive meta-program (MP) construction where programs are
        grown from a founder factor by iteratively adding the best
        overlapping factor and updating the MP gene set via frequency +
        NMF-score ranking.

    Both strategies populate a common set of cluster fields so downstream
    steps can treat the result uniformly.

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.gp.nmf_fit and juzi.gp.similarity.
    strategy : str
        One of "centroid" or "progressive".
    min_cluster : int
        Minimum number of unique samples per cluster/program.
    reorder : bool
        If True, sort clusters by size (largest first) and factors within
        clusters by internal similarity. The ordering is stored in
        .uns["juzi_cluster_order"] for downstream heatmap alignment.
    n_top_genes : int
        Centroid mode only. Number of canonical genes per cluster stored
        in .uns["juzi_cluster_genes"].
    threshold : float
        Centroid mode only. Merge clusters until maximum inter-cluster
        similarity falls below this value.
    method : str
        Centroid mode only. Linkage method: "average", "complete", "ward".
    top_k : int
        Progressive mode only. Number of genes per MP.
    min_overlap : int
        Progressive mode only. Minimum shared genes between a candidate
        factor and the current MP to be eligible for addition.
        Also the minimum pairwise overlap for counting qualifying partners
        during founder selection.
    min_founder_overlaps : int
        Progressive mode only. Minimum number of other unassigned factors
        with pairwise overlap >= min_overlap required for a factor to be
        chosen as a founder. A founder must have strictly more than this
        value qualifying partners, so this parameter sets the lower bound.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields populated:
            .uns["juzi_keep_cluster"]       : boolean mask length n_total
            .uns["juzi_keep"]               : intersection of all stage masks
            .uns["juzi_cluster_similarity"] : reordered factor similarity matrix
            .uns["juzi_cluster_labels"]     : cluster label per retained factor
            .uns["juzi_cluster_names"]      : sample name per retained factor
            .uns["juzi_cluster_order"]      : global factor indices in their final
                                              display order; use to align heatmap axes
            .uns["juzi_cluster_G"]          : centroid loading per cluster
            .uns["juzi_cluster_genes"]      : canonical gene set per cluster
            .uns["juzi_cluster_mp_genes"]   : MP gene sets (progressive mode only)
            .uns["juzi_cluster_samples"]    : unique contributing samples per cluster
            .uns["juzi_cluster_stats"]      : silhouette, inner/outer similarity
            .uns["juzi_cluster_meta"]       : clustering strategy + parameters
    """
    adata = adata.copy() if copy else adata

    if strategy == "centroid":
        programs_centroid(
            adata=adata,
            threshold=threshold,
            min_cluster=min_cluster,
            method=method,
            reorder=reorder,
            n_top_genes=n_top_genes,
            copy=False,
        )
    elif strategy == "progressive":
        programs_progressive(
            adata=adata,
            top_k=top_k,
            min_overlap=min_overlap,
            min_founder_overlaps=min_founder_overlaps,
            min_cluster=min_cluster,
            reorder=reorder,
            copy=False,
        )
    else:
        raise ValueError("strategy must be 'centroid' or 'progressive'.")

    return adata if copy else None


def programs_centroid(
    adata: AnnData,
    threshold: float = 0.1,
    min_cluster: int = 2,
    method: str = "average",
    reorder: bool = True,
    n_top_genes: int = 50,
    copy: bool = False,
) -> AnnData | None:
    """Cluster factors into consensus programs via hierarchical centroid merging."""
    adata = adata.copy() if copy else adata

    _validate_cluster_inputs(adata)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if min_cluster < 1:
        raise ValueError("min_cluster must be >= 1.")
    if method not in ("average", "complete", "ward"):
        raise ValueError("method must be 'average', 'complete', or 'ward'.")
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be >= 1.")
    if method == "ward":
        warnings.warn(
            "method='ward' requires Euclidean distances. 1 - similarity "
            "is not strictly Euclidean here, so Ward linkage is an approximation. "
            "Consider method='average'.",
            UserWarning,
            stacklevel=2,
        )

    exclude_intra = not adata.uns.get("juzi_similarity_intra", True)

    n_total = adata.varm["juzi_G"].shape[1]
    adata.uns["juzi_keep_cluster"] = np.ones(n_total, dtype=bool)
    _recompute_keep(adata)

    sim_idx = adata.uns["juzi_similarity_idx"]
    full_G = adata.varm["juzi_G"].T
    full_names = np.array(adata.uns["juzi_names"], dtype=object)
    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)

    global_keep = adata.uns["juzi_keep"]
    cluster_mask = global_keep[sim_idx].copy()

    clusters, cluster_mask = _cluster_centroid(
        S_full=adata.uns["juzi_similarity"],
        full_names=full_names,
        sim_idx=sim_idx,
        cluster_mask=cluster_mask,
        threshold=threshold,
        min_cluster=min_cluster,
        method=method,
        exclude_intra=exclude_intra,
    )

    _finalise_clusters(
        adata=adata,
        clusters=clusters,
        cluster_mask=cluster_mask,
        sim_idx=sim_idx,
        full_G=full_G,
        full_names=full_names,
        gene_names=gene_names,
        method=method,
        reorder=reorder,
        n_top_genes=n_top_genes,
        mp_genes=None,
        strategy="centroid",
        meta_extra={
            "threshold": threshold,
            "min_cluster": min_cluster,
            "method": method,
            "reorder": reorder,
            "n_top_genes": n_top_genes,
        },
    )

    return adata if copy else None


def programs_progressive(
    adata: AnnData,
    top_k: int = 50,
    min_overlap: int = 10,
    min_founder_overlaps: int = 6,
    min_cluster: int = 2,
    reorder: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Cluster factors into consensus programs via progressive MP construction.

    Outer loop — founder selection:
        For each iteration, count how many unassigned factors have pairwise
        gene overlap >= min_overlap with each candidate. Pick the candidate
        with the highest count (the "founder"). Stop when that count is
        not strictly greater than min_founder_overlaps

    Seed initialisation:
        The founder's own top-k gene set becomes the initial Genes_MP.
        The best-overlapping unassigned partner (>= min_overlap) is added
        immediately as the first step of the inner loop.

    Inner loop — cluster growth:
        Repeatedly add the unassigned factor with the highest overlap with
        the current Genes_MP (must be >= min_overlap). After each addition,
        recompute Genes_MP from the full running history of all member gene
        vectors using frequency ranking with ties broken by max NMF score.

    Post-cluster filter:
        A completed cluster is kept only if it has >= min_cluster unique
        contributing samples. Factors from rejected clusters return to the
        unassigned pool.
    """
    adata = adata.copy() if copy else adata

    _validate_cluster_inputs(adata)

    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    if min_overlap < 1:
        raise ValueError("min_overlap must be >= 1.")
    if min_overlap > top_k:
        raise ValueError("min_overlap must be <= top_k.")
    if min_founder_overlaps < 1:
        raise ValueError("min_founder_overlaps must be >= 1.")
    if min_cluster < 1:
        raise ValueError("min_cluster must be >= 1.")

    n_total = adata.varm["juzi_G"].shape[1]
    adata.uns["juzi_keep_cluster"] = np.ones(n_total, dtype=bool)
    _recompute_keep(adata)

    sim_idx = adata.uns["juzi_similarity_idx"]
    full_G = adata.varm["juzi_G"].T
    full_names = np.array(adata.uns["juzi_names"], dtype=object)
    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)

    global_keep = adata.uns["juzi_keep"]
    cluster_mask = global_keep[sim_idx].copy()

    clusters, cluster_mask, mp_genes = _cluster_progressive(
        full_G=full_G,
        full_names=full_names,
        sim_idx=sim_idx,
        gene_names=gene_names,
        cluster_mask=cluster_mask,
        top_k=top_k,
        min_overlap=min_overlap,
        min_founder_overlaps=min_founder_overlaps,
        min_cluster=min_cluster,
    )

    _finalise_clusters(
        adata=adata,
        clusters=clusters,
        cluster_mask=cluster_mask,
        sim_idx=sim_idx,
        full_G=full_G,
        full_names=full_names,
        gene_names=gene_names,
        method="average",
        reorder=reorder,
        n_top_genes=top_k,
        mp_genes=mp_genes,
        strategy="progressive",
        meta_extra={
            "top_k": top_k,
            "min_overlap": min_overlap,
            "min_founder_overlaps": min_founder_overlaps,
            "min_cluster": min_cluster,
            "reorder": reorder,
        },
    )

    return adata if copy else None


def programs_threshold(
    adata: AnnData,
    thresholds: np.ndarray | None = None,
    min_cluster: int = 2,
    method: str = "average",
    metric: str = "ratio",
    copy: bool = False,
    silent: bool = False,
) -> float:
    """Select the optimal clustering threshold for centroid mode.

    Sweeps over a range of similarity thresholds and scores each resulting
    partition with the chosen metric. Returns the optimal threshold and
    stores the full sweep in .uns["juzi_threshold_sweep"].

    This function is only meaningful for centroid clustering.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_similarity in .uns.
    thresholds : np.ndarray | None
        Threshold values to sweep. Defaults to np.linspace(0, 1, 50).
    min_cluster : int
        Minimum unique samples per cluster.
    method : str
        Linkage method for hierarchical clustering.
    metric : str
        Partition quality metric. One of "ratio" (inner/outer similarity),
        "delta" (inner - outer), or "silhouette".
    copy : bool
        If True, operate on a copy (only the returned float is meaningful;
        the copy itself is discarded).
    silent : bool
        If True, suppress the progress bar.

    Returns
    -------
    float
        Optimal threshold value.
    """
    adata = adata.copy() if copy else adata

    for field in ["juzi_similarity", "juzi_similarity_idx", "juzi_names"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. Run juzi.gp.similarity first."
            )

    if metric not in ("ratio", "delta", "silhouette"):
        raise ValueError("metric must be 'ratio', 'delta', or 'silhouette'.")
    if method not in ("average", "complete", "ward"):
        raise ValueError("method must be 'average', 'complete', or 'ward'.")
    if min_cluster < 1:
        raise ValueError("min_cluster must be >= 1.")

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 50)

    exclude_intra = not adata.uns.get("juzi_similarity_intra", True)
    sim_idx = adata.uns["juzi_similarity_idx"]
    full_names = np.array(adata.uns["juzi_names"], dtype=object)
    global_keep = adata.uns.get("juzi_keep", np.ones(len(full_names), dtype=bool))

    base_cluster_mask = global_keep[sim_idx].copy()
    S_full = adata.uns["juzi_similarity"]
    names = full_names[sim_idx]

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

        S = S_full[np.ix_(local_mask, local_mask)]
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

    valid = ~np.isnan(metric_values)
    if not valid.any():
        raise ValueError(
            "No valid partitions found across the threshold sweep. "
            "Lower min_cluster or widen the threshold range."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        finite_metric = np.where(np.isnan(metric_values), -np.inf, metric_values)
        sign_changes = np.diff(np.sign(np.diff(finite_metric)))

    local_max_idx = (sign_changes < 0).nonzero()[0] + 1
    local_max_idx = local_max_idx[~np.isnan(metric_values[local_max_idx])]

    local_max_thresholds = thresholds[local_max_idx]
    local_max_values = metric_values[local_max_idx]

    if len(local_max_idx) > 0:
        best = int(np.argmax(local_max_values))
        optimal = float(local_max_thresholds[best])
    else:
        valid_positions = np.where(valid)[0]
        best_valid_pos = int(np.argmax(metric_values[valid_positions]))
        optimal_full_idx = valid_positions[best_valid_pos]
        optimal = float(thresholds[optimal_full_idx])
        local_max_thresholds = np.array([optimal])
        local_max_values = np.array([metric_values[optimal_full_idx]])

    adata.uns["juzi_threshold_sweep"] = {
        "thresholds": thresholds,
        "metric": metric_values,
        "metric_name": metric,
        "optimal": optimal,
        "local_maxima": local_max_thresholds,
        "local_maxima_values": local_max_values,
        "min_cluster": min_cluster,
        "method": method,
        "strategy": "centroid",
    }

    return optimal


def programs_merge(
    adata: AnnData,
    clusters: List[int] | List[List[int]],
    copy: bool = False,
) -> AnnData | None:
    """Manually merge consensus programs post-hoc.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_labels in .uns.
    clusters : List[int] | List[List[int]]
        Either a flat list of cluster labels to merge into one, or a list
        of groups where each group is a list of labels to merge together.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
    """
    adata = adata.copy() if copy else adata

    _validate_posthoc_inputs(adata)

    labels = adata.uns["juzi_cluster_labels"].copy()
    unique_C = set(np.unique(labels).tolist())

    if len(clusters) == 0:
        raise ValueError("clusters must be a non-empty list.")

    if isinstance(clusters[0], (int, np.integer)):
        merge_groups = [list(clusters)]
    else:
        merge_groups = [list(g) for g in clusters]

    all_mentioned: List[int] = []
    for group in merge_groups:
        if len(group) < 2:
            raise ValueError(
                f"Each merge group must contain at least 2 labels. Got: {group}"
            )
        for c in group:
            if int(c) not in unique_C:
                raise ValueError(
                    f"Cluster label {c} not found. Available: {sorted(unique_C)}"
                )
            all_mentioned.append(int(c))

    if len(all_mentioned) != len(set(all_mentioned)):
        raise ValueError("Each cluster label may only appear in one merge group.")

    for group in merge_groups:
        group = [int(c) for c in group]
        target = min(group)
        for c in group:
            if c != target:
                labels[labels == c] = target

    _rebuild_cluster_state(adata, labels, source="merge")

    return adata if copy else None


def programs_remove(
    adata: AnnData,
    clusters: List[int],
    copy: bool = False,
) -> AnnData | None:
    """Remove consensus programs post-hoc.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_labels in .uns.
    clusters : List[int]
        Cluster labels to remove.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
    """
    adata = adata.copy() if copy else adata

    _validate_posthoc_inputs(adata)

    if len(clusters) == 0:
        raise ValueError("clusters must be a non-empty list.")

    labels = adata.uns["juzi_cluster_labels"].copy()
    unique_C = set(np.unique(labels).tolist())

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

    # Update the global keep_cluster mask before rebuilding cluster state
    sim_idx = adata.uns["juzi_similarity_idx"]
    keep_cluster_global = adata.uns["juzi_keep_cluster"].copy()
    in_cluster = keep_cluster_global[sim_idx]
    sim_cluster_global = sim_idx[in_cluster]

    remove_positions = np.where(np.isin(labels, list(remove_set)))[0]
    for pos in remove_positions:
        keep_cluster_global[sim_cluster_global[pos]] = False

    adata.uns["juzi_keep_cluster"] = keep_cluster_global
    _recompute_keep(adata)

    # Subset cluster-space arrays to surviving factors
    factor_keep = ~np.isin(labels, list(remove_set))
    labels = labels[factor_keep]

    _rebuild_cluster_state(adata, labels, source="remove", factor_keep=factor_keep)

    return adata if copy else None


def _finalise_clusters(
    adata: AnnData,
    clusters: np.ndarray,
    cluster_mask: np.ndarray,
    sim_idx: np.ndarray,
    full_G: np.ndarray,
    full_names: np.ndarray,
    gene_names: np.ndarray,
    method: str,
    reorder: bool,
    n_top_genes: int,
    mp_genes: Dict[int, Set[str]] | None,
    strategy: str,
    meta_extra: dict,
) -> None:
    """Shared post-clustering bookkeeping for both centroid and progressive modes.

    Writes all cluster fields to adata.uns (see programs_cluster docstring).

    juzi_cluster_order stores the global factor indices (into varm["juzi_G"].T)
    in the final display order. Downstream heatmap code should index both axes
    of the similarity matrix and the factor dimension with this array to get a
    correctly ordered block-diagonal plot.
    """
    n_total = adata.varm["juzi_G"].shape[1]

    keep_cluster_global = np.zeros(n_total, dtype=bool)
    keep_cluster_global[sim_idx[cluster_mask]] = True
    adata.uns["juzi_keep_cluster"] = keep_cluster_global
    _recompute_keep(adata)

    S = adata.uns["juzi_similarity"][np.ix_(cluster_mask, cluster_mask)]
    names = full_names[sim_idx[cluster_mask]]
    G_masked = full_G[sim_idx[cluster_mask]]

    # Track which global factor index sits at each display position
    global_order = sim_idx[cluster_mask].copy()

    if reorder:
        reorder_idx = _reorder_clusters(S, clusters, method=method)
        S = S[np.ix_(reorder_idx, reorder_idx)]
        names = names[reorder_idx]
        clusters = clusters[reorder_idx]
        G_masked = G_masked[reorder_idx]
        global_order = global_order[reorder_idx]

    # Remap cluster labels to contiguous 0-based integers in display order
    _, first_occurrence = np.unique(clusters, return_index=True)
    ordered_old = clusters[np.sort(first_occurrence)]

    remapped = np.empty_like(clusters)
    for new_label, old_label in enumerate(ordered_old):
        remapped[clusters == old_label] = new_label
    clusters = remapped

    unique_clusters = np.unique(clusters)
    cluster_G = np.array(
        [G_masked[clusters == c].mean(axis=0) for c in unique_clusters]
    )

    # Canonical gene sets
    if mp_genes is not None:
        mp_genes_remapped: Dict[int, List[str]] = {
            int(new_label): sorted(mp_genes[int(old_label)])
            for new_label, old_label in enumerate(ordered_old)
            if int(old_label) in mp_genes
        }
        cluster_genes: Dict[int, List[str]] = {
            int(c): mp_genes_remapped[int(c)]
            for c in unique_clusters
            if int(c) in mp_genes_remapped
        }
        adata.uns["juzi_cluster_mp_genes"] = cluster_genes
    else:
        G_rank = _combined_score(cluster_G)
        cluster_genes = {
            int(c): gene_names[np.argsort(G_rank[i])[-n_top_genes:][::-1]].tolist()
            for i, c in enumerate(unique_clusters)
        }
        if "juzi_cluster_mp_genes" in adata.uns:
            del adata.uns["juzi_cluster_mp_genes"]

    cluster_samples = {
        int(c): np.unique(names[clusters == c]).tolist() for c in unique_clusters
    }

    inner_mask = clusters[:, None] == clusters[None, :]
    outer_mask = ~inner_mask
    inner_sim = float(S[inner_mask].mean()) if inner_mask.any() else 0.0
    outer_sim = float(S[outer_mask].mean()) if outer_mask.any() else 0.0

    sil_score_val = None
    nc = len(unique_clusters)
    dist_matrix = 1.0 - S
    np.fill_diagonal(dist_matrix, 0.0)

    if nc > 1 and nc < S.shape[0] - 1:
        try:
            sil_score_val = float(
                silhouette_score(dist_matrix, clusters, metric="precomputed")
            )
        except Exception:
            pass

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"] = clusters
    adata.uns["juzi_cluster_names"] = names.tolist()
    adata.uns["juzi_cluster_order"] = global_order
    adata.uns["juzi_cluster_G"] = cluster_G
    adata.uns["juzi_cluster_genes"] = cluster_genes
    adata.uns["juzi_cluster_samples"] = cluster_samples
    adata.uns["juzi_cluster_stats"] = {
        "silhouette_score": sil_score_val,
        "inner_similarity": inner_sim,
        "outer_similarity": outer_sim,
    }
    adata.uns["juzi_cluster_meta"] = {
        "strategy": strategy,
        **meta_extra,
    }


def _rebuild_cluster_state(
    adata: AnnData,
    labels: np.ndarray,
    source: str,
    factor_keep: np.ndarray | None = None,
) -> None:
    """Shared bookkeeping for programs_merge and programs_remove.

    Takes the (possibly subset) labels array in current cluster-space order,
    reorders, remaps, recomputes centroids / gene sets, and writes all cluster
    fields. Centralising this logic means merge and remove always produce
    identical field layouts without duplicating code.

    Parameters
    ----------
    adata : AnnData
        Modified in place.
    labels : np.ndarray
        Cluster labels in cluster-space order. For remove, already subset
        by factor_keep. For merge, full-length with merged label values.
    source : str
        "merge" or "remove". Recorded in juzi_cluster_meta.
    factor_keep : np.ndarray | None
        Boolean mask (cluster-space length before removal) used by remove
        to subset the similarity matrix, names, and G. None for merge.
    """
    S = adata.uns["juzi_cluster_similarity"].copy()
    names = np.array(adata.uns["juzi_cluster_names"], dtype=object)
    prev_order = np.array(adata.uns.get("juzi_cluster_order", []), dtype=int)

    # Derive G_masked from the cluster-order index stored at cluster time.
    # We must NOT use juzi_keep_cluster here: programs_remove mutates it
    # before calling this function, so indexing varm with it would already
    # exclude the removed factors — making G_masked shorter than S / names
    # and causing an index mismatch on the factor_keep step below.
    # juzi_cluster_order always holds global factor indices (into varm columns)
    # in the pre-modification cluster-space order, which is what we need.
    if len(prev_order) > 0:
        G_masked = adata.varm["juzi_G"].T[prev_order]
    else:
        # Fallback for objects written before juzi_cluster_order existed:
        # reconstruct cluster-space G from the similarity idx + old keep mask.
        # This path should not be reached for newly clustered objects.
        sim_idx = adata.uns["juzi_similarity_idx"]
        old_labels_full = adata.uns["juzi_cluster_labels"]  # pre-modification
        n_cluster_space = len(old_labels_full)
        # sim_idx positions that were in cluster space
        in_cluster = np.zeros(len(sim_idx), dtype=bool)
        for i, idx in enumerate(sim_idx):
            if i < n_cluster_space:
                in_cluster[i] = True
        G_masked = adata.varm["juzi_G"].T[sim_idx[in_cluster[: len(sim_idx)]]]

    if factor_keep is not None:
        S = S[np.ix_(factor_keep, factor_keep)]
        names = names[factor_keep]
        G_masked = G_masked[factor_keep]
        if len(prev_order) > 0:
            prev_order = prev_order[factor_keep]

    method = adata.uns.get("juzi_threshold_sweep", {}).get("method", "average")
    reorder_idx = _reorder_clusters(S, labels, method=method)

    S = S[np.ix_(reorder_idx, reorder_idx)]
    names = names[reorder_idx]
    labels = labels[reorder_idx]
    G_masked = G_masked[reorder_idx]

    if len(prev_order) > 0:
        global_order = prev_order[reorder_idx]
    else:
        global_order = reorder_idx.copy()

    # Remap to contiguous labels
    _, first_occurrence = np.unique(labels, return_index=True)
    ordered_old = labels[np.sort(first_occurrence)]

    remapped = np.empty_like(labels)
    for new_label, old_label in enumerate(ordered_old):
        remapped[labels == old_label] = new_label
    labels = remapped

    unique_clusters = np.unique(labels)
    cluster_G = np.array([G_masked[labels == c].mean(axis=0) for c in unique_clusters])

    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)
    existing_genes = adata.uns.get("juzi_cluster_genes", {})
    n_top_genes = len(next(iter(existing_genes.values()))) if existing_genes else 50

    if "juzi_cluster_mp_genes" in adata.uns:
        old_mp = adata.uns["juzi_cluster_mp_genes"]
        # For both merge and remove: remap surviving keys to new contiguous labels.
        # For merge, the caller already collapsed merged labels into min(group),
        # so the target label's existing MP gene list is carried forward unchanged.
        new_mp: Dict[int, List[str]] = {
            new_label: old_mp[int(old_label)]
            for new_label, old_label in enumerate(ordered_old)
            if int(old_label) in old_mp
        }
        adata.uns["juzi_cluster_mp_genes"] = new_mp
        cluster_genes: Dict[int, List[str]] = {k: sorted(v) for k, v in new_mp.items()}
    else:
        G_rank = _combined_score(cluster_G)
        cluster_genes = {
            int(c): gene_names[np.argsort(G_rank[i])[-n_top_genes:][::-1]].tolist()
            for i, c in enumerate(unique_clusters)
        }

    cluster_samples = {
        int(c): np.unique(names[labels == c]).tolist() for c in unique_clusters
    }

    inner_mask = labels[:, None] == labels[None, :]
    outer_mask = ~inner_mask
    inner_sim = float(S[inner_mask].mean()) if inner_mask.any() else 0.0
    outer_sim = float(S[outer_mask].mean()) if outer_mask.any() else 0.0

    sil_score_val = None
    nc = len(unique_clusters)
    dist_matrix = 1.0 - S
    np.fill_diagonal(dist_matrix, 0.0)

    if nc > 1 and nc < S.shape[0] - 1:
        try:
            sil_score_val = float(
                silhouette_score(dist_matrix, labels, metric="precomputed")
            )
        except Exception:
            pass

    if "juzi_jackknife" in adata.uns:
        del adata.uns["juzi_jackknife"]

    meta = dict(adata.uns.get("juzi_cluster_meta", {}))
    meta[f"posthoc_{source}"] = True

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"] = labels
    adata.uns["juzi_cluster_names"] = names.tolist()
    adata.uns["juzi_cluster_order"] = global_order
    adata.uns["juzi_cluster_G"] = cluster_G
    adata.uns["juzi_cluster_genes"] = cluster_genes
    adata.uns["juzi_cluster_samples"] = cluster_samples
    adata.uns["juzi_cluster_stats"] = {
        "silhouette_score": sil_score_val,
        "inner_similarity": inner_sim,
        "outer_similarity": outer_sim,
    }
    adata.uns["juzi_cluster_meta"] = meta


def _cluster_progressive(
    full_G: np.ndarray,
    full_names: np.ndarray,
    sim_idx: np.ndarray,
    gene_names: np.ndarray,
    cluster_mask: np.ndarray,
    top_k: int,
    min_overlap: int,
    min_founder_overlaps: int,
    min_cluster: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Set[str]]]:
    """Progressive meta-program construction."""
    active_idx = np.where(cluster_mask)[0]
    n_active = len(active_idx)

    if n_active == 0:
        raise ValueError(
            "No factors remain after applying juzi_keep mask. "
            "Lower similarity or cluster thresholds."
        )

    global_idx = sim_idx[active_idx]
    names = full_names[global_idx]
    G_active = full_G[global_idx]

    if top_k > G_active.shape[1]:
        raise ValueError(
            f"top_k={top_k} exceeds number of genes ({G_active.shape[1]})."
        )

    gene_to_col: Dict[str, int] = {g: i for i, g in enumerate(gene_names.tolist())}

    # Precompute per-factor top-k gene sets and gene->NMF-score dicts
    top_gene_sets: List[Set[str]] = []
    factor_gene_scores: List[Dict[str, float]] = []

    for i in range(n_active):
        top_idx = np.argsort(G_active[i])[-top_k:]
        genes_i = gene_names[top_idx].tolist()
        top_gene_sets.append(set(genes_i))
        factor_gene_scores.append(
            {g: float(G_active[i, gene_to_col[g]]) for g in genes_i}
        )

    assigned = np.full(n_active, -1, dtype=int)
    mp_genes: Dict[int, Set[str]] = {}
    cluster_id = 0

    # Outer loop: founder selection
    while True:
        unassigned = np.where(assigned == -1)[0]
        if len(unassigned) == 0:
            break

        # Count qualifying partners for each unassigned factor
        overlap_counts = np.zeros(len(unassigned), dtype=int)
        for i, ui in enumerate(unassigned):
            for uj in unassigned:
                if ui == uj:
                    continue
                if len(top_gene_sets[ui] & top_gene_sets[uj]) >= min_overlap:
                    overlap_counts[i] += 1

        founder_pos = int(np.argmax(overlap_counts))
        founder_count = int(overlap_counts[founder_pos])

        if founder_count <= min_founder_overlaps:
            break

        founder = unassigned[founder_pos]
        assigned[founder] = cluster_id
        members: List[int] = [founder]

        gene_history: List[List[str]] = [list(top_gene_sets[founder])]

        # Initial Genes_MP = founder's own top-k genes
        genes_mp: Set[str] = set(top_gene_sets[founder])

        # Inner loop: cluster growth
        while True:
            current_unassigned = np.where(assigned == -1)[0]
            if len(current_unassigned) == 0:
                break

            best_candidate = -1
            best_overlap = -1
            for ui in current_unassigned:
                ov = len(top_gene_sets[ui] & genes_mp)
                if ov > best_overlap:
                    best_overlap = ov
                    best_candidate = ui

            if best_candidate == -1 or best_overlap < min_overlap:
                break

            members.append(best_candidate)
            assigned[best_candidate] = cluster_id
            gene_history.append(list(top_gene_sets[best_candidate]))

            # Recompute Genes_MP from full gene history
            genes_mp = _recompute_genes_mp(
                gene_history=gene_history,
                members=members,
                factor_gene_scores=factor_gene_scores,
                top_k=top_k,
            )

        # Post-cluster filter: require >= min_cluster unique samples
        unique_samples = len(np.unique(names[members]))
        if unique_samples >= min_cluster:
            mp_genes[cluster_id] = genes_mp
            cluster_id += 1
        else:
            for m in members:
                assigned[m] = -1

    # Build output arrays aligned to cluster_mask positions
    unassigned_local = active_idx[assigned == -1]
    cluster_mask[unassigned_local] = False

    kept_active = active_idx[assigned != -1]
    clusters_kept = assigned[assigned != -1]

    kept_positions = np.where(cluster_mask)[0]
    pos_to_out: Dict[int, int] = {pos: i for i, pos in enumerate(kept_positions)}

    clusters_out = np.full(cluster_mask.sum(), -1, dtype=int)
    for ai, label in zip(kept_active, clusters_kept):
        clusters_out[pos_to_out[ai]] = label

    return clusters_out, cluster_mask, mp_genes


def _recompute_genes_mp(
    gene_history: List[List[str]],
    members: List[int],
    factor_gene_scores: List[Dict[str, float]],
    top_k: int,
) -> Set[str]:
    """Recompute Genes_MP from the full running member gene history.

    Ties at position top_k are broken by the maximum NMF loading score
    for that gene across all current cluster members.

    Parameters
    ----------
    gene_history : List[List[str]]
        Running list of top-k gene vectors from each member in order of
        addition.
    members : List[int]
        Indices into factor_gene_scores for all current members (used only
        for the tie-breaking score lookup).
    factor_gene_scores : List[Dict[str, float]]
        Per-factor gene->NMF-score mappings.
    top_k : int
        Target MP size.

    Returns
    -------
    Set[str]
        Updated MP gene set of size <= top_k.
    """
    # Step 1 gene frequency = table(NMF_history)
    freq: Dict[str, int] = {}
    for gene_list in gene_history:
        for g in gene_list:
            freq[g] = freq.get(g, 0) + 1

    if not freq:
        return set()

    # Step 2 sort by frequency descending (primary key)
    sorted_genes = sorted(freq.keys(), key=lambda g: freq[g], reverse=True)

    if len(sorted_genes) <= top_k:
        return set(sorted_genes)

    # Step 3 find border frequency (frequency of the gene at position top_k)
    border_freq = freq[sorted_genes[top_k - 1]]

    above_border = [g for g in sorted_genes if freq[g] > border_freq]
    at_border = [g for g in sorted_genes if freq[g] == border_freq]

    n_needed = top_k - len(above_border)

    if n_needed <= 0:
        return set(above_border[:top_k])

    if len(at_border) <= n_needed:
        # All tied genes fit; no tie-breaking needed
        return set(above_border + at_border)

    # Step 4 tie-break by maximum NMF loading score across cluster members
    max_score: Dict[str, float] = {}
    for g in at_border:
        best = 0.0
        for m in members:
            s = factor_gene_scores[m].get(g, 0.0)
            if s > best:
                best = s
        max_score[g] = best

    at_border_sorted = sorted(at_border, key=lambda g: max_score[g], reverse=True)

    return set(above_border + at_border_sorted[:n_needed])


def _cluster_centroid(
    S_full: np.ndarray,
    full_names: np.ndarray,
    sim_idx: np.ndarray,
    cluster_mask: np.ndarray,
    threshold: float,
    min_cluster: int,
    method: str,
    exclude_intra: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterative hierarchical clustering on the similarity matrix."""
    max_merge_iter = 10_000

    while True:
        S = S_full[np.ix_(cluster_mask, cluster_mask)]
        names = full_names[sim_idx[cluster_mask]]
        n = S.shape[0]

        if n == 0:
            raise ValueError(
                "No factors remain after applying juzi_keep mask. "
                "Lower similarity or cluster thresholds."
            )

        Z = sp.cluster.hierarchy.linkage(1.0 - S, method=method)
        leaf_order = sp.cluster.hierarchy.leaves_list(Z)
        clusters = np.empty(n, dtype=int)
        for new_label, position in enumerate(leaf_order):
            clusters[position] = new_label

        for _ in range(max_merge_iter):
            max_pair = _find_max_similar(
                S,
                clusters,
                threshold,
                names=names if exclude_intra else None,
            )
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

        keep_clusters = unique_clusters[passes]
        factor_passes = np.isin(clusters, keep_clusters)
        local_indices = np.where(cluster_mask)[0]
        cluster_mask[local_indices[~factor_passes]] = False

    return clusters, cluster_mask


def _cluster_at_threshold(
    S_full: np.ndarray,
    names: np.ndarray,
    cluster_mask: np.ndarray,
    threshold: float,
    min_cluster: int,
    method: str = "average",
    exclude_intra: bool = False,
) -> np.ndarray | None:
    """Run centroid clustering at a single threshold value (for programs_threshold)."""
    max_outer = 100
    max_merge_iter = 10_000

    for _ in range(max_outer):
        S = S_full[np.ix_(cluster_mask, cluster_mask)]
        names_active = names[cluster_mask]
        n = S.shape[0]

        if n == 0:
            return None

        Z = sp.cluster.hierarchy.linkage(1.0 - S, method=method)
        leaf_order = sp.cluster.hierarchy.leaves_list(Z)
        clusters = np.empty(n, dtype=int)
        for new_label, position in enumerate(leaf_order):
            clusters[position] = new_label

        for _ in range(max_merge_iter):
            max_pair = _find_max_similar(
                S,
                clusters,
                threshold,
                names=names_active if exclude_intra else None,
            )
            if max_pair is None:
                break
            i, j = max_pair
            clusters[clusters == clusters[j]] = clusters[i]

        unique_clusters = np.unique(clusters)
        sample_counts = np.array(
            [len(np.unique(names_active[clusters == c])) for c in unique_clusters]
        )
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


def _find_max_similar(
    S: np.ndarray,
    clusters: np.ndarray,
    threshold: float,
    names: np.ndarray | None = None,
) -> Tuple[int, int] | None:
    """Find the pair of distinct clusters with the highest mean inter-cluster
    similarity, returning None if no pair exceeds `threshold`.

    When `names` is provided (inter-sample only mode), only factor pairs
    from different samples contribute to the mean. Self-similarities on the
    diagonal are excluded in both modes via explicit zeroing before accumulation.
    """
    c_unique, first_indices = np.unique(clusters, return_index=True)
    X = (clusters[:, None] == c_unique[None, :]).astype(float)

    if names is not None:
        cross_mask = (names[:, None] != names[None, :]).astype(float)
        S_off = S * cross_mask
        sum_sims = X.T @ S_off @ X
        sum_valid = X.T @ cross_mask @ X
        mean_sims = np.zeros_like(sum_sims)
        np.divide(sum_sims, sum_valid, out=mean_sims, where=sum_valid > 0)
    else:
        # Zero diagonal before accumulating to exclude self-similarities
        S_off = S.copy()
        np.fill_diagonal(S_off, 0.0)
        c_counts = np.array([np.sum(clusters == c) for c in c_unique], dtype=float)
        sum_sims = X.T @ S_off @ X
        # Off-diagonal pair (a,b): denominator = c_counts[a] * c_counts[b]
        # Diagonal (self) a:       denominator = c_counts[a] * (c_counts[a] - 1)
        denom = np.outer(c_counts, c_counts)
        np.fill_diagonal(denom, c_counts * (c_counts - 1))
        mean_sims = np.zeros_like(sum_sims)
        np.divide(sum_sims, denom, out=mean_sims, where=denom > 0)

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
    method: str = "average",
) -> np.ndarray:
    """Return a permutation index that places factors in heatmap-ready order.

    Clusters are sorted by descending size (largest program first). Within
    each cluster, factors are sorted by internal similarity via hierarchical
    clustering so the most similar factors are adjacent — the natural display
    order for a block-diagonal heatmap.

    Returns
    -------
    np.ndarray
        Integer index array usable as arr[reorder_idx] or
        S[np.ix_(reorder_idx, reorder_idx)].
    """
    c_unique = np.unique(clusters)
    c_sizes = {c: int(np.sum(clusters == c)) for c in c_unique}
    c_sorted = sorted(c_unique, key=lambda c: c_sizes[c], reverse=True)

    reorder_idx: List[int] = []
    for c in c_sorted:
        idx = np.where(clusters == c)[0]
        if len(idx) > 1:
            Si = S[np.ix_(idx, idx)]
            D = sp.spatial.distance.squareform(1.0 - Si, checks=False)
            Z = sp.cluster.hierarchy.linkage(D, method=method)
            sub_order = sp.cluster.hierarchy.leaves_list(Z)
            reorder_idx.extend(idx[sub_order].tolist())
        else:
            reorder_idx.extend(idx.tolist())

    return np.array(reorder_idx, dtype=int)


def _validate_cluster_inputs(adata: AnnData) -> None:
    """Raise KeyError if any field required before clustering is absent."""
    for field, store in [
        ("juzi_similarity", "uns"),
        ("juzi_similarity_idx", "uns"),
        ("juzi_names", "uns"),
        ("juzi_G", "varm"),
        ("juzi_G_genes", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. "
                "Run juzi.gp.nmf_fit and juzi.gp.similarity first."
            )


def _validate_posthoc_inputs(adata: AnnData) -> None:
    """Raise KeyError if any field required for post-hoc operations is absent."""
    for field, store in [
        ("juzi_cluster_labels", "uns"),
        ("juzi_cluster_G", "uns"),
        ("juzi_cluster_names", "uns"),
        ("juzi_cluster_samples", "uns"),
        ("juzi_cluster_similarity", "uns"),
        ("juzi_keep_cluster", "uns"),
        ("juzi_keep", "uns"),
        ("juzi_similarity_idx", "uns"),
        ("juzi_cluster_genes", "uns"),
        ("juzi_G_genes", "uns"),
        ("juzi_G", "varm"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. "
                "Run juzi.gp.programs_cluster before post-hoc operations."
            )
