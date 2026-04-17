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
    min_founder_overlaps: int = 5,
    copy: bool = False,
) -> AnnData | None:
    """Cluster NMF factors into consensus programs.

    Two clustering strategies are supported:

    - strategy="centroid":
        Iterative hierarchical clustering on the similarity matrix with
        cluster merging controlled by `threshold`.
    - strategy="progressive":
        Progressive meta-program (MP) construction style procedure.
        Programs are grown from a founder factor by iteratively adding
        the best-overlapping factor and updating the MP gene set.

    Both strategies populate a common set of cluster fields so downstream
    steps can treat the result uniformly. `juzi_cluster_genes` is the
    canonical gene definition for each program in both modes.

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.gp.nmf_fit and juzi.gp.similarity.
    strategy : str
        One of "centroid" or "progressive".
    min_cluster : int
        Minimum number of unique samples per cluster/program.
    reorder : bool
        If True, sort clusters by size and factors within clusters by
        internal similarity.
    n_top_genes : int
        Number of canonical genes per centroid cluster to store in
        `juzi_cluster_genes`.
    threshold : float
        Centroid mode only. Merge clusters until maximum inter-cluster
        similarity falls below this value.
        method : str
        Centroid mode linkage method. One of "average", "complete", "ward".
    top_k : int
        Progressive mode only. Number of genes per MP.
    min_overlap : int
        Progressive mode only. Minimum shared genes between a candidate
        factor and the current MP to be eligible for addition.
    min_founder_overlaps : int
        Progressive mode only. Minimum number of overlapping unassigned
        factors required for a founder to initiate a cluster.
        To mimic the paper's "exceeded 5 cases" rule, pass 6.
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
            .uns["juzi_cluster_G"]          : centroid loading per cluster
            .uns["juzi_cluster_genes"]      : canonical gene set per cluster
            .uns["juzi_cluster_mp_genes"]   : MP genes for progressive mode
            .uns["juzi_cluster_samples"]    : unique contributing samples
            .uns["juzi_cluster_stats"]      : silhouette, inner/outer similarity
            .uns["juzi_cluster_meta"]       : cluster strategy + parameters
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
    min_founder_overlaps: int = 5,
    min_cluster: int = 2,
    reorder: bool = True,
    copy: bool = False,
) -> AnnData | None:
    """Cluster factors into consensus programs via progressive MP construction."""
    adata = adata.copy() if copy else adata

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
    """Shared post-clustering bookkeeping."""
    n_total = adata.varm["juzi_G"].shape[1]

    keep_cluster_global = np.zeros(n_total, dtype=bool)
    keep_cluster_global[sim_idx[cluster_mask]] = True
    adata.uns["juzi_keep_cluster"] = keep_cluster_global
    _recompute_keep(adata)

    S = adata.uns["juzi_similarity"][np.ix_(cluster_mask, cluster_mask)]
    names = full_names[sim_idx[cluster_mask]]
    G_masked = full_G[sim_idx[cluster_mask]]

    if reorder:
        reorder_idx = _reorder_clusters(S, clusters, method=method)
        S = S[np.ix_(reorder_idx, reorder_idx)]
        names = names[reorder_idx]
        clusters = clusters[reorder_idx]
        G_masked = G_masked[reorder_idx]

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

    if mp_genes is not None:
        mp_genes_remapped = {
            int(new_label): mp_genes[int(old_label)]
            for new_label, old_label in enumerate(ordered_old)
            if int(old_label) in mp_genes
        }
        cluster_genes = {
            int(c): sorted(mp_genes_remapped[int(c)])
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

    adata.uns["juzi_cluster_genes"] = cluster_genes

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

        while True:
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

    gene_to_idx = {g: i for i, g in enumerate(gene_names.tolist())}

    top_gene_sets: List[Set[str]] = []
    factor_gene_scores: List[Dict[str, float]] = []

    for i in range(n_active):
        top_idx = np.argsort(G_active[i])[-top_k:]
        genes_i = gene_names[top_idx].tolist()
        top_gene_sets.append(set(genes_i))
        factor_gene_scores.append(
            {g: float(G_active[i, gene_to_idx[g]]) for g in genes_i}
        )

    assigned = np.full(n_active, -1, dtype=int)
    mp_genes: Dict[int, Set[str]] = {}
    cluster_id = 0

    while True:
        unassigned = np.where(assigned == -1)[0]
        if len(unassigned) == 0:
            break

        overlap_counts = np.zeros(len(unassigned), dtype=int)
        best_partner_for = np.full(len(unassigned), -1, dtype=int)
        best_partner_overlap = np.full(len(unassigned), -1, dtype=int)

        for i, ui in enumerate(unassigned):
            for uj in unassigned:
                if ui == uj:
                    continue
                ov = len(top_gene_sets[ui] & top_gene_sets[uj])
                if ov >= min_overlap:
                    overlap_counts[i] += 1
                    if ov > best_partner_overlap[i]:
                        best_partner_overlap[i] = ov
                        best_partner_for[i] = uj

        founder_pos = int(np.argmax(overlap_counts))
        founder = unassigned[founder_pos]

        if overlap_counts[founder_pos] < min_founder_overlaps:
            break

        seed_partner = int(best_partner_for[founder_pos])
        if seed_partner == -1:
            break

        members = [founder, seed_partner]
        assigned[founder] = cluster_id
        assigned[seed_partner] = cluster_id

        mp = _initialise_mp_from_seed(
            members=members,
            top_gene_sets=top_gene_sets,
            factor_gene_scores=factor_gene_scores,
            top_k=top_k,
        )

        while True:
            current_unassigned = np.where(assigned == -1)[0]
            if len(current_unassigned) == 0:
                break

            best_candidate = -1
            best_overlap = -1

            for ui in current_unassigned:
                ov = len(top_gene_sets[ui] & mp)
                if ov > best_overlap:
                    best_overlap = ov
                    best_candidate = ui

            if best_candidate == -1 or best_overlap < min_overlap:
                break

            members.append(best_candidate)
            assigned[best_candidate] = cluster_id

            mp = _recompute_mp_from_members(
                members=members,
                top_gene_sets=top_gene_sets,
                factor_gene_scores=factor_gene_scores,
                top_k=top_k,
            )

        unique_samples = len(np.unique(names[members]))
        if unique_samples >= min_cluster:
            mp_genes[cluster_id] = mp
            cluster_id += 1
        else:
            for m in members:
                assigned[m] = -1

    unassigned_local = active_idx[assigned == -1]
    cluster_mask[unassigned_local] = False

    kept_active = active_idx[assigned != -1]
    clusters_kept = assigned[assigned != -1]

    kept_positions = np.where(cluster_mask)[0]
    pos_to_out = {pos: i for i, pos in enumerate(kept_positions)}

    clusters_out = np.full(cluster_mask.sum(), -1, dtype=int)
    for ai, label in zip(kept_active, clusters_kept):
        clusters_out[pos_to_out[ai]] = label

    return clusters_out, cluster_mask, mp_genes


def _initialise_mp_from_seed(
    members: List[int],
    top_gene_sets: List[Set[str]],
    factor_gene_scores: List[Dict[str, float]],
    top_k: int,
) -> Set[str]:
    """Initialise MP from the two-program seed."""
    if len(members) != 2:
        raise ValueError("Seed initialisation requires exactly 2 members.")

    a, b = members
    common = set(top_gene_sets[a] & top_gene_sets[b])

    return _complete_mp(
        common=common,
        members=members,
        top_gene_sets=top_gene_sets,
        factor_gene_scores=factor_gene_scores,
        top_k=top_k,
        use_frequency=False,
    )


def _recompute_mp_from_members(
    members: List[int],
    top_gene_sets: List[Set[str]],
    factor_gene_scores: List[Dict[str, float]],
    top_k: int,
) -> Set[str]:
    """Recompute MP after adding a member."""
    common = set(top_gene_sets[members[0]])
    for m in members[1:]:
        common &= top_gene_sets[m]

    return _complete_mp(
        common=common,
        members=members,
        top_gene_sets=top_gene_sets,
        factor_gene_scores=factor_gene_scores,
        top_k=top_k,
        use_frequency=True,
    )


def _complete_mp(
    common: Set[str],
    members: List[int],
    top_gene_sets: List[Set[str]],
    factor_gene_scores: List[Dict[str, float]],
    top_k: int,
    use_frequency: bool,
) -> Set[str]:
    """Complete an MP gene set to top_k."""
    if len(common) >= top_k:
        common_scores = {}
        for g in common:
            common_scores[g] = sum(factor_gene_scores[m].get(g, 0.0) for m in members)
        ordered = sorted(common, key=lambda g: common_scores[g], reverse=True)
        return set(ordered[:top_k])

    gene_freq: Dict[str, int] = {}
    gene_score: Dict[str, float] = {}

    for m in members:
        for g in top_gene_sets[m]:
            if g in common:
                continue
            gene_freq[g] = gene_freq.get(g, 0) + 1
            gene_score[g] = gene_score.get(g, 0.0) + factor_gene_scores[m].get(g, 0.0)

    if use_frequency:
        ordered = sorted(
            gene_freq.keys(),
            key=lambda g: (gene_freq[g], gene_score[g], g),
            reverse=True,
        )
    else:
        ordered = sorted(
            gene_freq.keys(),
            key=lambda g: (gene_score[g], gene_freq[g], g),
            reverse=True,
        )

    mp = set(common)
    for g in ordered:
        if len(mp) >= top_k:
            break
        mp.add(g)

    return mp


def _find_max_similar(
    S: np.ndarray,
    clusters: np.ndarray,
    threshold: float,
    names: np.ndarray | None = None,
) -> Tuple[int, int] | None:
    """Find the pair of clusters with the highest mean inter-cluster similarity."""
    c_unique, first_indices = np.unique(clusters, return_index=True)
    X = (clusters[:, None] == c_unique[None, :]).astype(float)

    if names is not None:
        cross_mask = (names[:, None] != names[None, :]).astype(float)
        sum_sims = X.T @ (S * cross_mask) @ X
        sum_valid = X.T @ cross_mask @ X
        mean_sims = np.zeros_like(sum_sims)
        np.divide(sum_sims, sum_valid, out=mean_sims, where=sum_valid > 0)
    else:
        c_counts = np.array([np.sum(clusters == c) for c in c_unique])
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
    method: str = "average",
) -> np.ndarray:
    """Reorder factors so clusters are sorted by size and internally by similarity."""
    c_unique = np.unique(clusters)
    c_sizes = {c: np.sum(clusters == c) for c in c_unique}
    c_sorted = sorted(c_unique, key=lambda c: c_sizes[c], reverse=True)

    reorder_idx = []
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


def _cluster_at_threshold(
    S_full: np.ndarray,
    names: np.ndarray,
    cluster_mask: np.ndarray,
    threshold: float,
    min_cluster: int,
    method: str = "average",
    exclude_intra: bool = False,
) -> np.ndarray | None:
    """Run centroid clustering at a single threshold value."""
    max_iter = 100
    for _ in range(max_iter):
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

        for _ in range(n * n):
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

    This function is only meaningful for centroid clustering.
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
    """Manually merge consensus programs post-hoc."""
    adata = adata.copy() if copy else adata

    for field in [
        "juzi_cluster_labels",
        "juzi_cluster_G",
        "juzi_cluster_names",
        "juzi_cluster_samples",
        "juzi_cluster_similarity",
        "juzi_keep_cluster",
        "juzi_cluster_genes",
    ]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. "
                "Run juzi.gp.programs_cluster before merging."
            )

    labels = adata.uns["juzi_cluster_labels"].copy()
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

    S = adata.uns["juzi_cluster_similarity"].copy()
    names = np.array(adata.uns["juzi_cluster_names"], dtype=object)
    G_masked = adata.varm["juzi_G"].T[adata.uns["juzi_keep_cluster"]]
    method = adata.uns.get("juzi_threshold_sweep", {}).get("method", "average")

    reorder_idx = _reorder_clusters(S, labels, method=method)
    S = S[np.ix_(reorder_idx, reorder_idx)]
    names = names[reorder_idx]
    labels = labels[reorder_idx]
    G_masked = G_masked[reorder_idx]

    _, first_occurrence = np.unique(labels, return_index=True)
    ordered_old = labels[np.sort(first_occurrence)]

    remapped = np.empty_like(labels)
    for new_label, old_label in enumerate(ordered_old):
        remapped[labels == old_label] = new_label
    labels = remapped

    unique_clusters = np.unique(labels)
    cluster_G = np.array([G_masked[labels == c].mean(axis=0) for c in unique_clusters])

    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)
    existing = adata.uns.get("juzi_cluster_genes", {})
    n_top_genes = len(next(iter(existing.values()))) if existing else 50

    if "juzi_cluster_mp_genes" in adata.uns:
        old_mp = adata.uns["juzi_cluster_mp_genes"]
        new_mp: Dict[int, List[str]] = {}

        for group in merge_groups:
            group = [int(c) for c in group]
            target = min(group)
            merged: Set[str] = set()
            for c in group:
                if c in old_mp:
                    merged |= set(old_mp[c])
            new_label_idx = int(np.where(ordered_old == target)[0][0])
            new_mp[new_label_idx] = sorted(merged)

        mentioned = {int(c) for g in merge_groups for c in g}
        for new_label, old_label in enumerate(ordered_old):
            if int(old_label) not in mentioned and int(old_label) in old_mp:
                new_mp[new_label] = old_mp[int(old_label)]

        adata.uns["juzi_cluster_mp_genes"] = new_mp
        cluster_genes = {k: sorted(v) for k, v in new_mp.items()}
    else:
        G_rank = _combined_score(cluster_G)
        cluster_genes = {
            int(c): gene_names[np.argsort(G_rank[i])[-n_top_genes:][::-1]].tolist()
            for i, c in enumerate(unique_clusters)
        }

    adata.uns["juzi_cluster_genes"] = cluster_genes

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
    meta["posthoc_merge"] = True

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"] = labels
    adata.uns["juzi_cluster_names"] = names.tolist()
    adata.uns["juzi_cluster_G"] = cluster_G
    adata.uns["juzi_cluster_genes"] = cluster_genes
    adata.uns["juzi_cluster_samples"] = cluster_samples
    adata.uns["juzi_cluster_stats"] = {
        "silhouette_score": sil_score_val,
        "inner_similarity": inner_sim,
        "outer_similarity": outer_sim,
    }
    adata.uns["juzi_cluster_meta"] = meta

    return adata if copy else None


def programs_remove(
    adata: AnnData,
    clusters: List[int],
    copy: bool = False,
) -> AnnData | None:
    """Remove consensus programs post-hoc."""
    adata = adata.copy() if copy else adata

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
                "Run juzi.gp.programs_cluster before removing programs."
            )

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

    sim_idx = adata.uns["juzi_similarity_idx"]
    keep_cluster_global = adata.uns["juzi_keep_cluster"].copy()
    in_cluster = keep_cluster_global[sim_idx]
    sim_cluster_global = sim_idx[in_cluster]

    remove_positions = np.where(np.isin(labels, list(remove_set)))[0]
    for pos in remove_positions:
        keep_cluster_global[sim_cluster_global[pos]] = False

    adata.uns["juzi_keep_cluster"] = keep_cluster_global
    _recompute_keep(adata)

    factor_keep = ~np.isin(labels, list(remove_set))
    names = np.array(adata.uns["juzi_cluster_names"], dtype=object)[factor_keep]
    labels = labels[factor_keep]
    S = adata.uns["juzi_cluster_similarity"][np.ix_(factor_keep, factor_keep)]
    G_masked = adata.varm["juzi_G"].T[adata.uns["juzi_keep_cluster"]]

    method = adata.uns.get("juzi_threshold_sweep", {}).get("method", "average")
    reorder_idx = _reorder_clusters(S, labels, method=method)

    S = S[np.ix_(reorder_idx, reorder_idx)]
    names = names[reorder_idx]
    labels = labels[reorder_idx]
    G_masked = G_masked[reorder_idx]

    _, first_occurrence = np.unique(labels, return_index=True)
    ordered_old = labels[np.sort(first_occurrence)]

    remapped = np.empty_like(labels)
    for new_label, old_label in enumerate(ordered_old):
        remapped[labels == old_label] = new_label
    labels = remapped

    unique_clusters = np.unique(labels)
    cluster_G = np.array([G_masked[labels == c].mean(axis=0) for c in unique_clusters])

    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)
    existing = adata.uns.get("juzi_cluster_genes", {})
    n_top_genes = len(next(iter(existing.values()))) if existing else 50

    if "juzi_cluster_mp_genes" in adata.uns:
        old_mp = adata.uns["juzi_cluster_mp_genes"]
        new_mp = {
            new_label: old_mp[int(old_label)]
            for new_label, old_label in enumerate(ordered_old)
            if int(old_label) in old_mp
        }
        adata.uns["juzi_cluster_mp_genes"] = new_mp
        cluster_genes = {k: sorted(v) for k, v in new_mp.items()}
    else:
        G_rank = _combined_score(cluster_G)
        cluster_genes = {
            int(c): gene_names[np.argsort(G_rank[i])[-n_top_genes:][::-1]].tolist()
            for i, c in enumerate(unique_clusters)
        }

    adata.uns["juzi_cluster_genes"] = cluster_genes

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
    meta["posthoc_remove"] = True

    adata.uns["juzi_cluster_similarity"] = S
    adata.uns["juzi_cluster_labels"] = labels
    adata.uns["juzi_cluster_names"] = names.tolist()
    adata.uns["juzi_cluster_G"] = cluster_G
    adata.uns["juzi_cluster_genes"] = cluster_genes
    adata.uns["juzi_cluster_samples"] = cluster_samples
    adata.uns["juzi_cluster_stats"] = {
        "silhouette_score": sil_score_val,
        "inner_similarity": inner_sim,
        "outer_similarity": outer_sim,
    }
    adata.uns["juzi_cluster_meta"] = meta

    return adata if copy else None
