# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np

from anndata import AnnData
from joblib import Parallel, delayed
from tqdm import tqdm

from ._nmf import _combined_score
from ._cluster import _cluster_at_threshold


def programs_jackknife(
    adata: AnnData,
    n_top_genes: int = 50,
    use_combined: bool = False,
    n_jobs: int = 1,
    prefer: str = "threads",
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Assess consensus program stability via donor jackknife resampling.

    For each donor, removes that donor's factors from the similarity matrix
    and re-runs clustering with the same threshold and min_cluster as the
    reference run. The stability of each reference program is measured as
    the mean maximum Jaccard similarity between its top genes and any
    jackknife program's top genes across all N jackknife iterations.

    A stability score near 1.0 indicates the program's gene signature
    reappears consistently regardless of which donor is removed. A score
    near 0.0 indicates the program is fragile and likely driven by a
    small number of donors.

    Must be run after juzi.gp.programs_cluster. Reads threshold and
    min_cluster from juzi_threshold_sweep if available, otherwise
    falls back to threshold=0.1 and min_cluster=2.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_G, juzi_cluster_labels,
        juzi_similarity, juzi_similarity_idx in .uns, produced by
        juzi.gp.programs_cluster.
    n_top_genes : int
        Number of top genes per program used for Jaccard computation.
        Should match the value used in juzi.gp.score_cells.
    use_combined : bool
        If True, rank genes by combined loading x specificity score
        when extracting top genes from jackknife program centroids.
        If False, rank by raw loading magnitude.
    n_jobs : int
        Number of parallel workers. Parallelisation is across donors —
        each jackknife iteration is independent.
    prefer : str
        Joblib backend preference. "threads" is appropriate since each
        iteration operates on read-only numpy arrays with no shared
        mutable state.
    silent : bool
        If True, suppress progress bar.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields populated:
            .uns["juzi_jackknife"] : dict with keys:
                "stability"        : (K,) mean Jaccard per program
                "stability_matrix" : (K x N) per-program per-donor scores
                "donors"           : donor names aligned to matrix columns
                "n_top_genes"      : n_top_genes value used
                "threshold"        : clustering threshold used
                "min_cluster"      : min_cluster value used
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_cluster_G", "uns"),
        ("juzi_cluster_labels", "uns"),
        ("juzi_G_genes", "uns"),
        ("juzi_similarity", "uns"),
        ("juzi_similarity_idx", "uns"),
        ("juzi_names", "uns"),
        ("juzi_keep", "uns"),
        ("juzi_G", "varm"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. "
                "Run juzi.gp.programs_cluster first."
            )

    if n_top_genes < 1:
        raise ValueError("n_top_genes must be >= 1.")

    # Read clustering parameters

    sweep = adata.uns.get("juzi_threshold_sweep", {})
    threshold = sweep.get("optimal", 0.1)
    min_cluster = sweep.get("min_cluster", 2)
    method = sweep.get("method", "average")

    # Setup

    sim_idx = adata.uns["juzi_similarity_idx"]  # (n_kept,)
    S_full = adata.uns["juzi_similarity"]  # (n_kept × n_kept)
    full_names = np.array(adata.uns["juzi_names"])  # (n_total,)
    full_G = adata.varm["juzi_G"].T  # (n_total × n_genes)
    gene_names = np.array(adata.uns["juzi_G_genes"])
    global_keep = adata.uns["juzi_keep"]

    sim_names = full_names[sim_idx]  # (n_kept,)
    sim_G = full_G[sim_idx]  # (n_kept × n_genes)
    base_mask = global_keep[sim_idx].copy()  # (n_kept,)

    donors = sorted(np.unique(sim_names[base_mask]).tolist())
    N = len(donors)

    # Reference program top genes

    G_ref = adata.uns["juzi_cluster_G"]  # (K × n_genes)
    labels_ref = adata.uns["juzi_cluster_labels"]
    unique_C = np.unique(labels_ref)
    K = len(unique_C)

    G_rank_ref = _combined_score(G_ref) if use_combined else G_ref

    ref_genes = []
    for i, c in enumerate(unique_C):
        top_idx = np.argsort(G_rank_ref[i])[-n_top_genes:]
        ref_genes.append(set(gene_names[top_idx].tolist()))

    # Parallel jackknife iterations

    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_jackknife_iteration)(
            donor=donor,
            base_mask=base_mask,
            sim_names=sim_names,
            S_full=S_full,
            sim_G=sim_G,
            gene_names=gene_names,
            ref_genes=ref_genes,
            threshold=threshold,
            min_cluster=min_cluster,
            method=method,
            n_top_genes=n_top_genes,
            use_combined=use_combined,
            K=K,
        )
        for donor in tqdm(donors, desc="[juzi] Jackknife", disable=silent)
    )

    # results: list of N arrays each (K,)
    # Stack to (K × N)
    stability_matrix = np.vstack(results).T.astype(np.float32)

    # Aggregate
    # Mean over donors that actually have factors in the similarity space.
    # By construction all donors in sim_names[base_mask] are in donors,
    # so donor_in_sim is always all True — but kept for correctness.

    donor_in_sim = np.array(
        [(sim_names[base_mask] == d).any() for d in donors], dtype=bool
    )

    n_valid = donor_in_sim.sum()
    stability = (
        stability_matrix[:, donor_in_sim].mean(axis=1)
        if n_valid > 0
        else np.zeros(K, dtype=np.float32)
    )

    # Store results

    adata.uns["juzi_jackknife"] = {
        "stability": stability,
        "stability_matrix": stability_matrix,
        "donors": donors,
        "n_top_genes": n_top_genes,
        "threshold": threshold,
        "min_cluster": min_cluster,
    }

    return adata if copy else None


def _jackknife_iteration(
    donor: str,
    base_mask: np.ndarray,
    sim_names: np.ndarray,
    S_full: np.ndarray,
    sim_G: np.ndarray,
    gene_names: np.ndarray,
    ref_genes: list,
    threshold: float,
    min_cluster: int,
    method: str,
    n_top_genes: int,
    use_combined: bool,
    K: int,
) -> np.ndarray:
    """Run one jackknife iteration for a single held-out donor.

    All inputs are read-only — base_mask and S_full are never modified.
    local_mask is a fresh copy per call so parallel execution is safe.

    Parameters
    ----------
    donor : str
        Donor identifier to hold out.
    base_mask : np.ndarray
        Base local boolean mask of active factors (length n_kept).
        Not modified — a copy is made internally.
    sim_names : np.ndarray
        Donor name per factor in similarity matrix row order.
    S_full : np.ndarray
        Full (n_kept × n_kept) similarity matrix.
    sim_G : np.ndarray
        Factor loading matrix aligned to similarity rows (n_kept × n_genes).
    gene_names : np.ndarray
        Gene names corresponding to loading columns.
    ref_genes : list
        List of sets of top gene names per reference program.
    threshold : float
        Clustering threshold.
    min_cluster : int
        Minimum unique donors per cluster.
    method : str
        Linkage method for hierarchical clustering.
    n_top_genes : int
        Number of top genes per jackknife program for Jaccard computation.
    use_combined : bool
        If True, rank genes by combined score for jackknife program centroids.
    K : int
        Number of reference programs.

    Returns
    -------
    np.ndarray
        Shape (K,) — maximum Jaccard score per reference program for this
        held-out donor. Zero for programs with no matching jackknife program.
    """
    scores = np.zeros(K, dtype=np.float32)

    # Exclude held-out donor — fresh copy, never modifies base_mask
    jack_mask = base_mask.copy()
    jack_mask[sim_names == donor] = False

    if not jack_mask.any():
        return scores

    # local_mask is modified in place by _cluster_at_threshold during
    # min_cluster removal — must be a separate copy from jack_mask
    local_mask = jack_mask.copy()

    try:
        clusters_jack = _cluster_at_threshold(
            S_full=S_full,
            names=sim_names,
            cluster_mask=local_mask,
            threshold=threshold,
            min_cluster=min_cluster,
            method=method,
        )
    except ValueError:
        return scores

    if clusters_jack is None:
        return scores

    # Extract jackknife program top genes from final local_mask
    G_jack_active = sim_G[local_mask]  # (n_active × n_genes)
    unique_jack = np.unique(clusters_jack)

    jack_genes = []
    for c in unique_jack:
        centroid = G_jack_active[clusters_jack == c].mean(axis=0, keepdims=True)
        G_rank = _combined_score(centroid) if use_combined else centroid
        top_idx = np.argsort(G_rank[0])[-n_top_genes:]
        jack_genes.append(set(gene_names[top_idx].tolist()))

    # Max Jaccard between each reference program and any jackknife program
    for k, ref_set in enumerate(ref_genes):
        max_jacc = 0.0
        for jack_set in jack_genes:
            union = ref_set | jack_set
            if len(union) == 0:
                continue
            jacc = len(ref_set & jack_set) / len(union)
            if jacc > max_jacc:
                max_jacc = jacc
        scores[k] = max_jacc

    return scores
