# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np

from anndata import AnnData
from joblib import Parallel, delayed
from typing import Callable
from tqdm import tqdm

from ._nmf import _recompute_keep, _combined_score


def similarity_compute(
    adata: AnnData,
    distance: str | Callable = "jaccard",
    top_k: int | None = 50,
    intra_sample: bool = True,
    drop_zeros: bool = True,
    use_combined: bool = True,
    n_jobs: int = 1,
    prefer: str | None = None,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Compute pairwise similarity between gene program factors across samples.

    Builds a symmetric factor × factor similarity matrix using either Jaccard
    similarity on top-loaded genes or a user-provided distance function.
    Only factors retained by juzi_keep enter the computation. The resulting
    matrix has shape (n_kept × n_kept).

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.gp.nmf and juzi.gp.prune.
    distance : str | Callable
        Similarity metric. "jaccard" computes overlap of top-k gene sets.
        A callable must accept two 1-d float arrays and return a scalar.
    top_k : int | None
        Number of top genes used per factor. Required for "jaccard".
    intra_sample : bool
        If True, compute within-donor similarities. If False, only
        inter-donor similarities are computed.
    drop_zeros : bool
        If True, flag factors whose entire similarity row is zero.
    use_combined : bool
        If True, rank genes by combined loading × specificity score
        (G * G / G.sum(axis=0)) before selecting top_k genes. Downweights
        genes that load broadly across factors, making similarity more
        discriminative. If False, rank by raw loading magnitude.
    n_jobs : int
        Number of parallel workers.
    prefer : str | None
        Joblib parallelisation backend preference.
    silent : bool
        If True, suppress progress bar.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields populated:
            .uns["juzi_similarity"]      : (n_kept × n_kept) similarity matrix
            .uns["juzi_similarity_idx"]  : global factor indices
            .uns["juzi_keep_similarity"] : boolean mask length n_total
            .uns["juzi_keep"]            : intersection of all three stage masks
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_G", "varm"),
        ("juzi_k", "uns"),
        ("juzi_names", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(f"'{field}' not found in .{store}. Run juzi.gp.nmf first.")

    if distance != "jaccard" and not callable(distance):
        raise ValueError("distance must be 'jaccard' or a callable.")

    if distance == "jaccard" and top_k is None:
        raise ValueError("top_k must be set when using distance='jaccard'.")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if callable(distance):
        _validate_distance_fn(distance)

    # Subset to kept factors

    n_total = adata.varm["juzi_G"].shape[1]
    keep = (
        adata.uns["juzi_keep"]
        if "juzi_keep" in adata.uns
        else np.ones(n_total, dtype=bool)
    )
    sim_idx = np.where(keep)[0]
    G_all = adata.varm["juzi_G"].T  # (n_total × n_genes)
    G = G_all[sim_idx]  # (n_kept × n_genes)
    names_all = np.array(adata.uns["juzi_names"])
    names = names_all[sim_idx]
    n = G.shape[0]

    # Compute gene ranking scores

    G_score = _combined_score(G) if use_combined else G

    # Build index pairs

    rows, cols = np.triu_indices(n, k=1)

    if not intra_sample:
        inter_mask = names[rows] != names[cols]
        rows, cols = rows[inter_mask], cols[inter_mask]

    indices = list(zip(rows.tolist(), cols.tolist()))

    # Compute pairwise similarities

    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_compute_similarity)(
            i=i,
            j=j,
            G=G,
            G_score=G_score,
            top_k=top_k,
            distance=distance,
        )
        for i, j in tqdm(
            indices,
            desc="[juzi] Computing similarity",
            disable=silent,
        )
    )

    # Fill symmetric similarity matrix

    sim = np.zeros((n, n), dtype=np.float32)
    for i, j, s_xy in results:
        sim[i, j] = s_xy
        sim[j, i] = s_xy

    adata.uns["juzi_similarity"] = sim
    adata.uns["juzi_similarity_idx"] = sim_idx

    # Update juzi_keep_similarity

    keep_sim = np.zeros(n_total, dtype=bool)

    if drop_zeros:
        local_pass = ~np.isclose(sim, 0).all(axis=1)
        keep_sim[sim_idx[local_pass]] = True
    else:
        keep_sim[sim_idx] = True

    adata.uns["juzi_keep_similarity"] = keep_sim
    adata.uns["juzi_similarity_intra"] = intra_sample

    _recompute_keep(adata)

    return adata if copy else None


def similarity_filter(
    adata: AnnData,
    min_similarity: float,
    copy: bool = False,
) -> AnnData | None:
    """Filter factors by minimum similarity threshold.

    Updates juzi_keep_similarity to flag factors whose maximum similarity
    to any other factor falls below min_similarity. Can be re-run with
    different thresholds without re-running juzi.gp.similarity. The
    drop_zeros mask is preserved across re-runs.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_similarity in .uns.
    min_similarity : float
        Minimum similarity threshold. Must be in [0, 1].
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields updated:
            .uns["juzi_keep_similarity"] : updated boolean mask
            .uns["juzi_keep"]            : intersection of all three stage masks
    """
    adata = adata.copy() if copy else adata

    for field in ["juzi_similarity", "juzi_similarity_idx"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. " "Run juzi.gp.similarity_compute first."
            )

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1].")

    sim = adata.uns["juzi_similarity"]
    sim_idx = adata.uns["juzi_similarity_idx"]
    n_total = adata.varm["juzi_G"].shape[1]

    keep_sim = adata.uns.get(
        "juzi_keep_similarity",
        np.zeros(n_total, dtype=bool),
    ).copy()

    local_max = sim.max(axis=1)
    local_pass = local_max >= min_similarity

    keep_sim[sim_idx] = False
    keep_sim[sim_idx[local_pass]] = True

    # Preserve drop_zeros mask
    if "juzi_keep_similarity" in adata.uns:
        local_nonzero = ~np.isclose(sim, 0).all(axis=1)
        original_drop_zeros = np.zeros(n_total, dtype=bool)
        original_drop_zeros[sim_idx[local_nonzero]] = True
        keep_sim = keep_sim & original_drop_zeros

    adata.uns["juzi_keep_similarity"] = keep_sim
    _recompute_keep(adata)

    return adata if copy else None


def _validate_distance_fn(distance: Callable) -> None:
    x, y = np.random.default_rng(0).random(4), np.random.default_rng(1).random(4)
    try:
        result = distance(x, y)
    except Exception:
        raise ValueError("distance callable must accept two 1-d float arrays.")
    if not isinstance(result, (int, float, np.floating)):
        raise ValueError("distance callable must return a scalar value.")


def _compute_similarity(
    i: int,
    j: int,
    G: np.ndarray,
    G_score: np.ndarray,
    top_k: int | None,
    distance: str | Callable,
) -> tuple[int, int, float]:
    """Compute similarity between two factor loading vectors."""
    x, y = G[i], G[j]
    x_score, y_score = G_score[i], G_score[j]

    if np.sum(x) == 0 or np.sum(y) == 0:
        return (i, j, 0.0)

    if distance == "jaccard":
        top_x = np.argsort(x_score)[-int(top_k) :]
        top_y = np.argsort(y_score)[-int(top_k) :]
        union = np.union1d(top_x, top_y)
        if len(union) == 0:
            return (i, j, 0.0)
        s_xy = len(np.intersect1d(top_x, top_y)) / len(union)
    else:
        if top_k is not None:
            union = np.union1d(
                np.argsort(x_score)[-int(top_k) :],
                np.argsort(y_score)[-int(top_k) :],
            )
            x, y = x[union], y[union]
        s_xy = float(distance(x, y))

    if np.isnan(s_xy):
        return (i, j, 0.0)

    return (i, j, float(s_xy))
