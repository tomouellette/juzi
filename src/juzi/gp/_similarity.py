# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np

from anndata import AnnData
from joblib import Parallel, delayed
from typing import Callable
from tqdm import tqdm

from ._nmf import _recompute_keep


def similarity(
    adata: AnnData,
    distance: str | Callable = "jaccard",
    top_k: int | None = 50,
    intra_sample: bool = True,
    drop_zeros: bool = True,
    n_jobs: int = 1,
    prefer: str | None = None,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Compute pairwise similarity between gene program factors across samples.

    Builds a symmetric factor × factor similarity matrix using either Jaccard
    similarity on top-loaded genes or a user-provided distance function.
    Factors whose entire similarity row is zero are flagged in
    juzi_keep_similarity when drop_zeros=True.

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.gp.nmf and juzi.gp.prune.
    distance : str | Callable
        Similarity metric. "jaccard" computes overlap of top-k gene sets.
        A callable must accept two 1-d float arrays and return a scalar.
    top_k : int | None
        Number of top-loaded genes used per factor for Jaccard similarity.
        Required when distance="jaccard". Ignored for callable distances.
    intra_sample : bool
        If True, compute similarity between factors from the same sample.
        If False, only inter-sample similarities are computed and
        intra-sample entries remain zero.
    drop_zeros : bool
        If True, flag factors whose entire similarity row is zero in
        juzi_keep_similarity. These factors have no similarity to any
        other factor and contribute nothing to consensus program detection.
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
            .uns["juzi_similarity"]      : factor × factor similarity matrix
            .uns["juzi_keep_similarity"] : boolean mask updated by drop_zeros
            .uns["juzi_keep"]            : intersection of all three stage masks
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
                f"'{field}' not found in .{store}. Run juzi.gp.nmf first."
            )

    if distance != "jaccard" and not callable(distance):
        raise ValueError("distance must be 'jaccard' or a callable.")

    if distance == "jaccard" and top_k is None:
        raise ValueError("top_k must be set when using distance='jaccard'.")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if callable(distance):
        _validate_distance_fn(distance)

    # Setup

    G     = adata.varm["juzi_G"].T # (n_factors × n_genes)
    names = np.array(adata.uns["juzi_names"])
    n     = G.shape[0]

    # Build index pairs

    rows, cols = np.triu_indices(n, k=1)

    if not intra_sample:
        inter_mask = names[rows] != names[cols]
        rows, cols = rows[inter_mask], cols[inter_mask]

    indices = list(zip(rows.tolist(), cols.tolist()))

    # Compute pairwise similarities

    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_similarity)(
            i=i,
            j=j,
            G=G,
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

    # Update juzi_keep_similarity
    # Reset to all True then apply drop_zeros filter only
    # min_similarity filtering is handled separately by select_similarity

    keep_sim = np.ones(n, dtype=bool)

    if drop_zeros:
        keep_sim[np.isclose(sim, 0).all(axis=1)] = False

    adata.uns["juzi_keep_similarity"] = keep_sim
    _recompute_keep(adata)

    return adata if copy else None


def select_similarity(
    adata: AnnData,
    min_similarity: float,
    copy: bool = False,
) -> AnnData | None:
    """Filter factors by minimum similarity threshold.

    Updates juzi_keep_similarity to flag factors whose maximum similarity
    to any other factor falls below min_similarity. Can be re-run with
    different thresholds without re-running juzi.gp.similarity.

    Use juzi.pl.similarity to inspect the min_similarity vs factors
    retained curve before choosing a threshold.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_similarity in .uns, produced by
        juzi.gp.similarity.
    min_similarity : float
        Minimum similarity threshold. Factors whose maximum similarity
        to any other factor is below this value are flagged in
        juzi_keep_similarity. Must be in [0, 1].
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

    # Validate

    if "juzi_similarity" not in adata.uns:
        raise KeyError(
            "'juzi_similarity' not found in .uns. "
            "Run juzi.gp.similarity first."
        )

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1].")

    # Apply threshold

    sim      = adata.uns["juzi_similarity"]
    n        = sim.shape[0]

    # Start from drop_zeros result if already computed, otherwise all True
    keep_sim = adata.uns.get(
        "juzi_keep_similarity", np.ones(n, dtype=bool)
    ).copy()

    # Apply min_similarity on top of existing drop_zeros mask
    keep_sim[sim.max(axis=1) < min_similarity] = False

    adata.uns["juzi_keep_similarity"] = keep_sim
    _recompute_keep(adata)

    return adata if copy else None


def _validate_distance_fn(distance: Callable) -> None:
    """Validate that a callable distance function accepts two arrays and
    returns a scalar."""
    x, y = np.random.default_rng(0).random(4), np.random.default_rng(1).random(4)
    try:
        result = distance(x, y)
    except Exception:
        raise ValueError("distance callable must accept two 1-d float arrays.")
    if not isinstance(result, (int, float, np.floating)):
        raise ValueError("distance callable must return a scalar value.")


def _similarity(
    i: int,
    j: int,
    G: np.ndarray,
    top_k: int | None,
    distance: str | Callable,
) -> tuple[int, int, float]:
    """Compute similarity between two factor loading vectors.

    Parameters
    ----------
    i : int
        Row index of first factor in G.
    j : int
        Row index of second factor in G.
    G : np.ndarray
        Factor loading matrix, shape (n_factors × n_genes).
    top_k : int | None
        Number of top genes to use for Jaccard similarity.
    distance : str | Callable
        Similarity metric.

    Returns
    -------
    tuple[int, int, float]
        (i, j, similarity_score)
    """
    x, y = G[i], G[j]

    if np.sum(x) == 0 or np.sum(y) == 0:
        return (i, j, 0.0)

    if distance == "jaccard":
        top_x = np.argsort(x)[-int(top_k):]
        top_y = np.argsort(y)[-int(top_k):]
        union = np.union1d(top_x, top_y)

        if len(union) == 0:
            return (i, j, 0.0)

        s_xy = len(np.intersect1d(top_x, top_y)) / len(union)

    else:
        if top_k is not None:
            union = np.union1d(
                np.argsort(x)[-int(top_k):],
                np.argsort(y)[-int(top_k):],
            )
            x, y = x[union], y[union]

        s_xy = float(distance(x, y))

    if np.isnan(s_xy):
        return (i, j, 0.0)

    return (i, j, float(s_xy))
