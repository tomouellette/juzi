# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np

from anndata import AnnData
from joblib import Parallel, delayed
from typing import Callable
from tqdm import tqdm


def similarity(
    adata: AnnData,
    distance: str | Callable = "jaccard",
    top_k: int | None = 50,
    intra_sample: bool = True,
    drop_zeros: bool = True,
    min_similarity: float = 0.2,
    n_jobs: int = 1,
    prefer: str | None = None,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Compute pairwise similarity between gene program factors across samples.

    Builds a symmetric factor × factor similarity matrix using either Jaccard
    similarity on top-loaded genes or a user-provided distance function.
    Factors that are all-zero or fall below min_similarity are flagged in
    juzi_keep for exclusion from downstream clustering.

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
        juzi_keep. These factors have no similarity to any other factor
        and contribute nothing to consensus program detection.
    min_similarity : float
        Flag factors whose maximum similarity to any other factor is below
        this threshold in juzi_keep. Must be in [0, 1].
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
            .uns["juzi_similarity"] : factor × factor similarity matrix
            .uns["juzi_keep"]       : updated boolean mask
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

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1].")

    if callable(distance):
        _validate_distance_fn(distance)

    # Setup

    G = adata.varm["juzi_G"].T  # (n_factors × n_genes)
    names = np.array(adata.uns["juzi_names"])
    n = G.shape[0]

    # Initialise juzi_keep if not present (i.e. prune was not run)
    if "juzi_keep" not in adata.uns:
        adata.uns["juzi_keep"] = np.ones(n, dtype=bool)

    keep = adata.uns["juzi_keep"]

    # Build index pairs
    # Only compute upper triangle — result is symmetric
    # Optionally skip intra-sample pairs

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

    # Update juzi_keep
    # Apply both filters in a single pass — flag factors that are
    # all-zero or fall below min_similarity threshold

    if drop_zeros:
        keep[np.isclose(sim, 0).all(axis=1)] = False

    if min_similarity > 0.0:
        keep[sim.max(axis=1) < min_similarity] = False

    adata.uns["juzi_keep"] = keep
    adata.uns["juzi_similarity"] = sim

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

    # Zero vectors have no meaningful similarity
    if np.sum(x) == 0 or np.sum(y) == 0:
        return (i, j, 0.0)

    if distance == "jaccard":
        top_x = np.argsort(x)[-int(top_k) :]
        top_y = np.argsort(y)[-int(top_k) :]
        union = np.union1d(top_x, top_y)

        if len(union) == 0:
            return (i, j, 0.0)

        s_xy = len(np.intersect1d(top_x, top_y)) / len(union)

    else:
        if top_k is not None:
            union = np.union1d(
                np.argsort(x)[-int(top_k) :],
                np.argsort(y)[-int(top_k) :],
            )
            x, y = x[union], y[union]

        s_xy = float(distance(x, y))

    if np.isnan(s_xy):
        return (i, j, 0.0)

    return (i, j, float(s_xy))
