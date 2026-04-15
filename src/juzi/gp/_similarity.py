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
    Only factors retained by juzi_keep (i.e. passing prune) are included
    in the similarity computation. The resulting matrix has shape
    (n_kept × n_kept) where n_kept = juzi_keep.sum().

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
            .uns["juzi_similarity"]      : (n_kept × n_kept) similarity matrix
            .uns["juzi_similarity_idx"]  : global factor indices in similarity matrix
            .uns["juzi_keep_similarity"] : boolean mask length n_total, True where
                                           factor is kept and passes drop_zeros
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
    sim_idx = np.where(keep)[0]  # global indices to local rows/cols
    G_all = adata.varm["juzi_G"].T  # (n_total × n_genes)
    G = G_all[sim_idx]  # (n_kept × n_genes)
    names_all = np.array(adata.uns["juzi_names"])
    names = names_all[sim_idx]  # (n_kept,)
    n = G.shape[0]

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
    adata.uns["juzi_similarity_idx"] = sim_idx

    # Update juzi_keep_similarity
    # juzi_keep_similarity is length n_total — True only where juzi_keep is
    # True and the factor passes drop_zeros. Factors not in sim_idx are
    # always False since they were already excluded by pruning.

    keep_sim = np.zeros(n_total, dtype=bool)

    if drop_zeros:
        local_pass = ~np.isclose(sim, 0).all(axis=1)
        keep_sim[sim_idx[local_pass]] = True
    else:
        keep_sim[sim_idx] = True

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

    The drop_zeros mask from juzi.gp.similarity is preserved — factors
    already excluded by drop_zeros are not restored by select_similarity.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_similarity in .uns, produced by
        juzi.gp.similarity.
    min_similarity : float
        Minimum similarity threshold. Must be in [0, 1].
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields updated:
            .uns["juzi_keep_similarity"] : updated boolean mask (length n_total)
            .uns["juzi_keep"]            : intersection of all three stage masks
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field in ["juzi_similarity", "juzi_similarity_idx"]:
        if field not in adata.uns:
            raise KeyError(
                f"'{field}' not found in .uns. " "Run juzi.gp.similarity first."
            )

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be in [0, 1].")

    # Apply threshold

    sim = adata.uns["juzi_similarity"]  # (n_kept × n_kept)
    sim_idx = adata.uns["juzi_similarity_idx"]  # global indices
    n_total = adata.varm["juzi_G"].shape[1]

    # Start from drop_zeros result — factors not in sim_idx are already False
    keep_sim = adata.uns.get(
        "juzi_keep_similarity",
        np.zeros(n_total, dtype=bool),
    ).copy()

    # Compute local pass mask from max similarity per row
    local_max = sim.max(axis=1)  # (n_kept,)
    local_pass = local_max >= min_similarity  # (n_kept,)

    # Reset sim_idx entries then re-apply both drop_zeros and min_similarity
    # so re-running with a stricter threshold correctly removes factors
    # that were kept by a looser threshold
    keep_sim[sim_idx] = False
    keep_sim[sim_idx[local_pass]] = True

    # Preserve drop_zeros — factors excluded by drop_zeros must stay excluded
    # by intersecting with the original drop_zeros mask
    if "juzi_keep_similarity" in adata.uns:
        original_drop_zeros = np.zeros(n_total, dtype=bool)
        # Reconstruct drop_zeros mask: factors in sim_idx that had non-zero rows
        local_nonzero = ~np.isclose(sim, 0).all(axis=1)
        original_drop_zeros[sim_idx[local_nonzero]] = True
        keep_sim = keep_sim & original_drop_zeros

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
    """Compute similarity between two factor loading vectors."""
    x, y = G[i], G[j]

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
