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
    copy: bool = True
) -> AnnData | None:
    """Compute similarity between gene loadings computed across samples.

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.cs.nmf.
    distance : str or callable
        'jaccard' or a custom callable that takes two 1-d arrays of equal length.
    top_k : Optional[int]
        Compute the distance function on the union of the top K loadings.
    drop_zeros : bool
        Drop rows/columns in similarity matrix where the sum is zero.
    min_similarity : float
        Drop rows/columns that have maximum similarity lower than provided value.
    intra_sample : bool
        If True, compute similarity scores between factors from same sample.
    n_jobs : str
        Number of parallel processes/threads to run.
    prefer : str
        Joblib preference (e.g. "threads").
    silent : bool
        If True, disable progress bar.
    copy : bool
        If True, a copy of the anndata is returned.

    Returns
    -------
    AnnData | None
        Copied AnnData or AnnData modified in-place
    """
    if "juzi_G" not in adata.varm or "juzi_names" not in adata.uns:
        raise KeyError("Please run juzi.cs.nmf before computing similarity.")

    if distance != "jaccard" and not callable(distance):
        raise ValueError("distance must be 'jaccard' or callable")

    if distance == "jaccard" and top_k is None:
        raise ValueError("top_k must be set when using Jaccard similarity.")

    if min_similarity < 0. or min_similarity > 1.:
        raise ValueError("'min_similarity' must be in [0, 1]")

    if callable(distance):
        x, y = np.random.rand(4), np.random.rand(4)
        try:
            d = distance(x, y)
            if not isinstance(d, (int, float)):
                raise ValueError(
                    "'distance' must be a callable that returns a scalar.")
        except:
            raise ValueError(
                "'distance' must be callable that accepts two arrays.")

    juzi_G = adata.varm["juzi_G"].T

    indices = []
    for i in range(juzi_G.shape[0]):
        for j in range(i + 1, juzi_G.shape[0]):
            if np.all([
                adata.uns["juzi_names"][i] == adata.uns["juzi_names"][j],
                not intra_sample
            ]):
                continue

            indices.append((i, j))

    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_similarity)(
            i=i,
            j=j,
            G=juzi_G,
            top_k=top_k,
            distance=distance
        )
        for i, j in tqdm(
            indices,
            desc="juzi | Computing similarity",
            disable=silent
        )
    )

    similarity = np.zeros((juzi_G.shape[0], juzi_G.shape[0]))
    for (i, j, s_xy) in results:
        similarity[i, j] = s_xy
        similarity[j, i] = s_xy

    if drop_zeros:
        if "juzi_keep" in adata.uns:
            adata.uns["juzi_keep"][np.all(
                np.isclose(similarity, 0), axis=1)] = False
        else:
            adata.uns["juzi_keep"] = ~np.all(np.isclose(similarity, 0), axis=1)

    if min_similarity > 0.:
        mask = np.max(similarity, axis=0) < min_similarity
        adata.uns["juzi_keep"][mask] = False

    adata.uns["juzi_similarity"] = similarity

    return adata if copy else None


def _similarity(
    i: int,
    j: int,
    G: np.ndarray,
    top_k: int,
    distance: str | Callable,
) -> float:
    x = G[i]
    y = G[j]

    if np.sum(x) == 0 or np.sum(y) == 0:
        return (i, j, 0.0)

    if top_k is not None:
        top_x = np.argsort(x)[-int(top_k):]
        top_y = np.argsort(y)[-int(top_k):]
        union = np.union1d(top_x, top_y)

        if len(union) == 0:
            return (i, j, 0.0)

        x = x[union]
        y = y[union]

    if distance == "jaccard":
        s_xy = len(np.intersect1d(top_x, top_y)) / len(union)
    elif callable(distance):
        s_xy = distance(x, y)

    if np.isnan(s_xy):
        return (i, j, 0.0)

    return (i, j, s_xy)
