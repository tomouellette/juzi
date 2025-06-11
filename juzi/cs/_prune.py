# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np

from anndata import AnnData
from joblib import Parallel, delayed
from typing import List
from tqdm import tqdm


def prune(
    adata: AnnData,
    top_k: int = 50,
    min_similarity: float = 0.7,
    min_k: int = 1,
    n_jobs: int = 1,
    prefer: str | None = None,
    copy: bool = False,
    silent: bool = False,
) -> AnnData | None:
    """Prunes non-recurrent intra-sample factors based on overlapping top genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object with 'juzi_G' matrix stored in .varm.
    top_k : int
        Number of top genes to consider for Jaccard similarity.
    threshold : float
        Similarity threshold for identifying recurrent factors.
    similarity_metric : str
        Similarity metric to use: "jaccard" or "cosine".
    min_k : int
        Minimum number of K values a factor must appear in to be recurrent.
    n_jobs : str
        Number of parallel processes/threads to run.
    prefer : str
        Joblib preference (e.g. "threads").
    copy : bool
        If True, a copy of the anndata is returned.
    silent : bool
        If True, suppress output messages and progress bar.

    Returns
    -------
    np.ndarray
        Retained factors after pruning redundant factors.
    """
    if "juzi_G" not in adata.varm:
        raise KeyError("'juzi_G' not found in .varm")

    if "juzi_names" not in adata.uns:
        raise KeyError("'juzi_names' not found in .uns")

    if "juzi_k" not in adata.uns:
        raise KeyError("'juzi_k' not found in .uns")

    if top_k > adata.varm["juzi_G"].shape[0]:
        raise ValueError("'top_k' cannot be greater than number of genes")

    if min_similarity < 0. or min_similarity > 1.:
        raise ValueError("'min_similarity' must be in [0, 1]")

    nk = len(adata.uns["juzi_k"])
    if min_k > nk:
        raise ValueError(
            "'min_k' cannot be greater than number of factor " +
            f"levels (len(k) = {nk}")

    factors = np.split(
        adata.varm["juzi_G"].T,
        len(np.unique(adata.uns["juzi_names"]))
    )

    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_prune)(
            factors=f,
            k=adata.uns["juzi_k"],
            top_k=top_k,
            min_similarity=min_similarity,
            min_k=min_k
        ) for f in tqdm(
            factors,
            desc="juzi | Pruning recurrent factors",
            disable=silent
        ))

    k = sum(adata.uns["juzi_k"])

    mask = np.zeros(adata.varm["juzi_G"].shape[1])
    for i, idx_ in enumerate(results):
        if len(idx_) > 0:
            mask[np.array([i * k + j for j in idx_])] = True

    adata.uns["juzi_keep"] = mask

    return adata if copy else None


def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)


def _prune(
    factors: np.ndarray,
    k: List[int],
    top_k: int,
    min_similarity: float,
    min_k: int
) -> np.ndarray:
    keep_idx = []
    used_factors = set()

    cumsum_k = np.cumsum(k)[:-1]
    factors_split = np.split(factors, cumsum_k)

    for k_idx, factor_k in enumerate(factors_split):
        for i, factor_i in enumerate(factor_k):
            if (k_idx, i) in used_factors:
                continue

            top_genes_i = set(np.argsort(factor_i)[-top_k:])

            recurrent_k_count = 0
            for other_k_idx, factor_other_k in enumerate(factors_split):
                if other_k_idx == k_idx:
                    continue

                for j, factor_j in enumerate(factor_other_k):
                    if (other_k_idx, j) in used_factors:
                        continue

                    top_genes_j = set(np.argsort(factor_j)[-top_k:])
                    similarity = jaccard_similarity(top_genes_i, top_genes_j)

                    if similarity >= min_similarity:
                        recurrent_k_count += 1
                        used_factors.add((other_k_idx, j))
                        break

            if recurrent_k_count >= min_k - 1:
                keep_idx.append((k_idx, i))
                used_factors.add((k_idx, i))

    cumulative_offsets = np.concatenate(([0], np.cumsum(k)))

    idx = []
    for i, j in keep_idx:
        idx.append(cumulative_offsets[i] + j)

    return np.array(idx)
