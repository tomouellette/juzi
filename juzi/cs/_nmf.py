# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np
import scipy as sp

from anndata import AnnData
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from typing import List, Tuple
from tqdm import tqdm


def nmf(
    adata: AnnData,
    key: str,
    layer: str | None = None,
    genes: str | List[str] | np.ndarray | None = "highly_variable",
    min_cells: int = 5,
    k: List[int] | Tuple[int] | np.ndarray = [7, 8, 9, 10],
    target_sum: float = 1e5,
    normalize: bool = True,
    log1p: bool = True,
    center: bool = True,
    std: bool = False,
    clip_quantile: float = 0.999,
    max_iter: int = 1000,
    alpha: float = 0.,
    tol: float = 0.00005,
    loss: str = "frobenius",
    init: str = "nndsvda",
    l1_ratio: float = 0.,
    n_jobs: int = 1,
    prefer: str | None = None,
    keep_C: bool = False,
    seed: int = 123,
    silent: bool = False,
) -> AnnData:
    """Fit an NMF on each sample individually.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    key : str
        Key denoting sample column in .obs
    layer : str | None
        Optionally provide a layer to extract counts.
    genes : str | List[str] | None
        If a string is provided, then it must specify a boolean column in .var
        that indicates which genes should be retained for analysis.containing.
        Alternatively, a list of gene names can be provided for filtering. If
        no column name or gene list is provided, all present genes are used.
    min_cells : int
        Only perform NMF on samples with sufficient number of cells.
    target_sum : float
        Total counts per cell after normalization.
    log1p : bool
        If True, then normalized counts will be log1p transformed.
    center : bool
        If True, mean center each gene's expression.
    std : bool
        If True, scale each gene's expression to unit variance.
    clip_quantile : float
        Clip the data to the provided quantile.
    max_iter : int
        Maximum number of NMF iterations before timing out (see sklearn NMF).
    alpha : float
        Constant that multiplies regularization terms of W (see sklearn NMF).
    tol : float
        Tolerance of the stopping condition (see sklearn NMF).
    loss : str
        Beta divergence to be minimized (see sklearn NMF).
    init : str
        Method used to initialize the procedure (see sklearn NMF).
    l1_ratio : float
        Regularization mixing parameter (see sklearn NMF).
    n_jobs : str
        Number of parallel processes/threads to run.
    prefer : str
        Joblib preference (e.g. "threads").
    keep_C : bool
        If True, the cell factor loadings are retained.
    seed : int
        Random seed for reproducibility.
    silent : bool
        If True, suppress output messages and progress bar.

    Returns
    -------
    AnnData
        Anndata object with the following results added:
            - .uns['juzi_names']: Array the originating sample for each factor.
            - .uns['juzi_k']: List of k factor levels juzi.nmf was run with.
            - .varm['juzi_G']: Gene factor loading stacked across all samples.
            - .varm['juzi_C']: Optional factor cell loadings for each cell.
    """
    if key not in adata.obs:
        raise KeyError(f"'{key}' not a valid column in adata.obs")

    if not isinstance(k, (list, np.ndarray, tuple)):
        raise TypeError("k must be a list, tuple, or array of integers")

    gene_mask = np.ones(adata.shape[1], dtype=bool)
    if isinstance(genes, str):
        if genes not in adata.var:
            raise KeyError(f"'{genes}' was not found in .var")

        gene_mask = np.array(adata.var[genes])
    elif isinstance(genes, (list, np.ndarray)):
        gene_mask = adata.var_names.isin(genes)

    if np.sum(gene_mask) == 0:
        raise Exception(
            "No provided 'genes' were detected. " +
            "Check your provided .var column or gene list.")

    adata = adata[:, gene_mask]

    keep_obs = adata.obs.value_counts(key)
    keep_obs = keep_obs[keep_obs >= min_cells].index.to_numpy()
    adata = adata[np.isin(adata.obs[key], keep_obs)].copy()

    if isinstance(layer, str):
        if layer not in adata.layers:
            raise KeyError(f"'{layer}' is not a valid layer key")

        X = adata.layers[layer]
    else:
        X = adata.X

    idx = [np.where(adata.obs[key] == i)[0] for i in np.unique(adata.obs[key])]
    idx = [i for i in idx if len(i) >= min_cells]

    if len(idx) == 0:
        raise ValueError(f"No samples passed min_cells cutoff of {min_cells}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning
        )

        results = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(_nmf)(
                X=X[i, :].copy(),
                k=k,
                target_sum=target_sum,
                normalize=normalize,
                log1p=log1p,
                center=center,
                std=std,
                clip_quantile=clip_quantile,
                max_iter=max_iter,
                alpha=alpha,
                tol=tol,
                loss=loss,
                init=init,
                l1_ratio=l1_ratio,
                keep_C=keep_C,
                seed=seed,
            ) for i in tqdm(idx, desc="juzi | Fitting", disable=silent)
        )

    n_cells = np.sum([len(i) for i in idx])
    n_comps = np.sum(k)

    adata.uns["juzi_k"] = list(k)
    adata.uns["juzi_names"] = np.repeat(
        np.unique(adata.obs[key]), n_comps).tolist()

    if keep_C:
        adata.obsm["juzi_C"] = np.zeros((n_cells, n_comps))
        for i, ci in zip(idx, [i[1] for i in results]):
            adata.obsm["juzi_C"][i, :] = ci

        adata.varm["juzi_G"] = np.vstack([i[0] for i in results]).T
    else:
        adata.varm["juzi_G"] = np.vstack(results).T

    return adata


def _nmf(
    X: np.ndarray | csr_matrix,
    k: List[int] | Tuple[int] | np.ndarray = [7, 8, 9, 10],
    target_sum: float = 1e5,
    normalize: bool = True,
    log1p: bool = True,
    center: bool = True,
    std: bool = False,
    clip_quantile: float = 0.999,
    max_iter: int = 1000,
    alpha: float = 0.,
    tol: float = 0.00005,
    loss: str = "frobenius",
    init: str = "nndsvda",
    l1_ratio: float = 0.,
    keep_C: bool = False,
    seed: int = 123
) -> np.ndarray:
    if normalize:
        if isinstance(X, csr_matrix):
            X = sp.sparse.diags(target_sum / X.sum(axis=1).A.ravel()) @ X
            X = X.toarray()
        elif isinstance(X, np.ndarray):
            X[X < 0] = 0
            X = target_sum * (X / X.sum(axis=1)[:, np.newaxis])

    if log1p:
        X = np.log1p(X)

    if center:
        X -= X.mean(axis=0)

    if std:
        std = X.std(axis=0)
        mask = std == 0
        std[mask] = 1
        X = X / std

    X[X < 0] = 0
    q = np.quantile(X, clip_quantile)
    X[X > q] = q

    _H, _W = [], []
    for n_components in k:
        model = NMF(
            n_components=n_components,
            init=init,
            alpha_W=alpha,
            tol=tol,
            max_iter=max_iter,
            l1_ratio=l1_ratio,
            beta_loss=loss,
            random_state=seed,
        ).fit(X)

        if keep_C:
            _W.append(model.transform(X))

        _H.append(model.components_)

    if keep_C:
        return np.vstack(_H), np.hstack(_W)

    return np.vstack(_H)
