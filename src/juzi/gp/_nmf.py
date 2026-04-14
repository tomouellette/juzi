# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np
import scipy.sparse as sp

from anndata import AnnData
from joblib import Parallel, delayed
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from typing import List, Tuple, Union
from tqdm import tqdm


def nmf(
    adata: AnnData,
    key: str,
    layer: str | None = None,
    genes: str | List[str] | np.ndarray | None = "highly_variable",
    genes_force: List[str] | np.ndarray | None = None,
    gene_names_col: str | None = None,
    min_cells: int = 5,
    k: List[int] | Tuple[int] | np.ndarray = [7, 8, 9, 10],
    target_sum: float = 1e4,
    normalize: bool = True,
    log1p: bool = True,
    clip_quantile: float = 0.999,
    solver: str = "cd",
    loss: str = "frobenius",
    init: str = "nndsvda",
    alpha: float = 0.0,
    l1_ratio: float = 0.0,
    tol: float = 1e-5,
    max_iter: int = 10000,
    keep_scores: bool = False,
    n_jobs: int = 1,
    prefer: str | None = None,
    seed: int = 123,
    silent: bool = False,
) -> AnnData:
    """Fit NMF independently on each sample to extract gene programs.

    For each sample (e.g. donor), cells are normalised and factorised at
    multiple resolutions k. Gene loadings are stacked across all samples
    into a single matrix for downstream consensus program detection.

    Note: nmf always returns a new AnnData object subset to the selected
    genes and samples that passed min_cells filtering. The input adata is
    never modified in place.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Must contain a column in .obs matching key.
    key : str
        Column in .obs denoting sample identity (e.g. "donor_id").
    layer : str | None
        Layer containing raw counts. If None, uses .X.
    genes : str | List[str] | np.ndarray | None
        Gene selection. If str, must be a boolean column in .var. If list
        or array, must be gene names. If None, all genes are used.
    genes_force : List[str] | np.ndarray | None
        Gene names to always include regardless of the genes argument.
        Genes not present in adata.var_names are warned about and ignored.
    gene_names_col : str | None
        Column in .var containing gene names used for alignment in
        juzi.gp.score. If None, var_names is used. Must be consistent
        with the gene_names_col argument passed to juzi.gp.score.
    min_cells : int
        Minimum number of cells required per sample to fit NMF.
    k : List[int] | Tuple[int] | np.ndarray
        List of factorisation ranks to fit per sample. Multiple values
        improve program detection stability in downstream pruning.
    target_sum : float
        Total counts per cell after library size normalisation.
    normalize : bool
        If True, normalise counts to target_sum before log transformation.
    log1p : bool
        If True, apply log1p transformation after normalisation.
    clip_quantile : float
        Upper quantile at which to clip expression values after
        preprocessing. Reduces the influence of extreme outlier cells.
    solver : str
        NMF solver. "cd" (coordinate descent) is stable and recommended
        for Frobenius loss. "mu" (multiplicative update) is required for
        Kullback-Leibler loss and better motivated for count-like data.
    loss : str
        Beta divergence to minimise. "frobenius" for squared error,
        "kullback-leibler" for KL divergence (requires solver="mu").
    init : str
        Initialisation method. "nndsvda" is recommended for sparse data
        and gives deterministic initialisation independent of seed.
    alpha : float
        Regularisation strength applied to W (gene loadings).
    l1_ratio : float
        Mixing parameter between L1 and L2 regularisation (0 = L2 only,
        1 = L1 only).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations per fit.
    keep_scores : bool
        If True, per-cell factor scores are stored in obsm["juzi_scores"].
    n_jobs : int
        Number of parallel workers for fitting across samples.
    prefer : str | None
        Joblib parallelisation backend preference (e.g. "threads").
    seed : int
        Random seed. Only affects initialisations other than "nndsvda".
    silent : bool
        If True, suppress progress bar.

    Returns
    -------
    AnnData
        New AnnData object subset to selected genes and passing samples,
        with the following fields populated:
            .uns["juzi_k"]               : list of k values used
            .uns["juzi_names"]           : sample identity per factor column
            .uns["juzi_G_genes"]         : gene names for juzi_G rows
            .uns["juzi_keep_prune"]      : boolean mask, all True (set by prune)
            .uns["juzi_keep_similarity"] : boolean mask, all True (set by similarity)
            .uns["juzi_keep_cluster"]    : boolean mask, all True (set by cluster)
            .uns["juzi_keep"]            : intersection of all three masks
            .varm["juzi_G"]              : gene × factors loading matrix
            .obsm["juzi_scores"]         : cell × factors score matrix (if keep_scores)
    """
    # Validate

    if key not in adata.obs:
        raise KeyError(f"'{key}' not found in adata.obs.")

    if not isinstance(k, (list, np.ndarray, tuple)):
        raise TypeError("k must be a list, tuple, or array of integers.")

    if solver == "cd" and loss != "frobenius":
        raise ValueError(
            "solver='cd' only supports loss='frobenius'. "
            "Use solver='mu' for other loss functions."
        )

    # Gene selection

    if isinstance(genes, str):
        if genes not in adata.var:
            raise KeyError(f"'{genes}' not found in adata.var.")
        gene_mask = np.array(adata.var[genes], dtype=bool)
    elif isinstance(genes, (list, np.ndarray)):
        gene_mask = adata.var_names.isin(genes)
    else:
        gene_mask = np.ones(adata.n_vars, dtype=bool)

    # Force include genes

    if genes_force is not None:
        genes_force = np.asarray(genes_force)
        missing     = genes_force[~np.isin(genes_force, adata.var_names)]
        if len(missing) > 0:
            warnings.warn(
                f"{len(missing)} gene(s) in genes_force not found in "
                f"adata.var_names and will be ignored: {missing.tolist()}",
                UserWarning,
                stacklevel=2,
            )
        force_mask = adata.var_names.isin(genes_force)
        n_added    = (force_mask & ~gene_mask).sum()
        gene_mask  = gene_mask | force_mask

        if n_added > 0 and not silent:
            warnings.warn(
                f"{n_added} gene(s) from genes_force added to selection.",
                UserWarning,
                stacklevel=2,
            )

    if gene_mask.sum() == 0:
        raise ValueError(
            "No genes passed the selection filter. "
            "Check your 'genes' argument or .var column."
        )

    if gene_names_col is not None and gene_names_col not in adata.var:
        raise KeyError(
            f"'{gene_names_col}' not found in adata.var. "
            "Check your gene_names_col argument."
        )

    adata = adata[:, gene_mask].copy()

    # Sample filtering

    cell_counts   = adata.obs[key].value_counts()
    valid_samples = cell_counts[cell_counts >= min_cells].index.to_numpy()

    if len(valid_samples) == 0:
        raise ValueError(
            f"No samples passed min_cells={min_cells}. "
            "Check your 'key' column or lower min_cells."
        )

    adata = adata[adata.obs[key].isin(valid_samples)].copy()

    # Extract expression matrix

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        X = adata.layers[layer]
    else:
        X = adata.X

    # Build per-sample cell index arrays

    sample_order = np.array(sorted(valid_samples))
    idx          = [np.where(adata.obs[key].values == s)[0] for s in sample_order]

    # Parallel NMF fits

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        results = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(_nmf)(
                X=X[i, :].copy(),
                k=k,
                target_sum=target_sum,
                normalize=normalize,
                log1p=log1p,
                clip_quantile=clip_quantile,
                solver=solver,
                loss=loss,
                init=init,
                alpha=alpha,
                l1_ratio=l1_ratio,
                tol=tol,
                max_iter=max_iter,
                keep_scores=keep_scores,
                seed=seed,
            )
            for i in tqdm(idx, desc="[juzi] Fitting NMF", disable=silent)
        )

    # Store results

    n_comps  = int(np.sum(k))
    n_factors = len(sample_order) * n_comps

    adata.uns["juzi_k"]      = list(k)
    adata.uns["juzi_names"]  = np.repeat(sample_order, n_comps).tolist()
    adata.uns["juzi_G_genes"] = (
        adata.var[gene_names_col].tolist()
        if gene_names_col is not None
        else adata.var_names.tolist()
    )

    if keep_scores:
        H_list, W_list = zip(*results)
        adata.varm["juzi_G"] = np.vstack(H_list).T

        scores = np.full((adata.n_obs, n_comps), np.nan)
        for i, W in zip(idx, W_list):
            scores[i, :] = W
        adata.obsm["juzi_scores"] = scores

    else:
        adata.varm["juzi_G"] = np.vstack(results).T

    # Initialise stage keep masks

    adata.uns["juzi_keep_prune"]      = np.ones(n_factors, dtype=bool)
    adata.uns["juzi_keep_similarity"] = np.ones(n_factors, dtype=bool)
    adata.uns["juzi_keep_cluster"]    = np.ones(n_factors, dtype=bool)
    adata.uns["juzi_keep"]            = np.ones(n_factors, dtype=bool)

    return adata


def _recompute_keep(adata: AnnData) -> None:
    """Recompute juzi_keep as the intersection of all three stage masks.

    If any stage mask is absent it is treated as all True — this allows
    functions to be run independently without requiring all upstream steps.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_G in .varm.
    """
    n = adata.varm["juzi_G"].shape[1]
    for key in ["juzi_keep_prune", "juzi_keep_similarity", "juzi_keep_cluster"]:
        if key not in adata.uns:
            adata.uns[key] = np.ones(n, dtype=bool)

    adata.uns["juzi_keep"] = (
        adata.uns["juzi_keep_prune"]      &
        adata.uns["juzi_keep_similarity"] &
        adata.uns["juzi_keep_cluster"]
    )


def _nmf(
    X: Union[np.ndarray, sp.csr_matrix],
    k: List[int] | Tuple[int] | np.ndarray,
    target_sum: float,
    normalize: bool,
    log1p: bool,
    clip_quantile: float,
    solver: str,
    loss: str,
    init: str,
    alpha: float,
    l1_ratio: float,
    tol: float,
    max_iter: int,
    keep_scores: bool,
    seed: int,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Fit NMF at multiple resolutions on a single sample."""

    if sp.issparse(X):
        X = X.toarray()

    X = X.astype(np.float64)
    X = np.clip(X, 0, None)

    if normalize:
        row_sums            = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X                   = target_sum * X / row_sums

    if log1p:
        X = np.log1p(X)

    q = np.quantile(X, clip_quantile)
    X = np.clip(X, 0, q)

    H_list, W_list = [], []

    for n_components in k:
        model = NMF(
            n_components=n_components,
            init=init,
            solver=solver,
            beta_loss=loss,
            alpha_W=alpha,
            l1_ratio=l1_ratio,
            tol=tol,
            max_iter=max_iter,
            random_state=seed,
        )
        W = model.fit_transform(X)
        H_list.append(model.components_)

        if keep_scores:
            W_list.append(W)

    H = np.vstack(H_list)

    if keep_scores:
        return H, np.hstack(W_list)

    return H
