# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np

from anndata import AnnData
from joblib import Parallel, delayed
from typing import Callable

from tqdm import tqdm

from ._nmf import _recompute_keep, _combined_score


def similarity(
    adata: AnnData,
    metric: str | Callable = "jaccard",
    top_k: int | None = 50,
    intra_sample: bool = True,
    drop_zeros: bool = True,
    use_combined: bool = False,
    n_jobs: int = 1,
    prefer: str | None = None,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Compute pairwise similarity between NMF factors across samples.

    Builds a symmetric factor × factor similarity matrix using either
    Jaccard similarity on top-ranked gene sets or a user-provided metric.
    Only factors currently retained by juzi_keep enter the computation.
    The resulting matrix has shape (n_kept × n_kept).

    Parameters
    ----------
    adata : AnnData
        AnnData object fit with juzi.gp.nmf_fit and optionally juzi.gp.nmf_prune.
    metric : str | Callable
        Similarity metric. "jaccard" computes overlap of top-k gene sets.
        A callable must accept two 1-d float arrays and return a scalar.
    top_k : int | None
        Number of top genes used per factor. Required for metric="jaccard".
        When metric is callable and top_k is not None, the callable is applied
        only to the union of the top-k genes from the two factors.
    intra_sample : bool
        If True, compute within-sample similarities. If False, only
        inter-sample similarities are computed.
    drop_zeros : bool
        If True, factors whose entire similarity row is zero are removed
        from juzi_keep_similarity.
    use_combined : bool
        If True, rank genes by combined loading × specificity score before
        selecting top_k genes. If False, rank by raw loading magnitude.
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
            .uns["juzi_similarity"]       : (n_kept × n_kept) similarity matrix
            .uns["juzi_similarity_idx"]   : global factor indices
            .uns["juzi_keep_similarity"]  : boolean mask length n_total
            .uns["juzi_similarity_meta"]  : similarity parameter metadata
            .uns["juzi_similarity_scope"] : "all" or "inter"
            .uns["juzi_keep"]             : intersection of all three stage masks
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_G", "varm"),
        ("juzi_k", "uns"),
        ("juzi_names", "uns"),
        ("juzi_G_genes", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. Run juzi.gp.nmf_fit first."
            )

    if metric != "jaccard" and not callable(metric):
        raise ValueError("metric must be 'jaccard' or a callable.")

    if metric == "jaccard" and top_k is None:
        raise ValueError("top_k must be set when using metric='jaccard'.")

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if top_k is not None and top_k > adata.n_vars:
        raise ValueError(f"top_k={top_k} exceeds number of genes ({adata.n_vars}).")

    if callable(metric):
        _validate_metric_fn(metric)

    # Subset to currently kept factors

    n_total = adata.varm["juzi_G"].shape[1]
    keep = adata.uns.get("juzi_keep", np.ones(n_total, dtype=bool))
    sim_idx = np.where(keep)[0]

    G_all = adata.varm["juzi_G"].T  # (n_total × n_genes)
    G = G_all[sim_idx]  # (n_kept × n_genes)

    names_all = np.array(adata.uns["juzi_names"], dtype=object)
    names = names_all[sim_idx]

    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)
    n = G.shape[0]

    # Compute ranking scores for top-gene extraction

    G_score = _combined_score(G) if use_combined else G

    # Precompute top gene sets for Jaccard and callable subsetting

    top_gene_sets = None
    if top_k is not None:
        top_gene_sets = [
            set(gene_names[np.argsort(row)[-int(top_k) :]].tolist()) for row in G_score
        ]

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
            gene_names=gene_names,
            top_gene_sets=top_gene_sets,
            top_k=top_k,
            metric=metric,
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
    adata.uns["juzi_similarity_scope"] = "all" if intra_sample else "inter"
    adata.uns["juzi_similarity_intra"] = intra_sample  # backwards compatibility
    adata.uns["juzi_similarity_meta"] = {
        "metric": metric if isinstance(metric, str) else "callable",
        "top_k": top_k,
        "intra_sample": intra_sample,
        "drop_zeros": drop_zeros,
        "use_combined": use_combined,
    }

    _recompute_keep(adata)

    return adata if copy else None


def similarity_compute(
    adata: AnnData,
    distance: str | Callable = "jaccard",
    top_k: int | None = 50,
    intra_sample: bool = True,
    drop_zeros: bool = True,
    use_combined: bool = False,
    n_jobs: int = 1,
    prefer: str | None = None,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Backward-compatible wrapper for similarity().

    Parameters
    ----------
    distance : str | Callable
        Deprecated alias for metric.

    Returns
    -------
    AnnData | None
        Same as similarity().
    """
    return similarity(
        adata=adata,
        metric=distance,
        top_k=top_k,
        intra_sample=intra_sample,
        drop_zeros=drop_zeros,
        use_combined=use_combined,
        n_jobs=n_jobs,
        prefer=prefer,
        silent=silent,
        copy=copy,
    )


def similarity_filter(
    adata: AnnData,
    min_similarity: float,
    copy: bool = False,
) -> AnnData | None:
    """Filter factors by minimum similarity threshold.

    Updates juzi_keep_similarity to flag factors whose maximum similarity
    to any other factor falls below min_similarity. Can be re-run with
    different thresholds without re-running juzi.gp.similarity. The
    original drop_zeros behavior is preserved across re-runs.

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
                f"'{field}' not found in .uns. Run juzi.gp.similarity first."
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

    # Preserve original drop_zeros behavior if present
    if "juzi_keep_similarity" in adata.uns:
        local_nonzero = ~np.isclose(sim, 0).all(axis=1)
        original_drop_zeros = np.zeros(n_total, dtype=bool)
        original_drop_zeros[sim_idx[local_nonzero]] = True
        keep_sim = keep_sim & original_drop_zeros

    adata.uns["juzi_keep_similarity"] = keep_sim
    _recompute_keep(adata)

    return adata if copy else None


def _validate_metric_fn(metric: Callable) -> None:
    """Validate that a callable metric accepts two 1-d arrays and returns a scalar."""
    x = np.random.default_rng(0).random(4)
    y = np.random.default_rng(1).random(4)

    try:
        result = metric(x, y)
    except Exception as exc:
        raise ValueError("metric callable must accept two 1-d float arrays.") from exc

    if not isinstance(result, (int, float, np.floating)):
        raise ValueError("metric callable must return a scalar value.")


def _compute_similarity(
    i: int,
    j: int,
    G: np.ndarray,
    gene_names: np.ndarray,
    top_gene_sets: list[set[str]] | None,
    top_k: int | None,
    metric: str | Callable,
) -> tuple[int, int, float]:
    """Compute similarity between two factor loading vectors."""
    x = G[i]
    y = G[j]

    if np.sum(x) == 0 or np.sum(y) == 0:
        return (i, j, 0.0)

    if metric == "jaccard":
        assert top_gene_sets is not None
        top_x = top_gene_sets[i]
        top_y = top_gene_sets[j]

        union = top_x | top_y
        if len(union) == 0:
            return (i, j, 0.0)

        s_xy = len(top_x & top_y) / len(union)

    else:
        if top_k is not None:
            assert top_gene_sets is not None
            union_genes = top_gene_sets[i] | top_gene_sets[j]
            if len(union_genes) == 0:
                return (i, j, 0.0)

            union_mask = np.isin(gene_names, list(union_genes))
            x_use = x[union_mask]
            y_use = y[union_mask]
        else:
            x_use = x
            y_use = y

        s_xy = float(metric(x_use, y_use))

    if np.isnan(s_xy):
        return (i, j, 0.0)

    return (i, j, float(s_xy))
