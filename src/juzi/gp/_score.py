# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np

from anndata import AnnData
from typing import List, Dict
from tqdm import tqdm
import scipy.stats as stats
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from ._nmf import _combined_score


def score_cells(
    adata: AnnData,
    n_top_genes: int = 50,
    use_combined: bool = False,
    n_control_genes: int = 50,
    gene_pool: List[str] | None = None,
    gene_names_col: str | None = None,
    layer: str | None = None,
    seed: int = 123,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Score cells on each consensus gene program using a control-subtracted mean.

    For each consensus program, the top n_top_genes genes are selected by
    either raw loading magnitude or specificity (loading relative to other
    programs). Per-cell scores are computed as the mean expression of the
    program genes minus the mean expression of a matched set of control genes
    drawn from similar expression bins, following Tirosh et al. 2016.

    Scores are stored per program as columns in obsm["juzi_program_scores"]
    and are comparable across programs and donors since the control subtraction
    removes background expression level variation.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Should contain normalised log expression in .X
        or the specified layer. Must share gene names with the genes used
        during juzi.gp.nmf.
    n_top_genes : int
        Number of top genes per program used for scoring.
    use_combined : bool
        If True, rank genes by specificity score (loading in this program
        divided by total loading across all programs) rather than raw loading
        magnitude. Specificity-ranked genes are more discriminative between
        programs and produce cleaner scores.
    n_control_genes : int
        Number of control genes per program gene used for background
        subtraction. Control genes are drawn from the same expression bin
        as each program gene. Total control set size is up to
        n_top_genes × n_control_genes, deduplicated.
    gene_pool : List[str] | None
        Genes eligible to serve as control genes. Defaults to all genes
        in adata.var_names. Program genes are excluded from the control pool.
    gene_names_col : str | None
        Column in adata.var containing gene names. Must match the
        gene_names_col used in juzi.gp.nmf. If None, adata.var_names is used.
    layer : str | None
        Layer containing normalised log expression values. If None, uses .X.
    seed : int
        Random seed for control gene sampling.
    silent : bool
        If True, suppress progress bar.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields populated:
            .obsm["juzi_program_scores"] : (n_cells × n_programs) score matrix
            .uns["juzi_program_genes"]   : dict mapping program index to top genes
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_cluster_G", "uns"),
        ("juzi_cluster_labels", "uns"),
        ("juzi_G_genes", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. "
                "Run juzi.gp.nmf, juzi.gp.similarity, juzi.gp.cluster first."
            )

    if n_top_genes < 1:
        raise ValueError("n_top_genes must be >= 1.")

    if n_control_genes < 1:
        raise ValueError("n_control_genes must be >= 1.")

    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers.")

    if gene_names_col is not None and gene_names_col not in adata.var:
        raise KeyError(
            f"'{gene_names_col}' not found in adata.var. "
            "Check your gene_names_col argument."
        )

    # Extract gene names

    cell_genes = (
        adata.var[gene_names_col].to_numpy()
        if gene_names_col is not None
        else adata.var_names.to_numpy()
    )

    nmf_genes = np.array(adata.uns["juzi_G_genes"])

    # Align genes

    shared_genes = np.intersect1d(nmf_genes, cell_genes)
    n_shared = len(shared_genes)

    if n_shared == 0:
        raise ValueError(
            "No genes shared between adata gene names and juzi_G_genes. "
            "Ensure adata contains the same genes used in juzi.gp.nmf and "
            "that gene_names_col matches the value used in juzi.gp.nmf."
        )

    if n_shared < n_top_genes:
        raise ValueError(
            f"Only {n_shared} genes overlap between adata and the NMF gene "
            f"set, but n_top_genes={n_top_genes}. Lower n_top_genes or check "
            "that adata contains the genes used during juzi.gp.nmf."
        )

    if n_shared < len(nmf_genes):
        warnings.warn(
            f"{len(nmf_genes) - n_shared} genes from the NMF gene set are "
            "absent from adata and will be ignored during scoring.",
            UserWarning,
            stacklevel=2,
        )

    nmf_gene_to_idx = {g: i for i, g in enumerate(nmf_genes)}
    cell_gene_to_idx = {g: i for i, g in enumerate(cell_genes)}

    shared_nmf_idx = np.array([nmf_gene_to_idx[g] for g in shared_genes])
    shared_cell_idx = np.array([cell_gene_to_idx[g] for g in shared_genes])

    # H aligned to shared genes: (n_programs × n_shared_genes)
    H_full = adata.uns["juzi_cluster_G"]
    H_aligned = H_full[:, shared_nmf_idx]
    n_programs = H_aligned.shape[0]

    # Extract expression matrix

    X = adata.layers[layer] if layer is not None else adata.X

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.array(X, dtype=np.float32)

    if X.max() > 40:
        warnings.warn(
            "Expression values appear to be raw counts (max > 40). "
            "juzi.gp.score expects normalised log expression. "
            "Consider running sc.pp.normalize_total and sc.pp.log1p first.",
            UserWarning,
            stacklevel=2,
        )

    # Expression aligned to shared genes: (n_cells × n_shared_genes)
    X_aligned = X[:, shared_cell_idx]

    # Weight by gene specificity

    if use_combined:
        H_rank = _combined_score(H_aligned)
    else:
        H_rank = H_aligned

    # Control gene pool

    if gene_pool is not None:
        pool_mask = np.isin(cell_genes, gene_pool)
    else:
        pool_mask = np.ones(len(cell_genes), dtype=bool)

    # Expression bin assignment for all genes using mean expression
    gene_means = X.mean(axis=0)
    n_bins = 25
    bin_edges = np.percentile(gene_means, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6
    gene_bins = np.digitize(gene_means, bin_edges) - 1

    # Score each program

    rng = np.random.default_rng(seed)
    scores = np.zeros((adata.n_obs, n_programs), dtype=np.float32)
    program_genes = {}

    for p in tqdm(range(n_programs), desc="[juzi] Scoring", disable=silent):
        # Select top genes by rank (specificity or raw loading)
        gene_ranks = H_rank[p]
        top_local = np.argsort(gene_ranks)[-n_top_genes:]
        top_genes = shared_genes[top_local]
        top_cell_idx = np.array([cell_gene_to_idx[g] for g in top_genes])

        program_genes[p] = top_genes.tolist()

        # Program score — mean expression of top genes per cell
        program_score = X[:, top_cell_idx].mean(axis=1)

        # Control gene selection — bin-matched, excluding program genes
        top_cell_idx_set = set(top_cell_idx.tolist())
        control_idx = []

        for gene_idx in top_cell_idx:
            gene_bin = gene_bins[gene_idx]
            bin_members = np.where(
                (gene_bins == gene_bin)
                & pool_mask
                & ~np.isin(np.arange(len(cell_genes)), list(top_cell_idx_set))
            )[0]

            # Fall back to adjacent bins if current bin has no eligible genes
            if len(bin_members) == 0:
                for delta in [1, -1, 2, -2, 3, -3]:
                    adjacent = np.where(
                        (gene_bins == gene_bin + delta)
                        & pool_mask
                        & ~np.isin(np.arange(len(cell_genes)), list(top_cell_idx_set))
                    )[0]
                    if len(adjacent) > 0:
                        bin_members = adjacent
                        break

            if len(bin_members) > 0:
                n_draw = min(n_control_genes, len(bin_members))
                chosen = rng.choice(bin_members, size=n_draw, replace=False)
                control_idx.extend(chosen.tolist())

        control_idx = np.unique(control_idx)
        control_score = (
            X[:, control_idx].mean(axis=1)
            if len(control_idx) > 0
            else np.zeros(adata.n_obs, dtype=np.float32)
        )

        scores[:, p] = program_score - control_score

    adata.obsm["juzi_program_scores"] = scores
    adata.uns["juzi_program_genes"] = program_genes

    return adata if copy else None


def score_classify(
    adata: AnnData,
    n_shuffles: int = 20,
    n_cells_per_shuffle: int = 5000,
    padj_thresh: float = 0.05,
    n_jobs: int = 1,
    prefer: str = "threads",
    layer: str | None = None,
    gene_names_col: str | None = None,
    seed: int = 123,
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Classify cells into consensus programs using a permutation null.

    Generates null score distributions by shuffling expression values
    per gene across randomly subsampled cells, scoring the shuffled
    matrices on the real program gene sets, and fitting a normal
    distribution to the resulting null scores. Each cell is then
    classified via a z-test against the null, with BH correction
    for multiple testing across all program × cell pairs.

    Cells achieving padj < padj_thresh for exactly one program are
    assigned to that program. Cells achieving padj < padj_thresh for
    multiple programs are assigned to the program for which they scored
    maximally. Cells achieving padj < padj_thresh for no program are
    assigned "unresolved".

    Must be run after juzi.gp.score_cells since it reads
    juzi_program_scores and juzi_program_genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_program_scores and juzi_program_genes
        in .obsm and .uns respectively, produced by juzi.gp.score_cells.
    n_shuffles : int
        Number of shuffled expression matrices to generate. Each shuffle
        subsamples n_cells_per_shuffle cells and shuffles expression
        values per gene. Total null scores per program =
        n_shuffles × n_cells_per_shuffle.
    n_cells_per_shuffle : int
        Number of cells to subsample per shuffle. Does not need to equal
        the full dataset size — 5000 is sufficient to estimate null
        distribution parameters for most datasets.
    padj_thresh : float
        Adjusted p-value threshold for program assignment. Must be in
        (0, 1].
    n_jobs : int
        Number of parallel workers. Parallelisation is across shuffles.
    prefer : str
        Joblib backend preference. "threads" is recommended since the
        inner scoring loop is numpy-based with no GIL contention.
    layer : str | None
        Layer containing normalised log expression. If None, uses .X.
        Must match the layer used in juzi.gp.score_cells.
    gene_names_col : str | None
        Column in adata.var containing gene names. Must match the value
        used in juzi.gp.score_cells.
    seed : int
        Master random seed. Per-shuffle seeds are derived deterministically
        from this value.
    silent : bool
        If True, suppress progress bar.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields populated:
            .obsm["juzi_program_pvals"]  : (n_cells × n_programs) raw p-values
            .obsm["juzi_program_padj"]   : (n_cells × n_programs) BH-adjusted
            .obs["juzi_program_label"]   : per-cell program assignment or
                                           "unresolved"
            .uns["juzi_classify_params"] : dict storing n_shuffles,
                                           n_cells_per_shuffle, padj_thresh,
                                           null_mean, null_std per program
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_program_scores", "obsm"),
        ("juzi_program_genes", "uns"),
        ("juzi_cluster_labels", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. " "Run juzi.gp.score_cells first."
            )

    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1.")

    if n_cells_per_shuffle < 1:
        raise ValueError("n_cells_per_shuffle must be >= 1.")

    if not 0.0 < padj_thresh <= 1.0:
        raise ValueError("padj_thresh must be in (0, 1].")

    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers.")

    if gene_names_col is not None and gene_names_col not in adata.var:
        raise KeyError(f"'{gene_names_col}' not found in adata.var.")

    # Setup

    X = adata.layers[layer] if layer is not None else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    cell_genes = (
        adata.var[gene_names_col].to_numpy()
        if gene_names_col is not None
        else adata.var_names.to_numpy()
    )

    program_genes = adata.uns["juzi_program_genes"]  # dict: int -> List[str]
    n_programs = len(program_genes)
    n_cells = adata.n_obs

    # Map program genes to cell expression indices
    gene_to_idx = {g: i for i, g in enumerate(cell_genes)}
    prog_indices: List[np.ndarray] = []
    for p in range(n_programs):
        genes = program_genes[p]
        indices = np.array(
            [gene_to_idx[g] for g in genes if g in gene_to_idx], dtype=int
        )
        prog_indices.append(indices)

    if any(len(idx) == 0 for idx in prog_indices):
        warnings.warn(
            "Some programs have no genes matching adata gene names. "
            "Check that gene_names_col matches the value used in score_cells.",
            UserWarning,
            stacklevel=2,
        )

    # Per-shuffle seeds

    rng = np.random.default_rng(seed)
    shuffle_seeds = rng.integers(0, 2**31, size=n_shuffles)

    # Cap n_cells_per_shuffle at actual cell count
    n_sample = min(n_cells_per_shuffle, n_cells)

    # Run shuffles in parallel
    # Each shuffle returns a (n_programs,) array of mean scores across
    # the subsampled shuffled cells

    null_scores = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_score_shuffle)(
            X=X,
            prog_indices=prog_indices,
            n_cells=n_sample,
            seed=int(shuffle_seeds[i]),
        )
        for i in tqdm(
            range(n_shuffles),
            desc="[juzi] Classifying",
            disable=silent,
        )
    )

    # null_scores: list of n_shuffles arrays each (n_programs,)
    # Stack to (n_shuffles × n_programs)
    null_matrix = np.vstack(null_scores)  # (n_shuffles × n_programs)

    # Fit null distribution per program

    null_mean = null_matrix.mean(axis=0)  # (n_programs,)
    null_std = null_matrix.std(axis=0)  # (n_programs,)

    # Avoid division by zero for programs with zero-variance null
    null_std = np.where(null_std == 0, 1e-8, null_std)

    # Compute z-scores and p-values for real cells

    real_scores = adata.obsm["juzi_program_scores"]  # (n_cells × n_programs)

    # z-score each cell against the null for each program
    z_scores = (real_scores - null_mean[None, :]) / null_std[None, :]

    # One-sided p-value; probability of observing a score this high under null
    pvals = stats.norm.sf(z_scores)  # (n_cells × n_programs)

    # BH correction across all program by cell pairs
    # Flatten, correct, reshape

    flat_pvals = pvals.flatten()
    _, flat_padj, _, _ = multipletests(flat_pvals, method="fdr_bh")
    padj = flat_padj.reshape(n_cells, n_programs)

    # Cell assignment

    unique_C = np.unique(adata.uns["juzi_cluster_labels"])
    prog_names = [f"C{int(c)}" for c in unique_C]
    labels_out = np.full(n_cells, "unresolved", dtype=object)

    sig_mask = padj < padj_thresh  # (n_cells × n_programs)
    n_sig = sig_mask.sum(axis=1)  # (n_cells,)

    # Cells significant for exactly one program
    single_sig = n_sig == 1
    if single_sig.any():
        prog_idx = np.argmax(sig_mask[single_sig], axis=1)
        labels_out[single_sig] = [prog_names[p] for p in prog_idx]

    # Cells significant for multiple programs — assign to max score
    multi_sig = n_sig > 1
    if multi_sig.any():
        # Mask non-significant scores before taking argmax
        masked_scores = real_scores.copy()
        masked_scores[~sig_mask] = -np.inf
        prog_idx = np.argmax(masked_scores[multi_sig], axis=1)
        labels_out[multi_sig] = [prog_names[p] for p in prog_idx]

    # Store results

    adata.obsm["juzi_program_pvals"] = pvals.astype(np.float32)
    adata.obsm["juzi_program_padj"] = padj.astype(np.float32)
    adata.obs["juzi_program_label"] = labels_out

    adata.uns["juzi_classify_params"] = {
        "n_shuffles": n_shuffles,
        "n_cells_per_shuffle": n_cells_per_shuffle,
        "padj_thresh": padj_thresh,
        "null_mean": null_mean,
        "null_std": null_std,
    }

    return adata if copy else None


def _score_shuffle(
    X: np.ndarray,
    prog_indices: List[np.ndarray],
    n_cells: int,
    seed: int,
) -> np.ndarray:
    """Run one shuffle iteration.

    Subsamples n_cells cells, shuffles expression values per gene
    independently, and scores all programs on the shuffled matrix.

    Parameters
    ----------
    X : np.ndarray
        Full expression matrix (n_cells_total × n_genes), normalised log.
    prog_indices : List[np.ndarray]
        List of gene index arrays, one per program.
    n_cells : int
        Number of cells to subsample.
    seed : int
        Random seed for this shuffle.

    Returns
    -------
    np.ndarray
        Mean score per program across shuffled cells, shape (n_programs,).
    """
    rng = np.random.default_rng(seed)

    # Subsample cells
    cell_idx = rng.choice(X.shape[0], size=n_cells, replace=False)
    X_sub = X[cell_idx].copy()  # (n_cells × n_genes)

    # Shuffle expression values per gene independently
    for g in range(X_sub.shape[1]):
        X_sub[:, g] = rng.permutation(X_sub[:, g])

    # Score each program — mean expression of program genes per cell
    n_programs = len(prog_indices)
    mean_scores = np.zeros(n_programs, dtype=np.float32)

    for p, indices in enumerate(prog_indices):
        if len(indices) == 0:
            continue
        mean_scores[p] = X_sub[:, indices].mean()

    return mean_scores
