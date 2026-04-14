# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np

from anndata import AnnData
from typing import List
from tqdm import tqdm


def score(
    adata: AnnData,
    n_top_genes: int = 50,
    use_specificity: bool = True,
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
    use_specificity : bool
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

    # Gene specificity

    if use_specificity:
        total_loading = H_aligned.sum(axis=0, keepdims=True) + 1e-8
        H_rank = H_aligned / total_loading
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
