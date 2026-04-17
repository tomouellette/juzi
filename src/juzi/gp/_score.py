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

    Program genes are taken primarily from `adata.uns["juzi_cluster_genes"]`,
    which is the canonical program definition for both centroid and progressive
    clustering modes. If `juzi_cluster_genes` is absent, the function falls
    back to deriving genes from `juzi_cluster_G`.

    Per-cell scores are computed as the mean expression of the program genes
    minus the mean expression of a matched set of control genes drawn from
    similar expression bins, following Tirosh et al. 2016.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Should contain normalised log expression in .X
        or the specified layer. Must share gene names with the genes used
        during juzi.gp.nmf.
    n_top_genes : int
        Maximum number of genes per program used for scoring. If canonical
        program genes are present, the first `n_top_genes` are used. If the
        function falls back to `juzi_cluster_G`, the top `n_top_genes` genes
        are derived from cluster loadings.
    use_combined : bool
        Only used when falling back to `juzi_cluster_G`. If True, rank genes
        by specificity score rather than raw loading magnitude.
    n_control_genes : int
        Number of control genes per program gene used for background
        subtraction. Control genes are drawn from the same expression bin
        as each program gene. Total control set size is up to
        n_top_genes × n_control_genes, deduplicated.
    gene_pool : List[str] | None
        Genes eligible to serve as control genes. Defaults to all genes
        in adata. Program genes are excluded from the control pool.
    gene_names_col : str | None
        Column in adata.var containing gene names. Must match the namespace
        of the genes used in juzi.gp.nmf / clustering. If None, var_names
        is used.
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
            .uns["juzi_program_genes"]   : dict mapping program index to genes used
            .uns["juzi_score_meta"]      : scoring parameter metadata
    """
    adata = adata.copy() if copy else adata

    # Validate

    if "juzi_cluster_genes" not in adata.uns and "juzi_cluster_G" not in adata.uns:
        raise KeyError(
            "Neither 'juzi_cluster_genes' nor 'juzi_cluster_G' found in .uns. "
            "Run juzi.gp.programs_cluster first."
        )

    if "juzi_cluster_labels" not in adata.uns:
        raise KeyError(
            "'juzi_cluster_labels' not found in .uns. "
            "Run juzi.gp.programs_cluster first."
        )

    if "juzi_G_genes" not in adata.uns:
        raise KeyError(
            "'juzi_G_genes' not found in .uns. " "Run juzi.gp.nmf_fit first."
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

    # Extract cell gene names

    cell_genes = (
        adata.var[gene_names_col].to_numpy()
        if gene_names_col is not None
        else adata.var_names.to_numpy()
    )

    nmf_genes = np.array(adata.uns["juzi_G_genes"], dtype=object)

    # Align genes between adata and NMF/clustering gene namespace

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

    # Determine program genes
    # Primary path: canonical cluster genes
    # Fallback path: derive from juzi_cluster_G

    program_genes: Dict[int, List[str]] = {}
    gene_source = "juzi_cluster_genes"

    if "juzi_cluster_genes" in adata.uns:
        cluster_genes = adata.uns["juzi_cluster_genes"]

        # Ensure deterministic program order by sorted program index
        program_ids = sorted(int(k) for k in cluster_genes.keys())

        for p in program_ids:
            genes = list(cluster_genes[p])

            # Keep only genes present in both NMF and cell gene spaces
            genes = [g for g in genes if g in nmf_gene_to_idx and g in cell_gene_to_idx]

            if len(genes) == 0:
                warnings.warn(
                    f"Program C{p} has no genes shared with adata and will score as zero.",
                    UserWarning,
                    stacklevel=2,
                )
                program_genes[p] = []
                continue

            if len(genes) < n_top_genes:
                warnings.warn(
                    f"Program C{p} has only {len(genes)} usable genes after alignment; "
                    f"scoring will use all available genes instead of n_top_genes={n_top_genes}.",
                    UserWarning,
                    stacklevel=2,
                )
                program_genes[p] = genes
            else:
                program_genes[p] = genes[:n_top_genes]

        n_programs = len(program_ids)

    else:
        # Fallback for older objects that do not yet store canonical genes
        gene_source = "juzi_cluster_G"

        H_full = adata.uns["juzi_cluster_G"]  # (n_programs × n_genes)
        shared_nmf_idx = np.array([nmf_gene_to_idx[g] for g in shared_genes])
        H_aligned = H_full[:, shared_nmf_idx]
        n_programs = H_aligned.shape[0]

        H_rank = _combined_score(H_aligned) if use_combined else H_aligned

        for p in range(n_programs):
            gene_ranks = H_rank[p]
            top_local = np.argsort(gene_ranks)[-n_top_genes:][::-1]
            genes = shared_genes[top_local].tolist()
            program_genes[p] = genes

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
    used_program_genes: Dict[int, List[str]] = {}

    for p in tqdm(range(n_programs), desc="[juzi] Scoring", disable=silent):
        genes = program_genes[p]

        if len(genes) == 0:
            used_program_genes[p] = []
            scores[:, p] = 0.0
            continue

        top_cell_idx = np.array([cell_gene_to_idx[g] for g in genes], dtype=int)
        used_program_genes[p] = genes

        # Program score — mean expression of program genes per cell
        program_score = X[:, top_cell_idx].mean(axis=1)

        # Control gene selection — bin-matched, excluding program genes
        top_cell_idx_set = set(top_cell_idx.tolist())
        control_idx = []

        for gene_idx in top_cell_idx:
            gene_bin = gene_bins[gene_idx]
            eligible = ~np.isin(np.arange(len(cell_genes)), list(top_cell_idx_set))

            bin_members = np.where((gene_bins == gene_bin) & pool_mask & eligible)[0]

            # Fall back to adjacent bins if current bin has no eligible genes
            if len(bin_members) == 0:
                for delta in [1, -1, 2, -2, 3, -3]:
                    adjacent = np.where(
                        (gene_bins == gene_bin + delta) & pool_mask & eligible
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
    adata.uns["juzi_program_genes"] = used_program_genes
    adata.uns["juzi_score_meta"] = {
        "n_top_genes": n_top_genes,
        "use_combined": use_combined,
        "n_control_genes": n_control_genes,
        "gene_pool_provided": gene_pool is not None,
        "gene_names_col": gene_names_col,
        "layer": layer,
        "seed": seed,
        "gene_source": gene_source,
    }

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

    Must be run after juzi.gp.score_cells since it reads
    juzi_program_scores and juzi_program_genes.
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

    if "juzi_score_meta" in adata.uns and gene_names_col is None:
        gene_names_col = adata.uns["juzi_score_meta"].get(
            "gene_names_col", gene_names_col
        )

    cell_genes = (
        adata.var[gene_names_col].to_numpy()
        if gene_names_col is not None
        else adata.var_names.to_numpy()
    )

    program_genes = adata.uns["juzi_program_genes"]
    n_programs = len(program_genes)
    n_cells = adata.n_obs

    # Map program genes to cell expression indices
    gene_to_idx = {g: i for i, g in enumerate(cell_genes)}
    prog_indices: List[np.ndarray] = []

    for p in range(n_programs):
        genes = program_genes[p]
        indices = np.array(
            [gene_to_idx[g] for g in genes if g in gene_to_idx],
            dtype=int,
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

    null_matrix = np.vstack(null_scores)  # (n_shuffles × n_programs)

    # Fit null distribution per program

    null_mean = null_matrix.mean(axis=0)
    null_std = null_matrix.std(axis=0)
    null_std = np.where(null_std == 0, 1e-8, null_std)

    # Compute z-scores and p-values for real cells

    real_scores = adata.obsm["juzi_program_scores"]
    z_scores = (real_scores - null_mean[None, :]) / null_std[None, :]
    pvals = stats.norm.sf(z_scores)

    # BH correction across all program × cell pairs

    flat_pvals = pvals.flatten()
    _, flat_padj, _, _ = multipletests(flat_pvals, method="fdr_bh")
    padj = flat_padj.reshape(n_cells, n_programs)

    # Cell assignment

    unique_C = np.unique(adata.uns["juzi_cluster_labels"])
    prog_names = [f"C{int(c)}" for c in unique_C]
    labels_out = np.full(n_cells, "unresolved", dtype=object)

    sig_mask = padj < padj_thresh
    n_sig = sig_mask.sum(axis=1)

    single_sig = n_sig == 1
    if single_sig.any():
        prog_idx = np.argmax(sig_mask[single_sig], axis=1)
        labels_out[single_sig] = [prog_names[p] for p in prog_idx]

    multi_sig = n_sig > 1
    if multi_sig.any():
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
    """Run one shuffle iteration."""
    rng = np.random.default_rng(seed)

    cell_idx = rng.choice(X.shape[0], size=n_cells, replace=False)
    X_sub = X[cell_idx].copy()

    for g in range(X_sub.shape[1]):
        X_sub[:, g] = rng.permutation(X_sub[:, g])

    n_programs = len(prog_indices)
    mean_scores = np.zeros(n_programs, dtype=np.float32)

    for p, indices in enumerate(prog_indices):
        if len(indices) == 0:
            continue
        mean_scores[p] = X_sub[:, indices].mean()

    return mean_scores
