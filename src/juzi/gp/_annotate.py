# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import warnings
import numpy as np
import pandas as pd

from anndata import AnnData
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from typing import Dict, List


def programs_annotate(
    adata: AnnData,
    gene_sets: Dict[str, List[str]] | object,
    padj_method: str = "fdr_bh",
    min_overlap: int = 1,
    copy: bool = False,
) -> AnnData | None:
    """Annotate consensus programs by overlap with reference gene sets.

    For each consensus program, reads the canonical gene set from
    juzi_cluster_genes — computed at cluster time by programs_cluster.
    Computes Jaccard similarity and hypergeometric overlap statistics
    against every gene set in the provided reference dictionary.

    Results are stored as a tidy long-format DataFrame in
    uns["juzi_annotation"] with one row per program × gene_set pair,
    sorted by adjusted p-value.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_genes, juzi_cluster_labels,
        and juzi_G_genes in .uns, produced by juzi.gp.programs_cluster.
    gene_sets : Dict[str, List[str]] | object
        Reference gene sets to score against. Accepts:
            - A plain dict mapping gene set name to gene list
            - Any juzi.mg gene set object with an as_dict() method
              e.g. jz.mg.CellCycle(), jz.mg.Hallmark3CA()
            - Output of jz.mg.read_msigdb()
    padj_method : str
        Multiple testing correction method. Default is "fdr_bh"
        (Benjamini-Hochberg FDR) applied across all program × gene_set
        pairs simultaneously.
    min_overlap : int
        Minimum number of overlapping genes required for a program ×
        gene_set pair to be retained in the output. Must be >= 0.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following fields populated:
            .uns["juzi_annotation"] : DataFrame with columns:
                program            — program label (e.g. "C0")
                gene_set           — reference gene set name
                jaccard            — Jaccard similarity
                n_overlap          — number of overlapping genes
                n_program          — number of program genes
                n_geneset          — gene set size after intersection
                                     with background
                n_background       — total background gene count
                pval               — hypergeometric p-value
                padj               — FDR-adjusted p-value
                overlap_genes      — comma-separated overlapping gene names
                overlap_genes_list — sorted list of overlapping gene names
            .uns["juzi_annotation_meta"] : dict with annotation metadata
    """
    adata = adata.copy() if copy else adata

    # Validate

    for field, store in [
        ("juzi_cluster_genes", "uns"),
        ("juzi_cluster_labels", "uns"),
        ("juzi_G_genes", "uns"),
    ]:
        if field not in getattr(adata, store):
            raise KeyError(
                f"'{field}' not found in .{store}. "
                "Run juzi.gp.programs_cluster first."
            )

    if min_overlap < 0:
        raise ValueError("min_overlap must be >= 0.")

    # Resolve gene sets

    if hasattr(gene_sets, "as_dict"):
        gene_sets = gene_sets.as_dict()

    if not isinstance(gene_sets, dict) or len(gene_sets) == 0:
        raise ValueError(
            "gene_sets must be a non-empty dict or a juzi.mg gene set object."
        )

    # Setup

    cluster_genes = adata.uns["juzi_cluster_genes"]
    gene_names = np.array(adata.uns["juzi_G_genes"], dtype=object)
    labels = adata.uns["juzi_cluster_labels"]
    unique_C = np.unique(labels)
    n_background = len(gene_names)
    background = set(gene_names.tolist())

    # Build per-program gene sets

    program_gene_sets: Dict[int, set[str]] = {}
    for c in unique_C:
        genes = cluster_genes.get(int(c), [])
        program_gene_sets[int(c)] = set(genes)

    # Filter reference gene sets to background
    # Hypergeometric test requires gene sets restricted to the background

    ref_sets_filtered: Dict[str, set[str]] = {}
    for gs_name, gs_genes in gene_sets.items():
        filtered = set(gs_genes) & background
        if len(filtered) > 0:
            ref_sets_filtered[gs_name] = filtered

    if len(ref_sets_filtered) == 0:
        raise ValueError(
            "No reference gene set genes overlap with juzi_G_genes. "
            "Check that gene_sets use the same gene symbols as your AnnData."
        )

    # Compute overlap statistics

    rows = []

    for c in unique_C:
        prog_genes = program_gene_sets[int(c)]
        n_program = len(prog_genes)

        if n_program == 0:
            warnings.warn(
                f"Program C{int(c)} has no genes and was skipped during annotation.",
                UserWarning,
                stacklevel=2,
            )
            continue

        for gs_name, gs_genes in ref_sets_filtered.items():
            overlap = prog_genes & gs_genes
            n_overlap = len(overlap)

            if n_overlap < min_overlap:
                continue

            n_geneset = len(gs_genes)

            union = prog_genes | gs_genes
            jaccard = n_overlap / len(union) if len(union) > 0 else 0.0

            # P(X >= n_overlap) where X ~ Hypergeom(N, K, n)
            #   N = background size
            #   K = reference gene set size in background
            #   n = number of program genes
            pval = hypergeom.sf(
                n_overlap - 1,
                n_background,
                n_geneset,
                n_program,
            )

            overlap_genes_sorted = sorted(overlap)

            rows.append(
                {
                    "program": f"C{int(c)}",
                    "gene_set": gs_name,
                    "jaccard": round(jaccard, 6),
                    "n_overlap": n_overlap,
                    "n_program": n_program,
                    "n_geneset": n_geneset,
                    "n_background": n_background,
                    "pval": pval,
                    "overlap_genes": ",".join(overlap_genes_sorted),
                    "overlap_genes_list": overlap_genes_sorted,
                }
            )

    if len(rows) == 0:
        raise ValueError(
            "No program × gene_set pairs passed filtering. "
            "Check that juzi_cluster_genes is populated and non-empty, and "
            "consider lowering min_overlap."
        )

    # FDR correction

    annot_df = pd.DataFrame(rows)

    _, annot_df["padj"], _, _ = multipletests(
        annot_df["pval"],
        method=padj_method,
    )

    annot_df = annot_df.sort_values("padj").reset_index(drop=True)

    adata.uns["juzi_annotation"] = annot_df
    adata.uns["juzi_annotation_meta"] = {
        "padj_method": padj_method,
        "min_overlap": int(min_overlap),
        "n_programs": int(len(unique_C)),
        "n_gene_sets_input": int(len(gene_sets)),
        "n_gene_sets_tested": int(len(ref_sets_filtered)),
        "n_background": int(n_background),
    }

    return adata if copy else None
