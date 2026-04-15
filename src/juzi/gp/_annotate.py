# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import pandas as pd

from anndata import AnnData
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from typing import Dict, List


def annotate(
    adata: AnnData,
    gene_sets: Dict[str, List[str]] | object,
    n_top_genes: int = 50,
    use_specificity: bool = True,
    padj_method: str = "fdr_bh",
    copy: bool = False,
) -> AnnData | None:
    """Annotate consensus programs by overlap with reference gene sets.

    For each consensus program identified by juzi.gp.cluster, selects
    the top n_top_genes genes by loading or specificity and computes
    Jaccard similarity and hypergeometric overlap statistics against
    every gene set in the provided reference dictionary.

    Results are stored as a tidy long-format DataFrame in
    uns["juzi_annotation"] with one row per program × gene_set pair,
    sorted by adjusted p-value.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_cluster_G, juzi_cluster_labels, and
        juzi_G_genes in .uns, produced by juzi.gp.cluster.
    gene_sets : Dict[str, List[str]] | object
        Reference gene sets to score against. Accepts:
            - A plain dict mapping gene set name to gene list
            - Any juzi.mg gene set object with an as_dict() method
              e.g. jz.mg.CellCycle(), jz.mg.MsigDB3CA()
            - Output of jz.mg.read_msigdb()
    n_top_genes : int
        Number of top genes per program to use for overlap computation.
    use_specificity : bool
        If True, rank genes by specificity score (loading in this program
        divided by total loading across all programs). If False, rank by
        raw loading magnitude.
    padj_method : str
        Multiple testing correction method. Default is "fdr_bh"
        (Benjamini-Hochberg FDR) applied across all program × gene_set
        pairs simultaneously.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following field populated:
            .uns["juzi_annotation"] : DataFrame with columns:
                program      — program label (e.g. "C0")
                gene_set     — reference gene set name
                jaccard      — Jaccard similarity of top program genes
                               vs reference gene set
                n_overlap    — number of overlapping genes
                n_program    — number of top program genes
                n_geneset    — number of genes in reference gene set
                               after intersection with background
                n_background — total background gene count (juzi_G_genes)
                pval         — hypergeometric p-value
                padj         — FDR-adjusted p-value
                overlap_genes— comma-separated overlapping gene names
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
                "Run juzi.gp.nmf and juzi.gp.cluster first."
            )

    if n_top_genes < 1:
        raise ValueError("n_top_genes must be >= 1.")

    # Resolve gene sets

    if hasattr(gene_sets, "as_dict"):
        gene_sets = gene_sets.as_dict()

    if not isinstance(gene_sets, dict) or len(gene_sets) == 0:
        raise ValueError(
            "gene_sets must be a non-empty dict or a juzi.mg gene set object."
        )

    # Setup

    G = adata.uns["juzi_cluster_G"]  # (n_programs × n_genes)
    gene_names = np.array(adata.uns["juzi_G_genes"])
    labels = adata.uns["juzi_cluster_labels"]
    unique_C = np.unique(labels)
    n_programs = len(unique_C)
    n_background = len(gene_names)
    background = set(gene_names)

    if n_top_genes > G.shape[1]:
        raise ValueError(
            f"n_top_genes={n_top_genes} exceeds number of genes "
            f"({G.shape[1]}) used in NMF."
        )

    # Gene ranking

    if use_specificity:
        total = G.sum(axis=0, keepdims=True) + 1e-8
        G_rank = G / total
    else:
        G_rank = G

    # Precompute top program genes

    program_top_genes = {}
    for i, c in enumerate(unique_C):
        top_idx = np.argsort(G_rank[i])[-n_top_genes:]
        program_top_genes[c] = set(gene_names[top_idx])

    # Precompute reference gene sets intersected with background
    # Hypergeometric test requires gene sets restricted to the background

    ref_sets_filtered = {}
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
        program_genes = program_top_genes[c]
        n_program = len(program_genes)

        for gs_name, gs_genes in ref_sets_filtered.items():
            overlap = program_genes & gs_genes
            n_overlap = len(overlap)
            n_geneset = len(gs_genes)

            # Jaccard
            union = program_genes | gs_genes
            jaccard = n_overlap / len(union) if len(union) > 0 else 0.0

            # Hypergeometric p-value
            # P(X >= n_overlap) where X ~ Hypergeom(N, K, n)
            #   N = background size
            #   K = reference gene set size in background
            #   n = number of program top genes drawn
            pval = hypergeom.sf(
                n_overlap - 1,
                n_background,
                n_geneset,
                n_program,
            )

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
                    "overlap_genes": ",".join(sorted(overlap)),
                }
            )

    # FDR correction

    annot_df = pd.DataFrame(rows)

    _, annot_df["padj"], _, _ = multipletests(
        annot_df["pval"],
        method=padj_method,
    )

    annot_df = annot_df.sort_values("padj").reset_index(drop=True)

    adata.uns["juzi_annotation"] = annot_df

    return adata if copy else None
