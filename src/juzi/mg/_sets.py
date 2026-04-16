# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import json
import importlib.resources

from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field


def available_sets() -> List[str]:
    """List available gene set collections in juzi.mg."""
    return ["CancerBreast", "CancerPathways", "CellCycle", "MsigDB3CA"]


def _load_json(resource: str) -> Dict[str, List[str]]:
    """Load a bundled JSON gene set file from juzi.mg."""
    with importlib.resources.open_text("juzi.mg", resource) as f:
        return json.load(f)


def _make_gene_set_class(class_name: str, resource: str, docstring: str):
    """Dynamically create a gene set dataclass from a bundled JSON resource.

    The JSON must be a flat {gene_set_name: [gene, ...]} dictionary.
    Each key becomes an attribute on the class, accessible as:
        obj.GENE_SET_NAME  to  List[str]
        obj.as_dict()      to  Dict[str, List[str]]
        obj.all()          to  List[str] (all genes merged)
        obj.info()         to  List[str] (gene set names)
    """

    class GeneSetCollection:
        __doc__ = docstring

        def __init__(self):
            data = _load_json(resource)
            for key, genes in data.items():
                setattr(self, key, genes)
            self._data = data

        def info(self) -> List[str]:
            """Return list of gene set names."""
            return list(self._data.keys())

        def all(self) -> List[str]:
            """Return all genes across all sets, deduplicated."""
            seen, genes = set(), []
            for g_list in self._data.values():
                for g in g_list:
                    if g not in seen:
                        seen.add(g)
                        genes.append(g)
            return genes

        def as_dict(self) -> Dict[str, List[str]]:
            """Return the full gene set dictionary."""
            return dict(self._data)

        def __repr__(self) -> str:
            n_sets = len(self._data)
            n_genes = len(self.all())
            return f"{class_name}(" f"n_sets={n_sets}, " f"n_genes={n_genes})"

    GeneSetCollection.__name__ = class_name
    GeneSetCollection.__qualname__ = class_name
    return GeneSetCollection


# Gene set collections

CancerBreast = _make_gene_set_class(
    class_name="CancerBreast",
    resource="cancer_breast.json",
    docstring="""Breast cancer gene markers spanning subtypes and drivers.

    References
    ----------
    .. [1] J.S. Parker et al. Supervised Risk Predictor of Breast Cancer
       Based on Intrinsic Subtypes. Journal of Clinical Oncology. 2009.
    .. [2] M.D Burstein et al. Comprehensive genomic analysis identifies
       novel subtypes and targets of triple-negative breast cancer.
       Clinical Cancer Research. 2015.
    """,
)

CancerPathways = _make_gene_set_class(
    class_name="CancerPathways",
    resource="cancer_pathways.json",
    docstring="""Genes from canonical cancer signalling pathways.

    References
    ----------
    .. [1] F. Sanchez-Vega et al. Oncogenic Signaling Pathways in The
       Cancer Genome Atlas. Cell. 2018.
    .. [2] M. Kanehisa et al. KEGG: Kyoto Encyclopedia of Genes and
       Genomes. Nucleic Acids Research. 2000.
    """,
)

CellCycle = _make_gene_set_class(
    class_name="CellCycle",
    resource="cell_cycle.json",
    docstring="""Cell cycle gene markers.

    Attributes are loaded dynamically from cell_cycle.json.

    References
    ----------
    .. [1] I. Tirosh et al. Dissecting the multicellular ecosystem of
       metastatic melanoma by single-cell RNA-seq. Science. 2016.
    """,
)

Hallmark3CA = _make_gene_set_class(
    class_name="3CA",
    resource="3CA.json",
    docstring="""3CA meta-programs for malignant and normal cells.

    References
    ----------
    .. [1] Gavish et al. Hallmarks of transcriptional intratumour
       heterogeneity across a thousand tumours. Nature. 2023.
    """,
)
