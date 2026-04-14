# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License


def read_msigdb(
    path: str,
    collections: list[str] | None = None,
) -> dict[str, list[str]]:
    """Read an MSigDB text file and return a simplified gene set dictionary.

    MSigDB provides gene sets as a text file containing a single JSON
    object. This function reads that file, parses it, and returns a
    simplified mapping of gene set name to gene symbols.

    Parameters
    ----------
    path : str
        Path to the MSigDB .txt file downloaded from
        https://www.gsea-msigdb.org.
    collections : list[str] | None
        If provided, only gene sets whose 'collection' field matches one
        of the provided values are retained. If None, all gene sets are
        returned. Examples:
            ["C4:3CA"]       — 3CA meta-programs only
            ["H"]            — Hallmarks only
            ["C4:3CA", "C8"] — 3CA and cell type signatures

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping gene set name to list of gene symbols.

    Examples
    --------
    >>> gene_sets = jz.mg.read_msigdb("msigdb_human.txt")
    >>> gene_sets = jz.mg.read_msigdb("msigdb_human.txt", collections=["C4:3CA"])
    >>> jz.gp.annotate(adata, gene_sets=gene_sets)
    """
    import json
    from pathlib import Path

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        raw = json.loads(f.read())

    result = {}
    for name, entry in raw.items():
        if collections is not None:
            if entry.get("collection", "") not in collections:
                continue
        genes = entry.get("geneSymbols", [])
        if genes:
            result[name] = genes

    return result
