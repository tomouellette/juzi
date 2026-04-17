import pandas as pd
import pytest
import juzi as jz


def test_programs_annotate_requires_cluster_genes(adata_cluster_centroid, gene_sets):
    adata = adata_cluster_centroid.copy()
    del adata.uns["juzi_cluster_genes"]
    with pytest.raises(KeyError):
        jz.gp.programs_annotate(adata, gene_sets=gene_sets)


def test_programs_annotate_outputs(adata_cluster_centroid, gene_sets):
    adata = adata_cluster_centroid.copy()
    jz.gp.programs_annotate(adata, gene_sets=gene_sets, min_overlap=1)
    assert "juzi_annotation" in adata.uns
    assert "juzi_annotation_meta" in adata.uns
    assert isinstance(adata.uns["juzi_annotation"], pd.DataFrame)


def test_programs_annotate_has_new_columns(adata_annotated):
    df = adata_annotated.uns["juzi_annotation"]
    for col in ["program", "gene_set", "jaccard", "padj", "overlap_genes", "overlap_genes_list"]:
        assert col in df.columns


def test_programs_annotate_min_overlap_filters_rows(adata_cluster_centroid, gene_sets):
    a = adata_cluster_centroid.copy()
    b = adata_cluster_centroid.copy()
    jz.gp.programs_annotate(a, gene_sets=gene_sets, min_overlap=1)
    jz.gp.programs_annotate(b, gene_sets=gene_sets, min_overlap=5)
    assert len(b.uns["juzi_annotation"]) <= len(a.uns["juzi_annotation"])
