import pandas as pd
import pytest
import juzi as jz


def test_factor_loadings_returns_dataframe(adata_similarity):
    df = jz.ut.factor_loadings(adata_similarity)
    assert isinstance(df, pd.DataFrame)


def test_factor_scores_returns_dataframe(adata_nmf):
    df = jz.ut.factor_scores(adata_nmf)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == sum(adata_nmf.uns["juzi_k"])


def test_programs_genes_returns_cluster_gene_sets(adata_cluster_progressive):
    result = jz.ut.programs_genes(adata_cluster_progressive)
    assert isinstance(result, dict)
    assert all(k.startswith("C") for k in result)


def test_programs_compare_returns_dataframe(adata_cluster_centroid, adata_cluster_progressive):
    df = jz.ut.programs_compare(adata_cluster_centroid, adata_cluster_progressive)
    assert isinstance(df, pd.DataFrame)


def test_programs_donors_returns_dataframe(adata_cluster_centroid):
    df = jz.ut.programs_donors(adata_cluster_centroid)
    assert isinstance(df, pd.DataFrame)
    assert all(col.startswith("C") for col in df.columns)
