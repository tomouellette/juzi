import numpy as np
import pytest
import juzi as jz


def test_score_cells_prefers_cluster_genes(adata_cluster_progressive):
    adata = adata_cluster_progressive.copy()
    if "juzi_cluster_G" in adata.uns:
        del adata.uns["juzi_cluster_G"]
    jz.gp.score_cells(
        adata, n_top_genes=10, n_control_genes=10, seed=0, gene_names_col="gene_name"
    )
    assert "juzi_program_scores" in adata.obsm
    assert adata.uns["juzi_score_meta"]["gene_source"] == "juzi_cluster_genes"


def test_score_cells_fallback_to_cluster_G(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    del adata.uns["juzi_cluster_genes"]
    jz.gp.score_cells(
        adata, n_top_genes=10, n_control_genes=10, seed=0, gene_names_col="gene_name"
    )
    assert adata.uns["juzi_score_meta"]["gene_source"] == "juzi_cluster_G"


def test_score_cells_outputs(adata_scored):
    assert "juzi_program_scores" in adata_scored.obsm
    assert "juzi_program_genes" in adata_scored.uns
    assert "juzi_score_meta" in adata_scored.uns
    assert adata_scored.obsm["juzi_program_scores"].shape[0] == adata_scored.n_obs


def test_score_cells_reproducible(adata_cluster_centroid):
    a = adata_cluster_centroid.copy()
    b = adata_cluster_centroid.copy()
    jz.gp.score_cells(
        a, n_top_genes=10, n_control_genes=10, seed=1, gene_names_col="gene_name"
    )
    jz.gp.score_cells(
        b, n_top_genes=10, n_control_genes=10, seed=1, gene_names_col="gene_name"
    )
    np.testing.assert_allclose(
        a.obsm["juzi_program_scores"], b.obsm["juzi_program_scores"], atol=1e-6
    )


def test_score_cells_warns_on_raw_like_counts(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    adata.X = adata.X * 1000
    with pytest.warns(UserWarning, match="raw counts"):
        jz.gp.score_cells(
            adata,
            n_top_genes=10,
            n_control_genes=10,
            gene_names_col="gene_name",
            seed=0,
        )


def test_score_classify_outputs(adata_scored):
    adata = adata_scored.copy()
    jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50, seed=0)
    assert "juzi_program_pvals" in adata.obsm
    assert "juzi_program_padj" in adata.obsm
    assert "juzi_program_label" in adata.obs
    assert "juzi_classify_params" in adata.uns


def test_score_classify_reproducible(adata_scored):
    a = adata_scored.copy()
    b = adata_scored.copy()
    jz.gp.score_classify(a, n_shuffles=2, n_cells_per_shuffle=50, seed=1)
    jz.gp.score_classify(b, n_shuffles=2, n_cells_per_shuffle=50, seed=1)
    np.testing.assert_array_equal(
        a.obs["juzi_program_label"].values, b.obs["juzi_program_label"].values
    )
