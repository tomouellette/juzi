import numpy as np
import pytest
import juzi as jz


def test_similarity_requires_gene_names(adata_pruned):
    adata = adata_pruned.copy()
    del adata.uns["juzi_G_genes"]
    with pytest.raises(KeyError):
        jz.gp.similarity(adata)


def test_similarity_stores_expected_outputs(adata_pruned):
    adata = adata_pruned.copy()
    jz.gp.similarity(adata, metric="jaccard", top_k=20)
    assert "juzi_similarity" in adata.uns
    assert "juzi_similarity_idx" in adata.uns
    assert "juzi_keep_similarity" in adata.uns
    assert "juzi_similarity_meta" in adata.uns
    assert "juzi_similarity_scope" in adata.uns


def test_similarity_matrix_symmetric_and_zero_diag(adata_similarity):
    S = adata_similarity.uns["juzi_similarity"]
    np.testing.assert_allclose(S, S.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(S), 0.0)


def test_similarity_idx_matches_current_keep(adata_similarity):
    expected = np.where(adata_similarity.uns["juzi_keep"])[0]
    np.testing.assert_array_equal(adata_similarity.uns["juzi_similarity_idx"], expected)


def test_similarity_invalid_metric_raises(adata_pruned):
    with pytest.raises(ValueError):
        jz.gp.similarity(adata_pruned, metric="bad")


def test_similarity_top_k_required_for_jaccard(adata_pruned):
    with pytest.raises(ValueError):
        jz.gp.similarity(adata_pruned, metric="jaccard", top_k=None)


def test_similarity_filter_requires_similarity(adata_pruned):
    with pytest.raises(KeyError):
        jz.gp.similarity_filter(adata_pruned, min_similarity=0.3)


def test_similarity_filter_updates_keep_mask(adata_similarity):
    adata = adata_similarity.copy()
    before = adata.uns["juzi_keep_similarity"].sum()
    jz.gp.similarity_filter(adata, min_similarity=0.5)
    after = adata.uns["juzi_keep_similarity"].sum()
    assert after <= before


def test_similarity_backward_compat_wrapper_runs(adata_pruned):
    adata = adata_pruned.copy()
    jz.gp.similarity_compute(adata, distance="jaccard", top_k=20)
    assert "juzi_similarity" in adata.uns
