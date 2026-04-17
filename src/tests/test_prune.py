import numpy as np
import pytest
import juzi as jz


def test_prune_requires_core_nmf_fields(adata_nmf):
    adata = adata_nmf.copy()
    del adata.uns["juzi_G_genes"]
    with pytest.raises(KeyError):
        jz.gp.nmf_prune(adata)


def test_prune_stores_expected_fields(adata_nmf):
    adata = adata_nmf.copy()
    jz.gp.nmf_prune(
        adata,
        top_k=20,
        min_similarity=0.2,
        min_other_resolutions=1,
        matching="hungarian",
    )
    assert "juzi_keep_prune" in adata.uns
    assert "juzi_prune" in adata.uns
    assert "juzi_prune_matches" in adata.uns
    assert adata.uns["juzi_keep_prune"].dtype == bool


def test_prune_updates_juzi_keep_as_intersection(adata_nmf):
    adata = adata_nmf.copy()
    jz.gp.nmf_prune(adata, min_other_resolutions=1)
    expected = (
        adata.uns["juzi_keep_prune"]
        & adata.uns["juzi_keep_similarity"]
        & adata.uns["juzi_keep_cluster"]
    )
    np.testing.assert_array_equal(adata.uns["juzi_keep"], expected)


def test_prune_invalid_matching_raises(adata_nmf):
    with pytest.raises(ValueError):
        jz.gp.nmf_prune(adata_nmf, matching="bad")


def test_prune_invalid_top_k_raises(adata_nmf):
    with pytest.raises(ValueError):
        jz.gp.nmf_prune(adata_nmf, top_k=0)


def test_prune_min_other_resolutions_validation(adata_nmf):
    with pytest.raises(ValueError):
        jz.gp.nmf_prune(adata_nmf, min_other_resolutions=10)


def test_prune_can_run_greedy_and_hungarian(adata_nmf):
    a = adata_nmf.copy()
    b = adata_nmf.copy()
    jz.gp.nmf_prune(a, matching="greedy", min_other_resolutions=1)
    jz.gp.nmf_prune(b, matching="hungarian", min_other_resolutions=1)
    assert "juzi_keep_prune" in a.uns and "juzi_keep_prune" in b.uns
