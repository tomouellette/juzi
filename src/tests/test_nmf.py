import numpy as np
import pytest
from anndata import AnnData
import juzi as jz


def test_nmf_fit_returns_anndata(adata_raw):
    result = jz.gp.nmf_fit(
        adata_raw,
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
        gene_names_col="gene_name",
        keep_scores=True,
        center=False,
        seed=0,
    )
    assert isinstance(result, AnnData)


def test_nmf_fit_does_not_modify_input(adata_raw):
    result = jz.gp.nmf_fit(
        adata_raw,
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
        gene_names_col="gene_name",
        seed=0,
    )
    assert "juzi_G" not in adata_raw.varm
    assert "juzi_G" in result.varm


def test_nmf_fit_stores_expected_outputs(adata_nmf):
    assert "juzi_G" in adata_nmf.varm
    assert "juzi_G_genes" in adata_nmf.uns
    assert "juzi_names" in adata_nmf.uns
    assert "juzi_k" in adata_nmf.uns
    assert "juzi_keep_prune" in adata_nmf.uns
    assert "juzi_keep_similarity" in adata_nmf.uns
    assert "juzi_keep_cluster" in adata_nmf.uns
    assert "juzi_keep" in adata_nmf.uns
    assert "juzi_scores" in adata_nmf.obsm


def test_nmf_keep_masks_match_number_of_global_factors(adata_nmf):
    n_total = adata_nmf.varm["juzi_G"].shape[1]
    for key in [
        "juzi_keep_prune",
        "juzi_keep_similarity",
        "juzi_keep_cluster",
        "juzi_keep",
    ]:
        assert len(adata_nmf.uns[key]) == n_total


def test_nmf_scores_are_local_factor_space(adata_nmf):
    # local score columns are sum(k), not global number of factors
    assert adata_nmf.obsm["juzi_scores"].shape[1] == sum(adata_nmf.uns["juzi_k"])


def test_nmf_invalid_scalar_k_raises(adata_raw):
    with pytest.raises(TypeError):
        jz.gp.nmf_fit(adata_raw, key="donor_id", k=3, min_cells=10)


def test_nmf_invalid_k_values_raise(adata_raw):
    with pytest.raises(ValueError):
        jz.gp.nmf_fit(adata_raw, key="donor_id", k=[2, 0], min_cells=10)


def test_nmf_invalid_gene_names_col_raises(adata_raw):
    with pytest.raises(KeyError):
        jz.gp.nmf_fit(
            adata_raw, key="donor_id", k=[2], min_cells=10, gene_names_col="bad_col"
        )


def test_nmf_invalid_layer_raises(adata_raw):
    with pytest.raises(KeyError):
        jz.gp.nmf_fit(adata_raw, key="donor_id", k=[2], min_cells=10, layer="bad_layer")


def test_nmf_reproducible_given_seed(adata_raw):
    a = jz.gp.nmf_fit(
        adata_raw, key="donor_id", k=[2], min_cells=10, genes=None, seed=1, center=False
    )
    b = jz.gp.nmf_fit(
        adata_raw, key="donor_id", k=[2], min_cells=10, genes=None, seed=1, center=False
    )
    np.testing.assert_allclose(a.varm["juzi_G"], b.varm["juzi_G"], atol=1e-6)
