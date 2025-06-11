import pytest
import numpy as np
import juzi as jz

from anndata import AnnData


def make_anndata(k=[2, 2], init="nndsvda"):
    a = np.random.normal(10., 3., size=(8, 100)),
    b = np.random.normal(90., 9., size=(8, 100)),
    X = np.vstack((a[0], b[0]))
    adata = AnnData(
        X,
        var={"gene": np.arange(100).astype(str)},
        obs={"cell": 8*["a"] + 8*["b"]},
    )
    return jz.cs.nmf(
        adata,
        k=k,
        init=init,
        target_sum=10,
        min_cells=2,
        key="cell",
        genes=None,
    )


class TestPrune:
    def test_error_missing_factors(self):
        adata = make_anndata()
        del adata.varm["juzi_G"]
        with pytest.raises(KeyError):
            jz.cs.prune(adata)

    def test_error_missing_names(self):
        adata = make_anndata()
        del adata.uns["juzi_names"]
        with pytest.raises(KeyError):
            jz.cs.prune(adata)

    def test_error_missing_k(self):
        adata = make_anndata()
        del adata.uns["juzi_k"]
        with pytest.raises(KeyError):
            jz.cs.prune(adata)

    def test_error_top_k(self):
        adata = make_anndata()
        with pytest.raises(ValueError):
            jz.cs.prune(adata, top_k=100000)

    def test_error_min_similarity(self):
        adata = make_anndata()
        with pytest.raises(ValueError):
            jz.cs.prune(adata, min_similarity=1.1)

    def test_error_min_k(self):
        adata = make_anndata()
        with pytest.raises(ValueError):
            jz.cs.prune(adata, min_k=10)

    def test_prune_empty(self):
        adata = make_anndata(k=[1, 2, 7], init="random")
        jz.cs.prune(adata, min_k=3, min_similarity=1.0)
        assert np.sum(adata.uns["juzi_keep"]) == 0

    def test_prune_none(self):
        adata = make_anndata(k=[5])
        jz.cs.prune(adata, min_k=0, min_similarity=0.0)
        assert np.sum(adata.uns["juzi_keep"]) == 10.0

    def test_prune_half(self):
        adata = make_anndata(k=[2, 2])
        jz.cs.prune(adata, min_k=1, min_similarity=1.0)
        assert np.sum(adata.uns["juzi_keep"]) == 4.0
        assert np.all(adata.uns["juzi_keep"] == np.array(
            [1., 1., 0., 0., 1., 1., 0., 0.]))
