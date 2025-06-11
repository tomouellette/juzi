import pytest
import numpy as np
import juzi as jz

from anndata import AnnData


def make_anndata():
    X = np.vstack([
        np.random.normal(10., 3., size=100),
        np.random.normal(10., 3., size=100),
        np.random.normal(10., 3., size=100),
        np.random.normal(90., 1., size=100),
        np.random.normal(90., 1., size=100),
        np.random.normal(90., 1., size=100),
    ])
    return AnnData(
        X,
        var={
            "gene": np.arange(100).astype(str),
            "highly_variable": np.array(32*[True] + 68*[False]),
        },
        obs={"cell": 3*["a"] + 3*["b"]},
    )


class TestNMF:
    def test_error_key(self):
        with pytest.raises(KeyError):
            _ = jz.cs.nmf(make_anndata(), key="wrong")

    def test_error_layer(self):
        with pytest.raises(KeyError):
            _ = jz.cs.nmf(make_anndata(), key="cell", layer="counts")

    def test_error_genes(self):
        with pytest.raises(ValueError):
            _ = jz.cs.nmf(make_anndata(), key="cell", genes=None)

    def test_error_k(self):
        with pytest.raises(TypeError):
            _ = jz.cs.nmf(make_anndata(), k=2, key="cell", genes=None)

    def test_error_min_cells(self):
        with pytest.raises(ValueError):
            _ = jz.cs.nmf(make_anndata(), k=[2], key="cell", genes=None)

    def test_nmf_default(self):
        jdata = jz.cs.nmf(
            make_anndata(),
            k=[3],
            min_cells=2,
            key="cell",
            genes=None
        )

        n, k = jdata.varm["juzi_G"].shape
        assert n == 100
        assert k == 2 * 3
        assert "juzi_names" in jdata.uns

    def test_nmf_cell_loadings(self):
        jdata = jz.cs.nmf(
            make_anndata(),
            k=[3],
            min_cells=2,
            key="cell",
            genes=None,
            keep_C=True
        )

        n, k = jdata.varm["juzi_G"].shape
        assert n == 100
        assert k == 2 * 3

        n, k = jdata.obsm["juzi_C"].shape
        assert n == 6
        assert k == 3

    def test_nmf_gene_list(self):
        gene_list = np.arange(20).astype(str)
        jdata = jz.cs.nmf(
            make_anndata(),
            k=[3],
            min_cells=2,
            key="cell",
            genes=gene_list,
            keep_C=True
        )

        n, k = jdata.varm["juzi_G"].shape
        assert n == 20
        assert k == 2 * 3

        n, k = jdata.obsm["juzi_C"].shape
        assert n == 6
        assert k == 3

    def test_nmf_gene_column(self):
        jdata = jz.cs.nmf(
            make_anndata(),
            k=[3],
            min_cells=2,
            key="cell",
            genes="highly_variable",
            keep_C=True
        )

        n, k = jdata.varm["juzi_G"].shape
        assert n == 32
        assert k == 2 * 3

        n, k = jdata.obsm["juzi_C"].shape
        assert n == 6
        assert k == 3

    def test_nmf_parallel_threads(self):
        jdata = jz.cs.nmf(
            make_anndata(),
            k=[3],
            min_cells=2,
            key="cell",
            genes="highly_variable",
            keep_C=True,
            n_jobs=2,
            prefer="threads"
        )
        assert "juzi_names" in jdata.uns

    def test_nmf_parallel_processes(self):
        jdata = jz.cs.nmf(
            make_anndata(),
            k=[3],
            min_cells=2,
            key="cell",
            genes="highly_variable",
            keep_C=True,
            n_jobs=2,
            prefer="processes"
        )
        assert "juzi_names" in jdata.uns
