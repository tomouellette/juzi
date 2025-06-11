import pytest
import numpy as np
import juzi as jz

from anndata import AnnData


def make_anndata():
    a = np.random.normal(10., 3., size=100),
    b = np.random.normal(90., 9., size=100),
    c = np.random.normal(80., 9., size=100),
    X = np.vstack([a, a, b, b, c])
    adata = AnnData(
        X,
        var={"gene": np.arange(100).astype(str)},
        obs={"cell": 2*["a"] + 3*["b"]},
    )
    return jz.cs.nmf(
        adata,
        k=[2],
        target_sum=10,
        min_cells=2,
        key="cell",
        genes=None,
    )


class TestSimilarity:
    def test_error_missing_keys(self):
        with pytest.raises(KeyError):
            adata = make_anndata()
            del adata.uns["juzi_names"]
            jz.cs.similarity(adata, distance="jaccard")

    def test_error_min_similarity(self):
        with pytest.raises(ValueError):
            jz.cs.similarity(make_anndata(), min_similarity=10.)

    def test_error_distance(self):
        with pytest.raises(ValueError):
            jz.cs.similarity(make_anndata(), distance="wrong")

    def test_error_jaccard_top_k(self):
        with pytest.raises(ValueError):
            jz.cs.similarity(make_anndata(), distance="jaccard", top_k=None)

    def test_similarity_min(self):
        adata = make_anndata()
        jz.cs.similarity(adata, min_similarity=1.0, distance="jaccard")
        assert adata.uns["juzi_similarity"].shape[0] == 4

    def test_similarity_run(self):
        adata = make_anndata()
        jz.cs.similarity(adata, distance="jaccard")
        assert "juzi_keep" in adata.uns

    def test_similarity_threads(self):
        adata = make_anndata()
        jz.cs.similarity(
            adata,
            distance="jaccard",
            prefer="threads",
            n_jobs=-1
        )
        assert np.max(adata.uns["juzi_similarity"]) > 0.
        assert "juzi_keep" in adata.uns

    def test_similarity_processes(self):
        adata = make_anndata()
        jz.cs.similarity(
            adata,
            distance="jaccard",
            prefer="processes",
            n_jobs=-1
        )
        assert np.max(adata.uns["juzi_similarity"]) > 0.
        assert "juzi_keep" in adata.uns

    def test_similarity_custom_callable(self):
        adata = make_anndata()
        def _distance(x, y): return np.sum(x + y)
        jz.cs.similarity(
            adata,
            distance=_distance,
        )
        assert np.max(adata.uns["juzi_similarity"]) > 0.
        assert "juzi_keep" in adata.uns
