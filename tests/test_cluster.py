import pytest
import numpy as np
import juzi as jz

from anndata import AnnData


def make_anndata(k=[2]):
    np.random.seed(123456)
    a = np.random.normal(5., 4., size=(100, 8)).T
    b = np.random.beta(5., 1., size=(100, 8)).T
    X = np.vstack((4*[a] + 4*[b]))
    adata = AnnData(
        X,
        var={"gene": np.arange(100).astype(str)},
        obs={"cell": np.repeat(np.arange(8).astype(str), 8)}
    )
    adata = jz.cs.nmf(
        adata,
        k=k,
        init='nndsvda',
        target_sum=100,
        center=False,
        min_cells=2,
        key="cell",
        genes=None,
    )
    jz.cs.similarity(
        adata,
        top_k=10,
        intra_sample=False,
        distance="jaccard",
        min_similarity=0.,
        drop_zeros=False,
    )
    return adata


class TestCluster:
    def test_error_missing_keys(self):
        adata = make_anndata()
        del adata.uns["juzi_names"]
        with pytest.raises(KeyError):
            jz.cs.cluster(adata, min_cluster=1)

    def test_error_threshold(self):
        adata = make_anndata()
        with pytest.raises(ValueError):
            jz.cs.cluster(adata, threshold=10., min_cluster=1)

    def test_cluster_correct(self):
        adata = make_anndata()

        # Given we have two sets of duplicated adata and given that we are
        # learning a two components per sample - we would expect 4 clusters
        # with equal size sub-clusters (i.e. 16 samples / 4 = 4 per cluster)
        print(adata.uns["juzi_similarity"].shape)
        jz.cs.cluster(adata, threshold=0.5, min_cluster=1)

        assert "juzi_cluster_stats" in adata.uns
        assert "juzi_cluster_names" in adata.uns
        assert "juzi_cluster_labels" in adata.uns
        assert "juzi_cluster_G" in adata.uns
        assert "juzi_cluster_similarity" in adata.uns

        labels, counts = np.unique(
            adata.uns["juzi_cluster_labels"], return_counts=True)

        assert len(labels) == 4
        assert np.min(counts) == 4
        assert np.max(counts) == 4
        assert np.all(labels == np.array([0, 1, 2, 3]))

    def test_cluster_overcluster(self):
        adata = make_anndata()

        # A threshold of 1.0 should mean every sample is a single cluster
        jz.cs.cluster(adata, threshold=1.0, min_cluster=1)

        assert "juzi_cluster_stats" in adata.uns
        assert "juzi_cluster_names" in adata.uns
        assert "juzi_cluster_labels" in adata.uns
        assert "juzi_cluster_G" in adata.uns
        assert "juzi_cluster_similarity" in adata.uns

        labels, counts = np.unique(
            adata.uns["juzi_cluster_labels"], return_counts=True)

        assert len(labels) == len(adata.uns["juzi_names"])
        assert np.all(counts == 1)
