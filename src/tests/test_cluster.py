import numpy as np
import pytest
import juzi as jz


def test_programs_cluster_invalid_strategy_raises(adata_similarity):
    with pytest.raises(ValueError):
        jz.gp.programs_cluster(adata_similarity, strategy="bad")


def test_centroid_cluster_outputs(adata_cluster_centroid):
    for key in [
        "juzi_cluster_similarity",
        "juzi_cluster_labels",
        "juzi_cluster_names",
        "juzi_cluster_G",
        "juzi_cluster_genes",
        "juzi_cluster_samples",
        "juzi_cluster_stats",
        "juzi_cluster_meta",
    ]:
        assert key in adata_cluster_centroid.uns
    assert adata_cluster_centroid.uns["juzi_cluster_meta"]["strategy"] == "centroid"
    assert "juzi_cluster_mp_genes" not in adata_cluster_centroid.uns


def test_progressive_cluster_outputs(adata_cluster_progressive):
    for key in [
        "juzi_cluster_similarity",
        "juzi_cluster_labels",
        "juzi_cluster_names",
        "juzi_cluster_G",
        "juzi_cluster_genes",
        "juzi_cluster_samples",
        "juzi_cluster_stats",
        "juzi_cluster_meta",
        "juzi_cluster_mp_genes",
    ]:
        assert key in adata_cluster_progressive.uns
    assert adata_cluster_progressive.uns["juzi_cluster_meta"]["strategy"] == "progressive"


def test_cluster_genes_are_canonical_for_both_strategies(adata_cluster_centroid, adata_cluster_progressive):
    for adata in [adata_cluster_centroid, adata_cluster_progressive]:
        assert isinstance(adata.uns["juzi_cluster_genes"], dict)
        assert set(adata.uns["juzi_cluster_genes"].keys()) == set(
            int(c) for c in np.unique(adata.uns["juzi_cluster_labels"])
        )


def test_cluster_keep_is_global_length(adata_cluster_centroid):
    n_total = adata_cluster_centroid.varm["juzi_G"].shape[1]
    assert len(adata_cluster_centroid.uns["juzi_keep_cluster"]) == n_total
    assert adata_cluster_centroid.uns["juzi_keep_cluster"].dtype == bool


def test_cluster_keep_updates_master_keep(adata_cluster_centroid):
    expected = (
        adata_cluster_centroid.uns["juzi_keep_prune"]
        & adata_cluster_centroid.uns["juzi_keep_similarity"]
        & adata_cluster_centroid.uns["juzi_keep_cluster"]
    )
    np.testing.assert_array_equal(adata_cluster_centroid.uns["juzi_keep"], expected)


def test_centroid_cluster_labels_contiguous(adata_cluster_centroid):
    uniq = np.unique(adata_cluster_centroid.uns["juzi_cluster_labels"])
    np.testing.assert_array_equal(uniq, np.arange(len(uniq)))


def test_progressive_cluster_labels_contiguous(adata_cluster_progressive):
    uniq = np.unique(adata_cluster_progressive.uns["juzi_cluster_labels"])
    np.testing.assert_array_equal(uniq, np.arange(len(uniq)))


def test_programs_threshold_runs_for_centroid(adata_similarity):
    optimal = jz.gp.programs_threshold(adata_similarity, min_cluster=1)
    assert isinstance(optimal, float)
    assert "juzi_threshold_sweep" in adata_similarity.uns


def test_programs_merge_updates_meta_flag(adata_cluster_progressive):
    adata = adata_cluster_progressive.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test merge.")
    jz.gp.programs_merge(adata, [int(uniq[0]), int(uniq[1])])
    assert adata.uns["juzi_cluster_meta"].get("posthoc_merge", False) is True


def test_programs_remove_updates_meta_flag(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test remove.")
    jz.gp.programs_remove(adata, [int(uniq[0])])
    assert adata.uns["juzi_cluster_meta"].get("posthoc_remove", False) is True
