import numpy as np
import pytest
import juzi as jz


# Strategy validation


def test_programs_cluster_invalid_strategy_raises(adata_similarity):
    with pytest.raises(ValueError):
        jz.gp.programs_cluster(adata_similarity, strategy="bad")


# Centroid outputs


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
        "juzi_cluster_order",
    ]:
        assert key in adata_cluster_centroid.uns, f"Missing key: {key}"

    assert adata_cluster_centroid.uns["juzi_cluster_meta"]["strategy"] == "centroid"

    # Centroid mode must not write progressive-only fields
    assert "juzi_cluster_mp_genes" not in adata_cluster_centroid.uns


# Progressive outputs


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
        "juzi_cluster_order",
    ]:
        assert key in adata_cluster_progressive.uns, f"Missing key: {key}"

    assert (
        adata_cluster_progressive.uns["juzi_cluster_meta"]["strategy"] == "progressive"
    )


def test_progressive_mp_genes_match_cluster_genes(adata_cluster_progressive):
    # juzi_cluster_mp_genes and juzi_cluster_genes must be identical in
    # progressive mode — they are the same object written under two keys
    # for backward compatibility.
    mp = adata_cluster_progressive.uns["juzi_cluster_mp_genes"]
    cg = adata_cluster_progressive.uns["juzi_cluster_genes"]
    assert set(mp.keys()) == set(cg.keys())
    for k in mp:
        assert mp[k] == cg[k], f"Mismatch for cluster {k}"


def test_progressive_genes_ordered_by_frequency(adata_cluster_progressive):
    # Gene lists in progressive mode must be in descending frequency order.
    # We cannot recompute frequency directly here, but we can verify that
    # the stored gene list is a plain list (not a set) and has no duplicates,
    # which is necessary for the ordering contract to be meaningful.
    cluster_genes = adata_cluster_progressive.uns["juzi_cluster_genes"]
    for cluster_id, genes in cluster_genes.items():
        assert isinstance(genes, list), f"Cluster {cluster_id} genes is not a list"
        assert len(genes) == len(
            set(genes)
        ), f"Cluster {cluster_id} gene list contains duplicates"


def test_progressive_default_min_founder_overlaps(adata_similarity):
    # Default min_founder_overlaps is 6 (strictly > 5), matching the paper.
    # Verify the parameter is stored correctly in meta.
    adata = adata_similarity.copy()
    jz.gp.programs_cluster(
        adata,
        strategy="progressive",
        min_cluster=1,
        copy=False,
    )
    meta = adata.uns["juzi_cluster_meta"]
    assert meta["min_founder_overlaps"] == 6


# Shared contracts for both strategies


def test_cluster_genes_are_canonical_for_both_strategies(
    adata_cluster_centroid, adata_cluster_progressive
):
    for adata in [adata_cluster_centroid, adata_cluster_progressive]:
        cg = adata.uns["juzi_cluster_genes"]
        assert isinstance(cg, dict)
        assert set(cg.keys()) == {
            int(c) for c in np.unique(adata.uns["juzi_cluster_labels"])
        }


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


def test_heatmap_order_length_matches_labels(
    adata_cluster_centroid, adata_cluster_progressive
):
    # juzi_cluster_order holds global varm indices, not a local arange.
    # Assert length matches label count and all indices are valid.
    for adata in [adata_cluster_centroid, adata_cluster_progressive]:
        order = adata.uns["juzi_cluster_order"]
        n = len(adata.uns["juzi_cluster_labels"])
        n_total = adata.varm["juzi_G"].shape[1]
        assert len(order) == n
        assert np.all(order >= 0) and np.all(order < n_total)
        assert len(np.unique(order)) == n  # no duplicate global indices


def test_heatmap_order_clusters_are_contiguous_blocks(
    adata_cluster_centroid, adata_cluster_progressive
):
    # After reordering, each cluster must occupy a single contiguous block
    # in juzi_cluster_labels (no interleaving of different cluster labels).
    for adata in [adata_cluster_centroid, adata_cluster_progressive]:
        labels = adata.uns["juzi_cluster_labels"]
        seen = set()
        prev = labels[0]
        for label in labels:
            if label != prev:
                assert label not in seen, (
                    f"Cluster {label} appears non-contiguously in "
                    "juzi_cluster_labels — reordering failed"
                )
                seen.add(prev)
                prev = label


def test_cluster_similarity_shape_matches_labels(
    adata_cluster_centroid, adata_cluster_progressive
):
    for adata in [adata_cluster_centroid, adata_cluster_progressive]:
        n = len(adata.uns["juzi_cluster_labels"])
        S = adata.uns["juzi_cluster_similarity"]
        assert S.shape == (n, n), (
            f"Similarity matrix shape {S.shape} does not match " f"label count {n}"
        )


def test_cluster_names_length_matches_labels(
    adata_cluster_centroid, adata_cluster_progressive
):
    for adata in [adata_cluster_centroid, adata_cluster_progressive]:
        assert len(adata.uns["juzi_cluster_names"]) == len(
            adata.uns["juzi_cluster_labels"]
        )


def test_cluster_G_shape(adata_cluster_centroid, adata_cluster_progressive):
    for adata in [adata_cluster_centroid, adata_cluster_progressive]:
        n_clusters = len(np.unique(adata.uns["juzi_cluster_labels"]))
        n_genes = adata.varm["juzi_G"].shape[0]
        assert adata.uns["juzi_cluster_G"].shape == (n_clusters, n_genes)


# programs_threshold


def test_programs_threshold_runs_for_centroid(adata_similarity):
    optimal = jz.gp.programs_threshold(adata_similarity, min_cluster=1)
    assert isinstance(optimal, float)
    assert "juzi_threshold_sweep" in adata_similarity.uns


def test_programs_threshold_sweep_fields(adata_similarity):
    jz.gp.programs_threshold(adata_similarity, min_cluster=1)
    sweep = adata_similarity.uns["juzi_threshold_sweep"]
    for key in ["thresholds", "metric", "metric_name", "optimal", "method"]:
        assert key in sweep, f"Missing sweep key: {key}"


def test_programs_threshold_does_not_modify_cluster_state(adata_cluster_centroid):
    # programs_threshold must not overwrite existing cluster results.
    labels_before = adata_cluster_centroid.uns["juzi_cluster_labels"].copy()
    jz.gp.programs_threshold(adata_cluster_centroid, min_cluster=1)
    np.testing.assert_array_equal(
        adata_cluster_centroid.uns["juzi_cluster_labels"], labels_before
    )


# programs_merge


def test_programs_merge_reduces_cluster_count(adata_cluster_progressive):
    adata = adata_cluster_progressive.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test merge.")
    n_before = len(uniq)
    jz.gp.programs_merge(adata, [int(uniq[0]), int(uniq[1])])
    n_after = len(np.unique(adata.uns["juzi_cluster_labels"]))
    assert n_after == n_before - 1


def test_programs_merge_labels_remain_contiguous(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test merge.")
    jz.gp.programs_merge(adata, [int(uniq[0]), int(uniq[1])])
    uniq_after = np.unique(adata.uns["juzi_cluster_labels"])
    np.testing.assert_array_equal(uniq_after, np.arange(len(uniq_after)))


def test_programs_merge_sets_posthoc_flag(adata_cluster_progressive):
    adata = adata_cluster_progressive.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test merge.")
    jz.gp.programs_merge(adata, [int(uniq[0]), int(uniq[1])])
    assert adata.uns["juzi_cluster_meta"].get("posthoc_merge") is True


def test_programs_merge_invalid_label_raises(adata_cluster_centroid):
    with pytest.raises(ValueError, match="not found"):
        jz.gp.programs_merge(adata_cluster_centroid, [999, 0])


def test_programs_merge_single_label_raises(adata_cluster_centroid):
    with pytest.raises(ValueError, match="at least 2"):
        jz.gp.programs_merge(adata_cluster_centroid, [0])


def test_programs_merge_preserves_heatmap_order_field(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test merge.")
    jz.gp.programs_merge(adata, [int(uniq[0]), int(uniq[1])])
    assert "juzi_cluster_order" in adata.uns
    n = len(adata.uns["juzi_cluster_labels"])
    assert len(adata.uns["juzi_cluster_order"]) == n


# programs_remove


def test_programs_remove_reduces_cluster_count(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test remove.")
    n_before = len(uniq)
    jz.gp.programs_remove(adata, [int(uniq[0])])
    n_after = len(np.unique(adata.uns["juzi_cluster_labels"]))
    assert n_after == n_before - 1


def test_programs_remove_labels_remain_contiguous(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test remove.")
    jz.gp.programs_remove(adata, [int(uniq[0])])
    uniq_after = np.unique(adata.uns["juzi_cluster_labels"])
    np.testing.assert_array_equal(uniq_after, np.arange(len(uniq_after)))


def test_programs_remove_sets_posthoc_flag(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test remove.")
    jz.gp.programs_remove(adata, [int(uniq[0])])
    assert adata.uns["juzi_cluster_meta"].get("posthoc_remove") is True


def test_programs_remove_all_raises(adata_cluster_centroid):
    uniq = np.unique(adata_cluster_centroid.uns["juzi_cluster_labels"]).tolist()
    with pytest.raises(ValueError, match="Cannot remove all"):
        jz.gp.programs_remove(adata_cluster_centroid, [int(c) for c in uniq])


def test_programs_remove_invalid_label_raises(adata_cluster_centroid):
    with pytest.raises(ValueError, match="not found"):
        jz.gp.programs_remove(adata_cluster_centroid, [999])


def test_programs_remove_duplicate_labels_raises(adata_cluster_centroid):
    uniq = np.unique(adata_cluster_centroid.uns["juzi_cluster_labels"])
    if len(uniq) < 1:
        pytest.skip("Need at least one program.")
    with pytest.raises(ValueError, match="duplicate"):
        jz.gp.programs_remove(adata_cluster_centroid, [int(uniq[0]), int(uniq[0])])


def test_programs_remove_updates_keep_cluster(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test remove.")
    keep_before = adata.uns["juzi_keep_cluster"].sum()
    jz.gp.programs_remove(adata, [int(uniq[0])])
    keep_after = adata.uns["juzi_keep_cluster"].sum()
    assert keep_after < keep_before


def test_programs_remove_preserves_heatmap_order_field(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    uniq = np.unique(adata.uns["juzi_cluster_labels"])
    if len(uniq) < 2:
        pytest.skip("Need at least two programs to test remove.")
    jz.gp.programs_remove(adata, [int(uniq[0])])
    assert "juzi_cluster_order" in adata.uns
    n = len(adata.uns["juzi_cluster_labels"])
    assert len(adata.uns["juzi_cluster_order"]) == n
