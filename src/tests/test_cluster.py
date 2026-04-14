import pytest
import numpy as np
from anndata import AnnData
import juzi as jz


# Fixtures


def make_adata(
    n_cells_per_sample: int = 30,
    n_genes: int = 100,
    n_samples: int = 4,
    k: list[int] = [2, 3],
    seed: int = 42,
    drop_zeros: bool = False,
    intra_sample: bool = False,
) -> AnnData:
    """AnnData fit through nmf → similarity, ready for clustering."""
    rng = np.random.default_rng(seed)

    profile_a = rng.normal(5.0,  1.0, size=(1, n_genes))
    profile_b = rng.normal(90.0, 1.0, size=(1, n_genes))

    blocks, labels = [], []
    for i in range(n_samples):
        profile  = profile_a if i % 2 == 0 else profile_b
        noise    = rng.normal(0.0, 0.5, size=(n_cells_per_sample, n_genes))
        X_sample = np.clip(profile + noise, 0, None)
        blocks.append(X_sample)
        labels.extend([f"sample_{i}"] * n_cells_per_sample)

    adata = AnnData(
        X=np.vstack(blocks).astype(np.float32),
        obs={"donor_id": labels},
        var={"gene_name": np.arange(n_genes).astype(str)},
    )

    adata = jz.gp.nmf(adata, key="donor_id", k=k, min_cells=10, genes=None, seed=seed)
    jz.gp.similarity(
        adata,
        distance="jaccard",
        top_k=20,
        intra_sample=intra_sample,
        drop_zeros=drop_zeros,
    )

    return adata


def make_adata_pruned(seed: int = 42) -> AnnData:
    adata = make_adata(seed=seed)
    jz.gp.prune(adata, min_k=1, min_similarity=0.3)
    jz.gp.similarity(adata, distance="jaccard", top_k=20)
    return adata


# Input validation


def test_error_missing_juzi_similarity():
    adata = make_adata()
    del adata.uns["juzi_similarity"]
    with pytest.raises(KeyError):
        jz.gp.cluster(adata)


def test_error_missing_juzi_names():
    adata = make_adata()
    del adata.uns["juzi_names"]
    with pytest.raises(KeyError):
        jz.gp.cluster(adata)


def test_error_missing_juzi_G():
    adata = make_adata()
    del adata.varm["juzi_G"]
    with pytest.raises(KeyError):
        jz.gp.cluster(adata)


def test_error_threshold_above_one():
    with pytest.raises(ValueError):
        jz.gp.cluster(make_adata(), threshold=1.1)


def test_error_threshold_below_zero():
    with pytest.raises(ValueError):
        jz.gp.cluster(make_adata(), threshold=-0.1)


def test_error_min_cluster_below_one():
    with pytest.raises(ValueError):
        jz.gp.cluster(make_adata(), min_cluster=0)


# Output structure


def test_output_fields_present():
    adata = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    assert "juzi_cluster_similarity" in adata.uns
    assert "juzi_cluster_labels"     in adata.uns
    assert "juzi_cluster_G"          in adata.uns
    assert "juzi_cluster_samples"    in adata.uns
    assert "juzi_cluster_stats"      in adata.uns


def test_juzi_keep_cluster_in_uns():
    adata = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    assert "juzi_keep_cluster" in adata.uns


def test_juzi_keep_cluster_is_bool():
    adata = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    assert adata.uns["juzi_keep_cluster"].dtype == bool


def test_juzi_keep_cluster_length_matches_factors():
    adata     = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    n_factors = adata.varm["juzi_G"].shape[1]
    assert len(adata.uns["juzi_keep_cluster"]) == n_factors


def test_juzi_keep_is_intersection_after_cluster():
    """juzi_keep must equal the AND of all three stage masks after cluster."""
    adata = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    expected = (
        adata.uns["juzi_keep_prune"] &
        adata.uns["juzi_keep_similarity"] &
        adata.uns["juzi_keep_cluster"]
    )
    np.testing.assert_array_equal(adata.uns["juzi_keep"], expected)


def test_upstream_masks_not_modified_by_cluster():
    """cluster must only modify juzi_keep_cluster — not prune or similarity masks."""
    adata = make_adata_pruned()
    prune_before      = adata.uns["juzi_keep_prune"].copy()
    similarity_before = adata.uns["juzi_keep_similarity"].copy()
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)
    np.testing.assert_array_equal(adata.uns["juzi_keep_prune"],      prune_before)
    np.testing.assert_array_equal(adata.uns["juzi_keep_similarity"], similarity_before)


def test_cluster_rerun_resets_keep_cluster():
    """Re-running cluster with different params should reset juzi_keep_cluster."""
    adata = make_adata(n_samples=6, seed=0)
    jz.gp.cluster(adata, threshold=0.1, min_cluster=1)
    keep_loose = adata.uns["juzi_keep_cluster"].sum()
    jz.gp.cluster(adata, threshold=0.5, min_cluster=2)
    keep_strict = adata.uns["juzi_keep_cluster"].sum()
    # Stricter params should keep fewer or equal factors
    assert keep_strict <= keep_loose


def test_cluster_labels_contiguous():
    adata = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    labels = adata.uns["juzi_cluster_labels"]
    unique = np.unique(labels)
    np.testing.assert_array_equal(unique, np.arange(len(unique)))


def test_cluster_similarity_shape():
    adata  = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    n_kept = adata.uns["juzi_keep"].sum()
    S      = adata.uns["juzi_cluster_similarity"]
    assert S.shape == (n_kept, n_kept)


def test_cluster_similarity_symmetric():
    adata = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    S = adata.uns["juzi_cluster_similarity"]
    np.testing.assert_allclose(S, S.T, atol=1e-6)


def test_cluster_G_shape():
    adata   = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    n_clust = len(np.unique(adata.uns["juzi_cluster_labels"]))
    n_genes = adata.n_vars
    assert adata.uns["juzi_cluster_G"].shape == (n_clust, n_genes)


def test_cluster_G_non_negative():
    adata = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    assert (adata.uns["juzi_cluster_G"] >= 0).all()


def test_cluster_samples_is_dict():
    adata = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    assert isinstance(adata.uns["juzi_cluster_samples"], dict)


def test_cluster_samples_keys_match_labels():
    adata   = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    labels  = np.unique(adata.uns["juzi_cluster_labels"]).tolist()
    samples = adata.uns["juzi_cluster_samples"]
    assert set(samples.keys()) == set(labels)


def test_cluster_stats_keys():
    adata = make_adata()
    jz.gp.cluster(adata, min_cluster=1)
    stats = adata.uns["juzi_cluster_stats"]
    assert "silhouette_score" in stats
    assert "inner_similarity" in stats
    assert "outer_similarity" in stats


def test_inner_similarity_geq_outer():
    adata = make_adata(n_samples=6, seed=0)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)
    stats = adata.uns["juzi_cluster_stats"]
    assert stats["inner_similarity"] >= stats["outer_similarity"]


# Clustering behaviour


def test_low_threshold_fewer_clusters():
    adata_low  = make_adata(seed=0)
    adata_high = make_adata(seed=0)
    jz.gp.cluster(adata_low,  threshold=0.1, min_cluster=1)
    jz.gp.cluster(adata_high, threshold=0.9, min_cluster=1)
    n_low  = len(np.unique(adata_low.uns["juzi_cluster_labels"]))
    n_high = len(np.unique(adata_high.uns["juzi_cluster_labels"]))
    assert n_low <= n_high


def test_threshold_one_each_factor_own_cluster():
    adata = make_adata(drop_zeros=False)
    jz.gp.cluster(adata, threshold=1.0, min_cluster=1)
    labels = adata.uns["juzi_cluster_labels"]
    assert len(np.unique(labels)) == len(labels)


def test_min_cluster_removes_small_clusters():
    adata       = make_adata(n_samples=6, seed=0)
    min_cluster = 3
    jz.gp.cluster(adata, threshold=0.3, min_cluster=min_cluster)
    labels = adata.uns["juzi_cluster_labels"]
    names  = np.array(adata.uns["juzi_names"])[adata.uns["juzi_keep"]]
    for c in np.unique(labels):
        n_unique_samples = len(np.unique(names[labels == c]))
        assert n_unique_samples >= min_cluster


def test_reorder_largest_cluster_first():
    adata = make_adata(n_samples=6, seed=0)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1, reorder=True)
    labels    = adata.uns["juzi_cluster_labels"]
    _, counts = np.unique(labels, return_counts=True)
    assert counts[0] == counts.max()


def test_reorder_false_runs():
    adata = make_adata(seed=0)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1, reorder=False)
    assert "juzi_cluster_labels" in adata.uns


def test_juzi_keep_updated_by_min_cluster():
    """juzi_keep_cluster and juzi_keep must reflect min_cluster removal."""
    adata       = make_adata(n_samples=4, seed=0)
    keep_before = adata.uns["juzi_keep"].sum()
    jz.gp.cluster(adata, threshold=0.3, min_cluster=2)
    keep_after  = adata.uns["juzi_keep"].sum()
    assert keep_after <= keep_before


def test_juzi_keep_cluster_subset_of_juzi_keep():
    """Anything False in juzi_keep_cluster must also be False in juzi_keep."""
    adata = make_adata(n_samples=6, seed=0)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=2)
    assert not adata.uns["juzi_keep"][~adata.uns["juzi_keep_cluster"]].any()


# Relabelling correctness


def test_cluster_labels_no_gaps():
    adata  = make_adata(n_samples=6, seed=0)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)
    labels = adata.uns["juzi_cluster_labels"]
    unique = np.unique(labels)
    assert unique[0] == 0
    assert unique[-1] == len(unique) - 1
    assert len(unique) == unique[-1] + 1


def test_cluster_labels_count_matches_cluster_G():
    adata    = make_adata(n_samples=6, seed=0)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)
    n_unique = len(np.unique(adata.uns["juzi_cluster_labels"]))
    assert adata.uns["juzi_cluster_G"].shape[0] == n_unique


# copy parameter


def test_copy_false_modifies_inplace():
    adata  = make_adata()
    result = jz.gp.cluster(adata, min_cluster=1, copy=False)
    assert result is None
    assert "juzi_cluster_labels" in adata.uns


def test_copy_true_returns_new_object():
    adata  = make_adata()
    result = jz.gp.cluster(adata, min_cluster=1, copy=True)
    assert result is not None
    assert "juzi_cluster_labels" not in adata.uns
    assert "juzi_cluster_labels" in result.uns


# Pipeline integration


def test_full_pipeline_runs():
    adata = make_adata_pruned()
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)
    assert "juzi_cluster_labels" in adata.uns


def test_full_pipeline_cluster_G_is_centroid():
    adata = make_adata_pruned()
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)
    keep   = adata.uns["juzi_keep"]
    labels = adata.uns["juzi_cluster_labels"]
    for c in np.unique(labels):
        actual = adata.uns["juzi_cluster_G"][c]
        assert actual.shape == (adata.n_vars,)
        assert np.isfinite(actual).all()
        assert (actual >= 0).all()


def test_cluster_rerun_does_not_affect_prune_or_similarity():
    """Confirm upstream masks are stable across multiple cluster runs."""
    adata = make_adata_pruned()
    prune_mask      = adata.uns["juzi_keep_prune"].copy()
    similarity_mask = adata.uns["juzi_keep_similarity"].copy()

    jz.gp.cluster(adata, threshold=0.1, min_cluster=1)
    jz.gp.cluster(adata, threshold=0.5, min_cluster=2)

    np.testing.assert_array_equal(adata.uns["juzi_keep_prune"],      prune_mask)
    np.testing.assert_array_equal(adata.uns["juzi_keep_similarity"], similarity_mask)
