import pytest
import numpy as np
from anndata import AnnData
import juzi as jz


# Fixtures


def make_adata(
    n_cells_per_sample: int = 30,
    n_genes: int = 100,
    n_samples: int = 3,
    k: list[int] = [2, 3],
    seed: int = 42,
) -> AnnData:
    """AnnData fit with juzi.gp.nmf, ready for similarity computation."""
    rng = np.random.default_rng(seed)

    sample_names = [f"sample_{chr(97+i)}" for i in range(n_samples)]
    blocks, labels = [], []

    for i, sample in enumerate(sample_names):
        mean = 10.0 if i % 2 == 0 else 90.0
        X_sample = (
            rng.negative_binomial(
                n=5, p=0.5, size=(n_cells_per_sample, n_genes)
            ).astype(np.float32)
            + mean
        )
        blocks.append(X_sample)
        labels.extend([sample] * n_cells_per_sample)

    adata = AnnData(
        X=np.vstack(blocks),
        obs={"donor_id": labels},
        var={"gene_name": np.arange(n_genes).astype(str)},
    )

    return jz.gp.nmf_fit(
        adata,
        key="donor_id",
        k=k,
        min_cells=10,
        genes=None,
        seed=seed,
    )


def make_adata_pruned(seed: int = 42) -> AnnData:
    """AnnData fit with nmf and prune, ready for similarity."""
    adata = make_adata(seed=seed)
    jz.gp.nmf_prune(adata, min_k=1, min_similarity=0.3)
    return adata


# Input validation


def test_error_missing_juzi_G():
    adata = make_adata()
    del adata.varm["juzi_G"]
    with pytest.raises(KeyError):
        jz.gp.similarity_compute(adata)


def test_error_missing_juzi_names():
    adata = make_adata()
    del adata.uns["juzi_names"]
    with pytest.raises(KeyError):
        jz.gp.similarity_compute(adata)


def test_error_missing_juzi_k():
    adata = make_adata()
    del adata.uns["juzi_k"]
    with pytest.raises(KeyError):
        jz.gp.similarity_compute(adata)


def test_error_invalid_distance_string():
    with pytest.raises(ValueError):
        jz.gp.similarity_compute(make_adata(), distance="wrong")


def test_error_jaccard_requires_top_k():
    with pytest.raises(ValueError):
        jz.gp.similarity_compute(make_adata(), distance="jaccard", top_k=None)


def test_error_top_k_below_one():
    with pytest.raises(ValueError):
        jz.gp.similarity_compute(make_adata(), distance="jaccard", top_k=0)


def test_error_callable_wrong_signature():
    def bad_distance(x):
        return np.sum(x)

    with pytest.raises(ValueError):
        jz.gp.similarity_compute(make_adata(), distance=bad_distance)


def test_error_callable_non_scalar_return():
    def bad_distance(x, y):
        return x + y

    with pytest.raises(ValueError):
        jz.gp.similarity_compute(make_adata(), distance=bad_distance)


# select_similarity input validation


def test_select_similarity_error_missing_juzi_similarity():
    adata = make_adata()
    with pytest.raises(KeyError):
        jz.gp.similarity_filter(adata, min_similarity=0.3)


def test_select_similarity_error_above_one():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    with pytest.raises(ValueError):
        jz.gp.similarity_filter(adata, min_similarity=1.1)


def test_select_similarity_error_below_zero():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    with pytest.raises(ValueError):
        jz.gp.similarity_filter(adata, min_similarity=-0.1)


# Output structure


def test_similarity_matrix_in_uns():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    assert "juzi_similarity" in adata.uns


def test_similarity_idx_in_uns():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    assert "juzi_similarity_idx" in adata.uns


def test_juzi_keep_similarity_in_uns():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    assert "juzi_keep_similarity" in adata.uns


def test_juzi_keep_in_uns():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    assert "juzi_keep" in adata.uns


def test_similarity_matrix_shape_matches_kept_factors():
    """juzi_similarity shape must be (n_kept × n_kept) not (n_total × n_total)."""
    adata = make_adata_pruned()
    jz.gp.similarity_compute(adata)
    n_kept = adata.uns["juzi_keep"].sum()
    sim = adata.uns["juzi_similarity"]
    assert sim.shape == (n_kept, n_kept)


def test_similarity_idx_length_matches_kept_factors():
    adata = make_adata_pruned()
    jz.gp.similarity_compute(adata)
    n_kept = adata.uns["juzi_keep"].sum()
    assert len(adata.uns["juzi_similarity_idx"]) == n_kept


def test_similarity_idx_are_valid_global_indices():
    """sim_idx values must be valid indices into juzi_G columns."""
    adata = make_adata_pruned()
    jz.gp.similarity_compute(adata)
    n_total = adata.varm["juzi_G"].shape[1]
    sim_idx = adata.uns["juzi_similarity_idx"]
    assert (sim_idx >= 0).all()
    assert (sim_idx < n_total).all()


def test_similarity_idx_matches_juzi_keep():
    """sim_idx must be exactly the global indices where juzi_keep is True."""
    adata = make_adata_pruned()
    jz.gp.similarity_compute(adata)
    expected = np.where(adata.uns["juzi_keep"])[0]
    np.testing.assert_array_equal(adata.uns["juzi_similarity_idx"], expected)


def test_similarity_matrix_symmetric():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    sim = adata.uns["juzi_similarity"]
    np.testing.assert_allclose(sim, sim.T, atol=1e-6)


def test_similarity_matrix_diagonal_zero():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    sim = adata.uns["juzi_similarity"]
    np.testing.assert_allclose(np.diag(sim), 0.0)


def test_similarity_matrix_non_negative():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    assert (adata.uns["juzi_similarity"] >= 0).all()


def test_similarity_matrix_bounded():
    adata = make_adata()
    jz.gp.similarity_compute(adata, distance="jaccard")
    assert adata.uns["juzi_similarity"].max() <= 1.0 + 1e-6


def test_juzi_keep_similarity_is_bool():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    assert adata.uns["juzi_keep_similarity"].dtype == bool


def test_juzi_keep_similarity_length_is_n_total():
    """juzi_keep_similarity must be length n_total not n_kept."""
    adata = make_adata_pruned()
    jz.gp.similarity_compute(adata)
    n_total = adata.varm["juzi_G"].shape[1]
    assert len(adata.uns["juzi_keep_similarity"]) == n_total


def test_juzi_keep_similarity_false_outside_sim_idx():
    """Factors not in sim_idx must be False in juzi_keep_similarity."""
    adata = make_adata_pruned()
    jz.gp.similarity_compute(adata)
    sim_idx = adata.uns["juzi_similarity_idx"]
    n_total = adata.varm["juzi_G"].shape[1]
    outside = np.ones(n_total, dtype=bool)
    outside[sim_idx] = False
    assert not adata.uns["juzi_keep_similarity"][outside].any()


def test_juzi_keep_is_intersection():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    expected = (
        adata.uns["juzi_keep_prune"]
        & adata.uns["juzi_keep_similarity"]
        & adata.uns["juzi_keep_cluster"]
    )
    np.testing.assert_array_equal(adata.uns["juzi_keep"], expected)


def test_upstream_masks_not_modified_by_similarity():
    adata = make_adata_pruned()
    prune_before = adata.uns["juzi_keep_prune"].copy()
    jz.gp.similarity_compute(adata)
    np.testing.assert_array_equal(adata.uns["juzi_keep_prune"], prune_before)


# drop_zeros


def test_drop_zeros_removes_all_zero_rows():
    adata = make_adata()
    jz.gp.similarity_compute(adata, drop_zeros=True)
    sim = adata.uns["juzi_similarity"]
    sim_idx = adata.uns["juzi_similarity_idx"]
    keep = adata.uns["juzi_keep_similarity"]
    # Map keep back to local indices
    local_keep = keep[sim_idx]
    assert not np.isclose(sim[local_keep], 0).all(axis=1).any()


def test_drop_zeros_false_all_sim_idx_kept():
    adata = make_adata()
    jz.gp.similarity_compute(adata, drop_zeros=False)
    sim_idx = adata.uns["juzi_similarity_idx"]
    keep = adata.uns["juzi_keep_similarity"]
    assert keep[sim_idx].all()


# select_similarity


def test_select_similarity_filters_weak_factors():
    adata = make_adata()
    threshold = 0.3
    jz.gp.similarity_compute(adata, drop_zeros=False)
    jz.gp.similarity_filter(adata, min_similarity=threshold)
    sim = adata.uns["juzi_similarity"]
    sim_idx = adata.uns["juzi_similarity_idx"]
    keep = adata.uns["juzi_keep_similarity"]
    local_keep = keep[sim_idx]
    assert (sim[local_keep].max(axis=1) >= threshold).all()


def test_select_similarity_strict_keeps_fewer():
    adata_low = make_adata(seed=0)
    adata_high = make_adata(seed=0)
    jz.gp.similarity_compute(adata_low, drop_zeros=False)
    jz.gp.similarity_compute(adata_high, drop_zeros=False)
    jz.gp.similarity_filter(adata_low, min_similarity=0.1)
    jz.gp.similarity_filter(adata_high, min_similarity=0.9)
    assert (
        adata_high.uns["juzi_keep_similarity"].sum()
        <= adata_low.uns["juzi_keep_similarity"].sum()
    )


def test_select_similarity_reruns_update_keep():
    adata = make_adata()
    jz.gp.similarity_compute(adata, drop_zeros=False)
    jz.gp.similarity_filter(adata, min_similarity=0.1)
    keep_loose = adata.uns["juzi_keep"].sum()
    jz.gp.similarity_filter(adata, min_similarity=0.9)
    keep_strict = adata.uns["juzi_keep"].sum()
    assert keep_strict <= keep_loose


def test_select_similarity_preserves_drop_zeros():
    adata = make_adata()
    jz.gp.similarity_compute(adata, drop_zeros=True)
    drop_zeros_mask = adata.uns["juzi_keep_similarity"].copy()
    jz.gp.similarity_filter(adata, min_similarity=0.0)
    np.testing.assert_array_equal(
        adata.uns["juzi_keep_similarity"],
        drop_zeros_mask,
    )


def test_select_similarity_recomputes_juzi_keep():
    adata = make_adata_pruned()
    jz.gp.similarity_compute(adata, drop_zeros=False)
    jz.gp.similarity_filter(adata, min_similarity=0.3)
    expected = (
        adata.uns["juzi_keep_prune"]
        & adata.uns["juzi_keep_similarity"]
        & adata.uns["juzi_keep_cluster"]
    )
    np.testing.assert_array_equal(adata.uns["juzi_keep"], expected)


def test_select_similarity_does_not_modify_prune_mask():
    adata = make_adata_pruned()
    jz.gp.similarity_compute(adata)
    prune_before = adata.uns["juzi_keep_prune"].copy()
    jz.gp.similarity_filter(adata, min_similarity=0.3)
    np.testing.assert_array_equal(adata.uns["juzi_keep_prune"], prune_before)


def test_select_similarity_copy_false():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    result = jz.gp.similarity_filter(adata, min_similarity=0.3, copy=False)
    assert result is None
    assert "juzi_keep_similarity" in adata.uns


def test_select_similarity_copy_true():
    adata = make_adata()
    jz.gp.similarity_compute(adata)
    keep_before = adata.uns["juzi_keep_similarity"].copy()
    result = jz.gp.similarity_filter(adata, min_similarity=0.9, copy=True)
    assert result is not None
    np.testing.assert_array_equal(adata.uns["juzi_keep_similarity"], keep_before)


def test_juzi_keep_preserved_from_prune():
    adata = make_adata_pruned()
    keep_before = adata.uns["juzi_keep"].copy()
    jz.gp.similarity_compute(adata, drop_zeros=False)
    jz.gp.similarity_filter(adata, min_similarity=0.0)
    keep_after = adata.uns["juzi_keep"]
    assert not keep_after[~keep_before].any()


# intra_sample


def test_intra_sample_true_computes_all_pairs():
    adata = make_adata(n_samples=2, k=[2])
    jz.gp.similarity_compute(adata, intra_sample=True)
    sim = adata.uns["juzi_similarity"]
    sim_idx = adata.uns["juzi_similarity_idx"]
    names = np.array(adata.uns["juzi_names"])[sim_idx]
    n = len(names)
    intra = np.array(
        [sim[i, j] for i in range(n) for j in range(i + 1, n) if names[i] == names[j]]
    )
    assert intra.max() > 0


def test_intra_sample_false_zeros_intra_pairs():
    adata = make_adata(n_samples=2, k=[2])
    jz.gp.similarity_compute(adata, intra_sample=False)
    sim = adata.uns["juzi_similarity"]
    sim_idx = adata.uns["juzi_similarity_idx"]
    names = np.array(adata.uns["juzi_names"])[sim_idx]
    n = len(names)
    for i in range(n):
        for j in range(n):
            if names[i] == names[j] and i != j:
                assert sim[i, j] == 0.0


# Distance metrics


def test_jaccard_distance():
    adata = make_adata()
    jz.gp.similarity_compute(adata, distance="jaccard", top_k=20)
    assert adata.uns["juzi_similarity"].max() > 0


def test_callable_distance():
    adata = make_adata()

    def cosine(x, y):
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return float(np.dot(x, y) / denom) if denom > 0 else 0.0

    jz.gp.similarity_compute(adata, distance=cosine)
    assert adata.uns["juzi_similarity"].max() > 0


def test_callable_distance_with_top_k():
    adata = make_adata()
    called_sizes = []

    def recording_distance(x, y):
        called_sizes.append(len(x))
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))

    jz.gp.similarity_compute(adata, distance=recording_distance, top_k=10)
    assert all(s <= 20 for s in called_sizes)


# Parallelisation


def test_parallel_threads():
    adata = make_adata()
    jz.gp.similarity_compute(adata, n_jobs=2, prefer="threads")
    assert adata.uns["juzi_similarity"].max() > 0


def test_parallel_processes():
    adata = make_adata()
    jz.gp.similarity_compute(adata, n_jobs=2, prefer="processes")
    assert adata.uns["juzi_similarity"].max() > 0


def test_parallel_matches_serial():
    adata_serial = make_adata(seed=0)
    adata_parallel = make_adata(seed=0)
    jz.gp.similarity_compute(adata_serial, n_jobs=1)
    jz.gp.similarity_compute(adata_parallel, n_jobs=2, prefer="threads")
    np.testing.assert_allclose(
        adata_serial.uns["juzi_similarity"],
        adata_parallel.uns["juzi_similarity"],
        atol=1e-6,
    )


# copy parameter


def test_copy_false_modifies_inplace():
    adata = make_adata()
    result = jz.gp.similarity_compute(adata, copy=False)
    assert result is None
    assert "juzi_similarity" in adata.uns


def test_copy_true_returns_new_object():
    adata = make_adata()
    result = jz.gp.similarity_compute(adata, copy=True)
    assert result is not None
    assert "juzi_similarity" not in adata.uns
    assert "juzi_similarity" in result.uns
