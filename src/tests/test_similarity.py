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

    adata = jz.gp.nmf(
        adata,
        key="donor_id",
        k=k,
        min_cells=10,
        genes=None,
        seed=seed,
    )

    return adata


def make_adata_pruned(seed: int = 42) -> AnnData:
    """AnnData fit with nmf and prune, ready for similarity."""
    adata = make_adata(seed=seed)
    jz.gp.prune(adata, min_k=1, min_similarity=0.3)
    return adata


# Input validation


def test_error_missing_juzi_G():
    adata = make_adata()
    del adata.varm["juzi_G"]
    with pytest.raises(KeyError):
        jz.gp.similarity(adata)


def test_error_missing_juzi_names():
    adata = make_adata()
    del adata.uns["juzi_names"]
    with pytest.raises(KeyError):
        jz.gp.similarity(adata)


def test_error_missing_juzi_k():
    adata = make_adata()
    del adata.uns["juzi_k"]
    with pytest.raises(KeyError):
        jz.gp.similarity(adata)


def test_error_invalid_distance_string():
    with pytest.raises(ValueError):
        jz.gp.similarity(make_adata(), distance="wrong")


def test_error_jaccard_requires_top_k():
    with pytest.raises(ValueError):
        jz.gp.similarity(make_adata(), distance="jaccard", top_k=None)


def test_error_top_k_below_one():
    with pytest.raises(ValueError):
        jz.gp.similarity(make_adata(), distance="jaccard", top_k=0)


def test_error_min_similarity_above_one():
    with pytest.raises(ValueError):
        jz.gp.similarity(make_adata(), min_similarity=1.1)


def test_error_min_similarity_below_zero():
    with pytest.raises(ValueError):
        jz.gp.similarity(make_adata(), min_similarity=-0.1)


def test_error_callable_wrong_signature():
    def bad_distance(x):
        return np.sum(x)

    with pytest.raises(ValueError):
        jz.gp.similarity(make_adata(), distance=bad_distance)


def test_error_callable_non_scalar_return():
    def bad_distance(x, y):
        return x + y  # returns array not scalar

    with pytest.raises(ValueError):
        jz.gp.similarity(make_adata(), distance=bad_distance)


# Output structure


def test_similarity_matrix_in_uns():
    adata = make_adata()
    jz.gp.similarity(adata)
    assert "juzi_similarity" in adata.uns


def test_juzi_keep_in_uns():
    adata = make_adata()
    jz.gp.similarity(adata)
    assert "juzi_keep" in adata.uns


def test_similarity_matrix_shape():
    adata = make_adata(n_samples=2, k=[3])
    jz.gp.similarity(adata)
    n_factors = adata.varm["juzi_G"].shape[1]
    sim = adata.uns["juzi_similarity"]
    assert sim.shape == (n_factors, n_factors)


def test_similarity_matrix_symmetric():
    adata = make_adata()
    jz.gp.similarity(adata)
    sim = adata.uns["juzi_similarity"]
    np.testing.assert_allclose(sim, sim.T, atol=1e-6)


def test_similarity_matrix_diagonal_zero():
    """Diagonal should be zero — self-similarity is not computed."""
    adata = make_adata()
    jz.gp.similarity(adata)
    sim = adata.uns["juzi_similarity"]
    np.testing.assert_allclose(np.diag(sim), 0.0)


def test_similarity_matrix_non_negative():
    adata = make_adata()
    jz.gp.similarity(adata)
    assert (adata.uns["juzi_similarity"] >= 0).all()


def test_similarity_matrix_bounded():
    """Jaccard similarity is in [0, 1]."""
    adata = make_adata()
    jz.gp.similarity(adata, distance="jaccard")
    sim = adata.uns["juzi_similarity"]
    assert sim.max() <= 1.0 + 1e-6


def test_juzi_keep_is_bool():
    adata = make_adata()
    jz.gp.similarity(adata)
    assert adata.uns["juzi_keep"].dtype == bool


def test_juzi_keep_length_matches_factors():
    adata = make_adata(n_samples=2, k=[3, 4])
    jz.gp.similarity(adata)
    n_factors = adata.varm["juzi_G"].shape[1]
    assert len(adata.uns["juzi_keep"]) == n_factors


# juzi_keep filtering


def test_drop_zeros_removes_all_zero_rows():
    adata = make_adata()
    jz.gp.similarity(adata, drop_zeros=True)
    sim = adata.uns["juzi_similarity"]
    keep = adata.uns["juzi_keep"]
    # All kept factors must have at least one non-zero similarity
    assert not np.isclose(sim[keep], 0).all(axis=1).any()


def test_min_similarity_filters_weak_factors():
    """Factors kept after filtering must exceed min_similarity threshold."""
    adata = make_adata()
    threshold = 0.3
    jz.gp.similarity(adata, min_similarity=threshold)
    sim = adata.uns["juzi_similarity"]
    keep = adata.uns["juzi_keep"]
    assert (sim[keep].max(axis=1) >= threshold).all()


def test_strict_min_similarity_keeps_fewer():
    adata_low = make_adata(seed=0)
    adata_high = make_adata(seed=0)
    jz.gp.similarity(adata_low, min_similarity=0.1)
    jz.gp.similarity(adata_high, min_similarity=0.9)
    assert adata_high.uns["juzi_keep"].sum() <= adata_low.uns["juzi_keep"].sum()


def test_juzi_keep_preserved_from_prune():
    """Factors already masked by prune should remain masked after similarity."""
    adata = make_adata_pruned()
    keep_before = adata.uns["juzi_keep"].copy()
    jz.gp.similarity(adata, min_similarity=0.0, drop_zeros=False)
    keep_after = adata.uns["juzi_keep"]
    # Anything False before prune must still be False after similarity
    assert not keep_after[~keep_before].any()


# intra_sample


def test_intra_sample_true_computes_all_pairs():
    adata = make_adata(n_samples=2, k=[2])
    jz.gp.similarity(adata, intra_sample=True)
    sim = adata.uns["juzi_similarity"]
    # With intra_sample=True some intra-sample entries should be non-zero
    names = np.array(adata.uns["juzi_names"])
    n = len(names)
    intra = np.array(
        [sim[i, j] for i in range(n) for j in range(i + 1, n) if names[i] == names[j]]
    )
    assert intra.max() > 0


def test_intra_sample_false_zeros_intra_pairs():
    adata = make_adata(n_samples=2, k=[2])
    jz.gp.similarity(adata, intra_sample=False)
    sim = adata.uns["juzi_similarity"]
    names = np.array(adata.uns["juzi_names"])
    n = len(names)
    for i in range(n):
        for j in range(n):
            if names[i] == names[j] and i != j:
                assert sim[i, j] == 0.0


# Distance metrics


def test_jaccard_distance():
    adata = make_adata()
    jz.gp.similarity(adata, distance="jaccard", top_k=20)
    assert adata.uns["juzi_similarity"].max() > 0


def test_callable_distance():
    adata = make_adata()

    def cosine(x, y):
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return float(np.dot(x, y) / denom) if denom > 0 else 0.0

    jz.gp.similarity(adata, distance=cosine)
    assert adata.uns["juzi_similarity"].max() > 0


def test_callable_distance_with_top_k():
    """Callable distance with top_k set should subset vectors before calling."""
    adata = make_adata()

    called_sizes = []

    def recording_distance(x, y):
        called_sizes.append(len(x))
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))

    jz.gp.similarity(adata, distance=recording_distance, top_k=10)
    # All calls should use the union of top-10 genes, so vector length <= 20
    assert all(s <= 20 for s in called_sizes)


# Parallelisation


def test_parallel_threads():
    adata = make_adata()
    jz.gp.similarity(adata, n_jobs=2, prefer="threads")
    assert adata.uns["juzi_similarity"].max() > 0


def test_parallel_processes():
    adata = make_adata()
    jz.gp.similarity(adata, n_jobs=2, prefer="processes")
    assert adata.uns["juzi_similarity"].max() > 0


def test_parallel_matches_serial():
    adata_serial = make_adata(seed=0)
    adata_parallel = make_adata(seed=0)

    jz.gp.similarity(adata_serial, n_jobs=1)
    jz.gp.similarity(adata_parallel, n_jobs=2, prefer="threads")

    np.testing.assert_allclose(
        adata_serial.uns["juzi_similarity"],
        adata_parallel.uns["juzi_similarity"],
        atol=1e-6,
    )


# copy parameter


def test_copy_false_modifies_inplace():
    adata = make_adata()
    result = jz.gp.similarity(adata, copy=False)
    assert result is None
    assert "juzi_similarity" in adata.uns


def test_copy_true_returns_new_object():
    adata = make_adata()
    result = jz.gp.similarity(adata, copy=True)
    assert result is not None
    assert "juzi_similarity" not in adata.uns
    assert "juzi_similarity" in result.uns
