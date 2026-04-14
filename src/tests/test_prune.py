import pytest
import numpy as np
from anndata import AnnData
import juzi as jz


# Fixtures


def make_adata(
    n_cells_per_sample: int = 30,
    n_genes: int = 100,
    n_samples: int = 2,
    k: list[int] = [2, 3],
    init: str = "nndsvda",
    seed: int = 42,
) -> AnnData:
    """AnnData already fit with juzi.gp.nmf, ready for pruning."""
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

    return jz.gp.nmf(
        adata,
        key="donor_id",
        k=k,
        init=init,
        min_cells=10,
        genes=None,
        seed=seed,
    )


# Input validation


def test_error_missing_juzi_G():
    adata = make_adata()
    del adata.varm["juzi_G"]
    with pytest.raises(KeyError):
        jz.gp.prune(adata)


def test_error_missing_juzi_names():
    adata = make_adata()
    del adata.uns["juzi_names"]
    with pytest.raises(KeyError):
        jz.gp.prune(adata)


def test_error_missing_juzi_k():
    adata = make_adata()
    del adata.uns["juzi_k"]
    with pytest.raises(KeyError):
        jz.gp.prune(adata)


def test_error_top_k_exceeds_genes():
    adata = make_adata(n_genes=100)
    with pytest.raises(ValueError):
        jz.gp.prune(adata, top_k=100_000)


def test_error_min_similarity_above_one():
    adata = make_adata()
    with pytest.raises(ValueError):
        jz.gp.prune(adata, min_similarity=1.1)


def test_error_min_similarity_below_zero():
    adata = make_adata()
    with pytest.raises(ValueError):
        jz.gp.prune(adata, min_similarity=-0.1)


def test_error_min_k_exceeds_resolutions():
    adata = make_adata(k=[2, 3])
    with pytest.raises(ValueError):
        jz.gp.prune(adata, min_k=10)


def test_error_invalid_matching():
    adata = make_adata()
    with pytest.raises(ValueError):
        jz.gp.prune(adata, matching="invalid")


# Output structure


def test_juzi_keep_prune_in_uns():
    adata = make_adata()
    jz.gp.prune(adata)
    assert "juzi_keep_prune" in adata.uns


def test_juzi_keep_in_uns():
    adata = make_adata()
    jz.gp.prune(adata)
    assert "juzi_keep" in adata.uns


def test_juzi_keep_prune_is_bool():
    adata = make_adata()
    jz.gp.prune(adata)
    assert adata.uns["juzi_keep_prune"].dtype == bool


def test_juzi_keep_is_bool():
    adata = make_adata()
    jz.gp.prune(adata)
    assert adata.uns["juzi_keep"].dtype == bool


def test_juzi_keep_prune_length_matches_factors():
    adata = make_adata(n_samples=2, k=[3, 4])
    jz.gp.prune(adata)
    n_factors = adata.varm["juzi_G"].shape[1]
    assert len(adata.uns["juzi_keep_prune"]) == n_factors


def test_juzi_keep_length_matches_factors():
    adata = make_adata(n_samples=2, k=[3, 4])
    jz.gp.prune(adata)
    n_factors = adata.varm["juzi_G"].shape[1]
    assert len(adata.uns["juzi_keep"]) == n_factors


def test_juzi_keep_is_intersection():
    """juzi_keep must equal the AND of all three stage masks after prune."""
    adata = make_adata(k=[3, 4])
    jz.gp.prune(adata, min_k=1, min_similarity=0.3)
    expected = (
        adata.uns["juzi_keep_prune"] &
        adata.uns["juzi_keep_similarity"] &
        adata.uns["juzi_keep_cluster"]
    )
    np.testing.assert_array_equal(adata.uns["juzi_keep"], expected)


def test_juzi_keep_prune_subset_of_juzi_keep():
    """juzi_keep can only be a subset of juzi_keep_prune."""
    adata = make_adata(k=[3, 4])
    jz.gp.prune(adata, min_k=1, min_similarity=0.3)
    # Anything False in juzi_keep_prune must also be False in juzi_keep
    assert not adata.uns["juzi_keep"][~adata.uns["juzi_keep_prune"]].any()


def test_upstream_masks_not_modified_by_prune():
    """prune must only modify juzi_keep_prune — not similarity or cluster masks."""
    adata = make_adata(k=[3, 4])
    # Manually set similarity and cluster masks to known values
    n = adata.varm["juzi_G"].shape[1]
    adata.uns["juzi_keep_similarity"] = np.ones(n, dtype=bool)
    adata.uns["juzi_keep_cluster"]    = np.ones(n, dtype=bool)

    jz.gp.prune(adata, min_k=1, min_similarity=0.3)

    assert adata.uns["juzi_keep_similarity"].all()
    assert adata.uns["juzi_keep_cluster"].all()


# Pruning behaviour


def test_prune_keeps_all_at_zero_threshold():
    adata = make_adata(k=[3, 4])
    jz.gp.prune(adata, min_k=1, min_similarity=0.0)
    assert adata.uns["juzi_keep_prune"].all()
    assert adata.uns["juzi_keep"].all()


def test_prune_drops_all_at_impossible_threshold():
    adata = make_adata(k=[3, 4, 5], init="random", seed=0)
    jz.gp.prune(adata, min_k=3, min_similarity=1.0)
    assert not adata.uns["juzi_keep_prune"].any()
    assert not adata.uns["juzi_keep"].any()


def test_prune_single_k_keeps_all():
    adata = make_adata(k=[5])
    jz.gp.prune(adata, min_k=1, min_similarity=0.5)
    assert adata.uns["juzi_keep_prune"].all()


def test_prune_result_is_subset_of_all_factors():
    adata = make_adata(k=[3, 4])
    jz.gp.prune(adata, min_k=1, min_similarity=0.3)
    n_factors = adata.varm["juzi_G"].shape[1]
    assert len(adata.uns["juzi_keep_prune"]) == n_factors
    assert adata.uns["juzi_keep_prune"].sum() <= n_factors


def test_prune_stricter_threshold_keeps_fewer():
    adata_low  = make_adata(k=[3, 4], seed=0)
    adata_high = make_adata(k=[3, 4], seed=0)
    jz.gp.prune(adata_low,  min_k=1, min_similarity=0.1)
    jz.gp.prune(adata_high, min_k=1, min_similarity=0.9)
    assert adata_high.uns["juzi_keep_prune"].sum() <= adata_low.uns["juzi_keep_prune"].sum()


def test_prune_stricter_min_k_keeps_fewer():
    adata_low  = make_adata(k=[3, 4, 5], seed=0)
    adata_high = make_adata(k=[3, 4, 5], seed=0)
    jz.gp.prune(adata_low,  min_k=1, min_similarity=0.3)
    jz.gp.prune(adata_high, min_k=3, min_similarity=0.3)
    assert adata_high.uns["juzi_keep_prune"].sum() <= adata_low.uns["juzi_keep_prune"].sum()


# Matching strategies


def test_greedy_matching_runs():
    adata = make_adata(k=[3, 4])
    jz.gp.prune(adata, matching="greedy")
    assert "juzi_keep_prune" in adata.uns


def test_hungarian_matching_runs():
    adata = make_adata(k=[3, 4])
    jz.gp.prune(adata, matching="hungarian")
    assert "juzi_keep_prune" in adata.uns


def test_hungarian_keeps_leq_greedy():
    adata_greedy    = make_adata(k=[3, 4], seed=0)
    adata_hungarian = make_adata(k=[3, 4], seed=0)
    jz.gp.prune(adata_greedy,    min_k=1, min_similarity=0.3, matching="greedy")
    jz.gp.prune(adata_hungarian, min_k=1, min_similarity=0.3, matching="hungarian")
    assert (
        adata_hungarian.uns["juzi_keep_prune"].sum()
        <= adata_greedy.uns["juzi_keep_prune"].sum()
    )
