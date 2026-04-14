import pytest
import warnings
import numpy as np
import pandas as pd
from anndata import AnnData
import juzi as jz


# Fixtures


def make_adata(
    n_cells_per_sample: int = 50,
    n_genes: int = 200,
    n_samples: int = 4,
    k: list[int] = [2, 3],
    seed: int = 42,
) -> AnnData:
    """AnnData fit through full pipeline including score, ready for aggregate."""
    rng = np.random.default_rng(seed)

    profile_a = rng.normal(5.0, 1.0, size=(1, n_genes))
    profile_b = rng.normal(20.0, 1.0, size=(1, n_genes))

    blocks, labels = [], []
    for i in range(n_samples):
        profile = profile_a if i % 2 == 0 else profile_b
        noise = rng.normal(0.0, 0.5, size=(n_cells_per_sample, n_genes))
        X_sample = np.clip(profile + noise, 0, None)
        blocks.append(X_sample)
        labels.extend([f"sample_{i}"] * n_cells_per_sample)

    # Add donor-level covariates
    ages = np.repeat([30, 45, 55, 70][:n_samples], n_cells_per_sample)
    studies = np.repeat(
        [f"study_{i % 2}" for i in range(n_samples)], n_cells_per_sample
    )

    adata = AnnData(
        X=np.vstack(blocks).astype(np.float32),
        obs={
            "donor_id": labels,
            "age": ages.astype(float),
            "study_id": studies,
        },
        var={"gene_name": np.arange(n_genes).astype(str)},
    )

    adata = jz.gp.nmf(adata, key="donor_id", k=k, min_cells=10, genes=None, seed=seed)
    jz.gp.similarity(adata, distance="jaccard", top_k=20)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)
    jz.gp.score(adata, n_top_genes=10, seed=seed)

    return adata


def make_adata_raw_small(seed: int = 1) -> AnnData:
    """Raw AnnData with only 5 cells per sample — too small for nmf min_cells=10."""
    rng = np.random.default_rng(seed)
    X = rng.normal(5.0, 1.0, size=(5, 200)).astype(np.float32)
    return AnnData(
        X=X,
        obs={"donor_id": ["sample_small"] * 5, "age": [40.0] * 5},
        var={"gene_name": np.arange(200).astype(str)},
    )


# Input validation


def test_error_missing_program_scores():
    adata = make_adata()
    del adata.obsm["juzi_program_scores"]
    with pytest.raises(KeyError):
        jz.gp.aggregate(adata, key="donor_id")


def test_error_invalid_key():
    with pytest.raises(KeyError):
        jz.gp.aggregate(make_adata(), key="wrong_key")


def test_error_invalid_agg():
    with pytest.raises(ValueError):
        jz.gp.aggregate(make_adata(), key="donor_id", agg="sum")


def test_error_min_cells_below_one():
    with pytest.raises(ValueError):
        jz.gp.aggregate(make_adata(), key="donor_id", min_cells=0)


def test_error_invalid_obs_cols():
    with pytest.raises(KeyError):
        jz.gp.aggregate(make_adata(), key="donor_id", obs_cols=["nonexistent_col"])


def test_error_no_donors_pass_min_cells():
    """aggregate raises ValueError when all donors are below min_cells."""
    adata = make_adata()
    # Artificially inflate min_cells so all donors fail
    with pytest.raises(ValueError):
        jz.gp.aggregate(adata, key="donor_id", min_cells=100_000)


# Output structure


def test_aggregate_scores_in_uns():
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id")
    assert "juzi_aggregate_scores" in adata.uns


def test_aggregate_scores_is_dataframe():
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id")
    assert isinstance(adata.uns["juzi_aggregate_scores"], pd.DataFrame)


def test_aggregate_scores_shape():
    adata = make_adata(n_samples=4)
    jz.gp.aggregate(adata, key="donor_id")
    df = adata.uns["juzi_aggregate_scores"]
    n_programs = adata.obsm["juzi_program_scores"].shape[1]
    assert df.shape == (4, n_programs)


def test_aggregate_scores_index_is_donor_key():
    adata = make_adata(n_samples=4)
    jz.gp.aggregate(adata, key="donor_id")
    df = adata.uns["juzi_aggregate_scores"]
    assert df.index.name == "donor_id"
    assert set(df.index.tolist()) == set(adata.obs["donor_id"].unique())


def test_aggregate_program_columns_named_correctly():
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id")
    df = adata.uns["juzi_aggregate_scores"]
    n_programs = adata.obsm["juzi_program_scores"].shape[1]
    expected = [f"P{p}" for p in range(n_programs)]
    assert all(col in df.columns for col in expected)


# Covariate propagation


def test_obs_cols_propagated():
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id", obs_cols=["age", "study_id"])
    df = adata.uns["juzi_aggregate_scores"]
    assert "age" in df.columns
    assert "study_id" in df.columns


def test_obs_cols_values_correct():
    """Covariate values must match the donor-level constant in adata.obs."""
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id", obs_cols=["age"])
    df = adata.uns["juzi_aggregate_scores"]

    for donor in df.index:
        expected_age = adata.obs.loc[adata.obs["donor_id"] == donor, "age"].iloc[0]
        assert df.loc[donor, "age"] == expected_age


def test_no_obs_cols_only_program_columns():
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id", obs_cols=None)
    df = adata.uns["juzi_aggregate_scores"]
    n_programs = adata.obsm["juzi_program_scores"].shape[1]
    assert df.shape[1] == n_programs


# Aggregation functions


def test_agg_mean_runs():
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id", agg="mean")
    assert "juzi_aggregate_scores" in adata.uns


def test_agg_median_runs():
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id", agg="median")
    assert "juzi_aggregate_scores" in adata.uns


def test_agg_mean_correct_value():
    """Mean scores must equal per-cell mean within each donor."""
    adata = make_adata(n_samples=2)
    jz.gp.aggregate(adata, key="donor_id", agg="mean")
    df = adata.uns["juzi_aggregate_scores"]
    scores = adata.obsm["juzi_program_scores"]
    donors = adata.obs["donor_id"].values

    for donor in df.index:
        mask = donors == donor
        expected = scores[mask].mean(axis=0)
        actual = df.loc[donor, [f"P{p}" for p in range(scores.shape[1])]].values
        np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_agg_median_differs_from_mean():
    """Median and mean should differ when per-cell scores are skewed."""
    adata_mean = make_adata(seed=0)
    adata_median = make_adata(seed=0)
    jz.gp.aggregate(adata_mean, key="donor_id", agg="mean")
    jz.gp.aggregate(adata_median, key="donor_id", agg="median")

    mean_vals = adata_mean.uns["juzi_aggregate_scores"].values
    median_vals = adata_median.uns["juzi_aggregate_scores"].values
    assert not np.allclose(mean_vals, median_vals)


# min_cells filtering


def test_min_cells_excludes_small_donors():
    """Donors with fewer than min_cells cells must be excluded."""
    import anndata

    adata_main = make_adata(n_samples=3, n_cells_per_sample=50)
    adata_main.obs_names = [f"cell_{i}" for i in range(adata_main.n_obs)]

    # Build a small raw AnnData with matching var_names and add program scores
    rng = np.random.default_rng(1)
    X_small = rng.normal(5.0, 1.0, size=(5, adata_main.n_vars)).astype(np.float32)
    adata_small = AnnData(
        X=X_small,
        obs={"donor_id": ["sample_small"] * 5, "age": [40.0] * 5},
        var=adata_main.var.copy(),
    )

    combined = anndata.concat([adata_main, adata_small])
    combined.obs_names_make_unique()

    # Manually add program scores for all cells including small donor
    n_programs = adata_main.obsm["juzi_program_scores"].shape[1]
    combined.obsm["juzi_program_scores"] = np.zeros(
        (combined.n_obs, n_programs), dtype=np.float32
    )
    combined.obsm["juzi_program_scores"][: adata_main.n_obs] = adata_main.obsm[
        "juzi_program_scores"
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        jz.gp.aggregate(combined, key="donor_id", min_cells=10)

    df = combined.uns["juzi_aggregate_scores"]
    assert "sample_small" not in df.index


def test_min_cells_warning_on_exclusion():
    """A UserWarning should be raised when donors are excluded."""
    import anndata

    adata_main = make_adata(n_samples=3, n_cells_per_sample=50)
    adata_main.obs_names = [f"cell_{i}" for i in range(adata_main.n_obs)]

    rng = np.random.default_rng(1)
    X_small = rng.normal(5.0, 1.0, size=(5, adata_main.n_vars)).astype(np.float32)
    adata_small = AnnData(
        X=X_small,
        obs={"donor_id": ["sample_small"] * 5, "age": [40.0] * 5},
        var=adata_main.var.copy(),
    )

    combined = anndata.concat([adata_main, adata_small])
    combined.obs_names_make_unique()

    n_programs = adata_main.obsm["juzi_program_scores"].shape[1]
    combined.obsm["juzi_program_scores"] = np.zeros(
        (combined.n_obs, n_programs), dtype=np.float32
    )
    combined.obsm["juzi_program_scores"][: adata_main.n_obs] = adata_main.obsm[
        "juzi_program_scores"
    ]

    with pytest.warns(UserWarning):
        jz.gp.aggregate(combined, key="donor_id", min_cells=10)


# Numerical properties


def test_aggregate_scores_finite():
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id")
    df = adata.uns["juzi_aggregate_scores"]
    program_cols = [c for c in df.columns if c.startswith("P")]
    assert np.isfinite(df[program_cols].values.astype(float)).all()


def test_aggregate_scores_no_nan():
    adata = make_adata()
    jz.gp.aggregate(adata, key="donor_id")
    df = adata.uns["juzi_aggregate_scores"]
    program_cols = [c for c in df.columns if c.startswith("P")]
    assert not df[program_cols].isna().any().any()


# copy parameter


def test_copy_false_modifies_inplace():
    adata = make_adata()
    result = jz.gp.aggregate(adata, key="donor_id", copy=False)
    assert result is None
    assert "juzi_aggregate_scores" in adata.uns


def test_copy_true_returns_new_object():
    adata = make_adata()
    result = jz.gp.aggregate(adata, key="donor_id", copy=True)
    assert result is not None
    assert "juzi_aggregate_scores" not in adata.uns
    assert "juzi_aggregate_scores" in result.uns
