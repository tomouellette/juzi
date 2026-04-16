import warnings
import pytest
import numpy as np
from anndata import AnnData
import juzi as jz


# Fixtures


def make_adata_raw(
    n_cells_per_sample: int = 50,
    n_genes: int = 100,
    n_hvg: int = 32,
    n_samples: int = 3,
    seed: int = 42,
) -> AnnData:
    """Raw AnnData before nmf — used to test nmf directly."""
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

    X = np.vstack(blocks)
    hvg = np.zeros(n_genes, dtype=bool)
    hvg[:n_hvg] = True

    return AnnData(
        X=X,
        obs={"donor_id": labels},
        var={
            "gene_name": np.arange(n_genes).astype(str),
            "highly_variable": hvg,
        },
    )


def make_adata_raw_with_layer(seed: int = 42) -> AnnData:
    adata = make_adata_raw(seed=seed)
    adata.layers["counts"] = adata.X.copy()
    return adata


def make_adata(
    n_cells_per_sample: int = 50,
    n_genes: int = 100,
    n_samples: int = 3,
    k: list[int] = [2, 3],
    seed: int = 42,
) -> AnnData:
    """AnnData already fit with juzi.gp.nmf_fit."""
    return jz.gp.nmf_fit(
        make_adata_raw(
            n_cells_per_sample=n_cells_per_sample,
            n_genes=n_genes,
            n_samples=n_samples,
            seed=seed,
        ),
        key="donor_id",
        k=k,
        min_cells=10,
        genes=None,
        seed=seed,
    )


# Input validation


def test_error_invalid_key():
    with pytest.raises(KeyError):
        jz.gp.nmf_fit(make_adata_raw(), key="wrong_key", k=[2], min_cells=10)


def test_error_invalid_layer():
    with pytest.raises(KeyError):
        jz.gp.nmf_fit(
            make_adata_raw(),
            key="donor_id",
            layer="nonexistent",
            k=[2],
            min_cells=10,
        )


def test_error_invalid_genes_column():
    with pytest.raises(KeyError):
        jz.gp.nmf_fit(
            make_adata_raw(),
            key="donor_id",
            genes="nonexistent_column",
            k=[2],
            min_cells=10,
        )


def test_error_invalid_gene_names_col():
    with pytest.raises(KeyError):
        jz.gp.nmf_fit(
            make_adata_raw(),
            key="donor_id",
            gene_names_col="nonexistent",
            k=[2],
            min_cells=10,
        )


def test_error_no_genes_pass_filter():
    adata = make_adata_raw()
    adata.var["highly_variable"] = False
    with pytest.raises(ValueError):
        jz.gp.nmf_fit(
            adata, key="donor_id", genes="highly_variable", k=[2], min_cells=10
        )


def test_error_k_not_iterable():
    with pytest.raises(TypeError):
        jz.gp.nmf_fit(make_adata_raw(), key="donor_id", k=3, min_cells=10)


def test_error_no_samples_pass_min_cells():
    with pytest.raises(ValueError):
        jz.gp.nmf_fit(
            make_adata_raw(n_cells_per_sample=5),
            key="donor_id",
            k=[2],
            min_cells=100,
        )


def test_error_cd_solver_with_kl_loss():
    with pytest.raises(ValueError):
        jz.gp.nmf_fit(
            make_adata_raw(),
            key="donor_id",
            k=[2],
            min_cells=10,
            solver="cd",
            loss="kullback-leibler",
        )


# Always returns AnnData


def test_always_returns_adata():
    result = jz.gp.nmf_fit(
        make_adata_raw(), key="donor_id", k=[2], min_cells=10, genes=None
    )
    assert isinstance(result, AnnData)


def test_input_not_modified():
    adata = make_adata_raw()
    result = jz.gp.nmf_fit(adata, key="donor_id", k=[2], min_cells=10, genes=None)
    assert "juzi_G" not in adata.varm
    assert "juzi_G" in result.varm


# Output shapes


def test_output_shapes_default():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_genes=100, n_samples=2),
        key="donor_id",
        k=[3],
        min_cells=10,
        genes=None,
    )
    n_genes, n_factors = result.varm["juzi_G"].shape
    assert n_genes == 100
    assert n_factors == 2 * 3


def test_output_shapes_multiple_k():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_genes=100, n_samples=2),
        key="donor_id",
        k=[3, 4, 5],
        min_cells=10,
        genes=None,
    )
    n_genes, n_factors = result.varm["juzi_G"].shape
    assert n_genes == 100
    assert n_factors == 2 * (3 + 4 + 5)


def test_output_shapes_hvg_column():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_genes=100, n_hvg=32, n_samples=2),
        key="donor_id",
        k=[3],
        min_cells=10,
        genes="highly_variable",
    )
    n_genes, n_factors = result.varm["juzi_G"].shape
    assert n_genes == 32
    assert n_factors == 2 * 3


def test_output_shapes_gene_list():
    gene_list = np.arange(20).astype(str)
    result = jz.gp.nmf_fit(
        make_adata_raw(n_genes=100, n_samples=2),
        key="donor_id",
        k=[3],
        min_cells=10,
        genes=gene_list,
    )
    n_genes, n_factors = result.varm["juzi_G"].shape
    assert n_genes == 20
    assert n_factors == 2 * 3


def test_output_shapes_keep_scores():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_genes=100, n_samples=2, n_cells_per_sample=50),
        key="donor_id",
        k=[3],
        min_cells=10,
        genes=None,
        keep_scores=True,
    )
    n_genes, n_factors = result.varm["juzi_G"].shape
    n_cells, n_scores = result.obsm["juzi_scores"].shape
    assert n_genes == 100
    assert n_factors == 2 * 3
    assert n_cells == result.n_obs
    assert n_scores == 3


def test_output_shapes_layer():
    result = jz.gp.nmf_fit(
        make_adata_raw_with_layer(),
        key="donor_id",
        layer="counts",
        k=[3],
        min_cells=10,
        genes=None,
    )
    assert "juzi_G" in result.varm


# uns fields


def test_uns_juzi_k():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=2),
        key="donor_id",
        k=[3, 4],
        min_cells=10,
        genes=None,
    )
    assert result.uns["juzi_k"] == [3, 4]


def test_uns_juzi_names_length():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=2),
        key="donor_id",
        k=[3],
        min_cells=10,
        genes=None,
    )
    n_factors = result.varm["juzi_G"].shape[1]
    assert len(result.uns["juzi_names"]) == n_factors


def test_uns_juzi_names_alignment():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=2),
        key="donor_id",
        k=[3],
        min_cells=10,
        genes=None,
    )
    names = result.uns["juzi_names"]
    unique = list(dict.fromkeys(names))
    assert set(unique) == set(result.obs["donor_id"].unique())


def test_uns_juzi_G_genes_default():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=2),
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
    )
    assert result.uns["juzi_G_genes"] == result.var_names.tolist()


def test_uns_juzi_G_genes_from_col():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=2),
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
        gene_names_col="gene_name",
    )
    assert result.uns["juzi_G_genes"] == result.var["gene_name"].tolist()


# Stage keep masks


def test_keep_masks_initialised():
    """All four keep masks must be present and all True after nmf."""
    result = make_adata()
    for key in [
        "juzi_keep_prune",
        "juzi_keep_similarity",
        "juzi_keep_cluster",
        "juzi_keep",
    ]:
        assert key in result.uns, f"'{key}' not found in .uns"


def test_keep_masks_all_true():
    """All keep masks must be all True immediately after nmf."""
    result = make_adata()
    for key in [
        "juzi_keep_prune",
        "juzi_keep_similarity",
        "juzi_keep_cluster",
        "juzi_keep",
    ]:
        assert result.uns[key].all(), f"'{key}' not all True after nmf"


def test_keep_masks_dtype_bool():
    result = make_adata()
    for key in [
        "juzi_keep_prune",
        "juzi_keep_similarity",
        "juzi_keep_cluster",
        "juzi_keep",
    ]:
        assert result.uns[key].dtype == bool, f"'{key}' is not dtype bool"


def test_keep_masks_length_matches_factors():
    result = make_adata(n_samples=2, k=[3, 4])
    n_factors = result.varm["juzi_G"].shape[1]
    for key in [
        "juzi_keep_prune",
        "juzi_keep_similarity",
        "juzi_keep_cluster",
        "juzi_keep",
    ]:
        assert len(result.uns[key]) == n_factors


def test_juzi_keep_is_intersection():
    """juzi_keep must equal the AND of all three stage masks."""
    result = make_adata()
    expected = (
        result.uns["juzi_keep_prune"]
        & result.uns["juzi_keep_similarity"]
        & result.uns["juzi_keep_cluster"]
    )
    np.testing.assert_array_equal(result.uns["juzi_keep"], expected)


# genes_force


def test_genes_force_included():
    """Forced genes must appear in juzi_G_genes."""
    raw = make_adata_raw(n_genes=100)
    force_genes = raw.var_names[:5].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = jz.gp.nmf_fit(
            raw,
            key="donor_id",
            k=[2],
            min_cells=10,
            genes="highly_variable",
            genes_force=force_genes,
        )
    for gene in force_genes:
        assert gene in result.uns["juzi_G_genes"]


def test_genes_force_union_with_hvg():
    """Gene set with genes_force must be >= HVG-only gene set."""
    raw = make_adata_raw(n_genes=100, n_hvg=32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result_hvg = jz.gp.nmf_fit(
            make_adata_raw(n_genes=100, n_hvg=32),
            key="donor_id",
            k=[2],
            min_cells=10,
            genes="highly_variable",
        )
        result_force = jz.gp.nmf_fit(
            raw,
            key="donor_id",
            k=[2],
            min_cells=10,
            genes="highly_variable",
            genes_force=raw.var_names[50:55].tolist(),  # non-HVG genes
        )
    assert result_force.n_vars >= result_hvg.n_vars


def test_genes_force_missing_genes_warn():
    """Genes in genes_force not in var_names should raise a UserWarning."""
    with pytest.warns(UserWarning, match="not found"):
        jz.gp.nmf_fit(
            make_adata_raw(),
            key="donor_id",
            k=[2],
            min_cells=10,
            genes=None,
            genes_force=["FAKE_GENE_1", "FAKE_GENE_2"],
        )


def test_genes_force_all_missing_still_runs():
    """If all genes_force genes are missing the pipeline should still run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = jz.gp.nmf_fit(
            make_adata_raw(),
            key="donor_id",
            k=[2],
            min_cells=10,
            genes=None,
            genes_force=["FAKE_GENE_1"],
        )
    assert "juzi_G" in result.varm


# Numerical properties


def test_G_non_negative():
    result = make_adata()
    assert (result.varm["juzi_G"] >= 0).all()


def test_G_no_nan():
    result = make_adata()
    assert not np.isnan(result.varm["juzi_G"]).any()


def test_scores_non_negative():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=2),
        key="donor_id",
        k=[3],
        min_cells=10,
        genes=None,
        keep_scores=True,
    )
    scores = result.obsm["juzi_scores"]
    assert (scores[~np.isnan(scores)] >= 0).all()


def test_scores_nan_for_dropped_samples():
    import anndata

    adata_main = make_adata_raw(n_samples=2, n_cells_per_sample=50, seed=0)
    adata_small = make_adata_raw(n_samples=1, n_cells_per_sample=3, seed=1)
    adata_small.obs["donor_id"] = "sample_small"
    adata_small.var.index = adata_main.var.index

    adata_main.obs_names = [f"cell_{i}" for i in range(adata_main.n_obs)]
    adata_small.obs_names = [f"cell_small_{i}" for i in range(adata_small.n_obs)]

    combined = anndata.concat([adata_main, adata_small])
    result = jz.gp.nmf_fit(
        combined,
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
        keep_scores=True,
    )

    small_idx = np.where(result.obs["donor_id"] == "sample_small")[0]
    if len(small_idx) > 0:
        assert np.isnan(result.obsm["juzi_scores"][small_idx]).all()


# Solver and loss


def test_solver_mu_frobenius():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=2),
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
        solver="mu",
        loss="frobenius",
    )
    assert "juzi_G" in result.varm


def test_solver_mu_kl():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=2),
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
        solver="mu",
        loss="kullback-leibler",
    )
    assert "juzi_G" in result.varm


# Parallelisation


def test_parallel_threads():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=3),
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
        n_jobs=2,
        prefer="threads",
    )
    assert "juzi_G" in result.varm


def test_parallel_processes():
    result = jz.gp.nmf_fit(
        make_adata_raw(n_samples=3),
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
        n_jobs=2,
        prefer="processes",
    )
    assert "juzi_G" in result.varm


def test_parallel_matches_serial():
    raw_a = make_adata_raw(n_samples=2, seed=0)
    raw_b = make_adata_raw(n_samples=2, seed=0)

    serial = jz.gp.nmf_fit(
        raw_a, key="donor_id", k=[3], min_cells=10, genes=None, n_jobs=1
    )
    parallel = jz.gp.nmf_fit(
        raw_b, key="donor_id", k=[3], min_cells=10, genes=None, n_jobs=2
    )

    np.testing.assert_allclose(
        serial.varm["juzi_G"],
        parallel.varm["juzi_G"],
        rtol=1e-5,
    )
