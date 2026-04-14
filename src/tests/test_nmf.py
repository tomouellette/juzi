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
    """AnnData already fit with juzi.gp.nmf."""
    return jz.gp.nmf(
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
        jz.gp.nmf(make_adata_raw(), key="wrong_key", k=[2], min_cells=10)


def test_error_invalid_layer():
    with pytest.raises(KeyError):
        jz.gp.nmf(
            make_adata_raw(), key="donor_id", layer="nonexistent", k=[2], min_cells=10
        )


def test_error_invalid_genes_column():
    with pytest.raises(KeyError):
        jz.gp.nmf(
            make_adata_raw(),
            key="donor_id",
            genes="nonexistent_column",
            k=[2],
            min_cells=10,
        )


def test_error_invalid_gene_names_col():
    with pytest.raises(KeyError):
        jz.gp.nmf(
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
        jz.gp.nmf(adata, key="donor_id", genes="highly_variable", k=[2], min_cells=10)


def test_error_k_not_iterable():
    with pytest.raises(TypeError):
        jz.gp.nmf(make_adata_raw(), key="donor_id", k=3, min_cells=10)


def test_error_no_samples_pass_min_cells():
    with pytest.raises(ValueError):
        jz.gp.nmf(
            make_adata_raw(n_cells_per_sample=5), key="donor_id", k=[2], min_cells=100
        )


def test_error_cd_solver_with_kl_loss():
    with pytest.raises(ValueError):
        jz.gp.nmf(
            make_adata_raw(),
            key="donor_id",
            k=[2],
            min_cells=10,
            solver="cd",
            loss="kullback-leibler",
        )


# Always returns AnnData


def test_always_returns_adata():
    result = jz.gp.nmf(
        make_adata_raw(), key="donor_id", k=[2], min_cells=10, genes=None
    )
    assert isinstance(result, AnnData)


def test_input_not_modified():
    """nmf always returns a new object — input adata is never modified."""
    adata = make_adata_raw()
    result = jz.gp.nmf(adata, key="donor_id", k=[2], min_cells=10, genes=None)
    assert "juzi_G" not in adata.varm
    assert "juzi_G" in result.varm


# Output shapes


def test_output_shapes_default():
    result = jz.gp.nmf(
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
    result = jz.gp.nmf(
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
    result = jz.gp.nmf(
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
    result = jz.gp.nmf(
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
    result = jz.gp.nmf(
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
    result = jz.gp.nmf(
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
    result = jz.gp.nmf(
        make_adata_raw(n_samples=2), key="donor_id", k=[3, 4], min_cells=10, genes=None
    )
    assert result.uns["juzi_k"] == [3, 4]


def test_uns_juzi_names_length():
    result = jz.gp.nmf(
        make_adata_raw(n_samples=2), key="donor_id", k=[3], min_cells=10, genes=None
    )
    n_factors = result.varm["juzi_G"].shape[1]
    assert len(result.uns["juzi_names"]) == n_factors


def test_uns_juzi_names_alignment():
    result = jz.gp.nmf(
        make_adata_raw(n_samples=2), key="donor_id", k=[3], min_cells=10, genes=None
    )
    names = result.uns["juzi_names"]
    unique = list(dict.fromkeys(names))
    assert set(unique) == set(result.obs["donor_id"].unique())


def test_uns_juzi_G_genes_default():
    """juzi_G_genes should match var_names when gene_names_col is None."""
    result = jz.gp.nmf(
        make_adata_raw(n_samples=2), key="donor_id", k=[2], min_cells=10, genes=None
    )
    assert result.uns["juzi_G_genes"] == result.var_names.tolist()


def test_uns_juzi_G_genes_from_col():
    """juzi_G_genes should use the specified .var column."""
    result = jz.gp.nmf(
        make_adata_raw(n_samples=2),
        key="donor_id",
        k=[2],
        min_cells=10,
        genes=None,
        gene_names_col="gene_name",
    )
    assert result.uns["juzi_G_genes"] == result.var["gene_name"].tolist()


# Numerical properties


def test_G_non_negative():
    result = make_adata()
    assert (result.varm["juzi_G"] >= 0).all()


def test_G_no_nan():
    result = make_adata()
    assert not np.isnan(result.varm["juzi_G"]).any()


def test_scores_non_negative():
    result = jz.gp.nmf(
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
    """Cells from samples that fail min_cells must have NaN scores."""
    import anndata

    adata_main  = make_adata_raw(n_samples=2, n_cells_per_sample=50, seed=0)
    adata_small = make_adata_raw(n_samples=1, n_cells_per_sample=3, seed=1)
    adata_small.obs["donor_id"] = "sample_small"
    adata_small.var.index       = adata_main.var.index

    adata_main.obs_names = [f"cell_{i}" for i in range(adata_main.n_obs)]
    adata_small.obs_names = [f"cell_small_{i}" for i in range(adata_small.n_obs)]

    combined = anndata.concat([adata_main, adata_small])

    result    = jz.gp.nmf(
        combined, key="donor_id", k=[2],
        min_cells=10, genes=None, keep_scores=True
    )

    small_idx = np.where(result.obs["donor_id"] == "sample_small")[0]
    if len(small_idx) > 0:
        assert np.isnan(result.obsm["juzi_scores"][small_idx]).all()


# Solver and loss


def test_solver_mu_frobenius():
    result = jz.gp.nmf(
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
    result = jz.gp.nmf(
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
    result = jz.gp.nmf(
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
    result = jz.gp.nmf(
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
    """Parallel and serial fits must produce identical results with nndsvda."""
    raw_a = make_adata_raw(n_samples=2, seed=0)
    raw_b = make_adata_raw(n_samples=2, seed=0)

    serial = jz.gp.nmf(raw_a, key="donor_id", k=[3], min_cells=10, genes=None, n_jobs=1)
    parallel = jz.gp.nmf(
        raw_b, key="donor_id", k=[3], min_cells=10, genes=None, n_jobs=2
    )

    np.testing.assert_allclose(
        serial.varm["juzi_G"],
        parallel.varm["juzi_G"],
        rtol=1e-5,
    )
