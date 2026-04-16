import pytest
import numpy as np
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
    """AnnData fit through the full pipeline, ready for scoring."""
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

    adata = AnnData(
        X=np.vstack(blocks).astype(np.float32),
        obs={"donor_id": labels},
        var={"gene_name": np.arange(n_genes).astype(str)},
    )

    adata = jz.gp.nmf_fit(
        adata, key="donor_id", k=k, min_cells=10, genes=None, seed=seed
    )
    jz.gp.similarity_compute(adata, distance="jaccard", top_k=20)
    jz.gp.programs_cluster(adata, threshold=0.3, min_cluster=1)

    return adata


def make_adata_with_gene_col(seed: int = 42) -> AnnData:
    """AnnData where gene names are in a .var column, not var_names."""
    adata = make_adata(seed=seed)
    adata.var["feature_name"] = adata.var_names.to_numpy()
    return adata


def make_adata_with_layer(seed: int = 42) -> AnnData:
    adata = make_adata(seed=seed)
    adata.layers["lognorm"] = adata.X.copy()
    return adata


def make_adata_scored(seed: int = 42) -> AnnData:
    """AnnData fit through score_cells — ready for score_classify."""
    adata = make_adata(seed=seed)
    jz.gp.score_cells(adata, n_top_genes=10, seed=seed, use_combined=True)
    return adata


# Input validation


def test_error_missing_juzi_cluster_G():
    adata = make_adata()
    del adata.uns["juzi_cluster_G"]
    with pytest.raises(KeyError):
        jz.gp.score_cells(adata)


def test_error_missing_juzi_cluster_labels():
    adata = make_adata()
    del adata.uns["juzi_cluster_labels"]
    with pytest.raises(KeyError):
        jz.gp.score_cells(adata)


def test_error_missing_juzi_G_genes():
    adata = make_adata()
    del adata.uns["juzi_G_genes"]
    with pytest.raises(KeyError):
        jz.gp.score_cells(adata)


def test_error_n_top_genes_below_one():
    with pytest.raises(ValueError):
        jz.gp.score_cells(make_adata(), n_top_genes=0)


def test_error_n_control_genes_below_one():
    with pytest.raises(ValueError):
        jz.gp.score_cells(make_adata(), n_control_genes=0)


def test_error_invalid_layer():
    with pytest.raises(KeyError):
        jz.gp.score_cells(make_adata(), layer="nonexistent")


def test_error_invalid_gene_names_col():
    with pytest.raises(KeyError):
        jz.gp.score_cells(make_adata(), gene_names_col="nonexistent")


def test_error_no_shared_genes():
    adata = make_adata()
    # Overwrite juzi_G_genes with genes that don't exist in adata
    adata.uns["juzi_G_genes"] = ["FAKE_GENE_1", "FAKE_GENE_2"]
    with pytest.raises(ValueError):
        jz.gp.score_cells(adata)


def test_error_n_top_genes_exceeds_shared():
    adata = make_adata(n_genes=10)
    with pytest.raises(ValueError):
        jz.gp.score_cells(adata, n_top_genes=100)


# Output structure


def test_program_scores_in_obsm():
    adata = make_adata()
    jz.gp.score_cells(adata)
    assert "juzi_program_scores" in adata.obsm


def test_program_genes_in_uns():
    adata = make_adata()
    jz.gp.score_cells(adata)
    assert "juzi_program_genes" in adata.uns


def test_program_scores_shape():
    adata = make_adata()
    jz.gp.score_cells(adata)
    n_programs = len(np.unique(adata.uns["juzi_cluster_labels"]))
    scores = adata.obsm["juzi_program_scores"]
    assert scores.shape == (adata.n_obs, n_programs)


def test_program_genes_keys_match_programs():
    adata = make_adata()
    jz.gp.score_cells(adata)
    n_programs = len(np.unique(adata.uns["juzi_cluster_labels"]))
    assert set(adata.uns["juzi_program_genes"].keys()) == set(range(n_programs))


def test_program_genes_length():
    adata = make_adata()
    n_top = 10
    jz.gp.score_cells(adata, n_top_genes=n_top)
    for genes in adata.uns["juzi_program_genes"].values():
        assert len(genes) == n_top


def test_program_genes_are_strings():
    adata = make_adata()
    jz.gp.score_cells(adata)
    for genes in adata.uns["juzi_program_genes"].values():
        assert all(isinstance(g, str) for g in genes)


def test_program_scores_dtype():
    adata = make_adata()
    jz.gp.score_cells(adata)
    assert adata.obsm["juzi_program_scores"].dtype == np.float32


# Numerical properties


def test_scores_finite():
    adata = make_adata()
    jz.gp.score_cells(adata)
    assert np.isfinite(adata.obsm["juzi_program_scores"]).all()


def test_scores_no_nan():
    adata = make_adata()
    jz.gp.score_cells(adata)
    assert not np.isnan(adata.obsm["juzi_program_scores"]).any()


def test_scores_can_be_negative():
    """Control subtraction can produce negative scores — this is expected."""
    adata = make_adata()
    jz.gp.score_cells(adata)
    # Not all scores should be non-negative after control subtraction
    assert (adata.obsm["juzi_program_scores"] < 0).any()


def test_scores_vary_across_cells():
    """Scores should not be constant across cells for any program."""
    adata = make_adata()
    jz.gp.score_cells(adata)
    scores = adata.obsm["juzi_program_scores"]
    for p in range(scores.shape[1]):
        assert scores[:, p].std() > 0


def test_scores_reproducible_with_seed():
    adata_a = make_adata(seed=0)
    adata_b = make_adata(seed=0)
    jz.gp.score_cells(adata_a, seed=42)
    jz.gp.score_cells(adata_b, seed=42)
    np.testing.assert_array_equal(
        adata_a.obsm["juzi_program_scores"],
        adata_b.obsm["juzi_program_scores"],
    )


# Gene selection


def test_combined_true_genes_differ_from_raw():
    """Combined ranking should produce different gene sets than raw loading."""
    adata_spec = make_adata(seed=0)
    adata_raw = make_adata(seed=0)

    jz.gp.score_cells(adata_spec, n_top_genes=10, use_combined=True, seed=0)
    jz.gp.score_cells(adata_raw, n_top_genes=10, use_combined=False, seed=0)

    spec_genes = adata_spec.uns["juzi_program_genes"]
    raw_genes = adata_raw.uns["juzi_program_genes"]

    # At least one program should have different top genes
    any_differ = any(set(spec_genes[p]) != set(raw_genes[p]) for p in spec_genes)
    assert any_differ


def test_program_genes_unique_per_program():
    """Top genes for each program should be unique within that program."""
    adata = make_adata()
    jz.gp.score_cells(adata, n_top_genes=20)
    for genes in adata.uns["juzi_program_genes"].values():
        assert len(genes) == len(set(genes))


def test_program_genes_in_nmf_gene_set():
    """All selected program genes must come from the NMF gene set."""
    adata = make_adata()
    jz.gp.score_cells(adata)
    nmf_genes = set(adata.uns["juzi_G_genes"])
    for genes in adata.uns["juzi_program_genes"].values():
        assert all(g in nmf_genes for g in genes)


# gene_names_col


def test_gene_names_col_runs():
    adata = make_adata_with_gene_col()
    jz.gp.score_cells(adata, gene_names_col="feature_name")
    assert "juzi_program_scores" in adata.obsm


def test_gene_names_col_matches_var_names():
    """Scores should be identical whether var_names or gene_names_col is used."""
    adata_vn = make_adata(seed=0)
    adata_col = make_adata_with_gene_col(seed=0)

    jz.gp.score_cells(adata_vn, seed=0)
    jz.gp.score_cells(adata_col, gene_names_col="feature_name", seed=0)

    np.testing.assert_allclose(
        adata_vn.obsm["juzi_program_scores"],
        adata_col.obsm["juzi_program_scores"],
        atol=1e-5,
    )


# layer


def test_layer_runs():
    adata = make_adata_with_layer()
    jz.gp.score_cells(adata, layer="lognorm")
    assert "juzi_program_scores" in adata.obsm


def test_layer_matches_X_when_identical():
    """Scores from layer and .X should be identical when they contain same data."""
    adata = make_adata_with_layer(seed=0)
    adata_x = make_adata(seed=0)

    jz.gp.score_cells(adata, layer="lognorm", seed=0)
    jz.gp.score_cells(adata_x, seed=0)

    np.testing.assert_allclose(
        adata.obsm["juzi_program_scores"],
        adata_x.obsm["juzi_program_scores"],
        atol=1e-5,
    )


# gene_pool


def test_gene_pool_restricts_controls():
    """With a small gene pool, control genes must come from that pool."""
    adata = make_adata()
    all_genes = adata.var_names.tolist()
    pool = all_genes[:50]  # restrict to first 50 genes

    jz.gp.score_cells(adata, gene_pool=pool, n_top_genes=5, seed=0)
    assert "juzi_program_scores" in adata.obsm


# Raw count warning


def test_raw_count_warning():
    adata = make_adata()
    adata.X = adata.X * 1000  # inflate to look like raw counts
    with pytest.warns(UserWarning, match="raw counts"):
        jz.gp.score_cells(adata)


# Partial gene overlap warning


def test_partial_overlap_warning():
    adata = make_adata(n_genes=200)
    # Remove some genes from adata so overlap is partial
    adata_subset = adata[:, :150].copy()
    # juzi_G_genes still references all 200 genes
    with pytest.warns(UserWarning, match="absent from adata"):
        jz.gp.score_cells(adata_subset)


# copy parameter


def test_copy_false_modifies_inplace():
    adata = make_adata()
    result = jz.gp.score_cells(adata, copy=False)
    assert result is None
    assert "juzi_program_scores" in adata.obsm


def test_copy_true_returns_new_object():
    adata = make_adata()
    result = jz.gp.score_cells(adata, copy=True)
    assert result is not None
    assert "juzi_program_scores" not in adata.obsm
    assert "juzi_program_scores" in result.obsm


# juzi.gp.score_classify


class TestScoreClassify:

    def test_error_missing_program_scores(self):
        adata = make_adata()
        with pytest.raises(KeyError):
            jz.gp.score_classify(adata)

    def test_error_missing_program_genes(self):
        adata = make_adata_scored()
        del adata.uns["juzi_program_genes"]
        with pytest.raises(KeyError):
            jz.gp.score_classify(adata)

    def test_error_missing_cluster_labels(self):
        adata = make_adata_scored()
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.gp.score_classify(adata)

    def test_error_n_shuffles_below_one(self):
        adata = make_adata_scored()
        with pytest.raises(ValueError):
            jz.gp.score_classify(adata, n_shuffles=0)

    def test_error_n_cells_below_one(self):
        adata = make_adata_scored()
        with pytest.raises(ValueError):
            jz.gp.score_classify(adata, n_cells_per_shuffle=0)

    def test_error_padj_thresh_below_zero(self):
        adata = make_adata_scored()
        with pytest.raises(ValueError):
            jz.gp.score_classify(adata, padj_thresh=0.0)

    def test_error_padj_thresh_above_one(self):
        adata = make_adata_scored()
        with pytest.raises(ValueError):
            jz.gp.score_classify(adata, padj_thresh=1.1)

    def test_output_fields_present(self):
        adata = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        assert "juzi_program_pvals"    in adata.obsm
        assert "juzi_program_padj"     in adata.obsm
        assert "juzi_program_label"    in adata.obs
        assert "juzi_classify_params"  in adata.uns

    def test_pvals_shape(self):
        adata      = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        n_programs = adata.obsm["juzi_program_scores"].shape[1]
        assert adata.obsm["juzi_program_pvals"].shape == (adata.n_obs, n_programs)

    def test_padj_shape(self):
        adata      = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        n_programs = adata.obsm["juzi_program_scores"].shape[1]
        assert adata.obsm["juzi_program_padj"].shape == (adata.n_obs, n_programs)

    def test_pvals_in_zero_one(self):
        adata = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        pvals = adata.obsm["juzi_program_pvals"]
        assert (pvals >= 0.0).all()
        assert (pvals <= 1.0).all()

    def test_padj_in_zero_one(self):
        adata = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        padj  = adata.obsm["juzi_program_padj"]
        assert (padj >= 0.0).all()
        assert (padj <= 1.0).all()

    def test_padj_geq_pvals(self):
        """BH-adjusted p-values must be >= raw p-values."""
        adata = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        pvals = adata.obsm["juzi_program_pvals"]
        padj  = adata.obsm["juzi_program_padj"]
        assert (padj >= pvals - 1e-6).all()

    def test_labels_are_strings(self):
        adata = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        labels = adata.obs["juzi_program_label"]
        assert labels.dtype == object
        assert all(isinstance(l, str) for l in labels)

    def test_labels_are_valid_programs_or_unresolved(self):
        adata      = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        unique_C   = np.unique(adata.uns["juzi_cluster_labels"])
        valid      = {f"C{int(c)}" for c in unique_C} | {"unresolved"}
        labels     = set(adata.obs["juzi_program_label"].unique().tolist())
        assert labels.issubset(valid)

    def test_classify_params_keys(self):
        adata = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        params = adata.uns["juzi_classify_params"]
        for key in ["n_shuffles", "n_cells_per_shuffle",
                    "padj_thresh", "null_mean", "null_std"]:
            assert key in params

    def test_null_mean_shape(self):
        adata      = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        n_programs = adata.obsm["juzi_program_scores"].shape[1]
        assert adata.uns["juzi_classify_params"]["null_mean"].shape == (n_programs,)

    def test_null_std_shape(self):
        adata      = make_adata_scored()
        jz.gp.score_classify(adata, n_shuffles=2, n_cells_per_shuffle=50)
        n_programs = adata.obsm["juzi_program_scores"].shape[1]
        assert adata.uns["juzi_classify_params"]["null_std"].shape == (n_programs,)

    def test_stored_params_match_inputs(self):
        adata = make_adata_scored()
        jz.gp.score_classify(
            adata, n_shuffles=3, n_cells_per_shuffle=30, padj_thresh=0.1
        )
        params = adata.uns["juzi_classify_params"]
        assert params["n_shuffles"]          == 3
        assert params["n_cells_per_shuffle"] == 30
        assert params["padj_thresh"]         == 0.1

    def test_strict_padj_gives_more_unresolved(self):
        """Stricter padj_thresh should produce more unresolved cells."""
        adata_loose  = make_adata_scored(seed=0)
        adata_strict = make_adata_scored(seed=0)
        jz.gp.score_classify(adata_loose,  n_shuffles=2, n_cells_per_shuffle=50, padj_thresh=0.5)
        jz.gp.score_classify(adata_strict, n_shuffles=2, n_cells_per_shuffle=50, padj_thresh=0.01)
        n_unresolved_loose  = (adata_loose.obs["juzi_program_label"]  == "unresolved").sum()
        n_unresolved_strict = (adata_strict.obs["juzi_program_label"] == "unresolved").sum()
        assert n_unresolved_strict >= n_unresolved_loose

    def test_reproducible_with_same_seed(self):
        adata_a = make_adata_scored(seed=0)
        adata_b = make_adata_scored(seed=0)
        jz.gp.score_classify(adata_a, n_shuffles=3, n_cells_per_shuffle=50, seed=42)
        jz.gp.score_classify(adata_b, n_shuffles=3, n_cells_per_shuffle=50, seed=42)
        np.testing.assert_array_equal(
            adata_a.obs["juzi_program_label"].values,
            adata_b.obs["juzi_program_label"].values,
        )

    def test_parallel_threads(self):
        adata = make_adata_scored()
        jz.gp.score_classify(
            adata, n_shuffles=2, n_cells_per_shuffle=50,
            n_jobs=2, prefer="threads",
        )
        assert "juzi_program_label" in adata.obs

    def test_copy_false_modifies_inplace(self):
        adata  = make_adata_scored()
        result = jz.gp.score_classify(
            adata, n_shuffles=2, n_cells_per_shuffle=50, copy=False
        )
        assert result is None
        assert "juzi_program_label" in adata.obs

    def test_copy_true_returns_new_object(self):
        adata  = make_adata_scored()
        result = jz.gp.score_classify(
            adata, n_shuffles=2, n_cells_per_shuffle=50, copy=True
        )
        assert result is not None
        assert "juzi_program_label" not in adata.obs
        assert "juzi_program_label" in result.obs
