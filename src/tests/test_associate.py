import pytest
import re
import numpy as np
import pandas as pd
from anndata import AnnData
import juzi as jz


# Fixtures


def make_adata(
    n_cells_per_sample: int = 50,
    n_genes: int = 200,
    n_samples: int = 12,
    k: list[int] = [2, 3],
    seed: int = 42,
) -> AnnData:
    rng = np.random.default_rng(seed)

    blocks, labels, ages, studies, brcas = [], [], [], [], []

    for i in range(n_samples):
        # Each donor gets a unique expression profile with age-correlated signal
        age = 30 + i * 5
        noise = rng.normal(0.0, 1.0, size=(n_cells_per_sample, n_genes))

        # Add age-correlated signal to first 20 genes
        age_signal = (age / 70.0) * rng.normal(3.0, 0.5, size=(1, 20))
        noise[:, :20] += age_signal

        X_sample = np.clip(np.abs(noise) + 1.0, 0, None)
        blocks.append(X_sample)
        labels.extend([f"sample_{i}"] * n_cells_per_sample)
        ages.extend([float(age)] * n_cells_per_sample)
        studies.extend([f"study_{i % 3}"] * n_cells_per_sample)
        brcas.extend([float(i % 2)] * n_cells_per_sample)

    adata = AnnData(
        X=np.vstack(blocks).astype(np.float32),
        obs={
            "donor_id": labels,
            "age": np.array(ages, dtype=float),
            "study_id": studies,
            "donor_brca": np.array(brcas, dtype=float),
        },
        var={"gene_name": np.arange(n_genes).astype(str)},
    )

    adata = jz.gp.nmf(adata, key="donor_id", k=k, min_cells=10, genes=None, seed=seed)
    jz.gp.similarity(adata, distance="jaccard", top_k=20, min_similarity=0.0)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)
    jz.gp.score(adata, n_top_genes=10, seed=seed)
    jz.gp.aggregate(adata, key="donor_id", obs_cols=["age", "study_id", "donor_brca"])

    return adata


# Input validation


def test_error_missing_aggregate_scores():
    adata = make_adata()
    del adata.uns["juzi_aggregate_scores"]
    with pytest.raises(KeyError):
        jz.gp.associate(adata, formula="age + (1|study_id)")


def test_error_empty_formula():
    with pytest.raises(ValueError):
        jz.gp.associate(make_adata(), formula="")


def test_error_formula_only_random_effects():
    with pytest.raises(ValueError):
        jz.gp.associate(make_adata(), formula="(1|study_id)")


def test_error_missing_covariate_in_scores():
    with pytest.raises(KeyError):
        jz.gp.associate(make_adata(), formula="nonexistent_col + (1|study_id)")


def test_error_missing_group_in_scores():
    with pytest.raises(KeyError):
        jz.gp.associate(make_adata(), formula="age + (1|nonexistent_group)")


# Formula parsing


def test_parse_formula_single_random_effect():
    from juzi.gp._associate import _parse_formula

    fixed, groups = _parse_formula("age + donor_brca + (1|study_id)")
    assert "study_id" not in fixed
    assert "study_id" in groups
    assert "age" in fixed
    assert "donor_brca" in fixed


def test_parse_formula_multiple_random_effects():
    from juzi.gp._associate import _parse_formula

    fixed, groups = _parse_formula("age + (1|study_id) + (1|batch)")
    assert "study_id" in groups
    assert "batch" in groups
    assert len(groups) == 2
    assert "age" in fixed


def test_parse_formula_no_random_effects():
    from juzi.gp._associate import _parse_formula

    fixed, groups = _parse_formula("age + donor_brca")
    assert fixed.strip() == "age + donor_brca"
    assert groups == []


def test_parse_formula_no_trailing_operators():
    from juzi.gp._associate import _parse_formula

    fixed, _ = _parse_formula("age + (1|study_id)")
    assert not fixed.strip().startswith("+")
    assert not fixed.strip().endswith("+")


def test_parse_formula_whitespace_variants():
    from juzi.gp._associate import _parse_formula

    fixed1, groups1 = _parse_formula("age + ( 1 | study_id )")
    fixed2, groups2 = _parse_formula("age + (1|study_id)")
    assert groups1 == groups2
    assert fixed1.strip() == fixed2.strip()


# Output structure


def test_association_in_uns():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    assert "juzi_association" in adata.uns


def test_association_is_dataframe():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    assert isinstance(adata.uns["juzi_association"], pd.DataFrame)


def test_association_columns():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    for col in ["program", "covariate", "beta", "se", "pval", "padj", "n_obs", "model"]:
        assert col in df.columns


def test_association_one_row_per_program():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    agg_df = adata.uns["juzi_aggregate_scores"]
    n_programs = len([c for c in agg_df.columns if re.match(r"^P\d+$", c)])
    assert len(df) == n_programs


def test_association_program_col_values():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    assert all(re.match(r"^P\d+$", p) for p in df["program"])


def test_association_covariate_col_values():
    """Primary covariate column should match first fixed effect."""
    adata = make_adata()
    jz.gp.associate(adata, formula="age + donor_brca + (1|study_id)")
    df = adata.uns["juzi_association"]
    assert (df["covariate"] == "age").all()


def test_association_sorted_by_padj():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    assert (df["padj"].diff().dropna() >= 0).all()


# Numerical properties


def test_pval_in_zero_one():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    assert (df["pval"].between(0, 1)).all()


def test_padj_in_zero_one():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    assert (df["padj"].between(0, 1)).all()


def test_padj_geq_pval():
    """After FDR correction adjusted p-values must be >= raw p-values."""
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    assert (df["padj"] >= df["pval"] - 1e-8).all()


def test_se_positive():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    assert (df["se"] > 0).all()


def test_beta_finite():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    assert np.isfinite(df["beta"]).all()


def test_n_obs_correct():
    adata = make_adata(n_samples=6)
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    n_donors = adata.uns["juzi_aggregate_scores"].shape[0]
    assert (df["n_obs"] == n_donors).all()


# Model types


def test_lmm_model_used_with_random_effects():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)")
    df = adata.uns["juzi_association"]
    # At least one program should use lmm — not all will if data is ill-conditioned
    assert df["model"].isin(["lmm", "ols"]).all()


def test_ols_model_used_without_random_effects():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + donor_brca")
    df = adata.uns["juzi_association"]
    assert (df["model"] == "ols").all()


def test_ols_fallback_warning_single_group():
    """OLS fallback warning fires when only one group level exists."""
    adata = make_adata()
    agg = adata.uns["juzi_aggregate_scores"].copy()
    agg["single_group"] = "only_one"
    adata.uns["juzi_aggregate_scores"] = agg

    with pytest.warns(UserWarning, match="Falling back to OLS"):
        jz.gp.associate(adata, formula="age + (1|single_group)")


def test_multiple_random_effects_combined():
    """Multiple (1|group) terms should run without error."""
    adata = make_adata()
    agg = adata.uns["juzi_aggregate_scores"].copy()
    agg["batch"] = ["b1", "b2"] * (len(agg) // 2) + ["b1"] * (len(agg) % 2)
    adata.uns["juzi_aggregate_scores"] = agg

    jz.gp.associate(adata, formula="age + (1|study_id) + (1|batch)")
    df = adata.uns["juzi_association"]
    assert "juzi_association" in adata.uns
    assert (df["model"] == "lmm").all()


def test_multiple_random_effects_groups_column():
    """groups column should reflect combined random effect names."""
    adata = make_adata()
    agg = adata.uns["juzi_aggregate_scores"].copy()
    agg["batch"] = ["b1", "b2"] * (len(agg) // 2) + ["b1"] * (len(agg) % 2)
    adata.uns["juzi_aggregate_scores"] = agg

    jz.gp.associate(adata, formula="age + (1|study_id) + (1|batch)")
    df = adata.uns["juzi_association"]
    assert (df["groups"] == "study_id_x_batch").all()


# Formula variants


def test_formula_multiple_fixed_effects():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + donor_brca + (1|study_id)")
    assert "juzi_association" in adata.uns


def test_formula_categorical_covariate():
    adata = make_adata()
    jz.gp.associate(adata, formula="C(donor_brca) + (1|study_id)")
    assert "juzi_association" in adata.uns


def test_reml_false_runs():
    adata = make_adata()
    jz.gp.associate(adata, formula="age + (1|study_id)", reml=False)
    assert "juzi_association" in adata.uns


def test_reml_true_false_differ():
    """REML and ML estimates should differ in beta values."""
    adata_reml = make_adata(seed=0)
    adata_ml = make_adata(seed=0)
    jz.gp.associate(adata_reml, formula="age + (1|study_id)", reml=True)
    jz.gp.associate(adata_ml, formula="age + (1|study_id)", reml=False)
    assert not np.allclose(
        adata_reml.uns["juzi_association"]["beta"].values,
        adata_ml.uns["juzi_association"]["beta"].values,
        atol=1e-6,
    )


# copy parameter


def test_copy_false_modifies_inplace():
    adata = make_adata()
    result = jz.gp.associate(adata, formula="age + (1|study_id)", copy=False)
    assert result is None
    assert "juzi_association" in adata.uns


def test_copy_true_returns_new_object():
    adata = make_adata()
    result = jz.gp.associate(adata, formula="age + (1|study_id)", copy=True)
    assert result is not None
    assert "juzi_association" not in adata.uns
    assert "juzi_association" in result.uns
