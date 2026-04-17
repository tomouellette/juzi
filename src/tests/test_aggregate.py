import pandas as pd
import pytest
import juzi as jz


def test_score_aggregate_outputs(adata_scored):
    adata = adata_scored.copy()
    jz.gp.score_aggregate(
        adata,
        key="donor_id",
        obs_cols=["age", "study_id"],
        agg="mean",
        min_cells=10,
    )
    assert "juzi_aggregate_scores" in adata.uns
    assert "juzi_aggregate_meta" in adata.uns
    assert isinstance(adata.uns["juzi_aggregate_scores"], pd.DataFrame)


def test_score_aggregate_program_columns_named_P(adata_aggregated):
    df = adata_aggregated.uns["juzi_aggregate_scores"]
    program_cols = [c for c in df.columns if c.startswith("P")]
    assert len(program_cols) == adata_aggregated.obsm["juzi_program_scores"].shape[1]


def test_score_aggregate_propagates_covariates(adata_aggregated):
    df = adata_aggregated.uns["juzi_aggregate_scores"]
    assert "age" in df.columns
    assert "study_id" in df.columns


def test_score_aggregate_invalid_agg_raises(adata_scored):
    with pytest.raises(ValueError):
        jz.gp.score_aggregate(adata_scored, key="donor_id", agg="sum")
