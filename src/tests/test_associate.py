import pandas as pd
import pytest
import juzi as jz


def test_score_associate_outputs(adata_aggregated):
    adata = adata_aggregated.copy()
    jz.gp.score_associate(adata, formula="age + (1|study_id)")
    assert "juzi_association" in adata.uns
    assert "juzi_association_meta" in adata.uns
    assert isinstance(adata.uns["juzi_association"], pd.DataFrame)


def test_score_associate_program_column_uses_P_names(adata_associated):
    assert adata_associated.uns["juzi_association"]["program"].str.match(r"^P\d+$").all()


def test_score_associate_formula_only_random_effects_raises(adata_aggregated):
    with pytest.raises(ValueError):
        jz.gp.score_associate(adata_aggregated, formula="(1|study_id)")


def test_score_associate_missing_variable_raises(adata_aggregated):
    with pytest.raises(KeyError):
        jz.gp.score_associate(adata_aggregated, formula="missing + (1|study_id)")
