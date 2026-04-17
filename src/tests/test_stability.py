import numpy as np
import pytest
import juzi as jz


def test_programs_stability_outputs(adata_stable):
    assert "juzi_stability" in adata_stable.uns
    assert "juzi_stability_meta" in adata_stable.uns


def test_programs_stability_shapes(adata_stable):
    stab = adata_stable.uns["juzi_stability"]
    matrix = stab["matrix"]
    score = stab["score"]
    assert matrix.shape[0] == len(stab["programs"])
    assert matrix.shape[1] == len(stab["donors"])
    assert len(score) == len(stab["programs"])


def test_programs_stability_scores_bounded(adata_stable):
    score = adata_stable.uns["juzi_stability"]["score"]
    assert np.nanmin(score) >= 0.0
    assert np.nanmax(score) <= 1.0


def test_programs_stability_requires_cluster_genes(adata_cluster_progressive):
    adata = adata_cluster_progressive.copy()
    del adata.uns["juzi_cluster_genes"]
    with pytest.raises(KeyError):
        jz.gp.programs_stability(adata)
