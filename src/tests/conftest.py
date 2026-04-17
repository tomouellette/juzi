import numpy as np
import pytest
from anndata import AnnData
import matplotlib

matplotlib.use("Agg")

import juzi as jz


def _make_expression(
    n_cells_per_sample: int = 40,
    n_genes: int = 120,
    n_samples: int = 6,
    seed: int = 42,
    strong_groups: bool = True,
):
    rng = np.random.default_rng(seed)

    # Donors alternate between two latent programs so clustering is not trivial.
    profile_a = np.concatenate(
        [
            rng.normal(20, 1, size=n_genes // 3),
            rng.normal(4, 0.5, size=n_genes - n_genes // 3),
        ]
    )
    profile_b = np.concatenate(
        [
            rng.normal(4, 0.5, size=n_genes // 3),
            rng.normal(20, 1, size=n_genes - n_genes // 3),
        ]
    )

    if not strong_groups:
        profile_a = rng.normal(8, 1, size=n_genes)
        profile_b = rng.normal(9, 1, size=n_genes)

    blocks, donors, ages, studies, brcas = [], [], [], [], []
    for i in range(n_samples):
        base = profile_a if i % 2 == 0 else profile_b
        noise = rng.normal(0.0, 1.0, size=(n_cells_per_sample, n_genes))
        X = np.clip(base + noise, 0, None).astype(np.float32)
        blocks.append(X)
        donors.extend([f"sample_{i}"] * n_cells_per_sample)
        ages.extend([30.0 + 5.0 * i] * n_cells_per_sample)
        studies.extend([f"study_{i % 3}"] * n_cells_per_sample)
        brcas.extend([float(i % 2)] * n_cells_per_sample)

    adata = AnnData(
        X=np.vstack(blocks),
        obs={
            "donor_id": donors,
            "age": np.array(ages, dtype=float),
            "study_id": np.array(studies, dtype=object),
            "donor_brca": np.array(brcas, dtype=float),
        },
        var={
            "gene_name": np.array([f"G{i}" for i in range(n_genes)], dtype=object),
            "highly_variable": np.array(
                [True] * min(40, n_genes) + [False] * max(0, n_genes - min(40, n_genes))
            ),
        },
    )
    adata.obsm["X_umap"] = rng.normal(size=(adata.n_obs, 2)).astype(np.float32)
    return adata


@pytest.fixture
def adata_raw():
    return _make_expression()


@pytest.fixture
def adata_nmf():
    adata = _make_expression()
    return jz.gp.nmf_fit(
        adata,
        key="donor_id",
        k=[2, 3],
        min_cells=10,
        genes=None,
        gene_names_col="gene_name",
        keep_scores=True,
        center=False,
        seed=42,
    )


@pytest.fixture
def adata_pruned(adata_nmf):
    adata = adata_nmf.copy()
    jz.gp.nmf_prune(
        adata,
        top_k=20,
        min_similarity=0.2,
        min_other_resolutions=1,
        matching="hungarian",
    )
    return adata


@pytest.fixture
def adata_similarity(adata_pruned):
    adata = adata_pruned.copy()
    jz.gp.similarity(
        adata,
        metric="jaccard",
        top_k=20,
        intra_sample=False,
        drop_zeros=True,
        use_combined=False,
    )
    return adata


@pytest.fixture
def adata_cluster_centroid(adata_similarity):
    adata = adata_similarity.copy()
    jz.gp.programs_cluster(
        adata,
        strategy="centroid",
        threshold=0.25,
        min_cluster=1,
        method="average",
        n_top_genes=15,
    )
    return adata


@pytest.fixture
def adata_cluster_progressive(adata_similarity):
    adata = adata_similarity.copy()
    jz.gp.programs_cluster(
        adata,
        strategy="progressive",
        top_k=15,
        min_overlap=5,
        min_founder_overlaps=1,
        min_cluster=1,
    )
    return adata


@pytest.fixture
def adata_scored(adata_cluster_centroid):
    adata = adata_cluster_centroid.copy()
    jz.gp.score_cells(
        adata,
        n_top_genes=10,
        n_control_genes=10,
        gene_names_col="gene_name",
        seed=42,
    )
    return adata


@pytest.fixture
def adata_aggregated(adata_scored):
    adata = adata_scored.copy()
    jz.gp.score_aggregate(
        adata,
        key="donor_id",
        obs_cols=["age", "study_id", "donor_brca"],
        agg="mean",
        min_cells=10,
    )
    return adata


@pytest.fixture
def adata_associated(adata_aggregated):
    adata = adata_aggregated.copy()
    jz.gp.score_associate(adata, formula="age + (1|study_id)")
    return adata


@pytest.fixture
def gene_sets(adata_cluster_centroid):
    genes = list(adata_cluster_centroid.uns["juzi_G_genes"])
    return {
        "GS_A": genes[:20],
        "GS_B": genes[10:35],
        "GS_C": genes[30:50],
    }


@pytest.fixture
def adata_annotated(adata_cluster_centroid, gene_sets):
    adata = adata_cluster_centroid.copy()
    jz.gp.programs_annotate(adata, gene_sets=gene_sets, min_overlap=1)
    return adata


@pytest.fixture
def adata_stable(adata_cluster_progressive):
    adata = adata_cluster_progressive.copy()
    jz.gp.programs_stability(adata, top_k=10, min_program_genes=3)
    return adata
