import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from anndata import AnnData
import juzi as jz

matplotlib.use("Agg")


# Fixtures


def make_adata(
    n_cells_per_sample: int = 50,
    n_genes: int = 200,
    n_samples: int = 4,
    k: list[int] = [2, 3],
    seed: int = 42,
) -> AnnData:
    """AnnData fit through full pipeline, ready for annotate."""
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


def make_gene_sets(adata: AnnData, n_sets: int = 5, seed: int = 0) -> dict:
    """Make synthetic gene sets with guaranteed overlap with juzi_G_genes."""
    rng = np.random.default_rng(seed)
    all_genes = adata.uns["juzi_G_genes"]
    gene_sets = {}
    for i in range(n_sets):
        size = rng.integers(10, 30)
        genes = rng.choice(all_genes, size=size, replace=False).tolist()
        gene_sets[f"GS_{i}"] = genes
    return gene_sets


# juzi.gp.programs_annotate


class TestGpAnnotate:
    def test_error_missing_cluster_G(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_G"]
        with pytest.raises(KeyError):
            jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))

    def test_error_missing_cluster_labels(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))

    def test_error_missing_G_genes(self):
        adata = make_adata()
        del adata.uns["juzi_G_genes"]
        with pytest.raises(KeyError):
            jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))

    def test_error_n_top_genes_below_one(self):
        adata = make_adata()
        with pytest.raises(ValueError):
            jz.gp.programs_annotate(
                adata, gene_sets=make_gene_sets(adata), n_top_genes=0
            )

    def test_error_n_top_genes_exceeds_genes(self):
        adata = make_adata()
        with pytest.raises(ValueError):
            jz.gp.programs_annotate(
                adata, gene_sets=make_gene_sets(adata), n_top_genes=100_000
            )

    def test_error_empty_gene_sets(self):
        adata = make_adata()
        with pytest.raises(ValueError):
            jz.gp.programs_annotate(adata, gene_sets={})

    def test_error_no_overlap_with_background(self):
        adata = make_adata()
        with pytest.raises(ValueError):
            jz.gp.programs_annotate(
                adata,
                gene_sets={"FAKE": ["FAKE_GENE_1", "FAKE_GENE_2"]},
            )

    def test_error_invalid_gene_sets_type(self):
        adata = make_adata()
        with pytest.raises(ValueError):
            jz.gp.programs_annotate(adata, gene_sets="not_a_dict")

    # Output structure

    def test_annotation_in_uns(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        assert "juzi_annotation" in adata.uns

    def test_annotation_is_dataframe(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        assert isinstance(adata.uns["juzi_annotation"], pd.DataFrame)

    def test_annotation_columns(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        for col in [
            "program",
            "gene_set",
            "jaccard",
            "n_overlap",
            "n_program",
            "n_geneset",
            "n_background",
            "pval",
            "padj",
            "overlap_genes",
        ]:
            assert col in df.columns

    def test_annotation_row_count(self):
        """Should have one row per program × gene_set pair."""
        adata = make_adata()
        gene_sets = make_gene_sets(adata, n_sets=5)
        jz.gp.programs_annotate(adata, gene_sets=gene_sets)
        df = adata.uns["juzi_annotation"]
        n_programs = len(np.unique(adata.uns["juzi_cluster_labels"]))
        assert len(df) == n_programs * len(gene_sets)

    def test_annotation_program_labels(self):
        """Program column should contain C0, C1, ... format."""
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        pattern = r"^C\d+$"
        assert df["program"].str.match(pattern).all()

    def test_annotation_sorted_by_padj(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        assert (df["padj"].diff().dropna() >= 0).all()

    # Numerical properties

    def test_jaccard_in_zero_one(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        assert df["jaccard"].between(0, 1).all()

    def test_pval_in_zero_one(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        assert df["pval"].between(0, 1).all()

    def test_padj_in_zero_one(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        assert df["padj"].between(0, 1).all()

    def test_padj_geq_pval(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        assert (df["padj"] >= df["pval"] - 1e-8).all()

    def test_n_overlap_leq_n_program(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        assert (df["n_overlap"] <= df["n_program"]).all()

    def test_n_overlap_leq_n_geneset(self):
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        assert (df["n_overlap"] <= df["n_geneset"]).all()

    def test_n_background_constant(self):
        """n_background should equal len(juzi_G_genes) for all rows."""
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        assert (df["n_background"] == len(adata.uns["juzi_G_genes"])).all()

    def test_overlap_genes_consistent_with_n_overlap(self):
        """overlap_genes comma count should match n_overlap."""
        adata = make_adata()
        jz.gp.programs_annotate(adata, gene_sets=make_gene_sets(adata))
        df = adata.uns["juzi_annotation"]
        for _, row in df.iterrows():
            if row["n_overlap"] == 0:
                assert row["overlap_genes"] == ""
            else:
                assert len(row["overlap_genes"].split(",")) == row["n_overlap"]

    # Gene set input variants

    def test_accepts_plain_dict(self):
        adata = make_adata()
        gene_sets = make_gene_sets(adata)
        jz.gp.programs_annotate(adata, gene_sets=gene_sets)
        assert "juzi_annotation" in adata.uns

    def test_accepts_mg_object(self):
        """Objects with as_dict() method should be accepted."""
        adata = make_adata()

        class FakeGeneSet:
            def as_dict(self):
                return make_gene_sets(adata)

        jz.gp.programs_annotate(adata, gene_sets=FakeGeneSet())
        assert "juzi_annotation" in adata.uns

    def test_mg_object_equivalent_to_dict(self):
        """as_dict() object and plain dict should produce identical results."""
        adata = make_adata()
        gene_sets = make_gene_sets(adata, seed=0)

        class FakeGeneSet:
            def as_dict(self):
                return gene_sets

        adata_dict = make_adata(seed=42)
        adata_obj = make_adata(seed=42)

        jz.gp.programs_annotate(adata_dict, gene_sets=gene_sets)
        jz.gp.programs_annotate(adata_obj, gene_sets=FakeGeneSet())

        pd.testing.assert_frame_equal(
            adata_dict.uns["juzi_annotation"],
            adata_obj.uns["juzi_annotation"],
        )

    # Gene selection

    def test_combined_true_runs(self):
        adata = make_adata()
        jz.gp.programs_annotate(
            adata, gene_sets=make_gene_sets(adata), use_combined=True
        )
        assert "juzi_annotation" in adata.uns

    def test_combined_false_runs(self):
        adata = make_adata()
        jz.gp.programs_annotate(
            adata, gene_sets=make_gene_sets(adata), use_combined=False
        )
        assert "juzi_annotation" in adata.uns

    def test_combined_affects_results(self):
        adata_spec = make_adata(seed=0)
        adata_raw = make_adata(seed=0)
        jz.gp.programs_annotate(
            adata_spec, gene_sets=make_gene_sets(adata_spec), use_combined=True
        )
        jz.gp.programs_annotate(
            adata_raw, gene_sets=make_gene_sets(adata_raw), use_combined=False
        )
        assert not adata_spec.uns["juzi_annotation"]["jaccard"].equals(
            adata_raw.uns["juzi_annotation"]["jaccard"]
        )

    # copy parameter

    def test_copy_false_modifies_inplace(self):
        adata = make_adata()
        result = jz.gp.programs_annotate(
            adata, gene_sets=make_gene_sets(adata), copy=False
        )
        assert result is None
        assert "juzi_annotation" in adata.uns

    def test_copy_true_returns_new_object(self):
        adata = make_adata()
        result = jz.gp.programs_annotate(
            adata, gene_sets=make_gene_sets(adata), copy=True
        )
        assert result is not None
        assert "juzi_annotation" not in adata.uns
        assert "juzi_annotation" in result.uns
