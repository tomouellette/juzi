import pytest
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
    """AnnData fit through full pipeline, ready for ut functions."""
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

    adata = jz.gp.nmf(adata, key="donor_id", k=k, min_cells=10, genes=None, seed=seed)
    jz.gp.similarity(adata, distance="jaccard", top_k=20)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)

    return adata


# juzi.ut.program_genes


class TestProgramGenes:
    def test_error_missing_cluster_names(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_names"]
        with pytest.raises(KeyError):
            jz.ut.program_donors(adata)

    def test_error_missing_cluster_G(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_G"]
        with pytest.raises(KeyError):
            jz.ut.program_genes(adata)

    def test_error_missing_cluster_labels(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.ut.program_genes(adata)

    def test_error_missing_G_genes(self):
        adata = make_adata()
        del adata.uns["juzi_G_genes"]
        with pytest.raises(KeyError):
            jz.ut.program_genes(adata)

    def test_error_n_top_genes_below_one(self):
        adata = make_adata()
        with pytest.raises(ValueError):
            jz.ut.program_genes(adata, n_top_genes=0)

    def test_error_n_top_genes_exceeds_genes(self):
        adata = make_adata()
        with pytest.raises(ValueError):
            jz.ut.program_genes(adata, n_top_genes=100_000)

    def test_returns_dict(self):
        adata = make_adata()
        result = jz.ut.program_genes(adata)
        assert isinstance(result, dict)

    def test_keys_match_program_labels(self):
        adata = make_adata()
        result = jz.ut.program_genes(adata)
        unique_C = np.unique(adata.uns["juzi_cluster_labels"])
        expected = {f"C{int(c)}" for c in unique_C}
        assert set(result.keys()) == expected

    def test_values_are_lists(self):
        adata = make_adata()
        result = jz.ut.program_genes(adata)
        for genes in result.values():
            assert isinstance(genes, list)

    def test_genes_are_strings(self):
        adata = make_adata()
        result = jz.ut.program_genes(adata)
        for genes in result.values():
            assert all(isinstance(g, str) for g in genes)

    def test_n_top_genes_length(self):
        adata = make_adata()
        n_top = 10
        result = jz.ut.program_genes(adata, n_top_genes=n_top)
        for genes in result.values():
            assert len(genes) == n_top

    def test_genes_in_nmf_gene_set(self):
        """All returned genes must come from juzi_G_genes."""
        adata = make_adata()
        result = jz.ut.program_genes(adata)
        nmf_genes = set(adata.uns["juzi_G_genes"])
        for genes in result.values():
            assert all(g in nmf_genes for g in genes)

    def test_genes_unique_per_program(self):
        adata = make_adata()
        result = jz.ut.program_genes(adata, n_top_genes=20)
        for genes in result.values():
            assert len(genes) == len(set(genes))

    def test_specificity_true_differs_from_false(self):
        adata_spec = make_adata(seed=0)
        adata_raw = make_adata(seed=0)
        result_spec = jz.ut.program_genes(
            adata_spec, n_top_genes=10, use_specificity=True
        )
        result_raw = jz.ut.program_genes(
            adata_raw, n_top_genes=10, use_specificity=False
        )
        any_differ = any(set(result_spec[p]) != set(result_raw[p]) for p in result_spec)
        assert any_differ


# juzi.ut.program_compare


class TestProgramCompare:
    def test_error_missing_cluster_G_a(self):
        adata_a = make_adata(seed=0)
        adata_b = make_adata(seed=1)
        del adata_a.uns["juzi_cluster_G"]
        with pytest.raises(KeyError):
            jz.ut.program_compare(adata_a, adata_b)

    def test_error_missing_cluster_G_b(self):
        adata_a = make_adata(seed=0)
        adata_b = make_adata(seed=1)
        del adata_b.uns["juzi_cluster_G"]
        with pytest.raises(KeyError):
            jz.ut.program_compare(adata_a, adata_b)

    def test_returns_dataframe(self):
        adata_a = make_adata(seed=0)
        adata_b = make_adata(seed=1)
        result = jz.ut.program_compare(adata_a, adata_b)
        assert isinstance(result, pd.DataFrame)

    def test_shape(self):
        adata_a = make_adata(seed=0)
        adata_b = make_adata(seed=1)
        result = jz.ut.program_compare(adata_a, adata_b)
        n_prog_a = len(np.unique(adata_a.uns["juzi_cluster_labels"]))
        n_prog_b = len(np.unique(adata_b.uns["juzi_cluster_labels"]))
        assert result.shape == (n_prog_a, n_prog_b)

    def test_index_labels(self):
        adata_a = make_adata(seed=0)
        adata_b = make_adata(seed=1)
        result = jz.ut.program_compare(adata_a, adata_b)
        unique_a = np.unique(adata_a.uns["juzi_cluster_labels"])
        expected = [f"C{int(c)}" for c in unique_a]
        assert result.index.tolist() == expected

    def test_column_labels(self):
        adata_a = make_adata(seed=0)
        adata_b = make_adata(seed=1)
        result = jz.ut.program_compare(adata_a, adata_b)
        unique_b = np.unique(adata_b.uns["juzi_cluster_labels"])
        expected = [f"C{int(c)}" for c in unique_b]
        assert result.columns.tolist() == expected

    def test_values_in_zero_one(self):
        adata_a = make_adata(seed=0)
        adata_b = make_adata(seed=1)
        result = jz.ut.program_compare(adata_a, adata_b)
        assert (result.values >= 0).all()
        assert (result.values <= 1).all()

    def test_self_compare_diagonal(self):
        """Comparing an adata with itself should give high diagonal values."""
        adata = make_adata(seed=0)
        result = jz.ut.program_compare(adata, adata, n_top_genes=20)
        # Diagonal should be 1.0 since identical gene sets
        np.testing.assert_allclose(np.diag(result.values), 1.0, atol=1e-6)

    def test_symmetric_for_self_compare(self):
        adata = make_adata(seed=0)
        result = jz.ut.program_compare(adata, adata)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-6)

    def test_n_top_genes_parameter(self):
        adata_a = make_adata(seed=0)
        adata_b = make_adata(seed=1)
        result_10 = jz.ut.program_compare(adata_a, adata_b, n_top_genes=10)
        result_50 = jz.ut.program_compare(adata_a, adata_b, n_top_genes=50)
        # Results should differ with different n_top_genes
        assert not np.allclose(result_10.values, result_50.values)


# juzi.ut.program_donors


class TestProgramDonors:
    def test_error_missing_cluster_labels(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.ut.program_donors(adata)

    def test_error_missing_cluster_names(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_names"]
        with pytest.raises(KeyError):
            jz.ut.program_donors(adata)

    def test_error_missing_cluster_samples(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_samples"]
        with pytest.raises(KeyError):
            jz.ut.program_donors(adata)

    def test_returns_dataframe(self):
        adata = make_adata()
        result = jz.ut.program_donors(adata)
        assert isinstance(result, pd.DataFrame)

    def test_columns_match_program_labels(self):
        adata = make_adata()
        result = jz.ut.program_donors(adata)
        unique_C = np.unique(adata.uns["juzi_cluster_labels"])
        expected = [f"C{int(c)}" for c in unique_C]
        assert result.columns.tolist() == expected

    def test_index_contains_donors(self):
        """Index must contain donors from cluster_samples."""
        adata = make_adata()
        result = jz.ut.program_donors(adata)
        samples = adata.uns["juzi_cluster_samples"]
        all_donors = {d for donors in samples.values() for d in donors}
        assert all_donors.issubset(set(result.index.tolist()))

    def test_values_non_negative_integers(self):
        adata = make_adata()
        result = jz.ut.program_donors(adata)
        assert (result.values >= 0).all()
        assert result.values.dtype in [np.int32, np.int64, int]

    def test_zero_for_non_contributing_donors(self):
        """Donors not in a program's cluster_samples must have count 0."""
        adata = make_adata()
        result = jz.ut.program_donors(adata)
        samples = adata.uns["juzi_cluster_samples"]
        for col in result.columns:
            c_int = int(col.replace("C", ""))
            program_donors = set(samples[c_int])
            for donor in result.index:
                if donor not in program_donors:
                    assert result.loc[donor, col] == 0
                else:
                    assert result.loc[donor, col] > 0

    def test_total_counts_match_cluster_size(self):
        """Sum of factor counts per program must equal n_factors in that cluster."""
        adata = make_adata()
        result = jz.ut.program_donors(adata)
        labels = adata.uns["juzi_cluster_labels"]
        for col in result.columns:
            c_int = int(col.replace("C", ""))
            n_factors = (labels == c_int).sum()
            assert result[col].sum() == n_factors

    def test_shape_reasonable(self):
        """Shape must be (n_unique_donors × n_programs)."""
        adata = make_adata()
        result = jz.ut.program_donors(adata)
        n_programs = len(np.unique(adata.uns["juzi_cluster_labels"]))
        assert result.shape[1] == n_programs
        assert result.shape[0] >= 1
