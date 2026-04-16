# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from anndata import AnnData
import juzi as jz

matplotlib.use("Agg")


# Fixtures


def make_adata(
    n_cells_per_sample: int = 50,
    n_genes: int = 200,
    n_samples: int = 6,
    k: list[int] = [2, 3],
    seed: int = 42,
) -> AnnData:
    """AnnData fit through programs_cluster — ready for jackknife,
    programs_remove, and score_classify."""
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
    jz.gp.similarity_compute(adata, distance="jaccard", top_k=20, use_combined=True)
    jz.gp.programs_cluster(adata, threshold=0.3, min_cluster=2)

    return adata


def make_adata_scored(seed: int = 42) -> AnnData:
    """AnnData fit through score_cells — ready for score_classify."""
    adata = make_adata(seed=seed)
    jz.gp.score_cells(adata, n_top_genes=10, seed=seed, use_combined=True)
    return adata


# juzi.gp.programs_jackknife


class TestProgramsJackknife:
    def test_error_missing_cluster_G(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_G"]
        with pytest.raises(KeyError):
            jz.gp.programs_jackknife(adata)

    def test_error_missing_cluster_labels(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.gp.programs_jackknife(adata)

    def test_error_missing_juzi_similarity(self):
        adata = make_adata()
        del adata.uns["juzi_similarity"]
        with pytest.raises(KeyError):
            jz.gp.programs_jackknife(adata)

    def test_error_missing_juzi_similarity_idx(self):
        adata = make_adata()
        del adata.uns["juzi_similarity_idx"]
        with pytest.raises(KeyError):
            jz.gp.programs_jackknife(adata)

    def test_error_missing_juzi_names(self):
        adata = make_adata()
        del adata.uns["juzi_names"]
        with pytest.raises(KeyError):
            jz.gp.programs_jackknife(adata)

    def test_error_n_top_genes_below_one(self):
        adata = make_adata()
        with pytest.raises(ValueError):
            jz.gp.programs_jackknife(adata, n_top_genes=0)

    def test_output_fields_present(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        assert "juzi_jackknife" in adata.uns

    def test_jackknife_dict_keys(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        jack = adata.uns["juzi_jackknife"]
        for key in [
            "stability",
            "stability_matrix",
            "donors",
            "n_top_genes",
            "threshold",
            "min_cluster",
        ]:
            assert key in jack, f"'{key}' missing from juzi_jackknife"

    def test_stability_shape(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        K = len(np.unique(adata.uns["juzi_cluster_labels"]))
        assert adata.uns["juzi_jackknife"]["stability"].shape == (K,)

    def test_stability_matrix_shape(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        jack = adata.uns["juzi_jackknife"]
        K = len(np.unique(adata.uns["juzi_cluster_labels"]))
        N = len(jack["donors"])
        assert jack["stability_matrix"].shape == (K, N)

    def test_stability_in_zero_one(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        s = adata.uns["juzi_jackknife"]["stability"]
        assert (s >= 0.0).all()
        assert (s <= 1.0).all()

    def test_stability_matrix_in_zero_one(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        S = adata.uns["juzi_jackknife"]["stability_matrix"]
        assert (S >= 0.0).all()
        assert (S <= 1.0).all()

    def test_stability_equals_mean_of_matrix(self):
        """stability must equal mean of stability_matrix over valid donors."""
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        jack = adata.uns["juzi_jackknife"]
        np.testing.assert_allclose(
            jack["stability"],
            jack["stability_matrix"].mean(axis=1),
            atol=1e-5,
        )

    def test_donors_list_length_matches_matrix(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        jack = adata.uns["juzi_jackknife"]
        assert len(jack["donors"]) == jack["stability_matrix"].shape[1]

    def test_donors_are_valid_sample_names(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        donors = set(adata.uns["juzi_jackknife"]["donors"])
        names = set(adata.obs["donor_id"].unique().tolist())
        assert donors.issubset(names)

    def test_stored_params_match_inputs(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=15)
        jack = adata.uns["juzi_jackknife"]
        assert jack["n_top_genes"] == 15

    def test_use_combined_false_runs(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10, use_combined=False)
        assert "juzi_jackknife" in adata.uns

    def test_copy_false_modifies_inplace(self):
        adata = make_adata()
        result = jz.gp.programs_jackknife(adata, n_top_genes=10, copy=False)
        assert result is None
        assert "juzi_jackknife" in adata.uns

    def test_copy_true_returns_new_object(self):
        adata = make_adata()
        result = jz.gp.programs_jackknife(adata, n_top_genes=10, copy=True)
        assert result is not None
        assert "juzi_jackknife" not in adata.uns
        assert "juzi_jackknife" in result.uns

    def test_parallel_threads(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10, n_jobs=2, prefer="threads")
        assert "juzi_jackknife" in adata.uns

    def test_well_separated_programs_high_stability(self):
        """Programs from clearly distinct expression profiles should be stable."""
        adata = make_adata(n_samples=9, seed=0)
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        stability = adata.uns["juzi_jackknife"]["stability"]
        assert stability.mean() > 0.1


# juzi.pl.programs_jackknife


class TestPlProgramsJackknife:
    def test_error_missing_jackknife(self):
        adata = make_adata()
        with pytest.raises(KeyError):
            jz.pl.programs_jackknife(adata)

    def test_error_missing_cluster_labels(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.pl.programs_jackknife(adata)

    def test_returns_axes(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        ax = jz.pl.programs_jackknife(adata)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_accepts_ax(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        fig, ax = plt.subplots()
        result = jz.pl.programs_jackknife(adata, ax=ax)
        assert result is ax
        plt.close("all")

    def test_custom_palette(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        labels = np.unique(adata.uns["juzi_cluster_labels"])
        palette = {int(c): "#ff0000" for c in labels}
        ax = jz.pl.programs_jackknife(adata, palette=palette)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_cmap(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        ax = jz.pl.programs_jackknife(adata, cmap="Blues_r")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_hide_donors(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        ax = jz.pl.programs_jackknife(adata, show_donors=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_figsize(self):
        adata = make_adata()
        jz.gp.programs_jackknife(adata, n_top_genes=10)
        ax = jz.pl.programs_jackknife(adata, figsize=(8, 4))
        assert isinstance(ax, plt.Axes)
        plt.close("all")
