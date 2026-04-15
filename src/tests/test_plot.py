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
    n_samples: int = 4,
    k: list[int] = [2, 3],
    seed: int = 42,
) -> AnnData:
    """AnnData fit through full pipeline including associate."""
    rng = np.random.default_rng(seed)

    profile_a = rng.normal(5.0, 1.0, size=(1, n_genes))
    profile_b = rng.normal(20.0, 1.0, size=(1, n_genes))

    blocks, labels, ages, studies = [], [], [], []

    for i in range(n_samples):
        profile = profile_a if i % 2 == 0 else profile_b
        noise = rng.normal(0.0, 0.5, size=(n_cells_per_sample, n_genes))
        X_sample = np.clip(profile + noise, 0, None)
        blocks.append(X_sample)
        labels.extend([f"sample_{i}"] * n_cells_per_sample)
        ages.extend([30 + i * 10] * n_cells_per_sample)
        studies.extend([f"study_{i % 2}"] * n_cells_per_sample)

    adata = AnnData(
        X=np.vstack(blocks).astype(np.float32),
        obs={
            "donor_id": labels,
            "age": np.array(ages, dtype=float),
            "study_id": studies,
        },
        var={"gene_name": np.arange(n_genes).astype(str)},
    )

    adata.obsm["X_umap"] = rng.normal(size=(len(labels), 2)).astype(np.float32)

    adata = jz.gp.nmf(adata, key="donor_id", k=k, min_cells=10, genes=None, seed=seed)
    jz.gp.similarity(adata, distance="jaccard", top_k=20)
    jz.gp.cluster(adata, threshold=0.3, min_cluster=1)
    jz.gp.score(adata, n_top_genes=10, seed=seed)
    jz.gp.aggregate(adata, key="donor_id", obs_cols=["age", "study_id"])
    jz.gp.associate(adata, formula="age + (1|study_id)")

    return adata


# juzi.pl.similarity


class TestPlSimilarity:
    def test_error_missing_juzi_similarity(self):
        adata = make_adata()
        del adata.uns["juzi_similarity"]
        with pytest.raises(KeyError):
            jz.pl.similarity(adata)

    def test_error_missing_juzi_similarity_idx(self):
        adata = make_adata()
        del adata.uns["juzi_similarity_idx"]
        with pytest.raises(KeyError):
            jz.pl.similarity(adata)

    def test_error_missing_juzi_names(self):
        adata = make_adata()
        del adata.uns["juzi_names"]
        with pytest.raises(KeyError):
            jz.pl.similarity(adata)

    def test_error_partial_axes_injection(self):
        adata = make_adata()
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            jz.pl.similarity(adata, ax_retention=ax)
        plt.close("all")

    def test_returns_tuple_of_axes(self):
        adata = make_adata()
        ax_ret, ax_hist = jz.pl.similarity(adata)
        assert isinstance(ax_ret, plt.Axes)
        assert isinstance(ax_hist, plt.Axes)
        plt.close("all")

    def test_accepts_axes(self):
        adata = make_adata()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        result = jz.pl.similarity(adata, ax_retention=ax1, ax_hist=ax2)
        assert result[0] is ax1
        assert result[1] is ax2
        plt.close("all")

    def test_custom_thresholds(self):
        adata = make_adata()
        ax_ret, ax_hist = jz.pl.similarity(adata, thresholds=np.linspace(0, 1, 20))
        assert isinstance(ax_ret, plt.Axes)
        plt.close("all")

    def test_custom_bins(self):
        adata = make_adata()
        ax_ret, ax_hist = jz.pl.similarity(adata, bins=20)
        assert isinstance(ax_hist, plt.Axes)
        plt.close("all")

    def test_custom_figsize(self):
        adata = make_adata()
        ax_ret, ax_hist = jz.pl.similarity(adata, figsize=(10, 4))
        assert isinstance(ax_ret, plt.Axes)
        plt.close("all")

    def test_custom_color(self):
        adata = make_adata()
        ax_ret, ax_hist = jz.pl.similarity(adata, color="#ff0000")
        assert isinstance(ax_ret, plt.Axes)
        plt.close("all")

    def test_show_gmm_true(self):
        adata = make_adata()
        ax_ret, ax_hist = jz.pl.similarity(adata, show_gmm=True)
        assert isinstance(ax_hist, plt.Axes)
        plt.close("all")

    def test_show_gmm_false(self):
        adata = make_adata()
        ax_ret, ax_hist = jz.pl.similarity(adata, show_gmm=False)
        assert isinstance(ax_hist, plt.Axes)
        plt.close("all")


# juzi.pl.heatmap


class TestPlHeatmap:
    def test_error_missing_cluster_similarity(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_similarity"]
        with pytest.raises(KeyError):
            jz.pl.heatmap(adata)

    def test_error_missing_cluster_labels(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.pl.heatmap(adata)

    def test_returns_axes(self):
        adata = make_adata()
        ax = jz.pl.heatmap(adata)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_accepts_ax(self):
        adata = make_adata()
        fig, ax = plt.subplots()
        result = jz.pl.heatmap(adata, ax=ax)
        assert result is ax
        plt.close("all")

    def test_custom_palette(self):
        adata = make_adata()
        labels = np.unique(adata.uns["juzi_cluster_labels"])
        palette = {int(c): "#ff0000" for c in labels}
        ax = jz.pl.heatmap(adata, palette=palette)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_vmin_vmax(self):
        adata = make_adata()
        ax = jz.pl.heatmap(adata, vmin=0.1, vmax=0.9)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_no_cluster_colors(self):
        adata = make_adata()
        ax = jz.pl.heatmap(adata, add_cluster_colors=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_no_cluster_labels(self):
        adata = make_adata()
        ax = jz.pl.heatmap(adata, add_cluster_labels=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


# juzi.pl.loadings


class TestPlLoadings:
    def test_error_missing_cluster_G(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_G"]
        with pytest.raises(KeyError):
            jz.pl.loadings(adata)

    def test_error_missing_cluster_labels(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.pl.loadings(adata)

    def test_error_missing_G_genes(self):
        adata = make_adata()
        del adata.uns["juzi_G_genes"]
        with pytest.raises(KeyError):
            jz.pl.loadings(adata)

    def test_error_n_top_genes_below_one(self):
        with pytest.raises(ValueError):
            jz.pl.loadings(make_adata(), n_top_genes=0)

    def test_error_n_top_genes_exceeds_genes(self):
        with pytest.raises(ValueError):
            jz.pl.loadings(make_adata(), n_top_genes=100_000)

    def test_returns_figure(self):
        adata = make_adata()
        fig = jz.pl.loadings(adata)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_n_top_genes(self):
        adata = make_adata()
        fig = jz.pl.loadings(adata, n_top_genes=5)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_show_values(self):
        adata = make_adata()
        fig = jz.pl.loadings(adata, show_values=True)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_palette(self):
        adata = make_adata()
        labels = np.unique(adata.uns["juzi_cluster_labels"])
        palette = {int(c): "#0000ff" for c in labels}
        fig = jz.pl.loadings(adata, palette=palette)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_ncols(self):
        adata = make_adata()
        fig = jz.pl.loadings(adata, ncols=2)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_single_program(self):
        adata = make_adata()
        adata.uns["juzi_cluster_labels"] = np.zeros(
            len(adata.uns["juzi_cluster_labels"]), dtype=int
        )
        adata.uns["juzi_cluster_G"] = adata.uns["juzi_cluster_G"][[0]]
        fig = jz.pl.loadings(adata)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


# juzi.pl.associate


class TestPlAssociate:
    def test_error_missing_association(self):
        adata = make_adata()
        del adata.uns["juzi_association"]
        with pytest.raises(KeyError):
            jz.pl.associate(adata)

    def test_error_missing_cluster_labels(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.pl.associate(adata)

    def test_error_invalid_padj_thresh(self):
        with pytest.raises(ValueError):
            jz.pl.associate(make_adata(), padj_thresh=1.5)

    def test_returns_axes(self):
        adata = make_adata()
        ax = jz.pl.associate(adata)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_accepts_ax(self):
        adata = make_adata()
        fig, ax = plt.subplots()
        result = jz.pl.associate(adata, ax=ax)
        assert result is ax
        plt.close("all")

    def test_no_threshold_line(self):
        adata = make_adata()
        ax = jz.pl.associate(adata, show_threshold=False)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_padj_thresh(self):
        adata = make_adata()
        ax = jz.pl.associate(adata, padj_thresh=0.01)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_palette(self):
        adata = make_adata()
        labels = np.unique(adata.uns["juzi_cluster_labels"])
        palette = {int(c): "#00ff00" for c in labels}
        ax = jz.pl.associate(adata, palette=palette)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_figsize(self):
        adata = make_adata()
        ax = jz.pl.associate(adata, figsize=(4, 6))
        assert isinstance(ax, plt.Axes)
        plt.close("all")


# juzi.pl.score


class TestPlScores:
    def test_error_missing_program_scores(self):
        adata = make_adata()
        del adata.obsm["juzi_program_scores"]
        with pytest.raises(KeyError):
            jz.pl.score(adata)

    def test_error_missing_basis(self):
        adata = make_adata()
        with pytest.raises(KeyError):
            jz.pl.score(adata, basis="X_nonexistent")

    def test_error_missing_cluster_labels(self):
        adata = make_adata()
        del adata.uns["juzi_cluster_labels"]
        with pytest.raises(KeyError):
            jz.pl.score(adata)

    def test_returns_figure(self):
        adata = make_adata()
        fig = jz.pl.score(adata)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_no_colorbar(self):
        adata = make_adata()
        fig = jz.pl.score(adata, show_colorbar=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_cmap(self):
        adata = make_adata()
        fig = jz.pl.score(adata, cmap="viridis")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_vmin_vmax(self):
        adata = make_adata()
        fig = jz.pl.score(adata, vmin=-2.0, vmax=2.0)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_ncols(self):
        adata = make_adata()
        fig = jz.pl.score(adata, ncols=2)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_basis(self):
        adata = make_adata()
        rng = np.random.default_rng(0)
        adata.obsm["X_tsne"] = rng.normal(size=(adata.n_obs, 2)).astype(np.float32)
        fig = jz.pl.score(adata, basis="X_tsne")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_not_rasterized(self):
        adata = make_adata()
        fig = jz.pl.score(adata, rasterized=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
