import matplotlib.pyplot as plt
import pytest
import juzi as jz


def test_pl_similarity_returns_axes(adata_similarity):
    ax1, ax2 = jz.pl.similarity(adata_similarity)
    assert isinstance(ax1, plt.Axes)
    assert isinstance(ax2, plt.Axes)
    plt.close("all")


def test_pl_programs_heatmap_returns_axes(adata_cluster_centroid):
    ax = jz.pl.programs_heatmap(adata_cluster_centroid)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_pl_programs_loadings_returns_figure(adata_cluster_progressive):
    fig = jz.pl.programs_loadings(adata_cluster_progressive)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_pl_programs_annotate_returns_axes(adata_annotated):
    ax = jz.pl.programs_annotate(adata_annotated, top_n=3)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_pl_programs_stability_returns_axes(adata_stable):
    ax = jz.pl.programs_stability(adata_stable)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_pl_programs_gene_overlap_returns_axes(adata_cluster_progressive):
    ax = jz.pl.programs_gene_overlap(adata_cluster_progressive)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_pl_programs_samples_returns_axes(adata_cluster_centroid):
    ax = jz.pl.programs_samples(adata_cluster_centroid)
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_pl_score_embedding_returns_figure(adata_scored):
    fig = jz.pl.score_embedding(adata_scored, basis="X_umap")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_pl_score_associate_returns_axes(adata_associated):
    ax = jz.pl.score_associate(adata_associated)
    assert isinstance(ax, plt.Axes)
    plt.close("all")
