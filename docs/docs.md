# docs

## Overview

#### Analysis (`juzi.gp`)

```
juzi.gp.nmf_fit               — per-sample NMF at multiple resolutions
juzi.gp.nmf_prune             — remove non-recurrent intra-sample factors

juzi.gp.similarity_compute    — compute inter-sample factor similarity
juzi.gp.similarity_filter     — filter factors by minimum similarity threshold

juzi.gp.programs_threshold    — select optimal clustering threshold
juzi.gp.programs_cluster      — cluster factors into consensus programs
juzi.gp.programs_jackknife    — assess program stability via sample jackknife
juzi.gp.programs_remove       — remove unstable or artefactual programs
juzi.gp.programs_merge        — manually merge related programs
juzi.gp.programs_annotate     — overlap programs with reference gene sets

juzi.gp.score_cells           — score cells on consensus programs
juzi.gp.score_classify        — classify cells into programs via permutation null
juzi.gp.score_aggregate       — pseudobulk sample-level program scores
juzi.gp.score_associate       — LMM association testing
```

#### Plotting (`juzi.pl`)

```
juzi.pl.similarity            — similarity threshold selection curve + distribution

juzi.pl.programs_threshold    — clustering threshold sweep curve
juzi.pl.programs_heatmap      — factor similarity matrix with program annotations
juzi.pl.programs_loadings     — top gene loadings per program
juzi.pl.programs_jackknife    — jackknife stability heatmap
juzi.pl.programs_annotate     — gene set annotation dot plot

juzi.pl.score_embedding       — cells on embedding coloured by program scores
juzi.pl.score_associate       — coefficient plot of association results
```

#### Utilities (`juzi.ut`)

```
juzi.ut.programs_genes    — top genes per program as a dict
juzi.ut.programs_compare  — Jaccard similarity between programs across two datasets
juzi.ut.programs_donors   — per-sample factor contribution per program
juzi.ut.factors_loadings  — gene × factor loading matrix as a labelled DataFrame
juzi.ut.factors_scores    — cell × factor score matrix as a labelled DataFrame
```

#### Gene sets (`juzi.mg`)

```
juzi.mg.CellCycle       — cell cycle markers (Tirosh et al. 2016)
juzi.mg.CancerPathways  — canonical cancer pathways (Sanchez-Vega et al. 2018)
juzi.mg.CancerBreast    — breast cancer subtype markers (Parker et al. 2009)
juzi.mg.Hallmark3CA     — 3CA meta-programs (Gavish et al. 2023)
juzi.mg.read_msigdb     — load any MSigDB .txt file
```

---

## Quick start

```python
import scanpy as sc
import juzi as jz

adata = sc.read_h5ad("data.h5ad")

# adata.obs must contain a grouping column
# adata.X should contain raw counts
# The grouping column can be donors, samples, regions, time points, etc.
print(adata.obs["sample_id"].value_counts())
```

## A note on ranking genes

`juzi` ranks genes using raw loadings by default. However, genes can alternatively be ranked using a weighted log-ratio score that rewards both high
absolute loading and program exclusivity:

```
G = raw loadings matrix (n_programs × n_genes)
score = G * log(G / mean(G across programs) + e)
score = max(score, 0)   # anti-markers clipped to zero
```

The log term is positive when a gene loads more in this program than its
average across all programs (it measures exclusivity). The G weight ensures
near-zero genes contribute nothing regardless of their log-ratio. Together
they select genes that are both highly expressed in the program and not
shared with other programs.

Set `use_combined=True` in any of `nmf_prune`, `similarity_compute`, `score_cells`,
`programs_annotate`, `programs_genes`, `programs_compare`, and
`programs_loadings` to rank genes by the weighted log-ratio score.

---

## Step 1 — NMF

Fit NMF independently on each sample at multiple resolutions. juzi normalises and log-transforms counts internally. Always returns a new AnnData — input is never modified. All three keep masks are initialised to all True.

```python
adata = jz.gp.nmf_fit(
    adata,
    key="sample_id",          # any .obs column — donors, regions, time points, etc.
    k=[7, 8, 9, 10],          # factorisation ranks — use multiple for stability
    min_cells=10,             # minimum cells per sample to include
    genes="highly_variable",  # bool .var column, list of genes, or None
    genes_force=target_genes, # always include regardless of genes filter
    target_sum=1e4,
    seed=42,
)
```

| Field | Location | Description |
|---|---|---|
| `juzi_G` | `.varm` | Gene × factor loading matrix |
| `juzi_k` | `.uns` | List of k values used |
| `juzi_names` | `.uns` | Sample identity per factor |
| `juzi_G_genes` | `.uns` | Gene names corresponding to `juzi_G` rows |
| `juzi_keep_prune` | `.uns` | Boolean mask — all True |
| `juzi_keep_similarity` | `.uns` | Boolean mask — all True |
| `juzi_keep_cluster` | `.uns` | Boolean mask — all True |
| `juzi_keep` | `.uns` | Intersection of all three masks |

---

## Step 2 — Prune

Remove non-recurrent and redundant factors within each sample. Two sequential
filters are applied:

**Recurrence filter** — a factor is kept if it shares sufficient top-gene
overlap (Jaccard >= `min_similarity`) with at least one factor from each of
`min_k` other resolutions. Non-recurrent factors are masked before
cross-sample similarity is computed.

**Deduplication filter** — among factors that passed recurrence, any two
factors from the same sample with Jaccard >= `min_similarity` are considered
redundant. The most central factor per redundant group (highest mean Jaccard
to all other members) is retained.

```python
jz.gp.nmf_prune(
    adata,
    top_k=50,
    min_similarity=0.1,
    min_k=1,
    matching="hungarian",  # or, "greedy"
    deduplicate=True,      # remove within-sample redundant factors
    use_combined=True,
)
```

| Field | Location | Description |
|---|---|---|
| `juzi_keep_prune` | `.uns` | Boolean mask of recurrent non-redundant factors |
| `juzi_keep` | `.uns` | Recomputed intersection |

---

## Step 3 — Similarity

Compute pairwise Jaccard similarity between kept factors across all samples. Only factors where `juzi_keep` is True enter the computation. The resulting matrix is `(n_kept × n_kept)`. Gene ranking for top-k selection uses the combined score by default.

```python
jz.gp.similarity_compute(
    adata,
    distance="jaccard",
    top_k=50,
    intra_sample=False,    # recommended — exclude within-sample pairs
    drop_zeros=True,
    use_combined=True,
)
```

| Field | Location | Description |
|---|---|---|
| `juzi_similarity` | `.uns` | `(n_kept × n_kept)` similarity matrix |
| `juzi_similarity_idx` | `.uns` | Global factor indices of matrix rows/cols |
| `juzi_keep_similarity` | `.uns` | Boolean mask length `n_total` |
| `juzi_keep` | `.uns` | Recomputed intersection |

---

## Step 3b — Select similarity threshold

Inspect the retention curve and max-similarity distribution per factor. A two-component GMM is fitted to the distribution — the crossover between noise and signal components provides a principled threshold suggestion.

```python
# Two-panel plot: retention curve + max similarity distribution with GMM
ax_ret, ax_hist = jz.pl.similarity(adata)

# Apply threshold
jz.gp.similarity_filter(adata, min_similarity=0.2)
```

| Field | Location | Description |
|---|---|---|
| `juzi_keep_similarity` | `.uns` | Updated boolean mask |
| `juzi_keep` | `.uns` | Recomputed intersection |

---

## Step 4 — Select clustering threshold

Sweep threshold values, fit clustering at each, and find the threshold that maximises inner/outer cluster similarity contrast. All local maxima are stored — each represents a valid partition at a different resolution.

```python
optimal = jz.gp.programs_threshold(
    adata,
    min_cluster=3,
    metric="ratio",        # "ratio", "delta", or "silhouette"
    method="average",      # "average", "complete", or "ward"
)

# Global optimum marked solid, local maxima dashed
ax = jz.pl.programs_threshold(adata)

# Inspect all local maxima
sweep = adata.uns["juzi_threshold_sweep"]
for t, v in zip(sweep["local_maxima"], sweep["local_maxima_values"]):
    print(f"threshold = {t:.3f}  metric = {v:.3f}")
```

| Field | Location | Description |
|---|---|---|
| `juzi_threshold_sweep` | `.uns` | Thresholds, metric values, local maxima, optimal |

---

## Step 5 — Cluster

Cluster the factor similarity matrix into consensus programs using iterative hierarchical merging. Re-running resets only `juzi_keep_cluster` — upstream masks are never modified.

```python
jz.gp.programs_cluster(
    adata,
    threshold=optimal,
    min_cluster=3,         # minimum unique samples per program
    method="average",
    reorder=True,
)
```

| Field | Location | Description |
|---|---|---|
| `juzi_keep_cluster` | `.uns` | Boolean mask length `n_total` |
| `juzi_keep` | `.uns` | Recomputed intersection |
| `juzi_cluster_similarity` | `.uns` | Reordered factor similarity matrix |
| `juzi_cluster_labels` | `.uns` | Cluster label per retained factor |
| `juzi_cluster_names` | `.uns` | Sample name per retained factor, aligned to labels |
| `juzi_cluster_G` | `.uns` | Centroid gene loading per program |
| `juzi_cluster_samples` | `.uns` | Unique contributing samples per program |
| `juzi_cluster_stats` | `.uns` | Silhouette, inner/outer similarity |

---

## Step 5b — Inspect programs

Before refinement, inspect the clustering result and understand program composition.

```python
# Factor similarity matrix with program annotations
ax  = jz.pl.programs_heatmap(adata)

# Top gene loadings per program
fig = jz.pl.programs_loadings(adata, n_top_genes=15)

# Top genes per program as a dict — does not require score_cells
genes = jz.ut.programs_genes(adata, n_top_genes=20)

# Per-sample factor contribution — reveals sample dominance
df = jz.ut.programs_donors(adata)
```

---

## Step 5c — Jackknife stability

Assess whether each program is a genuine consensus signal or driven by a subset of samples. For each held-out sample, re-runs clustering and measures how well each reference program's gene signature reappears.

```python
jz.gp.programs_jackknife(
    adata,
    n_top_genes=50,
    use_combined=True,
    n_jobs=4,
)

ax = jz.pl.programs_jackknife(adata)

# Per-program mean stability and per-sample breakdown
jack = adata.uns["juzi_jackknife"]
print(jack["stability"])           # (K,) mean Jaccard per program
print(jack["stability_matrix"])    # (K × N) per-sample scores
```

| Field | Location | Description |
|---|---|---|
| `juzi_jackknife` | `.uns` | Dict with stability, stability_matrix, samples, params |

---

## Step 5d — Refine programs (optional)

Remove unstable or artefactual programs. Unlike `programs_merge`, `programs_remove` updates `juzi_keep_cluster` — factors are permanently excluded. `juzi_jackknife` is dropped and must be rerun.

```python
# Remove unstable programs
jz.gp.programs_remove(adata, clusters=[3, 5])

# Merge biologically related programs
jz.gp.programs_merge(adata, clusters=[0, 2])
jz.gp.programs_merge(adata, clusters=[[0, 2], [1, 4]])  # multiple merges

# Rerun jackknife on final programs
jz.gp.programs_jackknife(adata, n_top_genes=50)
```

---

## Step 5e — Annotate programs (optional)

Score programs against reference gene sets via Jaccard similarity and hypergeometric test.

```python
# Built-in gene sets
jz.gp.programs_annotate(adata, gene_sets=jz.mg.Hallmark3CA().as_dict())

# MSigDB
gene_sets = jz.mg.read_msigdb("msigdb_human.txt", collections=["C4:3CA"])
jz.gp.programs_annotate(adata, gene_sets=gene_sets, n_top_genes=50)

ax = jz.pl.programs_annotate(adata, top_n=10, padj_thresh=0.05)
```

| Field | Location | Description |
|---|---|---|
| `juzi_annotation` | `.uns` | Tidy DataFrame with Jaccard, pval, padj, overlap_genes |

---

## Step 6 — Score cells

Score each cell on each consensus program using control-subtracted mean expression (Tirosh et al. 2016). Gene selection uses the combined score by default.

```python
jz.gp.score_cells(
    adata,
    n_top_genes=50,
    use_combined=True,
    n_control_genes=50,
    seed=42,
)

fig = jz.pl.score_embedding(adata, basis="X_umap")
```

| Field | Location | Description |
|---|---|---|
| `juzi_program_scores` | `.obsm` | Cell × program score matrix |
| `juzi_program_genes` | `.uns` | Top genes used per program |

---

## Step 6b — Classify cells (optional)

Classify cells into programs using a permutation null distribution. Shuffles expression values per gene across subsampled cells to estimate the null, then assigns each cell via z-test with BH correction.

```python
jz.gp.score_classify(
    adata,
    n_shuffles=20,
    n_cells_per_shuffle=5000,
    padj_thresh=0.05,
    n_jobs=4,
    seed=42,
)
```

| Field | Location | Description |
|---|---|---|
| `juzi_program_pvals` | `.obsm` | Raw p-values (n_cells × n_programs) |
| `juzi_program_padj` | `.obsm` | BH-adjusted p-values |
| `juzi_program_label` | `.obs` | Per-cell program assignment or "unresolved" |
| `juzi_classify_params` | `.uns` | Null distribution parameters |

---

## Step 7 — Aggregate

Aggregate per-cell scores to per-sample pseudobulk scores. Sample-level covariates are propagated for direct use in `score_associate`.

```python
jz.gp.score_aggregate(
    adata,
    key="sample_id",
    obs_cols=["age", "study_id", "condition"],
    agg="mean",
    min_cells=10,
)
```

| Field | Location | Description |
|---|---|---|
| `juzi_aggregate_scores` | `.uns` | DataFrame (n_samples × n_programs + covariates) |

---

## Step 8 — Associate

Test whether program activity associates with a covariate using a linear mixed model. Supports R-style formula notation with random effects.

```python
jz.gp.score_associate(
    adata,
    formula="age + condition + (1|study_id)",
    reml=True,
)

# Multiple random effects → combined interaction grouping variable
jz.gp.score_associate(adata, formula="age + (1|study_id) + (1|batch)")

ax = jz.pl.score_associate(adata, padj_thresh=0.05)
```

| Field | Location | Description |
|---|---|---|
| `juzi_association` | `.uns` | DataFrame with beta, se, pval, padj per program |

---

## Full pipeline example

```python
import scanpy as sc
import juzi as jz

adata = sc.read_h5ad("data.h5ad")
sc.pp.highly_variable_genes(adata, n_top_genes=3000)

# NMF

adata = jz.gp.nmf_fit(
    adata,
    key="sample_id",           # donors, regions, time points, etc.
    k=[7, 8, 9, 10],
    min_cells=10,
    genes="highly_variable",
    genes_force=target_genes,
    seed=42,
)
jz.gp.nmf_prune(adata, top_k=50, min_similarity=0.1, min_k=1)

# Similarity

jz.gp.similarity_compute(adata, distance="jaccard", top_k=50, intra_sample=False)
ax_ret, ax_hist = jz.pl.similarity(adata)
jz.gp.similarity_filter(adata, min_similarity=0.2)

# Programs

optimal = jz.gp.programs_threshold(adata, min_cluster=3, metric="ratio")
ax      = jz.pl.programs_threshold(adata)

jz.gp.programs_cluster(adata, threshold=optimal, min_cluster=3)

ax  = jz.pl.programs_heatmap(adata)
fig = jz.pl.programs_loadings(adata, n_top_genes=15)

genes = jz.ut.programs_genes(adata, n_top_genes=20)
df    = jz.ut.programs_donors(adata)

# Stability assessment
jz.gp.programs_jackknife(adata, n_top_genes=50, n_jobs=4)
ax = jz.pl.programs_jackknife(adata)

# Refinement
jz.gp.programs_remove(adata, clusters=[3])
jz.gp.programs_merge(adata, clusters=[0, 2])
jz.gp.programs_jackknife(adata, n_top_genes=50)   # rerun on final programs

# Annotation
jz.gp.programs_annotate(adata, gene_sets=jz.mg.Hallmark3CA().as_dict())
ax = jz.pl.programs_annotate(adata, top_n=10, padj_thresh=0.05)

# Score

jz.gp.score_cells(adata, n_top_genes=50, seed=42)
fig = jz.pl.score_embedding(adata, basis="X_umap")

jz.gp.score_classify(adata, n_shuffles=20, n_cells_per_shuffle=5000, n_jobs=4)

jz.gp.score_aggregate(adata, key="sample_id", obs_cols=["age", "study_id"])
jz.gp.score_associate(adata, formula="age + (1|study_id)")
ax = jz.pl.score_associate(adata, padj_thresh=0.05)

# Utilities

df_loadings = jz.ut.factors_loadings(adata, kept_only=True)
df_scores   = jz.ut.factors_scores(adata, kept_only=True)
df_compare  = jz.ut.programs_compare(adata_a, adata_b, n_top_genes=50)
```

---

## Alternative use cases

**Spatial transcriptomics (programs recurring across slides or regions)**

```python
# key = slide ID or spatial region annotation
adata = jz.gp.nmf_fit(adata, key="slide_id", k=[5, 6, 7], min_cells=50)
```

**Tumour biopsies (programs recurring across regions from the same patient)**

```python
adata = jz.gp.nmf_fit(adata, key="biopsy_id", k=[5, 6, 7, 8], min_cells=20)
```

**Time course (programs recurring across time points)**

```python
adata = jz.gp.nmf_fit(adata, key="time_point", k=[4, 5, 6], min_cells=100)
```

**Cell type subsets (T cell states recurring across donors)**

```python
t_cells = adata[adata.obs["cell_type"] == "T cell"].copy()
t_cells = jz.gp.nmf_fit(t_cells, key="donor_id", k=[5, 6, 7], min_cells=10)
```

**Cross-dataset comparison (programs shared across cohorts)**

```python
programs_a = jz.ut.programs_genes(adata_cohort_a, n_top_genes=50)
programs_b = jz.ut.programs_genes(adata_cohort_b, n_top_genes=50)
df         = jz.ut.programs_compare(adata_cohort_a, adata_cohort_b)
```

---

## Parameters reference

### `jz.gp.nmf_fit`

| Parameter | Default | Description |
|---|---|---|
| `key` | required | Column in `.obs` for sample grouping — donors, regions, time points, etc. |
| `k` | `[7,8,9,10]` | Factorisation ranks |
| `min_cells` | `5` | Minimum cells per sample |
| `genes` | `"highly_variable"` | Gene selection — bool `.var` column, list, or `None` |
| `genes_force` | `None` | Genes to always include regardless of `genes` filter |
| `gene_names_col` | `None` | `.var` column for gene names |
| `target_sum` | `1e4` | Library size normalisation target |
| `normalize` | `True` | Normalise to target_sum |
| `log1p` | `True` | Apply log1p transformation |
| `solver` | `"cd"` | NMF solver (`"cd"` or `"mu"`) |
| `loss` | `"frobenius"` | Beta divergence |
| `init` | `"nndsvda"` | NMF initialisation |
| `seed` | `123` | Random seed |
| `n_jobs` | `1` | Parallel workers |

### `jz.gp.nmf_prune`

| Parameter | Default | Description |
|---|---|---|
| `top_k` | `50` | Top genes for Jaccard similarity |
| `min_similarity`  | `0.1` | Minimum Jaccard for recurrence detection across resolutions |
| `dedup_similarity`| `0.5` | Minimum Jaccard for within-sample deduplication — should be > min_similarity |
| `deduplicate`     | `True`| Apply within-sample deduplication after recurrence filter |
| `min_k` | `1` | Minimum other resolutions a factor must match |
| `matching` | `"greedy"` | `"greedy"` or `"hungarian"` |
| `use_combined` | `True` | Rank genes by combined loading × specificity score |
| `n_jobs` | `1` | Parallel workers |

### `jz.gp.similarity_compute`

| Parameter | Default | Description |
|---|---|---|
| `distance` | `"jaccard"` | `"jaccard"` or custom callable |
| `top_k` | `50` | Top genes per factor |
| `intra_sample` | `True` | Include within-sample pairs |
| `drop_zeros` | `True` | Flag all-zero similarity rows |
| `use_combined` | `True` | Rank genes by combined loading × specificity score |
| `n_jobs` | `1` | Parallel workers |

### `jz.gp.similarity_filter`

| Parameter | Default | Description |
|---|---|---|
| `min_similarity` | required | Minimum similarity threshold in [0, 1] |

### `jz.gp.programs_threshold`

| Parameter | Default | Description |
|---|---|---|
| `thresholds` | `linspace(0, 1, 50)` | Grid of threshold values to evaluate |
| `min_cluster` | `2` | Minimum unique samples per program |
| `method` | `"average"` | Linkage method — `"average"`, `"complete"`, or `"ward"` |
| `metric` | `"ratio"` | `"ratio"`, `"delta"`, or `"silhouette"` |

### `jz.gp.programs_cluster`

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.1` | Merge until max inter-cluster similarity < threshold |
| `min_cluster` | `2` | Minimum unique samples per program |
| `method` | `"average"` | Linkage method |
| `reorder` | `True` | Sort programs by size |

### `jz.gp.programs_jackknife`

| Parameter | Default | Description |
|---|---|---|
| `n_top_genes` | `50` | Top genes per program for Jaccard computation |
| `use_combined` | `True` | Rank genes by combined loading × specificity score |
| `n_jobs` | `1` | Parallel workers (across samples) |

### `jz.gp.programs_remove`

| Parameter | Default | Description |
|---|---|---|
| `clusters` | required | List of cluster label integers to remove |

### `jz.gp.programs_merge`

| Parameter | Default | Description |
|---|---|---|
| `clusters` | required | `[0, 2]` to merge C0 and C2, or `[[0,2],[1,3]]` for multiple merges |

### `jz.gp.programs_annotate`

| Parameter | Default | Description |
|---|---|---|
| `gene_sets` | required | Dict or `juzi.mg` object with `as_dict()` method |
| `n_top_genes` | `50` | Top genes per program |
| `use_combined` | `True` | Rank genes by combined loading × specificity score |
| `padj_method` | `"fdr_bh"` | Multiple testing correction |

### `jz.gp.score_cells`

| Parameter | Default | Description |
|---|---|---|
| `n_top_genes` | `50` | Top genes per program |
| `use_combined` | `True` | Rank genes by combined loading × specificity score |
| `n_control_genes` | `50` | Control genes per program gene |
| `gene_names_col` | `None` | `.var` column for gene names |
| `layer` | `None` | Layer for expression values |
| `seed` | `123` | Random seed |

### `jz.gp.score_classify`

| Parameter | Default | Description |
|---|---|---|
| `n_shuffles` | `20` | Number of shuffled expression matrices |
| `n_cells_per_shuffle` | `5000` | Cells subsampled per shuffle |
| `padj_thresh` | `0.05` | BH-adjusted p-value threshold for assignment |
| `n_jobs` | `1` | Parallel workers (across shuffles) |
| `seed` | `123` | Random seed |

### `jz.gp.score_aggregate`

| Parameter | Default | Description |
|---|---|---|
| `key` | required | Column in `.obs` for sample grouping |
| `obs_cols` | `None` | Sample-level covariates to propagate |
| `agg` | `"mean"` | `"mean"` or `"median"` |
| `min_cells` | `10` | Minimum cells per sample |

### `jz.gp.score_associate`

| Parameter | Default | Description |
|---|---|---|
| `formula` | required | R-style formula e.g. `"age + (1\|study_id)"` |
| `reml` | `True` | Use restricted maximum likelihood |
| `padj_method` | `"fdr_bh"` | Multiple testing correction |

### `jz.ut.programs_genes`

| Parameter | Default | Description |
|---|---|---|
| `n_top_genes` | `50` | Top genes per program |
| `use_combined` | `True` | Rank by combined loading × specificity score |

### `jz.ut.programs_compare`

| Parameter | Default | Description |
|---|---|---|
| `n_top_genes` | `50` | Top genes per program |
| `use_combined` | `True` | Rank by combined loading × specificity score |

### `jz.ut.programs_donors`

No parameters beyond `adata`.

### `jz.ut.factors_loadings`

| Parameter | Default | Description |
|---|---|---|
| `kept_only` | `True` | Return only factors where `juzi_keep` is True |

### `jz.ut.factors_scores`

| Parameter | Default | Description |
|---|---|---|
| `kept_only` | `True` | Return only factors where `juzi_keep` is True |

### `jz.pl.similarity`

| Parameter | Default | Description |
|---|---|---|
| `thresholds` | `linspace(0, 1, 100)` | Thresholds for retention curve |
| `bins` | `50` | Histogram bins for max similarity distribution |
| `show_gmm` | `True` | Fit and overlay two-component GMM |
| `figsize` | `(7, 3)` | Figure size |

### `jz.pl.programs_threshold`

| Parameter | Default | Description |
|---|---|---|
| `show_optimal` | `True` | Mark global optimum with solid line |
| `show_local_maxima` | `True` | Mark local maxima with dashed lines |
| `figsize` | `(4, 3)` | Figure size |

### `jz.pl.programs_loadings`

| Parameter | Default | Description |
|---|---|---|
| `n_top_genes` | `10` | Genes to display per program |
| `use_combined` | `True` | Rank and display by combined loading × specificity score |
| `ncols` | `4` | Columns in figure grid |
| `show_values` | `False` | Annotate bars with score values |

### `jz.pl.programs_jackknife`

| Parameter | Default | Description |
|---|---|---|
| `cmap` | `"Reds_r"` | Colormap — low stability appears dark |
| `show_donors` | `True` | Show sample names on x-axis |
| `figsize` | `None` | Inferred from K and N if None |

---

## Gene sets

```python
cc = jz.mg.CellCycle()
cc.info()
cc.as_dict()

cp = jz.mg.CancerPathways()
cb = jz.mg.CancerBreast()
mg = jz.mg.Hallmark3CA()

# Load your own MSigDB collection
gene_sets = jz.mg.read_msigdb("msigdb_human.txt", collections=["H"])
```

---

## License

BSD 3-Clause License. Copyright (c) 2025, Tom Ouellette.
