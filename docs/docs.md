# docs

## Overview

The juzi pipeline consists of eight analytical steps, five plotting functions, and a set of utility functions:

#### Analysis (`juzi.gp`)

```
juzi.gp.nmf               — per-donor NMF at multiple resolutions
juzi.gp.prune             — remove non-recurrent intra-donor factors
juzi.gp.similarity        — compute inter-donor factor similarity
juzi.gp.select_similarity — filter factors by minimum similarity threshold
juzi.gp.cluster           — cluster factors into consensus programs
juzi.gp.score             — score cells on consensus programs
juzi.gp.aggregate         — pseudobulk donor-level program scores
juzi.gp.associate         — LMM association testing
juzi.gp.annotate          — overlap with reference gene sets
```

#### Plotting (`juzi.pl`)

```
juzi.pl.similarity   — min_similarity vs factors retained curve
juzi.pl.heatmap      — factor similarity matrix heatmap with cluster annotations
juzi.pl.loadings     — top gene loadings per program
juzi.pl.associate    — coefficient plot of association results
juzi.pl.scores       — embedding coloured by program scores
juzi.pl.annotate     — dot plot of gene set overlaps
```

#### Utilities (`juzi.ut`)

```
juzi.ut.program_genes   — extract top genes per program as a dict
juzi.ut.program_compare — compare programs across two datasets via Jaccard
juzi.ut.program_donors  — per-donor factor contribution per program
```

#### Gene sets (`juzi.mg`)

```
juzi.mg.CellCycle       — cell cycle markers (Tirosh et al. 2016)
juzi.mg.CancerPathways  — canonical cancer pathways (Sanchez-Vega et al. 2018)
juzi.mg.CancerBreast    — breast cancer subtype markers (Parker et al. 2009)
juzi.mg.MsigDB3CA       — 3CA meta-programs (Gavish et al. 2023)
juzi.mg.read_msigdb     — load any MSigDB .txt file
```

---

## Quick start

```python
import scanpy as sc
import juzi as jz

# Load your AnnData object
adata = sc.read_h5ad("mammary_gland.h5ad")

# adata.obs must contain a donor identity column
# adata.X should contain raw counts
print(adata.obs["donor_id"].value_counts())
```

---

## Step 1 — NMF

Fit NMF independently on each donor at multiple resolutions. juzi normalises and log-transforms counts internally.

```python
adata = jz.gp.nmf(
    adata,
    key="donor_id",           # column in .obs denoting donor identity
    k=[7, 8, 9, 10],          # factorisation ranks — use multiple for stability
    min_cells=10,             # minimum cells per donor to include
    genes="highly_variable",  # boolean .var column, list of genes, or None
    genes_force=["PTHLH", "AREG"],  # always include regardless of genes filter
    target_sum=1e4,           # library size normalisation target
    seed=42,
)
```

> **Note:** `nmf` always returns a new AnnData object subset to the selected genes and donors that passed `min_cells` filtering. The input object is never modified in place.

> **Note:** All three stage keep masks (`juzi_keep_prune`, `juzi_keep_similarity`, `juzi_keep_cluster`) are initialised to all True after `nmf`. `juzi_keep` is the intersection of all three and is recomputed by each downstream function.

Key outputs stored in the returned AnnData:

| Field | Location | Description |
|---|---|---|
| `juzi_G` | `.varm` | Gene × factor loading matrix |
| `juzi_k` | `.uns` | List of k values used |
| `juzi_names` | `.uns` | Donor identity per factor column |
| `juzi_G_genes` | `.uns` | Gene names corresponding to `juzi_G` rows |
| `juzi_keep_prune` | `.uns` | Boolean mask — all True, updated by `prune` |
| `juzi_keep_similarity` | `.uns` | Boolean mask — all True, updated by `similarity` |
| `juzi_keep_cluster` | `.uns` | Boolean mask — all True, updated by `cluster` |
| `juzi_keep` | `.uns` | Intersection of all three stage masks |

---

## Step 2 — Prune

Remove non-recurrent factors within each donor. A factor is considered recurrent if it shares sufficient top-gene overlap with at least one factor from each of `min_k` other resolutions. Updates `juzi_keep_prune` and recomputes `juzi_keep`.

```python
jz.gp.prune(
    adata,
    top_k=50,            # top genes used to compute Jaccard similarity
    min_similarity=0.1,  # minimum Jaccard for two factors to match
    min_k=1,             # minimum other resolutions a factor must match
    matching="greedy",   # "greedy" or "hungarian"
)
```

Outputs:

| Field | Location | Description |
|---|---|---|
| `juzi_keep_prune` | `.uns` | Boolean mask of recurrent factors |
| `juzi_keep` | `.uns` | Recomputed intersection of all three stage masks |

---

## Step 3 — Similarity

Compute pairwise Jaccard similarity between all factors across all donors. Factors whose entire similarity row is zero are flagged via `drop_zeros`. Updates `juzi_keep_similarity` and recomputes `juzi_keep`.

```python
jz.gp.similarity(
    adata,
    distance="jaccard",  # "jaccard" or a custom callable
    top_k=50,            # top genes used per factor
    intra_sample=False,  # whether to compute within-donor similarities
    drop_zeros=True,     # flag factors with all-zero similarity rows
)
```

Outputs:

| Field | Location | Description |
|---|---|---|
| `juzi_similarity` | `.uns` | Factor × factor similarity matrix |
| `juzi_keep_similarity` | `.uns` | Boolean mask updated by drop_zeros |
| `juzi_keep` | `.uns` | Recomputed intersection of all three stage masks |

---

## Step 3b — Select similarity threshold

Inspect the retention curve and apply a `min_similarity` threshold. Can be re-run with different thresholds without re-running `similarity`. The `drop_zeros` mask from `similarity` is preserved across re-runs.

```python
# Inspect the curve first
ax = jz.pl.similarity(adata)

# Apply a threshold
jz.gp.select_similarity(adata, min_similarity=0.2)
```

Outputs:

| Field | Location | Description |
|---|---|---|
| `juzi_keep_similarity` | `.uns` | Updated boolean mask |
| `juzi_keep` | `.uns` | Recomputed intersection of all three stage masks |

---

## Step 4 — Cluster

Cluster the factor similarity matrix into consensus programs using iterative average-linkage merging. Factors removed by `min_cluster` are tracked in `juzi_keep_cluster`. Can be re-run with different parameters without affecting upstream masks — only `juzi_keep_cluster` is reset each run.

```python
jz.gp.cluster(
    adata,
    threshold=0.1,   # merge until max inter-cluster similarity < threshold
    min_cluster=2,   # minimum unique donors per cluster
    reorder=True,    # sort clusters by size, largest first
)
```

Outputs:

| Field | Location | Description |
|---|---|---|
| `juzi_keep_cluster` | `.uns` | Boolean mask of factors retained after min_cluster filtering |
| `juzi_keep` | `.uns` | Recomputed intersection of all three stage masks |
| `juzi_cluster_similarity` | `.uns` | Reordered factor similarity matrix |
| `juzi_cluster_labels` | `.uns` | Cluster label per retained factor |
| `juzi_cluster_names` | `.uns` | Donor name per retained factor, aligned to labels |
| `juzi_cluster_G` | `.uns` | Centroid gene loading per program |
| `juzi_cluster_samples` | `.uns` | Unique contributing donors per program |
| `juzi_cluster_stats` | `.uns` | Silhouette, inner/outer similarity |

---

## Step 5 — Score

Score each cell on each consensus program using the Tirosh et al. 2016 control-subtracted mean expression approach.

```python
jz.gp.score(
    adata,
    n_top_genes=50,       # top genes per program for scoring
    use_specificity=True, # rank by specificity rather than raw loading
    n_control_genes=50,   # control genes per program gene
    seed=42,
)
```

Outputs:

| Field | Location | Description |
|---|---|---|
| `juzi_program_scores` | `.obsm` | Cell × program score matrix |
| `juzi_program_genes` | `.uns` | Top genes used per program |

---

## Step 6 — Aggregate

Aggregate per-cell program scores to per-donor pseudobulk scores. Donor-level covariates are propagated alongside program scores for direct use in `associate`.

```python
jz.gp.aggregate(
    adata,
    key="donor_id",
    obs_cols=["age", "study_id", "donor_brca"],  # donor-level covariates
    agg="mean",   # "mean" or "median"
    min_cells=10, # minimum cells per donor to include
)
```

Outputs:

| Field | Location | Description |
|---|---|---|
| `juzi_aggregate_scores` | `.uns` | DataFrame (n_donors × n_programs + covariates) |

---

## Step 7 — Associate

Test whether consensus program activity associates with a covariate of interest using a linear mixed model. Supports R-style formula notation with random effects. Multiple random effects are combined into a single interaction grouping variable.

```python
jz.gp.associate(
    adata,
    formula="age + donor_brca + (1|study_id)",
    reml=True,  # use restricted maximum likelihood
)

# Multiple random effects: (1|study_id) + (1|batch) → groups = study_id_x_batch
jz.gp.associate(
    adata,
    formula="age + (1|study_id) + (1|batch)",
)
```

Outputs:

| Field | Location | Description |
|---|---|---|
| `juzi_association` | `.uns` | DataFrame with beta, se, pval, padj per program |

---

## Step 8 — Annotate

Score consensus programs against reference gene sets to aid biological interpretation. Computes Jaccard similarity and hypergeometric p-values for each program × gene set pair.

```python
# Using a built-in juzi.mg gene set
gene_sets = jz.mg.CellCycle().as_dict()

# Or from MSigDB
gene_sets = jz.mg.read_msigdb("msigdb_human.txt", collections=["C4:3CA"])

# Or a custom dict
gene_sets = {"MY_PROGRAM": ["GENE1", "GENE2", "GENE3"]}

jz.gp.annotate(
    adata,
    gene_sets=gene_sets,
    n_top_genes=50,
    use_specificity=True,
)
```

Outputs:

| Field | Location | Description |
|---|---|---|
| `juzi_annotation` | `.uns` | Tidy DataFrame with Jaccard, hypergeometric pval, padj, overlap_genes |

---

## Plotting

### Similarity threshold selection curve

Inspect how many factors are retained at each `min_similarity` threshold before calling `jz.gp.select_similarity`.

```python
ax = jz.pl.similarity(adata)
```

### Factor similarity matrix heatmap

```python
ax = jz.pl.heatmap(adata)
```

### Top gene loadings per program

```python
fig = jz.pl.loadings(adata, n_top_genes=15)
```

### Association results

```python
ax = jz.pl.associate(adata, padj_thresh=0.05)
```

### Per-cell program scores on embedding

```python
sc.pp.neighbors(adata)
sc.tl.umap(adata)
fig = jz.pl.scores(adata, basis="X_umap")
```

### Gene set annotation dot plot

```python
ax = jz.pl.annotate(adata, top_n=10, padj_thresh=0.05)
```

---

## Utilities

### Extract top genes per program

Returns a `{"C0": ["GENE1", ...], "C1": [...], ...}` dict without requiring `score` to be run first — useful for rapid biological inspection immediately after clustering.

```python
genes = jz.ut.program_genes(adata, n_top_genes=50, use_specificity=True)
```

### Compare programs across two datasets

Returns a `(n_programs_a × n_programs_b)` Jaccard similarity DataFrame. Useful for identifying shared programs across lineages or conditions.

```python
df = jz.ut.program_compare(adata_lhs, adata_lasp, n_top_genes=50)
```

### Per-donor factor contribution

Returns a `(n_donors × n_programs)` DataFrame of factor counts. Reveals whether any single donor dominates a program, which may indicate a donor-specific rather than consensus signal.

```python
df = jz.ut.program_donors(adata)
```

---

## Gene sets

juzi bundles several gene set collections accessible via `juzi.mg`:

```python
# List available collections
jz.mg.available_sets()

# Cell cycle (Tirosh et al. 2016)
cc = jz.mg.CellCycle()
cc.info()     # list gene set names
cc.G1S        # access individual gene sets as attributes
cc.as_dict()  # full dict for passing to annotate

# Cancer pathways (Sanchez-Vega et al. 2018)
cp = jz.mg.CancerPathways()

# Breast cancer subtypes (Parker et al. 2009)
cb = jz.mg.CancerBreast()

# MSigDB 3CA meta-programs (Gavish et al. 2023)
mg = jz.mg.MsigDB3CA()
mg.as_dict()  # pass all 3CA programs to annotate
```

To load your own MSigDB collection:

```python
# Download the .txt file from https://www.gsea-msigdb.org
gene_sets = jz.mg.read_msigdb(
    path="msigdb_human.txt",
    collections=["H"],   # Hallmarks only
)
```

---

## The keep mask system

juzi uses a three-stage boolean mask system to track which factors are retained at each step. This allows any step to be re-run with different parameters without invalidating upstream results.

| Mask | Set by | Meaning |
|---|---|---|
| `juzi_keep_prune` | `prune` | Factors that recur across k resolutions |
| `juzi_keep_similarity` | `similarity`, `select_similarity` | Factors with non-zero similarity and above min_similarity |
| `juzi_keep_cluster` | `cluster` | Factors surviving min_cluster filtering |
| `juzi_keep` | All of the above | Intersection — final set of factors used by cluster |

Re-running any step only updates its own mask and recomputes `juzi_keep`:

```python
# Run similarity once
jz.gp.similarity(adata, distance="jaccard", top_k=50)

# Inspect the retention curve
ax = jz.pl.similarity(adata)

# Try different thresholds without re-running similarity
jz.gp.select_similarity(adata, min_similarity=0.1)
jz.gp.cluster(adata, threshold=0.1, min_cluster=3)

# Try a stricter threshold — only juzi_keep_similarity changes
jz.gp.select_similarity(adata, min_similarity=0.3)
jz.gp.cluster(adata, threshold=0.1, min_cluster=3)

# Re-run cluster with different parameters — only juzi_keep_cluster changes
jz.gp.cluster(adata, threshold=0.2, min_cluster=5)
```

---

## Full pipeline example

```python
import scanpy as sc
import juzi as jz

# Load data
adata = sc.read_h5ad("mammary_gland.h5ad")

# Preprocessing (outside juzi)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)

# Step 1 — NMF (always returns new AnnData)
adata = jz.gp.nmf(
    adata,
    key="donor_id",
    k=[7, 8, 9, 10],
    min_cells=10,
    genes="highly_variable",
    genes_force=age_degs,  # force include age-associated DEGs
    seed=42,
)

# Step 2 — Prune
jz.gp.prune(adata, top_k=50, min_similarity=0.1, min_k=1)

# Step 3 — Similarity
jz.gp.similarity(adata, distance="jaccard", top_k=50)

# Step 3b — Inspect and apply threshold
ax = jz.pl.similarity(adata)
jz.gp.select_similarity(adata, min_similarity=0.2)

# Step 4 — Cluster
jz.gp.cluster(adata, threshold=0.1, min_cluster=3)

# Inspect programs before scoring
genes = jz.ut.program_genes(adata, n_top_genes=20)
df    = jz.ut.program_donors(adata)

# Step 5 — Score
jz.gp.score(adata, n_top_genes=50, seed=42)

# Step 6 — Aggregate
jz.gp.aggregate(
    adata,
    key="donor_id",
    obs_cols=["age", "study_id", "donor_brca"],
)

# Step 7 — Associate
jz.gp.associate(
    adata,
    formula="age + donor_brca + (1|study_id)",
)

# Step 8 — Annotate
jz.gp.annotate(
    adata,
    gene_sets=jz.mg.MsigDB3CA(),
    n_top_genes=50,
)

# Plot results
ax  = jz.pl.heatmap(adata)
fig = jz.pl.loadings(adata, n_top_genes=15)
ax  = jz.pl.associate(adata, padj_thresh=0.05)
ax  = jz.pl.annotate(adata, top_n=10, padj_thresh=0.05)
fig = jz.pl.scores(adata, basis="X_umap")

# Compare programs across lineages
df_compare = jz.ut.program_compare(adata_lhs, adata_lasp, n_top_genes=50)
```

---

## Parameters reference

### `jz.gp.nmf`

| Parameter | Default | Description |
|---|---|---|
| `key` | required | Column in `.obs` for donor identity |
| `k` | `[7,8,9,10]` | Factorisation ranks |
| `min_cells` | `5` | Minimum cells per donor |
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

### `jz.gp.prune`

| Parameter | Default | Description |
|---|---|---|
| `top_k` | `50` | Top genes for Jaccard similarity |
| `min_similarity` | `0.7` | Minimum Jaccard to consider factors matched |
| `min_k` | `1` | Minimum other resolutions a factor must match |
| `matching` | `"greedy"` | `"greedy"` or `"hungarian"` |
| `n_jobs` | `1` | Parallel workers |

### `jz.gp.similarity`

| Parameter | Default | Description |
|---|---|---|
| `distance` | `"jaccard"` | `"jaccard"` or custom callable |
| `top_k` | `50` | Top genes per factor |
| `intra_sample` | `True` | Include within-donor pairs |
| `drop_zeros` | `True` | Flag all-zero similarity rows |
| `n_jobs` | `1` | Parallel workers |

### `jz.gp.select_similarity`

| Parameter | Default | Description |
|---|---|---|
| `min_similarity` | required | Minimum similarity threshold in [0, 1] |

### `jz.gp.cluster`

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.1` | Merge until max inter-cluster similarity < threshold |
| `min_cluster` | `2` | Minimum unique donors per cluster |
| `reorder` | `True` | Sort clusters by size |

### `jz.gp.score`

| Parameter | Default | Description |
|---|---|---|
| `n_top_genes` | `50` | Top genes per program |
| `use_specificity` | `True` | Rank by specificity |
| `n_control_genes` | `50` | Control genes per program gene |
| `gene_names_col` | `None` | `.var` column for gene names |
| `layer` | `None` | Layer for expression values |
| `seed` | `123` | Random seed for control gene sampling |

### `jz.gp.aggregate`

| Parameter | Default | Description |
|---|---|---|
| `key` | required | Column in `.obs` for donor identity |
| `obs_cols` | `None` | Donor-level covariates to propagate |
| `agg` | `"mean"` | `"mean"` or `"median"` |
| `min_cells` | `10` | Minimum cells per donor |

### `jz.gp.associate`

| Parameter | Default | Description |
|---|---|---|
| `formula` | required | R-style formula e.g. `"age + (1\|study_id)"` |
| `reml` | `True` | Use restricted maximum likelihood |
| `padj_method` | `"fdr_bh"` | Multiple testing correction |

### `jz.gp.annotate`

| Parameter | Default | Description |
|---|---|---|
| `gene_sets` | required | Dict or `juzi.mg` object with `as_dict()` method |
| `n_top_genes` | `50` | Top genes per program |
| `use_specificity` | `True` | Rank by specificity |
| `padj_method` | `"fdr_bh"` | Multiple testing correction |

### `jz.ut.program_genes`

| Parameter | Default | Description |
|---|---|---|
| `n_top_genes` | `50` | Top genes per program |
| `use_specificity` | `True` | Rank by specificity |

### `jz.ut.program_compare`

| Parameter | Default | Description |
|---|---|---|
| `n_top_genes` | `50` | Top genes per program |
| `use_specificity` | `True` | Rank by specificity |

### `jz.ut.program_donors`

No parameters beyond `adata`.

---

## License

BSD 3-Clause License. Copyright (c) 2025, Tom Ouellette.
