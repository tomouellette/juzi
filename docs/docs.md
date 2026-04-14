# docs

## Overview

The juzi pipeline consists of seven analytical steps and four plotting functions:

#### Analysis

```
juzi.gp.nmf          — per-donor NMF at multiple resolutions
juzi.gp.prune        — remove non-recurrent intra-donor factors
juzi.gp.similarity   — compute inter-donor factor similarity
juzi.gp.cluster      — cluster factors into consensus programs
juzi.gp.score        — score cells on consensus programs
juzi.gp.aggregate    — pseudobulk donor-level program scores
juzi.gp.associate    — LMM association testing
juzi.gp.annotate     — overlap with reference gene sets
```

#### Plotting

```
juzi.pl.similarity   — factor similarity matrix heatmap
juzi.pl.loadings     — top gene loadings per program
juzi.pl.associate    — coefficient plot of association results
juzi.pl.scores       — embedding coloured by program scores
juzi.pl.annotate     — dot plot of gene set overlaps
```

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

### Step 1 — NMF

Fit NMF independently on each donor at multiple resolutions. juzi normalises and log-transforms counts internally.

```python
adata = jz.gp.nmf(
    adata,
    key="donor_id",           # column in .obs denoting donor identity
    k=[7, 8, 9, 10],          # factorisation ranks — use multiple for stability
    min_cells=10,             # minimum cells per donor to include
    genes="highly_variable",  # boolean .var column, list of genes, or None
    target_sum=1e4,           # library size normalisation target
    seed=42,
)
```

> **Note:** `nmf` always returns a new AnnData object subset to the selected genes and donors that passed `min_cells` filtering. The input object is never modified in place.

Key outputs stored in the returned AnnData:

| Field | Location | Description |
|---|---|---|
| `juzi_G` | `.varm` | Gene × factor loading matrix |
| `juzi_k` | `.uns` | List of k values used |
| `juzi_names` | `.uns` | Donor identity per factor column |
| `juzi_G_genes` | `.uns` | Gene names corresponding to `juzi_G` rows |

---

### Step 2 — Prune

Remove non-recurrent factors within each donor. A factor is considered recurrent if it shares sufficient top-gene overlap with at least one factor from each of `min_k` other resolutions.

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
| `juzi_keep` | `.uns` | Boolean mask of retained factors |

---

### Step 3 — Similarity

Compute pairwise Jaccard similarity between all retained factors across all donors.

```python
jz.gp.similarity(
    adata,
    distance="jaccard",  # "jaccard" or a custom callable
    top_k=50,            # top genes used per factor
    intra_sample=False,  # whether to compute within-donor similarities
    min_similarity=0.1,  # flag factors below this maximum similarity
    drop_zeros=True,     # flag factors with all-zero similarity rows
)
```

Outputs:

| Field | Location | Description |
|---|---|---|
| `juzi_similarity` | `.uns` | Factor × factor similarity matrix |
| `juzi_keep` | `.uns` | Updated boolean mask |

---

### Step 4 — Cluster

Cluster the factor similarity matrix into consensus programs using iterative average-linkage merging.

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
| `juzi_cluster_similarity` | `.uns` | Reordered factor similarity matrix |
| `juzi_cluster_labels` | `.uns` | Cluster label per factor |
| `juzi_cluster_G` | `.uns` | Centroid gene loading per program |
| `juzi_cluster_samples` | `.uns` | Contributing donors per program |
| `juzi_cluster_stats` | `.uns` | Silhouette, inner/outer similarity |

---

### Step 5 — Score

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

### Step 6 — Aggregate

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

### Step 7 — Associate

Test whether consensus program activity associates with a covariate of interest using a linear mixed model. Supports R-style formula notation with random effects.

```python
jz.gp.associate(
    adata,
    formula="age + donor_brca + (1|study_id)",
    reml=True,  # use restricted maximum likelihood
)
```

Multiple random effects are combined into an interaction grouping variable:

```python
# (1|study_id) + (1|batch) → groups = study_id_x_batch
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

### Step 8 — Annotate

Score consensus programs against reference gene sets to aid biological interpretation.

```python
# Using a built-in juzi.mg gene set
gene_sets = jz.mg.CellCycle().as_dict()

# Or from MSigDB
gene_sets = jz.mg.read_msigdb(
    "msigdb_human.txt",
    collections=["C4:3CA"],
)

# Or a custom dict
gene_sets = {
    "MY_PROGRAM": ["GENE1", "GENE2", "GENE3"],
}

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
| `juzi_annotation` | `.uns` | Tidy DataFrame with Jaccard and hypergeometric p-values |

---

## Plotting

### Factor similarity matrix

```python
ax = jz.pl.similarity(adata)
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
# Requires a 2D embedding in adata.obsm
sc.pp.neighbors(adata)
sc.tl.umap(adata)

fig = jz.pl.scores(adata, basis="X_umap")
```

### Gene set annotation dot plot

```python
ax = jz.pl.annotate(adata, top_n=10, padj_thresh=0.05)
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
```

To load your own MSigDB collection:

```python
# Download the .txt file from https://www.gsea-msigdb.org
gene_sets = jz.mg.read_msigdb(
    path="msigdb_human.txt",
    collections=["H"],  # Hallmarks only
)
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

# NMF (always returns new AnnData)
adata = jz.gp.nmf(
    adata,
    key="donor_id",
    k=[7, 8, 9, 10],
    min_cells=10,
    genes="highly_variable",
    seed=42,
)

# Prune
jz.gp.prune(adata, top_k=50, min_similarity=0.1, min_k=1)

# Similarity
jz.gp.similarity(adata, distance="jaccard", top_k=50, min_similarity=0.1)

# Cluster
jz.gp.cluster(adata, threshold=0.1, min_cluster=3)

# Score
jz.gp.score(adata, n_top_genes=50, seed=42)

# Aggregate
jz.gp.aggregate(
    adata,
    key="donor_id",
    obs_cols=["age", "study_id", "donor_brca"],
)

# Associate
jz.gp.associate(
    adata,
    formula="age + donor_brca + (1|study_id)",
)

# Annotate
jz.gp.annotate(
    adata,
    gene_sets=jz.mg.MsigDB3CA(),
    n_top_genes=50,
)

# Plot results
fig = jz.pl.loadings(adata, n_top_genes=15)
ax  = jz.pl.associate(adata, padj_thresh=0.05)
ax  = jz.pl.annotate(adata, top_n=10, padj_thresh=0.05)
fig = jz.pl.scores(adata, basis="X_umap")
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
| `min_similarity` | `0.2` | Flag factors below this threshold |
| `drop_zeros` | `True` | Flag all-zero similarity rows |
| `n_jobs` | `1` | Parallel workers |

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
| `gene_sets` | required | Dict or `juzi.mg` object |
| `n_top_genes` | `50` | Top genes per program |
| `use_specificity` | `True` | Rank by specificity |
| `padj_method` | `"fdr_bh"` | Multiple testing correction |

---

## License

BSD 3-Clause License. Copyright (c) 2025, Tom Ouellette.
