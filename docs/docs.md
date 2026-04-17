# docs

## Overview

### Analysis (`juzi.gp`)

```
juzi.gp.nmf_fit             — per-sample NMF across multiple ranks
juzi.gp.nmf_prune           — remove non-recurrent and redundant factors

juzi.gp.similarity_compute  — compute inter-sample factor similarity
juzi.gp.similarity_filter   — filter factors by similarity threshold

juzi.gp.programs_threshold  — select clustering threshold
juzi.gp.programs_cluster    — cluster factors into programs (centroid or progressive)
juzi.gp.programs_remove     — remove unwanted programs
juzi.gp.programs_merge      — merge related programs
juzi.gp.programs_annotate   — gene set enrichment

juzi.gp.programs_stability  — assess program reproducibility

juzi.gp.score_cells         — score cells on programs
juzi.gp.score_classify      — classify cells via permutation null
juzi.gp.score_aggregate     — aggregate to sample-level
juzi.gp.score_associate     — association testing
```

---

### Plotting (`juzi.pl`)

```
juzi.pl.similarity          — similarity threshold selection

juzi.pl.programs_threshold  — clustering threshold sweep
juzi.pl.programs_heatmap    — factor similarity + program structure
juzi.pl.programs_loadings   — program gene loadings
juzi.pl.programs_stability  — program reproducibility

juzi.pl.programs_annotate   — gene set enrichment heatmap

juzi.pl.score_embedding     — embedding coloured by program scores
juzi.pl.score_associate     — association results
```

---

### Utilities (`juzi.ut`)

```
juzi.ut.programs_genes      — canonical genes per program
juzi.ut.programs_compare    — program similarity across datasets
juzi.ut.programs_donors     — donor contribution per program

juzi.ut.factor_loadings     — gene × factor matrix
juzi.ut.factor_scores       — cell × factor matrix
```

---

## Full pipeline

```python
# Run NMF across multiple ranks
adata = jz.gp.nmf_fit(adata, key="sample_id", k=[7,8,9])

# Prune non-recurrent and redundant factors
jz.gp.nmf_prune(adata)

# Compute similarity matrix using top 50 genes per factor
jz.gp.similarity_compute(adata, top_k=50)

# Remove factors with low similarity
jz.gp.similarity_filter(adata, min_similarity=0.2)

# Cluster donor-level factors into programs
jz.gp.programs_cluster(adata, threshold=0.3, strategy="progressive")

# Optionally assess program stability using LOO
jz.gp.programs_stability(adata)

# Score cells on programs using top 50 genes per program
jz.gp.score_cells(adata, gene_names_col="gene_name", n_top_genes=50)
jz.gp.score_aggregate(adata, key="sample_id")
jz.gp.score_associate(adata, formula="age + (1|study)")
```

---

## License

BSD 3-Clause License. Copyright (c) 2025, Tom Ouellette.
