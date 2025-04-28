# :tangerine: juzi

Various methods for analyzing cell states and types in single-cell sequencing data (experimental).

### cell states (`cs`)

Identifying intra-sample programs.

```python
from juzi.cs.nmf import gaussian_nmf, poisson_nmf, fixed_gaussian_nmf, fixed_poisson_nmf

# Vanilla NMF on normalized and log1p counts
W, H, losses = gaussian_nmf(
    data,
    n_factors=8,
    max_iter=100,
    lambda_H=1e-2,
    init="random",
    eps=1e-7
) 

# Poisson NMF on counts
W, H, losses = poisson_nmf(
    data,
    n_factors=8,
    max_iter=100,
    lambda_H=1e-2,
    init="nndsvd",
    eps=1e-7
)

# Vanilla NMF with fixed H
W, losses = fixed_gaussian_nmf(
    data,
    fixed_H=H,
    max_iter=100,
    init="random",
    eps=1e-7,
    silent=False
)

# Poisson NMF with fixed H
W, losses = fixed_poisson_nmf(
    data,
    fixed_H=H,
    max_iter=100,
    init="random",
    eps=1e-7,
    silent=False
)
```

Identifying consensus (intra-sample) and shared (inter-sample) programs. 

```python
from juzi.cs.tools import consensus_factors, factor_similarity

# Compute a set of consensus factors between runs
HC, HS, labels, correlation = consensus_factors(
    [H1, H2, H3, ...],
    n_clusters=10,
    eps=1e-8,
    method="agglomerative",
    metric="euclidean",
    linkage="ward",
)


# Compute similarity matrix between factors computed across different samples
S, K, ids = factor_similarity(
    [H1, H2, H3, ...],
    distance="cosine",
    top_k=500,
    drop_zeros=True,
    intra_sample=False,
    eps=1e-8
)
```
