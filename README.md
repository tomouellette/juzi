# :tangerine: juzi

Various methods for analyzing cell states and types in single-cell sequencing data (experimental).

- [cell states (`cs`)](#cell-states-cs)
- [marker genes (`mg`)](#marker-genes-mg)

### Installation

```bash
pip install juzi
```

### cell states (`cs`)

Identifying intra-sample programs.

```python
from juzi.cs.nmf import multisample_nmf

# Perform NMF independently across multiple samples
Hs = multisample_nmf(
    [C1, C2, C3, ...],
    k=[5, 6, 7, 8, 9],
    center=False,
    scale=False,
    target_sum=1e5,
    max_exp=10.0,
    max_iter=500,
    alpha=0.,
    tol=0.0001,
    loss="frobenius",
    n_jobs=8,
    prefer="threads",
    seed=123
) 
```

Identifying consensus (intra-sample) and shared (inter-sample) programs. 

```python
from juzi.cs.nmf import factor_consensus, factor_similarity

# Compute a set of clustered consensus factors between NMF runs
HC, HS, labels, correlation = factor_consensus(
    np.array_split(Hs[0], [5, 6, 7, 8, 9]),
    n_clusters=8,
    eps=1e-8,
    method="agglomerative",
    metric="euclidean",
    linkage="ward",
)

# Compute similarity matrix between factors computed across different samples
S, K, ids = factor_similarity(
    Hs,
    distance="cosine",
    top_k=500,
    drop_zeros=True,
    intra_sample=False,
    eps=1e-8
)
```

Some additional tools for clustering and filtering factors.

```python
from juzi.cs.cluster import eigengap_heuristic, spectral_clustering

# Estimate an optimal number of clusters from a similarity matrix
results = eigengap_heuristic(
    S,
    min_clusters=2,
    max_clusters=10,
    normalize=True,
    eps=1e-8
)

results.plot(show=True)

# Cluster a similarity matrix given an input number of clusters
assignments = spectral_clustering(
    S,
    n_clusters=results.k,
    normalize=True,
    seed=123456,
    eps=1e-8
)
```

### marker genes (`mg`)

Various marker genes for cell types, subtypes, and pathways.

```python
from juzi.mg import available_sets

# Check available marker gene sets
print(available_sets())

# Load breast cancer gene sets (e.g. PAM50)
from juzi.mg import CancerBreast
markers = CancerBreast()

# Load cancer pathway gene sets (e.g. HIPPO)
from juzi.mg import CancerPathways
markers = CancerPathways()

# Load cell cycle gene sets (e.g. G1S)
from juzi.mg import CellCycle
markers = CellCycle()

# List available sets in a given marker class
markers.info()

# Get all genes from all sets in a flattened list
markers.all()
```
