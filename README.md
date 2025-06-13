# :tangerine: juzi

Various methods for analyzing cancer cell states and types in single-cell sequencing data (experimental).

- [Installation](#installation)
- [Usage](#usage)
  - [cell states (`cs`)](#cell-states-cs)
  - [marker genes (`mg`)](#marker-genes-mg)

## Installation

```bash
pip install juzi
```

## Usage

### cell states (`cs`)

Identification of recurrent gene programs across multiple cancer samples (inspired by [Gavish et al. 2023](https://www.nature.com/articles/s41586-023-06130-4)).

```python
import juzi as jz

# Perform NMF across each unique sample at multiple factor resolutions
subset = jz.cs.nmf(
    adata,
    key="sample_id",
    layer="counts",
    genes="highly_variable",
    min_cells=100,
    k=[7, 8, 9, 10],
    max_iter=1000,
    n_jobs=8,
    seed=123
)

# Prune recurrent intra-sample factors based on overlapping top genes
jz.cs.prune(
    subset,
    top_k=50,
    min_similarity=0.7,
    min_k=1,
    n_jobs=1
)

# Compute inter-sample similarity matrix between factors
jz.cs.similarity(
    subset,
    distance="jaccard",
    top_k=50,
    intra_sample=True,
    drop_zeros=True,
    min_similarity=0.2,
    n_jobs=8,
    prefer="threads"
)

# Cluster factor similarity matrix by iterative merging
jz.cs.cluster(
    subset,
    threshold=0.1,
    min_cluster=5
)

# Plot clustered factor similarity matrix (i.e. gene programs)
jz.cs.plot_programs(
    subset,
    vmin=0.,
    vmax=1.,
    figsize=(5., 5.),
    cbar_label="Similarity"
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
