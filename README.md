# :tangerine: juzi

Various methods for analyzing cell states and types in single-cell sequencing data.

### cell states (`cs`)

```python
from juzi.cs.nmf import gaussian_nmf, poisson_nmf, fixed_poisson_nmf

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
