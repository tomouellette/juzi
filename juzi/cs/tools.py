import numpy as np

from typing import List, Union, Callable, Optional, Tuple
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr


def factor_similarity(
    H: List[np.ndarray],
    distance: Union[str, Callable],
    top_k: Optional[int] = None,
    drop_zeros: bool = True,
    intra_sample: bool = False,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute similarity matrix between factors computed across S samples

    Parameters
    ----------
    H : List[np.ndarray]
        A list of K x M factors (column-wise e.g. genes) from S samples
    distance : Union[str, Callabe]
        One of 'jaccard', 'cosine', 'spearman', 'pearson' or a custom callable
    top_k : Optional[int]
        Compute factor similarity using only the union of the top K loadings
    drop_zeros : bool
        If True, then rows/columns with sum of zero will be dropped from matrix
    intra_sample : bool
        If True, pairwise similarity between intra-sample factors will be kept
    eps : float
        A small value to avoid division by zero

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A similarity matrix between factors and integer value specifying which
        sample (the i-th member in input list) the factor originated from
    """
    Hs = np.vstack(H)
    Ns = Hs.shape[0]
    samples = np.repeat(np.arange(len(H)), H[0].shape[0])

    if callable(distance):
        x, y = np.random.rand(4), np.random.rand(4)
        try:
            d = distance(x, y)
        except:
            raise ValueError("'distance' is not a valid callable")

        if not isinstance(d, (int, float)):
            raise ValueError(
                "If 'distance' is a callable, it must return a scalar")

    similarity = np.zeros((Ns, Ns))
    for i in range(Ns):
        for j in range(i + 1, Ns):
            if i == j:
                continue

            if (samples[i] == samples[j]) and not intra_sample:
                continue

            x = Hs[i]
            y = Hs[j]

            if np.sum(x) == 0 or np.sum(y) == 0:
                continue

            if isinstance(top_k, (int, float)) or distance == "jaccard":
                top_x = np.argsort(x)[-int(top_k):]
                top_y = np.argsort(x)[-int(top_k):]
                union = np.union1d(top_x, top_y)

            if isinstance(top_k, (int, float)):
                if len(union) == 0:
                    continue

                x = x[union]
                y = y[union]

            if distance == "jaccard":
                s_xy = len(np.intersect1d(top_x, top_y)) / len(union)
            elif distance == "cosine":
                s_xy = 1 - cosine(x + eps, y + eps)
            elif distance == "spearman":
                s_xy = spearmanr(x, y).statistic**2
            elif distance == "pearson":
                s_xy = pearsonr(x, y).statistic**2
            elif callable(distance):
                s_xy = distance(x, y)
            else:
                raise ValueError(
                    "'distance' must be one of 'jaccard', 'cosine', 'spearman'"
                    ", 'pearson', or a custom callable")

            if np.isnan(s_xy):
                continue

            similarity[i, j] = s_xy
            similarity[j, i] = s_xy

    if drop_zeros:
        nonzero_rows = ~np.all(similarity == 0, axis=1)
        nonzero_cols = ~np.all(similarity == 0, axis=0)
        kept_indices = np.where(nonzero_rows)[0]

        similarity = similarity[nonzero_rows][:, nonzero_cols]
        samples = samples[kept_indices]

    return similarity, samples
