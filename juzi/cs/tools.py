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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute similarity matrix between factors computed across samples.

    Parameters
    ----------
    H : List[np.ndarray]
        List of K x M matrices (factors from different samples).
    distance : str or callable
        'jaccard', 'cosine', 'spearman', 'pearson' or a custom function.
    top_k : Optional[int]
        Restrict comparison to top_k genes.
    drop_zeros : bool
        Drop factors with no similarities.
    intra_sample : bool
        Whether to allow comparing factors from the same sample.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Similarity matrix, concatenated factor matrix, sample IDs.
    """

    if distance == "jaccard" and top_k is None:
        raise ValueError("top_k must be set when using Jaccard similarity.")

    Hs = np.vstack(H)
    Ns = Hs.shape[0]
    samples = np.concatenate([np.full(H[i].shape[0], i)
                             for i in range(len(H))])

    kept_indices = np.arange(Ns)  # In case drop_zeros is False

    if callable(distance):
        x, y = np.random.rand(4), np.random.rand(4)
        try:
            d = distance(x, y)
            if not isinstance(d, (int, float)):
                raise ValueError()
        except:
            raise ValueError(
                "'distance' must be a valid callable that returns a scalar.")

    similarity = np.zeros((Ns, Ns))

    for i in range(Ns):
        for j in range(i + 1, Ns):
            if (samples[i] == samples[j]) and not intra_sample:
                continue

            x = Hs[i]
            y = Hs[j]

            if np.sum(x) == 0 or np.sum(y) == 0:
                continue

            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)

            if top_k is not None:
                top_x = np.argsort(x)[-int(top_k):]
                top_y = np.argsort(y)[-int(top_k):]
                union = np.union1d(top_x, top_y)

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
                raise ValueError(f"Unknown distance type: {distance}")

            if np.isnan(s_xy):
                continue

            similarity[i, j] = s_xy
            similarity[j, i] = s_xy

    if drop_zeros:
        nonzero = ~np.all(similarity == 0, axis=1)
        kept_indices = np.where(nonzero)[0]
        similarity = similarity[nonzero][:, nonzero]
        samples = samples[kept_indices]

    return similarity, Hs[kept_indices], samples
