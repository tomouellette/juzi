import itertools
import numpy as np

from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import fcluster, linkage as linkage_
from typing import List, Union, Callable, Optional, Tuple


def consensus_factors(
    H: List[np.ndarray],
    n_clusters: int = 10,
    eps: float = 1e-8,
    method: str = "agglomerative",
    max_iter: int = 300,
    metric: str = 'euclidean',
    linkage: str = 'ward',
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a consensus H matrix from multiple intra-sample NMF runs.

    Parameters
    ----------
    H : List[np.ndarray]
        List of K x M matrices (factors from different samples).
    n_clusters : int
        Number of consensus factors.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Consensus factors, merged factors, merged factors cluster labels,
        and the intracluster correlation between factors.
    """
    Hs = []
    for hi in H:
        hi = hi / (np.linalg.norm(hi, axis=1, keepdims=True) + eps)
        Hs.append(hi)

    Hs = np.vstack(Hs)

    if n_clusters > Hs.shape[0]:
        raise ValueError(
            "n_clusters must be less than or equal to the number of factors.")

    if method == "kmeans":
        _, labels = kmeans2(
            data=Hs,
            k=n_clusters,
            iter=300,
            thresh=1e-8,
            minit='++'
        )
    elif method == "agglomerative":
        distance_matrix = pdist(Hs, metric=metric)
        Z = linkage_(distance_matrix, method=linkage)
        labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    else:
        raise ValueError('method' must be one of 'kmeans' or 'agglomerative'")

    # Assign consensus as median within each cluster
    H_consensus = []
    for c in range(n_clusters):
        members = Hs[labels == c]
        median = np.median(members, axis=0)
        H_consensus.append(median)

    H_consensus = np.stack(H_consensus)

    # Compute average correlation between factors in a cluster
    cluster_correlation = np.zeros(n_clusters)
    for c in range(n_clusters):
        members = Hs[labels == c]
        if len(members) > 1:
            corr = []
            for i, j in itertools.product(range(len(members)), repeat=2):
                if i != j and np.sum(members[i]) > 0 and np.sum(members[j]) > 0:
                    corr.append(spearmanr(members[i], members[j]).statistic**2)

            if len(corr) > 0:
                cluster_correlation[c] = np.mean(corr)

    return H_consensus, Hs, labels, cluster_correlation


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
        A list of K x M matrices specifying factors across S different samples.
    distance : str or callable
        One of 'jaccard', 'cosine', 'spearman', 'pearson' or a custom callable. 
    top_k : Optional[int]
        Compute the distance function on the union of the top K loadings.
    drop_zeros : bool
        Drop rows/columns in similarity matrix where the sum is zero.
    intra_sample : bool
        If True, compute similarity scores between factors from same sample.
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

    kept_indices = np.arange(Ns)

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
