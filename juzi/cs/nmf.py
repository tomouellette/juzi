import itertools
import numpy as np

from tqdm import tqdm
from typing import List, Union, Callable, Optional, Tuple
from scipy.linalg import svd
from scipy.optimize import nnls
from scipy.cluster.hierarchy import fcluster, linkage as linkage_
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist
from scipy.cluster.vq import kmeans2


def random_init(
    n: int,
    m: int,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly initialize W and H matrices for NMF.

    Parameters
    ----------
    n : int
        Number of rows in the original matrix
    m : int
        Number of columns in the original matrix
    k : int
        Number of latent factors

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Randomly initialized W (n x k) and H (k x m) matrices
    """
    W = np.random.rand(n, k)
    H = np.random.rand(k, m)
    return W, H


def nndsvd_init(
    counts: np.ndarray,
    n_factors: int,
    variant: str = 'none',
    eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """NNDSVD initialization for NMF.

    Parameters
    ----------
    counts : np.ndarray
        N x M non-negative data matrix
    n_factors : int
        Number of latent factors
    variant : str
        Do not add random noise to zeros ('none') or add random noise ('add')
    eps : float
        Small value to add to zeros if variant='add' or 'ar'

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Initialized W (N x k) and H (k x M) matrices

    References
    ----------
    .. [1] C. Boutsidis & E. Gallopoulos. SVD-based initialization: A head
       start for nonnegative matrix factorization. Pattern Recognition. 2007.
    """
    counts = np.maximum(counts, eps)
    n, m = counts.shape

    # Compute truncated SVD
    U, S, Vh = svd(counts, full_matrices=False)

    W = np.zeros((n, n_factors))
    H = np.zeros((n_factors, m))

    # Initialize the first component
    W[:, 0] = np.sqrt(S[0]) * np.maximum(U[:, 0], 0)
    H[0, :] = np.sqrt(S[0]) * np.maximum(Vh[0, :], 0)

    # Initialize the other components
    for j in range(1, n_factors):
        u = U[:, j]
        v = Vh[j, :]

        u_pos = np.maximum(u, 0)
        u_neg = np.maximum(-u, 0)
        v_pos = np.maximum(v, 0)
        v_neg = np.maximum(-v, 0)

        u_pos_norm = np.linalg.norm(u_pos)
        v_pos_norm = np.linalg.norm(v_pos)
        u_neg_norm = np.linalg.norm(u_neg)
        v_neg_norm = np.linalg.norm(v_neg)

        pos_product = u_pos_norm * v_pos_norm
        neg_product = u_neg_norm * v_neg_norm

        if pos_product > neg_product:
            W[:, j] = np.sqrt(S[j] * pos_product) * \
                (u_pos / (u_pos_norm + eps))
            H[j, :] = np.sqrt(S[j] * pos_product) * \
                (v_pos / (v_pos_norm + eps))
        else:
            W[:, j] = np.sqrt(S[j] * neg_product) * \
                (u_neg / (u_neg_norm + eps))
            H[j, :] = np.sqrt(S[j] * neg_product) * \
                (v_neg / (v_neg_norm + eps))

    # Handle zeros
    if variant == 'add':
        W[W < eps] = np.random.rand(*W[W < eps].shape) * eps
        H[H < eps] = np.random.rand(*H[H < eps].shape) * eps

    return W, H


def gaussian_nmf(
    counts: np.ndarray,
    n_factors: int,
    max_iter: int = 300,
    init: str = 'nndsvd',
    lambda_H: float = 0.0,
    eps: float = 1e-6,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Gaussian non-negative matrix factorization

    Parameters
    ----------
    counts : np.ndarray
        A N x M count matrix
    n_factors : int
        Number of latent factors to fit
    max_iter : int
        Maximum number of iterations to perform
    init : str
        Type of initialization to perform ('random' or 'nndsvd')
    lambda_H : float
        Sparsity regularization on H
    eps : float
        A small amount to avoid division by zero
    silent : bool
        If True, progress bar will be suppressed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[float]]
        Row-wise factors, column-wise factors, and loss log

    References
    ----------
    .. [1] D.D. Lee, S. Seung. Algorithms for Non-negative Matrix Factorization.
       NeurIPS. 2001
    """
    counts = np.maximum(counts, eps)
    n, m = counts.shape

    if init == 'random':
        W, H = random_init(n, m, n_factors)
    elif init == 'nndsvd':
        W, H = nndsvd_init(counts, n_factors, 'add', eps)
    else:
        raise ValueError("'init' must be one of 'random' or 'nndsvd'")

    log = []
    with tqdm(
        range(max_iter),
        desc='juzi | Fitting',
        disable=silent
    ) as progress:
        for _ in progress:
            WH = np.dot(W, H)
            WH = np.maximum(WH, eps)

            # Update H
            numerator_H = np.dot(W.T, counts)
            denominator_H = np.dot(W.T, WH) + lambda_H
            H *= numerator_H / (denominator_H + eps)

            # Update W
            WH = np.dot(W, H)
            WH = np.maximum(WH, eps)
            numerator_W = np.dot(counts, H.T)
            denominator_W = np.dot(WH, H.T)
            W *= numerator_W / (denominator_W + eps)

            if not silent:
                loss = np.sum((counts - WH) ** 2)
                log.append(loss)
                progress.set_postfix({'loss': f'{loss}'})

    return W, H, log


def poisson_nmf(
    counts: np.ndarray,
    n_factors: int,
    max_iter: int = 300,
    init: str = 'nndsvd',
    lambda_H: float = 0.0,
    eps: float = 1e-6,
    silent: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Poisson non-negative matrix factorization

    Parameters
    ----------
    counts : np.ndarray
        A N x M count matrix
    n_factors : int
        Number of latent factors to fit
    max_iter : int
        Maximum number of iterations to perform
    init : str
        Type of initialization to perform ('random' or 'nndsvd')
    lambda_H : float
        Sparsity regularization on H
    eps : float
        A small amount of error to avoid division by zero
    silent : bool
        If True, progress bar will be suppressed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[float]]
        Row-wise factors, column-wise factors, and loss log

    References
    ----------
    .. [1] D.D. Lee, S. Seung. Algorithms for Non-negative Matrix Factorization.
       NeurIPS. 2001
    """
    counts = np.maximum(counts, eps)
    n, m = counts.shape

    if init == 'random':
        W, H = random_init(n, m, n_factors)
    elif init == 'nndsvd':
        W, H = nndsvd_init(counts, n_factors, 'add', eps)
    else:
        raise ValueError("'init' must be one of 'random' or 'nndsvd'")

    ones = np.ones_like(counts)

    log = []
    with tqdm(
        range(max_iter),
        desc='juzi | Fitting',
        disable=silent
    ) as progress:
        for _ in progress:
            WH = np.dot(W, H)
            WH = np.maximum(WH, eps)

            # Elementwise division: X / (WH)
            X_div_WH = counts / WH

            # Update H
            numerator_H = np.dot(W.T, X_div_WH)
            denominator_H = np.dot(W.T, ones) + lambda_H
            H *= numerator_H / (denominator_H + eps)

            # Recompute WH after H update
            WH = np.dot(W, H)
            WH = np.maximum(WH, eps)
            X_div_WH = counts / WH

            # Update W
            numerator_W = np.dot(X_div_WH, H.T)
            denominator_W = np.dot(ones, H.T)
            W *= numerator_W / (denominator_W + eps)

            if not silent:
                kl = np.sum(counts * (np.log(counts / WH) - 1) + WH)
                log.append(kl)
                progress.set_postfix({'loss': f'{kl}'})

    return W, H, log


def fixed_gaussian_nmf(
    counts: np.ndarray,
    fixed_H: np.ndarray,
    max_iter: int = 300,
    init: str = 'random',
    eps: float = 1e-6,
    silent: bool = False
) -> Tuple[np.ndarray, List[float]]:
    """Gaussian non-negative matrix factorization with fixed H

    Parameters
    ----------
    counts : np.ndarray
        A N x M count matrix
    fixed_H : np.ndarray
        Fixed column-wise factors (K x M)
    max_iter : int
        Maximum number of iterations to perform
    init : str
        Type of initialization for W ('random' or 'nnls')
    eps : float
        A small amount to avoid division by zero
    silent : bool
        If True, progress bar will be suppressed

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        Optimized row-wise factors (W) and loss log

    References
    ----------
    .. [1] D.D. Lee, S. Seung. Algorithms for Non-negative Matrix Factorization.
       NeurIPS. 2001
    """
    counts = np.maximum(counts, eps)
    n, m = counts.shape
    k = fixed_H.shape[0]

    if init == 'random':
        W = np.random.rand(n, k)
    elif init == 'nnls':
        W = np.zeros((n, k))
        for i in range(n):
            W[i], _ = nnls(fixed_H.T, counts[i])
    else:
        raise ValueError("'init' must be one of 'random' or 'nnls'")

    log = []
    with tqdm(
        range(max_iter),
        desc='juzi | Fitting W with fixed H',
        disable=silent
    ) as progress:
        for _ in progress:
            WH = np.dot(W, fixed_H)
            WH = np.maximum(WH, eps)

            # Update W only (H is fixed)
            numerator_W = np.dot(counts, fixed_H.T)
            denominator_W = np.dot(WH, fixed_H.T)
            W *= numerator_W / (denominator_W + eps)

            if not silent:
                # Compute squared Frobenius norm loss
                WH = np.dot(W, fixed_H)
                loss = np.sum((counts - WH) ** 2)
                log.append(loss)
                progress.set_postfix({'loss': f'{loss:.4f}'})

    return W, log


def fixed_poisson_nmf(
    counts: np.ndarray,
    fixed_H: np.ndarray,
    max_iter: int = 300,
    init: str = 'random',
    eps: float = 1e-6,
    silent: bool = False
) -> Tuple[np.ndarray, List[float]]:
    """Poisson non-negative matrix factorization with fixed H

    Parameters
    ----------
    counts : np.ndarray
        A N x M count matrix
    fixed_H : np.ndarray
        Fixed column-wise factors (K x M)
    max_iter : int
        Maximum number of iterations to perform
    init : str
        Type of initialization for W ('random' or 'nnls')
    eps : float
        A small amount of error to avoid division by zero
    silent : bool
        If True, progress bar will be suppressed

    Returns
    -------
    Tuple[np.ndarray, List[float]]
        Optimized row-wise factors (W) and loss log

    References
    ----------
    .. [1] D.D. Lee, S. Seung. Algorithms for Non-negative Matrix Factorization.
       NeurIPS. 2001
    """
    counts = np.maximum(counts, eps)
    n, m = counts.shape
    k = fixed_H.shape[0]

    if init == 'random':
        W = np.random.rand(n, k)
    elif init == 'nnls':
        W = np.zeros((n, k))
        for i in range(n):
            W[i], _ = nnls(fixed_H.T, counts[i])
    else:
        raise ValueError("'init_W' must be one of 'random' or 'nnls'")

    ones = np.ones_like(counts)

    log = []
    with tqdm(
        range(max_iter),
        desc='juzi | Fitting',
        disable=silent
    ) as progress:
        for _ in progress:
            WH = np.dot(W, fixed_H)
            WH = np.maximum(WH, eps)

            # Elementwise division: counts / (WH)
            X_div_WH = counts / WH

            # Update W only (H is fixed)
            numerator_W = np.dot(X_div_WH, fixed_H.T)
            denominator_W = np.dot(ones, fixed_H.T)
            W *= numerator_W / (denominator_W + eps)

            if not silent:
                # Compute KL divergence loss
                WH = np.dot(W, fixed_H)
                WH = np.maximum(WH, eps)
                kl = np.sum(counts * (np.log((counts + eps) / WH) - 1) + WH)
                log.append(kl)
                progress.set_postfix({'loss': f'{kl:.4f}'})

    return W, log


def factor_consensus(
    H: List[np.ndarray],
    n_clusters: int = 10,
    eps: float = 1e-8,
    method: str = "agglomerative",
    max_iter: int = 300,
    metric: str = 'euclidean',
    linkage: str = 'ward',
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a consensus H matrix from multiple intra-sample NMF runs.

    Parameters
    ----------
    H : List[np.ndarray]
        List of K x M matrices (factors from different samples).
    n_clusters : int
        Number of consensus factors. 
    method : str
        Clustering method ('agglomerative' or 'kmeans')
    max_iter : int
        If method is 'kmeans', maximum number of iterations
    metric : str
        If method is 'agglomerative', the pairwise distance metric
    linkage : str
        If method is 'agglomerative', the linkage function
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Consensus factors, merged factors, merged factors cluster labels,
        and the intracluster correlation between factors.
    """
    rng = np.random.default_rng(seed)

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
            iter=max_iter,
            thresh=1e-8,
            minit='++',
            seed=rng
        )
    elif method == "agglomerative":
        distance_matrix = pdist(Hs, metric=metric)
        Z = linkage_(distance_matrix, method=linkage)
        labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    else:
        raise ValueError("'method' must be one of 'kmeans' or 'agglomerative'")

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
                raise ValueError(
                    "'distance' must be a callable that returns a scalar.")
        except:
            raise ValueError(
                "'distance' must be callable that accepts two arrays.")

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
