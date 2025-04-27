import numpy as np

from tqdm import tqdm
from typing import Tuple, List
from scipy.linalg import svd
from scipy.optimize import nnls


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
    .. [1] D.D Lee & S. Seung. Algorithms for Non-negative Matrix Factorization.
       Advances in Neural Information Processing Systems (NeurIPS) 13. 2000.
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
    .. [1] D.D Lee & S. Seung. Algorithms for Non-negative Matrix Factorization.
       Advances in Neural Information Processing Systems (NeurIPS) 13. 2000.
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
    .. [1] D.D Lee & S. Seung. Algorithms for Non-negative Matrix Factorization.
       Advances in Neural Information Processing Systems (NeurIPS) 13. 2000.
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
                WH = np.dot(W, fixed_H)  # Recompute with updated W
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
    .. [1] D.D Lee & S. Seung. Algorithms for Non-negative Matrix Factorization.
       Advances in Neural Information Processing Systems (NeurIPS) 13. 2000.
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
                WH = np.dot(W, fixed_H)  # Recompute with updated W
                WH = np.maximum(WH, eps)
                kl = np.sum(counts * (np.log((counts + eps) / WH) - 1) + WH)
                log.append(kl)
                progress.set_postfix({'loss': f'{kl:.4f}'})

    return W, log
