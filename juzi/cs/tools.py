import scipy
import numpy as np
from scipy import sparse
from typing import Union


def score_gene_set(
    X: Union[scipy.sparse.csr_matrix, np.ndarray],
    scale_by_sv: bool = True
) -> np.ndarray:
    """Compute gene set scores using the SVD.

    Parameters
    ----------
    X : Union[np.ndarray sparse.csr_matrix
        Expression matrix subsetted with genes in gene_sets.
    scale_by_sv : bool
        If True, scale scores by singular value (adds size/magnitude to score)

    Returns
    -------
    np.ndarray
        Gene set score per cell.

    References
    ----------
    .. [1] J. Tomfohr, J. Lu, T.B. Kepler. Pathway level analysis of gene
       expression using singular value decomposition. BMC Bioinformatics. 2005.
    """
    if sparse.issparse(X):
        X = X.toarray()

    X -= X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    X /= std

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # First right singular vector similar to PLAGE
    scores = U[:, 0]

    if scale_by_sv:
        scores *= S[0]

    return scores
