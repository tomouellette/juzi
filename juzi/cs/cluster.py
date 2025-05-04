import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.cluster.vq import kmeans2


@dataclass
class EigengapData:
    """Container for eigengap_heuristic results.

    Attributes
    ----------
    k: int
        Estimated number of clusters.
    min_clusters: int
        Minimum number of clusters to evaluate.
    max_clusters: int
        Maximum number of clusters to evaluate.
    eigengaps: np.ndarray
        Array of eigengaps (differences between consecutive eigenvalues).
    sorted_eigenvalues: np.ndarray
        Sorted array of eigenvalues.
    """

    k: int
    min_clusters: int
    max_clusters: int
    eigengaps: np.ndarray
    sorted_eigenvalues: np.ndarray

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (5.25, 3),
        fontsize: int = 9,
        legend_anchor: Tuple[float, float] = (0.5, 1.2),
        show: bool = False
    ) -> plt.Axes:
        """Visualize eigengap across the evaluated cluster range.

        Parameters
        ----------
        ax : Optional[plt.Axes]
            An optional axis from existing matplotlib figure
        figsize : Tuple[int, int]
            Width and height of plot if axis is not provided
        fontsize : int
            Plot fontsize
        legend_anchor : Tuple[float, float]
            Position of legend in terms of horiztonal and vertical fraction
        show: bool
            Show the plot instead of returning a plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            np.arange(self.min_clusters, self.max_clusters + 1),
            self.eigengaps,
            'o-',
            label='Eigengaps',
            color='#467e1b'
        )
        ax.set_xlabel("Clusters (k)", fontsize=fontsize)
        ax.set_ylabel("Eigengap (λₖ₊₁ - λₖ)", fontsize=fontsize)

        ax2 = ax.twinx()
        ax2.plot(
            np.arange(self.min_clusters, len(self.sorted_eigenvalues)),
            self.sorted_eigenvalues[self.min_clusters:],
            marker='.',
            label='Eigenvalues',
            linewidth=2,
            linestyle="dotted",
            color='#fd9d34'
        )
        ax2.set_ylabel("Eigenvalue", fontsize=fontsize)
        ax2.tick_params(axis='y', labelsize=fontsize)

        ax.axvspan(xmin=self.k - 0.1, xmax=self.k +
                   0.1, color='black', alpha=0.1)
        ax.axvline(x=self.k, color='black', linestyle='--',
                   label=f'Optimal K={self.k}')
        ax.legend()

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        combined_legend = ax.legend(
            lines + lines2,
            labels + labels2,
            loc='upper center',
            ncol=3,
            frameon=False,
            fontsize=fontsize
        )

        combined_legend.set_bbox_to_anchor(
            legend_anchor, transform=ax.transAxes)

        if show:
            plt.tight_layout()
            plt.show()
        else:
            return ax


def eigengap_heuristic(
    similarity_matrix: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 10,
    normalize: bool = True,
    eps: float = 1e-8,
) -> EigengapData:
    """Estimate number of clusters using the eigengap heuristic.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        A pairwise similarity matrix.
    min_clusters : int
        Minimum clusters to evaluate.
    max_clusters : int
        Maximum clusters to evaluate.
    normalize : bool
        If True, compute the normalized Laplacian before computing eigengaps
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    EigengapData
        Dataclass storing estimated k, sorted eigenvalues, and eigengaps
    """

    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2.")
    if min_clusters > max_clusters:
        raise ValueError(
            "min_clusters must be less than or equal to max_clusters.")

    # Symmetrize the matrix
    W = (similarity_matrix + similarity_matrix.T) / 2

    d = np.sum(W, axis=1)
    if normalize:
        d_safe = np.maximum(d, eps)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(d_safe))
        L = np.eye(W.shape[0]) - D_sqrt_inv @ W @ D_sqrt_inv
    else:
        D = np.diag(d)
        L = D - W

    n = L.shape[0]
    max_k = max_clusters + 1

    if max_k >= n:
        raise ValueError(
            "max_clusters is too large for the size of the matrix.")

    # Ensure the matrix is symmetric and convert to sparse for performance
    L = (L + L.T) / 2
    L_sparse = sparse.csr_matrix(L)

    # Compute smallest eigenvalues of the Laplacian
    eigenvalues, _ = eigsh(L_sparse, k=max_k, which="SM")
    sorted_eigenvalues = np.sort(eigenvalues)

    # Compute eigengaps
    eigengaps = np.diff(sorted_eigenvalues)
    relevant_eigengaps = eigengaps[min_clusters - 1:max_clusters]

    max_gap_index = np.argmax(relevant_eigengaps)
    optimal_k = max_gap_index + min_clusters

    return EigengapData(
        k=optimal_k,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        eigengaps=relevant_eigengaps,
        sorted_eigenvalues=sorted_eigenvalues,
    )


def spectral_clustering(
    similarity_matrix: np.ndarray,
    n_clusters: int,
    normalize: bool = True,
    seed: int = 123,
    eps: float = 1e-8,
) -> np.ndarray:
    """Estimate number of clusters using the eigengap heuristic.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        A pairwise similarity matrix.
    n_clusters : int
        Number of clusters to fit.
    normalize : bool
        If True, compute the normalized Laplacian before computing eigengaps
    seed : int
        Random seed for reproducibility.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Cluster assignments

    References
    ----------
    .. [1] A.Y. Ng, M.I. Jordan, Y. Weiss. On Spectral Clustering: Analysis and
       an algorithm. NeurIPS. 2001.
    """
    rng = np.random.default_rng(seed)

    W = (similarity_matrix + similarity_matrix.T) / 2

    d = np.sum(W, axis=1)

    if normalize:
        d_safe = np.maximum(d, eps)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(d_safe))
        L = np.eye(W.shape[0]) - D_sqrt_inv @ W @ D_sqrt_inv
    else:
        D = np.diag(d)
        L = D - W

    L = (L + L.T) / 2
    L_sparse = sparse.csr_matrix(L)

    n_eigvecs = n_clusters
    eigenvalues, eigenvectors = eigsh(L_sparse, k=n_eigvecs, which='SM')

    if normalize:
        row_norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        embedding = eigenvectors / (row_norms + eps)
    else:
        embedding = eigenvectors

    _, labels = kmeans2(
        data=embedding,
        k=n_clusters,
        iter=300,
        thresh=1e-9,
        minit='++',
        seed=rng
    )

    return labels
