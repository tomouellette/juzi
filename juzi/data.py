import warnings
import numpy as np
import polars as pl
import scipy as sp

from typing import Union, List, Self
from juzi.types import Index, Indices


class CountData:
    """Store and access single-cell sequencing count data

    Parameters
    ----------
    barcodes : Union[List[str], np.ndarray, pl.DataFrame]
        A list or array of barcode identifiers or polars dataframe. If a polars
        dataframe is provided, then identifiers must be in column 'barcode'.
    features : Union[List[str], np.ndarray, pl.DataFrame]
        A list or array of feature identifiers or polars dataframe. If a polars
        dataframe is provided, then identifiers must be in column 'feature'.
    counts : Union[np.ndarray, sp.sparse.csr_matrix]
        A barcode (row) by feature (column) array or sparse count matrix

    Returns
    -------
    Self
        An initialized SingleCellRNA object

    Attributes
    ----------
    bc : pl.DataFrame
        A polars DataFrame storing barcode information
    ft : pl.DataFrame
        A polars DataFrame storing feature information
    counts : sp.sparse.csr_matrix
        A scipy sparse matrix storing count information
    """

    def __init__(
        self,
        barcodes: Union[List[str], np.ndarray, pl.DataFrame],
        features: Union[List[str], np.ndarray, pl.DataFrame],
        counts: Union[np.ndarray, sp.sparse.csr_matrix],
    ) -> Self:
        if not isinstance(barcodes, (np.ndarray, list, pl.DataFrame)):
            raise ValueError(
                "barcodes must be a np.ndarray, list, or polars.DataFrame.")

        if not isinstance(features, (np.ndarray, list, pl.DataFrame)):
            raise ValueError(
                "features must be a np.ndarray, list, or polars.DataFrame.")

        if isinstance(barcodes, (np.ndarray, list)):
            self.bc = pl.DataFrame({"barcode": barcodes})
        else:
            if "barcode" not in barcodes.columns:
                raise ValueError("'barcode' key not in barcodes columns")
            self.bc = barcodes

        if isinstance(features, (np.ndarray, list)):
            self.ft = pl.DataFrame({"gene": features})
        else:
            if "gene" not in features.columns:
                raise ValueError("'gene' key not in features columns")
            self.ft = features

        if not isinstance(counts, (np.ndarray, sp.sparse.csr_matrix)):
            raise ValueError(
                "Argument counts must be a numpy array or sparse matrix.")

        h, w = counts.shape

        nb = self.bc.shape[0]
        nf = self.ft.shape[0]

        if nb != h and nb != w:
            raise ValueError(
                "Barcodes length does not equal row/column number in counts.")

        if nf != h and nf != w:
            raise ValueError(
                "Features length does not equal row/column number in counts.")

        self.counts = counts

        if nb == w and nf == h:
            self.counts = self.counts.T

        if isinstance(self.counts, sp.sparse.csr_matrix):
            self.counts = sp.sparse.csr_matrix(self.counts)

        if nb == nf:
            warnings.warn(
                "Number of barcodes and features are equal. Please ensure data"
                " is in cell (row) by gene (column) orientation."
            )

    def __repr__(self) -> str:
        """Display the current barcode and feature count"""
        return (
            f"{self.__class__.__name__}("
            f"n_barcodes={self.bc.shape[0]}, "
            f"n_features={self.ft.shape[0]}, "
            ")"
        )

    def __getitem__(self, i: Union[Index, Indices]) -> Self:
        """Subset the SingleCellRNA object

        Parameters
        ----------
        i : Union[Index, Indices]
            An integer or boolean index or indices to subset

        Returns
        -------
        Self
            A subsetted copy of the SingleCellRNA object
        """
        if isinstance(i, tuple):
            bc_idx, ft_idx = i
        else:
            bc_idx = i
            ft_idx = slice(None)

        return SingleCellRNA(
            barcodes=self.bc.iloc[bc_idx],
            features=self.ft.iloc[ft_idx],
            counts=self.counts[bc_idx, :][:, ft_idx]
        )
