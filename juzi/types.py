import numpy as np
import polars as pl

from typing import Union, List, Tuple, TypeAlias

Index: TypeAlias = Union[int, slice, np.ndarray, List[int], pl.Series]
Indices: TypeAlias = Tuple[Index, Index]
