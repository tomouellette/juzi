# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import numpy as np
import pandas as pd

from anndata import AnnData
from typing import List


def score_aggregate(
    adata: AnnData,
    key: str,
    obs_cols: List[str] | None = None,
    agg: str = "mean",
    min_cells: int = 10,
    copy: bool = False,
) -> AnnData | None:
    """Aggregate per-cell program scores to per-donor pseudobulk scores.

    Takes per-cell program scores from obsm["juzi_program_scores"] and
    aggregates within each donor to produce a per-donor score matrix.
    Donor-level covariates from adata.obs can be propagated alongside
    scores for direct use in juzi.gp.associate.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_program_scores in .obsm, produced by
        juzi.gp.score.
    key : str
        Column in adata.obs denoting donor identity. Must match the key
        used in juzi.gp.nmf.
    obs_cols : List[str] | None
        Columns in adata.obs to propagate as donor-level covariates.
        For each column, the first observed value per donor is taken —
        these should be donor-level constants (e.g. age, BRCA status,
        study ID). If None, no covariates are propagated.
    agg : str
        Aggregation function applied to per-cell scores within each donor.
        "mean" is standard for program activity. "median" is more robust
        to outlier cells. Must be "mean" or "median".
    min_cells : int
        Minimum number of cells required for a donor to be included in
        the aggregate. Donors below this threshold are excluded.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following field populated:
            .uns["juzi_aggregate_scores"] : DataFrame (n_donors × n_programs)
                with optional covariate columns appended. Donors with fewer
                than min_cells cells are excluded.
    """
    adata = adata.copy() if copy else adata

    # Validate

    if "juzi_program_scores" not in adata.obsm:
        raise KeyError(
            "'juzi_program_scores' not found in .obsm. " "Run juzi.gp.score first."
        )

    if key not in adata.obs:
        raise KeyError(f"'{key}' not found in adata.obs. " "Check your key argument.")

    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'.")

    if min_cells < 1:
        raise ValueError("min_cells must be >= 1.")

    if obs_cols is not None:
        missing = [c for c in obs_cols if c not in adata.obs]
        if missing:
            raise KeyError(
                f"The following obs_cols were not found in adata.obs: " f"{missing}."
            )

    # Setup

    scores = adata.obsm["juzi_program_scores"]  # (n_cells × n_programs)
    n_programs = scores.shape[1]
    program_cols = [f"P{p}" for p in range(n_programs)]

    donors = adata.obs[key].values
    unique_donors = np.unique(donors)

    # Filter by min_cells

    cell_counts = pd.Series(donors).value_counts()
    valid_donors = cell_counts[cell_counts >= min_cells].index.to_numpy()

    if len(valid_donors) == 0:
        raise ValueError(
            f"No donors passed min_cells={min_cells}. "
            "Lower min_cells or check your key argument."
        )

    n_excluded = len(unique_donors) - len(valid_donors)
    if n_excluded > 0:
        import warnings

        warnings.warn(
            f"{n_excluded} donor(s) excluded for having fewer than "
            f"min_cells={min_cells} cells.",
            UserWarning,
            stacklevel=2,
        )

    # Aggregate scores per donor

    agg_fn = np.mean if agg == "mean" else np.median

    rows = []
    for donor in valid_donors:
        donor_mask = donors == donor
        donor_scores = scores[donor_mask]  # (n_donor_cells × n_programs)
        agg_scores = agg_fn(donor_scores, axis=0)  # (n_programs,)
        rows.append(agg_scores)

    agg_matrix = np.vstack(rows)  # (n_donors × n_programs)

    agg_df = pd.DataFrame(
        agg_matrix,
        index=valid_donors,
        columns=program_cols,
    )
    agg_df.index.name = key

    # Propagate donor-level covariates

    if obs_cols is not None:
        obs_df = adata.obs[[key] + obs_cols].copy()

        # Take first value per donor — these should be donor-level constants
        covariate_df = (
            obs_df.groupby(key, sort=False, observed=True)[obs_cols]
            .first()
            .loc[valid_donors]
        )

        agg_df = pd.concat([agg_df, covariate_df], axis=1)

    adata.uns["juzi_aggregate_scores"] = agg_df

    return adata if copy else None
