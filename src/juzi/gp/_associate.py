# Copyright (c) 2025, Tom Ouellette
# Licensed under the BSD 3-Clause License

import re
import warnings
import numpy as np
import pandas as pd

from anndata import AnnData
from statsmodels.stats.multitest import multipletests

try:
    import statsmodels.formula.api as smf
except ImportError:
    raise ImportError(
        "statsmodels is required for juzi.gp.associate. "
        "Install it with: pip install statsmodels"
    )


def associate(
    adata: AnnData,
    formula: str,
    method: str = "lbfgs",
    reml: bool = True,
    padj_method: str = "fdr_bh",
    silent: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """Test association between consensus program scores and covariates.

    For each consensus program, fits a linear mixed model of the form:

        program_score ~ formula

    where random effect terms of the form (1|group) are parsed from the
    formula string and passed as the grouping variable to statsmodels
    MixedLM. Multiple random effect terms are combined into an interaction
    grouping variable. Program scores are standardised before fitting so
    beta coefficients are comparable across programs.

    Parameters
    ----------
    adata : AnnData
        AnnData object with juzi_aggregate_scores in .uns, produced by
        juzi.gp.aggregate.
    formula : str
        R-style formula string for the right-hand side of the model.
        Fixed effects follow standard patsy syntax. Random intercepts
        are specified as (1|group). Examples:
            "age + donor_brca + (1|study_id)"
            "age + (1|study_id) + (1|batch)"
            "age"  (no random effects — falls back to OLS)
    method : str
        Optimisation method passed to MixedLM.fit(). "lbfgs" is stable
        for most cases. "nm" or "powell" can help if lbfgs fails to
        converge.
    reml : bool
        If True, use restricted maximum likelihood estimation. Recommended
        for random effect variance estimation.
    padj_method : str
        Multiple testing correction method passed to
        statsmodels.stats.multitest.multipletests. Default is "fdr_bh"
        (Benjamini-Hochberg FDR).
    silent : bool
        If True, suppress warnings from failed model fits.
    copy : bool
        If True, return a modified copy. If False, modify in place.

    Returns
    -------
    AnnData | None
        AnnData with the following field populated:
            .uns["juzi_association"] : DataFrame with one row per program
                containing beta, se, pval, padj for the first fixed effect
                covariate in the formula, plus model metadata.
    """
    adata = adata.copy() if copy else adata

    # Validate

    if "juzi_aggregate_scores" not in adata.uns:
        raise KeyError(
            "'juzi_aggregate_scores' not found in .uns. " "Run juzi.gp.aggregate first."
        )

    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a non-empty string.")

    if padj_method not in multipletests.__doc__:
        pass  # let multipletests raise its own error

    # Parse formula

    fixed_formula, group_vars = _parse_formula(formula)

    if not fixed_formula.strip():
        raise ValueError(
            "No fixed effects remain after parsing random effects. "
            "Provide at least one fixed effect covariate."
        )

    df = adata.uns["juzi_aggregate_scores"].copy()

    # Validate all referenced columns exist in aggregate scores
    all_vars = _extract_vars(fixed_formula) + group_vars
    missing = [v for v in all_vars if v not in df.columns]
    if missing:
        raise KeyError(
            f"The following variables are not in juzi_aggregate_scores: "
            f"{missing}. Check your formula and obs_cols in juzi.gp.aggregate."
        )

    # Build grouping variable
    # Combine multiple random effects into an interaction grouping variable

    use_lmm = len(group_vars) > 0
    groups = None

    if use_lmm:
        if len(group_vars) == 1:
            groups = df[group_vars[0]].astype(str)
        else:
            # Interaction of all grouping variables
            groups = df[group_vars[0]].astype(str)
            for gv in group_vars[1:]:
                groups = groups + "_" + df[gv].astype(str)
            groups.name = "_x_".join(group_vars)

        n_groups = groups.nunique()
        if n_groups < 2:
            warnings.warn(
                f"Only {n_groups} unique group level(s) found for random "
                "effects. Falling back to OLS.",
                UserWarning,
                stacklevel=2,
            )
            use_lmm = False

    # Identify program columns

    program_cols = [c for c in df.columns if re.match(r"^P\d+$", c)]

    if len(program_cols) == 0:
        raise ValueError(
            "No program columns (P0, P1, ...) found in juzi_aggregate_scores."
        )

    # Identify primary covariate for result extraction
    # First term in fixed formula after stripping interactions/transformations

    primary_covariate = _extract_vars(fixed_formula)[0]

    # Fit model per program

    results = []

    for prog in program_cols:
        prog_std = df[prog].std()

        if prog_std == 0 or np.isnan(prog_std):
            if not silent:
                warnings.warn(
                    f"Program {prog} has zero variance — skipping.",
                    UserWarning,
                    stacklevel=2,
                )
            continue

        df[f"{prog}_z"] = (df[prog] - df[prog].mean()) / prog_std
        full_formula = f"{prog}_z ~ {fixed_formula}"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if use_lmm:
                    try:
                        fit = smf.mixedlm(
                            formula=full_formula,
                            data=df,
                            groups=groups,
                        ).fit(reml=reml, method=method)
                    except (np.linalg.LinAlgError, Exception) as lmm_error:
                        if "singular" in str(lmm_error).lower():
                            # Fall back to OLS when LMM covariance is singular
                            if not silent:
                                warnings.warn(
                                    f"LMM singular for {prog}, falling back to OLS.",
                                    UserWarning,
                                    stacklevel=2,
                                )
                            fit = smf.ols(
                                formula=full_formula,
                                data=df,
                            ).fit()
                        else:
                            raise
                else:
                    fit = smf.ols(
                        formula=full_formula,
                        data=df,
                    ).fit()

            # Extract primary covariate result
            beta = fit.params.get(primary_covariate, np.nan)
            se = fit.bse.get(primary_covariate, np.nan)
            pval = fit.pvalues.get(primary_covariate, np.nan)

            results.append(
                {
                    "program": prog,
                    "covariate": primary_covariate,
                    "beta": beta,
                    "se": se,
                    "pval": pval,
                    "n_obs": len(df),
                    "model": "lmm" if use_lmm else "ols",
                    "groups": "_x_".join(group_vars) if group_vars else None,
                }
            )

        except Exception as e:
            if not silent:
                warnings.warn(
                    f"Model failed for {prog}: {e}",
                    UserWarning,
                    stacklevel=2,
                )

    if len(results) == 0:
        raise ValueError(
            "No models were successfully fitted. " "Check your formula and data."
        )

    # FDR correction

    assoc_df = pd.DataFrame(results)

    _, assoc_df["padj"], _, _ = multipletests(
        assoc_df["pval"].fillna(1.0),
        method=padj_method,
    )

    assoc_df = assoc_df.sort_values("padj").reset_index(drop=True)

    adata.uns["juzi_association"] = assoc_df

    return adata if copy else None


def _parse_formula(formula: str) -> tuple[str, list[str]]:
    """Extract random effect terms from an R-style formula string.

    Parses (1|group) terms, removes them from the formula, and returns
    the cleaned fixed-effects formula alongside the group variable names.

    Parameters
    ----------
    formula : str
        R-style formula string, e.g. "age + donor_brca + (1|study_id)".

    Returns
    -------
    tuple[str, list[str]]
        Cleaned fixed-effects formula string and list of group variable names.
    """
    re_term = re.compile(r"\(\s*1\s*\|\s*(\w+)\s*\)")
    groups = re_term.findall(formula)
    cleaned = re_term.sub("", formula)

    # Remove consecutive or leading/trailing + operators left by substitution
    cleaned = re.sub(r"\+\s*\+", "+", cleaned)
    cleaned = re.sub(r"^\s*\+|\+\s*$", "", cleaned)
    cleaned = cleaned.strip()

    return cleaned, groups


def _extract_vars(formula: str) -> list[str]:
    """Extract bare variable names from a patsy formula string.

    Strips transformations (C(), np.log(), etc.) and interaction operators
    to return the underlying column names referenced in the formula.

    Parameters
    ----------
    formula : str
        Patsy formula right-hand side string.

    Returns
    -------
    list[str]
        Variable names in order of appearance, deduplicated.
    """
    # Remove known function wrappers — C(), np.fn(), Q()
    cleaned = re.sub(r"\bC\s*\(([^)]+)\)", r"\1", formula)
    cleaned = re.sub(r"\bnp\.\w+\s*\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\bQ\s*\([^)]+\)", "", cleaned)

    # Split on operators and extract word tokens
    tokens = re.split(r"[\+\*\:\|~\s]+", cleaned)
    seen, vars_ = set(), []
    for t in tokens:
        t = t.strip()
        if t and re.match(r"^[A-Za-z_]\w*$", t) and t not in seen:
            seen.add(t)
            vars_.append(t)

    return vars_
