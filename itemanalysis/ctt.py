"""Classical Test Theory (CTT) item and test statistics.

All functions work on a *scored* matrix: rows are examinees, columns are items,
values are 1 (correct), 0 (incorrect). Omitted responses should be scored as 0
before calling these functions (see :func:`score`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score(responses: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    """Score raw option responses against an answer key.

    Parameters
    ----------
    responses : DataFrame
        Examinees x items, values are the chosen option (e.g. ``"A"``..``"E"``).
        Missing values are treated as omitted (scored 0).
    key : Series
        Indexed by item id, values are the correct option.

    Returns
    -------
    DataFrame of 0/1 integers with the same shape as ``responses``.
    """
    missing = set(responses.columns) - set(key.index)
    if missing:
        raise ValueError(f"Items without an answer key: {sorted(missing)[:5]}")
    key = key.reindex(responses.columns)
    scored = responses.eq(key, axis=1)
    return scored.astype(int)


def difficulty(scored: pd.DataFrame) -> pd.Series:
    """Proportion correct (p-value). Higher means *easier*."""
    return scored.mean(axis=0).rename("p")


def item_rest_correlation(scored: pd.DataFrame) -> pd.Series:
    """Corrected point-biserial: correlation between an item and the total score
    computed *without* that item (avoids the part-whole inflation)."""
    total = scored.sum(axis=1)
    out = {}
    for item in scored.columns:
        x = scored[item]
        rest = total - x
        if x.std(ddof=0) == 0 or rest.std(ddof=0) == 0:
            out[item] = np.nan
        else:
            out[item] = float(np.corrcoef(x, rest)[0, 1])
    return pd.Series(out, name="r_pbis")


def discrimination_index(scored: pd.DataFrame, fraction: float = 0.27) -> pd.Series:
    """Upper-lower discrimination index D = p(upper group) - p(lower group).

    Groups are the top and bottom ``fraction`` of examinees by total score
    (Kelley's classic 27% rule).
    """
    if not 0 < fraction <= 0.5:
        raise ValueError("fraction must be in (0, 0.5]")
    total = scored.sum(axis=1)
    n = max(1, int(round(len(total) * fraction)))
    order = total.sort_values(kind="mergesort").index
    lower, upper = order[:n], order[-n:]
    d = scored.loc[upper].mean() - scored.loc[lower].mean()
    return d.rename("D")


def cronbach_alpha(scored: pd.DataFrame) -> float:
    """Cronbach's alpha. For dichotomous items it equals KR-20."""
    k = scored.shape[1]
    if k < 2:
        return float("nan")
    item_var = scored.var(axis=0, ddof=1).sum()
    total_var = scored.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float(k / (k - 1) * (1 - item_var / total_var))


kr20 = cronbach_alpha


def alpha_if_deleted(scored: pd.DataFrame) -> pd.Series:
    """Test reliability recomputed after removing each item in turn."""
    return pd.Series(
        {c: cronbach_alpha(scored.drop(columns=c)) for c in scored.columns},
        name="alpha_if_deleted",
    )


def standard_error_of_measurement(scored: pd.DataFrame) -> float:
    """SEM = SD(total) * sqrt(1 - alpha), in raw-score points."""
    sd = scored.sum(axis=1).std(ddof=1)
    return float(sd * np.sqrt(max(0.0, 1 - cronbach_alpha(scored))))


def item_statistics(scored: pd.DataFrame) -> pd.DataFrame:
    """All CTT item indicators in one table."""
    return pd.concat(
        [
            difficulty(scored),
            discrimination_index(scored),
            item_rest_correlation(scored),
            alpha_if_deleted(scored),
        ],
        axis=1,
    )


def test_summary(scored: pd.DataFrame) -> dict:
    """Headline numbers for the whole test."""
    total = scored.sum(axis=1)
    return {
        "examinees": int(scored.shape[0]),
        "items": int(scored.shape[1]),
        "mean_score": float(total.mean()),
        "sd_score": float(total.std(ddof=1)),
        "mean_percent": float(100 * total.mean() / scored.shape[1]),
        "alpha": cronbach_alpha(scored),
        "sem": standard_error_of_measurement(scored),
        "mean_p": float(difficulty(scored).mean()),
    }
