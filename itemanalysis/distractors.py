"""Distractor (option) analysis for multiple-choice items."""
from __future__ import annotations

import numpy as np
import pandas as pd

OMIT = "(omit)"


def ability_groups(total: pd.Series, n_groups: int = 3) -> pd.Series:
    """Split examinees into ``n_groups`` ordered groups by total score
    (1 = lowest). Ties are broken by rank so groups have similar sizes."""
    ranks = total.rank(method="first")
    return pd.qcut(ranks, n_groups, labels=range(1, n_groups + 1)).astype(int)


def option_table(responses: pd.DataFrame, key: pd.Series, scored: pd.DataFrame,
                 n_groups: int = 3) -> pd.DataFrame:
    """Long table with one row per (item, option).

    Columns
    -------
    item, option, is_key, prop          overall proportion choosing the option
    prop_g1..prop_gK                    proportion within each ability group
    r_option                            point-biserial between choosing the option
                                        and the total score (excluding the item)
    """
    total = scored.sum(axis=1)
    groups = ability_groups(total, n_groups)
    rows = []
    for item in responses.columns:
        choice = responses[item].fillna(OMIT)
        rest = total - scored[item]
        options = sorted(set(choice.unique()) | {key[item]})
        for opt in options:
            chose = (choice == opt).astype(int)
            row = {
                "item": item,
                "option": opt,
                "is_key": opt == key[item],
                "prop": float(chose.mean()),
            }
            for g in range(1, n_groups + 1):
                mask = groups == g
                row[f"prop_g{g}"] = float(chose[mask].mean()) if mask.any() else np.nan
            if chose.std(ddof=0) > 0 and rest.std(ddof=0) > 0:
                row["r_option"] = float(np.corrcoef(chose, rest)[0, 1])
            else:
                row["r_option"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def distractor_flags(options: pd.DataFrame, min_prop: float = 0.05,
                     positive_r: float = 0.05) -> pd.DataFrame:
    """Summarize distractor quality per item.

    * non-functional distractor: chosen by fewer than ``min_prop`` of examinees
    * positive distractor: a wrong option whose option-rest correlation exceeds
      ``positive_r`` (stronger examinees are attracted to it: possible key
      error or ambiguity)
    """
    d = options[(~options["is_key"]) & (options["option"] != OMIT)]
    summary = d.groupby("item").agg(
        n_distractors=("option", "size"),
        nonfunctional=("prop", lambda s: int((s < min_prop).sum())),
        positive_distractors=("r_option", lambda s: int((s > positive_r).sum())),
    )
    summary["functional_distractors"] = summary["n_distractors"] - summary["nonfunctional"]
    return summary
