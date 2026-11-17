"""Rule-based item classification: keep, review or discard.

Thresholds follow widely used CTT rules of thumb (Ebel & Frisbie; Haladyna)
and can be adjusted through :class:`Thresholds`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    p_too_hard: float = 0.20
    p_too_easy: float = 0.90
    r_discard: float = 0.10
    r_review: float = 0.20
    min_functional_distractors: int = 2


def classify(stats: pd.DataFrame, distractors: pd.DataFrame | None = None,
             t: Thresholds = Thresholds()) -> pd.DataFrame:
    """Return ``stats`` with two extra columns: ``decision`` and ``reasons``."""
    out = stats.copy()
    decisions, reasons = [], []
    for item, row in out.iterrows():
        why, level = [], 0  # 0 keep, 1 review, 2 discard
        r = row.get("r_pbis")
        if pd.notna(r) and r < 0:
            why.append("negative discrimination (check the key)")
            level = 2
        elif pd.notna(r) and r < t.r_discard:
            why.append(f"very low discrimination (r < {t.r_discard})")
            level = 2
        elif pd.notna(r) and r < t.r_review:
            why.append(f"low discrimination (r < {t.r_review})")
            level = max(level, 1)
        p = row.get("p")
        if p < t.p_too_hard:
            why.append(f"very hard (p < {t.p_too_hard})")
            level = max(level, 1)
        elif p > t.p_too_easy:
            why.append(f"very easy (p > {t.p_too_easy})")
            level = max(level, 1)
        if distractors is not None and item in distractors.index:
            drow = distractors.loc[item]
            if drow["positive_distractors"] > 0:
                why.append("distractor attracting high scorers")
                level = max(level, 1)
            if drow["functional_distractors"] < t.min_functional_distractors:
                why.append("few functional distractors")
                level = max(level, 1)
        decisions.append(["keep", "review", "discard"][level])
        reasons.append("; ".join(why))
    out["decision"] = decisions
    out["reasons"] = reasons
    return out
