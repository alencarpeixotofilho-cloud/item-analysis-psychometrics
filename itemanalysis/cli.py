"""Command-line interface.

Example
-------
    python -m itemanalysis --responses examples/responses.csv \
        --key examples/key.csv --irt 2PL --out report.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import ctt, distractors, flags, irt, report


def run(responses_path: str, key_path: str, out: str = "report.html",
        irt_model: str | None = "2PL", csv_out: str | None = None) -> pd.DataFrame:
    responses = pd.read_csv(responses_path, index_col=0, dtype=str)
    key_df = pd.read_csv(key_path, dtype=str)
    key = key_df.set_index(key_df.columns[0])[key_df.columns[1]]

    scored = ctt.score(responses, key)
    stats = ctt.item_statistics(scored)
    options = distractors.option_table(responses, key, scored)
    dsum = distractors.distractor_flags(options)
    stats = stats.join(dsum)
    stats = flags.classify(stats, dsum)

    irt_result = None
    if irt_model:
        irt_result = irt.fit(scored, model=irt_model)
        stats = stats.join(irt_result.items[["a", "b"]].add_prefix("irt_"))

    summary = ctt.test_summary(scored)
    Path(out).write_text(report.build_html(summary, stats, options, scored, irt_result),
                         encoding="utf-8")
    if csv_out:
        stats.to_csv(csv_out)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Psychometric item analysis (CTT + IRT).")
    ap.add_argument("--responses", required=True, help="CSV: examinee id + one column per item")
    ap.add_argument("--key", required=True, help="CSV: item, key")
    ap.add_argument("--out", default="report.html", help="HTML report path")
    ap.add_argument("--csv", default=None, help="optional CSV with item statistics")
    ap.add_argument("--irt", default="2PL", choices=["1PL", "2PL", "none"])
    a = ap.parse_args(argv)
    stats = run(a.responses, a.key, a.out, None if a.irt == "none" else a.irt, a.csv)
    counts = stats["decision"].value_counts().to_dict()
    print(f"Report written to {a.out} · decisions: {counts}")


if __name__ == "__main__":
    main()
