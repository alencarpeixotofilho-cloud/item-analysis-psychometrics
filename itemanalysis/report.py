"""Self-contained HTML report (figures embedded as base64 PNG)."""
from __future__ import annotations

import base64
import html
import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"keep": "#2da44e", "review": "#d29922", "discard": "#cf222e"}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_difficulty_discrimination(stats: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for decision, grp in stats.groupby("decision"):
        ax.scatter(grp["p"], grp["r_pbis"], s=36, alpha=0.85,
                   color=COLORS.get(decision, "grey"), label=decision)
    ax.axhline(0.20, ls="--", lw=0.8, color="grey")
    ax.axhline(0.10, ls=":", lw=0.8, color="grey")
    ax.axvspan(0, 0.20, color="grey", alpha=0.06)
    ax.axvspan(0.90, 1, color="grey", alpha=0.06)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Difficulty (proportion correct)")
    ax.set_ylabel("Discrimination (item-rest r)")
    ax.set_title("Item map: difficulty × discrimination")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_score_distribution(scored: pd.DataFrame):
    total = scored.sum(axis=1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(total, bins=min(30, scored.shape[1]), color="#1f6feb", alpha=0.85)
    ax.axvline(total.mean(), color="black", lw=1)
    ax.set_xlabel("Total score")
    ax.set_ylabel("Examinees")
    ax.set_title("Score distribution")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_option_curves(options: pd.DataFrame, item: str):
    sub = options[options["item"] == item]
    gcols = [c for c in sub.columns if c.startswith("prop_g")]
    x = np.arange(1, len(gcols) + 1)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for _, row in sub.iterrows():
        ax.plot(x, row[gcols].to_numpy(dtype=float), marker="o",
                lw=2.4 if row["is_key"] else 1.2,
                color="#1f6feb" if row["is_key"] else None,
                label=f"{row['option']}{' (key)' if row['is_key'] else ''}")
    ax.set_xticks(x, ["low", "mid", "high"][: len(x)] if len(x) == 3 else x)
    ax.set_ylim(0, 1)
    ax.set_title(item)
    ax.set_xlabel("Ability group")
    ax.set_ylabel("Proportion choosing")
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_test_information(irt_result):
    theta = np.linspace(-4, 4, 161)
    info = irt_result.test_information(theta)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(theta, info.to_numpy(), color="#8250df", lw=2)
    ax.set_xlabel("Ability (θ)")
    ax.set_ylabel("Information")
    ax.set_title(f"Test information function ({irt_result.model})")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def _table(df: pd.DataFrame) -> str:
    return df.to_html(classes="tbl", float_format=lambda v: f"{v:.3f}", border=0)


CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1f2328}
h1{margin-bottom:.2rem} .muted{color:#656d76}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;margin:1.2rem 0}
.card{border:1px solid #d0d7de;border-radius:8px;padding:.8rem}
.card b{display:block;font-size:1.4rem}
.tbl{border-collapse:collapse;font-size:.85rem;width:100%}
.tbl th,.tbl td{padding:.3rem .5rem;border-bottom:1px solid #eaeef2;text-align:right}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}
img{max-width:100%}
.keep{color:#2da44e}.review{color:#d29922}.discard{color:#cf222e}
"""


def build_html(summary: dict, stats: pd.DataFrame, options: pd.DataFrame,
               scored: pd.DataFrame, irt_result=None, title: str = "Item Analysis Report") -> str:
    counts = stats["decision"].value_counts()
    cards = [
        ("Examinees", f"{summary['examinees']}"),
        ("Items", f"{summary['items']}"),
        ("Mean score", f"{summary['mean_percent']:.1f}%"),
        ("Reliability (α)", f"{summary['alpha']:.3f}"),
        ("SEM (points)", f"{summary['sem']:.2f}"),
        ("Keep / Review / Discard",
         f"{counts.get('keep', 0)} / {counts.get('review', 0)} / {counts.get('discard', 0)}"),
    ]
    cards_html = "".join(f'<div class="card"><span class="muted">{k}</span><b>{v}</b></div>' for k, v in cards)

    flagged = stats[stats["decision"] != "keep"].sort_values("r_pbis")
    curves = "".join(
        f'<img src="data:image/png;base64,{_png(fig_option_curves(options, it))}" alt="{html.escape(it)}">'
        for it in flagged.index[:12]
    )
    irt_html = ""
    if irt_result is not None:
        irt_html = (
            f"<h2>Item Response Theory ({irt_result.model})</h2>"
            f"<p class='muted'>Marginal ML (EM), {irt_result.iterations} iterations, "
            f"converged: {irt_result.converged}, log-likelihood {irt_result.log_likelihood:.1f}.</p>"
            f'<img src="data:image/png;base64,{_png(fig_test_information(irt_result))}">'
            + _table(irt_result.items[["a", "b"]].round(3))
        )

    view = stats.copy()
    view["decision"] = [f'<span class="{d}">{d}</span>' for d in view["decision"]]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body>
<h1>{html.escape(title)}</h1><p class="muted">Generated on {date.today().isoformat()} · item-analysis-psychometrics</p>
<div class="cards">{cards_html}</div>
<div class="grid">
<img src="data:image/png;base64,{_png(fig_score_distribution(scored))}">
<img src="data:image/png;base64,{_png(fig_difficulty_discrimination(stats))}">
</div>
<h2>Items flagged for review</h2>
<div class="grid">{curves or '<p>No items flagged.</p>'}</div>
<h2>Item statistics</h2>
{view.to_html(classes="tbl", float_format=lambda v: f"{v:.3f}", border=0, escape=False)}
{irt_html}
</body></html>"""
