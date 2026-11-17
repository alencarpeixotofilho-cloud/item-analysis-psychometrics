"""Generate a synthetic multiple-choice exam (no real data involved).

Abilities are drawn from N(0, 1) and responses follow a 2PL model. Wrong
answers are spread over the distractors with ability-dependent attractiveness.
A few items are deliberately flawed so the analysis has something to find:

* one miskeyed item (the "key" is actually a distractor)
* one item with an implausible distractor nobody picks
* one item that is far too easy and one that is far too hard
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OPTIONS = list("ABCDE")


def simulate(n_examinees: int = 400, n_items: int = 60, seed: int = 2026):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_examinees)
    a = rng.lognormal(0.0, 0.3, n_items)
    b = rng.normal(0, 1, n_items)
    items = [f"Q{i:03d}" for i in range(1, n_items + 1)]

    # deliberate flaws
    b[2], b[5] = -3.5, 3.0              # far too easy / far too hard
    a[8] = 0.15                          # barely discriminating

    key = rng.choice(OPTIONS, n_items)
    p = 1 / (1 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    correct = rng.random((n_examinees, n_items)) < p

    responses = np.empty((n_examinees, n_items), dtype=object)
    for j in range(n_items):
        distractors = [o for o in OPTIONS if o != key[j]]
        base = rng.dirichlet(np.ones(4) * 2)
        if j == 11:                       # one implausible distractor
            base[0] = 0.002
            base /= base.sum()
        slope = 0.5 + rng.normal(0, 0.1, 4)  # weaker examinees favour distractors
        for i in range(n_examinees):
            if correct[i, j]:
                responses[i, j] = key[j]
            else:
                w = base * np.exp(-slope * theta[i])
                responses[i, j] = rng.choice(distractors, p=w / w.sum())
    omit = rng.random((n_examinees, n_items)) < 0.01
    responses[omit] = None

    published_key = key.copy()
    # miskey item 15: publish a distractor as the key
    published_key[14] = [o for o in OPTIONS if o != key[14]][0]

    ids = [f"S{i:04d}" for i in range(1, n_examinees + 1)]
    resp_df = pd.DataFrame(responses, index=pd.Index(ids, name="examinee"), columns=items)
    key_df = pd.DataFrame({"item": items, "key": published_key})
    truth = pd.DataFrame({"item": items, "a": a, "b": b, "true_key": key})
    return resp_df, key_df, truth


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--examinees", type=int, default=400)
    ap.add_argument("--items", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()
    resp, key, truth = simulate(args.examinees, args.items, args.seed)
    out = Path(args.out)
    resp.to_csv(out / "responses.csv")
    key.to_csv(out / "key.csv", index=False)
    truth.to_csv(out / "true_parameters.csv", index=False)
    print(f"Wrote {len(resp)} examinees x {resp.shape[1]} items to {out}")


if __name__ == "__main__":
    main()
