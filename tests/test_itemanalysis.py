import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from itemanalysis import ctt, distractors, flags, irt  # noqa: E402
from generate_sample_data import simulate  # noqa: E402


@pytest.fixture(scope="module")
def data():
    resp, key_df, truth = simulate(n_examinees=600, n_items=40, seed=7)
    key = key_df.set_index("item")["key"]
    return resp, key, truth, ctt.score(resp, key)


def test_score_is_binary(data):
    _, _, _, scored = data
    assert set(np.unique(scored.to_numpy())) <= {0, 1}


def test_score_rejects_missing_key():
    resp = pd.DataFrame({"Q1": ["A"], "Q2": ["B"]})
    with pytest.raises(ValueError):
        ctt.score(resp, pd.Series({"Q1": "A"}))


def test_alpha_known_value():
    # perfectly consistent Guttman pattern -> high alpha
    m = pd.DataFrame(np.tril(np.ones((6, 5)), -1)[:, :5].astype(int))
    assert ctt.cronbach_alpha(m) > 0.8


def test_alpha_reasonable(data):
    *_, scored = data
    assert 0.6 < ctt.cronbach_alpha(scored) < 0.99


def test_difficulty_matches_mean(data):
    *_, scored = data
    assert np.allclose(ctt.difficulty(scored), scored.mean())


def test_miskeyed_item_is_flagged(data):
    resp, key, _, scored = data
    stats = ctt.item_statistics(scored)
    opts = distractors.option_table(resp, key, scored)
    dsum = distractors.distractor_flags(opts)
    out = flags.classify(stats.join(dsum), dsum)
    assert out.loc["Q015", "decision"] == "discard"
    assert out.loc["Q015", "r_pbis"] < 0


def test_option_props_sum_to_one(data):
    resp, key, _, scored = data
    opts = distractors.option_table(resp, key, scored)
    sums = opts.groupby("item")["prop"].sum()
    assert np.allclose(sums, 1.0)


def test_2pl_recovers_parameters(data):
    _, _, truth, scored = data
    ok = truth["item"] != "Q015"          # skip the miskeyed item
    res = irt.fit(scored.loc[:, ok.values], model="2PL")
    t = truth[ok].set_index("item")
    assert res.converged
    assert np.corrcoef(res.items["b"], t["b"])[0, 1] > 0.9
    assert np.corrcoef(res.items["a"], t["a"])[0, 1] > 0.6


def test_rasch_has_unit_slopes(data):
    *_, scored = data
    res = irt.fit(scored, model="1PL")
    assert np.allclose(res.items["a"], 1.0)


def test_eap_correlates_with_total(data):
    *_, scored = data
    res = irt.fit(scored, model="2PL")
    eap = irt.eap_scores(scored, res)
    assert np.corrcoef(eap["theta"], scored.sum(axis=1))[0, 1] > 0.9
