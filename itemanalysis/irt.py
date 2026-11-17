"""Item Response Theory: Rasch (1PL) and 2PL models.

Parameters are estimated by **marginal maximum likelihood** with the
Bock–Aitkin EM algorithm on a fixed quadrature grid, assuming a standard
normal ability distribution. The M-step uses vectorized Newton–Raphson in the
slope–intercept parameterization ``logit P = a*theta + d`` and reports the
usual IRT difficulty ``b = -d / a``.

This is a compact, dependency-light implementation intended for teaching,
auditing and small/medium exams (hundreds to a few thousand examinees).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _quadrature(n_points: int = 41, bound: float = 4.0):
    nodes = np.linspace(-bound, bound, n_points)
    weights = np.exp(-0.5 * nodes**2)
    return nodes, weights / weights.sum()


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


@dataclass
class IRTResult:
    model: str
    items: pd.DataFrame                 # a, b (and d) per item
    log_likelihood: float
    iterations: int
    converged: bool
    nodes: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)

    def icc(self, theta: np.ndarray) -> pd.DataFrame:
        """Item characteristic curves evaluated at ``theta``."""
        a = self.items["a"].to_numpy()
        b = self.items["b"].to_numpy()
        p = _sigmoid(a[None, :] * (np.asarray(theta)[:, None] - b[None, :]))
        return pd.DataFrame(p, index=theta, columns=self.items.index)

    def item_information(self, theta: np.ndarray) -> pd.DataFrame:
        p = self.icc(theta)
        a = self.items["a"]
        return p * (1 - p) * a**2

    def test_information(self, theta: np.ndarray) -> pd.Series:
        return self.item_information(theta).sum(axis=1).rename("information")


def fit(scored: pd.DataFrame, model: str = "2PL", max_iter: int = 500,
        tol: float = 1e-4, n_quad: int = 41) -> IRTResult:
    """Fit a Rasch ("1PL") or "2PL" model to a 0/1 scored matrix."""
    model = model.upper()
    if model not in {"1PL", "2PL"}:
        raise ValueError("model must be '1PL' or '2PL'")
    X = scored.to_numpy(dtype=float)
    n_items = X.shape[1]
    nodes, prior = _quadrature(n_quad)

    p_obs = np.clip(X.mean(axis=0), 0.01, 0.99)
    a = np.ones(n_items)
    d = np.log(p_obs / (1 - p_obs))           # start from the observed logit

    prev_ll = -np.inf
    converged = False
    for it in range(1, max_iter + 1):
        # ---- E-step: posterior over quadrature nodes for each examinee ----
        P = _sigmoid(a[None, :] * nodes[:, None] + d[None, :])         # Q x I
        logP, log1mP = np.log(P), np.log1p(-P)
        loglik = X @ logP.T + (1 - X) @ log1mP.T                       # N x Q
        loglik += np.log(prior)[None, :]
        m = loglik.max(axis=1, keepdims=True)
        post = np.exp(loglik - m)
        marg = post.sum(axis=1, keepdims=True)
        post /= marg
        ll = float((np.log(marg) + m).sum())

        n_q = post.sum(axis=0)                                          # Q
        r_qi = post.T @ X                                               # Q x I

        # ---- M-step: Newton-Raphson per item (vectorized) ----
        for _ in range(5):
            P = _sigmoid(a[None, :] * nodes[:, None] + d[None, :])
            resid = r_qi - n_q[:, None] * P                             # Q x I
            w = n_q[:, None] * P * (1 - P)
            g_d = resid.sum(axis=0)
            h_dd = w.sum(axis=0)
            if model == "2PL":
                g_a = (resid * nodes[:, None]).sum(axis=0)
                h_aa = (w * nodes[:, None] ** 2).sum(axis=0)
                h_ad = (w * nodes[:, None]).sum(axis=0)
                det = h_aa * h_dd - h_ad**2
                det = np.where(np.abs(det) < 1e-10, 1e-10, det)
                step_a = (h_dd * g_a - h_ad * g_d) / det
                step_d = (h_aa * g_d - h_ad * g_a) / det
                a = np.clip(a + np.clip(step_a, -0.5, 0.5), 0.05, 6.0)
            else:
                step_d = g_d / np.maximum(h_dd, 1e-10)
            d = np.clip(d + np.clip(step_d, -1.0, 1.0), -15, 15)

        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    items = pd.DataFrame({"a": a, "d": d, "b": -d / a}, index=scored.columns)
    return IRTResult(model, items, ll, it, converged, nodes, prior)


def eap_scores(scored: pd.DataFrame, result: IRTResult) -> pd.DataFrame:
    """Expected-a-posteriori ability estimates and posterior SDs."""
    X = scored.to_numpy(dtype=float)
    a = result.items["a"].to_numpy()
    d = result.items["d"].to_numpy()
    nodes, prior = result.nodes, result.weights
    P = _sigmoid(a[None, :] * nodes[:, None] + d[None, :])
    loglik = X @ np.log(P).T + (1 - X) @ np.log1p(-P).T + np.log(prior)[None, :]
    post = np.exp(loglik - loglik.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)
    theta = post @ nodes
    sd = np.sqrt(post @ nodes**2 - theta**2)
    return pd.DataFrame({"theta": theta, "se": sd}, index=scored.index)
