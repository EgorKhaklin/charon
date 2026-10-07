"""Pandora: open many small vessels, gather what comes out, and let the data certify it.

THEORY.md's ceiling says that no fixed elementwise wormhole beats L1 uniformly. E16
showed that past L1, the obstacle is the ranking: the true support is not among the
top coordinates of any single solve. Pandora builds the ranking from many small
solves instead of one big one:

1. score every column by |w| from basis pursuit on all of them;
2. repeat: a vessel is the `top` best-scored columns plus `rest` random others;
   basis pursuit restricted to the vessel (fewer unknowns, an easier problem);
   score <- decay * score + |w_vessel|;
3. certify: least squares on the n - 1 best-scored columns. If it fits y exactly,
   the true support lies inside them (generically, THEORY.md) and least squares
   returns the true w. Every few rounds, stop as soon as a certificate appears.
"""

import numpy as np
from scipy.optimize import linprog


def basis_pursuit(X, y, cols=None, weights=None):
    """min sum weights_i |w_i| subject to X w = y, using only `cols` (default all).

    Returns (w over all d columns, dual multipliers of X w = y)."""
    d = X.shape[1]
    cols = np.arange(d) if cols is None else np.asarray(cols)
    Xc = X[:, cols]
    c = np.ones(len(cols)) if weights is None else np.asarray(weights)[cols]
    r = linprog(np.concatenate([c, c]), A_eq=np.hstack([Xc, -Xc]), b_eq=y, bounds=(0, None),
                method="highs")
    w = np.zeros(d)
    if r.x is not None:
        w[cols] = r.x[:len(cols)] - r.x[len(cols):]
    return w, (r.eqlin.marginals if r.eqlin is not None else np.zeros(len(y)))


def certify(X, y, score, tol=1e-8):
    """Least squares on the n - 1 best-scored columns; the answer if the fit is exact, else None."""
    n = X.shape[0]
    C = np.argsort(-score)[:n - 1]
    c, *_ = np.linalg.lstsq(X[:, C], y, rcond=None)
    if np.linalg.norm(X[:, C] @ c - y) > tol * np.linalg.norm(y):
        return None
    w = np.zeros(X.shape[1])
    w[C] = c
    w[np.abs(w) < 1e-9] = 0.0
    return w


def pandora(X, y, rng, rounds=200, top=20, rest=40, decay=0.7, check_every=5):
    """Returns (certified w or None, rounds used)."""
    score = np.abs(basis_pursuit(X, y)[0])
    others = np.arange(X.shape[1])
    for r in range(1, rounds + 1):
        T = np.argsort(-score)[:top]
        R = rng.choice(np.setdiff1d(others, T), rest, replace=False)
        score = decay * score + np.abs(basis_pursuit(X, y, np.concatenate([T, R]))[0])
        if r % check_every == 0:
            w = certify(X, y, score)
            if w is not None:
                return w, r
    return certify(X, y, score), rounds


def reweighted_l1(X, y, iters=5, eps=0.1):
    """Candes, Wakin and Boyd (2008)."""
    w = basis_pursuit(X, y)[0]
    for _ in range(iters - 1):
        w = basis_pursuit(X, y, weights=1 / (np.abs(w) + eps))[0]
    return w


def iterative_support_detection(X, y, iters=8, beta=2.0):
    """Wang and Yin (2010): basis pursuit with the detected support left unpenalized,
    support = {|w_i| > max|w| / beta^(t+1)} from the previous solution."""
    w = basis_pursuit(X, y)[0]
    for t in range(iters):
        detected = np.abs(w) > np.abs(w).max() / beta ** (t + 1)
        w = basis_pursuit(X, y, weights=np.where(detected, 0.0, 1.0))[0]
    return w
