import numpy as np
import pytest

from charon.pandora import basis_pursuit, certify, iterative_support_detection, pandora
from charon.wormholes import sparse_problem


def test_certificate_accepts_a_superset_and_returns_the_truth():
    X, y, wt = sparse_problem(0, k=11)
    score = np.abs(wt) + 1e-3 * np.random.default_rng(0).random(200)  # support first, then 28 others
    w = certify(X, y, score)
    assert w is not None and w == pytest.approx(wt, abs=1e-8)


def test_certificate_refuses_when_one_true_column_is_missing():
    X, y, wt = sparse_problem(0, k=11)
    score = np.abs(wt) + 1e-3 * np.random.default_rng(0).random(200)
    score[np.flatnonzero(wt)[0]] = -1.0  # push one true column out of the top 39
    assert certify(X, y, score) is None


def test_restricted_basis_pursuit_stays_on_its_columns():
    X, y, _ = sparse_problem(1, k=5)
    cols = np.arange(100)
    w, _ = basis_pursuit(X, y, cols)
    assert np.all(w[100:] == 0) and np.linalg.norm(X @ w - y) < 1e-8 * np.linalg.norm(y)


def test_pandora_and_isd_recover_an_easy_problem():
    X, y, wt = sparse_problem(2, k=5)
    w, rounds = pandora(X, y, np.random.default_rng(2), rounds=20)
    assert w is not None and w == pytest.approx(wt, abs=1e-6) and rounds <= 20
    assert iterative_support_detection(X, y) == pytest.approx(wt, abs=1e-6)
