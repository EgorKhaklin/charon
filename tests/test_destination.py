import numpy as np
import pytest

from charon.destination import descend, destination, hadamard, log_wormhole, potential


def small(seed=0, n=10, d=30, k=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[rng.choice(d, k, replace=False)] = rng.uniform(1, 2, k)
    return X, X @ w


METRICS = [hadamard(1e-2), hadamard(0.3), log_wormhole(1e-2, 3e-2)]


def test_flat_metric_destination_is_min_norm():
    X, y = small()
    w, res = destination(X, y, lambda a: np.ones_like(a))
    assert res < 1e-10
    assert w == pytest.approx(np.linalg.pinv(X) @ y, abs=1e-8)


@pytest.mark.parametrize("m", METRICS)
def test_destination_meets_theorem_1(m):
    """It fits, and phi'(w) lies in the row space of X."""
    X, y = small(1)
    w, res = destination(X, y, m)
    assert res < 1e-10
    phi_prime, _ = potential(m)
    g = phi_prime(w)
    coef, *_ = np.linalg.lstsq(X.T, g, rcond=None)
    assert np.linalg.norm(X.T @ coef - g) < 1e-6 * np.linalg.norm(g)


@pytest.mark.parametrize("m", METRICS)
def test_speed_does_not_change_the_destination(m):
    X, y = small(2)
    w1, _ = destination(X, y, m)
    w2, _ = destination(X, y, lambda a: 0.1 * m(a))
    assert w2 == pytest.approx(w1, abs=1e-6)


def test_gradient_descent_lands_on_the_prediction_with_step_size_error():
    X, y = small(3)
    m = hadamard(0.1)
    w_star, _ = destination(X, y, m)
    gap = [np.linalg.norm(descend(X, y, m, steps, lr) - w_star)
           for lr, steps in ((0.2, 4000), (0.02, 40000))]
    assert gap[1] < 0.2 * gap[0]  # first order in the step size
    assert gap[1] < 1e-2 * np.linalg.norm(w_star)
