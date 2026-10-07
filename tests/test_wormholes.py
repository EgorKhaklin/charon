import numpy as np
import pytest

from charon.wormholes import (SIGNED_SQUARE, SINH, balanced, compose, fit_transform, nest,
                              sparse_problem, tanh_cap)

NESTS = [nest(SIGNED_SQUARE, 2), nest(SIGNED_SQUARE, 3), compose(tanh_cap(3.0), SIGNED_SQUARE),
         compose(SINH, SIGNED_SQUARE), compose(SIGNED_SQUARE, SINH)]


@pytest.mark.parametrize("tf", NESTS, ids=lambda t: t.name)
def test_nested_derivative_is_the_chain_rule(tf):
    w, h = np.array([-1.1, -0.3, 0.2, 0.9]), 1e-6
    fd = (tf.T(w + h) - tf.T(w - h)) / (2 * h)
    assert tf.dT(w) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("tf", NESTS, ids=lambda t: t.name)
def test_nested_inverse(tf):
    for v in (-0.7, 0.05, 1.3):
        assert tf.T(np.array(tf.inverse(v))) == pytest.approx(v, rel=1e-9)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_nesting_signed_square_is_the_signed_power(depth):
    w = np.linspace(-1.5, 1.5, 13)
    p = 2**depth
    assert nest(SIGNED_SQUARE, depth).T(w) == pytest.approx(np.sign(w) * np.abs(w) ** p)


def test_jump_keeps_the_function_and_sets_the_conserved_quantity():
    w = np.array([-2.0, -1e-3, 0.0, 0.4, 3.0])
    for beta in (1e-1, 1e-4):
        u, v = balanced(w, beta)
        assert u * u - v * v == pytest.approx(w, abs=1e-12)
        assert u * v == pytest.approx(np.full_like(w, beta), rel=1e-6)


def test_identity_fit_is_min_norm():
    """With no wormhole, GD from (near) zero lands on the minimum-norm interpolant."""
    from charon.transforms import IDENTITY
    X, y, _ = sparse_problem(0)
    w = fit_transform(IDENTITY, X, y, lr=1.0, init=0.0, steps=3000)
    assert w == pytest.approx(np.linalg.pinv(X) @ y, abs=1e-8)
