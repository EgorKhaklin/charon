import numpy as np
import pytest

from charon.core import curvature_limit, loss, make_data, ols, steps_to, train
from charon.transforms import DEFAULT, IDENTITY, SQUARE, mass_energy

x, y = make_data()
v_star, b_star, L_star = ols(x, y)


@pytest.mark.parametrize("tf", DEFAULT, ids=lambda t: t.name)
def test_gradient_matches_finite_difference(tf):
    """The analytic dL/dw (chain rule through T) equals a central difference."""
    for w in (-1.3, -0.4, 0.7, 1.9):
        run = train(x, y, tf, lr=0.0, steps=1, w0=w, b0=0.5)
        h = 1e-6
        fd = (loss(x, y, tf.T(np.array(w + h)), 0.5) - loss(x, y, tf.T(np.array(w - h)), 0.5)) / (2 * h)
        assert run.grad_w[0, 0] == pytest.approx(abs(fd), rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("tf", DEFAULT, ids=lambda t: t.name)
def test_one_step_is_preconditioned_gd_on_v(tf):
    """dv = -lr T'(w)^2 dL/dv + O(lr^2): the transform is a preconditioner."""
    w0, lr = 0.8, 1e-7
    run = train(x, y, tf, lr=lr, steps=2, w0=w0, b0=b_star, lr_b=0.0)
    g_v = np.mean((tf.T(np.array(w0)) * x + b_star - y) * x)
    predicted = -lr * tf.dT(np.array(w0)) ** 2 * g_v
    assert run.v[1, 0] - run.v[0, 0] == pytest.approx(predicted, rel=1e-4)


def test_identity_reaches_ols():
    run = train(x, y, IDENTITY, lr=0.02, steps=2000, w0=0.5, lr_b=1.0)
    assert run.v[-1, 0] == pytest.approx(v_star, rel=1e-9)
    assert run.b[0] == pytest.approx(b_star, rel=1e-9)


def test_identity_at_inverse_curvature_is_one_newton_step():
    run = train(x, y, IDENTITY, lr=1.0 / np.mean(x**2), steps=3, w0=0.5, lr_b=1.0)
    assert steps_to(run, L_star)[0] == 1


def test_square_saddle_at_zero_never_moves():
    """T'(0) = 0 for w^2: w = 0 is a stationary point whatever the data say."""
    run = train(x, y, SQUARE, lr=0.01, steps=500, w0=0.0, lr_b=1.0)
    assert np.all(run.w[:, 0] == 0.0)


@pytest.mark.parametrize("tf", DEFAULT, ids=lambda t: t.name)
def test_curvature_limit_predicts_stability(tf):
    """Just under the predicted limit converges from near w*; just over diverges."""
    lim = curvature_limit(x, tf, v_star)
    w_star = tf.inverse(v_star)
    run = train(x, y, tf, lr=[0.9 * lim, 1.1 * lim], steps=4000, w0=w_star + 1e-3, lr_b=1.0)
    s = steps_to(run, L_star)
    assert s[0] >= 0
    assert s[1] == -1


def test_mass_energy_is_identity_with_lr_times_c4():
    """T(w) = c^2 w: the trajectory of c^2 w equals identity GD at lr * c^4."""
    c = 7.0
    me = mass_energy(c)
    a = train(x, y, me, lr=1e-5, steps=200, w0=0.5 / c**2, lr_b=1.0)
    b = train(x, y, IDENTITY, lr=1e-5 * c**4, steps=200, w0=0.5, lr_b=1.0)
    np.testing.assert_allclose(a.v[:, 0], b.v[:, 0], rtol=1e-10)
