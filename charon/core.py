"""Data, the closed-form optimum, and batched gradient descent on y_hat = T(w) x + b."""

from dataclasses import dataclass

import numpy as np

from .transforms import Transform


def make_data(n=100, w=3.0, b=2.0, noise=2.0, seed=42):
    """x on [-10, 10], y = w x + b + N(0, noise^2). Same setup as the original sketch."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-10.0, 10.0, n)
    y = w * x + b + rng.normal(0.0, noise, n)
    return x, y


def ols(x, y):
    """The best any model y_hat = v x + b can do: (v*, b*, loss*)."""
    v, b = np.polyfit(x, y, 1)
    return v, b, loss(x, y, v, b)


def loss(x, y, v, b):
    """L = (1/2N) sum((v x + b - y)^2), broadcast over v and b."""
    v, b = np.asarray(v, float), np.asarray(b, float)
    err = v[..., None] * x + b[..., None] - y
    return 0.5 * np.mean(err**2, axis=-1)


@dataclass
class Run:
    loss: np.ndarray       # (steps, B)
    w: np.ndarray          # (steps, B), the raw parameter
    v: np.ndarray          # (steps, B), the effective weight T(w)
    grad_w: np.ndarray     # (steps, B), |dL/dw|
    b: np.ndarray          # (B,), final bias
    b_hist: np.ndarray     # (steps, B), the bias at each step


def train(x, y, tf: Transform, lr, steps=1000, w0=0.5, b0=0.0, lr_b=None, adam=None):
    """Gradient descent on w (and b), for a whole batch of (lr, w0) at once.

    adam=(beta1, beta2, eps) swaps the w update for Adam; b keeps plain
    gradient descent with lr_b, so the comparison stays about w.

    lr and w0 broadcast to a common shape B; one run per element. The bias
    gets its own step size lr_b (default lr). With x centred, the Hessian is
    block-diagonal and lr_b = 1 / mean(1) = 1 solves b in one step, which
    keeps the comparison about w alone.

    dL/dw = mean((y_hat - y) x) * T'(w)  -- the chain rule, nothing more.
    A run that leaves the finite numbers is frozen at its last finite state
    and its later losses are inf.

    The loss is quadratic in (v, b), so each step needs only the data's
    moments, not the data: with d = (v - v*, b - b*),

        L = L* + (1/2) [d_v^2 E[x^2] + 2 d_v d_b E[x] + d_b^2]

    which costs O(B) per step instead of O(B N), and measures the excess
    over L* directly instead of as a difference of two large numbers.
    """
    lr, w = np.broadcast_arrays(np.asarray(lr, float), np.asarray(w0, float))
    lr, w = lr.ravel().copy(), w.ravel().copy()
    lr_b = lr if lr_b is None else np.full_like(lr, lr_b)
    b = np.full_like(w, b0)
    alive = np.ones_like(w, bool)
    out = {k: np.empty((steps, w.size)) for k in ("loss", "w", "v", "grad_w", "b")}
    v_star, b_star, L_star = ols(x, y)
    mx, mxx = np.mean(x), np.mean(x * x)
    m1, m2 = np.zeros_like(w), np.zeros_like(w)

    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(steps):
            v = tf.T(w)
            dv, db = v - v_star, b - b_star
            g_v = dv * mxx + db * mx           # mean(err * x)
            g_b = dv * mx + db                 # mean(err)
            L = L_star + 0.5 * (dv * g_v + db * g_b)
            g_w = g_v * tf.dT(w)

            alive &= np.isfinite(L) & np.isfinite(g_w) & (np.abs(w) < 1e150)
            out["loss"][t] = np.where(alive, L, np.inf)
            out["w"][t], out["v"][t], out["grad_w"][t], out["b"][t] = w, v, np.abs(g_w), b

            if adam is None:
                step = lr * g_w
            else:
                b1, b2, eps = adam
                m1 = b1 * m1 + (1 - b1) * g_w
                m2 = b2 * m2 + (1 - b2) * g_w**2
                step = lr * (m1 / (1 - b1 ** (t + 1))) / (np.sqrt(m2 / (1 - b2 ** (t + 1))) + eps)
            w = np.where(alive, w - step, w)
            b = np.where(alive, b - lr_b * g_b, b)

    return Run(out["loss"], out["w"], out["v"], out["grad_w"], b, out["b"])


ADAM = (0.9, 0.999, 1e-8)


def steps_to(run: Run, target, rtol=1e-6):
    """First step at which loss <= target * (1 + rtol); -1 if never (per run)."""
    hit = run.loss <= target * (1.0 + rtol)
    first = hit.argmax(axis=0)
    return np.where(hit.any(axis=0), first, -1)


def classify(run: Run, tf: Transform, v_star, target, rtol=1e-6):
    """Per run: converged | diverged | stalled (finite, off the optimum at the end)
    | unreachable (finite, and T cannot express the optimum at all)."""
    off = "unreachable" if tf.inverse(v_star) is None else "stalled"
    final = run.loss[-1]
    return np.where(~np.isfinite(final), "diverged",
                    np.where(final <= target * (1 + rtol), "converged", off))


def curvature_limit(x, tf: Transform, v_star):
    """Largest stable lr at the optimum, predicted from the local curvature.

    At the optimum d^2L/dw^2 = T'(w*)^2 mean(x^2) (the error is orthogonal to
    x there, so the T'' term drops), and GD on a quadratic is stable only for
    lr < 2 / curvature.
    """
    w_star = tf.inverse(v_star)
    if w_star is None:
        return np.nan
    return 2.0 / (float(tf.dT(np.asarray(w_star))) ** 2 * np.mean(x**2))
