"""Wormholes within wormholes: composed transforms, and jumps that keep the function.

compose(outer, inner) is the transform w -> outer(inner(w)). The chain rule gives
its step factor:

    T'(w)^2 = outer'(inner(w))^2 * inner'(w)^2

so nesting multiplies preconditioners. Nesting w|w| k times gives the signed power
p = 2^k, whose step factor in terms of the effective weight v is

    lr * T'(T^-1(v))^2 = lr * p^2 * |v|^(2 - 2/p)

which approaches |v|^2 as the depth grows: each nest makes small weights slower
and the step more rich-get-richer.

A wormhole jump changes the raw parameters while keeping the function. For
w = u^2 - v^2, gradient flow conserves u*v in every coordinate, and that
conserved quantity sets the implicit bias (small u*v, sparse answer).
jump_hadamard resets u*v while keeping u^2 - v^2 exactly.
"""

import numpy as np

from .transforms import Transform

SIGNED_SQUARE = Transform(
    "w|w|", lambda w: w * np.abs(w), lambda w: 2.0 * np.abs(w),
    lambda v: np.sign(v) * np.sqrt(abs(v)),
    "full range, T'(w) = 2|w|; the one-tensor cousin of u^2 - v^2",
)


def tanh_cap(c: float = 3.0) -> Transform:
    return Transform(
        f"{c:g}*tanh(w/{c:g})", lambda w: c * np.tanh(w / c),
        lambda w: 1.0 / np.cosh(np.clip(w / c, -350, 350)) ** 2,
        lambda v: c * np.arctanh(v / c) if abs(v) < c else None,
        "a bounded weight",
    )


SINH = Transform(
    "sinh(w)", np.sinh, np.cosh, np.arcsinh, "steps grow exponentially with |w|",
)


def compose(outer: Transform, inner: Transform) -> Transform:
    """outer o inner: the model sees outer(inner(w))."""

    def inverse(v):
        mid = outer.inverse(v)
        return None if mid is None else inner.inverse(mid)

    return Transform(
        f"{outer.name} o {inner.name}",
        lambda w: outer.T(inner.T(w)),
        lambda w: outer.dT(inner.T(w)) * inner.dT(w),
        inverse,
        f"{outer.name} applied to {inner.name}",
    )


def nest(tf: Transform, depth: int) -> Transform:
    """tf o tf o ... o tf, depth times."""
    out = tf
    for _ in range(depth - 1):
        out = compose(tf, out)
    return out


def sparse_problem(seed, n=40, d=200, k=5):
    """charon's E6 problem: n random equations, d unknowns, a k-sparse true weight."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    support = rng.choice(d, k, replace=False)
    w[support] = rng.choice([-1.0, 1.0], k) * rng.uniform(1, 2, k)
    return X, X @ w, w


def fit_transform(tf: Transform, X, y, lr, init, steps, seed=5):
    """Gradient descent on raw theta for the model X @ T(theta), from theta = +-init.

    Each step is scaled by 1 / (lambda_max * max T'(theta)^2), the stability limit
    of the current position (charon's E6 rule), so a transform is judged on the
    answer it finds, not on surviving one shared step size. Returns T(theta), or
    None if the run left the finite numbers.
    """
    n = X.shape[0]
    lam = np.linalg.eigvalsh(X.T @ X / n).max()
    theta = init * np.random.default_rng(seed).choice([-1.0, 1.0], X.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):  # a diverging run is reported as None
        for _ in range(steps):
            g = X.T @ (X @ tf.T(theta) - y) / n
            dT = tf.dT(theta)
            theta = theta - lr * g * dT / (lam * max((dT**2).max(), 1e-12))
            if not np.isfinite(theta).all():
                return None
        return tf.T(theta)


def balanced(w, beta):
    """(u, v) with u^2 - v^2 = w and u*v = beta, both nonnegative."""
    s = np.sqrt(w**2 + 4 * beta**2)
    return np.sqrt((s + w) / 2), np.sqrt((s - w) / 2)


def fit_hadamard(X, y, alpha, jump_every=None, beta=None, lr=0.1, budget=400_000, tol=1e-8):
    """GD on w = u^2 - v^2 from u = v = alpha; optionally jump to u*v = beta every
    jump_every steps (the function is unchanged by each jump). Returns (w, steps),
    steps = -1 if the residual never reached tol."""
    n = X.shape[0]
    lam = np.linalg.eigvalsh(X.T @ X / n).max()
    u = np.full(X.shape[1], alpha)
    v = u.copy()
    for t in range(1, budget + 1):
        r = X @ (u * u - v * v) - y
        g = X.T @ r / n
        step = lr / (lam * max(1.0, 4 * np.max(u * u + v * v)))
        u, v = u - step * 2 * u * g, v + step * 2 * v * g
        if jump_every and t % jump_every == 0:
            u, v = balanced(u * u - v * v, beta)
        if t % 100 == 0 and np.linalg.norm(r) / np.linalg.norm(y) < tol:
            return u * u - v * v, t
    return u * u - v * v, -1
