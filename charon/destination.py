"""Where a wormhole ends, computed without running it.

Gradient flow under a fixed elementwise wormhole, dw_i/dt = -m(w_i) dL/dw_i, is
mirror descent with the potential phi, phi'' = 1/m. Along the flow

    d/dt phi'(w) = -dL/dw = -X^T (X w - y) / n,

so phi'(w(t)) - phi'(w(0)) stays in the row space of X. If the flow reaches an
exact fit from w(0) = 0, that fit is the minimizer of sum_i phi(w_i) subject to
X w = y (THEORY.md, Theorem 1). Its optimality condition is phi'(w) = X^T nu,
so w = psi(X^T nu) with psi the inverse of phi'. `destination` solves for the
n numbers nu by Newton's method on the convex dual: no gradient descent at all.
"""

import numpy as np


def potential(m, wmax=50.0, points=20000):
    """phi' and its inverse psi for the metric m (a function of |w|, positive).

    phi'(w) = int_0^w du / m(u) is tabulated on a log grid up to wmax."""
    grid = np.concatenate([[0.0], np.logspace(-14, np.log10(wmax), points)])
    mid = 0.5 * (grid[1:] + grid[:-1])
    dphi = np.concatenate([[0.0], np.cumsum(np.diff(grid) / m(mid))])

    def phi_prime(w):
        return np.sign(w) * np.interp(np.abs(w), grid, dphi)

    def psi(z):
        return np.sign(z) * np.interp(np.abs(z), dphi, grid)

    return phi_prime, psi


def destination(X, y, m, iters=200):
    """The exact fit sum_i phi(w_i)-minimal, phi'' = 1/m. Returns (w, relative residual)."""
    _, psi = potential(m)
    nu = np.zeros(X.shape[0])
    F = X @ psi(X.T @ nu) - y
    for _ in range(iters):
        if np.linalg.norm(F) < 1e-12 * np.linalg.norm(y):
            break
        z = X.T @ nu
        H = (X * m(np.abs(psi(z)))) @ X.T + 1e-14 * np.eye(len(y))  # psi'(z) = m(psi(z))
        step = np.linalg.solve(H, F)
        t = 1.0
        while t > 1e-8:
            F_new = X @ psi(X.T @ (nu - t * step)) - y
            if np.linalg.norm(F_new) < np.linalg.norm(F):
                break
            t /= 2
        nu, F = nu - t * step, F_new
    return psi(X.T @ nu), np.linalg.norm(F) / np.linalg.norm(y)


def descend(X, y, m, steps, lr=0.25):
    """Gradient descent w <- w - (lr / lambda_max) m(|w|) grad from w = 0 (m <= 4 keeps it stable)."""
    n = X.shape[0]
    lam = np.linalg.eigvalsh(X.T @ X / n).max()
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w = w - (lr / lam) * m(np.abs(w)) * (X.T @ (X @ w - y) / n)
    return w


def basis_pursuit(X, y):
    """min ||w||_1 subject to X w = y, as a linear program."""
    from scipy.optimize import linprog
    d = X.shape[1]
    r = linprog(np.ones(2 * d), A_eq=np.hstack([X, -X]), b_eq=y, bounds=(0, None), method="highs")
    return r.x[:d] - r.x[d:]


def hadamard(a):
    """The metric of u^2 - v^2 with u*v = a/2: 4 sqrt(w^2 + a^2), capped at 4."""
    return lambda w: np.minimum(4 * np.sqrt(w * w + a * a), 4.0)


def log_wormhole(f, tau):
    """styx's distilled gate: f sqrt(|w|) + 4 w^2 / (w^2 + tau^2), capped at 4.

    The 1e-12 keeps m(0) > 0: with m(0) = 0 exactly, w = 0 is a dead point and
    gradient descent from zero never moves, although phi is finite."""
    return lambda w: np.minimum(f * np.sqrt(w + 1e-12) + 4 * w * w / (w * w + tau * tau), 4.0)
