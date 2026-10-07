"""Quantum wormholes: the non-commuting case, on quantum state tomography.

Everything in THEORY.md is about elementwise wormholes, where every coordinate
commutes. The quantum version replaces the vector by a density matrix rho (Hermitian,
positive semidefinite, trace 1), and the wormhole by a matrix map:

    Born wormhole    rho = A A^dagger       the matrix u^2 (the Born rule)
    Gibbs wormhole   rho = exp(H) / tr      mirror descent with the von Neumann entropy

The task: an unknown n-qubit state, measured on m random Pauli strings (expectation
values tr(P rho)), with m far below the d^2 - 1 unknowns. Many states fit, so the
wormhole picks which one comes out: its implicit bias is a state-estimation principle.
"""

import itertools

import numpy as np

_P1 = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]


def paulis(nq):
    """All 4^nq Pauli strings as a (4^nq, 2^nq, 2^nq) complex array; index 0 is the identity."""
    out = []
    for idx in itertools.product(range(4), repeat=nq):
        M = np.array([[1.0 + 0j]])
        for i in idx:
            M = np.kron(M, _P1[i])
        out.append(M)
    return np.array(out)


def measure(P, rho):
    """tr(P_k rho) for every k."""
    return np.einsum("kij,ji->k", P, rho).real


def grad(P, b, rho):
    """Gradient in rho of |measure(P, rho) - b|^2 / (2m): a Hermitian matrix."""
    return np.einsum("k,kij->ij", measure(P, rho) - b, P) / len(b)


def eig_fn(M, f):
    e, V = np.linalg.eigh((M + M.conj().T) / 2)
    return (V * f(e)) @ V.conj().T


def fidelity(rho, sigma):
    """Uhlmann fidelity (tr sqrt(sqrt(sigma) rho sqrt(sigma)))^2, with rho normalized to trace 1."""
    rho = rho / np.trace(rho).real
    s = eig_fn(sigma, lambda e: np.sqrt(np.maximum(e, 0)))
    return float(np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(s @ rho @ s), 0))) ** 2)


def purity(rho):
    rho = rho / np.trace(rho).real
    return float(np.trace(rho @ rho).real)


def random_state(d, rank, rng):
    G = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    rho = G @ G.conj().T
    return rho / np.trace(rho).real


def _simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.nonzero(u * np.arange(1, len(v) + 1) > css - 1)[0][-1]
    return np.maximum(v - (css[k] - 1) / (k + 1), 0)


def least_squares(P, b):
    """The minimum-norm Hermitian fit (the Paulis are orthogonal): not positive in general."""
    d = P.shape[1]
    return (np.eye(d) + np.einsum("k,kij->ij", b[1:], P[1:])) / d


def psd_least_squares(P, b, steps=4000):
    """Projected gradient descent onto {rho >= 0, tr rho = 1}: the convex baseline."""
    d, m = P.shape[1], len(b)
    rho, eta = np.eye(d) / d, m / d
    for _ in range(steps):
        rho = eig_fn(rho - eta * grad(P, b, rho), _simplex)
    return rho


def born(P, b, start, rng, steps=20000):
    """Gradient descent on A, rho = A A^dagger, A full d x d (no rank assumed), from size `start`."""
    d, m = P.shape[1], len(b)
    A = start * (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2 * d)
    eta = 0.25 * m / d
    for _ in range(steps):
        A = A - eta * grad(P, b, A @ A.conj().T) @ A
    return A @ A.conj().T


def gibbs(P, b, steps=20000):
    """Gradient steps on H, rho = exp(H) / tr exp(H), from the maximally mixed state."""
    d, m = P.shape[1], len(b)
    H, eta = np.zeros((d, d), complex), 0.5 * m / d
    for _ in range(steps):
        rho = eig_fn(H, np.exp)
        H = H - eta * grad(P, b, rho / np.trace(rho).real)
    rho = eig_fn(H, np.exp)
    return rho / np.trace(rho).real


def born_held_out(P, b, Pv, bv, start, rng, steps=6000, every=50):
    """born(), keeping the iterate whose normalized state best predicts the held-out
    measurements (Pv, bv): early stopping without looking at the true state."""
    d, m = P.shape[1], len(b)
    A = start * (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2 * d)
    eta, best = 0.25 * m / d, (np.inf, None)
    for t in range(steps):
        A = A - eta * grad(P, b, A @ A.conj().T) @ A
        if t % every == 0 or t == steps - 1:
            rho = A @ A.conj().T
            rho = rho / np.trace(rho).real
            v = np.linalg.norm(measure(Pv, rho) - bv)
            if v < best[0]:
                best = (v, rho)
    return best[1]


def projected_least_squares(P, b):
    """Least squares, then the nearest state (positive, trace 1): Guta et al.'s estimator."""
    return eig_fn(least_squares(P, b), _simplex)


def shots(P, rho, n_shots, rng):
    """Each Pauli measured n_shots times: the mean of +-1 outcomes (the identity is exact)."""
    p = np.clip((1 + measure(P, rho)) / 2, 0, 1)
    b = 2 * rng.binomial(n_shots, p) / n_shots - 1
    b[0] = 1.0
    return b
