import numpy as np
import pytest

from charon.quantum import (born, fidelity, gibbs, grad, least_squares, measure, paulis,
                            psd_least_squares, purity, random_state)

P2 = paulis(2)


def test_paulis_are_orthogonal_and_hermitian():
    d = P2.shape[1]
    gram = np.einsum("aij,bji->ab", P2, P2).real
    assert gram == pytest.approx(d * np.eye(16))
    assert np.allclose(P2, np.conj(np.transpose(P2, (0, 2, 1))))


def test_full_tomography_inverts():
    rho = random_state(4, 2, np.random.default_rng(0))
    assert least_squares(P2, measure(P2, rho)) == pytest.approx(rho)


def test_grad_matches_finite_difference():
    rng = np.random.default_rng(1)
    rho, sigma = random_state(4, 1, rng), random_state(4, 2, rng)
    b = measure(P2[:7], sigma)
    E = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    E = (E + E.conj().T) / 2
    loss = lambda r: 0.5 * np.mean((measure(P2[:7], r) - b) ** 2)  # noqa: E731
    h = 1e-6
    fd = (loss(rho + h * E) - loss(rho - h * E)) / (2 * h)
    assert np.trace(grad(P2[:7], b, rho) @ E).real == pytest.approx(fd, rel=1e-6)


def test_fidelity_and_purity():
    rng = np.random.default_rng(2)
    rho = random_state(4, 1, rng)
    assert fidelity(rho, rho) == pytest.approx(1.0)
    assert purity(rho) == pytest.approx(1.0)
    assert purity(np.eye(4) / 4) == pytest.approx(0.25)
    assert fidelity(np.eye(4) / 4, rho) == pytest.approx(0.25)


@pytest.mark.parametrize("method", ["psd", "born", "gibbs"])
def test_every_estimator_recovers_a_fully_measured_state(method):
    rng = np.random.default_rng(3)
    rho = random_state(4, 1, rng)
    b = measure(P2, rho)
    est = {"psd": lambda: psd_least_squares(P2, b, 2000),
           "born": lambda: born(P2, b, 1e-2, rng, 4000),
           "gibbs": lambda: gibbs(P2, b, 4000)}[method]()
    assert fidelity(est, rho) > 0.99
