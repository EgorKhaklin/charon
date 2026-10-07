"""Parameter transforms T(w): the model sees T(w), the optimizer moves w.

Gradient descent on w is gradient descent on the effective weight
v = T(w) with a position-dependent step size:

    dv ~= -lr * T'(w)^2 * dL/dv            (first order in lr)

so every transform here is a preconditioner, T'(w)^2, plus whatever the
range of T forbids (w^2 and exp cannot go negative; c*tanh cannot pass c).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


@dataclass(frozen=True)
class Transform:
    name: str
    T: Callable[[np.ndarray], np.ndarray]
    dT: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[float], float | None]  # a w with T(w) = v, or None if v is out of range
    note: str


def _clip(w):
    return np.clip(w, -50.0, 50.0)


IDENTITY = Transform(
    "identity", lambda w: w, lambda w: np.ones_like(w), lambda v: v,
    "ordinary regression; the baseline",
)

SQUARE = Transform(
    "w^2", lambda w: w**2, lambda w: 2.0 * w,
    lambda v: np.sqrt(v) if v >= 0 else None,
    "range [0, inf); two minima at +-sqrt(v*), a saddle at w=0",
)

CUBE = Transform(
    "w^3", lambda w: w**3, lambda w: 3.0 * w**2, np.cbrt,
    "full range, but T'(0)=0: the step size vanishes near zero",
)

EXP = Transform(
    "exp(w)", lambda w: np.exp(_clip(w)), lambda w: np.exp(_clip(w)),
    lambda v: np.log(v) if v > 0 else None,
    "range (0, inf); multiplicative updates, like exponentiated gradient",
)


def rapidity(c: float = 5.0) -> Transform:
    """Special relativity's own reparameterization: velocity = c * tanh(rapidity).

    Rapidities add where velocities don't, and no rapidity reaches c. As a
    weight transform it is a speed limit: |T(w)| < c, with steps that shrink
    as the effective weight approaches the limit.
    """
    return Transform(
        f"c*tanh(w/c), c={c:g}",
        lambda w: c * np.tanh(w / c),
        lambda w: 1.0 / np.cosh(np.clip(w / c, -350, 350)) ** 2,
        lambda v: c * np.arctanh(v / c) if abs(v) < c else None,
        "range (-c, c); a bounded weight, the relativistic speed limit",
    )


def mass_energy(c: float = C_LIGHT) -> Transform:
    """T(w) = c^2 * w: a constant rescale, so GD on it is identity GD with lr * c^4."""
    return Transform(
        f"c^2*w, c={c:g}", lambda w: c**2 * w, lambda w: np.full_like(w, c**2),
        lambda v: v / c**2,
        "E = mc^2 as a transform: identity with the step size multiplied by c^4",
    )


DEFAULT = (IDENTITY, SQUARE, CUBE, EXP, rapidity(5.0))
