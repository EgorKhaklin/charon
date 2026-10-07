"""Wormholes within wormholes (E9), wormhole jumps (E10), where a wormhole ends (E11),
the ceiling on what it can recover (E12), quantum wormholes (E13, E14), and schedules
from scripture (E15).

    python -m charon.frontier [e9 ... e15]     # about 25 minutes for all seven
"""

import itertools
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiments import COLORS, FIG, INK, INK2, RES, table  # noqa: E402
from .destination import basis_pursuit, descend, destination, hadamard, log_wormhole  # noqa: E402
from . import quantum as Q  # noqa: E402
from .transforms import IDENTITY  # noqa: E402
from .wormholes import (SIGNED_SQUARE, SINH, compose, fit_hadamard, fit_transform, nest,  # noqa: E402
                        sparse_problem, tanh_cap)

STEPS = 6000
TUNE, TEST = range(8), range(100, 140)
GRID = list(itertools.product([0.1, 0.3, 1.0], [1e-1, 1e-2, 1e-3, 1e-4]))  # (lr, init)


def rel_err(w, w_true):
    return np.linalg.norm(w - w_true) / np.linalg.norm(w_true)


def e9_nested():
    print("## E9. Wormholes within wormholes: nested transforms on the sparse problem\n")
    cands = {
        "w (no wormhole)": IDENTITY,
        "w|w|": SIGNED_SQUARE,
        "w|w| o w|w| (2 deep)": nest(SIGNED_SQUARE, 2),
        "w|w| o w|w| o w|w| (3 deep)": nest(SIGNED_SQUARE, 3),
        "3tanh(w/3) o w|w|": compose(tanh_cap(3.0), SIGNED_SQUARE),
        "sinh o w|w|": compose(SINH, SIGNED_SQUARE),
        "w|w| o sinh": compose(SIGNED_SQUARE, SINH),
    }
    tune = [sparse_problem(s) for s in TUNE]
    test = [sparse_problem(s) for s in TEST]
    rows, out = [], {}
    for name, tf in cands.items():
        t0 = time.time()

        def score(lr, a, probs):
            errs, res = [], []
            for X, y, wt in probs:
                w = fit_transform(tf, X, y, lr, a, STEPS)
                if w is None:
                    return None
                errs.append(rel_err(w, wt))
                res.append(np.linalg.norm(X @ w - y) / np.linalg.norm(y))
            return np.array(errs), np.array(res)

        scored = {p: score(*p, tune) for p in GRID}
        lr, a = min((p for p in GRID if scored[p] is not None), key=lambda p: scored[p][0].mean())
        e, r = score(lr, a, test)
        out[name] = e
        rows.append([name, f"{lr:g}", f"{a:g}", f"{np.median(e):.3f}", f"{e.mean():.3f}",
                     f"{np.median(r):.0e}"])
        print(f"  ({name}: {time.time() - t0:.0f}s)")
    md = table(["transform", "lr (x stability limit)", "init", "median error vs true w",
                "mean error", f"median residual after {STEPS} steps"], rows)
    print(f"40 test problems (40 equations, 200 unknowns, 5 nonzeros), lr and init picked on 8 "
          f"separate problems. A residual far above 0 means the run had not fit the data.\n")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(out)
    ax.boxplot([out[k] for k in names], orientation="horizontal", widths=0.55, medianprops={"color": COLORS[1]},
               boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2},
               flierprops={"markeredgecolor": INK2, "markersize": 3})
    ax.set_yticks(range(1, len(names) + 1), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("error vs the true sparse w (0 = exact recovery)")
    ax.set_title("Two wormholes deep finds the sparse answer; three is too many", loc="left",
                 color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "nested.png")
    plt.close(fig)
    return md


def e10_jumps():
    print("## E10. Wormhole jumps: reset u*v mid-training, keep the function (falsified)\n")
    rows = []
    for seed in range(3):
        X, y, wt = sparse_problem(seed)
        runs = [("init 0.1", dict(alpha=0.1)),
                ("init 0.0001", dict(alpha=1e-4)),
                ("init 0.1, jump to u*v = 1e-4 every 200 steps",
                 dict(alpha=0.1, jump_every=200, beta=1e-4)),
                ("init 0.1, jump to u*v = 1e-8 every 200 steps",
                 dict(alpha=0.1, jump_every=200, beta=1e-8))]
        for name, kw in runs:
            w, st = fit_hadamard(X, y, **kw)
            rows.append([seed, name, st if st >= 0 else "over 400000",
                         f"{rel_err(w, wt):.3f}", f"{np.abs(w).sum():.2f}",
                         f"{np.abs(wt).sum():.2f}"])
    return table(["problem", "run", "steps to residual 1e-8", "error vs true w", "L1 norm",
                  "true L1 norm"], rows)


def e11_destination():
    print("## E11. The destination without the route (THEORY.md, Theorem 1)\n")
    rows = []
    for name, m in [("u^2 - v^2 metric, a = 0.01", hadamard(1e-2)),
                    ("u^2 - v^2 metric, a = 0.1", hadamard(1e-1)),
                    ("log wormhole, f = 1e-3, tau = 3e-3", log_wormhole(1e-3, 3e-3))]:
        for seed in range(3):
            X, y, wt = sparse_problem(seed)
            wd, res = destination(X, y, m)
            gaps = [np.linalg.norm(descend(X, y, m, steps, lr) - wd) / np.linalg.norm(wd)
                    for lr, steps in ((0.25, 20_000), (0.025, 200_000))]
            rows.append([name, seed, f"{res:.0e}", f"{rel_err(wd, wt):.4f}", f"{gaps[0]:.1e}",
                         f"{gaps[1]:.1e}"])
    return table(["wormhole", "problem", "predicted fit residual", "predicted error vs true w",
                  "GD vs predicted, lr 0.25", "GD vs predicted, lr 0.025"], rows)


def e12_ceiling():
    print("## E12. The ceiling: does any wormhole recover what L1 cannot? (Theorem 2)\n")
    worms = {"log wormhole (f 1e-4, tau 1e-3)": log_wormhole(1e-4, 1e-3),
             "u^2 - v^2 metric (a 1e-3)": hadamard(1e-3)}
    rows = []
    for magnitudes in ("all 1.5", "uniform 1 to 2"):
        for k in (6, 8, 10, 12):
            probs = []
            for seed in range(1000, 1100):
                X, y, wt = sparse_problem(seed, k=k)
                if magnitudes == "all 1.5":
                    wt = 1.5 * np.sign(wt)
                    y = X @ wt
                probs.append((X, y, wt))
            l1 = np.array([rel_err(basis_pursuit(X, y), wt) < 1e-2 for X, y, wt in probs])
            for name, m in worms.items():
                ok = np.array([rel_err(destination(X, y, m)[0], wt) < 1e-2 for X, y, wt in probs])
                rows.append([magnitudes, k, name, f"{ok.sum()}", f"{l1.sum()}",
                             f"{(ok & ~l1).sum()}", f"{(l1 & ~ok).sum()}"])
    return table(["nonzero sizes", "nonzeros", "wormhole", "wormhole exact (of 100)",
                  "L1 exact", "wormhole only", "L1 only"], rows)


def e13_quantum_noiseless():
    print("## E13. Quantum wormholes: which state comes out when many fit (4 qubits, no noise)\n")
    P = Q.paulis(4)
    rows = []
    for m in (24, 48, 96):
        res = {}
        for trial in range(10):
            rng = np.random.default_rng(1000 * m + trial)
            idx = np.concatenate([[0], 1 + rng.choice(255, m - 1, replace=False)])
            rho = Q.random_state(16, 1, rng)
            b = Q.measure(P[idx], rho)
            ests = {"least squares (min norm)": Q.least_squares(P[idx], b),
                    "positivity, convex": Q.psd_least_squares(P[idx], b),
                    "Born rho = AA†, start 0.001": Q.born(P[idx], b, 1e-3, rng),
                    "Born rho = AA†, start 1": Q.born(P[idx], b, 1.0, rng),
                    "Gibbs rho = exp(H)": Q.gibbs(P[idx], b)}
            for k, r in ests.items():
                res.setdefault(k, []).append((Q.fidelity(r, rho), Q.purity(r)))
        for k, v in res.items():
            v = np.array(v)
            rows.append([m, k, f"{v[:, 0].mean():.3f}", f"{(v[:, 0] > 0.99).sum()}/10",
                         f"{v[:, 1].mean():.2f}"])
    return table(["Pauli measurements (of 255)", "estimator", "fidelity with the true pure state",
                  "fidelity > 0.99", "purity of the estimate"], rows)


def e14_quantum_noisy():
    print("## E14. Quantum wormholes under shot noise: pure and mixed states (4 qubits)\n")
    P = Q.paulis(4)
    rows = []
    for rank in (1, 4):
        for n_shots in (100, 1000):
            for m in (96, 160):
                res = {}
                for trial in range(8):
                    rng = np.random.default_rng(11 * m + trial + n_shots + 1000 * rank)
                    idx = np.concatenate([[0], 1 + rng.choice(255, m - 1, replace=False)])
                    rho = Q.random_state(16, rank, rng)
                    b = Q.shots(P[idx], rho, n_shots, rng)
                    hold = np.zeros(m, bool)
                    hold[1 + rng.choice(m - 1, (m - 1) // 5, replace=False)] = True
                    Pt, bt = P[idx][~hold], b[~hold]
                    full = Q.psd_least_squares(Pt, bt)
                    v = np.linalg.eigh(full)[1][:, -1:]
                    ests = {"projected least squares": Q.projected_least_squares(Pt, bt),
                            "positivity, convex": full,
                            "rank 1 of the convex fit (told it is pure)": v @ v.conj().T,
                            "Born, start 0.001, held-out stop":
                                Q.born_held_out(Pt, bt, P[idx][hold], b[hold], 1e-3, rng)}
                    for k, r in ests.items():
                        res.setdefault(k, []).append(Q.fidelity(r, rho))
                rows.append([rank, n_shots, m] + [f"{np.mean(v):.3f}" for v in res.values()])
    return table(["true rank", "shots per Pauli", "measurements", "projected least squares",
                  "positivity, convex", "rank 1 of convex (told pure)",
                  "Born wormhole, small start"], rows)


def e15_scriptures():
    print("## E15. Schedules from scripture: time-varying, coupled and restarted wormholes vs L1\n")
    from .destination import basis_pursuit as bp

    def run(X, y, metric, steps=3000, w0=None):
        n = X.shape[0]
        lam = np.linalg.eigvalsh(X.T @ X / n).max()
        w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
        for t in range(steps):
            w = w - (0.25 / lam) * metric(w, t / steps) * (X.T @ (X @ w - y) / n)
        return w

    def gate(a, f, tau):
        return np.minimum(f * np.sqrt(a + 1e-12) + 4 * a * a / (a * a + tau * tau), 4.0)

    fams = {
        "log wormhole, fixed": (lambda f, tau: lambda X, y: run(
            X, y, lambda w, t: gate(np.abs(w), f, tau)),
            [(f, tau) for f in (1e-4, 1e-3) for tau in (1e-3, 1e-2)]),
        "Ezekiel 37: the breath (floor) withdrawn over training": (lambda f0, tau: lambda X, y: run(
            X, y, lambda w, t: gate(np.abs(w), f0 * 1e-4**t, tau)),
            [(f0, tau) for f0 in (1e-2, 1e-1) for tau in (1e-3, 1e-2)]),
        "Jacob's ladder: tau climbs up and down three times": (lambda f, tm: lambda X, y: run(
            X, y, lambda w, t: gate(np.abs(w), f, 1e-3 * (tm / 1e-3) ** (0.5 - 0.5 * np.cos(6 * np.pi * t))
                                    * (1 - t) + 1e-3 * t)),
            [(f, tm) for f in (1e-4, 1e-3) for tm in (1e-1, 1.0)]),
        "Solomon: speed set by size relative to the largest (coupled)": (lambda f, tau: lambda X, y: run(
            X, y, lambda w, t: gate(np.abs(w) / (np.abs(w).max() + 1e-12), f, tau)),
            [(f, tau) for f in (1e-4, 1e-3) for tau in (1e-3, 1e-2, 3e-2)]),
        "Revelation 21: a new, sharper creation started from the end": (lambda f, tau: lambda X, y: run(
            X, y, lambda w, t: gate(np.abs(w), f, tau),
            w0=run(X, y, lambda w, t: gate(np.abs(w), 1e-4, 1e-3))),
            [(1e-4, 1e-4), (1e-5, 1e-4), (1e-4, 3e-4)]),
    }
    rows = []
    for k in (8, 11):
        tune = [sparse_problem(s, k=k) for s in range(300, 312)]
        test = [sparse_problem(s, k=k) for s in range(400, 460)]
        l1 = np.array([rel_err(bp(X, y), wt) < 1e-2 for X, y, wt in test])
        rows.append([k, "L1 minimization", f"{l1.sum()}/60", "", ""])
        for name, (make, grid) in fams.items():
            p = min(grid, key=lambda p: np.median([rel_err(make(*p)(X, y), wt) for X, y, wt in tune]))
            ok = np.array([rel_err(make(*p)(X, y), wt) < 1e-2 for X, y, wt in test])
            rows.append([k, name, f"{ok.sum()}/60", f"{(ok & ~l1).sum()}", f"{(l1 & ~ok).sum()}"])
    return table(["nonzeros", "schedule", "exact recoveries", "beyond L1", "L1 only"], rows)


def main():
    import sys
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    runs = {"e9": e9_nested, "e10": e10_jumps, "e11": e11_destination, "e12": e12_ceiling,
            "e13": e13_quantum_noiseless, "e14": e14_quantum_noisy, "e15": e15_scriptures}
    want = set(sys.argv[1:]) or set(runs)
    md = [f() for k, f in runs.items() if k in want]
    if want == set(runs):
        Path(RES / "frontier.md").write_text("\n\n".join(md) + "\n")


if __name__ == "__main__":
    main()
