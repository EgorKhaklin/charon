"""Wormholes within wormholes (E9), wormhole jumps (E10), where a wormhole ends (E11)
and the ceiling on what it can recover (E12).

    python -m charon.frontier [e9 e10 e11 e12]     # about 10 minutes for all four
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


def main():
    import sys
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    want = set(sys.argv[1:]) or {"e9", "e10", "e11", "e12"}
    runs = {"e9": e9_nested, "e10": e10_jumps, "e11": e11_destination, "e12": e12_ceiling}
    md = [runs[k]() for k in ("e9", "e10", "e11", "e12") if k in want]
    if want == set(runs):
        Path(RES / "frontier.md").write_text("\n\n".join(md) + "\n")


if __name__ == "__main__":
    main()
