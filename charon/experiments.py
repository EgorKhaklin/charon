"""Run every experiment, print the tables, write figures/ and results/summary.json.

    python -m charon.experiments
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .core import classify, curvature_limit, loss, make_data, ols, steps_to, train  # noqa: E402
from .transforms import C_LIGHT, DEFAULT, IDENTITY, mass_energy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG, RES = ROOT / "figures", ROOT / "results"

# Categorical slots 1-5 of the reference palette, light mode, in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 12,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "legend.frameon": False,
    "legend.labelcolor": INK2, "figure.dpi": 140,
})

STEPS = 5000
RTOL = 1e-6
LRS = np.logspace(-6, 0, 61)


def color(tf):
    return COLORS[[t.name for t in DEFAULT].index(tf.name)]


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    out = "\n".join(lines)
    print(out + "\n")
    return out


def e1_landscape(x, y, v_star, b_star, L_star):
    print("## E1. The landscape each transform makes (b held at its optimum)\n")
    w = np.linspace(-4, 4, 2001)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rows = []
    for tf in DEFAULT:
        L = loss(x, y, tf.T(w), np.full_like(w, b_star))
        ax.plot(w, L / L_star, color=color(tf), label=tf.name)
        mins = w[1:-1][(L[1:-1] < L[:-2]) & (L[1:-1] < L[2:])]
        flat = "yes" if tf.name in ("w^2", "w^3") else "no"
        rows.append([tf.name, ", ".join(f"{m:+.2f}" for m in mins) or "none", flat, tf.note])
    ax.set_yscale("log")
    ax.set_xlabel("raw parameter w")
    ax.set_ylabel("loss / best possible loss")
    ax.set_title("Same data, five landscapes")
    ax.legend(loc="upper center", ncol=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "landscape.png")
    plt.close(fig)
    return table(["transform", "minima in w (on [-4, 4])", "T'(0)=0", "what the range does"], rows)


def e2_learning_rate(x, y, v_star, L_star):
    print(f"## E2. Fair learning rates: steps to reach the optimum (w0 = 0.5, {STEPS} step budget)\n")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    rows, best = [], {}
    for tf in DEFAULT:
        run = train(x, y, tf, lr=LRS, steps=STEPS, w0=0.5, lr_b=1.0)
        s = steps_to(run, L_star, RTOL)
        ok = s >= 0
        i = np.argmin(np.where(ok, s, np.iinfo(int).max))
        best[tf.name] = LRS[i]
        lim = curvature_limit(x, tf, v_star)
        ax.plot(LRS[ok], s[ok], color=color(tf), marker="o", markersize=3, label=tf.name)
        ax.axvline(lim, color=color(tf), linestyle=":", linewidth=1)
        rows.append([tf.name, f"{LRS[i]:.2g}", int(s[i]), f"{lim:.3g}",
                     f"{LRS[ok].max():.3g}", f"{LRS[ok].max() / LRS[ok].min():.0f}x"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("steps to the optimum")
    ax.set_title("Steps to the optimum, by learning rate")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "learning_rate.png")
    plt.close(fig)
    md = table(["transform", "best lr", "steps at best lr", "predicted max lr 2/(T'(w*)^2 E[x^2])",
                "largest lr that converged", "lr window"], rows)
    return md, best


def e2b_trajectories(x, y, v_star, best):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for tf in DEFAULT:
        run = train(x, y, tf, lr=best[tf.name], steps=15, w0=0.5, lr_b=1.0)
        ax.plot(run.v[:, 0], color=color(tf), label=f"{tf.name} (lr {best[tf.name]:.2g})")
    ax.axhline(v_star, color=INK2, linestyle="--", linewidth=1)
    ax.annotate("least-squares slope", (14, v_star), color=INK2, fontsize=9,
                ha="right", va="bottom")
    ax.set_xlabel("step")
    ax.set_ylabel("effective weight T(w)")
    ax.set_title("Effective weight, each at its best learning rate")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "trajectories.png")
    plt.close(fig)


def e3_init(x, y, best):
    print("## E3. Where you start: 201 starts on w0 in [-5, 5], each transform at its best lr\n")
    w0 = np.linspace(-5, 5, 201)
    rows, grid = [], {}
    for sign in (+1, -1):
        ys = sign * y
        v_s, _, L_s = ols(x, ys)
        for tf in DEFAULT:
            run = train(x, ys, tf, lr=best[tf.name], steps=STEPS, w0=w0, lr_b=1.0)
            lab = classify(run, tf, v_s, L_s, RTOL)
            grid[(sign, tf.name)] = lab
            n = {k: int((lab == k).sum()) for k in ("converged", "diverged", "stalled", "unreachable")}
            rows.append([f"{v_s:+.2f}", tf.name, n["converged"], n["diverged"], n["stalled"],
                         n["unreachable"]])
    md = table(["true slope", "transform", "converged", "diverged", "stalled", "unreachable"], rows)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
    shade = {"converged": "#2a78d6", "diverged": "#e34948", "stalled": "#c3c2b7",
             "unreachable": "#52514e"}
    for ax, sign in zip(axes, (+1, -1)):
        for j, tf in enumerate(DEFAULT):
            lab = grid[(sign, tf.name)]
            ax.scatter(w0, np.full_like(w0, j), c=[shade[k] for k in lab], marker="s", s=9,
                       linewidths=0)
        ax.set_yticks(range(len(DEFAULT)), [t.name for t in DEFAULT], fontsize=8)
        ax.set_ylim(-0.7, len(DEFAULT) - 0.3)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_title(f"true slope {'+' if sign > 0 else '-'}3", fontsize=10, loc="left")
    axes[-1].set_xlabel("starting w0")
    handles = [plt.Line2D([], [], marker="s", linestyle="", color=c, label=k) for k, c in shade.items()]
    fig.legend(handles=handles, loc="upper right", ncol=4, fontsize=8)
    fig.suptitle("Outcome by starting point", x=0.02, ha="left", color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "init_map.png")
    plt.close(fig)
    return md


def e4_noise(best):
    print("## E4. Noise: excess loss over least squares after the budget (w0 = 0.5, best lr)\n")
    rows = []
    for s in (0.1, 1.0, 2.0, 5.0, 10.0):
        x, y = make_data(noise=s, seed=7)
        v_s, _, L_s = ols(x, y)
        row = [s, f"{v_s:.3f}"]
        for tf in DEFAULT:
            run = train(x, y, tf, lr=best[tf.name], steps=STEPS, w0=0.5, lr_b=1.0)
            ex = (run.loss[-1, 0] - L_s) / L_s
            row.append("0" if abs(ex) < 1e-12 else f"{ex:.1e}")
        rows.append(row)
    return table(["noise sd", "least-squares slope"] + [t.name for t in DEFAULT], rows)


def e5_mass_energy(x, y, v_star, L_star):
    print("## E5. E = mc^2 as a transform: T(w) = c^2 w with the real c\n")
    me = mass_energy(C_LIGHT)
    lim = curvature_limit(x, me, v_star)
    rows = []
    for lr in (1e-2, 1e-6, 0.5 * lim):
        run = train(x, y, me, lr=lr, steps=200, w0=0.5 / C_LIGHT**2, lr_b=1.0)
        s = int(steps_to(run, L_star, RTOL)[0])
        rows.append([f"{lr:.3g}", "diverged" if not np.isfinite(run.loss[-1, 0]) else
                     (f"converged in {s} step{'s' * (s != 1)}" if s >= 0 else "not converged")])
    md = table(["lr", "outcome"], rows)
    print(f"Predicted stable limit: lr < 2 / (c^4 E[x^2]) = {lim:.3g}. "
          f"c^4 = {C_LIGHT**4:.3g}.\n")
    return md, lim


def e6_sparse():
    print("## E6. Where a transform does change the answer: an underdetermined problem\n")
    rng = np.random.default_rng(0)
    n, d, k = 40, 200, 5
    X = rng.normal(size=(n, d))
    w_true = np.zeros(d)
    support = rng.choice(d, k, replace=False)
    w_true[support] = rng.choice([-1.0, 1.0], k) * rng.uniform(1, 2, k)
    y = X @ w_true
    lam = np.linalg.eigvalsh(X.T @ X / n).max()

    def grad(w):
        return X.T @ (X @ w - y) / n

    w = np.zeros(d)
    for _ in range(20000):
        w -= (1.0 / lam) * grad(w)
    w_id = w

    rows, fits = [], {"identity: w": w_id}
    for alpha in (1e-1, 1e-2, 1e-4):
        u = np.full(d, alpha)
        v = np.full(d, alpha)
        for _ in range(60000):
            g = grad(u**2 - v**2)
            u, v = u - (0.1 / lam) * 2 * u * g, v + (0.1 / lam) * 2 * v * g
        fits[f"w = u^2 - v^2, init {alpha:g}"] = u**2 - v**2

    for name, w in fits.items():
        top = set(np.argsort(-np.abs(w))[:k])
        rows.append([name, f"{np.linalg.norm(X @ w - y) / np.linalg.norm(y):.1e}",
                     f"{np.linalg.norm(w - w_true) / np.linalg.norm(w_true):.3f}",
                     f"{len(top & set(support))}/{k}", f"{np.abs(w).sum():.2f}"])
    md = table(["parameterization", "train residual", "error vs true w", "support found",
                "L1 norm"], rows)
    print(f"True w: {k} nonzeros of {d}, {n} equations, L1 norm {np.abs(w_true).sum():.2f}.\n")

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, sharey=True)
    for ax, (name, w) in zip(axes, [("identity: w", w_id),
                                     ("w = u^2 - v^2, init 0.0001", fits["w = u^2 - v^2, init 0.0001"])]):
        ax.vlines(np.arange(d), 0, w, color=COLORS[0], linewidth=1.2)
        ax.scatter(support, w_true[support], s=36, facecolors="none", edgecolors=INK, zorder=3,
                   label="true nonzeros")
        ax.set_title(name, fontsize=10, loc="left")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("coordinate")
    fig.suptitle("Same data, same zero training error, different answers", x=0.02, ha="left",
                 color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "sparse.png")
    plt.close(fig)
    return md


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    x, y = make_data()
    v_star, b_star, L_star = ols(x, y)
    print(f"Data: y = 3x + 2 + N(0, 2^2), N = 100. Least squares: slope {v_star:.4f}, "
          f"intercept {b_star:.4f}, loss {L_star:.4f}.\n")

    out: dict[str, object] = {"ols": {"slope": v_star, "intercept": b_star, "loss": L_star}}
    out["e1"] = e1_landscape(x, y, v_star, b_star, L_star)
    out["e2"], best = e2_learning_rate(x, y, v_star, L_star)
    e2b_trajectories(x, y, v_star, best)
    out["best_lr"] = best
    out["e3"] = e3_init(x, y, best)
    out["e4"] = e4_noise(best)
    out["e5"], out["e5_limit"] = e5_mass_energy(x, y, v_star, L_star)
    out["e6"] = e6_sparse()
    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
